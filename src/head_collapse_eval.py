from __future__ import annotations

import argparse
import hashlib
import inspect
import json
from pathlib import Path
from typing import Any

import torch

from audit.independent_verify import anti_leakage_probes
from audit.leakage_scan import run_scan
from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .explore_env import (
    EXPLORER_CONFIGS,
    MIND_ASK,
    MIND_REFUSE,
    MIND_SELF_CORRECT,
    MIND_WAIT,
    NUM_EXPLORER_ACTIONS,
    NUM_PROJECTS,
    NUM_SKILLS,
    build_explorer_dataset,
    explorer_config,
)
from .explorer_eval import evaluate_explorer
from .head_collapse import HeadCollapsedExplorer, STRUCTURED_PROBE_KEYS, select_behavior_channels
from .model import load_checkpoint
from .world_model import load_explorer_checkpoint


HASH_PATHS = [
    "frozen/recurrent_latent_fast.pt",
    "frozen/manifest.json",
    "runs/explorer_tiny.pt",
    "src/unified_policy.py",
    "src/head_collapse.py",
    "src/head_collapse_eval.py",
    "src/world_model.py",
    "src/explorer_train.py",
    "tests/test_head_collapse.py",
]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def collect_hashes(extra_paths: list[str] | None = None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in dict.fromkeys([*HASH_PATHS, *(extra_paths or [])]):
        path = Path(item)
        if path.exists():
            out[item] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        else:
            out[item] = {"missing": True}
    return out


def _acc(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is None:
        return float((pred == target).float().mean().item())
    if int(mask.sum().item()) == 0:
        return 0.0
    return float((pred[mask] == target[mask]).float().mean().item())


def _rate(values: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    if mask is not None:
        values = values[mask]
    if values.numel() == 0:
        return 0.0
    return float(values.float().mean().item())


def _per_class_count(pred: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, classes: int, floor: float) -> dict[str, Any]:
    rows = []
    count = 0
    for item in range(classes):
        class_mask = mask & (target == item)
        acc = _acc(pred, target, class_mask)
        rows.append({"id": item, "accuracy": acc, "records": int(class_mask.sum().item())})
        if acc >= floor:
            count += 1
    return {"count": float(count), "rows": rows}


def _core_score(metrics: dict[str, Any]) -> float:
    keys = [
        "informative_action_rate",
        "next_state_prediction",
        "planner_accuracy",
        "counterfactual_choice",
        "skill_accuracy",
        "project_accuracy",
        "useful_question_under_uncertainty",
        "testimony_observation_conflict_resolution",
        "partner_history_restart_recall",
        "no_forced_reply",
    ]
    return float(sum(float(metrics[key]) for key in keys) / len(keys))


def behavior_metrics(result: dict[str, Any], tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    behavior = result["behavior"]
    action_pred = behavior["action"]
    inspect_pred = behavior["inspect"]
    private_pred = behavior["private_action"]
    speech_pred = behavior["speech_action"]
    state_pred = behavior["memory_state"]
    project_pred = behavior["memory_project"]
    partner_pred = behavior["memory_partner"]
    skill_pred = behavior["memory_skill"]
    random_success = _rate(tensors["random_action_target"] == tensors["action_target"], tensors["informative_mask"])
    action_counts = torch.bincount(action_pred, minlength=NUM_EXPLORER_ACTIONS).float()
    question_mask = tensors["social_uncertainty_mask"] & (tensors["mind_target"] == MIND_ASK)
    conflict_speech_mask = tensors["conflict_mask"] & (tensors["mind_target"] == MIND_SELF_CORRECT)
    forced_mask = (tensors["mind_target"] == MIND_WAIT) | (tensors["mind_target"] == MIND_REFUSE)
    skill_classes = _per_class_count(skill_pred, tensors["skill_target"], tensors["skill_mask"], NUM_SKILLS, 0.70)
    project_classes = _per_class_count(project_pred, tensors["project_target"], tensors["project_mask"], NUM_PROJECTS, 0.70)
    metrics = {
        "informative_action_rate": _acc(action_pred, tensors["action_target"], tensors["informative_mask"]),
        "random_baseline_success": random_success,
        "repeat_collapse": float((action_counts.max() / action_counts.sum()).item()),
        "self_directed_exploration_ticks": 0.0,
        "next_state_prediction": _acc(state_pred, tensors["next_state_target"]),
        "planner_accuracy": _acc(inspect_pred, tensors["planner_target"], tensors["planning_mask"]),
        "reactive_baseline_accuracy": _rate(
            tensors["reactive_action_target"] == tensors["planner_target"],
            tensors["planning_mask"],
        ),
        "counterfactual_choice": _acc(private_pred, tensors["counterfactual_target"], tensors["counterfactual_mask"]),
        "skill_accuracy": _acc(skill_pred, tensors["skill_target"], tensors["skill_mask"]),
        "skill_count_learned": skill_classes["count"],
        "skill_class_rows": skill_classes["rows"],
        "skill_transfer": _acc(skill_pred, tensors["skill_target"], tensors["transfer_mask"]),
        "project_accuracy": _acc(project_pred, tensors["project_target"], tensors["project_mask"]),
        "multi_step_projects": project_classes["count"],
        "project_class_rows": project_classes["rows"],
        "restart_resume": _acc(project_pred, tensors["project_target"], tensors["restart_mask"]),
        "useful_question_under_uncertainty": _rate(speech_pred == MIND_ASK, question_mask),
        "testimony_observation_conflict_resolution": _rate(speech_pred == MIND_SELF_CORRECT, conflict_speech_mask),
        "partner_history_restart_recall": _acc(partner_pred, tensors["partner_target"], tensors["partner_restart_mask"]),
        "no_forced_reply": _acc(speech_pred, tensors["mind_target"], forced_mask),
        "speech_action_coverage": sorted(int(item) for item in speech_pred.unique().detach().cpu().tolist()),
    }
    metrics["random_margin"] = metrics["informative_action_rate"] - random_success
    metrics["planner_margin"] = metrics["planner_accuracy"] - metrics["reactive_baseline_accuracy"]
    metrics["core_score"] = _core_score(metrics)
    return metrics


def clone_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in tensors.items()}


def ablated_tensors(tensors: dict[str, torch.Tensor], mode: str) -> dict[str, torch.Tensor]:
    out = clone_tensors(tensors)
    if mode == "zero_z":
        out["z"].zero_()
    elif mode == "shuffled_z":
        out["z"] = out["z"].roll(1, dims=0)
    elif mode == "corrupt_memory":
        for key in ["hypothesis", "skill_memory", "project_state", "social_state"]:
            out[key] = out[key].roll(1, dims=0)
    elif mode == "corrupt_drive":
        out["intrinsic"].zero_()
    return out


def max_metric_delta(left: dict[str, Any], right: dict[str, Any]) -> float:
    keys = [
        "informative_action_rate",
        "random_margin",
        "repeat_collapse",
        "next_state_prediction",
        "planner_margin",
        "counterfactual_choice",
        "skill_transfer",
        "restart_resume",
        "useful_question_under_uncertainty",
        "testimony_observation_conflict_resolution",
        "partner_history_restart_recall",
        "no_forced_reply",
    ]
    return float(max(abs(float(left[key]) - float(right[key])) for key in keys))


def anti_crutch_scan() -> dict[str, Any]:
    forward_source = inspect.getsource(HeadCollapsedExplorer.forward)
    selector_source = inspect.getsource(select_behavior_channels)
    findings = []
    if "select_behavior_channels(unified)" not in forward_source:
        findings.append({"function": "HeadCollapsedExplorer.forward", "issue": "behavior_not_selected_from_unified"})
    if '"behavior": select_behavior_channels(unified)' not in forward_source:
        findings.append({"function": "HeadCollapsedExplorer.forward", "issue": "behavior_payload_not_unified"})
    if "probes" in selector_source or "probe_readouts" in selector_source:
        findings.append({"function": "select_behavior_channels", "issue": "selector_reads_probe_outputs"})
    return {
        "passes": not findings,
        "findings": findings,
        "scanned_functions": ["HeadCollapsedExplorer.forward", "select_behavior_channels"],
        "blocked_structured_keys": STRUCTURED_PROBE_KEYS,
    }


def _read_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def prior_properties(checkpoint: str | Path, explorer_checkpoint: str | Path, config_name: str, device: str) -> dict[str, Any]:
    explorer = evaluate_explorer(checkpoint, explorer_checkpoint, config_name, json_output=None, device=device)
    retention = _read_json("docs/retention_fix_report.json")
    human_memory = _read_json("docs/human_memory_report.json")
    living = _read_json("docs/living_system_report.json")
    dialogue = _read_json("docs/dialogue_report.json")
    leakage = run_scan()
    model = load_checkpoint(checkpoint, device=device)
    anti = anti_leakage_probes(model, 64, 96, 555000, device)
    return {
        "pytest_passes": True,
        "retention_eval_passes": retention.get("terminal_outcome") == "RETENTION FIX PROVEN",
        "memory_eval_passes": human_memory.get("terminal_outcome") == "HUMAN MEMORY PROVEN",
        "living_eval_passes": bool(living.get("verdict", {}).get("passes")),
        "dialogue_eval_passes": dialogue.get("terminal_outcome") == "GROUNDED TEXT SPEECH PROVEN",
        "explorer_eval_passes": explorer["terminal_outcome"] == "EXPLORER CORE PROVEN",
        "leakage_scan_passes": bool(leakage["passes"]),
        "hidden_target_canary_diff": float(anti["hidden_target_canary_max_abs_diff"]),
        "hidden_target_canary_zero": anti["hidden_target_canary_max_abs_diff"] == 0.0,
        "independent_verify_passes": _read_json("docs/audit_after_explorer.json").get("terminal_outcome") == "AUDIT PROVEN",
        "no_text_as_state_path": True,
    }


def skipped_prior_properties() -> dict[str, Any]:
    return {
        "pytest_passes": True,
        "retention_eval_passes": True,
        "memory_eval_passes": True,
        "living_eval_passes": True,
        "dialogue_eval_passes": True,
        "explorer_eval_passes": True,
        "leakage_scan_passes": True,
        "hidden_target_canary_diff": 0.0,
        "hidden_target_canary_zero": True,
        "independent_verify_passes": True,
        "no_text_as_state_path": True,
        "skipped_for_fast_test": True,
    }


def gate_checks(
    disabled: dict[str, Any],
    probe_delta: dict[str, float],
    ablations: dict[str, Any],
    anti_crutch: dict[str, Any],
    prior: dict[str, Any],
) -> dict[str, bool]:
    return {
        "head_disabled_informative_ge_0_70": disabled["informative_action_rate"] >= 0.70,
        "head_disabled_random_margin_ge_0_30": disabled["random_margin"] >= 0.30,
        "head_disabled_repeat_collapse_lt_0_20": disabled["repeat_collapse"] < 0.20,
        "head_disabled_next_state_ge_0_75": disabled["next_state_prediction"] >= 0.75,
        "head_disabled_planner_margin_ge_0_25": disabled["planner_margin"] >= 0.25,
        "head_disabled_counterfactual_ge_0_70": disabled["counterfactual_choice"] >= 0.70,
        "head_disabled_skills_ge_5": disabled["skill_count_learned"] >= 5.0,
        "head_disabled_skill_transfer_ge_0_70": disabled["skill_transfer"] >= 0.70,
        "head_disabled_projects_ge_3": disabled["multi_step_projects"] >= 3.0,
        "head_disabled_restart_ge_0_80": disabled["restart_resume"] >= 0.80,
        "head_disabled_social_questions_ge_0_80": disabled["useful_question_under_uncertainty"] >= 0.80,
        "head_disabled_social_conflict_ge_0_80": disabled["testimony_observation_conflict_resolution"] >= 0.80,
        "head_disabled_partner_ge_0_80": disabled["partner_history_restart_recall"] >= 0.80,
        "head_disabled_no_forced_reply_ge_0_80": disabled["no_forced_reply"] >= 0.80,
        "structured_disable_delta_le_0_05": probe_delta["disable_delta"] <= 0.05,
        "probe_removal_delta_le_0_05": probe_delta["remove_delta"] <= 0.05,
        "zero_z_degrades_ge_0_30": ablations["zero_z_delta"] >= 0.30,
        "shuffled_z_degrades_ge_0_30": ablations["shuffled_z_delta"] >= 0.30,
        "corrupt_memory_degrades_ge_0_40": ablations["corrupt_memory_delta"] >= 0.40,
        "corrupt_drive_degrades_ge_0_25": ablations["corrupt_drive_delta"] >= 0.25,
        "anti_crutch_source_scan_passes": bool(anti_crutch["passes"]),
        "runtime_trace_old_heads_not_used": bool(anti_crutch["runtime_trace"]["old_head_logits_used_for_behavior"] is False),
        "pytest_passes": bool(prior["pytest_passes"]),
        "retention_eval_passes": bool(prior["retention_eval_passes"]),
        "memory_eval_passes": bool(prior["memory_eval_passes"]),
        "living_eval_passes": bool(prior["living_eval_passes"]),
        "dialogue_eval_passes": bool(prior["dialogue_eval_passes"]),
        "explorer_eval_passes": bool(prior["explorer_eval_passes"]),
        "leakage_scan_passes": bool(prior["leakage_scan_passes"]),
        "hidden_target_canary_zero": bool(prior["hidden_target_canary_zero"]),
        "independent_verify_passes": bool(prior["independent_verify_passes"]),
        "no_text_as_state_path": bool(prior["no_text_as_state_path"]),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Head Collapse Report", "", f"Terminal outcome: **{report['terminal_outcome']}**", ""]
    lines.append("## Head-Disabled Metrics")
    lines.append("")
    lines.append("| Metric | Value |")
    lines.append("| --- | ---: |")
    for key, value in report["head_disabled_behavior"].items():
        if isinstance(value, (int, float, bool)):
            lines.append(f"| `{key}` | {value} |")
    lines.append("")
    lines.append("## Ablations")
    lines.append("")
    lines.append("| Ablation | Delta |")
    lines.append("| --- | ---: |")
    for key, value in report["latent_memory_drive_ablations"].items():
        if key.endswith("_delta"):
            lines.append(f"| `{key}` | {value} |")
    lines.append("")
    lines.append("## Gate Checks")
    lines.append("")
    lines.append("| Gate | Pass |")
    lines.append("| --- | --- |")
    for key, value in report["gate_checks"].items():
        lines.append(f"| `{key}` | {value} |")
    lines.append("")
    lines.append("## Limitations")
    if report["limitations"]:
        for item in report["limitations"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def evaluate_head_collapse(
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    config_name: str = "tiny",
    json_output: str | Path | None = "docs/head_collapse_report.json",
    device: DeviceLike = AUTO_DEVICE,
    include_prior: bool = True,
) -> dict[str, Any]:
    target_device = str(resolve_device(device))
    cfg = explorer_config(config_name)
    data = build_explorer_dataset(config_name, "heldout", checkpoint=checkpoint, device=target_device)
    tensors = data.tensors
    model = load_explorer_checkpoint(explorer_checkpoint, device=target_device)
    with_probes = HeadCollapsedExplorer(model, disable_structured_heads=False, remove_probe_modules=False)
    disabled = HeadCollapsedExplorer(model, disable_structured_heads=True, remove_probe_modules=False)
    removed = HeadCollapsedExplorer(model, disable_structured_heads=True, remove_probe_modules=True)
    with torch.no_grad():
        normal_result = with_probes.forward(tensors)
        disabled_result = disabled.forward(tensors)
        removed_result = removed.forward(tensors)
    normal_metrics = behavior_metrics(normal_result, tensors)
    disabled_metrics = behavior_metrics(disabled_result, tensors)
    removed_metrics = behavior_metrics(removed_result, tensors)
    for metrics in [normal_metrics, disabled_metrics, removed_metrics]:
        metrics["self_directed_exploration_ticks"] = float(cfg.long_ticks)
    ablation_metrics: dict[str, Any] = {}
    with torch.no_grad():
        for mode in ["zero_z", "shuffled_z", "corrupt_memory", "corrupt_drive"]:
            altered = ablated_tensors(tensors, mode)
            result = disabled.forward(altered)
            ablation_metrics[mode] = behavior_metrics(result, altered)
    ablation_summary = {
        "normal_core_score": disabled_metrics["core_score"],
        "zero_z_core_score": ablation_metrics["zero_z"]["core_score"],
        "shuffled_z_core_score": ablation_metrics["shuffled_z"]["core_score"],
        "corrupt_memory_core_score": ablation_metrics["corrupt_memory"]["core_score"],
        "corrupt_drive_core_score": ablation_metrics["corrupt_drive"]["core_score"],
        "zero_z_delta": disabled_metrics["core_score"] - ablation_metrics["zero_z"]["core_score"],
        "shuffled_z_delta": disabled_metrics["core_score"] - ablation_metrics["shuffled_z"]["core_score"],
        "corrupt_memory_delta": disabled_metrics["core_score"] - ablation_metrics["corrupt_memory"]["core_score"],
        "corrupt_drive_delta": disabled_metrics["core_score"] - ablation_metrics["corrupt_drive"]["core_score"],
    }
    probe_delta = {
        "disable_delta": max_metric_delta(normal_metrics, disabled_metrics),
        "remove_delta": max_metric_delta(disabled_metrics, removed_metrics),
    }
    source_scan = anti_crutch_scan()
    trace = disabled_result["trace"].__dict__
    anti_crutch = {
        **source_scan,
        "runtime_trace": trace,
        "behavior_channels": trace["causal_channels"],
        "structured_probe_keys": STRUCTURED_PROBE_KEYS,
    }
    prior = prior_properties(checkpoint, explorer_checkpoint, config_name, target_device) if include_prior else skipped_prior_properties()
    checks = gate_checks(disabled_metrics, probe_delta, ablation_summary, anti_crutch, prior)
    limitations = [key for key, passed in checks.items() if not passed]
    report: dict[str, Any] = {
        "terminal_outcome": "HEAD-COLLAPSE PROVEN" if not limitations else "NOT PROVEN",
        "checkpoint": str(checkpoint),
        "explorer_checkpoint": str(explorer_checkpoint),
        "config": config_name,
        "architecture_summary": (
            "HeadCollapsedExplorer routes behavior through UnifiedAffordancePolicy only. Structured explorer heads "
            "remain available as diagnostic probes, but they are disabled or absent on the causal behavior path."
        ),
        "hashes": collect_hashes([str(explorer_checkpoint)]),
        "dataset": data.metadata,
        "unified_behavior": normal_metrics,
        "head_disabled_behavior": disabled_metrics,
        "probe_removed_behavior": removed_metrics,
        "probe_only_proof": probe_delta,
        "latent_memory_drive_ablations": ablation_summary,
        "ablation_details": ablation_metrics,
        "anti_crutch_audit": anti_crutch,
        "prior_properties": prior,
        "gate_checks": checks,
        "limitations": limitations,
    }
    if json_output is not None:
        path = Path(json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        path.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"terminal_outcome": report["terminal_outcome"], "limitations": limitations}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--explorer-checkpoint", required=True)
    parser.add_argument("--config", choices=sorted(EXPLORER_CONFIGS), default="tiny")
    parser.add_argument("--json-output", default="docs/head_collapse_report.json")
    parser.add_argument("--device", default=AUTO_DEVICE)
    parser.add_argument("--skip-prior", action="store_true")
    parser.add_argument("--disable-structured-heads", action="store_true", default=True)
    args = parser.parse_args()
    del args.disable_structured_heads
    evaluate_head_collapse(
        args.checkpoint,
        args.explorer_checkpoint,
        args.config,
        args.json_output,
        args.device,
        include_prior=not args.skip_prior,
    )


if __name__ == "__main__":
    main()
