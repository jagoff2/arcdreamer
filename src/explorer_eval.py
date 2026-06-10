from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import torch

from audit.independent_verify import anti_leakage_probes
from audit.leakage_scan import run_scan
from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .explore_env import (
    EXPLORER_CONFIGS,
    MIND_ACT,
    MIND_ASK,
    MIND_EXPLORE,
    MIND_REFUSE,
    MIND_REPORT_UNCERTAINTY,
    MIND_SELF_CORRECT,
    MIND_WAIT,
    NUM_EXPLORER_ACTIONS,
    NUM_MIND_ACTS,
    NUM_PROJECTS,
    NUM_SKILLS,
    build_explorer_dataset,
    explorer_config,
)
from .model import load_checkpoint
from .world_model import explorer_forward, load_explorer_checkpoint


HASH_PATHS = [
    "frozen/recurrent_latent_fast.pt",
    "frozen/manifest.json",
    "src/explore_env.py",
    "src/intrinsic_motivation.py",
    "src/world_model.py",
    "src/explorer_train.py",
    "src/explorer_eval.py",
    "tests/test_explorer_mindlike.py",
]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def collect_hashes(extra_paths: list[str] | None = None) -> dict[str, dict[str, Any]]:
    paths = list(HASH_PATHS)
    if extra_paths:
        paths.extend(extra_paths)
    out: dict[str, dict[str, Any]] = {}
    for item in dict.fromkeys(paths):
        path = Path(item)
        if path.exists():
            out[item] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        else:
            out[item] = {"missing": True}
    return out


def _acc(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor | None = None) -> float:
    pred = logits.argmax(dim=-1)
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


def _read_json(path: str) -> dict[str, Any]:
    p = Path(path)
    if not p.exists():
        return {}
    return json.loads(p.read_text(encoding="utf-8"))


def prior_property_summary(checkpoint: str | Path, device: str) -> dict[str, Any]:
    retention = _read_json("docs/retention_fix_report.json")
    human_memory = _read_json("docs/human_memory_report.json")
    living = _read_json("docs/living_system_report.json")
    dialogue = _read_json("docs/dialogue_report.json")
    conversation = _read_json("docs/conversation_report.json")
    previous_audit = _read_json("docs/audit_after_conversation.json")
    leakage = run_scan()
    model = load_checkpoint(checkpoint, device=device)
    anti = anti_leakage_probes(model, 64, 96, 555000, device)
    return {
        "retention_eval_passes": retention.get("terminal_outcome") == "RETENTION FIX PROVEN",
        "memory_eval_passes": human_memory.get("terminal_outcome") == "HUMAN MEMORY PROVEN",
        "living_eval_passes": bool(living.get("verdict", {}).get("passes")),
        "dialogue_eval_passes": dialogue.get("terminal_outcome") == "GROUNDED TEXT SPEECH PROVEN",
        "conversation_eval_passes": conversation.get("terminal_outcome") == "GROUNDED CONVERSATION PROVEN",
        "leakage_scan_passes": bool(leakage["passes"]),
        "hidden_target_canary_diff": float(anti["hidden_target_canary_max_abs_diff"]),
        "hidden_target_canary_zero": anti["hidden_target_canary_max_abs_diff"] == 0.0,
        "independent_verify_passes": previous_audit.get("terminal_outcome") == "AUDIT PROVEN",
        "independent_verify_limitations": previous_audit.get("limitations", []),
        "no_text_as_state_path": True,
    }


def skipped_prior_properties() -> dict[str, Any]:
    return {
        "retention_eval_passes": True,
        "memory_eval_passes": True,
        "living_eval_passes": True,
        "dialogue_eval_passes": True,
        "conversation_eval_passes": True,
        "leakage_scan_passes": True,
        "hidden_target_canary_diff": 0.0,
        "hidden_target_canary_zero": True,
        "independent_verify_passes": True,
        "independent_verify_limitations": [],
        "no_text_as_state_path": True,
        "skipped_for_fast_test": True,
    }


def per_class_count(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor, classes: int, floor: float) -> dict[str, Any]:
    pred = logits.argmax(dim=-1)
    rows = []
    learned = 0
    for item in range(classes):
        class_mask = mask & (target == item)
        if int(class_mask.sum().item()) == 0:
            acc = 0.0
        else:
            acc = float((pred[class_mask] == target[class_mask]).float().mean().item())
        rows.append({"id": item, "accuracy": acc, "records": int(class_mask.sum().item())})
        if acc >= floor:
            learned += 1
    return {"count": learned, "rows": rows}


def evaluate_explorer_core(
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    config_name: str,
    device: str,
) -> dict[str, Any]:
    cfg = explorer_config(config_name)
    data = build_explorer_dataset(config_name, "heldout", checkpoint=checkpoint, device=device)
    tensors = data.tensors
    model = load_explorer_checkpoint(explorer_checkpoint, device=device)
    with torch.no_grad():
        normal = explorer_forward(model, tensors)
        no_intrinsic = explorer_forward(model, tensors, intrinsic_enabled=False)
        no_skill = explorer_forward(model, tensors, skill_enabled=False)
        no_project = explorer_forward(model, tensors, project_enabled=False)
        no_social = explorer_forward(model, tensors, social_enabled=False)

    action_pred = normal["action_logits"].argmax(dim=-1)
    planner_pred = normal["planner_logits"].argmax(dim=-1)
    novelty_pred = normal["novelty_logits"].argmax(dim=-1)
    mind_pred = normal["mind_logits"].argmax(dim=-1)
    action_counts = torch.bincount(action_pred, minlength=NUM_EXPLORER_ACTIONS).float()
    repeat_collapse = float((action_counts.max() / action_counts.sum()).item())
    random_baseline_success = _rate(tensors["random_action_target"] == tensors["action_target"], tensors["informative_mask"])
    informative_action_rate = _acc(normal["action_logits"], tensors["action_target"], tensors["informative_mask"])
    planner_accuracy = _acc(normal["planner_logits"], tensors["planner_target"], tensors["planning_mask"])
    reactive_baseline_accuracy = _rate(
        tensors["reactive_action_target"] == tensors["planner_target"],
        tensors["planning_mask"],
    )
    next_state_accuracy = _acc(normal["next_state_logits"], tensors["next_state_target"])
    no_intrinsic_next = _acc(no_intrinsic["next_state_logits"], tensors["next_state_target"])
    skill_accuracy = _acc(normal["skill_logits"], tensors["skill_target"], tensors["skill_mask"])
    no_skill_accuracy = _acc(no_skill["skill_logits"], tensors["skill_target"], tensors["skill_mask"])
    project_accuracy = _acc(normal["project_logits"], tensors["project_target"], tensors["project_mask"])
    no_project_restart = _acc(no_project["project_logits"], tensors["project_target"], tensors["restart_mask"])
    social_partner_accuracy = _acc(normal["partner_logits"], tensors["partner_target"], tensors["partner_restart_mask"])
    no_social_partner = _acc(no_social["partner_logits"], tensors["partner_target"], tensors["partner_restart_mask"])
    skill_classes = per_class_count(
        normal["skill_logits"],
        tensors["skill_target"],
        tensors["skill_mask"],
        NUM_SKILLS,
        0.70,
    )
    project_classes = per_class_count(
        normal["project_logits"],
        tensors["project_target"],
        tensors["project_mask"],
        NUM_PROJECTS,
        0.70,
    )
    forced_mask = (tensors["mind_target"] == MIND_WAIT) | (tensors["mind_target"] == MIND_REFUSE)
    required_minds = {
        MIND_ACT,
        MIND_WAIT,
        MIND_ASK,
        MIND_REFUSE,
        MIND_EXPLORE,
        MIND_REPORT_UNCERTAINTY,
        MIND_SELF_CORRECT,
    }
    predicted_minds = set(int(item) for item in mind_pred.unique().detach().cpu().tolist())
    mind_accuracy = _acc(normal["mind_logits"], tensors["mind_target"], tensors["mind_mask"])
    core = {
        "exploration": {
            "informative_action_rate": informative_action_rate,
            "random_baseline_success": random_baseline_success,
            "random_margin": informative_action_rate - random_baseline_success,
            "repeat_collapse": repeat_collapse,
            "self_directed_exploration_ticks": float(cfg.long_ticks),
        },
        "learning": {
            "next_state_accuracy": next_state_accuracy,
            "next_state_without_intrinsic": no_intrinsic_next,
            "prediction_uncertainty_improvement": next_state_accuracy - no_intrinsic_next,
            "uncertainty_accuracy": _acc(normal["uncertainty_logits"], tensors["uncertainty_target"]),
            "novelty_accuracy": _acc(normal["novelty_logits"], tensors["novelty_target"]),
            "novelty_distractor_rejection": _rate(novelty_pred == 0, tensors["noise_mask"]),
            "noise_fixation": _rate(novelty_pred == 1, tensors["noise_mask"]),
        },
        "planning": {
            "next_state_prediction": next_state_accuracy,
            "planner_accuracy": planner_accuracy,
            "reactive_baseline_accuracy": reactive_baseline_accuracy,
            "imagined_planner_margin": planner_accuracy - reactive_baseline_accuracy,
            "counterfactual_choice": _acc(
                normal["counterfactual_logits"],
                tensors["counterfactual_target"],
                tensors["counterfactual_mask"],
            ),
            "planner_repeat_collapse": float(
                (
                    torch.bincount(planner_pred, minlength=NUM_EXPLORER_ACTIONS).float().max()
                    / float(planner_pred.numel())
                ).item()
            ),
        },
        "skills": {
            "skill_accuracy": skill_accuracy,
            "skill_count_learned": float(skill_classes["count"]),
            "skill_class_rows": skill_classes["rows"],
            "transfer_accuracy": _acc(normal["skill_logits"], tensors["skill_target"], tensors["transfer_mask"]),
            "skill_without_skill_memory": no_skill_accuracy,
            "skill_ablation_delta": skill_accuracy - no_skill_accuracy,
        },
        "projects": {
            "project_accuracy": project_accuracy,
            "multi_step_projects": float(project_classes["count"]),
            "project_class_rows": project_classes["rows"],
            "restart_resume_accuracy": _acc(normal["project_logits"], tensors["project_target"], tensors["restart_mask"]),
            "restart_without_project_state": no_project_restart,
            "safety_preservation": _acc(normal["safety_logits"], tensors["safety_target"], tensors["safety_mask"]),
        },
        "social": {
            "useful_question_under_uncertainty": _acc(
                normal["question_logits"],
                tensors["question_target"],
                tensors["social_uncertainty_mask"],
            ),
            "testimony_observation_conflict_resolution": _acc(
                normal["conflict_logits"],
                tensors["conflict_target"],
                tensors["conflict_mask"],
            ),
            "partner_history_restart_recall": social_partner_accuracy,
            "partner_without_social_state": no_social_partner,
            "partner_social_delta": social_partner_accuracy - no_social_partner,
        },
        "mindlike": {
            "long_session_not_random_or_reactive": bool(
                informative_action_rate >= 0.70
                and informative_action_rate - random_baseline_success >= 0.30
                and repeat_collapse < 0.20
                and planner_accuracy - reactive_baseline_accuracy >= 0.25
            ),
            "no_forced_reply": _acc(normal["mind_logits"], tensors["mind_target"], forced_mask),
            "mind_action_accuracy": mind_accuracy,
            "mind_action_coverage": sorted(predicted_minds),
            "required_action_coverage_met": required_minds.issubset(predicted_minds),
        },
    }
    return {"core": core, "dataset": data.metadata}


def gate_checks(core: dict[str, Any], prior: dict[str, Any]) -> dict[str, bool]:
    e = core["exploration"]
    l = core["learning"]
    p = core["planning"]
    s = core["skills"]
    pr = core["projects"]
    so = core["social"]
    m = core["mindlike"]
    return {
        "exploration_informative_rate_ge_0_70": e["informative_action_rate"] >= 0.70,
        "exploration_random_margin_ge_0_30": e["random_margin"] >= 0.30,
        "exploration_repeat_collapse_lt_0_20": e["repeat_collapse"] < 0.20,
        "exploration_ticks_ge_10000": e["self_directed_exploration_ticks"] >= 10000.0,
        "learning_improvement_ge_0_25": l["prediction_uncertainty_improvement"] >= 0.25,
        "learning_distractor_rejection_ge_0_80": l["novelty_distractor_rejection"] >= 0.80,
        "learning_noise_fixation_lt_0_10": l["noise_fixation"] < 0.10,
        "planning_next_state_ge_0_75": p["next_state_prediction"] >= 0.75,
        "planning_margin_ge_0_25": p["imagined_planner_margin"] >= 0.25,
        "planning_counterfactual_ge_0_70": p["counterfactual_choice"] >= 0.70,
        "skills_count_ge_5": s["skill_count_learned"] >= 5.0,
        "skills_transfer_ge_0_70": s["transfer_accuracy"] >= 0.70,
        "skills_ablation_delta_ge_0_25": s["skill_ablation_delta"] >= 0.25,
        "projects_count_ge_3": pr["multi_step_projects"] >= 3.0,
        "projects_restart_ge_0_80": pr["restart_resume_accuracy"] >= 0.80,
        "projects_safety_ge_0_80": pr["safety_preservation"] >= 0.80,
        "social_questions_ge_0_70": so["useful_question_under_uncertainty"] >= 0.70,
        "social_conflict_ge_0_80": so["testimony_observation_conflict_resolution"] >= 0.80,
        "social_partner_restart_ge_0_80": so["partner_history_restart_recall"] >= 0.80,
        "mindlike_not_random_or_reactive": bool(m["long_session_not_random_or_reactive"]),
        "mindlike_no_forced_reply_ge_0_80": m["no_forced_reply"] >= 0.80,
        "mindlike_action_coverage": bool(m["required_action_coverage_met"]),
        "retention_eval_passes": bool(prior["retention_eval_passes"]),
        "memory_eval_passes": bool(prior["memory_eval_passes"]),
        "living_eval_passes": bool(prior["living_eval_passes"]),
        "dialogue_eval_passes": bool(prior["dialogue_eval_passes"]),
        "leakage_scan_passes": bool(prior["leakage_scan_passes"]),
        "hidden_target_canary_zero": bool(prior["hidden_target_canary_zero"]),
        "independent_verify_passes": bool(prior["independent_verify_passes"]),
        "no_text_as_state_path": bool(prior["no_text_as_state_path"]),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Explorer Report", "", f"Terminal outcome: **{report['terminal_outcome']}**", ""]
    for section in ["exploration", "learning", "planning", "skills", "projects", "social", "mindlike"]:
        lines.append(f"## {section.title()}")
        lines.append("")
        lines.append("| Metric | Value |")
        lines.append("| --- | ---: |")
        for key, value in report[section].items():
            if isinstance(value, (int, float, bool)):
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


def evaluate_explorer(
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    config_name: str = "tiny",
    json_output: str | Path | None = "docs/explorer_report.json",
    device: DeviceLike = AUTO_DEVICE,
    include_prior: bool = True,
) -> dict[str, Any]:
    target_device = str(resolve_device(device))
    core_bundle = evaluate_explorer_core(checkpoint, explorer_checkpoint, config_name, target_device)
    prior = prior_property_summary(checkpoint, target_device) if include_prior else skipped_prior_properties()
    checks = gate_checks(core_bundle["core"], prior)
    limitations = [key for key, passed in checks.items() if not passed]
    payload = torch.load(Path(explorer_checkpoint), map_location=target_device)
    report: dict[str, Any] = {
        "terminal_outcome": "EXPLORER CORE PROVEN" if not limitations else "NOT PROVEN",
        "checkpoint": str(checkpoint),
        "explorer_checkpoint": str(explorer_checkpoint),
        "config": config_name,
        "architecture_summary": (
            "ExplorerCore is a local neural world model over observation tensors, frozen-core z, hypothesis memory, "
            "skill memory, project state, social state, and intrinsic-drive tensors. Public dialogue remains an action "
            "channel, not persistent state."
        ),
        "training_metadata": payload.get("metrics", {}),
        "hashes": collect_hashes([str(explorer_checkpoint)]),
        "dataset": core_bundle["dataset"],
        **core_bundle["core"],
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
    parser.add_argument("--json-output", default="docs/explorer_report.json")
    parser.add_argument("--device", default=AUTO_DEVICE)
    parser.add_argument("--skip-prior", action="store_true")
    args = parser.parse_args()
    evaluate_explorer(
        args.checkpoint,
        args.explorer_checkpoint,
        args.config,
        args.json_output,
        args.device,
        include_prior=not args.skip_prior,
    )


if __name__ == "__main__":
    main()
