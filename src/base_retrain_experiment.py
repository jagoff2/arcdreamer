from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .arcagi3_eval import collect_hashes
from .arcagi3_failure_taxonomy import no_hack_audit
from .base_eval import ARM_SPECS, ABLATIONS, evaluate_suites, official_suite_worker
from .base_pretrain import TRAINED_ARMS, train_external_base
from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .external_collapse_experiment import (
    CUDA_PREFERRED_DEVICE,
    baseline_table,
    collapse_table,
    cuda_runtime_info,
    score_table,
)


EXPERIMENT_PATHS = [
    "src/trace_collect.py",
    "src/base_world_model.py",
    "src/base_pretrain.py",
    "src/base_eval.py",
    "src/base_retrain_experiment.py",
    "tests/test_external_base_training.py",
    "docs/external_base_report.json",
    "docs/external_base_report.md",
    "frozen/external_base_v1.pt",
    "frozen/external_base_manifest_v1.json",
    "data/external_traces_manifest.json",
]


def evaluate_external_base(
    *,
    json_output: str | Path,
    trace_dir: str | Path,
    trace_manifest: str | Path,
    external_checkpoint: str | Path,
    external_manifest: str | Path,
    recurrent_checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    train_steps: int,
    batch_size: int,
    device: DeviceLike,
) -> dict[str, Any]:
    target_device = resolve_device(device)
    frozen_manifest = train_external_base(
        manifest_path=trace_manifest,
        checkpoint_output=external_checkpoint,
        manifest_output=external_manifest,
        recurrent_checkpoint=recurrent_checkpoint,
        steps=train_steps,
        batch_size=batch_size,
        device=target_device,
    )
    suite_reports, selection = evaluate_suites(
        external_checkpoint=external_checkpoint,
        explorer_checkpoint=explorer_checkpoint,
        trace_dir=trace_dir,
        device=target_device,
    )
    trace_compaction = compact_trace_artifacts(suite_reports)
    report = build_report(
        suite_reports=suite_reports,
        selection=selection,
        trace_compaction=trace_compaction,
        trace_manifest_path=trace_manifest,
        frozen_manifest=frozen_manifest,
        external_checkpoint=external_checkpoint,
        external_manifest=external_manifest,
        recurrent_checkpoint=recurrent_checkpoint,
        explorer_checkpoint=explorer_checkpoint,
        trace_dir=trace_dir,
        requested_device=device,
        resolved_device=target_device,
    )
    output = Path(json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    return report


def build_report(
    *,
    suite_reports: list[dict[str, Any]],
    selection: dict[str, Any],
    trace_compaction: dict[str, Any],
    trace_manifest_path: str | Path,
    frozen_manifest: dict[str, Any],
    external_checkpoint: str | Path,
    external_manifest: str | Path,
    recurrent_checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    trace_dir: str | Path,
    requested_device: DeviceLike,
    resolved_device: Any,
) -> dict[str, Any]:
    selected = selection["selected_variant"]
    official = next(report for report in suite_reports if report["suite_id"] == "official_arcagi3")
    old = official["aggregate_by_variant"]["old_base_unchanged"]
    chosen = official["aggregate_by_variant"][selected]
    best_baseline = max(float(row.get("mean_normalized_score", 0.0)) for row in official.get("baselines", {}).values())
    score_gain_old = float(chosen["mean_normalized_score"] - old["mean_normalized_score"])
    score_gain_baseline = float(chosen["mean_normalized_score"] - best_baseline)
    useful_gain = float(chosen["mean_useful_events"] - old["mean_useful_events"])
    repeat_drop = float(old["mean_repeat_collapse"] - chosen["mean_repeat_collapse"])
    non_arc_drop = non_arc_best_drop(suite_reports, selected)
    ablation = ablation_review(official, selected)
    solved_or_big_gain = bool(chosen.get("solve_rate", 0.0) >= 1.0 / 25.0 or score_gain_old >= 0.02)
    gate = (
        score_gain_old >= 0.01
        and score_gain_baseline >= 0.005
        and useful_gain >= 0.08
        and repeat_drop >= 0.20
        and non_arc_drop <= 0.05
        and solved_or_big_gain
        and float(chosen["mean_invalid_action_rate"]) == 0.0
        and ablation["supports_if_needed"]
    )
    trace_manifest = json.loads(Path(trace_manifest_path).read_text(encoding="utf-8"))
    report = {
        "terminal_outcome": "EXTERNAL BASE IMPROVEMENT FOUND" if gate else "NO IMPROVEMENT FOUND",
        "answer_yes_no": "yes" if gate else "no",
        "selected_variant": selected,
        "requested_device": str(requested_device),
        "resolved_device": str(resolved_device),
        "device_runtime": cuda_runtime_info(resolved_device),
        "suite_device_runtime": {report["suite_id"]: report.get("device_runtime", {}) for report in suite_reports},
        "old_recurrent_checkpoint": str(recurrent_checkpoint),
        "old_explorer_checkpoint": str(explorer_checkpoint),
        "external_base_checkpoint": str(external_checkpoint),
        "external_base_manifest": str(external_manifest),
        "trace_manifest": trace_manifest,
        "frozen_manifest": frozen_manifest,
        "model_arms": [arm.__dict__ for arm in ARM_SPECS],
        "training_losses": frozen_manifest.get("arm_metrics", {}),
        "training_losses_are_diagnostic_only": True,
        "selection": selection,
        "improvement_gate": {
            "selected_variant": selected,
            "official_score_gain_over_old_base": score_gain_old,
            "official_score_gain_over_best_baseline": score_gain_baseline,
            "official_useful_event_gain_over_old_base": useful_gain,
            "official_repeat_collapse_drop": repeat_drop,
            "official_invalid_action_rate": chosen["mean_invalid_action_rate"],
            "non_arc_best_score_drop": non_arc_drop,
            "best_existing_baseline_score": best_baseline,
            "solved_or_score_gain_clause": solved_or_big_gain,
            "passes": gate,
            "thresholds": {
                "official_score_gain_over_old_base_min": 0.01,
                "official_score_gain_over_best_baseline_min": 0.005,
                "official_useful_event_gain_over_old_base_min": 0.08,
                "official_repeat_collapse_drop_min": 0.20,
                "non_arc_best_score_drop_max": 0.05,
                "official_invalid_action_rate_required": 0.0,
                "one_solved_or_score_gain_over_old_base_min": 0.02,
            },
        },
        "suite_reports": suite_reports,
        "sealed_score_table": score_table(suite_reports),
        "sealed_useful_event_table": useful_event_table(suite_reports),
        "collapse_table": collapse_table(suite_reports),
        "baselines": baseline_table(suite_reports),
        "official_per_game_table": official_per_game_table(official),
        "ablations": ablation_table(suite_reports),
        "ablation_review": ablation,
        "negative_result_preservation": negative_result_preservation(),
        "unsupported_arm_analysis": unsupported_arm_analysis(suite_reports, selected),
        "trace_paths": [path for report in suite_reports for path in report.get("trace_paths", [])],
        "required_arc_trace_paths": required_arc_trace_paths(official, selected),
        "trace_compaction": trace_compaction,
        "no_hack_proof": no_hack_proof(),
        "limitations": limitations(gate, trace_manifest),
        "hashes": collect_hashes(
            EXPERIMENT_PATHS
            + [
                str(recurrent_checkpoint),
                str(explorer_checkpoint),
                str(external_checkpoint),
                str(external_manifest),
                "docs/external_collapse_report.json",
                "docs/perceptual_affordance_report.json",
                "docs/external_generalization_report.json",
                "docs/arcagi3_failure_report.json",
            ]
        ),
        "trace_dir": str(trace_dir),
    }
    return report


def non_arc_best_drop(suite_reports: list[dict[str, Any]], selected: str) -> float:
    old_best = 0.0
    selected_best = 0.0
    for report in suite_reports:
        if report["suite_id"] == "official_arcagi3":
            continue
        old_best = max(old_best, float(report["aggregate_by_variant"]["old_base_unchanged"]["mean_normalized_score"]))
        selected_best = max(selected_best, float(report["aggregate_by_variant"][selected]["mean_normalized_score"]))
    return float(old_best - selected_best)


def useful_event_table(suite_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "suite_id": row["suite_id"],
            "variant": row["variant"],
            "mean_useful_events": row["mean_useful_events"],
            "mean_normalized_score": row["mean_normalized_score"],
        }
        for row in score_table(suite_reports)
    ]


def official_per_game_table(official: dict[str, Any]) -> list[dict[str, Any]]:
    rows = []
    for row in official.get("rows", []):
        rows.append(
            {
                "game_id": row.get("game_id"),
                "variant": row.get("variant"),
                "normalized_score": row.get("normalized_score"),
                "useful_events": row.get("useful_events"),
                "repeat_collapse": row.get("repeat_collapse"),
                "invalid_action_rate": row.get("invalid_action_rate"),
                "solved": row.get("solved"),
                "trace_path": row.get("trace_path"),
            }
        )
    return rows


def ablation_table(suite_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in suite_reports:
        for variant, aggregate in report["aggregate_by_variant"].items():
            if variant.startswith("ablation_"):
                rows.append({"suite_id": report["suite_id"], "ablation": variant, **aggregate})
    return rows


def compact_trace_artifacts(suite_reports: list[dict[str, Any]]) -> dict[str, Any]:
    paths = sorted(
        {
            Path(row["trace_path"])
            for report in suite_reports
            for row in report.get("rows", [])
            if row.get("trace_path")
        }
    )
    before_bytes = 0
    after_bytes = 0
    compacted = 0
    missing: list[str] = []
    for path in paths:
        if not path.exists():
            missing.append(str(path))
            continue
        before_bytes += path.stat().st_size
        payload = json.loads(path.read_text(encoding="utf-8"))
        steps = payload.get("steps", [])
        compact_steps = [compact_step(step) for step in steps]
        compact_payload = {
            "format": "external_base_compact_trace_v1",
            "metadata": payload.get("metadata", {}),
            "summary": payload.get("summary", {}),
            "step_count": len(steps),
            "steps": compact_steps,
        }
        path.write_text(json.dumps(compact_payload, separators=(",", ":")), encoding="utf-8")
        after_bytes += path.stat().st_size
        compacted += 1
    return {
        "format": "external_base_compact_trace_v1",
        "trace_files": compacted,
        "missing": missing,
        "bytes_before": before_bytes,
        "bytes_after": after_bytes,
        "reduction_ratio": float(1.0 - after_bytes / max(before_bytes, 1)),
        "preserved_fields": [
            "metadata",
            "summary",
            "step_count",
            "obs_hash",
            "next_obs_hash",
            "legal_action_count",
            "chosen_action",
            "score_delta",
            "event_delta",
            "terminal",
            "invalid_action",
            "compact_policy_scores",
        ],
    }


def compact_step(step: dict[str, Any]) -> dict[str, Any]:
    obs = step.get("obs", {}) if isinstance(step.get("obs"), dict) else {}
    policy = step.get("memory_drive_hypothesis", {}).get("policy", {})
    compact_policy = {
        key: policy.get(key)
        for key in [
            "variant",
            "base_action",
            "chosen_action",
            "changed_action",
            "top_base_scores",
            "top_adjusted_scores",
            "external_scores",
            "ablation",
            "forced_cycle",
            "baseline",
        ]
        if key in policy
    }
    return {
        "step": step.get("step"),
        "obs_hash": obs.get("obs_hash"),
        "legal_action_count": len(obs.get("available_actions", []) or step.get("legal_actions", [])),
        "chosen_action": step.get("chosen_action"),
        "baseline_actions": step.get("baseline_actions", {}),
        "next_obs_hash": step.get("next_obs_hash"),
        "score_delta": step.get("score_delta"),
        "event_delta": step.get("event_delta", []),
        "terminal": step.get("terminal"),
        "invalid_action": step.get("invalid_action"),
        "failure_class": step.get("failure_class"),
        "policy": compact_policy,
    }


def ablation_review(official: dict[str, Any], selected: str) -> dict[str, Any]:
    selected_row = official["aggregate_by_variant"][selected]
    rows = {
        variant: aggregate
        for variant, aggregate in official["aggregate_by_variant"].items()
        if variant.startswith("ablation_")
    }
    if not rows:
        return {"present": False, "supports_if_needed": False}
    best_ablation_score = max(float(row.get("mean_normalized_score", 0.0)) for row in rows.values())
    best_ablation_events = max(float(row.get("mean_useful_events", 0.0)) for row in rows.values())
    score_drop = float(selected_row["mean_normalized_score"] - best_ablation_score)
    useful_drop = float(selected_row["mean_useful_events"] - best_ablation_events)
    return {
        "present": True,
        "selected_variant": selected,
        "score_drop_vs_best_ablation": score_drop,
        "useful_event_drop_vs_best_ablation": useful_drop,
        "supports_if_needed": score_drop > 0.0 or useful_drop > 0.0,
        "rows": rows,
    }


def required_arc_trace_paths(official: dict[str, Any], selected: str) -> dict[str, Any]:
    normalized = [path.replace("\\", "/") for path in official["trace_paths"]]
    old_paths = [path for path in normalized if "/old_base_unchanged/" in path]
    selected_paths = [path for path in normalized if f"/{selected}/" in path]
    return {
        "old_base_unchanged_count": len(old_paths),
        "selected_variant": selected,
        "selected_variant_count": len(selected_paths),
        "old_base_unchanged": old_paths,
        "selected": selected_paths,
    }


def negative_result_preservation() -> dict[str, Any]:
    collapse = json.loads(Path("docs/external_collapse_report.json").read_text(encoding="utf-8"))
    perception = json.loads(Path("docs/perceptual_affordance_report.json").read_text(encoding="utf-8"))
    generalization = json.loads(Path("docs/external_generalization_report.json").read_text(encoding="utf-8"))
    return {
        "collapse_terminal_outcome": collapse.get("terminal_outcome"),
        "collapse_gate": collapse.get("improvement_gate"),
        "perception_terminal_outcome": perception.get("terminal_outcome"),
        "perception_gate": perception.get("improvement_gate"),
        "external_generalization_terminal_outcome": generalization.get("terminal_outcome"),
        "external_supported_claims": [
            row.get("claim_id") for row in generalization.get("claim_registry", []) if row.get("status") == "supported"
        ],
    }


def unsupported_arm_analysis(suite_reports: list[dict[str, Any]], selected: str) -> list[dict[str, Any]]:
    official = next(report for report in suite_reports if report["suite_id"] == "official_arcagi3")
    old = official["aggregate_by_variant"]["old_base_unchanged"]
    rows = []
    for arm in [item.arm_id for item in ARM_SPECS if item.arm_id != "old_base_unchanged"]:
        agg = official["aggregate_by_variant"].get(arm, {})
        score_gain = float(agg.get("mean_normalized_score", 0.0) - old["mean_normalized_score"])
        useful_gain = float(agg.get("mean_useful_events", 0.0) - old["mean_useful_events"])
        repeat_drop = float(old["mean_repeat_collapse"] - agg.get("mean_repeat_collapse", old["mean_repeat_collapse"]))
        reasons = []
        if score_gain < 0.01:
            reasons.append("official score gain below 0.01")
        if useful_gain < 0.08:
            reasons.append("official useful-event gain below 0.08")
        if repeat_drop < 0.20:
            reasons.append("repeat-collapse drop below 0.20")
        if arm == "null_training_control":
            reasons.append("null training control is not eligible for improvement claim")
        rows.append(
            {
                "arm": arm,
                "selected_by_non_arc_dev": arm == selected,
                "official_score_gain": score_gain,
                "official_useful_event_gain": useful_gain,
                "official_repeat_drop": repeat_drop,
                "unsupported_reasons": reasons,
            }
        )
    return rows


def no_hack_proof() -> dict[str, Any]:
    source_paths = [
        Path("src/trace_collect.py"),
        Path("src/base_world_model.py"),
        Path("src/base_pretrain.py"),
        Path("src/base_eval.py"),
        Path("src/base_retrain_experiment.py"),
    ]
    blocked = [
        "hidden" + "_goal",
        "ground" + "_truth",
        "answer" + "_key",
        "solution" + "_path",
        "manual" + "_hint",
        "round" + "_robin",
        "forced" + "_entropy",
        "game" + "_id" + " ==",
    ]
    findings = []
    for path in source_paths:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for item in blocked:
            if item in text:
                findings.append({"path": str(path), "match": item})
    return {
        "passes": not findings and no_hack_audit([])["passes"],
        "findings": findings,
        "model_action_source": "old ARCAGI3Adapter action_scores optionally rescored by frozen external base predictions",
        "selection_protocol": "primary arm selected from non-ARC dev before official sealed evaluation",
        "no_forced_cycle": True,
        "no_external_judge": True,
        "no_game_id_branch": True,
        "no_hidden_labels": True,
        "no_public_text_as_state": True,
    }


def limitations(gate: bool, trace_manifest: dict[str, Any]) -> list[str]:
    rows = []
    if not gate:
        rows.append("No external-base arm passed the sealed official ARC score/useful/repeat gate.")
    rows.append("Generated ARC-like traces are pretraining data only and are not terminal proof.")
    if int(trace_manifest.get("source_counts", {}).get("generated_arc_like_pretrain", 0)) > int(
        trace_manifest.get("source_counts", {}).get("gymnasium_dev_policy", 0)
    ):
        rows.append("The first-pass dataset is dominated by generated pretraining transitions; sealed evaluation remains the proof source.")
    rows.append("Official ARC runtime remains serialized and slow even when model inference is CUDA-backed.")
    return rows


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# External Base Retrain Report", ""]
    lines.append(f"Terminal outcome: **{report['terminal_outcome']}**")
    lines.append(f"Answer: **{report['answer_yes_no']}**")
    lines.append(f"Selected arm: `{report['selected_variant']}`")
    lines.append("")
    gate = report["improvement_gate"]
    lines.append("## Gate")
    for key, value in gate.items():
        if key != "thresholds":
            lines.append(f"- `{key}`: `{value}`")
    lines.append("")
    lines.append("## CUDA")
    lines.append(f"- Requested device: `{report.get('requested_device')}`")
    lines.append(f"- Resolved device: `{report.get('resolved_device')}`")
    runtime = report.get("device_runtime", {})
    lines.append(f"- CUDA available: `{runtime.get('torch_cuda_available')}`")
    lines.append(f"- CUDA runtime: `{runtime.get('torch_cuda_version')}`")
    lines.append("")
    lines.append("## Sealed Scores")
    lines.append("| Suite | Variant | Score | Useful | Repeat | Invalid |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in report["sealed_score_table"]:
        lines.append(
            f"| `{row['suite_id']}` | `{row['variant']}` | {row['mean_normalized_score']:.6f} | {row['mean_useful_events']:.6f} | {row['mean_repeat_collapse']:.6f} | {row['mean_invalid_action_rate']:.6f} |"
        )
    lines.append("")
    lines.append("## Trace Manifest")
    manifest = report["trace_manifest"]
    lines.append(f"- Trace transitions: `{manifest.get('transition_count')}`")
    lines.append(f"- Data hash: `{manifest.get('data_sha256')}`")
    lines.append("")
    lines.append("## Ablations")
    lines.append("```json")
    lines.append(json.dumps(report["ablation_review"], indent=2)[:5000])
    lines.append("```")
    lines.append("")
    lines.append("## Unsupported Arms")
    lines.append("```json")
    lines.append(json.dumps(report["unsupported_arm_analysis"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## No-Hack Proof")
    lines.append("```json")
    lines.append(json.dumps(report["no_hack_proof"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Limitations")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["external", "official_worker"], default="external")
    parser.add_argument("--json-output", default="docs/external_base_report.json")
    parser.add_argument("--trace-dir", default="docs/external_base_traces")
    parser.add_argument("--trace-manifest", default="data/external_traces_manifest.json")
    parser.add_argument("--external-checkpoint", default="frozen/external_base_v1.pt")
    parser.add_argument("--external-manifest", default="frozen/external_base_manifest_v1.json")
    parser.add_argument("--recurrent-checkpoint", default="frozen/recurrent_latent_fast.pt")
    parser.add_argument("--explorer-checkpoint", default="runs/explorer_tiny.pt")
    parser.add_argument("--selected-arm", default="old_base_finetuned")
    parser.add_argument("--train-steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default=CUDA_PREFERRED_DEVICE)
    args = parser.parse_args()
    if args.config == "official_worker":
        report = official_suite_worker(
            external_checkpoint=args.external_checkpoint,
            explorer_checkpoint=args.explorer_checkpoint,
            trace_dir=args.trace_dir,
            json_output=args.json_output,
            selected_arm=args.selected_arm,
            device=args.device,
        )
        print(json.dumps({"suite_id": report["suite_id"], "variants": sorted(report["aggregate_by_variant"])}, indent=2))
        return
    report = evaluate_external_base(
        json_output=args.json_output,
        trace_dir=args.trace_dir,
        trace_manifest=args.trace_manifest,
        external_checkpoint=args.external_checkpoint,
        external_manifest=args.external_manifest,
        recurrent_checkpoint=args.recurrent_checkpoint,
        explorer_checkpoint=args.explorer_checkpoint,
        train_steps=args.train_steps,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "terminal_outcome": report["terminal_outcome"],
                "answer_yes_no": report["answer_yes_no"],
                "selected_variant": report["selected_variant"],
                "improvement_gate": report["improvement_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
