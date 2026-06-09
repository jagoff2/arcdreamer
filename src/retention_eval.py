from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

import torch

from audit.independent_verify import (
    anti_leakage_probes,
    durable_memory_corrupt_probe,
    score_outputs,
)
from audit.leakage_scan import run_scan
from .continual_learning import PersistentConceptMemory, learn_concept_sequence, recall_accuracy
from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .env import generate_batch
from .living_eval import (
    LIVING_CONFIGS,
    durable_restart_eval,
    idle_mode_eval,
    private_language_eval,
    richer_dynamics_eval,
    run_autoregressive_private,
)
from .model import load_checkpoint


RETENTION_CONFIGS = {
    "smoke": {"batch_size": 16, "seq_len": 96, "seed": 88200, "concept_ids": list(range(5))},
    "fast": {"batch_size": 128, "seq_len": 112, "seed": 88200, "concept_ids": list(range(5))},
}


HASH_PATHS = [
    "frozen/recurrent_latent_fast.pt",
    "frozen/manifest.json",
    "src/device.py",
    "src/model.py",
    "src/continual_learning.py",
    "src/retention_eval.py",
    "src/curriculum.py",
    "audit/independent_verify.py",
]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def hash_summary() -> dict[str, Any]:
    hashes = {path: {"sha256": sha256(path), "size_bytes": Path(path).stat().st_size} for path in HASH_PATHS}
    manifest = json.loads(Path("frozen/manifest.json").read_text(encoding="utf-8"))
    return {
        "hashes": hashes,
        "manifest": manifest,
        "model_hash_matches_manifest": hashes["src/model.py"]["sha256"] == manifest["model_sha256"],
        "checkpoint_hash_matches_manifest": hashes[manifest["checkpoint"]]["sha256"] == manifest["checkpoint_sha256"],
        "checkpoint_size_matches_manifest": hashes[manifest["checkpoint"]]["size_bytes"] == manifest["checkpoint_size_bytes"],
    }


def old_task_scores(model, batch_size: int, seq_len: int, seed: int, device: str) -> dict[str, Any]:
    batch = generate_batch(batch_size, seq_len, seed, device=device)
    with torch.no_grad():
        outputs = run_autoregressive_private(model, batch)
    return score_outputs(outputs, batch)


def submetric_floor_checks(before: dict[str, float], after: dict[str, float]) -> dict[str, bool]:
    checks: dict[str, bool] = {}
    for key, value in after.items():
        if key == "core_mean":
            continue
        checks[key] = value >= 0.85 or (before.get(key, 0.0) < 0.85 and value < 0.85)
    return checks


def concept_restart_probe(memory_path: str | Path, concept_ids: Iterable[int], old_core: float, device: DeviceLike) -> dict[str, Any]:
    device = str(resolve_device(device))
    ids = [int(item) for item in concept_ids]
    loaded = PersistentConceptMemory.load(memory_path, device=device)
    zero = loaded.zeroed()
    corrupt = loaded.corrupted()
    loaded_recall = recall_accuracy(loaded, ids, device=device)
    zero_recall = recall_accuracy(zero, ids, device=device)
    corrupt_recall = recall_accuracy(corrupt, ids, device=device)
    return {
        "memory_path": str(memory_path),
        "loaded_new_concept_recall": loaded_recall,
        "zero_memory_new_concept_recall": zero_recall,
        "corrupt_memory_new_concept_recall": corrupt_recall,
        "old_task_core_after_reload": old_core,
        "checks": {
            "loaded_new_recall_ge_0_85": loaded_recall >= 0.85,
            "old_task_core_ge_0_90": old_core >= 0.90,
            "zero_memory_degrades_new_recall": loaded_recall - zero_recall >= 0.50,
            "corrupt_memory_degrades_new_recall": loaded_recall - corrupt_recall >= 0.50,
        },
    }


def prior_living_preservation(model, config_name: str, device: DeviceLike) -> dict[str, Any]:
    device = str(resolve_device(device))
    cfg = LIVING_CONFIGS[config_name]
    batch_size = int(cfg["batch_size"])
    seq_len = int(cfg["seq_len"])
    seed = int(cfg["seed"])
    durable = durable_restart_eval(model, batch_size, seq_len, seed, device=device)
    idle = idle_mode_eval(model, batch_size, 16 + int(cfg["idle_ticks"]), seed + 1000, device=device)
    private = private_language_eval(model, batch_size, seq_len, seed + 2000, device=device)
    dynamics = richer_dynamics_eval()
    checks = {
        "durable_restart_memory_ge_0_85": durable["memory_file_restart_final_memory_accuracy"] >= 0.85,
        "memory_beats_zero_reset_by_0_40": durable["memory_file_restart_final_memory_accuracy"] - durable["zero_reset_final_memory_accuracy"] >= 0.40,
        "idle_memory_ge_0_95": idle["final_memory_accuracy"] >= 0.95,
        "idle_object_pos_ge_0_95": idle["final_object_pos_accuracy"] >= 0.95,
        "idle_goal_action_ge_0_70": idle["endogenous_goal_action_accuracy"] >= 0.70,
        "public_repetition_lt_0_40": idle["public_language_repetition_ratio"] < 0.40,
        "private_repetition_lt_0_40": idle["private_token_repetition_ratio"] < 0.40,
        "private_tokens_generated_and_causal": (
            private["generated_private_unique_count"] >= 3.0
            and (
                private["private_channel_action_shift_rate"] > 0.0
                or private["private_channel_language_shift_rate"] > 0.0
            )
        ),
        "richer_dynamics_all_present": all(value > 0.0 for value in dynamics.values()),
    }
    return {
        "durable_restart": durable,
        "idle_mode": idle,
        "private_internal_language": private,
        "richer_dynamics": dynamics,
        "checks": checks,
        "passes": all(checks.values()),
    }


def audit_preservation(model, batch_size: int, seq_len: int, seed: int, device: DeviceLike) -> dict[str, Any]:
    device = str(resolve_device(device))
    leakage = run_scan()
    anti = anti_leakage_probes(model, batch_size, seq_len, seed + 31, device)
    corrupt = durable_memory_corrupt_probe(model, batch_size, seq_len, seed + 41, device)
    normal_core = anti["scores"]["normal_generated_private"]["core_mean"]
    checks = {
        "leakage_scan_passes": leakage["passes"],
        "hidden_target_canary_diff_zero": anti["hidden_target_canary_max_abs_diff"] == 0.0,
        "tensor_z_stream_no_text_loop": leakage["checks"]["runtime_not_prompt_loop"],
        "zero_z_degrades": anti["checks"]["zero_z_degrades_core"],
        "shuffled_z_degrades": anti["checks"]["shuffled_z_degrades_core"],
        "no_language_degrades_grounding": anti["checks"]["no_language_degrades_grounding"],
        "no_private_perturbs_outputs": anti["no_private_action_shift_rate"] > 0.0 or anti["no_private_language_shift_rate"] > 0.0,
        "random_private_perturbs_outputs": normal_core - anti["scores"]["random_private_token"]["core_mean"] > 0.10,
        "corrupt_tensor_memory_degrades": (
            corrupt["memory_file_restart_final_memory_accuracy"] - corrupt["corrupt_memory_final_memory_accuracy"] >= 0.30
        ),
    }
    return {
        "leakage_scan": leakage,
        "anti_leakage": anti,
        "tensor_memory_corrupt_probe": corrupt,
        "checks": checks,
        "passes": all(checks.values()),
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Retention Fix Report", ""]
    lines.append(f"Terminal outcome: **{report['terminal_outcome']}**")
    lines.append("")
    lines.append("## New Concepts")
    lines.append("")
    lines.append("| Concept ID | Before | After | Improvement | Pass |")
    lines.append("| ---: | ---: | ---: | ---: | --- |")
    for row in report["new_concepts"]["rows"]:
        lines.append(
            f"| {int(row['concept_id'])} | {row['accuracy_before']:.6f} | {row['accuracy_after']:.6f} | {row['improvement']:.6f} | {row['passes']} |"
        )
    lines.append("")
    lines.append("## Old-Task Retention")
    lines.append("")
    lines.append("| Metric | Before | After |")
    lines.append("| --- | ---: | ---: |")
    for key, before in report["old_task_retention"]["before"].items():
        after = report["old_task_retention"]["after"][key]
        lines.append(f"| `{key}` | {before:.6f} | {after:.6f} |")
    lines.append("")
    lines.append("## Restart")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["restart"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Prior Living Preservation")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["prior_living"]["checks"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Audit Preservation")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["audit_preservation"]["checks"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Limitations")
    if report["limitations"]:
        for item in report["limitations"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None.")
    lines.append("")
    return "\n".join(lines)


def evaluate_retention_fix(
    checkpoint: str | Path,
    config_name: str = "fast",
    json_output: str | Path | None = None,
    device: DeviceLike = AUTO_DEVICE,
) -> dict[str, Any]:
    device = str(resolve_device(device))
    cfg = RETENTION_CONFIGS[config_name]
    concept_ids = [int(item) for item in cfg["concept_ids"]]
    model = load_checkpoint(checkpoint, device=device)
    model.eval()

    before_old = old_task_scores(model, int(cfg["batch_size"]), 80, int(cfg["seed"]) + 100, device)
    memory_path = Path("runs") / f"retention_concepts_{config_name}.pt"
    learned = learn_concept_sequence(concept_ids, memory_path, device=device)
    after_old = old_task_scores(model, int(cfg["batch_size"]), 80, int(cfg["seed"]) + 100, device)
    submetric_checks = submetric_floor_checks(before_old, after_old)
    new_rows = []
    for row in learned["concept_rows"]:
        item = dict(row)
        item["passes"] = (
            item["accuracy_before"] <= 0.20
            and item["accuracy_after"] >= 0.85
            and item["improvement"] >= 0.50
        )
        new_rows.append(item)
    old_delta = after_old["core_mean"] - before_old["core_mean"]
    restart = concept_restart_probe(memory_path, concept_ids, after_old["core_mean"], device=device)
    living = prior_living_preservation(model, config_name, device)
    audit = audit_preservation(model, int(cfg["batch_size"]), int(cfg["seq_len"]), int(cfg["seed"]) + 200, device)
    retention_checks = {
        "old_task_core_ge_0_90": after_old["core_mean"] >= 0.90,
        "old_task_delta_ge_minus_0_05": old_delta >= -0.05,
        "old_submetrics_ge_0_85_or_na": all(submetric_checks.values()),
    }
    top_checks = {
        "new_concepts": all(row["passes"] for row in new_rows) and len(new_rows) >= 5,
        "old_task_retention": all(retention_checks.values()),
        "restart": all(restart["checks"].values()),
        "prior_living": living["passes"],
        "audit_preservation": audit["passes"],
    }
    limitations = []
    if not top_checks["new_concepts"]:
        limitations.append("One or more sequential concept IDs failed before/after/improvement gates.")
    if not top_checks["old_task_retention"]:
        limitations.append(f"Old-task retention failed: core after {after_old['core_mean']:.6f}, delta {old_delta:.6f}.")
    if not top_checks["restart"]:
        limitations.append("Concept-memory restart, zero-memory, or corrupt-memory gate failed.")
    if not top_checks["prior_living"]:
        limitations.append("Prior living-system preservation gate failed.")
    if not top_checks["audit_preservation"]:
        limitations.append("Audit-preservation gate failed.")
    report: dict[str, Any] = {
        "terminal_outcome": "RETENTION FIX PROVEN" if all(top_checks.values()) else "RETENTION FIX NOT PROVEN",
        "checkpoint": str(checkpoint),
        "config": config_name,
        "hash_manifest_summary": hash_summary(),
        "new_concepts": {
            "rows": new_rows,
            "final_recall_accuracy": learned["final_recall_accuracy"],
            "passes": top_checks["new_concepts"],
        },
        "old_task_retention": {
            "before": before_old,
            "after": after_old,
            "delta": old_delta,
            "submetric_floor_checks": submetric_checks,
            "checks": retention_checks,
            "passes": top_checks["old_task_retention"],
        },
        "restart": restart,
        "prior_living": living,
        "audit_preservation": audit,
        "top_level_checks": top_checks,
        "limitations": limitations,
    }
    if json_output is not None:
        json_path = Path(json_output)
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        json_path.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"terminal_outcome": report["terminal_outcome"], "limitations": limitations}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", choices=sorted(RETENTION_CONFIGS), default="fast")
    parser.add_argument("--json-output", default="docs/retention_fix_report.json")
    parser.add_argument("--device", default=AUTO_DEVICE)
    args = parser.parse_args()
    evaluate_retention_fix(args.checkpoint, args.config, args.json_output, args.device)


if __name__ == "__main__":
    main()
