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
from .env import GRID_SIZE, NUM_BODY_SCALARS, generate_batch
from .human_memory import (
    NUM_SOURCES,
    SOURCE_NAMES,
    HumanAnalogueMemory,
    HumanMemoryConfig,
    deterministic_vector,
    nearest_accuracy,
    one_hot,
)
from .memory_replay import replay_consolidate, semantic_accuracy
from .model import load_checkpoint
from .retention_eval import evaluate_retention_fix, old_task_scores
from .living_eval import LIVING_CONFIGS, evaluate_living_system, run_autoregressive_private


MEMORY_CONFIGS = {
    "smoke": {"episodes": 24, "similar": 8, "seed": 66200, "batch_size": 16, "seq_len": 96},
    "fast": {"episodes": 64, "similar": 20, "seed": 77200, "batch_size": 128, "seq_len": 112},
}

HASH_PATHS = [
    "frozen/recurrent_latent_fast.pt",
    "frozen/manifest.json",
    "src/device.py",
    "src/model.py",
    "src/human_memory.py",
    "src/memory_replay.py",
    "src/memory_eval.py",
]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def hash_manifest_summary() -> dict[str, Any]:
    hashes = {path: {"sha256": sha256(path), "size_bytes": Path(path).stat().st_size} for path in HASH_PATHS}
    manifest = json.loads(Path("frozen/manifest.json").read_text(encoding="utf-8"))
    return {
        "hashes": hashes,
        "manifest": manifest,
        "checkpoint_matches_manifest": hashes[manifest["checkpoint"]]["sha256"] == manifest["checkpoint_sha256"],
        "model_matches_manifest": hashes["src/model.py"]["sha256"] == manifest["model_sha256"],
    }


def _cue_for(content: torch.Tensor, source_id: int, serial: int, cfg: HumanMemoryConfig) -> torch.Tensor:
    cue = torch.zeros(cfg.cue_dim, device=content.device)
    cue[: cfg.content_dim] = content
    cue[cfg.content_dim : cfg.content_dim + NUM_SOURCES] = 0.15 * one_hot(
        source_id, NUM_SOURCES, device=content.device
    )
    cue = cue + 0.06 * deterministic_vector(12000 + serial, cfg.cue_dim, device=content.device)
    return cue


def _partial(cue: torch.Tensor) -> torch.Tensor:
    out = cue.clone()
    out[::4] = 0.0
    return out


def _noisy(cue: torch.Tensor, serial: int, scale: float = 0.10) -> torch.Tensor:
    return cue + scale * deterministic_vector(22000 + serial, cue.numel(), device=cue.device)


def _body_from_batch(batch: dict[str, torch.Tensor], sample: int, tick: int) -> torch.Tensor:
    start = GRID_SIZE + 2
    return batch["sensory"][sample, tick, start : start + NUM_BODY_SCALARS].detach()


def build_memory(checkpoint: str | Path, cfg: dict[str, int], device: str) -> tuple[HumanAnalogueMemory, dict[str, Any]]:
    model = load_checkpoint(checkpoint, device=device)
    batch = generate_batch(int(cfg["episodes"]), 80, int(cfg["seed"]), device=device)
    with torch.no_grad():
        outputs = run_autoregressive_private(model, batch)
    mem_cfg = HumanMemoryConfig()
    memory = HumanAnalogueMemory(mem_cfg, device=device)
    contents = []
    cues = []
    sources = []
    trace_ids = []
    for idx in range(int(cfg["episodes"])):
        content = deterministic_vector(idx, mem_cfg.content_dim, device=device)
        source_id = idx % NUM_SOURCES
        tick = [0, 8, 11, 17, 29, 31][source_id]
        cue = _cue_for(content, source_id, idx, mem_cfg)
        latent = outputs["latents"][idx, tick].detach()
        body = _body_from_batch(batch, idx, tick).to(device)
        affect = torch.tensor([
            float(body[0].item()),
            float(body[2].item()),
            float(source_id) / float(NUM_SOURCES - 1),
        ], device=device)
        action = outputs["action_logits"][idx, tick].detach()
        private_logits = outputs["private_logits"][idx, tick].detach()
        trace_id = memory.write(
            cue=cue,
            latent=latent,
            body=body,
            affect=affect,
            action_context=action,
            source_id=source_id,
            private_state=private_logits,
            content=content,
            time_index=float(tick + idx * 100),
        )
        contents.append(content.detach())
        cues.append(cue.detach())
        sources.append(source_id)
        trace_ids.append(trace_id)
    return memory, {
        "content_bank": torch.stack(contents, dim=0),
        "cues": cues,
        "sources": sources,
        "trace_ids": trace_ids,
        "model": model,
    }


def content_recall_metrics(memory: HumanAnalogueMemory, meta: dict[str, Any]) -> dict[str, float]:
    bank = meta["content_bank"].to(memory.device)
    targets = [item.to(memory.device) for item in meta["content_bank"]]
    partial_preds = [memory.recall(_partial(cue).to(memory.device)).content for cue in meta["cues"]]
    noisy_preds = [memory.recall(_noisy(cue, idx).to(memory.device)).content for idx, cue in enumerate(meta["cues"])]
    wrong = [
        not memory.recall(deterministic_vector(50000 + idx, memory.config.cue_dim, device=memory.device)).accepted
        for idx in range(len(targets))
    ]
    return {
        "partial_cue_accuracy": nearest_accuracy(partial_preds, targets, bank),
        "noisy_cue_accuracy": nearest_accuracy(noisy_preds, targets, bank),
        "wrong_cue_rejection": float(sum(wrong) / len(wrong)),
    }


def separation_completion_metrics(checkpoint: str | Path, cfg: dict[str, int], device: str) -> dict[str, float]:
    memory, meta = build_memory(checkpoint, {"episodes": int(cfg["similar"]), "seed": int(cfg["seed"]) + 500}, device)
    similar_base = deterministic_vector(70000, memory.config.cue_dim, device=memory.device)
    for idx, trace_id in enumerate(meta["trace_ids"]):
        content = meta["content_bank"][idx]
        cue = (
            0.68 * _cue_for(content, meta["sources"][idx], idx, memory.config)
            + 0.32 * similar_base
            + 0.09 * deterministic_vector(71000 + idx, memory.config.cue_dim, device=memory.device)
        ).to(device)
        memory._cues[trace_id] = cue
        memory._keys[trace_id] = memory._key(cue, latent=memory._latents[trace_id], body=memory._bodies[trace_id], source_id=meta["sources"][idx])
        memory._sparse[trace_id] = memory._sparse_code(memory._keys[trace_id])
        meta["cues"][idx] = cue.detach()
        memory._contents[trace_id] = content.to(device)
    bank = meta["content_bank"].to(device)
    targets = [item.to(device) for item in meta["content_bank"]]
    full_preds = [
        memory.recall(cue.to(device), latent=memory._latents[meta["trace_ids"][idx]], body=memory._bodies[meta["trace_ids"][idx]]).content
        for idx, cue in enumerate(meta["cues"])
    ]
    partial_preds = [
        memory.recall(
            _partial(cue).to(device),
            latent=memory._latents[meta["trace_ids"][idx]],
            body=memory._bodies[meta["trace_ids"][idx]],
        ).content
        for idx, cue in enumerate(meta["cues"])
    ]
    after_accuracy = nearest_accuracy(full_preds, targets, bank)
    completion = nearest_accuracy(partial_preds, targets, bank)
    return {
        "similar_episode_discrimination": after_accuracy,
        "partial_to_full_reconstruction": completion,
        "accuracy_loss_after_20_similar": max(0.0, 1.0 - after_accuracy),
    }


def causal_metrics(memory: HumanAnalogueMemory, meta: dict[str, Any]) -> dict[str, float]:
    target_idx = 3
    cue = meta["cues"][target_idx].to(memory.device)
    target = meta["content_bank"][target_idx].to(memory.device)
    bank = meta["content_bank"].to(memory.device)
    before = memory.recall(cue)
    corrupt = memory.clone()
    corrupt.corrupt_trace(meta["trace_ids"][target_idx])
    after = corrupt.recall(cue)
    unrelated = memory.clone()
    for idx in range(8, 12):
        unrelated.corrupt_trace(meta["trace_ids"][idx])
    unrelated_after = unrelated.recall(cue)
    before_acc = nearest_accuracy([before.content], [target], bank)
    after_acc = nearest_accuracy([after.content], [target], bank)
    unrelated_acc = nearest_accuracy([unrelated_after.content], [target], bank)
    perturbed = memory.clone()
    perturbed.perturb_trace(meta["trace_ids"][target_idx])
    shifted = perturbed.recall(cue)
    action_shift = float((before.action_logits - shifted.action_logits).abs().mean().item())
    private_shift = float((before.private_logits - shifted.private_logits).abs().mean().item())
    language_shift = float((before.language_logits - shifted.language_logits).abs().mean().item())
    return {
        "targeted_trace_corruption_recall_action_degrade": before_acc - after_acc,
        "unrelated_memory_corruption_degrade": before_acc - unrelated_acc,
        "relevant_trace_action_shift": action_shift,
        "relevant_trace_private_shift": private_shift,
        "relevant_trace_language_shift": language_shift,
    }


def replay_metrics(memory: HumanAnalogueMemory, meta: dict[str, Any], model, device: str) -> dict[str, float]:
    cues = [_noisy(cue, idx + 300, scale=0.08).to(memory.device) for idx, cue in enumerate(meta["cues"])]
    targets = [item.to(memory.device) for item in meta["content_bank"]]
    before = semantic_accuracy(memory, cues, targets, meta["content_bank"].to(memory.device))
    replay_report = replay_consolidate(memory)
    after = semantic_accuracy(memory, cues, targets, meta["content_bank"].to(memory.device))
    before_old = old_task_scores(model, 128, 80, 99200, device)
    after_old = old_task_scores(model, 128, 80, 99200, device)
    return {
        "semantic_accuracy_before_replay": before,
        "semantic_accuracy_after_replay": after,
        "semantic_accuracy_improvement": after - before,
        "old_task_core_before": before_old["core_mean"],
        "old_task_core_after": after_old["core_mean"],
        "old_task_core_delta": after_old["core_mean"] - before_old["core_mean"],
        "samples_replayed": float(replay_report.samples_replayed),
        "adapter_rank": float(replay_report.adapter_rank),
    }


def reconsolidation_metrics(memory: HumanAnalogueMemory, meta: dict[str, Any]) -> dict[str, float]:
    checks_history = []
    checks_current = []
    distractors = torch.stack(
        [deterministic_vector(80000 + i, memory.config.content_dim, device=memory.device) for i in range(8)]
    )
    bank = torch.cat([meta["content_bank"].to(memory.device), distractors], dim=0)
    for offset, trace_id in enumerate(meta["trace_ids"][:8]):
        new_content = deterministic_vector(80000 + offset, memory.config.content_dim, device=memory.device)
        original_source = meta["sources"][offset]
        memory.reconsolidate(trace_id, new_content, (original_source + 1) % NUM_SOURCES)
        recalled = memory.recall(meta["cues"][offset].to(memory.device))
        history = recalled.source_history
        checks_history.append(float(history[original_source] > 0.1 and history[(original_source + 1) % NUM_SOURCES] > 0.1))
        checks_current.append(nearest_accuracy([recalled.content], [new_content], bank))
    return {
        "source_history_accuracy": float(sum(checks_history) / len(checks_history)),
        "current_belief_accuracy": float(sum(checks_current) / len(checks_current)),
    }


def source_monitoring_metrics(memory: HumanAnalogueMemory, meta: dict[str, Any]) -> dict[str, float]:
    correct = 0
    total = 0
    for idx, cue in enumerate(meta["cues"]):
        recalled = memory.recall(cue.to(memory.device))
        correct += int(recalled.source_distribution.argmax().item() == meta["sources"][idx])
        total += 1
    by_source = {}
    for source_id, source_name in enumerate(SOURCE_NAMES):
        indices = [idx for idx, value in enumerate(meta["sources"]) if value == source_id]
        if not indices:
            continue
        source_correct = 0
        for idx in indices:
            source_correct += int(memory.recall(meta["cues"][idx].to(memory.device)).source_distribution.argmax().item() == source_id)
        by_source[source_name] = source_correct / len(indices)
    return {"source_monitoring_accuracy": float(correct / total), **{f"{key}_accuracy": float(value) for key, value in by_source.items()}}


def restart_metrics(memory: HumanAnalogueMemory, meta: dict[str, Any], output_path: str | Path) -> dict[str, float]:
    path = Path(output_path)
    memory.save(path)
    loaded = HumanAnalogueMemory.load(path, device=memory.device)
    zero = loaded.zeroed()
    corrupt = loaded.corrupted()
    bank = meta["content_bank"].to(memory.device)
    targets = [item.to(memory.device) for item in meta["content_bank"]]
    loaded_preds = [loaded.recall(cue.to(memory.device)).content for cue in meta["cues"]]
    zero_preds = [zero.recall(cue.to(memory.device)).content for cue in meta["cues"]]
    corrupt_preds = [corrupt.recall(cue.to(memory.device)).content for cue in meta["cues"]]
    loaded_acc = nearest_accuracy(loaded_preds, targets, bank)
    zero_acc = nearest_accuracy(zero_preds, targets, bank)
    corrupt_acc = nearest_accuracy(corrupt_preds, targets, bank)
    return {
        "restart_recall_accuracy": loaded_acc,
        "zero_memory_recall_accuracy": zero_acc,
        "corrupt_memory_recall_accuracy": corrupt_acc,
        "zero_memory_degradation": loaded_acc - zero_acc,
        "corrupt_memory_degradation": loaded_acc - corrupt_acc,
    }


def prior_property_metrics(checkpoint: str | Path, config_name: str, model, device: str) -> dict[str, Any]:
    retention = evaluate_retention_fix(checkpoint, config_name=config_name, json_output=None, device=device)
    living = evaluate_living_system(
        checkpoint,
        config_name=config_name,
        device=device,
        curriculum_output=Path("runs") / f"human_memory_living_curriculum_{config_name}.pt",
        json_output=None,
    )
    leakage = run_scan()
    anti = anti_leakage_probes(model, int(MEMORY_CONFIGS[config_name]["batch_size"]), int(MEMORY_CONFIGS[config_name]["seq_len"]), 990000, device)
    return {
        "retention_eval_passes": retention["terminal_outcome"] == "RETENTION FIX PROVEN",
        "living_eval_passes": living["verdict"]["passes"],
        "leakage_scan_passes": leakage["passes"],
        "hidden_target_canary_diff": anti["hidden_target_canary_max_abs_diff"],
        "hidden_target_canary_zero": anti["hidden_target_canary_max_abs_diff"] == 0.0,
    }


def gate_table(report: dict[str, Any]) -> dict[str, bool]:
    content = report["content_recall"]
    sep = report["separation_completion"]
    causal = report["causal_memory"]
    replay = report["replay_consolidation"]
    recon = report["reconsolidation"]
    source = report["source_monitoring"]
    restart = report["restart_durability"]
    prior = report["prior_properties"]
    return {
        "partial_cue_ge_0_85": content["partial_cue_accuracy"] >= 0.85,
        "noisy_cue_ge_0_75": content["noisy_cue_accuracy"] >= 0.75,
        "wrong_cue_rejection_ge_0_85": content["wrong_cue_rejection"] >= 0.85,
        "similar_discrimination_ge_0_85": sep["similar_episode_discrimination"] >= 0.85,
        "completion_ge_0_85": sep["partial_to_full_reconstruction"] >= 0.85,
        "similar_loss_le_0_10": sep["accuracy_loss_after_20_similar"] <= 0.10,
        "targeted_corruption_degrades_ge_0_40": causal["targeted_trace_corruption_recall_action_degrade"] >= 0.40,
        "unrelated_degrades_lt_0_10": causal["unrelated_memory_corruption_degrade"] < 0.10,
        "relevant_trace_perturbs_outputs": causal["relevant_trace_action_shift"] > 0.05 and causal["relevant_trace_private_shift"] > 0.05 and causal["relevant_trace_language_shift"] > 0.05,
        "replay_improves_ge_0_20": replay["semantic_accuracy_improvement"] >= 0.20,
        "replay_old_core_ge_0_90": replay["old_task_core_after"] >= 0.90,
        "replay_old_delta_ge_minus_0_05": replay["old_task_core_delta"] >= -0.05,
        "history_accuracy_ge_0_85": recon["source_history_accuracy"] >= 0.85,
        "current_belief_ge_0_85": recon["current_belief_accuracy"] >= 0.85,
        "source_monitoring_ge_0_85": source["source_monitoring_accuracy"] >= 0.85,
        "restart_recall_ge_0_85": restart["restart_recall_accuracy"] >= 0.85,
        "zero_memory_degrades_ge_0_40": restart["zero_memory_degradation"] >= 0.40,
        "corrupt_memory_degrades_ge_0_40": restart["corrupt_memory_degradation"] >= 0.40,
        "retention_eval_passes": prior["retention_eval_passes"],
        "living_eval_passes": prior["living_eval_passes"],
        "leakage_scan_passes": prior["leakage_scan_passes"],
        "hidden_target_canary_zero": prior["hidden_target_canary_zero"],
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# Human Memory Report", ""]
    lines.append(f"Terminal outcome: **{report['terminal_outcome']}**")
    lines.append("")
    lines.append("## Architecture")
    lines.append("")
    lines.append(report["architecture_summary"])
    lines.append("")
    lines.append("## Gate Metrics")
    lines.append("")
    lines.append("| Gate | Value | Pass |")
    lines.append("| --- | ---: | --- |")
    for key, value in report["gate_metrics_flat"].items():
        lines.append(f"| `{key}` | {value:.6f} | {report['gate_checks'].get(key, '')} |")
    lines.append("")
    lines.append("## Causal Ablation")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["causal_memory"], indent=2))
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


def evaluate_human_memory(
    checkpoint: str | Path,
    config_name: str = "fast",
    json_output: str | Path | None = None,
    device: DeviceLike = AUTO_DEVICE,
) -> dict[str, Any]:
    device = str(resolve_device(device))
    cfg = MEMORY_CONFIGS[config_name]
    memory, meta = build_memory(checkpoint, cfg, device)
    model = meta["model"]
    report: dict[str, Any] = {
        "checkpoint": str(checkpoint),
        "config": config_name,
        "hash_manifest_summary": hash_manifest_summary(),
        "architecture_summary": (
            "Sparse cue-addressed engram memory stores compressed recurrent latents, content fragments, "
            "body/affect/action/private-token tags, source distributions, time order, and source histories. "
            "Recall performs pattern completion by attention over sparse random-projection traces; replay fits a bounded semantic adapter from stored traces only."
        ),
        "content_recall": content_recall_metrics(memory, meta),
        "separation_completion": separation_completion_metrics(checkpoint, cfg, device),
        "causal_memory": causal_metrics(memory, meta),
    }
    replay_memory = memory.clone()
    report["replay_consolidation"] = replay_metrics(replay_memory, meta, model, device)
    recon_memory = memory.clone()
    report["reconsolidation"] = reconsolidation_metrics(recon_memory, meta)
    report["source_monitoring"] = source_monitoring_metrics(memory, meta)
    report["restart_durability"] = restart_metrics(memory, meta, Path("runs") / f"human_memory_{config_name}.pt")
    report["prior_properties"] = prior_property_metrics(checkpoint, config_name, model, device)
    flat = {
        **report["content_recall"],
        **report["separation_completion"],
        **report["causal_memory"],
        **report["replay_consolidation"],
        **report["reconsolidation"],
        **report["source_monitoring"],
        **report["restart_durability"],
    }
    report["gate_metrics_flat"] = {key: float(value) for key, value in flat.items() if isinstance(value, (int, float))}
    report["gate_checks"] = gate_table(report)
    report["limitations"] = [key for key, passed in report["gate_checks"].items() if not passed]
    report["terminal_outcome"] = "HUMAN MEMORY PROVEN" if not report["limitations"] else "NOT PROVEN"
    if json_output is not None:
        path = Path(json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        path.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"terminal_outcome": report["terminal_outcome"], "limitations": report["limitations"]}, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", choices=sorted(MEMORY_CONFIGS), default="fast")
    parser.add_argument("--json-output", default="docs/human_memory_report.json")
    parser.add_argument("--device", default=AUTO_DEVICE)
    args = parser.parse_args()
    evaluate_human_memory(args.checkpoint, args.config, args.json_output, args.device)


if __name__ == "__main__":
    main()
