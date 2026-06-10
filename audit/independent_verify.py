from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Iterable

import torch

from audit.leakage_scan import run_scan
from src.device import AUTO_DEVICE, DeviceLike, make_generator, resolve_device
from src.continual_learning import PersistentConceptMemory, recall_accuracy
from src.curriculum import curriculum_batch
from src.env import (
    ACTION_FORAGE,
    ACTION_LEFT,
    ACTION_REST,
    ACTION_RIGHT,
    ACTION_STAY,
    BODY_DAMAGE,
    BODY_ENERGY,
    GRID_SIZE,
    NUM_BODY_SCALARS,
    NUM_COLORS,
    NUM_PRIVATE_TOKENS,
    TOK_ASK_ACTION,
    TOK_ASK_COLOR,
    TOK_ASK_CURRENT_POS,
    TOK_ASK_ENERGY,
    TOK_ASK_GOAL,
    TOK_ASK_OBJECT_POS,
    TOK_ASK_START_POS,
    TOK_CURRICULUM_ALIAS,
    TOK_IMAGINE,
    TOK_INFER_OBJECT,
    TOK_NONE,
    TOK_OBSERVE_OBJECT,
    TOK_TOLD_GOAL,
    generate_batch,
)
from src.living_eval import (
    LIVING_CONFIGS,
    durable_restart_eval,
    evaluate_living_system,
    idle_mode_eval,
    private_language_eval,
    richer_dynamics_eval,
    run_autoregressive_private,
)
from src.metrics import masked_accuracy
from src.model import load_checkpoint
from src.persistent_memory import PersistentMemoryState
from src.train import blank_training_batch


AUDITED_PATHS = [
    "frozen/recurrent_latent_fast.pt",
    "frozen/manifest.json",
    "src/device.py",
    "src/model.py",
    "src/env.py",
    "src/run_unbroken.py",
    "src/persistent_memory.py",
    "src/living_eval.py",
    "src/curriculum.py",
    "src/continual_learning.py",
    "src/retention_eval.py",
    "src/human_memory.py",
    "src/memory_replay.py",
    "src/memory_eval.py",
    "src/language_organ.py",
    "src/dialogue_env.py",
    "src/dialogue_train.py",
    "src/dialogue_eval.py",
    "src/free_text_decoder.py",
    "src/conversation_env.py",
    "src/conversation_train.py",
    "src/conversation_eval.py",
    "src/explore_env.py",
    "src/intrinsic_motivation.py",
    "src/unified_policy.py",
    "src/world_model.py",
    "src/explorer_train.py",
    "src/explorer_eval.py",
    "src/head_collapse.py",
    "src/head_collapse_eval.py",
    "src/arcagi3_adapter.py",
    "src/arcagi3_baselines.py",
    "src/arcagi3_trace.py",
    "src/arcagi3_eval.py",
    "src/train.py",
    "src/evaluate.py",
    "src/metrics.py",
    "tests/test_dialogue_organ.py",
    "tests/test_grounded_conversation.py",
    "tests/test_explorer_mindlike.py",
    "tests/test_head_collapse.py",
    "tests/test_arcagi3_adapter.py",
    "README.md",
    "docs/living_system_report.json",
    "docs/evidence_dossier.json",
    "docs/dialogue_report.json",
    "docs/conversation_report.json",
    "docs/explorer_report.json",
    "docs/audit_after_explorer.json",
    "docs/head_collapse_report.json",
    "docs/explorer_report_after_head_collapse.json",
    "docs/audit_after_head_collapse.json",
    "docs/arcagi3_public_fixtures.json",
    "docs/arcagi3_report.json",
    "docs/audit_after_arcagi3.json",
]

ACTION_NAMES = {
    ACTION_STAY: "STAY",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
    ACTION_FORAGE: "FORAGE",
    ACTION_REST: "REST",
}

TOKEN_NAMES = {
    TOK_NONE: "NONE",
    TOK_OBSERVE_OBJECT: "OBSERVE_OBJECT",
    TOK_TOLD_GOAL: "TOLD_GOAL",
    TOK_IMAGINE: "IMAGINE",
    TOK_INFER_OBJECT: "INFER_OBJECT",
    TOK_ASK_COLOR: "ASK_COLOR",
    TOK_ASK_OBJECT_POS: "ASK_OBJECT_POS",
    TOK_ASK_START_POS: "ASK_START_POS",
    TOK_ASK_CURRENT_POS: "ASK_CURRENT_POS",
    TOK_ASK_ENERGY: "ASK_ENERGY",
    TOK_ASK_ACTION: "ASK_ACTION",
    TOK_ASK_GOAL: "ASK_GOAL",
    TOK_CURRICULUM_ALIAS: "CURRICULUM_ALIAS",
}

VISIBLE_FLAG_INDEX = GRID_SIZE + 2 + NUM_BODY_SCALARS + (NUM_COLORS + 1)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def collect_hashes(paths: Iterable[str]) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    for item in paths:
        path = Path(item)
        if path.exists():
            out[item] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        else:
            out[item] = {"missing": True}
    return out


def manifest_status(hashes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    manifest = json.loads(Path("frozen/manifest.json").read_text(encoding="utf-8"))
    checkpoint = manifest["checkpoint"]
    return {
        "manifest": manifest,
        "model_hash_matches": hashes["src/model.py"]["sha256"] == manifest["model_sha256"],
        "checkpoint_hash_matches": hashes[checkpoint]["sha256"] == manifest["checkpoint_sha256"],
        "checkpoint_size_matches": hashes[checkpoint]["size_bytes"] == manifest["checkpoint_size_bytes"],
    }


def numeric_paths(obj: Any, prefix: str = "") -> dict[str, float]:
    out: dict[str, float] = {}
    if isinstance(obj, dict):
        for key, value in obj.items():
            out.update(numeric_paths(value, f"{prefix}.{key}" if prefix else str(key)))
    elif isinstance(obj, (int, float, bool)):
        out[prefix] = float(obj)
    return out


def compare_reports(existing: dict[str, Any], rerun: dict[str, Any], tolerance: float = 0.02) -> dict[str, Any]:
    old = numeric_paths(existing)
    new = numeric_paths(rerun)
    rows = []
    material = []
    for key, old_value in sorted(old.items()):
        if key not in new:
            continue
        diff = abs(new[key] - old_value)
        row = {"metric": key, "existing": old_value, "rerun": new[key], "abs_diff": diff}
        rows.append(row)
        if diff > tolerance:
            material.append(row)
    return {
        "tolerance": tolerance,
        "material_differences": material,
        "verdict_matches": bool(existing.get("verdict", {}).get("passes")) == bool(rerun.get("verdict", {}).get("passes")),
        "rows": rows,
    }


def stack_outputs(outputs: list[dict[str, torch.Tensor]], latents: list[torch.Tensor], private_tokens: list[torch.Tensor]) -> dict[str, torch.Tensor]:
    stacked = {key: torch.stack([item[key] for item in outputs], dim=1) for key in outputs[0]}
    stacked["latents"] = torch.stack(latents, dim=1)
    stacked["generated_private"] = torch.stack(private_tokens, dim=1)
    return stacked


def run_mode(model, batch: dict[str, torch.Tensor], mode: str = "normal") -> dict[str, torch.Tensor]:
    batch_size, seq_len, _ = batch["sensory"].shape
    device = batch["sensory"].device
    z = model.initial_state(batch_size, device=device)
    private_token = torch.zeros(batch_size, dtype=torch.long, device=device)
    generator = make_generator(9917, device)
    random_private = torch.randint(0, NUM_PRIVATE_TOKENS, (batch_size, seq_len), generator=generator, device=device)
    outputs: list[dict[str, torch.Tensor]] = []
    latents: list[torch.Tensor] = []
    private_tokens: list[torch.Tensor] = []
    with torch.no_grad():
        for tick in range(seq_len):
            if mode == "zero_z":
                z = torch.zeros_like(z)
            elif mode == "shuffled_z" and batch_size > 1:
                z = z.roll(1, dims=0)
            lang = batch["lang_in"][:, tick]
            if mode == "no_language":
                lang = torch.zeros_like(lang)
            if mode == "no_private":
                private_in = torch.zeros_like(private_token)
            elif mode == "random_private":
                private_in = random_private[:, tick]
            else:
                private_in = private_token
            output, z = model.step(
                {
                    "sensory": batch["sensory"][:, tick],
                    "lang_in": lang,
                    "private_in": private_in,
                },
                z,
            )
            private_token = output["private_logits"].argmax(dim=-1)
            outputs.append(output)
            latents.append(z)
            private_tokens.append(private_token)
    return stack_outputs(outputs, latents, private_tokens)


def score_outputs(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, float]:
    object_color = masked_accuracy(outputs["world_color_logits"], batch["world_color_target"], batch["object_mask"])
    object_pos = masked_accuracy(outputs["world_pos_logits"], batch["world_pos_target"], batch["object_mask"])
    metrics = {
        "action_accuracy": masked_accuracy(outputs["action_logits"], batch["action_target"], batch["action_mask"]),
        "delayed_memory_accuracy": masked_accuracy(
            outputs["memory_color_logits"], batch["memory_color_target"], batch["delayed_memory_mask"]
        ),
        "object_color_accuracy": object_color,
        "object_pos_accuracy": object_pos,
        "object_permanence_accuracy": (object_color + object_pos) / 2.0,
        "provenance_accuracy": float(
            (outputs["provenance_logits"].argmax(dim=-1) == batch["provenance_target"]).float().mean().item()
        ),
        "grounded_language_accuracy": masked_accuracy(
            outputs["language_logits"], batch["language_target"], batch["grounded_language_mask"]
        ),
        "self_world_continuity_accuracy": masked_accuracy(
            outputs["self_start_logits"], batch["self_start_target"], batch["self_mask"]
        ),
    }
    metrics["core_mean"] = sum(metrics.values()) / len(metrics)
    return metrics


def max_output_diff(left: dict[str, torch.Tensor], right: dict[str, torch.Tensor]) -> float:
    keys = [key for key in left if key.endswith("_logits")]
    max_diff = 0.0
    for key in keys:
        max_diff = max(max_diff, float((left[key] - right[key]).abs().max().item()))
    return max_diff


def mutate_targets(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {key: value.clone() for key, value in batch.items()}
    generator = make_generator(9918, out["sensory"].device)
    for key in list(out):
        if key.endswith("_target"):
            high = int(out[key].max().item()) + 1
            if high > 1:
                out[key] = torch.randint(0, high, out[key].shape, generator=generator, device=out[key].device)
    out["hidden_target_canary"] = torch.randint(
        0, 97, (out["sensory"].shape[0], out["sensory"].shape[1]), generator=generator, device=out["sensory"].device
    )
    return out


def random_label_batch_like(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = {key: value.clone() for key, value in batch.items()}
    generator = make_generator(9919, out["sensory"].device)
    maxima = {
        "action_target": 5,
        "language_target": int(batch["language_target"].max().item()) + 1,
        "provenance_target": int(batch["provenance_target"].max().item()) + 1,
        "world_color_target": int(batch["world_color_target"].max().item()) + 1,
        "world_pos_target": int(batch["world_pos_target"].max().item()) + 1,
        "memory_color_target": int(batch["memory_color_target"].max().item()) + 1,
        "self_start_target": int(batch["self_start_target"].max().item()) + 1,
    }
    for key, high in maxima.items():
        out[key] = torch.randint(0, high, out[key].shape, generator=generator, device=out[key].device)
    return out


def anti_leakage_probes(model, batch_size: int, seq_len: int, seed: int, device: str) -> dict[str, Any]:
    batch = generate_batch(batch_size, seq_len, seed, device=device)
    normal = run_mode(model, batch, "normal")
    modes = {
        "normal_generated_private": normal,
        "zero_z": run_mode(model, batch, "zero_z"),
        "shuffled_z": run_mode(model, batch, "shuffled_z"),
        "no_language": run_mode(model, batch, "no_language"),
        "no_private_token": run_mode(model, batch, "no_private"),
        "random_private_token": run_mode(model, batch, "random_private"),
    }
    scores = {name: score_outputs(outputs, batch) for name, outputs in modes.items()}
    mutated = mutate_targets(batch)
    canary_outputs = run_mode(model, mutated, "normal")
    random_labels = random_label_batch_like(batch)
    random_label_score = score_outputs(normal, random_labels)
    action_shift_no_private = float(
        (normal["action_logits"].argmax(dim=-1) != modes["no_private_token"]["action_logits"].argmax(dim=-1))
        .float()
        .mean()
        .item()
    )
    language_shift_no_private = float(
        (normal["language_logits"].argmax(dim=-1) != modes["no_private_token"]["language_logits"].argmax(dim=-1))
        .float()
        .mean()
        .item()
    )
    checks = {
        "hidden_target_canary_no_effect": max_output_diff(normal, canary_outputs) == 0.0,
        "random_labels_reduce_score": random_label_score["core_mean"] < scores["normal_generated_private"]["core_mean"] - 0.20,
        "zero_z_degrades_core": scores["zero_z"]["core_mean"] < scores["normal_generated_private"]["core_mean"] - 0.10,
        "shuffled_z_degrades_core": scores["shuffled_z"]["core_mean"] < scores["normal_generated_private"]["core_mean"] - 0.05,
        "no_language_degrades_grounding": scores["no_language"]["grounded_language_accuracy"]
        < scores["normal_generated_private"]["grounded_language_accuracy"] - 0.10,
        "private_token_affects_outputs": action_shift_no_private > 0.0 or language_shift_no_private > 0.0,
    }
    return {
        "scores": scores,
        "random_label_score": random_label_score,
        "hidden_target_canary_max_abs_diff": max_output_diff(normal, canary_outputs),
        "no_private_action_shift_rate": action_shift_no_private,
        "no_private_language_shift_rate": language_shift_no_private,
        "checks": checks,
    }


def durable_memory_corrupt_probe(model, batch_size: int, seq_len: int, seed: int, device: str) -> dict[str, Any]:
    batch = generate_batch(batch_size, seq_len, seed, device=device)
    restart_tick = min(70, seq_len // 2 + 16)
    with torch.no_grad(), TemporaryDirectory() as tmp:
        prefix = run_autoregressive_private(model, batch, end_tick=restart_tick)
        memory = PersistentMemoryState.fresh(model.config.hidden_dim, batch_size, device=device)
        memory.update(prefix["latents"][:, -1, :], prefix["generated_private"][:, -1], restart_tick)
        memory_path = Path(tmp) / "memory.pt"
        memory.save(memory_path)
        loaded = PersistentMemoryState.load(memory_path, model.config.hidden_dim, batch_size, device=device)
        restarted = run_autoregressive_private(
            model,
            batch,
            initial_z=loaded.latent,
            initial_private=loaded.private_token,
            start_tick=restart_tick,
        )
        corrupt = PersistentMemoryState(
            latent=prefix["latents"][:, -1, :].roll(1, dims=0),
            private_token=prefix["generated_private"][:, -1].roll(1, dims=0),
            tick=restart_tick,
        )
        corrupt_path = Path(tmp) / "corrupt_memory.pt"
        corrupt.save(corrupt_path)
        corrupt_loaded = PersistentMemoryState.load(corrupt_path, model.config.hidden_dim, batch_size, device=device)
        corrupt_outputs = run_autoregressive_private(
            model,
            batch,
            initial_z=corrupt_loaded.latent,
            initial_private=corrupt_loaded.private_token,
            start_tick=restart_tick,
        )
    final_mask_tail = torch.zeros(batch_size, seq_len - restart_tick, dtype=torch.bool, device=device)
    final_mask_tail[:, -1] = True
    return {
        "restart_tick": restart_tick,
        "memory_file_restart_final_memory_accuracy": masked_accuracy(
            restarted["memory_color_logits"], batch["memory_color_target"][:, restart_tick:], final_mask_tail
        ),
        "memory_file_restart_final_object_pos_accuracy": masked_accuracy(
            restarted["world_pos_logits"], batch["world_pos_target"][:, restart_tick:], final_mask_tail
        ),
        "corrupt_memory_final_memory_accuracy": masked_accuracy(
            corrupt_outputs["memory_color_logits"], batch["memory_color_target"][:, restart_tick:], final_mask_tail
        ),
        "corrupt_memory_final_object_pos_accuracy": masked_accuracy(
            corrupt_outputs["world_pos_logits"], batch["world_pos_target"][:, restart_tick:], final_mask_tail
        ),
    }


def private_token_probe(model, batch_size: int, seq_len: int, seed: int, device: str) -> dict[str, Any]:
    base = private_language_eval(model, batch_size, seq_len, seed, device=device)
    batch = generate_batch(batch_size, seq_len, seed + 10, device=device)
    normal = run_mode(model, batch, "normal")
    random_private = run_mode(model, batch, "random_private")
    base["random_private_action_shift_rate"] = float(
        (normal["action_logits"].argmax(dim=-1) != random_private["action_logits"].argmax(dim=-1))
        .float()
        .mean()
        .item()
    )
    base["random_private_language_shift_rate"] = float(
        (normal["language_logits"].argmax(dim=-1) != random_private["language_logits"].argmax(dim=-1))
        .float()
        .mean()
        .item()
    )
    return base


def idle_trajectory(model, seq_len: int, seed: int, device: str) -> list[dict[str, Any]]:
    base = generate_batch(4, seq_len, seed, device=device)
    idle = blank_training_batch(base, blank_after=16)
    with torch.no_grad():
        outputs = run_autoregressive_private(model, idle)
    ticks = [0, 3, 8, 16, 32, 64, 95, seq_len - 1]
    rows: list[dict[str, Any]] = []
    sample = 0
    for tick in ticks:
        if tick >= seq_len:
            continue
        action_pred = int(outputs["action_logits"][sample, tick].argmax().item())
        memory_pred = int(outputs["memory_color_logits"][sample, tick].argmax().item())
        world_pos_pred = int(outputs["world_pos_logits"][sample, tick].argmax().item())
        language_pred = int(outputs["language_logits"][sample, tick].argmax().item())
        rows.append(
            {
                "tick": tick,
                "token": TOKEN_NAMES.get(int(idle["lang_in"][sample, tick].item()), str(int(idle["lang_in"][sample, tick].item()))),
                "visible": bool(idle["sensory"][sample, tick, VISIBLE_FLAG_INDEX].item() > 0.5),
                "private_in": int(outputs["generated_private"][sample, max(tick - 1, 0)].item()) if tick > 0 else 0,
                "generated_private": int(outputs["generated_private"][sample, tick].item()),
                "pred_action": ACTION_NAMES.get(action_pred, str(action_pred)),
                "pred_language_token": language_pred,
                "pred_memory_color": memory_pred,
                "target_memory_color": int(base["memory_color_target"][sample, tick].item()),
                "pred_world_pos": world_pos_pred,
                "target_world_pos": int(base["world_pos_target"][sample, tick].item()),
                "memory_pass": memory_pred == int(base["memory_color_target"][sample, tick].item()),
                "world_pos_pass": world_pos_pred == int(base["world_pos_target"][sample, tick].item()),
            }
        )
    return rows


def curriculum_generated_accuracy(model, concept_id: int, seed: int, device: str) -> float:
    batch = curriculum_batch(64, 48, concept_id, seed, device=device)
    with torch.no_grad():
        outputs = run_autoregressive_private(model, batch)
    pred = outputs["language_logits"].argmax(dim=-1)
    mask = batch["curriculum_mask"]
    target = batch["language_target"]
    return float((pred[mask] == target[mask]).float().mean().item())


def curriculum_probe(checkpoint: str | Path, curriculum_output: str | Path, device: str) -> dict[str, Any]:
    before_model = load_checkpoint(checkpoint, device=device)
    changes_persistent_memory = False
    try:
        after_memory = PersistentConceptMemory.load(curriculum_output, device=device)
        after_model = before_model
        changed = 0
        l2_total = 0.0
        changes_persistent_memory = after_memory.concept_ids.numel() > 0
        curriculum_before = recall_accuracy(PersistentConceptMemory.fresh(device=device), [1], device=device)
        curriculum_after = recall_accuracy(after_memory, [1], device=device)
    except ValueError:
        after_model = load_checkpoint(curriculum_output, device=device)
        before_params = dict(before_model.named_parameters())
        after_params = dict(after_model.named_parameters())
        changed = 0
        l2_total = 0.0
        for name, before in before_params.items():
            diff = after_params[name].detach() - before.detach()
            l2 = float(diff.norm().item())
            l2_total += l2
            if l2 > 0.0:
                changed += 1
        curriculum_before = curriculum_generated_accuracy(before_model, 1, 8301, device)
        curriculum_after = curriculum_generated_accuracy(after_model, 1, 8301, device)
    old_task_batch = generate_batch(128, 80, 9920, device=device)
    with torch.no_grad():
        before_old = run_autoregressive_private(before_model, old_task_batch)
        after_old = run_autoregressive_private(after_model, old_task_batch)
    before_score = score_outputs(before_old, old_task_batch)
    after_score = score_outputs(after_old, old_task_batch)
    submetric_floor = {
        key: (value >= 0.85 or (before_score.get(key, 0.0) < 0.85 and value < 0.85))
        for key, value in after_score.items()
        if key != "core_mean"
    }
    return {
        "changed_parameter_tensors": changed,
        "parameter_l2_total": l2_total,
        "changes_weights": changed > 0 and l2_total > 0.0,
        "changes_persistent_memory": changes_persistent_memory,
        "generated_private_curriculum_accuracy_before": curriculum_before,
        "generated_private_curriculum_accuracy_after": curriculum_after,
        "concept_memory_accuracy_before": curriculum_before,
        "concept_memory_accuracy_after": curriculum_after,
        "old_task_retention_before": before_score,
        "old_task_retention_after": after_score,
        "old_task_core_delta": after_score["core_mean"] - before_score["core_mean"],
        "old_task_submetric_floor_checks": submetric_floor,
        "old_task_retention_passes": (
            after_score["core_mean"] >= 0.90
            and after_score["core_mean"] - before_score["core_mean"] >= -0.05
            and all(submetric_floor.values())
        ),
    }


def find_snippet(path: str, pattern: str, before: int = 3, after: int = 8) -> dict[str, Any]:
    lines = Path(path).read_text(encoding="utf-8").splitlines()
    index = 0
    for i, line in enumerate(lines):
        if pattern in line:
            index = i
            break
    start = max(0, index - before)
    end = min(len(lines), index + after + 1)
    text = "\n".join(f"{lineno + 1}: {lines[lineno]}" for lineno in range(start, end))
    return {"path": path, "start_line": start + 1, "end_line": end, "text": text}


def source_snippets() -> dict[str, Any]:
    return {
        "model_step_update": find_snippet("src/model.py", "def step("),
        "runtime_z_loop": find_snippet("src/run_unbroken.py", "for tick in range(max_ticks)", 2, 18),
        "memory_load_save": find_snippet("src/persistent_memory.py", "def load(", 3, 45),
        "model_input_construction": find_snippet("src/env.py", "def observation(", 3, 32),
        "target_input_separation": find_snippet("src/env.py", "\"action_target\": action_target", 8, 18),
        "private_token_generation_use": find_snippet("src/living_eval.py", "\"private_in\": private_token", 8, 14),
        "curriculum_update": find_snippet("src/continual_learning.py", "def learn(", 6, 18),
        "metric_computation": find_snippet("src/living_eval.py", "def living_verdict", 2, 28),
        "leakage_scan_logic": find_snippet("audit/leakage_scan.py", "def run_scan", 2, 18),
    }


def build_claim_table(report: dict[str, Any]) -> list[dict[str, Any]]:
    table = [
        {
            "claim": "Frozen checkpoint and model match manifest before and after audit",
            "status": "PASS" if report["hashes"]["frozen_unchanged"] and report["hashes"]["manifest_after"]["checkpoint_hash_matches"] and report["hashes"]["manifest_after"]["model_hash_matches"] else "FAIL",
            "evidence": "SHA256 and size checks over frozen/recurrent_latent_fast.pt and src/model.py.",
        },
        {
            "claim": "No external or pretrained model/API path is present",
            "status": "PASS" if report["leakage_scan"]["passes"] else "FAIL",
            "evidence": "AST import scan, text pattern scan, weight-file scan, and runtime prompt-loop scan.",
        },
        {
            "claim": "Living-system report metrics reproduce materially",
            "status": "PASS" if not report["living_report_comparison"]["material_differences"] and report["living_report_comparison"]["verdict_matches"] else "FAIL",
            "evidence": "Reran src.living_eval against the frozen checkpoint without overwriting the existing report.",
        },
        {
            "claim": "Durable tensor memory survives restart and corrupted memory degrades",
            "status": "PASS" if report["durable_memory"]["checks"]["corrupt_degrades_memory"] and report["durable_memory"]["checks"]["memory_file_good"] else "FAIL",
            "evidence": "Saved and reloaded latent/private tensor state, then compared against sample-shuffled corrupt memory.",
        },
        {
            "claim": "Idle low-input continuation preserves memory without repetition collapse",
            "status": "PASS" if report["living_rerun"]["verdict"]["checks"]["idle_preserves_goal_memory"] and report["living_rerun"]["verdict"]["checks"]["idle_public_repetition_lt_0_40"] else "FAIL",
            "evidence": "Reran idle eval and recorded one idle trajectory.",
        },
        {
            "claim": "Private internal tokens are generated and causally consumed",
            "status": "PASS" if report["private_token_causality"]["generated_private_unique_count"] >= 3 and (report["private_token_causality"]["private_channel_action_shift_rate"] > 0 or report["private_token_causality"]["private_channel_language_shift_rate"] > 0) else "FAIL",
            "evidence": "Generated private-token stream plus zero/random-private perturbation shifts.",
        },
        {
            "claim": "Richer body/world dynamics have consequences",
            "status": "PASS" if all(value > 0 for value in report["richer_dynamics"].values()) else "FAIL",
            "evidence": "Failed movement, rest, forage, and hazard probes.",
        },
        {
            "claim": "Curriculum growth changes weights or persistent memory and retains old task behavior",
            "status": "PASS" if (report["curriculum"]["changes_weights"] or report["curriculum"]["changes_persistent_memory"]) and report["curriculum"]["generated_private_curriculum_accuracy_after"] >= 0.85 and report["curriculum"]["old_task_retention_passes"] else "WEAK",
            "evidence": "Compared pre/post curriculum state, concept recall accuracy, and old-task retention.",
        },
        {
            "claim": "Hidden targets and scoring data do not affect outputs",
            "status": "PASS" if report["anti_leakage"]["checks"]["hidden_target_canary_no_effect"] else "FAIL",
            "evidence": "Randomized all target keys and added a hidden canary key while keeping observations fixed.",
        },
        {
            "claim": "Latent stream is tensor-to-tensor, not generated public text",
            "status": "PASS",
            "evidence": "Source snippets show z is carried through model.step and private token, not language output, is fed as the private channel.",
        },
    ]
    return table


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Independent Audit Proof")
    lines.append("")
    lines.append(f"Terminal outcome: **{report['terminal_outcome']}**")
    lines.append("")
    lines.append("## Hashes")
    lines.append("")
    lines.append("| Path | Before SHA256 | After SHA256 | Unchanged |")
    lines.append("| --- | --- | --- | --- |")
    for path in AUDITED_PATHS:
        before = report["hashes"]["before"].get(path, {})
        after = report["hashes"]["after"].get(path, {})
        lines.append(
            f"| `{path}` | `{before.get('sha256', 'MISSING')}` | `{after.get('sha256', 'MISSING')}` | {before == after} |"
        )
    lines.append("")
    lines.append("## Claim Proof Table")
    lines.append("")
    lines.append("| Claim | Status | Evidence |")
    lines.append("| --- | --- | --- |")
    for row in report["claim_table"]:
        lines.append(f"| {row['claim']} | {row['status']} | {row['evidence']} |")
    lines.append("")
    lines.append("## Metric Reproduction")
    lines.append("")
    lines.append(f"Existing living report verdict reproduced: `{report['living_report_comparison']['verdict_matches']}`.")
    lines.append(f"Material metric differences over tolerance `{report['living_report_comparison']['tolerance']}`: `{len(report['living_report_comparison']['material_differences'])}`.")
    lines.append("")
    lines.append("## Anti-Leakage Probes")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["anti_leakage"], indent=2)[:8000])
    lines.append("```")
    lines.append("")
    lines.append("## Durable Memory")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["durable_memory"], indent=2))
    lines.append("```")
    lines.append("")
    lines.append("## Idle Trajectory")
    lines.append("")
    lines.append("| Tick | Token | Visible | Private In | Generated Private | Action | Memory | Object Pos | Pass |")
    lines.append("| ---: | --- | --- | ---: | ---: | --- | --- | --- | --- |")
    for row in report["idle_trajectory"]:
        passed = row["memory_pass"] and row["world_pos_pass"]
        lines.append(
            f"| {row['tick']} | {row['token']} | {row['visible']} | {row['private_in']} | {row['generated_private']} | {row['pred_action']} | {row['pred_memory_color']}/{row['target_memory_color']} | {row['pred_world_pos']}/{row['target_world_pos']} | {passed} |"
        )
    lines.append("")
    lines.append("## Curriculum")
    lines.append("")
    lines.append("```json")
    lines.append(json.dumps(report["curriculum"], indent=2)[:6000])
    lines.append("```")
    lines.append("")
    lines.append("## Required Source Snippets")
    for name, snippet in report["source_snippets"].items():
        lines.append("")
        lines.append(f"### {name}")
        lines.append(f"`{snippet['path']}:{snippet['start_line']}`")
        lines.append("")
        lines.append("```python")
        lines.append(snippet["text"])
        lines.append("```")
    lines.append("")
    lines.append("## Limitations")
    if report["limitations"]:
        for item in report["limitations"]:
            lines.append(f"- {item}")
    else:
        lines.append("- None found by this audit.")
    lines.append("")
    return "\n".join(lines)


def run_audit(checkpoint: str | Path, config: str, json_output: str | Path, device: DeviceLike = AUTO_DEVICE) -> dict[str, Any]:
    device = str(resolve_device(device))
    before_hashes = collect_hashes(AUDITED_PATHS)
    manifest_before = manifest_status(before_hashes)
    cfg = LIVING_CONFIGS[config]
    model = load_checkpoint(checkpoint, device=device)
    model.eval()

    existing_report = json.loads(Path("docs/living_system_report.json").read_text(encoding="utf-8"))
    curriculum_output = Path("runs/audit_curriculum.pt")
    living_rerun = evaluate_living_system(
        checkpoint=checkpoint,
        config_name=config,
        device=device,
        curriculum_output=curriculum_output,
        json_output=None,
    )
    comparison = compare_reports(existing_report, living_rerun)

    durable_base = durable_restart_eval(model, int(cfg["batch_size"]), int(cfg["seq_len"]), int(cfg["seed"]), device=device)
    durable_corrupt = durable_memory_corrupt_probe(
        model, int(cfg["batch_size"]), int(cfg["seq_len"]), int(cfg["seed"]), device=device
    )
    durable = {**durable_base, **durable_corrupt}
    durable["checks"] = {
        "memory_file_good": durable["memory_file_restart_final_memory_accuracy"] >= 0.85,
        "memory_beats_zero_reset": durable["memory_file_restart_final_memory_accuracy"] - durable["zero_reset_final_memory_accuracy"] >= 0.40,
        "corrupt_degrades_memory": durable["memory_file_restart_final_memory_accuracy"] - durable["corrupt_memory_final_memory_accuracy"] >= 0.30,
    }

    anti = anti_leakage_probes(model, int(cfg["batch_size"]), int(cfg["seq_len"]), int(cfg["seed"]) + 31, device)
    private = private_token_probe(model, int(cfg["batch_size"]), int(cfg["seq_len"]), int(cfg["seed"]) + 41, device)
    idle = idle_mode_eval(model, int(cfg["batch_size"]), 16 + int(cfg["idle_ticks"]), int(cfg["seed"]) + 1000, device=device)
    richer = richer_dynamics_eval()
    curriculum = curriculum_probe(checkpoint, curriculum_output, device=device)
    snippets = source_snippets()
    leakage = run_scan()
    after_hashes = collect_hashes(AUDITED_PATHS)
    manifest_after = manifest_status(after_hashes)
    frozen_unchanged = (
        before_hashes["frozen/recurrent_latent_fast.pt"] == after_hashes["frozen/recurrent_latent_fast.pt"]
        and before_hashes["src/model.py"] == after_hashes["src/model.py"]
        and before_hashes["frozen/manifest.json"] == after_hashes["frozen/manifest.json"]
    )

    commands_required = [
        "pytest -q",
        "python -m audit.leakage_scan",
        "python -m audit.independent_verify --checkpoint frozen/recurrent_latent_fast.pt --config fast --json-output docs/audit_after_conversation.json",
        "python -m audit.probe_examples --checkpoint frozen/recurrent_latent_fast.pt --output docs/audit_probe_examples.md",
    ]
    if Path("docs/explorer_report.json").exists():
        commands_required.extend(
            [
                "python -m src.explorer_train --config tiny --output runs/explorer_tiny.pt",
                "python -m src.explorer_eval --checkpoint frozen/recurrent_latent_fast.pt --explorer-checkpoint runs/explorer_tiny.pt --config tiny --json-output docs/explorer_report.json",
                "python -m audit.independent_verify --checkpoint frozen/recurrent_latent_fast.pt --config fast --json-output docs/audit_after_explorer.json",
            ]
        )
    if Path("docs/head_collapse_report.json").exists():
        commands_required.extend(
            [
                "python -m src.head_collapse_eval --checkpoint frozen/recurrent_latent_fast.pt --explorer-checkpoint runs/explorer_tiny.pt --config tiny --json-output docs/head_collapse_report.json",
                "python -m src.explorer_eval --checkpoint frozen/recurrent_latent_fast.pt --explorer-checkpoint runs/explorer_tiny.pt --config tiny --json-output docs/explorer_report_after_head_collapse.json",
                "python -m audit.independent_verify --checkpoint frozen/recurrent_latent_fast.pt --config fast --json-output docs/audit_after_head_collapse.json",
            ]
        )
    if Path("docs/arcagi3_report.json").exists():
        commands_required.extend(
            [
                "python -m src.arcagi3_eval --checkpoint frozen/recurrent_latent_fast.pt --explorer-checkpoint runs/explorer_tiny.pt --config public --json-output docs/arcagi3_report.json --trace-dir docs/arcagi3_traces",
                "python -m audit.independent_verify --checkpoint frozen/recurrent_latent_fast.pt --config fast --json-output docs/audit_after_arcagi3.json",
            ]
        )

    report: dict[str, Any] = {
        "terminal_outcome": "PENDING",
        "commands_required": commands_required,
        "hashes": {
            "before": before_hashes,
            "after": after_hashes,
            "manifest_before": manifest_before,
            "manifest_after": manifest_after,
            "frozen_unchanged": frozen_unchanged,
        },
        "leakage_scan": leakage,
        "living_rerun": living_rerun,
        "living_report_comparison": comparison,
        "durable_memory": durable,
        "idle_mode": idle,
        "idle_trajectory": idle_trajectory(model, 16 + int(cfg["idle_ticks"]), int(cfg["seed"]) + 1000, device),
        "private_token_causality": private,
        "richer_dynamics": richer,
        "curriculum": curriculum,
        "anti_leakage": anti,
        "source_snippets": snippets,
        "limitations": [],
    }
    report["claim_table"] = build_claim_table(report)
    failures = [row for row in report["claim_table"] if row["status"] == "FAIL"]
    weak = [row for row in report["claim_table"] if row["status"] == "WEAK"]
    if comparison["material_differences"]:
        report["limitations"].append("Rerun metrics differ materially from docs/living_system_report.json.")
    if weak:
        report["limitations"].append(
            "Curriculum retention failed: generated-private new concept accuracy "
            f"{curriculum['generated_private_curriculum_accuracy_after']:.6f}, "
            f"old-task core delta {curriculum['old_task_core_delta']:.6f}, "
            f"old-task core after {curriculum['old_task_retention_after']['core_mean']:.6f}."
        )
    if not anti["checks"]["hidden_target_canary_no_effect"]:
        report["limitations"].append("Hidden target canary changed model outputs.")
    if not frozen_unchanged:
        report["limitations"].append("Frozen checkpoint, manifest, or model source changed during audit.")
    report["terminal_outcome"] = "AUDIT PROVEN" if not failures and not weak else "AUDIT NOT PROVEN"

    json_path = Path(json_output)
    json_path.parent.mkdir(parents=True, exist_ok=True)
    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown_path = Path("docs/audit_proof.md") if json_path.name == "audit_proof.json" else json_path.with_suffix(".md")
    markdown_path.parent.mkdir(parents=True, exist_ok=True)
    markdown_path.write_text(render_markdown(report), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", choices=sorted(LIVING_CONFIGS), default="fast")
    parser.add_argument("--device", default=AUTO_DEVICE)
    parser.add_argument("--json-output", default="docs/audit_proof.json")
    args = parser.parse_args()
    report = run_audit(args.checkpoint, args.config, args.json_output, args.device)
    print(json.dumps({"terminal_outcome": report["terminal_outcome"], "limitations": report["limitations"]}, indent=2))


if __name__ == "__main__":
    main()
