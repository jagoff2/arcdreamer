from __future__ import annotations

import argparse
import hashlib
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .arcagi3_adapter import ArcAGI3Observation, ArcAGI3StepResult
from .attempt_buffer import AttemptBuffer, AttemptRecord, tensors_from_attempts
from .device import AUTO_DEVICE, resolve_device
from .video_jepa import VideoJEPA, VideoJEPAConfig, checkpoint_payload, jepa_loss, null_future_loss


DEV_TRACE_ROOTS = [
    Path("docs/external_base_traces"),
    Path("docs/external_collapse_traces"),
    Path("docs/perceptual_affordance_traces"),
]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def synthetic_attempts(count: int = 128, *, seed: int = 2027) -> list[AttemptRecord]:
    rng = random.Random(seed)
    records: list[AttemptRecord] = []
    for attempt_index in range(count):
        grid_size = 64 if attempt_index % 2 else 8
        low = 1 if grid_size == 8 else 4
        high = 7 if grid_size == 8 else 60
        y = rng.randrange(low, high)
        x = rng.randrange(low, high)
        target = (rng.randrange(low, high), rng.randrange(low, high))
        click_target = f"click:{target[1]}:{target[0]}"
        click_decoy = f"click:{rng.randrange(0, grid_size)}:{rng.randrange(0, grid_size)}"
        legal = ("1", "2", "3", "4", "5", click_target, click_decoy)
        steps = rng.randrange(10, 24)
        buffer = AttemptBuffer(
            suite_id="generated_jepa_dev",
            task_id=f"synthetic_{grid_size}_{attempt_index % 8}",
            variant="generated",
            split="dev",
            seed=seed + attempt_index,
            attempt_index=1,
        )
        for step in range(steps):
            grid = np.zeros((grid_size, grid_size), dtype=np.int64)
            grid[y, x] = 2
            grid[target] = 7
            obs = ArcAGI3Observation(
                task_id=f"generated_jepa_dev/synthetic_{grid_size}_{attempt_index % 8}",
                episode_id=f"generated/{attempt_index}",
                step_index=step,
                grid=grid,
                available_actions=legal,
                extras={"source": "generated_non_arc_dev"},
            )
            if abs(target[0] - y) > abs(target[1] - x):
                action = "2" if target[0] > y else "1"
            elif target[1] != x:
                action = "4" if target[1] > x else "3"
            else:
                action = "5" if rng.random() < 0.5 else click_target
            if rng.random() < 0.18:
                action = rng.choice(legal)
            old_distance = abs(target[0] - y) + abs(target[1] - x)
            if action in {"up", "1"}:
                y = max(0, y - 1)
            elif action in {"down", "2"}:
                y = min(grid_size - 1, y + 1)
            elif action in {"left", "3"}:
                x = max(0, x - 1)
            elif action in {"right", "4"}:
                x = min(grid_size - 1, x + 1)
            new_grid = np.zeros((grid_size, grid_size), dtype=np.int64)
            new_grid[y, x] = 2
            new_grid[target] = 7
            new_distance = abs(target[0] - y) + abs(target[1] - x)
            clicked_target = action == click_target and new_distance <= 1
            reward = 1.0 if new_distance == 0 or clicked_target else (0.05 if new_distance < old_distance else -0.01)
            terminal = new_distance == 0 or clicked_target or step == steps - 1
            events = ["positive_reward"] if reward > 0.0 else []
            if new_distance == 0 or clicked_target:
                events.append("resource_collected")
            result = ArcAGI3StepResult(
                ArcAGI3Observation(
                    task_id=obs.task_id,
                    episode_id=obs.episode_id,
                    step_index=step + 1,
                    grid=new_grid,
                    available_actions=legal,
                    extras={"source": "generated_non_arc_dev"},
                ),
                reward,
                terminal,
                False,
                {"events": events},
            )
            buffer.append_transition(obs, action, result)
            if terminal:
                break
        records.append(buffer.to_record())
    return records


def index_dev_traces() -> list[str]:
    paths: list[str] = []
    for root in DEV_TRACE_ROOTS:
        if not root.exists():
            continue
        for path in root.rglob("*.json"):
            parts = {part.lower() for part in path.parts}
            if "dev" in parts and "sealed_eval" not in parts:
                paths.append(str(path))
    return sorted(paths)


def train_jepa(records: list[AttemptRecord], *, device: torch.device, seed: int) -> tuple[VideoJEPA, dict[str, Any]]:
    torch.manual_seed(seed)
    shuffled_records = list(records)
    random.Random(seed + 19).shuffle(shuffled_records)
    split = max(8, int(len(shuffled_records) * 0.8))
    train_records = shuffled_records[:split]
    val_records = shuffled_records[split:] or shuffled_records[-8:]
    model = VideoJEPA(VideoJEPAConfig()).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3, weight_decay=1.0e-4)
    train_batch = tensors_from_attempts(train_records, max_steps=32, device=device)
    val_batch = tensors_from_attempts(val_records, max_steps=32, device=device)
    losses: list[float] = []
    model.train()
    for _ in range(140):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            train_batch["frames"],
            train_batch["action_ids"],
            train_batch["legal_counts"],
            train_batch["valid"],
            train_batch.get("action_features"),
        )
        loss = jepa_loss(output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        train_out = model(
            train_batch["frames"],
            train_batch["action_ids"],
            train_batch["legal_counts"],
            train_batch["valid"],
            train_batch.get("action_features"),
        )
        val_out = model(
            val_batch["frames"],
            val_batch["action_ids"],
            val_batch["legal_counts"],
            val_batch["valid"],
            val_batch.get("action_features"),
        )
        train_loss = float(jepa_loss(train_out).cpu())
        val_loss = float(jepa_loss(val_out).cpu())
        null_loss = float(null_future_loss(val_out).cpu())
    metrics = {
        "seed": seed,
        "device": str(device),
        "train_attempts": len(train_records),
        "val_attempts": len(val_records),
        "train_loss": train_loss,
        "val_loss": val_loss,
        "null_loss": null_loss,
        "jepa_beats_null": bool(val_loss < null_loss),
        "loss_tail": [round(item, 8) for item in losses[-8:]],
        "predicts_actions": False,
        "emits_text": False,
    }
    return model, metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["dev"], default="dev")
    parser.add_argument("--output", default="runs/video_jepa.pt")
    parser.add_argument("--manifest", default="data/jepa_trace_manifest.json")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else AUTO_DEVICE)
    parser.add_argument("--seed", type=int, default=20270612)
    args = parser.parse_args()
    del args.config
    device = resolve_device(args.device)
    dev_trace_index = index_dev_traces()
    records = synthetic_attempts(seed=args.seed)
    manifest = {
        "format": "jepa_attempt_trace_manifest_v1",
        "training_sources": ["generated_non_arc_dev_attempts"],
        "generated_attempts": len(records),
        "indexed_non_arc_dev_traces": len(dev_trace_index),
        "dev_trace_index_sample": dev_trace_index[:20],
        "used_official_sealed_data": False,
        "used_hidden_labels": False,
        "used_action_advice": False,
        "attempt_unit_fields": ["frames", "actions", "legal_actions", "score_delta", "event_delta", "terminal"],
    }
    model, metrics = train_jepa(records, device=device, seed=args.seed)
    manifest["metrics"] = metrics
    manifest_path = Path(args.manifest)
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    payload = checkpoint_payload(model, metrics, manifest)
    torch.save(payload, output)
    manifest["checkpoint"] = str(output)
    manifest["checkpoint_sha256"] = sha256_file(output)
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output), "manifest": str(manifest_path), "metrics": metrics}, indent=2))


if __name__ == "__main__":
    main()
