from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from .base_world_model import (
    ExternalBaseConfig,
    ExternalBaseWorldModel,
    save_external_base_checkpoint,
    sha256_file,
    training_targets,
    write_external_manifest,
)
from .device import AUTO_DEVICE, DeviceLike, resolve_device


TRAINED_ARMS = [
    "old_base_world_model",
    "old_base_finetuned",
    "from_scratch_external_base",
    "null_training_control",
]


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32 - 1))
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_trace_tensors(manifest_path: str | Path) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    data = np.load(manifest["data_path"])
    arrays = {key: data[key] for key in data.files}
    return arrays, manifest


def sample_batch(
    arrays: dict[str, np.ndarray],
    batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    *,
    positive_fraction: float = 0.0,
) -> dict[str, torch.Tensor]:
    count = int(arrays["obs"].shape[0])
    batch_size = max(int(batch_size), 1)
    positive_fraction = max(min(float(positive_fraction), 1.0), 0.0)
    reward = arrays.get("reward")
    positive_signal = arrays.get("future_progress", reward)
    positive_idx = (
        np.flatnonzero(np.asarray(positive_signal) > 0.0)
        if positive_signal is not None and positive_fraction > 0.0
        else np.asarray([], dtype=np.int64)
    )
    if positive_idx.size:
        positive_count = min(batch_size, max(1, int(round(float(batch_size) * positive_fraction))))
        rest_count = batch_size - positive_count
        positive_sample = rng.choice(positive_idx, size=positive_count, replace=positive_idx.size < positive_count)
        rest_sample = rng.integers(0, count, size=rest_count) if rest_count else np.asarray([], dtype=np.int64)
        idx = np.concatenate([positive_sample, rest_sample]).astype(np.int64, copy=False)
        rng.shuffle(idx)
    else:
        idx = rng.integers(0, count, size=batch_size)
    batch = {
        "obs": torch.as_tensor(arrays["obs"][idx], dtype=torch.float32, device=device),
        "next_obs": torch.as_tensor(arrays["next_obs"][idx], dtype=torch.float32, device=device),
        "action_features": torch.as_tensor(arrays["action_features"][idx], dtype=torch.float32, device=device),
        "reward": torch.as_tensor(arrays["reward"][idx], dtype=torch.float32, device=device),
        "family": torch.as_tensor(arrays["family"][idx], dtype=torch.long, device=device),
    }
    if "action_id" in arrays:
        batch["action_id"] = torch.as_tensor(arrays["action_id"][idx], dtype=torch.long, device=device)
    if "future_progress" in arrays:
        batch["future_progress"] = torch.as_tensor(arrays["future_progress"][idx], dtype=torch.float32, device=device)
    return batch


def shuffle_targets(batch: dict[str, torch.Tensor], rng: torch.Generator) -> dict[str, torch.Tensor]:
    perm = torch.randperm(batch["obs"].shape[0], generator=rng, device=batch["obs"].device)
    out = dict(batch)
    out["next_obs"] = batch["next_obs"][perm]
    out["reward"] = batch["reward"][perm]
    out["family"] = batch["family"][perm]
    return out


def objective_losses_from_output(
    model: ExternalBaseWorldModel,
    output: dict[str, torch.Tensor],
    batch: dict[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    targets = training_targets(batch)
    zero_actions = torch.zeros_like(batch["action_features"])
    next_memory_target = model.encode(targets["next_obs"], zero_actions)["memory"].detach()
    progress_target = targets["progress"]
    positive_count = progress_target.sum()
    negative_count = progress_target.numel() - positive_count
    positive_weight = torch.clamp(negative_count / positive_count.clamp_min(1.0), min=1.0, max=100.0)
    future_progress_target = targets["future_progress"]
    future_positive_count = (future_progress_target > 0.0).float().sum()
    future_negative_count = future_progress_target.numel() - future_positive_count
    future_positive_weight = torch.clamp(
        future_negative_count / future_positive_count.clamp_min(1.0),
        min=1.0,
        max=100.0,
    )
    if "action_id" in batch:
        policy_weight = 0.25 + (4.0 * future_progress_target)
        action_policy_prior = (
            F.cross_entropy(output["action_prior_logits"], batch["action_id"].long(), reduction="none") * policy_weight
        ).mean()
    else:
        action_policy_prior = output["action_prior_logits"].sum() * 0.0
    return {
        "next_observation": F.mse_loss(output["next_obs"], targets["next_obs"]),
        "change_mask": F.binary_cross_entropy_with_logits(output["change_logits"], targets["change_mask"]),
        "reward_event_noop": F.mse_loss(output["reward"], targets["reward"])
        + F.binary_cross_entropy_with_logits(output["noop_logits"], targets["noop"]),
        "progress_event": F.binary_cross_entropy_with_logits(
            output["progress_logits"],
            progress_target,
            pos_weight=positive_weight,
        ),
        "future_progress": F.binary_cross_entropy_with_logits(
            output["future_progress_logits"],
            future_progress_target,
            pos_weight=future_positive_weight,
        ),
        "action_policy_prior": action_policy_prior,
        "inverse_dynamics": F.cross_entropy(output["inverse_logits"], targets["family"]),
        "action_affordance": F.mse_loss(torch.sigmoid(output["affordance"]), targets["affordance"]),
        "temporal_object_persistence": F.mse_loss(output["memory"], next_memory_target),
        "latent_rollout_consistency": F.mse_loss(output["rollout_latent"], next_memory_target),
        "memory_utility": F.mse_loss(output["memory_next_obs"], targets["next_obs"]),
    }


def objective_losses(model: ExternalBaseWorldModel, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    output = model(batch["obs"], batch["action_features"])
    return objective_losses_from_output(model, output, batch)


def weighted_loss(losses: dict[str, torch.Tensor]) -> torch.Tensor:
    weights = {
        "next_observation": 1.0,
        "change_mask": 0.7,
        "reward_event_noop": 0.8,
        "progress_event": 1.4,
        "future_progress": 1.8,
        "action_policy_prior": 1.0,
        "inverse_dynamics": 0.4,
        "action_affordance": 0.6,
        "temporal_object_persistence": 0.2,
        "latent_rollout_consistency": 0.2,
        "memory_utility": 0.5,
    }
    return sum(losses[name] * weights[name] for name in losses)


@torch.no_grad()
def evaluate_losses(model: ExternalBaseWorldModel, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    losses = objective_losses(model, batch)
    out = {name: float(value.detach().item()) for name, value in losses.items()}
    out["total"] = float(weighted_loss(losses).detach().item())
    return out


@torch.no_grad()
def memory_utility_diagnostic(model: ExternalBaseWorldModel, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    output = model(batch["obs"], batch["action_features"])
    no_memory = model(batch["obs"], batch["action_features"], torch.zeros_like(output["memory"]))
    targets = training_targets(batch)
    with_memory = float(F.mse_loss(output["memory_next_obs"], targets["next_obs"]).item())
    without_memory = float(F.mse_loss(no_memory["memory_next_obs"], targets["next_obs"]).item())
    return {
        "with_memory_mse": with_memory,
        "without_memory_mse": without_memory,
        "memory_improves_delayed_partial_prediction": with_memory <= without_memory,
    }


def train_arm(
    *,
    arm: str,
    arrays: dict[str, np.ndarray],
    config: ExternalBaseConfig,
    steps: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    positive_fraction: float = 0.0,
) -> tuple[ExternalBaseWorldModel, dict[str, Any]]:
    set_seed(seed)
    model = ExternalBaseWorldModel(config, device=device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2.0e-3 if arm != "old_base_finetuned" else 1.0e-3, weight_decay=1.0e-4)
    rng = np.random.default_rng(seed + 17)
    torch_rng = torch.Generator(device=device).manual_seed(seed + 29)
    eval_batch = sample_batch(arrays, min(batch_size * 2, 2048), device, rng)
    if arm == "null_training_control":
        eval_batch = shuffle_targets(eval_batch, torch_rng)
    initial = evaluate_losses(model, eval_batch)
    tail_losses: list[float] = []
    for _ in range(steps):
        batch = sample_batch(arrays, batch_size, device, rng, positive_fraction=positive_fraction)
        if arm == "null_training_control":
            batch = shuffle_targets(batch, torch_rng)
        losses = objective_losses(model, batch)
        loss = weighted_loss(losses)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        tail_losses.append(float(loss.detach().item()))
    final = evaluate_losses(model, eval_batch)
    utility = memory_utility_diagnostic(model, eval_batch)
    return model, {
        "arm": arm,
        "seed": seed,
        "steps": steps,
        "batch_size": batch_size,
        "positive_fraction": float(max(min(float(positive_fraction), 1.0), 0.0)),
        "initial_losses": initial,
        "final_losses": final,
        "recent_total_loss_mean": float(np.mean(tail_losses[-min(len(tail_losses), 20) :])) if tail_losses else final["total"],
        "loss_delta": initial["total"] - final["total"],
        "training_losses_are_diagnostic_only": True,
        "memory_utility": utility,
    }


def train_external_base(
    *,
    manifest_path: str | Path,
    checkpoint_output: str | Path,
    manifest_output: str | Path,
    recurrent_checkpoint: str | Path,
    steps: int,
    batch_size: int,
    device: DeviceLike,
) -> dict[str, Any]:
    target_device = resolve_device(device)
    arrays, trace_manifest = load_trace_tensors(manifest_path)
    config = ExternalBaseConfig()
    arm_states: dict[str, dict[str, Any]] = {}
    arm_metrics: dict[str, Any] = {}
    seed_by_arm = {
        "old_base_world_model": 7101,
        "old_base_finetuned": 7102,
        "from_scratch_external_base": 7103,
        "null_training_control": 7104,
    }
    for arm in TRAINED_ARMS:
        model, metrics = train_arm(
            arm=arm,
            arrays=arrays,
            config=config,
            steps=steps,
            batch_size=batch_size,
            device=target_device,
            seed=seed_by_arm[arm],
        )
        arm_metrics[arm] = metrics
        arm_states[arm] = {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metrics": metrics,
        }
    manifest = {
        "format": "external_base_manifest_v1",
        "checkpoint": str(checkpoint_output),
        "old_recurrent_checkpoint": str(recurrent_checkpoint),
        "old_recurrent_checkpoint_sha256": sha256_file(recurrent_checkpoint),
        "trace_manifest": str(manifest_path),
        "trace_manifest_sha256": sha256_file(manifest_path),
        "trace_data_sha256": trace_manifest.get("data_sha256"),
        "trace_transition_count": trace_manifest.get("transition_count"),
        "source_counts": trace_manifest.get("source_counts"),
        "model_arms": ["old_base_unchanged", *TRAINED_ARMS],
        "frozen_before_sealed_eval": True,
        "no_sealed_official_training": True,
        "training_objectives": [
            "next observation/change prediction",
            "reward/event/no-op prediction",
            "positive progress event classification",
            "discounted future progress classification",
            "future-positive recurrent action prior",
            "inverse dynamics",
            "action-affordance prediction",
            "temporal object/region persistence",
            "latent rollout consistency",
            "memory improves delayed/partial prediction",
            "intrinsic value from controllability, prediction improvement, progress, and no-op avoidance",
        ],
        "arm_metrics": arm_metrics,
    }
    save_external_base_checkpoint(
        recurrent_checkpoint=recurrent_checkpoint,
        output_path=checkpoint_output,
        external_config=config,
        arm_states=arm_states,
        manifest=manifest,
        metrics=arm_metrics,
    )
    manifest["checkpoint_sha256"] = sha256_file(checkpoint_output)
    write_external_manifest(manifest_output, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-manifest", default="data/external_traces_manifest.json")
    parser.add_argument("--checkpoint-output", default="frozen/external_base_v1.pt")
    parser.add_argument("--manifest-output", default="frozen/external_base_manifest_v1.json")
    parser.add_argument("--recurrent-checkpoint", default="frozen/recurrent_latent_fast.pt")
    parser.add_argument("--steps", type=int, default=160)
    parser.add_argument("--batch-size", type=int, default=512)
    parser.add_argument("--device", default=AUTO_DEVICE)
    args = parser.parse_args()
    manifest = train_external_base(
        manifest_path=args.trace_manifest,
        checkpoint_output=args.checkpoint_output,
        manifest_output=args.manifest_output,
        recurrent_checkpoint=args.recurrent_checkpoint,
        steps=args.steps,
        batch_size=args.batch_size,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "checkpoint": manifest["checkpoint"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "trace_transition_count": manifest["trace_transition_count"],
                "model_arms": manifest["model_arms"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
