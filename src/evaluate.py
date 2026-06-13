from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .env import (
    GRID_SIZE,
    NUM_ACTIONS,
    NUM_BODY_SCALARS,
    TOK_ASK_ACTION,
    TinyWorldRuntime,
    generate_batch,
    token_for_tick,
)
from .metrics import (
    categorical_head_stats,
    masked_accuracy,
    memorization_overfit_gates,
    memorization_overfit_stats,
    pass_fail,
)
from .model import load_checkpoint
from .run_unbroken import run_unbroken


EVAL_CONFIGS = {
    "smoke": {"episodes": 16, "batch_size": 16, "seq_len": 80, "runtime_ticks": 128},
    "fast": {"episodes": 512, "batch_size": 128, "seq_len": 80, "runtime_ticks": 10000},
    "extended": {"episodes": 1024, "batch_size": 128, "seq_len": 88, "runtime_ticks": 10000},
}


SENSOR_BODY = slice(GRID_SIZE + 2, GRID_SIZE + 2 + NUM_BODY_SCALARS)


def closed_loop_action_success(
    model,
    episodes: int,
    seq_len: int,
    seed: int,
    device: DeviceLike = AUTO_DEVICE,
) -> float:
    device = resolve_device(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for episode in range(episodes):
            world = TinyWorldRuntime(seed=seed + episode, episode_len=seq_len)
            z = model.initial_state(1, device=device)
            private_token = torch.zeros(1, dtype=torch.long, device=device)
            previous_sensory = None
            previous_action = NUM_ACTIONS
            for tick in range(seq_len):
                observation = world.observation(device=device, private_in=int(private_token.item()))
                current_sensory = observation["sensory"]
                if previous_sensory is None:
                    prev_delta = torch.zeros_like(current_sensory)
                else:
                    prev_delta = current_sensory - previous_sensory
                model_observation = {
                    **observation,
                    "prev_action": torch.tensor([previous_action], dtype=torch.long, device=device),
                    "prev_delta": prev_delta,
                    "dt": torch.ones(1, dtype=torch.float32, device=device),
                }
                output, z = model.step(model_observation, z)
                private_token = output["private_logits"].argmax(dim=-1)
                action = int(output["action_logits"].argmax(dim=-1).item())
                if tick >= 64 and token_for_tick(tick) == TOK_ASK_ACTION:
                    correct += int(action == world.expected_action())
                    total += 1
                world.step(action)
                previous_sensory = current_sensory.detach()
                previous_action = action
    return float(correct / total) if total else 0.0


def _offline_action_accuracy(model, batch: Dict[str, torch.Tensor]) -> float:
    outputs = model(
        batch["sensory"],
        batch["lang_in"],
        batch["private_in"],
        batch["prev_action"],
        batch["prev_delta"],
        batch["dt"],
    )
    return masked_accuracy(outputs["action_logits"], batch["action_target"], batch["action_mask"])


def _perturb_public_observations(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    perturbed = {key: value.clone() for key, value in batch.items()}
    sensory = perturbed["sensory"]
    tick_pattern = torch.where(
        torch.arange(sensory.shape[1], device=sensory.device) % 2 == 0,
        torch.tensor(1.0, device=sensory.device),
        torch.tensor(-1.0, device=sensory.device),
    ).view(1, -1, 1)
    body_pattern = torch.tensor([0.50, -0.50, 0.25, -0.25], device=sensory.device).view(1, 1, -1)
    sensory[:, :, SENSOR_BODY] = torch.clamp(
        sensory[:, :, SENSOR_BODY] + 0.010 * tick_pattern * body_pattern,
        0.0,
        1.0,
    )
    perturbed["prev_delta"].zero_()
    if sensory.shape[1] > 1:
        perturbed["prev_delta"][:, 1:] = sensory[:, 1:] - sensory[:, :-1]
    return perturbed


def memorization_overfit_audit(
    model,
    *,
    batch_size: int,
    seq_len: int,
    device: DeviceLike = AUTO_DEVICE,
) -> Dict[str, float]:
    device = resolve_device(device)
    audit_batch_size = min(int(batch_size), 64)
    train_like = generate_batch(
        batch_size=audit_batch_size,
        seq_len=seq_len,
        base_seed=1234 + audit_batch_size,
        device=device,
    )
    heldout = generate_batch(
        batch_size=audit_batch_size,
        seq_len=seq_len,
        base_seed=4_000_000,
        device=device,
    )
    perturbed = _perturb_public_observations(heldout)
    adversarial = generate_batch(
        batch_size=audit_batch_size,
        seq_len=seq_len,
        base_seed=9_000_000,
        device=device,
    )
    train_accuracy = _offline_action_accuracy(model, train_like)
    heldout_accuracy = _offline_action_accuracy(model, heldout)
    perturbation_accuracy = _offline_action_accuracy(model, perturbed)
    adversarial_accuracy = _offline_action_accuracy(model, adversarial)
    stats = memorization_overfit_stats(
        train_accuracy=train_accuracy,
        heldout_accuracy=heldout_accuracy,
        perturbation_accuracy=perturbation_accuracy,
        adversarial_accuracy=adversarial_accuracy,
    )
    gates = memorization_overfit_gates(stats)
    stats["memorization_overfit_pass"] = float(gates["memorization_overfit"])
    for key, value in gates.items():
        stats[f"{key}_pass"] = float(value)
    return stats


def evaluate_checkpoint(
    checkpoint: str | Path,
    config_name: str = "fast",
    runtime_ticks: int | None = None,
    device: DeviceLike = AUTO_DEVICE,
) -> Dict[str, float]:
    device = str(resolve_device(device))
    cfg = dict(EVAL_CONFIGS[config_name])
    if runtime_ticks is not None:
        cfg["runtime_ticks"] = runtime_ticks

    model = load_checkpoint(checkpoint, device=device)
    model.eval()

    totals: Dict[str, float] = {
        "goal_action_success": 0.0,
        "delayed_memory_accuracy": 0.0,
        "object_color_accuracy": 0.0,
        "object_pos_accuracy": 0.0,
        "provenance_accuracy": 0.0,
        "grounded_language_accuracy": 0.0,
        "self_world_continuity_accuracy": 0.0,
    }
    head_totals: Dict[str, float] = {}
    batches = 0
    with torch.no_grad():
        for offset in range(0, cfg["episodes"], cfg["batch_size"]):
            batch_size = min(cfg["batch_size"], cfg["episodes"] - offset)
            batch = generate_batch(
                batch_size=batch_size,
                seq_len=cfg["seq_len"],
                base_seed=1_000_000 + offset,
                device=device,
            )
            outputs = model(
                batch["sensory"],
                batch["lang_in"],
                batch["private_in"],
                batch["prev_action"],
                batch["prev_delta"],
                batch["dt"],
            )
            totals["goal_action_success"] += masked_accuracy(
                outputs["action_logits"], batch["action_target"], batch["action_mask"]
            )
            totals["delayed_memory_accuracy"] += masked_accuracy(
                outputs["memory_color_logits"],
                batch["memory_color_target"],
                batch["delayed_memory_mask"],
            )
            totals["object_color_accuracy"] += masked_accuracy(
                outputs["world_color_logits"],
                batch["world_color_target"],
                batch["object_mask"],
            )
            totals["object_pos_accuracy"] += masked_accuracy(
                outputs["world_pos_logits"],
                batch["world_pos_target"],
                batch["object_mask"],
            )
            totals["provenance_accuracy"] += float(
                (outputs["provenance_logits"].argmax(dim=-1) == batch["provenance_target"])
                .float()
                .mean()
                .item()
            )
            totals["grounded_language_accuracy"] += masked_accuracy(
                outputs["language_logits"],
                batch["language_target"],
                batch["grounded_language_mask"],
            )
            totals["self_world_continuity_accuracy"] += masked_accuracy(
                outputs["self_start_logits"],
                batch["self_start_target"],
                batch["self_mask"],
            )
            head_stats = {
                **categorical_head_stats(
                    outputs["action_logits"],
                    prefix="action_head",
                    mask=batch["action_mask"],
                ),
                **categorical_head_stats(
                    outputs["language_logits"],
                    prefix="language_head",
                    mask=batch["grounded_language_mask"],
                ),
                **categorical_head_stats(outputs["private_logits"], prefix="private_head"),
            }
            for key, value in head_stats.items():
                head_totals[key] = head_totals.get(key, 0.0) + float(value)
            batches += 1

    metrics = {key: value / batches for key, value in totals.items()}
    metrics.update({key: value / batches for key, value in head_totals.items()})
    metrics["offline_goal_action_accuracy"] = metrics["goal_action_success"]
    metrics["goal_action_success"] = closed_loop_action_success(
        model=model,
        episodes=int(cfg["episodes"]),
        seq_len=int(cfg["seq_len"]),
        seed=1_500_000,
        device=device,
    )
    metrics["object_permanence_accuracy"] = (
        metrics["object_color_accuracy"] + metrics["object_pos_accuracy"]
    ) / 2.0
    with torch.no_grad():
        metrics.update(
            memorization_overfit_audit(
                model,
                batch_size=int(cfg["batch_size"]),
                seq_len=int(cfg["seq_len"]),
                device=device,
            )
        )

    runtime = run_unbroken(
        checkpoint=checkpoint,
        max_ticks=int(cfg["runtime_ticks"]),
        log_every=0,
        seed=2_000_000,
        device=device,
    )
    metrics.update(runtime)
    gates = pass_fail(metrics)
    metrics["all_gates_pass"] = float(all(gates.values()))
    print(json.dumps({"metrics": metrics, "gates": gates}, indent=2))
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", choices=sorted(EVAL_CONFIGS), default="fast")
    parser.add_argument("--runtime-ticks", type=int, default=None)
    parser.add_argument("--device", default=AUTO_DEVICE)
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.config, args.runtime_ticks, args.device)


if __name__ == "__main__":
    main()
