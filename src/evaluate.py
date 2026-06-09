from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from .env import TOK_ASK_ACTION, TinyWorldRuntime, generate_batch, token_for_tick
from .metrics import masked_accuracy, pass_fail
from .model import load_checkpoint
from .run_unbroken import run_unbroken


EVAL_CONFIGS = {
    "smoke": {"episodes": 16, "batch_size": 16, "seq_len": 80, "runtime_ticks": 128},
    "fast": {"episodes": 512, "batch_size": 128, "seq_len": 80, "runtime_ticks": 10000},
    "extended": {"episodes": 1024, "batch_size": 128, "seq_len": 88, "runtime_ticks": 10000},
}


def closed_loop_action_success(
    model,
    episodes: int,
    seq_len: int,
    seed: int,
    device: str = "cpu",
) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for episode in range(episodes):
            world = TinyWorldRuntime(seed=seed + episode, episode_len=seq_len)
            z = model.initial_state(1, device=device)
            private_token = torch.zeros(1, dtype=torch.long, device=device)
            for tick in range(seq_len):
                observation = world.observation(device=device, private_in=int(private_token.item()))
                output, z = model.step(observation, z)
                private_token = output["private_logits"].argmax(dim=-1)
                action = int(output["action_logits"].argmax(dim=-1).item())
                if tick >= 64 and token_for_tick(tick) == TOK_ASK_ACTION:
                    correct += int(action == world.expected_action())
                    total += 1
                world.step(action)
    return float(correct / total) if total else 0.0


def evaluate_checkpoint(
    checkpoint: str | Path,
    config_name: str = "fast",
    runtime_ticks: int | None = None,
    device: str = "cpu",
) -> Dict[str, float]:
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
            outputs = model(batch["sensory"], batch["lang_in"], batch["private_in"])
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
            batches += 1

    metrics = {key: value / batches for key, value in totals.items()}
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
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    evaluate_checkpoint(args.checkpoint, args.config, args.runtime_ticks, args.device)


if __name__ == "__main__":
    main()
