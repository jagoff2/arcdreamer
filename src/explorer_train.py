from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .explore_env import EXPLORER_CONFIGS, build_explorer_dataset, explorer_config
from .metrics import plain_accuracy
from .world_model import ExplorerCore, ExplorerModelConfig, explorer_forward, save_explorer_checkpoint


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return logits.sum() * 0.0
    loss = F.cross_entropy(logits, target, reduction="none")
    return (loss * mask.float()).sum() / mask.float().sum()


def batch_slice(tensors: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in tensors.items()}


def compute_explorer_losses(outputs: dict[str, torch.Tensor], batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    action = _masked_ce(outputs["action_logits"], batch["action_target"], batch["informative_mask"])
    planner = _masked_ce(outputs["planner_logits"], batch["planner_target"], batch["planning_mask"])
    counter = _masked_ce(outputs["counterfactual_logits"], batch["counterfactual_target"], batch["counterfactual_mask"])
    next_state = F.cross_entropy(outputs["next_state_logits"], batch["next_state_target"])
    uncertainty = F.cross_entropy(outputs["uncertainty_logits"], batch["uncertainty_target"])
    novelty = F.cross_entropy(outputs["novelty_logits"], batch["novelty_target"])
    skill = _masked_ce(outputs["skill_logits"], batch["skill_target"], batch["skill_mask"])
    transfer = _masked_ce(outputs["skill_logits"], batch["skill_target"], batch["transfer_mask"])
    project = _masked_ce(outputs["project_logits"], batch["project_target"], batch["project_mask"])
    restart = _masked_ce(outputs["project_logits"], batch["project_target"], batch["restart_mask"])
    question = _masked_ce(outputs["question_logits"], batch["question_target"], batch["social_uncertainty_mask"])
    conflict = _masked_ce(outputs["conflict_logits"], batch["conflict_target"], batch["conflict_mask"])
    safety = _masked_ce(outputs["safety_logits"], batch["safety_target"], batch["safety_mask"])
    partner = _masked_ce(outputs["partner_logits"], batch["partner_target"], batch["partner_restart_mask"])
    mind = _masked_ce(outputs["mind_logits"], batch["mind_target"], batch["mind_mask"])
    total = (
        1.5 * action
        + 1.4 * planner
        + 1.2 * counter
        + 1.4 * next_state
        + uncertainty
        + novelty
        + 1.4 * skill
        + 0.7 * transfer
        + 1.3 * project
        + 0.8 * restart
        + question
        + conflict
        + safety
        + partner
        + 1.2 * mind
    )
    return {
        "total": total,
        "action": action.detach(),
        "planner": planner.detach(),
        "counterfactual": counter.detach(),
        "next_state": next_state.detach(),
        "uncertainty": uncertainty.detach(),
        "novelty": novelty.detach(),
        "skill": skill.detach(),
        "transfer": transfer.detach(),
        "project": project.detach(),
        "restart": restart.detach(),
        "question": question.detach(),
        "conflict": conflict.detach(),
        "safety": safety.detach(),
        "partner": partner.detach(),
        "mind": mind.detach(),
    }


def train_explorer(
    config_name: str = "tiny",
    output: str | Path = "runs/explorer_tiny.pt",
    checkpoint: str | Path = "frozen/recurrent_latent_fast.pt",
    device: DeviceLike = AUTO_DEVICE,
    steps: int | None = None,
) -> dict[str, Any]:
    target_device = resolve_device(device)
    cfg = explorer_config(config_name)
    actual_steps = int(steps if steps is not None else cfg.steps)
    torch.manual_seed(cfg.seed + 41)
    train_data = build_explorer_dataset(config_name, "train", checkpoint=checkpoint, device=target_device)
    tensors = train_data.tensors
    model = ExplorerCore(ExplorerModelConfig(hidden_dim=cfg.hidden_dim), device=target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1.0e-4)
    model.train()
    generator = torch.Generator(device=target_device)
    generator.manual_seed(cfg.seed + 99)
    count = tensors["obs"].shape[0]
    last_losses: dict[str, torch.Tensor] = {}
    for step in range(1, actual_steps + 1):
        indices = torch.randint(0, count, (cfg.batch_size,), generator=generator, device=target_device)
        batch = batch_slice(tensors, indices)
        outputs = explorer_forward(model, batch)
        losses = compute_explorer_losses(outputs, batch)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last_losses = losses
        if step == 1 or step == actual_steps or step % max(1, cfg.steps // 5) == 0:
            print(
                json.dumps(
                    {
                        "step": step,
                        **{key: round(float(value.item()), 5) for key, value in losses.items()},
                    }
                )
            )
    model.eval()
    with torch.no_grad():
        sample = batch_slice(tensors, torch.arange(min(2048, count), device=target_device))
        sample_out = explorer_forward(model, sample)
    metrics = {
        "sample_action_accuracy": plain_accuracy(sample_out["action_logits"], sample["action_target"]),
        "sample_planner_accuracy": plain_accuracy(sample_out["planner_logits"], sample["planner_target"]),
        "sample_next_state_accuracy": plain_accuracy(sample_out["next_state_logits"], sample["next_state_target"]),
        "sample_skill_accuracy": plain_accuracy(sample_out["skill_logits"], sample["skill_target"]),
        "records": float(count),
    }
    summary: dict[str, Any] = {
        "config": config_name,
        "checkpoint": str(checkpoint),
        "output": str(output),
        "steps": actual_steps,
        "losses": {f"loss_{key}": float(value.item()) for key, value in last_losses.items()},
        "metrics": metrics,
        "dataset": train_data.metadata,
    }
    save_explorer_checkpoint(
        output,
        model,
        {**asdict(cfg), "steps": actual_steps, "checkpoint": str(checkpoint)},
        summary,
    )
    print(json.dumps({"checkpoint": str(output), **metrics}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(EXPLORER_CONFIGS), default="tiny")
    parser.add_argument("--output", default="runs/explorer_tiny.pt")
    parser.add_argument("--checkpoint", default="frozen/recurrent_latent_fast.pt")
    parser.add_argument("--device", default=AUTO_DEVICE)
    parser.add_argument("--steps", type=int, default=None)
    args = parser.parse_args()
    train_explorer(args.config, args.output, args.checkpoint, args.device, args.steps)


if __name__ == "__main__":
    main()
