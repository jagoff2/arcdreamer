from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from .env import GRID_SIZE, NUM_COLORS, TOK_TOLD_GOAL, generate_batch
from .metrics import masked_ce
from .model import ModelConfig, RecurrentLatentModel, save_checkpoint


@dataclass
class TrainConfig:
    name: str
    steps: int
    batch_size: int
    seq_len: int
    lr: float
    hidden_dim: int = 64
    seed: int = 1234
    log_every: int = 50


CONFIGS = {
    "smoke": TrainConfig("smoke", steps=2, batch_size=8, seq_len=80, lr=2e-3, log_every=1),
    "fast": TrainConfig("fast", steps=700, batch_size=64, seq_len=80, lr=2e-3, log_every=100),
    "extended": TrainConfig("extended", steps=1800, batch_size=96, seq_len=88, lr=1.5e-3, log_every=100),
}

SENSOR_COLOR = slice(GRID_SIZE + 4, GRID_SIZE + 4 + NUM_COLORS + 1)
SENSOR_VISIBLE = GRID_SIZE + 4 + NUM_COLORS + 1
SENSOR_OBJECT_POS = slice(SENSOR_VISIBLE + 1, SENSOR_VISIBLE + 1 + GRID_SIZE + 1)


def blank_training_batch(batch: Dict[str, torch.Tensor], blank_after: int = 4) -> Dict[str, torch.Tensor]:
    altered = {key: value.clone() for key, value in batch.items()}
    altered["sensory"][:, blank_after:, :] = 0.0
    altered["lang_in"][:, blank_after:] = 0
    return altered


def told_only_training_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = {key: value.clone() for key, value in batch.items()}
    altered["sensory"][:, :4, SENSOR_COLOR] = torch.nn.functional.one_hot(
        torch.full_like(batch["world_color_target"][:, :4], NUM_COLORS), NUM_COLORS + 1
    ).float()
    altered["sensory"][:, :4, SENSOR_VISIBLE] = 0.0
    altered["sensory"][:, :4, SENSOR_OBJECT_POS] = torch.nn.functional.one_hot(
        torch.full_like(batch["world_pos_target"][:, :4], GRID_SIZE), GRID_SIZE + 1
    ).float()
    altered["sensory"][:, 8, SENSOR_COLOR] = torch.nn.functional.one_hot(
        batch["world_color_target"][:, 8], NUM_COLORS + 1
    ).float()
    altered["sensory"][:, 8, SENSOR_VISIBLE] = 1.0
    altered["sensory"][:, 8, SENSOR_OBJECT_POS] = torch.nn.functional.one_hot(
        batch["world_pos_target"][:, 8], GRID_SIZE + 1
    ).float()
    altered["lang_in"][:, 8] = TOK_TOLD_GOAL
    return altered


def false_told_conflict_training_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = {key: value.clone() for key, value in batch.items()}
    false_color = (batch["world_color_target"][:, 8] + 1) % NUM_COLORS
    false_pos = (batch["world_pos_target"][:, 8] + 1) % GRID_SIZE
    altered["sensory"][:, 8, SENSOR_COLOR] = torch.nn.functional.one_hot(false_color, NUM_COLORS + 1).float()
    altered["sensory"][:, 8, SENSOR_VISIBLE] = 1.0
    altered["sensory"][:, 8, SENSOR_OBJECT_POS] = torch.nn.functional.one_hot(false_pos, GRID_SIZE + 1).float()
    altered["lang_in"][:, 8] = TOK_TOLD_GOAL
    return altered


def compute_losses(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    action_loss = masked_ce(outputs["action_logits"], batch["action_target"], batch["action_mask"])
    language_loss = F.cross_entropy(
        outputs["language_logits"].reshape(-1, outputs["language_logits"].shape[-1]),
        batch["language_target"].reshape(-1),
    )
    provenance_loss = F.cross_entropy(
        outputs["provenance_logits"].reshape(-1, outputs["provenance_logits"].shape[-1]),
        batch["provenance_target"].reshape(-1),
    )
    world_color_loss = F.cross_entropy(
        outputs["world_color_logits"].reshape(-1, outputs["world_color_logits"].shape[-1]),
        batch["world_color_target"].reshape(-1),
    )
    world_pos_loss = F.cross_entropy(
        outputs["world_pos_logits"].reshape(-1, outputs["world_pos_logits"].shape[-1]),
        batch["world_pos_target"].reshape(-1),
    )
    memory_loss = masked_ce(
        outputs["memory_color_logits"],
        batch["memory_color_target"],
        batch["delayed_memory_mask"],
    )
    self_loss = masked_ce(outputs["self_start_logits"], batch["self_start_target"], batch["self_mask"])
    latent_std = outputs["latents"].reshape(-1, outputs["latents"].shape[-1]).std(dim=0)
    collapse_loss = torch.relu(torch.tensor(0.06, device=latent_std.device) - latent_std).mean()

    total = (
        action_loss
        + language_loss
        + provenance_loss
        + world_color_loss
        + world_pos_loss
        + 1.5 * memory_loss
        + 1.5 * self_loss
        + 0.05 * collapse_loss
    )
    return {
        "total": total,
        "action": action_loss.detach(),
        "language": language_loss.detach(),
        "provenance": provenance_loss.detach(),
        "world_color": world_color_loss.detach(),
        "world_pos": world_pos_loss.detach(),
        "memory": memory_loss.detach(),
        "self": self_loss.detach(),
        "collapse": collapse_loss.detach(),
    }


def train_model(
    config_name: str = "fast",
    output: str | Path = "runs/latest.pt",
    steps: int | None = None,
    device: str = "cpu",
) -> Dict[str, float]:
    cfg = CONFIGS[config_name]
    if steps is not None:
        cfg = TrainConfig(**{**asdict(cfg), "steps": steps})

    torch.manual_seed(cfg.seed)
    model = RecurrentLatentModel(ModelConfig(hidden_dim=cfg.hidden_dim)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    model.train()

    last_losses: Dict[str, torch.Tensor] = {}
    for step in range(1, cfg.steps + 1):
        batch = generate_batch(
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            base_seed=cfg.seed + step * cfg.batch_size,
            device=device,
        )
        outputs = model(batch["sensory"], batch["lang_in"])
        losses = compute_losses(outputs, batch)
        blank_batch = blank_training_batch(batch)
        blank_outputs = model(blank_batch["sensory"], blank_batch["lang_in"])
        blank_losses = compute_losses(blank_outputs, blank_batch)
        told_batch = told_only_training_batch(batch)
        told_outputs = model(told_batch["sensory"], told_batch["lang_in"])
        told_losses = compute_losses(told_outputs, told_batch)
        conflict_batch = false_told_conflict_training_batch(batch)
        conflict_outputs = model(conflict_batch["sensory"], conflict_batch["lang_in"])
        conflict_losses = compute_losses(conflict_outputs, conflict_batch)
        total_loss = (
            losses["total"]
            + 0.20 * blank_losses["total"]
            + 0.35 * told_losses["total"]
            + 0.45 * conflict_losses["total"]
        )
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses["total"] = total_loss.detach()
        losses["blank"] = blank_losses["total"].detach()
        losses["told"] = told_losses["total"].detach()
        losses["conflict"] = conflict_losses["total"].detach()
        last_losses = losses
        if step == 1 or step % cfg.log_every == 0 or step == cfg.steps:
            printable = {key: round(float(value.item()), 4) for key, value in losses.items()}
            print(json.dumps({"step": step, **printable}))

    summary = {f"loss_{key}": float(value.item()) for key, value in last_losses.items()}
    save_checkpoint(output, model, asdict(cfg), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONFIGS), default="fast")
    parser.add_argument("--output", default="runs/latest.pt")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    summary = train_model(args.config, args.output, args.steps, args.device)
    print(json.dumps({"checkpoint": args.output, **summary}, indent=2))


if __name__ == "__main__":
    main()
