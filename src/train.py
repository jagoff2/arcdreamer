from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .curriculum import (
    REQUIRED_SCIENTIST_CAPABILITIES,
    SCIENTIST_CURRICULUM_FAMILIES,
    generate_scientist_curriculum_batch,
)
from .env import (
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
    TOK_NONE,
    TOK_TOLD_GOAL,
    generate_batch,
    idle_language_target,
    private_target,
    shortest_action,
)
from .metrics import masked_ce
from .model import (
    ModelConfig,
    PROGRAM_FAMILIES,
    PROGRAM_FIELDS,
    PROGRAM_TRANSFORMS,
    RecurrentLatentModel,
    save_checkpoint,
)


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
    curriculum_every: int = 10


CONFIGS = {
    "smoke": TrainConfig("smoke", steps=2, batch_size=8, seq_len=80, lr=2e-3, log_every=1),
    "fast": TrainConfig("fast", steps=900, batch_size=64, seq_len=80, lr=2e-3, log_every=100),
    "extended": TrainConfig("extended", steps=1800, batch_size=96, seq_len=88, lr=1.5e-3, log_every=100),
}

SENSOR_POS = slice(0, GRID_SIZE)
SENSOR_BODY = slice(GRID_SIZE + 2, GRID_SIZE + 2 + NUM_BODY_SCALARS)
SENSOR_COLOR = slice(GRID_SIZE + 2 + NUM_BODY_SCALARS, GRID_SIZE + 2 + NUM_BODY_SCALARS + NUM_COLORS + 1)
SENSOR_VISIBLE = GRID_SIZE + 2 + NUM_BODY_SCALARS + NUM_COLORS + 1
SENSOR_OBJECT_POS = slice(SENSOR_VISIBLE + 1, SENSOR_VISIBLE + 1 + GRID_SIZE + 1)

SCIENTIST_LOOP_STAGES = (
    "perceive",
    "perturb",
    "infer",
    "compress",
    "plan",
    "test",
    "consolidate",
)


def refresh_transition_impulses(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if "prev_delta" in batch:
        batch["prev_delta"].zero_()
        if batch["sensory"].shape[1] > 1:
            batch["prev_delta"][:, 1:] = batch["sensory"][:, 1:] - batch["sensory"][:, :-1]
    return batch


def blank_training_batch(batch: Dict[str, torch.Tensor], blank_after: int = 4) -> Dict[str, torch.Tensor]:
    altered = {key: value.clone() for key, value in batch.items()}
    altered["sensory"][:, blank_after:, SENSOR_COLOR] = torch.nn.functional.one_hot(
        torch.full_like(batch["world_color_target"][:, blank_after:], NUM_COLORS),
        NUM_COLORS + 1,
    ).float()
    altered["sensory"][:, blank_after:, SENSOR_VISIBLE] = 0.0
    altered["sensory"][:, blank_after:, SENSOR_OBJECT_POS] = torch.nn.functional.one_hot(
        torch.full_like(batch["world_pos_target"][:, blank_after:], GRID_SIZE),
        GRID_SIZE + 1,
    ).float()
    altered["lang_in"][:, blank_after:] = TOK_NONE
    for tick in range(blank_after, altered["sensory"].shape[1]):
        current_pos = batch["sensory"][:, tick, SENSOR_POS].argmax(dim=-1)
        energy = batch["sensory"][:, tick, SENSOR_BODY.start + BODY_ENERGY]
        damage = batch["sensory"][:, tick, SENSOR_BODY.start + BODY_DAMAGE]
        altered["language_target"][:, tick] = idle_language_target(
            tick,
            batch["world_color_target"][:, tick],
            batch["world_pos_target"][:, tick],
            current_pos,
            energy,
            damage,
        )
        altered["private_target"][:, tick] = private_target(
            tick,
            TOK_NONE,
            batch["world_color_target"][:, tick],
            batch["world_pos_target"][:, tick],
            energy,
            damage,
        )
        altered["action_target"][:, tick] = shortest_action(
            current_pos,
            batch["world_pos_target"][:, tick],
        )
    altered["private_in"].zero_()
    altered["private_in"][:, :blank_after] = batch["private_in"][:, :blank_after]
    altered["private_in"][:, blank_after + 1 :] = altered["private_target"][:, blank_after:-1]
    return refresh_transition_impulses(altered)


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
    return refresh_transition_impulses(altered)


def false_told_conflict_training_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = {key: value.clone() for key, value in batch.items()}
    false_color = (batch["world_color_target"][:, 8] + 1) % NUM_COLORS
    false_pos = (batch["world_pos_target"][:, 8] + 1) % GRID_SIZE
    altered["sensory"][:, 8, SENSOR_COLOR] = torch.nn.functional.one_hot(false_color, NUM_COLORS + 1).float()
    altered["sensory"][:, 8, SENSOR_VISIBLE] = 1.0
    altered["sensory"][:, 8, SENSOR_OBJECT_POS] = torch.nn.functional.one_hot(false_pos, GRID_SIZE + 1).float()
    altered["lang_in"][:, 8] = TOK_TOLD_GOAL
    return refresh_transition_impulses(altered)


def _sequence_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1))


def _scientist_program_targets(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    actions = batch["action_target"]
    family_target = torch.full_like(actions, PROGRAM_FAMILIES.index("field_change"))
    family_target = torch.where(
        (actions == ACTION_LEFT) | (actions == ACTION_RIGHT),
        torch.full_like(family_target, PROGRAM_FAMILIES.index("move_color")),
        family_target,
    )
    family_target = torch.where(
        actions == ACTION_STAY,
        torch.full_like(family_target, PROGRAM_FAMILIES.index("no_op")),
        family_target,
    )

    field_target = torch.full_like(actions, PROGRAM_FIELDS.index("grid"))
    transform_target = torch.full_like(actions, PROGRAM_TRANSFORMS.index("change"))
    transform_target = torch.where(
        actions == ACTION_STAY,
        torch.full_like(transform_target, PROGRAM_TRANSFORMS.index("identity")),
        transform_target,
    )
    transform_target = torch.where(
        actions == ACTION_LEFT,
        torch.full_like(transform_target, PROGRAM_TRANSFORMS.index("translate_left")),
        transform_target,
    )
    transform_target = torch.where(
        actions == ACTION_RIGHT,
        torch.full_like(transform_target, PROGRAM_TRANSFORMS.index("translate_right")),
        transform_target,
    )
    transform_target = torch.where(
        (actions == ACTION_FORAGE) | (actions == ACTION_REST),
        torch.full_like(transform_target, PROGRAM_TRANSFORMS.index("change")),
        transform_target,
    )

    family_id = batch.get("curriculum_family_id")
    if family_id is not None:
        row_family = family_id.view(-1, 1).expand_as(actions)
        object_permanence = row_family == 0
        inventory_or_resource = (row_family == 4) | (row_family == 8)
        sparse_unknown_goal = row_family == 9
        no_op_trap_reversal = row_family == 11
        family_target = torch.where(
            object_permanence,
            torch.full_like(family_target, PROGRAM_FAMILIES.index("field_stable")),
            family_target,
        )
        field_target = torch.where(
            inventory_or_resource,
            torch.full_like(field_target, PROGRAM_FIELDS.index("sensory")),
            field_target,
        )
        field_target = torch.where(
            sparse_unknown_goal,
            torch.full_like(field_target, PROGRAM_FIELDS.index("lang_in")),
            field_target,
        )
        family_target = torch.where(
            no_op_trap_reversal & (actions == ACTION_STAY),
            torch.full_like(family_target, PROGRAM_FAMILIES.index("no_op")),
            family_target,
        )

    return {
        "family": family_target,
        "field": field_target,
        "transform": transform_target,
        "color": batch["world_color_target"],
    }


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
    private_loss = F.cross_entropy(
        outputs["private_logits"].reshape(-1, outputs["private_logits"].shape[-1]),
        batch["private_target"].reshape(-1),
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
    latent_magnitude_loss = outputs["latents"].float().pow(2).mean()
    zero_program_loss = outputs["action_logits"].sum() * 0.0
    if {
        "program_family_logits",
        "program_field_logits",
        "program_transform_logits",
        "program_color_logits",
    }.issubset(outputs):
        program_targets = _scientist_program_targets(batch)
        program_family_loss = _sequence_ce(outputs["program_family_logits"], program_targets["family"])
        program_field_loss = _sequence_ce(outputs["program_field_logits"], program_targets["field"])
        program_transform_loss = _sequence_ce(outputs["program_transform_logits"], program_targets["transform"])
        program_color_loss = _sequence_ce(outputs["program_color_logits"], program_targets["color"])
        program_loss = (
            program_family_loss
            + 0.5 * program_field_loss
            + 0.5 * program_transform_loss
            + 0.25 * program_color_loss
        )
    else:
        program_family_loss = zero_program_loss
        program_field_loss = zero_program_loss
        program_transform_loss = zero_program_loss
        program_color_loss = zero_program_loss
        program_loss = zero_program_loss

    loop_perceive = 0.5 * (world_color_loss + world_pos_loss)
    loop_perturb = action_loss + 0.25 * program_transform_loss
    loop_infer = program_loss + 0.25 * provenance_loss
    loop_compress = collapse_loss + 0.002 * latent_magnitude_loss
    loop_plan = action_loss + 0.25 * self_loss
    loop_test = provenance_loss + 0.25 * program_field_loss
    loop_consolidate = memory_loss + self_loss + 0.25 * private_loss
    loop_objective = (
        loop_perceive
        + loop_perturb
        + loop_infer
        + loop_compress
        + loop_plan
        + loop_test
        + loop_consolidate
    )

    total = (
        2.2 * action_loss
        + 1.3 * language_loss
        + provenance_loss
        + 1.0 * private_loss
        + world_color_loss
        + world_pos_loss
        + 1.5 * memory_loss
        + 5.0 * self_loss
        + 0.05 * collapse_loss
        + 0.10 * loop_objective
    )
    return {
        "total": total,
        "action": action_loss.detach(),
        "language": language_loss.detach(),
        "provenance": provenance_loss.detach(),
        "private": private_loss.detach(),
        "world_color": world_color_loss.detach(),
        "world_pos": world_pos_loss.detach(),
        "memory": memory_loss.detach(),
        "self": self_loss.detach(),
        "collapse": collapse_loss.detach(),
        "program": program_loss.detach(),
        "loop_perceive": loop_perceive.detach(),
        "loop_perturb": loop_perturb.detach(),
        "loop_infer": loop_infer.detach(),
        "loop_compress": loop_compress.detach(),
        "loop_plan": loop_plan.detach(),
        "loop_test": loop_test.detach(),
        "loop_consolidate": loop_consolidate.detach(),
    }


def train_model(
    config_name: str = "fast",
    output: str | Path = "runs/latest.pt",
    steps: int | None = None,
    device: DeviceLike = AUTO_DEVICE,
) -> Dict[str, float]:
    target_device = resolve_device(device)
    cfg = CONFIGS[config_name]
    if steps is not None:
        cfg = TrainConfig(**{**asdict(cfg), "steps": steps})

    torch.manual_seed(cfg.seed)
    model = RecurrentLatentModel(ModelConfig(hidden_dim=cfg.hidden_dim)).to(target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    model.train()

    last_losses: Dict[str, torch.Tensor] = {}
    family_covered = torch.zeros(len(SCIENTIST_CURRICULUM_FAMILIES), dtype=torch.bool)
    capability_covered = torch.zeros(len(REQUIRED_SCIENTIST_CAPABILITIES), dtype=torch.bool)
    for step in range(1, cfg.steps + 1):
        core_batch = generate_batch(
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            base_seed=cfg.seed + step * cfg.batch_size,
            device=target_device,
        )
        curriculum_batch = generate_scientist_curriculum_batch(
            batch_size=cfg.batch_size,
            seq_len=cfg.seq_len,
            base_seed=cfg.seed + 1_000_000 + step * cfg.batch_size,
            device=target_device,
        )
        family_covered[curriculum_batch["curriculum_family_id"].detach().cpu().unique()] = True
        capability_covered |= curriculum_batch["curriculum_capability_mask"].detach().cpu().any(dim=0)
        outputs = model(
            core_batch["sensory"],
            core_batch["lang_in"],
            core_batch["private_in"],
            core_batch["prev_action"],
            core_batch["prev_delta"],
            core_batch["dt"],
        )
        losses = compute_losses(outputs, core_batch)
        zero_aux = outputs["action_logits"].sum() * 0.0
        curriculum_losses: Dict[str, torch.Tensor] = {"total": zero_aux}
        should_train_curriculum = step == 1 or step == cfg.steps or step % cfg.curriculum_every == 0
        if should_train_curriculum:
            curriculum_outputs = model(
                curriculum_batch["sensory"],
                curriculum_batch["lang_in"],
                curriculum_batch["private_in"],
                curriculum_batch["prev_action"],
                curriculum_batch["prev_delta"],
                curriculum_batch["dt"],
            )
            curriculum_losses = compute_losses(curriculum_outputs, curriculum_batch)
        blank_batch = blank_training_batch(core_batch)
        blank_outputs = model(
            blank_batch["sensory"],
            blank_batch["lang_in"],
            blank_batch["private_in"],
            blank_batch["prev_action"],
            blank_batch["prev_delta"],
            blank_batch["dt"],
        )
        blank_losses = compute_losses(blank_outputs, blank_batch)
        told_batch = told_only_training_batch(core_batch)
        told_outputs = model(
            told_batch["sensory"],
            told_batch["lang_in"],
            told_batch["private_in"],
            told_batch["prev_action"],
            told_batch["prev_delta"],
            told_batch["dt"],
        )
        told_losses = compute_losses(told_outputs, told_batch)
        conflict_batch = false_told_conflict_training_batch(core_batch)
        conflict_outputs = model(
            conflict_batch["sensory"],
            conflict_batch["lang_in"],
            conflict_batch["private_in"],
            conflict_batch["prev_action"],
            conflict_batch["prev_delta"],
            conflict_batch["dt"],
        )
        conflict_losses = compute_losses(conflict_outputs, conflict_batch)
        total_loss = (
            losses["total"]
            + 0.85 * blank_losses["total"]
            + 0.35 * told_losses["total"]
            + 0.45 * conflict_losses["total"]
            + 0.35 * curriculum_losses["total"]
        )
        late_blank_batch = blank_training_batch(core_batch, blank_after=16)
        late_blank_outputs = model(
            late_blank_batch["sensory"],
            late_blank_batch["lang_in"],
            late_blank_batch["private_in"],
            late_blank_batch["prev_action"],
            late_blank_batch["prev_delta"],
            late_blank_batch["dt"],
        )
        late_blank_losses = compute_losses(late_blank_outputs, late_blank_batch)
        total_loss = total_loss + 0.55 * late_blank_losses["total"]
        optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        losses["total"] = total_loss.detach()
        losses["blank"] = blank_losses["total"].detach()
        losses["late_blank"] = late_blank_losses["total"].detach()
        losses["told"] = told_losses["total"].detach()
        losses["conflict"] = conflict_losses["total"].detach()
        losses["curriculum"] = curriculum_losses["total"].detach()
        last_losses = losses
        if step == 1 or step % cfg.log_every == 0 or step == cfg.steps:
            printable = {key: round(float(value.item()), 4) for key, value in losses.items()}
            print(json.dumps({"step": step, **printable}))

    summary = {f"loss_{key}": float(value.item()) for key, value in last_losses.items()}
    summary["curriculum_family_count"] = float(family_covered.sum().item())
    summary["curriculum_capability_count"] = float(capability_covered.sum().item())
    summary["scientist_loop_stage_count"] = float(len(SCIENTIST_LOOP_STAGES))
    save_checkpoint(output, model, asdict(cfg), summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONFIGS), default="fast")
    parser.add_argument("--output", default="runs/latest.pt")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--device", default=AUTO_DEVICE)
    args = parser.parse_args()
    summary = train_model(args.config, args.output, args.steps, args.device)
    print(json.dumps({"checkpoint": args.output, **summary}, indent=2))


if __name__ == "__main__":
    main()
