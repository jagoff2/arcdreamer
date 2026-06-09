from __future__ import annotations

import argparse
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Dict, List

import torch

from .curriculum import acquire_new_concept
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
    TinyWorldRuntime,
    generate_batch,
    shortest_action,
)
from .metrics import latent_noncollapse_stats, masked_accuracy
from .model import RecurrentLatentModel, load_checkpoint
from .persistent_memory import PersistentMemoryState
from .train import blank_training_batch

SENSOR_POS = slice(0, GRID_SIZE)
SENSOR_BODY = slice(GRID_SIZE + 2, GRID_SIZE + 2 + NUM_BODY_SCALARS)

LIVING_CONFIGS = {
    "smoke": {"batch_size": 16, "seq_len": 96, "seed": 55200, "idle_ticks": 48, "curriculum_steps": 32},
    "fast": {"batch_size": 128, "seq_len": 112, "seed": 75200, "idle_ticks": 96, "curriculum_steps": 80},
}


def run_autoregressive_private(
    model: RecurrentLatentModel,
    batch: Dict[str, torch.Tensor],
    initial_z: torch.Tensor | None = None,
    start_tick: int = 0,
    end_tick: int | None = None,
    initial_private: torch.Tensor | None = None,
) -> Dict[str, torch.Tensor]:
    batch_size, seq_len, _ = batch["sensory"].shape
    if end_tick is None:
        end_tick = seq_len
    device = batch["sensory"].device
    z = initial_z if initial_z is not None else model.initial_state(batch_size, device=device)
    private_token = (
        initial_private.clone()
        if initial_private is not None
        else torch.zeros(batch_size, dtype=torch.long, device=device)
    )
    outputs: List[Dict[str, torch.Tensor]] = []
    latents: List[torch.Tensor] = []
    private_tokens: List[torch.Tensor] = []
    for tick in range(start_tick, end_tick):
        output, z = model.step(
            {
                "sensory": batch["sensory"][:, tick],
                "lang_in": batch["lang_in"][:, tick],
                "private_in": private_token,
            },
            z,
        )
        private_token = output["private_logits"].argmax(dim=-1)
        outputs.append(output)
        latents.append(z)
        private_tokens.append(private_token)
    stacked = {key: torch.stack([item[key] for item in outputs], dim=1) for key in outputs[0]}
    stacked["latents"] = torch.stack(latents, dim=1)
    stacked["generated_private"] = torch.stack(private_tokens, dim=1)
    return stacked


def durable_restart_eval(
    model: RecurrentLatentModel,
    batch_size: int,
    seq_len: int,
    seed: int,
    device: str = "cpu",
) -> Dict[str, float]:
    batch = generate_batch(batch_size, seq_len, seed, device=device)
    restart_tick = min(70, seq_len // 2 + 16)
    with torch.no_grad(), TemporaryDirectory() as tmp:
        normal = run_autoregressive_private(model, batch)
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
        reset = run_autoregressive_private(
            model,
            batch,
            initial_z=model.initial_state(batch_size, device=device),
            initial_private=torch.zeros(batch_size, dtype=torch.long, device=device),
            start_tick=restart_tick,
        )

    final_mask_tail = torch.zeros(batch_size, seq_len - restart_tick, dtype=torch.bool, device=device)
    final_mask_tail[:, -1] = True
    final_mask_full = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    final_mask_full[:, -1] = True
    return {
        "restart_tick": float(restart_tick),
        "normal_final_memory_accuracy": masked_accuracy(
            normal["memory_color_logits"], batch["memory_color_target"], final_mask_full
        ),
        "memory_file_restart_final_memory_accuracy": masked_accuracy(
            restarted["memory_color_logits"], batch["memory_color_target"][:, restart_tick:], final_mask_tail
        ),
        "zero_reset_final_memory_accuracy": masked_accuracy(
            reset["memory_color_logits"], batch["memory_color_target"][:, restart_tick:], final_mask_tail
        ),
        "normal_final_object_pos_accuracy": masked_accuracy(
            normal["world_pos_logits"], batch["world_pos_target"], final_mask_full
        ),
        "memory_file_restart_final_object_pos_accuracy": masked_accuracy(
            restarted["world_pos_logits"], batch["world_pos_target"][:, restart_tick:], final_mask_tail
        ),
        "zero_reset_final_object_pos_accuracy": masked_accuracy(
            reset["world_pos_logits"], batch["world_pos_target"][:, restart_tick:], final_mask_tail
        ),
    }


def idle_mode_eval(
    model: RecurrentLatentModel,
    batch_size: int,
    seq_len: int,
    seed: int,
    device: str = "cpu",
) -> Dict[str, float]:
    base = generate_batch(batch_size, seq_len, seed, device=device)
    idle = blank_training_batch(base, blank_after=16)
    with torch.no_grad():
        outputs = run_autoregressive_private(model, idle)
    final_mask = torch.zeros_like(base["delayed_memory_mask"])
    final_mask[:, -1] = True
    lang_tokens = outputs["language_logits"][:, 16:, :].argmax(dim=-1)
    private_tokens = outputs["generated_private"][:, 16:]
    latent_stats = latent_noncollapse_stats(outputs["latents"][:, 16:, :], lang_tokens)
    current_pos = base["sensory"][:, -1, SENSOR_POS].argmax(dim=-1)
    target_pos = base["world_pos_target"][:, -1]
    target_action = shortest_action(current_pos, target_pos)
    final_action = outputs["action_logits"][:, -1, :].argmax(dim=-1)
    return {
        "idle_ticks": float(seq_len - 16),
        "final_memory_accuracy": masked_accuracy(outputs["memory_color_logits"], base["memory_color_target"], final_mask),
        "final_object_pos_accuracy": masked_accuracy(outputs["world_pos_logits"], base["world_pos_target"], final_mask),
        "endogenous_goal_action_accuracy": float((final_action == target_action).float().mean().item()),
        "public_language_repetition_ratio": latent_stats["language_repetition_ratio"],
        "private_token_repetition_ratio": (
            float((private_tokens[:, 1:] == private_tokens[:, :-1]).float().mean().item())
            if private_tokens.shape[1] > 1
            else 0.0
        ),
        "private_token_unique_count": float(private_tokens.unique().numel()),
        "latent_active_fraction": latent_stats["latent_active_fraction"],
        "latent_effective_rank": latent_stats["latent_effective_rank"],
        "latent_max_quantized_fraction": latent_stats["latent_max_quantized_fraction"],
    }


def richer_dynamics_eval() -> Dict[str, float]:
    world = TinyWorldRuntime(seed=6100)
    world.energy = 0.04
    world.damage = 0.10
    start_pos = world.current_pos
    start_damage = world.damage
    world.step(ACTION_LEFT)
    failed_no_move = float(world.current_pos == start_pos and world.failed_actions == 1)
    failed_damage_delta = world.damage - start_damage

    world.energy = 0.20
    world.damage = 0.72
    damage_before_rest = world.damage
    energy_before_rest = world.energy
    world.step(ACTION_REST)
    rest_healed = float(world.damage < damage_before_rest and world.energy > energy_before_rest)

    world.energy = 0.20
    world.resource = 0.10
    energy_before_forage = world.energy
    resource_before_forage = world.resource
    world.step(ACTION_FORAGE)
    forage_gained = float(world.energy > energy_before_forage and world.resource > resource_before_forage)

    world.current_pos = world.hazard_pos
    damage_before_hazard = world.damage
    world.step(ACTION_STAY)
    hazard_damaged = float(world.damage > damage_before_hazard and world.damage_events > 0)

    return {
        "failed_action_no_move": failed_no_move,
        "failed_action_damage_delta": failed_damage_delta,
        "rest_healed_and_restored_energy": rest_healed,
        "forage_restored_energy_and_resource": forage_gained,
        "hazard_caused_damage": hazard_damaged,
    }


def private_language_eval(
    model: RecurrentLatentModel,
    batch_size: int,
    seq_len: int,
    seed: int,
    device: str = "cpu",
) -> Dict[str, float]:
    batch = generate_batch(batch_size, seq_len, seed, device=device)
    zero_private = {key: value.clone() for key, value in batch.items()}
    zero_private["private_in"].zero_()
    with torch.no_grad():
        feedback = run_autoregressive_private(model, batch)
        no_private = model(zero_private["sensory"], zero_private["lang_in"], zero_private["private_in"])
    action_shift = (
        feedback["action_logits"][:, :, :].argmax(dim=-1) != no_private["action_logits"].argmax(dim=-1)
    ).float()
    language_shift = (
        feedback["language_logits"].argmax(dim=-1) != no_private["language_logits"].argmax(dim=-1)
    ).float()
    private_tokens = feedback["generated_private"]
    return {
        "generated_private_unique_count": float(private_tokens.unique().numel()),
        "generated_private_repetition_ratio": (
            float((private_tokens[:, 1:] == private_tokens[:, :-1]).float().mean().item())
            if private_tokens.shape[1] > 1
            else 0.0
        ),
        "private_channel_action_shift_rate": float(action_shift.mean().item()),
        "private_channel_language_shift_rate": float(language_shift.mean().item()),
    }


def evaluate_living_system(
    checkpoint: str | Path,
    config_name: str = "fast",
    device: str = "cpu",
    curriculum_output: str | Path = "runs/curriculum_latest.pt",
    json_output: str | Path | None = None,
) -> Dict[str, object]:
    cfg = LIVING_CONFIGS[config_name]
    model = load_checkpoint(checkpoint, device=device)
    model.eval()
    batch_size = int(cfg["batch_size"])
    seq_len = int(cfg["seq_len"])
    seed = int(cfg["seed"])
    report: Dict[str, object] = {
        "durable_restart": durable_restart_eval(model, batch_size, seq_len, seed, device=device),
        "idle_mode": idle_mode_eval(model, batch_size, 16 + int(cfg["idle_ticks"]), seed + 1000, device=device),
        "richer_dynamics": richer_dynamics_eval(),
        "private_internal_language": private_language_eval(model, batch_size, seq_len, seed + 2000, device=device),
        "curriculum_growth": acquire_new_concept(
            checkpoint=checkpoint,
            output=curriculum_output,
            concept_id=1,
            steps=int(cfg["curriculum_steps"]),
            device=device,
        ),
    }
    report["verdict"] = living_verdict(report)
    if json_output is not None:
        path = Path(json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2))
    return report


def living_verdict(report: Dict[str, object]) -> Dict[str, object]:
    restart = report["durable_restart"]  # type: ignore[assignment]
    idle = report["idle_mode"]  # type: ignore[assignment]
    dynamics = report["richer_dynamics"]  # type: ignore[assignment]
    private = report["private_internal_language"]  # type: ignore[assignment]
    curriculum = report["curriculum_growth"]  # type: ignore[assignment]
    checks = {
        "memory_file_restart_memory_ge_0_85": restart["memory_file_restart_final_memory_accuracy"] >= 0.85,
        "memory_file_beats_zero_reset_by_0_40": (
            restart["memory_file_restart_final_memory_accuracy"] - restart["zero_reset_final_memory_accuracy"] >= 0.40
        ),
        "idle_public_repetition_lt_0_40": idle["public_language_repetition_ratio"] < 0.40,
        "idle_private_repetition_lt_0_40": idle["private_token_repetition_ratio"] < 0.40,
        "idle_preserves_goal_memory": idle["final_memory_accuracy"] >= 0.85 and idle["final_object_pos_accuracy"] >= 0.85,
        "idle_endogenous_goal_action_ge_0_70": idle["endogenous_goal_action_accuracy"] >= 0.70,
        "richer_dynamics_all_present": all(value > 0.0 for value in dynamics.values()),
        "private_tokens_generated_and_used": (
            private["generated_private_unique_count"] >= 3.0
            and (
                private["private_channel_action_shift_rate"] > 0.0
                or private["private_channel_language_shift_rate"] > 0.0
            )
        ),
        "curriculum_accuracy_after_ge_0_80": curriculum["accuracy_after"] >= 0.80,
        "curriculum_improves_by_0_30": curriculum["accuracy_delta"] >= 0.30,
    }
    return {"passes": all(checks.values()), "checks": checks}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", choices=sorted(LIVING_CONFIGS), default="fast")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--curriculum-output", default="runs/curriculum_latest.pt")
    parser.add_argument("--json-output", default=None)
    args = parser.parse_args()
    evaluate_living_system(
        args.checkpoint,
        args.config,
        args.device,
        args.curriculum_output,
        args.json_output,
    )


if __name__ == "__main__":
    main()
