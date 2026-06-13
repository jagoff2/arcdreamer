from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from .action_selection import select_experimental_action
from .adaptation import fast_adaptation_summary, model_state_fingerprint
from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .env import NUM_ACTIONS, TinyWorldRuntime
from .metrics import latent_noncollapse_stats
from .model import decode_program_proposals, load_checkpoint
from .persistent_memory import PersistentMemoryState


def _prediction_error(output: Dict[str, torch.Tensor], *, action: int | str | None = None) -> Dict[str, object]:
    errors: Dict[str, object] = {}
    for key, logits in output.items():
        if not key.endswith("_logits"):
            continue
        probs = torch.softmax(logits.detach().float(), dim=-1)
        confidence = probs.max(dim=-1).values
        entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum(dim=-1)
        stem = key[: -len("_logits")]
        errors[f"{stem}_uncertainty"] = float((1.0 - confidence).mean().item())
        errors[f"{stem}_entropy"] = float(entropy.mean().item())
    if action is not None:
        errors["neural_program_proposals"] = decode_program_proposals(output, action=action)
    return errors


def _transition_metadata(
    world: TinyWorldRuntime,
    *,
    before_global_tick: int,
    before_local_tick: int,
    external_action_index: int,
    action_selection: Dict[str, object] | None = None,
) -> Dict[str, object]:
    boundary = world.local_tick == 0 and world.global_tick != before_global_tick
    return {
        "global_tick_before": int(before_global_tick),
        "local_tick_before": int(before_local_tick),
        "global_tick_after": int(world.global_tick),
        "local_tick_after": int(world.local_tick),
        "external_action_index": int(external_action_index),
        "available_action_mask": [True for _ in range(NUM_ACTIONS)],
        "score_delta": 0.0,
        "terminal": bool(boundary),
        "boundary": "episode" if boundary else None,
        "action_selection": action_selection or {},
    }


def run_unbroken(
    checkpoint: str | Path,
    max_ticks: int = 100000,
    log_every: int = 10000,
    seed: int = 900000,
    device: DeviceLike = AUTO_DEVICE,
    memory_file: str | Path | None = None,
) -> Dict[str, float]:
    target_device = resolve_device(device)
    model = load_checkpoint(checkpoint, device=target_device)
    model.eval()
    base_fingerprint_before = model_state_fingerprint(model)
    world = TinyWorldRuntime(seed=seed)
    if memory_file is not None:
        memory = PersistentMemoryState.load(memory_file, model.config.hidden_dim, device=target_device)
    else:
        memory = PersistentMemoryState.fresh(model.config.hidden_dim, device=target_device)
    z = memory.latent.clone()
    private_token = memory.private_token.clone()
    start_tick = int(memory.tick)
    latents = []
    language_tokens = []
    private_tokens = []
    previous_sensory = None
    previous_action = NUM_ACTIONS

    with torch.no_grad():
        for tick in range(max_ticks):
            event_tick = start_tick + tick + 1
            before_global_tick = world.global_tick
            before_local_tick = world.local_tick
            observation = world.observation(device=target_device, private_in=int(private_token.item()))
            current_sensory = observation["sensory"]
            if previous_sensory is None:
                prev_delta = torch.zeros_like(current_sensory)
            else:
                prev_delta = current_sensory - previous_sensory
            model_observation = {
                **observation,
                "prev_action": torch.tensor([previous_action], dtype=torch.long, device=target_device),
                "prev_delta": prev_delta,
                "dt": torch.ones(1, dtype=torch.float32, device=target_device),
            }
            output, z = model.step(model_observation, z)
            available_action_mask = [True for _ in range(NUM_ACTIONS)]
            action, action_selection = select_experimental_action(
                output["action_logits"].squeeze(0),
                memory,
                observation,
                available_action_mask=available_action_mask,
            )
            language_token = output["language_logits"].argmax(dim=-1)
            language = int(language_token.item())
            next_private_token = output["private_logits"].argmax(dim=-1)
            latents.append(z.squeeze(0).detach())
            language_tokens.append(language_token.squeeze(0).detach())
            private_tokens.append(next_private_token.squeeze(0).detach())
            world.step(action)
            next_observation = world.observation(device=target_device, private_in=int(next_private_token.item()))
            memory.append_event(
                tick=event_tick,
                observation=observation,
                action=action,
                next_observation=next_observation,
                metadata=_transition_metadata(
                    world,
                    before_global_tick=before_global_tick,
                    before_local_tick=before_local_tick,
                    external_action_index=tick + 1,
                    action_selection=action_selection,
                ),
                prediction_error=_prediction_error(output, action=action),
            )
            memory.update(z, next_private_token, event_tick)
            private_token = next_private_token
            previous_sensory = current_sensory.detach()
            previous_action = action
            if log_every > 0 and (tick + 1) % log_every == 0:
                print(
                    json.dumps(
                        {
                            "tick": tick + 1,
                            "z_norm": round(float(z.norm().item()), 6),
                            "action": action,
                            "language": language,
                            "private": int(private_token.item()),
                            "event_journal_length": len(memory.event_journal),
                        }
                    )
                )
    if memory_file is not None:
        memory.save(memory_file)

    latent_tensor = torch.stack(latents, dim=0)
    language_tensor = torch.stack(language_tokens, dim=0).long()
    private_tensor = torch.stack(private_tokens, dim=0).long()
    stats = latent_noncollapse_stats(latent_tensor, language_tensor)
    stats["private_repetition_ratio"] = (
        float((private_tensor[1:] == private_tensor[:-1]).float().mean().item())
        if private_tensor.numel() > 1
        else 0.0
    )
    stats["unbroken_ticks"] = float(max_ticks)
    stats["requested_ticks"] = float(max_ticks)
    stats["external_action_count"] = float(max_ticks)
    stats["latent_dim"] = float(latent_tensor.shape[-1])
    stats["event_journal_length"] = float(len(memory.event_journal))
    stats["base_weights_unchanged"] = float(base_fingerprint_before == model_state_fingerprint(model))
    stats.update(fast_adaptation_summary({"plastic_memory": memory.plastic_memory}))
    print(json.dumps({"final_runtime": stats}, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-ticks", type=int, default=100000)
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--device", default=AUTO_DEVICE)
    parser.add_argument("--memory-file", default=None)
    args = parser.parse_args()
    run_unbroken(args.checkpoint, args.max_ticks, args.log_every, args.seed, args.device, args.memory_file)


if __name__ == "__main__":
    main()
