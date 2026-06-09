from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from .env import TinyWorldRuntime
from .metrics import latent_noncollapse_stats
from .model import load_checkpoint
from .persistent_memory import PersistentMemoryState


def run_unbroken(
    checkpoint: str | Path,
    max_ticks: int = 100000,
    log_every: int = 10000,
    seed: int = 900000,
    device: str = "cpu",
    memory_file: str | Path | None = None,
) -> Dict[str, float]:
    model = load_checkpoint(checkpoint, device=device)
    model.eval()
    world = TinyWorldRuntime(seed=seed)
    if memory_file is not None:
        memory = PersistentMemoryState.load(memory_file, model.config.hidden_dim, device=device)
        z = memory.latent.clone()
        private_token = memory.private_token.clone()
    else:
        memory = None
        z = model.initial_state(1, device=device)
        private_token = torch.zeros(1, dtype=torch.long, device=device)
    latents = []
    language_tokens = []
    private_tokens = []

    with torch.no_grad():
        for tick in range(max_ticks):
            observation = world.observation(device=device, private_in=int(private_token.item()))
            output, z = model.step(observation, z)
            action = int(output["action_logits"].argmax(dim=-1).item())
            language = int(output["language_logits"].argmax(dim=-1).item())
            private_token = output["private_logits"].argmax(dim=-1)
            latents.append(z.squeeze(0).detach().cpu())
            language_tokens.append(language)
            private_tokens.append(int(private_token.item()))
            if memory is not None:
                memory.update(z, private_token, tick + 1)
            world.step(action)
            if log_every > 0 and (tick + 1) % log_every == 0:
                print(
                    json.dumps(
                        {
                            "tick": tick + 1,
                            "z_norm": round(float(z.norm().item()), 6),
                            "action": action,
                            "language": language,
                            "private": int(private_token.item()),
                        }
                    )
                )
    if memory is not None and memory_file is not None:
        memory.save(memory_file)

    latent_tensor = torch.stack(latents, dim=0)
    language_tensor = torch.tensor(language_tokens, dtype=torch.long)
    private_tensor = torch.tensor(private_tokens, dtype=torch.long)
    stats = latent_noncollapse_stats(latent_tensor, language_tensor)
    stats["private_repetition_ratio"] = (
        float((private_tensor[1:] == private_tensor[:-1]).float().mean().item())
        if private_tensor.numel() > 1
        else 0.0
    )
    stats["unbroken_ticks"] = float(max_ticks)
    stats["requested_ticks"] = float(max_ticks)
    stats["latent_dim"] = float(latent_tensor.shape[-1])
    print(json.dumps({"final_runtime": stats}, indent=2))
    return stats


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--max-ticks", type=int, default=100000)
    parser.add_argument("--log-every", type=int, default=10000)
    parser.add_argument("--seed", type=int, default=900000)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--memory-file", default=None)
    args = parser.parse_args()
    run_unbroken(args.checkpoint, args.max_ticks, args.log_every, args.seed, args.device, args.memory_file)


if __name__ == "__main__":
    main()
