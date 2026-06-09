from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from .env import TinyWorldRuntime
from .metrics import latent_noncollapse_stats
from .model import load_checkpoint


def run_unbroken(
    checkpoint: str | Path,
    max_ticks: int = 100000,
    log_every: int = 10000,
    seed: int = 900000,
    device: str = "cpu",
) -> Dict[str, float]:
    model = load_checkpoint(checkpoint, device=device)
    model.eval()
    world = TinyWorldRuntime(seed=seed)
    z = model.initial_state(1, device=device)
    latents = []
    language_tokens = []

    with torch.no_grad():
        for tick in range(max_ticks):
            observation = world.observation(device=device)
            output, z = model.step(observation, z)
            action = int(output["action_logits"].argmax(dim=-1).item())
            language = int(output["language_logits"].argmax(dim=-1).item())
            latents.append(z.squeeze(0).detach().cpu())
            language_tokens.append(language)
            world.step(action)
            if log_every > 0 and (tick + 1) % log_every == 0:
                print(
                    json.dumps(
                        {
                            "tick": tick + 1,
                            "z_norm": round(float(z.norm().item()), 6),
                            "action": action,
                            "language": language,
                        }
                    )
                )

    latent_tensor = torch.stack(latents, dim=0)
    language_tensor = torch.tensor(language_tokens, dtype=torch.long)
    stats = latent_noncollapse_stats(latent_tensor, language_tensor)
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
    args = parser.parse_args()
    run_unbroken(args.checkpoint, args.max_ticks, args.log_every, args.seed, args.device)


if __name__ == "__main__":
    main()
