from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import torch

from .env import PRIVATE_NONE


@dataclass
class PersistentMemoryState:
    latent: torch.Tensor
    private_token: torch.Tensor
    tick: int = 0

    @classmethod
    def fresh(
        cls,
        hidden_dim: int,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
    ) -> "PersistentMemoryState":
        return cls(
            latent=torch.zeros(batch_size, hidden_dim, device=device),
            private_token=torch.full((batch_size,), PRIVATE_NONE, dtype=torch.long, device=device),
            tick=0,
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        hidden_dim: int,
        batch_size: int = 1,
        device: torch.device | str = "cpu",
    ) -> "PersistentMemoryState":
        path = Path(path)
        if not path.exists():
            return cls.fresh(hidden_dim, batch_size=batch_size, device=device)
        payload = torch.load(path, map_location=device)
        latent = payload["latent"].to(device).float()
        private_token = payload["private_token"].to(device).long()
        if latent.ndim == 1:
            latent = latent.view(1, -1)
        if private_token.ndim == 0:
            private_token = private_token.view(1)
        if latent.shape[-1] != hidden_dim:
            raise ValueError(f"memory latent width {latent.shape[-1]} does not match model hidden_dim {hidden_dim}")
        if latent.shape[0] != batch_size:
            latent = latent[:1].repeat(batch_size, 1)
        if private_token.shape[0] != batch_size:
            private_token = private_token[:1].repeat(batch_size)
        return cls(latent=latent, private_token=private_token, tick=int(payload.get("tick", 0)))

    def update(self, latent: torch.Tensor, private_token: torch.Tensor, tick: int) -> None:
        self.latent = latent.detach().clone()
        self.private_token = private_token.detach().clone().long()
        self.tick = int(tick)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latent": self.latent.detach().cpu(),
                "private_token": self.private_token.detach().cpu(),
                "tick": self.tick,
                "format": "persistent_differentiable_tensor_memory_v1",
            },
            path,
        )
