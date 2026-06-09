from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Tuple

import torch
from torch import nn

from .env import (
    GRID_SIZE,
    NUM_ACTIONS,
    NUM_COLORS,
    NUM_INPUT_TOKENS,
    NUM_LANGUAGE_TOKENS,
    NUM_PROVENANCE,
    SENSOR_DIM,
)


@dataclass
class ModelConfig:
    sensor_dim: int = SENSOR_DIM
    input_tokens: int = NUM_INPUT_TOKENS
    language_tokens: int = NUM_LANGUAGE_TOKENS
    hidden_dim: int = 64
    embed_dim: int = 16


class RecurrentLatentModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.sensor_encoder = nn.Sequential(
            nn.Linear(self.config.sensor_dim, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
        )
        self.token_embedding = nn.Embedding(self.config.input_tokens, self.config.embed_dim)
        self.input_mixer = nn.Sequential(
            nn.Linear(48 + self.config.embed_dim, self.config.hidden_dim),
            nn.Tanh(),
        )
        self.core = nn.GRUCell(self.config.hidden_dim, self.config.hidden_dim)
        self.norm = nn.LayerNorm(self.config.hidden_dim)
        self.action_head = nn.Linear(self.config.hidden_dim, NUM_ACTIONS)
        self.language_head = nn.Linear(self.config.hidden_dim, self.config.language_tokens)
        self.provenance_head = nn.Linear(self.config.hidden_dim, NUM_PROVENANCE)
        self.world_color_head = nn.Linear(self.config.hidden_dim, NUM_COLORS)
        self.world_pos_head = nn.Linear(self.config.hidden_dim, GRID_SIZE)
        self.memory_color_head = nn.Linear(self.config.hidden_dim, NUM_COLORS)
        self.self_start_head = nn.Linear(self.config.hidden_dim, GRID_SIZE)

    def initial_state(self, batch_size: int, device: torch.device | str = "cpu") -> torch.Tensor:
        return torch.zeros(batch_size, self.config.hidden_dim, device=device)

    def step(
        self, observation: Dict[str, torch.Tensor], z_prev: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        sensory = observation["sensory"]
        lang_in = observation["lang_in"]
        sensor_features = self.sensor_encoder(sensory)
        token_features = self.token_embedding(lang_in)
        mixed = self.input_mixer(torch.cat([sensor_features, token_features], dim=-1))
        z_next = self.core(mixed, z_prev)
        z_view = self.norm(z_next)
        output = {
            "action_logits": self.action_head(z_view),
            "language_logits": self.language_head(z_view),
            "provenance_logits": self.provenance_head(z_view),
            "world_color_logits": self.world_color_head(z_view),
            "world_pos_logits": self.world_pos_head(z_view),
            "memory_color_logits": self.memory_color_head(z_view),
            "self_start_logits": self.self_start_head(z_view),
        }
        return output, z_next

    def forward(self, sensory: torch.Tensor, lang_in: torch.Tensor) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = sensory.shape
        z = self.initial_state(batch_size, sensory.device)
        outputs = []
        latents = []
        for tick in range(seq_len):
            output, z = self.step({"sensory": sensory[:, tick], "lang_in": lang_in[:, tick]}, z)
            outputs.append(output)
            latents.append(z)
        stacked: Dict[str, torch.Tensor] = {}
        for key in outputs[0]:
            stacked[key] = torch.stack([item[key] for item in outputs], dim=1)
        stacked["latents"] = torch.stack(latents, dim=1)
        return stacked


def save_checkpoint(
    path: str | Path,
    model: RecurrentLatentModel,
    train_config: Dict[str, object],
    metrics: Dict[str, float] | None = None,
) -> None:
    payload = {
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "train_config": train_config,
        "metrics": metrics or {},
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: torch.device | str = "cpu") -> RecurrentLatentModel:
    payload = torch.load(Path(path), map_location=device)
    config = ModelConfig(**payload["model_config"])
    model = RecurrentLatentModel(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model
