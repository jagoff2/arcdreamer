from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .attempt_buffer import ACTION_BUCKETS
from .base_world_model import GRID_SIZE, NUM_CELLS


@dataclass
class VideoJEPAConfig:
    grid_size: int = GRID_SIZE
    action_buckets: int = ACTION_BUCKETS
    hidden_dim: int = 96
    latent_dim: int = 48
    action_dim: int = 24
    max_legal_actions: int = 256

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class VideoJEPA(nn.Module):
    emits_text = False

    def __init__(self, config: VideoJEPAConfig | None = None) -> None:
        super().__init__()
        self.config = config or VideoJEPAConfig()
        frame_dim = self.config.grid_size * self.config.grid_size
        self.frame_encoder = nn.Sequential(
            nn.Linear(frame_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim),
            nn.LayerNorm(self.config.latent_dim),
        )
        self.action_marker = nn.Embedding(self.config.action_buckets, self.config.action_dim)
        self.legal_encoder = nn.Sequential(nn.Linear(1, self.config.action_dim), nn.Tanh())
        self.input_projection = nn.Sequential(
            nn.Linear(self.config.latent_dim + self.config.action_dim * 2, self.config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_dim),
        )
        self.temporal = nn.GRU(self.config.hidden_dim, self.config.latent_dim, batch_first=True)
        self.predictor = nn.Sequential(
            nn.Linear(self.config.latent_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.Linear(self.config.hidden_dim, self.config.latent_dim),
        )

    def encode_frames(self, frames: torch.Tensor) -> torch.Tensor:
        scale = float(max(NUM_CELLS - 1, 1))
        return self.frame_encoder(frames.float() / scale)

    def forward(
        self,
        frames: torch.Tensor,
        action_ids: torch.Tensor,
        legal_counts: torch.Tensor,
        valid: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if frames.dim() == 2:
            frames = frames.unsqueeze(0)
            action_ids = action_ids.unsqueeze(0)
            legal_counts = legal_counts.unsqueeze(0)
            if valid is not None:
                valid = valid.unsqueeze(0)
        frame_latent = self.encode_frames(frames)
        action_latent = self.action_marker(action_ids.clamp(0, self.config.action_buckets - 1))
        legal = legal_counts.float().clamp(0.0, float(self.config.max_legal_actions)) / float(self.config.max_legal_actions)
        legal_latent = self.legal_encoder(legal)
        sequence = self.input_projection(torch.cat([frame_latent, action_latent, legal_latent], dim=-1))
        context, _ = self.temporal(sequence)
        if context.shape[1] < 2:
            prediction = context[:, :0]
            target = frame_latent[:, :0]
            mask = torch.zeros_like(legal[:, :0])
        else:
            prediction = self.predictor(context[:, :-1])
            target = frame_latent[:, 1:].detach()
            mask = torch.ones_like(legal[:, 1:]) if valid is None else valid[:, 1:].float()
        return {
            "context_tokens": context,
            "frame_tokens": frame_latent,
            "predicted_future": prediction,
            "target_future": target,
            "loss_mask": mask,
        }

    @torch.no_grad()
    def attempt_tokens(self, batch: dict[str, torch.Tensor]) -> torch.Tensor:
        output = self(batch["frames"], batch["action_ids"], batch["legal_counts"], batch.get("valid"))
        valid = batch.get("valid")
        tokens = output["context_tokens"]
        if valid is None:
            return tokens.mean(dim=1)
        weights = valid.float().clamp(0.0, 1.0)
        return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


def jepa_loss(output: dict[str, torch.Tensor]) -> torch.Tensor:
    prediction = output["predicted_future"]
    target = output["target_future"]
    if prediction.numel() == 0:
        return prediction.sum() * 0.0
    mask = output["loss_mask"].expand_as(prediction)
    squared = (prediction - target).pow(2) * mask
    return squared.sum() / mask.sum().clamp_min(1.0)


def null_future_loss(output: dict[str, torch.Tensor]) -> torch.Tensor:
    target = output["target_future"]
    if target.numel() == 0:
        return target.sum() * 0.0
    mask = output["loss_mask"].expand_as(target)
    baseline = target.mean(dim=1, keepdim=True).detach()
    squared = (baseline - target).pow(2) * mask
    return squared.sum() / mask.sum().clamp_min(1.0)


def checkpoint_payload(model: VideoJEPA, metrics: dict[str, Any], manifest: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "action_conditioned_video_jepa_v1",
        "config": model.config.to_dict(),
        "model_state": model.state_dict(),
        "metrics": metrics,
        "manifest": manifest,
        "emits_text": False,
        "predicts": "future latent visual regions",
        "does_not_predict": ["actions", "goals", "labels", "solutions"],
    }


def load_video_jepa(path: str, device: torch.device | str = "cpu") -> tuple[VideoJEPA, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = VideoJEPAConfig(**payload.get("config", {}))
    model = VideoJEPA(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
