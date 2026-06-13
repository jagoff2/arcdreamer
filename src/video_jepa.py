from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from .attempt_buffer import ACTION_BUCKETS
from .base_world_model import ACTION_FEATURE_DIM, GRID_SIZE, NUM_CELLS


@dataclass
class VideoJEPAConfig:
    grid_size: int = GRID_SIZE
    action_buckets: int = ACTION_BUCKETS
    hidden_dim: int = 96
    latent_dim: int = 48
    action_dim: int = 24
    action_feature_dim: int = ACTION_FEATURE_DIM
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
        self.action_feature_encoder = nn.Sequential(
            nn.Linear(self.config.action_feature_dim, self.config.action_dim),
            nn.GELU(),
        )
        self.legal_encoder = nn.Sequential(nn.Linear(1, self.config.action_dim), nn.Tanh())
        self.input_projection = nn.Sequential(
            nn.Linear(self.config.latent_dim + self.config.action_dim * 3, self.config.hidden_dim),
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
        action_features: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        if frames.dim() == 2:
            frames = frames.unsqueeze(0)
            action_ids = action_ids.unsqueeze(0)
            legal_counts = legal_counts.unsqueeze(0)
            if action_features is not None:
                action_features = action_features.unsqueeze(0)
            if valid is not None:
                valid = valid.unsqueeze(0)
        frame_latent = self.encode_frames(frames)
        action_latent = self.action_marker(action_ids.clamp(0, self.config.action_buckets - 1))
        if action_features is None:
            action_features = torch.zeros(
                (*action_ids.shape, self.config.action_feature_dim),
                dtype=frames.dtype,
                device=frames.device,
            )
        action_feature_latent = self.action_feature_encoder(action_features.float())
        legal = legal_counts.float().clamp(0.0, float(self.config.max_legal_actions)) / float(self.config.max_legal_actions)
        legal_latent = self.legal_encoder(legal)
        sequence = self.input_projection(torch.cat([frame_latent, action_latent, action_feature_latent, legal_latent], dim=-1))
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
        output = self(
            batch["frames"],
            batch["action_ids"],
            batch["legal_counts"],
            batch.get("valid"),
            batch.get("action_features"),
        )
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
    loss_mask = output["loss_mask"]
    if loss_mask.ndim == prediction.ndim - 1:
        loss_mask = loss_mask.unsqueeze(-1)
    mask = loss_mask.expand_as(prediction)
    squared = (prediction - target).pow(2) * mask
    return squared.sum() / mask.sum().clamp_min(1.0)


def null_future_loss(output: dict[str, torch.Tensor]) -> torch.Tensor:
    target = output["target_future"]
    if target.numel() == 0:
        return target.sum() * 0.0
    loss_mask = output["loss_mask"]
    if loss_mask.ndim == target.ndim - 1:
        loss_mask = loss_mask.unsqueeze(-1)
    mask = loss_mask.expand_as(target)
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


def synthetic_latent_video_batch(
    batch_size: int,
    seq_len: int,
    *,
    seed: int = 0,
    device: torch.device | str = "cpu",
    config: VideoJEPAConfig | None = None,
) -> dict[str, torch.Tensor]:
    cfg = config or VideoJEPAConfig()
    target_device = torch.device(device)
    generator = torch.Generator(device=target_device)
    generator.manual_seed(int(seed))
    grid_size = int(cfg.grid_size)
    y = torch.randint(0, grid_size, (batch_size,), generator=generator, device=target_device)
    x = torch.randint(0, grid_size, (batch_size,), generator=generator, device=target_device)
    goal_y = torch.randint(0, grid_size, (batch_size,), generator=generator, device=target_device)
    goal_x = torch.randint(0, grid_size, (batch_size,), generator=generator, device=target_device)
    action_ids = torch.zeros(batch_size, seq_len, dtype=torch.long, device=target_device)
    legal_counts = torch.full((batch_size, seq_len, 1), 5.0, dtype=torch.float32, device=target_device)
    valid = torch.ones(batch_size, seq_len, 1, dtype=torch.bool, device=target_device)
    action_features = torch.zeros(
        batch_size,
        seq_len,
        int(cfg.action_feature_dim),
        dtype=torch.float32,
        device=target_device,
    )
    frames = torch.zeros(
        batch_size,
        seq_len,
        grid_size * grid_size,
        dtype=torch.float32,
        device=target_device,
    )
    for tick in range(seq_len):
        frame = torch.zeros(batch_size, grid_size, grid_size, dtype=torch.float32, device=target_device)
        frame[torch.arange(batch_size, device=target_device), goal_y, goal_x] = 7.0
        frame[torch.arange(batch_size, device=target_device), y, x] = 2.0
        frames[:, tick] = frame.reshape(batch_size, -1)
        dy = torch.sign((goal_y - y).float()).long()
        dx = torch.sign((goal_x - x).float()).long()
        move_vertical = dy != 0
        action = torch.full((batch_size,), 4, dtype=torch.long, device=target_device)
        action = torch.where(move_vertical & (dy < 0), torch.zeros_like(action), action)
        action = torch.where(move_vertical & (dy > 0), torch.ones_like(action), action)
        action = torch.where((~move_vertical) & (dx < 0), torch.full_like(action, 2), action)
        action = torch.where((~move_vertical) & (dx > 0), torch.full_like(action, 3), action)
        random_action = torch.randint(0, 5, (batch_size,), generator=generator, device=target_device)
        explore = torch.rand(batch_size, generator=generator, device=target_device) < 0.15
        action = torch.where(explore, random_action, action)
        action_ids[:, tick] = action.clamp(0, int(cfg.action_buckets) - 1)
        feature_limit = min(5, int(cfg.action_feature_dim))
        action_features[:, tick, :feature_limit] = F.one_hot(action.clamp(0, 4), 5).float()[:, :feature_limit]
        action_features[:, tick, 5:9] = torch.stack(
            [
                y.float() / max(grid_size - 1, 1),
                x.float() / max(grid_size - 1, 1),
                goal_y.float() / max(grid_size - 1, 1),
                goal_x.float() / max(grid_size - 1, 1),
            ],
            dim=-1,
        )[:, : max(0, min(4, int(cfg.action_feature_dim) - 5))]
        y = torch.where(action == 0, torch.clamp(y - 1, 0, grid_size - 1), y)
        y = torch.where(action == 1, torch.clamp(y + 1, 0, grid_size - 1), y)
        x = torch.where(action == 2, torch.clamp(x - 1, 0, grid_size - 1), x)
        x = torch.where(action == 3, torch.clamp(x + 1, 0, grid_size - 1), x)
    return {
        "frames": frames,
        "action_ids": action_ids,
        "legal_counts": legal_counts,
        "valid": valid,
        "action_features": action_features,
    }


def train_local_latent_jepa_smoke(
    *,
    steps: int = 24,
    batch_size: int = 16,
    seq_len: int = 18,
    seed: int = 0,
    device: torch.device | str = "cpu",
) -> tuple[VideoJEPA, dict[str, Any]]:
    target_device = torch.device(device)
    torch.manual_seed(int(seed))
    config = VideoJEPAConfig(hidden_dim=48, latent_dim=24, action_dim=12)
    model = VideoJEPA(config).to(target_device)
    batch = synthetic_latent_video_batch(
        batch_size=batch_size,
        seq_len=seq_len,
        seed=seed,
        device=target_device,
        config=config,
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    model.train()
    with torch.no_grad():
        initial_output = model(
            batch["frames"],
            batch["action_ids"],
            batch["legal_counts"],
            batch["valid"],
            batch["action_features"],
        )
        initial_loss = float(jepa_loss(initial_output).detach().cpu())
        initial_null = float(null_future_loss(initial_output).detach().cpu())
    tail: list[float] = []
    for _ in range(max(1, int(steps))):
        optimizer.zero_grad(set_to_none=True)
        output = model(
            batch["frames"],
            batch["action_ids"],
            batch["legal_counts"],
            batch["valid"],
            batch["action_features"],
        )
        loss = jepa_loss(output)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        tail.append(float(loss.detach().cpu()))
    model.eval()
    with torch.no_grad():
        final_output = model(
            batch["frames"],
            batch["action_ids"],
            batch["legal_counts"],
            batch["valid"],
            batch["action_features"],
        )
        final_loss = float(jepa_loss(final_output).detach().cpu())
        null_loss = float(null_future_loss(final_output).detach().cpu())
    metrics = {
        "seed": int(seed),
        "steps": float(max(1, int(steps))),
        "batch_size": float(batch_size),
        "seq_len": float(seq_len),
        "initial_loss": initial_loss,
        "final_loss": final_loss,
        "initial_null_loss": initial_null,
        "null_loss": null_loss,
        "loss_improved": bool(final_loss < initial_loss),
        "jepa_beats_null": bool(final_loss < null_loss),
        "uses_pretrained_backbone": False,
        "emits_text": bool(model.emits_text),
        "predicts_actions": False,
        "loss_tail": [round(item, 8) for item in tail[-6:]],
    }
    return model, metrics


def jepa_self_supervision_manifest(model: VideoJEPA, metrics: dict[str, Any]) -> dict[str, Any]:
    return {
        "format": "local_latent_video_jepa_self_supervision_v1",
        "training_source": "synthetic_public_frame_sequences",
        "initialization": "local_random_init",
        "uses_pretrained_backbone": False,
        "emits_text": bool(model.emits_text),
        "predicts": "future latent visual regions",
        "does_not_predict": ["actions", "goals", "labels", "solutions", "text"],
        "metrics": dict(metrics),
    }


def load_video_jepa(path: str, device: torch.device | str = "cpu") -> tuple[VideoJEPA, dict[str, Any]]:
    payload = torch.load(path, map_location=device, weights_only=False)
    config = VideoJEPAConfig(**payload.get("config", {}))
    model = VideoJEPA(config).to(device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model, payload
