from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from torch import nn

from arcagi.core.types import StructuredState

MAX_OBJECTS = 64
OBJECT_FEATURES = 12
SUMMARY_FEATURES = 25
SPATIAL_FEATURES = 18
MAX_GRID_SIZE = 32
GRID_EMBED_DIM = 16
GRID_VALUE_BUCKETS = 256


@dataclass(frozen=True)
class EncodedState:
    latent: torch.Tensor
    object_rows: torch.Tensor
    summary: torch.Tensor
    spatial: torch.Tensor
    grid: torch.Tensor


def state_to_arrays(state: StructuredState) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    return (
        state.object_feature_rows(MAX_OBJECTS),
        state.summary_vector(),
        state.spatial_vector(),
        state_to_grid_array(state),
    )


def state_to_grid_array(state: StructuredState, max_grid_size: int = MAX_GRID_SIZE) -> np.ndarray:
    grid = state.as_grid().astype(np.int64)
    clipped_height = min(grid.shape[0], max_grid_size)
    clipped_width = min(grid.shape[1], max_grid_size)
    padded = np.zeros((max_grid_size, max_grid_size), dtype=np.int64)
    padded[:clipped_height, :clipped_width] = grid[:clipped_height, :clipped_width]
    return padded


class StructuredStateEncoder(nn.Module):
    def __init__(self, latent_dim: int = 128, hidden_dim: int = 128) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.object_mlp = nn.Sequential(
            nn.Linear(OBJECT_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.summary_mlp = nn.Sequential(
            nn.Linear(SUMMARY_FEATURES, hidden_dim),
            nn.ReLU(),
        )
        self.spatial_mlp = nn.Sequential(
            nn.Linear(SPATIAL_FEATURES, hidden_dim),
            nn.ReLU(),
        )
        self.output_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.object_attention = nn.MultiheadAttention(hidden_dim, num_heads=4, batch_first=True)
        self.object_query = nn.Parameter(torch.zeros(1, 1, hidden_dim))
        self.grid_embedding = nn.Embedding(GRID_VALUE_BUCKETS, GRID_EMBED_DIM)
        self.grid_conv = nn.Sequential(
            nn.Conv2d(GRID_EMBED_DIM, hidden_dim // 2, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_dim // 2, hidden_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d(1),
        )
        self.enhancement_mlp = nn.Sequential(
            nn.Linear(hidden_dim * 4, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, latent_dim),
        )
        self.enhancement_gate = nn.Parameter(torch.tensor(0.1))
        nn.init.normal_(self.object_query, mean=0.0, std=0.02)

    def forward(
        self,
        object_rows: torch.Tensor,
        summary: torch.Tensor,
        spatial: torch.Tensor,
        grid: torch.Tensor | None = None,
    ) -> torch.Tensor:
        object_embeds = self.object_mlp(object_rows)
        mask = object_rows.abs().sum(dim=-1) > 0
        masked = object_embeds * mask.unsqueeze(-1)
        counts = mask.sum(dim=1, keepdim=True).clamp(min=1)
        pooled_mean = masked.sum(dim=1) / counts
        pooled_max = object_embeds.masked_fill(~mask.unsqueeze(-1), -1e9).max(dim=1).values
        pooled_max = torch.where(mask.any(dim=1, keepdim=True), pooled_max, torch.zeros_like(pooled_max))
        summary_embed = self.summary_mlp(summary)
        spatial_embed = self.spatial_mlp(spatial)
        base_latent = self.output_mlp(torch.cat([pooled_mean, pooled_max, summary_embed, spatial_embed], dim=-1))
        if grid is None:
            return base_latent
        object_context = torch.zeros_like(summary_embed)
        if mask.any():
            query = self.object_query.expand(object_rows.shape[0], -1, -1)
            attention_output, _ = self.object_attention(
                query,
                object_embeds,
                object_embeds,
                key_padding_mask=~mask,
                need_weights=False,
            )
            object_context = attention_output.squeeze(1)
        grid_embeds = self.grid_embedding(grid.long().remainder(GRID_VALUE_BUCKETS)).permute(0, 3, 1, 2)
        grid_context = self.grid_conv(grid_embeds).flatten(1)
        enhancement = self.enhancement_mlp(torch.cat([object_context, summary_embed, spatial_embed, grid_context], dim=-1))
        return base_latent + (torch.tanh(self.enhancement_gate) * enhancement)

    def encode_state(self, state: StructuredState, device: torch.device | None = None) -> EncodedState:
        object_rows, summary, spatial, grid = state_to_arrays(state)
        object_tensor = torch.tensor(object_rows, dtype=torch.float32, device=device).unsqueeze(0)
        summary_tensor = torch.tensor(summary, dtype=torch.float32, device=device).unsqueeze(0)
        spatial_tensor = torch.tensor(spatial, dtype=torch.float32, device=device).unsqueeze(0)
        grid_tensor = torch.tensor(grid, dtype=torch.long, device=device).unsqueeze(0)
        latent = self.forward(object_tensor, summary_tensor, spatial_tensor, grid_tensor)
        return EncodedState(
            latent=latent,
            object_rows=object_tensor,
            summary=summary_tensor,
            spatial=spatial_tensor,
            grid=grid_tensor,
        )
