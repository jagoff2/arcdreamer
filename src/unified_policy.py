from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import torch
import torch.nn.functional as F
from torch import nn

from .explore_env import (
    NUM_EXPLORER_ACTIONS,
    NUM_MIND_ACTS,
    NUM_PARTNERS,
    NUM_PROJECTS,
    NUM_SKILLS,
    NUM_WORLD_STATES,
)


class ChannelSpec(NamedTuple):
    start: int
    end: int


ACTION = ChannelSpec(0, NUM_EXPLORER_ACTIONS)
INSPECT = ChannelSpec(ACTION.end, ACTION.end + NUM_EXPLORER_ACTIONS)
PRIVATE_ACTION = ChannelSpec(INSPECT.end, INSPECT.end + NUM_EXPLORER_ACTIONS)
SPEECH_ACTION = ChannelSpec(PRIVATE_ACTION.end, PRIVATE_ACTION.end + NUM_MIND_ACTS)
MEMORY_STATE = ChannelSpec(SPEECH_ACTION.end, SPEECH_ACTION.end + NUM_WORLD_STATES)
MEMORY_PROJECT = ChannelSpec(MEMORY_STATE.end, MEMORY_STATE.end + NUM_PROJECTS)
MEMORY_PARTNER = ChannelSpec(MEMORY_PROJECT.end, MEMORY_PROJECT.end + NUM_PARTNERS)
MEMORY_SKILL = ChannelSpec(MEMORY_PARTNER.end, MEMORY_PARTNER.end + NUM_SKILLS)
AFFORDANCE_DIM = MEMORY_SKILL.end


CHANNEL_SPECS = {
    "action": ACTION,
    "inspect": INSPECT,
    "private_action": PRIVATE_ACTION,
    "speech_action": SPEECH_ACTION,
    "memory_state": MEMORY_STATE,
    "memory_project": MEMORY_PROJECT,
    "memory_partner": MEMORY_PARTNER,
    "memory_skill": MEMORY_SKILL,
}


@dataclass(frozen=True)
class UnifiedPolicyConfig:
    hidden_dim: int = 128
    affordance_dim: int = AFFORDANCE_DIM


class UnifiedAffordancePolicy(nn.Module):
    """Single causal decoder split into a small set of allowed behavior channels."""

    def __init__(self, config: UnifiedPolicyConfig) -> None:
        super().__init__()
        self.config = config
        self.z_binder = nn.Linear(64, config.hidden_dim)
        self.memory_binder = nn.LazyLinear(config.hidden_dim)
        self.drive_binder = nn.LazyLinear(config.hidden_dim)
        self.memory_to_z = nn.LazyLinear(64)
        self.affordance_decoder = nn.Sequential(
            nn.Linear(config.hidden_dim, config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(config.hidden_dim),
            nn.Linear(config.hidden_dim, config.affordance_dim),
        )

    def forward(
        self,
        shared_state: torch.Tensor,
        z: torch.Tensor,
        memory_state: torch.Tensor,
        drive_state: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        predicted_z = self.memory_to_z(memory_state)
        alignment = z_memory_alignment(z, predicted_z)
        integrity = state_integrity_gate(z, memory_state, drive_state) * alignment
        bound_state = shared_state
        bound_state = bound_state * (1.0 + torch.tanh(self.z_binder(z)))
        bound_state = bound_state * (1.0 + 0.50 * torch.tanh(self.memory_binder(memory_state)))
        bound_state = bound_state * (1.0 + 0.50 * torch.tanh(self.drive_binder(drive_state)))
        raw = self.affordance_decoder(bound_state)
        flat = blend_with_low_integrity_fallback(raw, integrity)
        channels = {
            "affordance_logits": flat,
            "integrity_gate": integrity,
            "z_memory_alignment": alignment,
            "predicted_z": predicted_z,
        }
        for name, spec in CHANNEL_SPECS.items():
            channels[f"{name}_logits"] = flat[:, spec.start : spec.end]
        return channels


def state_integrity_gate(z: torch.Tensor, memory_state: torch.Tensor, drive_state: torch.Tensor) -> torch.Tensor:
    z_score = torch.clamp(z.float().abs().mean(dim=-1, keepdim=True) / 0.20, 0.0, 1.0)
    memory_score = torch.clamp(memory_state.float().abs().mean(dim=-1, keepdim=True) / 0.18, 0.0, 1.0)
    drive_score = torch.clamp(drive_state.float().abs().mean(dim=-1, keepdim=True) / 0.18, 0.0, 1.0)
    return z_score * memory_score * drive_score


def z_memory_alignment(z: torch.Tensor, predicted_z: torch.Tensor) -> torch.Tensor:
    cosine = F.cosine_similarity(z.float(), predicted_z.float(), dim=-1).view(-1, 1)
    return torch.clamp((cosine + 1.0) * 0.5, 0.0, 1.0).pow(2.0)


def blend_with_low_integrity_fallback(logits: torch.Tensor, integrity: torch.Tensor) -> torch.Tensor:
    fallback = torch.zeros_like(logits)
    for spec in CHANNEL_SPECS.values():
        fallback[:, spec.start] = 5.0
    return logits * integrity + fallback * (1.0 - integrity)


def fused_memory_state(tensors: dict[str, torch.Tensor]) -> torch.Tensor:
    return torch.cat(
        [
            tensors["hypothesis"],
            tensors["skill_memory"],
            tensors["project_state"],
            tensors["social_state"],
        ],
        dim=-1,
    )
