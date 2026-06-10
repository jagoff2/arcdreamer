from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn

from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .explore_env import (
    HYPOTHESIS_DIM,
    INTRINSIC_DIM,
    NUM_EXPLORER_ACTIONS,
    NUM_MIND_ACTS,
    NUM_PARTNERS,
    NUM_PROJECTS,
    NUM_SKILLS,
    NUM_UNCERTAINTY,
    NUM_WORLD_STATES,
    OBS_DIM,
    PROJECT_STATE_DIM,
    SKILL_MEMORY_DIM,
    SOCIAL_STATE_DIM,
    Z_DIM,
)


@dataclass
class ExplorerModelConfig:
    obs_dim: int = OBS_DIM
    z_dim: int = Z_DIM
    hypothesis_dim: int = HYPOTHESIS_DIM
    skill_memory_dim: int = SKILL_MEMORY_DIM
    project_state_dim: int = PROJECT_STATE_DIM
    social_state_dim: int = SOCIAL_STATE_DIM
    intrinsic_dim: int = INTRINSIC_DIM
    hidden_dim: int = 128
    embed_dim: int = 64


class ExplorerCore(nn.Module):
    def __init__(self, config: ExplorerModelConfig | None = None, device: DeviceLike = AUTO_DEVICE) -> None:
        super().__init__()
        self.config = config or ExplorerModelConfig()
        e = self.config.embed_dim
        self.obs_encoder = nn.Sequential(nn.Linear(self.config.obs_dim, e), nn.GELU())
        self.z_encoder = nn.Sequential(nn.Linear(self.config.z_dim, e), nn.GELU())
        self.hypothesis_encoder = nn.Sequential(nn.Linear(self.config.hypothesis_dim, e), nn.GELU())
        self.skill_encoder = nn.Sequential(nn.Linear(self.config.skill_memory_dim, e), nn.GELU())
        self.project_encoder = nn.Sequential(nn.Linear(self.config.project_state_dim, e), nn.GELU())
        self.social_encoder = nn.Sequential(nn.Linear(self.config.social_state_dim, e), nn.GELU())
        self.intrinsic_encoder = nn.Sequential(nn.Linear(self.config.intrinsic_dim, e), nn.GELU())
        self.trunk = nn.Sequential(
            nn.Linear(e * 7, self.config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_dim),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_dim),
        )
        self.action_head = nn.Linear(self.config.hidden_dim, NUM_EXPLORER_ACTIONS)
        self.planner_head = nn.Linear(self.config.hidden_dim, NUM_EXPLORER_ACTIONS)
        self.counterfactual_head = nn.Linear(self.config.hidden_dim, NUM_EXPLORER_ACTIONS)
        self.next_state_head = nn.Linear(self.config.hidden_dim, NUM_WORLD_STATES)
        self.uncertainty_head = nn.Linear(self.config.hidden_dim, NUM_UNCERTAINTY)
        self.novelty_head = nn.Linear(self.config.hidden_dim, 2)
        self.skill_head = nn.Linear(self.config.hidden_dim, NUM_SKILLS)
        self.project_head = nn.Linear(self.config.hidden_dim, NUM_PROJECTS)
        self.question_head = nn.Linear(self.config.hidden_dim, 2)
        self.conflict_head = nn.Linear(self.config.hidden_dim, 2)
        self.safety_head = nn.Linear(self.config.hidden_dim, 2)
        self.partner_head = nn.Linear(self.config.hidden_dim, NUM_PARTNERS)
        self.mind_head = nn.Linear(self.config.hidden_dim, NUM_MIND_ACTS)
        self.to(resolve_device(device))

    def forward(
        self,
        obs: torch.Tensor,
        z: torch.Tensor,
        hypothesis: torch.Tensor,
        skill_memory: torch.Tensor,
        project_state: torch.Tensor,
        social_state: torch.Tensor,
        intrinsic: torch.Tensor,
        *,
        z_enabled: bool = True,
        hypothesis_enabled: bool = True,
        skill_enabled: bool = True,
        project_enabled: bool = True,
        social_enabled: bool = True,
        intrinsic_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        if not z_enabled:
            z = torch.zeros_like(z)
        if not hypothesis_enabled:
            hypothesis = torch.zeros_like(hypothesis)
        if not skill_enabled:
            skill_memory = torch.zeros_like(skill_memory)
        if not project_enabled:
            project_state = torch.zeros_like(project_state)
        if not social_enabled:
            social_state = torch.zeros_like(social_state)
        if not intrinsic_enabled:
            intrinsic = torch.zeros_like(intrinsic)
        features = torch.cat(
            [
                self.obs_encoder(obs),
                self.z_encoder(z),
                self.hypothesis_encoder(hypothesis),
                self.skill_encoder(skill_memory),
                self.project_encoder(project_state),
                self.social_encoder(social_state),
                self.intrinsic_encoder(intrinsic),
            ],
            dim=-1,
        )
        h = self.trunk(features)
        return {
            "action_logits": self.action_head(h),
            "planner_logits": self.planner_head(h),
            "counterfactual_logits": self.counterfactual_head(h),
            "next_state_logits": self.next_state_head(h),
            "uncertainty_logits": self.uncertainty_head(h),
            "novelty_logits": self.novelty_head(h),
            "skill_logits": self.skill_head(h),
            "project_logits": self.project_head(h),
            "question_logits": self.question_head(h),
            "conflict_logits": self.conflict_head(h),
            "safety_logits": self.safety_head(h),
            "partner_logits": self.partner_head(h),
            "mind_logits": self.mind_head(h),
        }


def explorer_forward(
    model: ExplorerCore,
    tensors: dict[str, torch.Tensor],
    **kwargs: Any,
) -> dict[str, torch.Tensor]:
    return model(
        tensors["obs"],
        tensors["z"],
        tensors["hypothesis"],
        tensors["skill_memory"],
        tensors["project_state"],
        tensors["social_state"],
        tensors["intrinsic"],
        **kwargs,
    )


def save_explorer_checkpoint(
    path: str | Path,
    model: ExplorerCore,
    train_config: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> None:
    payload = {
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "train_config": train_config,
        "metrics": metrics or {},
    }
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_explorer_checkpoint(path: str | Path, device: DeviceLike = AUTO_DEVICE) -> ExplorerCore:
    target_device = resolve_device(device)
    payload = torch.load(Path(path), map_location=target_device)
    model = ExplorerCore(ExplorerModelConfig(**payload["model_config"]), device=target_device)
    model.load_state_dict(payload["model_state"])
    model.eval()
    return model
