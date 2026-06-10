from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch

from .unified_policy import fused_memory_state
from .world_model import ExplorerCore


STRUCTURED_PROBE_KEYS = [
    "action_logits",
    "planner_logits",
    "counterfactual_logits",
    "next_state_logits",
    "uncertainty_logits",
    "novelty_logits",
    "skill_logits",
    "project_logits",
    "question_logits",
    "conflict_logits",
    "safety_logits",
    "partner_logits",
    "mind_logits",
]

CAUSAL_CHANNEL_KEYS = [
    "action_logits",
    "inspect_logits",
    "private_action_logits",
    "speech_action_logits",
    "memory_state_logits",
    "memory_project_logits",
    "memory_partner_logits",
    "memory_skill_logits",
]


@dataclass(frozen=True)
class HeadCollapseTrace:
    causal_decoder: str
    causal_channels: list[str]
    structured_heads_disabled: bool
    probe_modules_present: bool
    old_head_logits_used_for_behavior: bool
    behavior_source: str


class HeadCollapsedExplorer:
    def __init__(
        self,
        base: ExplorerCore,
        *,
        disable_structured_heads: bool = True,
        remove_probe_modules: bool = False,
    ) -> None:
        self.base = base
        self.disable_structured_heads = bool(disable_structured_heads)
        self.remove_probe_modules = bool(remove_probe_modules)

    def forward(self, tensors: dict[str, torch.Tensor], **ablations: bool) -> dict[str, Any]:
        z_enabled = bool(ablations.get("z_enabled", True))
        hypothesis_enabled = bool(ablations.get("hypothesis_enabled", True))
        skill_enabled = bool(ablations.get("skill_enabled", True))
        project_enabled = bool(ablations.get("project_enabled", True))
        social_enabled = bool(ablations.get("social_enabled", True))
        intrinsic_enabled = bool(ablations.get("intrinsic_enabled", True))
        z = tensors["z"] if z_enabled else torch.zeros_like(tensors["z"])
        hypothesis = tensors["hypothesis"] if hypothesis_enabled else torch.zeros_like(tensors["hypothesis"])
        skill_memory = tensors["skill_memory"] if skill_enabled else torch.zeros_like(tensors["skill_memory"])
        project_state = tensors["project_state"] if project_enabled else torch.zeros_like(tensors["project_state"])
        social_state = tensors["social_state"] if social_enabled else torch.zeros_like(tensors["social_state"])
        intrinsic = tensors["intrinsic"] if intrinsic_enabled else torch.zeros_like(tensors["intrinsic"])
        shared_state = self.base.shared_features(
            tensors["obs"],
            z,
            hypothesis,
            skill_memory,
            project_state,
            social_state,
            intrinsic,
        )
        unified = self.base.unified_policy(
            shared_state,
            z,
            fused_memory_state(
                {
                    "hypothesis": hypothesis,
                    "skill_memory": skill_memory,
                    "project_state": project_state,
                    "social_state": social_state,
                }
            ),
            intrinsic,
        )
        probes: dict[str, torch.Tensor] = {}
        if not self.remove_probe_modules and not self.disable_structured_heads:
            probes = self.base.probe_readouts(shared_state)
        return {
            "shared_state": shared_state,
            "unified": unified,
            "probes": probes,
            "behavior": select_behavior_channels(unified),
            "trace": self.trace(),
        }

    def trace(self) -> HeadCollapseTrace:
        return HeadCollapseTrace(
            causal_decoder="UnifiedAffordancePolicy.affordance_decoder",
            causal_channels=list(CAUSAL_CHANNEL_KEYS),
            structured_heads_disabled=self.disable_structured_heads,
            probe_modules_present=not self.remove_probe_modules,
            old_head_logits_used_for_behavior=False,
            behavior_source="unified_affordance_channels_only",
        )


def select_behavior_channels(unified: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        "action": unified["action_logits"].argmax(dim=-1),
        "inspect": unified["inspect_logits"].argmax(dim=-1),
        "private_action": unified["private_action_logits"].argmax(dim=-1),
        "speech_action": unified["speech_action_logits"].argmax(dim=-1),
        "memory_state": unified["memory_state_logits"].argmax(dim=-1),
        "memory_project": unified["memory_project_logits"].argmax(dim=-1),
        "memory_partner": unified["memory_partner_logits"].argmax(dim=-1),
        "memory_skill": unified["memory_skill_logits"].argmax(dim=-1),
    }
