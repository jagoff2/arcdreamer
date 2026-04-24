from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np

from arcagi.core.progress_signals import action_family
from arcagi.core.types import ActionName, StructuredState


def _state_delta(before: StructuredState, after: StructuredState) -> float:
    return float(np.linalg.norm(after.transition_vector() - before.transition_vector()))


def _has_visible_active_target(state: StructuredState) -> bool:
    return any("target" in obj.tags and "active" in obj.tags for obj in state.objects)


@dataclass
class InferredStateTracker:
    progress_level: int = 0
    setback_level: int = 0
    contradiction_count: int = 0
    recent_progress: bool = False
    last_effect_family: str = "none"
    mode: str = "explore"

    def reset(self) -> None:
        self.progress_level = 0
        self.setback_level = 0
        self.contradiction_count = 0
        self.recent_progress = False
        self.last_effect_family = "none"
        self.mode = "explore"

    def reset_level(self) -> None:
        self.recent_progress = False
        self.last_effect_family = "none"
        self.mode = "explore"

    def augment(self, state: StructuredState) -> StructuredState:
        inventory = dict(state.inventory)
        flags = dict(state.flags)
        inventory["belief_progress_level"] = str(self.progress_level)
        inventory["belief_setback_level"] = str(self.setback_level)
        inventory["belief_contradiction_count"] = str(self.contradiction_count)
        flags["belief_mode"] = self.mode
        flags["belief_last_effect_family"] = self.last_effect_family
        flags["belief_recent_progress"] = "1" if self.recent_progress else "0"
        return replace(
            state,
            inventory=tuple(sorted((str(key), str(value)) for key, value in inventory.items())),
            flags=tuple(sorted((str(key), str(value)) for key, value in flags.items())),
        )

    def observe_transition(
        self,
        before: StructuredState,
        action: ActionName,
        reward: float,
        after: StructuredState,
        *,
        terminated: bool,
    ) -> None:
        family = action_family(action)
        delta_norm = _state_delta(before, after)
        before_active = _has_visible_active_target(before)
        after_active = _has_visible_active_target(after)
        strong_positive = (
            (reward >= 0.5)
            or (reward > 0.0 and delta_norm >= 0.25)
            or (after_active and not before_active)
            or (terminated and reward > 0.0)
        )
        positive = strong_positive or reward > 0.0 or delta_norm >= 0.35
        strong_setback = reward <= -0.15
        no_effect = reward <= 0.0 and delta_norm < 0.05

        if strong_positive:
            self.progress_level = min(self.progress_level + 1, 4)
            self.setback_level = max(self.setback_level - 1, 0)
            self.recent_progress = True
            self.last_effect_family = family
            self.mode = "commit"
        elif positive:
            self.progress_level = min(self.progress_level + 1, 4)
            self.setback_level = max(self.setback_level - 1, 0)
            self.recent_progress = True
            self.last_effect_family = family
            if family in {"interact", "click"}:
                self.mode = "probe"
        else:
            self.recent_progress = False

        if strong_setback:
            self.progress_level = max(self.progress_level - 1, 0)
            self.setback_level = min(self.setback_level + 1, 4)
            self.contradiction_count = min(self.contradiction_count + 1, 4)
            self.mode = "explore"
        elif no_effect and family in {"interact", "click"}:
            self.contradiction_count = min(self.contradiction_count + 1, 4)
            if self.contradiction_count >= 2:
                self.mode = "explore"
        elif positive:
            self.contradiction_count = max(self.contradiction_count - 1, 0)

        if after_active or (terminated and reward > 0.0):
            self.progress_level = max(self.progress_level, 2)
            self.recent_progress = True
            self.mode = "commit"
            self.last_effect_family = family
