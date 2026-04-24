from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from arcagi.core.types import StructuredState, Transition


@dataclass(frozen=True)
class TransitionLabels:
    visible_change: float = 0.0
    objective_progress: float = 0.0
    reward_progress: float = 0.0
    terminal_progress: float = 0.0
    action_availability_changed: float = 0.0
    appeared_or_disappeared: float = 0.0
    mechanic_change: float = 0.0
    no_effect_nonprogress: float = 0.0
    visible_only_nonprogress: float = 0.0
    harm: float = 0.0

    @property
    def useful_change(self) -> float:
        return float(max(self.objective_progress, self.mechanic_change))


def labels_from_transition(transition: Transition) -> TransitionLabels:
    before = transition.state
    after = transition.next_state
    reward = float(transition.reward)
    info = dict(transition.info or {})

    visible_change = _visible_change(before, after)
    before_levels = _levels_completed(before, info_key="levels_completed_before", info=info)
    after_levels = _levels_completed(after, info_key="levels_completed_after", info=info)
    level_delta = max(0, after_levels - before_levels)
    terminal_success = _terminal_success(after, info)
    reward_progress = 1.0 if reward > 0.0 or _positive_number(info.get("score_delta", 0.0)) > 0.0 else 0.0
    objective_progress = 1.0 if reward_progress > 0.0 or level_delta > 0 or terminal_success else 0.0
    terminal_progress = 1.0 if terminal_success else 0.0
    action_availability_changed = 1.0 if set(before.affordances) != set(after.affordances) else 0.0
    appeared_or_disappeared = 1.0 if len(before.objects) != len(after.objects) else 0.0
    mechanic_change = 1.0 if action_availability_changed or appeared_or_disappeared else 0.0
    visible_only_nonprogress = 1.0 if visible_change and not objective_progress and not mechanic_change else 0.0
    no_effect_nonprogress = 1.0 if not visible_change and not objective_progress and not mechanic_change else 0.0
    harm = 1.0 if reward < 0.0 or _terminal_failure(after, info) else 0.0
    return TransitionLabels(
        visible_change=float(visible_change),
        objective_progress=float(objective_progress),
        reward_progress=float(reward_progress),
        terminal_progress=float(terminal_progress),
        action_availability_changed=float(action_availability_changed),
        appeared_or_disappeared=float(appeared_or_disappeared),
        mechanic_change=float(mechanic_change),
        no_effect_nonprogress=float(no_effect_nonprogress),
        visible_only_nonprogress=float(visible_only_nonprogress),
        harm=float(harm),
    )


def _visible_change(before: StructuredState, after: StructuredState) -> bool:
    if before.grid_shape != after.grid_shape:
        return True
    if before.grid_signature != after.grid_signature:
        return True
    return before.inventory != after.inventory or before.flags != after.flags


def _levels_completed(state: StructuredState, *, info_key: str, info: dict[str, Any]) -> int:
    if info_key in info:
        try:
            return int(info.get(info_key, 0) or 0)
        except Exception:
            return 0
    values = dict(state.inventory)
    try:
        return int(values.get("interface_levels_completed", "0") or 0)
    except Exception:
        return 0


def _terminal_success(state: StructuredState, info: dict[str, Any]) -> bool:
    if bool(info.get("won", False) or info.get("success", False) or info.get("terminal_success", False)):
        return True
    values = dict(state.inventory)
    game_state = str(values.get("interface_game_state", info.get("game_state_after", ""))).upper()
    return game_state.endswith("WIN")


def _terminal_failure(state: StructuredState, info: dict[str, Any]) -> bool:
    if bool(info.get("terminal_failure", False)):
        return True
    values = dict(state.inventory)
    game_state = str(values.get("interface_game_state", info.get("game_state_after", ""))).upper()
    return game_state.endswith("GAME_OVER") or game_state.endswith("LOSE")


def _positive_number(value: Any) -> float:
    try:
        parsed = float(value)
    except Exception:
        return 0.0
    if not np.isfinite(parsed):
        return 0.0
    return max(parsed, 0.0)
