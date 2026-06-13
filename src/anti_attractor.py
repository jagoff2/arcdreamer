from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .external_eval import json_safe


PROGRESS_EVENTS = {
    "goal_reached",
    "positive_reward",
    "level_completed",
    "level_changed",
    "win",
    "WIN",
}
HARM_EVENTS = {
    "game_over",
    "hazard_hit",
    "invalid_action",
    "already_terminal",
}
SIGNIFICANT_HARM_REWARD = -0.05


@dataclass(frozen=True)
class ActionEventSignature:
    state_hash: str
    action: str
    action_class: str
    coordinate_class: str
    delta_class: str
    score_delta: float
    level_delta: int
    useful_effect: bool
    harmful_effect: bool

    @property
    def abstract_action(self) -> str:
        if self.action_class == "coordinate":
            return self.coordinate_class
        return self.action_class if self.action_class in {"undo", "wait"} else self.action

    def action_cycle_key(self) -> tuple[str, str]:
        return (self.action_class, self.abstract_action)

    def state_action_cycle_key(self) -> tuple[str, str, str]:
        return (self.state_hash, self.action_class, self.abstract_action)

    def event_cycle_key(self) -> tuple[str, str, str]:
        return (self.action_class, self.abstract_action, self.delta_class)


def action_class(action: str) -> str:
    text = str(action)
    if text == "7":
        return "undo"
    if text in {"1", "2", "3", "4", "5"}:
        return "simple"
    if text.startswith("click:"):
        return "coordinate"
    if text.lower() in {"wait", "noop", "no_op"}:
        return "wait"
    return "other"


def stable_observation_hash(observation: Any) -> str:
    grid = getattr(observation, "grid", None)
    extras = getattr(observation, "extras", {}) or {}
    payload = {
        "grid": np.asarray(grid, dtype=np.int64).tolist() if grid is not None else str(observation),
        "actions": list(getattr(observation, "available_actions", ())),
        "stable_extras": {
            str(key): value
            for key, value in dict(extras).items()
            if str(key)
            not in {
                "score",
                "normalized_score",
                "game_state",
                "levels_completed",
                "win_levels",
                "total_levels_completed",
                "total_levels",
                "terminated",
                "truncated",
            }
        },
    }
    import hashlib

    blob = json.dumps(json_safe(payload), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(blob).hexdigest()[:16]


def parse_click(action: str) -> tuple[int, int] | None:
    parts = str(action).split(":")
    if len(parts) != 3 or parts[0] != "click":
        return None
    try:
        return int(parts[1]), int(parts[2])
    except ValueError:
        return None


def coordinate_equivalence_class(action: str, observation: Any | None = None) -> str:
    click = parse_click(action)
    if click is None:
        return action_class(action)
    x, y = click
    grid = getattr(observation, "grid", None)
    if grid is None:
        bucket_x = max(0, min(7, x // 8))
        bucket_y = max(0, min(7, y // 8))
        return f"click:unknown:bucket:{bucket_x}:{bucket_y}"
    arr = np.asarray(grid, dtype=np.int64)
    if arr.ndim != 2 or arr.size == 0:
        return "click:empty-grid"
    height, width = arr.shape
    clamped_x = max(0, min(width - 1, int(x)))
    clamped_y = max(0, min(height - 1, int(y)))
    color = int(arr[clamped_y, clamped_x])
    kind = "background" if color == 0 else "object"
    bucket_x = int(clamped_x * 8 / max(width, 1))
    bucket_y = int(clamped_y * 8 / max(height, 1))
    return f"click:{kind}:color:{color}:bucket:{bucket_x}:{bucket_y}"


def transition_delta_class(
    before_observation: Any,
    action: str,
    result: Any,
) -> tuple[str, bool, bool, int]:
    info = getattr(result, "info", {}) or {}
    events = {str(item) for item in info.get("events", []) or []}
    score_delta = float(getattr(result, "reward", 0.0))
    before_extras = getattr(before_observation, "extras", {}) or {}
    next_observation = getattr(result, "observation", None)
    after_extras = getattr(next_observation, "extras", {}) or {}
    before_level = int(before_extras.get("levels_completed", before_extras.get("total_levels_completed", 0)) or 0)
    after_level = int(after_extras.get("levels_completed", after_extras.get("total_levels_completed", before_level)) or before_level)
    level_delta = int(after_level - before_level)
    terminal = bool(getattr(result, "terminated", False) or getattr(result, "truncated", False))
    game_state = str(info.get("game_state", after_extras.get("game_state", "")))
    useful = bool(score_delta > 0.0 or level_delta > 0 or game_state == "WIN" or events.intersection(PROGRESS_EVENTS))
    harmful = bool(score_delta <= SIGNIFICANT_HARM_REWARD or game_state == "GAME_OVER" or events.intersection(HARM_EVENTS))

    changed_cells = 0
    try:
        before_grid = np.asarray(getattr(before_observation, "grid"), dtype=np.int64)
        after_grid = np.asarray(getattr(next_observation, "grid"), dtype=np.int64)
        if before_grid.shape == after_grid.shape:
            changed_cells = int(np.count_nonzero(before_grid != after_grid))
    except Exception:
        changed_cells = 0

    if useful:
        return "progress", useful, harmful, level_delta
    if harmful:
        return "harm", useful, harmful, level_delta
    if changed_cells > 0:
        return "visible_nuisance", useful, harmful, level_delta
    if terminal:
        return "terminal_no_progress", useful, harmful, level_delta
    return "no_effect", useful, harmful, level_delta


def projected_signature(observation: Any, action: str) -> ActionEventSignature:
    return ActionEventSignature(
        state_hash=stable_observation_hash(observation),
        action=str(action),
        action_class=action_class(str(action)),
        coordinate_class=coordinate_equivalence_class(str(action), observation),
        delta_class="pending",
        score_delta=0.0,
        level_delta=0,
        useful_effect=False,
        harmful_effect=False,
    )


def observed_signature(before_observation: Any, action: str, result: Any) -> ActionEventSignature:
    delta_class, useful, harmful, level_delta = transition_delta_class(before_observation, str(action), result)
    return ActionEventSignature(
        state_hash=stable_observation_hash(before_observation),
        action=str(action),
        action_class=action_class(str(action)),
        coordinate_class=coordinate_equivalence_class(str(action), before_observation),
        delta_class=delta_class,
        score_delta=float(getattr(result, "reward", 0.0)),
        level_delta=int(level_delta),
        useful_effect=bool(useful),
        harmful_effect=bool(harmful),
    )


def _cycle_score(keys: Sequence[Any], period: int) -> float:
    if period <= 0 or len(keys) <= period:
        return 0.0
    matches = sum(1 for index in range(period, len(keys)) if keys[index] == keys[index - period])
    return float(matches / max(len(keys) - period, 1))


class HardAntiAttractorGate:
    """Hard veto for action cycles that produce no progress evidence."""

    def __init__(
        self,
        *,
        window: int = 12,
        max_period: int = 6,
        threshold: float = 0.82,
        no_progress_repeat_limit: int = 2,
        history_limit: int = 96,
    ) -> None:
        self.window = int(window)
        self.max_period = int(max_period)
        self.threshold = float(threshold)
        self.no_progress_repeat_limit = int(no_progress_repeat_limit)
        self.history_limit = int(history_limit)
        self.history: list[ActionEventSignature] = []
        self.undo_probe_used = False
        self.veto_count = 0
        self.last_diagnostics: dict[str, Any] = {"active": True, "vetoed_actions": []}

    def reset(self) -> None:
        self.history.clear()
        self.undo_probe_used = False
        self.last_diagnostics = {"active": True, "vetoed_actions": []}

    def observe_transition(self, before_observation: Any, action: str, result: Any) -> None:
        signature = observed_signature(before_observation, str(action), result)
        self.history.append(signature)
        self.history = self.history[-self.history_limit :]
        if signature.action_class == "undo":
            self.undo_probe_used = True

    def select_action(
        self,
        preferred_action: str,
        ranked_actions: Iterable[str],
        observation: Any,
    ) -> tuple[str, dict[str, Any]]:
        ranked = []
        seen: set[str] = set()
        for action in [str(preferred_action), *[str(item) for item in ranked_actions]]:
            if action not in seen:
                ranked.append(action)
                seen.add(action)
        vetoed: list[dict[str, Any]] = []
        selected = str(preferred_action)
        selected_reasons: list[str] = []
        for action in ranked:
            reasons = self.veto_reasons(action, observation)
            if reasons:
                vetoed.append({"action": action, "reasons": reasons})
                continue
            selected = action
            break
        else:
            selected_reasons = ["all_ranked_actions_vetoed_fail_open"]

        changed = selected != str(preferred_action)
        self.veto_count += int(changed)
        diagnostics = {
            "active": True,
            "schema": "hard_anti_attractor_gate_v1",
            "preferred_action": str(preferred_action),
            "selected_action": selected,
            "changed_action": bool(changed),
            "history_len": len(self.history),
            "window": self.window,
            "max_period": self.max_period,
            "threshold": self.threshold,
            "vetoed_actions": vetoed[:16],
            "fail_open_reasons": selected_reasons,
            "recent_progress": self._recent_progress(),
        }
        self.last_diagnostics = diagnostics
        return selected, diagnostics

    def veto_reasons(self, action: str, observation: Any) -> list[str]:
        candidate = projected_signature(observation, str(action))
        reasons: list[str] = []
        if candidate.action_class == "undo" and not self._undo_allowed():
            reasons.append("undo_without_harm_deadend_verified_macro_or_first_probe")
        reasons.extend(self._repeat_veto_reasons(candidate))
        reasons.extend(self._cycle_veto_reasons(candidate))
        return reasons

    def _recent(self) -> list[ActionEventSignature]:
        return self.history[-self.window :]

    def _recent_progress(self) -> bool:
        return any(item.useful_effect for item in self._recent())

    def _last_transition_harmful(self) -> bool:
        return bool(self.history and self.history[-1].harmful_effect)

    def _last_transition_nontrivial(self) -> bool:
        if not self.history:
            return False
        return self.history[-1].delta_class in {"visible_nuisance", "progress", "harm"}

    def _undo_allowed(self) -> bool:
        if self._last_transition_harmful():
            return True
        if not self.undo_probe_used and self._last_transition_nontrivial():
            return True
        return False

    def _repeat_veto_reasons(self, candidate: ActionEventSignature) -> list[str]:
        recent = self._recent()
        if not recent or any(item.useful_effect for item in recent):
            return []
        immediate_reasons: list[str] = []
        if recent[-1].action_cycle_key() == candidate.action_cycle_key():
            immediate_reasons.append(f"immediate_repeat:{candidate.action_class}:no_progress_window")
        if len(recent) < self.no_progress_repeat_limit:
            return immediate_reasons
        limit = max(int(self.no_progress_repeat_limit), 1)
        action_count = sum(1 for item in recent if item.action_cycle_key() == candidate.action_cycle_key())
        state_action_count = sum(
            1 for item in recent if item.state_action_cycle_key() == candidate.state_action_cycle_key()
        )
        reasons: list[str] = list(immediate_reasons)
        if action_count >= limit:
            reasons.append(
                f"action_repeat:{candidate.action_class}:count_{action_count}:limit_{limit}:no_progress_window"
            )
        if state_action_count >= limit:
            reasons.append(
                f"state_action_repeat:{candidate.action_class}:count_{state_action_count}:limit_{limit}:no_progress_window"
            )
        return reasons

    def _cycle_veto_reasons(self, candidate: ActionEventSignature) -> list[str]:
        recent = self._recent()
        if len(recent) < 3 or any(item.useful_effect for item in recent):
            return []
        projected = [*recent, candidate]
        checks = {
            "action_cycle": [item.action_cycle_key() for item in projected],
            "state_action_cycle": [item.state_action_cycle_key() for item in projected],
            "event_cycle": [item.event_cycle_key() for item in projected],
        }
        reasons: list[str] = []
        for name, keys in checks.items():
            for period in range(1, min(self.max_period, len(keys) - 1) + 1):
                recent_score = _cycle_score(keys[:-1], period)
                cycle_members = set(keys[-period - 1 : -1])
                if recent_score >= self.threshold and keys[-1] in cycle_members:
                    reasons.append(f"{name}:active_period_{period}:score_{recent_score:.3f}:cycle_member:no_progress_window")
                    break
                score = _cycle_score(keys, period)
                if score >= self.threshold:
                    reasons.append(f"{name}:period_{period}:score_{score:.3f}:no_progress_window")
                    break
        return reasons

    def summary(self) -> dict[str, Any]:
        return {
            "schema": "hard_anti_attractor_gate_v1",
            "history_len": len(self.history),
            "undo_probe_used": bool(self.undo_probe_used),
            "veto_count": int(self.veto_count),
            "no_progress_repeat_limit": int(self.no_progress_repeat_limit),
            "last": self.last_diagnostics,
        }
