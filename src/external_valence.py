from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .arcagi3_adapter import ArcAGI3Observation, ArcAGI3StepResult


def observation_key(observation: ArcAGI3Observation) -> str:
    grid = np.asarray(observation.grid, dtype=np.int64)
    payload = {
        "shape": list(grid.shape),
        "grid": grid.reshape(-1).tolist(),
        "actions": list(observation.available_actions),
    }
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def grid_changed(before: ArcAGI3Observation, after: ArcAGI3Observation) -> bool:
    return not np.array_equal(np.asarray(before.grid), np.asarray(after.grid))


def public_valence(before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult) -> dict[str, Any]:
    events = list(result.info.get("events", []))
    changed = grid_changed(before, result.observation)
    invalid = "invalid_action" in events or action not in before.available_actions
    positive_event = any(
        item
        in {
            "positive_reward",
            "key_collected",
            "door_opened",
            "goal_reached",
            "goal_clicked",
            "resource_collected",
            "useful_click",
            "level_completed",
            "terminated",
        }
        for item in events
    )
    terminal_bad = any(item in {"game_over", "runtime_no_response"} for item in events)
    reward = float(result.reward)
    no_effect = reward <= 0.0 and not positive_event and not changed and not invalid
    value = reward
    value += 0.25 if positive_event else 0.0
    value += 0.04 if changed else 0.0
    value -= 0.08 if no_effect else 0.0
    value -= 0.25 if invalid or terminal_bad else 0.0
    return {
        "state_key": observation_key(before),
        "action": action,
        "reward": reward,
        "events": events,
        "changed": changed,
        "positive_event": positive_event,
        "no_effect": no_effect,
        "invalid": invalid,
        "value": float(value),
    }


@dataclass
class ConsequenceValence:
    alpha: float = 0.35
    action_value: dict[str, float] = field(default_factory=dict)
    state_action_value: dict[tuple[str, str], float] = field(default_factory=dict)
    no_effect_count: dict[tuple[str, str], int] = field(default_factory=dict)
    updates: int = 0
    nonzero_updates: int = 0

    def reset(self) -> None:
        self.action_value.clear()
        self.state_action_value.clear()
        self.no_effect_count.clear()
        self.updates = 0
        self.nonzero_updates = 0

    def score(self, observation: ArcAGI3Observation, action: str) -> float:
        key = observation_key(observation)
        return 0.65 * self.state_action_value.get((key, action), 0.0) + 0.35 * self.action_value.get(action, 0.0)

    def loop_penalty(self, observation: ArcAGI3Observation, action: str) -> float:
        count = self.no_effect_count.get((observation_key(observation), action), 0)
        return float(min(3, count))

    def observe(self, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult) -> dict[str, Any]:
        item = public_valence(before, action, result)
        key = (str(item["state_key"]), action)
        value = float(item["value"])
        self.action_value[action] = self._update(self.action_value.get(action, 0.0), value)
        self.state_action_value[key] = self._update(self.state_action_value.get(key, 0.0), value)
        if item["no_effect"]:
            self.no_effect_count[key] = self.no_effect_count.get(key, 0) + 1
        self.updates += 1
        self.nonzero_updates += int(abs(value) > 1.0e-9)
        item["action_value"] = self.action_value[action]
        item["state_action_value"] = self.state_action_value[key]
        item["no_effect_count"] = self.no_effect_count.get(key, 0)
        return item

    def diagnostics(self) -> dict[str, Any]:
        return {
            "updates": self.updates,
            "nonzero_updates": self.nonzero_updates,
            "known_actions": len(self.action_value),
            "known_state_actions": len(self.state_action_value),
            "no_effect_state_actions": len(self.no_effect_count),
        }

    def _update(self, old: float, value: float) -> float:
        return float((1.0 - self.alpha) * old + self.alpha * value)
