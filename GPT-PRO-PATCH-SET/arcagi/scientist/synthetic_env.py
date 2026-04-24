"""Small black-box synthetic environments for testing online rule learning.

These are not ARC-AGI-3 games.  They are intentionally tiny hidden-rule worlds
that exercise the same mechanics: sparse reward, latent affordances, object
movement, interaction, and no natural-language instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import GridFrame, action_delta, action_family, parse_action_target


@dataclass(frozen=True)
class SyntheticConfig:
    size: int = 7
    requires_key: bool = True
    seed: int = 0
    max_steps: int = 80


class HiddenRuleGridEnv:
    """A minimal grid game with hidden key/goal mechanics.

    Colors:
        0 background, 1 avatar, 2 goal, 3 key, 4 wall, 5 distractor.
    """

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.step_index = 0
        self.episode_index = 0
        self.avatar = (1, 1)
        self.goal = (self.config.size - 2, self.config.size - 2)
        self.key = (1, self.config.size - 2)
        self.has_key = False
        self.done = False
        self.score = 0.0
        self.available_actions = ("up", "down", "left", "right", "interact", "click")

    @property
    def task_id(self) -> str:
        return "synthetic/hidden_rule_grid"

    def reset(self, seed: int | None = None) -> GridFrame:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.episode_index += 1
        self.step_index = 0
        self.done = False
        self.has_key = False
        self.score = 0.0
        self.avatar = (1, 1)
        self.goal = (self.config.size - 2, self.config.size - 2)
        self.key = (1, self.config.size - 2)
        return self._frame()

    def step(self, action: str) -> tuple[GridFrame, float, bool, dict[str, Any]]:
        if self.done:
            return self._frame(), 0.0, True, {"score": self.score, "already_done": True}
        old_score = self.score
        reward = 0.0
        fam = action_family(action)
        target = parse_action_target(action)
        if target is not None and fam in {"click", "action6"}:
            # Clicking the key is a legal but hidden affordance.  It gives the
            # agent a way to test targeted actions without requiring text.
            if _manhattan(target, self.key) <= 0 and not self.has_key:
                self.has_key = True
                self.score += 0.10
                reward += 0.10
        elif fam == "interact":
            if _manhattan(self.avatar, self.key) <= 1 and not self.has_key:
                self.has_key = True
                self.score += 0.10
                reward += 0.10
        else:
            delta = action_delta(action)
            if delta is not None:
                nr = min(max(self.avatar[0] + delta[0], 0), self.config.size - 1)
                nc = min(max(self.avatar[1] + delta[1], 0), self.config.size - 1)
                if (nr, nc) not in self._walls():
                    self.avatar = (nr, nc)

        if _manhattan(self.avatar, self.goal) <= 0:
            if (not self.config.requires_key) or self.has_key:
                self.score += 1.0
                reward += 1.0
                self.done = True
            else:
                reward -= 0.02

        self.step_index += 1
        if self.step_index >= self.config.max_steps:
            self.done = True
        info = {"score": self.score, "score_delta": self.score - old_score, "has_key": self.has_key}
        return self._frame(), float(reward), bool(self.done), info

    def _walls(self) -> set[tuple[int, int]]:
        size = self.config.size
        walls = {(3, c) for c in range(1, size - 1) if c != size // 2}
        return walls

    def _grid(self) -> np.ndarray:
        size = self.config.size
        grid = np.zeros((size, size), dtype=np.int64)
        for r, c in self._walls():
            grid[r, c] = 4
        grid[self.goal] = 2
        if not self.has_key:
            grid[self.key] = 3
        grid[size - 2, 1] = 5
        grid[self.avatar] = 1
        return grid

    def _frame(self) -> GridFrame:
        return GridFrame(
            task_id=self.task_id,
            episode_id=f"{self.task_id}/episode_{self.episode_index}",
            step_index=self.step_index,
            grid=self._grid(),
            available_actions=self.available_actions,
            extras={"score": self.score, "has_key": self.has_key},
        )


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])
