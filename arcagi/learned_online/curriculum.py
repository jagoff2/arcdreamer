from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arcagi.core.types import ActionName, GridObservation, StepResult


@dataclass
class VisibleUsefulTrapTask:
    seed: int = 0
    step_index: int = 0
    trap_on: bool = False

    def reset(self, seed: int | None = None) -> GridObservation:
        if seed is not None:
            self.seed = int(seed)
        self.step_index = 0
        self.trap_on = False
        return self._observation()

    def step(self, action: ActionName) -> StepResult:
        self.step_index += 1
        reward = 0.0
        terminated = False
        if action == "trap":
            self.trap_on = not self.trap_on
        elif action == "useful":
            reward = 1.0
            terminated = True
        return StepResult(self._observation(), reward, terminated, False, {})

    def _observation(self) -> GridObservation:
        grid = np.array([[0, int(self.trap_on)], [0, 0]], dtype=np.int64)
        return GridObservation("visible_useful_trap", str(self.seed), self.step_index, grid, ("trap", "useful", "noop"))


@dataclass
class RandomizedBindingTask:
    seed: int = 0
    step_index: int = 0
    correct_action: ActionName = "bind_a"

    def reset(self, seed: int | None = None) -> GridObservation:
        if seed is not None:
            self.seed = int(seed)
        self.step_index = 0
        actions = self.actions()
        self.correct_action = actions[int(np.random.default_rng(self.seed).integers(len(actions)))]
        return self._observation()

    def actions(self) -> tuple[ActionName, ...]:
        return ("bind_a", "bind_b", "bind_c", "bind_d")

    def step(self, action: ActionName) -> StepResult:
        self.step_index += 1
        reward = 1.0 if str(action) == self.correct_action else 0.0
        return StepResult(self._observation(), reward, bool(reward > 0.0), False, {})

    def _observation(self) -> GridObservation:
        grid = np.zeros((2, 2), dtype=np.int64)
        grid[0, 0] = (self.seed % 3) + 1
        return GridObservation("randomized_binding", str(self.seed), self.step_index, grid, self.actions())


@dataclass
class DenseCoordinateGroundingTask:
    seed: int = 0
    size: int = 6
    step_index: int = 0
    target_color: int = 2
    object_cell: tuple[int, int] = (1, 1)

    def reset(self, seed: int | None = None) -> GridObservation:
        if seed is not None:
            self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.step_index = 0
        self.target_color = 2 + int(rng.integers(3))
        self.object_cell = (int(rng.integers(self.size)), int(rng.integers(self.size)))
        return self._observation()

    def step(self, action: ActionName) -> StepResult:
        self.step_index += 1
        reward = 0.0
        parts = str(action).split(":")
        if len(parts) == 3 and parts[0] == "click":
            x = int(parts[1])
            y = int(parts[2])
            if (y, x) == self.object_cell:
                reward = 1.0
        return StepResult(self._observation(), reward, bool(reward > 0.0), False, {})

    def _observation(self) -> GridObservation:
        grid = np.zeros((self.size, self.size), dtype=np.int64)
        y, x = self.object_cell
        grid[y, x] = self.target_color
        actions = tuple(f"click:{x}:{y}" for y in range(self.size) for x in range(self.size))
        return GridObservation("dense_coordinate_grounding", str(self.seed), self.step_index, grid, actions)


@dataclass
class NullDenseClickTask:
    seed: int = 0
    width: int = 8
    height: int = 8
    step_index: int = 0

    def reset(self, seed: int | None = None) -> GridObservation:
        if seed is not None:
            self.seed = int(seed)
        self.step_index = 0
        return self._observation()

    def step(self, action: ActionName) -> StepResult:
        self.step_index += 1
        return StepResult(self._observation(), 0.0, False, False, {})

    def _observation(self) -> GridObservation:
        grid = np.zeros((self.height, self.width), dtype=np.int64)
        actions = tuple(f"click:{x}:{y}" for y in range(self.height) for x in range(self.width))
        return GridObservation("null_dense_click", str(self.seed), self.step_index, grid, actions)


class RowMajorSweepPolicy:
    def __init__(self) -> None:
        self.index = 0

    def act(self, observation: GridObservation) -> ActionName:
        actions = tuple(observation.available_actions)
        action = actions[self.index % len(actions)]
        self.index += 1
        return action


class FrozenFirstPolicy:
    name = "frozen_first"
    latest_language: tuple[str, ...] = ()

    def reset_episode(self) -> None:
        return None

    def act(self, observation: GridObservation) -> ActionName:
        return tuple(observation.available_actions)[0]

    def update_after_step(self, **_kwargs) -> None:
        return None


class ExactActionCountBaseline:
    name = "exact_action_count"
    latest_language: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.counts: dict[ActionName, tuple[int, float]] = {}

    def reset_episode(self) -> None:
        self.counts.clear()

    def act(self, observation: GridObservation) -> ActionName:
        actions = tuple(observation.available_actions)
        unseen = [action for action in actions if action not in self.counts]
        if unseen:
            return unseen[0]
        return max(actions, key=lambda action: self.counts[action][1] / max(float(self.counts[action][0]), 1.0))

    def update_after_step(self, **kwargs) -> None:
        action = str(kwargs.get("action", ""))
        reward = float(kwargs.get("reward", 0.0))
        if not action:
            return
        trials, total = self.counts.get(action, (0, 0.0))
        self.counts[action] = (trials + 1, total + reward)


class FamilyCountBaseline:
    name = "family_count"
    latest_language: tuple[str, ...] = ()

    def __init__(self) -> None:
        self.counts: dict[str, tuple[int, float]] = {}

    def reset_episode(self) -> None:
        self.counts.clear()

    def act(self, observation: GridObservation) -> ActionName:
        actions = tuple(observation.available_actions)
        unseen = [action for action in actions if _family(action) not in self.counts]
        if unseen:
            return unseen[0]
        return max(actions, key=lambda action: self.counts[_family(action)][1] / max(float(self.counts[_family(action)][0]), 1.0))

    def update_after_step(self, **kwargs) -> None:
        action = str(kwargs.get("action", ""))
        reward = float(kwargs.get("reward", 0.0))
        if not action:
            return
        key = _family(action)
        trials, total = self.counts.get(key, (0, 0.0))
        self.counts[key] = (trials + 1, total + reward)


def _family(action: ActionName) -> str:
    action = str(action)
    if action.startswith("click:"):
        return "click"
    return action.split("_", 1)[0]
