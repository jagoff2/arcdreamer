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


@dataclass
class VisibleMovementTrapTask:
    seed: int = 0
    step_index: int = 0
    position: int = 0

    def reset(self, seed: int | None = None) -> GridObservation:
        if seed is not None:
            self.seed = int(seed)
        self.step_index = 0
        self.position = 0
        return self._observation()

    def step(self, action: ActionName) -> StepResult:
        self.step_index += 1
        if action == "move_left":
            self.position = max(0, self.position - 1)
        elif action == "move_right":
            self.position = min(3, self.position + 1)
        reward = 1.0 if action == "commit" else 0.0
        return StepResult(self._observation(), reward, bool(reward > 0.0), False, {})

    def _observation(self) -> GridObservation:
        grid = np.zeros((1, 4), dtype=np.int64)
        grid[0, self.position] = 1
        return GridObservation("visible_movement_trap", str(self.seed), self.step_index, grid, ("move_left", "move_right", "commit"))


@dataclass
class MovementRequiredAfterModeTask:
    seed: int = 0
    step_index: int = 0
    mode: bool = False
    position: int = 0

    def reset(self, seed: int | None = None) -> GridObservation:
        if seed is not None:
            self.seed = int(seed)
        self.step_index = 0
        self.mode = False
        self.position = 0
        return self._observation()

    def step(self, action: ActionName) -> StepResult:
        self.step_index += 1
        reward = 0.0
        if action == "mode":
            self.mode = True
        elif action == "move_right" and self.mode:
            self.position += 1
            if self.position >= 2:
                reward = 1.0
        elif action == "move_right":
            self.position = 0
        return StepResult(self._observation(), reward, bool(reward > 0.0), False, {})

    def _observation(self) -> GridObservation:
        grid = np.zeros((1, 3), dtype=np.int64)
        grid[0, min(self.position, 2)] = 2 if self.mode else 1
        return GridObservation("movement_required_after_mode", str(self.seed), self.step_index, grid, ("mode", "move_right", "noop"))


@dataclass
class DelayedUnlockTask:
    seed: int = 0
    step_index: int = 0
    unlocked: bool = False

    def reset(self, seed: int | None = None) -> GridObservation:
        if seed is not None:
            self.seed = int(seed)
        self.step_index = 0
        self.unlocked = False
        return self._observation()

    def step(self, action: ActionName) -> StepResult:
        self.step_index += 1
        reward = 0.0
        if action == "unlock":
            self.unlocked = True
        elif action == "goal" and self.unlocked:
            reward = 1.0
        return StepResult(self._observation(), reward, bool(reward > 0.0), False, {})

    def _observation(self) -> GridObservation:
        grid = np.array([[1, 2 if self.unlocked else 0]], dtype=np.int64)
        actions = ("goal", "noop", "unlock") if self.unlocked else ("unlock", "noop")
        return GridObservation("delayed_unlock", str(self.seed), self.step_index, grid, actions)


@dataclass
class DenseFamilyMassArbitrationTask:
    seed: int = 0
    size: int = 20
    step_index: int = 0

    def reset(self, seed: int | None = None) -> GridObservation:
        if seed is not None:
            self.seed = int(seed)
        self.step_index = 0
        return self._observation()

    def step(self, action: ActionName) -> StepResult:
        self.step_index += 1
        reward = 1.0 if str(action) == "5" else 0.0
        return StepResult(self._observation(), reward, bool(reward > 0.0), False, {})

    def _observation(self) -> GridObservation:
        grid = np.zeros((self.size, self.size), dtype=np.int64)
        grid[self.size // 2, self.size // 2] = 1
        actions = tuple(f"click:{x}:{y}" for y in range(self.size) for x in range(self.size)) + ("1", "5", "7")
        return GridObservation(
            "dense_family_mass_arbitration",
            str(self.seed),
            self.step_index,
            grid,
            actions,
            extras={"action_roles": {"1": "move_up", "5": "select_cycle", "7": "undo"}},
        )


@dataclass
class ModeThenDenseClickTask:
    seed: int = 0
    size: int = 8
    step_index: int = 0
    mode: bool = False
    target_cell: tuple[int, int] = (0, 0)

    def reset(self, seed: int | None = None) -> GridObservation:
        if seed is not None:
            self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.step_index = 0
        self.mode = False
        self.target_cell = (int(rng.integers(self.size)), int(rng.integers(self.size)))
        return self._observation()

    def step(self, action: ActionName) -> StepResult:
        self.step_index += 1
        reward = 0.0
        if str(action) == "5":
            self.mode = True
        elif self.mode:
            parts = str(action).split(":")
            if len(parts) == 3 and parts[0] == "click":
                x = int(parts[1])
                y = int(parts[2])
                if (y, x) == self.target_cell:
                    reward = 1.0
        return StepResult(self._observation(), reward, bool(reward > 0.0), False, {})

    def _observation(self) -> GridObservation:
        grid = np.zeros((self.size, self.size), dtype=np.int64)
        y, x = self.target_cell
        grid[y, x] = 2 if self.mode else 1
        actions = tuple(f"click:{x}:{y}" for y in range(self.size) for x in range(self.size)) + ("5", "noop")
        return GridObservation(
            "mode_then_dense_click",
            str(self.seed),
            self.step_index,
            grid,
            actions,
            extras={"action_roles": {"5": "select_cycle", "noop": "wait"}},
        )


@dataclass
class ActionNameRemapHeldoutTask:
    seed: int = 0
    step_index: int = 0
    action_names: tuple[ActionName, ...] = ("a0", "a1", "a2", "a3")
    roles: tuple[str, ...] = ("move_up", "move_down", "select_cycle", "undo")
    target_role: str = "select_cycle"

    def reset(self, seed: int | None = None) -> GridObservation:
        if seed is not None:
            self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.step_index = 0
        names = [f"a{self.seed % 997}_{idx}" for idx in range(4)]
        rng.shuffle(names)
        self.action_names = tuple(names)
        self.roles = ("move_up", "move_down", "select_cycle", "undo")
        self.target_role = self.roles[int(rng.integers(len(self.roles)))]
        return self._observation()

    def step(self, action: ActionName) -> StepResult:
        self.step_index += 1
        roles = self._action_roles()
        reward = 1.0 if roles.get(str(action)) == self.target_role else 0.0
        return StepResult(self._observation(), reward, bool(reward > 0.0), False, {})

    def _action_roles(self) -> dict[ActionName, str]:
        return {action: role for action, role in zip(self.action_names, self.roles)}

    def _observation(self) -> GridObservation:
        role_index = self.roles.index(self.target_role)
        grid = np.zeros((1, len(self.roles)), dtype=np.int64)
        grid[0, role_index] = role_index + 1
        return GridObservation(
            "action_name_remap_heldout",
            str(self.seed),
            self.step_index,
            grid,
            self.action_names,
            extras={"action_roles": self._action_roles()},
        )


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
