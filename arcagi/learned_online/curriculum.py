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


@dataclass
class LongSparseMixedFamilyChainTask:
    seed: int = 0
    chain_length: int = 96
    size: int = 8
    step_index: int = 0
    progress: int = 0
    plan: tuple[str, ...] = ()
    target_cells: tuple[tuple[int, int], ...] = ()

    def reset(self, seed: int | None = None) -> GridObservation:
        if seed is not None:
            self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        families = ("click", "select", "move_right", "move_left", "undo")
        self.plan = tuple(str(families[int(rng.integers(len(families)))]) for _ in range(max(int(self.chain_length), 1)))
        self.target_cells = tuple(
            (int(rng.integers(1, max(int(self.size) - 1, 2))), int(rng.integers(1, max(int(self.size) - 1, 2))))
            for _ in range(len(self.plan))
        )
        self.step_index = 0
        self.progress = 0
        return self._observation()

    def step(self, action: ActionName) -> StepResult:
        self.step_index += 1
        reward = 0.0
        terminated = False
        if str(action) == self.expected_action():
            self.progress += 1
            if self.progress >= len(self.plan):
                reward = 1.0
                terminated = True
        return StepResult(self._observation(), reward, terminated, False, {})

    def expected_action(self) -> ActionName:
        if not self.plan:
            return "noop"
        family = self.plan[min(self.progress, len(self.plan) - 1)]
        if family == "click":
            y, x = self.target_cells[min(self.progress, len(self.target_cells) - 1)]
            return f"click:{x}:{y}"
        if family == "select":
            return "5"
        if family == "move_right":
            return "4"
        if family == "move_left":
            return "3"
        if family == "undo":
            return "7"
        return "noop"

    def expected_action_sequence(self) -> tuple[ActionName, ...]:
        progress = self.progress
        actions: list[ActionName] = []
        for index in range(progress, len(self.plan)):
            family = self.plan[index]
            if family == "click":
                y, x = self.target_cells[index]
                actions.append(f"click:{x}:{y}")
            elif family == "select":
                actions.append("5")
            elif family == "move_right":
                actions.append("4")
            elif family == "move_left":
                actions.append("3")
            elif family == "undo":
                actions.append("7")
            else:
                actions.append("noop")
        return tuple(actions)

    def expert_action(self, _observation: GridObservation | None = None) -> ActionName:
        return self.expected_action()

    def recommended_max_steps(self) -> int:
        return int(self.chain_length) + 8

    def _observation(self) -> GridObservation:
        grid = np.zeros((self.size, self.size), dtype=np.int64)
        family = self.plan[min(self.progress, len(self.plan) - 1)] if self.plan else "done"
        colors = {"click": 2, "select": 3, "move_right": 4, "move_left": 5, "undo": 6}
        grid[0, 0] = colors.get(family, 1)
        if self.target_cells and self.progress < len(self.target_cells):
            y, x = self.target_cells[self.progress]
            grid[y, x] = 9 if family == "click" else colors.get(family, 1)
        actions = tuple(f"click:{x}:{y}" for y in range(self.size) for x in range(self.size)) + (
            "1",
            "3",
            "4",
            "5",
            "7",
            "noop",
        )
        return GridObservation(
            "long_sparse_mixed_family_chain",
            str(self.seed),
            self.step_index,
            grid,
            actions,
            extras={
                "action_roles": {
                    "1": "move_up",
                    "3": "move_left",
                    "4": "move_right",
                    "5": "select_cycle",
                    "7": "undo",
                    "noop": "wait",
                },
                "inventory": {
                    "chain_progress": str(int(self.progress)),
                    "chain_total": str(int(len(self.plan))),
                    "chain_remaining": str(max(int(len(self.plan)) - int(self.progress), 0)),
                },
                "flags": {"phase_family": family},
            },
        )


@dataclass
class LongDenseDecoyTask:
    seed: int = 0
    chain_length: int = 96
    size: int = 10
    step_index: int = 0
    progress: int = 0
    target_cells: tuple[tuple[int, int], ...] = ()
    decoy_cells: tuple[tuple[tuple[int, int], ...], ...] = ()
    decoy_flash: bool = False

    def reset(self, seed: int | None = None) -> GridObservation:
        if seed is not None:
            self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        targets: list[tuple[int, int]] = []
        decoys: list[tuple[tuple[int, int], ...]] = []
        for _index in range(max(int(self.chain_length), 1)):
            target = (int(rng.integers(self.size)), int(rng.integers(self.size)))
            cells = {target}
            phase_decoys: list[tuple[int, int]] = []
            while len(phase_decoys) < 3:
                candidate = (int(rng.integers(self.size)), int(rng.integers(self.size)))
                if candidate in cells:
                    continue
                cells.add(candidate)
                phase_decoys.append(candidate)
            targets.append(target)
            decoys.append(tuple(phase_decoys))
        self.target_cells = tuple(targets)
        self.decoy_cells = tuple(decoys)
        self.step_index = 0
        self.progress = 0
        self.decoy_flash = False
        return self._observation()

    def step(self, action: ActionName) -> StepResult:
        self.step_index += 1
        reward = 0.0
        terminated = False
        self.decoy_flash = False
        if str(action) == self.expected_action():
            self.progress += 1
            if self.progress >= len(self.target_cells):
                reward = 1.0
                terminated = True
        elif str(action).startswith("click:"):
            self.decoy_flash = True
        return StepResult(self._observation(), reward, terminated, False, {})

    def expected_action(self) -> ActionName:
        if not self.target_cells:
            return "noop"
        y, x = self.target_cells[min(self.progress, len(self.target_cells) - 1)]
        return f"click:{x}:{y}"

    def expected_action_sequence(self) -> tuple[ActionName, ...]:
        return tuple(f"click:{x}:{y}" for y, x in self.target_cells[self.progress :])

    def expert_action(self, _observation: GridObservation | None = None) -> ActionName:
        return self.expected_action()

    def recommended_max_steps(self) -> int:
        return int(self.chain_length) + 8

    def _observation(self) -> GridObservation:
        grid = np.zeros((self.size, self.size), dtype=np.int64)
        if self.progress < len(self.target_cells):
            y, x = self.target_cells[self.progress]
            grid[y, x] = 9
            for dy, dx in self.decoy_cells[self.progress]:
                grid[dy, dx] = 4 if self.decoy_flash else 3
        actions = tuple(f"click:{x}:{y}" for y in range(self.size) for x in range(self.size)) + ("noop",)
        return GridObservation(
            "long_dense_decoy",
            str(self.seed),
            self.step_index,
            grid,
            actions,
            extras={
                "inventory": {
                    "chain_progress": str(int(self.progress)),
                    "chain_total": str(int(len(self.target_cells))),
                    "chain_remaining": str(max(int(len(self.target_cells)) - int(self.progress), 0)),
                },
                "flags": {"decoy_flash": "1" if self.decoy_flash else "0"},
            },
        )


@dataclass
class RoleOpaqueRemapTask:
    seed: int = 0
    step_index: int = 0
    target_action: ActionName = "up"

    def reset(self, seed: int | None = None) -> GridObservation:
        if seed is not None:
            self.seed = int(seed)
        actions = self.actions()
        self.target_action = actions[int(np.random.default_rng(self.seed).integers(len(actions) - 1))]
        self.step_index = 0
        return self._observation()

    def actions(self) -> tuple[ActionName, ...]:
        return ("up", "down", "left", "right", "select", "noop")

    def step(self, action: ActionName) -> StepResult:
        self.step_index += 1
        reward = 1.0 if str(action) == self.target_action else 0.0
        return StepResult(self._observation(), reward, bool(reward > 0.0), False, {})

    def expert_action(self, _observation: GridObservation | None = None) -> ActionName:
        return self.target_action

    def recommended_max_steps(self) -> int:
        return 4

    def _observation(self) -> GridObservation:
        actions = self.actions()
        grid = np.zeros((1, len(actions) - 1), dtype=np.int64)
        grid[0, actions.index(self.target_action)] = actions.index(self.target_action) + 1
        return GridObservation(
            "role_opaque_remap",
            str(self.seed),
            self.step_index,
            grid,
            actions,
            extras={},
        )


@dataclass
class LongPostBoundaryCarryoverTask:
    seed: int = 0
    level_count: int = 4
    size: int = 8
    step_index: int = 0
    levels_completed: int = 0
    armed: bool = False
    selector_action: ActionName = "5"
    target_cells: tuple[tuple[int, int], ...] = ()

    def reset(self, seed: int | None = None) -> GridObservation:
        if seed is not None:
            self.seed = int(seed)
        rng = np.random.default_rng(self.seed)
        self.selector_action = ("1", "5", "7")[int(rng.integers(3))]
        self.target_cells = tuple(
            (int(rng.integers(1, max(int(self.size) - 1, 2))), int(rng.integers(1, max(int(self.size) - 1, 2))))
            for _ in range(max(int(self.level_count), 1))
        )
        self.step_index = 0
        self.levels_completed = 0
        self.armed = False
        return self._observation()

    def step(self, action: ActionName) -> StepResult:
        self.step_index += 1
        reward = 0.0
        terminated = False
        if not self.armed and str(action) == self.selector_action:
            self.armed = True
        elif self.armed and str(action) == self.expected_click_action():
            reward = 1.0
            self.levels_completed += 1
            self.armed = False
            if self.levels_completed >= int(self.level_count):
                terminated = True
        return StepResult(self._observation(), reward, terminated, False, {})

    def expected_click_action(self) -> ActionName:
        if not self.target_cells:
            return "noop"
        y, x = self.target_cells[min(self.levels_completed, len(self.target_cells) - 1)]
        return f"click:{x}:{y}"

    def expert_action(self, _observation: GridObservation | None = None) -> ActionName:
        return self.expected_click_action() if self.armed else self.selector_action

    def expected_action_sequence(self) -> tuple[ActionName, ...]:
        actions: list[ActionName] = []
        armed = bool(self.armed)
        for level in range(self.levels_completed, int(self.level_count)):
            if not armed:
                actions.append(self.selector_action)
            y, x = self.target_cells[min(level, len(self.target_cells) - 1)]
            actions.append(f"click:{x}:{y}")
            armed = False
        return tuple(actions)

    def recommended_max_steps(self) -> int:
        return (2 * int(self.level_count)) + 8

    def _observation(self) -> GridObservation:
        grid = np.zeros((self.size, self.size), dtype=np.int64)
        selector_colors = {"1": 2, "5": 3, "7": 4}
        if self.levels_completed == 0:
            grid[0, 0] = selector_colors.get(self.selector_action, 1)
        if self.levels_completed < len(self.target_cells):
            y, x = self.target_cells[self.levels_completed]
            grid[y, x] = 9 if self.armed else 8
        actions = tuple(f"click:{x}:{y}" for y in range(self.size) for x in range(self.size)) + (
            "1",
            "5",
            "7",
            "noop",
        )
        game_state = "GameState.WIN" if self.levels_completed >= int(self.level_count) else "GameState.NOT_FINISHED"
        return GridObservation(
            "long_post_boundary_carryover",
            str(self.seed),
            self.step_index,
            grid,
            actions,
            extras={
                "levels_completed": int(self.levels_completed),
                "game_state": game_state,
                "action_roles": {
                    "1": "move_up",
                    "5": "select_cycle",
                    "7": "undo",
                    "noop": "wait",
                },
                "inventory": {
                    "interface_levels_completed": str(int(self.levels_completed)),
                    "interface_game_state": game_state,
                    "carryover_level_count": str(int(self.level_count)),
                    "carryover_armed": "1" if self.armed else "0",
                },
                "flags": {"selector_visible": "1" if self.levels_completed == 0 else "0"},
            },
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
