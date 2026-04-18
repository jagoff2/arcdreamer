from __future__ import annotations

from dataclasses import dataclass
import random
from typing import Any

import numpy as np

from arcagi.core.types import ACTION_ORDER, ActionName, GridObservation, StepResult
from arcagi.envs.base import BaseEnvironment

EMPTY = 0
WALL = 1
TARGET = 2
SWITCH_RED = 3
SWITCH_BLUE = 4
SWITCH_GREEN = 5
SWITCH_YELLOW = 6
COLLECT_RED = 7
COLLECT_BLUE = 8
AGENT = 9
GOAL_ACTIVE = 10

SWITCH_COLORS: tuple[int, ...] = (SWITCH_RED, SWITCH_BLUE, SWITCH_GREEN, SWITCH_YELLOW)
SELECTOR_COLORS: tuple[int, ...] = (SWITCH_RED, SWITCH_BLUE, SWITCH_GREEN)
COLLECT_COLORS: tuple[int, ...] = (COLLECT_RED, COLLECT_BLUE)
DEFAULT_SYNTHETIC_FAMILY_MODES: tuple[str, ...] = (
    "switch_unlock",
    "order_collect",
    "selector_unlock",
    "delayed_order_unlock",
    "selector_sequence_unlock",
)

MOVE_DELTAS: dict[ActionName, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
}

INTERACT_DELTAS: dict[ActionName, tuple[int, int]] = {
    "interact_up": (-1, 0),
    "interact_down": (1, 0),
    "interact_left": (0, -1),
    "interact_right": (0, 1),
}


@dataclass(frozen=True)
class RuleSpec:
    kind: str
    family_id: str
    answer_tokens: tuple[str, ...]
    question_tokens: tuple[str, ...]
    target_color: int
    order: tuple[int, ...] = ()
    decoy_color: int = 0


def _random_empty_cell(grid: np.ndarray, rng: random.Random) -> tuple[int, int]:
    empties = np.argwhere(grid == EMPTY)
    choice = empties[rng.randrange(len(empties))]
    return int(choice[0]), int(choice[1])


def _click_action(position: tuple[int, int]) -> ActionName:
    y, x = position
    return f"click:{x}:{y}"


def _parse_click_action(action: ActionName) -> tuple[int, int] | None:
    if not action.startswith("click:"):
        return None
    _, x_str, y_str, *_ = action.split(":") + ["", ""]
    if not x_str.isdigit() or not y_str.isdigit():
        return None
    return int(y_str), int(x_str)


class HiddenRuleEnv(BaseEnvironment):
    def __init__(
        self,
        size: int = 7,
        max_steps: int = 36,
        family_mode: str = "switch_unlock",
        family_variant: str | None = None,
        seed: int = 0,
    ) -> None:
        self.size = size
        self.max_steps = max_steps
        self.family_mode = family_mode
        self.family_variant = family_variant
        self._base_seed = seed
        self._rng = random.Random(seed)
        self._episode_index = 0
        self._task_id = "synthetic/hidden_rule"
        self._family_id = ""
        self._rule: RuleSpec | None = None
        self._grid = np.zeros((size, size), dtype=np.int64)
        self._agent = (1, 1)
        self._step = 0
        self._terminated = False
        self._inventory: dict[str, str] = {}
        self._flags: dict[str, str] = {}
        self._target_pos = (self.size - 2, self.size - 2)
        self._available_actions: tuple[ActionName, ...] = ACTION_ORDER
        self._action_roles: dict[ActionName, str] = {}
        self._cell_tag_overrides: dict[tuple[int, int], tuple[str, ...]] = {}
        self._selector_actions: dict[ActionName, int] = {}
        self._selector_positions: dict[int, tuple[int, int]] = {}
        self._switch_positions: dict[int, tuple[int, int]] = {}
        self._collect_positions: dict[int, tuple[int, int]] = {}
        self._selected_color: int | None = None
        self._progress_index = 0

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def family_id(self) -> str:
        return self._family_id

    def legal_actions(self) -> tuple[ActionName, ...]:
        return self._available_actions

    def reset(self, seed: int | None = None) -> GridObservation:
        if seed is not None:
            self._rng = random.Random(seed)
        else:
            self._rng = random.Random(self._base_seed + self._episode_index)
        self._episode_index += 1
        self._step = 0
        self._terminated = False
        self._inventory = {}
        self._flags = {"goal_active": "0"}
        self._grid = np.full((self.size, self.size), EMPTY, dtype=np.int64)
        self._grid[[0, -1], :] = WALL
        self._grid[:, [0, -1]] = WALL
        self._available_actions = ACTION_ORDER
        self._action_roles = {}
        self._cell_tag_overrides = {}
        self._selector_actions = {}
        self._selector_positions = {}
        self._switch_positions = {}
        self._collect_positions = {}
        self._selected_color = None
        self._progress_index = 0
        self._place_target()
        if self.family_mode == "switch_unlock":
            self._build_switch_unlock()
        elif self.family_mode == "order_collect":
            self._build_order_collect()
        elif self.family_mode == "selector_unlock":
            self._build_selector_unlock()
        elif self.family_mode == "delayed_order_unlock":
            self._build_delayed_order_unlock()
        elif self.family_mode == "selector_sequence_unlock":
            self._build_selector_sequence_unlock()
        else:
            raise ValueError(f"unknown family_mode: {self.family_mode}")
        self._agent = _random_empty_cell(self._grid, self._rng)
        self._grid[self._agent] = AGENT
        return self._observation()

    def step(self, action: ActionName) -> StepResult:
        if self._terminated:
            raise RuntimeError("environment already terminated")
        self._step += 1
        reward = -0.01
        event = "noop"
        if action in MOVE_DELTAS:
            reward, event = self._handle_move(action)
        elif action in INTERACT_DELTAS:
            reward, event = self._handle_interaction(action)
        elif action.startswith("click:"):
            reward, event = self._handle_click(action)
        elif action != "wait":
            raise ValueError(f"unsupported action: {action}")
        terminated = self._terminated or self._step >= self.max_steps
        truncated = not self._terminated and self._step >= self.max_steps
        observation = self._observation()
        info: dict[str, Any] = {
            "family_id": self._family_id,
            "rule_tokens": self._rule.answer_tokens if self._rule else (),
            "question_tokens": self._rule.question_tokens if self._rule else (),
            "inventory": dict(self._inventory),
            "flags": dict(self._flags),
            "event": event,
        }
        return StepResult(
            observation=observation,
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            info=info,
        )

    def _place_target(self) -> None:
        y, x = self._target_pos
        self._grid[y, x] = TARGET

    def _build_switch_unlock(self) -> None:
        if self.family_variant is None:
            unlock_color = self._rng.choice(SWITCH_COLORS)
        else:
            unlock_color = _name_to_color(self.family_variant)
        color_name = self._color_name(unlock_color)
        self._family_id = f"switch_unlock/{color_name}"
        self._rule = RuleSpec(
            kind="switch_unlock",
            family_id=self._family_id,
            answer_tokens=("goal", "unlock_target", color_name, "switch"),
            question_tokens=("need", "test", color_name, "switch"),
            target_color=unlock_color,
        )
        for cell_value in SWITCH_COLORS[:3]:
            pos = _random_empty_cell(self._grid, self._rng)
            self._grid[pos] = cell_value
            self._switch_positions[cell_value] = pos
            self._cell_tag_overrides[pos] = ("interactable",)

    def _build_order_collect(self) -> None:
        if self.family_variant == "red_then_blue":
            order = (COLLECT_RED, COLLECT_BLUE)
        elif self.family_variant == "blue_then_red":
            order = (COLLECT_BLUE, COLLECT_RED)
        else:
            order = (COLLECT_RED, COLLECT_BLUE)
            if self._rng.random() < 0.5:
                order = tuple(reversed(order))
        first, second = order
        name_first = self._color_name(first)
        name_second = self._color_name(second)
        self._family_id = f"order_collect/{name_first}_then_{name_second}"
        self._rule = RuleSpec(
            kind="order_collect",
            family_id=self._family_id,
            answer_tokens=("goal", "collect", name_first, "then", name_second),
            question_tokens=("need", "test", name_first, "before", name_second),
            target_color=first,
            order=order,
        )
        for cell_value in COLLECT_COLORS:
            pos = _random_empty_cell(self._grid, self._rng)
            self._grid[pos] = cell_value
            self._collect_positions[cell_value] = pos
            self._cell_tag_overrides[pos] = ("interactable",)

    def _build_selector_unlock(self) -> None:
        if self.family_variant is None:
            unlock_color = self._rng.choice(SELECTOR_COLORS)
        else:
            unlock_color = _name_to_color(self.family_variant)
        color_name = self._color_name(unlock_color)
        self._family_id = f"selector_unlock/{color_name}"
        self._rule = RuleSpec(
            kind="selector_unlock",
            family_id=self._family_id,
            answer_tokens=("goal", "select", color_name, "then", "switch"),
            question_tokens=("need", "test", "selector", "before", color_name),
            target_color=unlock_color,
        )
        click_actions: list[ActionName] = list(ACTION_ORDER)
        for cell_value in SELECTOR_COLORS:
            selector_pos = _random_empty_cell(self._grid, self._rng)
            self._grid[selector_pos] = cell_value
            self._selector_positions[cell_value] = selector_pos
            self._cell_tag_overrides[selector_pos] = ("selector",)
            click_action = _click_action(selector_pos)
            self._selector_actions[click_action] = cell_value
            self._action_roles[click_action] = "click"
            click_actions.append(click_action)
        for cell_value in SELECTOR_COLORS:
            switch_pos = _random_empty_cell(self._grid, self._rng)
            self._grid[switch_pos] = cell_value
            self._switch_positions[cell_value] = switch_pos
            self._cell_tag_overrides[switch_pos] = ("interactable",)
        self._available_actions = tuple(click_actions)

    def _build_delayed_order_unlock(self) -> None:
        if self.family_variant == "red_then_blue":
            order = (COLLECT_RED, COLLECT_BLUE)
        elif self.family_variant == "blue_then_red":
            order = (COLLECT_BLUE, COLLECT_RED)
        else:
            order = (COLLECT_RED, COLLECT_BLUE)
            if self._rng.random() < 0.5:
                order = tuple(reversed(order))
        decoy = SWITCH_YELLOW
        first, second = order
        name_first = self._color_name(first)
        name_second = self._color_name(second)
        decoy_name = self._color_name(decoy)
        self._family_id = f"delayed_order_unlock/{name_first}_then_{name_second}"
        self._rule = RuleSpec(
            kind="delayed_order_unlock",
            family_id=self._family_id,
            answer_tokens=("goal", "delayed_collect", name_first, "then", name_second, "avoid", decoy_name),
            question_tokens=("need", "test", "delayed", name_first, "before", name_second),
            target_color=first,
            order=order,
            decoy_color=decoy,
        )
        for cell_value in (*COLLECT_COLORS, decoy):
            pos = _random_empty_cell(self._grid, self._rng)
            self._grid[pos] = cell_value
            self._collect_positions[cell_value] = pos
            self._cell_tag_overrides[pos] = ("interactable",)

    def _build_selector_sequence_unlock(self) -> None:
        if self.family_variant is None:
            gate_color = self._rng.choice(SELECTOR_COLORS)
            order = (COLLECT_RED, COLLECT_BLUE)
            if self._rng.random() < 0.5:
                order = tuple(reversed(order))
        else:
            gate_name, order_name = self.family_variant.split("__", maxsplit=1)
            gate_color = _name_to_color(gate_name)
            order = _order_variant_to_tuple(order_name)
        decoy = SWITCH_YELLOW
        first, second = order
        gate_name = self._color_name(gate_color)
        first_name = self._color_name(first)
        second_name = self._color_name(second)
        self._family_id = f"selector_sequence_unlock/{gate_name}__{first_name}_then_{second_name}"
        self._rule = RuleSpec(
            kind="selector_sequence_unlock",
            family_id=self._family_id,
            answer_tokens=("goal", "select", gate_name, "then", first_name, "then", second_name, "avoid", "yellow"),
            question_tokens=("need", "test", "selector", "and", "delayed_order"),
            target_color=gate_color,
            order=order,
            decoy_color=decoy,
        )
        click_actions: list[ActionName] = list(ACTION_ORDER)
        for cell_value in SELECTOR_COLORS:
            selector_pos = _random_empty_cell(self._grid, self._rng)
            self._grid[selector_pos] = cell_value
            self._selector_positions[cell_value] = selector_pos
            self._cell_tag_overrides[selector_pos] = ("selector",)
            click_action = _click_action(selector_pos)
            self._selector_actions[click_action] = cell_value
            self._action_roles[click_action] = "click"
            click_actions.append(click_action)
        for cell_value in (*COLLECT_COLORS, decoy):
            pos = _random_empty_cell(self._grid, self._rng)
            self._grid[pos] = cell_value
            self._collect_positions[cell_value] = pos
            self._cell_tag_overrides[pos] = ("interactable",)
        self._available_actions = tuple(click_actions)

    def _handle_move(self, action: ActionName) -> tuple[float, str]:
        dy, dx = MOVE_DELTAS[action]
        ny = self._agent[0] + dy
        nx = self._agent[1] + dx
        cell = int(self._grid[ny, nx])
        if cell == WALL:
            return -0.05, "wall"
        if cell in SWITCH_COLORS or cell in COLLECT_COLORS:
            return -0.02, "blocked_by_object"
        if cell in (TARGET, GOAL_ACTIVE):
            if self._goal_is_active():
                self._terminated = True
                return 1.0, "goal_reached"
            return -0.03, "inactive_goal_blocked"
        self._move_agent((ny, nx))
        return -0.01, "move"

    def _handle_click(self, action: ActionName) -> tuple[float, str]:
        if action not in self._selector_actions:
            return -0.05, "invalid_click"
        self._selected_color = self._selector_actions[action]
        if self._rule is not None and self._rule.kind == "selector_unlock":
            if self._selected_color == self._rule.target_color:
                return 0.0, "selector_candidate"
            return 0.0, "selector_probe"
        return -0.01, "unused_click"

    def _handle_interaction(self, action: ActionName) -> tuple[float, str]:
        dy, dx = INTERACT_DELTAS[action]
        ny = self._agent[0] + dy
        nx = self._agent[1] + dx
        cell = int(self._grid[ny, nx])
        if self._rule is None:
            return -0.05, "missing_rule"
        if self._goal_is_active():
            if self._rule.kind == "switch_unlock" and cell in SWITCH_COLORS:
                return -0.05, "redundant_post_goal_interaction"
            if self._rule.kind == "selector_unlock" and (ny, nx) in self._switch_positions.values():
                return -0.05, "redundant_post_goal_interaction"
            if self._rule.kind == "order_collect" and cell in COLLECT_COLORS:
                return -0.05, "redundant_post_goal_interaction"
            if self._rule.kind == "delayed_order_unlock" and (
                cell in self._rule.order or cell == self._rule.decoy_color
            ):
                return -0.05, "redundant_post_goal_interaction"
            if self._rule.kind == "selector_sequence_unlock" and (
                cell in self._rule.order or cell == self._rule.decoy_color
            ):
                return -0.05, "redundant_post_goal_interaction"
        if self._rule.kind == "switch_unlock" and cell in SWITCH_COLORS:
            if cell == self._rule.target_color:
                self._flags["goal_active"] = "1"
                self._mark_goal_active()
                return 0.25, "correct_switch"
            return -0.1, "wrong_switch"
        if self._rule.kind == "selector_unlock" and (ny, nx) in self._switch_positions.values():
            if self._selected_color == self._rule.target_color and cell == self._rule.target_color:
                self._flags["goal_active"] = "1"
                self._mark_goal_active()
                return 0.2, "selector_unlock_complete"
            if self._selected_color == cell:
                return 0.03, "local_match_no_unlock"
            return -0.08, "wrong_selector_or_switch"
        if self._rule.kind == "order_collect" and cell in COLLECT_COLORS:
            collected = self._inventory.get("sequence", "")
            expected = self._rule.order[len(collected)] if len(collected) < len(self._rule.order) else None
            if expected is None:
                return -0.05, "redundant_collect"
            if cell == expected:
                collected += self._color_name(cell)[0]
                self._inventory["sequence"] = collected
                self._grid[ny, nx] = EMPTY
                if len(collected) == len(self._rule.order):
                    self._flags["goal_active"] = "1"
                    self._mark_goal_active()
                    return 0.35, "correct_order_complete"
                return 0.15, "correct_collect"
            self._inventory["sequence"] = ""
            return -0.2, "wrong_order"
        if self._rule.kind == "delayed_order_unlock" and (
            cell in self._rule.order or cell == self._rule.decoy_color
        ):
            if cell == self._rule.decoy_color:
                self._progress_index = 0
                return 0.08, "decoy_reward_reset"
            expected = self._rule.order[self._progress_index] if self._progress_index < len(self._rule.order) else None
            if expected is None:
                return -0.05, "redundant_collect"
            if cell == expected:
                self._progress_index += 1
                if self._progress_index >= len(self._rule.order):
                    self._flags["goal_active"] = "1"
                    self._mark_goal_active()
                    return 0.0, "delayed_sequence_complete"
                return 0.0, "delayed_correct_collect"
            self._progress_index = 0
            return -0.05, "wrong_order_reset"
        if self._rule.kind == "selector_sequence_unlock" and (
            cell in self._rule.order or cell == self._rule.decoy_color
        ):
            if cell == self._rule.decoy_color:
                self._progress_index = 0
                return 0.08, "decoy_reward_reset"
            if self._selected_color != self._rule.target_color:
                self._progress_index = 0
                return 0.02, "false_progress_under_wrong_selector"
            expected = self._rule.order[self._progress_index] if self._progress_index < len(self._rule.order) else None
            if expected is None:
                return -0.05, "redundant_collect"
            if cell == expected:
                self._progress_index += 1
                if self._progress_index >= len(self._rule.order):
                    self._flags["goal_active"] = "1"
                    self._mark_goal_active()
                    return 0.0, "selector_sequence_complete"
                return 0.0, "selector_sequence_progress"
            self._progress_index = 0
            return -0.05, "wrong_order_reset"
        return -0.05, "empty_interaction"

    def _move_agent(self, next_pos: tuple[int, int]) -> None:
        current = self._agent
        if current == next_pos:
            return
        if self._grid[current] == AGENT:
            self._grid[current] = (
                GOAL_ACTIVE
                if current == self._target_pos and self._goal_is_active()
                else TARGET
                if current == self._target_pos
                else EMPTY
            )
        self._agent = next_pos
        self._grid[next_pos] = AGENT

    def _goal_is_active(self) -> bool:
        return self._flags.get("goal_active") == "1"

    def _mark_goal_active(self) -> None:
        self._grid[self._target_pos] = GOAL_ACTIVE

    def _observation(self) -> GridObservation:
        extras = {
            "cell_tags": self._build_cell_tags(),
            "world_type": "synthetic_hidden_rule",
            "action_roles": dict(self._action_roles),
        }
        return GridObservation(
            task_id=self.task_id,
            episode_id=f"synthetic_hidden_rule/{self._base_seed}_{self._episode_index - 1}",
            step_index=self._step,
            grid=self._grid.copy(),
            available_actions=self.legal_actions(),
            extras=extras,
        )

    @staticmethod
    def _color_name(color: int) -> str:
        return {
            SWITCH_RED: "red",
            COLLECT_RED: "red",
            SWITCH_BLUE: "blue",
            COLLECT_BLUE: "blue",
            SWITCH_GREEN: "green",
            SWITCH_YELLOW: "yellow",
        }.get(color, f"color_{color}")

    def _build_cell_tags(self) -> dict[tuple[int, int], tuple[str, ...]]:
        tags: dict[tuple[int, int], tuple[str, ...]] = {}
        for y in range(self._grid.shape[0]):
            for x in range(self._grid.shape[1]):
                value = int(self._grid[y, x])
                if value == EMPTY:
                    continue
                if (y, x) in self._cell_tag_overrides:
                    tags[(y, x)] = self._cell_tag_overrides[(y, x)]
                    continue
                if value == WALL:
                    tags[(y, x)] = ("blocking", "wall")
                elif value == TARGET:
                    tags[(y, x)] = ("target",)
                elif value in SWITCH_COLORS:
                    tags[(y, x)] = ("interactable",)
                elif value in COLLECT_COLORS:
                    tags[(y, x)] = ("interactable",)
                elif value == AGENT:
                    tags[(y, x)] = ("agent",)
                elif value == GOAL_ACTIVE:
                    tags[(y, x)] = ("target", "active")
        return tags


def _name_to_color(name: str) -> int:
    return {
        "red": SWITCH_RED,
        "blue": SWITCH_BLUE,
        "green": SWITCH_GREEN,
        "yellow": SWITCH_YELLOW,
    }[name]


def family_variants_for_mode(family_mode: str) -> tuple[str, ...]:
    if family_mode == "switch_unlock":
        return ("red", "blue", "green")
    if family_mode == "order_collect":
        return ("red_then_blue", "blue_then_red")
    if family_mode == "selector_unlock":
        return ("red", "blue", "green")
    if family_mode == "delayed_order_unlock":
        return ("red_then_blue", "blue_then_red")
    if family_mode == "selector_sequence_unlock":
        return tuple(
            f"{gate}__{order}"
            for gate in ("red", "blue", "green")
            for order in ("red_then_blue", "blue_then_red")
        )
    raise ValueError(f"unknown family_mode: {family_mode}")


def _order_variant_to_tuple(name: str) -> tuple[int, ...]:
    if name == "red_then_blue":
        return (COLLECT_RED, COLLECT_BLUE)
    if name == "blue_then_red":
        return (COLLECT_BLUE, COLLECT_RED)
    raise ValueError(f"unknown order variant: {name}")
