from __future__ import annotations

from collections import deque

import numpy as np

from arcagi.core.types import ActionName
from arcagi.envs.synthetic import (
    AGENT,
    COLLECT_COLORS,
    EMPTY,
    GOAL_ACTIVE,
    HiddenRuleEnv,
    INTERACT_DELTAS,
    MOVE_DELTAS,
    SWITCH_COLORS,
    TARGET,
    WALL,
)


def oracle_action(env: HiddenRuleEnv) -> ActionName:
    if env._rule is None:
        return "wait"
    if env._rule.kind == "switch_unlock" and not env._goal_is_active():
        target = env._switch_positions.get(env._rule.target_color)
        return _approach_and_interact_position(env, target)
    if env._rule.kind == "order_collect" and not env._goal_is_active():
        sequence = env._inventory.get("sequence", "")
        if len(sequence) < len(env._rule.order):
            target = env._collect_positions.get(env._rule.order[len(sequence)])
            return _approach_and_interact_position(env, target)
    if env._rule.kind == "selector_unlock" and not env._goal_is_active():
        if env._selected_color != env._rule.target_color:
            return _click_selector(env, env._rule.target_color)
        target = env._switch_positions.get(env._rule.target_color)
        return _approach_and_interact_position(env, target)
    if env._rule.kind == "delayed_order_unlock" and not env._goal_is_active():
        if env._progress_index < len(env._rule.order):
            target = env._collect_positions.get(env._rule.order[env._progress_index])
            return _approach_and_interact_position(env, target)
    if env._rule.kind == "selector_sequence_unlock" and not env._goal_is_active():
        if env._selected_color != env._rule.target_color:
            return _click_selector(env, env._rule.target_color)
        if env._progress_index < len(env._rule.order):
            target = env._collect_positions.get(env._rule.order[env._progress_index])
            return _approach_and_interact_position(env, target)
    return _move_toward(env, env._target_pos, goal_required=True)


def _approach_and_interact_position(env: HiddenRuleEnv, target: tuple[int, int] | None) -> ActionName:
    if target is None:
        return "wait"
    for action, (delta_y, delta_x) in INTERACT_DELTAS.items():
        if (env._agent[0] + delta_y, env._agent[1] + delta_x) == target:
            return action
    candidate_cells = [
        cell
        for cell in _adjacent_cells(target)
        if _cell_passable(env._grid, cell, allow_goal=False)
    ]
    path = _shortest_path(env._grid, env._agent, set(candidate_cells), allow_goal=False)
    if not path:
        return "wait"
    return _move_action(env._agent, path[0])


def _click_selector(env: HiddenRuleEnv, target_color: int) -> ActionName:
    target = env._selector_positions.get(target_color)
    if target is None:
        return "wait"
    return f"click:{target[1]}:{target[0]}"


def _move_toward(env: HiddenRuleEnv, target: tuple[int, int], goal_required: bool) -> ActionName:
    path = _shortest_path(env._grid, env._agent, {target}, allow_goal=goal_required)
    if not path:
        return "wait"
    return _move_action(env._agent, path[0])


def _shortest_path(
    grid: np.ndarray,
    start: tuple[int, int],
    goals: set[tuple[int, int]],
    allow_goal: bool,
) -> list[tuple[int, int]]:
    frontier: deque[tuple[int, int]] = deque([start])
    parent: dict[tuple[int, int], tuple[int, int] | None] = {start: None}
    while frontier:
        cell = frontier.popleft()
        if cell in goals:
            return _reconstruct_path(cell, parent)
        for next_cell in _adjacent_cells(cell):
            if next_cell in parent:
                continue
            if not _cell_passable(grid, next_cell, allow_goal=allow_goal or next_cell in goals):
                continue
            parent[next_cell] = cell
            frontier.append(next_cell)
    return []


def _reconstruct_path(
    goal: tuple[int, int],
    parent: dict[tuple[int, int], tuple[int, int] | None],
) -> list[tuple[int, int]]:
    path: list[tuple[int, int]] = []
    current: tuple[int, int] | None = goal
    while current is not None and parent[current] is not None:
        path.append(current)
        current = parent[current]
    path.reverse()
    return path


def _adjacent_cells(cell: tuple[int, int]) -> list[tuple[int, int]]:
    return [(cell[0] - 1, cell[1]), (cell[0] + 1, cell[1]), (cell[0], cell[1] - 1), (cell[0], cell[1] + 1)]


def _cell_passable(grid: np.ndarray, cell: tuple[int, int], allow_goal: bool) -> bool:
    y, x = cell
    if y < 0 or x < 0 or y >= grid.shape[0] or x >= grid.shape[1]:
        return False
    value = int(grid[y, x])
    if value in {EMPTY, AGENT}:
        return True
    if allow_goal and value in {TARGET, GOAL_ACTIVE}:
        return True
    return value not in {WALL, *SWITCH_COLORS, *COLLECT_COLORS, TARGET, GOAL_ACTIVE}


def _move_action(start: tuple[int, int], next_cell: tuple[int, int]) -> ActionName:
    delta = (next_cell[0] - start[0], next_cell[1] - start[1])
    for action, action_delta in MOVE_DELTAS.items():
        if action_delta == delta:
            return action
    return "wait"
