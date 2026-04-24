from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np

from arcagi.core.progress_signals import action_family
from arcagi.core.types import ActionName, Position, StructuredState

INTERACT_DELTAS: dict[ActionName, tuple[int, int]] = {
    "interact_up": (-1, 0),
    "interact_down": (1, 0),
    "interact_left": (0, -1),
    "interact_right": (0, 1),
}


def _agent_position(state: StructuredState) -> Position | None:
    agent = next((obj for obj in state.objects if "agent" in obj.tags), None)
    if agent is None or not agent.cells:
        return None
    return tuple(sorted(agent.cells)[0])


def _state_delta(before: StructuredState, after: StructuredState) -> float:
    return float(np.linalg.norm(after.transition_vector() - before.transition_vector()))


def _changed_positions(before: StructuredState, after: StructuredState) -> list[Position]:
    if before.grid_shape != after.grid_shape:
        return []
    before_grid = before.as_grid()
    after_grid = after.as_grid()
    ys, xs = np.nonzero(before_grid != after_grid)
    return [(int(y), int(x)) for y, x in zip(ys.tolist(), xs.tolist(), strict=False)]


def _centroid_position(positions: list[Position]) -> Position | None:
    if not positions:
        return None
    mean_y = int(round(sum(y for y, _ in positions) / len(positions)))
    mean_x = int(round(sum(x for _, x in positions) / len(positions)))
    return (mean_y, mean_x)


def _nearest_distance(origin: Position | None, sites: set[Position]) -> int | None:
    if origin is None or not sites:
        return None
    return min(abs(origin[0] - y) + abs(origin[1] - x) for y, x in sites)


def _distance_bucket(distance: int | None) -> str:
    if distance is None:
        return "none"
    if distance <= 1:
        return "near"
    if distance <= 3:
        return "mid"
    return "far"


def _direction_from_to(origin: Position | None, target: Position | None) -> str:
    if origin is None or target is None:
        return "none"
    dy = target[0] - origin[0]
    dx = target[1] - origin[1]
    if abs(dy) >= abs(dx) and dy != 0:
        return "down" if dy > 0 else "up"
    if dx != 0:
        return "right" if dx > 0 else "left"
    return "none"


def _visited_ratio_bucket(visited_cells: int, grid_shape: tuple[int, int]) -> str:
    total_cells = max(grid_shape[0] * grid_shape[1], 1)
    ratio = visited_cells / float(total_cells)
    if ratio <= 0.1:
        return "0"
    if ratio <= 0.3:
        return "1"
    if ratio <= 0.6:
        return "2"
    if ratio <= 0.9:
        return "3"
    return "4"


def _nearest_unvisited(origin: Position | None, visited_positions: set[Position], grid_shape: tuple[int, int]) -> Position | None:
    if origin is None:
        return None
    best: Position | None = None
    best_distance: int | None = None
    for y in range(grid_shape[0]):
        for x in range(grid_shape[1]):
            candidate = (y, x)
            if candidate in visited_positions:
                continue
            distance = abs(origin[0] - y) + abs(origin[1] - x)
            if best_distance is None or distance < best_distance:
                best = candidate
                best_distance = distance
    return best


def _nearest_site(origin: Position | None, sites: set[Position]) -> Position | None:
    if origin is None or not sites:
        return None
    return min(sites, key=lambda site: abs(origin[0] - site[0]) + abs(origin[1] - site[1]))


@dataclass
class SpatialBeliefWorkspace:
    visited_positions: set[Position] = field(default_factory=set)
    tested_sites: set[Position] = field(default_factory=set)
    effect_sites: set[Position] = field(default_factory=set)
    contradiction_sites: set[Position] = field(default_factory=set)
    last_anchor: Position | None = None

    def reset(self) -> None:
        self.visited_positions.clear()
        self.tested_sites.clear()
        self.effect_sites.clear()
        self.contradiction_sites.clear()
        self.last_anchor = None

    def reset_level(self) -> None:
        self.visited_positions.clear()

    def observe_state(self, state: StructuredState) -> None:
        position = _agent_position(state)
        if position is not None:
            self.visited_positions.add(position)

    def observe_transition(
        self,
        before: StructuredState,
        action: ActionName,
        reward: float,
        after: StructuredState,
        *,
        terminated: bool,
    ) -> None:
        before_position = _agent_position(before)
        after_position = _agent_position(after)
        if before_position is not None:
            self.visited_positions.add(before_position)
        if after_position is not None:
            self.visited_positions.add(after_position)

        family = action_family(action)
        delta_norm = _state_delta(before, after)
        changed_positions = _changed_positions(before, after)
        attempted_site = None
        if before_position is not None and action in INTERACT_DELTAS:
            dy, dx = INTERACT_DELTAS[action]
            attempted_site = (before_position[0] + dy, before_position[1] + dx)

        if attempted_site is not None:
            self.tested_sites.add(attempted_site)

        positive = reward > 0.0 or delta_norm >= 0.35 or (terminated and reward >= 0.0)
        contradiction = reward <= -0.05 or (family in {"interact", "click"} and delta_norm < 0.05 and reward <= 0.0)

        effect_site = _centroid_position(changed_positions)
        if effect_site is None:
            if positive and after_position is not None and family == "move":
                effect_site = after_position
            else:
                effect_site = attempted_site

        if positive and effect_site is not None:
            self.effect_sites.add(effect_site)
            self.last_anchor = effect_site
        if contradiction and attempted_site is not None:
            self.contradiction_sites.add(attempted_site)

    def augment(self, state: StructuredState) -> StructuredState:
        inventory = dict(state.inventory)
        flags = dict(state.flags)
        position = _agent_position(state)
        nearest_frontier = _nearest_unvisited(position, self.visited_positions, state.grid_shape)
        nearest_anchor_site = _nearest_site(position, self.effect_sites)
        nearest_contradiction_site = _nearest_site(position, self.contradiction_sites)
        nearest_anchor = _nearest_distance(position, self.effect_sites)
        nearest_tested = _nearest_distance(position, self.tested_sites)
        nearest_contradiction = _nearest_distance(position, self.contradiction_sites)
        inventory["belief_visited_cells"] = str(len(self.visited_positions))
        inventory["belief_visited_ratio_bucket"] = _visited_ratio_bucket(len(self.visited_positions), state.grid_shape)
        inventory["belief_tested_sites"] = str(len(self.tested_sites))
        inventory["belief_effect_sites"] = str(len(self.effect_sites))
        inventory["belief_contradiction_sites"] = str(len(self.contradiction_sites))
        inventory["belief_frontier_distance"] = _distance_bucket(
            None if nearest_frontier is None or position is None else abs(position[0] - nearest_frontier[0]) + abs(position[1] - nearest_frontier[1])
        )
        inventory["belief_frontier_direction"] = _direction_from_to(position, nearest_frontier)
        inventory["belief_nearest_anchor_distance"] = _distance_bucket(nearest_anchor)
        inventory["belief_nearest_tested_distance"] = _distance_bucket(nearest_tested)
        inventory["belief_nearest_contradiction_distance"] = _distance_bucket(nearest_contradiction)
        inventory["belief_anchor_direction"] = _direction_from_to(position, nearest_anchor_site)
        inventory["belief_contradiction_direction"] = _direction_from_to(position, nearest_contradiction_site)
        flags["belief_has_spatial_anchor"] = "1" if self.effect_sites else "0"
        flags["belief_near_spatial_anchor"] = "1" if nearest_anchor is not None and nearest_anchor <= 1 else "0"
        flags["belief_near_tested_site"] = "1" if nearest_tested is not None and nearest_tested <= 1 else "0"
        flags["belief_has_contradiction_hotspot"] = "1" if self.contradiction_sites else "0"
        return replace(
            state,
            inventory=tuple(sorted((str(key), str(value)) for key, value in inventory.items())),
            flags=tuple(sorted((str(key), str(value)) for key, value in flags.items())),
        )
