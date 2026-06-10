from __future__ import annotations

import random
from collections import deque
from dataclasses import dataclass, field
from typing import Protocol

import numpy as np

from .arcagi3_adapter import (
    ArcAGI3Observation,
    ArcAGI3StepResult,
    MOVE_DELTAS,
    action_family,
    find_agent,
    in_bounds,
    parse_click,
)


class BaselinePolicy(Protocol):
    name: str

    def reset(self, seed: int) -> None:
        ...

    def choose(self, observation: ArcAGI3Observation) -> str:
        ...

    def observe(self, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult) -> None:
        ...


@dataclass
class RandomLegalPolicy:
    name: str = "random_legal"
    rng: random.Random = field(default_factory=random.Random)

    def reset(self, seed: int) -> None:
        self.rng.seed(seed)

    def choose(self, observation: ArcAGI3Observation) -> str:
        return self.rng.choice(list(observation.available_actions))

    def observe(self, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult) -> None:
        return None


@dataclass
class RepeatLastPolicy:
    name: str = "repeat_last_action"
    last_action: str | None = None

    def reset(self, seed: int) -> None:
        self.last_action = None

    def choose(self, observation: ArcAGI3Observation) -> str:
        if self.last_action in observation.available_actions:
            return str(self.last_action)
        return observation.available_actions[0]

    def observe(self, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult) -> None:
        self.last_action = action


@dataclass
class CoverageGraphPolicy:
    name: str = "coverage_graph_exploration"
    visited: set[tuple[int, int]] = field(default_factory=set)
    action_counts: dict[str, int] = field(default_factory=dict)

    def reset(self, seed: int) -> None:
        self.visited.clear()
        self.action_counts.clear()

    def choose(self, observation: ArcAGI3Observation) -> str:
        grid = observation.grid
        agent = find_agent(grid)
        if agent is not None:
            self.visited.add(agent)
        best = None
        best_score = -1.0e9
        for action in observation.available_actions:
            score = -float(self.action_counts.get(action, 0))
            if action in MOVE_DELTAS and agent is not None:
                dy, dx = MOVE_DELTAS[action]
                ny, nx = agent[0] + dy, agent[1] + dx
                if in_bounds(grid, ny, nx) and int(grid[ny, nx]) != 1:
                    score += 2.0 if (ny, nx) not in self.visited else 0.0
            if action.startswith("click:"):
                score -= 0.25
            if score > best_score:
                best = action
                best_score = score
        return str(best if best is not None else observation.available_actions[0])

    def observe(self, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult) -> None:
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        agent = find_agent(result.observation.grid)
        if agent is not None:
            self.visited.add(agent)


@dataclass
class NoveltyFirstPolicy:
    name: str = "novelty_first"
    seen_cells: set[tuple[int, int]] = field(default_factory=set)
    tried_clicks: set[str] = field(default_factory=set)

    def reset(self, seed: int) -> None:
        self.seen_cells.clear()
        self.tried_clicks.clear()

    def choose(self, observation: ArcAGI3Observation) -> str:
        grid = np.asarray(observation.grid)
        for y, x in np.argwhere(grid != 8):
            self.seen_cells.add((int(y), int(x)))
        agent = find_agent(grid)
        if agent is not None:
            for action in observation.available_actions:
                if action in MOVE_DELTAS:
                    dy, dx = MOVE_DELTAS[action]
                    ny, nx = agent[0] + dy, agent[1] + dx
                    if in_bounds(grid, ny, nx) and (ny, nx) not in self.seen_cells:
                        return action
        for action in observation.available_actions:
            click = parse_click(action)
            if click is not None and action not in self.tried_clicks:
                return action
        return observation.available_actions[0]

    def observe(self, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult) -> None:
        if action.startswith("click:"):
            self.tried_clicks.add(action)


@dataclass
class GreedyObservableDeltaPolicy:
    name: str = "greedy_observable_score_delta"
    rewards_by_family: dict[str, list[float]] = field(default_factory=dict)
    fallback: RandomLegalPolicy = field(default_factory=RandomLegalPolicy)

    def reset(self, seed: int) -> None:
        self.rewards_by_family.clear()
        self.fallback.reset(seed + 19)

    def choose(self, observation: ArcAGI3Observation) -> str:
        best_action = None
        best_value = -1.0e9
        for action in observation.available_actions:
            family = action_family(action)
            values = self.rewards_by_family.get(family, [])
            if not values:
                continue
            value = sum(values) / len(values)
            if value > best_value:
                best_action = action
                best_value = value
        return best_action if best_action is not None else self.fallback.choose(observation)

    def observe(self, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult) -> None:
        self.rewards_by_family.setdefault(action_family(action), []).append(float(result.reward))


@dataclass
class ObservedGraphBFSPolicy:
    name: str = "oracle_free_observed_graph_bfs"
    graph: dict[tuple[str, str], str] = field(default_factory=dict)
    seen_states: set[str] = field(default_factory=set)
    action_counts: dict[str, int] = field(default_factory=dict)

    def reset(self, seed: int) -> None:
        self.graph.clear()
        self.seen_states.clear()
        self.action_counts.clear()

    def choose(self, observation: ArcAGI3Observation) -> str:
        state = state_key(observation)
        self.seen_states.add(state)
        known = self._first_known_path_to_unseen(state)
        if known in observation.available_actions:
            return known
        return min(observation.available_actions, key=lambda action: self.action_counts.get(action, 0))

    def observe(self, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult) -> None:
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        self.graph[(state_key(before), action)] = state_key(result.observation)
        self.seen_states.add(state_key(result.observation))

    def _first_known_path_to_unseen(self, start: str) -> str | None:
        queue: deque[tuple[str, str | None]] = deque([(start, None)])
        visited = {start}
        while queue:
            state, first = queue.popleft()
            for (src, action), dst in self.graph.items():
                if src != state or dst in visited:
                    continue
                candidate_first = action if first is None else first
                if dst not in self.seen_states:
                    return candidate_first
                visited.add(dst)
                queue.append((dst, candidate_first))
        return None


def state_key(observation: ArcAGI3Observation) -> str:
    grid = np.asarray(observation.grid, dtype=np.int64)
    return f"{grid.shape}:{','.join(str(int(v)) for v in grid.reshape(-1))}"


def build_baselines() -> list[BaselinePolicy]:
    return [
        RandomLegalPolicy(),
        RepeatLastPolicy(),
        CoverageGraphPolicy(),
        NoveltyFirstPolicy(),
        GreedyObservableDeltaPolicy(),
        ObservedGraphBFSPolicy(),
    ]

