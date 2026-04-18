from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, field
import math

from arcagi.core.types import ActionName, StructuredState, Transition


@dataclass
class ActionStats:
    visits: int = 0
    reward_sum: float = 0.0
    next_counts: Counter[str] = field(default_factory=Counter)

    @property
    def mean_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.reward_sum / self.visits

    @property
    def outcome_entropy(self) -> float:
        if self.visits <= 1:
            return 0.0
        entropy = 0.0
        for count in self.next_counts.values():
            probability = count / self.visits
            entropy -= probability * math.log(probability + 1e-8)
        return entropy


@dataclass
class StateNode:
    fingerprint: str
    affordances: tuple[ActionName, ...]
    visits: int = 0
    last_seen_tick: int = 0
    action_stats: dict[ActionName, ActionStats] = field(default_factory=dict)
    state: StructuredState | None = None


class StateGraph:
    def __init__(self) -> None:
        self.nodes: dict[str, StateNode] = {}
        self._tick = 0

    def clear(self) -> None:
        self.nodes.clear()
        self._tick = 0

    def ensure_node(self, state: StructuredState) -> StateNode:
        fingerprint = state.fingerprint()
        node = self.nodes.get(fingerprint)
        if node is None:
            node = StateNode(
                fingerprint=fingerprint,
                affordances=state.affordances,
                visits=0,
                last_seen_tick=0,
                state=state,
            )
            self.nodes[fingerprint] = node
        node.affordances = state.affordances
        node.state = state
        return node

    def visit(self, state: StructuredState) -> StateNode:
        node = self.ensure_node(state)
        self._tick += 1
        node.visits += 1
        node.last_seen_tick = self._tick
        return node

    def update(self, transition: Transition) -> None:
        source = self.visit(transition.state)
        self.visit(transition.next_state)
        stats = source.action_stats.setdefault(transition.action, ActionStats())
        stats.visits += 1
        stats.reward_sum += transition.reward
        stats.next_counts[transition.next_state.fingerprint()] += 1

    def get_action_stats(self, state: StructuredState, action: ActionName) -> ActionStats:
        node = self.ensure_node(state)
        return node.action_stats.setdefault(action, ActionStats())

    def action_novelty(self, state: StructuredState, action: ActionName) -> float:
        stats = self.get_action_stats(state, action)
        if stats.visits == 0:
            return 1.0
        return 1.0 / math.sqrt(stats.visits + 1)

    def action_outcome_entropy(self, state: StructuredState, action: ActionName) -> float:
        return self.get_action_stats(state, action).outcome_entropy

    def action_cycle_penalty(self, state: StructuredState, action: ActionName, recency_horizon: int = 6) -> float:
        stats = self.get_action_stats(state, action)
        if stats.visits == 0 or not stats.next_counts:
            return 0.0
        current_node = self.ensure_node(state)
        next_fingerprint, next_count = stats.next_counts.most_common(1)[0]
        next_node = self.nodes.get(next_fingerprint)
        if next_node is None:
            return 0.0
        tick_gap = max(current_node.last_seen_tick - next_node.last_seen_tick, 0)
        if tick_gap > recency_horizon:
            return 0.0
        revisit_strength = (recency_horizon - tick_gap + 1) / float(recency_horizon + 1)
        determinism = next_count / max(stats.visits, 1)
        self_loop_bonus = 0.5 if next_fingerprint == state.fingerprint() else 0.0
        return (determinism * revisit_strength) + self_loop_bonus

    def frontier_actions(self, state: StructuredState) -> list[ActionName]:
        node = self.ensure_node(state)
        return [
            action
            for action in node.affordances
            if node.action_stats.get(action, ActionStats()).visits == 0
        ] or list(node.affordances)

    def best_frontier_action(self, state: StructuredState) -> ActionName:
        best_action = state.affordances[0]
        best_score = float("-inf")
        for action in state.affordances:
            score = self.action_novelty(state, action) + 0.5 * self.action_outcome_entropy(state, action)
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def shortest_path_to_frontier(
        self,
        start_state: StructuredState,
        max_depth: int = 4,
    ) -> tuple[ActionName, ...]:
        start_fingerprint = start_state.fingerprint()
        if start_fingerprint not in self.nodes:
            self.visit(start_state)
        frontier: deque[tuple[str, tuple[ActionName, ...], int]] = deque([(start_fingerprint, (), 0)])
        visited = {start_fingerprint}
        while frontier:
            fingerprint, path, depth = frontier.popleft()
            node = self.nodes[fingerprint]
            if any(
                node.action_stats.get(action, ActionStats()).visits == 0
                for action in node.affordances
            ):
                return path
            if depth >= max_depth:
                continue
            for action, stats in node.action_stats.items():
                if not stats.next_counts:
                    continue
                next_fingerprint, _ = stats.next_counts.most_common(1)[0]
                if next_fingerprint in visited or next_fingerprint not in self.nodes:
                    continue
                visited.add(next_fingerprint)
                frontier.append((next_fingerprint, path + (action,), depth + 1))
        return ()
