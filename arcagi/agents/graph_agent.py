from __future__ import annotations

from collections import defaultdict
import math

import numpy as np

from arcagi.agents.base import BaseAgent
from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.types import ActionName, GridObservation, Transition
from arcagi.planning.rule_induction import EpisodeRuleInducer


class GraphExplorerAgent(BaseAgent):
    def __init__(self) -> None:
        super().__init__(name="graph")
        self.global_action_counts: dict[ActionName, int] = defaultdict(int)
        self.global_action_delta_sum: dict[ActionName, float] = defaultdict(float)
        self.global_action_reward_sum: dict[ActionName, float] = defaultdict(float)
        self.family_counts: dict[str, int] = defaultdict(int)
        self.family_bins: dict[str, set[tuple[int, int]]] = defaultdict(set)
        self.rule_inducer = EpisodeRuleInducer()
        self.stuck_steps = 0

    def act(self, observation: GridObservation) -> ActionName:
        state = self.observe(observation)
        frontier_path = self.graph.shortest_path_to_frontier(state, max_depth=3)
        frontier_action = frontier_path[0] if frontier_path else None
        action = self._score_actions(state, frontier_action)
        self.last_state = state
        self.last_action = action
        return action

    def reset_episode(self) -> None:
        super().reset_episode()
        self.global_action_counts.clear()
        self.global_action_delta_sum.clear()
        self.global_action_reward_sum.clear()
        self.family_counts.clear()
        self.family_bins.clear()
        self.rule_inducer.clear()
        self.stuck_steps = 0

    def reset_level(self) -> None:
        super().reset_level()
        self.global_action_counts.clear()
        self.global_action_delta_sum.clear()
        self.global_action_reward_sum.clear()
        self.family_counts.clear()
        self.family_bins.clear()
        self.stuck_steps = 0

    def on_transition(self, transition: Transition) -> None:
        action = transition.action
        delta_pixels = self._grid_delta_pixels(transition)
        self.global_action_counts[action] += 1
        self.global_action_delta_sum[action] += delta_pixels
        self.global_action_reward_sum[action] += transition.reward
        if delta_pixels <= 1.0 and transition.reward <= 0.0:
            self.stuck_steps += 1
        else:
            self.stuck_steps = 0
        context = build_action_schema_context(transition.state.affordances, dict(transition.state.action_roles))
        schema = build_action_schema(action, context)
        self.family_counts[schema.family] += 1
        if schema.coarse_bin is not None:
            self.family_bins[schema.family].add(schema.coarse_bin)
        self.rule_inducer.record(transition)

    def _score_actions(self, state, frontier_action: ActionName | None) -> ActionName:
        best_action = state.affordances[0]
        best_score = float("-inf")
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        for action in state.affordances:
            schema = build_action_schema(action, context)
            local_novelty = self.graph.action_novelty(state, action)
            entropy = self.graph.action_outcome_entropy(state, action)
            global_count = self.global_action_counts[action]
            global_novelty = 1.0 / math.sqrt(global_count + 1.0)
            mean_delta = self.global_action_delta_sum[action] / max(global_count, 1)
            mean_reward = self.global_action_reward_sum[action] / max(global_count, 1)
            repeat_penalty = 0.35 if action == self.last_action else 0.0
            frontier_bonus = 0.3 if frontier_action == action else 0.0
            stuck_bonus = (0.1 * min(self.stuck_steps, 8)) * global_novelty
            induced_score = self.rule_inducer.action_score(state, action)
            family_count = self.family_counts[schema.family]
            family_novelty = 1.0 / math.sqrt(family_count + 1.0)
            parameter_bonus = 0.0
            if schema.coarse_bin is not None and schema.coarse_bin not in self.family_bins[schema.family]:
                parameter_bonus = 0.35
            score = (
                (1.1 * local_novelty)
                + (0.8 * global_novelty)
                + (0.6 * family_novelty)
                + (0.25 * entropy)
                + (0.003 * mean_delta)
                + (0.5 * mean_reward)
                + frontier_bonus
                + stuck_bonus
                + parameter_bonus
                + induced_score
                - repeat_penalty
            )
            if score > best_score:
                best_score = score
                best_action = action
        return best_action

    def _grid_delta_pixels(self, transition: Transition) -> float:
        current = np.asarray(transition.state.grid_signature, dtype=np.int16)
        nxt = np.asarray(transition.next_state.grid_signature, dtype=np.int16)
        overlap = min(current.size, nxt.size)
        delta = float(np.count_nonzero(current[:overlap] != nxt[:overlap]))
        delta += float(abs(current.size - nxt.size))
        return delta
