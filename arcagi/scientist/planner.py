"""Information-gain planner for the scientist agent."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Iterable, Mapping

import numpy as np

from .effects import state_numeric_channels, transition_numeric_deltas
from .hypotheses import HypothesisEngine
from .language import GroundedLanguage
from .memory import EpisodicMemory
from .types import (
    ActionDecision,
    ActionName,
    ObjectToken,
    StructuredState,
    TransitionRecord,
    action_delta,
    action_family,
    combined_progress_signal,
    grid_cell_to_action_coordinates,
    is_failure_terminal_game_state,
    is_interact_action,
    is_move_action,
    is_reset_action,
    is_win_game_state,
    make_targeted_action,
    observation_game_state,
    parse_action_target,
)
from .world_model import OnlineWorldModel


@dataclass(frozen=True)
class PlannerConfig:
    reward_weight: float = 2.4
    information_weight: float = 1.7
    novelty_weight: float = 0.85
    change_weight: float = 0.45
    uncertainty_weight: float = 0.35
    memory_weight: float = 0.75
    mechanic_weight: float = 0.90
    option_weight: float = 0.75
    efficiency_weight: float = 0.65
    cost_weight: float = 0.50
    spatial_goal_weight: float = 1.15
    navigation_weight: float = 2.10
    repeat_penalty: float = 0.55
    max_candidates: int = 0
    random_tie_noise: float = 1e-4


class ScientistPlanner:
    def __init__(self, *, config: PlannerConfig | None = None, seed: int = 0) -> None:
        self.config = config or PlannerConfig()
        self.rng = np.random.default_rng(seed)
        self.state_action_visits: dict[tuple[str, ActionName], int] = defaultdict(int)
        self.action_visits: dict[ActionName, int] = defaultdict(int)
        self.last_action: ActionName | None = None
        self.last_state_fp: str | None = None
        self.stall_count = 0
        self.ineffective_actions: dict[tuple[str, ActionName], int] = defaultdict(int)
        self.blocked_cells: set[tuple[int, int]] = set()
        self.visited_actor_positions: set[tuple[int, int]] = set()
        self.nonproductive_colors: dict[int, int] = defaultdict(int)
        self.productive_colors: dict[int, int] = defaultdict(int)
        self.pending_interactions = 0
        self.family_cost_sum: dict[str, float] = defaultdict(float)
        self.family_cost_count: dict[str, int] = defaultdict(int)
        self.family_effect_sum: dict[str, float] = defaultdict(float)
        self.family_effect_count: dict[str, int] = defaultdict(int)
        self.channel_min: dict[str, float] = {}
        self.channel_max: dict[str, float] = {}
        self.channel_spend_sum: dict[str, float] = defaultdict(float)
        self.channel_spend_count: dict[str, int] = defaultdict(int)

    def reset_episode(self) -> None:
        self.state_action_visits.clear()
        self.action_visits.clear()
        self.last_action = None
        self.last_state_fp = None
        self.stall_count = 0
        self.ineffective_actions.clear()
        self.blocked_cells.clear()
        self.visited_actor_positions.clear()
        self.nonproductive_colors.clear()
        self.productive_colors.clear()
        self.pending_interactions = 0
        self.family_cost_sum.clear()
        self.family_cost_count.clear()
        self.family_effect_sum.clear()
        self.family_effect_count.clear()
        self.channel_min.clear()
        self.channel_max.clear()
        self.channel_spend_sum.clear()
        self.channel_spend_count.clear()

    def reset_level(self) -> None:
        self.state_action_visits.clear()
        self.action_visits.clear()
        self.last_action = None
        self.last_state_fp = None
        self.stall_count = 0
        self.ineffective_actions.clear()
        self.blocked_cells.clear()
        self.visited_actor_positions.clear()
        self.pending_interactions = 0

    def state_dict(self) -> dict[str, Any]:
        return {
            "rng_state": self.rng.bit_generator.state,
            "state_action_visits": [(list(key), int(value)) for key, value in self.state_action_visits.items()],
            "action_visits": [(str(key), int(value)) for key, value in self.action_visits.items()],
            "last_action": self.last_action,
            "last_state_fp": self.last_state_fp,
            "stall_count": int(self.stall_count),
            "ineffective_actions": [(list(key), int(value)) for key, value in self.ineffective_actions.items()],
            "blocked_cells": [list(cell) for cell in sorted(self.blocked_cells)],
            "visited_actor_positions": [list(cell) for cell in sorted(self.visited_actor_positions)],
            "nonproductive_colors": [(int(key), int(value)) for key, value in self.nonproductive_colors.items()],
            "productive_colors": [(int(key), int(value)) for key, value in self.productive_colors.items()],
            "pending_interactions": int(self.pending_interactions),
            "family_cost_sum": [(str(key), float(value)) for key, value in self.family_cost_sum.items()],
            "family_cost_count": [(str(key), int(value)) for key, value in self.family_cost_count.items()],
            "family_effect_sum": [(str(key), float(value)) for key, value in self.family_effect_sum.items()],
            "family_effect_count": [(str(key), int(value)) for key, value in self.family_effect_count.items()],
            "channel_min": dict(self.channel_min),
            "channel_max": dict(self.channel_max),
            "channel_spend_sum": [(str(key), float(value)) for key, value in self.channel_spend_sum.items()],
            "channel_spend_count": [(str(key), int(value)) for key, value in self.channel_spend_count.items()],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.reset_episode()
        rng_state = state.get("rng_state")
        if isinstance(rng_state, Mapping):
            try:
                self.rng.bit_generator.state = dict(rng_state)
            except Exception:
                pass
        for key, value in state.get("state_action_visits", []):
            self.state_action_visits[(str(key[0]), str(key[1]))] = int(value)
        for key, value in state.get("action_visits", []):
            self.action_visits[str(key)] = int(value)
        last_action = state.get("last_action")
        self.last_action = None if last_action is None else str(last_action)
        last_state_fp = state.get("last_state_fp")
        self.last_state_fp = None if last_state_fp is None else str(last_state_fp)
        self.stall_count = int(state.get("stall_count", 0))
        for key, value in state.get("ineffective_actions", []):
            self.ineffective_actions[(str(key[0]), str(key[1]))] = int(value)
        self.blocked_cells = {
            (int(cell[0]), int(cell[1]))
            for cell in state.get("blocked_cells", [])
            if isinstance(cell, (list, tuple)) and len(cell) == 2
        }
        self.visited_actor_positions = {
            (int(cell[0]), int(cell[1]))
            for cell in state.get("visited_actor_positions", [])
            if isinstance(cell, (list, tuple)) and len(cell) == 2
        }
        for key, value in state.get("nonproductive_colors", []):
            self.nonproductive_colors[int(key)] = int(value)
        for key, value in state.get("productive_colors", []):
            self.productive_colors[int(key)] = int(value)
        self.pending_interactions = int(state.get("pending_interactions", 0))
        for key, value in state.get("family_cost_sum", []):
            self.family_cost_sum[str(key)] = float(value)
        for key, value in state.get("family_cost_count", []):
            self.family_cost_count[str(key)] = int(value)
        for key, value in state.get("family_effect_sum", []):
            self.family_effect_sum[str(key)] = float(value)
        for key, value in state.get("family_effect_count", []):
            self.family_effect_count[str(key)] = int(value)
        channel_min = state.get("channel_min", {})
        self.channel_min = {str(key): float(value) for key, value in channel_min.items()} if isinstance(channel_min, Mapping) else {}
        channel_max = state.get("channel_max", {})
        self.channel_max = {str(key): float(value) for key, value in channel_max.items()} if isinstance(channel_max, Mapping) else {}
        for key, value in state.get("channel_spend_sum", []):
            self.channel_spend_sum[str(key)] = float(value)
        for key, value in state.get("channel_spend_count", []):
            self.channel_spend_count[str(key)] = int(value)

    def notify_transition(
        self,
        *,
        changed: bool,
        record: TransitionRecord | None = None,
        engine: HypothesisEngine | None = None,
    ) -> None:
        if changed:
            self.stall_count = 0
        else:
            self.stall_count += 1

        if record is None:
            return
        if engine is not None:
            self._update_contact_statistics(record, engine)
        fam = action_family(record.action)
        observed_effect = self._observed_effect_value(record)
        observed_cost = self._observed_action_cost(record, observed_effect=observed_effect)
        self.family_cost_sum[fam] += float(observed_cost)
        self.family_cost_count[fam] += 1
        self.family_effect_sum[fam] += float(observed_effect)
        self.family_effect_count[fam] += 1
        self._update_numeric_channel_stats(record)
        key = (record.before.exact_fingerprint, record.action)
        if changed:
            self.ineffective_actions.pop(key, None)
            return
        self.ineffective_actions[key] += 1
        delta = action_delta(record.action)
        if delta is None or engine is None:
            return
        actor = self._controlled_object(record.before, engine)
        if actor is None:
            return
        ar, ac = actor.center_cell
        nr = ar + delta[0]
        nc = ac + delta[1]
        rows, cols = record.before.grid.shape
        if 0 <= nr < rows and 0 <= nc < cols:
            self.blocked_cells.add((nr, nc))

    def choose_action(
        self,
        state: StructuredState,
        *,
        engine: HypothesisEngine,
        world_model: OnlineWorldModel,
        memory: EpisodicMemory,
        language: GroundedLanguage,
    ) -> ActionDecision:
        lang_tokens = language.memory_tokens(state, engine)
        candidates = self.candidate_actions(state, engine=engine, memory=memory, language_tokens=lang_tokens)
        if not candidates:
            raise RuntimeError("No legal actions are available for ScientistPlanner")

        game_state = observation_game_state(state)
        if is_failure_terminal_game_state(game_state):
            for action in candidates:
                if is_reset_action(action):
                    return ActionDecision(
                        action=action,
                        score=10.0,
                        components={
                            "expected_reward": 0.0,
                            "information_gain": 0.0,
                            "novelty": 0.0,
                            "expected_change": 0.0,
                            "world_uncertainty": 0.0,
                            "memory_bonus": 0.0,
                            "spatial_goal": 0.0,
                            "navigation_goal": 0.0,
                            "risk": 0.0,
                            "repeat_penalty": 0.0,
                            "posterior_mass": 0.0,
                            "reset_retry_bonus": 1.0,
                        },
                        language=(f"plan: do reset; because terminal state {game_state} requires retry",),
                        candidate_count=len(candidates),
                        chosen_reason=f"terminal state {game_state} requires reset to continue the environment",
                    )

        navigation_action = self._navigation_next_action(state, engine, candidates)
        scored: list[tuple[float, ActionName, dict[str, float], tuple[str, ...]]] = []
        budget_pressure = self._budget_pressure(state)
        for action in candidates:
            hyp_score = engine.score_action(state, action)
            world = world_model.predict(state, action)
            novelty = self._novelty(state, action)
            memory_bonus = memory.action_memory_bonus(state, action, lang_tokens)
            option_profile = self._option_profile(memory, state, action, lang_tokens)
            mechanic_goal = self._mechanic_goal_value(state, action, engine)
            repeat = self._repeat_penalty(state, action)
            spatial_goal = self._spatial_goal_value(state, action, engine)
            navigation_goal = 1.0 if navigation_action == action else 0.0
            action_cost = self._action_cost_estimate(action, option_profile=option_profile)
            efficiency = self._action_efficiency_prior(action, option_profile=option_profile)
            components = {
                "expected_reward": hyp_score.expected_reward + world.reward_mean,
                "information_gain": hyp_score.information_gain,
                "novelty": novelty,
                "expected_change": hyp_score.expected_change + world.change_mean,
                "world_uncertainty": world.total_uncertainty,
                "memory_bonus": memory_bonus,
                "option_schema_bonus": option_profile["schema_bonus"],
                "option_continuation": option_profile["continuation_depth"],
                "option_efficiency": option_profile["efficiency"],
                "action_cost": action_cost,
                "budget_pressure": budget_pressure,
                "family_efficiency": efficiency,
                "mechanic_goal": mechanic_goal,
                "spatial_goal": spatial_goal,
                "navigation_goal": navigation_goal,
                "risk": hyp_score.risk,
                "repeat_penalty": repeat,
                "posterior_mass": hyp_score.posterior_mass,
            }
            score = (
                self.config.reward_weight * components["expected_reward"]
                + self.config.information_weight * components["information_gain"]
                + self.config.novelty_weight * components["novelty"]
                + self.config.change_weight * components["expected_change"]
                + self.config.uncertainty_weight * components["world_uncertainty"]
                + self.config.memory_weight * components["memory_bonus"]
                + self.config.option_weight * (components["option_schema_bonus"] + (0.25 * components["option_continuation"]))
                + self.config.efficiency_weight * components["family_efficiency"]
                + self.config.mechanic_weight * components["mechanic_goal"]
                + self.config.spatial_goal_weight * components["spatial_goal"]
                + self.config.navigation_weight * components["navigation_goal"]
                - self.config.cost_weight * components["action_cost"] * components["budget_pressure"]
                - components["risk"]
                - repeat
            )
            score += float(self.rng.normal(0.0, self.config.random_tie_noise))
            scored.append((float(score), action, components, hyp_score.rationale))

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_action, best_components, rationale = scored[0]
        self.state_action_visits[(state.abstract_fingerprint, best_action)] += 1
        self.action_visits[best_action] += 1
        if is_interact_action(best_action) and self.pending_interactions > 0:
            self.pending_interactions -= 1
        self.last_action = best_action
        self.last_state_fp = state.exact_fingerprint
        reason = " ".join(rationale[:5]) if rationale else "maximize online experiment value"
        plan = language.plan_sentence(best_action, components=best_components, reason=reason)
        return ActionDecision(
            action=best_action,
            score=best_score,
            components=best_components,
            language=(*language.belief_sentences(engine, limit=4), *language.questions(engine, limit=3), plan),
            candidate_count=len(candidates),
            chosen_reason=reason,
        )

    def candidate_actions(
        self,
        state: StructuredState,
        *,
        engine: HypothesisEngine,
        memory: EpisodicMemory | None = None,
        language_tokens: Iterable[str] = (),
    ) -> tuple[ActionName, ...]:
        legal = tuple(state.available_actions)
        if not legal:
            legal = ("up", "down", "left", "right", "interact", "click")
        candidates: list[ActionName] = list(legal)
        candidates.extend(engine.diagnostic_actions(state, legal, limit=48))
        if memory is not None:
            legal_families = {action_family(action) for action in legal}
            for option in memory.retrieve_options(state, language_tokens, k=8):
                entry = option.first_action
                if entry in legal or action_family(entry) in legal_families:
                    candidates.append(entry)

        # Expand click-like actions to the full grid. Restricting this before
        # training would hide legal action parameters from the learner.
        click_bases = [a for a in legal if action_family(a) in {"click", "action6", "interact_at", "select_at"}]
        if click_bases:
            base = click_bases[0]
            rows, cols = state.grid.shape
            for r in range(rows):
                for c in range(cols):
                    tr, tc = grid_cell_to_action_coordinates(r, c, state.frame.extras)
                    candidates.append(make_targeted_action(base, tr, tc))

        # If all recent actions are stalling, force broad coverage before repeating.
        if self.stall_count >= 3:
            candidates = sorted(candidates, key=lambda a: (self.action_visits[a], self.state_action_visits[(state.abstract_fingerprint, a)]))

        deduped = tuple(dict.fromkeys(candidates))
        max_candidates = int(self.config.max_candidates)
        if max_candidates > 0:
            legal_set = set(legal)
            extras = [action for action in deduped if action not in legal_set]
            return (*legal, *extras[: max(0, max_candidates - len(legal))])
        return deduped



    def _navigation_next_action(
        self,
        state: StructuredState,
        engine: HypothesisEngine,
        legal_actions: tuple[ActionName, ...],
    ) -> ActionName | None:
        move_by_delta: dict[tuple[int, int], ActionName] = {}
        for action in legal_actions:
            delta = action_delta(action)
            if delta is not None:
                move_by_delta.setdefault(delta, action)
        if self.pending_interactions > 0:
            for action in legal_actions:
                if is_interact_action(action) and parse_action_target(action) is None:
                    return action
        if not move_by_delta:
            return None
        actor = self._controlled_object(state, engine)
        if actor is None:
            return None
        start = actor.center_cell
        self.visited_actor_positions.add(start)
        color_priors = engine.color_progress_priors()
        mechanic_priors = engine.mechanic_color_priors()
        best: tuple[float, ObjectToken, int] | None = None
        for obj in state.objects:
            if obj.object_id == actor.object_id:
                continue
            if obj.color == actor.color and obj.shape_hash == actor.shape_hash:
                continue
            dist = self._learned_grid_distance(start, obj.center_cell, state)
            if dist is None or dist == 0:
                continue
            weight = 0.20
            weight += color_priors.get(obj.color, 0.0)
            weight += 0.35 * mechanic_priors.get(obj.color, 0.0)
            weight += 0.20 * self.productive_colors.get(obj.color, 0)
            weight -= 0.35 * self.nonproductive_colors.get(obj.color, 0)
            if "point" in obj.role_tags or "small" in obj.role_tags:
                weight += 0.10
            if "large" in obj.role_tags or "boundary_touching" in obj.role_tags:
                weight *= 0.45
            if weight <= 0.02:
                continue
            score = weight / (1.0 + 0.15 * dist)
            if best is None or score > best[0]:
                best = (float(score), obj, dist)
        if best is None:
            return self._frontier_action(state, start, move_by_delta)
        goal = best[1].center_cell
        best_step: tuple[int, ActionName] | None = None
        for delta, action in move_by_delta.items():
            nxt = (start[0] + delta[0], start[1] + delta[1])
            if nxt in self.blocked_cells:
                continue
            dist = self._learned_grid_distance(nxt, goal, state)
            if dist is None:
                continue
            if best_step is None or dist < best_step[0]:
                best_step = (dist, action)
        return None if best_step is None else best_step[1]

    def _frontier_action(
        self,
        state: StructuredState,
        start: tuple[int, int],
        move_by_delta: dict[tuple[int, int], ActionName],
    ) -> ActionName | None:
        rows, cols = state.grid.shape
        best: tuple[int, ActionName] | None = None
        for delta, action in move_by_delta.items():
            nxt = (start[0] + delta[0], start[1] + delta[1])
            if nxt in self.blocked_cells:
                continue
            if nxt[0] < 0 or nxt[1] < 0 or nxt[0] >= rows or nxt[1] >= cols:
                continue
            visits = 1 if nxt in self.visited_actor_positions else 0
            if best is None or visits < best[0]:
                best = (visits, action)
        return None if best is None else best[1]

    def _update_contact_statistics(self, record: TransitionRecord, engine: HypothesisEngine) -> None:
        actor = self._controlled_object(record.after, engine) or self._controlled_object(record.before, engine)
        if actor is None:
            return
        progress = combined_progress_signal(record.reward, record.delta.score_delta)
        colors: set[int] = set()
        for state in (record.before, record.after):
            current_actor = self._controlled_object(state, engine) or actor
            ar, ac = current_actor.center_cell
            for obj in state.objects:
                if obj.object_id == current_actor.object_id:
                    continue
                if obj.color == current_actor.color and obj.shape_hash == current_actor.shape_hash:
                    continue
                if abs(ar - obj.center_cell[0]) + abs(ac - obj.center_cell[1]) <= 1:
                    colors.add(obj.color)
        if record.delta.disappeared or colors:
            self.pending_interactions = max(self.pending_interactions, 2)
        if not colors:
            return
        if progress > 1e-8:
            for color in colors:
                self.productive_colors[color] += 1
        elif record.delta.has_visible_effect or is_move_action(record.action) or is_interact_action(record.action):
            for color in colors:
                self.nonproductive_colors[color] += 1

    def _spatial_goal_value(self, state: StructuredState, action: ActionName, engine: HypothesisEngine) -> float:
        """Give move actions a model-based incentive to test object contact hypotheses.

        The benchmark gives no instructions, so the agent must create its own
        experiments. This term identifies the currently controllable object,
        estimates its next position under the candidate action, and rewards moves
        that reduce distance to potentially useful or still-unexplained objects.
        It is intentionally weak relative to observed reward and information gain.
        """

        delta = action_delta(action)
        if delta is None or not state.objects:
            return 0.0
        actor = self._controlled_object(state, engine)
        if actor is None:
            return 0.0
        ar, ac = actor.center_cell
        rows, cols = state.grid.shape
        nr = min(max(ar + delta[0], 0), rows - 1)
        nc = min(max(ac + delta[1], 0), cols - 1)
        if (nr, nc) in self.blocked_cells:
            return -0.75
        color_priors = engine.color_progress_priors()
        mechanic_priors = engine.mechanic_color_priors()
        value = 0.0
        for obj in state.objects:
            if obj.object_id == actor.object_id:
                continue
            if obj.color == actor.color and obj.shape_hash == actor.shape_hash:
                continue
            before_dist = self._learned_grid_distance((ar, ac), obj.center_cell, state)
            after_dist = self._learned_grid_distance((nr, nc), obj.center_cell, state)
            if before_dist is None or after_dist is None or before_dist == 0:
                continue
            progress = (before_dist - after_dist) / max(before_dist, 1)
            if progress <= 0:
                continue
            weight = 0.10
            weight += color_priors.get(obj.color, 0.0)
            weight += 0.25 * mechanic_priors.get(obj.color, 0.0)
            if "point" in obj.role_tags or "small" in obj.role_tags:
                weight += 0.08
            if "large" in obj.role_tags or "boundary_touching" in obj.role_tags:
                weight *= 0.55
            value += weight * progress / (1.0 + after_dist)
        return float(min(value, 1.25))

    def _mechanic_goal_value(self, state: StructuredState, action: ActionName, engine: HypothesisEngine) -> float:
        mechanic_priors = engine.mechanic_color_priors()
        if not mechanic_priors:
            return 0.0

        target = parse_action_target(action)
        value = 0.0
        if target is not None:
            for obj in state.objects:
                prior = mechanic_priors.get(obj.color, 0.0)
                if prior <= 0.0:
                    continue
                if abs(obj.center_cell[0] - target[0]) + abs(obj.center_cell[1] - target[1]) <= 1:
                    value += 0.35 * prior

        delta = action_delta(action)
        actor = self._controlled_object(state, engine)
        if delta is None or actor is None:
            return float(min(value, 1.25))
        start = actor.center_cell
        rows, cols = state.grid.shape
        nxt = (min(max(start[0] + delta[0], 0), rows - 1), min(max(start[1] + delta[1], 0), cols - 1))
        for obj in state.objects:
            if obj.object_id == actor.object_id:
                continue
            prior = mechanic_priors.get(obj.color, 0.0)
            if prior <= 0.0:
                continue
            before_dist = self._learned_grid_distance(start, obj.center_cell, state)
            after_dist = self._learned_grid_distance(nxt, obj.center_cell, state)
            if before_dist is None or after_dist is None or before_dist == 0:
                continue
            progress = (before_dist - after_dist) / max(before_dist, 1)
            if progress > 0:
                value += 0.45 * prior * progress / (1.0 + after_dist)
        return float(min(value, 1.25))

    def _option_profile(
        self,
        memory: EpisodicMemory,
        state: StructuredState,
        action: ActionName,
        language_tokens: Iterable[str],
    ) -> dict[str, float]:
        profile_fn = getattr(memory, "action_option_profile", None)
        if callable(profile_fn):
            try:
                return dict(profile_fn(state, action, language_tokens))
            except Exception:
                pass
        return {
            "schema_bonus": 0.0,
            "relative_cost": 1.0,
            "efficiency": 0.0,
            "support": 0.0,
            "contradiction": 0.0,
            "continuation_depth": 0.0,
        }

    def _action_cost_estimate(self, action: ActionName, *, option_profile: dict[str, float]) -> float:
        fam = action_family(action)
        empirical = self.family_cost_sum[fam] / self.family_cost_count[fam] if self.family_cost_count[fam] else 1.0
        option_cost = float(option_profile.get("relative_cost", 1.0) or 1.0)
        blended = 0.75 * empirical + 0.25 * min(option_cost, 6.0) / 2.0
        return float(max(blended, 0.25))

    def _action_efficiency_prior(self, action: ActionName, *, option_profile: dict[str, float]) -> float:
        fam = action_family(action)
        empirical = self.family_effect_sum[fam] / self.family_effect_count[fam] if self.family_effect_count[fam] else 0.0
        option_efficiency = float(option_profile.get("efficiency", 0.0) or 0.0)
        return float(max(0.0, (0.70 * empirical) + (0.30 * option_efficiency)))

    def _budget_pressure(self, state: StructuredState) -> float:
        extras = state.frame.extras if isinstance(state.frame.extras, dict) else {}
        retry_index = 0.0
        if isinstance(extras, dict):
            try:
                retry_index = float(extras.get("session_retry_index", 0) or 0)
            except Exception:
                retry_index = 0.0
        pressure = 0.20
        pressure += 0.35 * min(float(state.step_index), 96.0) / 96.0
        pressure += 0.30 * min(retry_index, 4.0) / 4.0
        pressure += 0.25 * min(float(self.stall_count), 8.0) / 8.0
        pressure += 0.45 * self._numeric_budget_pressure(state)
        return float(min(pressure, 1.25))

    def _observed_effect_value(self, record: TransitionRecord) -> float:
        progress = max(0.0, combined_progress_signal(record.reward, record.delta.score_delta))
        action_regime_changed = tuple(record.before.available_actions) != tuple(record.after.available_actions)
        numeric_deltas = transition_numeric_deltas(record)
        structural = bool(
            progress > 0.0
            or record.delta.disappeared
            or action_regime_changed
            or any(delta > 0.0 for delta in numeric_deltas.values())
        )
        value = progress
        if structural:
            value += 0.20 * min(record.delta.changed_fraction, 1.0)
            if record.delta.disappeared:
                value += 0.20
            large_motions = sum(1 for motion in record.delta.moved_objects if motion.distance > 1.25)
            value += 0.18 * min(float(large_motions), 2.0)
            if action_regime_changed:
                value += 0.16
            if any(delta > 0.0 for delta in numeric_deltas.values()):
                value += 0.12
            if len(numeric_deltas) >= 2:
                value += 0.06
        else:
            value += 0.02 * min(record.delta.changed_fraction, 1.0)
        return float(min(value, 2.5))

    def _observed_action_cost(self, record: TransitionRecord, *, observed_effect: float) -> float:
        cost = 1.0
        progress = combined_progress_signal(record.reward, record.delta.score_delta)
        if observed_effect <= 0.05:
            cost += 0.35
        if not record.delta.has_visible_effect and progress <= 0.0:
            cost += 0.35
        if record.delta.terminated and progress <= 0.0:
            cost += 0.45
        if is_reset_action(record.action):
            cost += 0.25
        numeric_spend = sum(max(-delta, 0.0) for delta in transition_numeric_deltas(record).values())
        cost += 0.20 * min(numeric_spend, 2.0)
        return float(min(cost, 3.0))

    def _update_numeric_channel_stats(self, record: TransitionRecord) -> None:
        before = state_numeric_channels(record.before)
        after = state_numeric_channels(record.after)
        for key, value in {**before, **after}.items():
            current = float(value)
            self.channel_min[key] = current if key not in self.channel_min else min(self.channel_min[key], current)
            self.channel_max[key] = current if key not in self.channel_max else max(self.channel_max[key], current)
        for key, delta in transition_numeric_deltas(record).items():
            if delta >= 0.0:
                continue
            self.channel_spend_sum[key] += float(-delta)
            self.channel_spend_count[key] += 1

    def _numeric_budget_pressure(self, state: StructuredState) -> float:
        channels = state_numeric_channels(state)
        if not channels:
            return 0.0
        pressure = 0.0
        for key, current in channels.items():
            span = self.channel_max.get(key, current) - self.channel_min.get(key, current)
            if span <= 1e-6:
                continue
            spend_count = self.channel_spend_count.get(key, 0)
            if spend_count <= 0:
                continue
            scarcity = 1.0 - ((float(current) - self.channel_min.get(key, current)) / span)
            spend_intensity = min(self.channel_spend_sum.get(key, 0.0), 8.0) / 8.0
            pressure += max(0.0, scarcity) * (0.35 + (0.65 * spend_intensity))
        return float(min(pressure, 1.0))


    def _learned_grid_distance(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        state: StructuredState,
    ) -> int | None:
        rows, cols = state.grid.shape
        if start == goal:
            return 0
        q: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
        seen = {start}
        while q:
            (r, c), dist = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if nr < 0 or nc < 0 or nr >= rows or nc >= cols or nxt in seen:
                    continue
                if nxt in self.blocked_cells and nxt != goal:
                    continue
                if nxt == goal:
                    return dist + 1
                seen.add(nxt)
                q.append((nxt, dist + 1))
        return None

    def _controlled_object(self, state: StructuredState, engine: HypothesisEngine) -> ObjectToken | None:
        color_scores = engine.controlled_object_colors()
        if color_scores:
            candidates = [obj for obj in state.objects if obj.color in color_scores]
            if candidates:
                return sorted(candidates, key=lambda o: (-color_scores.get(o.color, 0.0), o.area, o.object_id))[0]
        point_like = [obj for obj in state.objects if "point" in obj.role_tags or obj.area <= 4]
        if point_like:
            return sorted(point_like, key=lambda o: (o.area, "boundary_touching" in o.role_tags, o.object_id))[0]
        return min(state.objects, key=lambda o: (o.area, o.object_id))

    def _novelty(self, state: StructuredState, action: ActionName) -> float:
        local = self.state_action_visits[(state.abstract_fingerprint, action)]
        global_visits = self.action_visits[action]
        target_bonus = 0.10 if parse_action_target(action) is not None else 0.0
        move_or_interact_bonus = 0.08 if is_move_action(action) or is_interact_action(action) else 0.0
        return float(1.0 / (1.0 + local) + 0.25 / (1.0 + global_visits) + target_bonus + move_or_interact_bonus)

    def _repeat_penalty(self, state: StructuredState, action: ActionName) -> float:
        blocked = self.ineffective_actions[(state.exact_fingerprint, action)]
        penalty = 0.40 * min(blocked, 4)
        if self.last_action == action:
            local = self.state_action_visits[(state.abstract_fingerprint, action)]
            penalty += self.config.repeat_penalty * min(local, 4) / 4.0
        if self.stall_count:
            penalty += 0.15 * self.stall_count
        return float(penalty)
