"""Information-gain planner for the scientist agent."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

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
    is_interact_action,
    is_move_action,
    make_targeted_action,
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
    spatial_goal_weight: float = 1.15
    navigation_weight: float = 2.10
    repeat_penalty: float = 0.55
    max_candidates: int = 96
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
        candidates = self.candidate_actions(state, engine=engine)
        if not candidates:
            raise RuntimeError("No legal actions are available for ScientistPlanner")

        lang_tokens = language.memory_tokens(state, engine)
        navigation_action = self._navigation_next_action(state, engine, candidates)
        scored: list[tuple[float, ActionName, dict[str, float], tuple[str, ...]]] = []
        for action in candidates:
            hyp_score = engine.score_action(state, action)
            world = world_model.predict(state, action)
            novelty = self._novelty(state, action)
            memory_bonus = memory.action_memory_bonus(state, action, lang_tokens)
            repeat = self._repeat_penalty(state, action)
            spatial_goal = self._spatial_goal_value(state, action, engine)
            navigation_goal = 1.0 if navigation_action == action else 0.0
            components = {
                "expected_reward": hyp_score.expected_reward + world.reward_mean,
                "information_gain": hyp_score.information_gain,
                "novelty": novelty,
                "expected_change": hyp_score.expected_change + world.change_mean,
                "world_uncertainty": world.total_uncertainty,
                "memory_bonus": memory_bonus,
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
                + self.config.spatial_goal_weight * components["spatial_goal"]
                + self.config.navigation_weight * components["navigation_goal"]
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

    def candidate_actions(self, state: StructuredState, *, engine: HypothesisEngine) -> tuple[ActionName, ...]:
        legal = tuple(state.available_actions)
        if not legal:
            legal = ("up", "down", "left", "right", "interact", "click")
        candidates: list[ActionName] = list(legal)
        candidates.extend(engine.diagnostic_actions(state, legal, limit=48))

        # Expand click-like actions to object centers and a small frontier set.  The
        # official ARC docs expose ACTION6 as a coordinate-bearing action in the
        # direct API, so this representation is adapter-friendly.
        click_bases = [a for a in legal if action_family(a) in {"click", "action6", "interact_at", "select_at"}]
        if click_bases:
            base = click_bases[0]
            for obj in sorted(state.objects, key=lambda o: ("background_candidate" in o.role_tags, -o.area))[:24]:
                r, c = obj.center_cell
                tr, tc = grid_cell_to_action_coordinates(r, c, state.frame.extras)
                candidates.append(make_targeted_action(base, tr, tc))
            rows, cols = state.grid.shape
            for r, c in ((0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1), (rows // 2, cols // 2)):
                tr, tc = grid_cell_to_action_coordinates(r, c, state.frame.extras)
                candidates.append(make_targeted_action(base, tr, tc))

        # If all recent actions are stalling, force broad coverage before repeating.
        if self.stall_count >= 3:
            candidates = sorted(candidates, key=lambda a: (self.action_visits[a], self.state_action_visits[(state.abstract_fingerprint, a)]))

        deduped = tuple(dict.fromkeys(candidates))
        return deduped[: self.config.max_candidates]



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
            if "point" in obj.role_tags or "small" in obj.role_tags:
                weight += 0.08
            if "large" in obj.role_tags or "boundary_touching" in obj.role_tags:
                weight *= 0.55
            value += weight * progress / (1.0 + after_dist)
        return float(min(value, 1.25))


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
