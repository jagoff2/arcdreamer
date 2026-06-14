"""Runtime plan-level cognition patch for real ARC attempts.

This layer is deliberately small and invasive: it changes the *control surface*, not
just the diagnostics.  Attempt memory gets a persistent active goal object, a
committed execution trace, candidate-goal distance scoring, and contradiction
revision.  The controller already asks attempt memory for scores and experiments;
this patch makes those calls prefer a held goal and an executable next step until
observed evidence contradicts it.
"""
from __future__ import annotations

import math
from typing import Any

_APPLIED = False


def apply_plan_cognition_patch() -> None:
    """Apply idempotent plan-cognition hooks to ``jepa_attempt_memory``."""

    global _APPLIED
    if _APPLIED:
        return

    from . import jepa_attempt_memory as m

    if bool(getattr(m, "_arc_plan_cognition_patch_applied", False)):
        _APPLIED = True
        return

    orig_init = m.JEPAAttemptMemory.__init__
    orig_reset = m.JEPAAttemptMemory.reset
    orig_start_attempt = m.JEPAAttemptMemory.start_attempt
    orig_plan_scores = m.JEPAAttemptMemory.plan_scores_for_observation
    orig_select_experiment = m.JEPAAttemptMemory.select_experiment
    orig_observe_live_transition = m.JEPAAttemptMemory.observe_live_transition
    orig_has_public_goal_evidence = m.JEPAAttemptMemory._has_public_goal_evidence
    orig_summary = m.JEPAAttemptMemory.summary

    def _init_plan_state(self: Any) -> None:
        self.active_goal_object: dict[str, Any] | None = None
        self.active_plan_trace: dict[str, Any] = {}
        self.plan_bank: dict[str, dict[str, Any]] = {}
        self.plan_execution_log: list[dict[str, Any]] = []
        self.plan_contradictions: dict[str, int] = {}
        self.plan_successes: dict[str, int] = {}
        self.plan_revision_count = 0
        self.pending_plan_step: dict[str, Any] | None = None
        self.plan_cognition_ticks = 0

    def _frame_components(observation: Any) -> tuple[Any | None, list[Any]]:
        frame = m._public_frame(observation)
        if frame is None:
            return None, []
        try:
            arr = m.np.asarray(frame, dtype=m.np.int64)
        except Exception:
            return None, []
        if arr.ndim != 2 or arr.size == 0:
            return None, []
        return arr, m._frame_components(arr)

    def _component_by_token(components: list[Any], token: str) -> Any | None:
        token = str(token)
        if not token:
            return None
        for component in components:
            if m._goal_component_token(component) == token:
                return component
        return None

    def _target_centroid(predicate: dict[str, Any]) -> tuple[float, float] | None:
        value = predicate.get("target_centroid") or predicate.get("component_centroid") or predicate.get("anomaly_centroid")
        if isinstance(value, (list, tuple)) and len(value) >= 2:
            try:
                return float(value[0]), float(value[1])
            except (TypeError, ValueError):
                return None
        axis = str(predicate.get("axis", ""))
        if axis in {"row", "col"}:
            try:
                scalar = float(predicate.get("target_scalar", 0.0))
            except (TypeError, ValueError):
                return None
            return (scalar, 0.0) if axis == "row" else (0.0, scalar)
        return None

    def _source_component(predicate: dict[str, Any], components: list[Any], memory: Any) -> Any | None:
        for key in ("source_token", "component_token"):
            component = _component_by_token(components, str(predicate.get(key, "")))
            if component is not None:
                return component
        preferred_values = set()
        try:
            preferred_values = set(memory._positive_component_values())
        except Exception:
            preferred_values = set()
        if preferred_values:
            preferred = [component for component in components if int(component.value) in preferred_values]
            if preferred:
                return sorted(preferred, key=lambda item: (-item.area, item.value, item.bbox))[0]
        target_token = str(predicate.get("target_token", ""))
        non_target = [component for component in components if m._goal_component_token(component) != target_token]
        pool = non_target or components
        if not pool:
            return None
        return sorted(pool, key=lambda item: (-item.area, item.value, item.bbox))[0]

    def _goal_distance(predicate: dict[str, Any], frame: Any | None, components: list[Any], memory: Any) -> float:
        if frame is None or not components:
            return float("inf")
        ptype = str(predicate.get("type", ""))
        if ptype == "clear_value":
            target_value = int(predicate.get("value", 0) or 0)
            return float(sum(component.area for component in components if int(component.value) == target_value))
        if ptype == "activate_all_switches":
            target_value = int(predicate.get("switch_value", 0) or 0)
            return float(sum(1 for component in components if int(component.value) == target_value))
        if ptype in {"transform_to_value", "transform_to_exemplar"}:
            target_value = int(predicate.get("target_value", predicate.get("exemplar_value", 0)) or 0)
            return float(sum(component.area for component in components if int(component.value) != target_value))
        source = _source_component(predicate, components, memory)
        target = _target_centroid(predicate)
        if source is None or target is None:
            return float("inf")
        if ptype == "align_row_with_target":
            return abs(float(source.centroid[0]) - float(target[0]))
        if ptype == "align_col_with_target":
            return abs(float(source.centroid[1]) - float(target[1]))
        if ptype == "align_axis_bucket":
            axis = str(predicate.get("axis", ""))
            scalar = float(predicate.get("target_scalar", 0.0))
            current = float(source.centroid[0] if axis == "row" else source.centroid[1])
            return abs(current - scalar)
        return float(math.dist(source.centroid, target))

    def _predicate_rank(memory: Any, predicate: dict[str, Any], frame: Any | None, components: list[Any]) -> float:
        predicate_id = str(predicate.get("id", ""))
        bank_item = memory.progress_hypothesis_bank.get(predicate_id, {}) if hasattr(memory, "progress_hypothesis_bank") else {}
        support = float(bank_item.get("support", 0.0))
        posterior = float(bank_item.get("posterior_score", 0.0))
        observations = float(bank_item.get("observations", 0.0))
        contradictions = float(memory.plan_contradictions.get(predicate_id, 0))
        ptype = str(predicate.get("type", ""))
        terminal = 0.34 if bool(predicate.get("terminal_candidate", False)) else 0.08
        relation_bonus = 0.0
        if ptype in {"move_to_target_component", "touch_target_component", "reach_adjacent_component", "align_row_with_target", "align_col_with_target"}:
            relation_bonus = 0.18
        if ptype in {"clear_value", "transform_to_value", "activate_all_switches", "transform_to_exemplar"}:
            relation_bonus = 0.12
        distance = _goal_distance(predicate, frame, components, memory)
        distance_score = 0.18 / (1.0 + min(distance if math.isfinite(distance) else 99.0, 99.0))
        return float(terminal + relation_bonus + 0.28 * support + 0.20 * posterior + 0.015 * min(observations, 10.0) + distance_score - 0.55 * contradictions)

    def _candidate_goals(memory: Any, observation: Any, limit: int = 24) -> list[dict[str, Any]]:
        frame, components = _frame_components(observation)
        if frame is None:
            return []
        predicates = memory._ensure_goal_predicates(frame, components)
        ranked: list[tuple[float, dict[str, Any]]] = []
        for predicate in predicates:
            predicate_id = str(predicate.get("id", ""))
            if not predicate_id:
                continue
            score = _predicate_rank(memory, predicate, frame, components)
            if score <= -0.20:
                continue
            goal = dict(predicate)
            goal["schema"] = "persistent_candidate_goal_v1"
            goal["rank_score"] = round(float(score), 6)
            goal["distance"] = round(float(_goal_distance(predicate, frame, components, memory)), 6) if components else float("inf")
            goal["contradictions"] = int(memory.plan_contradictions.get(predicate_id, 0))
            goal["successes"] = int(memory.plan_successes.get(predicate_id, 0))
            ranked.append((score, goal))
        ranked.sort(key=lambda item: (item[0], str(item[1].get("id", ""))), reverse=True)
        return [item for _score, item in ranked[: max(int(limit), 1)]]

    def _relation_bonus_for_action(memory: Any, action: str, predicate: dict[str, Any], frame: Any | None, components: list[Any]) -> float:
        if frame is None or m._action_family(action) != "contact":
            return 0.0
        relation = m._component_target_relation(frame, m._action_target_cell(action, frame), components)
        value = int(relation.get("value", 0) or 0)
        ptype = str(predicate.get("type", ""))
        if ptype == "clear_value" and value == int(predicate.get("value", 0) or 0):
            return 0.34
        if ptype == "activate_all_switches" and value == int(predicate.get("switch_value", 0) or 0):
            return 0.30
        if ptype in {"transform_to_value", "transform_to_exemplar"} and value and value != int(predicate.get("target_value", predicate.get("exemplar_value", 0)) or 0):
            return 0.22
        if ptype in {"touch_target_component", "reach_component", "reach_adjacent_component"} and value == int(predicate.get("target_value", 0) or 0):
            return 0.26
        return 0.0

    def _best_plan_step(memory: Any, observation: Any, legal_actions: list[str] | tuple[str, ...], scores: dict[str, float] | None = None) -> dict[str, Any] | None:
        legal = [str(action) for action in legal_actions]
        if not legal:
            return None
        frame, components = _frame_components(observation)
        if frame is None:
            return None
        goals = _candidate_goals(memory, observation, limit=32)
        if not goals:
            return None
        active = memory.active_goal_object if isinstance(getattr(memory, "active_goal_object", None), dict) else None
        if active is not None:
            active_id = str(active.get("id", ""))
            matched = next((goal for goal in goals if str(goal.get("id", "")) == active_id), None)
            if matched is not None and int(memory.plan_contradictions.get(active_id, 0)) < 3:
                goals.insert(0, matched)
        base_scores = scores or {}
        best: tuple[float, dict[str, Any], str, dict[str, Any]] | None = None
        for goal in goals[:20]:
            predicate_id = str(goal.get("id", ""))
            before_distance = _goal_distance(goal, frame, components, memory)
            for index, action in enumerate(legal):
                try:
                    evidence = m._counterfactual_goal_progress_score(
                        action,
                        frame,
                        components=components,
                        predicates=[goal],
                        preferred_component_values=memory._positive_component_values(),
                        return_evidence=True,
                    )
                except Exception:
                    evidence = {"score": 0.0, "attached": [], "reachability_delta": 0.0}
                reach_delta = max(float(evidence.get("reachability_delta", 0.0)), 0.0)
                attached = [str(item) for item in evidence.get("attached", [])]
                relation_bonus = _relation_bonus_for_action(memory, action, goal, frame, components)
                prior = max(float(base_scores.get(action, 0.0)), -0.25)
                action_count = float(memory.action_counts.get(action, 0))
                family = m._action_family(action)
                nuisance = float(memory.action_nuisance.get(action, 0) + memory.family_nuisance.get(family, 0))
                failure = float(memory.action_failures.get(action, 0))
                repeat_cost = 0.018 * min(action_count, 16.0) + 0.030 * min(failure, 8.0) + 0.020 * min(nuisance, 12.0)
                persistence_bonus = 0.10 if active is not None and str(active.get("id", "")) == predicate_id else 0.0
                terminal_bonus = 0.05 if bool(goal.get("terminal_candidate", False)) else 0.0
                value = (
                    1.05 * reach_delta
                    + 0.45 * float(evidence.get("score", 0.0))
                    + 0.58 * relation_bonus
                    + 0.12 * max(prior, 0.0)
                    + persistence_bonus
                    + terminal_bonus
                    - repeat_cost
                    - 0.02 * float(index)
                )
                if attached and predicate_id in attached:
                    value += 0.24
                if value <= 0.0:
                    continue
                step = {
                    "schema": "plan_level_step_v1",
                    "action": action,
                    "goal_id": predicate_id,
                    "goal_type": str(goal.get("type", "")),
                    "value": round(float(value), 6),
                    "before_distance": round(float(before_distance), 6) if math.isfinite(before_distance) else float("inf"),
                    "expected_reachability_delta": round(float(reach_delta), 6),
                    "expected_attached_goals": attached[:8],
                    "relation_bonus": round(float(relation_bonus), 6),
                    "prior_score": round(float(prior), 6),
                    "repeat_cost": round(float(repeat_cost), 6),
                }
                if best is None or value > best[0]:
                    best = (value, goal, action, step)
        if best is None:
            return None
        _value, goal, action, step = best
        return {"goal": goal, "action": action, "step": step}

    def _commit_plan(memory: Any, decision: dict[str, Any]) -> dict[str, Any]:
        goal = dict(decision["goal"])
        step = dict(decision["step"])
        goal_id = str(goal.get("id", ""))
        current = memory.active_plan_trace if isinstance(getattr(memory, "active_plan_trace", None), dict) else {}
        same_goal = bool(memory.active_goal_object and str(memory.active_goal_object.get("id", "")) == goal_id)
        if not same_goal:
            memory.plan_revision_count = int(getattr(memory, "plan_revision_count", 0)) + 1
            memory.active_plan_trace = {
                "schema": "persistent_plan_trace_v1",
                "goal_id": goal_id,
                "goal_type": str(goal.get("type", "")),
                "revision": int(memory.plan_revision_count),
                "status": "active",
                "created_tick": int(getattr(memory, "plan_cognition_ticks", 0)),
                "steps": [],
                "contradictions": 0,
                "successes": 0,
            }
        elif current:
            current["status"] = "active"
            memory.active_plan_trace = current
        memory.active_goal_object = goal
        trace = memory.active_plan_trace
        planned_steps = list(trace.get("steps", []))[-31:]
        planned_steps.append(step)
        trace["steps"] = planned_steps
        trace["last_action"] = str(step.get("action", ""))
        trace["last_value"] = float(step.get("value", 0.0))
        trace["last_expected_reachability_delta"] = float(step.get("expected_reachability_delta", 0.0))
        memory.pending_plan_step = step
        memory.plan_bank[goal_id] = {"goal": goal, "trace": dict(trace)}
        return trace

    def _plan_diagnostics(memory: Any) -> dict[str, Any]:
        return {
            "schema": "plan_level_cognition_v1",
            "active_goal": dict(memory.active_goal_object or {}),
            "active_trace": dict(memory.active_plan_trace or {}),
            "pending_step": dict(memory.pending_plan_step or {}),
            "bank_size": int(len(memory.plan_bank)),
            "revision_count": int(getattr(memory, "plan_revision_count", 0)),
            "contradictions": dict(sorted(memory.plan_contradictions.items())[-12:]),
            "successes": dict(sorted(memory.plan_successes.items())[-12:]),
            "recent_execution": list(memory.plan_execution_log[-8:]),
        }

    def patched_init(self: Any, *args: Any, **kwargs: Any) -> None:
        orig_init(self, *args, **kwargs)
        _init_plan_state(self)

    def patched_reset(self: Any) -> None:
        orig_reset(self)
        _init_plan_state(self)

    def patched_start_attempt(self: Any) -> None:
        orig_start_attempt(self)
        if getattr(self, "active_plan_trace", None):
            if self.active_plan_trace.get("status") in {"failed", "contradicted"}:
                self.active_goal_object = None
                self.active_plan_trace = {}
                self.pending_plan_step = None
            else:
                self.active_plan_trace["status"] = "carried_into_attempt"
                self.pending_plan_step = None

    def patched_has_public_goal_evidence(self: Any) -> bool:
        if bool(orig_has_public_goal_evidence(self)):
            return True
        return bool(getattr(self, "active_goal_object", None) and getattr(self, "active_plan_trace", None))

    def patched_plan_scores_for_observation(self: Any, observation: Any) -> dict[str, float]:
        scores = dict(orig_plan_scores(self, observation))
        self.plan_cognition_ticks = int(getattr(self, "plan_cognition_ticks", 0)) + 1
        legal = [str(action) for action in getattr(observation, "available_actions", ())]
        decision = _best_plan_step(self, observation, legal, scores)
        if decision is not None:
            trace = _commit_plan(self, decision)
            action = str(decision["action"])
            plan_value = float(decision["step"].get("value", 0.0))
            scores[action] = max(float(scores.get(action, 0.0)), 0.0) + 1.05 + min(plan_value, 1.25)
            for other in legal:
                if other != action and m._action_family(other) == m._action_family(action):
                    scores[other] = float(scores.get(other, 0.0)) - 0.08
            trace["last_score_bias"] = round(float(scores[action]), 6)
        return {action: float(max(min(score, 2.25), -2.25)) for action, score in scores.items()}

    def patched_select_experiment(self: Any, observation: Any, legal_actions: list[str] | tuple[str, ...], *, scores: dict[str, float] | None = None) -> Any | None:
        legal = [str(action) for action in legal_actions]
        decision = _best_plan_step(self, observation, legal, scores or {})
        if decision is not None:
            trace = _commit_plan(self, decision)
            step = decision["step"]
            action = str(step["action"])
            return m.Experiment(
                hypothesis_ids=[f"goal:{step['goal_id']}", "h_plan_distance_delta", "h_execute_until_contradiction"],
                action=action,
                predicted_outcomes={
                    "h_plan_distance_delta": "candidate_goal_distance_decreases",
                    "h_execute_until_contradiction": "continue_trace_or_revise",
                    "h_no_progress": "retire_goal_or_replan",
                },
                useful_if=[
                    "score_delta > 0",
                    "level_changed",
                    "candidate_goal_reachability_improved > threshold",
                    "observed_goal_distance_decreased",
                ],
                max_repeats=1,
                retire_if="contradiction_or_no_distance_delta",
                action_class=f"plan:{m._action_family(action)}",
                coordinate_equiv_class=m._experiment_coordinate_equiv_class(action, _frame_components(observation)[0]),
                abstract_state_class=f"plan_goal:{step['goal_id']}",
                experiment_value={
                    "plan_level_value": float(step.get("value", 0.0)),
                    "expected_goal_reachability_delta": float(step.get("expected_reachability_delta", 0.0)),
                    "trace_revision": float(trace.get("revision", 0)),
                    "total": float(step.get("value", 0.0)),
                },
                attached_goal_predicates=[str(step["goal_id"])],
            )
        return orig_select_experiment(self, observation, legal_actions, scores=scores)

    def patched_observe_live_transition(self: Any, before_observation: Any, action: str, result: Any) -> Any:
        before_frame, before_components = _frame_components(before_observation)
        before_goal = dict(getattr(self, "active_goal_object", {}) or {})
        before_distance = _goal_distance(before_goal, before_frame, before_components, self) if before_goal else float("inf")
        label = orig_observe_live_transition(self, before_observation, action, result)
        after_frame, after_components = _frame_components(getattr(result, "observation", None))
        pending = dict(getattr(self, "pending_plan_step", None) or {})
        if not before_goal or not pending:
            return label
        goal_id = str(before_goal.get("id", ""))
        after_distance = _goal_distance(before_goal, after_frame, after_components, self) if after_frame is not None else float("inf")
        improved = bool(math.isfinite(before_distance) and math.isfinite(after_distance) and after_distance < before_distance - 1.0e-6)
        progress = bool(label and (getattr(label, "progress_effect", False) or getattr(label, "terminal_win", False) or getattr(label, "useful_effect", False)))
        matched_action = str(action) == str(pending.get("action", ""))
        event = {
            "schema": "plan_step_observation_v1",
            "goal_id": goal_id,
            "goal_type": str(before_goal.get("type", "")),
            "action": str(action),
            "expected_action": str(pending.get("action", "")),
            "matched_action": matched_action,
            "before_distance": round(float(before_distance), 6) if math.isfinite(before_distance) else float("inf"),
            "after_distance": round(float(after_distance), 6) if math.isfinite(after_distance) else float("inf"),
            "distance_improved": improved,
            "posthoc_progress": progress,
        }
        self.plan_execution_log.append(event)
        self.plan_execution_log = self.plan_execution_log[-128:]
        trace = self.active_plan_trace if isinstance(getattr(self, "active_plan_trace", None), dict) else {}
        if progress or improved:
            self.plan_successes[goal_id] = int(self.plan_successes.get(goal_id, 0)) + 1
            trace["successes"] = int(trace.get("successes", 0)) + 1
            trace["status"] = "active_progressing"
            trace["last_observed_distance"] = event["after_distance"]
        elif matched_action:
            self.plan_contradictions[goal_id] = int(self.plan_contradictions.get(goal_id, 0)) + 1
            trace["contradictions"] = int(trace.get("contradictions", 0)) + 1
            trace["status"] = "contradicted" if trace["contradictions"] >= 1 else "active"
            if int(self.plan_contradictions[goal_id]) >= 2:
                self.active_goal_object = None
                self.active_plan_trace = {}
        self.pending_plan_step = None
        return label

    def patched_summary(self: Any) -> dict[str, Any]:
        out = dict(orig_summary(self))
        out["plan_level_cognition"] = _plan_diagnostics(self)
        return out

    m.JEPAAttemptMemory.__init__ = patched_init
    m.JEPAAttemptMemory.reset = patched_reset
    m.JEPAAttemptMemory.start_attempt = patched_start_attempt
    m.JEPAAttemptMemory._has_public_goal_evidence = patched_has_public_goal_evidence
    m.JEPAAttemptMemory.plan_scores_for_observation = patched_plan_scores_for_observation
    m.JEPAAttemptMemory.select_experiment = patched_select_experiment
    m.JEPAAttemptMemory.observe_live_transition = patched_observe_live_transition
    m.JEPAAttemptMemory.summary = patched_summary
    setattr(m, "_arc_plan_cognition_patch_applied", True)
    _APPLIED = True
