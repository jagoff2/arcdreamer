from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from .attempt_buffer import AttemptRecord, observation_frame, stable_hash, tensors_from_attempts
from .video_jepa import VideoJEPA


POSITIVE_EVENTS = {"positive_reward", "resource_collected", "level_completed", "goal_reached", "goal_clicked", "useful_click"}

PREDICTIVE_COMPONENT_MECHANISMS = (
    "component_movement",
    "component_appearance",
    "component_disappearance",
    "component_color_transform",
    "component_split_merge",
    "component_visual_transform",
    "blocked_or_no_effect",
    "stable",
)
PRODUCTIVE_COMPONENT_MECHANISMS = {
    "component_movement",
    "component_appearance",
    "component_disappearance",
    "component_color_transform",
    "component_split_merge",
    "component_visual_transform",
}
NON_PRODUCTIVE_COMPONENT_MECHANISMS = {"blocked_or_no_effect", "stable"}


@dataclass
class AttemptMemoryEntry:
    attempt_index: int
    token_mean: list[float]
    failed_actions: dict[str, int]
    effect_actions: dict[str, int]
    repeated_actions: dict[str, float]
    event_candidates: dict[str, float]
    causal_hypotheses: dict[str, float]
    next_attempt_plan: dict[str, float]
    transition_graph_summary: dict[str, Any] = field(default_factory=dict)
    object_causal_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    component_causal_hypotheses: list[dict[str, Any]] = field(default_factory=list)
    sequence_plan_summary: dict[str, Any] = field(default_factory=dict)
    jepa_action_evidence: dict[str, dict[str, float]] = field(default_factory=dict)
    action_distribution_delta: dict[str, float] = field(default_factory=dict)
    causal_substrate_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JEPAAttemptMemory:
    def __init__(self, *, use_jepa_tokens: bool, device: torch.device | str = "cpu") -> None:
        self.use_jepa_tokens = bool(use_jepa_tokens)
        self.device = torch.device(device)
        self.entries: list[AttemptMemoryEntry] = []
        self.transition_counts: dict[tuple[str, str], int] = {}
        self.transition_values: dict[tuple[str, str], float] = {}
        self.transition_effects: dict[tuple[str, str], int] = {}
        self.transition_failures: dict[tuple[str, str], int] = {}
        self.transition_events: dict[tuple[str, str], int] = {}
        self.state_seen_actions: dict[str, set[str]] = {}
        self.family_counts: dict[str, int] = {}
        self.family_values: dict[str, float] = {}
        self.region_counts: dict[tuple[int, int], int] = {}
        self.region_values: dict[tuple[int, int], float] = {}
        self.region_failures: dict[tuple[int, int], int] = {}
        self.family_region_counts: dict[tuple[str, int, int], int] = {}
        self.family_region_values: dict[tuple[str, int, int], float] = {}
        self.color_counts: dict[int, int] = {}
        self.color_values: dict[int, float] = {}
        self.component_relation_counts: dict[str, int] = {}
        self.component_relation_values: dict[str, float] = {}
        self.component_relation_failures: dict[str, int] = {}
        self.family_component_counts: dict[tuple[str, str], int] = {}
        self.family_component_values: dict[tuple[str, str], float] = {}
        self.component_value_counts: dict[int, int] = {}
        self.component_value_scores: dict[int, float] = {}
        self.component_goal_relations: dict[str, float] = {}
        self.component_prediction_counts: dict[str, int] = {}
        self.component_prediction_values: dict[str, float] = {}
        self.component_prediction_goal_values: dict[str, float] = {}
        self.component_prediction_contradictions: dict[str, int] = {}
        self.family_component_prediction_counts: dict[tuple[str, str], int] = {}
        self.family_component_prediction_values: dict[tuple[str, str], float] = {}
        self.component_chain_counts: dict[tuple[str, str, str], int] = {}
        self.component_chain_values: dict[tuple[str, str, str], float] = {}
        self.component_chain_goal_values: dict[tuple[str, str, str], float] = {}
        self.component_chain_failures: dict[tuple[str, str, str], int] = {}
        self.component_chain_contradictions: dict[tuple[str, str, str], int] = {}
        self.component_chain_expectations: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.component_chain_edges_by_state: dict[str, set[tuple[str, str]]] = {}
        self.component_goal_state_values: dict[str, float] = {}
        self.component_relation_chain_counts: dict[tuple[str, str, str], int] = {}
        self.component_relation_chain_values: dict[tuple[str, str, str], float] = {}
        self.component_relation_chain_goal_values: dict[tuple[str, str, str], float] = {}
        self.component_relation_chain_failures: dict[tuple[str, str, str], int] = {}
        self.component_relation_chain_contradictions: dict[tuple[str, str, str], int] = {}
        self.component_relation_chain_expectations: dict[tuple[str, str, str], dict[str, Any]] = {}
        self.component_relation_chain_edges_by_state: dict[str, set[tuple[str, str]]] = {}
        self.component_relation_goal_state_values: dict[str, float] = {}
        self.sequence_contradictions = 0
        self.sequence_candidates: list[dict[str, Any]] = []
        self.active_sequence: list[str] = []
        self.active_sequence_expectations: list[dict[str, Any]] = []
        self.sequence_cursor = 0
        self.active_sequence_source = ""

    def reset(self) -> None:
        self.entries.clear()
        self.transition_counts.clear()
        self.transition_values.clear()
        self.transition_effects.clear()
        self.transition_failures.clear()
        self.transition_events.clear()
        self.state_seen_actions.clear()
        self.family_counts.clear()
        self.family_values.clear()
        self.region_counts.clear()
        self.region_values.clear()
        self.region_failures.clear()
        self.family_region_counts.clear()
        self.family_region_values.clear()
        self.color_counts.clear()
        self.color_values.clear()
        self.component_relation_counts.clear()
        self.component_relation_values.clear()
        self.component_relation_failures.clear()
        self.family_component_counts.clear()
        self.family_component_values.clear()
        self.component_value_counts.clear()
        self.component_value_scores.clear()
        self.component_goal_relations.clear()
        self.component_prediction_counts.clear()
        self.component_prediction_values.clear()
        self.component_prediction_goal_values.clear()
        self.component_prediction_contradictions.clear()
        self.family_component_prediction_counts.clear()
        self.family_component_prediction_values.clear()
        self.component_chain_counts.clear()
        self.component_chain_values.clear()
        self.component_chain_goal_values.clear()
        self.component_chain_failures.clear()
        self.component_chain_contradictions.clear()
        self.component_chain_expectations.clear()
        self.component_chain_edges_by_state.clear()
        self.component_goal_state_values.clear()
        self.component_relation_chain_counts.clear()
        self.component_relation_chain_values.clear()
        self.component_relation_chain_goal_values.clear()
        self.component_relation_chain_failures.clear()
        self.component_relation_chain_contradictions.clear()
        self.component_relation_chain_expectations.clear()
        self.component_relation_chain_edges_by_state.clear()
        self.component_relation_goal_state_values.clear()
        self.sequence_contradictions = 0
        self.sequence_candidates.clear()
        self.active_sequence.clear()
        self.active_sequence_expectations.clear()
        self.sequence_cursor = 0
        self.active_sequence_source = ""

    def ingest_attempt(self, record: AttemptRecord, model: VideoJEPA | None = None) -> AttemptMemoryEntry:
        failed: dict[str, int] = {}
        effects: dict[str, int] = {}
        events: dict[str, float] = {}
        action_counts: dict[str, int] = {}
        transition_outcomes: list[dict[str, Any]] = []
        no_effect = 0
        visible_effect = 0
        positive = 0
        for step in record.steps:
            action = str(step.action)
            action_counts[action] = action_counts.get(action, 0) + 1
            event_hit = bool(POSITIVE_EVENTS.intersection(step.event_delta)) or float(step.score_delta) > 0.0
            changed = step.obs_hash != step.next_obs_hash
            no_effect_event = _is_no_effect_event(step.event_delta)
            outcome_value = _transition_credit(
                event_hit=event_hit,
                changed=changed,
                no_effect_event=no_effect_event,
                invalid_action=bool(step.invalid_action),
                score_delta=float(step.score_delta),
                terminal=bool(step.terminal),
            )
            transition_outcomes.append(
                {
                    "edge": (str(step.obs_hash), action),
                    "action": action,
                    "value": outcome_value,
                    "event_hit": event_hit,
                    "changed": changed,
                    "no_effect": (not changed) or no_effect_event or bool(step.invalid_action),
                }
            )
            if event_hit:
                positive += 1
                events[action] = events.get(action, 0.0) + max(1.0, float(step.score_delta))
            elif not changed or no_effect_event:
                no_effect += 1
                failed[action] = failed.get(action, 0) + 1
            elif changed:
                visible_effect += 1
                effects[action] = effects.get(action, 0) + 1
        object_hypotheses = _object_hypotheses_from_record(record, transition_outcomes)
        component_hypotheses = _component_hypotheses_from_record(record, transition_outcomes)
        token_mean: list[float] = []
        jepa_action_evidence: dict[str, dict[str, float]] = {}
        if model is not None and self.use_jepa_tokens and record.steps:
            batch = tensors_from_attempts([record], device=self.device)
            model = model.to(self.device)
            model.eval()
            with torch.no_grad():
                output = model(batch["frames"], batch["action_ids"], batch["legal_counts"], batch.get("valid"))
                token_mean = [round(float(item), 6) for item in model.attempt_tokens(batch).detach().cpu().reshape(-1).tolist()[:24]]
                jepa_action_evidence = _jepa_action_evidence(record, output)
        total = max(len(record.steps), 1)
        repeated = {
            action: float(count / total)
            for action, count in sorted(action_counts.items())
            if count >= max(4, int(0.35 * total)) and count / total > 0.35
        }
        jepa_active = bool(jepa_action_evidence)
        effect_values = [item.get("effect", 0.0) for item in jepa_action_evidence.values()]
        surprise_values = [item.get("surprise", 0.0) for item in jepa_action_evidence.values()]
        effect_mean = sum(effect_values) / max(len(effect_values), 1)
        surprise_mean = sum(surprise_values) / max(len(surprise_values), 1)
        effect_span = (max(effect_values) - min(effect_values)) if effect_values else 0.0
        surprise_span = (max(surprise_values) - min(surprise_values)) if surprise_values else 0.0
        causal = {
            "no_effect_rate": float(no_effect / total),
            "visible_effect_rate": float(visible_effect / total),
            "positive_event_rate": float(positive / total),
            "terminal_seen": float(any(step.terminal for step in record.steps)),
            "max_repeated_action_fraction": float(max(repeated.values(), default=0.0)),
            "jepa_token_norm": float(torch.tensor(token_mean).norm().item()) if token_mean else 0.0,
            "jepa_action_effect_mean": float(effect_mean),
            "jepa_action_effect_span": float(effect_span),
            "jepa_prediction_surprise_mean": float(surprise_mean),
            "jepa_prediction_surprise_span": float(surprise_span),
            "jepa_causal_substrate_active": float(jepa_active),
        }
        next_plan: dict[str, float] = {}
        for action, value in events.items():
            next_plan[action] = next_plan.get(action, 0.0) + min(float(value), 3.0)
        for action, count in failed.items():
            next_plan[action] = next_plan.get(action, 0.0) - min(float(count), 4.0) * 0.25
        for action, count in effects.items():
            if action not in failed and action not in events:
                next_plan[action] = next_plan.get(action, 0.0) + min(float(count), 4.0) * 0.05
        if positive == 0:
            for action, fraction in repeated.items():
                next_plan[action] = next_plan.get(action, 0.0) - min(float(fraction), 1.0) * 0.45
        delayed_credit_edges = self._update_transition_graph(transition_outcomes)
        delayed_region_links = self._update_object_memory(object_hypotheses, transition_outcomes)
        delayed_component_links = self._update_component_memory(component_hypotheses, transition_outcomes)
        delayed_component_chain_links = self._update_component_chain_graph(component_hypotheses, transition_outcomes)
        delayed_relation_chain_links = self._update_component_relation_chain_graph(component_hypotheses, transition_outcomes)
        sequence_candidates_added = self._update_sequence_candidates(
            record,
            transition_outcomes,
            object_hypotheses,
            component_hypotheses,
        )
        graph_summary = self.transition_graph_summary()
        object_summary = self.object_memory_summary()
        causal.update(
            {
                "transition_graph_edges": float(graph_summary["observed_edges"]),
                "transition_graph_positive_edges": float(graph_summary["positive_edges"]),
                "transition_graph_no_effect_edges": float(graph_summary["no_effect_edges"]),
                "transition_graph_delayed_credit_edges": float(delayed_credit_edges),
                "object_changed_region_count": float(object_summary["observed_regions"]),
                "object_changed_color_count": float(object_summary["observed_colors"]),
                "object_delayed_region_links": float(delayed_region_links),
                "component_relation_count": float(object_summary["component_relation_count"]),
                "component_goal_relation_count": float(object_summary["component_goal_relation_count"]),
                "component_delayed_relation_links": float(delayed_component_links),
                "component_transition_prediction_count": float(object_summary["component_transition_prediction_count"]),
                "component_transition_contradictions": float(object_summary["component_transition_contradictions"]),
                "component_chain_edge_count": float(object_summary["component_chain_edge_count"]),
                "component_chain_goal_edge_count": float(object_summary["component_chain_goal_edge_count"]),
                "component_goal_state_count": float(object_summary["component_goal_state_count"]),
                "component_chain_contradictions": float(object_summary["component_chain_contradictions"]),
                "component_chain_delayed_links": float(delayed_component_chain_links),
                "component_relation_chain_edge_count": float(object_summary["component_relation_chain_edge_count"]),
                "component_relation_chain_goal_edge_count": float(object_summary["component_relation_chain_goal_edge_count"]),
                "component_relation_goal_state_count": float(object_summary["component_relation_goal_state_count"]),
                "component_relation_chain_contradictions": float(object_summary["component_relation_chain_contradictions"]),
                "component_relation_chain_delayed_links": float(delayed_relation_chain_links),
                "sequence_contradictions": float(object_summary["sequence_contradictions"]),
                "sequence_candidates_added": float(sequence_candidates_added),
                "sequence_candidate_count": float(object_summary["sequence_candidate_count"]),
            }
        )
        if jepa_active:
            for action, evidence in jepa_action_evidence.items():
                effect_centered = float(evidence.get("effect", 0.0) - effect_mean)
                surprise_centered = float(evidence.get("surprise", 0.0) - surprise_mean)
                count_scale = math.log1p(float(evidence.get("count", 0.0)))
                latent_rule_bias = (0.55 * effect_centered + 0.25 * surprise_centered) * max(count_scale, 1.0)
                if action in events:
                    latent_rule_bias += 0.20 * min(float(events[action]), 3.0)
                if action in failed and effect_centered < 0.0:
                    latent_rule_bias -= 0.15 * min(float(failed[action]), 4.0)
                next_plan[action] = next_plan.get(action, 0.0) + latent_rule_bias
        action_distribution_delta = _centered_distribution_delta(next_plan)
        entry = AttemptMemoryEntry(
            attempt_index=int(record.attempt_index),
            token_mean=token_mean,
            failed_actions=failed,
            effect_actions=effects,
            repeated_actions=repeated,
            event_candidates=events,
            causal_hypotheses=causal,
            next_attempt_plan=next_plan,
            transition_graph_summary=graph_summary,
            object_causal_hypotheses=object_hypotheses[:12],
            component_causal_hypotheses=component_hypotheses[:12],
            sequence_plan_summary=object_summary,
            jepa_action_evidence=jepa_action_evidence,
            action_distribution_delta=action_distribution_delta,
            causal_substrate_active=jepa_active,
        )
        self.entries.append(entry)
        return entry

    def score_action(self, action: str) -> float:
        if not self.entries:
            return 0.0
        score = 0.0
        for entry in self.entries[-3:]:
            score += 0.45 * float(entry.next_attempt_plan.get(action, 0.0))
        return float(max(min(score, 0.55), -0.55))

    def plan_scores(self, legal_actions: tuple[str, ...] | list[str]) -> dict[str, float]:
        return {str(action): self.score_action(str(action)) for action in legal_actions}

    def plan_scores_for_observation(self, observation: Any) -> dict[str, float]:
        legal_actions = [str(action) for action in getattr(observation, "available_actions", ())]
        if not legal_actions:
            return {}
        obs_key = _observation_key(observation)
        frame = _public_frame(observation)
        component_relations: dict[str, dict[str, Any]] = {}
        components: list[_FrameComponent] | None = None
        if frame is not None and frame.size:
            components = _frame_components(frame)
            for action in legal_actions:
                component_relations[action] = _component_target_relation(
                    frame,
                    _action_target_cell(action, frame),
                    components,
                )
        scores = self.plan_scores(legal_actions)
        seen_here = self.state_seen_actions.get(obs_key, set()) if obs_key else set()
        stuck = self._recent_stuck()
        for action in legal_actions:
            edge = (obs_key, action)
            edge_count = self.transition_counts.get(edge, 0)
            if obs_key and edge_count:
                edge_mean = self.transition_values.get(edge, 0.0) / max(edge_count, 1)
                scores[action] += 0.65 * edge_mean
                scores[action] -= 0.08 * min(float(self.transition_failures.get(edge, 0)), 4.0)
                if self.transition_events.get(edge, 0):
                    scores[action] += 0.12
            else:
                family = _action_family(action)
                family_count = self.family_counts.get(family, 0)
                if family_count:
                    scores[action] += 0.20 * (self.family_values.get(family, 0.0) / max(family_count, 1))
                if stuck:
                    scores[action] += 0.10
                if obs_key and seen_here and action not in seen_here:
                    scores[action] += 0.08
            scores[action] += self._object_region_score(action, frame)
            target_relation = component_relations.get(action)
            scores[action] += self._component_relation_score(action, frame, target_relation)
            scores[action] += self._component_transition_prediction_score(action, frame, target_relation)
        chain_plan = self._component_chain_plan(frame, legal_actions, components)
        relation_chain_plan = self._component_relation_chain_plan(frame, legal_actions, components)
        selected_plan = chain_plan
        if relation_chain_plan is not None and (
            selected_plan is None
            or float(relation_chain_plan.get("value", 0.0)) > float(selected_plan.get("value", 0.0)) + 0.08
        ):
            selected_plan = relation_chain_plan
        if selected_plan is not None:
            self._activate_component_chain_plan(selected_plan)
            first_action = str(selected_plan.get("actions", [""])[0])
            if first_action in scores:
                scale = 0.30 if str(selected_plan.get("source", "")) == "component_relation_goal_chain" else 0.34
                scores[first_action] += scale * min(max(float(selected_plan.get("value", 0.0)), 0.0), 1.5)
        planned_action = self.sequence_plan_action(legal_actions)
        if planned_action is not None:
            plan_support = self._sequence_plan_support(planned_action, frame, component_relations.get(planned_action))
            scores[planned_action] = scores.get(planned_action, 0.0) + (0.85 * plan_support)
        return {action: float(max(min(score, 0.75), -0.75)) for action, score in scores.items()}

    def action_distribution(self, legal_actions: tuple[str, ...] | list[str]) -> dict[str, float]:
        actions = [str(action) for action in legal_actions]
        if not actions:
            return {}
        scores = self.plan_scores(actions)
        return self.action_distribution_from_scores(scores)

    def action_distribution_from_scores(self, scores: dict[str, float]) -> dict[str, float]:
        if not scores:
            return {}
        actions = list(scores)
        mean_score = sum(scores.values()) / len(actions)
        logits = {action: max(min((scores[action] - mean_score) * 4.0, 8.0), -8.0) for action in actions}
        denom = sum(math.exp(value) for value in logits.values())
        return {action: float(math.exp(logits[action]) / max(denom, 1.0e-12)) for action in actions}

    def transition_graph_summary(self) -> dict[str, Any]:
        return {
            "observed_edges": len(self.transition_counts),
            "observed_states": len(self.state_seen_actions),
            "positive_edges": sum(1 for value in self.transition_events.values() if value > 0),
            "effect_edges": sum(1 for value in self.transition_effects.values() if value > 0),
            "no_effect_edges": sum(1 for value in self.transition_failures.values() if value > 0),
            "action_families": sorted(self.family_counts),
        }

    def object_memory_summary(self) -> dict[str, Any]:
        active_remaining = max(len(self.active_sequence) - self.sequence_cursor, 0)
        return {
            "observed_regions": len(self.region_counts),
            "observed_colors": len(self.color_counts),
            "region_failures": sum(1 for value in self.region_failures.values() if value > 0),
            "component_relation_count": len(self.component_relation_counts),
            "component_goal_relation_count": len(self.component_goal_relations),
            "component_failures": sum(1 for value in self.component_relation_failures.values() if value > 0),
            "component_values_seen": len(self.component_value_counts),
            "component_transition_prediction_count": len(self.component_prediction_counts),
            "component_transition_contradictions": sum(self.component_prediction_contradictions.values()),
            "component_chain_edge_count": len(self.component_chain_counts),
            "component_chain_goal_edge_count": len(self.component_chain_goal_values),
            "component_goal_state_count": len(self.component_goal_state_values),
            "component_chain_contradictions": sum(self.component_chain_contradictions.values()),
            "component_relation_chain_edge_count": len(self.component_relation_chain_counts),
            "component_relation_chain_goal_edge_count": len(self.component_relation_chain_goal_values),
            "component_relation_goal_state_count": len(self.component_relation_goal_state_values),
            "component_relation_chain_contradictions": sum(self.component_relation_chain_contradictions.values()),
            "sequence_contradictions": self.sequence_contradictions,
            "sequence_candidate_count": len(self.sequence_candidates),
            "active_sequence_length": len(self.active_sequence),
            "active_sequence_remaining": active_remaining,
            "active_sequence_source": self.active_sequence_source,
        }

    def start_attempt(self) -> None:
        self.active_sequence = []
        self.active_sequence_expectations = []
        self.sequence_cursor = 0
        self.active_sequence_source = ""
        candidates = sorted(
            self.sequence_candidates,
            key=lambda item: (float(item.get("value", 0.0)), int(item.get("length", 0))),
            reverse=True,
        )
        for candidate in candidates:
            actions = [str(action) for action in candidate.get("actions", []) if str(action)]
            if actions:
                self.active_sequence = actions[:12]
                expectations = candidate.get("component_expectations", [])
                self.active_sequence_expectations = [
                    item if isinstance(item, dict) else {} for item in list(expectations)[: len(self.active_sequence)]
                ]
                while len(self.active_sequence_expectations) < len(self.active_sequence):
                    self.active_sequence_expectations.append({})
                self.active_sequence_source = str(candidate.get("source", "prior_attempt_event_window"))
                return

    def sequence_plan_action(self, legal_actions: list[str] | tuple[str, ...]) -> str | None:
        if not self.active_sequence:
            return None
        legal = {str(action) for action in legal_actions}
        while self.sequence_cursor < len(self.active_sequence):
            action = self.active_sequence[self.sequence_cursor]
            if action in legal:
                return action
            self.sequence_cursor += 1
        return None

    def advance_sequence(self, chosen_action: str) -> None:
        if not self.active_sequence or self.sequence_cursor >= len(self.active_sequence):
            return
        if str(chosen_action) == self.active_sequence[self.sequence_cursor]:
            self.sequence_cursor += 1

    def observe_live_transition(self, before_observation: Any, action: str, result: Any) -> None:
        if not self.active_sequence or self.sequence_cursor >= len(self.active_sequence):
            return
        action = str(action)
        expected_action = self.active_sequence[self.sequence_cursor]
        if action != expected_action:
            return
        expected = (
            self.active_sequence_expectations[self.sequence_cursor]
            if self.sequence_cursor < len(self.active_sequence_expectations)
            else {}
        )
        before = _public_frame(before_observation)
        after = _public_frame(getattr(result, "observation", None))
        event_hit = bool(POSITIVE_EVENTS.intersection(getattr(result, "info", {}).get("events", []))) or float(
            getattr(result, "reward", 0.0)
        ) > 0.0
        if expected and before is not None and after is not None:
            actual = _component_hypothesis_from_frames(
                before=before,
                after=after,
                action=action,
                step_index=-1,
                event_hit=event_hit,
                score_delta=float(getattr(result, "reward", 0.0)),
                no_effect=not bool(np.any(before != after)),
            )
            if not _component_expectation_matches(expected, actual) and not event_hit:
                self.sequence_contradictions += 1
                self._penalize_component_prediction(expected)
                self.active_sequence = []
                self.active_sequence_expectations = []
                self.sequence_cursor = 0
                self.active_sequence_source = "aborted_by_public_component_contradiction"
                return
        self.advance_sequence(action)

    def summary(self) -> dict[str, Any]:
        return {
            "entry_count": len(self.entries),
            "use_jepa_tokens": self.use_jepa_tokens,
            "entries": [entry.to_dict() for entry in self.entries[-3:]],
            "emits_text": False,
            "direct_action_source": False,
            "causal_chain": [
                "attempt_video_action_history",
                "jepa_temporal_representation",
                "attempt_memory",
                "rule_causal_hypothesis_update",
                "changed_next_attempt_action_distribution",
            ],
            "planner_chain": [
                "public_observation_hash",
                "state_action_transition_edge",
                "public_frame_region_diff",
                "object_causal_hypothesis",
                "public_component_causal_graph",
                "delayed_public_event_credit",
                "prior_event_sequence_candidate",
                "component_grounded_sequence_check",
                "predicted_component_transition_planner",
                "component_transition_goal_chain_search",
                "component_relation_goal_chain_search",
                "targeted_next_attempt_experiment",
            ],
            "transition_graph": self.transition_graph_summary(),
            "object_memory": self.object_memory_summary(),
            "causal_substrate_active": any(entry.causal_substrate_active for entry in self.entries),
        }

    def _update_transition_graph(self, outcomes: list[dict[str, Any]]) -> int:
        delayed_credit_edges: set[tuple[str, str]] = set()
        for outcome in outcomes:
            edge = outcome["edge"]
            action = str(outcome["action"])
            value = float(outcome["value"])
            family = _action_family(action)
            self.transition_counts[edge] = self.transition_counts.get(edge, 0) + 1
            self.transition_values[edge] = self.transition_values.get(edge, 0.0) + value
            self.state_seen_actions.setdefault(edge[0], set()).add(action)
            self.family_counts[family] = self.family_counts.get(family, 0) + 1
            self.family_values[family] = self.family_values.get(family, 0.0) + value
            if outcome["event_hit"]:
                self.transition_events[edge] = self.transition_events.get(edge, 0) + 1
            if outcome["changed"]:
                self.transition_effects[edge] = self.transition_effects.get(edge, 0) + 1
            if outcome["no_effect"]:
                self.transition_failures[edge] = self.transition_failures.get(edge, 0) + 1

        event_indices = [index for index, outcome in enumerate(outcomes) if outcome["event_hit"]]
        for event_index in event_indices:
            for prior_index in range(max(0, event_index - 6), event_index):
                edge = outcomes[prior_index]["edge"]
                distance = event_index - prior_index
                credit = 0.42 / float(distance + 1)
                self.transition_values[edge] = self.transition_values.get(edge, 0.0) + credit
                delayed_credit_edges.add(edge)
        return len(delayed_credit_edges)

    def _update_object_memory(self, hypotheses: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> int:
        event_indices = [index for index, outcome in enumerate(outcomes) if outcome["event_hit"]]
        delayed_links: set[tuple[int, int]] = set()
        delayed_credit_by_index: dict[int, float] = {}
        for event_index in event_indices:
            for prior_index in range(max(0, event_index - 8), event_index):
                distance = event_index - prior_index
                delayed_credit_by_index[prior_index] = delayed_credit_by_index.get(prior_index, 0.0) + 0.36 / float(distance + 1)
        for hypothesis in hypotheses:
            region = _hypothesis_region(hypothesis)
            if region is None:
                continue
            index = int(hypothesis.get("step_index", -1))
            value = _object_hypothesis_value(hypothesis) + delayed_credit_by_index.get(index, 0.0)
            family = str(hypothesis.get("action_family", "other"))
            self.region_counts[region] = self.region_counts.get(region, 0) + 1
            self.region_values[region] = self.region_values.get(region, 0.0) + value
            family_key = (family, region[0], region[1])
            self.family_region_counts[family_key] = self.family_region_counts.get(family_key, 0) + 1
            self.family_region_values[family_key] = self.family_region_values.get(family_key, 0.0) + value
            if str(hypothesis.get("mechanism")) == "blocked_or_no_effect":
                self.region_failures[region] = self.region_failures.get(region, 0) + 1
            if index in delayed_credit_by_index:
                delayed_links.add(region)
            for color in hypothesis.get("changed_colors", []):
                color_id = int(color)
                self.color_counts[color_id] = self.color_counts.get(color_id, 0) + 1
                self.color_values[color_id] = self.color_values.get(color_id, 0.0) + value
        return len(delayed_links)

    def _update_component_memory(self, hypotheses: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> int:
        event_indices = [index for index, outcome in enumerate(outcomes) if outcome["event_hit"]]
        delayed_links: set[str] = set()
        delayed_credit_by_index: dict[int, float] = {}
        for event_index in event_indices:
            for prior_index in range(max(0, event_index - 10), event_index):
                distance = event_index - prior_index
                delayed_credit_by_index[prior_index] = delayed_credit_by_index.get(prior_index, 0.0) + 0.44 / float(
                    distance + 1
                )
        for hypothesis in hypotheses:
            relation_key = str(hypothesis.get("relation_key", ""))
            if not relation_key:
                continue
            index = int(hypothesis.get("step_index", -1))
            value = _component_hypothesis_value(hypothesis) + delayed_credit_by_index.get(index, 0.0)
            family = str(hypothesis.get("action_family", "other"))
            self.component_relation_counts[relation_key] = self.component_relation_counts.get(relation_key, 0) + 1
            self.component_relation_values[relation_key] = self.component_relation_values.get(relation_key, 0.0) + value
            family_key = (family, relation_key)
            self.family_component_counts[family_key] = self.family_component_counts.get(family_key, 0) + 1
            self.family_component_values[family_key] = self.family_component_values.get(family_key, 0.0) + value
            value_id = int(hypothesis.get("target_value", 0) or hypothesis.get("component_value", 0) or 0)
            if value_id:
                self.component_value_counts[value_id] = self.component_value_counts.get(value_id, 0) + 1
                self.component_value_scores[value_id] = self.component_value_scores.get(value_id, 0.0) + value
            mechanism = str(hypothesis.get("mechanism", ""))
            fallback_relation = str(hypothesis.get("fallback_relation", ""))
            prediction_relations = [(relation_key, 1.0)]
            if fallback_relation and fallback_relation != relation_key:
                prediction_relations.append((fallback_relation, 0.55))
            for prediction_relation, weight in prediction_relations:
                prediction_key = _component_prediction_key(prediction_relation, mechanism)
                weighted_value = value * weight
                self.component_prediction_counts[prediction_key] = (
                    self.component_prediction_counts.get(prediction_key, 0) + 1
                )
                self.component_prediction_values[prediction_key] = (
                    self.component_prediction_values.get(prediction_key, 0.0) + weighted_value
                )
                family_prediction_key = (family, prediction_key)
                self.family_component_prediction_counts[family_prediction_key] = (
                    self.family_component_prediction_counts.get(family_prediction_key, 0) + 1
                )
                self.family_component_prediction_values[family_prediction_key] = (
                    self.family_component_prediction_values.get(family_prediction_key, 0.0) + weighted_value
                )
                if bool(hypothesis.get("event_linked")) or index in delayed_credit_by_index:
                    self.component_prediction_goal_values[prediction_key] = (
                        self.component_prediction_goal_values.get(prediction_key, 0.0) + max(weighted_value, 0.1 * weight)
                    )
            if str(hypothesis.get("mechanism")) == "blocked_or_no_effect":
                self.component_relation_failures[relation_key] = self.component_relation_failures.get(relation_key, 0) + 1
            if bool(hypothesis.get("event_linked")) or index in delayed_credit_by_index:
                self.component_goal_relations[relation_key] = self.component_goal_relations.get(relation_key, 0.0) + max(
                    value,
                    0.1,
                )
                delayed_links.add(relation_key)
        return len(delayed_links)

    def _update_component_chain_graph(self, hypotheses: list[dict[str, Any]], outcomes: list[dict[str, Any]]) -> int:
        event_indices = [index for index, outcome in enumerate(outcomes) if outcome["event_hit"]]
        delayed_goal_edges: set[tuple[str, str, str]] = set()
        delayed_credit_by_index: dict[int, float] = {}
        for event_index in event_indices:
            for prior_index in range(max(0, event_index - 10), event_index):
                distance = event_index - prior_index
                delayed_credit_by_index[prior_index] = delayed_credit_by_index.get(prior_index, 0.0) + 0.48 / float(
                    distance + 1
                )
        for hypothesis in hypotheses:
            before_state = str(hypothesis.get("before_state_signature", ""))
            after_state = str(hypothesis.get("after_state_signature", ""))
            action = str(hypothesis.get("action", ""))
            if not before_state or not after_state or not action:
                continue
            index = int(hypothesis.get("step_index", -1))
            outcome = outcomes[index] if 0 <= index < len(outcomes) else {}
            key = (before_state, action, after_state)
            value = (
                _component_hypothesis_value(hypothesis)
                + 0.35 * float(outcome.get("value", 0.0))
                + delayed_credit_by_index.get(index, 0.0)
            )
            self.component_chain_counts[key] = self.component_chain_counts.get(key, 0) + 1
            self.component_chain_values[key] = self.component_chain_values.get(key, 0.0) + value
            self.component_chain_edges_by_state.setdefault(before_state, set()).add((action, after_state))
            self.component_chain_expectations[key] = _sequence_component_expectation(hypothesis)
            mechanism = str(hypothesis.get("mechanism", ""))
            if mechanism in NON_PRODUCTIVE_COMPONENT_MECHANISMS or bool(outcome.get("no_effect", False)):
                self.component_chain_failures[key] = self.component_chain_failures.get(key, 0) + 1
            if bool(hypothesis.get("event_linked")) or index in delayed_credit_by_index or bool(outcome.get("event_hit", False)):
                goal_value = max(value, 0.1)
                self.component_chain_goal_values[key] = self.component_chain_goal_values.get(key, 0.0) + goal_value
                self.component_goal_state_values[after_state] = self.component_goal_state_values.get(after_state, 0.0) + goal_value
                delayed_goal_edges.add(key)
        return len(delayed_goal_edges)

    def _update_component_relation_chain_graph(
        self,
        hypotheses: list[dict[str, Any]],
        outcomes: list[dict[str, Any]],
    ) -> int:
        event_indices = [index for index, outcome in enumerate(outcomes) if outcome["event_hit"]]
        delayed_goal_edges: set[tuple[str, str, str]] = set()
        delayed_credit_by_index: dict[int, float] = {}
        for event_index in event_indices:
            for prior_index in range(max(0, event_index - 10), event_index):
                distance = event_index - prior_index
                delayed_credit_by_index[prior_index] = delayed_credit_by_index.get(prior_index, 0.0) + 0.42 / float(
                    distance + 1
                )
        for hypothesis in hypotheses:
            before_state = str(hypothesis.get("before_relation_signature", ""))
            after_state = str(hypothesis.get("after_relation_signature", ""))
            action_template = str(hypothesis.get("action_template", ""))
            if not before_state or not after_state or not action_template:
                continue
            index = int(hypothesis.get("step_index", -1))
            outcome = outcomes[index] if 0 <= index < len(outcomes) else {}
            key = (before_state, action_template, after_state)
            value = (
                _component_hypothesis_value(hypothesis)
                + 0.30 * float(outcome.get("value", 0.0))
                + delayed_credit_by_index.get(index, 0.0)
            )
            self.component_relation_chain_counts[key] = self.component_relation_chain_counts.get(key, 0) + 1
            self.component_relation_chain_values[key] = self.component_relation_chain_values.get(key, 0.0) + value
            self.component_relation_chain_edges_by_state.setdefault(before_state, set()).add((action_template, after_state))
            self.component_relation_chain_expectations[key] = _relation_sequence_component_expectation(hypothesis)
            mechanism = str(hypothesis.get("mechanism", ""))
            if mechanism in NON_PRODUCTIVE_COMPONENT_MECHANISMS or bool(outcome.get("no_effect", False)):
                self.component_relation_chain_failures[key] = self.component_relation_chain_failures.get(key, 0) + 1
            if bool(hypothesis.get("event_linked")) or index in delayed_credit_by_index or bool(outcome.get("event_hit", False)):
                goal_value = max(value, 0.1)
                self.component_relation_chain_goal_values[key] = (
                    self.component_relation_chain_goal_values.get(key, 0.0) + goal_value
                )
                self.component_relation_goal_state_values[after_state] = (
                    self.component_relation_goal_state_values.get(after_state, 0.0) + goal_value
                )
                delayed_goal_edges.add(key)
        return len(delayed_goal_edges)

    def _update_sequence_candidates(
        self,
        record: AttemptRecord,
        outcomes: list[dict[str, Any]],
        hypotheses: list[dict[str, Any]],
        component_hypotheses: list[dict[str, Any]],
    ) -> int:
        added = 0
        hypothesis_by_index = {int(item.get("step_index", -1)): item for item in hypotheses}
        component_by_index = {int(item.get("step_index", -1)): item for item in component_hypotheses}
        existing = {tuple(str(action) for action in item.get("actions", [])) for item in self.sequence_candidates}
        for event_index, outcome in enumerate(outcomes):
            if not outcome["event_hit"]:
                continue
            start = max(0, event_index - 8)
            window_steps = record.steps[start : event_index + 1]
            actions = [str(step.action) for step in window_steps]
            if not actions:
                continue
            key = tuple(actions)
            if key in existing:
                continue
            linked = [
                hypothesis_by_index[index]
                for index in range(start, event_index + 1)
                if index in hypothesis_by_index and _hypothesis_region(hypothesis_by_index[index]) is not None
            ]
            linked_components = [
                component_by_index[index]
                for index in range(start, event_index + 1)
                if index in component_by_index and str(component_by_index[index].get("relation_key", ""))
            ]
            component_expectations = [
                _sequence_component_expectation(component_by_index.get(index, {})) for index in range(start, event_index + 1)
            ]
            value = (
                1.0
                + max(float(record.steps[event_index].score_delta), 0.0)
                + 0.08 * len(linked)
                + 0.10 * len(linked_components)
            )
            self.sequence_candidates.append(
                {
                    "source": "positive_public_event_window",
                    "actions": actions,
                    "value": float(value),
                    "length": len(actions),
                    "event_index": event_index,
                    "linked_regions": [_hypothesis_region(item) for item in linked[:4]],
                    "linked_mechanisms": [str(item.get("mechanism")) for item in linked[:4]],
                    "linked_component_relations": [str(item.get("relation_key")) for item in linked_components[:6]],
                    "component_expectations": component_expectations[: len(actions)],
                }
            )
            existing.add(key)
            added += 1
        self.sequence_candidates = sorted(
            self.sequence_candidates,
            key=lambda item: (float(item.get("value", 0.0)), int(item.get("length", 0))),
            reverse=True,
        )[:24]
        return added

    def _object_region_score(self, action: str, frame: np.ndarray | None) -> float:
        score = 0.0
        family = _action_family(action)
        click_region = _click_cell(action, frame.shape if frame is not None and frame.size else (8, 8))
        candidate_regions = [click_region] if click_region is not None else []
        if not candidate_regions and frame is not None and frame.size:
            candidate_regions = _nonzero_regions(frame)[:8]
        seen_regions: set[tuple[int, int]] = set()
        for region in candidate_regions:
            if region in seen_regions:
                continue
            seen_regions.add(region)
            for near_region, weight in _neighbor_regions(region):
                count = self.region_counts.get(near_region, 0)
                if count:
                    score += weight * 0.22 * (self.region_values.get(near_region, 0.0) / max(count, 1))
                family_key = (family, near_region[0], near_region[1])
                family_count = self.family_region_counts.get(family_key, 0)
                if family_count:
                    score += weight * 0.18 * (self.family_region_values.get(family_key, 0.0) / max(family_count, 1))
                score -= weight * 0.05 * min(float(self.region_failures.get(near_region, 0)), 4.0)
            if frame is not None and frame.size:
                y, x = region
                if 0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]:
                    color = int(frame[y, x])
                    color_count = self.color_counts.get(color, 0)
                    if color and color_count:
                        score += 0.10 * (self.color_values.get(color, 0.0) / max(color_count, 1))
        return float(max(min(score, 0.35), -0.35))

    def _component_relation_score(
        self,
        action: str,
        frame: np.ndarray | None,
        target_relation: dict[str, Any] | None = None,
    ) -> float:
        if target_relation is None and (frame is None or not frame.size):
            return 0.0
        family = _action_family(action)
        if target_relation is None and frame is not None:
            target = _action_target_cell(action, frame)
            target_relation = _component_target_relation(frame, target)
        if target_relation is None:
            return 0.0
        relation_keys = _candidate_relation_keys(family, target_relation)
        score = 0.0
        for relation_key, weight in relation_keys:
            count = self.component_relation_counts.get(relation_key, 0)
            if count:
                score += weight * 0.26 * (self.component_relation_values.get(relation_key, 0.0) / max(count, 1))
            family_key = (family, relation_key)
            family_count = self.family_component_counts.get(family_key, 0)
            if family_count:
                score += weight * 0.18 * (self.family_component_values.get(family_key, 0.0) / max(family_count, 1))
            if relation_key in self.component_goal_relations:
                score += weight * 0.16 * self.component_goal_relations[relation_key]
            score -= weight * 0.06 * min(float(self.component_relation_failures.get(relation_key, 0)), 4.0)
        value = int(target_relation.get("value", 0))
        if value:
            count = self.component_value_counts.get(value, 0)
            if count:
                score += 0.12 * (self.component_value_scores.get(value, 0.0) / max(count, 1))
        if family == "move" and not relation_keys:
            score += 0.05
        return float(max(min(score, 0.45), -0.45))

    def _component_transition_prediction_score(
        self,
        action: str,
        frame: np.ndarray | None,
        target_relation: dict[str, Any] | None = None,
    ) -> float:
        if target_relation is None and (frame is None or not frame.size):
            return 0.0
        family = _action_family(action)
        if target_relation is None and frame is not None:
            target_relation = _component_target_relation(frame, _action_target_cell(action, frame))
        if target_relation is None:
            return 0.0
        productive_score = 0.0
        nonproductive_penalty = 0.0
        for relation_key, relation_weight in _candidate_relation_keys(family, target_relation):
            for mechanism in PREDICTIVE_COMPONENT_MECHANISMS:
                prediction_key = _component_prediction_key(relation_key, mechanism)
                count = self.component_prediction_counts.get(prediction_key, 0)
                if not count:
                    continue
                mean_value = self.component_prediction_values.get(prediction_key, 0.0) / max(count, 1)
                confidence = min(1.0, math.log1p(float(count)) / math.log(5.0))
                contradictions = float(self.component_prediction_contradictions.get(prediction_key, 0))
                contradiction_rate = contradictions / max(float(count) + contradictions, 1.0)
                goal_value = self.component_prediction_goal_values.get(prediction_key, 0.0) / max(count, 1)
                family_key = (family, prediction_key)
                family_count = self.family_component_prediction_counts.get(family_key, 0)
                family_mean = (
                    self.family_component_prediction_values.get(family_key, 0.0) / max(family_count, 1)
                    if family_count
                    else 0.0
                )
                weighted_confidence = relation_weight * confidence * max(0.0, 1.0 - contradiction_rate)
                if mechanism in PRODUCTIVE_COMPONENT_MECHANISMS:
                    candidate_score = weighted_confidence * (
                        0.22 * max(mean_value, 0.0)
                        + 0.16 * max(goal_value, 0.0)
                        + 0.08 * max(family_mean, 0.0)
                    )
                    productive_score = max(productive_score, candidate_score)
                elif mechanism in NON_PRODUCTIVE_COMPONENT_MECHANISMS:
                    penalty_mean = max(-mean_value, 0.0) + 0.18 * max(1.0 - max(mean_value, 0.0), 0.0)
                    nonproductive_penalty += relation_weight * confidence * penalty_mean * 0.20
        score = productive_score - min(nonproductive_penalty, 0.28)
        return float(max(min(score, 0.38), -0.32))

    def _component_chain_edge_score(self, key: tuple[str, str, str]) -> float:
        count = self.component_chain_counts.get(key, 0)
        if count <= 0:
            return 0.0
        mean_value = self.component_chain_values.get(key, 0.0) / max(count, 1)
        goal_value = self.component_chain_goal_values.get(key, 0.0) / max(count, 1)
        state_goal = min(max(self.component_goal_state_values.get(key[2], 0.0), 0.0), 3.0)
        failures = float(self.component_chain_failures.get(key, 0))
        contradictions = float(self.component_chain_contradictions.get(key, 0))
        confidence = min(1.0, math.log1p(float(count)) / math.log(6.0))
        reliability = max(0.0, 1.0 - (0.45 * failures + contradictions) / max(float(count) + failures + contradictions, 1.0))
        score = confidence * reliability * (
            0.38 * max(mean_value, 0.0) + 0.45 * max(goal_value, 0.0) + 0.15 * state_goal
        )
        if mean_value < 0.0:
            score += 0.18 * mean_value
        score -= 0.10 * min(failures, 4.0)
        score -= 0.28 * min(contradictions, 4.0)
        return float(max(min(score, 1.25), -0.50))

    def _component_relation_chain_edge_score(self, key: tuple[str, str, str]) -> float:
        count = self.component_relation_chain_counts.get(key, 0)
        if count <= 0:
            return 0.0
        mean_value = self.component_relation_chain_values.get(key, 0.0) / max(count, 1)
        goal_value = self.component_relation_chain_goal_values.get(key, 0.0) / max(count, 1)
        state_goal = min(max(self.component_relation_goal_state_values.get(key[2], 0.0), 0.0), 3.0)
        failures = float(self.component_relation_chain_failures.get(key, 0))
        contradictions = float(self.component_relation_chain_contradictions.get(key, 0))
        confidence = min(1.0, math.log1p(float(count)) / math.log(5.0))
        reliability = max(0.0, 1.0 - (0.50 * failures + contradictions) / max(float(count) + failures + contradictions, 1.0))
        score = confidence * reliability * (
            0.34 * max(mean_value, 0.0) + 0.50 * max(goal_value, 0.0) + 0.18 * state_goal
        )
        if mean_value < 0.0:
            score += 0.16 * mean_value
        score -= 0.12 * min(failures, 4.0)
        score -= 0.30 * min(contradictions, 4.0)
        return float(max(min(score, 1.10), -0.45))

    def _component_chain_plan(
        self,
        frame: np.ndarray | None,
        legal_actions: list[str] | tuple[str, ...],
        components: list[_FrameComponent] | None = None,
    ) -> dict[str, Any] | None:
        if frame is None or not frame.size or not self.component_chain_edges_by_state:
            return None
        start_state = _component_state_signature(frame, components)
        if not start_state:
            return None
        legal = {str(action) for action in legal_actions}
        best: dict[str, Any] | None = None
        max_depth = 3

        def remember(
            actions: list[str],
            expectations: list[dict[str, Any]],
            edge_keys: list[tuple[str, str, str]],
            value: float,
        ) -> None:
            nonlocal best
            if not actions:
                return
            adjusted = float(value) - 0.035 * float(max(len(actions) - 1, 0))
            if adjusted <= 0.03:
                return
            if best is None or adjusted > float(best.get("value", 0.0)):
                best = {
                    "source": "component_transition_goal_chain",
                    "actions": list(actions),
                    "component_expectations": [dict(item) for item in expectations],
                    "edge_keys": [list(item) for item in edge_keys],
                    "value": adjusted,
                    "length": len(actions),
                    "start_state": start_state,
                }

        def dfs(
            state: str,
            depth: int,
            actions: list[str],
            expectations: list[dict[str, Any]],
            edge_keys: list[tuple[str, str, str]],
            value: float,
            visited: set[str],
        ) -> None:
            if depth >= max_depth:
                remember(actions, expectations, edge_keys, value)
                return
            candidates: list[tuple[float, str, str, tuple[str, str, str]]] = []
            for action, after_state in self.component_chain_edges_by_state.get(state, set()):
                if depth == 0 and action not in legal:
                    continue
                key = (state, action, after_state)
                edge_score = self._component_chain_edge_score(key)
                if edge_score <= -0.20:
                    continue
                candidates.append((edge_score, action, after_state, key))
            candidates.sort(reverse=True)
            for edge_score, action, after_state, key in candidates[:10]:
                expectation = dict(self.component_chain_expectations.get(key, {}))
                expectation.setdefault("action", action)
                expectation.setdefault("before_state_signature", state)
                expectation.setdefault("after_state_signature", after_state)
                new_actions = actions + [action]
                new_expectations = expectations + [expectation]
                new_edge_keys = edge_keys + [key]
                state_goal = min(max(self.component_goal_state_values.get(after_state, 0.0), 0.0), 3.0)
                goal_bonus = 0.16 * state_goal
                new_value = value + edge_score + goal_bonus
                remember(new_actions, new_expectations, new_edge_keys, new_value)
                if after_state not in visited:
                    dfs(
                        after_state,
                        depth + 1,
                        new_actions,
                        new_expectations,
                        new_edge_keys,
                        new_value * 0.92,
                        visited | {after_state},
                    )

        dfs(start_state, 0, [], [], [], 0.0, {start_state})
        return best

    def _component_relation_chain_plan(
        self,
        frame: np.ndarray | None,
        legal_actions: list[str] | tuple[str, ...],
        components: list[_FrameComponent] | None = None,
    ) -> dict[str, Any] | None:
        if frame is None or not frame.size or not self.component_relation_chain_edges_by_state:
            return None
        start_state = _component_relation_state_signature(frame, components)
        if not start_state:
            return None
        best: dict[str, Any] | None = None
        max_depth = 3

        def remember(
            actions: list[str],
            expectations: list[dict[str, Any]],
            edge_keys: list[tuple[str, str, str]],
            value: float,
        ) -> None:
            nonlocal best
            if not actions:
                return
            adjusted = float(value) - 0.045 * float(max(len(actions) - 1, 0))
            if adjusted <= 0.04:
                return
            if best is None or adjusted > float(best.get("value", 0.0)):
                best = {
                    "source": "component_relation_goal_chain",
                    "actions": list(actions),
                    "component_expectations": [dict(item) for item in expectations],
                    "edge_keys": [list(item) for item in edge_keys],
                    "value": adjusted,
                    "length": len(actions),
                    "start_relation_state": start_state,
                }

        def dfs(
            state: str,
            depth: int,
            actions: list[str],
            expectations: list[dict[str, Any]],
            edge_keys: list[tuple[str, str, str]],
            value: float,
            visited: set[str],
        ) -> None:
            if depth >= max_depth:
                remember(actions, expectations, edge_keys, value)
                return
            candidates: list[tuple[float, str, str, tuple[str, str, str], str | None]] = []
            for action_template, after_state in self.component_relation_chain_edges_by_state.get(state, set()):
                resolved_action = None
                if depth == 0:
                    resolved_action = _resolve_component_relation_action_template(
                        action_template,
                        legal_actions,
                        frame,
                        components,
                    )
                    if resolved_action is None:
                        continue
                key = (state, action_template, after_state)
                edge_score = self._component_relation_chain_edge_score(key)
                if edge_score <= -0.18:
                    continue
                candidates.append((edge_score, action_template, after_state, key, resolved_action))
            candidates.sort(reverse=True)
            for edge_score, action_template, after_state, key, resolved_action in candidates[:10]:
                action = str(resolved_action or action_template)
                expectation = dict(self.component_relation_chain_expectations.get(key, {}))
                expectation.setdefault("action_template", action_template)
                expectation.setdefault("before_relation_signature", state)
                expectation.setdefault("after_relation_signature", after_state)
                if resolved_action is not None:
                    expectation["action"] = str(resolved_action)
                else:
                    expectation.setdefault("action", action)
                new_actions = actions + [action]
                new_expectations = expectations + [expectation]
                new_edge_keys = edge_keys + [key]
                state_goal = min(max(self.component_relation_goal_state_values.get(after_state, 0.0), 0.0), 3.0)
                goal_bonus = 0.14 * state_goal
                new_value = value + edge_score + goal_bonus
                remember(new_actions, new_expectations, new_edge_keys, new_value)
                if after_state not in visited:
                    dfs(
                        after_state,
                        depth + 1,
                        new_actions,
                        new_expectations,
                        new_edge_keys,
                        new_value * 0.90,
                        visited | {after_state},
                    )

        dfs(start_state, 0, [], [], [], 0.0, {start_state})
        return best

    def _activate_component_chain_plan(self, plan: dict[str, Any]) -> bool:
        actions = [str(action) for action in plan.get("actions", []) if str(action)]
        if not actions:
            return False
        source = str(plan.get("source", "component_transition_goal_chain"))
        remaining = self.active_sequence[self.sequence_cursor :] if self.active_sequence else []
        if self.active_sequence_source == source and remaining:
            if remaining == actions[: len(remaining)]:
                return False
            if float(plan.get("value", 0.0)) < 0.20:
                return False
        expectations = [
            item if isinstance(item, dict) else {} for item in list(plan.get("component_expectations", []))[: len(actions)]
        ]
        while len(expectations) < len(actions):
            expectations.append({})
        self.active_sequence = actions[:8]
        self.active_sequence_expectations = expectations[: len(self.active_sequence)]
        self.sequence_cursor = 0
        self.active_sequence_source = source
        return True

    def _sequence_plan_support(
        self,
        action: str,
        frame: np.ndarray | None,
        target_relation: dict[str, Any] | None = None,
    ) -> float:
        if not self.active_sequence_expectations or self.sequence_cursor >= len(self.active_sequence_expectations):
            return 1.0
        expected = self.active_sequence_expectations[self.sequence_cursor]
        relation_key = str(expected.get("relation_key", ""))
        if not relation_key or frame is None or not frame.size:
            return 1.0
        expected_before = str(expected.get("before_state_signature", ""))
        if expected_before and _component_state_signature(frame) != expected_before:
            return 0.25
        expected_relation_before = str(expected.get("before_relation_signature", ""))
        if expected_relation_before and _component_relation_state_signature(frame) != expected_relation_before:
            return 0.25
        action_template = str(expected.get("action_template", ""))
        if action_template:
            resolved_action = _resolve_component_relation_action_template(action_template, [action], frame)
            if resolved_action == action:
                return 1.0
        if target_relation is None:
            target_relation = _component_target_relation(frame, _action_target_cell(action, frame))
        current_keys = {key for key, _ in _candidate_relation_keys(_action_family(action), target_relation)}
        if relation_key in current_keys or str(expected.get("fallback_relation", "")) in current_keys:
            return 1.0
        expected_value = int(expected.get("target_value", 0) or 0)
        if expected_value and target_relation.get("value") == expected_value:
            return 0.85
        return 0.35

    def _penalize_component_prediction(self, expected: dict[str, Any]) -> None:
        mechanism = str(expected.get("mechanism", ""))
        if not mechanism:
            return
        relation_keys = [str(expected.get("relation_key", "")), str(expected.get("fallback_relation", ""))]
        for relation_key in relation_keys:
            if not relation_key:
                continue
            prediction_key = _component_prediction_key(relation_key, mechanism)
            self.component_prediction_contradictions[prediction_key] = (
                self.component_prediction_contradictions.get(prediction_key, 0) + 1
            )
        before_state = str(expected.get("before_state_signature", ""))
        after_state = str(expected.get("after_state_signature", ""))
        action = str(expected.get("action", ""))
        if before_state and after_state and action:
            chain_key = (before_state, action, after_state)
            self.component_chain_contradictions[chain_key] = self.component_chain_contradictions.get(chain_key, 0) + 1
        before_relation = str(expected.get("before_relation_signature", ""))
        after_relation = str(expected.get("after_relation_signature", ""))
        action_template = str(expected.get("action_template", ""))
        if before_relation and after_relation and action_template:
            relation_key = (before_relation, action_template, after_relation)
            self.component_relation_chain_contradictions[relation_key] = (
                self.component_relation_chain_contradictions.get(relation_key, 0) + 1
            )

    def _recent_stuck(self) -> bool:
        if not self.entries:
            return False
        for entry in self.entries[-2:]:
            no_effect_rate = float(entry.causal_hypotheses.get("no_effect_rate", 0.0))
            repeated_fraction = float(entry.causal_hypotheses.get("max_repeated_action_fraction", 0.0))
            if no_effect_rate >= 0.35 or repeated_fraction >= 0.50:
                return True
        return False


def _jepa_action_evidence(record: AttemptRecord, output: dict[str, torch.Tensor]) -> dict[str, dict[str, float]]:
    context = output["context_tokens"][0].detach().cpu()
    predicted = output["predicted_future"][0].detach().cpu()
    target = output["target_future"][0].detach().cpu()
    evidence: dict[str, dict[str, float]] = {}
    for index, step in enumerate(record.steps):
        action = str(step.action)
        if action not in evidence:
            evidence[action] = {"effect": 0.0, "surprise": 0.0, "count": 0.0}
        if index + 1 < context.shape[0]:
            effect = float(torch.norm(context[index + 1] - context[index]).item())
        else:
            effect = 0.0
        if index < predicted.shape[0]:
            surprise = float(torch.mean((predicted[index] - target[index]).pow(2)).item())
        else:
            surprise = 0.0
        evidence[action]["effect"] += effect
        evidence[action]["surprise"] += surprise
        evidence[action]["count"] += 1.0
    for action, values in evidence.items():
        count = max(float(values.get("count", 0.0)), 1.0)
        values["effect"] = float(values["effect"] / count)
        values["surprise"] = float(values["surprise"] / count)
        values["count"] = float(count)
    return evidence


def _is_no_effect_event(events: list[str]) -> bool:
    for event in events:
        lowered = str(event).lower()
        if (
            "no_effect" in lowered
            or "blocked" in lowered
            or "invalid" in lowered
            or "bad_click" in lowered
            or "already_terminal" in lowered
            or "runtime_no_response" in lowered
        ):
            return True
    return False


def _transition_credit(
    *,
    event_hit: bool,
    changed: bool,
    no_effect_event: bool,
    invalid_action: bool,
    score_delta: float,
    terminal: bool,
) -> float:
    if invalid_action:
        return -1.0
    if event_hit:
        return 1.15 + min(max(float(score_delta), 0.0), 2.0)
    if no_effect_event or not changed:
        return -0.65
    if terminal:
        return 0.05
    return 0.18


def _observation_key(observation: Any) -> str:
    try:
        return stable_hash(
            {
                "grid": np.asarray(observation.grid, dtype=np.int64),
                "extras": getattr(observation, "extras", {}),
                "actions": getattr(observation, "available_actions", ()),
            }
        )
    except Exception:
        return ""


def _action_family(action: str) -> str:
    lowered = str(action).lower()
    if lowered in {"up", "down", "left", "right", "north", "south", "east", "west"}:
        return "move"
    if "click" in lowered or "press" in lowered or "tap" in lowered or "touch" in lowered:
        return "contact"
    if lowered in {"wait", "noop", "no_op", "none"}:
        return "wait"
    if "toggle" in lowered or "use" in lowered or "pickup" in lowered or "drop" in lowered:
        return "object"
    return "other"


def _object_hypotheses_from_record(record: AttemptRecord, outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hypotheses: list[dict[str, Any]] = []
    for index, step in enumerate(record.steps):
        before = np.asarray(step.frame, dtype=np.int64)
        after = _step_after_frame(record, index)
        if after is None or before.size == 0:
            continue
        outcome = outcomes[index] if index < len(outcomes) else {}
        hypothesis = _frame_diff_hypothesis(
            before=before,
            after=after,
            action=str(step.action),
            step_index=index,
            event_hit=bool(outcome.get("event_hit", False)),
            score_delta=float(step.score_delta),
            no_effect=bool(outcome.get("no_effect", False)),
        )
        if hypothesis is not None:
            hypotheses.append(hypothesis)
    return hypotheses


@dataclass(frozen=True)
class _FrameComponent:
    component_id: int
    value: int
    cells: tuple[tuple[int, int], ...]
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]

    @property
    def area(self) -> int:
        return len(self.cells)

    def compact(self) -> dict[str, Any]:
        return {
            "id": int(self.component_id),
            "value": int(self.value),
            "area": int(self.area),
            "bbox": [int(item) for item in self.bbox],
            "centroid": [round(float(self.centroid[0]), 3), round(float(self.centroid[1]), 3)],
        }


def _component_hypotheses_from_record(record: AttemptRecord, outcomes: list[dict[str, Any]]) -> list[dict[str, Any]]:
    hypotheses: list[dict[str, Any]] = []
    for index, step in enumerate(record.steps):
        before = np.asarray(step.frame, dtype=np.int64)
        after = _step_after_frame(record, index)
        if after is None or before.size == 0:
            continue
        outcome = outcomes[index] if index < len(outcomes) else {}
        hypothesis = _component_hypothesis_from_frames(
            before=before,
            after=after,
            action=str(step.action),
            step_index=index,
            event_hit=bool(outcome.get("event_hit", False)),
            score_delta=float(step.score_delta),
            no_effect=bool(outcome.get("no_effect", False)),
        )
        if hypothesis is not None:
            hypotheses.append(hypothesis)
    return hypotheses


def _component_hypothesis_from_frames(
    *,
    before: np.ndarray,
    after: np.ndarray,
    action: str,
    step_index: int,
    event_hit: bool,
    score_delta: float,
    no_effect: bool,
) -> dict[str, Any] | None:
    if before.shape != after.shape:
        size_y = min(before.shape[0], after.shape[0])
        size_x = min(before.shape[1], after.shape[1])
        before = before[:size_y, :size_x]
        after = after[:size_y, :size_x]
    if before.size == 0 or after.size == 0:
        return None
    family = _action_family(action)
    before_components = _frame_components(before)
    after_components = _frame_components(after)
    before_state_signature = _component_state_signature(before, before_components)
    after_state_signature = _component_state_signature(after, after_components)
    before_relation_signature = _component_relation_state_signature(before, before_components)
    after_relation_signature = _component_relation_state_signature(after, after_components)
    target = _action_target_cell(action, before)
    target_relation = _component_target_relation(before, target, before_components)
    relation_keys = _candidate_relation_keys(family, target_relation)
    relation_key = relation_keys[0][0] if relation_keys else f"{family}:no_target"
    fallback_relation = relation_keys[1][0] if len(relation_keys) > 1 else relation_key
    action_template = _component_relation_action_template(action, family, relation_key, target_relation)
    changed_mask = before != after
    changed_pixels = int(np.count_nonzero(changed_mask))
    matches = _match_components(before_components, after_components)
    matched_before = set(matches)
    matched_after = set(matches.values())
    appeared = [item for idx, item in enumerate(after_components) if idx not in matched_after]
    disappeared = [item for idx, item in enumerate(before_components) if idx not in matched_before]
    moved: list[dict[str, Any]] = []
    for before_idx, after_idx in matches.items():
        old = before_components[before_idx]
        new = after_components[after_idx]
        dy = float(new.centroid[0] - old.centroid[0])
        dx = float(new.centroid[1] - old.centroid[1])
        distance = math.sqrt(dy * dy + dx * dx)
        if distance >= 0.60:
            moved.append(
                {
                    "value": int(old.value),
                    "area": int(old.area),
                    "from": [round(float(old.centroid[0]), 3), round(float(old.centroid[1]), 3)],
                    "to": [round(float(new.centroid[0]), 3), round(float(new.centroid[1]), 3)],
                    "delta": [round(dy, 3), round(dx, 3)],
                    "distance": round(distance, 3),
                }
            )
    transforms = _component_transforms(before_components, after_components, matched_before, matched_after)
    if changed_pixels == 0:
        mechanism = "blocked_or_no_effect" if no_effect or family in {"move", "object", "contact"} else "stable"
    elif moved:
        mechanism = "component_movement"
    elif appeared and not disappeared:
        mechanism = "component_appearance"
    elif disappeared and not appeared:
        mechanism = "component_disappearance"
    elif transforms:
        mechanism = "component_color_transform"
    elif len(before_components) != len(after_components):
        mechanism = "component_split_merge"
    else:
        mechanism = "component_visual_transform"
    affected = moved[0]["value"] if moved else None
    if affected is None and appeared:
        affected = appeared[0].value
    if affected is None and disappeared:
        affected = disappeared[0].value
    if affected is None and transforms:
        affected = int(transforms[0].get("after_value", transforms[0].get("before_value", 0)))
    if affected is None:
        affected = int(target_relation.get("value", 0))
    return {
        "step_index": int(step_index),
        "action": str(action),
        "action_family": family,
        "mechanism": mechanism,
        "changed_pixels": changed_pixels,
        "component_count_before": len(before_components),
        "component_count_after": len(after_components),
        "before_state_signature": before_state_signature,
        "after_state_signature": after_state_signature,
        "before_relation_signature": before_relation_signature,
        "after_relation_signature": after_relation_signature,
        "action_template": action_template,
        "moved_components": moved[:4],
        "appeared_components": [item.compact() for item in appeared[:4]],
        "disappeared_components": [item.compact() for item in disappeared[:4]],
        "transformed_components": transforms[:4],
        "target_relation": target_relation,
        "relation_key": relation_key,
        "fallback_relation": fallback_relation,
        "target_value": int(target_relation.get("value", 0)),
        "component_value": int(affected or 0),
        "event_linked": bool(event_hit),
        "score_delta": float(score_delta),
    }


def _background_value(frame: np.ndarray) -> int:
    values, counts = np.unique(np.asarray(frame, dtype=np.int64), return_counts=True)
    if len(values) == 0:
        return 0
    return int(values[int(np.argmax(counts))])


def _frame_components(frame: np.ndarray) -> list[_FrameComponent]:
    arr = np.asarray(frame, dtype=np.int64)
    if arr.ndim != 2:
        return []
    background = _background_value(arr)
    seen = np.zeros(arr.shape, dtype=bool)
    components: list[_FrameComponent] = []
    height, width = arr.shape
    component_id = 0
    for y0 in range(height):
        for x0 in range(width):
            if seen[y0, x0] or int(arr[y0, x0]) == background:
                continue
            value = int(arr[y0, x0])
            stack = [(y0, x0)]
            seen[y0, x0] = True
            cells: list[tuple[int, int]] = []
            while stack:
                y, x = stack.pop()
                cells.append((int(y), int(x)))
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if (
                        0 <= ny < height
                        and 0 <= nx < width
                        and not seen[ny, nx]
                        and int(arr[ny, nx]) == value
                    ):
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            cells_tuple = tuple(sorted(cells))
            ys = [cell[0] for cell in cells_tuple]
            xs = [cell[1] for cell in cells_tuple]
            components.append(
                _FrameComponent(
                    component_id=component_id,
                    value=value,
                    cells=cells_tuple,
                    bbox=(min(ys), min(xs), max(ys), max(xs)),
                    centroid=(float(sum(ys) / len(ys)), float(sum(xs) / len(xs))),
                )
            )
            component_id += 1
    components.sort(key=lambda item: (-item.area, item.value, item.component_id))
    return components


def _component_state_signature(frame: np.ndarray, components: list[_FrameComponent] | None = None) -> str:
    arr = np.asarray(frame, dtype=np.int64)
    if arr.ndim != 2 or arr.size == 0:
        return ""
    if components is None:
        components = _frame_components(arr)
    compact_components = []
    for component in sorted(components, key=lambda item: (item.value, item.area, item.bbox, item.cells)):
        compact_components.append(
            {
                "value": int(component.value),
                "area": int(component.area),
                "bbox": [int(item) for item in component.bbox],
                "cells": [[int(y), int(x)] for y, x in component.cells],
            }
        )
    return stable_hash(
        {
            "shape": [int(arr.shape[0]), int(arr.shape[1])],
            "background": _background_value(arr),
            "components": compact_components,
        }
    )


def _component_relation_state_signature(frame: np.ndarray, components: list[_FrameComponent] | None = None) -> str:
    arr = np.asarray(frame, dtype=np.int64)
    if arr.ndim != 2 or arr.size == 0:
        return ""
    if components is None:
        components = _frame_components(arr)
    tokens: list[str] = [
        f"shape:{int(arr.shape[0])}:{int(arr.shape[1])}",
        f"background:{_background_value(arr)}",
        f"component_count:{_area_bucket(len(components))}",
    ]
    descriptor_counts: dict[tuple[int, int], int] = {}
    area_counts: dict[int, int] = {}
    for component in components:
        area_bucket = _area_bucket(component.area)
        descriptor = (int(component.value), int(area_bucket))
        descriptor_counts[descriptor] = descriptor_counts.get(descriptor, 0) + 1
        area_counts[area_bucket] = area_counts.get(area_bucket, 0) + 1
    for (value, area_bucket), count in sorted(descriptor_counts.items()):
        tokens.append(f"value:{value}:area:{area_bucket}:count:{count}")
    for area_bucket, count in sorted(area_counts.items()):
        tokens.append(f"area:{area_bucket}:count:{count}")
    paired_components = sorted(components, key=lambda item: (-item.area, item.value, item.component_id))[:12]
    for left_index, left in enumerate(paired_components):
        for right in paired_components[left_index + 1 :]:
            value_a, value_b = sorted((int(left.value), int(right.value)))
            area_a, area_b = sorted((_area_bucket(left.area), _area_bucket(right.area)))
            pair = f"{value_a}:{value_b}:area:{area_a}:{area_b}"
            if int(left.value) == int(right.value):
                tokens.append(f"same_value:{int(left.value)}:area:{area_a}:{area_b}")
            if abs(float(left.centroid[0] - right.centroid[0])) <= 0.5:
                tokens.append(f"align_row:{pair}")
            if abs(float(left.centroid[1] - right.centroid[1])) <= 0.5:
                tokens.append(f"align_col:{pair}")
            if _bbox_touches(left.bbox, right.bbox, margin=1):
                tokens.append(f"touch:{pair}")
            if _bbox_contains(left.bbox, right.bbox):
                tokens.append(f"contains:{int(left.value)}:{int(right.value)}:area:{_area_bucket(left.area)}:{_area_bucket(right.area)}")
            elif _bbox_contains(right.bbox, left.bbox):
                tokens.append(f"contains:{int(right.value)}:{int(left.value)}:area:{_area_bucket(right.area)}:{_area_bucket(left.area)}")
    return "rel:" + stable_hash({"tokens": sorted(tokens)})


def _match_components(before: list[_FrameComponent], after: list[_FrameComponent]) -> dict[int, int]:
    candidates: list[tuple[float, int, int]] = []
    for before_idx, old in enumerate(before):
        old_cells = set(old.cells)
        for after_idx, new in enumerate(after):
            if old.value != new.value:
                continue
            area_ratio = min(old.area, new.area) / max(max(old.area, new.area), 1)
            if area_ratio < 0.45:
                continue
            new_cells = set(new.cells)
            overlap = len(old_cells & new_cells) / max(len(old_cells | new_cells), 1)
            distance = math.dist(old.centroid, new.centroid)
            if overlap <= 0.0 and distance > max(3.0, math.sqrt(max(old.area, new.area)) + 1.5):
                continue
            score = overlap + max(0.0, 4.0 - distance) * 0.12 + area_ratio * 0.15
            candidates.append((score, before_idx, after_idx))
    candidates.sort(reverse=True)
    matched_before: set[int] = set()
    matched_after: set[int] = set()
    matches: dict[int, int] = {}
    for _score, before_idx, after_idx in candidates:
        if before_idx in matched_before or after_idx in matched_after:
            continue
        matches[before_idx] = after_idx
        matched_before.add(before_idx)
        matched_after.add(after_idx)
    return matches


def _component_transforms(
    before: list[_FrameComponent],
    after: list[_FrameComponent],
    matched_before: set[int],
    matched_after: set[int],
) -> list[dict[str, Any]]:
    transforms: list[dict[str, Any]] = []
    for before_idx, old in enumerate(before):
        if before_idx in matched_before:
            continue
        old_cells = set(old.cells)
        for after_idx, new in enumerate(after):
            if after_idx in matched_after or old.value == new.value:
                continue
            new_cells = set(new.cells)
            overlap = len(old_cells & new_cells) / max(min(len(old_cells), len(new_cells)), 1)
            if overlap <= 0.0 and not _bbox_touches(old.bbox, new.bbox, margin=1):
                continue
            transforms.append(
                {
                    "before_value": int(old.value),
                    "after_value": int(new.value),
                    "before_area": int(old.area),
                    "after_area": int(new.area),
                    "overlap": round(float(overlap), 3),
                    "before_bbox": [int(item) for item in old.bbox],
                    "after_bbox": [int(item) for item in new.bbox],
                }
            )
            break
    return transforms


def _bbox_touches(a: tuple[int, int, int, int], b: tuple[int, int, int, int], *, margin: int = 0) -> bool:
    ay0, ax0, ay1, ax1 = a
    by0, bx0, by1, bx1 = b
    return not (ay1 + margin < by0 or by1 + margin < ay0 or ax1 + margin < bx0 or bx1 + margin < ax0)


def _bbox_contains(outer: tuple[int, int, int, int], inner: tuple[int, int, int, int]) -> bool:
    oy0, ox0, oy1, ox1 = outer
    iy0, ix0, iy1, ix1 = inner
    return oy0 <= iy0 and ox0 <= ix0 and oy1 >= iy1 and ox1 >= ix1


def _action_target_cell(action: str, frame: np.ndarray) -> tuple[int, int] | None:
    return _click_cell(action, frame.shape)


def _component_target_relation(
    frame: np.ndarray,
    target: tuple[int, int] | None,
    components: list[_FrameComponent] | None = None,
) -> dict[str, Any]:
    components = _frame_components(frame) if components is None else components
    if target is None:
        return {"kind": "no_target", "value": 0, "area_bucket": 0, "component_count": len(components)}
    y, x = target
    if not (0 <= y < frame.shape[0] and 0 <= x < frame.shape[1]):
        return {"kind": "out_of_bounds", "value": 0, "area_bucket": 0, "component_count": len(components)}
    for component in components:
        if (y, x) in component.cells:
            return {
                "kind": "on_component",
                "value": int(component.value),
                "area_bucket": _area_bucket(component.area),
                "component_id": int(component.component_id),
                "component_count": len(components),
            }
    nearest: tuple[_FrameComponent, float] | None = None
    for component in components:
        distance = math.dist((float(y), float(x)), component.centroid)
        if nearest is None or distance < nearest[1]:
            nearest = (component, distance)
    if nearest is not None and nearest[1] <= 1.75:
        component = nearest[0]
        return {
            "kind": "adjacent_component",
            "value": int(component.value),
            "area_bucket": _area_bucket(component.area),
            "component_id": int(component.component_id),
            "component_count": len(components),
        }
    return {
        "kind": "background",
        "value": int(frame[y, x]),
        "area_bucket": 0,
        "cell": [int(y), int(x)],
        "component_count": len(components),
    }


def _area_bucket(area: int) -> int:
    if area <= 1:
        return 1
    if area <= 4:
        return 4
    if area <= 16:
        return 16
    return 64


def _candidate_relation_keys(family: str, relation: dict[str, Any]) -> list[tuple[str, float]]:
    kind = str(relation.get("kind", "no_target"))
    value = int(relation.get("value", 0))
    area_bucket = int(relation.get("area_bucket", 0))
    keys: list[tuple[str, float]] = []
    if kind == "no_target":
        keys.append((f"{family}:no_target", 1.0))
        keys.append(("*:no_target", 0.35))
        return keys
    if kind == "background":
        cell = relation.get("cell", [])
        if isinstance(cell, list) and len(cell) >= 2:
            keys.append((f"{family}:background:cell:{int(cell[0])}:{int(cell[1])}", 1.0))
        keys.append((f"{family}:background", 0.18))
        keys.append(("*:background", 0.08))
        return keys
    if value:
        keys.append((f"{family}:{kind}:value:{value}:area:{area_bucket}", 1.0))
        keys.append((f"{family}:{kind}:value:{value}", 0.74))
        keys.append((f"*:{kind}:value:{value}", 0.52))
    keys.append((f"{family}:{kind}", 0.45))
    keys.append((f"*:{kind}", 0.28))
    return keys


def _component_relation_action_template(
    action: str,
    family: str,
    relation_key: str,
    target_relation: dict[str, Any],
) -> str:
    family = str(family)
    kind = str(target_relation.get("kind", ""))
    if family == "contact":
        if kind == "background":
            return "relation:contact:background"
        if relation_key:
            return f"relation:{relation_key}"
    if family == "move":
        return f"action:{str(action)}"
    if action:
        return f"action:{str(action)}"
    if relation_key:
        return f"relation:{relation_key}"
    return ""


def _resolve_component_relation_action_template(
    template: str,
    legal_actions: list[str] | tuple[str, ...],
    frame: np.ndarray | None,
    components: list[_FrameComponent] | None = None,
) -> str | None:
    template = str(template)
    legal = [str(action) for action in legal_actions]
    if not template:
        return None
    if template.startswith("action:"):
        action = template[len("action:") :]
        return action if action in set(legal) else None
    if not template.startswith("relation:"):
        return template if template in set(legal) else None
    if frame is None or not frame.size:
        return None
    expected_relation = template[len("relation:") :]
    if components is None:
        components = _frame_components(frame)
    for action in legal:
        target_relation = _component_target_relation(frame, _action_target_cell(action, frame), components)
        current_keys = {key for key, _ in _candidate_relation_keys(_action_family(action), target_relation)}
        if expected_relation in current_keys:
            return action
    return None


def _component_prediction_key(relation_key: str, mechanism: str) -> str:
    return f"{str(relation_key)}=>{str(mechanism)}"


def _component_hypothesis_value(hypothesis: dict[str, Any]) -> float:
    mechanism = str(hypothesis.get("mechanism", ""))
    changed_pixels = int(hypothesis.get("changed_pixels", 0))
    value = 0.0
    if bool(hypothesis.get("event_linked")):
        value += 1.12 + min(max(float(hypothesis.get("score_delta", 0.0)), 0.0), 2.0)
    if changed_pixels:
        value += 0.16 + min(float(changed_pixels), 32.0) * 0.007
    if mechanism in {
        "component_movement",
        "component_appearance",
        "component_disappearance",
        "component_color_transform",
        "component_split_merge",
    }:
        value += 0.10
    if mechanism == "blocked_or_no_effect":
        value -= 0.46
    return float(value)


def _sequence_component_expectation(hypothesis: dict[str, Any]) -> dict[str, Any]:
    if not hypothesis:
        return {}
    return {
        "action": str(hypothesis.get("action", "")),
        "before_state_signature": str(hypothesis.get("before_state_signature", "")),
        "after_state_signature": str(hypothesis.get("after_state_signature", "")),
        "relation_key": str(hypothesis.get("relation_key", "")),
        "fallback_relation": str(hypothesis.get("fallback_relation", "")),
        "mechanism": str(hypothesis.get("mechanism", "")),
        "target_value": int(hypothesis.get("target_value", 0) or hypothesis.get("component_value", 0) or 0),
        "changed_expected": int(hypothesis.get("changed_pixels", 0)) > 0,
    }


def _relation_sequence_component_expectation(hypothesis: dict[str, Any]) -> dict[str, Any]:
    if not hypothesis:
        return {}
    return {
        "action": str(hypothesis.get("action", "")),
        "action_template": str(hypothesis.get("action_template", "")),
        "before_relation_signature": str(hypothesis.get("before_relation_signature", "")),
        "after_relation_signature": str(hypothesis.get("after_relation_signature", "")),
        "relation_key": str(hypothesis.get("relation_key", "")),
        "fallback_relation": str(hypothesis.get("fallback_relation", "")),
        "mechanism": str(hypothesis.get("mechanism", "")),
        "target_value": int(hypothesis.get("target_value", 0) or hypothesis.get("component_value", 0) or 0),
        "changed_expected": int(hypothesis.get("changed_pixels", 0)) > 0,
    }


def _component_expectation_matches(expected: dict[str, Any], actual: dict[str, Any] | None) -> bool:
    if not expected:
        return True
    if actual is None:
        return not bool(expected.get("changed_expected", False))
    expected_before = str(expected.get("before_state_signature", ""))
    actual_before = str(actual.get("before_state_signature", ""))
    if expected_before and actual_before and expected_before != actual_before:
        return False
    expected_after = str(expected.get("after_state_signature", ""))
    actual_after = str(actual.get("after_state_signature", ""))
    if expected_after:
        return expected_after == actual_after
    expected_relation_before = str(expected.get("before_relation_signature", ""))
    actual_relation_before = str(actual.get("before_relation_signature", ""))
    if expected_relation_before and actual_relation_before and expected_relation_before != actual_relation_before:
        return False
    expected_relation_after = str(expected.get("after_relation_signature", ""))
    actual_relation_after = str(actual.get("after_relation_signature", ""))
    if expected_relation_after:
        return expected_relation_after == actual_relation_after
    if bool(expected.get("changed_expected", False)) and int(actual.get("changed_pixels", 0)) == 0:
        return False
    expected_keys = {str(expected.get("relation_key", "")), str(expected.get("fallback_relation", ""))}
    expected_keys.discard("")
    if expected_keys and str(actual.get("relation_key", "")) in expected_keys:
        return True
    if expected_keys and str(actual.get("fallback_relation", "")) in expected_keys:
        return True
    expected_value = int(expected.get("target_value", 0) or 0)
    actual_value = int(actual.get("target_value", 0) or actual.get("component_value", 0) or 0)
    if expected_value and expected_value == actual_value and int(actual.get("changed_pixels", 0)) > 0:
        return True
    expected_mechanism = str(expected.get("mechanism", ""))
    if expected_mechanism and expected_mechanism == str(actual.get("mechanism", "")):
        return True
    return False


def _step_after_frame(record: AttemptRecord, index: int) -> np.ndarray | None:
    step = record.steps[index]
    direct = getattr(step, "next_frame", None)
    if direct is not None:
        return np.asarray(direct, dtype=np.int64)
    if index + 1 < len(record.steps):
        return np.asarray(record.steps[index + 1].frame, dtype=np.int64)
    return None


def _frame_diff_hypothesis(
    *,
    before: np.ndarray,
    after: np.ndarray,
    action: str,
    step_index: int,
    event_hit: bool,
    score_delta: float,
    no_effect: bool,
) -> dict[str, Any] | None:
    if before.shape != after.shape:
        size_y = min(before.shape[0], after.shape[0])
        size_x = min(before.shape[1], after.shape[1])
        before = before[:size_y, :size_x]
        after = after[:size_y, :size_x]
    family = _action_family(action)
    click_cell = _click_cell(action, before.shape)
    changed_mask = before != after
    changed_pixels = int(np.count_nonzero(changed_mask))
    if changed_pixels == 0 and click_cell is None and family not in {"move", "object", "contact"} and not no_effect:
        return None

    bbox: list[int] | None = None
    centroid: list[float] | None = None
    changed_colors: list[int] = []
    appeared_pixels = 0
    removed_pixels = 0
    replacement_pixels = 0
    mechanism = "blocked_or_no_effect"
    contact = False
    movement_colors: list[int] = []
    if changed_pixels:
        coords = np.argwhere(changed_mask)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        bbox = [int(y0), int(x0), int(y1), int(x1)]
        centroid_values = coords.astype(np.float32).mean(axis=0)
        centroid = [round(float(centroid_values[0]), 3), round(float(centroid_values[1]), 3)]
        before_changed = before[changed_mask]
        after_changed = after[changed_mask]
        changed_colors = sorted(
            int(value)
            for value in set(before_changed.reshape(-1).tolist() + after_changed.reshape(-1).tolist())
            if int(value) != 0
        )[:12]
        appeared_pixels = int(np.count_nonzero((before == 0) & (after != 0)))
        removed_pixels = int(np.count_nonzero((before != 0) & (after == 0)))
        replacement_pixels = int(np.count_nonzero((before != 0) & (after != 0) & changed_mask))
        movement_colors = _movement_colors(before, after)
        if movement_colors:
            mechanism = "movement"
        elif appeared_pixels and not removed_pixels:
            mechanism = "spawn"
        elif removed_pixels and not appeared_pixels:
            mechanism = "removal"
        elif replacement_pixels:
            mechanism = "toggle_or_transform"
        else:
            mechanism = "visual_transform"
        contact = bool(click_cell is not None and bbox is not None and _cell_touches_bbox(click_cell, bbox))
    elif no_effect or family in {"move", "object", "contact"}:
        centroid = [float(click_cell[0]), float(click_cell[1])] if click_cell is not None else None

    return {
        "step_index": int(step_index),
        "action": str(action),
        "action_family": family,
        "mechanism": mechanism,
        "changed_pixels": changed_pixels,
        "appeared_pixels": appeared_pixels,
        "removed_pixels": removed_pixels,
        "replacement_pixels": replacement_pixels,
        "bbox": bbox,
        "centroid": centroid,
        "changed_colors": changed_colors,
        "movement_colors": movement_colors,
        "click_cell": list(click_cell) if click_cell is not None else None,
        "click_contacts_change": contact,
        "event_linked": bool(event_hit),
        "score_delta": float(score_delta),
    }


def _movement_colors(before: np.ndarray, after: np.ndarray) -> list[int]:
    colors = sorted(int(value) for value in set(before.reshape(-1).tolist() + after.reshape(-1).tolist()) if int(value) != 0)
    moved: list[int] = []
    for color in colors:
        before_coords = np.argwhere(before == color)
        after_coords = np.argwhere(after == color)
        if before_coords.size == 0 or after_coords.size == 0 or before_coords.shape[0] != after_coords.shape[0]:
            continue
        before_centroid = before_coords.astype(np.float32).mean(axis=0)
        after_centroid = after_coords.astype(np.float32).mean(axis=0)
        if float(np.linalg.norm(after_centroid - before_centroid)) >= 0.75:
            moved.append(color)
    return moved[:8]


def _hypothesis_region(hypothesis: dict[str, Any]) -> tuple[int, int] | None:
    centroid = hypothesis.get("centroid")
    if isinstance(centroid, list) and len(centroid) >= 2:
        return (int(round(float(centroid[0]))), int(round(float(centroid[1]))))
    click_cell = hypothesis.get("click_cell")
    if isinstance(click_cell, list) and len(click_cell) >= 2:
        return (int(click_cell[0]), int(click_cell[1]))
    return None


def _object_hypothesis_value(hypothesis: dict[str, Any]) -> float:
    mechanism = str(hypothesis.get("mechanism", ""))
    changed_pixels = int(hypothesis.get("changed_pixels", 0))
    value = 0.0
    if bool(hypothesis.get("event_linked")):
        value += 1.05 + min(max(float(hypothesis.get("score_delta", 0.0)), 0.0), 2.0)
    if changed_pixels:
        value += 0.18 + min(float(changed_pixels), 16.0) * 0.012
    if mechanism in {"movement", "spawn", "removal", "toggle_or_transform"}:
        value += 0.06
    if bool(hypothesis.get("click_contacts_change")):
        value += 0.08
    if mechanism == "blocked_or_no_effect":
        value -= 0.42
    return float(value)


def _public_frame(observation: Any) -> np.ndarray | None:
    try:
        return observation_frame(observation).astype(np.int64)
    except Exception:
        try:
            return np.asarray(observation.grid, dtype=np.int64)
        except Exception:
            return None


def _click_cell(action: str, shape: tuple[int, ...] | list[int] = (8, 8)) -> tuple[int, int] | None:
    lowered = str(action).lower()
    if not lowered.startswith("click:"):
        return None
    parts = lowered.split(":")
    if len(parts) < 3:
        return None
    try:
        x_coord = int(float(parts[1]))
        y_coord = int(float(parts[2]))
    except ValueError:
        return None
    height = int(shape[0]) if len(shape) >= 1 else 8
    width = int(shape[1]) if len(shape) >= 2 else 8
    y_cell = min(max(int(y_coord * height // 64), 0), max(height - 1, 0))
    x_cell = min(max(int(x_coord * width // 64), 0), max(width - 1, 0))
    return (y_cell, x_cell)


def _cell_touches_bbox(cell: tuple[int, int], bbox: list[int], margin: int = 1) -> bool:
    y, x = cell
    y0, x0, y1, x1 = [int(value) for value in bbox]
    return y0 - margin <= y <= y1 + margin and x0 - margin <= x <= x1 + margin


def _neighbor_regions(region: tuple[int, int]) -> list[tuple[tuple[int, int], float]]:
    y, x = region
    neighbors = [((y, x), 1.0)]
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny = y + dy
        nx = x + dx
        if 0 <= ny < 8 and 0 <= nx < 8:
            neighbors.append(((ny, nx), 0.45))
    return neighbors


def _nonzero_regions(frame: np.ndarray) -> list[tuple[int, int]]:
    coords = np.argwhere(frame != 0)
    return [(int(y), int(x)) for y, x in coords[:16]]


def _centered_distribution_delta(plan: dict[str, float]) -> dict[str, float]:
    if not plan:
        return {}
    mean = sum(float(value) for value in plan.values()) / len(plan)
    return {action: float(value - mean) for action, value in sorted(plan.items())}
