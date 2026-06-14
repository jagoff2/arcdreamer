from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from .attempt_buffer import AttemptRecord, observation_frame, public_observation_key_payload, stable_hash, tensors_from_attempts
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
PLANNER_DIAGNOSTIC_KINDS = ("component_chain", "relation_chain", "relation_delta", "relation_delta_sequence")
PLANNER_DIAGNOSTIC_EVENTS = (
    "activations",
    "score_hits",
    "legal_resolutions",
    "visible_changes",
    "useful_events",
    "contradiction_aborts",
    "stale_penalties",
)
RELATION_SCORING_ACTION_LIMIT = 96
EXPERIMENT_CLASS_RETIRE_FAILURES = 2


@dataclass(frozen=True)
class TransitionUsefulness:
    visible_effect: bool
    useful_effect: bool
    nuisance_effect: bool
    no_effect: bool
    progress_effect: bool
    terminal_win: bool
    controllability_effect: bool
    reachable_state_class_effect: bool
    reasons: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "visible_effect": self.visible_effect,
            "useful_effect": self.useful_effect,
            "nuisance_effect": self.nuisance_effect,
            "no_effect": self.no_effect,
            "progress_effect": self.progress_effect,
            "terminal_win": self.terminal_win,
            "controllability_effect": self.controllability_effect,
            "reachable_state_class_effect": self.reachable_state_class_effect,
            "reasons": list(self.reasons),
        }


@dataclass(frozen=True)
class Experiment:
    hypothesis_ids: list[str]
    action: str
    predicted_outcomes: dict[str, str]
    useful_if: list[str]
    max_repeats: int
    retire_if: str
    action_class: str
    coordinate_equiv_class: str
    abstract_state_class: str
    repeat_count: int = 0
    retired: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _empty_planner_activation_stats() -> dict[str, dict[str, int]]:
    return {
        kind: {event: 0 for event in PLANNER_DIAGNOSTIC_EVENTS}
        for kind in PLANNER_DIAGNOSTIC_KINDS
    }


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
    jepa_planner_state: dict[str, Any] = field(default_factory=dict)
    sidecar_action_priors: dict[str, float] = field(default_factory=dict)
    action_distribution_delta: dict[str, float] = field(default_factory=dict)
    causal_substrate_active: bool = False

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


class JEPAAttemptMemory:
    def __init__(
        self,
        *,
        use_jepa_tokens: bool,
        device: torch.device | str = "cpu",
        jepa_token_mode: str = "normal",
    ) -> None:
        self.use_jepa_tokens = bool(use_jepa_tokens)
        self.device = torch.device(device)
        self.jepa_token_mode = str(jepa_token_mode)
        self.entries: list[AttemptMemoryEntry] = []
        self.transition_counts: dict[tuple[str, str], int] = {}
        self.transition_values: dict[tuple[str, str], float] = {}
        self.transition_effects: dict[tuple[str, str], int] = {}
        self.transition_failures: dict[tuple[str, str], int] = {}
        self.transition_events: dict[tuple[str, str], int] = {}
        self.transition_useful: dict[tuple[str, str], int] = {}
        self.transition_nuisance: dict[tuple[str, str], int] = {}
        self.reachable_state_classes: set[str] = set()
        self.state_seen_actions: dict[str, set[str]] = {}
        self.action_counts: dict[str, int] = {}
        self.action_values: dict[str, float] = {}
        self.action_failures: dict[str, int] = {}
        self.action_events: dict[str, int] = {}
        self.action_useful: dict[str, int] = {}
        self.action_nuisance: dict[str, int] = {}
        self.family_counts: dict[str, int] = {}
        self.family_values: dict[str, float] = {}
        self.family_useful: dict[str, int] = {}
        self.family_nuisance: dict[str, int] = {}
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
        self.component_relation_delta_counts: dict[str, int] = {}
        self.component_relation_delta_values: dict[str, float] = {}
        self.component_relation_delta_goal_values: dict[str, float] = {}
        self.component_relation_delta_failures: dict[str, int] = {}
        self.component_relation_delta_contradictions: dict[str, int] = {}
        self.component_relation_delta_tokens_by_scope: dict[str, set[str]] = {}
        self.sequence_contradictions = 0
        self.sequence_candidates: list[dict[str, Any]] = []
        self.active_sequence: list[str] = []
        self.active_sequence_expectations: list[dict[str, Any]] = []
        self.sequence_cursor = 0
        self.active_sequence_source = ""
        self.best_event_prefix: list[str] = []
        self.best_event_prefix_value = 0.0
        self.positive_prefix_completed = False
        self.discovery_phase = 0
        self.phase_action_counts: dict[tuple[int, str], int] = {}
        self.phase_family_counts: dict[tuple[int, str], int] = {}
        self.phase_state_seen_actions: dict[tuple[int, str], set[str]] = {}
        self.experiment_counts: dict[str, int] = {}
        self.experiment_failures: dict[str, int] = {}
        self.experiment_retired: set[str] = set()
        self.experiment_class_failures: dict[str, int] = {}
        self.experiment_class_retired: set[str] = set()
        self.pending_experiment: dict[str, Any] | None = None
        self.experiment_history: list[dict[str, Any]] = []
        self.last_transition_nontrivial = False
        self.last_transition_usefulness: TransitionUsefulness | None = None
        self.undo_probe_state_classes: set[str] = set()
        self.planner_activation_stats = _empty_planner_activation_stats()

    def reset(self) -> None:
        self.entries.clear()
        self.transition_counts.clear()
        self.transition_values.clear()
        self.transition_effects.clear()
        self.transition_failures.clear()
        self.transition_events.clear()
        self.transition_useful.clear()
        self.transition_nuisance.clear()
        self.reachable_state_classes.clear()
        self.state_seen_actions.clear()
        self.action_counts.clear()
        self.action_values.clear()
        self.action_failures.clear()
        self.action_events.clear()
        self.action_useful.clear()
        self.action_nuisance.clear()
        self.family_counts.clear()
        self.family_values.clear()
        self.family_useful.clear()
        self.family_nuisance.clear()
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
        self.component_relation_delta_counts.clear()
        self.component_relation_delta_values.clear()
        self.component_relation_delta_goal_values.clear()
        self.component_relation_delta_failures.clear()
        self.component_relation_delta_contradictions.clear()
        self.component_relation_delta_tokens_by_scope.clear()
        self.sequence_contradictions = 0
        self.sequence_candidates.clear()
        self.active_sequence.clear()
        self.active_sequence_expectations.clear()
        self.sequence_cursor = 0
        self.active_sequence_source = ""
        self.best_event_prefix.clear()
        self.best_event_prefix_value = 0.0
        self.positive_prefix_completed = False
        self.discovery_phase = 0
        self.phase_action_counts.clear()
        self.phase_family_counts.clear()
        self.phase_state_seen_actions.clear()
        self.experiment_counts.clear()
        self.experiment_failures.clear()
        self.experiment_retired.clear()
        self.experiment_class_failures.clear()
        self.experiment_class_retired.clear()
        self.pending_experiment = None
        self.experiment_history.clear()
        self.last_transition_nontrivial = False
        self.last_transition_usefulness = None
        self.undo_probe_state_classes.clear()
        self.planner_activation_stats = _empty_planner_activation_stats()

    def ingest_attempt(self, record: AttemptRecord, model: VideoJEPA | None = None) -> AttemptMemoryEntry:
        failed: dict[str, int] = {}
        effects: dict[str, int] = {}
        useful_actions: dict[str, int] = {}
        events: dict[str, float] = {}
        action_counts: dict[str, int] = {}
        transition_outcomes: list[dict[str, Any]] = []
        no_effect = 0
        visible_effect = 0
        useful_effect = 0
        nuisance_effect = 0
        controllability_effect = 0
        reachable_state_class_effect = 0
        positive = 0
        attempt_family_counts: dict[str, int] = {}
        for index, step in enumerate(record.steps):
            action = str(step.action)
            family = _action_family(action)
            prior_action_count = action_counts.get(action, 0)
            prior_family_count = attempt_family_counts.get(family, 0)
            action_counts[action] = action_counts.get(action, 0) + 1
            attempt_family_counts[family] = attempt_family_counts.get(family, 0) + 1
            event_hit = bool(POSITIVE_EVENTS.intersection(step.event_delta)) or float(step.score_delta) > 0.0
            changed = step.obs_hash != step.next_obs_hash
            no_effect_event = _is_no_effect_event(step.event_delta)
            after_frame = _step_after_frame(record, index)
            next_state_class = _abstract_frame_state_class(after_frame) if after_frame is not None else ""
            state_class_seen = bool(next_state_class and next_state_class in self.reachable_state_classes)
            public_controllability = _public_controllability_evidence(step.frame, after_frame, action)
            label = _classify_transition_usefulness(
                visible_effect=changed,
                events=step.event_delta,
                score_delta=float(step.score_delta),
                terminal=bool(step.terminal),
                no_effect_event=no_effect_event,
                invalid_action=bool(step.invalid_action),
                prior_action_count=prior_action_count,
                prior_family_count=prior_family_count,
                prior_useful_action_count=int(useful_actions.get(action, 0)),
                public_controllability_evidence=public_controllability,
                public_reachable_state_evidence=bool(public_controllability and not state_class_seen),
                state_class_seen=state_class_seen,
            )
            outcome_value = _transition_credit(
                useful=label,
                no_effect_event=no_effect_event,
                invalid_action=bool(step.invalid_action),
                score_delta=float(step.score_delta),
                terminal=bool(step.terminal),
            )
            if next_state_class:
                self.reachable_state_classes.add(next_state_class)
            transition_outcomes.append(
                {
                    "edge": (str(step.obs_hash), action),
                    "action": action,
                    "value": outcome_value,
                    "event_hit": event_hit,
                    "changed": label.visible_effect,
                    "visible_effect": label.visible_effect,
                    "useful_effect": label.useful_effect,
                    "nuisance_effect": label.nuisance_effect,
                    "no_effect": label.no_effect,
                    "progress_effect": label.progress_effect,
                    "controllability_effect": label.controllability_effect,
                    "reachable_state_class_effect": label.reachable_state_class_effect,
                    "usefulness_reasons": list(label.reasons),
                }
            )
            if event_hit:
                positive += 1
                events[action] = events.get(action, 0.0) + max(1.0, float(step.score_delta))
            if label.no_effect:
                no_effect += 1
                failed[action] = failed.get(action, 0) + 1
            if label.visible_effect:
                visible_effect += 1
            if label.useful_effect:
                useful_effect += 1
                useful_actions[action] = useful_actions.get(action, 0) + 1
            if label.nuisance_effect:
                nuisance_effect += 1
            if label.controllability_effect:
                controllability_effect += 1
            if label.reachable_state_class_effect:
                reachable_state_class_effect += 1
            if label.visible_effect and not label.no_effect:
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
                output = model(
                    batch["frames"],
                    batch["action_ids"],
                    batch["legal_counts"],
                    batch.get("valid"),
                    batch.get("action_features"),
                )
                if self.jepa_token_mode == "shuffled":
                    output = _shuffled_jepa_output(output)
                token_mean_tensor = _attempt_token_mean_from_output(output, batch.get("valid"))
                token_mean = [round(float(item), 6) for item in token_mean_tensor.detach().cpu().reshape(-1).tolist()[:24]]
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
            "useful_effect_rate": float(useful_effect / total),
            "nuisance_effect_rate": float(nuisance_effect / total),
            "controllability_effect_rate": float(controllability_effect / total),
            "reachable_state_class_effect_rate": float(reachable_state_class_effect / total),
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
        for action, count in useful_actions.items():
            if positive > 0 and action not in failed and action not in events:
                next_plan[action] = next_plan.get(action, 0.0) + min(float(count), 4.0) * 0.05
        if positive == 0:
            for action, fraction in repeated.items():
                next_plan[action] = next_plan.get(action, 0.0) - min(float(fraction), 1.0) * 0.45
        delayed_credit_edges = self._update_transition_graph(transition_outcomes)
        delayed_region_links = self._update_object_memory(object_hypotheses, transition_outcomes)
        delayed_component_links = self._update_component_memory(component_hypotheses, transition_outcomes)
        delayed_component_chain_links = self._update_component_chain_graph(component_hypotheses, transition_outcomes)
        delayed_relation_chain_links = self._update_component_relation_chain_graph(component_hypotheses, transition_outcomes)
        delayed_relation_delta_links = self._update_component_relation_delta_memory(
            component_hypotheses,
            transition_outcomes,
        )
        sequence_candidates_added = self._update_sequence_candidates(
            record,
            transition_outcomes,
            object_hypotheses,
            component_hypotheses,
        )
        self._update_best_event_prefix(record, transition_outcomes)
        graph_summary = self.transition_graph_summary()
        object_summary = self.object_memory_summary()
        planner_stats = self.planner_activation_summary()
        causal.update(
            {
                "transition_graph_edges": float(graph_summary["observed_edges"]),
                "transition_graph_positive_edges": float(graph_summary["positive_edges"]),
                "transition_graph_no_effect_edges": float(graph_summary["no_effect_edges"]),
                "transition_graph_useful_edges": float(graph_summary["useful_edges"]),
                "transition_graph_nuisance_edges": float(graph_summary["nuisance_edges"]),
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
                "component_relation_delta_count": float(object_summary["component_relation_delta_count"]),
                "component_relation_delta_goal_count": float(object_summary["component_relation_delta_goal_count"]),
                "component_relation_delta_contradictions": float(object_summary["component_relation_delta_contradictions"]),
                "component_relation_delta_delayed_links": float(delayed_relation_delta_links),
                "sequence_contradictions": float(object_summary["sequence_contradictions"]),
                "sequence_candidates_added": float(sequence_candidates_added),
                "sequence_candidate_count": float(object_summary["sequence_candidate_count"]),
            }
        )
        for planner_name, stats in planner_stats.items():
            for event_name, value in stats.items():
                causal[f"{planner_name}_{event_name}"] = float(value)
        if jepa_active:
            jepa_planner_biases: dict[str, float] = {}
            for action, evidence in jepa_action_evidence.items():
                effect_centered = float(evidence.get("effect", 0.0) - effect_mean)
                surprise_centered = float(evidence.get("surprise", 0.0) - surprise_mean)
                count_scale = math.log1p(float(evidence.get("count", 0.0)))
                latent_rule_bias = (0.90 * effect_centered + 0.40 * surprise_centered) * max(count_scale, 1.0)
                if len(jepa_action_evidence) == 1:
                    quality_signal = float(evidence.get("effect", 0.0)) - 0.05 * float(evidence.get("surprise", 0.0))
                    latent_rule_bias += 0.35 * math.tanh(quality_signal * 8.0)
                if action in events:
                    latent_rule_bias += 0.25 * min(float(events[action]), 3.0)
                if action in failed and effect_centered < 0.0:
                    latent_rule_bias -= 0.20 * min(float(failed[action]), 4.0)
                if positive == 0 and action not in events:
                    latent_rule_bias = min(latent_rule_bias, 0.0)
                jepa_planner_biases[action] = float(latent_rule_bias)
            causal["jepa_proposer_prior_l1"] = float(sum(abs(value) for value in jepa_planner_biases.values()))
            causal["jepa_proposer_actions"] = float(sum(1 for value in jepa_planner_biases.values() if abs(value) > 1.0e-9))
            causal["jepa_planner_bias_l1"] = 0.0
            causal["jepa_planner_consumed_actions"] = 0.0
        else:
            jepa_planner_biases = {}
            causal["jepa_proposer_prior_l1"] = 0.0
            causal["jepa_proposer_actions"] = 0.0
            causal["jepa_planner_bias_l1"] = 0.0
            causal["jepa_planner_consumed_actions"] = 0.0
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
            jepa_planner_state={
                "mode": self.jepa_token_mode,
                "planner_role": "proposer_read_only",
                "consumed_by_planner": False,
                "bias_l1": 0.0,
                "proposal_l1": float(sum(abs(value) for value in jepa_planner_biases.values())),
                "action_biases": {},
                "action_priors": {action: round(float(value), 6) for action, value in sorted(jepa_planner_biases.items())},
                "evidence": {
                    action: {key: round(float(value), 6) for key, value in sorted(values.items())}
                    for action, values in sorted(jepa_action_evidence.items())
                },
            },
            sidecar_action_priors={action: float(value) for action, value in jepa_planner_biases.items()},
            action_distribution_delta=action_distribution_delta,
            causal_substrate_active=jepa_active,
        )
        self.entries.append(entry)
        return entry

    def _planner_kind(self, source: str | None) -> str | None:
        source_text = str(source or "")
        if "relation_delta_sequence" in source_text:
            return "relation_delta_sequence"
        if source_text == "positive_public_relation_delta_sequence":
            return "relation_delta_sequence"
        if "relation_delta" in source_text:
            return "relation_delta"
        if "predicted_component_transition" in source_text:
            return "component_chain"
        if "component_relation_goal_chain" in source_text:
            return "relation_chain"
        if "component_transition_goal_chain" in source_text:
            return "component_chain"
        return None

    def _record_planner_event(self, source: str | None, event: str, amount: int = 1) -> None:
        kind = self._planner_kind(source)
        if kind is None or event not in PLANNER_DIAGNOSTIC_EVENTS:
            return
        self.planner_activation_stats.setdefault(kind, {name: 0 for name in PLANNER_DIAGNOSTIC_EVENTS})
        self.planner_activation_stats[kind][event] = int(self.planner_activation_stats[kind].get(event, 0)) + int(amount)

    def planner_activation_summary(self) -> dict[str, dict[str, int]]:
        return {
            kind: {event: int(self.planner_activation_stats.get(kind, {}).get(event, 0)) for event in PLANNER_DIAGNOSTIC_EVENTS}
            for kind in PLANNER_DIAGNOSTIC_KINDS
        }

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
        has_goal_evidence = self._has_public_goal_evidence()
        obs_key = _observation_key(observation)
        frame = _public_frame(observation)
        component_relations: dict[str, dict[str, Any]] = {}
        components: list[_FrameComponent] | None = None
        relation_actions = _bounded_relation_scoring_actions(legal_actions)
        relation_action_set = set(relation_actions)
        if frame is not None and frame.size:
            components = _frame_components(frame)
            for action in relation_actions:
                component_relations[action] = _component_target_relation(
                    frame,
                    _action_target_cell(action, frame),
                    components,
                )
        scores = self.plan_scores(legal_actions)
        seen_here = self.state_seen_actions.get(obs_key, set()) if obs_key else set()
        phase_seen_here = self.phase_state_seen_actions.get((self.discovery_phase, obs_key), set()) if obs_key else set()
        stuck = self._recent_stuck()
        useful_family_total = int(sum(max(int(value), 0) for value in self.family_useful.values()))
        for action in legal_actions:
            family = _action_family(action)
            edge = (obs_key, action)
            edge_count = self.transition_counts.get(edge, 0)
            if obs_key and edge_count:
                edge_mean = self.transition_values.get(edge, 0.0) / max(edge_count, 1)
                scores[action] += 0.65 * edge_mean
                scores[action] -= 0.22 * min(float(self.transition_failures.get(edge, 0)), 6.0)
                if self.transition_events.get(edge, 0):
                    scores[action] += 0.12
            else:
                family_count = self.family_counts.get(family, 0)
                if family_count:
                    family_mean = self.family_values.get(family, 0.0) / max(family_count, 1)
                    if has_goal_evidence or family_mean > 0.0:
                        scores[action] += 0.20 * family_mean
                    elif family_count >= 24:
                        evidence_scale = min(max((float(family_count) - 24.0) / 24.0, 0.0), 1.0)
                        scores[action] += 0.70 * evidence_scale * family_mean
                if stuck:
                    scores[action] += 0.10
                if obs_key and seen_here and action not in seen_here:
                    scores[action] += 0.08
            if self.discovery_phase > 0 and family != "wait":
                phase_action_count = self.phase_action_counts.get((self.discovery_phase, action), 0)
                phase_family_count = self.phase_family_counts.get((self.discovery_phase, family), 0)
                scores[action] += 0.18 / math.sqrt(float(phase_action_count + 1))
                scores[action] += 0.12 / math.sqrt(float(phase_family_count + 1))
                if obs_key and phase_seen_here and action not in phase_seen_here:
                    scores[action] += 0.10
            action_count = self.action_counts.get(action, 0)
            if action_count:
                action_mean = self.action_values.get(action, 0.0) / max(action_count, 1)
                scores[action] += 0.45 * action_mean
                if not has_goal_evidence and action_mean <= 0.0:
                    scores[action] -= 0.030 * min(float(action_count), 24.0)
                if self.action_events.get(action, 0):
                    scores[action] += 0.10
            action_useful_count = int(self.action_useful.get(action, 0))
            family_useful_count = int(self.family_useful.get(family, 0))
            if action_useful_count > 0:
                scores[action] += 0.34 / math.sqrt(float(action_count + 1))
            elif family in {"move", "object"} and family_useful_count > 0:
                scores[action] += 0.18 / math.sqrt(float(action_count + 1))
            if useful_family_total > 0 and family == "contact" and family_useful_count <= 0:
                family_count = int(self.family_counts.get(family, 0))
                nuisance_rate = float(self.family_nuisance.get(family, 0)) / max(float(family_count), 1.0)
                scores[action] -= 0.08 + 0.18 * min(max(nuisance_rate, 0.0), 1.0)
            if action in relation_action_set:
                scores[action] += self._object_region_score(action, frame)
                target_relation = component_relations.get(action)
                scores[action] += self._component_relation_score(action, frame, target_relation)
                component_prediction_score = self._component_transition_prediction_score(action, frame, target_relation)
                scores[action] += component_prediction_score
                if abs(component_prediction_score) > 1.0e-9:
                    self._record_planner_event("predicted_component_transition_planner", "score_hits")
                    if component_prediction_score > 0.0:
                        self._record_planner_event("predicted_component_transition_planner", "activations")
                        self._record_planner_event("predicted_component_transition_planner", "legal_resolutions")
                relation_delta_score = self._component_relation_delta_score(action, frame, target_relation)
                scores[action] += relation_delta_score
                if abs(relation_delta_score) > 1.0e-9:
                    self._record_planner_event("component_relation_delta_event_miner", "score_hits")
                    if relation_delta_score > 0.0:
                        self._record_planner_event("component_relation_delta_event_miner", "activations")
                        self._record_planner_event("component_relation_delta_event_miner", "legal_resolutions")
        chain_plan = self._component_chain_plan(frame, legal_actions, components)
        relation_chain_plan = self._component_relation_chain_plan(frame, relation_actions, components)
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
        planned_action = self.sequence_plan_action(
            legal_actions,
            observation=observation,
            frame=frame,
            components=components,
        )
        if planned_action is not None:
            plan_support = self._sequence_plan_support(planned_action, frame, component_relations.get(planned_action))
            scores[planned_action] = scores.get(planned_action, 0.0) + (0.85 * plan_support)
        if not has_goal_evidence:
            capped_scores: dict[str, float] = {}
            for action, score in scores.items():
                family = _action_family(action)
                has_public_control = bool(
                    self.action_useful.get(action, 0) > 0
                    or (family in {"move", "object"} and self.family_useful.get(family, 0) > 0)
                )
                upper_bound = 0.45 if has_public_control else 0.0
                capped_scores[action] = min(float(score), upper_bound)
            scores = capped_scores
        lower_bound = -1.75 if not has_goal_evidence else -0.75
        return {action: float(max(min(score, 0.75), lower_bound)) for action, score in scores.items()}

    def _has_public_goal_evidence(self) -> bool:
        return bool(
            self.transition_events
            or self.component_goal_relations
            or self.component_prediction_goal_values
            or self.component_chain_goal_values
            or self.component_goal_state_values
            or self.component_relation_chain_goal_values
            or self.component_relation_goal_state_values
            or self.component_relation_delta_goal_values
            or any(entry.event_candidates for entry in self.entries)
        )

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

    def bridge_public_support_weight(self, action: str) -> float:
        action = str(action)
        if self.action_events.get(action, 0) > 0:
            return 1.0
        count = int(self.action_counts.get(action, 0))
        family = _action_family(action)
        if count <= 0:
            family_count = int(self.family_counts.get(family, 0))
            if family_count >= 24:
                family_mean = self.family_values.get(family, 0.0) / max(family_count, 1)
                nuisance_rate = float(self.family_nuisance.get(family, 0)) / max(float(family_count), 1.0)
                if family_mean <= 0.0 or nuisance_rate >= 0.25:
                    evidence_scale = min(max((float(family_count) - 24.0) / 24.0, 0.0), 1.0)
                    nuisance_penalty = 0.45 * evidence_scale * nuisance_rate
                    return float(max(0.02, 0.18 + 0.37 * (1.0 - evidence_scale) + 0.70 * evidence_scale * family_mean - nuisance_penalty))
            return 0.18
        mean_value = self.action_values.get(action, 0.0) / max(count, 1)
        nuisance = min(float(self.action_nuisance.get(action, 0)), 8.0)
        if mean_value > 0.0 and nuisance <= 0.0:
            return 0.85
        failures = min(float(self.action_failures.get(action, 0)), 8.0)
        return float(max(0.0, 0.35 - 0.075 * failures - 0.070 * nuisance))

    def discovery_action_counts(self, action: str) -> tuple[int, int]:
        action = str(action)
        family = _action_family(action)
        return (
            int(self.phase_action_counts.get((self.discovery_phase, action), 0)),
            int(self.phase_family_counts.get((self.discovery_phase, family), 0)),
        )

    def discovery_exploration_candidates(
        self,
        legal_actions: list[str] | tuple[str, ...],
        *,
        observation: Any | None = None,
        scores: dict[str, float] | None = None,
        limit: int = 16,
    ) -> list[str]:
        if self.discovery_phase <= 0 and not self.positive_prefix_completed:
            return []
        legal = [str(action) for action in legal_actions if _action_family(str(action)) != "wait"]
        if not legal:
            return [str(action) for action in legal_actions]
        obs_key = _observation_key(observation) if observation is not None else ""
        unseen_here = [
            action
            for action in legal
            if obs_key
            and action not in self.phase_state_seen_actions.get((self.discovery_phase, obs_key), set())
        ]
        pool = unseen_here or legal
        min_family_count = min((self.discovery_action_counts(action)[1] for action in pool), default=0)
        pool = [action for action in pool if self.discovery_action_counts(action)[1] == min_family_count]
        min_action_count = min((self.discovery_action_counts(action)[0] for action in pool), default=0)
        pool = [action for action in pool if self.discovery_action_counts(action)[0] == min_action_count] or pool
        if scores:
            index = {action: idx for idx, action in enumerate(legal)}
            pool = sorted(pool, key=lambda action: (float(scores.get(action, 0.0)), -index.get(action, 0)), reverse=True)
        return pool[: max(int(limit), 1)]

    def select_experiment(
        self,
        observation: Any,
        legal_actions: list[str] | tuple[str, ...],
        *,
        scores: dict[str, float] | None = None,
    ) -> Experiment | None:
        legal = [str(action) for action in legal_actions]
        if not legal:
            return None
        score_map = scores or {}

        def score_for_action(action: str) -> float:
            try:
                value = float(score_map.get(action, 0.0))
            except (TypeError, ValueError):
                return 0.0
            return value if math.isfinite(value) else 0.0

        frame = _public_frame(observation)
        fallback_abstract_state = _abstract_frame_state_class(frame)
        if not fallback_abstract_state:
            fallback_abstract_state = _observation_key(observation)
        candidates: list[tuple[tuple[float, float, float, float, int, float, int], Experiment]] = []
        contact_by_coordinate_class: dict[
            str,
            tuple[tuple[float, int], tuple[float, float, float, float, int, float, int], Experiment],
        ] = {}
        useful_family_total = int(sum(max(int(value), 0) for value in self.family_useful.values()))
        for legal_index, action in enumerate(legal):
            experiment = self._build_experiment(
                action,
                frame,
                _experiment_abstract_state_class(action, frame, fallback_abstract_state),
            )
            if experiment is None:
                continue
            key = self._experiment_key(experiment)
            if key in self.experiment_retired:
                continue
            class_key = self._experiment_class_key(experiment)
            if class_key in self.experiment_class_retired:
                continue
            repeat_count = int(self.experiment_counts.get(key, 0))
            if repeat_count >= int(experiment.max_repeats):
                continue
            family = _action_family(action)
            phase_family_count = int(self.phase_family_counts.get((self.discovery_phase, family), 0))
            phase_action_count = int(self.phase_action_counts.get((self.discovery_phase, action), 0))
            priority = _experiment_action_priority(action)
            remaining = int(experiment.max_repeats) - repeat_count
            action_score = score_for_action(action)
            family_useful = int(self.family_useful.get(family, 0))
            family_count = int(self.family_counts.get(family, 0))
            family_mean = self.family_values.get(family, 0.0) / max(family_count, 1)
            family_nuisance_rate = float(self.family_nuisance.get(family, 0)) / max(float(family_count), 1.0)
            if useful_family_total <= 0:
                useful_family_priority = 0.0
            elif family_useful > 0:
                useful_family_priority = 0.0
            elif family_count >= 8 and (family_mean < -0.12 or family_nuisance_rate >= 0.25):
                useful_family_priority = 2.0
            else:
                useful_family_priority = 1.0
            candidate_key = (
                useful_family_priority,
                float(phase_family_count),
                float(priority),
                float(phase_action_count),
                -remaining,
                -action_score,
                legal_index,
            )
            if family == "contact":
                representative_key = (-action_score, legal_index)
                current = contact_by_coordinate_class.get(experiment.coordinate_equiv_class)
                if current is None or representative_key < current[0]:
                    contact_by_coordinate_class[experiment.coordinate_equiv_class] = (
                        representative_key,
                        candidate_key,
                        experiment,
                    )
                continue
            candidates.append((candidate_key, experiment))
        candidates.extend((candidate_key, experiment) for _, candidate_key, experiment in contact_by_coordinate_class.values())
        if not candidates:
            return None
        candidates.sort(key=lambda item: item[0])
        experiment = candidates[0][1]
        repeat_count = int(self.experiment_counts.get(self._experiment_key(experiment), 0))
        return Experiment(
            hypothesis_ids=list(experiment.hypothesis_ids),
            action=str(experiment.action),
            predicted_outcomes=dict(experiment.predicted_outcomes),
            useful_if=list(experiment.useful_if),
            max_repeats=int(experiment.max_repeats),
            retire_if=str(experiment.retire_if),
            action_class=str(experiment.action_class),
            coordinate_equiv_class=str(experiment.coordinate_equiv_class),
            abstract_state_class=str(experiment.abstract_state_class),
            repeat_count=repeat_count,
            retired=False,
        )

    def experiment_for_action(
        self,
        observation: Any,
        action: str,
        *,
        respect_limits: bool = True,
    ) -> Experiment | None:
        frame = _public_frame(observation)
        fallback_abstract_state = _abstract_frame_state_class(frame)
        if not fallback_abstract_state:
            fallback_abstract_state = _observation_key(observation)
        experiment = self._build_experiment(
            str(action),
            frame,
            _experiment_abstract_state_class(str(action), frame, fallback_abstract_state),
        )
        if experiment is None:
            return None
        key = self._experiment_key(experiment)
        class_key = self._experiment_class_key(experiment)
        repeat_count = int(self.experiment_counts.get(key, 0))
        if respect_limits:
            if key in self.experiment_retired:
                return None
            if class_key in self.experiment_class_retired:
                return None
            if repeat_count >= int(experiment.max_repeats):
                return None
        return Experiment(
            hypothesis_ids=list(experiment.hypothesis_ids),
            action=str(experiment.action),
            predicted_outcomes=dict(experiment.predicted_outcomes),
            useful_if=list(experiment.useful_if),
            max_repeats=int(experiment.max_repeats),
            retire_if=str(experiment.retire_if),
            action_class=str(experiment.action_class),
            coordinate_equiv_class=str(experiment.coordinate_equiv_class),
            abstract_state_class=str(experiment.abstract_state_class),
            repeat_count=repeat_count,
            retired=bool(key in self.experiment_retired),
        )

    def begin_experiment(self, experiment: Experiment) -> None:
        payload = experiment.to_dict()
        key = self._experiment_key(experiment)
        class_key = self._experiment_class_key(experiment)
        self.experiment_counts[key] = int(self.experiment_counts.get(key, 0)) + 1
        payload["experiment_key"] = key
        payload["experiment_class_key"] = class_key
        payload["started_count"] = int(self.experiment_counts[key])
        payload["class_failure_count"] = int(self.experiment_class_failures.get(class_key, 0))
        self.pending_experiment = payload

    def experiment_protocol_summary(self) -> dict[str, Any]:
        return {
            "schema": "arc_explicit_experiment_protocol_v1",
            "pending_experiment": dict(self.pending_experiment or {}),
            "experiments_started": int(sum(self.experiment_counts.values())),
            "retired_experiment_count": int(len(self.experiment_retired)),
            "retired_experiment_class_count": int(len(self.experiment_class_retired)),
            "retired_experiment_classes": sorted(self.experiment_class_retired)[-16:],
            "recent_results": list(self.experiment_history[-8:]),
        }

    def _build_experiment(self, action: str, frame: np.ndarray | None, abstract_state: str) -> Experiment | None:
        family = _action_family(action)
        if family == "wait" and str(action) != "7":
            return None
        if str(action) == "7":
            if not self.last_transition_nontrivial or abstract_state in self.undo_probe_state_classes:
                return None
        action_class = _experiment_action_class(action)
        coordinate_class = _experiment_coordinate_equiv_class(action, frame)
        hypothesis_ids = _experiment_hypothesis_ids(action, frame)
        if not hypothesis_ids:
            return None
        predicted_outcomes = _experiment_predicted_outcomes(action, family, coordinate_class)
        if len(set(predicted_outcomes.values())) < 2:
            return None
        useful_if = [
            "score_delta > 0",
            "level_changed",
            "posterior_entropy_drops",
            "new_controllable_object_found",
            "new_reachable_state_class_found",
        ]
        return Experiment(
            hypothesis_ids=hypothesis_ids,
            action=str(action),
            predicted_outcomes=predicted_outcomes,
            useful_if=useful_if,
            max_repeats=1,
            retire_if="nuisance_or_noop_without_entropy_drop",
            action_class=action_class,
            coordinate_equiv_class=coordinate_class,
            abstract_state_class=str(abstract_state),
        )

    def _experiment_key(self, experiment: Experiment | dict[str, Any]) -> str:
        getter = experiment.get if isinstance(experiment, dict) else lambda name, default=None: getattr(experiment, name, default)
        return "|".join(
            [
                str(self.discovery_phase),
                str(getter("abstract_state_class", "")),
                str(getter("action_class", "")),
                str(getter("coordinate_equiv_class", "")),
            ]
        )

    def _experiment_class_key(self, experiment: Experiment | dict[str, Any]) -> str:
        getter = experiment.get if isinstance(experiment, dict) else lambda name, default=None: getattr(experiment, name, default)
        return "|".join(
            [
                str(self.discovery_phase),
                str(getter("action_class", "")),
                str(getter("coordinate_equiv_class", "")),
            ]
        )

    def public_no_effect_suppresses_action(self, action: str) -> bool:
        if self._has_public_goal_evidence():
            return False
        family = _action_family(str(action))
        family_count = int(self.family_counts.get(family, 0))
        useful_family_total = int(sum(max(int(value), 0) for value in self.family_useful.values()))
        family_useful = int(self.family_useful.get(family, 0))
        min_family_count = 48
        mean_threshold = -0.35
        nuisance_threshold = 0.35
        if useful_family_total > family_useful and family == "contact":
            min_family_count = 24
            mean_threshold = -0.22
            nuisance_threshold = 0.24
        if family_count < min_family_count:
            return False
        family_mean = self.family_values.get(family, 0.0) / max(family_count, 1)
        nuisance_rate = float(self.family_nuisance.get(family, 0)) / max(float(family_count), 1.0)
        return bool(family_mean <= mean_threshold or nuisance_rate >= nuisance_threshold)

    def transition_graph_summary(self) -> dict[str, Any]:
        return {
            "observed_edges": len(self.transition_counts),
            "observed_states": len(self.state_seen_actions),
            "positive_edges": sum(1 for value in self.transition_events.values() if value > 0),
            "effect_edges": sum(1 for value in self.transition_effects.values() if value > 0),
            "useful_edges": sum(1 for value in self.transition_useful.values() if value > 0),
            "nuisance_edges": sum(1 for value in self.transition_nuisance.values() if value > 0),
            "no_effect_edges": sum(1 for value in self.transition_failures.values() if value > 0),
            "observed_actions": len(self.action_counts),
            "negative_actions": sum(
                1
                for action, count in self.action_counts.items()
                if self.action_values.get(action, 0.0) / max(count, 1) < 0.0
            ),
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
            "component_relation_delta_count": len(self.component_relation_delta_counts),
            "component_relation_delta_goal_count": len(self.component_relation_delta_goal_values),
            "component_relation_delta_failures": sum(self.component_relation_delta_failures.values()),
            "component_relation_delta_contradictions": sum(self.component_relation_delta_contradictions.values()),
            "sequence_contradictions": self.sequence_contradictions,
            "sequence_candidate_count": len(self.sequence_candidates),
            "relation_delta_sequence_candidate_count": sum(
                1
                for item in self.sequence_candidates
                if str(item.get("source", "")) == "positive_public_relation_delta_sequence"
            ),
            "active_sequence_length": len(self.active_sequence),
            "active_sequence_remaining": active_remaining,
            "active_sequence_source": self.active_sequence_source,
            "best_event_prefix_length": len(self.best_event_prefix),
            "best_event_prefix_value": float(self.best_event_prefix_value),
            "positive_prefix_completed": bool(self.positive_prefix_completed),
            "discovery_phase": int(self.discovery_phase),
            "phase_action_count": sum(
                count for (phase, _action), count in self.phase_action_counts.items() if phase == self.discovery_phase
            ),
            "phase_family_counts": {
                family: int(count)
                for (phase, family), count in sorted(self.phase_family_counts.items())
                if phase == self.discovery_phase
            },
            "planner_activation": self.planner_activation_summary(),
        }

    def start_attempt(self) -> None:
        self.active_sequence = []
        self.active_sequence_expectations = []
        self.sequence_cursor = 0
        self.active_sequence_source = ""
        self.positive_prefix_completed = False
        self.discovery_phase = 0
        self.phase_action_counts.clear()
        self.phase_family_counts.clear()
        self.phase_state_seen_actions.clear()
        self.experiment_counts.clear()
        self.experiment_failures.clear()
        self.experiment_retired.clear()
        self.experiment_class_failures.clear()
        self.experiment_class_retired.clear()
        self.pending_experiment = None
        self.experiment_history.clear()
        self.last_transition_nontrivial = False
        self.last_transition_usefulness = None
        self.undo_probe_state_classes.clear()
        if self.best_event_prefix:
            self.active_sequence = list(self.best_event_prefix[:128])
            self.active_sequence_expectations = [{} for _ in self.active_sequence]
            self.active_sequence_source = "positive_public_control_prefix"
            self._record_planner_event(self.active_sequence_source, "activations")
            return
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
                self._record_planner_event(self.active_sequence_source, "activations")
                return

    def sequence_plan_action(
        self,
        legal_actions: list[str] | tuple[str, ...],
        observation: Any | None = None,
        frame: np.ndarray | None = None,
        components: list[_FrameComponent] | None = None,
    ) -> str | None:
        if not self.active_sequence:
            return None
        legal = {str(action) for action in legal_actions}
        if frame is None and observation is not None:
            frame = _public_frame(observation)
        if frame is not None and frame.size and components is None:
            components = _frame_components(frame)
        while self.sequence_cursor < len(self.active_sequence):
            action = self.active_sequence[self.sequence_cursor]
            expected = (
                self.active_sequence_expectations[self.sequence_cursor]
                if self.sequence_cursor < len(self.active_sequence_expectations)
                else {}
            )
            resolved_action = _resolve_sequence_expected_action(
                expected,
                legal_actions,
                frame,
                components,
            )
            if resolved_action is not None:
                self._record_planner_event(self.active_sequence_source, "legal_resolutions")
                return resolved_action
            if action in legal:
                self._record_planner_event(self.active_sequence_source, "legal_resolutions")
                return action
            self.sequence_cursor += 1
        return None

    def advance_sequence(self, chosen_action: str) -> None:
        if not self.active_sequence or self.sequence_cursor >= len(self.active_sequence):
            return
        if str(chosen_action) == self.active_sequence[self.sequence_cursor]:
            self.sequence_cursor += 1
            if self.active_sequence_source == "positive_public_control_prefix" and self.sequence_cursor >= len(self.active_sequence):
                self.positive_prefix_completed = True

    def observe_live_transition(self, before_observation: Any, action: str, result: Any) -> TransitionUsefulness | None:
        action = str(action)
        before = _public_frame(before_observation)
        after = _public_frame(getattr(result, "observation", None))
        events = [str(item) for item in getattr(result, "info", {}).get("events", [])]
        level_completed = any(event == "level_completed" for event in events)
        event_hit = bool(POSITIVE_EVENTS.intersection(events)) or float(getattr(result, "reward", 0.0)) > 0.0
        no_effect_event = _is_no_effect_event(events)
        changed = bool(before is not None and after is not None and before.shape == after.shape and np.any(before != after))
        obs_key = _observation_key(before_observation)
        family = _action_family(action)
        state_class_seen = (
            _abstract_frame_state_class(after) in self.reachable_state_classes
            if after is not None and after.size
            else True
        )
        public_controllability = _public_controllability_evidence(before, after, action)
        label = _classify_transition_usefulness(
            visible_effect=changed,
            events=events,
            score_delta=float(getattr(result, "reward", 0.0)),
            terminal=bool(getattr(result, "terminated", False) or getattr(result, "truncated", False)),
            no_effect_event=no_effect_event,
            invalid_action=any("invalid" in event.lower() for event in events),
            prior_action_count=int(self.action_counts.get(action, 0)),
            prior_family_count=int(self.family_counts.get(family, 0)),
            prior_useful_action_count=int(self.action_events.get(action, 0)),
            public_controllability_evidence=public_controllability,
            public_reachable_state_evidence=bool(public_controllability and not state_class_seen),
            state_class_seen=state_class_seen,
        )
        self.last_transition_usefulness = label
        if after is not None and after.size:
            state_class = _abstract_frame_state_class(after)
            if state_class:
                self.reachable_state_classes.add(state_class)
        self._record_phase_action(obs_key, action)
        if obs_key:
            outcome = {
                "edge": (obs_key, action),
                "action": action,
                "value": _transition_credit(
                    useful=label,
                    no_effect_event=no_effect_event,
                    invalid_action=any("invalid" in event.lower() for event in events),
                    score_delta=float(getattr(result, "reward", 0.0)),
                    terminal=bool(getattr(result, "terminated", False) or getattr(result, "truncated", False)),
                ),
                "event_hit": event_hit,
                "score_delta": float(getattr(result, "reward", 0.0)),
                "changed": label.visible_effect,
                "visible_effect": label.visible_effect,
                "useful_effect": label.useful_effect,
                "nuisance_effect": label.nuisance_effect,
                "no_effect": label.no_effect,
                "progress_effect": label.progress_effect,
                "controllability_effect": label.controllability_effect,
                "reachable_state_class_effect": label.reachable_state_class_effect,
                "usefulness_reasons": list(label.reasons),
            }
            self._update_transition_graph([outcome])
            self._update_live_public_memory(before, after, action, outcome)
        self._complete_pending_experiment(action, label)
        if not self.active_sequence or self.sequence_cursor >= len(self.active_sequence):
            if level_completed:
                self._enter_next_discovery_phase()
            return label
        source = self.active_sequence_source
        expected = (
            self.active_sequence_expectations[self.sequence_cursor]
            if self.sequence_cursor < len(self.active_sequence_expectations)
            else {}
        )
        legal_actions = list(getattr(before_observation, "available_actions", ()))
        expected_action = self.active_sequence[self.sequence_cursor]
        resolved_expected_action = _resolve_sequence_expected_action(
            expected,
            legal_actions,
            before,
        )
        if action != expected_action and (resolved_expected_action is None or action != resolved_expected_action):
            return label
        if before is not None and after is not None and bool(np.any(before != after)):
            self._record_planner_event(source, "visible_changes")
        if label.useful_effect:
            self._record_planner_event(source, "useful_events")
        if expected and before is not None and after is not None:
            actual = _component_hypothesis_from_frames(
                before=before,
                after=after,
                action=action,
                step_index=-1,
                event_hit=event_hit,
                score_delta=float(getattr(result, "reward", 0.0)),
                no_effect=label.no_effect,
                useful_effect=label.useful_effect,
                nuisance_effect=label.nuisance_effect,
                progress_effect=label.progress_effect,
                controllability_effect=label.controllability_effect,
                reachable_state_class_effect=label.reachable_state_class_effect,
            )
            if not _component_expectation_matches(expected, actual) and not event_hit:
                self.sequence_contradictions += 1
                self._record_planner_event(source, "contradiction_aborts")
                self._penalize_component_prediction(expected)
                self._record_planner_event(source, "stale_penalties")
                self.active_sequence = []
                self.active_sequence_expectations = []
                self.sequence_cursor = 0
                self.active_sequence_source = "aborted_by_public_component_contradiction"
                return label
        self.sequence_cursor += 1
        if source == "positive_public_control_prefix" and self.sequence_cursor >= len(self.active_sequence):
            self.positive_prefix_completed = True
        if level_completed:
            self._enter_next_discovery_phase()
        return label

    def _update_live_public_memory(
        self,
        before: np.ndarray | None,
        after: np.ndarray | None,
        action: str,
        outcome: dict[str, Any],
    ) -> None:
        if before is None or after is None:
            return
        try:
            before_arr = np.asarray(before, dtype=np.int64)
            after_arr = np.asarray(after, dtype=np.int64)
        except Exception:
            return
        if before_arr.ndim != 2 or after_arr.ndim != 2 or before_arr.size == 0 or after_arr.size == 0:
            return
        label_flags = {
            "event_hit": bool(outcome.get("event_hit", False)),
            "score_delta": float(outcome.get("score_delta", outcome.get("value", 0.0))),
            "no_effect": bool(outcome.get("no_effect", False)),
            "useful_effect": bool(outcome.get("useful_effect", False)),
            "nuisance_effect": bool(outcome.get("nuisance_effect", False)),
            "progress_effect": bool(outcome.get("progress_effect", False)),
            "controllability_effect": bool(outcome.get("controllability_effect", False)),
            "reachable_state_class_effect": bool(outcome.get("reachable_state_class_effect", False)),
        }
        object_hypothesis = _frame_diff_hypothesis(
            before=before_arr,
            after=after_arr,
            action=str(action),
            step_index=0,
            **label_flags,
        )
        component_hypothesis = _component_hypothesis_from_frames(
            before=before_arr,
            after=after_arr,
            action=str(action),
            step_index=0,
            **label_flags,
        )
        live_outcomes = [dict(outcome)]
        if object_hypothesis is not None:
            self._update_object_memory([object_hypothesis], live_outcomes)
        if component_hypothesis is not None:
            component_hypotheses = [component_hypothesis]
            self._update_component_memory(component_hypotheses, live_outcomes)
            self._update_component_chain_graph(component_hypotheses, live_outcomes)
            self._update_component_relation_chain_graph(component_hypotheses, live_outcomes)
            self._update_component_relation_delta_memory(component_hypotheses, live_outcomes)

    def _record_phase_action(self, obs_key: str, action: str) -> None:
        phase = int(self.discovery_phase)
        self.phase_action_counts[(phase, action)] = self.phase_action_counts.get((phase, action), 0) + 1
        family = _action_family(action)
        self.phase_family_counts[(phase, family)] = self.phase_family_counts.get((phase, family), 0) + 1
        if obs_key:
            key = (phase, obs_key)
            self.phase_state_seen_actions.setdefault(key, set()).add(action)

    def _complete_pending_experiment(self, action: str, label: TransitionUsefulness) -> None:
        pending = dict(self.pending_experiment or {})
        self.pending_experiment = None
        if not pending or str(pending.get("action", "")) != str(action):
            self.last_transition_nontrivial = bool(label.visible_effect and not label.no_effect)
            return
        key = str(pending.get("experiment_key", ""))
        class_key = str(pending.get("experiment_class_key", ""))
        classification = _experiment_result_class(label)
        posthoc_useful = bool(label.useful_effect)
        posthoc_progress = bool(label.progress_effect or label.terminal_win)
        failed = classification in {"nuisance", "no_op", "loop"} and not posthoc_useful
        if failed and key:
            self.experiment_failures[key] = int(self.experiment_failures.get(key, 0)) + 1
            if int(self.experiment_failures[key]) >= int(pending.get("max_repeats", 1)):
                self.experiment_retired.add(key)
        if failed and class_key:
            self.experiment_class_failures[class_key] = int(self.experiment_class_failures.get(class_key, 0)) + 1
            if int(self.experiment_class_failures[class_key]) >= EXPERIMENT_CLASS_RETIRE_FAILURES:
                self.experiment_class_retired.add(class_key)
        if str(pending.get("action", "")) == "7" and posthoc_useful:
            self.undo_probe_state_classes.add(str(pending.get("abstract_state_class", "")))
        result = {
            "experiment_key": key,
            "experiment_class_key": class_key,
            "action": str(action),
            "predicted_outcomes": dict(pending.get("predicted_outcomes", {})),
            "classification": classification,
            "posthoc_useful": posthoc_useful,
            "posthoc_progress": posthoc_progress,
            "posthoc_entropy_drop": 0.0,
            "retired": bool(key in self.experiment_retired),
            "class_retired": bool(class_key in self.experiment_class_retired),
            "usefulness_reasons": list(label.reasons),
        }
        self.experiment_history.append(result)
        self.experiment_history = self.experiment_history[-64:]
        self.last_transition_nontrivial = bool(label.visible_effect and not label.no_effect)

    def _enter_next_discovery_phase(self) -> None:
        self.discovery_phase += 1
        self.active_sequence = []
        self.active_sequence_expectations = []
        self.sequence_cursor = 0
        self.active_sequence_source = "post_level_discovery"
        self.positive_prefix_completed = True
        self.pending_experiment = None

    def summary(self) -> dict[str, Any]:
        return {
            "entry_count": len(self.entries),
            "use_jepa_tokens": self.use_jepa_tokens,
            "jepa_token_mode": self.jepa_token_mode,
            "entries": [entry.to_dict() for entry in self.entries[-3:]],
            "emits_text": False,
            "direct_action_source": False,
            "causal_chain": [
                "attempt_video_action_history",
                "jepa_temporal_representation",
                "attempt_memory",
                "rule_causal_hypothesis_update",
                "sidecar_action_prior_proposals",
                "symbolic_causal_controller_approval",
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
                "component_relation_delta_event_miner",
                "relation_delta_sequence_planner",
                "targeted_next_attempt_experiment",
            ],
            "transition_graph": self.transition_graph_summary(),
            "object_memory": self.object_memory_summary(),
            "planner_activation": self.planner_activation_summary(),
            "experiment_protocol": self.experiment_protocol_summary(),
            "jepa_evidence_state": [entry.jepa_planner_state for entry in self.entries[-3:]],
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
            self.action_counts[action] = self.action_counts.get(action, 0) + 1
            self.action_values[action] = self.action_values.get(action, 0.0) + value
            self.family_counts[family] = self.family_counts.get(family, 0) + 1
            self.family_values[family] = self.family_values.get(family, 0.0) + value
            if outcome["event_hit"]:
                self.transition_events[edge] = self.transition_events.get(edge, 0) + 1
                self.action_events[action] = self.action_events.get(action, 0) + 1
            if outcome["changed"]:
                self.transition_effects[edge] = self.transition_effects.get(edge, 0) + 1
            if outcome.get("useful_effect", False):
                self.transition_useful[edge] = self.transition_useful.get(edge, 0) + 1
                self.action_useful[action] = self.action_useful.get(action, 0) + 1
                self.family_useful[family] = self.family_useful.get(family, 0) + 1
            if outcome.get("nuisance_effect", False):
                self.transition_nuisance[edge] = self.transition_nuisance.get(edge, 0) + 1
                self.action_nuisance[action] = self.action_nuisance.get(action, 0) + 1
                self.family_nuisance[family] = self.family_nuisance.get(family, 0) + 1
            if outcome["no_effect"]:
                self.transition_failures[edge] = self.transition_failures.get(edge, 0) + 1
                self.action_failures[action] = self.action_failures.get(action, 0) + 1

        event_indices = [index for index, outcome in enumerate(outcomes) if outcome["event_hit"]]
        for event_index in event_indices:
            for prior_index in range(max(0, event_index - 6), event_index):
                edge = outcomes[prior_index]["edge"]
                action = str(outcomes[prior_index]["action"])
                distance = event_index - prior_index
                credit = 0.42 / float(distance + 1)
                self.transition_values[edge] = self.transition_values.get(edge, 0.0) + credit
                self.action_values[action] = self.action_values.get(action, 0.0) + credit
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

    def _update_component_relation_delta_memory(
        self,
        hypotheses: list[dict[str, Any]],
        outcomes: list[dict[str, Any]],
    ) -> int:
        event_indices = [index for index, outcome in enumerate(outcomes) if outcome["event_hit"]]
        delayed_goal_tokens: set[str] = set()
        delayed_credit_by_index: dict[int, float] = {}
        for event_index in event_indices:
            for prior_index in range(max(0, event_index - 10), event_index):
                distance = event_index - prior_index
                delayed_credit_by_index[prior_index] = delayed_credit_by_index.get(prior_index, 0.0) + 0.40 / float(
                    distance + 1
                )
        for hypothesis in hypotheses:
            index = int(hypothesis.get("step_index", -1))
            outcome = outcomes[index] if 0 <= index < len(outcomes) else {}
            token_weights = _component_relation_delta_tokens(hypothesis)
            if not token_weights:
                continue
            value = (
                _component_hypothesis_value(hypothesis)
                + 0.28 * float(outcome.get("value", 0.0))
                + delayed_credit_by_index.get(index, 0.0)
            )
            event_linked = (
                bool(hypothesis.get("event_linked"))
                or index in delayed_credit_by_index
                or bool(outcome.get("event_hit", False))
            )
            nonproductive = str(hypothesis.get("mechanism", "")) in NON_PRODUCTIVE_COMPONENT_MECHANISMS or bool(
                outcome.get("no_effect", False)
            )
            for token, weight in token_weights:
                token = str(token)
                scope = _relation_delta_scope(token)
                if not token or not scope:
                    continue
                self.component_relation_delta_counts[token] = self.component_relation_delta_counts.get(token, 0) + 1
                self.component_relation_delta_values[token] = (
                    self.component_relation_delta_values.get(token, 0.0) + value * float(weight)
                )
                self.component_relation_delta_tokens_by_scope.setdefault(scope, set()).add(token)
                if event_linked:
                    goal_value = max(value * float(weight), 0.08 * float(weight))
                    self.component_relation_delta_goal_values[token] = (
                        self.component_relation_delta_goal_values.get(token, 0.0) + goal_value
                    )
                    delayed_goal_tokens.add(token)
                if nonproductive:
                    self.component_relation_delta_failures[token] = (
                        self.component_relation_delta_failures.get(token, 0) + 1
                    )
        return len(delayed_goal_tokens)

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
        existing = {_sequence_candidate_key(item) for item in self.sequence_candidates}
        for event_index, outcome in enumerate(outcomes):
            if not outcome["event_hit"]:
                continue
            start = max(0, event_index - 8)
            window_steps = record.steps[start : event_index + 1]
            actions = [str(step.action) for step in window_steps]
            if not actions:
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
            component_expectations: list[dict[str, Any]] = []
            action_templates: list[str] = []
            relation_delta_step_count = 0
            productive_delta_count = 0
            for index in range(start, event_index + 1):
                component_hypothesis = component_by_index.get(index, {})
                if component_hypothesis and component_hypothesis.get("relation_delta_tokens"):
                    expectation = _relation_delta_sequence_component_expectation(component_hypothesis)
                    relation_delta_step_count += 1
                    productive_delta_count += sum(
                        1
                        for token in expectation.get("relation_delta_tokens", [])
                        if _relation_delta_token_is_productive(str(token))
                    )
                else:
                    expectation = _sequence_component_expectation(component_hypothesis)
                component_expectations.append(expectation)
                action_template = str(expectation.get("action_template", ""))
                action_templates.append(action_template or str(expectation.get("action", "")))
            source = (
                "positive_public_relation_delta_sequence"
                if relation_delta_step_count > 0
                else "positive_public_event_window"
            )
            value = (
                1.0
                + max(float(record.steps[event_index].score_delta), 0.0)
                + 0.08 * len(linked)
                + 0.10 * len(linked_components)
                + 0.06 * relation_delta_step_count
                + 0.015 * min(float(productive_delta_count), 12.0)
            )
            candidate = {
                "source": source,
                "actions": actions,
                "action_templates": action_templates[: len(actions)],
                "value": float(value),
                "length": len(actions),
                "event_index": event_index,
                "linked_regions": [_hypothesis_region(item) for item in linked[:4]],
                "linked_mechanisms": [str(item.get("mechanism")) for item in linked[:4]],
                "linked_component_relations": [str(item.get("relation_key")) for item in linked_components[:6]],
                "relation_delta_steps": relation_delta_step_count,
                "component_expectations": component_expectations[: len(actions)],
            }
            key = _sequence_candidate_key(candidate)
            if key in existing:
                continue
            self.sequence_candidates.append(candidate)
            existing.add(key)
            added += 1
        self.sequence_candidates = sorted(
            self.sequence_candidates,
            key=lambda item: (float(item.get("value", 0.0)), int(item.get("length", 0))),
            reverse=True,
        )[:24]
        return added

    def _update_best_event_prefix(self, record: AttemptRecord, outcomes: list[dict[str, Any]]) -> None:
        cumulative_positive = 0.0
        for event_index, outcome in enumerate(outcomes):
            step = record.steps[event_index]
            if bool(outcome.get("event_hit", False)):
                cumulative_positive += max(1.0, float(step.score_delta))
            if cumulative_positive <= 0.0:
                continue
            prefix_actions = [str(item.action) for item in record.steps[: event_index + 1]]
            if not prefix_actions or any(_action_family(action) == "contact" for action in prefix_actions):
                continue
            shorter_existing = self.best_event_prefix and len(prefix_actions) < len(self.best_event_prefix)
            if cumulative_positive > self.best_event_prefix_value + 1.0e-6 or (
                abs(cumulative_positive - self.best_event_prefix_value) <= 1.0e-6 and shorter_existing
            ):
                self.best_event_prefix = prefix_actions[:128]
                self.best_event_prefix_value = float(cumulative_positive)

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

    def _component_relation_delta_score(
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
        scopes = _component_relation_delta_action_scopes(action, family, target_relation)
        if not scopes:
            return 0.0
        token_scope_weights: dict[str, float] = {}
        for scope, scope_weight in scopes.items():
            for token in self.component_relation_delta_tokens_by_scope.get(scope, set()):
                token_scope_weights[token] = max(token_scope_weights.get(token, 0.0), float(scope_weight))
        if not token_scope_weights:
            return 0.0
        productive_score = 0.0
        nonproductive_penalty = 0.0
        for token, scope_weight in token_scope_weights.items():
            count = self.component_relation_delta_counts.get(token, 0)
            if count <= 0:
                continue
            mean_value = self.component_relation_delta_values.get(token, 0.0) / max(count, 1)
            goal_value = self.component_relation_delta_goal_values.get(token, 0.0) / max(count, 1)
            failures = float(self.component_relation_delta_failures.get(token, 0))
            contradictions = float(self.component_relation_delta_contradictions.get(token, 0))
            confidence = min(1.0, math.log1p(float(count)) / math.log(5.0))
            reliability = max(
                0.0,
                1.0 - (0.40 * failures + contradictions) / max(float(count) + failures + contradictions, 1.0),
            )
            if _relation_delta_token_is_productive(token):
                candidate_score = scope_weight * confidence * reliability * (
                    0.18 * max(mean_value, 0.0) + 0.36 * max(goal_value, 0.0)
                )
                productive_score = max(productive_score, candidate_score)
            else:
                penalty_mean = max(-mean_value, 0.0) + 0.12 * max(1.0 - max(mean_value, 0.0), 0.0)
                nonproductive_penalty += scope_weight * confidence * penalty_mean * 0.22
        score = productive_score - min(nonproductive_penalty, 0.22)
        return float(max(min(score, 0.22), -0.24))

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
        if self.active_sequence_source == "positive_public_control_prefix" and remaining:
            return False
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
        self._record_planner_event(source, "activations")
        self._record_planner_event(source, "legal_resolutions")
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
        for token in expected.get("relation_delta_tokens", []) or []:
            token = str(token)
            if token:
                self.component_relation_delta_contradictions[token] = (
                    self.component_relation_delta_contradictions.get(token, 0) + 1
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


def _shuffled_jepa_output(output: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    shuffled: dict[str, torch.Tensor] = {}
    for key, value in output.items():
        if not torch.is_tensor(value):
            shuffled[key] = value
            continue
        if key in {"context_tokens", "frame_tokens"} and value.dim() >= 2 and value.shape[1] > 1:
            shuffled[key] = torch.roll(value, shifts=1, dims=1)
        elif key in {"predicted_future", "target_future", "loss_mask"} and value.dim() >= 2 and value.shape[1] > 1:
            shuffled[key] = torch.roll(value, shifts=1, dims=1)
        else:
            shuffled[key] = value
    return shuffled


def _attempt_token_mean_from_output(output: dict[str, torch.Tensor], valid: torch.Tensor | None = None) -> torch.Tensor:
    tokens = output["context_tokens"]
    if valid is None:
        return tokens.mean(dim=1)
    weights = valid.float().clamp(0.0, 1.0)
    return (tokens * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


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


def _is_terminal_win_event(events: list[str]) -> bool:
    positive_tokens = ("win", "won", "solved", "success", "all_levels_completed")
    negative_tokens = ("game_over", "step_limit", "timeout", "failed", "loss", "lose")
    for event in events:
        lowered = str(event).lower()
        if any(token in lowered for token in negative_tokens):
            continue
        if any(token in lowered for token in positive_tokens):
            return True
    return False


def _classify_transition_usefulness(
    *,
    visible_effect: bool,
    events: list[str],
    score_delta: float,
    terminal: bool,
    no_effect_event: bool,
    invalid_action: bool,
    prior_action_count: int = 0,
    prior_family_count: int = 0,
    prior_useful_action_count: int = 0,
    public_controllability_evidence: bool = False,
    public_reachable_state_evidence: bool = False,
    state_class_seen: bool = True,
) -> TransitionUsefulness:
    progress_effect = bool(POSITIVE_EVENTS.intersection(str(event) for event in events)) or float(score_delta) > 0.0
    terminal_win = bool(terminal and _is_terminal_win_event(events))
    no_effect = bool((not visible_effect) or no_effect_event or invalid_action)
    has_prior_public_usefulness = bool(prior_useful_action_count > 0)
    controllability_effect = bool(
        visible_effect
        and not no_effect
        and not progress_effect
        and (public_controllability_evidence or has_prior_public_usefulness)
        and prior_family_count < 8
    )
    reachable_state_class_effect = bool(
        visible_effect
        and not no_effect
        and not progress_effect
        and public_reachable_state_evidence
        and not state_class_seen
        and prior_family_count < 4
    )
    useful_effect = bool(
        progress_effect
        or terminal_win
        or controllability_effect
        or reachable_state_class_effect
    )
    nuisance_effect = bool(visible_effect and not useful_effect)
    reasons: list[str] = []
    if progress_effect:
        reasons.append("progress")
    if terminal_win:
        reasons.append("terminal_win")
    if controllability_effect:
        reasons.append("publicly_supported_controllability")
    if reachable_state_class_effect:
        reasons.append("publicly_supported_reachable_state_class")
    if nuisance_effect:
        reasons.append("visible_without_useful_effect")
    if no_effect:
        reasons.append("no_effect")
    return TransitionUsefulness(
        visible_effect=bool(visible_effect),
        useful_effect=useful_effect,
        nuisance_effect=nuisance_effect,
        no_effect=no_effect,
        progress_effect=progress_effect,
        terminal_win=terminal_win,
        controllability_effect=controllability_effect,
        reachable_state_class_effect=reachable_state_class_effect,
        reasons=tuple(reasons),
    )


def _transition_credit(
    *,
    useful: TransitionUsefulness,
    no_effect_event: bool,
    invalid_action: bool,
    score_delta: float,
    terminal: bool,
) -> float:
    if invalid_action:
        return -1.0
    if useful.progress_effect or useful.terminal_win:
        return 1.15 + min(max(float(score_delta), 0.0), 2.0)
    if useful.controllability_effect or useful.reachable_state_class_effect:
        return 0.0
    if no_effect_event or useful.no_effect:
        return -0.65
    if useful.nuisance_effect:
        return min(float(score_delta), 0.0) - 0.18
    if terminal:
        return -0.65
    return min(float(score_delta), 0.0) - 0.04


def _observation_key(observation: Any) -> str:
    try:
        return stable_hash(public_observation_key_payload(observation))
    except Exception:
        return ""


def _abstract_frame_state_class(frame: np.ndarray | None) -> str:
    if frame is None:
        return ""
    arr = np.asarray(frame, dtype=np.int64)
    if arr.size == 0:
        return ""
    nonzero = arr[arr != 0]
    colors, counts = np.unique(nonzero, return_counts=True) if nonzero.size else (np.asarray([], dtype=np.int64), np.asarray([], dtype=np.int64))
    components = _frame_components(arr)
    bbox: list[int] | None = None
    if nonzero.size:
        coords = np.argwhere(arr != 0)
        y0, x0 = coords.min(axis=0)
        y1, x1 = coords.max(axis=0)
        height = max(int(arr.shape[0]), 1)
        width = max(int(arr.shape[1]), 1)
        bbox = [
            int(round(3.0 * float(y0) / float(height))),
            int(round(3.0 * float(x0) / float(width))),
            int(round(3.0 * float(y1) / float(height))),
            int(round(3.0 * float(x1) / float(width))),
        ]
    payload = {
        "shape": [int(arr.shape[0]), int(arr.shape[1])],
        "nonzero_count_bucket": int(min(nonzero.size // 4, 32)),
        "colors": [(int(color), int(min(count, 64))) for color, count in zip(colors.tolist(), counts.tolist())],
        "component_count_bucket": int(min(len(components), 32)),
        "component_areas": sorted(int(min(component.area, 64)) for component in components)[:16],
        "bbox_bucket": bbox,
    }
    return stable_hash(payload)


def _experiment_abstract_state_class(action: str, frame: np.ndarray | None, fallback: str) -> str:
    family = _action_family(action)
    if family not in {"move", "object"} or frame is None:
        return str(fallback)
    arr = np.asarray(frame, dtype=np.int64)
    if arr.ndim != 2 or arr.size == 0:
        return str(fallback)
    signature = _component_state_signature(arr)
    return f"component_state:{signature}" if signature else str(fallback)


def _action_family(action: str) -> str:
    lowered = str(action).lower()
    if lowered in {"up", "down", "left", "right", "north", "south", "east", "west", "1", "2", "3", "4"}:
        return "move"
    if "click" in lowered or "press" in lowered or "tap" in lowered or "touch" in lowered:
        return "contact"
    if lowered in {"7", "wait", "noop", "no_op", "none"}:
        return "wait"
    if "toggle" in lowered or "use" in lowered or "pickup" in lowered or "drop" in lowered:
        return "object"
    return "other"


def _experiment_action_class(action: str) -> str:
    family = _action_family(action)
    if family == "contact":
        return "coordinate_contact"
    if str(action) == "7":
        return "undo_probe"
    return f"{family}:{str(action)}"


def _experiment_action_priority(action: str) -> int:
    family = _action_family(action)
    if family in {"move", "object"}:
        return 0
    if family == "contact":
        return 1
    if family == "other":
        return 2
    if str(action) == "7":
        return 3
    return 4


def _experiment_coordinate_equiv_class(action: str, frame: np.ndarray | None) -> str:
    family = _action_family(action)
    if family != "contact":
        return str(action)
    if frame is None or not frame.size:
        return "contact:unknown_frame"
    target = _action_target_cell(action, frame)
    relation = _component_target_relation(frame, target)
    relation_keys = _candidate_relation_keys(family, relation)
    relation_key = relation_keys[0][0] if relation_keys else "contact:no_relation"
    if target is None:
        return f"{relation_key}:no_target"
    height = max(int(frame.shape[0]), 1)
    width = max(int(frame.shape[1]), 1)
    y, x = target
    bucket_y = int(min(max(4 * int(y) // height, 0), 3))
    bucket_x = int(min(max(4 * int(x) // width, 0), 3))
    value = int(relation.get("value", 0))
    if value == 0:
        value_class = "background"
    elif bool(relation.get("on_component")):
        value_class = f"object:{value}"
    else:
        value_class = f"color:{value}"
    local_cell = relation.get("component_local_cell")
    component_suffix = ""
    if isinstance(local_cell, list) and len(local_cell) >= 2 and str(relation.get("kind", "")) in {
        "on_component",
        "adjacent_component",
    }:
        component_suffix = f":component_cell:{int(local_cell[0])}:{int(local_cell[1])}"
    return f"{relation_key}:{value_class}:bucket:{bucket_y}:{bucket_x}{component_suffix}"


def _experiment_hypothesis_ids(action: str, frame: np.ndarray | None) -> list[str]:
    family = _action_family(action)
    ids = [
        "h_progress",
        f"h_{family}_visible_or_control",
        "h_no_change",
    ]
    if family == "contact" and frame is not None and frame.size:
        target = _action_target_cell(action, frame)
        relation = _component_target_relation(frame, target)
        ids.append(f"h_relation:{str(relation.get('relation_key', 'target'))}")
    if str(action) == "7":
        ids.append("h_reversibility")
    return ids


def _experiment_predicted_outcomes(action: str, family: str, coordinate_class: str) -> dict[str, str]:
    if str(action) == "7":
        return {
            "h_progress": "undo_recovers_from_harm_or_deadend",
            "h_reversibility": "state_reverts_to_previous_public_class",
            "h_no_change": "no_change_or_noop",
        }
    if family == "contact":
        return {
            "h_progress": "score_delta>0 or level_changed",
            "h_contact_visible_or_control": f"target_equiv_class_changes:{coordinate_class}",
            "h_no_change": "no_change_or_bad_click",
        }
    return {
        "h_progress": "score_delta>0 or level_changed",
        f"h_{family}_visible_or_control": f"{family}_action_changes_public_state",
        "h_no_change": "no_change_or_blocked",
    }


def _experiment_result_class(label: TransitionUsefulness) -> str:
    if label.progress_effect or label.terminal_win:
        return "progress"
    if label.controllability_effect:
        return "controllability"
    if label.reachable_state_class_effect:
        return "reachable_state_class"
    if label.nuisance_effect:
        return "nuisance"
    if label.no_effect:
        return "no_op"
    return "discriminating_evidence"


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
            useful_effect=bool(outcome.get("useful_effect", False)),
            nuisance_effect=bool(outcome.get("nuisance_effect", False)),
            progress_effect=bool(outcome.get("progress_effect", False)),
            controllability_effect=bool(outcome.get("controllability_effect", False)),
            reachable_state_class_effect=bool(outcome.get("reachable_state_class_effect", False)),
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
            useful_effect=bool(outcome.get("useful_effect", False)),
            nuisance_effect=bool(outcome.get("nuisance_effect", False)),
            progress_effect=bool(outcome.get("progress_effect", False)),
            controllability_effect=bool(outcome.get("controllability_effect", False)),
            reachable_state_class_effect=bool(outcome.get("reachable_state_class_effect", False)),
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
    useful_effect: bool = False,
    nuisance_effect: bool = False,
    progress_effect: bool = False,
    controllability_effect: bool = False,
    reachable_state_class_effect: bool = False,
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
    hypothesis = {
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
        "useful_effect": bool(useful_effect),
        "nuisance_effect": bool(nuisance_effect),
        "progress_effect": bool(progress_effect),
        "controllability_effect": bool(controllability_effect),
        "reachable_state_class_effect": bool(reachable_state_class_effect),
        "score_delta": float(score_delta),
    }
    hypothesis["relation_delta_tokens"] = [token for token, _weight in _component_relation_delta_tokens(hypothesis)[:16]]
    return hypothesis


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
            y0, x0, _y1, _x1 = component.bbox
            return {
                "kind": "on_component",
                "value": int(component.value),
                "area_bucket": _area_bucket(component.area),
                "component_id": int(component.component_id),
                "component_bbox": [int(item) for item in component.bbox],
                "component_local_cell": [int(y - y0), int(x - x0)],
                "component_count": len(components),
            }
    nearest: tuple[_FrameComponent, float] | None = None
    for component in components:
        distance = math.dist((float(y), float(x)), component.centroid)
        if nearest is None or distance < nearest[1]:
            nearest = (component, distance)
    if nearest is not None and nearest[1] <= 1.75:
        component = nearest[0]
        y0, x0, _y1, _x1 = component.bbox
        return {
            "kind": "adjacent_component",
            "value": int(component.value),
            "area_bucket": _area_bucket(component.area),
            "component_id": int(component.component_id),
            "component_bbox": [int(item) for item in component.bbox],
            "component_local_cell": [int(y - y0), int(x - x0)],
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


def _general_component_relation_key(relation_key: str) -> str:
    key = str(relation_key)
    if not key:
        return ""
    parts = key.split(":")
    if "background" in parts:
        background_index = parts.index("background")
        return ":".join(parts[: background_index + 1])
    return key


def _general_component_action_template(template: str) -> str:
    template = str(template)
    if template.startswith("relation:"):
        relation = _general_component_relation_key(template[len("relation:") :])
        return f"relation:{relation}" if relation else ""
    return template


def _relation_delta_scope(token: str) -> str:
    return str(token).split("|", 1)[0] if "|" in str(token) else ""


def _relation_delta_token_is_productive(token: str) -> bool:
    delta = str(token).split("|", 1)[-1]
    return "blocked_or_no_effect" not in delta and delta != "mechanism:stable"


def _relation_delta_suffix(token: str) -> str:
    return str(token).split("|", 1)[-1]


def _relation_delta_tokens_overlap(expected_tokens: list[str], actual_tokens: list[str]) -> bool:
    expected = {str(token) for token in expected_tokens if str(token)}
    actual = {str(token) for token in actual_tokens if str(token)}
    if expected & actual:
        return True
    expected_deltas = {_relation_delta_suffix(token) for token in expected if _relation_delta_token_is_productive(token)}
    actual_deltas = {_relation_delta_suffix(token) for token in actual if _relation_delta_token_is_productive(token)}
    expected_deltas.discard("")
    actual_deltas.discard("")
    return bool(expected_deltas & actual_deltas)


def _movement_direction_token(delta: Any) -> str:
    if not isinstance(delta, (list, tuple)) or len(delta) < 2:
        return "unknown"
    try:
        dy = float(delta[0])
        dx = float(delta[1])
    except (TypeError, ValueError):
        return "unknown"
    if abs(dy) < 0.25 and abs(dx) < 0.25:
        return "stationary"
    if abs(dx) >= abs(dy) * 1.35:
        return "right" if dx > 0.0 else "left"
    if abs(dy) >= abs(dx) * 1.35:
        return "down" if dy > 0.0 else "up"
    vertical = "down" if dy > 0.0 else "up"
    horizontal = "right" if dx > 0.0 else "left"
    return f"{vertical}_{horizontal}"


def _component_relation_delta_action_scopes(
    action: str,
    family: str,
    target_relation: dict[str, Any],
) -> dict[str, float]:
    scopes: dict[str, float] = {f"family:{family}": 0.30}
    relation_keys = _candidate_relation_keys(family, target_relation)
    relation_key = relation_keys[0][0] if relation_keys else f"{family}:no_target"
    template = _component_relation_action_template(action, family, relation_key, target_relation)
    general_template = _general_component_action_template(template)
    if general_template:
        scopes[f"template:{general_template}"] = 1.0
    for relation_key, weight in relation_keys:
        general_relation = _general_component_relation_key(relation_key)
        if general_relation:
            scopes[f"relation:{general_relation}"] = max(scopes.get(f"relation:{general_relation}", 0.0), 0.70 * weight)
    target_value = int(target_relation.get("value", 0) or 0)
    if target_value:
        scopes[f"target_value:{target_value}"] = 0.44
        scopes[f"component_value:{target_value}"] = 0.38
    return scopes


def _resolve_sequence_expected_action(
    expected: dict[str, Any],
    legal_actions: list[str] | tuple[str, ...],
    frame: np.ndarray | None,
    components: list[_FrameComponent] | None = None,
) -> str | None:
    legal = [str(action) for action in legal_actions]
    if not legal or not expected or frame is None or not frame.size:
        return None
    if components is None:
        components = _frame_components(frame)
    action_template = str(expected.get("action_template", ""))
    expected_tokens = [str(token) for token in list(expected.get("relation_delta_tokens", []))[:16] if str(token)]
    expected_scopes = {_relation_delta_scope(token) for token in expected_tokens}
    expected_scopes.discard("")
    candidates: list[tuple[float, int, str]] = []
    for index, action in enumerate(legal):
        target_relation = _component_target_relation(frame, _action_target_cell(action, frame), components)
        family = _action_family(action)
        current_keys = {key for key, _ in _candidate_relation_keys(family, target_relation)}
        current_general_keys = {_general_component_relation_key(key) for key in current_keys}
        scopes = _component_relation_delta_action_scopes(action, family, target_relation)
        score = 0.0
        if action_template:
            if action_template.startswith("action:"):
                if action == action_template[len("action:") :]:
                    score += 1.15
                elif expected_tokens:
                    score += 0.05
            elif action_template.startswith("relation:"):
                expected_relation = _general_component_relation_key(action_template[len("relation:") :])
                if expected_relation in current_general_keys:
                    score += 1.00
            elif action == action_template:
                score += 0.80
        relation_key = _general_component_relation_key(str(expected.get("relation_key", "")))
        fallback_relation = _general_component_relation_key(str(expected.get("fallback_relation", "")))
        if relation_key and relation_key in current_general_keys:
            score += 0.72
        if fallback_relation and fallback_relation in current_general_keys:
            score += 0.45
        expected_value = int(expected.get("target_value", 0) or 0)
        if expected_value and int(target_relation.get("value", 0) or 0) == expected_value:
            score += 0.36
        if expected_scopes:
            scope_score = max((float(scopes.get(scope, 0.0)) for scope in expected_scopes), default=0.0)
            score += 0.68 * scope_score
        if score > 0.20:
            candidates.append((score, -index, action))
    if not candidates:
        resolved = _resolve_component_relation_action_template(action_template, legal, frame, components)
        return resolved
    candidates.sort(reverse=True)
    return candidates[0][2]


def _component_relation_delta_tokens(hypothesis: dict[str, Any]) -> list[tuple[str, float]]:
    if not hypothesis:
        return []
    family = str(hypothesis.get("action_family", "other"))
    relation_key = _general_component_relation_key(str(hypothesis.get("relation_key", "")))
    fallback_relation = _general_component_relation_key(str(hypothesis.get("fallback_relation", "")))
    action_template = _general_component_action_template(str(hypothesis.get("action_template", "")))
    target_relation = hypothesis.get("target_relation", {})
    if not isinstance(target_relation, dict):
        target_relation = {}
    component_value = int(hypothesis.get("component_value", 0) or 0)
    target_value = int(hypothesis.get("target_value", 0) or 0)
    mechanism = str(hypothesis.get("mechanism", ""))

    scopes: dict[str, float] = {f"family:{family}": 0.30}
    if action_template:
        scopes[f"template:{action_template}"] = 1.0
    if relation_key:
        scopes[f"relation:{relation_key}"] = max(scopes.get(f"relation:{relation_key}", 0.0), 0.82)
    if fallback_relation and fallback_relation != relation_key:
        scopes[f"relation:{fallback_relation}"] = max(scopes.get(f"relation:{fallback_relation}", 0.0), 0.56)
    if target_value:
        scopes[f"target_value:{target_value}"] = 0.42
    if component_value:
        scopes[f"component_value:{component_value}"] = 0.46

    deltas: dict[str, float] = {}
    if mechanism:
        deltas[f"mechanism:{mechanism}"] = 1.0 if mechanism in PRODUCTIVE_COMPONENT_MECHANISMS else 0.58

    for moved in list(hypothesis.get("moved_components", []) or [])[:3]:
        if not isinstance(moved, dict):
            continue
        value = int(moved.get("value", component_value) or 0)
        area = _area_bucket(int(moved.get("area", 0) or 0))
        direction = _movement_direction_token(moved.get("delta", []))
        deltas[f"move:dir:{direction}"] = max(deltas.get(f"move:dir:{direction}", 0.0), 0.70)
        if value:
            deltas[f"move:value:{value}:dir:{direction}"] = 1.00
            deltas[f"move:value:{value}:area:{area}:dir:{direction}"] = 1.08

    for item in list(hypothesis.get("transformed_components", []) or [])[:3]:
        if not isinstance(item, dict):
            continue
        before_value = int(item.get("before_value", 0) or 0)
        after_value = int(item.get("after_value", 0) or 0)
        before_area = _area_bucket(int(item.get("before_area", 0) or 0))
        after_area = _area_bucket(int(item.get("after_area", 0) or 0))
        if before_value or after_value:
            deltas[f"transform:{before_value}->{after_value}"] = 1.00
            deltas[f"transform:{before_value}->{after_value}:area:{before_area}->{after_area}"] = 1.06
        if after_value:
            deltas[f"transform_to:{after_value}"] = max(deltas.get(f"transform_to:{after_value}", 0.0), 0.76)

    for item in list(hypothesis.get("appeared_components", []) or [])[:3]:
        if not isinstance(item, dict):
            continue
        value = int(item.get("value", 0) or 0)
        area = _area_bucket(int(item.get("area", 0) or 0))
        if value:
            deltas[f"appear:value:{value}:area:{area}"] = 0.98
            deltas[f"appear:value:{value}"] = max(deltas.get(f"appear:value:{value}", 0.0), 0.76)

    for item in list(hypothesis.get("disappeared_components", []) or [])[:3]:
        if not isinstance(item, dict):
            continue
        value = int(item.get("value", 0) or 0)
        area = _area_bucket(int(item.get("area", 0) or 0))
        if value:
            deltas[f"remove:value:{value}:area:{area}"] = 0.98
            deltas[f"remove:value:{value}"] = max(deltas.get(f"remove:value:{value}", 0.0), 0.76)

    if mechanism == "component_visual_transform":
        deltas["visual_transform"] = max(deltas.get("visual_transform", 0.0), 0.76)

    token_weights: dict[str, float] = {}
    for scope, scope_weight in scopes.items():
        for delta, delta_weight in deltas.items():
            token = f"{scope}|{delta}"
            token_weights[token] = max(token_weights.get(token, 0.0), float(scope_weight) * float(delta_weight))
    return sorted(token_weights.items(), key=lambda item: (-item[1], item[0]))[:36]


def _component_prediction_key(relation_key: str, mechanism: str) -> str:
    return f"{str(relation_key)}=>{str(mechanism)}"


def _component_hypothesis_value(hypothesis: dict[str, Any]) -> float:
    mechanism = str(hypothesis.get("mechanism", ""))
    changed_pixels = int(hypothesis.get("changed_pixels", 0))
    value = 0.0
    progress_linked = bool(hypothesis.get("event_linked")) or bool(hypothesis.get("progress_effect"))
    nuisance = bool(hypothesis.get("nuisance_effect"))
    if bool(hypothesis.get("event_linked")):
        value += 1.12 + min(max(float(hypothesis.get("score_delta", 0.0)), 0.0), 2.0)
    if changed_pixels and progress_linked:
        value += 0.16 + min(float(changed_pixels), 32.0) * 0.007
    elif changed_pixels and bool(hypothesis.get("useful_effect")) and not nuisance:
        if bool(hypothesis.get("reachable_state_class_effect")):
            value += 0.32 + min(float(changed_pixels), 32.0) * 0.004
        elif bool(hypothesis.get("controllability_effect")):
            value += 0.26 + min(float(changed_pixels), 32.0) * 0.003
        else:
            value += 0.035
    elif changed_pixels and nuisance:
        value -= 0.18 + min(float(changed_pixels), 32.0) * 0.004
    if progress_linked and mechanism in {
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
        "relation_delta_tokens": [str(token) for token in list(hypothesis.get("relation_delta_tokens", []))[:16]],
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
        "relation_delta_tokens": [str(token) for token in list(hypothesis.get("relation_delta_tokens", []))[:16]],
    }


def _relation_delta_sequence_component_expectation(hypothesis: dict[str, Any]) -> dict[str, Any]:
    if not hypothesis:
        return {}
    return {
        "action": str(hypothesis.get("action", "")),
        "action_template": _general_component_action_template(str(hypothesis.get("action_template", ""))),
        "relation_key": _general_component_relation_key(str(hypothesis.get("relation_key", ""))),
        "fallback_relation": _general_component_relation_key(str(hypothesis.get("fallback_relation", ""))),
        "mechanism": str(hypothesis.get("mechanism", "")),
        "target_value": int(hypothesis.get("target_value", 0) or hypothesis.get("component_value", 0) or 0),
        "changed_expected": int(hypothesis.get("changed_pixels", 0)) > 0,
        "relation_delta_tokens": [str(token) for token in list(hypothesis.get("relation_delta_tokens", []))[:16]],
        "requires_relation_delta_match": True,
    }


def _sequence_candidate_key(candidate: dict[str, Any]) -> tuple[Any, ...]:
    actions = tuple(str(action) for action in candidate.get("actions", []))
    templates = tuple(str(template) for template in candidate.get("action_templates", []))
    delta_suffixes: list[str] = []
    for expectation in candidate.get("component_expectations", []) or []:
        if not isinstance(expectation, dict):
            continue
        tokens = [str(token) for token in list(expectation.get("relation_delta_tokens", []))[:4]]
        if tokens:
            delta_suffixes.append(",".join(sorted({_relation_delta_suffix(token) for token in tokens})))
        else:
            delta_suffixes.append(str(expectation.get("mechanism", "")))
    return (str(candidate.get("source", "")), actions, templates, tuple(delta_suffixes))


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
    expected_delta_tokens = [str(token) for token in list(expected.get("relation_delta_tokens", []))[:16]]
    actual_delta_tokens = [str(token) for token in list(actual.get("relation_delta_tokens", []))[:16]]
    if bool(expected.get("requires_relation_delta_match", False)):
        return _relation_delta_tokens_overlap(expected_delta_tokens, actual_delta_tokens)
    if expected_delta_tokens and actual_delta_tokens and _relation_delta_tokens_overlap(expected_delta_tokens, actual_delta_tokens):
        return True
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
    useful_effect: bool = False,
    nuisance_effect: bool = False,
    progress_effect: bool = False,
    controllability_effect: bool = False,
    reachable_state_class_effect: bool = False,
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
        "useful_effect": bool(useful_effect),
        "nuisance_effect": bool(nuisance_effect),
        "progress_effect": bool(progress_effect),
        "controllability_effect": bool(controllability_effect),
        "reachable_state_class_effect": bool(reachable_state_class_effect),
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


def _public_controllability_evidence(before: Any, after: Any, action: str) -> bool:
    if before is None or after is None:
        return False
    try:
        before_arr = np.asarray(before, dtype=np.int64)
        after_arr = np.asarray(after, dtype=np.int64)
    except Exception:
        return False
    if before_arr.ndim != 2 or after_arr.ndim != 2 or before_arr.shape != after_arr.shape:
        return False
    if before_arr.size == 0 or after_arr.size == 0 or not bool(np.any(before_arr != after_arr)):
        return False
    family = _action_family(action)
    if family not in {"move", "object"}:
        return False
    if _movement_colors(before_arr, after_arr):
        return True
    component = _component_hypothesis_from_frames(
        before=before_arr,
        after=after_arr,
        action=str(action),
        step_index=-1,
        event_hit=False,
        score_delta=0.0,
        no_effect=False,
    )
    mechanism = str((component or {}).get("mechanism", ""))
    if family == "move":
        return mechanism == "component_movement"
    return mechanism in {
        "component_movement",
        "component_color_transform",
        "component_appearance",
        "component_disappearance",
        "component_split_merge",
    }


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
    progress_linked = bool(hypothesis.get("event_linked")) or bool(hypothesis.get("progress_effect"))
    nuisance = bool(hypothesis.get("nuisance_effect"))
    if bool(hypothesis.get("event_linked")):
        value += 1.05 + min(max(float(hypothesis.get("score_delta", 0.0)), 0.0), 2.0)
    if changed_pixels and progress_linked:
        value += 0.18 + min(float(changed_pixels), 16.0) * 0.012
    elif changed_pixels and bool(hypothesis.get("useful_effect")) and not nuisance:
        if bool(hypothesis.get("reachable_state_class_effect")):
            value += 0.22 + min(float(changed_pixels), 16.0) * 0.006
        elif bool(hypothesis.get("controllability_effect")):
            value += 0.16 + min(float(changed_pixels), 16.0) * 0.004
        else:
            value += 0.035
    elif changed_pixels and nuisance:
        value -= 0.18 + min(float(changed_pixels), 16.0) * 0.006
    if progress_linked and mechanism in {"movement", "spawn", "removal", "toggle_or_transform"}:
        value += 0.06
    if progress_linked and bool(hypothesis.get("click_contacts_change")):
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


def _bounded_relation_scoring_actions(actions: list[str] | tuple[str, ...]) -> list[str]:
    legal = [str(action) for action in actions]
    if len(legal) <= RELATION_SCORING_ACTION_LIMIT:
        return legal
    non_click = [action for action in legal if not action.lower().startswith("click:")]
    click_budget = max(RELATION_SCORING_ACTION_LIMIT - len(non_click), 0)
    bounded = [*non_click, *[action for action in legal if action.lower().startswith("click:")][:click_budget]]
    return list(dict.fromkeys(bounded))


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
