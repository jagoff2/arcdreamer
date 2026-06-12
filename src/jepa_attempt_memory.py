from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from .attempt_buffer import AttemptRecord, observation_frame, stable_hash, tensors_from_attempts
from .video_jepa import VideoJEPA


POSITIVE_EVENTS = {"positive_reward", "resource_collected", "level_completed", "goal_reached", "goal_clicked", "useful_click"}


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
        self.sequence_candidates: list[dict[str, Any]] = []
        self.active_sequence: list[str] = []
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
        self.sequence_candidates.clear()
        self.active_sequence.clear()
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
        sequence_candidates_added = self._update_sequence_candidates(record, transition_outcomes, object_hypotheses)
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
        planned_action = self.sequence_plan_action(legal_actions)
        if planned_action is not None:
            scores[planned_action] = scores.get(planned_action, 0.0) + 0.85
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
            "sequence_candidate_count": len(self.sequence_candidates),
            "active_sequence_length": len(self.active_sequence),
            "active_sequence_remaining": active_remaining,
            "active_sequence_source": self.active_sequence_source,
        }

    def start_attempt(self) -> None:
        self.active_sequence = []
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
                "delayed_public_event_credit",
                "prior_event_sequence_candidate",
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

    def _update_sequence_candidates(
        self,
        record: AttemptRecord,
        outcomes: list[dict[str, Any]],
        hypotheses: list[dict[str, Any]],
    ) -> int:
        added = 0
        hypothesis_by_index = {int(item.get("step_index", -1)): item for item in hypotheses}
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
            value = 1.0 + max(float(record.steps[event_index].score_delta), 0.0) + 0.08 * len(linked)
            self.sequence_candidates.append(
                {
                    "source": "positive_public_event_window",
                    "actions": actions,
                    "value": float(value),
                    "length": len(actions),
                    "event_index": event_index,
                    "linked_regions": [_hypothesis_region(item) for item in linked[:4]],
                    "linked_mechanisms": [str(item.get("mechanism")) for item in linked[:4]],
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
