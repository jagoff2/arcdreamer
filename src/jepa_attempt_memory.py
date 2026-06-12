from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np
import torch

from .attempt_buffer import AttemptRecord, stable_hash, tensors_from_attempts
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
        graph_summary = self.transition_graph_summary()
        causal.update(
            {
                "transition_graph_edges": float(graph_summary["observed_edges"]),
                "transition_graph_positive_edges": float(graph_summary["positive_edges"]),
                "transition_graph_no_effect_edges": float(graph_summary["no_effect_edges"]),
                "transition_graph_delayed_credit_edges": float(delayed_credit_edges),
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
                "delayed_public_event_credit",
                "targeted_next_attempt_experiment",
            ],
            "transition_graph": self.transition_graph_summary(),
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


def _centered_distribution_delta(plan: dict[str, float]) -> dict[str, float]:
    if not plan:
        return {}
    mean = sum(float(value) for value in plan.values()) / len(plan)
    return {action: float(value - mean) for action, value in sorted(plan.items())}
