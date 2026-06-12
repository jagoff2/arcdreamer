from __future__ import annotations

import math
from dataclasses import asdict, dataclass, field
from typing import Any

import torch

from .attempt_buffer import AttemptRecord, tensors_from_attempts
from .video_jepa import VideoJEPA


POSITIVE_EVENTS = {"positive_reward", "resource_collected", "level_completed", "goal_reached", "goal_clicked", "useful_click"}


@dataclass
class AttemptMemoryEntry:
    attempt_index: int
    token_mean: list[float]
    failed_actions: dict[str, int]
    event_candidates: dict[str, float]
    causal_hypotheses: dict[str, float]
    next_attempt_plan: dict[str, float]
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

    def reset(self) -> None:
        self.entries.clear()

    def ingest_attempt(self, record: AttemptRecord, model: VideoJEPA | None = None) -> AttemptMemoryEntry:
        failed: dict[str, int] = {}
        events: dict[str, float] = {}
        no_effect = 0
        positive = 0
        for step in record.steps:
            event_hit = bool(POSITIVE_EVENTS.intersection(step.event_delta)) or float(step.score_delta) > 0.0
            changed = step.obs_hash != step.next_obs_hash
            if event_hit:
                positive += 1
                events[step.action] = events.get(step.action, 0.0) + max(1.0, float(step.score_delta))
            elif not changed or float(step.score_delta) <= 0.0:
                no_effect += 1
                failed[step.action] = failed.get(step.action, 0) + 1
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
        jepa_active = bool(jepa_action_evidence)
        effect_values = [item.get("effect", 0.0) for item in jepa_action_evidence.values()]
        surprise_values = [item.get("surprise", 0.0) for item in jepa_action_evidence.values()]
        effect_mean = sum(effect_values) / max(len(effect_values), 1)
        surprise_mean = sum(surprise_values) / max(len(surprise_values), 1)
        effect_span = (max(effect_values) - min(effect_values)) if effect_values else 0.0
        surprise_span = (max(surprise_values) - min(surprise_values)) if surprise_values else 0.0
        causal = {
            "no_effect_rate": float(no_effect / total),
            "positive_event_rate": float(positive / total),
            "terminal_seen": float(any(step.terminal for step in record.steps)),
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
            event_candidates=events,
            causal_hypotheses=causal,
            next_attempt_plan=next_plan,
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
            score += 0.35 * float(entry.next_attempt_plan.get(action, 0.0))
        return float(max(min(score, 0.35), -0.35))

    def plan_scores(self, legal_actions: tuple[str, ...] | list[str]) -> dict[str, float]:
        return {str(action): self.score_action(str(action)) for action in legal_actions}

    def action_distribution(self, legal_actions: tuple[str, ...] | list[str]) -> dict[str, float]:
        actions = [str(action) for action in legal_actions]
        if not actions:
            return {}
        scores = self.plan_scores(actions)
        mean_score = sum(scores.values()) / len(actions)
        logits = {action: max(min((scores[action] - mean_score) * 4.0, 8.0), -8.0) for action in actions}
        denom = sum(math.exp(value) for value in logits.values())
        return {action: float(math.exp(logits[action]) / max(denom, 1.0e-12)) for action in actions}

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
            "causal_substrate_active": any(entry.causal_substrate_active for entry in self.entries),
        }


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


def _centered_distribution_delta(plan: dict[str, float]) -> dict[str, float]:
    if not plan:
        return {}
    mean = sum(float(value) for value in plan.values()) / len(plan)
    return {action: float(value - mean) for action, value in sorted(plan.items())}
