from __future__ import annotations

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
        if model is not None and self.use_jepa_tokens and record.steps:
            batch = tensors_from_attempts([record], device=self.device)
            model = model.to(self.device)
            model.eval()
            with torch.no_grad():
                token_mean = [round(float(item), 6) for item in model.attempt_tokens(batch).detach().cpu().reshape(-1).tolist()[:24]]
        total = max(len(record.steps), 1)
        causal = {
            "no_effect_rate": float(no_effect / total),
            "positive_event_rate": float(positive / total),
            "terminal_seen": float(any(step.terminal for step in record.steps)),
            "jepa_token_norm": float(torch.tensor(token_mean).norm().item()) if token_mean else 0.0,
        }
        next_plan: dict[str, float] = {}
        for action, value in events.items():
            next_plan[action] = next_plan.get(action, 0.0) + min(float(value), 3.0)
        for action, count in failed.items():
            next_plan[action] = next_plan.get(action, 0.0) - min(float(count), 4.0) * 0.25
        entry = AttemptMemoryEntry(
            attempt_index=int(record.attempt_index),
            token_mean=token_mean,
            failed_actions=failed,
            event_candidates=events,
            causal_hypotheses=causal,
            next_attempt_plan=next_plan,
        )
        self.entries.append(entry)
        return entry

    def score_action(self, action: str) -> float:
        if not self.entries:
            return 0.0
        score = 0.0
        for entry in self.entries[-3:]:
            score += 0.12 * float(entry.event_candidates.get(action, 0.0))
            score -= 0.08 * float(entry.failed_actions.get(action, 0))
            score += 0.01 * float(entry.causal_hypotheses.get("jepa_token_norm", 0.0))
        return float(max(min(score, 0.35), -0.35))

    def summary(self) -> dict[str, Any]:
        return {
            "entry_count": len(self.entries),
            "use_jepa_tokens": self.use_jepa_tokens,
            "entries": [entry.to_dict() for entry in self.entries[-3:]],
            "emits_text": False,
            "direct_action_source": False,
        }
