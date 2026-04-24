from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np

from arcagi.core.types import ActionName, StructuredState
from arcagi.learned_online.action_features import action_context_key
from arcagi.learned_online.questions import QuestionToken
from arcagi.learned_online.signals import TransitionLabels

MEMORY_FEATURE_DIM = 6


@dataclass
class OnlineMemoryEntry:
    state_key: np.ndarray
    context_key: str
    action: ActionName
    question: QuestionToken
    labels: TransitionLabels
    realized_info_gain: float
    feature: np.ndarray | None = None
    hidden: np.ndarray | None = None
    level_epoch: int = 0
    level_step: int = 0
    return_credit: float = 0.0


class OnlineEpisodicMemory:
    def __init__(self, capacity: int = 512) -> None:
        self.capacity = int(capacity)
        self.entries: list[OnlineMemoryEntry] = []
        self.level_epoch = 0

    def reset(self) -> None:
        self.entries.clear()
        self.level_epoch = 0

    def start_new_level(self, level_epoch: int) -> None:
        self.level_epoch = int(level_epoch)

    def write(
        self,
        *,
        state: StructuredState,
        action: ActionName,
        question: QuestionToken,
        labels: TransitionLabels,
        realized_info_gain: float,
        feature: np.ndarray | None = None,
        hidden: np.ndarray | None = None,
        level_epoch: int = 0,
        level_step: int = 0,
    ) -> None:
        self.entries.append(
            OnlineMemoryEntry(
                state_key=_state_key(state),
                context_key=action_context_key(state, action),
                action=str(action),
                question=question,
                labels=labels,
                realized_info_gain=float(realized_info_gain),
                feature=None if feature is None else np.asarray(feature, dtype=np.float32).copy(),
                hidden=None if hidden is None else np.asarray(hidden, dtype=np.float32).copy(),
                level_epoch=int(level_epoch),
                level_step=int(level_step),
            )
        )
        if len(self.entries) > self.capacity:
            self.entries = self.entries[-self.capacity :]

    def features_for(self, state: StructuredState, action: ActionName, *, top_k: int = 8) -> np.ndarray:
        if not self.entries:
            return np.zeros((MEMORY_FEATURE_DIM,), dtype=np.float32)
        key = _state_key(state)
        context_key = action_context_key(state, action)
        ranked: list[tuple[float, OnlineMemoryEntry]] = []
        for entry in self.entries:
            score = _cosine(key, entry.state_key)
            if entry.context_key == context_key:
                score += 0.35
            if entry.action == str(action):
                score += 0.15
            ranked.append((score, entry))
        ranked.sort(key=lambda item: item[0], reverse=True)
        selected = [entry for _score, entry in ranked[:top_k]]
        denom = max(float(len(selected)), 1.0)
        return np.asarray(
            [
                math.tanh(float(len(selected)) / float(top_k)),
                sum(entry.labels.useful_change for entry in selected) / denom,
                sum(entry.labels.visible_change for entry in selected) / denom,
                sum(max(entry.labels.visible_only_nonprogress, entry.labels.no_effect_nonprogress) for entry in selected)
                / denom,
                sum(max(entry.labels.reward_progress, entry.return_credit) for entry in selected) / denom,
                sum(max(entry.realized_info_gain, 0.0) for entry in selected) / denom,
            ],
            dtype=np.float32,
        )

    def credit_recent_success(
        self,
        *,
        level_epoch: int,
        gamma: float = 0.985,
        max_entries: int | None = None,
    ) -> list[OnlineMemoryEntry]:
        candidates = [
            entry
            for entry in self.entries
            if int(entry.level_epoch) == int(level_epoch) and entry.feature is not None
        ]
        if max_entries is not None:
            candidates = candidates[-max(int(max_entries), 1) :]
        horizon = len(candidates)
        credited: list[OnlineMemoryEntry] = []
        for index, entry in enumerate(candidates):
            distance = max(horizon - index - 1, 0)
            credit = float(gamma**distance)
            entry.return_credit = max(float(entry.return_credit), credit)
            credited.append(entry)
        return credited

    def summary(self) -> dict[str, int]:
        return {"entries": int(len(self.entries))}

    def sample_probe_batch(
        self,
        *,
        k: int = 8,
        exclude_action: ActionName | None = None,
    ) -> list[OnlineMemoryEntry]:
        candidates = [
            entry
            for entry in self.entries
            if entry.feature is not None and (exclude_action is None or entry.action != str(exclude_action))
        ]
        if not candidates:
            return []
        return candidates[-max(int(k), 1) :]


def _state_key(state: StructuredState) -> np.ndarray:
    return np.concatenate([state.summary_vector(), state.spatial_vector()]).astype(np.float32)


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 1e-8:
        return 0.0
    return float(np.dot(left, right) / denom)
