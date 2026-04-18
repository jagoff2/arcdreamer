from __future__ import annotations

from dataclasses import dataclass, field
import heapq
from typing import Any

import numpy as np

from arcagi.core.types import ActionName
from arcagi.core.utils import cosine_similarity

GENERIC_QUERY_TOKENS: frozenset[str] = frozenset(
    {
        "goal",
        "need",
        "test",
        "unknown",
        "uncertain",
        "rule",
        "question",
        "plan",
        "explore",
        "confirm",
        "move",
        "interact",
        "toward",
        "target",
        "active",
        "inactive",
        "high",
        "mid",
        "low",
    }
)


@dataclass
class EpisodicEntry:
    key: np.ndarray
    belief_tokens: tuple[str, ...]
    question_tokens: tuple[str, ...]
    context_tokens: tuple[str, ...]
    action_history: tuple[ActionName, ...]
    reward: float
    salience: float
    payload: dict[str, Any] = field(default_factory=dict)


class EpisodicMemory:
    def __init__(self, capacity: int = 1024) -> None:
        self.capacity = capacity
        self.entries: list[EpisodicEntry] = []

    def clear(self) -> None:
        self.entries.clear()

    def write(
        self,
        key: np.ndarray,
        belief_tokens: tuple[str, ...],
        question_tokens: tuple[str, ...],
        action_history: tuple[ActionName, ...],
        reward: float,
        salience: float,
        context_tokens: tuple[str, ...] = (),
        payload: dict[str, Any] | None = None,
    ) -> None:
        self.entries.append(
            EpisodicEntry(
                key=np.asarray(key, dtype=np.float32).copy(),
                belief_tokens=belief_tokens,
                question_tokens=question_tokens,
                context_tokens=context_tokens,
                action_history=action_history,
                reward=reward,
                salience=salience,
                payload=payload or {},
            )
        )
        if len(self.entries) > self.capacity:
            self.entries.sort(key=lambda item: item.salience, reverse=True)
            self.entries = self.entries[: self.capacity]

    def query(
        self,
        key: np.ndarray,
        query_tokens: tuple[str, ...] = (),
        top_k: int = 3,
    ) -> list[tuple[float, EpisodicEntry]]:
        if not self.entries:
            return []
        key = np.asarray(key, dtype=np.float32)
        query_set = set(query_tokens)
        heap: list[tuple[float, int]] = []
        for idx, entry in enumerate(self.entries):
            score = cosine_similarity(key, entry.key)
            if query_set:
                entry_tokens = entry.belief_tokens + entry.question_tokens + entry.context_tokens
                generic_overlap = sum(
                    1 for token in query_set.intersection(entry_tokens) if token in GENERIC_QUERY_TOKENS
                )
                content_overlap = sum(
                    1 for token in query_set.intersection(entry_tokens) if token not in GENERIC_QUERY_TOKENS
                )
                score += 0.02 * generic_overlap
                score += 0.18 * content_overlap
            score += 0.02 * entry.reward
            score += 0.02 * entry.salience
            if len(heap) < top_k:
                heapq.heappush(heap, (score, idx))
            else:
                heapq.heappushpop(heap, (score, idx))
        results = sorted(heap, reverse=True)
        return [(score, self.entries[idx]) for score, idx in results]
