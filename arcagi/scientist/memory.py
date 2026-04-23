"""Surprise-weighted episodic and option memory for online adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .features import cosine, jaccard, state_features
from .types import ActionName, StructuredState, TransitionRecord, combined_progress_signal


@dataclass
class MemoryItem:
    step: int
    before_fingerprint: str
    after_fingerprint: str
    action: ActionName
    reward: float
    surprise: float
    state_vector: np.ndarray
    language_tokens: frozenset[str]
    note: str = ""


@dataclass
class OptionItem:
    action_sequence: tuple[ActionName, ...]
    state_vector: np.ndarray
    language_tokens: frozenset[str]
    reward: float
    uses: int = 0
    successes: int = 0

    @property
    def value(self) -> float:
        return float((self.reward + self.successes) / max(self.uses + 1, 1))


class EpisodicMemory:
    def __init__(self, *, capacity: int = 2048, feature_dim: int = 256) -> None:
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.items: list[MemoryItem] = []
        self.options: list[OptionItem] = []
        self.recent_actions: list[ActionName] = []

    def reset_episode(self) -> None:
        self.items.clear()
        self.options.clear()
        self.recent_actions.clear()

    def reset_level(self) -> None:
        self.recent_actions.clear()

    def write_transition(self, record: TransitionRecord, *, surprise: float, language_tokens: Iterable[str]) -> None:
        self.recent_actions.append(record.action)
        if len(self.recent_actions) > 12:
            self.recent_actions.pop(0)

        reward = float(combined_progress_signal(record.reward, record.delta.score_delta))
        should_write = surprise > 0.18 or abs(reward) > 1e-8 or record.delta.has_visible_effect
        if not should_write:
            return
        item = MemoryItem(
            step=record.step_index,
            before_fingerprint=record.before.abstract_fingerprint,
            after_fingerprint=record.after.abstract_fingerprint,
            action=record.action,
            reward=reward,
            surprise=float(surprise),
            state_vector=state_features(record.before, dim=self.feature_dim),
            language_tokens=frozenset(language_tokens),
            note=_transition_note(record),
        )
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity :]

        if reward > 0.0:
            seq = tuple(self.recent_actions[-6:])
            self.options.append(
                OptionItem(
                    action_sequence=seq,
                    state_vector=state_features(record.before, dim=self.feature_dim),
                    language_tokens=frozenset(language_tokens),
                    reward=reward,
                    uses=1,
                    successes=1,
                )
            )
            if len(self.options) > self.capacity // 4:
                self.options = sorted(self.options, key=lambda o: o.value, reverse=True)[: self.capacity // 4]

    def retrieve(self, state: StructuredState, language_tokens: Iterable[str], *, k: int = 8) -> tuple[MemoryItem, ...]:
        if not self.items:
            return ()
        vec = state_features(state, dim=self.feature_dim)
        tokens = frozenset(language_tokens)
        scored: list[tuple[float, MemoryItem]] = []
        for item in self.items:
            score = 0.65 * cosine(vec, item.state_vector) + 0.25 * jaccard(tokens, item.language_tokens)
            score += 0.15 * max(item.reward, 0.0) + 0.05 * item.surprise
            scored.append((float(score), item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return tuple(item for _, item in scored[:k])

    def retrieve_options(self, state: StructuredState, language_tokens: Iterable[str], *, k: int = 4) -> tuple[OptionItem, ...]:
        if not self.options:
            return ()
        vec = state_features(state, dim=self.feature_dim)
        tokens = frozenset(language_tokens)
        scored: list[tuple[float, OptionItem]] = []
        for option in self.options:
            score = 0.55 * cosine(vec, option.state_vector) + 0.20 * jaccard(tokens, option.language_tokens)
            score += 0.35 * option.value
            scored.append((float(score), option))
        scored.sort(key=lambda x: x[0], reverse=True)
        return tuple(item for _, item in scored[:k])

    def action_memory_bonus(self, state: StructuredState, action: ActionName, language_tokens: Iterable[str]) -> float:
        bonus = 0.0
        for item in self.retrieve(state, language_tokens, k=8):
            if item.action == action:
                bonus += 0.08 * item.surprise + 0.12 * max(item.reward, 0.0)
        for option in self.retrieve_options(state, language_tokens, k=4):
            if option.action_sequence and option.action_sequence[0] == action:
                bonus += 0.25 * option.value
        return float(bonus)


def _transition_note(record: TransitionRecord) -> str:
    parts = [f"action={record.action}"]
    if record.delta.moved_objects:
        parts.append(f"moved={len(record.delta.moved_objects)}")
    if record.delta.changed_cells:
        parts.append(f"changed={record.delta.changed_cells}")
    if record.reward:
        parts.append(f"reward={record.reward:.3f}")
    if record.delta.score_delta:
        parts.append(f"score_delta={record.delta.score_delta:.3f}")
    return ";".join(parts)
