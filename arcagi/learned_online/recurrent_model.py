from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from arcagi.learned_online.minimal_model import MinimalOnlineModel, MinimalPredictions


@dataclass
class RecurrentOnlineModel:
    candidate_input_dim: int
    event_input_dim: int
    hidden_dim: int = 64
    learning_rate: float = 0.16
    seed: int = 0

    def __post_init__(self) -> None:
        rng = np.random.default_rng(self.seed)
        scale = 1.0 / max(float(self.event_input_dim) ** 0.5, 1.0)
        self.event_projection = rng.normal(0.0, scale, size=(self.event_input_dim, self.hidden_dim)).astype(np.float32)
        self.hidden_recurrence = rng.normal(0.0, 0.05, size=(self.hidden_dim, self.hidden_dim)).astype(np.float32)
        self.hidden = np.zeros((self.hidden_dim,), dtype=np.float32)
        self.fast_heads = MinimalOnlineModel(
            input_dim=self.candidate_input_dim + self.hidden_dim,
            learning_rate=self.learning_rate,
        )

    @property
    def updates(self) -> int:
        return int(self.fast_heads.updates)

    def reset_episode(self, *, keep_weights: bool = True) -> None:
        self.hidden = np.zeros((self.hidden_dim,), dtype=np.float32)
        if not keep_weights:
            self.fast_heads = MinimalOnlineModel(
                input_dim=self.candidate_input_dim + self.hidden_dim,
                learning_rate=self.learning_rate,
            )

    def observe_event(self, event: np.ndarray) -> None:
        event = np.asarray(event, dtype=np.float32)
        projected = event @ self.event_projection
        recurrent = self.hidden @ self.hidden_recurrence
        self.hidden = np.tanh(projected + recurrent).astype(np.float32)

    def augment_features(self, candidate_features: np.ndarray) -> np.ndarray:
        features = np.asarray(candidate_features, dtype=np.float32)
        if features.ndim == 1:
            return np.concatenate([features, self.hidden]).astype(np.float32)
        hidden = np.repeat(self.hidden[None, :], features.shape[0], axis=0)
        return np.concatenate([features, hidden], axis=1).astype(np.float32)

    def predict(self, candidate_features: np.ndarray) -> MinimalPredictions:
        return self.fast_heads.predict(self.augment_features(candidate_features))

    def prediction_loss(self, feature: np.ndarray, labels, *, realized_info_gain: float) -> float:
        return self.fast_heads.prediction_loss(
            self.augment_features(np.asarray(feature, dtype=np.float32)),
            labels,
            realized_info_gain=realized_info_gain,
        )

    def batch_prediction_loss(self, entries: Sequence[object]) -> float:
        losses: list[float] = []
        for entry in entries:
            feature = getattr(entry, "feature", None)
            labels = getattr(entry, "labels", None)
            if feature is None or labels is None:
                continue
            realized_info_gain = float(getattr(entry, "realized_info_gain", 0.0) or 0.0)
            losses.append(self.prediction_loss(feature, labels, realized_info_gain=realized_info_gain))
        if not losses:
            return 0.0
        return float(np.mean(losses))

    def online_update(self, feature: np.ndarray, labels, *, realized_info_gain: float = 0.0) -> float:
        return self.fast_heads.online_update(
            self.augment_features(np.asarray(feature, dtype=np.float32)),
            labels,
            realized_info_gain=realized_info_gain,
        )

    def state_dict(self) -> dict[str, object]:
        return {
            "candidate_input_dim": int(self.candidate_input_dim),
            "event_input_dim": int(self.event_input_dim),
            "hidden_dim": int(self.hidden_dim),
            "learning_rate": float(self.learning_rate),
            "seed": int(self.seed),
            "event_projection": self.event_projection.copy(),
            "hidden_recurrence": self.hidden_recurrence.copy(),
            "fast_heads": self.fast_heads.state_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        if "event_projection" in state:
            self.event_projection = np.asarray(state["event_projection"], dtype=np.float32).copy()
        if "hidden_recurrence" in state:
            self.hidden_recurrence = np.asarray(state["hidden_recurrence"], dtype=np.float32).copy()
        fast = state.get("fast_heads", {})
        if isinstance(fast, dict):
            self.fast_heads.load_state_dict(fast)
