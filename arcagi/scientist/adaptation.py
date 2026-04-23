"""Online scorer for attempt-to-attempt improvement.

The target for this model is not immediate reward. It is whether the current
attempt ends up better than the previous attempt on the same level/session
slice. This is the first concrete learning-to-learn signal in the scientist
stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


@dataclass(frozen=True)
class AdaptationPrediction:
    value_mean: float


class AdaptationScorer:
    """Tiny online linear scorer for future attempt improvement."""

    def __init__(
        self,
        *,
        feature_dim: int = 384,
        learning_rate: float = 0.08,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.rng = np.random.default_rng(seed)
        self.weights = np.zeros((self.feature_dim,), dtype=np.float32)
        self.updates = 0

    def state_dict(self) -> dict[str, object]:
        return {
            "feature_dim": self.feature_dim,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "updates": self.updates,
            "weights": self.weights.copy(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        feature_dim = int(state.get("feature_dim", self.feature_dim))
        weights = np.asarray(state.get("weights"), dtype=np.float32)
        if weights.shape != (feature_dim,):
            raise ValueError(f"adaptation scorer weights shape mismatch: expected {(feature_dim,)}, got {weights.shape}")
        self.feature_dim = feature_dim
        self.learning_rate = float(state.get("learning_rate", self.learning_rate))
        self.weight_decay = float(state.get("weight_decay", self.weight_decay))
        self.updates = int(state.get("updates", 0))
        self.weights = weights.copy()

    def predict_encoded(self, feature_vector: np.ndarray) -> AdaptationPrediction:
        value = float(np.dot(self.weights, feature_vector))
        return AdaptationPrediction(value_mean=float(np.tanh(value)))

    def score(self, prediction: AdaptationPrediction) -> float:
        return float(prediction.value_mean)

    def update(self, feature_vector: np.ndarray, target: float, *, weight: float = 1.0) -> float:
        clipped_target = _clip(target, -2.0, 2.0)
        pred = float(np.dot(self.weights, feature_vector))
        err = _clip(clipped_target - pred, -3.0, 3.0)
        self.weights *= 1.0 - self.learning_rate * self.weight_decay
        self.weights += (self.learning_rate * float(weight) * err * feature_vector).astype(np.float32)
        self.updates += 1
        return float(err * err)
