"""Teacher-shaped habit policy over spotlight workspace features.

The habit policy is intentionally simple: it learns a reusable action prior
from synthetic teacher labels over the same explicit workspace features that
the executive sees.  The executive can then learn when to override that prior.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


def _sigmoid(value: float) -> float:
    clipped = _clip(value, -12.0, 12.0)
    return float(1.0 / (1.0 + np.exp(-clipped)))


@dataclass(frozen=True)
class HabitPrediction:
    logit: float
    probability: float


class HabitPolicy:
    """Tiny online logistic policy used as the agent's default habit prior."""

    def __init__(
        self,
        *,
        feature_dim: int = 384,
        learning_rate: float = 0.25,
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
            raise ValueError(f"habit policy weights shape mismatch: expected {(feature_dim,)}, got {weights.shape}")
        self.feature_dim = feature_dim
        self.learning_rate = float(state.get("learning_rate", self.learning_rate))
        self.weight_decay = float(state.get("weight_decay", self.weight_decay))
        self.updates = int(state.get("updates", 0))
        self.weights = weights.copy()

    def predict_encoded(self, feature_vector: np.ndarray) -> HabitPrediction:
        logit = float(np.dot(self.weights, feature_vector))
        probability = _sigmoid(logit)
        return HabitPrediction(logit=logit, probability=probability)

    def score(self, prediction: HabitPrediction) -> float:
        return float((2.0 * prediction.probability) - 1.0)

    def update(self, feature_vector: np.ndarray, label: float, *, weight: float = 1.0) -> float:
        target = _clip(label, 0.0, 1.0)
        prediction = self.predict_encoded(feature_vector)
        error = target - prediction.probability
        self.weights *= 1.0 - self.learning_rate * self.weight_decay
        self.weights += (self.learning_rate * float(weight) * error * feature_vector).astype(np.float32)
        self.updates += 1
        eps = 1e-6
        loss = -(target * np.log(max(prediction.probability, eps)) + ((1.0 - target) * np.log(max(1.0 - prediction.probability, eps))))
        return float(loss)

    def update_preference(
        self,
        preferred_vector: np.ndarray,
        dispreferred_vector: np.ndarray,
        *,
        weight: float = 1.0,
    ) -> float:
        diff = preferred_vector - dispreferred_vector
        margin = float(np.dot(self.weights, diff))
        probability = _sigmoid(margin)
        error = 1.0 - probability
        self.weights *= 1.0 - self.learning_rate * self.weight_decay
        self.weights += (self.learning_rate * float(weight) * error * diff).astype(np.float32)
        self.updates += 1
        return float(-np.log(max(probability, 1e-6)))
