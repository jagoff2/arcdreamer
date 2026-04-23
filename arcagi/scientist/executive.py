"""Learned executive scorer over explicit workspace features.

This module does not replace the explicit spotlight protocol.  It replaces the
hand-weighted action combiner with a small learned model that scores candidate
actions from structured workspace state.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import numpy as np

from .features import add_hash_feature, normalize


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


@dataclass(frozen=True)
class ExecutivePrediction:
    value_mean: float
    value_uncertainty: float

    @property
    def optimistic_value(self) -> float:
        return float(self.value_mean + self.value_uncertainty)


def encode_feature_map(feature_map: Mapping[str, float], *, feature_dim: int) -> np.ndarray:
    vec = np.zeros(int(feature_dim), dtype=np.float32)
    vec[0] = 1.0
    for key, raw_value in sorted(feature_map.items()):
        value = float(raw_value)
        vec[1] += 0.01 * np.tanh(value)
        add_hash_feature(vec, f"feat:{key}", value)
        add_hash_feature(vec, f"feat_sign:{key}:{'pos' if value >= 0.0 else 'neg'}", abs(value) ** 0.5)
        bucket = int(round(_clip(value, -2.0, 2.0) * 4.0))
        add_hash_feature(vec, f"feat_bucket:{key}:{bucket}", 0.35)
    return normalize(vec)


class ExecutiveScorer:
    """Tiny online ensemble used to score spotlight candidates.

    The scorer consumes explicit feature maps produced by the spotlight
    workspace, then updates from real transitions with a TD-style target.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 384,
        ensemble_size: int = 5,
        learning_rate: float = 0.035,
        weight_decay: float = 1e-4,
        gamma: float = 0.92,
        exploration_bonus: float = 0.30,
        seed: int = 0,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.ensemble_size = int(ensemble_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.gamma = float(gamma)
        self.exploration_bonus = float(exploration_bonus)
        self.rng = np.random.default_rng(seed)
        scale = 0.035
        self.weights = self.rng.normal(0.0, scale, size=(self.ensemble_size, self.feature_dim)).astype(np.float32)
        self.updates = 0

    def state_dict(self) -> dict[str, object]:
        return {
            "feature_dim": self.feature_dim,
            "ensemble_size": self.ensemble_size,
            "learning_rate": self.learning_rate,
            "weight_decay": self.weight_decay,
            "gamma": self.gamma,
            "exploration_bonus": self.exploration_bonus,
            "updates": self.updates,
            "weights": self.weights.copy(),
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        feature_dim = int(state.get("feature_dim", self.feature_dim))
        ensemble_size = int(state.get("ensemble_size", self.ensemble_size))
        weights = np.asarray(state.get("weights"), dtype=np.float32)
        if weights.shape != (ensemble_size, feature_dim):
            raise ValueError(
                f"executive scorer weights shape mismatch: expected {(ensemble_size, feature_dim)}, got {weights.shape}"
            )
        self.feature_dim = feature_dim
        self.ensemble_size = ensemble_size
        self.learning_rate = float(state.get("learning_rate", self.learning_rate))
        self.weight_decay = float(state.get("weight_decay", self.weight_decay))
        self.gamma = float(state.get("gamma", self.gamma))
        self.exploration_bonus = float(state.get("exploration_bonus", self.exploration_bonus))
        self.updates = int(state.get("updates", 0))
        self.weights = weights.copy()

    def encode(self, feature_map: Mapping[str, float]) -> np.ndarray:
        return encode_feature_map(feature_map, feature_dim=self.feature_dim)

    def predict_encoded(self, feature_vector: np.ndarray) -> ExecutivePrediction:
        preds = self.weights @ feature_vector
        mean = float(np.mean(preds))
        uncertainty = float(np.var(preds) + (0.75 / np.sqrt(max(self.updates, 1))))
        return ExecutivePrediction(
            value_mean=float(np.tanh(mean)),
            value_uncertainty=float(uncertainty),
        )

    def predict(self, feature_map: Mapping[str, float]) -> tuple[np.ndarray, ExecutivePrediction]:
        feature_vector = self.encode(feature_map)
        return feature_vector, self.predict_encoded(feature_vector)

    def score(self, prediction: ExecutivePrediction) -> float:
        return float(prediction.value_mean + (self.exploration_bonus * prediction.value_uncertainty))

    def update(self, feature_vector: np.ndarray, target: float) -> float:
        clipped_target = _clip(target, -2.0, 2.0)
        losses: list[float] = []
        for idx in range(self.ensemble_size):
            if self.rng.random() >= 0.85:
                continue
            pred = float(self.weights[idx] @ feature_vector)
            err = _clip(clipped_target - pred, -3.0, 3.0)
            self.weights[idx] *= 1.0 - self.learning_rate * self.weight_decay
            self.weights[idx] += self.learning_rate * err * feature_vector
            losses.append(err * err)
        self.updates += 1
        return float(np.mean(losses)) if losses else 0.0
