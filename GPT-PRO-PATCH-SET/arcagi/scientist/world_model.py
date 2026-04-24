"""Small online world model with uncertainty via bootstrap ensemble.

The model is intentionally tiny: linear heads over hashed structured features,
updated after every transition.  It is not meant to solve ARC-AGI-3 alone; it
supplies calibrated uncertainty and weak reward/change predictions to the
experiment planner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import state_action_features, transition_target_features
from .types import ActionName, StructuredState, TransitionRecord


@dataclass(frozen=True)
class WorldPrediction:
    reward_mean: float
    reward_uncertainty: float
    change_mean: float
    change_uncertainty: float

    @property
    def total_uncertainty(self) -> float:
        return float(self.reward_uncertainty + 0.5 * self.change_uncertainty)


class OnlineWorldModel:
    """Bootstrap SGD ensemble for in-episode reward/change prediction."""

    def __init__(
        self,
        *,
        feature_dim: int = 320,
        ensemble_size: int = 7,
        learning_rate: float = 0.08,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.ensemble_size = int(ensemble_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.rng = np.random.default_rng(seed)
        scale = 0.015
        self.reward_w = self.rng.normal(0.0, scale, size=(self.ensemble_size, self.feature_dim)).astype(np.float32)
        self.change_w = self.rng.normal(0.0, scale, size=(self.ensemble_size, self.feature_dim)).astype(np.float32)
        self.updates = 0

    def reset_episode(self, *, keep_weights: bool = True) -> None:
        self.updates = 0
        if not keep_weights:
            scale = 0.015
            self.reward_w[:] = self.rng.normal(0.0, scale, size=self.reward_w.shape)
            self.change_w[:] = self.rng.normal(0.0, scale, size=self.change_w.shape)

    def predict(self, state: StructuredState, action: ActionName) -> WorldPrediction:
        x = state_action_features(state, action, dim=self.feature_dim)
        reward_preds = self.reward_w @ x
        change_preds = self.change_w @ x
        reward_mean = float(np.mean(reward_preds))
        change_mean = float(np.mean(change_preds))
        return WorldPrediction(
            reward_mean=float(np.tanh(reward_mean)),
            reward_uncertainty=float(np.var(reward_preds) + 1.0 / np.sqrt(max(self.updates, 1))),
            change_mean=float(1.0 / (1.0 + np.exp(-change_mean))),
            change_uncertainty=float(np.var(change_preds) + 0.5 / np.sqrt(max(self.updates, 1))),
        )

    def update(self, record: TransitionRecord) -> float:
        x = state_action_features(record.before, record.action, dim=self.feature_dim)
        reward_target, change_target = transition_target_features(record)
        reward_target = float(np.tanh(reward_target))
        change_target = float(change_target)
        losses: list[float] = []
        for idx in range(self.ensemble_size):
            # Bootstrap mask keeps member disagreement meaningful.
            if self.rng.random() < 0.80:
                pred = float(self.reward_w[idx] @ x)
                err = _clip(reward_target - pred, -2.0, 2.0)
                self.reward_w[idx] *= 1.0 - self.learning_rate * self.weight_decay
                self.reward_w[idx] += self.learning_rate * err * x
                losses.append(err * err)
            if self.rng.random() < 0.80:
                pred = float(self.change_w[idx] @ x)
                err = _clip(change_target - pred, -2.0, 2.0)
                self.change_w[idx] *= 1.0 - self.learning_rate * self.weight_decay
                self.change_w[idx] += self.learning_rate * err * x
                losses.append(err * err)
        self.updates += 1
        return float(np.mean(losses)) if losses else 0.0


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))
