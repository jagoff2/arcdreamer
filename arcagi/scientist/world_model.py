"""Small online recurrent world model with uncertainty via bootstrap ensemble.

The scientist path needs a sequence model, not just one-step tabular action
statistics.  This module keeps the old hashed linear heads for checkpoint
compatibility, then adds a compact trainable recurrent state.  The recurrent
state is updated online from the agent's own transitions and contributes to
reward/change predictions for planning.
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
    """Bootstrap SGD ensemble with a compact recurrent online state.

    The model is deliberately small enough to update every environment step on
    CPU.  It is not a pretrained policy.  It learns from the current run:

    - hashed structured state/action features give immediate generalization;
    - a recurrent hidden state carries sequence context such as mode switches;
    - bootstrap heads expose disagreement as uncertainty;
    - one-step truncated gradients update recurrent/input weights online.

    ``reward_w`` and ``change_w`` are preserved so legacy checkpoints and tests
    can still load and compare the old linear component.
    """

    def __init__(
        self,
        *,
        feature_dim: int = 320,
        ensemble_size: int = 7,
        recurrent_dim: int = 48,
        learning_rate: float = 0.08,
        recurrent_learning_rate: float = 0.01,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.ensemble_size = int(ensemble_size)
        self.recurrent_dim = int(recurrent_dim)
        self.learning_rate = float(learning_rate)
        self.recurrent_learning_rate = float(recurrent_learning_rate)
        self.weight_decay = float(weight_decay)
        self.rng = np.random.default_rng(seed)
        scale = 0.015
        self.reward_w = self.rng.normal(0.0, scale, size=(self.ensemble_size, self.feature_dim)).astype(np.float32)
        self.change_w = self.rng.normal(0.0, scale, size=(self.ensemble_size, self.feature_dim)).astype(np.float32)
        self.recurrent_input_w = self.rng.normal(
            0.0,
            0.06 / np.sqrt(max(self.feature_dim, 1)),
            size=(self.recurrent_dim, self.feature_dim),
        ).astype(np.float32)
        self.recurrent_w = self.rng.normal(
            0.0,
            0.05 / np.sqrt(max(self.recurrent_dim, 1)),
            size=(self.recurrent_dim, self.recurrent_dim),
        ).astype(np.float32)
        self.recurrent_bias = np.zeros(self.recurrent_dim, dtype=np.float32)
        self.reward_recurrent_w = self.rng.normal(
            0.0,
            scale,
            size=(self.ensemble_size, self.recurrent_dim),
        ).astype(np.float32)
        self.change_recurrent_w = self.rng.normal(
            0.0,
            scale,
            size=(self.ensemble_size, self.recurrent_dim),
        ).astype(np.float32)
        self.hidden = np.zeros(self.recurrent_dim, dtype=np.float32)
        self.updates = 0

    def state_dict(self) -> dict[str, object]:
        return {
            "model_schema_version": 2,
            "feature_dim": self.feature_dim,
            "ensemble_size": self.ensemble_size,
            "recurrent_dim": self.recurrent_dim,
            "learning_rate": self.learning_rate,
            "recurrent_learning_rate": self.recurrent_learning_rate,
            "weight_decay": self.weight_decay,
            "updates": self.updates,
            "reward_w": self.reward_w.copy(),
            "change_w": self.change_w.copy(),
            "recurrent_input_w": self.recurrent_input_w.copy(),
            "recurrent_w": self.recurrent_w.copy(),
            "recurrent_bias": self.recurrent_bias.copy(),
            "reward_recurrent_w": self.reward_recurrent_w.copy(),
            "change_recurrent_w": self.change_recurrent_w.copy(),
            "hidden": self.hidden.copy(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        feature_dim = int(state["feature_dim"])
        ensemble_size = int(state["ensemble_size"])
        reward_w = np.asarray(state["reward_w"], dtype=np.float32)
        change_w = np.asarray(state["change_w"], dtype=np.float32)
        if reward_w.shape != (ensemble_size, feature_dim):
            raise ValueError(
                f"scientist world-model reward weights shape mismatch: expected {(ensemble_size, feature_dim)}, got {reward_w.shape}"
            )
        if change_w.shape != (ensemble_size, feature_dim):
            raise ValueError(
                f"scientist world-model change weights shape mismatch: expected {(ensemble_size, feature_dim)}, got {change_w.shape}"
            )
        self.feature_dim = feature_dim
        self.ensemble_size = ensemble_size
        self.recurrent_dim = int(state.get("recurrent_dim", getattr(self, "recurrent_dim", 48)))
        self.learning_rate = float(state.get("learning_rate", self.learning_rate))
        self.recurrent_learning_rate = float(state.get("recurrent_learning_rate", getattr(self, "recurrent_learning_rate", 0.01)))
        self.weight_decay = float(state.get("weight_decay", self.weight_decay))
        self.reward_w = reward_w.copy()
        self.change_w = change_w.copy()
        self._load_or_init_recurrent_state(state)
        self.updates = int(state.get("updates", 0))

    def reset_episode(self, *, keep_weights: bool = True) -> None:
        self.hidden = np.zeros(self.recurrent_dim, dtype=np.float32)
        self.updates = 0
        if not keep_weights:
            scale = 0.015
            self.reward_w[:] = self.rng.normal(0.0, scale, size=self.reward_w.shape)
            self.change_w[:] = self.rng.normal(0.0, scale, size=self.change_w.shape)
            self.recurrent_input_w[:] = self.rng.normal(
                0.0,
                0.06 / np.sqrt(max(self.feature_dim, 1)),
                size=self.recurrent_input_w.shape,
            )
            self.recurrent_w[:] = self.rng.normal(
                0.0,
                0.05 / np.sqrt(max(self.recurrent_dim, 1)),
                size=self.recurrent_w.shape,
            )
            self.recurrent_bias[:] = 0.0
            self.reward_recurrent_w[:] = self.rng.normal(0.0, scale, size=self.reward_recurrent_w.shape)
            self.change_recurrent_w[:] = self.rng.normal(0.0, scale, size=self.change_recurrent_w.shape)

    def predict(self, state: StructuredState, action: ActionName) -> WorldPrediction:
        x = state_action_features(state, action, dim=self.feature_dim)
        candidate_hidden = self._candidate_hidden(x)
        reward_preds = (self.reward_w @ x) + (self.reward_recurrent_w @ candidate_hidden)
        change_preds = (self.change_w @ x) + (self.change_recurrent_w @ candidate_hidden)
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
        previous_hidden = self.hidden.copy()
        candidate_hidden = self._candidate_hidden(x)
        reward_target, change_target = transition_target_features(record)
        reward_target = float(np.tanh(reward_target))
        change_target = float(change_target)
        losses: list[float] = []
        reward_errors: list[float] = []
        change_errors: list[float] = []
        for idx in range(self.ensemble_size):
            # Bootstrap mask keeps member disagreement meaningful.
            if self.rng.random() < 0.80:
                pred = float((self.reward_w[idx] @ x) + (self.reward_recurrent_w[idx] @ candidate_hidden))
                err = _clip(reward_target - pred, -2.0, 2.0)
                self.reward_w[idx] *= 1.0 - self.learning_rate * self.weight_decay
                self.reward_recurrent_w[idx] *= 1.0 - self.learning_rate * self.weight_decay
                self.reward_w[idx] += self.learning_rate * err * x
                self.reward_recurrent_w[idx] += self.learning_rate * err * candidate_hidden
                losses.append(err * err)
                reward_errors.append(err)
            if self.rng.random() < 0.80:
                pred = float((self.change_w[idx] @ x) + (self.change_recurrent_w[idx] @ candidate_hidden))
                err = _clip(change_target - pred, -2.0, 2.0)
                self.change_w[idx] *= 1.0 - self.learning_rate * self.weight_decay
                self.change_recurrent_w[idx] *= 1.0 - self.learning_rate * self.weight_decay
                self.change_w[idx] += self.learning_rate * err * x
                self.change_recurrent_w[idx] += self.learning_rate * err * candidate_hidden
                losses.append(err * err)
                change_errors.append(err)
        self._update_recurrent_weights(
            x,
            candidate_hidden=candidate_hidden,
            previous_hidden=previous_hidden,
            reward_error=float(np.mean(reward_errors)) if reward_errors else 0.0,
            change_error=float(np.mean(change_errors)) if change_errors else 0.0,
        )
        self.hidden = candidate_hidden.astype(np.float32, copy=False)
        self.updates += 1
        return float(np.mean(losses)) if losses else 0.0

    def diagnostics(self) -> dict[str, float | int | str]:
        return {
            "model": "online_recurrent_bootstrap",
            "model_schema_version": 2,
            "updates": int(self.updates),
            "feature_dim": int(self.feature_dim),
            "recurrent_dim": int(self.recurrent_dim),
            "hidden_norm": float(np.linalg.norm(self.hidden)),
            "ensemble_size": int(self.ensemble_size),
        }

    def _candidate_hidden(self, x: np.ndarray) -> np.ndarray:
        pre = (self.recurrent_input_w @ x) + (self.recurrent_w @ self.hidden) + self.recurrent_bias
        return np.tanh(np.clip(pre, -8.0, 8.0)).astype(np.float32)

    def _update_recurrent_weights(
        self,
        x: np.ndarray,
        *,
        candidate_hidden: np.ndarray,
        previous_hidden: np.ndarray,
        reward_error: float,
        change_error: float,
    ) -> None:
        if self.recurrent_dim <= 0:
            return
        reward_credit = np.mean(self.reward_recurrent_w, axis=0) * float(reward_error)
        change_credit = np.mean(self.change_recurrent_w, axis=0) * float(change_error)
        hidden_error = np.clip(reward_credit + 0.5 * change_credit, -1.0, 1.0).astype(np.float32)
        gate_grad = (1.0 - np.square(candidate_hidden)).astype(np.float32)
        grad_pre = np.clip(hidden_error * gate_grad, -0.25, 0.25)
        lr = self.recurrent_learning_rate
        wd = 1.0 - lr * self.weight_decay
        self.recurrent_input_w *= wd
        self.recurrent_w *= wd
        self.recurrent_bias *= wd
        self.recurrent_input_w += lr * np.outer(grad_pre, x).astype(np.float32)
        self.recurrent_w += lr * np.outer(grad_pre, previous_hidden).astype(np.float32)
        self.recurrent_bias += lr * grad_pre

    def _load_or_init_recurrent_state(self, state: dict[str, object]) -> None:
        expected = (self.recurrent_dim, self.feature_dim)
        recurrent_input_w = state.get("recurrent_input_w")
        recurrent_w = state.get("recurrent_w")
        reward_recurrent_w = state.get("reward_recurrent_w")
        change_recurrent_w = state.get("change_recurrent_w")
        if recurrent_input_w is None or recurrent_w is None or reward_recurrent_w is None or change_recurrent_w is None:
            self._init_missing_recurrent_weights(preserve_linear_predictions=True)
            return

        self.recurrent_input_w = np.asarray(recurrent_input_w, dtype=np.float32).copy()
        self.recurrent_w = np.asarray(recurrent_w, dtype=np.float32).copy()
        self.reward_recurrent_w = np.asarray(reward_recurrent_w, dtype=np.float32).copy()
        self.change_recurrent_w = np.asarray(change_recurrent_w, dtype=np.float32).copy()
        if self.recurrent_input_w.shape != expected:
            raise ValueError(
                f"scientist recurrent input weights shape mismatch: expected {expected}, got {self.recurrent_input_w.shape}"
            )
        if self.recurrent_w.shape != (self.recurrent_dim, self.recurrent_dim):
            raise ValueError(
                "scientist recurrent weights shape mismatch: "
                f"expected {(self.recurrent_dim, self.recurrent_dim)}, got {self.recurrent_w.shape}"
            )
        if self.reward_recurrent_w.shape != (self.ensemble_size, self.recurrent_dim):
            raise ValueError(
                "scientist recurrent reward weights shape mismatch: "
                f"expected {(self.ensemble_size, self.recurrent_dim)}, got {self.reward_recurrent_w.shape}"
            )
        if self.change_recurrent_w.shape != (self.ensemble_size, self.recurrent_dim):
            raise ValueError(
                "scientist recurrent change weights shape mismatch: "
                f"expected {(self.ensemble_size, self.recurrent_dim)}, got {self.change_recurrent_w.shape}"
            )
        self.recurrent_bias = np.asarray(state.get("recurrent_bias", np.zeros(self.recurrent_dim)), dtype=np.float32).copy()
        if self.recurrent_bias.shape != (self.recurrent_dim,):
            raise ValueError(
                f"scientist recurrent bias shape mismatch: expected {(self.recurrent_dim,)}, got {self.recurrent_bias.shape}"
            )
        hidden = np.asarray(state.get("hidden", np.zeros(self.recurrent_dim)), dtype=np.float32).copy()
        self.hidden = hidden if hidden.shape == (self.recurrent_dim,) else np.zeros(self.recurrent_dim, dtype=np.float32)

    def _init_missing_recurrent_weights(self, *, preserve_linear_predictions: bool = False) -> None:
        scale = 0.015
        self.recurrent_input_w = self.rng.normal(
            0.0,
            0.06 / np.sqrt(max(self.feature_dim, 1)),
            size=(self.recurrent_dim, self.feature_dim),
        ).astype(np.float32)
        self.recurrent_w = self.rng.normal(
            0.0,
            0.05 / np.sqrt(max(self.recurrent_dim, 1)),
            size=(self.recurrent_dim, self.recurrent_dim),
        ).astype(np.float32)
        self.recurrent_bias = np.zeros(self.recurrent_dim, dtype=np.float32)
        if preserve_linear_predictions:
            self.reward_recurrent_w = np.zeros((self.ensemble_size, self.recurrent_dim), dtype=np.float32)
            self.change_recurrent_w = np.zeros((self.ensemble_size, self.recurrent_dim), dtype=np.float32)
        else:
            self.reward_recurrent_w = self.rng.normal(
                0.0,
                scale,
                size=(self.ensemble_size, self.recurrent_dim),
            ).astype(np.float32)
            self.change_recurrent_w = self.rng.normal(
                0.0,
                scale,
                size=(self.ensemble_size, self.recurrent_dim),
            ).astype(np.float32)
        self.hidden = np.zeros(self.recurrent_dim, dtype=np.float32)


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))
