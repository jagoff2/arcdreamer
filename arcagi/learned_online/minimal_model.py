from __future__ import annotations

from dataclasses import dataclass, field
from typing import Sequence

import numpy as np

from arcagi.learned_online.action_features import ACTION_FEATURE_DIM
from arcagi.learned_online.fast_belief import BELIEF_FEATURE_DIM
from arcagi.learned_online.memory import MEMORY_FEATURE_DIM
from arcagi.learned_online.signals import TransitionLabels


@dataclass(frozen=True)
class MinimalPredictions:
    reward: np.ndarray
    useful: np.ndarray
    visible: np.ndarray
    info_gain: np.ndarray
    cost: np.ndarray
    uncertainty: np.ndarray


@dataclass
class MinimalOnlineModel:
    input_dim: int
    learning_rate: float = 0.18
    weights: dict[str, np.ndarray] = field(init=False)
    biases: dict[str, float] = field(init=False)
    updates: int = 0

    def __post_init__(self) -> None:
        self.weights = {name: np.zeros((self.input_dim,), dtype=np.float32) for name in _HEADS}
        self.biases = {
            "reward": -2.0,
            "useful": -0.8,
            "visible": -0.4,
            "info_gain": -0.2,
            "cost": -1.2,
        }
        self._initialize_generic_belief_priors()

    def predict(self, features: np.ndarray) -> MinimalPredictions:
        x = np.asarray(features, dtype=np.float32)
        reward = _sigmoid(x @ self.weights["reward"] + self.biases["reward"])
        useful = _sigmoid(x @ self.weights["useful"] + self.biases["useful"])
        visible = _sigmoid(x @ self.weights["visible"] + self.biases["visible"])
        info_gain = _sigmoid(x @ self.weights["info_gain"] + self.biases["info_gain"])
        cost = _sigmoid(x @ self.weights["cost"] + self.biases["cost"])
        uncertainty = np.clip((1.0 - np.maximum(useful, cost)) + 0.25 * info_gain, 0.0, 1.0)
        return MinimalPredictions(
            reward=reward.astype(np.float32),
            useful=useful.astype(np.float32),
            visible=visible.astype(np.float32),
            info_gain=info_gain.astype(np.float32),
            cost=cost.astype(np.float32),
            uncertainty=uncertainty.astype(np.float32),
        )

    def prediction_loss(self, feature: np.ndarray, labels: TransitionLabels, *, realized_info_gain: float) -> float:
        pred = self.predict(np.asarray(feature, dtype=np.float32)[None, :])
        targets = _targets(labels, realized_info_gain=realized_info_gain)
        loss = 0.0
        for name, target in targets.items():
            value = float(getattr(pred, name)[0])
            loss += (value - float(target)) ** 2
        return float(loss / max(float(len(targets)), 1.0))

    def batch_prediction_loss(self, entries: Sequence[object]) -> float:
        losses: list[float] = []
        for entry in entries:
            feature = getattr(entry, "feature", None)
            labels = getattr(entry, "labels", None)
            if feature is None or labels is None:
                continue
            realized_info_gain = float(getattr(entry, "realized_info_gain", 0.0) or 0.0)
            losses.append(
                self.prediction_loss(
                    np.asarray(feature, dtype=np.float32),
                    labels,
                    realized_info_gain=realized_info_gain,
                )
            )
        if not losses:
            return 0.0
        return float(np.mean(losses))

    def online_update(
        self,
        feature: np.ndarray,
        labels: TransitionLabels,
        *,
        realized_info_gain: float = 0.0,
    ) -> float:
        x = np.asarray(feature, dtype=np.float32)
        targets = _targets(labels, realized_info_gain=realized_info_gain)
        pred = self.predict(x[None, :])
        loss = 0.0
        for name, target in targets.items():
            value = float(getattr(pred, name)[0])
            error = float(target) - value
            loss += error * error
            rate = self.learning_rate / float((self.updates + 1) ** 0.5)
            self.weights[name] += np.asarray(rate * error * x, dtype=np.float32)
            self.biases[name] += rate * error
        self.updates += 1
        return float(loss / max(float(len(targets)), 1.0))

    def state_dict(self) -> dict[str, object]:
        return {
            "input_dim": int(self.input_dim),
            "learning_rate": float(self.learning_rate),
            "weights": {key: value.copy() for key, value in self.weights.items()},
            "biases": dict(self.biases),
            "updates": int(self.updates),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        weights = state.get("weights", {})
        if isinstance(weights, dict):
            for key in _HEADS:
                if key in weights:
                    self.weights[key] = np.asarray(weights[key], dtype=np.float32).copy()
        biases = state.get("biases", {})
        if isinstance(biases, dict):
            for key in _HEADS:
                if key in biases:
                    self.biases[key] = float(biases[key])
        self.updates = int(state.get("updates", self.updates) or 0)

    def _initialize_generic_belief_priors(self) -> None:
        if self.input_dim < ACTION_FEATURE_DIM + BELIEF_FEATURE_DIM + MEMORY_FEATURE_DIM:
            return
        belief = ACTION_FEATURE_DIM
        memory = ACTION_FEATURE_DIM + BELIEF_FEATURE_DIM
        family = belief
        exact = belief + 7
        context = belief + 14
        for offset, value_scale, cost_scale in ((family, 0.30, 0.85), (exact, 0.85, 2.10), (context, 0.60, 1.65)):
            self.weights["visible"][offset + 1] += value_scale
            self.weights["useful"][offset + 2] += value_scale
            self.weights["reward"][offset + 3] += value_scale
            self.weights["info_gain"][offset + 4] += 0.40 * value_scale
            self.weights["cost"][offset + 5] += cost_scale
            self.weights["info_gain"][offset + 6] += 0.25 * value_scale
        self.weights["useful"][memory + 1] += 0.55
        self.weights["visible"][memory + 2] += 0.35
        self.weights["cost"][memory + 3] += 0.55
        self.weights["reward"][memory + 4] += 0.45
        self.weights["info_gain"][memory + 5] += 0.45


_HEADS: tuple[str, ...] = ("reward", "useful", "visible", "info_gain", "cost")


def _targets(labels: TransitionLabels, *, realized_info_gain: float) -> dict[str, float]:
    return {
        "reward": float(labels.reward_progress),
        "useful": float(labels.useful_change),
        "visible": float(labels.visible_change),
        "info_gain": float(max(0.0, min(1.0, realized_info_gain))),
        "cost": float(max(labels.visible_only_nonprogress, labels.no_effect_nonprogress, labels.harm)),
    }


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))
