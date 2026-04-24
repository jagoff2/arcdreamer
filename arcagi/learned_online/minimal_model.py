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
    value: np.ndarray
    imitation: np.ndarray
    uncertainty: np.ndarray


@dataclass
class MinimalOnlineModel:
    input_dim: int
    learning_rate: float = 0.18
    weights: dict[str, np.ndarray] = field(init=False)
    biases: dict[str, float] = field(init=False)
    updates: int = 0
    pretrain_updates: int = 0
    online_adapt_updates: int = 0

    def __post_init__(self) -> None:
        self.weights = {name: np.zeros((self.input_dim,), dtype=np.float32) for name in _PREDICTION_HEADS}
        self.biases = {
            "reward": -2.0,
            "useful": -0.8,
            "visible": -0.4,
            "info_gain": -0.2,
            "cost": -1.2,
            "value": -1.5,
            "imitation": -6.0,
        }
        self._initialize_generic_belief_priors()

    def predict(self, features: np.ndarray) -> MinimalPredictions:
        x = np.asarray(features, dtype=np.float32)
        reward = _sigmoid(x @ self.weights["reward"] + self.biases["reward"])
        useful = _sigmoid(x @ self.weights["useful"] + self.biases["useful"])
        visible = _sigmoid(x @ self.weights["visible"] + self.biases["visible"])
        info_gain = _sigmoid(x @ self.weights["info_gain"] + self.biases["info_gain"])
        cost = _sigmoid(x @ self.weights["cost"] + self.biases["cost"])
        value = _sigmoid(x @ self.weights["value"] + self.biases["value"])
        imitation = _sigmoid(x @ self.weights["imitation"] + self.biases["imitation"])
        uncertainty = np.clip((1.0 - np.maximum(useful, cost)) + 0.25 * info_gain, 0.0, 1.0)
        return MinimalPredictions(
            reward=reward.astype(np.float32),
            useful=useful.astype(np.float32),
            visible=visible.astype(np.float32),
            info_gain=info_gain.astype(np.float32),
            cost=cost.astype(np.float32),
            value=value.astype(np.float32),
            imitation=imitation.astype(np.float32),
            uncertainty=uncertainty.astype(np.float32),
        )

    def prediction_loss(
        self,
        feature: np.ndarray,
        labels: TransitionLabels,
        *,
        realized_info_gain: float,
        return_credit: float = 0.0,
    ) -> float:
        pred = self.predict(np.asarray(feature, dtype=np.float32)[None, :])
        targets = _targets(labels, realized_info_gain=realized_info_gain, return_credit=return_credit)
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
            return_credit = float(getattr(entry, "return_credit", 0.0) or 0.0)
            losses.append(
                self.prediction_loss(
                    np.asarray(feature, dtype=np.float32),
                    labels,
                    realized_info_gain=realized_info_gain,
                    return_credit=return_credit,
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
        return_credit: float = 0.0,
    ) -> float:
        x = np.asarray(feature, dtype=np.float32)
        targets = _targets(labels, realized_info_gain=realized_info_gain, return_credit=return_credit)
        pred = self.predict(x[None, :])
        loss = 0.0
        rate = self.learning_rate / float((self.online_adapt_updates + 1) ** 0.5)
        for name, target in targets.items():
            value = float(getattr(pred, name)[0])
            error = float(target) - value
            loss += error * error
            self.weights[name] += np.asarray(rate * error * x, dtype=np.float32)
            self.biases[name] += rate * error
        self.online_adapt_updates += 1
        self.updates += 1
        return float(loss / max(float(len(targets)), 1.0))

    def reset_online_adaptation(self) -> None:
        self.online_adapt_updates = 0

    def value_logit(self, feature: np.ndarray) -> float:
        x = np.asarray(feature, dtype=np.float32)
        return float(x @ self.weights["value"] + self.biases["value"])

    def ranking_update_value(
        self,
        positive_feature: np.ndarray,
        negative_feature: np.ndarray,
        *,
        margin: float = 0.10,
        weight: float = 0.25,
    ) -> float:
        positive = np.asarray(positive_feature, dtype=np.float32)
        negative = np.asarray(negative_feature, dtype=np.float32)
        violation = float(margin + self.value_logit(negative) - self.value_logit(positive))
        if violation <= 0.0:
            return 0.0
        rate = float(weight) * self.learning_rate / float((self.online_adapt_updates + 1) ** 0.5)
        self.weights["value"] += np.asarray(rate * (positive - negative), dtype=np.float32)
        self.online_adapt_updates += 1
        self.updates += 1
        return violation

    def imitation_logit(self, feature: np.ndarray) -> float:
        x = np.asarray(feature, dtype=np.float32)
        return float(x @ self.weights["imitation"] + self.biases["imitation"])

    def imitation_update(
        self,
        positive_feature: np.ndarray,
        negative_features: np.ndarray,
        *,
        margin: float = 0.20,
        weight: float = 0.75,
    ) -> float:
        positive = np.asarray(positive_feature, dtype=np.float32)
        negatives = np.asarray(negative_features, dtype=np.float32)
        if negatives.ndim == 1:
            negatives = negatives[None, :]
        if negatives.size == 0:
            return 0.0
        positive_logit = self.imitation_logit(positive)
        negative_logits = negatives @ self.weights["imitation"] + self.biases["imitation"]
        violations = margin + negative_logits - positive_logit
        active = violations > 0.0
        if not bool(np.any(active)):
            return 0.0
        active_negatives = negatives[active]
        rate = float(weight) * self.learning_rate / float((self.online_adapt_updates + 1) ** 0.5)
        gradient = positive - np.mean(active_negatives, axis=0)
        self.weights["imitation"] += np.asarray(rate * gradient, dtype=np.float32)
        self.online_adapt_updates += 1
        self.updates += 1
        return float(np.mean(violations[active]))

    def state_dict(self) -> dict[str, object]:
        return {
            "input_dim": int(self.input_dim),
            "learning_rate": float(self.learning_rate),
            "weights": {key: value.copy() for key, value in self.weights.items()},
            "biases": dict(self.biases),
            "updates": int(self.updates),
            "pretrain_updates": int(max(self.pretrain_updates, self.updates)),
            "online_adapt_updates": int(self.online_adapt_updates),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        weights = state.get("weights", {})
        if isinstance(weights, dict):
            for key in _PREDICTION_HEADS:
                if key in weights:
                    loaded = np.asarray(weights[key], dtype=np.float32).copy()
                    target = self.weights[key]
                    target.fill(0.0)
                    overlap = min(int(target.shape[0]), int(loaded.shape[0]))
                    if overlap > 0:
                        target[:overlap] = loaded[:overlap]
        biases = state.get("biases", {})
        if isinstance(biases, dict):
            for key in _PREDICTION_HEADS:
                if key in biases:
                    self.biases[key] = float(biases[key])
        self.updates = int(state.get("updates", self.updates) or 0)
        self.pretrain_updates = int(state.get("pretrain_updates", self.updates) or 0)
        self.online_adapt_updates = 0

    def _initialize_generic_belief_priors(self) -> None:
        if self.input_dim < ACTION_FEATURE_DIM + BELIEF_FEATURE_DIM + MEMORY_FEATURE_DIM:
            return
        belief = ACTION_FEATURE_DIM
        memory = ACTION_FEATURE_DIM + BELIEF_FEATURE_DIM
        family = belief
        exact = belief + 7
        context = belief + 14
        level_family = belief + 21
        level_exact = belief + 28
        level_context = belief + 35
        stat_blocks = (
            (family, 0.25, 0.70),
            (exact, 0.70, 1.70),
            (context, 0.50, 1.35),
            (level_family, 0.35, 1.10),
            (level_exact, 1.00, 2.60),
            (level_context, 0.75, 2.05),
        )
        for offset, value_scale, cost_scale in stat_blocks:
            self.weights["visible"][offset + 1] += value_scale
            self.weights["useful"][offset + 2] += value_scale
            self.weights["reward"][offset + 3] += value_scale
            self.weights["value"][offset + 3] += value_scale
            self.weights["info_gain"][offset + 4] += 0.40 * value_scale
            self.weights["cost"][offset + 5] += cost_scale
            self.weights["info_gain"][offset + 6] += 0.25 * value_scale
        self.weights["useful"][memory + 1] += 0.55
        self.weights["visible"][memory + 2] += 0.35
        self.weights["cost"][memory + 3] += 0.55
        self.weights["reward"][memory + 4] += 0.45
        self.weights["value"][memory + 4] += 0.45
        self.weights["info_gain"][memory + 5] += 0.45


_UPDATE_HEADS: tuple[str, ...] = ("reward", "useful", "visible", "info_gain", "cost", "value")
_PREDICTION_HEADS: tuple[str, ...] = (*_UPDATE_HEADS, "imitation")


def _targets(labels: TransitionLabels, *, realized_info_gain: float, return_credit: float = 0.0) -> dict[str, float]:
    nonprogress = max(float(labels.visible_only_nonprogress), float(labels.no_effect_nonprogress))
    return {
        "reward": float(labels.reward_progress),
        "useful": float(labels.useful_change),
        "visible": float(labels.visible_change),
        "info_gain": float(max(0.0, min(1.0, realized_info_gain))),
        "cost": float(max(float(labels.harm), 0.25 * nonprogress)),
        "value": float(max(float(labels.reward_progress), float(return_credit))),
    }


def _sigmoid(value: np.ndarray) -> np.ndarray:
    clipped = np.clip(value, -30.0, 30.0)
    return 1.0 / (1.0 + np.exp(-clipped))
