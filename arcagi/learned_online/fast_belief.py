from __future__ import annotations

from dataclasses import dataclass, field
import math

import numpy as np

from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.types import ActionName, StructuredState
from arcagi.learned_online.action_features import action_context_key
from arcagi.learned_online.signals import TransitionLabels

BELIEF_FEATURE_DIM = 49


@dataclass
class BeliefStats:
    trials: int = 0
    visible_sum: float = 0.0
    useful_sum: float = 0.0
    reward_sum: float = 0.0
    info_sum: float = 0.0
    cost_sum: float = 0.0

    def observe(self, labels: TransitionLabels, *, realized_info_gain: float) -> None:
        self.trials += 1
        self.visible_sum += float(labels.visible_change)
        self.useful_sum += float(labels.useful_change)
        self.reward_sum += float(labels.reward_progress)
        self.info_sum += float(max(realized_info_gain, 0.0))
        nonprogress = max(float(labels.visible_only_nonprogress), float(labels.no_effect_nonprogress))
        self.cost_sum += float(max(float(labels.harm), 0.25 * nonprogress))

    def features(self) -> list[float]:
        denom = max(float(self.trials), 1.0)
        return [
            math.tanh(float(self.trials) / 8.0),
            self.visible_sum / denom,
            self.useful_sum / denom,
            self.reward_sum / denom,
            self.info_sum / denom,
            self.cost_sum / denom,
            1.0 / math.sqrt(float(self.trials) + 1.0),
        ]


@dataclass
class OnlineBeliefState:
    family_stats: dict[str, BeliefStats] = field(default_factory=dict)
    exact_stats: dict[ActionName, BeliefStats] = field(default_factory=dict)
    context_stats: dict[str, BeliefStats] = field(default_factory=dict)
    level_family_stats: dict[str, BeliefStats] = field(default_factory=dict)
    level_exact_stats: dict[ActionName, BeliefStats] = field(default_factory=dict)
    level_context_stats: dict[str, BeliefStats] = field(default_factory=dict)
    online_update_count: int = 0
    level_epoch: int = 0
    level_step: int = 0
    recent_prediction_error: float = 0.0
    recent_visible_only_rate: float = 0.0
    recent_objective_rate: float = 0.0

    def reset(self) -> None:
        self.family_stats.clear()
        self.exact_stats.clear()
        self.context_stats.clear()
        self.level_family_stats.clear()
        self.level_exact_stats.clear()
        self.level_context_stats.clear()
        self.online_update_count = 0
        self.level_epoch = 0
        self.level_step = 0
        self.recent_prediction_error = 0.0
        self.recent_visible_only_rate = 0.0
        self.recent_objective_rate = 0.0

    def start_new_level(self) -> None:
        self.level_epoch += 1
        self.level_step = 0
        self.level_family_stats.clear()
        self.level_exact_stats.clear()
        self.level_context_stats.clear()
        self.recent_prediction_error *= 0.5
        self.recent_visible_only_rate *= 0.35
        self.recent_objective_rate = 0.0

    def features_for(self, state: StructuredState, action: ActionName) -> np.ndarray:
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        schema = build_action_schema(str(action), context)
        family_key = self.family_key(state, action)
        context_key = action_context_key(state, action)
        family_stats = self.family_stats.get(family_key, BeliefStats())
        context_stats = self.context_stats.get(context_key, BeliefStats())
        exact_stats = self.exact_stats.get(str(action), BeliefStats())
        level_family_stats = self.level_family_stats.get(family_key, BeliefStats())
        level_context_stats = self.level_context_stats.get(context_key, BeliefStats())
        level_exact_stats = self.level_exact_stats.get(str(action), BeliefStats())
        if schema.click is not None:
            exact_stats = context_stats if context_stats.trials > 0 else family_stats
            level_exact_stats = level_context_stats if level_context_stats.trials > 0 else level_family_stats
        values: list[float] = []
        values.extend(family_stats.features())
        values.extend(exact_stats.features())
        values.extend(context_stats.features())
        values.extend(level_family_stats.features())
        values.extend(level_exact_stats.features())
        values.extend(level_context_stats.features())
        values.extend(
            [
                math.tanh(float(self.online_update_count) / 32.0),
                math.tanh(float(self.level_epoch) / 8.0),
                math.tanh(float(self.level_step) / 64.0),
                float(self.recent_prediction_error),
                float(self.recent_visible_only_rate),
                float(self.recent_objective_rate),
                self.uncertainty_for(state, action),
            ]
        )
        if len(values) < BELIEF_FEATURE_DIM:
            values.extend([0.0] * (BELIEF_FEATURE_DIM - len(values)))
        return np.asarray(values[:BELIEF_FEATURE_DIM], dtype=np.float32)

    def observe(
        self,
        state: StructuredState,
        action: ActionName,
        labels: TransitionLabels,
        *,
        realized_info_gain: float,
        prediction_error: float,
    ) -> None:
        action = str(action)
        family_key = self.family_key(state, action)
        context_key = action_context_key(state, action)
        self.family_stats.setdefault(family_key, BeliefStats()).observe(labels, realized_info_gain=realized_info_gain)
        self.exact_stats.setdefault(action, BeliefStats()).observe(labels, realized_info_gain=realized_info_gain)
        self.context_stats.setdefault(context_key, BeliefStats()).observe(labels, realized_info_gain=realized_info_gain)
        self.level_family_stats.setdefault(family_key, BeliefStats()).observe(labels, realized_info_gain=realized_info_gain)
        self.level_exact_stats.setdefault(action, BeliefStats()).observe(labels, realized_info_gain=realized_info_gain)
        self.level_context_stats.setdefault(context_key, BeliefStats()).observe(labels, realized_info_gain=realized_info_gain)
        self.online_update_count += 1
        self.level_step += 1
        self.recent_prediction_error = _ema(self.recent_prediction_error, prediction_error, rate=0.35)
        self.recent_visible_only_rate = _ema(
            self.recent_visible_only_rate,
            float(labels.visible_only_nonprogress),
            rate=0.25,
        )
        self.recent_objective_rate = _ema(self.recent_objective_rate, float(labels.objective_progress), rate=0.25)

    def family_key(self, state: StructuredState, action: ActionName) -> str:
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        schema = build_action_schema(str(action), context)
        return f"{schema.action_type}:{schema.direction or 'none'}"

    def uncertainty_for(self, state: StructuredState, action: ActionName) -> float:
        schema_context = build_action_schema_context(state.affordances, dict(state.action_roles))
        schema = build_action_schema(str(action), schema_context)
        family = self.family_stats.get(self.family_key(state, action), BeliefStats())
        context = self.context_stats.get(action_context_key(state, action), BeliefStats())
        exact = self.exact_stats.get(str(action), BeliefStats())
        level_family = self.level_family_stats.get(self.family_key(state, action), BeliefStats())
        level_context = self.level_context_stats.get(action_context_key(state, action), BeliefStats())
        level_exact = self.level_exact_stats.get(str(action), BeliefStats())
        if schema.click is not None:
            exact = context if context.trials > 0 else family
            level_exact = level_context if level_context.trials > 0 else level_family
        return float(
            np.mean(
                [
                    family.features()[-1],
                    exact.features()[-1],
                    context.features()[-1],
                    level_family.features()[-1],
                    level_exact.features()[-1],
                    level_context.features()[-1],
                ]
            )
        )

    def summary(self) -> dict[str, float | int]:
        return {
            "online_update_count": int(self.online_update_count),
            "level_epoch": int(self.level_epoch),
            "level_step": int(self.level_step),
            "recent_prediction_error": float(self.recent_prediction_error),
            "recent_visible_only_rate": float(self.recent_visible_only_rate),
            "recent_objective_rate": float(self.recent_objective_rate),
            "family_count": int(len(self.family_stats)),
            "exact_count": int(len(self.exact_stats)),
            "context_count": int(len(self.context_stats)),
            "level_family_count": int(len(self.level_family_stats)),
            "level_exact_count": int(len(self.level_exact_stats)),
            "level_context_count": int(len(self.level_context_stats)),
        }


def _ema(old: float, new: float, *, rate: float) -> float:
    return float((1.0 - rate) * float(old) + rate * float(new))
