from __future__ import annotations

import numpy as np

from arcagi.core.types import ActionName, StructuredState
from arcagi.learned_online.action_features import ACTION_FEATURE_DIM, encode_action_candidates
from arcagi.learned_online.questions import QUESTION_FEATURE_DIM, QuestionToken, question_features
from arcagi.learned_online.signals import TransitionLabels

LABEL_FEATURE_DIM = 9
EVENT_FEATURE_DIM = ACTION_FEATURE_DIM + LABEL_FEATURE_DIM + QUESTION_FEATURE_DIM + 2


def label_features(labels: TransitionLabels) -> np.ndarray:
    return np.asarray(
        [
            labels.visible_change,
            labels.objective_progress,
            labels.reward_progress,
            labels.terminal_progress,
            labels.action_availability_changed,
            labels.appeared_or_disappeared,
            labels.mechanic_change,
            labels.no_effect_nonprogress,
            labels.visible_only_nonprogress,
        ],
        dtype=np.float32,
    )


def event_features(
    state: StructuredState,
    action: ActionName,
    labels: TransitionLabels,
    *,
    question: QuestionToken,
    prediction_error: float,
    realized_info_gain: float,
) -> np.ndarray:
    action_features = encode_action_candidates(state, (str(action),)).features[0]
    return np.concatenate(
        [
            action_features,
            label_features(labels),
            question_features(question),
            np.asarray([float(prediction_error), float(realized_info_gain)], dtype=np.float32),
        ]
    ).astype(np.float32)
