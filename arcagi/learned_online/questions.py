from __future__ import annotations

from enum import IntEnum

import numpy as np

from arcagi.learned_online.fast_belief import OnlineBeliefState


class QuestionToken(IntEnum):
    TEST_ACTION_MEANING = 0
    TEST_USEFUL_VS_VISIBLE = 1
    TEST_OBJECT_ROLE = 2
    TEST_PARAMETER_GROUNDING = 3
    TEST_MODE_STATE = 4
    TEST_GOAL_PREDICATE = 5
    TEST_ACTION_AVAILABILITY = 6


QUESTION_FEATURE_DIM = len(QuestionToken)


def select_question(belief: OnlineBeliefState) -> QuestionToken:
    if belief.recent_visible_only_rate >= 0.35:
        return QuestionToken.TEST_USEFUL_VS_VISIBLE
    if belief.recent_objective_rate <= 0.02 and belief.online_update_count >= 3:
        return QuestionToken.TEST_GOAL_PREDICATE
    if belief.online_update_count <= 2:
        return QuestionToken.TEST_ACTION_MEANING
    return QuestionToken.TEST_PARAMETER_GROUNDING


def question_features(question: QuestionToken) -> np.ndarray:
    features = np.zeros((QUESTION_FEATURE_DIM,), dtype=np.float32)
    features[int(question)] = 1.0
    return features


def question_tokens(question: QuestionToken) -> tuple[str, ...]:
    return ("question", question.name.lower())
