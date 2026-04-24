from __future__ import annotations

import numpy as np

from arcagi.agents.learned_online_minimal_agent import LearnedOnlineMinimalAgent
from arcagi.core.types import GridObservation
from arcagi.learned_online.questions import QuestionToken


def test_question_token_changes_controlled_scores_when_model_uses_question_features() -> None:
    agent = LearnedOnlineMinimalAgent(seed=0)
    observation = GridObservation("question", "0", 0, np.zeros((2, 2), dtype=np.int64), ("a", "b"))
    state = agent.observe(observation)
    question_start = agent.policy.model.input_dim - len(QuestionToken)
    agent.policy.model.weights["info_gain"][question_start + int(QuestionToken.TEST_ACTION_MEANING)] = 0.8
    agent.policy.model.weights["info_gain"][question_start + int(QuestionToken.TEST_GOAL_PREDICATE)] = -0.8

    action_meaning = agent.score_actions_for_state(state, ("a", "b"), question=QuestionToken.TEST_ACTION_MEANING)
    goal_predicate = agent.score_actions_for_state(state, ("a", "b"), question=QuestionToken.TEST_GOAL_PREDICATE)

    assert action_meaning["a"].components["pred_info_gain"] > goal_predicate["a"].components["pred_info_gain"]


def test_memory_features_change_controlled_scores_when_model_uses_memory_features() -> None:
    agent = LearnedOnlineMinimalAgent(seed=0)
    observation = GridObservation("memory", "0", 0, np.zeros((2, 2), dtype=np.int64), ("a", "b"))
    state = agent.observe(observation)
    memory_start = 38 + 26
    agent.policy.model.weights["useful"][memory_start + 1] = 1.2

    before = agent.score_actions_for_state(state, ("a",), question=QuestionToken.TEST_ACTION_MEANING)
    agent.last_state = state
    agent.last_action = "a"
    agent.last_question = QuestionToken.TEST_ACTION_MEANING
    after = GridObservation("memory", "0", 1, np.ones((2, 2), dtype=np.int64), ("a", "b"))
    agent.update_after_step(after, reward=1.0, terminated=True, info={})
    state_again = agent.observe(observation)
    after_scores = agent.score_actions_for_state(state_again, ("a",), question=QuestionToken.TEST_ACTION_MEANING)

    assert after_scores["a"].components["pred_useful"] > before["a"].components["pred_useful"]
