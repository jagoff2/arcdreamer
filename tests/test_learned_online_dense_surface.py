from __future__ import annotations

import numpy as np

from arcagi.agents.learned_online_minimal_agent import LearnedOnlineMinimalAgent
from arcagi.core.types import GridObservation
from arcagi.learned_online.questions import QuestionToken


def test_score_actions_is_order_equivariant() -> None:
    agent = LearnedOnlineMinimalAgent(seed=3)
    actions = tuple(f"click:{x}:{y}" for y in range(5) for x in range(5))
    observation = GridObservation("order", "0", 0, np.zeros((5, 5), dtype=np.int64), actions)
    state = agent.observe(observation)

    scores_a = agent.score_actions_for_state(state, actions, question=QuestionToken.TEST_PARAMETER_GROUNDING)
    shuffled = list(actions)
    np.random.default_rng(17).shuffle(shuffled)
    scores_b = agent.score_actions_for_state(state, shuffled, question=QuestionToken.TEST_PARAMETER_GROUNDING)

    assert set(scores_a) == set(scores_b)
    for action in actions:
        assert abs(scores_a[action].score - scores_b[action].score) < 1e-6


def test_equal_score_ties_do_not_choose_first_action_systematically() -> None:
    actions = tuple(f"click:{x}:0" for x in range(12))
    chosen_ranks: list[int] = []
    for seed in range(24):
        agent = LearnedOnlineMinimalAgent(seed=seed)
        shuffled = list(actions)
        np.random.default_rng(seed + 99).shuffle(shuffled)
        observation = GridObservation("ties", str(seed), 0, np.zeros((1, 12), dtype=np.int64), tuple(shuffled))
        chosen = agent.act(observation)
        chosen_ranks.append(tuple(shuffled).index(chosen))

    assert len(set(chosen_ranks)) > 1
    assert chosen_ranks.count(0) < len(chosen_ranks) // 2
