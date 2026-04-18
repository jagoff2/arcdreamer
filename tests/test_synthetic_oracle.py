from __future__ import annotations

import numpy as np

from arcagi.envs.synthetic import GOAL_ACTIVE, HiddenRuleEnv
from arcagi.training.synthetic_oracle import oracle_action


def test_synthetic_oracle_solves_switch_unlock() -> None:
    env = HiddenRuleEnv(family_mode="switch_unlock", family_variant="red", seed=11)
    observation = env.reset(seed=11)

    for _ in range(env.max_steps):
        result = env.step(oracle_action(env))
        observation = result.observation
        if result.reward > 0.9:
            break

    assert np.count_nonzero(observation.grid == GOAL_ACTIVE) == 1
    assert result.reward > 0.9


def test_synthetic_oracle_solves_selector_sequence_unlock() -> None:
    env = HiddenRuleEnv(family_mode="selector_sequence_unlock", family_variant="red__red_then_blue", seed=23)
    observation = env.reset(seed=23)

    for _ in range(env.max_steps):
        result = env.step(oracle_action(env))
        observation = result.observation
        if result.reward > 0.9:
            break

    assert np.count_nonzero(observation.grid == GOAL_ACTIVE) == 1
    assert result.reward > 0.9
