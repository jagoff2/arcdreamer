from __future__ import annotations

import numpy as np

from arcagi.envs.session import PersistentLevelSessionEnv
from arcagi.envs.synthetic import GOAL_ACTIVE, HiddenRuleEnv
from arcagi.scientist.synthetic_env import HiddenRuleGridEnv, SyntheticConfig
from arcagi.training.synthetic_oracle import oracle_action, teacher_action


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


def test_teacher_action_solves_simple_hidden_rule_grid() -> None:
    env = HiddenRuleGridEnv(SyntheticConfig(size=7, requires_key=True, seed=7, max_steps=48))
    observation = env.reset(seed=7)

    for _ in range(env.config.max_steps):
        action = teacher_action(env, observation=observation)
        observation, reward, done, _info = env.step(action)
        if reward >= 1.0:
            break
        if done:
            break

    assert reward >= 1.0


def test_teacher_action_handles_retryable_session_wrapper() -> None:
    env = PersistentLevelSessionEnv(
        level_builders=(
            lambda seed: HiddenRuleEnv(family_mode="selector_unlock", family_variant="red", seed=seed, max_steps=6),
        ),
        task_id="synthetic/session_oracle",
        family_id="synthetic/session_oracle/red",
        seed=19,
    )
    observation = env.reset(seed=19)

    assert teacher_action(env, observation=observation) != "0"

    failed = None
    for _ in range(8):
        failed = env.step("up")
        if failed.terminated:
            break

    assert failed is not None
    assert failed.terminated is True
    assert failed.observation.extras["game_state"] == "GAME_OVER"
    assert teacher_action(env, observation=failed.observation) == "0"
