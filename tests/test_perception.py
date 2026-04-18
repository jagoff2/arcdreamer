from __future__ import annotations

import numpy as np

from arcagi.core.types import GridObservation
from arcagi.envs.synthetic import HiddenRuleEnv
from arcagi.perception.object_encoder import extract_structured_state


def test_extract_structured_state_contains_agent_and_target() -> None:
    env = HiddenRuleEnv(family_mode="order_collect", family_variant="red_then_blue", seed=3)
    observation = env.reset(seed=3)
    state = extract_structured_state(observation)
    assert any("agent" in obj.tags for obj in state.objects)
    assert any("target" in obj.tags for obj in state.objects)
    assert state.grid_shape == observation.grid.shape
    assert state.fingerprint()


def test_extract_structured_state_respects_nonzero_background_color() -> None:
    observation = GridObservation(
        task_id="perception/test",
        episode_id="episode-0",
        step_index=0,
        grid=np.array(
            [
                [7, 7, 7],
                [7, 2, 7],
                [7, 7, 7],
            ],
            dtype=np.int64,
        ),
        available_actions=("wait",),
        extras={"background_color": 7},
    )

    state = extract_structured_state(observation)

    assert len(state.objects) == 1
    assert state.objects[0].color == 2
