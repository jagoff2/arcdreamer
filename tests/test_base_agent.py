from __future__ import annotations

import numpy as np

from arcagi.agents.base import BaseAgent
from arcagi.core.types import GridObservation


class _NoopAgent(BaseAgent):
    def act(self, observation: GridObservation) -> str:
        state = self.observe(observation)
        self.last_state = state
        self.last_action = "wait"
        return "wait"


def test_base_agent_merges_public_info_into_visible_observation_state() -> None:
    agent = _NoopAgent("noop")
    initial = GridObservation(
        task_id="base/test",
        episode_id="base/test/episode_0",
        step_index=0,
        grid=np.zeros((3, 3), dtype=np.int64),
        available_actions=("wait",),
        extras={},
    )
    state = agent.observe(initial)
    agent.last_state = state
    agent.last_action = "wait"

    next_observation = GridObservation(
        task_id="base/test",
        episode_id="base/test/episode_0",
        step_index=1,
        grid=np.zeros((3, 3), dtype=np.int64),
        available_actions=("wait",),
        extras={},
    )
    next_state = agent.update_after_step(
        next_observation=next_observation,
        reward=0.0,
        terminated=False,
        info={
            "public_inventory": {"interface_selected_color": "red"},
            "public_flags": {"interface_selection_active": "1"},
        },
    )

    assert next_state.inventory_dict()["interface_selected_color"] == "red"
    assert next_state.flags_dict()["interface_selection_active"] == "1"
