from __future__ import annotations

import numpy as np

from arcagi.agents.base import BaseAgent
from arcagi.core.types import GridObservation


class _NoopAgent(BaseAgent):
    def __init__(self, name: str) -> None:
        super().__init__(name)
        self.transitions = []

    def act(self, observation: GridObservation) -> str:
        state = self.observe(observation)
        self.last_state = state
        self.last_action = "wait"
        return "wait"

    def on_transition(self, transition) -> None:
        self.transitions.append(transition)


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


def test_base_agent_transition_info_exposes_public_level_boundary_without_hidden_event() -> None:
    agent = _NoopAgent("noop")
    initial = GridObservation(
        task_id="base/test",
        episode_id="base/test/episode_0",
        step_index=0,
        grid=np.zeros((3, 3), dtype=np.int64),
        available_actions=("1",),
        extras={
            "action_roles": {"1": "move_up"},
            "inventory": {"interface_levels_completed": "0", "interface_game_state": "GameState.NOT_FINISHED"},
        },
    )
    state = agent.observe(initial)
    agent.last_state = state
    agent.last_action = "1"

    next_observation = GridObservation(
        task_id="base/test",
        episode_id="base/test/episode_0",
        step_index=1,
        grid=np.ones((3, 3), dtype=np.int64),
        available_actions=("1",),
        extras={
            "action_roles": {"1": "move_up"},
            "inventory": {"interface_levels_completed": "1", "interface_game_state": "GameState.NOT_FINISHED"},
        },
    )
    agent.update_after_step(
        next_observation=next_observation,
        reward=1.0,
        terminated=False,
        info={"event": "hidden_should_not_leak", "public_flags": {"surface_progress": "1"}},
    )

    assert agent.transitions
    transition_info = agent.transitions[-1].info
    assert transition_info["levels_completed_before"] == 0
    assert transition_info["levels_completed_after"] == 1
    assert transition_info["level_delta"] == 1
    assert transition_info["level_boundary"] is True
    assert "event" not in transition_info
    assert transition_info["public_flags"] == {"surface_progress": "1"}
