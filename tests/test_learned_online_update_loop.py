from __future__ import annotations

import numpy as np

from arcagi.agents.learned_online_minimal_agent import LearnedOnlineMinimalAgent
from arcagi.core.types import GridObservation
from arcagi.evaluation.harness import run_episode
from arcagi.learned_online.curriculum import VisibleUsefulTrapTask


def test_same_state_scores_change_after_contradictory_evidence() -> None:
    agent = LearnedOnlineMinimalAgent(seed=5)
    task = VisibleUsefulTrapTask(seed=0)
    observation = task.reset(seed=0)
    state = agent.observe(observation)
    before = agent.score_actions_for_state(state, ("trap", "useful"))
    trap_before = before["trap"].components["pred_visible"]

    agent.last_state = state
    agent.last_action = "trap"
    agent.last_question = before["trap"].question
    after = GridObservation(
        task_id="visible_useful_trap",
        episode_id="0",
        step_index=1,
        grid=np.array([[0, 1], [0, 0]], dtype=np.int64),
        available_actions=("trap", "useful", "noop"),
    )
    agent.update_after_step(after, reward=0.0, terminated=False, info={})
    state_again = agent.observe(observation)
    rescored = agent.score_actions_for_state(state_again, ("trap", "useful"))

    assert agent.diagnostics()["online_updates"] == 1
    assert abs(rescored["trap"].components["pred_visible"] - trap_before) > 1e-4


def test_online_agent_beats_frozen_on_visible_useful_trap_sanity() -> None:
    class FrozenTrapAgent:
        name = "frozen_trap"
        latest_language = ()

        def reset_episode(self) -> None:
            return None

        def act(self, observation: GridObservation) -> str:
            return "trap"

        def update_after_step(self, **_kwargs) -> None:
            return None

    online_total = 0.0
    frozen_total = 0.0
    online_updates = 0
    for seed in range(10):
        online_result = run_episode(LearnedOnlineMinimalAgent(seed=seed), VisibleUsefulTrapTask(seed=seed), seed=seed, max_steps=6)
        frozen_result = run_episode(FrozenTrapAgent(), VisibleUsefulTrapTask(seed=seed), seed=seed, max_steps=6)
        online_total += float(online_result["return"])
        frozen_total += float(frozen_result["return"])
        online_updates += int(online_result["diagnostics"]["online_updates"])

    assert online_total > frozen_total
    assert online_updates >= 1
