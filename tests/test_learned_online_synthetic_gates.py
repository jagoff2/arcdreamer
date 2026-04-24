from __future__ import annotations

from arcagi.agents.learned_online_minimal_agent import LearnedOnlineMinimalAgent
from arcagi.core.types import GridObservation
from arcagi.evaluation.harness import run_episode
from arcagi.learned_online.curriculum import (
    FrozenFirstPolicy,
    RandomizedBindingTask,
    VisibleUsefulTrapTask,
)


def test_online_minimal_beats_frozen_first_on_visible_useful_trap() -> None:
    online_returns = []
    frozen_returns = []
    for seed in range(12):
        online_returns.append(
            float(
                run_episode(
                    LearnedOnlineMinimalAgent(seed=seed),
                    VisibleUsefulTrapTask(seed=seed),
                    seed=seed,
                    max_steps=5,
                )["return"]
            )
        )
        frozen_returns.append(
            float(run_episode(FrozenFirstPolicy(), VisibleUsefulTrapTask(seed=seed), seed=seed, max_steps=5)["return"])
        )

    assert sum(online_returns) > sum(frozen_returns)


def test_online_minimal_beats_frozen_first_on_randomized_binding() -> None:
    online_returns = []
    frozen_returns = []
    for seed in range(20):
        online_returns.append(
            float(
                run_episode(
                    LearnedOnlineMinimalAgent(seed=seed + 100),
                    RandomizedBindingTask(seed=seed),
                    seed=seed,
                    max_steps=6,
                )["return"]
            )
        )
        frozen_returns.append(
            float(run_episode(FrozenFirstPolicy(), RandomizedBindingTask(seed=seed), seed=seed, max_steps=6)["return"])
        )

    assert sum(online_returns) >= sum(frozen_returns) + 3.0


def test_learned_online_run_reports_claim_and_dense_metadata() -> None:
    result = run_episode(LearnedOnlineMinimalAgent(seed=0), RandomizedBindingTask(seed=3), seed=3, max_steps=2)

    assert result["controller_kind"] == "learned_online_minimal"
    assert result["claim_eligible"] is True
    assert result["learned_online_controller"] is True
    assert result["scored_action_count"] == result["legal_action_count"]
