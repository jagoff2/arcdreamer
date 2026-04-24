from __future__ import annotations

from arcagi.agents.learned_online_minimal_agent import LearnedOnlineMinimalAgent
from arcagi.learned_online.curriculum import NullDenseClickTask, RowMajorSweepPolicy


def test_agent_less_sweep_like_than_row_major_baseline_on_null_task() -> None:
    agent_coords = _rollout_click_coords(LearnedOnlineMinimalAgent(seed=13), NullDenseClickTask(width=10, height=10), steps=30)
    sweep_coords = _rollout_click_coords(RowMajorSweepPolicy(), NullDenseClickTask(width=10, height=10), steps=30)

    assert _monotone_sweep_score(agent_coords) < _monotone_sweep_score(sweep_coords)


def test_null_dense_task_does_not_keep_covering_unique_clicks_after_family_failure() -> None:
    coords = _rollout_click_coords(LearnedOnlineMinimalAgent(seed=19), NullDenseClickTask(width=10, height=10), steps=60)

    assert len(set(coords)) < 45


def _rollout_click_coords(agent, env: NullDenseClickTask, *, steps: int) -> list[tuple[int, int]]:
    observation = env.reset(seed=0)
    reset = getattr(agent, "reset_episode", None)
    if callable(reset):
        reset()
    coords: list[tuple[int, int]] = []
    for _ in range(steps):
        action = str(agent.act(observation))
        parts = action.split(":")
        if len(parts) == 3 and parts[0] == "click":
            coords.append((int(parts[1]), int(parts[2])))
        result = env.step(action)
        update = getattr(agent, "update_after_step", None)
        if callable(update):
            update(
                next_observation=result.observation,
                reward=result.reward,
                terminated=result.terminated or result.truncated,
                info=result.info,
            )
        observation = result.observation
    return coords


def _monotone_sweep_score(coords: list[tuple[int, int]]) -> float:
    if len(coords) < 2:
        return 0.0
    ranks = [y * 1000 + x for x, y in coords]
    adjacent_increments = sum(1 for left, right in zip(ranks, ranks[1:]) if right == left + 1)
    monotone = sum(1 for left, right in zip(ranks, ranks[1:]) if right > left)
    return max(adjacent_increments, monotone) / float(len(ranks) - 1)
