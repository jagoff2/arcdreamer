from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from arcagi.agents.learned_online_recurrent_agent import LearnedOnlineRecurrentAgent
from arcagi.learned_online.curriculum import (
    DelayedUnlockTask,
    DenseCoordinateGroundingTask,
    MovementRequiredAfterModeTask,
    RandomizedBindingTask,
    VisibleMovementTrapTask,
    VisibleUsefulTrapTask,
)
from arcagi.learned_online.questions import select_question


def train(
    *,
    output: Path,
    episodes: int,
    max_steps: int,
    seed: int,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    agent = LearnedOnlineRecurrentAgent(seed=seed)
    returns: list[float] = []
    task_counts: dict[str, int] = {}
    for episode in range(int(episodes)):
        task = _make_task(int(rng.integers(6)), seed=seed + episode)
        observation = task.reset(seed=seed + episode)
        agent.reset_episode()
        episode_return = 0.0
        for _step in range(int(max_steps)):
            state = agent.observe(observation)
            question = select_question(agent.belief)
            actions = tuple(state.affordances)
            action = _exploration_action(actions, rng)
            agent.last_state = state
            agent.last_action = action
            agent.last_question = question
            result = task.step(action)
            agent.update_after_step(
                next_observation=result.observation,
                reward=result.reward,
                terminated=result.terminated or result.truncated,
                info=result.info,
            )
            observation = result.observation
            episode_return += float(result.reward)
            if result.terminated or result.truncated:
                break
        returns.append(episode_return)
        task_counts[observation.task_id] = task_counts.get(observation.task_id, 0) + 1
    agent.save_checkpoint(output)
    return {
        "checkpoint": str(output),
        "episodes": int(episodes),
        "max_steps": int(max_steps),
        "seed": int(seed),
        "avg_return": float(np.mean(returns)) if returns else 0.0,
        "success_rate": float(np.mean([ret > 0.0 for ret in returns])) if returns else 0.0,
        "task_counts": task_counts,
        "model_updates": int(agent.model.updates),
    }


def _make_task(kind: int, *, seed: int):
    if kind == 0:
        return VisibleUsefulTrapTask(seed=seed)
    if kind == 1:
        return RandomizedBindingTask(seed=seed)
    if kind == 2:
        return DenseCoordinateGroundingTask(seed=seed, size=6)
    if kind == 3:
        return VisibleMovementTrapTask(seed=seed)
    if kind == 4:
        return MovementRequiredAfterModeTask(seed=seed)
    return DelayedUnlockTask(seed=seed)


def _exploration_action(actions: tuple[str, ...], rng: np.random.Generator) -> str:
    return str(actions[int(rng.integers(len(actions)))])


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/learned_online_recurrent_latest.pkl"))
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=23)
    args = parser.parse_args()
    result = train(output=args.output, episodes=args.episodes, max_steps=args.max_steps, seed=args.seed)
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
