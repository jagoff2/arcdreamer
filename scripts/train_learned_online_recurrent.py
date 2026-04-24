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
    behavior_policy: str = "mixed",
    epsilon: float = 0.2,
) -> dict[str, object]:
    rng = np.random.default_rng(seed)
    agent = LearnedOnlineRecurrentAgent(seed=seed)
    returns: list[float] = []
    task_counts: dict[str, int] = {}
    behavior_counts: dict[str, int] = {}
    for episode in range(int(episodes)):
        task = _make_task(int(rng.integers(6)), seed=seed + episode)
        observation = task.reset(seed=seed + episode)
        agent.reset_episode()
        episode_return = 0.0
        for _step in range(int(max_steps)):
            state = agent.observe(observation)
            question = select_question(agent.belief)
            actions = tuple(state.affordances)
            behavior, action = _choose_training_action(
                agent=agent,
                state=state,
                actions=actions,
                question=question,
                rng=rng,
                behavior_policy=behavior_policy,
                epsilon=epsilon,
            )
            behavior_counts[behavior] = behavior_counts.get(behavior, 0) + 1
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
        "behavior_policy": behavior_policy,
        "behavior_counts": behavior_counts,
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


def _choose_training_action(
    *,
    agent: LearnedOnlineRecurrentAgent,
    state,
    actions: tuple[str, ...],
    question,
    rng: np.random.Generator,
    behavior_policy: str,
    epsilon: float,
) -> tuple[str, str]:
    policy = str(behavior_policy).strip().lower()
    if policy == "mixed":
        draw = float(rng.random())
        if draw < 0.4:
            policy = "random"
        elif draw < 0.8:
            policy = "agent"
        else:
            policy = "epsilon_agent"
    if policy == "random":
        return "random", _exploration_action(actions, rng)
    if policy == "epsilon_agent":
        if float(rng.random()) < float(epsilon):
            return "epsilon_random", _exploration_action(actions, rng)
        decision = agent.policy.choose_action(
            state,
            actions,
            question=question,
            chunk_size=agent.chunk_size,
        )
        return "epsilon_agent", decision.action
    if policy == "agent":
        decision = agent.policy.choose_action(
            state,
            actions,
            question=question,
            chunk_size=agent.chunk_size,
        )
        return "agent", decision.action
    raise ValueError(f"unknown behavior_policy={behavior_policy!r}")


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=Path("artifacts/learned_online_recurrent_latest.pkl"))
    parser.add_argument("--episodes", type=int, default=2000)
    parser.add_argument("--max-steps", type=int, default=12)
    parser.add_argument("--seed", type=int, default=23)
    parser.add_argument(
        "--behavior-policy",
        type=str,
        default="mixed",
        choices=("random", "agent", "epsilon_agent", "mixed"),
    )
    parser.add_argument("--epsilon", type=float, default=0.2)
    args = parser.parse_args()
    result = train(
        output=args.output,
        episodes=args.episodes,
        max_steps=args.max_steps,
        seed=args.seed,
        behavior_policy=args.behavior_policy,
        epsilon=args.epsilon,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
