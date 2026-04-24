from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from typing import Any

from arcagi.perception.object_encoder import extract_structured_state
from arcagi.scientist.types import (
    is_failure_terminal_game_state,
    is_reset_action,
    is_win_game_state,
    observation_game_state,
    observation_levels_completed,
)


@dataclass(frozen=True)
class ArcSessionSummary:
    session_id: str
    game_id: str
    session_index: int
    seed: int
    policy: str
    steps: int
    reward: float
    positive_rewards: int
    reset_steps: int
    levels_completed: int
    level_completions: int
    level_boundaries: int
    failure_terminals: int
    won: bool
    terminated: bool
    steps_to_first_level_completion: int
    sequence_count: int


def should_continue_after_terminal(observation: Any) -> bool:
    available_actions = tuple(str(item) for item in getattr(observation, "available_actions", ()) or ())
    if "0" not in available_actions:
        return False
    return is_failure_terminal_game_state(observation_game_state(observation))


def reset_agent_level(agent: Any) -> None:
    reset_level = getattr(agent, "reset_level", None)
    if callable(reset_level):
        reset_level()
        return
    reset_episode = getattr(agent, "reset_episode", None)
    if callable(reset_episode):
        reset_episode()


def collect_arc_session(
    env: Any,
    *,
    agent: Any,
    game_id: str,
    session_index: int,
    seed: int,
    max_steps: int,
    policy_name: str,
    sample_builder: Callable[[Any, Any, str, float], dict[str, Any]],
) -> tuple[list[dict[str, Any]], ArcSessionSummary]:
    observation = env.reset(seed=seed)
    session_id = f"{game_id}/session_{session_index}"
    samples: list[dict[str, Any]] = []
    sequence_index = 0
    steps = 0
    total_reward = 0.0
    positive_rewards = 0
    reset_steps = 0
    level_completions = 0
    level_boundaries = 0
    failure_terminals = 0
    steps_to_first_level_completion = 0
    max_levels_completed = observation_levels_completed(observation)
    won = False
    terminated = False

    while steps < max_steps:
        state = extract_structured_state(observation)
        action = str(agent.act(observation))
        result = env.step(action)
        next_observation = result.observation
        reward = float(result.reward)
        next_state = extract_structured_state(next_observation)
        terminated = bool(result.terminated or result.truncated)
        before_levels = observation_levels_completed(observation)
        after_levels = observation_levels_completed(next_observation)
        level_delta = max(after_levels - before_levels, 0)
        if level_delta > 0 and steps_to_first_level_completion <= 0:
            steps_to_first_level_completion = steps + 1
        max_levels_completed = max(max_levels_completed, after_levels)
        game_state_before = observation_game_state(observation)
        game_state_after = observation_game_state(next_observation)
        won = won or is_win_game_state(game_state_after)
        failure_terminal = terminated and not won and is_failure_terminal_game_state(game_state_after)
        continue_after_terminal = failure_terminal and should_continue_after_terminal(next_observation)
        session_terminal = won or (terminated and not continue_after_terminal)
        reset_action = is_reset_action(action)
        level_boundary = reset_action or level_delta > 0
        if reset_action:
            reset_steps += 1
        if level_delta > 0:
            level_completions += level_delta
        if level_boundary:
            level_boundaries += 1
        if failure_terminal:
            failure_terminals += 1
        if reward > 0.0:
            positive_rewards += 1
        total_reward += reward

        sample = dict(sample_builder(state, next_state, action, reward))
        sample.update(
            {
                "session_id": session_id,
                "sequence_id": f"{session_id}/segment_{sequence_index}",
                "game_id": str(game_id),
                "session_index": int(session_index),
                "policy": str(policy_name),
                "session_step_index": int(steps),
                "terminated": bool(terminated),
                "session_terminal": bool(session_terminal),
                "failure_terminal": bool(failure_terminal),
                "continued_session": bool(continue_after_terminal),
                "won_session": bool(won),
                "reset_action": bool(reset_action),
                "level_boundary": bool(level_boundary),
                "levels_completed_before": int(before_levels),
                "levels_completed_after": int(after_levels),
                "level_delta": int(level_delta),
                "game_state_before": str(game_state_before),
                "game_state_after": str(game_state_after),
                "info": dict(result.info or {}),
            }
        )
        samples.append(sample)

        agent.update_after_step(
            next_observation=next_observation,
            reward=reward,
            terminated=terminated,
            info=result.info,
        )
        if level_boundary:
            reset_agent_level(agent)
            sequence_index += 1
        observation = next_observation
        steps += 1
        if session_terminal:
            break

    summary = ArcSessionSummary(
        session_id=session_id,
        game_id=str(game_id),
        session_index=int(session_index),
        seed=int(seed),
        policy=str(policy_name),
        steps=int(steps),
        reward=float(total_reward),
        positive_rewards=int(positive_rewards),
        reset_steps=int(reset_steps),
        levels_completed=int(max_levels_completed),
        level_completions=int(level_completions),
        level_boundaries=int(level_boundaries),
        failure_terminals=int(failure_terminals),
        won=bool(won),
        terminated=bool(terminated),
        steps_to_first_level_completion=int(steps_to_first_level_completion),
        sequence_count=int(sequence_index + (0 if not samples or samples[-1].get("level_boundary") else 1)),
    )
    return samples, summary


def session_progress_signal(sample: Mapping[str, Any]) -> float:
    reward = float(sample.get("reward", 0.0))
    level_delta = float(sample.get("level_delta", 0.0))
    won_session = bool(sample.get("won_session", False))
    failure_terminal = bool(sample.get("failure_terminal", False))
    continued_session = bool(sample.get("continued_session", False))
    reset_action = bool(sample.get("reset_action", False))

    progress = reward + (1.25 * level_delta)
    if won_session:
        progress += 1.0
    if failure_terminal:
        progress -= 0.35 if continued_session else 0.65
    return float(progress)


def annotate_session_returns(samples: list[dict[str, Any]], *, gamma: float = 0.92) -> list[dict[str, Any]]:
    if gamma <= 0.0 or gamma > 1.0:
        raise ValueError("gamma must be in (0, 1]")
    sequence_buckets: dict[str, list[int]] = defaultdict(list)
    for index, sample in enumerate(samples):
        sequence_buckets[str(sample.get("sequence_id", index))].append(index)

    for indices in sequence_buckets.values():
        running = 0.0
        for sample_index in reversed(indices):
            sample = samples[sample_index]
            immediate = session_progress_signal(sample)
            sample["session_progress"] = float(immediate)
            sample["discounted_return"] = float(immediate + (gamma * running))
            sample["sequence_return"] = float(running)
            base_usefulness = float(sample.get("usefulness", 0.0))
            sample["usefulness"] = max(base_usefulness, max(immediate, 0.0))
            running = immediate + (gamma * running)
    return samples
