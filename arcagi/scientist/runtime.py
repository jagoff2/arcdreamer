"""Runtime helpers for evaluating ScientistAgent on ARC-like environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .agent import ScientistAgent
from .types import ActionName


@dataclass(frozen=True)
class EpisodeResult:
    steps: int
    total_reward: float
    terminated: bool
    won: bool
    reset_steps: int
    levels_completed: int
    final_info: Mapping[str, Any]
    diagnostics: Mapping[str, Any]


def run_episode(env: Any, agent: ScientistAgent, *, max_steps: int = 256, seed: int | None = None) -> EpisodeResult:
    agent.reset_episode()
    try:
        obs = env.reset(seed=seed)
    except TypeError:
        obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    total_reward = 0.0
    terminated = False
    won = False
    reset_steps = 0
    levels_completed = _levels_completed(obs)
    final_info: Mapping[str, Any] = {}
    steps = 0
    for step in range(max_steps):
        action = agent.act(obs)
        if str(action) == "0":
            reset_steps += 1
        result = env.step(action)
        next_obs, reward, terminated, info = _unpack_step_result(result)
        agent.observe_result(
            action=action,
            before_observation=obs,
            after_observation=next_obs,
            reward=reward,
            terminated=terminated,
            info=info,
        )
        total_reward += float(reward)
        obs = next_obs
        steps = step + 1
        final_info = info
        levels_completed = max(levels_completed, _levels_completed(next_obs))
        won = won or _is_win_state(next_obs)
        if won:
            terminated = True
            break
        if terminated and not _should_continue_after_terminal(next_obs):
            break
    return EpisodeResult(
        steps=steps,
        total_reward=total_reward,
        terminated=terminated,
        won=won,
        reset_steps=reset_steps,
        levels_completed=levels_completed,
        final_info=final_info,
        diagnostics=agent.diagnostics(),
    )


def _unpack_step_result(result: Any) -> tuple[Any, float, bool, Mapping[str, Any]]:
    # Existing arcagi StepResult-style object.
    if hasattr(result, "observation") or hasattr(result, "next_observation"):
        obs = getattr(result, "observation", None)
        if obs is None:
            obs = getattr(result, "next_observation")
        reward = float(getattr(result, "reward", 0.0))
        terminated = bool(getattr(result, "terminated", False) or getattr(result, "done", False))
        info = getattr(result, "info", None) or getattr(result, "extras", None) or {}
        return obs, reward, terminated, info
    # Gymnasium shape.
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated or truncated), info or {}
    # Classic Gym shape.
    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward), bool(done), info or {}
    raise TypeError(f"Unsupported env.step result: {type(result)!r}")


def _observation_extras(observation: Any) -> Mapping[str, Any]:
    extras = getattr(observation, "extras", None)
    if isinstance(extras, Mapping):
        return extras
    return {}


def _levels_completed(observation: Any) -> int:
    extras = _observation_extras(observation)
    try:
        return int(extras.get("levels_completed", getattr(observation, "levels_completed", 0)) or 0)
    except Exception:
        return 0


def _game_state(observation: Any) -> str:
    extras = _observation_extras(observation)
    return str(extras.get("game_state", getattr(observation, "state", "")) or "")


def _is_win_state(observation: Any) -> bool:
    return _game_state(observation).strip().upper().endswith("WIN")


def _should_continue_after_terminal(observation: Any) -> bool:
    available_actions = tuple(str(item) for item in getattr(observation, "available_actions", ()) or ())
    game_state = _game_state(observation).strip().upper()
    if "0" not in available_actions:
        return False
    return game_state.endswith("GAME_OVER") or game_state.endswith("SESSION_ENDED")
