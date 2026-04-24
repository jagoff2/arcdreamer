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
    final_info: Mapping[str, Any] = {}
    steps = 0
    for step in range(max_steps):
        action = agent.act(obs)
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
        if terminated:
            break
    return EpisodeResult(
        steps=steps,
        total_reward=total_reward,
        terminated=terminated,
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
