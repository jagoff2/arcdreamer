from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from typing import Any

from arcagi.core.types import ActionName, GridObservation, StepResult
from arcagi.envs.base import BaseEnvironment


LevelBuilder = Callable[[int], Any]
LevelSuccessFn = Callable[[GridObservation, float, bool, bool, dict[str, Any]], bool]


@dataclass(frozen=True)
class SessionLevelResult:
    observation: GridObservation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any]


class PersistentLevelSessionEnv(BaseEnvironment):
    """Generic retryable multi-level session wrapper.

    One session contains multiple related levels.  The wrapper always exposes a
    reset action `"0"`, carries the same agent across failures, and advances to
    the next level only after a level-specific success condition is met.
    """

    def __init__(
        self,
        *,
        level_builders: Sequence[LevelBuilder],
        task_id: str,
        family_id: str,
        success_fn: LevelSuccessFn | None = None,
        reset_action: ActionName = "0",
        seed: int = 0,
    ) -> None:
        if not level_builders:
            raise ValueError("PersistentLevelSessionEnv requires at least one level builder")
        self._level_builders = tuple(level_builders)
        self._task_id = str(task_id)
        self._family_id = str(family_id)
        self._success_fn = success_fn or _default_success
        self._reset_action = str(reset_action)
        self._base_seed = int(seed)
        self._episode_index = 0
        self._session_step = 0
        self._levels_completed = 0
        self._current_level_index = 0
        self._current_retry = 0
        self._current_env: Any | None = None

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def family_id(self) -> str:
        return self._family_id

    @property
    def current_level_env(self) -> Any | None:
        return self._current_env

    def legal_actions(self) -> tuple[ActionName, ...]:
        if self._current_env is None:
            return (self._reset_action,)
        legal = tuple(str(item) for item in getattr(self._current_env, "legal_actions", lambda: ())() or ())
        return tuple(dict.fromkeys((*legal, self._reset_action)))

    def reset(self, seed: int | None = None) -> GridObservation:
        if seed is not None:
            self._base_seed = int(seed)
        self._episode_index += 1
        self._session_step = 0
        self._levels_completed = 0
        self._current_level_index = 0
        self._current_retry = 0
        return self._reset_current_level()

    def step(self, action: ActionName) -> StepResult:
        if self._current_env is None:
            raise RuntimeError("session environment must be reset before stepping")
        action_text = str(action)
        self._session_step += 1
        if action_text == self._reset_action:
            self._current_retry += 1
            observation = self._reset_current_level()
            info = {
                "event": "session_reset",
                "levels_completed": self._levels_completed,
                "session_level_index": self._current_level_index,
                "session_retry_index": self._current_retry,
            }
            return StepResult(
                observation=observation,
                reward=0.0,
                terminated=False,
                truncated=False,
                info=info,
            )

        level_result = _coerce_level_result(self._current_env.step(action_text))
        success = self._success_fn(
            level_result.observation,
            level_result.reward,
            level_result.terminated,
            level_result.truncated,
            dict(level_result.info),
        )
        if success:
            self._levels_completed += 1
            if self._current_level_index + 1 >= len(self._level_builders):
                final_observation = self._augment_observation(level_result.observation, game_state="WIN")
                return StepResult(
                    observation=final_observation,
                    reward=level_result.reward,
                    terminated=True,
                    truncated=False,
                    info={
                        **dict(level_result.info),
                        "event": "session_win",
                        "levels_completed": self._levels_completed,
                        "session_level_index": self._current_level_index,
                    },
                )
            self._current_level_index += 1
            self._current_retry = 0
            next_observation = self._reset_current_level()
            return StepResult(
                observation=next_observation,
                reward=level_result.reward,
                terminated=False,
                truncated=False,
                info={
                    **dict(level_result.info),
                    "event": "level_advanced",
                    "levels_completed": self._levels_completed,
                    "session_level_index": self._current_level_index,
                },
            )

        if level_result.terminated or level_result.truncated:
            failed_observation = self._augment_observation(level_result.observation, game_state="GAME_OVER")
            return StepResult(
                observation=failed_observation,
                reward=level_result.reward,
                terminated=True,
                truncated=False,
                info={
                    **dict(level_result.info),
                    "event": "level_failed",
                    "levels_completed": self._levels_completed,
                    "session_level_index": self._current_level_index,
                    "session_retry_index": self._current_retry,
                },
            )

        live_observation = self._augment_observation(level_result.observation, game_state="NOT_FINISHED")
        return StepResult(
            observation=live_observation,
            reward=level_result.reward,
            terminated=False,
            truncated=False,
            info={
                **dict(level_result.info),
                "levels_completed": self._levels_completed,
                "session_level_index": self._current_level_index,
                "session_retry_index": self._current_retry,
            },
        )

    def close(self) -> None:
        if self._current_env is None:
            return
        close = getattr(self._current_env, "close", None)
        if callable(close):
            close()

    def _reset_current_level(self) -> GridObservation:
        builder = self._level_builders[self._current_level_index]
        self._current_env = builder(self._level_seed(self._current_level_index, self._current_retry))
        reset = getattr(self._current_env, "reset", None)
        if not callable(reset):
            raise TypeError("session level builder must return an environment with a reset(seed=...) method")
        observation = reset(seed=self._level_seed(self._current_level_index, self._current_retry))
        if isinstance(observation, tuple):
            observation = observation[0]
        if not isinstance(observation, GridObservation):
            raise TypeError(f"session level reset must return GridObservation, got {type(observation)!r}")
        return self._augment_observation(observation, game_state="NOT_FINISHED")

    def _augment_observation(self, observation: GridObservation, *, game_state: str) -> GridObservation:
        extras = dict(observation.extras)
        action_roles = dict(extras.get("action_roles", {})) if isinstance(extras.get("action_roles"), dict) else {}
        action_roles[self._reset_action] = "reset_level"
        extras.update(
            {
                "game_state": game_state,
                "levels_completed": int(self._levels_completed),
                "session_level_index": int(self._current_level_index),
                "session_level_count": int(len(self._level_builders)),
                "session_retry_index": int(self._current_retry),
                "action_roles": action_roles,
            }
        )
        available_actions = tuple(dict.fromkeys((*tuple(str(item) for item in observation.available_actions), self._reset_action)))
        return GridObservation(
            task_id=self.task_id,
            episode_id=f"{self.task_id}/episode_{self._episode_index}",
            step_index=int(self._session_step),
            grid=observation.grid,
            available_actions=available_actions,
            extras=extras,
        )

    def _level_seed(self, level_index: int, retry_index: int) -> int:
        return int(self._base_seed + (level_index * 10_000) + retry_index)


def _default_success(
    observation: GridObservation,
    reward: float,
    terminated: bool,
    truncated: bool,
    info: dict[str, Any],
) -> bool:
    extras = observation.extras if isinstance(observation.extras, dict) else {}
    game_state = str(extras.get("game_state", "") or "").strip().upper()
    if game_state.endswith("WIN"):
        return True
    event = str(info.get("event", "") or "").strip().lower()
    if event in {"goal_reached", "session_win"}:
        return True
    return bool(reward > 0.9 and not truncated and terminated)


def _coerce_level_result(result: Any) -> SessionLevelResult:
    if isinstance(result, StepResult):
        return SessionLevelResult(
            observation=result.observation,
            reward=float(result.reward),
            terminated=bool(result.terminated),
            truncated=bool(result.truncated),
            info=dict(result.info),
        )
    if isinstance(result, tuple) and len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return SessionLevelResult(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info or {}),
        )
    if isinstance(result, tuple) and len(result) == 4:
        observation, reward, terminated, info = result
        return SessionLevelResult(
            observation=observation,
            reward=float(reward),
            terminated=bool(terminated),
            truncated=False,
            info=dict(info or {}),
        )
    raise TypeError(f"unsupported level result type: {type(result)!r}")
