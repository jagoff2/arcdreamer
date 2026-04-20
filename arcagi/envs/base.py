from __future__ import annotations

from abc import ABC, abstractmethod

from arcagi.core.types import ACTION_ORDER, ActionName, GridObservation, StepResult


class BaseEnvironment(ABC):
    action_space: tuple[ActionName, ...] = ACTION_ORDER

    @property
    @abstractmethod
    def task_id(self) -> str:
        raise NotImplementedError

    @property
    @abstractmethod
    def family_id(self) -> str:
        raise NotImplementedError

    @abstractmethod
    def reset(self, seed: int | None = None) -> GridObservation:
        raise NotImplementedError

    @abstractmethod
    def step(self, action: ActionName) -> StepResult:
        raise NotImplementedError

    @abstractmethod
    def legal_actions(self) -> tuple[ActionName, ...]:
        raise NotImplementedError

    def close(self) -> None:
        return None
