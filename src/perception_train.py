from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

from .arcagi3_adapter import ArcAGI3Observation, ArcAGI3StepResult
from .perceptual_affordance import OnlinePerceptualAffordance


Transition = tuple[ArcAGI3Observation, str, ArcAGI3StepResult]


@dataclass
class OnlinePerceptionTrainer:
    memory: OnlinePerceptualAffordance = field(default_factory=OnlinePerceptualAffordance)
    updates: int = 0

    def reset(self) -> None:
        self.memory.reset()
        self.updates = 0

    def observe(self, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult) -> dict:
        self.memory.perceive(before)
        row = self.memory.observe_transition(before, action, result)
        self.memory.perceive(result.observation)
        self.updates += 1
        return row

    def fit_online(self, transitions: Iterable[Transition]) -> dict:
        rows = []
        for before, action, result in transitions:
            rows.append(self.observe(before, action, result))
        return {"updates": self.updates, "rows": rows, "diagnostics": self.memory.diagnostics()}


def fit_online(transitions: Iterable[Transition]) -> dict:
    trainer = OnlinePerceptionTrainer()
    return trainer.fit_online(transitions)
