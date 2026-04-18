from __future__ import annotations

import numpy as np

from arcagi.agents.base import BaseAgent
from arcagi.core.types import ActionName, GridObservation


class RandomHeuristicAgent(BaseAgent):
    def __init__(self, seed: int = 0) -> None:
        super().__init__(name="random")
        self.rng = np.random.default_rng(seed)

    def act(self, observation: GridObservation) -> ActionName:
        state = self.observe(observation)
        non_wait = [action for action in state.affordances if action != "wait"]
        action = non_wait[int(self.rng.integers(len(non_wait)))] if non_wait else "wait"
        self.last_state = state
        self.last_action = action
        return action

