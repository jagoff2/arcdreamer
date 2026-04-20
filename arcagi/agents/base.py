from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from arcagi.core.inferred_state import InferredStateTracker
from arcagi.core.representation_repair import RepresentationRepairWorkspace
from arcagi.core.spatial_workspace import SpatialBeliefWorkspace
from arcagi.core.types import ActionName, GridObservation, StructuredState, Transition
from arcagi.core.types import StructuredClaim
from arcagi.memory.graph import StateGraph
from arcagi.perception.object_encoder import extract_structured_state


def _agent_visible_info(info: dict[str, object]) -> dict[str, object]:
    visible: dict[str, object] = {}
    for key, value in info.items():
        if str(key).startswith(("public_", "surface_")):
            visible[str(key)] = value
    return visible


class BaseAgent(ABC):
    def __init__(self, name: str) -> None:
        self.name = name
        self.graph = StateGraph()
        self.inferred_state = InferredStateTracker()
        self.representation_repair = RepresentationRepairWorkspace()
        self.spatial_workspace = SpatialBeliefWorkspace()
        self.last_state: StructuredState | None = None
        self.last_raw_state: StructuredState | None = None
        self.last_action: ActionName | None = None
        self.last_reward: float = 0.0
        self.last_info: dict[str, object] = {}
        self.total_reward: float = 0.0
        self.latest_language: tuple[str, ...] = ()
        self.latest_claims: tuple[StructuredClaim, ...] = ()

    def reset_episode(self) -> None:
        self.inferred_state.reset()
        self.representation_repair.reset()
        self.spatial_workspace.reset()
        self.last_state = None
        self.last_raw_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.last_info = {}
        self.total_reward = 0.0
        self.latest_language = ()
        self.latest_claims = ()

    def reset_all(self) -> None:
        self.reset_episode()
        self.graph.clear()

    def observe(self, observation: GridObservation) -> StructuredState:
        raw_state = self.representation_repair.augment(extract_structured_state(observation))
        self.last_raw_state = raw_state
        self.spatial_workspace.observe_state(raw_state)
        state = self.inferred_state.augment(raw_state)
        state = self.spatial_workspace.augment(state)
        if self.last_state is None:
            self.graph.visit(state)
        return state

    def update_after_step(
        self,
        next_observation: GridObservation,
        reward: float,
        terminated: bool,
        info: dict[str, object],
    ) -> StructuredState:
        agent_info = _agent_visible_info(info)
        extracted_next_raw_state = extract_structured_state(next_observation)
        next_raw_state = self.representation_repair.augment(extracted_next_raw_state, commit=False)
        if self.last_raw_state is not None and self.last_action is not None:
            self.representation_repair.observe_transition(
                before=self.last_raw_state,
                action=self.last_action,
                reward=reward,
                after=next_raw_state,
                terminated=terminated,
            )
            next_raw_state = self.representation_repair.augment(extracted_next_raw_state, commit=True)
            self.inferred_state.observe_transition(
                before=self.last_raw_state,
                action=self.last_action,
                reward=reward,
                after=next_raw_state,
                terminated=terminated,
            )
            self.spatial_workspace.observe_transition(
                before=self.last_raw_state,
                action=self.last_action,
                reward=reward,
                after=next_raw_state,
                terminated=terminated,
            )
        else:
            next_raw_state = self.representation_repair.augment(extracted_next_raw_state, commit=True)
        self.spatial_workspace.observe_state(next_raw_state)
        next_state = self.inferred_state.augment(next_raw_state)
        next_state = self.spatial_workspace.augment(next_state)
        if self.last_state is not None and self.last_action is not None:
            transition = Transition(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=next_state,
                terminated=terminated,
                info=agent_info,
            )
            self.graph.update(transition)
            self.on_transition(transition)
        self.total_reward += reward
        self.last_info = agent_info
        self.last_reward = reward
        self.last_raw_state = next_raw_state
        return next_state

    def sample_action(self, state: StructuredState, rng: np.random.Generator | None = None) -> ActionName:
        generator = rng or np.random.default_rng()
        return state.affordances[int(generator.integers(len(state.affordances)))]

    def on_transition(self, transition: Transition) -> None:
        return None

    @abstractmethod
    def act(self, observation: GridObservation) -> ActionName:
        raise NotImplementedError
