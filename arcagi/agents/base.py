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


def _state_inventory_value(state: StructuredState, key: str, default: str = "") -> str:
    inventory = state.inventory_dict()
    flags = state.flags_dict()
    value = inventory.get(key, flags.get(key, default))
    return str(value)


def _state_levels_completed(state: StructuredState) -> int:
    raw_value = _state_inventory_value(state, "interface_levels_completed", "0")
    try:
        return int(raw_value)
    except Exception:
        return 0


def _state_game_state(state: StructuredState) -> str:
    return _state_inventory_value(state, "interface_game_state", "")


def _is_reset_action(state: StructuredState, action: ActionName | None) -> bool:
    if action is None:
        return False
    role = dict(state.action_roles).get(action, "")
    return str(action) == "0" or "reset" in str(role)


def _visible_transition_info(
    raw_info: dict[str, object],
    *,
    before: StructuredState,
    after: StructuredState,
    action: ActionName,
) -> dict[str, object]:
    visible = _agent_visible_info(raw_info)
    before_levels = _state_levels_completed(before)
    after_levels = _state_levels_completed(after)
    level_delta = max(after_levels - before_levels, 0)
    reset_action = _is_reset_action(before, action)
    visible.update(
        {
            "levels_completed_before": before_levels,
            "levels_completed_after": after_levels,
            "level_delta": level_delta,
            "level_boundary": bool(reset_action or level_delta > 0),
            "reset_action": bool(reset_action),
            "game_state_before": _state_game_state(before),
            "game_state_after": _state_game_state(after),
        }
    )
    return visible


def _merge_visible_info_into_observation(
    observation: GridObservation,
    info: dict[str, object],
) -> GridObservation:
    visible = _agent_visible_info(info)
    if not visible:
        return observation
    extras = dict(observation.extras)
    inventory = {str(key): str(value) for key, value in extras.get("inventory", {}).items()}
    flags = {str(key): str(value) for key, value in extras.get("flags", {}).items()}
    cell_tags = {
        tuple(cell): tuple(str(tag) for tag in tags)
        for cell, tags in extras.get("cell_tags", {}).items()
    }
    for key, value in visible.items():
        if key in {"public_inventory", "surface_inventory"} and isinstance(value, dict):
            inventory.update({str(item_key): str(item_value) for item_key, item_value in value.items()})
            continue
        if key in {"public_flags", "surface_flags"} and isinstance(value, dict):
            flags.update({str(item_key): str(item_value) for item_key, item_value in value.items()})
            continue
        if key in {"public_cell_tags", "surface_cell_tags"} and isinstance(value, dict):
            for cell, tags_value in value.items():
                normalized_cell = tuple(cell)
                existing = set(cell_tags.get(normalized_cell, ()))
                if isinstance(tags_value, (tuple, list)):
                    existing.update(str(tag) for tag in tags_value)
                else:
                    existing.add(str(tags_value))
                cell_tags[normalized_cell] = tuple(sorted(existing))
            continue
        extras[key] = value
    if inventory:
        extras["inventory"] = inventory
    if flags:
        extras["flags"] = flags
    if cell_tags:
        extras["cell_tags"] = cell_tags
    return GridObservation(
        task_id=observation.task_id,
        episode_id=observation.episode_id,
        step_index=observation.step_index,
        grid=observation.grid,
        available_actions=observation.available_actions,
        extras=extras,
    )


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

    def reset_level(self) -> None:
        reset_level = getattr(self.inferred_state, "reset_level", None)
        if callable(reset_level):
            reset_level()
        else:
            self.inferred_state.reset()
        reset_level = getattr(self.representation_repair, "reset_level", None)
        if callable(reset_level):
            reset_level()
        else:
            self.representation_repair.reset()
        reset_level = getattr(self.spatial_workspace, "reset_level", None)
        if callable(reset_level):
            reset_level()
        else:
            self.spatial_workspace.reset()
        self.last_state = None
        self.last_raw_state = None
        self.last_action = None
        self.last_reward = 0.0
        self.last_info = {}
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
        merged_observation = _merge_visible_info_into_observation(next_observation, info)
        extracted_next_raw_state = extract_structured_state(merged_observation)
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
            transition_info = _visible_transition_info(
                info,
                before=self.last_state,
                after=next_state,
                action=self.last_action,
            )
            transition = Transition(
                state=self.last_state,
                action=self.last_action,
                reward=reward,
                next_state=next_state,
                terminated=terminated,
                info=transition_info,
            )
            self.graph.update(transition)
            self.on_transition(transition)
        self.total_reward += reward
        self.last_info = {}
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
