from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from arcagi.core.types import ActionName, ObjectState, StructuredState, Transition
from arcagi.learned_online.event_tokens import (
    OUT_NO_EFFECT_NONPROGRESS,
    OUT_OBJECTIVE_PROGRESS,
    OUT_REWARD_PROGRESS,
    OUT_VISIBLE_CHANGE,
    OUTCOME_DIM,
    STATE_DELTA_DIM,
    ActionTokenBatch,
    StateTokenBatch,
    TransitionEventTargets,
    build_transition_targets,
    encode_action_tokens,
    encode_state_tokens,
    stack_action_tokens,
    stack_state_tokens,
)


@dataclass(frozen=True)
class ObjectEventCurriculumConfig:
    seed: int = 0
    train_geometries: int = 16
    heldout_geometries: int = 16
    max_objects: int = 6
    grid_size: int = 8
    include_distractors: bool = True
    require_full_dense_actions: bool = True


@dataclass(frozen=True)
class CandidateEventTargets:
    outcome: np.ndarray
    value: np.ndarray
    delta: np.ndarray
    action_mask: np.ndarray


@dataclass(frozen=True)
class ObjectEventExample:
    state: StructuredState
    next_state: StructuredState
    action: ActionName
    legal_actions: tuple[ActionName, ...]
    correct_action_index: int
    state_tokens: StateTokenBatch
    action_tokens: ActionTokenBatch
    transition_targets: TransitionEventTargets
    candidate_targets: CandidateEventTargets
    metadata: dict[str, Any]

    def transition(self) -> Transition:
        return Transition(
            state=self.state,
            action=self.action,
            reward=1.0,
            next_state=self.next_state,
            terminated=False,
            info={"score_delta": 1.0},
        )


@dataclass(frozen=True)
class ObjectEventCurriculumSplit:
    train: tuple[ObjectEventExample, ...]
    heldout: tuple[ObjectEventExample, ...]


@dataclass(frozen=True)
class ObjectEventBatch:
    state_numeric: np.ndarray
    state_type_ids: np.ndarray
    state_mask: np.ndarray
    action_numeric: np.ndarray
    action_type_ids: np.ndarray
    direction_ids: np.ndarray
    action_mask: np.ndarray
    target_outcome: np.ndarray
    target_delta: np.ndarray
    actual_action_index: np.ndarray
    candidate_outcome_targets: np.ndarray
    candidate_value_targets: np.ndarray
    candidate_delta_targets: np.ndarray
    metadata: tuple[dict[str, Any], ...]


def build_paired_color_click_curriculum(
    config: ObjectEventCurriculumConfig,
) -> ObjectEventCurriculumSplit:
    train = _examples_for_split(
        base_seed=int(config.seed) * 100_000 + 11,
        count=int(config.train_geometries),
        config=config,
        split="train",
    )
    heldout = _examples_for_split(
        base_seed=int(config.seed) * 100_000 + 53_011,
        count=int(config.heldout_geometries),
        config=config,
        split="heldout",
    )
    return ObjectEventCurriculumSplit(train=train, heldout=heldout)


def collate_object_event_examples(examples: Sequence[ObjectEventExample]) -> ObjectEventBatch:
    if not examples:
        raise ValueError("cannot collate an empty object-event example sequence")
    state_numeric, state_type_ids, state_mask = stack_state_tokens([example.state_tokens for example in examples])
    action_numeric, action_type_ids, direction_ids, action_mask = stack_action_tokens([example.action_tokens for example in examples])
    max_actions = action_mask.shape[1]
    candidate_outcome = np.zeros((len(examples), max_actions, OUTCOME_DIM), dtype=np.float32)
    candidate_value = np.zeros((len(examples), max_actions), dtype=np.float32)
    candidate_delta = np.zeros((len(examples), max_actions, STATE_DELTA_DIM), dtype=np.float32)
    target_outcome = np.zeros((len(examples), OUTCOME_DIM), dtype=np.float32)
    target_delta = np.zeros((len(examples), STATE_DELTA_DIM), dtype=np.float32)
    actual_action_index = np.zeros((len(examples),), dtype=np.int64)
    for row, example in enumerate(examples):
        count = len(example.legal_actions)
        candidate_outcome[row, :count] = example.candidate_targets.outcome
        candidate_value[row, :count] = example.candidate_targets.value
        candidate_delta[row, :count] = example.candidate_targets.delta
        target_outcome[row] = example.transition_targets.outcome
        target_delta[row] = example.transition_targets.delta
        actual_action_index[row] = int(example.correct_action_index)
    return ObjectEventBatch(
        state_numeric=state_numeric,
        state_type_ids=state_type_ids,
        state_mask=state_mask,
        action_numeric=action_numeric,
        action_type_ids=action_type_ids,
        direction_ids=direction_ids,
        action_mask=action_mask,
        target_outcome=target_outcome,
        target_delta=target_delta,
        actual_action_index=actual_action_index,
        candidate_outcome_targets=candidate_outcome,
        candidate_value_targets=candidate_value,
        candidate_delta_targets=candidate_delta,
        metadata=tuple(dict(example.metadata) for example in examples),
    )


def _examples_for_split(
    *,
    base_seed: int,
    count: int,
    config: ObjectEventCurriculumConfig,
    split: str,
) -> tuple[ObjectEventExample, ...]:
    examples: list[ObjectEventExample] = []
    for geometry_index in range(max(int(count), 0)):
        geometry_seed = int(base_seed + geometry_index)
        examples.extend(_paired_examples_for_geometry(geometry_seed=geometry_seed, config=config, split=split))
    return tuple(examples)


def _paired_examples_for_geometry(
    *,
    geometry_seed: int,
    config: ObjectEventCurriculumConfig,
    split: str,
) -> tuple[ObjectEventExample, ObjectEventExample]:
    geometry = _sample_geometry(geometry_seed=geometry_seed, config=config)
    return (
        _example_from_geometry(geometry, cue_mode=0, split=split),
        _example_from_geometry(geometry, cue_mode=1, split=split),
    )


def _sample_geometry(*, geometry_seed: int, config: ObjectEventCurriculumConfig) -> dict[str, Any]:
    rng = np.random.default_rng(int(geometry_seed))
    grid_size = int(config.grid_size)
    if grid_size < 4:
        raise ValueError("object-event curriculum requires grid_size >= 4")
    occupied: set[tuple[int, int]] = {(0, 0)}
    red_pos = _random_free_cell(rng, grid_size=grid_size, occupied=occupied)
    occupied.add(red_pos)
    blue_pos = _random_free_cell(rng, grid_size=grid_size, occupied=occupied)
    occupied.add(blue_pos)
    distractors: list[tuple[int, int, int]] = []
    if config.include_distractors:
        for index in range(max(int(config.max_objects) - 3, 0)):
            cell = _random_free_cell(rng, grid_size=grid_size, occupied=occupied)
            occupied.add(cell)
            color = (6 + index) % 12
            if color in {1, 2, 5}:
                color = 9
            distractors.append((int(cell[0]), int(cell[1]), int(color)))
    actions = _dense_actions(grid_size=grid_size) if config.require_full_dense_actions else _sparse_actions(red_pos, blue_pos, grid_size)
    roles = {action: "click" for action in actions if action.startswith("click:")}
    roles.update({"0": "reset_level", "undo": "undo", "up": "move_up", "down": "move_down"})
    return {
        "geometry_seed": int(geometry_seed),
        "grid_size": int(grid_size),
        "red_pos": red_pos,
        "blue_pos": blue_pos,
        "distractors": tuple(distractors),
        "actions": tuple(actions),
        "action_roles": tuple(sorted(roles.items())),
    }


def _example_from_geometry(
    geometry: dict[str, Any],
    *,
    cue_mode: int,
    split: str,
) -> ObjectEventExample:
    grid_size = int(geometry["grid_size"])
    cue_color = 1 if int(cue_mode) == 0 else 2
    cue = _object("cue", cue_color, (0, 0), tags=("agent", "clickable"))
    red_pos = tuple(geometry["red_pos"])
    blue_pos = tuple(geometry["blue_pos"])
    red = _object("red", 2, red_pos)
    blue = _object("blue", 5, blue_pos)
    distractors = tuple(
        _object(f"distractor_{index}", int(color), (int(row), int(col)))
        for index, (row, col, color) in enumerate(geometry["distractors"])
    )
    objects = (cue, red, blue, *distractors)
    grid = np.zeros((grid_size, grid_size), dtype=np.int64)
    for obj in objects:
        for row, col in obj.cells:
            grid[row, col] = int(obj.color)
    legal_actions = tuple(geometry["actions"])
    correct_action = _click_action(red_pos) if int(cue_mode) == 0 else _click_action(blue_pos)
    state = StructuredState(
        task_id="synthetic_object_event",
        episode_id=f"{split}_{geometry['geometry_seed']}",
        step_index=0,
        grid_shape=grid.shape,
        grid_signature=tuple(int(value) for value in grid.reshape(-1)),
        objects=objects,
        relations=(),
        affordances=legal_actions,
        action_roles=tuple(geometry["action_roles"]),
    )
    next_state = StructuredState(
        task_id=state.task_id,
        episode_id=state.episode_id,
        step_index=1,
        grid_shape=state.grid_shape,
        grid_signature=state.grid_signature,
        objects=state.objects,
        relations=(),
        affordances=state.affordances,
        action_roles=state.action_roles,
        flags=(("synthetic_success_latch", "1"),),
    )
    transition = Transition(
        state=state,
        action=correct_action,
        reward=1.0,
        next_state=next_state,
        terminated=False,
        info={"score_delta": 1.0},
    )
    transition_targets = build_transition_targets(transition, actions=legal_actions)
    candidate_targets = _candidate_targets(
        legal_actions=legal_actions,
        correct_action_index=int(transition_targets.actual_action_index),
        transition_targets=transition_targets,
    )
    metadata = {
        "curriculum": "paired_color_click",
        "split": str(split),
        "geometry_seed": int(geometry["geometry_seed"]),
        "cue_mode": int(cue_mode),
        "red_action_index": int(legal_actions.index(_click_action(red_pos))),
        "blue_action_index": int(legal_actions.index(_click_action(blue_pos))),
        "correct_action_index": int(transition_targets.actual_action_index),
        "legal_action_count": int(len(legal_actions)),
    }
    return ObjectEventExample(
        state=state,
        next_state=next_state,
        action=correct_action,
        legal_actions=legal_actions,
        correct_action_index=int(transition_targets.actual_action_index),
        state_tokens=encode_state_tokens(state),
        action_tokens=encode_action_tokens(state, legal_actions),
        transition_targets=transition_targets,
        candidate_targets=candidate_targets,
        metadata=metadata,
    )


def _candidate_targets(
    *,
    legal_actions: tuple[ActionName, ...],
    correct_action_index: int,
    transition_targets: TransitionEventTargets,
) -> CandidateEventTargets:
    action_count = len(legal_actions)
    outcome = np.zeros((action_count, OUTCOME_DIM), dtype=np.float32)
    outcome[:, OUT_NO_EFFECT_NONPROGRESS] = 1.0
    value = np.zeros((action_count,), dtype=np.float32)
    delta = np.zeros((action_count, STATE_DELTA_DIM), dtype=np.float32)
    index = int(correct_action_index)
    outcome[index, :] = 0.0
    outcome[index, OUT_VISIBLE_CHANGE] = 1.0
    outcome[index, OUT_OBJECTIVE_PROGRESS] = 1.0
    outcome[index, OUT_REWARD_PROGRESS] = 1.0
    value[index] = 1.0
    delta[index] = transition_targets.delta
    return CandidateEventTargets(
        outcome=outcome,
        value=value,
        delta=delta,
        action_mask=np.ones((action_count,), dtype=bool),
    )


def _dense_actions(*, grid_size: int) -> tuple[ActionName, ...]:
    clicks = tuple(f"click:{col}:{row}" for row in range(int(grid_size)) for col in range(int(grid_size)))
    return clicks + ("0", "undo", "up", "down")


def _sparse_actions(red_pos: tuple[int, int], blue_pos: tuple[int, int], grid_size: int) -> tuple[ActionName, ...]:
    actions = [_click_action(red_pos), _click_action(blue_pos)]
    occupied = {red_pos, blue_pos}
    rng = np.random.default_rng(int(red_pos[0] * 997 + blue_pos[1] * 101 + grid_size))
    while len(actions) < 12:
        cell = _random_free_cell(rng, grid_size=grid_size, occupied=occupied)
        occupied.add(cell)
        actions.append(_click_action(cell))
    return tuple(actions) + ("0", "undo", "up", "down")


def _object(name: str, color: int, cell: tuple[int, int], *, tags: tuple[str, ...] = ()) -> ObjectState:
    row, col = cell
    return ObjectState(
        object_id=name,
        color=int(color),
        cells=((int(row), int(col)),),
        bbox=(int(row), int(col), int(row), int(col)),
        centroid=(float(row), float(col)),
        area=1,
        tags=tags,
    )


def _click_action(cell: tuple[int, int]) -> ActionName:
    row, col = cell
    return f"click:{int(col)}:{int(row)}"


def _random_free_cell(
    rng: np.random.Generator,
    *,
    grid_size: int,
    occupied: set[tuple[int, int]],
) -> tuple[int, int]:
    while True:
        cell = (int(rng.integers(1, int(grid_size))), int(rng.integers(0, int(grid_size))))
        if cell not in occupied:
            return cell
