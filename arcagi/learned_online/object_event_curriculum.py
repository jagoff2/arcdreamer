from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

from arcagi.core.action_schema import click_action_to_grid_cell
from arcagi.core.types import ActionName, GridObservation, ObjectState, StructuredState, Transition
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
class ParametricActionSurfaceConfig:
    kind: str = "dense_8x8"
    action_surface_size: int = 68
    coordinate_grid_size: int = 64
    empty_click_fraction: float = 0.80
    positive_region_radius: int = 1
    include_controls: bool = True
    misc_actions: tuple[ActionName, ...] = ("0", "undo", "up", "down")


@dataclass(frozen=True)
class ObjectEventCurriculumConfig:
    seed: int = 0
    train_geometries: int = 16
    heldout_geometries: int = 16
    max_objects: int = 6
    grid_size: int = 8
    include_distractors: bool = True
    require_full_dense_actions: bool = True
    action_surface: str = "dense_8x8"
    action_surface_size: int = 68
    coordinate_grid_size: int = 64
    empty_click_fraction: float = 0.80
    positive_region_radius: int = 1


@dataclass(frozen=True)
class OnlineObjectEventCurriculumConfig:
    seed: int = 0
    train_sessions: int = 32
    heldout_sessions: int = 32
    levels_per_session: int = 3
    max_objects: int = 6
    grid_size: int = 8
    include_distractors: bool = True
    require_full_dense_actions: bool = True
    curriculum: str = "latent_rule_color_click"
    palette_size: int = 8
    require_role_balanced_colors: bool = False
    action_surface: str = "dense_8x8"
    action_surface_size: int = 68
    coordinate_grid_size: int = 64
    empty_click_fraction: float = 0.80
    positive_region_radius: int = 1


@dataclass(frozen=True)
class ActiveOnlineObjectEventConfig:
    seed: int = 0
    train_sessions: int = 32
    heldout_sessions: int = 32
    levels_per_session: int = 3
    grid_size: int = 8
    max_distractors: int = 1
    include_distractors: bool = True
    max_steps_per_level: int = 3
    curriculum: str = "latent_rule_color_click"
    palette_size: int = 8
    require_role_balanced_colors: bool = False
    action_surface: str = "dense_8x8"
    action_surface_size: int = 68
    coordinate_grid_size: int = 64
    empty_click_fraction: float = 0.80
    positive_region_radius: int = 1


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
    positive_action_mask: np.ndarray | None = None
    positive_action_indices: tuple[int, ...] = ()

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
class OnlineObjectEventLevel:
    session_index: int
    level_index: int
    geometry_seed: int
    cue_mode: int
    latent_rule: int
    example: ObjectEventExample


@dataclass(frozen=True)
class OnlineObjectEventSession:
    session_index: int
    latent_rule: int
    levels: tuple[OnlineObjectEventLevel, ...]


@dataclass(frozen=True)
class OnlineObjectEventCurriculumSplit:
    train: tuple[OnlineObjectEventSession, ...]
    heldout: tuple[OnlineObjectEventSession, ...]


@dataclass(frozen=True)
class ActiveStepResult:
    transition_targets: TransitionEventTargets
    selected_action_index: int
    reward: float
    success: bool
    no_effect: bool
    levels_completed: int
    metadata: dict[str, Any]


PUBLIC_SYNTHETIC_CELL_TAGS = frozenset({"agent", "clickable"})


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


def build_online_object_event_curriculum(
    config: OnlineObjectEventCurriculumConfig,
) -> OnlineObjectEventCurriculumSplit:
    if config.curriculum not in {"latent_rule_color_click", "latent_rule_variable_palette"}:
        raise ValueError(f"unknown online object-event curriculum: {config.curriculum!r}")
    return OnlineObjectEventCurriculumSplit(
        train=generate_online_object_event_sessions(config, split="train"),
        heldout=generate_online_object_event_sessions(config, split="heldout"),
    )


def build_active_online_object_event_curriculum(
    config: ActiveOnlineObjectEventConfig,
) -> OnlineObjectEventCurriculumSplit:
    return build_online_object_event_curriculum(
        OnlineObjectEventCurriculumConfig(
            seed=int(config.seed),
            train_sessions=int(config.train_sessions),
            heldout_sessions=int(config.heldout_sessions),
            levels_per_session=int(config.levels_per_session),
            max_objects=3 + max(int(config.max_distractors), 0),
            grid_size=int(config.grid_size),
            include_distractors=bool(config.include_distractors and int(config.max_distractors) > 0),
            require_full_dense_actions=True,
            curriculum=str(config.curriculum),
            palette_size=int(config.palette_size),
            require_role_balanced_colors=bool(config.require_role_balanced_colors),
            action_surface=str(config.action_surface),
            action_surface_size=int(config.action_surface_size),
            coordinate_grid_size=int(config.coordinate_grid_size),
            empty_click_fraction=float(config.empty_click_fraction),
            positive_region_radius=int(config.positive_region_radius),
        )
    )


def apply_synthetic_object_event_action(
    level: OnlineObjectEventLevel,
    selected_action_index: int,
) -> ActiveStepResult:
    example = level.example
    index = int(selected_action_index)
    if index < 0 or index >= len(example.legal_actions):
        raise IndexError(f"selected action index {index} outside legal surface of {len(example.legal_actions)} actions")
    if example.positive_action_mask is None:
        success = index == int(example.correct_action_index)
    else:
        success = bool(np.asarray(example.positive_action_mask, dtype=bool)[index])
    outcome = np.asarray(example.candidate_targets.outcome[index], dtype=np.float32)
    delta = np.asarray(example.candidate_targets.delta[index], dtype=np.float32)
    reward = float(example.candidate_targets.value[index])
    targets = TransitionEventTargets(
        outcome=outcome,
        delta=delta,
        actual_action_index=index,
        reward=reward,
        terminated=False,
    )
    return ActiveStepResult(
        transition_targets=targets,
        selected_action_index=index,
        reward=reward,
        success=bool(success),
        no_effect=not bool(success),
        levels_completed=1 if success else 0,
        metadata={
            "curriculum": str(example.metadata.get("curriculum", "latent_rule_color_click")),
            "session_index": int(level.session_index),
            "level_index": int(level.level_index),
            "latent_rule": int(level.latent_rule),
            "cue_mode": int(level.cue_mode),
            "selected_action_index": index,
            "correct_action_index": int(example.correct_action_index),
            "positive_action_count": int(len(example.positive_action_indices) if example.positive_action_indices else 1),
            "oracle_support_used": False,
        },
    )


def state_to_grid_observation(
    state: StructuredState,
    actions: Sequence[ActionName] | None = None,
    *,
    step_index: int | None = None,
) -> GridObservation:
    action_tuple = tuple(state.affordances if actions is None else actions)
    inventory = state.inventory_dict()
    extras: dict[str, Any] = {
        "action_roles": dict(state.action_roles),
        "cell_tags": _public_cell_tags(state),
        "background_color": 0,
    }
    if state.inventory:
        extras["inventory"] = dict(state.inventory)
    if state.flags:
        extras["flags"] = dict(state.flags)
    if "interface_display_scale" in inventory:
        extras["display_scale"] = int(inventory["interface_display_scale"])
    if "interface_display_pad_x" in inventory or "interface_display_pad_y" in inventory:
        extras["display_padding"] = (
            int(inventory.get("interface_display_pad_x", "0")),
            int(inventory.get("interface_display_pad_y", "0")),
        )
    if "interface_camera_height" in inventory and "interface_camera_width" in inventory:
        extras["camera_grid_shape"] = (
            int(inventory["interface_camera_height"]),
            int(inventory["interface_camera_width"]),
        )
    return GridObservation(
        task_id=state.task_id,
        episode_id=state.episode_id,
        step_index=int(state.step_index if step_index is None else step_index),
        grid=state.as_grid(),
        available_actions=action_tuple,
        extras=extras,
    )


def level_to_grid_observation(
    level: OnlineObjectEventLevel,
    *,
    step_index: int = 0,
) -> GridObservation:
    return state_to_grid_observation(level.example.state, level.example.legal_actions, step_index=step_index)


def apply_synthetic_object_event_action_to_grid(
    level: OnlineObjectEventLevel,
    selected_action_index: int,
    *,
    before_step_index: int,
) -> tuple[GridObservation, GridObservation, ActiveStepResult]:
    result = apply_synthetic_object_event_action(level, selected_action_index)
    before = level_to_grid_observation(level, step_index=int(before_step_index))
    after_state = level.example.next_state if bool(result.success) else level.example.state
    after = state_to_grid_observation(after_state, level.example.legal_actions, step_index=int(before_step_index) + 1)
    return before, after, result


def rebuild_object_event_example_with_states(
    example: ObjectEventExample,
    *,
    state: StructuredState,
    next_state: StructuredState,
    metadata_extra: dict[str, Any] | None = None,
) -> ObjectEventExample:
    transition = Transition(
        state=state,
        action=example.action,
        reward=1.0,
        next_state=next_state,
        terminated=False,
        info={"score_delta": 1.0},
    )
    transition_targets = build_transition_targets(transition, actions=example.legal_actions)
    candidate_targets = _candidate_targets(
        legal_actions=example.legal_actions,
        correct_action_index=int(transition_targets.actual_action_index),
        transition_targets=transition_targets,
        positive_action_indices=example.positive_action_indices,
    )
    metadata = dict(example.metadata)
    if metadata_extra:
        metadata.update(dict(metadata_extra))
    return ObjectEventExample(
        state=state,
        next_state=next_state,
        action=example.action,
        legal_actions=example.legal_actions,
        correct_action_index=int(transition_targets.actual_action_index),
        state_tokens=encode_state_tokens(state),
        action_tokens=encode_action_tokens(state, example.legal_actions),
        transition_targets=transition_targets,
        candidate_targets=candidate_targets,
        metadata=metadata,
        positive_action_mask=None if example.positive_action_mask is None else np.asarray(example.positive_action_mask, dtype=bool).copy(),
        positive_action_indices=tuple(int(index) for index in example.positive_action_indices),
    )


def generate_online_object_event_sessions(
    config: OnlineObjectEventCurriculumConfig,
    *,
    split: str,
) -> tuple[OnlineObjectEventSession, ...]:
    if int(config.levels_per_session) < 2:
        raise ValueError("online object-event sessions require levels_per_session >= 2")
    count = int(config.train_sessions if split == "train" else config.heldout_sessions)
    base_seed = int(config.seed) * 1_000_000 + (101 if split == "train" else 701_101)
    sessions: list[OnlineObjectEventSession] = []
    for session_index in range(max(count, 0)):
        session_rng = np.random.default_rng(int(base_seed + session_index * 9_973))
        latent_rule = int(session_rng.integers(0, 2))
        cue_modes = [int(session_rng.integers(0, 2)) for _ in range(int(config.levels_per_session))]
        if len(set(cue_modes)) == 1:
            cue_modes[-1] = 1 - cue_modes[0]
        levels: list[OnlineObjectEventLevel] = []
        for level_index in range(int(config.levels_per_session)):
            cue_mode = int(cue_modes[level_index])
            geometry_seed = int(base_seed + session_index * 10_000 + level_index)
            if config.curriculum == "latent_rule_variable_palette":
                example = make_latent_rule_variable_palette_example(
                    config=config,
                    geometry_seed=geometry_seed,
                    cue_mode=cue_mode,
                    latent_rule=latent_rule,
                    split=split,
                    session_index=session_index,
                    level_index=level_index,
                )
            else:
                example = make_latent_rule_color_click_example(
                    config=config,
                    geometry_seed=geometry_seed,
                    cue_mode=cue_mode,
                    latent_rule=latent_rule,
                    split=split,
                    session_index=session_index,
                    level_index=level_index,
                )
            levels.append(
                OnlineObjectEventLevel(
                    session_index=int(session_index),
                    level_index=int(level_index),
                    geometry_seed=geometry_seed,
                    cue_mode=cue_mode,
                    latent_rule=latent_rule,
                    example=example,
                )
            )
        sessions.append(
            OnlineObjectEventSession(
                session_index=int(session_index),
                latent_rule=latent_rule,
                levels=tuple(levels),
            )
        )
    return tuple(sessions)


def make_latent_rule_color_click_example(
    *,
    config: OnlineObjectEventCurriculumConfig,
    geometry_seed: int,
    cue_mode: int,
    latent_rule: int,
    split: str,
    session_index: int = 0,
    level_index: int = 0,
) -> ObjectEventExample:
    geometry = _sample_geometry(
        geometry_seed=int(geometry_seed),
        config=ObjectEventCurriculumConfig(
            seed=int(config.seed),
            train_geometries=0,
            heldout_geometries=0,
            max_objects=int(config.max_objects),
            grid_size=int(config.grid_size),
            include_distractors=bool(config.include_distractors),
            require_full_dense_actions=bool(config.require_full_dense_actions),
            action_surface=str(config.action_surface),
            action_surface_size=int(config.action_surface_size),
            coordinate_grid_size=int(config.coordinate_grid_size),
            empty_click_fraction=float(config.empty_click_fraction),
            positive_region_radius=int(config.positive_region_radius),
        ),
    )
    target_mode = int(cue_mode) if int(latent_rule) == 0 else 1 - int(cue_mode)
    return _example_from_geometry(
        geometry,
        cue_mode=int(cue_mode),
        split=split,
        target_mode=target_mode,
        metadata_extra={
            "curriculum": "latent_rule_color_click",
            "session_index": int(session_index),
            "level_index": int(level_index),
            "latent_rule": int(latent_rule),
        },
    )


def make_latent_rule_variable_palette_example(
    *,
    config: OnlineObjectEventCurriculumConfig,
    geometry_seed: int,
    cue_mode: int,
    latent_rule: int,
    split: str,
    session_index: int = 0,
    level_index: int = 0,
) -> ObjectEventExample:
    palette = _variable_palette_colors(int(config.palette_size))
    target_mode = int(cue_mode) if int(latent_rule) == 0 else 1 - int(cue_mode)
    target_colors = _variable_palette_slot_colors(
        palette=palette,
        session_index=int(session_index),
        level_index=int(level_index),
        split=split,
    )
    distractor_colors = _variable_palette_distractor_colors(
        palette=palette,
        used_colors=set(target_colors),
        session_index=int(session_index),
        level_index=int(level_index),
        count=max(int(config.max_objects) - 3, 0),
    )
    geometry = _sample_geometry_with_colors(
        geometry_seed=int(geometry_seed),
        config=ObjectEventCurriculumConfig(
            seed=int(config.seed),
            train_geometries=0,
            heldout_geometries=0,
            max_objects=int(config.max_objects),
            grid_size=int(config.grid_size),
            include_distractors=bool(config.include_distractors),
            require_full_dense_actions=bool(config.require_full_dense_actions),
            action_surface=str(config.action_surface),
            action_surface_size=int(config.action_surface_size),
            coordinate_grid_size=int(config.coordinate_grid_size),
            empty_click_fraction=float(config.empty_click_fraction),
            positive_region_radius=int(config.positive_region_radius),
        ),
        slot_colors=target_colors,
        distractor_colors=distractor_colors,
        generic_object_ids=True,
    )
    return _example_from_geometry(
        geometry,
        cue_mode=int(cue_mode),
        split=split,
        target_mode=target_mode,
        metadata_extra={
            "curriculum": "latent_rule_variable_palette",
            "session_index": int(session_index),
            "level_index": int(level_index),
            "latent_rule": int(latent_rule),
            "palette_colors": tuple(int(color) for color in palette),
            "slot0_color": int(target_colors[0]),
            "slot1_color": int(target_colors[1]),
            "distractor_colors": tuple(int(color) for color in distractor_colors),
        },
    )
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
    distractor_count = max(int(config.max_objects) - 3, 0) if config.include_distractors else 0
    return _sample_geometry_with_colors(
        geometry_seed=geometry_seed,
        config=config,
        slot_colors=(2, 5),
        distractor_colors=tuple(_default_distractor_color(index) for index in range(distractor_count)),
        generic_object_ids=False,
    )


def _sample_geometry_with_colors(
    *,
    geometry_seed: int,
    config: ObjectEventCurriculumConfig,
    slot_colors: tuple[int, int],
    distractor_colors: tuple[int, ...],
    generic_object_ids: bool,
) -> dict[str, Any]:
    rng = np.random.default_rng(int(geometry_seed))
    grid_size = int(config.grid_size)
    if grid_size < 4:
        raise ValueError("object-event curriculum requires grid_size >= 4")
    occupied: set[tuple[int, int]] = {(0, 0)}
    use_spatial_hint = bool(generic_object_ids and int(geometry_seed) % 3 != 0)
    red_pos = (
        _random_free_cell_near_origin(rng, grid_size=grid_size, occupied=occupied)
        if use_spatial_hint
        else _random_free_cell(rng, grid_size=grid_size, occupied=occupied)
    )
    occupied.add(red_pos)
    blue_pos = (
        _random_free_cell_near_origin(rng, grid_size=grid_size, occupied=occupied)
        if use_spatial_hint
        else _random_free_cell(rng, grid_size=grid_size, occupied=occupied)
    )
    occupied.add(blue_pos)
    distractors: list[tuple[int, int, int]] = []
    if config.include_distractors:
        for index, color in enumerate(tuple(distractor_colors)[: max(int(config.max_objects) - 3, 0)]):
            cell = (
                _random_free_cell_far_from_origin(rng, grid_size=grid_size, occupied=occupied)
                if use_spatial_hint
                else _random_free_cell(rng, grid_size=grid_size, occupied=occupied)
            )
            occupied.add(cell)
            distractors.append((int(cell[0]), int(cell[1]), int(color)))
    actions, surface_metadata = _actions_for_surface(
        red_pos=red_pos,
        blue_pos=blue_pos,
        distractors=tuple((int(row), int(col)) for row, col, _color in distractors),
        geometry_seed=int(geometry_seed),
        grid_size=grid_size,
        config=config,
    )
    roles = {action: "click" for action in actions if action.startswith("click:")}
    roles.update({"0": "reset_level", "undo": "undo", "up": "move_up", "down": "move_down"})
    slot0_id = "object_0" if generic_object_ids else "red"
    slot1_id = "object_1" if generic_object_ids else "blue"
    distractor_ids = tuple(f"object_{index + 2}" if generic_object_ids else f"distractor_{index}" for index in range(len(distractors)))
    return {
        "geometry_seed": int(geometry_seed),
        "grid_size": int(grid_size),
        "red_pos": red_pos,
        "blue_pos": blue_pos,
        "slot0_color": int(slot_colors[0]),
        "slot1_color": int(slot_colors[1]),
        "slot0_object_id": slot0_id,
        "slot1_object_id": slot1_id,
        "distractors": tuple(distractors),
        "distractor_object_ids": distractor_ids,
        "actions": tuple(actions),
        "action_roles": tuple(sorted(roles.items())),
        "action_surface_metadata": surface_metadata,
    }


def _default_distractor_color(index: int) -> int:
    color = (6 + int(index)) % 12
    if color in {1, 2, 5}:
        color = 9
    return int(color)


def _variable_palette_colors(palette_size: int) -> tuple[int, ...]:
    size = max(min(int(palette_size), 9), 3)
    colors: list[int] = []
    cursor = 3
    while len(colors) < size:
        color = int(cursor % 12)
        cursor += 1
        if color in {0, 1, 2} or color in colors:
            continue
        colors.append(color)
    return tuple(colors)


def _variable_palette_slot_colors(
    *,
    palette: tuple[int, ...],
    session_index: int,
    level_index: int,
    split: str,
) -> tuple[int, int]:
    offset = 0 if split == "train" else max(len(palette) // 2, 1)
    del level_index
    base = int(session_index) + offset
    first = int(palette[base % len(palette)])
    second_cursor = base + max(len(palette) // 2, 1)
    second = int(palette[second_cursor % len(palette)])
    if second == first:
        second = int(palette[(second_cursor + 1) % len(palette)])
    return first, second


def _variable_palette_distractor_colors(
    *,
    palette: tuple[int, ...],
    used_colors: set[int],
    session_index: int,
    level_index: int,
    count: int,
) -> tuple[int, ...]:
    colors: list[int] = []
    cursor = int(session_index) * 3 + int(level_index) * 5 + 1
    while len(colors) < max(int(count), 0):
        color = int(palette[cursor % len(palette)])
        cursor += 1
        if color in used_colors or color in colors:
            continue
        colors.append(color)
    return tuple(colors)


def _example_from_geometry(
    geometry: dict[str, Any],
    *,
    cue_mode: int,
    split: str,
    target_mode: int | None = None,
    metadata_extra: dict[str, Any] | None = None,
) -> ObjectEventExample:
    grid_size = int(geometry["grid_size"])
    cue_color = 1 if int(cue_mode) == 0 else 2
    cue = _object("cue", cue_color, (0, 0), tags=("agent", "clickable"))
    red_pos = tuple(geometry["red_pos"])
    blue_pos = tuple(geometry["blue_pos"])
    slot0_color = int(geometry.get("slot0_color", 2))
    slot1_color = int(geometry.get("slot1_color", 5))
    slot0_id = str(geometry.get("slot0_object_id", "red"))
    slot1_id = str(geometry.get("slot1_object_id", "blue"))
    red = _object(slot0_id, slot0_color, red_pos)
    blue = _object(slot1_id, slot1_color, blue_pos)
    distractor_ids = tuple(str(item) for item in geometry.get("distractor_object_ids", ()))
    distractors = tuple(
        _object(
            distractor_ids[index] if index < len(distractor_ids) else f"distractor_{index}",
            int(color),
            (int(row), int(col)),
        )
        for index, (row, col, color) in enumerate(geometry["distractors"])
    )
    objects = (cue, red, blue, *distractors)
    grid = np.zeros((grid_size, grid_size), dtype=np.int64)
    for obj in objects:
        for row, col in obj.cells:
            grid[row, col] = int(obj.color)
    legal_actions = tuple(geometry["actions"])
    effective_target_mode = int(cue_mode) if target_mode is None else int(target_mode)
    surface_metadata = dict(geometry.get("action_surface_metadata", {}))
    red_positive_actions = _slot_positive_actions(
        surface_metadata,
        slot="red",
        fallback=_click_action(red_pos),
    )
    blue_positive_actions = _slot_positive_actions(
        surface_metadata,
        slot="blue",
        fallback=_click_action(blue_pos),
    )
    target_positive_actions = red_positive_actions if effective_target_mode == 0 else blue_positive_actions
    wrong_positive_actions = blue_positive_actions if effective_target_mode == 0 else red_positive_actions
    positive_action_indices = tuple(int(legal_actions.index(action)) for action in target_positive_actions if action in legal_actions)
    if not positive_action_indices:
        raise ValueError("object-event example has no positive legal target actions")
    correct_action = legal_actions[int(positive_action_indices[0])]
    red_action_index = int(legal_actions.index(red_positive_actions[0]))
    blue_action_index = int(legal_actions.index(blue_positive_actions[0]))
    red_action_indices = tuple(int(legal_actions.index(action)) for action in red_positive_actions if action in legal_actions)
    blue_action_indices = tuple(int(legal_actions.index(action)) for action in blue_positive_actions if action in legal_actions)
    wrong_action_indices = tuple(int(legal_actions.index(action)) for action in wrong_positive_actions if action in legal_actions)
    positive_action_mask = np.zeros((len(legal_actions),), dtype=bool)
    positive_action_mask[list(positive_action_indices)] = True
    inventory = _surface_inventory(surface_metadata, grid_size=grid_size)
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
        inventory=inventory,
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
        inventory=state.inventory,
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
        positive_action_indices=positive_action_indices,
    )
    metadata = {
        "curriculum": "paired_color_click",
        "split": str(split),
        "geometry_seed": int(geometry["geometry_seed"]),
        "cue_mode": int(cue_mode),
        "target_mode": int(effective_target_mode),
        "red_action_index": red_action_index,
        "blue_action_index": blue_action_index,
        "slot0_action_index": red_action_index,
        "slot1_action_index": blue_action_index,
        "red_action_indices": red_action_indices,
        "blue_action_indices": blue_action_indices,
        "positive_action_indices": positive_action_indices,
        "wrong_action_indices": wrong_action_indices,
        "candidate_action_indices": tuple(red_action_indices + blue_action_indices),
        "target_color": int(slot0_color if int(effective_target_mode) == 0 else slot1_color),
        "correct_action_index": int(transition_targets.actual_action_index),
        "legal_action_count": int(len(legal_actions)),
        "positive_action_count": int(len(positive_action_indices)),
        "action_surface_kind": str(surface_metadata.get("action_surface_kind", "dense_8x8")),
        "coordinate_grid_size": int(surface_metadata.get("coordinate_grid_size", grid_size)),
        "interface_display_scale": int(surface_metadata.get("display_scale", 1)),
        "empty_action_count": int(len(tuple(surface_metadata.get("empty_action_strings", ())))),
        "object_action_count": int(len(tuple(surface_metadata.get("object_action_strings", ())))),
    }
    if metadata_extra:
        metadata.update(dict(metadata_extra))
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
        positive_action_mask=positive_action_mask,
        positive_action_indices=positive_action_indices,
    )


def _candidate_targets(
    *,
    legal_actions: tuple[ActionName, ...],
    correct_action_index: int,
    transition_targets: TransitionEventTargets,
    positive_action_indices: Sequence[int] | None = None,
) -> CandidateEventTargets:
    action_count = len(legal_actions)
    outcome = np.zeros((action_count, OUTCOME_DIM), dtype=np.float32)
    outcome[:, OUT_NO_EFFECT_NONPROGRESS] = 1.0
    value = np.zeros((action_count,), dtype=np.float32)
    delta = np.zeros((action_count, STATE_DELTA_DIM), dtype=np.float32)
    if positive_action_indices:
        indices = tuple(sorted({int(index) for index in positive_action_indices}))
    else:
        indices = (int(correct_action_index),)
    for index in indices:
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


def parametric_click_to_grid_cell(
    action: ActionName,
    *,
    grid_size: int,
    coordinate_grid_size: int,
) -> tuple[int, int] | None:
    scale = _display_scale(grid_size=grid_size, coordinate_grid_size=coordinate_grid_size)
    return click_action_to_grid_cell(
        action,
        grid_shape=(int(grid_size), int(grid_size)),
        inventory={
            "interface_display_scale": str(scale),
            "interface_display_pad_x": "0",
            "interface_display_pad_y": "0",
        },
    )


def _actions_for_surface(
    *,
    red_pos: tuple[int, int],
    blue_pos: tuple[int, int],
    distractors: tuple[tuple[int, int], ...],
    geometry_seed: int,
    grid_size: int,
    config: ObjectEventCurriculumConfig,
) -> tuple[tuple[ActionName, ...], dict[str, Any]]:
    kind = str(config.action_surface)
    if kind == "arc_scale_parametric":
        return _arc_scale_parametric_actions(
            red_pos=red_pos,
            blue_pos=blue_pos,
            distractors=distractors,
            geometry_seed=int(geometry_seed),
            grid_size=int(grid_size),
            action_surface_size=int(config.action_surface_size),
            coordinate_grid_size=int(config.coordinate_grid_size),
            empty_click_fraction=float(config.empty_click_fraction),
            positive_region_radius=int(config.positive_region_radius),
        )
    if kind != "dense_8x8":
        raise ValueError(f"unknown object-event action surface: {kind!r}")
    actions = _dense_actions(grid_size=grid_size) if config.require_full_dense_actions else _sparse_actions(red_pos, blue_pos, grid_size)
    return tuple(actions), {
        "action_surface_kind": "dense_8x8",
        "coordinate_grid_size": int(grid_size),
        "display_scale": 1,
        "empty_action_strings": tuple(),
        "object_action_strings": tuple(action for action in actions if action.startswith("click:")),
    }


def _slot_positive_actions(
    surface_metadata: dict[str, Any],
    *,
    slot: str,
    fallback: ActionName,
) -> tuple[ActionName, ...]:
    raw = surface_metadata.get(f"{slot}_positive_action_strings")
    if isinstance(raw, (list, tuple)) and raw:
        return tuple(str(action) for action in raw)
    return (str(fallback),)


def _surface_inventory(surface_metadata: dict[str, Any], *, grid_size: int) -> tuple[tuple[str, str], ...]:
    if str(surface_metadata.get("action_surface_kind", "dense_8x8")) != "arc_scale_parametric":
        return ()
    coordinate_grid_size = int(surface_metadata.get("coordinate_grid_size", grid_size))
    display_scale = int(surface_metadata.get("display_scale", _display_scale(grid_size=grid_size, coordinate_grid_size=coordinate_grid_size)))
    return tuple(
        sorted(
            {
                "interface_display_scale": str(display_scale),
                "interface_display_pad_x": "0",
                "interface_display_pad_y": "0",
                "interface_camera_height": str(int(grid_size)),
                "interface_camera_width": str(int(grid_size)),
                "interface_coordinate_grid_size": str(int(coordinate_grid_size)),
            }.items()
        )
    )


def _arc_scale_parametric_actions(
    *,
    red_pos: tuple[int, int],
    blue_pos: tuple[int, int],
    distractors: tuple[tuple[int, int], ...],
    geometry_seed: int,
    grid_size: int,
    action_surface_size: int,
    coordinate_grid_size: int,
    empty_click_fraction: float,
    positive_region_radius: int,
) -> tuple[tuple[ActionName, ...], dict[str, Any]]:
    controls: tuple[ActionName, ...] = ("0", "undo", "up", "down")
    total = max(int(action_surface_size), len(controls) + 1)
    click_budget = total - len(controls)
    coord_size = max(int(coordinate_grid_size), int(grid_size))
    scale = _display_scale(grid_size=grid_size, coordinate_grid_size=coord_size)
    rng = np.random.default_rng(int(geometry_seed) * 9_973 + 4_211)
    occupied = {(0, 0), tuple(red_pos), tuple(blue_pos), *tuple(tuple(cell) for cell in distractors)}
    actions: list[ActionName] = []
    seen: set[ActionName] = set()
    red_actions: set[ActionName] = set()
    blue_actions: set[ActionName] = set()
    object_actions: set[ActionName] = set()
    empty_actions: set[ActionName] = set()

    def add_click(x: int, y: int, *, category: str) -> None:
        if len(actions) >= click_budget:
            return
        x = max(0, min(int(x), coord_size - 1))
        y = max(0, min(int(y), coord_size - 1))
        action = f"click:{x}:{y}"
        if action in seen:
            return
        cell = parametric_click_to_grid_cell(action, grid_size=grid_size, coordinate_grid_size=coord_size)
        if cell is None:
            return
        seen.add(action)
        actions.append(action)
        if category == "red":
            red_actions.add(action)
            object_actions.add(action)
        elif category == "blue":
            blue_actions.add(action)
            object_actions.add(action)
        elif category == "object":
            object_actions.add(action)
        elif category == "empty":
            empty_actions.add(action)

    radius = max(int(positive_region_radius), 0)
    for category, cell in (("red", red_pos), ("blue", blue_pos)):
        cx, cy = _cell_center_xy(cell, scale=scale, coordinate_grid_size=coord_size)
        for dy in range(-radius, radius + 1):
            for dx in range(-radius, radius + 1):
                add_click(cx + dx, cy + dy, category=category)
    for cell in ((0, 0), *distractors):
        cx, cy = _cell_center_xy(cell, scale=scale, coordinate_grid_size=coord_size)
        add_click(cx, cy, category="object")
        add_click(cx + 1, cy, category="object")
    desired_empty = int(round(float(empty_click_fraction) * float(click_budget)))
    while len(empty_actions) < desired_empty and len(actions) < click_budget:
        x = int(rng.integers(0, coord_size))
        y = int(rng.integers(0, coord_size))
        action = f"click:{x}:{y}"
        cell = parametric_click_to_grid_cell(action, grid_size=grid_size, coordinate_grid_size=coord_size)
        if cell is None or cell in occupied or action in seen:
            continue
        add_click(x, y, category="empty")
    for y in range(coord_size):
        if len(actions) >= click_budget:
            break
        for x in range(coord_size):
            if len(actions) >= click_budget:
                break
            action = f"click:{x}:{y}"
            if action in seen:
                continue
            cell = parametric_click_to_grid_cell(action, grid_size=grid_size, coordinate_grid_size=coord_size)
            if cell is None:
                continue
            add_click(x, y, category="empty" if cell not in occupied else "object")
    if len(actions) != click_budget:
        raise ValueError(f"could not build {click_budget} unique parametric click actions")
    all_actions = actions + list(controls)
    rng.shuffle(all_actions)
    action_tuple = tuple(all_actions)
    return action_tuple, {
        "action_surface_kind": "arc_scale_parametric",
        "coordinate_grid_size": int(coord_size),
        "display_scale": int(scale),
        "empty_action_strings": tuple(sorted(empty_actions)),
        "object_action_strings": tuple(sorted(object_actions)),
        "red_positive_action_strings": tuple(sorted(red_actions)),
        "blue_positive_action_strings": tuple(sorted(blue_actions)),
    }


def _display_scale(*, grid_size: int, coordinate_grid_size: int) -> int:
    return max(int(coordinate_grid_size) // max(int(grid_size), 1), 1)


def _cell_center_xy(
    cell: tuple[int, int],
    *,
    scale: int,
    coordinate_grid_size: int,
) -> tuple[int, int]:
    row, col = int(cell[0]), int(cell[1])
    x = int(col * int(scale) + int(scale) // 2)
    y = int(row * int(scale) + int(scale) // 2)
    return (
        max(0, min(x, int(coordinate_grid_size) - 1)),
        max(0, min(y, int(coordinate_grid_size) - 1)),
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


def _public_cell_tags(state: StructuredState) -> dict[tuple[int, int], tuple[str, ...]]:
    cell_tags: dict[tuple[int, int], set[str]] = {}
    for obj in state.objects:
        public_tags = tuple(tag for tag in obj.tags if tag in PUBLIC_SYNTHETIC_CELL_TAGS)
        if not public_tags:
            continue
        for row, col in obj.cells:
            cell_tags.setdefault((int(row), int(col)), set()).update(public_tags)
    return {cell: tuple(sorted(tags)) for cell, tags in cell_tags.items()}


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


def _random_free_cell_near_origin(
    rng: np.random.Generator,
    *,
    grid_size: int,
    occupied: set[tuple[int, int]],
) -> tuple[int, int]:
    threshold = max(int(grid_size) - 1, 2)
    for _ in range(512):
        cell = _random_free_cell(rng, grid_size=grid_size, occupied=occupied)
        if cell[0] + cell[1] <= threshold:
            return cell
    return _random_free_cell(rng, grid_size=grid_size, occupied=occupied)


def _random_free_cell_far_from_origin(
    rng: np.random.Generator,
    *,
    grid_size: int,
    occupied: set[tuple[int, int]],
) -> tuple[int, int]:
    threshold = max(int(grid_size), 3)
    for _ in range(512):
        cell = _random_free_cell(rng, grid_size=grid_size, occupied=occupied)
        if cell[0] + cell[1] >= threshold:
            return cell
    return _random_free_cell(rng, grid_size=grid_size, occupied=occupied)
