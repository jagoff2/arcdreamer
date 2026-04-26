from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from arcagi.core.action_schema import build_action_schema, build_action_schema_context, click_action_to_grid_cell
from arcagi.core.types import ActionName, ObjectState, StructuredState, Transition
from arcagi.learned_online.signals import TransitionLabels, labels_from_transition

MAX_OBJECT_TOKENS = 32
GRID_BINS_Y = 8
GRID_BINS_X = 8
MAX_GRID_TOKENS = GRID_BINS_Y * GRID_BINS_X
MAX_RELATION_TOKENS = 24
MAX_STATE_TOKENS = 1 + MAX_OBJECT_TOKENS + MAX_GRID_TOKENS + MAX_RELATION_TOKENS + 2
STATE_NUMERIC_DIM = 40
ACTION_NUMERIC_DIM = 48
ACTION_FEATURE_IS_CLICK = 4
ACTION_FEATURE_CLICK_X = 5
ACTION_FEATURE_CLICK_Y = 6
ACTION_FEATURE_HAS_GRID_CELL = 8
ACTION_FEATURE_GRID_ROW = 9
ACTION_FEATURE_GRID_COL = 10
OUTCOME_DIM = 10
STATE_DELTA_DIM = 25

PAD = 0
META = 1
OBJECT = 2
GRID = 3
RELATION = 4
GLOBAL_BELIEF = 5
LEVEL_BELIEF = 6

ACTION_PAD = 0
ACTION_CLICK = 1
ACTION_MOVE = 2
ACTION_INTERACT = 3
ACTION_SELECT = 4
ACTION_RESET = 5
ACTION_UNDO = 6
ACTION_WAIT = 7
ACTION_RAW = 8
ACTION_OTHER = 9

DIR_NONE = 0
DIR_UP = 1
DIR_DOWN = 2
DIR_LEFT = 3
DIR_RIGHT = 4

OUT_VISIBLE_CHANGE = 0
OUT_OBJECTIVE_PROGRESS = 1
OUT_REWARD_PROGRESS = 2
OUT_TERMINAL_PROGRESS = 3
OUT_ACTION_AVAIL_CHANGED = 4
OUT_APPEARED_OR_DISAPPEARED = 5
OUT_MECHANIC_CHANGE = 6
OUT_NO_EFFECT_NONPROGRESS = 7
OUT_VISIBLE_ONLY_NONPROGRESS = 8
OUT_HARM = 9

ALLOWED_OBJECT_TAGS = ("agent", "interactable", "blocking", "clickable")

_ACTION_TYPE_IDS = {
    "click": ACTION_CLICK,
    "move": ACTION_MOVE,
    "interact": ACTION_INTERACT,
    "select": ACTION_SELECT,
    "reset": ACTION_RESET,
    "undo": ACTION_UNDO,
    "wait": ACTION_WAIT,
    "raw": ACTION_RAW,
}

_DIRECTION_IDS = {
    None: DIR_NONE,
    "none": DIR_NONE,
    "up": DIR_UP,
    "down": DIR_DOWN,
    "left": DIR_LEFT,
    "right": DIR_RIGHT,
}


@dataclass(frozen=True)
class StateTokenBatch:
    numeric: np.ndarray
    type_ids: np.ndarray
    mask: np.ndarray
    token_names: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionTokenBatch:
    actions: tuple[ActionName, ...]
    numeric: np.ndarray
    action_type_ids: np.ndarray
    direction_ids: np.ndarray
    mask: np.ndarray


@dataclass(frozen=True)
class TransitionEventTargets:
    outcome: np.ndarray
    delta: np.ndarray
    actual_action_index: int
    reward: float
    terminated: bool


def encode_state_tokens(
    state: StructuredState,
    *,
    previous_state: StructuredState | None = None,
) -> StateTokenBatch:
    numeric = np.zeros((MAX_STATE_TOKENS, STATE_NUMERIC_DIM), dtype=np.float32)
    type_ids = np.zeros((MAX_STATE_TOKENS,), dtype=np.int64)
    mask = np.zeros((MAX_STATE_TOKENS,), dtype=bool)
    names: list[str] = []
    cursor = 0

    _write_token(numeric, type_ids, mask, names, cursor, META, _meta_features(state, previous_state), "meta")
    cursor += 1

    for index, obj in enumerate(_sorted_objects(state)[:MAX_OBJECT_TOKENS]):
        _write_token(numeric, type_ids, mask, names, cursor, OBJECT, _object_features(obj, state), f"object:{index}")
        cursor += 1

    grid = _state_grid(state)
    previous_grid = _state_grid(previous_state) if previous_state is not None and previous_state.grid_shape == state.grid_shape else None
    for bin_y in range(GRID_BINS_Y):
        for bin_x in range(GRID_BINS_X):
            _write_token(
                numeric,
                type_ids,
                mask,
                names,
                cursor,
                GRID,
                _grid_bin_features(grid, previous_grid, bin_y, bin_x),
                f"grid:{bin_y}:{bin_x}",
            )
            cursor += 1

    for index, relation in enumerate(state.relations[:MAX_RELATION_TOKENS]):
        _write_token(
            numeric,
            type_ids,
            mask,
            names,
            cursor,
            RELATION,
            _relation_features(relation_type=relation.relation_type, value=relation.value),
            f"relation:{index}",
        )
        cursor += 1

    _write_token(numeric, type_ids, mask, names, cursor, GLOBAL_BELIEF, np.zeros((STATE_NUMERIC_DIM,), dtype=np.float32), "global_belief")
    cursor += 1
    _write_token(numeric, type_ids, mask, names, cursor, LEVEL_BELIEF, np.zeros((STATE_NUMERIC_DIM,), dtype=np.float32), "level_belief")

    return StateTokenBatch(numeric=numeric, type_ids=type_ids, mask=mask, token_names=tuple(names))


def encode_action_tokens(
    state: StructuredState,
    actions: Sequence[ActionName] | None = None,
) -> ActionTokenBatch:
    action_tuple = tuple(state.affordances if actions is None else actions)
    numeric = np.zeros((len(action_tuple), ACTION_NUMERIC_DIM), dtype=np.float32)
    action_type_ids = np.zeros((len(action_tuple),), dtype=np.int64)
    direction_ids = np.zeros((len(action_tuple),), dtype=np.int64)
    mask = np.ones((len(action_tuple),), dtype=bool)
    context = build_action_schema_context(state.affordances, dict(state.action_roles))
    inventory = state.inventory_dict()
    for index, action in enumerate(action_tuple):
        schema = build_action_schema(action, context)
        direction = schema.direction or _direction_from_role(schema.role)
        action_type_ids[index] = _ACTION_TYPE_IDS.get(schema.action_type, ACTION_OTHER)
        direction_ids[index] = _DIRECTION_IDS.get(direction, DIR_NONE)
        numeric[index] = _action_features(state, action, inventory=inventory)
    return ActionTokenBatch(
        actions=action_tuple,
        numeric=numeric,
        action_type_ids=action_type_ids,
        direction_ids=direction_ids,
        mask=mask,
    )


def labels_to_outcome(labels: TransitionLabels) -> np.ndarray:
    return np.asarray(
        [
            labels.visible_change,
            labels.objective_progress,
            labels.reward_progress,
            labels.terminal_progress,
            labels.action_availability_changed,
            labels.appeared_or_disappeared,
            labels.mechanic_change,
            labels.no_effect_nonprogress,
            labels.visible_only_nonprogress,
            labels.harm,
        ],
        dtype=np.float32,
    )


def build_transition_targets(
    transition: Transition,
    *,
    actions: Sequence[ActionName] | None = None,
) -> TransitionEventTargets:
    action_tuple = tuple(transition.state.affordances if actions is None else actions)
    try:
        action_index = action_tuple.index(transition.action)
    except ValueError:
        raise ValueError(f"transition action {transition.action!r} is not in the supplied legal action surface")
    before = transition.state.transition_vector(max_objects=4)
    after = transition.next_state.transition_vector(max_objects=4)
    return TransitionEventTargets(
        outcome=labels_to_outcome(labels_from_transition(transition)),
        delta=np.asarray(after - before, dtype=np.float32),
        actual_action_index=int(action_index),
        reward=float(transition.reward),
        terminated=bool(transition.terminated),
    )


def stack_state_tokens(items: Sequence[StateTokenBatch]) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    return (
        np.stack([item.numeric for item in items]).astype(np.float32, copy=False),
        np.stack([item.type_ids for item in items]).astype(np.int64, copy=False),
        np.stack([item.mask for item in items]).astype(bool, copy=False),
    )


def stack_action_tokens(items: Sequence[ActionTokenBatch]) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    max_actions = max((len(item.actions) for item in items), default=0)
    numeric = np.zeros((len(items), max_actions, ACTION_NUMERIC_DIM), dtype=np.float32)
    action_type_ids = np.zeros((len(items), max_actions), dtype=np.int64)
    direction_ids = np.zeros((len(items), max_actions), dtype=np.int64)
    mask = np.zeros((len(items), max_actions), dtype=bool)
    for batch_index, item in enumerate(items):
        count = len(item.actions)
        numeric[batch_index, :count] = item.numeric
        action_type_ids[batch_index, :count] = item.action_type_ids
        direction_ids[batch_index, :count] = item.direction_ids
        mask[batch_index, :count] = item.mask
    return numeric, action_type_ids, direction_ids, mask


def _write_token(
    numeric: np.ndarray,
    type_ids: np.ndarray,
    mask: np.ndarray,
    names: list[str],
    index: int,
    token_type: int,
    features: np.ndarray,
    name: str,
) -> None:
    numeric[index, : min(features.shape[0], STATE_NUMERIC_DIM)] = features[:STATE_NUMERIC_DIM]
    type_ids[index] = int(token_type)
    mask[index] = True
    names.append(name)


def _meta_features(state: StructuredState, previous_state: StructuredState | None) -> np.ndarray:
    grid = _state_grid(state)
    height, width = state.grid_shape
    total_cells = max(float(height * width), 1.0)
    palette = _palette_histogram(grid, buckets=12)
    object_area = sum(float(obj.area) for obj in state.objects) / total_cells
    previous_object_count = len(previous_state.objects) if previous_state is not None else len(state.objects)
    features = np.zeros((STATE_NUMERIC_DIM,), dtype=np.float32)
    values = [
        float(height) / 64.0,
        float(width) / 64.0,
        float(len(state.objects)) / max(float(MAX_OBJECT_TOKENS), 1.0),
        object_area,
        float(len(state.affordances)) / 512.0,
        float(len(state.relations)) / max(float(MAX_RELATION_TOKENS), 1.0),
        float(len(state.objects) - previous_object_count) / max(float(MAX_OBJECT_TOKENS), 1.0),
        float(np.count_nonzero(grid)) / total_cells,
        *palette.tolist(),
    ]
    features[: len(values)] = np.asarray(values, dtype=np.float32)
    return features


def _object_features(obj: ObjectState, state: StructuredState) -> np.ndarray:
    height, width = state.grid_shape
    total_cells = max(float(height * width), 1.0)
    y0, x0, y1, x1 = obj.bbox
    box_h = max(int(y1) - int(y0) + 1, 1)
    box_w = max(int(x1) - int(x0) + 1, 1)
    density = float(obj.area) / max(float(box_h * box_w), 1.0)
    tags = set(obj.tags)
    features = np.zeros((STATE_NUMERIC_DIM,), dtype=np.float32)
    values = [
        float(int(obj.color) % 12) / 11.0,
        float(obj.area) / total_cells,
        float(obj.centroid[0]) / max(float(height), 1.0),
        float(obj.centroid[1]) / max(float(width), 1.0),
        float(y0) / max(float(height), 1.0),
        float(x0) / max(float(width), 1.0),
        float(y1) / max(float(height), 1.0),
        float(x1) / max(float(width), 1.0),
        density,
        *(1.0 if tag in tags else 0.0 for tag in ALLOWED_OBJECT_TAGS),
    ]
    features[: len(values)] = np.asarray(values, dtype=np.float32)
    return features


def _grid_bin_features(grid: np.ndarray, previous_grid: np.ndarray | None, bin_y: int, bin_x: int) -> np.ndarray:
    height, width = grid.shape
    y0 = int(np.floor(bin_y * height / GRID_BINS_Y))
    y1 = int(np.floor((bin_y + 1) * height / GRID_BINS_Y))
    x0 = int(np.floor(bin_x * width / GRID_BINS_X))
    x1 = int(np.floor((bin_x + 1) * width / GRID_BINS_X))
    if y1 <= y0:
        y1 = min(y0 + 1, height)
    if x1 <= x0:
        x1 = min(x0 + 1, width)
    patch = grid[y0:y1, x0:x1]
    if patch.size == 0:
        patch = np.zeros((1, 1), dtype=np.int64)
    colors, counts = np.unique(patch, return_counts=True)
    dominant = int(colors[int(np.argmax(counts))]) if colors.size else 0
    features = np.zeros((STATE_NUMERIC_DIM,), dtype=np.float32)
    palette = _palette_histogram(patch, buckets=12)
    changed = 0.0
    if previous_grid is not None and previous_grid.shape == grid.shape:
        changed = float(np.mean(previous_grid[y0:y1, x0:x1] != patch)) if patch.size else 0.0
    values = [
        float(bin_y) / max(float(GRID_BINS_Y - 1), 1.0),
        float(bin_x) / max(float(GRID_BINS_X - 1), 1.0),
        float(dominant % 12) / 11.0,
        float(np.count_nonzero(patch)) / max(float(patch.size), 1.0),
        changed,
        *palette.tolist(),
    ]
    features[: len(values)] = np.asarray(values, dtype=np.float32)
    return features


def _relation_features(*, relation_type: str, value: float) -> np.ndarray:
    features = np.zeros((STATE_NUMERIC_DIM,), dtype=np.float32)
    known_types = ("adjacent", "near", "overlap", "contains", "same_color", "touching")
    normalized = relation_type.strip().lower()
    features[0] = float(value)
    for index, known in enumerate(known_types, start=1):
        features[index] = 1.0 if normalized == known else 0.0
    return features


def _action_features(state: StructuredState, action: ActionName, *, inventory: Mapping[str, str]) -> np.ndarray:
    context = build_action_schema_context(state.affordances, dict(state.action_roles))
    schema = build_action_schema(action, context)
    features = np.zeros((ACTION_NUMERIC_DIM,), dtype=np.float32)
    action_type_id = _ACTION_TYPE_IDS.get(schema.action_type, ACTION_OTHER)
    direction = schema.direction or _direction_from_role(schema.role)
    direction_id = _DIRECTION_IDS.get(direction, DIR_NONE)
    features[0] = float(action_type_id) / float(ACTION_OTHER)
    features[1] = float(direction_id) / float(DIR_RIGHT)
    if direction == "up":
        features[2] = -1.0
    elif direction == "down":
        features[2] = 1.0
    if direction == "left":
        features[3] = -1.0
    elif direction == "right":
        features[3] = 1.0
    if schema.normalized_click is not None:
        features[4] = 1.0
        features[5] = float(schema.normalized_click[0])
        features[6] = float(schema.normalized_click[1])
    if schema.raw_action is not None:
        features[7] = float(schema.raw_action) / max(context.max_raw_action, 1.0)
    grid_cell = click_action_to_grid_cell(action, grid_shape=state.grid_shape, inventory=inventory)
    if grid_cell is not None:
        row, col = grid_cell
        height, width = state.grid_shape
        features[8] = 1.0
        features[9] = float(row) / max(float(height - 1), 1.0)
        features[10] = float(col) / max(float(width - 1), 1.0)
        containing = _containing_object(state, row, col)
        nearest = containing or _nearest_object(state, row, col)
        if containing is not None:
            features[11] = 1.0
        if nearest is not None:
            _write_object_relative_action_features(features, nearest, state, row=row, col=col, offset=12)
    return features


def _write_object_relative_action_features(
    features: np.ndarray,
    obj: ObjectState,
    state: StructuredState,
    *,
    row: int,
    col: int,
    offset: int,
) -> None:
    height, width = state.grid_shape
    total_cells = max(float(height * width), 1.0)
    dy = (float(row) - float(obj.centroid[0])) / max(float(height), 1.0)
    dx = (float(col) - float(obj.centroid[1])) / max(float(width), 1.0)
    distance = float(np.sqrt((float(row) - obj.centroid[0]) ** 2 + (float(col) - obj.centroid[1]) ** 2))
    y0, x0, y1, x1 = obj.bbox
    values = [
        1.0,
        float(int(obj.color) % 12) / 11.0,
        float(obj.area) / total_cells,
        float(obj.centroid[0]) / max(float(height), 1.0),
        float(obj.centroid[1]) / max(float(width), 1.0),
        dy,
        dx,
        distance / max(float(max(height, width)), 1.0),
        float(y0) / max(float(height), 1.0),
        float(x0) / max(float(width), 1.0),
        float(y1) / max(float(height), 1.0),
        float(x1) / max(float(width), 1.0),
        *(1.0 if tag in set(obj.tags) else 0.0 for tag in ALLOWED_OBJECT_TAGS),
    ]
    features[offset : offset + len(values)] = np.asarray(values, dtype=np.float32)
    # Duplicate the clicked-object color in a stable low index for candidate-level
    # training diagnostics and compact action grounding.
    if offset != 13 and offset < features.shape[0]:
        features[13] = values[1]


def _sorted_objects(state: StructuredState) -> tuple[ObjectState, ...]:
    return tuple(
        sorted(
            state.objects,
            key=lambda obj: (
                0 if "agent" in obj.tags else 1,
                int(obj.color) % 12,
                -int(obj.area),
                float(obj.centroid[0]),
                float(obj.centroid[1]),
                tuple(int(v) for v in obj.bbox),
            ),
        )
    )


def _state_grid(state: StructuredState | None) -> np.ndarray:
    if state is None:
        return np.zeros((1, 1), dtype=np.int64)
    try:
        grid = state.as_grid()
        if grid.shape == state.grid_shape:
            return np.asarray(grid, dtype=np.int64)
    except Exception:
        pass
    grid = np.zeros(state.grid_shape, dtype=np.int64)
    for obj in state.objects:
        for row, col in obj.cells:
            if 0 <= row < state.grid_shape[0] and 0 <= col < state.grid_shape[1]:
                grid[row, col] = int(obj.color)
    return grid


def _palette_histogram(values: np.ndarray, *, buckets: int) -> np.ndarray:
    hist = np.zeros((buckets,), dtype=np.float32)
    flat = np.asarray(values).reshape(-1)
    if flat.size == 0:
        return hist
    for value in flat:
        hist[int(value) % buckets] += 1.0
    return hist / max(float(flat.size), 1.0)


def _direction_from_role(role: str) -> str | None:
    normalized = role.lower()
    for direction in ("up", "down", "left", "right"):
        if direction in normalized:
            return direction
    return None


def _containing_object(state: StructuredState, row: int, col: int) -> ObjectState | None:
    for obj in _sorted_objects(state):
        if (row, col) in obj.cells:
            return obj
        y0, x0, y1, x1 = obj.bbox
        if y0 <= row <= y1 and x0 <= col <= x1:
            return obj
    return None


def _nearest_object(state: StructuredState, row: int, col: int) -> ObjectState | None:
    best: tuple[float, ObjectState] | None = None
    for obj in _sorted_objects(state):
        distance = (float(row) - obj.centroid[0]) ** 2 + (float(col) - obj.centroid[1]) ** 2
        if best is None or distance < best[0]:
            best = (distance, obj)
    return None if best is None else best[1]
