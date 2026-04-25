from __future__ import annotations

import inspect

import numpy as np

from arcagi.core.types import ObjectState, StructuredState
from arcagi.learned_online import event_tokens
from arcagi.learned_online.event_tokens import (
    ACTION_CLICK,
    ACTION_NUMERIC_DIM,
    ACTION_RESET,
    GRID,
    GRID_BINS_X,
    GRID_BINS_Y,
    MAX_STATE_TOKENS,
    OBJECT,
    STATE_NUMERIC_DIM,
    encode_action_tokens,
    encode_state_tokens,
)


def test_state_tokens_have_fixed_shape_and_object_grid_tokens_coexist() -> None:
    state = _state_with_objects()

    tokens = encode_state_tokens(state)

    assert tokens.numeric.shape == (MAX_STATE_TOKENS, STATE_NUMERIC_DIM)
    assert tokens.type_ids.shape == (MAX_STATE_TOKENS,)
    assert tokens.mask.shape == (MAX_STATE_TOKENS,)
    assert tokens.mask.dtype == bool
    assert int(np.sum(tokens.type_ids == OBJECT)) >= 3
    assert int(np.sum(tokens.type_ids == GRID)) == GRID_BINS_Y * GRID_BINS_X
    assert any(name.startswith("object:") for name in tokens.token_names)
    assert any(name.startswith("grid:") for name in tokens.token_names)


def test_action_tokens_preserve_full_dense_surface_and_reset_as_normal_action() -> None:
    clicks = tuple(f"click:{x}:{y}" for y in range(16) for x in range(28))
    actions = clicks + ("0", "undo", "up")
    state = _state_with_objects(actions=actions)

    tokens = encode_action_tokens(state, actions)

    assert tokens.actions == actions
    assert tokens.numeric.shape == (len(actions), ACTION_NUMERIC_DIM)
    assert tokens.mask.shape == (len(actions),)
    assert int(tokens.mask.sum()) == len(actions)
    assert int(np.sum(tokens.action_type_ids == ACTION_CLICK)) == len(clicks)
    reset_index = actions.index("0")
    assert int(tokens.action_type_ids[reset_index]) == ACTION_RESET
    assert len({tuple(tokens.numeric[index, 5:11]) for index in range(20)}) > 10


def test_event_token_source_does_not_use_identity_hash_or_action_string_projection() -> None:
    source = inspect.getsource(event_tokens)
    forbidden_fragments = (
        ".task_id",
        ".episode_id",
        ".object_id",
        "fingerprint(",
        "hash(",
        "state_hash",
        "teacher_action",
        "trace_cursor",
        "action_sequence",
        "full action",
        "action_string",
    )
    for fragment in forbidden_fragments:
        assert fragment not in source


def test_object_token_order_uses_visible_geometry_not_object_id() -> None:
    state_a = _state_with_objects(object_ids=("z", "y", "x"))
    state_b = _state_with_objects(object_ids=("a", "b", "c"))

    tokens_a = encode_state_tokens(state_a)
    tokens_b = encode_state_tokens(state_b)
    object_rows_a = tokens_a.numeric[tokens_a.type_ids == OBJECT]
    object_rows_b = tokens_b.numeric[tokens_b.type_ids == OBJECT]

    assert np.allclose(object_rows_a, object_rows_b)


def _state_with_objects(
    *,
    actions: tuple[str, ...] | None = None,
    object_ids: tuple[str, str, str] = ("cue", "red", "blue"),
) -> StructuredState:
    grid = np.zeros((8, 8), dtype=np.int64)
    objects = (
        _object(object_ids[0], 3, ((0, 0),), tags=("clickable",)),
        _object(object_ids[1], 2, ((3, 2), (3, 3))),
        _object(object_ids[2], 5, ((5, 6),)),
    )
    for obj in objects:
        for row, col in obj.cells:
            grid[row, col] = obj.color
    action_tuple = actions or ("click:2:3", "click:6:5", "click:0:0", "0", "undo", "up")
    roles = {action: "click" for action in action_tuple if action.startswith("click:")}
    roles.update({"0": "reset_level", "undo": "undo", "up": "move_up"})
    return StructuredState(
        task_id="identity_must_not_enter_tokens",
        episode_id="episode_must_not_enter_tokens",
        step_index=7,
        grid_shape=grid.shape,
        grid_signature=tuple(int(value) for value in grid.reshape(-1)),
        objects=objects,
        relations=(),
        affordances=action_tuple,
        action_roles=tuple(sorted(roles.items())),
    )


def _object(
    object_id: str,
    color: int,
    cells: tuple[tuple[int, int], ...],
    *,
    tags: tuple[str, ...] = (),
) -> ObjectState:
    rows = [row for row, _col in cells]
    cols = [col for _row, col in cells]
    return ObjectState(
        object_id=object_id,
        color=color,
        cells=cells,
        bbox=(min(rows), min(cols), max(rows), max(cols)),
        centroid=(float(sum(rows)) / len(rows), float(sum(cols)) / len(cols)),
        area=len(cells),
        tags=tags,
    )
