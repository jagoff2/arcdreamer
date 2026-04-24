from __future__ import annotations

import numpy as np

from arcagi.envs.arc_adapter import (
    _click_actions_for_grid,
    _display_to_grid_cell,
    _downsample_display_grid,
    _expand_actions,
    _with_terminal_reset,
)


def test_downsample_display_grid_recovers_camera_grid() -> None:
    camera_grid = np.array(
        [
            [0, 1, 2],
            [3, 4, 5],
        ],
        dtype=np.int64,
    )
    display_grid = np.repeat(np.repeat(camera_grid, 3, axis=0), 3, axis=1)
    camera_meta = {
        "width": 3,
        "height": 2,
        "x": 0,
        "y": 0,
        "scale": 3,
        "pad_x": 0,
        "pad_y": 0,
    }

    recovered = _downsample_display_grid(display_grid, camera_meta)

    assert recovered is not None
    np.testing.assert_array_equal(recovered, camera_grid)


def test_click_actions_for_grid_emit_display_space_centers() -> None:
    camera_grid = np.array(
        [
            [0, 0, 0, 0],
            [0, 2, 0, 3],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    camera_meta = {
        "width": 4,
        "height": 3,
        "x": 0,
        "y": 0,
        "scale": 3,
        "pad_x": 0,
        "pad_y": 0,
    }

    actions = _click_actions_for_grid(camera_grid, camera_meta)

    assert "click:4:4" in actions
    assert "click:10:4" in actions


def test_click_actions_for_grid_use_camera_local_display_coordinates_when_origin_moves() -> None:
    camera_grid = np.array(
        [
            [0, 0, 0],
            [0, 2, 0],
            [0, 0, 0],
        ],
        dtype=np.int64,
    )
    camera_meta = {
        "width": 3,
        "height": 3,
        "x": 12,
        "y": 9,
        "scale": 4,
        "pad_x": 2,
        "pad_y": 6,
    }

    actions = _click_actions_for_grid(camera_grid, camera_meta)

    assert "click:8:12" in actions
    assert _display_to_grid_cell(8, 12, camera_meta) == (1, 1)


def test_click_actions_for_grid_tolerates_empty_grid() -> None:
    camera_meta = {
        "width": 4,
        "height": 3,
        "x": 0,
        "y": 0,
        "scale": 3,
        "pad_x": 0,
        "pad_y": 0,
    }

    assert _click_actions_for_grid(np.zeros((0, 1), dtype=np.int64), camera_meta) == []


def test_click_actions_for_grid_keeps_large_salient_components_under_cap() -> None:
    camera_grid = np.zeros((12, 12), dtype=np.int64)
    camera_grid[0:4, 0:4] = 2
    color = 3
    for row in range(4, 12, 2):
        for col in range(4, 12, 2):
            camera_grid[row, col] = color
            color += 1
    camera_meta = {
        "width": 12,
        "height": 12,
        "x": 0,
        "y": 0,
        "scale": 2,
        "pad_x": 0,
        "pad_y": 0,
    }

    actions = _click_actions_for_grid(camera_grid, camera_meta, max_candidates=4)

    assert any(action in actions for action in ("click:3:3", "click:5:3", "click:3:5", "click:5:5"))


def test_click_actions_for_grid_has_no_default_candidate_cap() -> None:
    camera_grid = np.zeros((22, 22), dtype=np.int64)
    component_count = 0
    for row in range(0, 22, 2):
        for col in range(0, 22, 2):
            camera_grid[row, col] = 2 + (component_count % 8)
            component_count += 1
    camera_meta = {
        "width": 22,
        "height": 22,
        "x": 0,
        "y": 0,
        "scale": 1,
        "pad_x": 0,
        "pad_y": 0,
    }

    actions = _click_actions_for_grid(camera_grid, camera_meta)

    assert len(actions) == component_count


def test_click_actions_for_grid_dense_mode_emits_every_cell() -> None:
    camera_grid = np.zeros((5, 4), dtype=np.int64)
    camera_grid[1, 1] = 2
    camera_meta = {
        "width": 4,
        "height": 5,
        "x": 0,
        "y": 0,
        "scale": 1,
        "pad_x": 0,
        "pad_y": 0,
    }

    actions = _click_actions_for_grid(camera_grid, camera_meta, dense=True)

    assert len(actions) == camera_grid.size
    assert "click:0:0" in actions
    assert "click:3:4" in actions


def test_expand_actions_uses_dense_clicks_by_default(monkeypatch) -> None:
    monkeypatch.delenv("ARCAGI_DENSE_CLICKS", raising=False)
    camera_grid = np.zeros((3, 4), dtype=np.int64)
    camera_grid[1, 1] = 2
    camera_meta = {
        "width": 4,
        "height": 3,
        "x": 0,
        "y": 0,
        "scale": 1,
        "pad_x": 0,
        "pad_y": 0,
    }

    actions = _expand_actions(("1", "6"), camera_grid, camera_meta)

    assert len([action for action in actions if action.startswith("click:")]) == camera_grid.size


def test_expand_actions_rejects_legacy_dense_click_disable(monkeypatch) -> None:
    import pytest

    monkeypatch.setenv("ARCAGI_DENSE_CLICKS", "0")
    camera_grid = np.zeros((3, 4), dtype=np.int64)
    camera_grid[1, 1] = 2
    camera_meta = {
        "width": 4,
        "height": 3,
        "x": 0,
        "y": 0,
        "scale": 1,
        "pad_x": 0,
        "pad_y": 0,
    }

    with pytest.raises(RuntimeError, match="hides legal ARC click actions"):
        _expand_actions(("1", "6"), camera_grid, camera_meta)


def test_expand_actions_can_opt_into_component_click_smoke_mode(monkeypatch) -> None:
    monkeypatch.delenv("ARCAGI_DENSE_CLICKS", raising=False)
    monkeypatch.setenv("ARCAGI_SPARSE_CLICKS_BASELINE", "1")
    camera_grid = np.zeros((3, 4), dtype=np.int64)
    camera_grid[1, 1] = 2
    camera_meta = {
        "width": 4,
        "height": 3,
        "x": 0,
        "y": 0,
        "scale": 1,
        "pad_x": 0,
        "pad_y": 0,
    }

    actions = _expand_actions(("1", "6"), camera_grid, camera_meta)

    assert tuple(action for action in actions if action.startswith("click:")) == ("click:1:1",)


def test_expand_actions_requires_camera_metadata_for_legal_click_surface(monkeypatch) -> None:
    import pytest

    class _GameAction:
        RESET = object()

    monkeypatch.setattr("arcagi.envs.arc_adapter.GameAction", _GameAction)

    with pytest.raises(RuntimeError, match="camera metadata is unavailable"):
        _expand_actions(
            ("1", "2", "6"),
            np.zeros((2, 2), dtype=np.int64),
            None,
        )


def test_expand_actions_does_not_inject_unadvertised_reset_affordance(monkeypatch) -> None:
    class _GameAction:
        RESET = object()

    monkeypatch.setattr("arcagi.envs.arc_adapter.GameAction", _GameAction)

    actions = _expand_actions(
        ("1", "2"),
        np.zeros((2, 2), dtype=np.int64),
        None,
    )

    assert "0" not in actions


def test_terminal_reset_is_exposed_only_after_game_over(monkeypatch) -> None:
    class _GameAction:
        RESET = object()

    class _Observation:
        def __init__(self, state: str) -> None:
            self.state = state

    monkeypatch.setattr("arcagi.envs.arc_adapter.GameAction", _GameAction)

    assert _with_terminal_reset(("1", "2"), _Observation("GameState.NOT_FINISHED")) == ("1", "2")
    assert _with_terminal_reset(("1", "2"), _Observation("GameState.GAME_OVER")) == ("0", "1", "2")
