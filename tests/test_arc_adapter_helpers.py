from __future__ import annotations

import numpy as np

from arcagi.envs.arc_adapter import _click_actions_for_grid, _downsample_display_grid, _expand_actions


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


def test_expand_actions_injects_reset_affordance_when_supported(monkeypatch) -> None:
    class _GameAction:
        RESET = object()

    monkeypatch.setattr("arcagi.envs.arc_adapter.GameAction", _GameAction)

    actions = _expand_actions(
        ("1", "2", "6"),
        np.zeros((2, 2), dtype=np.int64),
        None,
    )

    assert actions[0] == "0"
