from __future__ import annotations

from collections.abc import Iterable
import os
from typing import Any

import numpy as np

from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.types import ActionName, GridObservation, StepResult
from arcagi.envs.base import BaseEnvironment

try:
    from arc_agi import Arcade
    from arc_agi import OperationMode
    from arcengine import GameAction, GameState
except Exception:  # pragma: no cover - optional dependency
    Arcade = None
    OperationMode = None
    GameAction = None
    GameState = None


ARC_ACTION_ROLES: dict[str, str] = {
    "0": "reset_level",
    "1": "move_up",
    "2": "move_down",
    "3": "move_left",
    "4": "move_right",
    "5": "select_cycle",
    "6": "click",
    "7": "undo",
}


def arc_toolkit_available() -> bool:
    return Arcade is not None


def list_arc_games(operation_mode: Any | None = None) -> list[str]:
    if Arcade is None:
        return []
    arcade = Arcade(operation_mode=operation_mode if operation_mode is not None else _default_operation_mode())
    environments = getattr(arcade, "get_environments", lambda: [])()
    if isinstance(environments, dict):
        return sorted(str(key) for key in environments.keys())
    game_ids = []
    for item in environments:
        game_ids.append(str(getattr(item, "game_id", item)))
    return sorted(game_ids)


class ArcToolkitEnv(BaseEnvironment):
    def __init__(self, game_id: str, operation_mode: Any | None = None, arcade: Any | None = None) -> None:
        if Arcade is None:
            raise RuntimeError("ARC toolkit is not installed. Install with `pip install -e .[arc]`.")
        self._operation_mode = operation_mode if operation_mode is not None else _default_operation_mode()
        self.arcade = arcade if arcade is not None else Arcade(operation_mode=self._operation_mode)
        self._owns_arcade = arcade is None
        self.env = self.arcade.make(game_id)
        self._task_id = f"arc/{game_id}"
        self._family_id = self._task_id
        self._episode_index = 0
        self._step_index = 0
        self._last_actions: tuple[ActionName, ...] = self._discover_actions()
        self._last_base_actions: tuple[ActionName, ...] = self._last_actions
        self._camera_meta: dict[str, int] | None = _camera_metadata(self.env)
        self._last_levels_completed = 0
        self._last_grid = np.zeros((1, 1), dtype=np.int64)

    @property
    def task_id(self) -> str:
        return self._task_id

    @property
    def family_id(self) -> str:
        return self._family_id

    def legal_actions(self) -> tuple[ActionName, ...]:
        return self._last_actions

    def close(self) -> None:
        close_method = getattr(self.env, "close", None)
        if callable(close_method):
            try:
                close_method()
            except Exception:
                pass
        if not self._owns_arcade:
            return
        close_scorecard = getattr(self.arcade, "close_scorecard", None)
        if callable(close_scorecard):
            try:
                close_scorecard()
            except Exception:
                pass

    def reset(self, seed: int | None = None) -> GridObservation:
        self._episode_index += 1
        self._step_index = 0
        self._last_levels_completed = 0
        try:
            result = self.env.reset(seed=seed) if seed is not None else self.env.reset()
        except TypeError:
            result = self.env.reset()
        observation = result[0] if isinstance(result, tuple) else result
        display = _extract_display_grid(observation)
        self._camera_meta = _camera_metadata(self.env) or _infer_camera_metadata_from_display(display)
        grid = _coerce_grid(display, self._camera_meta)
        if grid.size <= 0:
            grid = np.zeros((1, 1), dtype=np.int64)
        self._last_base_actions = self._discover_actions(observation)
        self._last_actions = _with_terminal_reset(
            _expand_actions(
                self._last_base_actions,
                grid,
                self._camera_meta,
            ),
            observation,
        )
        self._last_levels_completed = int(getattr(observation, "levels_completed", 0) or 0)
        self._last_grid = grid
        return GridObservation(
            task_id=self.task_id,
            episode_id=f"{self.task_id}/episode_{self._episode_index}",
            step_index=self._step_index,
            grid=grid,
            available_actions=self._last_actions,
            extras=_build_arc_extras(
                observation=observation,
                grid=grid,
                base_actions=self._last_base_actions,
                affordances=self._last_actions,
                camera_meta=self._camera_meta,
                step_index=self._step_index,
            ),
        )

    def step(self, action: ActionName) -> StepResult:
        env_action, action_data = self._convert_action(action)
        result = self.env.step(env_action, data=action_data)
        self._step_index += 1
        if isinstance(result, tuple) and len(result) == 5:
            observation, reward, terminated, truncated, info = result
        elif isinstance(result, tuple) and len(result) == 4:
            observation, reward, terminated, info = result
            truncated = False
        else:
            observation = result
            info = {}
            levels_completed = int(getattr(observation, "levels_completed", 0) or 0)
            reward = float(levels_completed - self._last_levels_completed)
            game_state = getattr(observation, "state", None)
            terminated = bool(
                game_state is not None
                and str(getattr(game_state, "value", game_state)) in {"WIN", "GAME_OVER"}
            )
            truncated = False
            if terminated and reward <= 0.0 and str(getattr(game_state, "value", game_state)) == "WIN":
                reward = 1.0
            self._last_levels_completed = levels_completed
        if observation is None:
            return StepResult(
                observation=GridObservation(
                    task_id=self.task_id,
                    episode_id=f"{self.task_id}/episode_{self._episode_index}",
                    step_index=self._step_index,
                    grid=self._last_grid.copy(),
                    available_actions=self._last_actions,
                    extras={
                        "adapter": "arc_toolkit",
                        "game_state": "transport_error",
                        "transport_error": True,
                        **(dict(info or {})),
                    },
                ),
                reward=float(reward),
                terminated=True,
                truncated=True,
                info={"transport_error": True, **dict(info or {})},
            )
        display = _extract_display_grid(observation)
        self._camera_meta = _camera_metadata(self.env) or _infer_camera_metadata_from_display(display)
        grid = _coerce_grid(display, self._camera_meta)
        if grid.size <= 0:
            grid = self._last_grid.copy() if self._last_grid.size > 0 else np.zeros((1, 1), dtype=np.int64)
        self._last_base_actions = self._discover_actions(observation)
        self._last_actions = _with_terminal_reset(
            _expand_actions(
                self._last_base_actions,
                grid,
                self._camera_meta,
            ),
            observation,
        )
        self._last_grid = grid
        step_index = self._step_index
        return StepResult(
            observation=GridObservation(
                task_id=self.task_id,
                episode_id=f"{self.task_id}/episode_{self._episode_index}",
                step_index=step_index,
                grid=grid,
                available_actions=self._last_actions,
                extras=_build_arc_extras(
                    observation=observation,
                    grid=grid,
                    base_actions=self._last_base_actions,
                    affordances=self._last_actions,
                    camera_meta=self._camera_meta,
                    step_index=step_index,
                    info=dict(info or {}),
                ),
            ),
            reward=float(reward),
            terminated=bool(terminated),
            truncated=bool(truncated),
            info=dict(info or {}),
        )

    def _discover_actions(self, observation: Any | None = None) -> tuple[ActionName, ...]:
        if observation is not None and hasattr(observation, "available_actions"):
            actions = tuple(str(item) for item in getattr(observation, "available_actions"))
            if actions:
                return actions
        for attr in ("available_actions", "actions"):
            value = getattr(self.env, attr, None)
            if value is None:
                continue
            if callable(value):
                value = value()
            if isinstance(value, Iterable):
                actions = tuple(str(item) for item in value)
                if actions:
                    return actions
        action_space = getattr(self.env, "action_space", None)
        if action_space is not None and hasattr(action_space, "n"):
            return tuple(str(idx) for idx in range(int(action_space.n)))
        return tuple(str(idx) for idx in range(9))

    def _convert_action(self, action: ActionName) -> tuple[Any, dict[str, int] | None]:
        if isinstance(action, str) and action.startswith("click:"):
            parts = action.split(":")
            if len(parts) == 3 and all(part.lstrip("-").isdigit() for part in parts[1:]):
                click_x = int(parts[1])
                click_y = int(parts[2])
                if GameAction is not None and hasattr(GameAction, "ACTION6"):
                    return getattr(GameAction, "ACTION6"), {"x": click_x, "y": click_y}
                return 6, {"x": click_x, "y": click_y}
        if GameAction is not None:
            if isinstance(action, str) and action.isdigit():
                action_id = int(action)
                if action_id == 0 and hasattr(GameAction, "RESET"):
                    return getattr(GameAction, "RESET"), None
                member_name = f"ACTION{action_id}"
                if hasattr(GameAction, member_name):
                    return getattr(GameAction, member_name), None
            if isinstance(action, str) and hasattr(GameAction, action):
                return getattr(GameAction, action), None
        if action in self._last_actions:
            if all(item.isdigit() for item in self._last_actions):
                return int(action), None
            return action, None
        if isinstance(action, str) and action.isdigit():
            return int(action), None
        return action, None


def _extract_display_grid(observation: Any) -> np.ndarray:
    return _extract_array(observation)


def _coerce_grid(array: np.ndarray, camera_meta: dict[str, int] | None = None) -> np.ndarray:
    if array.ndim == 2:
        if camera_meta is not None:
            downsampled = _downsample_display_grid(array.astype(np.int64), camera_meta)
            if downsampled is not None:
                return downsampled
        return array.astype(np.int64)
    if array.ndim == 3 and array.shape[-1] in (1, 3, 4):
        indexed = _rgb_to_index_grid(array)
        if camera_meta is not None:
            downsampled = _downsample_display_grid(indexed, camera_meta)
            if downsampled is not None:
                return downsampled
        return indexed
    return np.asarray(array).reshape(-1, 1).astype(np.int64)


def _extract_array(value: Any) -> np.ndarray:
    if isinstance(value, np.ndarray):
        return value
    if hasattr(value, "frame"):
        frame = getattr(value, "frame")
        if isinstance(frame, list) and frame:
            return _extract_array(frame[0])
    if isinstance(value, (list, tuple)):
        try:
            return np.asarray(value)
        except Exception:
            pass
    if isinstance(value, dict):
        for key in ("grid", "observation", "pixels", "frame", "state"):
            if key in value:
                return _extract_array(value[key])
    for attr in ("grid", "observation", "pixels", "frame", "state"):
        if hasattr(value, attr):
            return _extract_array(getattr(value, attr))
    return np.asarray(value)


def _rgb_to_index_grid(array: np.ndarray) -> np.ndarray:
    flat = array.reshape(-1, array.shape[-1])
    unique, inverse = np.unique(flat, axis=0, return_inverse=True)
    return inverse.reshape(array.shape[0], array.shape[1]).astype(np.int64)


def _camera_metadata(env: Any) -> dict[str, int] | None:
    game = getattr(env, "_game", None)
    camera = getattr(game, "camera", None)
    if camera is None:
        return None
    width = int(getattr(camera, "width", 0) or 0)
    height = int(getattr(camera, "height", 0) or 0)
    if width <= 0 or height <= 0:
        return None
    scale_x = max(int(64 / width), 1)
    scale_y = max(int(64 / height), 1)
    scale = min(scale_x, scale_y)
    return {
        "width": width,
        "height": height,
        "x": int(getattr(camera, "x", 0) or 0),
        "y": int(getattr(camera, "y", 0) or 0),
        "scale": scale,
        "pad_x": int((64 - (width * scale)) / 2),
        "pad_y": int((64 - (height * scale)) / 2),
    }


def _infer_camera_metadata_from_display(array: np.ndarray) -> dict[str, int] | None:
    if array.ndim != 2 or array.shape[0] != 64 or array.shape[1] != 64:
        return None
    best: tuple[float, int, int, int, int, int] | None = None
    for scale in range(2, 9):
        for pad_y in range(scale):
            height = (64 - pad_y) // scale
            if height < 8:
                continue
            for pad_x in range(scale):
                width = (64 - pad_x) // scale
                if width < 8:
                    continue
                purity = _block_purity_score(array, scale=scale, pad_x=pad_x, pad_y=pad_y, width=width, height=height)
                candidate = (purity, scale, width, height, pad_x, pad_y)
                if best is None or candidate > best:
                    best = candidate
    if best is None:
        return None
    purity, scale, width, height, pad_x, pad_y = best
    if purity < 0.92:
        return None
    return {
        "width": width,
        "height": height,
        "x": 0,
        "y": 0,
        "scale": scale,
        "pad_x": pad_x,
        "pad_y": pad_y,
    }


def _block_purity_score(
    array: np.ndarray,
    scale: int,
    pad_x: int,
    pad_y: int,
    width: int,
    height: int,
) -> float:
    total_purity = 0.0
    block_count = 0
    for grid_y in range(height):
        for grid_x in range(width):
            y0 = pad_y + (grid_y * scale)
            x0 = pad_x + (grid_x * scale)
            block = array[y0 : y0 + scale, x0 : x0 + scale]
            if block.shape != (scale, scale):
                continue
            values, counts = np.unique(block, return_counts=True)
            total_purity += float(counts[int(np.argmax(counts))]) / float(scale * scale)
            block_count += 1
    if block_count == 0:
        return 0.0
    return total_purity / float(block_count)


def _downsample_display_grid(array: np.ndarray, camera_meta: dict[str, int]) -> np.ndarray | None:
    if array.ndim != 2:
        return None
    scale = int(camera_meta.get("scale", 0) or 0)
    width = int(camera_meta.get("width", 0) or 0)
    height = int(camera_meta.get("height", 0) or 0)
    pad_x = int(camera_meta.get("pad_x", 0) or 0)
    pad_y = int(camera_meta.get("pad_y", 0) or 0)
    if scale <= 0 or width <= 0 or height <= 0:
        return None
    if pad_y + (height * scale) > array.shape[0] or pad_x + (width * scale) > array.shape[1]:
        return None
    grid = np.zeros((height, width), dtype=np.int64)
    for grid_y in range(height):
        for grid_x in range(width):
            y0 = pad_y + (grid_y * scale)
            x0 = pad_x + (grid_x * scale)
            block = array[y0 : y0 + scale, x0 : x0 + scale]
            values, counts = np.unique(block, return_counts=True)
            grid[grid_y, grid_x] = int(values[int(np.argmax(counts))])
    return grid


def _expand_actions(
    base_actions: tuple[ActionName, ...],
    grid: np.ndarray,
    camera_meta: dict[str, int] | None,
) -> tuple[ActionName, ...]:
    actions = [action for action in base_actions if action != "6"]
    if "6" in base_actions:
        if camera_meta is None:
            raise RuntimeError(
                "ARC click action 6 is legal but camera metadata is unavailable, so dense click actions cannot be exposed."
            )
        legacy_dense_setting = os.environ.get("ARCAGI_DENSE_CLICKS")
        if legacy_dense_setting is not None and str(legacy_dense_setting).strip().lower() in {"0", "false", "no", "off"}:
            raise RuntimeError(
                "ARCAGI_DENSE_CLICKS=0 silently hides legal ARC click actions. "
                "Use ARCAGI_SPARSE_CLICKS_BASELINE=1 only for explicitly labeled smoke/debug runs."
            )
        sparse_clicks = str(os.environ.get("ARCAGI_SPARSE_CLICKS_BASELINE", "")).strip().lower() in {
            "1",
            "true",
            "yes",
            "on",
        }
        dense_clicks = not sparse_clicks
        click_actions = _click_actions_for_grid(grid, camera_meta, dense=dense_clicks)
        if click_actions:
            actions.extend(click_actions)
        else:
            raise RuntimeError("ARC click action 6 is legal but no dense click actions could be generated.")
    return tuple(dict.fromkeys(actions))


def sparse_click_baseline_requested() -> bool:
    return str(os.environ.get("ARCAGI_SPARSE_CLICKS_BASELINE", "")).strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def require_dense_arc_action_surface(*, context: str, allow_sparse_click_smoke: bool = False) -> dict[str, object]:
    """Fail clean ARC train/eval paths if a debug action-surface reduction is enabled."""

    sparse_clicks = sparse_click_baseline_requested()
    dense_surface = not sparse_clicks
    metadata = {
        "dense_action_surface": dense_surface,
        "sparse_click_baseline": sparse_clicks,
        "allow_sparse_click_smoke": bool(allow_sparse_click_smoke),
    }
    if sparse_clicks and not allow_sparse_click_smoke:
        raise RuntimeError(
            f"{context} cannot run with ARCAGI_SPARSE_CLICKS_BASELINE=1 because it hides legal ARC click parameters. "
            "Unset it for train/eval, or pass the explicit sparse-click smoke/debug flag and do not claim ARC success."
        )
    return metadata


def _with_terminal_reset(actions: tuple[ActionName, ...], observation: Any) -> tuple[ActionName, ...]:
    if "0" in actions:
        return actions
    if GameAction is None or not hasattr(GameAction, "RESET"):
        return actions
    if not _is_game_over_observation(observation):
        return actions
    return ("0", *actions)


def _is_game_over_observation(observation: Any) -> bool:
    state = getattr(observation, "state", "")
    value = str(getattr(state, "value", state) or "")
    return value.endswith("GAME_OVER")


def _click_actions_for_grid(
    grid: np.ndarray,
    camera_meta: dict[str, int],
    max_candidates: int | None = None,
    *,
    dense: bool = False,
) -> list[ActionName]:
    if grid.ndim != 2 or grid.size <= 0:
        return []
    if dense and (max_candidates is None or int(max_candidates) <= 0):
        actions: list[ActionName] = []
        height, width = grid.shape
        for grid_y in range(height):
            for grid_x in range(width):
                display_x, display_y = _grid_cell_to_display(grid_x, grid_y, camera_meta)
                actions.append(f"click:{display_x}:{display_y}")
        return actions
    representatives = _component_representatives(grid)
    actions: list[ActionName] = []
    for _, _, grid_y, grid_x in _rank_component_representatives(representatives, grid.shape, max_candidates=max_candidates):
        display_x, display_y = _grid_cell_to_display(grid_x, grid_y, camera_meta)
        actions.append(f"click:{display_x}:{display_y}")
    return actions


def _component_representatives(grid: np.ndarray) -> list[tuple[int, int, int, int]]:
    if grid.ndim != 2 or grid.size <= 0:
        return []
    background = int(np.bincount(grid.flatten()).argmax())
    visited = np.zeros_like(grid, dtype=bool)
    height, width = grid.shape
    components: list[tuple[int, int, int, int]] = []
    for start_y in range(height):
        for start_x in range(width):
            color = int(grid[start_y, start_x])
            if visited[start_y, start_x] or color == background:
                continue
            stack = [(start_y, start_x)]
            visited[start_y, start_x] = True
            cells: list[tuple[int, int]] = []
            while stack:
                cell_y, cell_x = stack.pop()
                cells.append((cell_y, cell_x))
                for delta_y, delta_x in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    next_y = cell_y + delta_y
                    next_x = cell_x + delta_x
                    if next_y < 0 or next_x < 0 or next_y >= height or next_x >= width:
                        continue
                    if visited[next_y, next_x] or int(grid[next_y, next_x]) != color:
                        continue
                    visited[next_y, next_x] = True
                    stack.append((next_y, next_x))
            centroid_y = sum(cell[0] for cell in cells) / len(cells)
            centroid_x = sum(cell[1] for cell in cells) / len(cells)
            representative_y, representative_x = min(
                cells,
                key=lambda cell: abs(cell[0] - centroid_y) + abs(cell[1] - centroid_x),
            )
            components.append((color, len(cells), representative_y, representative_x))
    return components


def _rank_component_representatives(
    representatives: list[tuple[int, int, int, int]],
    grid_shape: tuple[int, int],
    max_candidates: int | None,
) -> list[tuple[int, int, int, int]]:
    if max_candidates is None or int(max_candidates) <= 0:
        return sorted(representatives, key=lambda item: (-item[1], item[0], item[2], item[3]))
    max_candidates = int(max_candidates)
    if len(representatives) <= max_candidates:
        return sorted(representatives, key=lambda item: (-item[1], item[0], item[2], item[3]))
    height, width = grid_shape
    span = max(float(height + width), 1.0)
    remaining = sorted(representatives, key=lambda item: (-item[1], item[0], item[2], item[3]))
    selected: list[tuple[int, int, int, int]] = []
    while remaining and len(selected) < max_candidates:
        if not selected:
            selected.append(remaining.pop(0))
            continue
        best_index = 0
        best_score = float("-inf")
        max_area = max(float(item[1]) for item in remaining)
        for index, candidate in enumerate(remaining):
            _, area, grid_y, grid_x = candidate
            nearest = min(abs(grid_y - chosen[2]) + abs(grid_x - chosen[3]) for chosen in selected)
            score = (0.7 * (float(area) / max(max_area, 1.0))) + (0.3 * (nearest / span))
            if score > best_score:
                best_score = score
                best_index = index
        selected.append(remaining.pop(best_index))
    return selected


def _grid_cell_to_display(grid_x: int, grid_y: int, camera_meta: dict[str, int]) -> tuple[int, int]:
    display_x = int(camera_meta["pad_x"] + (grid_x * camera_meta["scale"]) + (camera_meta["scale"] // 2))
    display_y = int(camera_meta["pad_y"] + (grid_y * camera_meta["scale"]) + (camera_meta["scale"] // 2))
    return display_x, display_y


def _build_arc_extras(
    observation: Any,
    grid: np.ndarray,
    base_actions: tuple[ActionName, ...],
    affordances: tuple[ActionName, ...],
    camera_meta: dict[str, int] | None,
    step_index: int,
    info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    action_roles = _action_roles(base_actions, affordances)
    inventory, flags = _arc_interface_inventory_flags(
        observation=observation,
        affordances=affordances,
        action_roles=action_roles,
        camera_meta=camera_meta,
    )
    extras: dict[str, Any] = {
        "adapter": "arc_toolkit",
        "raw_observation_type": type(observation).__name__,
        "game_state": str(getattr(observation, "state", "")),
        "levels_completed": int(getattr(observation, "levels_completed", 0) or 0),
        "background_color": int(np.bincount(np.asarray(grid).astype(np.int64).reshape(-1)).argmax()),
        "raw_available_actions": base_actions,
        "action_roles": action_roles,
        "inventory": inventory,
        "flags": flags,
        "cell_tags": _arc_cell_tags(grid, affordances, action_roles, camera_meta),
        "arc_step_index": step_index,
    }
    if camera_meta is not None:
        extras["camera_grid_shape"] = (camera_meta["height"], camera_meta["width"])
        extras["camera_origin"] = (camera_meta["x"], camera_meta["y"])
        extras["display_scale"] = camera_meta["scale"]
        extras["display_padding"] = (camera_meta["pad_x"], camera_meta["pad_y"])
    if info:
        extras.update(info)
    return extras


def _action_roles(base_actions: tuple[ActionName, ...], affordances: tuple[ActionName, ...]) -> dict[str, str]:
    roles = {action: ARC_ACTION_ROLES.get(action, "raw") for action in base_actions}
    for action in affordances:
        if action == "0":
            roles[action] = "reset_level"
        if action.startswith("click:"):
            roles[action] = "click"
    return roles


def _arc_interface_inventory_flags(
    *,
    observation: Any,
    affordances: tuple[ActionName, ...],
    action_roles: dict[str, str],
    camera_meta: dict[str, int] | None,
) -> tuple[dict[str, str], dict[str, str]]:
    context = build_action_schema_context(affordances, action_roles)
    counts: dict[str, int] = {key: 0 for key in ("move", "click", "select", "interact", "undo", "reset", "wait", "raw", "other")}
    click_bins: set[tuple[int, int]] = set()
    for action in affordances:
        schema = build_action_schema(action, context)
        counts[schema.action_type if schema.action_type in counts else "other"] += 1
        if schema.coarse_bin is not None:
            click_bins.add(schema.coarse_bin)
    inventory: dict[str, str] = {
        "interface_move_actions": str(counts["move"]),
        "interface_click_actions": str(counts["click"]),
        "interface_select_actions": str(counts["select"]),
        "interface_interact_actions": str(counts["interact"]),
        "interface_undo_actions": str(counts["undo"]),
        "interface_reset_actions": str(counts["reset"]),
        "interface_raw_actions": str(counts["raw"]),
        "interface_click_bin_count": str(len(click_bins)),
        "interface_levels_completed": str(int(getattr(observation, "levels_completed", 0) or 0)),
        "interface_game_state": str(getattr(observation, "state", "") or "unknown"),
    }
    if camera_meta is not None:
        inventory["interface_camera_width"] = str(int(camera_meta["width"]))
        inventory["interface_camera_height"] = str(int(camera_meta["height"]))
        inventory["interface_display_scale"] = str(int(camera_meta["scale"]))
        inventory["interface_display_pad_x"] = str(int(camera_meta["pad_x"]))
        inventory["interface_display_pad_y"] = str(int(camera_meta["pad_y"]))
    flags: dict[str, str] = {
        "interface_has_click": "1" if counts["click"] > 0 else "0",
        "interface_has_select": "1" if counts["select"] > 0 else "0",
        "interface_has_interact": "1" if counts["interact"] > 0 else "0",
        "interface_has_undo": "1" if counts["undo"] > 0 else "0",
        "interface_has_reset": "1" if counts["reset"] > 0 else "0",
        "interface_has_mode_actions": "1" if (counts["click"] + counts["select"]) > 0 else "0",
        "interface_dense_clicks": "1" if counts["click"] >= 8 else "0",
        "interface_parametric_clicks": "1" if any(action.startswith("click:") for action in affordances) else "0",
    }
    return inventory, flags


def _arc_cell_tags(
    grid: np.ndarray,
    affordances: tuple[ActionName, ...],
    action_roles: dict[str, str],
    camera_meta: dict[str, int] | None,
) -> dict[tuple[int, int], tuple[str, ...]]:
    if grid.ndim != 2:
        return {}
    height, width = grid.shape
    context = build_action_schema_context(affordances, action_roles)
    selector_count = sum(
        1 for action in affordances if build_action_schema(action, context).action_type in {"click", "select"}
    )
    background = int(np.bincount(np.asarray(grid).astype(np.int64).reshape(-1)).argmax())
    tags_by_cell: dict[tuple[int, int], set[str]] = {}
    for action in affordances:
        schema = build_action_schema(action, context)
        if schema.action_type != "click" or schema.click is None:
            continue
        if camera_meta is not None:
            grid_cell = _display_to_grid_cell(schema.click[0], schema.click[1], camera_meta)
        else:
            grid_cell = (schema.click[1], schema.click[0])
        if grid_cell is None:
            continue
        grid_y, grid_x = grid_cell
        if grid_y < 0 or grid_x < 0 or grid_y >= height or grid_x >= width:
            continue
        if int(grid[grid_y, grid_x]) == background:
            continue
        tags = tags_by_cell.setdefault((grid_y, grid_x), set())
        tags.add("clickable")
        tags.add("interface_target")
        if schema.coarse_bin is not None:
            tags.add(f"click_bin_{schema.coarse_bin[0]}_{schema.coarse_bin[1]}")
        if selector_count >= 2:
            tags.add("selector_candidate")
    return {cell: tuple(sorted(tags)) for cell, tags in tags_by_cell.items()}


def _display_to_grid_cell(
    display_x: int,
    display_y: int,
    camera_meta: dict[str, int],
) -> tuple[int, int] | None:
    scale = int(camera_meta.get("scale", 0) or 0)
    if scale <= 0:
        return None
    pad_x = int(camera_meta.get("pad_x", 0) or 0)
    pad_y = int(camera_meta.get("pad_y", 0) or 0)
    grid_x = int((display_x - pad_x) // scale)
    grid_y = int((display_y - pad_y) // scale)
    return (grid_y, grid_x)


def _default_operation_mode():
    if OperationMode is None:
        return None
    return getattr(OperationMode, "OFFLINE", None)


def arc_operation_mode(name: str) -> Any | None:
    if OperationMode is None:
        return None
    upper_name = name.strip().upper()
    return getattr(OperationMode, upper_name, None)
