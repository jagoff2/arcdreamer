from __future__ import annotations

import copy
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from .arcagi3_adapter import (
    CELL_AGENT,
    CELL_EMPTY,
    CELL_GOAL,
    CELL_RESOURCE,
    ArcAGI3Observation,
    ArcAGI3StepResult,
)


DEFAULT_ENVIRONMENTS_DIR = "runs/arcagi3_official_envs"
DEFAULT_RECORDINGS_DIR = "runs/arcagi3_official_recordings"


@dataclass
class OfficialGameSpec:
    game_id: str
    title: str
    tags: list[str] = field(default_factory=list)
    baseline_actions: list[int] = field(default_factory=list)
    class_name: str | None = None


@dataclass
class OfficialFixtureProxy:
    fixture_id: str
    title: str
    max_steps: int
    actions: tuple[str, ...] = field(default_factory=tuple)
    max_score: float = 1.0
    public_rules: dict[str, Any] = field(default_factory=dict)


def require_official_runtime() -> tuple[Any, Any, Any, Any]:
    try:
        from arc_agi import Arcade, OperationMode
        from arcengine import GameAction, GameState
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        raise RuntimeError(
            "Official ARC-AGI-3 runtime is not installed for this interpreter. "
            "Install with `python -m pip install arc-agi` under Python >=3.12."
        ) from exc
    return Arcade, OperationMode, GameAction, GameState


def make_arcade(
    *,
    operation_mode: str = "normal",
    environments_dir: str | Path = DEFAULT_ENVIRONMENTS_DIR,
    recordings_dir: str | Path = DEFAULT_RECORDINGS_DIR,
) -> Any:
    Arcade, OperationMode, _, _ = require_official_runtime()
    return Arcade(
        operation_mode=OperationMode(operation_mode),
        environments_dir=str(environments_dir),
        recordings_dir=str(recordings_dir),
    )


def discover_official_games(
    arcade: Any,
    *,
    game_ids: list[str] | None = None,
    limit: int | None = None,
) -> list[OfficialGameSpec]:
    requested = set(game_ids or [])
    specs: list[OfficialGameSpec] = []
    for env in arcade.get_environments():
        game_id = str(env.game_id)
        if requested and game_id not in requested and game_id.split("-", 1)[0] not in requested:
            continue
        specs.append(
            OfficialGameSpec(
                game_id=game_id,
                title=str(env.title or game_id),
                tags=list(env.tags or []),
                baseline_actions=[int(item) for item in (env.baseline_actions or [])],
                class_name=env.class_name,
            )
        )
    specs.sort(key=lambda item: item.game_id)
    return specs[:limit] if limit is not None else specs


def official_runtime_versions() -> dict[str, Any]:
    versions: dict[str, Any] = {}
    for name in ["arc_agi", "arcengine", "torch", "numpy"]:
        try:
            module = __import__(name)
            versions[name] = {
                "file": getattr(module, "__file__", None),
                "version": getattr(module, "__version__", None),
            }
        except Exception as exc:  # pragma: no cover - diagnostic only
            versions[name] = {"error": repr(exc)}
    try:
        import importlib.metadata as metadata

        versions["arc-agi"] = {"version": metadata.version("arc-agi")}
        versions["arcengine-package"] = {"version": metadata.version("arcengine")}
    except Exception:
        pass
    return versions


class OfficialArcAGI3Env:
    def __init__(
        self,
        arcade: Any,
        spec: OfficialGameSpec,
        *,
        seed: int = 0,
        max_steps: int | None = None,
        max_click_actions: int = 192,
        save_recording: bool = False,
    ) -> None:
        self.arcade = arcade
        self.spec = spec
        self.seed = int(seed)
        self.max_steps = int(max_steps or default_max_steps(spec))
        self.max_click_actions = int(max_click_actions)
        self.save_recording = bool(save_recording)
        self.fixture = OfficialFixtureProxy(
            fixture_id=spec.game_id,
            title=spec.title,
            max_steps=self.max_steps,
            max_score=1.0,
            public_rules={"official_runtime": True, "tags": list(spec.tags)},
        )
        self.scorecard_id: str | None = None
        self.wrapper: Any | None = None
        self.response: Any | None = None
        self.step_index = 0
        self.score = 0.0
        self.terminated = False
        self.truncated = False
        self.levels_completed = 0
        self.win_levels = max(1, len(spec.baseline_actions))
        self._game_state_name = "NOT_PLAYED"

    @property
    def task_id(self) -> str:
        return f"arcagi3-official/{self.spec.game_id}"

    @property
    def episode_id(self) -> str:
        return f"{self.task_id}/episode_{self.seed}"

    def clone(self) -> "OfficialArcAGI3Env":
        return copy.deepcopy(self)

    def legal_actions(self) -> tuple[str, ...]:
        if self.response is None:
            return tuple()
        return official_legal_actions(self.response, self.max_click_actions)

    def reset(self, seed: int | None = None) -> ArcAGI3Observation:
        if seed is not None:
            self.seed = int(seed)
        self.close()
        self.scorecard_id = self.arcade.create_scorecard(tags=["agent", "official-runtime"])
        self.wrapper = self.arcade.make(
            self.spec.game_id,
            seed=self.seed,
            scorecard_id=self.scorecard_id,
            save_recording=self.save_recording,
            include_frame_data=True,
        )
        if self.wrapper is None:
            raise RuntimeError(f"official runtime could not create game {self.spec.game_id}")
        self.response = self.wrapper.observation_space or self.wrapper.reset()
        if self.response is None:
            raise RuntimeError(f"official runtime returned no initial frame for {self.spec.game_id}")
        self.step_index = 0
        self.score = 0.0
        self.terminated = False
        self.truncated = False
        self.levels_completed = int(getattr(self.response, "levels_completed", 0))
        self.win_levels = max(1, int(getattr(self.response, "win_levels", 0) or len(self.spec.baseline_actions) or 1))
        self._game_state_name = state_name(getattr(self.response, "state", "NOT_FINISHED"))
        self.fixture.actions = self.legal_actions()
        return self._observation()

    def step(self, action: str) -> ArcAGI3StepResult:
        _, _, GameAction, _ = require_official_runtime()
        if self.response is None or self.wrapper is None:
            raise RuntimeError("reset must be called before step")
        if self.terminated or self.truncated:
            return ArcAGI3StepResult(
                self._observation(),
                0.0,
                self.terminated,
                self.truncated,
                {"events": ["already_terminal"], "score": self.score, "normalized_score": self.normalized_score()},
            )
        legal = self.legal_actions()
        events: list[str] = []
        invalid = action not in legal
        if invalid:
            action = legal[0] if legal else "wait"
            events.append("invalid_action")
        previous_levels = int(getattr(self.response, "levels_completed", 0))
        previous_state = state_name(getattr(self.response, "state", "NOT_FINISHED"))
        game_action, data = parse_official_action(action, GameAction)
        response = self.wrapper.step(game_action, data=data, reasoning={"action": action})
        self.step_index += 1
        reward = -0.001
        if response is None:
            self.truncated = True
            events.append("runtime_no_response")
        else:
            self.response = response
            new_levels = int(getattr(response, "levels_completed", 0))
            self.levels_completed = new_levels
            self.win_levels = max(1, int(getattr(response, "win_levels", 0) or self.win_levels))
            if new_levels > previous_levels:
                events.extend(["resource_collected", "level_completed"])
                reward += float(new_levels - previous_levels)
            self._game_state_name = state_name(getattr(response, "state", "NOT_FINISHED"))
            if self._game_state_name == "WIN":
                self.terminated = True
                events.append("goal_reached")
                if previous_state != "WIN":
                    reward += 1.0
            elif self._game_state_name == "GAME_OVER":
                self.truncated = True
                events.append("game_over")
                reward -= 0.1
        if self.step_index >= self.max_steps and not self.terminated:
            self.truncated = True
            events.append("step_limit")
        self.score = self.normalized_score()
        obs = self._observation()
        info = {
            "events": events,
            "score": self.score,
            "positive_score": max(0.0, self.score),
            "normalized_score": self.normalized_score(),
            "levels_completed": self.levels_completed,
            "win_levels": self.win_levels,
            "game_state": self._game_state_name,
            "official_game_id": self.spec.game_id,
        }
        return ArcAGI3StepResult(obs, float(reward), self.terminated, self.truncated, info)

    def normalized_score(self) -> float:
        if self._game_state_name == "WIN":
            return 1.0
        return float(max(0.0, min(1.0, self.levels_completed / max(self.win_levels, 1))))

    def close(self) -> dict[str, Any] | None:
        if self.scorecard_id is None:
            return None
        try:
            scorecard = self.arcade.close_scorecard(self.scorecard_id)
            if scorecard is None:
                return None
            return scorecard.model_dump(mode="json")
        finally:
            self.scorecard_id = None

    def _observation(self) -> ArcAGI3Observation:
        if self.response is None:
            grid = np.full((64, 64), CELL_EMPTY, dtype=np.int64)
            frame_stats: dict[str, Any] = {}
        else:
            grid, frame_stats = official_frame_to_grid(
                self.response,
                legal_ids=tuple(int(item) for item in getattr(self.response, "available_actions", []) or []),
                max_click_actions=self.max_click_actions,
            )
        return ArcAGI3Observation(
            task_id=self.task_id,
            episode_id=self.episode_id,
            step_index=self.step_index,
            grid=grid,
            available_actions=self.legal_actions(),
            extras={
                "adapter": "official_arcagi3_runtime",
                "fixture_id": self.spec.game_id,
                "title": self.spec.title,
                "score": self.score,
                "normalized_score": self.normalized_score(),
                "game_state": self._game_state_name,
                "levels_completed": self.levels_completed,
                "win_levels": self.win_levels,
                "public_rules": self.fixture.public_rules,
                "max_score": 1.0,
                "official_tags": list(self.spec.tags),
                "official_baseline_actions": list(self.spec.baseline_actions),
                "frame_stats": frame_stats,
            },
        )


def default_max_steps(spec: OfficialGameSpec) -> int:
    if spec.baseline_actions:
        return int(min(240, max(32, sum(spec.baseline_actions[:2]) * 2)))
    return 96


def state_name(value: Any) -> str:
    return str(getattr(value, "name", value)).split(".")[-1]


def parse_official_action(action: str, GameAction: Any) -> tuple[Any, dict[str, int]]:
    if action.startswith("click:"):
        _, x_raw, y_raw = action.split(":", 2)
        return GameAction.from_id(6), {"x": int(x_raw), "y": int(y_raw)}
    if action == "wait":
        return GameAction.from_id(7), {}
    try:
        return GameAction.from_id(int(action)), {}
    except Exception:
        return GameAction.from_id(7), {}


def official_legal_actions(response: Any, max_click_actions: int) -> tuple[str, ...]:
    available = tuple(int(item) for item in (getattr(response, "available_actions", []) or []))
    actions: list[str] = []
    for action_id in available:
        if action_id == 0:
            continue
        if action_id == 6:
            for y, x in click_candidates(response, max_click_actions=max_click_actions):
                actions.append(f"click:{x}:{y}")
        else:
            actions.append(str(action_id))
    if not actions:
        actions.append("wait")
    return tuple(dict.fromkeys(actions))


def official_frame_to_grid(
    response: Any,
    *,
    legal_ids: tuple[int, ...],
    max_click_actions: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    frame = collapse_frame(response)
    grid = np.full(frame.shape, CELL_EMPTY, dtype=np.int64)
    if frame.size == 0:
        return grid, {"empty_frame": True}
    background = mode_value(frame)
    visible = frame != background
    grid[visible] = CELL_RESOURCE
    candidates = click_candidates(response, max_click_actions=max_click_actions)
    for y, x in candidates:
        if 0 <= y < grid.shape[0] and 0 <= x < grid.shape[1]:
            grid[y, x] = CELL_GOAL
    if any(action_id in legal_ids for action_id in (1, 2, 3, 4, 5, 7)):
        y = min(grid.shape[0] - 1, max(0, grid.shape[0] // 2))
        x = min(grid.shape[1] - 1, max(0, grid.shape[1] // 2))
        grid[y, x] = CELL_AGENT
    values, counts = np.unique(frame, return_counts=True)
    return grid, {
        "shape": list(frame.shape),
        "background": int(background),
        "unique_values": int(len(values)),
        "non_background": int(np.count_nonzero(visible)),
        "click_candidates": len(candidates),
        "value_histogram_head": [
            {"value": int(v), "count": int(c)}
            for v, c in sorted(zip(values.tolist(), counts.tolist()), key=lambda item: item[1], reverse=True)[:10]
        ],
    }


def collapse_frame(response: Any) -> np.ndarray:
    layers = [np.asarray(layer, dtype=np.int64) for layer in getattr(response, "frame", [])]
    if not layers:
        return np.full((64, 64), 0, dtype=np.int64)
    shape = layers[0].shape
    valid = [layer for layer in layers if layer.shape == shape]
    if len(valid) == 1:
        return valid[0].copy()
    stacked = np.stack(valid, axis=0)
    return stacked.max(axis=0)


def mode_value(frame: np.ndarray) -> int:
    values, counts = np.unique(frame.reshape(-1), return_counts=True)
    return int(values[int(np.argmax(counts))])


def click_candidates(response: Any, *, max_click_actions: int) -> tuple[tuple[int, int], ...]:
    frame = collapse_frame(response)
    if frame.size == 0:
        return ((32, 32),)
    background = mode_value(frame)
    candidates: list[tuple[int, int]] = []
    for value in rare_values(frame, background):
        candidates.extend(component_points(frame == value))
        if len(candidates) >= max_click_actions:
            break
    if len(candidates) < max_click_actions:
        ys, xs = np.nonzero(frame != background)
        if len(ys):
            order = np.linspace(0, len(ys) - 1, num=min(max_click_actions - len(candidates), len(ys)), dtype=int)
            for idx in order:
                candidates.append((int(ys[idx]), int(xs[idx])))
    if not candidates:
        candidates.append((frame.shape[0] // 2, frame.shape[1] // 2))
    deduped: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in candidates:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
        if len(deduped) >= max_click_actions:
            break
    return tuple(deduped)


def rare_values(frame: np.ndarray, background: int) -> list[int]:
    values, counts = np.unique(frame.reshape(-1), return_counts=True)
    pairs = [
        (int(value), int(count))
        for value, count in zip(values.tolist(), counts.tolist())
        if int(value) != int(background)
    ]
    pairs.sort(key=lambda item: (item[1], item[0]))
    return [value for value, _ in pairs]


def component_points(mask: np.ndarray) -> list[tuple[int, int]]:
    mask = np.asarray(mask, dtype=bool)
    visited = np.zeros(mask.shape, dtype=bool)
    points: list[tuple[int, int]] = []
    height, width = mask.shape
    for y0, x0 in np.argwhere(mask):
        y = int(y0)
        x = int(x0)
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        count = 0
        sum_y = 0
        sum_x = 0
        min_y = max_y = y
        min_x = max_x = x
        while stack:
            cy, cx = stack.pop()
            count += 1
            sum_y += cy
            sum_x += cx
            min_y = min(min_y, cy)
            max_y = max(max_y, cy)
            min_x = min(min_x, cx)
            max_x = max(max_x, cx)
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        center = (int(round(sum_y / max(count, 1))), int(round(sum_x / max(count, 1))))
        box_center = ((min_y + max_y) // 2, (min_x + max_x) // 2)
        points.append(center)
        if box_center != center:
            points.append(box_center)
    return points


def write_official_game_manifest(path: str | Path, specs: list[OfficialGameSpec]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps({"games": [spec.__dict__ for spec in specs]}, indent=2),
        encoding="utf-8",
    )
