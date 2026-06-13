from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Mapping

import numpy as np
import torch

from .grid_perception import PerceptualScene, parse_grid


Coord = tuple[int, int]


@dataclass(frozen=True)
class CoordinateCandidate:
    y: int
    x: int
    reason: str
    score: float

    def compact(self) -> dict[str, Any]:
        return asdict(self)


def _add_candidate(
    candidates: dict[Coord, CoordinateCandidate],
    y: int,
    x: int,
    *,
    reason: str,
    score: float,
    shape: tuple[int, int],
) -> None:
    height, width = shape
    if not (0 <= y < height and 0 <= x < width):
        return
    key = (int(y), int(x))
    existing = candidates.get(key)
    if existing is None or float(score) > existing.score:
        candidates[key] = CoordinateCandidate(int(y), int(x), reason, float(score))


def _neighbors4(y: int, x: int, shape: tuple[int, int]) -> list[Coord]:
    height, width = shape
    out: list[Coord] = []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width:
            out.append((ny, nx))
    return out


def propose_coordinate_candidates(
    scene: PerceptualScene,
    *,
    max_candidates: int = 32,
) -> list[CoordinateCandidate]:
    candidates: dict[Coord, CoordinateCandidate] = {}
    shape = scene.shape
    for obj in scene.objects:
        if obj.kind != "component":
            continue
        cy, cx = int(round(obj.centroid[0])), int(round(obj.centroid[1]))
        _add_candidate(candidates, cy, cx, reason="object_center", score=2.00 + obj.salience * 0.02, shape=shape)
        top, left, bottom, right = obj.bbox
        for y, x in [(top, left), (top, right), (bottom, left), (bottom, right)]:
            _add_candidate(candidates, y, x, reason="object_corner", score=1.40 + obj.salience * 0.01, shape=shape)
        for y, x in obj.boundary_cells:
            _add_candidate(candidates, y, x, reason="object_boundary", score=1.10 + obj.changed_count * 0.20, shape=shape)
            for ny, nx in _neighbors4(y, x, shape):
                if scene.salience_map[ny][nx] == 0.0:
                    _add_candidate(candidates, ny, nx, reason="adjacent_empty", score=0.80, shape=shape)
    for y, row in enumerate(scene.salience_map):
        for x, value in enumerate(row):
            if float(value) >= 0.75:
                _add_candidate(candidates, y, x, reason="high_salience", score=1.80 + float(value), shape=shape)
    color_regions = [obj for obj in scene.objects if obj.kind == "component"]
    by_canonical: dict[tuple[int, str], list[Any]] = {}
    for obj in color_regions:
        by_canonical.setdefault((obj.color, obj.canonical_hash), []).append(obj)
    for group in by_canonical.values():
        if len(group) < 2:
            continue
        ys = sorted(round(obj.centroid[0], 3) for obj in group)
        xs = sorted(round(obj.centroid[1], 3) for obj in group)
        expected_y_step = _common_step(ys)
        expected_x_step = _common_step(xs)
        for obj in group:
            cy, cx = int(round(obj.centroid[0])), int(round(obj.centroid[1]))
            if expected_y_step is not None or expected_x_step is not None:
                _add_candidate(candidates, cy, cx, reason="repeated_pattern", score=1.65, shape=shape)
    ordered = sorted(candidates.values(), key=lambda item: (-item.score, item.y, item.x, item.reason))
    return ordered[: max(0, int(max_candidates))]


def _common_step(values: list[float]) -> float | None:
    if len(values) < 2:
        return None
    deltas = [round(values[i + 1] - values[i], 3) for i in range(len(values) - 1)]
    if not deltas:
        return None
    counts: dict[float, int] = {}
    for delta in deltas:
        if delta > 0:
            counts[delta] = counts.get(delta, 0) + 1
    if not counts:
        return None
    return max(counts.items(), key=lambda item: (item[1], -item[0]))[0]


@dataclass
class CoordinateAffordanceModel:
    shape: tuple[int, int]
    trials: dict[str, int]
    changed: dict[str, int]
    no_op: dict[str, int]
    score_sum: dict[str, float]

    @classmethod
    def fresh(cls, shape: tuple[int, int]) -> "CoordinateAffordanceModel":
        return cls(tuple(int(item) for item in shape), {}, {}, {}, {})

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "CoordinateAffordanceModel":
        return cls(
            shape=tuple(int(item) for item in payload.get("shape", (0, 0))),
            trials={str(k): int(v) for k, v in payload.get("trials", {}).items()},
            changed={str(k): int(v) for k, v in payload.get("changed", {}).items()},
            no_op={str(k): int(v) for k, v in payload.get("no_op", {}).items()},
            score_sum={str(k): float(v) for k, v in payload.get("score_sum", {}).items()},
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": "coordinate_affordance_model_v1",
            "shape": [int(item) for item in self.shape],
            "trials": dict(self.trials),
            "changed": dict(self.changed),
            "no_op": dict(self.no_op),
            "score_sum": dict(self.score_sum),
            "heatmap": self.heatmap().tolist(),
        }

    def update(
        self,
        before_grid: Any,
        after_grid: Any,
        coordinate: Coord,
        *,
        score_delta: float = 0.0,
    ) -> dict[str, Any]:
        before = np.asarray(before_grid, dtype=np.int64)
        after = np.asarray(after_grid, dtype=np.int64)
        if before.shape != after.shape:
            raise ValueError(f"before/after grid shapes differ: {before.shape} vs {after.shape}")
        if tuple(before.shape) != self.shape:
            raise ValueError(f"model shape {self.shape} does not match grid shape {before.shape}")
        y, x = int(coordinate[0]), int(coordinate[1])
        if not (0 <= y < self.shape[0] and 0 <= x < self.shape[1]):
            raise ValueError(f"coordinate {(y, x)} is outside shape {self.shape}")
        key = self._key(y, x)
        changed = bool(np.any(before != after))
        self.trials[key] = self.trials.get(key, 0) + 1
        self.changed[key] = self.changed.get(key, 0) + int(changed)
        self.no_op[key] = self.no_op.get(key, 0) + int(not changed)
        self.score_sum[key] = self.score_sum.get(key, 0.0) + float(score_delta)
        return {
            "coordinate": [y, x],
            "changed": changed,
            "trials": self.trials[key],
            "p_change": self.p_change(y, x),
            "mean_score_delta": self.score_sum[key] / max(self.trials[key], 1),
        }

    def p_change(self, y: int, x: int) -> float:
        key = self._key(int(y), int(x))
        return float(self.changed.get(key, 0) / max(self.trials.get(key, 0), 1))

    def heatmap(self) -> np.ndarray:
        heat = np.zeros(self.shape, dtype=np.float32)
        for key in self.trials:
            y, x = self._parse_key(key)
            heat[y, x] = self.p_change(y, x)
        return heat

    @staticmethod
    def _key(y: int, x: int) -> str:
        return f"{int(y)},{int(x)}"

    @staticmethod
    def _parse_key(key: str) -> Coord:
        y, x = key.split(",", 1)
        return int(y), int(x)


def fresh_coordinate_affordances() -> dict[str, Any]:
    return {
        "schema": "runtime_coordinate_affordances_v1",
        "models": {},
        "candidate_history": [],
        "updates": 0,
    }


def _grid_from_observation(observation: Mapping[str, Any]) -> np.ndarray | None:
    explicit_grid = "grid" in observation
    value = observation.get("grid") if explicit_grid else observation.get("sensory")
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim == 2 and (explicit_grid or not torch.is_floating_point(tensor)):
            return tensor.numpy().astype(np.int64)
    return None


def _coordinate_from_metadata(metadata: Mapping[str, Any] | None) -> Coord | None:
    if not metadata:
        return None
    value = metadata.get("action_coordinate", metadata.get("coordinate_action"))
    if value is None:
        return None
    if isinstance(value, Mapping):
        if "y" in value and "x" in value:
            return int(value["y"]), int(value["x"])
        if "row" in value and "col" in value:
            return int(value["row"]), int(value["col"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return None


def update_runtime_coordinate_affordances(
    state: dict[str, Any],
    *,
    observation: Mapping[str, Any],
    next_observation: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    max_candidates: int = 32,
) -> dict[str, Any]:
    if not state:
        state.update(fresh_coordinate_affordances())
    before = _grid_from_observation(observation)
    after = _grid_from_observation(next_observation)
    if before is None or after is None or before.shape != after.shape:
        return state
    scene = parse_grid(before)
    candidates = propose_coordinate_candidates(scene, max_candidates=max_candidates)
    state.setdefault("candidate_history", []).append([candidate.compact() for candidate in candidates])
    if len(state["candidate_history"]) > 64:
        del state["candidate_history"][: len(state["candidate_history"]) - 64]
    coordinate = _coordinate_from_metadata(metadata)
    if coordinate is not None:
        shape_key = f"{before.shape[0]}x{before.shape[1]}"
        models = state.setdefault("models", {})
        model = CoordinateAffordanceModel.from_dict(models[shape_key]) if shape_key in models else CoordinateAffordanceModel.fresh(tuple(before.shape))
        model.update(before, after, coordinate, score_delta=float((metadata or {}).get("score_delta", 0.0)))
        models[shape_key] = model.to_dict()
        state["updates"] = int(state.get("updates", 0)) + 1
    return state
