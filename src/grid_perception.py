from __future__ import annotations

import hashlib
import json
from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
from typing import Any

import numpy as np


Cell = tuple[int, int]


def _to_grid(grid: Any) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    if arr.ndim != 2:
        raise ValueError(f"grid perception expects a 2D grid, got shape {arr.shape}")
    if arr.shape[0] > 64 or arr.shape[1] > 64:
        raise ValueError(f"grid perception supports grids up to 64x64, got shape {arr.shape}")
    return arr


def _stable_hash(value: Any, length: int = 16) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def raw_grid_hash(grid: Any) -> str:
    arr = _to_grid(grid)
    return _stable_hash({"shape": list(arr.shape), "values": arr.tolist()})


def _background_value(arr: np.ndarray) -> int:
    counts = Counter(int(item) for item in arr.reshape(-1))
    return int(counts.most_common(1)[0][0]) if counts else 0


def _neighbors4(y: int, x: int, height: int, width: int) -> list[Cell]:
    out: list[Cell] = []
    for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
        ny, nx = y + dy, x + dx
        if 0 <= ny < height and 0 <= nx < width:
            out.append((ny, nx))
    return out


def _neighbors8(y: int, x: int, height: int, width: int) -> list[Cell]:
    out: list[Cell] = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if dy == 0 and dx == 0:
                continue
            ny, nx = y + dy, x + dx
            if 0 <= ny < height and 0 <= nx < width:
                out.append((ny, nx))
    return out


def _bbox(cells: tuple[Cell, ...]) -> tuple[int, int, int, int]:
    ys = [cell[0] for cell in cells]
    xs = [cell[1] for cell in cells]
    return (min(ys), min(xs), max(ys), max(xs))


def _centroid(cells: tuple[Cell, ...]) -> tuple[float, float]:
    return (
        float(sum(cell[0] for cell in cells) / len(cells)),
        float(sum(cell[1] for cell in cells) / len(cells)),
    )


def _canonical_cells(cells: tuple[Cell, ...]) -> tuple[Cell, ...]:
    top, left, _bottom, _right = _bbox(cells)
    return tuple(sorted((y - top, x - left) for y, x in cells))


def _boundary_cells(cells: tuple[Cell, ...]) -> tuple[Cell, ...]:
    cell_set = set(cells)
    top, left, bottom, right = _bbox(cells)
    result = []
    for y, x in cells:
        if y in (top, bottom) or x in (left, right):
            result.append((y, x))
            continue
        if any(neighbor not in cell_set for neighbor in _neighbors4(y, x, bottom + 1, right + 1)):
            result.append((y, x))
    return tuple(sorted(result))


def _hole_count(cells: tuple[Cell, ...]) -> int:
    cell_set = set(cells)
    top, left, bottom, right = _bbox(cells)
    height = bottom - top + 1
    width = right - left + 1
    blocked = {(y - top, x - left) for y, x in cell_set}
    seen: set[Cell] = set()
    holes = 0
    for y in range(height):
        for x in range(width):
            if (y, x) in blocked or (y, x) in seen:
                continue
            q: deque[Cell] = deque([(y, x)])
            seen.add((y, x))
            touches_border = False
            while q:
                cy, cx = q.popleft()
                touches_border = touches_border or cy in (0, height - 1) or cx in (0, width - 1)
                for ny, nx in _neighbors4(cy, cx, height, width):
                    if (ny, nx) not in blocked and (ny, nx) not in seen:
                        seen.add((ny, nx))
                        q.append((ny, nx))
            if not touches_border:
                holes += 1
    return holes


def _symmetry(cells: tuple[Cell, ...]) -> dict[str, bool]:
    canon = set(_canonical_cells(cells))
    top, left, bottom, right = _bbox(cells)
    height = bottom - top
    width = right - left
    vertical = {(y, width - x) for y, x in canon} == canon
    horizontal = {(height - y, x) for y, x in canon} == canon
    diagonal = height == width and {(x, y) for y, x in canon} == canon
    return {"vertical": bool(vertical), "horizontal": bool(horizontal), "diagonal": bool(diagonal)}


@dataclass(frozen=True)
class ObjectSlot:
    object_id: str
    canonical_hash: str
    kind: str
    color: int
    cells: tuple[Cell, ...]
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    area: int
    holes: int
    canonical_cells: tuple[Cell, ...]
    boundary_cells: tuple[Cell, ...]
    changed_count: int
    symmetry: dict[str, bool]
    salience: float

    def compact(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["cells"] = [list(cell) for cell in self.cells]
        payload["bbox"] = list(self.bbox)
        payload["centroid"] = [round(float(self.centroid[0]), 4), round(float(self.centroid[1]), 4)]
        payload["canonical_cells"] = [list(cell) for cell in self.canonical_cells]
        payload["boundary_cells"] = [list(cell) for cell in self.boundary_cells]
        payload["salience"] = round(float(self.salience), 6)
        return payload


@dataclass(frozen=True)
class Relation:
    source: str
    target: str
    relation: str
    detail: dict[str, Any] = field(default_factory=dict)

    def compact(self) -> dict[str, Any]:
        return {
            "source": self.source,
            "target": self.target,
            "relation": self.relation,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class EditScript:
    changed_count: int
    unchanged_count: int
    cell_edits: tuple[dict[str, Any], ...]
    appears: tuple[dict[str, Any], ...]
    disappears: tuple[dict[str, Any], ...]
    moves: tuple[dict[str, Any], ...]

    def compact(self) -> dict[str, Any]:
        return {
            "changed_count": self.changed_count,
            "unchanged_count": self.unchanged_count,
            "cell_edits": list(self.cell_edits),
            "appears": list(self.appears),
            "disappears": list(self.disappears),
            "moves": list(self.moves),
        }


@dataclass(frozen=True)
class Binding:
    key: str
    vector: tuple[int, ...]

    def compact(self) -> dict[str, Any]:
        return {"key": self.key, "vector": list(self.vector)}


@dataclass(frozen=True)
class PerceptualScene:
    raw_hash: str
    shape: tuple[int, int]
    background: int
    grid_embedding: tuple[float, ...]
    objects: tuple[ObjectSlot, ...]
    relation_graph: tuple[Relation, ...]
    bindings: tuple[Binding, ...]
    salience_map: tuple[tuple[float, ...], ...]
    active_cells: tuple[Cell, ...]
    edit_script: EditScript

    def compact(self) -> dict[str, Any]:
        return {
            "raw_hash": self.raw_hash,
            "shape": list(self.shape),
            "background": self.background,
            "grid_embedding": [round(float(item), 6) for item in self.grid_embedding],
            "objects": [obj.compact() for obj in self.objects],
            "relation_graph": [rel.compact() for rel in self.relation_graph],
            "bindings": [binding.compact() for binding in self.bindings],
            "salience_map": [[round(float(item), 6) for item in row] for row in self.salience_map],
            "active_cells": [list(cell) for cell in self.active_cells],
            "edit_script": self.edit_script.compact(),
        }


def _component_objects(arr: np.ndarray, changed: np.ndarray) -> list[ObjectSlot]:
    height, width = arr.shape
    bg = _background_value(arr)
    seen = np.zeros(arr.shape, dtype=bool)
    objects: list[ObjectSlot] = []
    for y0 in range(height):
        for x0 in range(width):
            if seen[y0, x0] or int(arr[y0, x0]) == bg:
                continue
            color = int(arr[y0, x0])
            q: deque[Cell] = deque([(y0, x0)])
            seen[y0, x0] = True
            cells: list[Cell] = []
            while q:
                y, x = q.popleft()
                cells.append((int(y), int(x)))
                for ny, nx in _neighbors4(y, x, height, width):
                    if not seen[ny, nx] and int(arr[ny, nx]) == color:
                        seen[ny, nx] = True
                        q.append((ny, nx))
            objects.append(_make_object("component", color, tuple(sorted(cells)), changed))
    return objects


def _color_region_objects(arr: np.ndarray, changed: np.ndarray) -> list[ObjectSlot]:
    bg = _background_value(arr)
    objects: list[ObjectSlot] = []
    for color in sorted(int(item) for item in np.unique(arr) if int(item) != bg):
        cells = tuple(sorted((int(y), int(x)) for y, x in zip(*np.where(arr == color))))
        if cells:
            objects.append(_make_object("color_region", color, cells, changed))
    return objects


def _make_object(kind: str, color: int, cells: tuple[Cell, ...], changed: np.ndarray) -> ObjectSlot:
    bbox = _bbox(cells)
    canon = _canonical_cells(cells)
    changed_count = int(sum(1 for y, x in cells if bool(changed[y, x])))
    object_id = _stable_hash({"kind": kind, "color": color, "cells": cells}, length=12)
    canonical_hash = _stable_hash({"kind": kind, "color": color, "canonical_cells": canon}, length=12)
    area = len(cells)
    salience = float(area + changed_count * 3 + _hole_count(cells) * 2)
    return ObjectSlot(
        object_id=object_id,
        canonical_hash=canonical_hash,
        kind=kind,
        color=int(color),
        cells=cells,
        bbox=bbox,
        centroid=_centroid(cells),
        area=area,
        holes=_hole_count(cells),
        canonical_cells=canon,
        boundary_cells=_boundary_cells(cells),
        changed_count=changed_count,
        symmetry=_symmetry(cells),
        salience=salience,
    )


def _touching(a: ObjectSlot, b: ObjectSlot, shape: tuple[int, int]) -> bool:
    b_cells = set(b.cells)
    height, width = shape
    return any(neighbor in b_cells for y, x in a.cells for neighbor in _neighbors8(y, x, height, width))


def _contains(a: ObjectSlot, b: ObjectSlot) -> bool:
    atop, aleft, abottom, aright = a.bbox
    btop, bleft, bbottom, bright = b.bbox
    return atop < btop and aleft < bleft and abottom > bbottom and aright > bright


def _aligned(a: ObjectSlot, b: ObjectSlot) -> list[str]:
    rels: list[str] = []
    if a.bbox[0] == b.bbox[0] or a.bbox[2] == b.bbox[2] or round(a.centroid[0], 6) == round(b.centroid[0], 6):
        rels.append("row_aligned")
    if a.bbox[1] == b.bbox[1] or a.bbox[3] == b.bbox[3] or round(a.centroid[1], 6) == round(b.centroid[1], 6):
        rels.append("col_aligned")
    return rels


def _relation_graph(objects: tuple[ObjectSlot, ...], shape: tuple[int, int]) -> tuple[Relation, ...]:
    relations: list[Relation] = []
    components = [obj for obj in objects if obj.kind == "component"]
    for obj in components:
        for axis, enabled in sorted(obj.symmetry.items()):
            if enabled:
                relations.append(Relation(obj.object_id, obj.object_id, "symmetry", {"axis": axis}))
        if obj.holes:
            relations.append(Relation(obj.object_id, obj.object_id, "has_hole", {"count": obj.holes}))
    for i, left in enumerate(components):
        for right in components[i + 1 :]:
            if left.color == right.color:
                relations.append(Relation(left.object_id, right.object_id, "same_color", {"color": left.color}))
            if _touching(left, right, shape):
                relations.append(Relation(left.object_id, right.object_id, "adjacent", {}))
            if _contains(left, right):
                relations.append(Relation(left.object_id, right.object_id, "contains", {}))
            if _contains(right, left):
                relations.append(Relation(right.object_id, left.object_id, "contains", {}))
            for rel in _aligned(left, right):
                relations.append(Relation(left.object_id, right.object_id, rel, {}))
            if left.canonical_cells == right.canonical_cells and left.color == right.color:
                dy = round(float(right.centroid[0] - left.centroid[0]), 4)
                dx = round(float(right.centroid[1] - left.centroid[1]), 4)
                relations.append(Relation(left.object_id, right.object_id, "repetition", {"delta": [dy, dx]}))
    relations.sort(key=lambda item: (item.source, item.relation, item.target, json.dumps(item.detail, sort_keys=True)))
    return tuple(relations)


def _embedding(arr: np.ndarray) -> tuple[float, ...]:
    height, width = arr.shape
    counts = Counter(int(item) for item in arr.reshape(-1))
    total = float(arr.size)
    colors = sorted(counts)
    histogram = [float(counts[color] / total) for color in colors[:16]]
    if len(histogram) < 16:
        histogram.extend([0.0] * (16 - len(histogram)))
    return tuple([float(height) / 64.0, float(width) / 64.0, float(len(colors)) / 16.0, float(arr.mean() / 16.0)] + histogram)


def _binding_vector(key: str, dim: int = 16) -> tuple[int, ...]:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    return tuple(1 if digest[index % len(digest)] >= 128 else -1 for index in range(dim))


def _bindings(
    objects: tuple[ObjectSlot, ...],
    relations: tuple[Relation, ...],
    *,
    action: Any,
    time_index: int,
) -> tuple[Binding, ...]:
    keys: list[str] = [f"time:{int(time_index)}"]
    if action is not None:
        keys.append(f"action:{action}")
    for obj in objects:
        if obj.kind != "component":
            continue
        keys.extend(
            [
                f"{obj.object_id}:color:{obj.color}",
                f"{obj.object_id}:bbox:{obj.bbox}",
                f"{obj.object_id}:area:{obj.area}",
                f"{obj.object_id}:centroid:{round(obj.centroid[0], 3)}:{round(obj.centroid[1], 3)}",
            ]
        )
    for rel in relations:
        keys.append(f"relation:{rel.source}:{rel.relation}:{rel.target}:{json.dumps(rel.detail, sort_keys=True)}")
    return tuple(Binding(key=key, vector=_binding_vector(key)) for key in sorted(keys))


def _edit_script(previous: np.ndarray | None, current: np.ndarray) -> EditScript:
    if previous is None:
        return EditScript(0, int(current.size), tuple(), tuple(), tuple(), tuple())
    prev = _to_grid(previous)
    if prev.shape != current.shape:
        raise ValueError(f"previous/current grid shapes differ: {prev.shape} vs {current.shape}")
    changed = prev != current
    edits = []
    for y, x in zip(*np.where(changed)):
        edits.append({"cell": [int(y), int(x)], "before": int(prev[y, x]), "after": int(current[y, x])})
    prev_bg = _background_value(prev)
    cur_bg = _background_value(current)
    appears = tuple(edit for edit in edits if int(edit["before"]) == prev_bg and int(edit["after"]) != cur_bg)
    disappears = tuple(edit for edit in edits if int(edit["before"]) != prev_bg and int(edit["after"]) == cur_bg)
    moves = tuple(_detect_moves(prev, current))
    return EditScript(
        changed_count=int(np.count_nonzero(changed)),
        unchanged_count=int(current.size - np.count_nonzero(changed)),
        cell_edits=tuple(edits),
        appears=appears,
        disappears=disappears,
        moves=moves,
    )


def _object_signature(obj: ObjectSlot) -> tuple[int, tuple[Cell, ...], int]:
    return (obj.color, obj.canonical_cells, obj.area)


def _detect_moves(previous: np.ndarray, current: np.ndarray) -> list[dict[str, Any]]:
    empty = np.zeros(previous.shape, dtype=bool)
    prev_objects = _component_objects(previous, empty)
    cur_objects = _component_objects(current, empty)
    by_signature: dict[tuple[int, tuple[Cell, ...], int], list[ObjectSlot]] = defaultdict(list)
    for obj in cur_objects:
        by_signature[_object_signature(obj)].append(obj)
    moves: list[dict[str, Any]] = []
    used: set[str] = set()
    for prev_obj in prev_objects:
        candidates = by_signature.get(_object_signature(prev_obj), [])
        candidates = [candidate for candidate in candidates if candidate.object_id not in used]
        if not candidates:
            continue
        candidate = min(
            candidates,
            key=lambda item: abs(item.centroid[0] - prev_obj.centroid[0]) + abs(item.centroid[1] - prev_obj.centroid[1]),
        )
        if candidate.cells != prev_obj.cells:
            used.add(candidate.object_id)
            moves.append(
                {
                    "object_id": prev_obj.object_id,
                    "color": prev_obj.color,
                    "from_centroid": [round(prev_obj.centroid[0], 4), round(prev_obj.centroid[1], 4)],
                    "to_centroid": [round(candidate.centroid[0], 4), round(candidate.centroid[1], 4)],
                    "delta": [
                        round(float(candidate.centroid[0] - prev_obj.centroid[0]), 4),
                        round(float(candidate.centroid[1] - prev_obj.centroid[1]), 4),
                    ],
                }
            )
    return moves


def _salience(
    arr: np.ndarray,
    objects: tuple[ObjectSlot, ...],
    changed: np.ndarray,
) -> tuple[tuple[float, ...], ...]:
    salience = changed.astype(np.float32) * 1.0
    for obj in objects:
        if obj.kind != "component":
            continue
        cy, cx = int(round(obj.centroid[0])), int(round(obj.centroid[1]))
        salience[cy, cx] += 0.40
        for y, x in obj.boundary_cells:
            salience[y, x] += 0.15
        if obj.holes:
            top, left, bottom, right = obj.bbox
            salience[top : bottom + 1, left : right + 1] += 0.05
    max_value = float(salience.max()) if salience.size else 0.0
    if max_value > 0:
        salience = salience / max_value
    return tuple(tuple(float(item) for item in row) for row in salience)


def _active_cells(
    objects: tuple[ObjectSlot, ...],
    changed: np.ndarray,
    salience_map: tuple[tuple[float, ...], ...],
) -> tuple[Cell, ...]:
    cells: set[Cell] = set((int(y), int(x)) for y, x in zip(*np.where(changed)))
    for obj in objects:
        if obj.kind != "component":
            continue
        cells.add((int(round(obj.centroid[0])), int(round(obj.centroid[1]))))
        cells.update(obj.boundary_cells)
    threshold = 0.75
    for y, row in enumerate(salience_map):
        for x, value in enumerate(row):
            if value >= threshold:
                cells.add((int(y), int(x)))
    return tuple(sorted(cells))


def parse_grid(
    grid: Any,
    previous_grid: Any | None = None,
    *,
    action: Any = None,
    time_index: int = 0,
) -> PerceptualScene:
    arr = _to_grid(grid)
    prev = None if previous_grid is None else _to_grid(previous_grid)
    if prev is not None and prev.shape != arr.shape:
        raise ValueError(f"previous/current grid shapes differ: {prev.shape} vs {arr.shape}")
    changed = np.zeros(arr.shape, dtype=bool) if prev is None else prev != arr
    components = _component_objects(arr, changed)
    color_regions = _color_region_objects(arr, changed)
    objects = tuple(sorted(components + color_regions, key=lambda obj: (obj.kind, obj.color, obj.bbox, obj.object_id)))
    relations = _relation_graph(objects, tuple(int(item) for item in arr.shape))
    salience_map = _salience(arr, objects, changed)
    return PerceptualScene(
        raw_hash=raw_grid_hash(arr),
        shape=tuple(int(item) for item in arr.shape),
        background=_background_value(arr),
        grid_embedding=_embedding(arr),
        objects=objects,
        relation_graph=relations,
        bindings=_bindings(objects, relations, action=action, time_index=time_index),
        salience_map=salience_map,
        active_cells=_active_cells(objects, changed, salience_map),
        edit_script=_edit_script(prev, arr),
    )
