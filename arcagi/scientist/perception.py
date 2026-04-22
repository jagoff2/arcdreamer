"""Object-centric grid perception and transition differencing.

This module replaces brittle per-game perception with a cheap, generic object
extractor.  It builds compact symbolic states from raw grids, computes spatial
relations, and compares consecutive states to supply causal evidence to the
hypothesis engine.
"""

from __future__ import annotations

from collections import Counter, deque
from hashlib import blake2b
from typing import Any, Iterable, Mapping

import numpy as np

from .types import (
    ActionName,
    GridFrame,
    ObjectMotion,
    ObjectToken,
    RelationToken,
    StructuredState,
    TransitionDelta,
    TransitionRecord,
    action_target_to_grid_cell,
    coerce_grid_frame,
)


def _digest(parts: Iterable[str], *, size: int = 10) -> str:
    h = blake2b(digest_size=size)
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    if len(values) == 0:
        return 0
    return int(values[int(np.argmax(counts))])


def _shape_hash(cells: tuple[tuple[int, int], ...]) -> str:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return _digest((f"{r},{c}" for r, c in norm), size=8)


def _component_role_tags(
    *,
    color: int,
    area: int,
    bbox: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
    dominant_color: int,
) -> tuple[str, ...]:
    tags: list[str] = []
    rows, cols = grid_shape
    r0, c0, r1, c1 = bbox
    grid_area = max(rows * cols, 1)
    if color == dominant_color and area > 0.35 * grid_area:
        tags.append("background_candidate")
    if r0 == 0 or c0 == 0 or r1 == rows - 1 or c1 == cols - 1:
        tags.append("boundary_touching")
    if area == 1:
        tags.append("point")
    if (r1 - r0 + 1) == 1 or (c1 - c0 + 1) == 1:
        tags.append("line_like")
    if area <= 4:
        tags.append("small")
    if area >= 0.10 * grid_area:
        tags.append("large")
    return tuple(tags)


def connected_components(grid: np.ndarray, *, include_dominant: bool = False) -> tuple[ObjectToken, ...]:
    """Segment same-color 4-connected components into object tokens."""

    grid = np.asarray(grid, dtype=np.int64)
    if grid.ndim != 2:
        raise ValueError(f"expected rank-2 grid, got {grid.shape!r}")
    rows, cols = grid.shape
    dominant = _dominant_color(grid)
    visited = np.zeros_like(grid, dtype=bool)
    objects: list[ObjectToken] = []
    color_counts: Counter[int] = Counter(int(x) for x in grid.ravel())
    dominant_area = color_counts[dominant]

    object_index = 0
    for start_r in range(rows):
        for start_c in range(cols):
            if visited[start_r, start_c]:
                continue
            color = int(grid[start_r, start_c])
            q: deque[tuple[int, int]] = deque([(start_r, start_c)])
            visited[start_r, start_c] = True
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            cell_tuple = tuple(sorted(cells))
            area = len(cell_tuple)
            # Drop the main background slab, but keep small components that happen to
            # share the dominant color.  ARC-like games often use color zero as empty
            # space, but relying only on color zero is too brittle for ARC-AGI-3.
            if (
                not include_dominant
                and color == dominant
                and area == dominant_area
                and area > 0.35 * rows * cols
            ):
                continue
            min_r = min(r for r, _ in cell_tuple)
            max_r = max(r for r, _ in cell_tuple)
            min_c = min(c for _, c in cell_tuple)
            max_c = max(c for _, c in cell_tuple)
            centroid = (
                float(sum(r for r, _ in cell_tuple)) / area,
                float(sum(c for _, c in cell_tuple)) / area,
            )
            bbox = (min_r, min_c, max_r, max_c)
            shape = _shape_hash(cell_tuple)
            tags = _component_role_tags(
                color=color,
                area=area,
                bbox=bbox,
                grid_shape=(rows, cols),
                dominant_color=dominant,
            )
            object_id = f"o{object_index:03d}_c{color}_{shape[:6]}_{min_r}_{min_c}"
            objects.append(
                ObjectToken(
                    object_id=object_id,
                    color=color,
                    area=area,
                    bbox=bbox,
                    centroid=centroid,
                    cells=cell_tuple,
                    shape_hash=shape,
                    role_tags=tags,
                )
            )
            object_index += 1
    return tuple(objects)


def _bbox_touching(a: ObjectToken, b: ObjectToken, *, margin: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a.bbox
    br0, bc0, br1, bc1 = b.bbox
    return not (ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0)


def spatial_relations(objects: tuple[ObjectToken, ...], *, max_pairs: int = 160) -> tuple[RelationToken, ...]:
    relations: list[RelationToken] = []
    pair_count = 0
    for a in objects:
        for b in objects:
            if a.object_id == b.object_id:
                continue
            pair_count += 1
            if pair_count > max_pairs:
                return tuple(relations)
            if a.color == b.color:
                relations.append(RelationToken("same_color", a.object_id, b.object_id))
            dr = b.centroid[0] - a.centroid[0]
            dc = b.centroid[1] - a.centroid[1]
            if abs(dc) >= abs(dr) and dc > 0:
                relations.append(RelationToken("left_of", a.object_id, b.object_id, float(abs(dc))))
            if abs(dc) >= abs(dr) and dc < 0:
                relations.append(RelationToken("right_of", a.object_id, b.object_id, float(abs(dc))))
            if abs(dr) > abs(dc) and dr > 0:
                relations.append(RelationToken("above", a.object_id, b.object_id, float(abs(dr))))
            if abs(dr) > abs(dc) and dr < 0:
                relations.append(RelationToken("below", a.object_id, b.object_id, float(abs(dr))))
            if _bbox_touching(a, b, margin=1):
                relations.append(RelationToken("near_or_touching", a.object_id, b.object_id, float(abs(dr) + abs(dc))))
    return tuple(relations)


def abstract_fingerprint(objects: tuple[ObjectToken, ...], relations: tuple[RelationToken, ...], grid_shape: tuple[int, int]) -> str:
    object_parts = sorted(obj.signature for obj in objects)
    rel_parts = sorted(rel.relation for rel in relations)
    parts = [f"shape:{grid_shape[0]}x{grid_shape[1]}", *object_parts, *rel_parts[:64]]
    return _digest(parts, size=16)


def extract_state(observation: Any, *, include_dominant: bool = False) -> StructuredState:
    frame = coerce_grid_frame(observation)
    objects = connected_components(frame.grid, include_dominant=include_dominant)
    relations = spatial_relations(objects)
    dominant = _dominant_color(frame.grid)
    abstract = abstract_fingerprint(objects, relations, frame.shape)
    return StructuredState(
        frame=frame,
        objects=objects,
        relations=relations,
        dominant_color=dominant,
        abstract_fingerprint=abstract,
        exact_fingerprint=frame.fingerprint,
        metadata={"object_count": len(objects), "relation_count": len(relations)},
    )


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    inter_r0 = max(ar0, br0)
    inter_c0 = max(ac0, bc0)
    inter_r1 = min(ar1, br1)
    inter_c1 = min(ac1, bc1)
    if inter_r1 < inter_r0 or inter_c1 < inter_c0:
        return 0.0
    inter = (inter_r1 - inter_r0 + 1) * (inter_c1 - inter_c0 + 1)
    area_a = (ar1 - ar0 + 1) * (ac1 - ac0 + 1)
    area_b = (br1 - br0 + 1) * (bc1 - bc0 + 1)
    return float(inter) / float(max(area_a + area_b - inter, 1))


def _match_objects(before: tuple[ObjectToken, ...], after: tuple[ObjectToken, ...]) -> dict[str, str]:
    candidates: list[tuple[float, str, str]] = []
    for b in before:
        for a in after:
            color_score = 2.0 if b.color == a.color else -1.0
            shape_score = 2.0 if b.shape_hash == a.shape_hash else 0.0
            area_score = 1.0 / (1.0 + abs(b.area - a.area))
            dist = abs(b.centroid[0] - a.centroid[0]) + abs(b.centroid[1] - a.centroid[1])
            iou = _bbox_iou(b.bbox, a.bbox)
            score = color_score + shape_score + area_score + iou - 0.12 * dist
            candidates.append((score, b.object_id, a.object_id))
    candidates.sort(reverse=True)
    mapping: dict[str, str] = {}
    used_after: set[str] = set()
    for score, before_id, after_id in candidates:
        if score < 0.25:
            continue
        if before_id in mapping or after_id in used_after:
            continue
        mapping[before_id] = after_id
        used_after.add(after_id)
    return mapping


def _objects_near_cells(objects: tuple[ObjectToken, ...], cells: Iterable[tuple[int, int]], *, radius: int = 1) -> tuple[str, ...]:
    interesting = tuple(cells)
    if not interesting:
        return ()
    touched: set[str] = set()
    for obj in objects:
        for r, c in interesting:
            if any(abs(orow - r) + abs(ocol - c) <= radius for orow, ocol in obj.cells[:256]):
                touched.add(obj.object_id)
                break
    return tuple(sorted(touched))


def compare_states(
    before: StructuredState,
    after: StructuredState,
    *,
    action: ActionName,
    reward: float = 0.0,
    score_delta: float = 0.0,
    terminated: bool = False,
    info: Mapping[str, Any] | None = None,
) -> TransitionRecord:
    grid_before = before.grid
    grid_after = after.grid
    if grid_before.shape == grid_after.shape:
        changed_mask = grid_before != grid_after
        changed_cells = int(np.count_nonzero(changed_mask))
        changed_coords = tuple((int(r), int(c)) for r, c in np.argwhere(changed_mask)[:512])
        changed_fraction = float(changed_cells) / float(max(grid_before.size, 1))
    else:
        changed_cells = int(max(grid_before.size, grid_after.size))
        changed_fraction = 1.0
        changed_coords = ()

    match = _match_objects(before.objects, after.objects)
    after_by_id = {obj.object_id: obj for obj in after.objects}
    before_by_id = {obj.object_id: obj for obj in before.objects}
    moved: list[ObjectMotion] = []
    for before_id, after_id in match.items():
        b = before_by_id[before_id]
        a = after_by_id[after_id]
        dr = int(round(a.centroid[0] - b.centroid[0]))
        dc = int(round(a.centroid[1] - b.centroid[1]))
        distance = abs(a.centroid[0] - b.centroid[0]) + abs(a.centroid[1] - b.centroid[1])
        if distance > 0.10 or b.bbox != a.bbox:
            moved.append(
                ObjectMotion(
                    before_id=before_id,
                    after_id=after_id,
                    before_signature=b.signature,
                    after_signature=a.signature,
                    color=b.color,
                    delta=(dr, dc),
                    distance=float(distance),
                )
            )
    appeared = tuple(sorted(set(after_by_id) - set(match.values())))
    disappeared = tuple(sorted(set(before_by_id) - set(match.keys())))

    target = action_target_to_grid_cell(action, before.frame.extras)
    touched_cells: list[tuple[int, int]] = list(changed_coords[:128])
    if target is not None:
        touched_cells.append(target)
    touched = sorted(
        set(_objects_near_cells(before.objects, touched_cells, radius=1))
        | set(_objects_near_cells(after.objects, touched_cells, radius=1))
    )

    delta = TransitionDelta(
        action=action,
        changed_cells=changed_cells,
        changed_fraction=changed_fraction,
        moved_objects=tuple(moved),
        appeared=appeared,
        disappeared=disappeared,
        touched_objects=tuple(touched),
        reward=float(reward),
        score_delta=float(score_delta),
        terminated=bool(terminated),
        info=info or {},
    )
    return TransitionRecord(before=before, after=after, delta=delta)
