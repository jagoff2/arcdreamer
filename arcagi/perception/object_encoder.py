from __future__ import annotations

from collections import deque

import numpy as np

from arcagi.core.types import GridObservation, ObjectState, Relation, StructuredState


def extract_structured_state(observation: GridObservation) -> StructuredState:
    grid = observation.grid
    height, width = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    object_specs: list[tuple[int, tuple[tuple[int, int], ...], tuple[int, int, int, int], tuple[float, float], int, tuple[str, ...]]] = []
    cell_tags = observation.extras.get("cell_tags", {})
    background_color = int(observation.extras.get("background_color", 0))

    for y in range(height):
        for x in range(width):
            color = int(grid[y, x])
            if color == background_color or visited[y, x]:
                continue
            cells = _component(grid, visited, y, x, color)
            ys = [cell[0] for cell in cells]
            xs = [cell[1] for cell in cells]
            bbox = (min(ys), min(xs), max(ys), max(xs))
            centroid = (float(sum(ys)) / len(ys), float(sum(xs)) / len(xs))
            sorted_cells = tuple(sorted(cells))
            object_specs.append(
                (
                    color,
                    sorted_cells,
                    bbox,
                    centroid,
                    len(cells),
                    _component_tags(cells, cell_tags),
                )
            )
    object_specs.sort(key=lambda item: (item[0], item[4], item[2], item[5], item[1]))
    objects = tuple(
        ObjectState(
            object_id=f"obj_{index}",
            color=color,
            cells=cells,
            bbox=bbox,
            centroid=centroid,
            area=area,
            tags=tags,
        )
        for index, (color, cells, bbox, centroid, area, tags) in enumerate(object_specs)
    )

    relations = _build_relations(objects)
    inventory = tuple(sorted((str(key), str(value)) for key, value in observation.extras.get("inventory", {}).items()))
    flags = tuple(sorted((str(key), str(value)) for key, value in observation.extras.get("flags", {}).items()))
    action_roles = tuple(
        sorted(
            (str(action), str(role))
            for action, role in observation.extras.get("action_roles", {}).items()
        )
    )
    grid_signature = tuple(int(value) for value in grid.flatten())
    return StructuredState(
        task_id=observation.task_id,
        episode_id=observation.episode_id,
        step_index=observation.step_index,
        grid_shape=(height, width),
        grid_signature=grid_signature,
        objects=objects,
        relations=tuple(relations),
        affordances=observation.available_actions,
        action_roles=action_roles,
        inventory=inventory,
        flags=flags,
    )


def _component_tags(
    cells: list[tuple[int, int]],
    cell_tags: dict[tuple[int, int], tuple[str, ...]],
) -> tuple[str, ...]:
    tags: set[str] = set()
    for cell in cells:
        tags.update(cell_tags.get(cell, ()))
    return tuple(sorted(tags))


def _component(
    grid: np.ndarray,
    visited: np.ndarray,
    start_y: int,
    start_x: int,
    color: int,
) -> list[tuple[int, int]]:
    queue: deque[tuple[int, int]] = deque([(start_y, start_x)])
    visited[start_y, start_x] = True
    cells: list[tuple[int, int]] = []
    height, width = grid.shape
    while queue:
        y, x = queue.popleft()
        cells.append((y, x))
        for dy, dx in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            ny, nx = y + dy, x + dx
            if ny < 0 or nx < 0 or ny >= height or nx >= width:
                continue
            if visited[ny, nx] or int(grid[ny, nx]) != color:
                continue
            visited[ny, nx] = True
            queue.append((ny, nx))
    return cells


def _build_relations(objects: tuple[ObjectState, ...]) -> list[Relation]:
    relations: list[Relation] = []
    for source in objects:
        for target in objects:
            if source.object_id == target.object_id:
                continue
            dy = target.centroid[0] - source.centroid[0]
            dx = target.centroid[1] - source.centroid[1]
            manhattan = abs(dy) + abs(dx)
            if manhattan <= 3.0:
                relations.append(
                    Relation(
                        relation_type="near",
                        source_id=source.object_id,
                        target_id=target.object_id,
                        value=float(manhattan),
                    )
                )
            if dy < 0:
                relations.append(
                    Relation(
                        relation_type="above",
                        source_id=source.object_id,
                        target_id=target.object_id,
                        value=float(abs(dy)),
                    )
                )
            if dx < 0:
                relations.append(
                    Relation(
                        relation_type="left_of",
                        source_id=source.object_id,
                        target_id=target.object_id,
                        value=float(abs(dx)),
                    )
                )
    return relations
