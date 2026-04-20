from __future__ import annotations

from collections import deque

import numpy as np

from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.types import GridObservation, ObjectState, Relation, StructuredState


def extract_structured_state(observation: GridObservation) -> StructuredState:
    grid = observation.grid
    height, width = grid.shape
    visited = np.zeros_like(grid, dtype=bool)
    object_specs: list[tuple[int, tuple[tuple[int, int], ...], tuple[int, int, int, int], tuple[float, float], int, tuple[str, ...]]] = []
    action_roles_dict = {str(action): str(role) for action, role in observation.extras.get("action_roles", {}).items()}
    inferred_inventory, inferred_flags = _inferred_interface_state(observation.available_actions, action_roles_dict)
    inventory_dict = {**inferred_inventory, **{str(key): str(value) for key, value in observation.extras.get("inventory", {}).items()}}
    flags_dict = {**inferred_flags, **{str(key): str(value) for key, value in observation.extras.get("flags", {}).items()}}
    cell_tags = _merge_cell_tags(
        observation.extras.get("cell_tags", {}),
        _inferred_cell_tags(grid.shape, observation.available_actions, action_roles_dict),
    )
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
    inventory = tuple(sorted(inventory_dict.items()))
    flags = tuple(sorted(flags_dict.items()))
    action_roles = tuple(
        sorted(
            (str(action), str(role))
            for action, role in action_roles_dict.items()
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


def _inferred_interface_state(
    affordances: tuple[str, ...],
    action_roles: dict[str, str],
) -> tuple[dict[str, str], dict[str, str]]:
    context = build_action_schema_context(affordances, action_roles)
    counts: dict[str, int] = {key: 0 for key in ("move", "click", "select", "interact", "undo", "wait", "raw", "other")}
    click_bins: set[tuple[int, int]] = set()
    for action in affordances:
        schema = build_action_schema(action, context)
        counts[schema.action_type if schema.action_type in counts else "other"] += 1
        if schema.coarse_bin is not None:
            click_bins.add(schema.coarse_bin)
    inventory = {
        "interface_move_actions": str(counts["move"]),
        "interface_click_actions": str(counts["click"]),
        "interface_select_actions": str(counts["select"]),
        "interface_interact_actions": str(counts["interact"]),
        "interface_undo_actions": str(counts["undo"]),
        "interface_raw_actions": str(counts["raw"]),
        "interface_click_bin_count": str(len(click_bins)),
    }
    flags = {
        "interface_has_click": "1" if counts["click"] > 0 else "0",
        "interface_has_select": "1" if counts["select"] > 0 else "0",
        "interface_has_interact": "1" if counts["interact"] > 0 else "0",
        "interface_has_undo": "1" if counts["undo"] > 0 else "0",
        "interface_has_mode_actions": "1" if (counts["click"] + counts["select"]) > 0 else "0",
        "interface_parametric_clicks": "1" if any(action.startswith("click:") for action in affordances) else "0",
    }
    return inventory, flags


def _inferred_cell_tags(
    grid_shape: tuple[int, int],
    affordances: tuple[str, ...],
    action_roles: dict[str, str],
) -> dict[tuple[int, int], tuple[str, ...]]:
    height, width = grid_shape
    context = build_action_schema_context(affordances, action_roles)
    selector_count = sum(
        1 for action in affordances if build_action_schema(action, context).action_type in {"click", "select"}
    )
    tags_by_cell: dict[tuple[int, int], set[str]] = {}
    for action in affordances:
        schema = build_action_schema(action, context)
        if schema.action_type != "click" or schema.click is None:
            continue
        grid_x, grid_y = schema.click
        if grid_y < 0 or grid_x < 0 or grid_y >= height or grid_x >= width:
            continue
        tags = tags_by_cell.setdefault((grid_y, grid_x), set())
        tags.add("clickable")
        tags.add("interface_target")
        if selector_count >= 2:
            tags.add("selector_candidate")
        if schema.coarse_bin is not None:
            tags.add(f"click_bin_{schema.coarse_bin[0]}_{schema.coarse_bin[1]}")
    return {cell: tuple(sorted(tags)) for cell, tags in tags_by_cell.items()}


def _merge_cell_tags(
    primary: dict[tuple[int, int], tuple[str, ...]] | dict[tuple[int, int], list[str]] | None,
    secondary: dict[tuple[int, int], tuple[str, ...]] | None,
) -> dict[tuple[int, int], tuple[str, ...]]:
    merged: dict[tuple[int, int], set[str]] = {}
    for source in (secondary or {}, primary or {}):
        for cell, tags in source.items():
            merged.setdefault(cell, set()).update(str(tag) for tag in tags)
    return {cell: tuple(sorted(tags)) for cell, tags in merged.items()}
