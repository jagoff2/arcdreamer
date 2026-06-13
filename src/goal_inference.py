from __future__ import annotations

import json
import math
from typing import Any, Mapping

import torch

from .grid_perception import parse_grid


def fresh_goal_posterior() -> dict[str, Any]:
    return {
        "schema": "runtime_goal_posterior_v1",
        "updates": 0,
        "goals": {},
        "subgoals": {},
        "posterior_mass": 0.0,
        "top_goals": [],
    }


def _tensor_from_snapshot(snapshot: Any) -> torch.Tensor | None:
    if not isinstance(snapshot, Mapping) or "values" not in snapshot:
        return None
    try:
        return torch.as_tensor(snapshot["values"])
    except (TypeError, ValueError):
        return None


def _grid_from_observation_snapshot(snapshot: Mapping[str, Any]) -> torch.Tensor | None:
    if "grid" in snapshot:
        return _tensor_from_snapshot(snapshot["grid"])
    if "sensory" in snapshot:
        tensor = _tensor_from_snapshot(snapshot["sensory"])
        if tensor is not None and tensor.ndim == 2 and not torch.is_floating_point(tensor):
            return tensor
    return None


def _events(event: Mapping[str, Any]) -> list[str]:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return []
    values = metadata.get("events", [])
    if isinstance(values, (list, tuple)):
        return [str(item) for item in values]
    if isinstance(values, str):
        return [values]
    return []


def _score_delta(event: Mapping[str, Any]) -> float:
    metadata = event.get("metadata", {})
    if isinstance(metadata, Mapping):
        return float(metadata.get("score_delta", 0.0))
    return 0.0


def _terminal(event: Mapping[str, Any]) -> bool:
    metadata = event.get("metadata", {})
    return bool(isinstance(metadata, Mapping) and metadata.get("terminal", False))


def _description_length(candidate: Mapping[str, Any]) -> float:
    kind = str(candidate.get("kind", ""))
    base = {
        "positive_reward": 1.0,
        "terminal_win": 1.2,
        "event": 1.4,
        "object_disappears": 1.8,
        "object_appears": 1.8,
        "alignment": 2.0,
        "clearing": 1.6,
        "filling": 1.7,
        "matching": 2.1,
        "height_equality": 2.0,
        "area_equality": 2.1,
        "sorting": 2.2,
        "containment": 2.0,
        "transformation_closure": 2.4,
        "movement_subgoal": 1.4,
        "target_changed_cell": 1.5,
        "controllable_object": 1.7,
        "target_object": 1.8,
        "counter_tracking": 1.6,
        "constraint_equality": 1.9,
        "relation_constraint": 2.0,
        "avoid_no_op": 1.2,
        "reversible_hint": 1.5,
        "gate_or_counter": 1.6,
    }.get(kind, 2.0)
    return float(base + 0.20 * len(candidate.get("selector", {})) + 0.20 * len(candidate.get("progress_test", {})))


def _candidate_id(candidate: Mapping[str, Any]) -> str:
    scope = str(candidate.get("scope", "goal"))
    kind = str(candidate.get("kind"))
    selector = candidate.get("selector", {})
    if kind in {"alignment", "matching", "positive_reward", "terminal_win", "clearing", "filling"}:
        return f"{scope}|{kind}"
    if "event" in selector:
        return f"{scope}|{kind}|event:{selector['event']}"
    if "color" in selector:
        return f"{scope}|{kind}|color:{selector['color']}"
    if "field" in selector:
        return f"{scope}|{kind}|field:{selector['field']}"
    if "height" in selector:
        return f"{scope}|{kind}|height:{selector['height']}"
    if "area" in selector:
        return f"{scope}|{kind}|area:{selector['area']}"
    if "relation" in selector:
        return f"{scope}|{kind}|relation:{selector['relation']}"
    if selector:
        selector_key = json.dumps(selector, sort_keys=True, separators=(",", ":"))
        return f"{scope}|{kind}|selector:{selector_key}"
    return f"{scope}|{kind}"


def _grid_scene(event: Mapping[str, Any]):
    before = event.get("observation", {})
    after = event.get("next_observation", {})
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return None
    before_grid = _grid_from_observation_snapshot(before)
    after_grid = _grid_from_observation_snapshot(after)
    if before_grid is None or after_grid is None or before_grid.ndim != 2 or after_grid.ndim != 2:
        return None
    try:
        return parse_grid(after_grid.numpy(), previous_grid=before_grid.numpy(), action=event.get("action"))
    except ValueError:
        return None


def _before_grid_scene(event: Mapping[str, Any]):
    before = event.get("observation", {})
    if not isinstance(before, Mapping):
        return None
    before_grid = _grid_from_observation_snapshot(before)
    if before_grid is None or before_grid.ndim != 2:
        return None
    try:
        return parse_grid(before_grid.numpy(), action=event.get("action"))
    except ValueError:
        return None


def _component_objects(scene: Any) -> list[Any]:
    return [obj for obj in scene.objects if getattr(obj, "kind", "") == "component"]


def _value_counts(values: list[int]) -> dict[int, int]:
    counts: dict[int, int] = {}
    for value in values:
        counts[int(value)] = int(counts.get(int(value), 0)) + 1
    return counts


def _object_descriptor(obj: Any) -> dict[str, Any]:
    return {
        "canonical_hash": str(getattr(obj, "canonical_hash", "unknown")),
        "color": int(getattr(obj, "color", -1)),
        "area": int(getattr(obj, "area", 0)),
    }


def _object_key(descriptor: Mapping[str, Any]) -> str:
    return "object|" + "|".join(f"{name}:{descriptor[name]}" for name in sorted(descriptor))


def _metadata_counter_values(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return []
    counters: list[dict[str, Any]] = []

    def visit(prefix: str, value: Any) -> None:
        lower = prefix.lower()
        if isinstance(value, Mapping):
            for key, item in value.items():
                child = f"{prefix}.{key}" if prefix else str(key)
                visit(child, item)
            return
        if isinstance(value, bool):
            return
        if isinstance(value, (int, float)) and any(
            marker in lower for marker in ("counter", "count", "keys", "resource", "inventory")
        ):
            counters.append({"field": prefix, "value": float(value)})

    visit("", metadata)
    for item in _events(event):
        lower = item.lower()
        if any(marker in lower for marker in ("counter", "count", "key", "resource")):
            counters.append({"field": "event", "event": item})
    return counters


def _delta_counter_fields(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    delta = event.get("delta", {})
    if not isinstance(delta, Mapping):
        return []
    candidates: list[dict[str, Any]] = []
    for field, item in sorted(delta.items()):
        lower = str(field).lower()
        if not any(marker in lower for marker in ("counter", "count", "key", "resource", "inventory")):
            continue
        if isinstance(item, Mapping):
            changed = bool(item.get("changed", False)) or int(item.get("changed_count", 0) or 0) > 0
            magnitude = float(item.get("l1", 0.0) or 0.0)
        else:
            changed = bool(item)
            magnitude = 1.0 if changed else 0.0
        if changed:
            candidates.append({"field": str(field), "delta_l1": magnitude})
    return candidates


def _grid_background(grid: torch.Tensor) -> int:
    values, counts = torch.unique(grid.reshape(-1).to(dtype=torch.long), return_counts=True)
    index = int(torch.argmax(counts).item())
    return int(values[index].item())


def _color_transform_mapping(event: Mapping[str, Any]) -> dict[str, int]:
    before = event.get("observation", {})
    after = event.get("next_observation", {})
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return {}
    before_grid = _grid_from_observation_snapshot(before)
    after_grid = _grid_from_observation_snapshot(after)
    if before_grid is None or after_grid is None or before_grid.shape != after_grid.shape:
        return {}
    before_grid = before_grid.to(dtype=torch.long)
    after_grid = after_grid.to(dtype=torch.long)
    before_bg = _grid_background(before_grid)
    after_bg = _grid_background(after_grid)
    mapping: dict[int, int] = {}
    for before_value, after_value in zip(before_grid.reshape(-1).tolist(), after_grid.reshape(-1).tolist()):
        before_color = int(before_value)
        after_color = int(after_value)
        if before_color == after_color:
            continue
        if before_color == before_bg or after_color == after_bg:
            continue
        existing = mapping.get(before_color)
        if existing is not None and existing != after_color:
            return {}
        mapping[before_color] = after_color
    return {str(key): int(value) for key, value in sorted(mapping.items())}


def _monotonic_direction(values: list[int]) -> str | None:
    if len(values) < 3 or len(set(values)) < 2:
        return None
    if values == sorted(values):
        return "ascending"
    if values == sorted(values, reverse=True):
        return "descending"
    return None


def _sorted_sequence_descriptors(scene: Any) -> list[dict[str, Any]]:
    objects = _component_objects(scene)
    descriptors: dict[str, dict[str, Any]] = {}

    def add_group(axis: str, group: list[Any]) -> None:
        if len(group) < 3:
            return
        position_index = 1 if axis == "horizontal" else 0
        ordered = sorted(group, key=lambda obj: (float(obj.centroid[position_index]), obj.bbox, obj.object_id))
        values_by_key = {
            "color": [int(obj.color) for obj in ordered],
            "area": [int(obj.area) for obj in ordered],
            "height": [int(obj.bbox[2] - obj.bbox[0] + 1) for obj in ordered],
            "width": [int(obj.bbox[3] - obj.bbox[1] + 1) for obj in ordered],
        }
        for key, values in values_by_key.items():
            direction = _monotonic_direction(values)
            if direction is None:
                continue
            selector = {"axis": axis, "key": key, "direction": direction}
            signature = json.dumps(selector, sort_keys=True, separators=(",", ":"))
            descriptors[signature] = selector

    by_top: dict[int, list[Any]] = {}
    by_left: dict[int, list[Any]] = {}
    by_center_y: dict[int, list[Any]] = {}
    by_center_x: dict[int, list[Any]] = {}
    for obj in objects:
        by_top.setdefault(int(obj.bbox[0]), []).append(obj)
        by_left.setdefault(int(obj.bbox[1]), []).append(obj)
        by_center_y.setdefault(int(round(float(obj.centroid[0]))), []).append(obj)
        by_center_x.setdefault(int(round(float(obj.centroid[1]))), []).append(obj)

    for group in list(by_top.values()) + list(by_center_y.values()):
        add_group("horizontal", group)
    for group in list(by_left.values()) + list(by_center_x.values()):
        add_group("vertical", group)
    return list(descriptors.values())


def _scene_goal_structure_candidates(scene: Any, event: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    objects = _component_objects(scene)
    heights = _value_counts([int(obj.bbox[2] - obj.bbox[0] + 1) for obj in objects])
    areas = _value_counts([int(obj.area) for obj in objects])
    for height, count in sorted(heights.items()):
        if count >= 2:
            candidates.append(
                {
                    "scope": "goal",
                    "kind": "height_equality",
                    "selector": {"height": int(height)},
                    "progress_test": {"equal_height_count_gte": 2},
                }
            )
    for area, count in sorted(areas.items()):
        if count >= 2:
            candidates.append(
                {
                    "scope": "goal",
                    "kind": "area_equality",
                    "selector": {"area": int(area)},
                    "progress_test": {"equal_area_count_gte": 2},
                }
            )
    for selector in _sorted_sequence_descriptors(scene):
        candidates.append(
            {
                "scope": "goal",
                "kind": "sorting",
                "selector": selector,
                "progress_test": {"ordered_sequence": dict(selector)},
            }
        )
    relation_names = {relation.relation for relation in scene.relation_graph}
    if "contains" in relation_names:
        candidates.append(
            {
                "scope": "goal",
                "kind": "containment",
                "selector": {"relation": "contains"},
                "progress_test": {"relation_present": "contains"},
            }
        )
    mapping = _color_transform_mapping(event)
    if mapping:
        candidates.append(
            {
                "scope": "goal",
                "kind": "transformation_closure",
                "selector": {"mapping": mapping},
                "progress_test": {"consistent_color_mapping": mapping},
            }
        )
    return candidates


def _scene_subgoal_structure_candidates(scene: Any, event: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    before_scene = _before_grid_scene(event)
    before_by_id = {}
    if before_scene is not None:
        before_by_id = {str(obj.object_id): obj for obj in _component_objects(before_scene)}

    for move in scene.edit_script.moves:
        before_obj = before_by_id.get(str(move.get("object_id")))
        if before_obj is not None:
            descriptor = _object_descriptor(before_obj)
        else:
            descriptor = {
                "canonical_hash": "unknown",
                "color": int(move.get("color", -1)),
                "area": 0,
            }
        delta = [round(float(item), 4) for item in move.get("delta", [0.0, 0.0])[:2]]
        candidates.append(
            {
                "scope": "subgoal",
                "kind": "controllable_object",
                "selector": {"object_key": _object_key(descriptor), "action": str(event.get("action"))},
                "progress_test": {"moved_object": descriptor, "delta": delta},
            }
        )

    for obj in _component_objects(scene):
        if int(getattr(obj, "changed_count", 0)) <= 0:
            continue
        descriptor = _object_descriptor(obj)
        candidates.append(
            {
                "scope": "subgoal",
                "kind": "target_object",
                "selector": {"object_key": _object_key(descriptor)},
                "progress_test": {
                    "changed_object": descriptor,
                    "changed_count_gt": 0,
                    "centroid": [round(float(obj.centroid[0]), 4), round(float(obj.centroid[1]), 4)],
                },
            }
        )

    objects = _component_objects(scene)
    heights = _value_counts([int(obj.bbox[2] - obj.bbox[0] + 1) for obj in objects])
    areas = _value_counts([int(obj.area) for obj in objects])
    for height, count in sorted(heights.items()):
        if count >= 2:
            candidates.append(
                {
                    "scope": "subgoal",
                    "kind": "constraint_equality",
                    "selector": {"field": "height", "height": int(height)},
                    "progress_test": {"equal_height_count_gte": 2, "count": int(count)},
                }
            )
    for area, count in sorted(areas.items()):
        if count >= 2:
            candidates.append(
                {
                    "scope": "subgoal",
                    "kind": "constraint_equality",
                    "selector": {"field": "area", "area": int(area)},
                    "progress_test": {"equal_area_count_gte": 2, "count": int(count)},
                }
            )
    relation_names = {relation.relation for relation in scene.relation_graph}
    for relation in ("contains", "row_aligned", "col_aligned", "same_color", "repetition"):
        if relation in relation_names:
            candidates.append(
                {
                    "scope": "subgoal",
                    "kind": "relation_constraint",
                    "selector": {"relation": relation},
                    "progress_test": {"relation_present": relation},
                }
            )
    for item in _metadata_counter_values(event):
        selector = {"field": str(item.get("field", ""))}
        if "event" in item:
            selector["event"] = str(item["event"])
        candidates.append(
            {
                "scope": "subgoal",
                "kind": "counter_tracking",
                "selector": selector,
                "progress_test": dict(item),
            }
        )
    for item in _delta_counter_fields(event):
        candidates.append(
            {
                "scope": "subgoal",
                "kind": "counter_tracking",
                "selector": {"field": str(item["field"])},
                "progress_test": dict(item),
            }
        )
    return candidates


def _goal_candidates(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    score = _score_delta(event)
    events = _events(event)
    if score > 0:
        candidates.append(
            {
                "scope": "goal",
                "kind": "positive_reward",
                "selector": {"signal": "score_delta"},
                "progress_test": {"score_delta_gt": 0.0},
            }
        )
    if _terminal(event):
        candidates.append(
            {
                "scope": "goal",
                "kind": "terminal_win",
                "selector": {"signal": "terminal"},
                "progress_test": {"terminal": True},
            }
        )
    for item in events:
        candidates.append(
            {
                "scope": "goal",
                "kind": "event",
                "selector": {"event": item},
                "progress_test": {"event_present": item},
            }
        )
        lower = item.lower()
        if "gate" in lower or "counter" in lower or "key" in lower or "door" in lower:
            candidates.append(
                {
                    "scope": "subgoal",
                    "kind": "gate_or_counter",
                    "selector": {"event": item},
                    "progress_test": {"event_present": item},
                }
            )
    for item in _metadata_counter_values(event):
        selector = {"field": str(item.get("field", ""))}
        if "event" in item:
            selector["event"] = str(item["event"])
        candidates.append(
            {
                "scope": "subgoal",
                "kind": "counter_tracking",
                "selector": selector,
                "progress_test": dict(item),
            }
        )
    for item in _delta_counter_fields(event):
        candidates.append(
            {
                "scope": "subgoal",
                "kind": "counter_tracking",
                "selector": {"field": str(item["field"])},
                "progress_test": dict(item),
            }
        )
    scene = _grid_scene(event)
    if scene is None:
        return candidates
    if scene.edit_script.disappears:
        colors = sorted({int(edit["before"]) for edit in scene.edit_script.disappears})
        for color in colors:
            candidates.append(
                {
                    "scope": "goal",
                    "kind": "object_disappears",
                    "selector": {"color": color},
                    "progress_test": {"disappears_color": color},
                }
            )
    if scene.edit_script.appears:
        colors = sorted({int(edit["after"]) for edit in scene.edit_script.appears})
        for color in colors:
            candidates.append(
                {
                    "scope": "goal",
                    "kind": "object_appears",
                    "selector": {"color": color},
                    "progress_test": {"appears_color": color},
                }
            )
    if scene.edit_script.moves:
        for move in scene.edit_script.moves:
            candidates.append(
                {
                    "scope": "subgoal",
                    "kind": "movement_subgoal",
                    "selector": {"color": int(move["color"])},
                    "progress_test": {"move_color": int(move["color"])},
                }
            )
    if scene.edit_script.changed_count > 0:
        candidates.append(
            {
                "scope": "subgoal",
                "kind": "target_changed_cell",
                "selector": {"field": "grid"},
                "progress_test": {"changed_count_gt": 0},
            }
        )
    if scene.edit_script.changed_count == 0:
        candidates.append(
            {
                "scope": "subgoal",
                "kind": "avoid_no_op",
                "selector": {"action": str(event.get("action"))},
                "progress_test": {"changed_count_eq": 0},
            }
        )
    if scene.edit_script.disappears and not scene.edit_script.appears:
        candidates.append(
            {
                "scope": "goal",
                "kind": "clearing",
                "selector": {"field": "grid"},
                "progress_test": {"disappears_without_appears": True},
            }
        )
    if scene.edit_script.appears and not scene.edit_script.disappears:
        candidates.append(
            {
                "scope": "goal",
                "kind": "filling",
                "selector": {"field": "grid"},
                "progress_test": {"appears_without_disappears": True},
            }
        )
    relation_names = {relation.relation for relation in scene.relation_graph}
    if "row_aligned" in relation_names or "col_aligned" in relation_names:
        candidates.append(
            {
                "scope": "goal",
                "kind": "alignment",
                "selector": {"relation": "alignment"},
                "progress_test": {"relation_present": "alignment"},
            }
        )
    if "same_color" in relation_names or "repetition" in relation_names:
        candidates.append(
            {
                "scope": "goal",
                "kind": "matching",
                "selector": {"relation": "matching"},
                "progress_test": {"relation_present": "matching"},
            }
        )
    candidates.extend(_scene_goal_structure_candidates(scene, event))
    candidates.extend(_scene_subgoal_structure_candidates(scene, event))
    return candidates


def _candidate_supported(candidate: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    kind = str(candidate.get("kind"))
    if kind == "positive_reward":
        return _score_delta(event) > 0
    if kind == "terminal_win":
        return _terminal(event)
    if kind in {"event", "gate_or_counter"}:
        expected = str(candidate.get("selector", {}).get("event"))
        return expected in _events(event)
    if kind == "counter_tracking":
        expected_field = str(candidate.get("selector", {}).get("field", ""))
        expected_event = candidate.get("selector", {}).get("event")
        for item in _metadata_counter_values(event):
            if expected_event is not None and str(item.get("event")) == str(expected_event):
                return True
            if str(item.get("field", "")) == expected_field:
                return True
        return any(str(item.get("field", "")) == expected_field for item in _delta_counter_fields(event))
    scene = _grid_scene(event)
    if scene is None:
        return False
    selector = candidate.get("selector", {})
    if kind == "object_disappears":
        color = int(selector.get("color"))
        return any(int(edit["before"]) == color for edit in scene.edit_script.disappears)
    if kind == "object_appears":
        color = int(selector.get("color"))
        return any(int(edit["after"]) == color for edit in scene.edit_script.appears)
    if kind == "movement_subgoal":
        color = int(selector.get("color"))
        return any(int(move["color"]) == color for move in scene.edit_script.moves)
    if kind == "controllable_object":
        expected_key = str(selector.get("object_key", ""))
        expected_action = str(selector.get("action", event.get("action")))
        if expected_action != str(event.get("action")):
            return False
        before_scene = _before_grid_scene(event)
        before_by_id = {}
        if before_scene is not None:
            before_by_id = {str(obj.object_id): obj for obj in _component_objects(before_scene)}
        for move in scene.edit_script.moves:
            before_obj = before_by_id.get(str(move.get("object_id")))
            if before_obj is None:
                descriptor = {
                    "canonical_hash": "unknown",
                    "color": int(move.get("color", -1)),
                    "area": 0,
                }
            else:
                descriptor = _object_descriptor(before_obj)
            if _object_key(descriptor) == expected_key:
                return True
        return False
    if kind == "target_object":
        expected_key = str(selector.get("object_key", ""))
        return any(
            int(getattr(obj, "changed_count", 0)) > 0 and _object_key(_object_descriptor(obj)) == expected_key
            for obj in _component_objects(scene)
        )
    if kind == "target_changed_cell":
        return scene.edit_script.changed_count > 0
    if kind == "avoid_no_op":
        return scene.edit_script.changed_count == 0
    if kind == "clearing":
        return bool(scene.edit_script.disappears and not scene.edit_script.appears)
    if kind == "filling":
        return bool(scene.edit_script.appears and not scene.edit_script.disappears)
    relation_names = {relation.relation for relation in scene.relation_graph}
    if kind == "alignment":
        return "row_aligned" in relation_names or "col_aligned" in relation_names
    if kind == "matching":
        return "same_color" in relation_names or "repetition" in relation_names
    selector = candidate.get("selector", {})
    if kind == "height_equality":
        height = int(selector.get("height"))
        heights = _value_counts([int(obj.bbox[2] - obj.bbox[0] + 1) for obj in _component_objects(scene)])
        return int(heights.get(height, 0)) >= 2
    if kind == "area_equality":
        area = int(selector.get("area"))
        areas = _value_counts([int(obj.area) for obj in _component_objects(scene)])
        return int(areas.get(area, 0)) >= 2
    if kind == "constraint_equality":
        field = str(selector.get("field", ""))
        if field == "height":
            height = int(selector.get("height"))
            heights = _value_counts([int(obj.bbox[2] - obj.bbox[0] + 1) for obj in _component_objects(scene)])
            return int(heights.get(height, 0)) >= 2
        if field == "area":
            area = int(selector.get("area"))
            areas = _value_counts([int(obj.area) for obj in _component_objects(scene)])
            return int(areas.get(area, 0)) >= 2
        return False
    if kind == "sorting":
        expected = {
            "axis": str(selector.get("axis")),
            "key": str(selector.get("key")),
            "direction": str(selector.get("direction")),
        }
        return expected in _sorted_sequence_descriptors(scene)
    if kind == "containment":
        return "contains" in relation_names
    if kind == "relation_constraint":
        return str(selector.get("relation", "")) in relation_names
    if kind == "transformation_closure":
        return dict(selector.get("mapping", {})) == _color_transform_mapping(event)
    return False


def goal_candidates_from_event(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    return _goal_candidates(event)


def _score_entry(entry: Mapping[str, Any]) -> float:
    return float(
        1.25 * int(entry.get("support", 0))
        - 1.45 * int(entry.get("inconsistency", 0))
        - 0.30 * float(entry.get("description_length", 1.0))
    )


def _normalize(posterior: dict[str, Any]) -> None:
    goals = posterior.setdefault("goals", {})
    if not goals:
        posterior["posterior_mass"] = 0.0
        posterior["top_goals"] = []
        return
    max_score = max(float(goal.get("score", 0.0)) for goal in goals.values())
    weights = {key: math.exp(float(goal.get("score", 0.0)) - max_score) for key, goal in goals.items()}
    total = sum(weights.values())
    for key, goal in goals.items():
        goal["posterior"] = float(weights[key] / max(total, 1.0e-12))
    posterior["posterior_mass"] = float(sum(goal["posterior"] for goal in goals.values()))
    top = sorted(goals.values(), key=lambda goal: (goal["posterior"], goal["support"]), reverse=True)
    posterior["top_goals"] = [
        {
            "id": goal["id"],
            "kind": goal["kind"],
            "posterior": goal["posterior"],
            "support": goal["support"],
            "inconsistency": goal["inconsistency"],
        }
        for goal in top[:8]
    ]


def _ensure_entry(store: dict[str, Any], candidate: Mapping[str, Any]) -> dict[str, Any]:
    cid = _candidate_id(candidate)
    if cid not in store:
        store[cid] = {
            "id": cid,
            "scope": candidate["scope"],
            "kind": candidate["kind"],
            "selector": dict(candidate.get("selector", {})),
            "progress_test": dict(candidate.get("progress_test", {})),
            "description_length": _description_length(candidate),
            "support": 0,
            "inconsistency": 0,
            "score": 0.0,
            "posterior": 0.0,
        }
    return store[cid]


def update_goal_posterior(posterior: dict[str, Any], event: Mapping[str, Any]) -> dict[str, Any]:
    if not posterior:
        posterior.update(fresh_goal_posterior())
    goals = posterior.setdefault("goals", {})
    subgoals = posterior.setdefault("subgoals", {})
    candidates = _goal_candidates(event)
    candidate_ids = set()
    for candidate in candidates:
        store = goals if candidate["scope"] == "goal" else subgoals
        entry = _ensure_entry(store, candidate)
        candidate_ids.add(entry["id"])
    for store in (goals, subgoals):
        for entry in store.values():
            if _candidate_supported(entry, event):
                entry["support"] = int(entry.get("support", 0)) + 1
            elif entry["id"] in candidate_ids or _score_delta(event) <= 0.0:
                entry["inconsistency"] = int(entry.get("inconsistency", 0)) + 1
            entry["score"] = _score_entry(entry)
    posterior["updates"] = int(posterior.get("updates", 0)) + 1
    _normalize(posterior)
    return posterior
