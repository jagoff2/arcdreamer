from __future__ import annotations

import json
from typing import Any, Mapping

import torch

from .grid_perception import parse_grid


def fresh_controllability_model() -> dict[str, Any]:
    return {
        "schema": "runtime_controllability_model_v1",
        "updates": 0,
        "action_effects": {},
        "controlled_objects": {},
        "controlled_variables": {},
        "intervention_traces": [],
        "top_controlled": [],
    }


def _tensor_grid(value: Any) -> torch.Tensor | None:
    if isinstance(value, torch.Tensor) and value.ndim == 2:
        return value.detach().cpu().long()
    return None


def _changed(delta_item: Any) -> bool:
    if isinstance(delta_item, Mapping):
        if "changed_count" in delta_item:
            return int(delta_item.get("changed_count", 0)) > 0
        if "changed" in delta_item:
            return bool(delta_item.get("changed"))
    return bool(delta_item)


def _l1(delta_item: Any) -> float:
    if isinstance(delta_item, Mapping) and "l1" in delta_item:
        return float(delta_item.get("l1", 0.0))
    return 1.0 if _changed(delta_item) else 0.0


def _effect_bucket(model: dict[str, Any], action: int | str) -> dict[str, Any]:
    return model.setdefault("action_effects", {}).setdefault(
        str(action),
        {
            "action": str(action),
            "trials": 0,
            "changed_trials": 0,
            "no_op_trials": 0,
            "movement_trials": 0,
            "color_change_trials": 0,
            "count_change_trials": 0,
            "field_change_trials": 0,
            "variables": {},
            "object_keys": {},
            "confidence": 0.0,
            "last_tick": 0,
        },
    )


def _variable_bucket(model: dict[str, Any], variable: str) -> dict[str, Any]:
    return model.setdefault("controlled_variables", {}).setdefault(
        variable,
        {
            "variable": variable,
            "support": 0,
            "actions": {},
            "confidence": 0.0,
            "last_tick": 0,
        },
    )


def _object_bucket(model: dict[str, Any], key: str, descriptor: Mapping[str, Any]) -> dict[str, Any]:
    return model.setdefault("controlled_objects", {}).setdefault(
        key,
        {
            "key": key,
            "descriptor": dict(descriptor),
            "support": 0,
            "actions": {},
            "deltas": {},
            "confidence": 0.0,
            "last_tick": 0,
        },
    )


def _bump_action_count(container: dict[str, Any], key: str) -> None:
    values = container.setdefault("actions", {})
    values[key] = int(values.get(key, 0)) + 1


def _bump_variable(
    model: dict[str, Any],
    action_effect: dict[str, Any],
    *,
    variable: str,
    action: str,
    tick: int,
) -> None:
    action_effect.setdefault("variables", {})[variable] = int(action_effect.setdefault("variables", {}).get(variable, 0)) + 1
    bucket = _variable_bucket(model, variable)
    bucket["support"] = int(bucket.get("support", 0)) + 1
    _bump_action_count(bucket, action)
    bucket["last_tick"] = int(tick)
    bucket["confidence"] = float(bucket["support"] / max(model.get("updates", 1), 1))


def _record_object(
    model: dict[str, Any],
    action_effect: dict[str, Any],
    *,
    action: str,
    tick: int,
    descriptor: Mapping[str, Any],
    delta: tuple[float, float] | None = None,
) -> str:
    key = "object|" + "|".join(f"{name}:{descriptor[name]}" for name in sorted(descriptor))
    action_effect.setdefault("object_keys", {})[key] = int(action_effect.setdefault("object_keys", {}).get(key, 0)) + 1
    bucket = _object_bucket(model, key, descriptor)
    bucket["support"] = int(bucket.get("support", 0)) + 1
    _bump_action_count(bucket, action)
    if delta is not None:
        delta_key = f"{round(float(delta[0]), 4)},{round(float(delta[1]), 4)}"
        deltas = bucket.setdefault("deltas", {})
        deltas[delta_key] = int(deltas.get(delta_key, 0)) + 1
    bucket["last_tick"] = int(tick)
    bucket["confidence"] = float(bucket["support"] / max(model.get("updates", 1), 1))
    return key


def _object_descriptor(before_scene: Any, move: Mapping[str, Any]) -> dict[str, Any]:
    moved = next((obj for obj in before_scene.objects if obj.object_id == move.get("object_id")), None)
    return {
        "color": int(move.get("color", moved.color if moved is not None else -1)),
        "canonical_hash": str(moved.canonical_hash if moved is not None else "unknown"),
        "area": int(moved.area if moved is not None else 0),
    }


def _grid_effects(
    *,
    model: dict[str, Any],
    action_effect: dict[str, Any],
    action: str,
    tick: int,
    before_grid: torch.Tensor,
    after_grid: torch.Tensor,
) -> dict[str, Any]:
    before_scene = parse_grid(before_grid.numpy())
    scene = parse_grid(after_grid.numpy(), previous_grid=before_grid.numpy(), action=action, time_index=tick)
    variables: list[str] = []
    object_keys: list[str] = []
    for move in scene.edit_script.moves:
        action_effect["movement_trials"] = int(action_effect.get("movement_trials", 0)) + 1
        _bump_variable(model, action_effect, variable="position", action=action, tick=tick)
        variables.append("position")
        delta = tuple(float(item) for item in move.get("delta", [0.0, 0.0])[:2])
        object_keys.append(
            _record_object(
                model,
                action_effect,
                action=action,
                tick=tick,
                descriptor=_object_descriptor(before_scene, move),
                delta=(float(delta[0]), float(delta[1])),
            )
        )
    color_changes = [
        edit
        for edit in scene.edit_script.cell_edits
        if int(edit.get("before", 0)) != int(edit.get("after", 0))
        and edit not in scene.edit_script.appears
        and edit not in scene.edit_script.disappears
    ]
    if color_changes:
        action_effect["color_change_trials"] = int(action_effect.get("color_change_trials", 0)) + 1
        _bump_variable(model, action_effect, variable="color", action=action, tick=tick)
        variables.append("color")
    if scene.edit_script.appears or scene.edit_script.disappears:
        action_effect["count_change_trials"] = int(action_effect.get("count_change_trials", 0)) + 1
        _bump_variable(model, action_effect, variable="count", action=action, tick=tick)
        variables.append("count")
        for edit in scene.edit_script.appears:
            descriptor = {"color": int(edit["after"]), "canonical_hash": "cell", "area": 1}
            object_keys.append(_record_object(model, action_effect, action=action, tick=tick, descriptor=descriptor))
        for edit in scene.edit_script.disappears:
            descriptor = {"color": int(edit["before"]), "canonical_hash": "cell", "area": 1}
            object_keys.append(_record_object(model, action_effect, action=action, tick=tick, descriptor=descriptor))
    return {
        "grid_hash_before": before_scene.raw_hash,
        "grid_hash_after": scene.raw_hash,
        "changed_count": int(scene.edit_script.changed_count),
        "variables": sorted(set(variables)),
        "object_keys": sorted(set(object_keys)),
    }


def _non_grid_effects(
    *,
    model: dict[str, Any],
    action_effect: dict[str, Any],
    action: str,
    tick: int,
    delta: Mapping[str, Any],
) -> dict[str, Any]:
    variables: list[str] = []
    for field, item in sorted(delta.items()):
        if not _changed(item):
            continue
        variable = f"field:{field}"
        action_effect["field_change_trials"] = int(action_effect.get("field_change_trials", 0)) + 1
        _bump_variable(model, action_effect, variable=variable, action=action, tick=tick)
        variables.append(variable)
    return {"variables": variables, "delta_l1": float(sum(_l1(item) for item in delta.values()))}


def _normalize(model: dict[str, Any]) -> None:
    total_updates = max(int(model.get("updates", 0)), 1)
    for action_effect in model.get("action_effects", {}).values():
        action_effect["confidence"] = float(int(action_effect.get("changed_trials", 0)) / max(int(action_effect.get("trials", 0)), 1))
    for bucket in model.get("controlled_variables", {}).values():
        bucket["confidence"] = float(int(bucket.get("support", 0)) / total_updates)
    for bucket in model.get("controlled_objects", {}).values():
        bucket["confidence"] = float(int(bucket.get("support", 0)) / total_updates)
    items = []
    for bucket in model.get("controlled_variables", {}).values():
        items.append(
            {
                "kind": "variable",
                "key": bucket["variable"],
                "support": int(bucket.get("support", 0)),
                "confidence": float(bucket.get("confidence", 0.0)),
            }
        )
    for bucket in model.get("controlled_objects", {}).values():
        items.append(
            {
                "kind": "object",
                "key": bucket["key"],
                "support": int(bucket.get("support", 0)),
                "confidence": float(bucket.get("confidence", 0.0)),
                "descriptor": dict(bucket.get("descriptor", {})),
            }
        )
    items.sort(key=lambda item: (float(item["confidence"]), int(item["support"]), str(item["key"])), reverse=True)
    model["top_controlled"] = items[:12]


def update_controllability_model(
    model: dict[str, Any],
    semantic_memory: dict[str, Any],
    *,
    tick: int,
    observation: Mapping[str, Any],
    action: int | str,
    next_observation: Mapping[str, Any],
    delta: Mapping[str, Any],
    capacity: int = 64,
) -> dict[str, Any]:
    if not model:
        model.update(fresh_controllability_model())
    action_text = str(action)
    model["updates"] = int(model.get("updates", 0)) + 1
    action_effect = _effect_bucket(model, action_text)
    action_effect["trials"] = int(action_effect.get("trials", 0)) + 1
    action_effect["last_tick"] = int(tick)
    changed = any(_changed(item) for item in delta.values())
    action_effect["changed_trials"] = int(action_effect.get("changed_trials", 0)) + int(changed)
    action_effect["no_op_trials"] = int(action_effect.get("no_op_trials", 0)) + int(not changed)

    before_grid = _tensor_grid(observation.get("grid"))
    after_grid = _tensor_grid(next_observation.get("grid"))
    if before_grid is not None and after_grid is not None and tuple(before_grid.shape) == tuple(after_grid.shape):
        trace = _grid_effects(
            model=model,
            action_effect=action_effect,
            action=action_text,
            tick=tick,
            before_grid=before_grid,
            after_grid=after_grid,
        )
        trace["mode"] = "grid"
    else:
        trace = _non_grid_effects(
            model=model,
            action_effect=action_effect,
            action=action_text,
            tick=tick,
            delta=delta,
        )
        trace["mode"] = "field"
    trace.update({"tick": int(tick), "action": action_text, "changed": bool(changed)})
    traces = model.setdefault("intervention_traces", [])
    traces.append(trace)
    if len(traces) > int(capacity):
        del traces[: len(traces) - int(capacity)]
    _normalize(model)

    fact_key = f"controllable:action:{action_text}"
    fact = {
        "fact": fact_key,
        "kind": "controllability",
        "action": action_text,
        "support": int(action_effect.get("changed_trials", 0)),
        "trials": int(action_effect.get("trials", 0)),
        "variables": dict(action_effect.get("variables", {})),
        "object_keys": dict(action_effect.get("object_keys", {})),
        "confidence": float(action_effect.get("confidence", 0.0)),
        "last_tick": int(tick),
    }
    semantic_memory.setdefault("facts", {})[fact_key] = fact
    semantic_memory.setdefault("controllability_facts", {})[fact_key] = fact
    semantic_memory.setdefault("procedural_facts", {}).setdefault(fact_key, fact)
    semantic_memory["procedural_facts"][fact_key] = fact
    return model
