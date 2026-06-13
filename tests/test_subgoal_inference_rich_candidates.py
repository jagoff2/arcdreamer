from __future__ import annotations

import torch

from src.grid_perception import parse_grid
from src.persistent_memory import PersistentMemoryState


def _object_key(descriptor: dict[str, object]) -> str:
    return "object|" + "|".join(f"{name}:{value}" for name, value in sorted(descriptor.items()))


def _find_subgoal(
    subgoals: dict[str, dict[str, object]],
    *,
    kind: str,
    selector: dict[str, object] | None = None,
) -> dict[str, object] | None:
    for entry in subgoals.values():
        if entry["kind"] != kind:
            continue
        current = entry.get("selector", {})
        if selector is None:
            return entry
        if all(current.get(key) == value for key, value in selector.items()):
            return entry
    return None


def _moved_object_descriptor(grid: torch.Tensor) -> dict[str, int]:
    scene = parse_grid(grid.numpy())
    components = [obj for obj in scene.objects if obj.kind == "component"]
    assert len(components) == 1
    obj = components[0]
    return {
        "canonical_hash": str(obj.canonical_hash),
        "color": int(obj.color),
        "area": int(obj.area),
    }


def _changed_component(scene: object, *, predicate) -> dict[str, int]:
    objects = [obj for obj in scene.objects if obj.kind == "component" and predicate(obj)]
    assert len(objects) == 1
    obj = objects[0]
    return {
        "canonical_hash": str(obj.canonical_hash),
        "color": int(obj.color),
        "area": int(obj.area),
    }


def test_subgoal_from_moved_object_tracks_controllable_object_identity() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = torch.tensor([[2, 2, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.long)
    after = torch.tensor([[0, 0, 0], [2, 2, 0], [0, 0, 0]], dtype=torch.long)

    moved = _moved_object_descriptor(before)
    expected_key = _object_key(moved)
    memory.append_event(
        tick=1,
        observation={"grid": before},
        action=0,
        next_observation={"grid": after},
        metadata={"score_delta": 0.0, "terminal": False},
    )

    controllable = _find_subgoal(memory.goal_posterior["subgoals"], kind="controllable_object", selector={"object_key": expected_key, "action": "0"})
    target_like = _find_subgoal(memory.goal_posterior["subgoals"], kind="target_object", selector={"object_key": expected_key})
    movement = _find_subgoal(memory.goal_posterior["subgoals"], kind="movement_subgoal", selector={"color": 2})
    assert controllable is not None
    assert target_like is not None
    assert movement is not None
    assert controllable["progress_test"]["moved_object"] == moved
    assert target_like["progress_test"]["changed_object"] == moved
    assert controllable["selector"]["object_key"] == expected_key
    assert target_like["selector"]["object_key"] == expected_key


def test_subgoal_from_color_change_tracks_changed_object_identity() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = torch.tensor([[0, 1, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.long)
    after = torch.tensor([[0, 2, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.long)

    next_scene = parse_grid(after.numpy(), previous_grid=before.numpy(), action=1)
    changed_descriptor = _changed_component(
        next_scene,
        predicate=lambda obj: obj.changed_count > 0,
    )
    changed_key = _object_key(changed_descriptor)

    memory.append_event(
        tick=1,
        observation={"grid": before},
        action=1,
        next_observation={"grid": after},
        metadata={"score_delta": 0.0, "terminal": False},
    )

    target_like = _find_subgoal(memory.goal_posterior["subgoals"], kind="target_object", selector={"object_key": changed_key})
    assert target_like is not None
    assert target_like["progress_test"]["changed_object"] == changed_descriptor


def test_counter_metadata_generates_deterministic_counter_tracking_subgoals() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    metadata = {
        "attempt_count": 3,
        "inventory": {"coins": 5, "keys": 1},
        "events": ["collected key"],
    }

    memory.append_event(
        tick=1,
        observation={"metric": torch.tensor([1.0])},
        action=2,
        next_observation={"metric": torch.tensor([1.0])},
        metadata=metadata,
    )

    assert _find_subgoal(
        memory.goal_posterior["subgoals"],
        kind="counter_tracking",
        selector={"field": "attempt_count"},
    ) is not None
    assert _find_subgoal(
        memory.goal_posterior["subgoals"],
        kind="counter_tracking",
        selector={"field": "inventory.coins"},
    ) is not None
    assert _find_subgoal(
        memory.goal_posterior["subgoals"],
        kind="counter_tracking",
        selector={"field": "inventory.keys"},
    ) is not None
    assert _find_subgoal(
        memory.goal_posterior["subgoals"],
        kind="counter_tracking",
        selector={"field": "event", "event": "collected key"},
    ) is not None


def test_constraint_and_equality_subgoals_are_emitted_with_matching_goal_equality_candidates() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = torch.tensor(
        [
            [0, 2, 2, 0, 0],
            [0, 0, 0, 2, 2],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    memory.append_event(
        tick=1,
        observation={"grid": before},
        action=0,
        next_observation={"grid": before},
        metadata={"score_delta": 0.0, "terminal": False},
    )

    assert _find_subgoal(
        memory.goal_posterior["subgoals"],
        kind="constraint_equality",
        selector={"field": "height", "height": 1},
    ) is not None
    assert _find_subgoal(
        memory.goal_posterior["subgoals"],
        kind="constraint_equality",
        selector={"field": "area", "area": 2},
    ) is not None
    assert memory.goal_posterior["goals"]["goal|height_equality|height:1"]["support"] == 1
    assert memory.goal_posterior["goals"]["goal|area_equality|area:2"]["support"] == 1
