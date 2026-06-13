from __future__ import annotations

from pathlib import Path

import torch

from src.grid_perception import parse_grid
from src.persistent_memory import PersistentMemoryState


def _object_key(descriptor: dict[str, object]) -> str:
    return "object|" + "|".join(f"{name}:{value}" for name, value in sorted(descriptor.items()))


def _single_component(scene):
    components = [obj for obj in scene.objects if obj.kind == "component"]
    assert len(components) == 1
    return components[0]


def test_fresh_memory_has_controllability_model_schema() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    model = memory.controllability_model

    assert model["schema"] == "runtime_controllability_model_v1"
    assert model["updates"] == 0
    assert model["action_effects"] == {}
    assert model["controlled_objects"] == {}
    assert model["controlled_variables"] == {}
    assert model["intervention_traces"] == []
    assert model["top_controlled"] == []


def test_grid_movement_records_position_control_and_semantic_fact_mirror() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")

    before_grid = torch.tensor(
        [
            [0, 2, 2],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=torch.long,
    )
    after_grid = torch.tensor(
        [
            [0, 0, 0],
            [0, 2, 2],
            [0, 0, 0],
        ],
        dtype=torch.long,
    )
    before_scene = parse_grid(before_grid.numpy())
    moved_obj = _single_component(before_scene)
    key = _object_key(
        {
            "canonical_hash": moved_obj.canonical_hash,
            "color": moved_obj.color,
            "area": moved_obj.area,
        }
    )

    memory.append_event(
        tick=1,
        observation={"grid": before_grid},
        action=1,
        next_observation={"grid": after_grid},
        metadata={"terminal": False, "score_delta": 0.0},
    )

    model = memory.controllability_model
    action_effect = model["action_effects"]["1"]
    variable_bucket = model["controlled_variables"]["position"]
    object_bucket = model["controlled_objects"][key]
    fact = memory.semantic_memory["facts"]["controllable:action:1"]

    assert action_effect["movement_trials"] == 1
    assert action_effect["variables"]["position"] == 1
    assert variable_bucket["support"] == 1
    assert variable_bucket["actions"] == {"1": 1}
    assert key in model["controlled_objects"]
    assert object_bucket["actions"] == {"1": 1}
    assert object_bucket["support"] == 1
    assert object_bucket["deltas"] == {"1.0,0.0": 1}
    assert object_bucket["descriptor"] == {
        "canonical_hash": moved_obj.canonical_hash,
        "color": moved_obj.color,
        "area": moved_obj.area,
    }
    assert fact["fact"] == "controllable:action:1"
    assert fact["kind"] == "controllability"
    assert fact["action"] == "1"
    assert fact["support"] == 1
    assert fact["trials"] == 1
    assert fact["variables"] == action_effect["variables"]
    assert fact["object_keys"] == action_effect["object_keys"]
    assert fact["confidence"] == action_effect["confidence"]
    assert fact["last_tick"] == 1


def test_grid_color_and_count_transforms_record_controlled_variables() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")

    memory.append_event(
        tick=1,
        observation={"grid": torch.tensor([[0, 0], [0, 1]], dtype=torch.long)},
        action=2,
        next_observation={"grid": torch.tensor([[0, 0], [0, 2]], dtype=torch.long)},
        metadata={"terminal": False, "score_delta": 0.0},
    )

    color_effect = memory.controllability_model["action_effects"]["2"]
    assert color_effect["color_change_trials"] == 1
    assert color_effect["variables"]["color"] == 1
    assert memory.controllability_model["controlled_variables"]["color"]["support"] == 1

    memory.append_event(
        tick=2,
        observation={"grid": torch.tensor([[0, 0], [0, 0]], dtype=torch.long)},
        action=3,
        next_observation={"grid": torch.tensor([[0, 0], [3, 0]], dtype=torch.long)},
        metadata={"terminal": False, "score_delta": 0.0},
    )

    count_effect = memory.controllability_model["action_effects"]["3"]
    assert count_effect["count_change_trials"] == 1
    assert count_effect["variables"]["count"] == 1
    object_keys = set(count_effect["object_keys"])
    assert "object|area:1|canonical_hash:cell|color:3" in object_keys
    assert memory.controllability_model["controlled_variables"]["count"]["support"] == 1


def test_non_grid_tensor_field_change_records_field_level_control() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")

    memory.append_event(
        tick=1,
        observation={"score": torch.tensor([1.0], dtype=torch.float32)},
        action=4,
        next_observation={"score": torch.tensor([4.0], dtype=torch.float32)},
        metadata={"terminal": False, "score_delta": 0.0},
    )

    model = memory.controllability_model
    field_effect = model["action_effects"]["4"]
    field_bucket = model["controlled_variables"]["field:score"]
    trace = model["intervention_traces"][-1]

    assert field_effect["field_change_trials"] == 1
    assert field_effect["variables"]["field:score"] == 1
    assert field_bucket["support"] == 1
    assert field_bucket["actions"] == {"4": 1}
    assert trace["mode"] == "field"
    assert "field:score" in trace["variables"]


def test_repeated_consistent_interventions_increase_support_confidence_and_round_trip(tmp_path: Path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    pre_grid = torch.tensor(
        [
            [0, 5, 0],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=torch.long,
    )
    mid_grid = torch.tensor(
        [
            [0, 0, 5],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=torch.long,
    )
    post_grid = pre_grid.clone()

    memory.append_event(
        tick=1,
        observation={"signal": torch.tensor([1.0])},
        action=0,
        next_observation={"signal": torch.tensor([2.0])},
        metadata={"terminal": False, "score_delta": 0.0},
    )
    memory.append_event(
        tick=2,
        observation={"grid": pre_grid},
        action=9,
        next_observation={"grid": mid_grid},
        metadata={"terminal": False, "score_delta": 0.0},
    )

    movement_after_first = memory.controllability_model["controlled_variables"]["position"]
    first_confidence = movement_after_first["confidence"]
    first_support = movement_after_first["support"]

    memory.append_event(
        tick=3,
        observation={"grid": mid_grid},
        action=9,
        next_observation={"grid": post_grid},
        metadata={"terminal": False, "score_delta": 0.0},
    )

    movement_after_second = memory.controllability_model["controlled_variables"]["position"]
    second_support = movement_after_second["support"]
    second_confidence = movement_after_second["confidence"]

    assert second_support == first_support + 1
    assert second_confidence > first_confidence

    action_effect = memory.controllability_model["action_effects"]["9"]
    assert action_effect["movement_trials"] == 2

    before_scene = parse_grid(pre_grid.numpy())
    moved_obj = _single_component(before_scene)
    expected_key = _object_key(
        {
            "canonical_hash": moved_obj.canonical_hash,
            "area": moved_obj.area,
            "color": moved_obj.color,
        }
    )
    assert memory.controllability_model["controlled_objects"][expected_key]["support"] == 2

    path = tmp_path / "memory.pt"
    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=4, batch_size=1, device="cpu")

    assert loaded.controllability_model == memory.controllability_model
