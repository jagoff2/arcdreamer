from __future__ import annotations

import torch

from src.grid_perception import parse_grid
from src.persistent_memory import PersistentMemoryState


def _object_keys_from_grid(grid: torch.Tensor) -> set[str]:
    scene = parse_grid(grid.numpy())
    return {
        f"object:{obj.canonical_hash}:color:{obj.color}:area:{obj.area}"
        for obj in scene.objects
        if obj.kind == "component"
    }


def _single_object_key_and_colors(grid: torch.Tensor) -> tuple[str, str, int]:
    scene = parse_grid(grid.numpy())
    objects = [obj for obj in scene.objects if obj.kind == "component"]
    assert len(objects) == 1, "expected a single object in test fixture"
    obj = objects[0]
    object_key = f"object:{obj.canonical_hash}:color:{obj.color}:area:{obj.area}"
    return object_key, str(obj.canonical_hash), int(obj.color)


def test_object_identity_facts_are_created_from_grid_observations() -> None:
    before = torch.tensor([[1, 0, 2], [0, 0, 0], [0, 0, 0]], dtype=torch.long)
    after = before.clone()

    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    memory.append_event(
        tick=1,
        observation={"grid": before},
        action=4,
        next_observation={"grid": after},
        metadata={"terminal": False, "score_delta": 0.0},
    )

    object_facts = memory.semantic_memory["object_facts"]
    relation_facts = memory.semantic_memory["relation_facts"]

    expected_keys = _object_keys_from_grid(before)
    assert expected_keys.issubset(set(object_facts))
    for key in expected_keys:
        fact = object_facts[key]
        assert fact["kind"] == "object_identity"
        assert fact["support"] >= 1
        assert fact["examples"]
    assert any(fact["relation"] == "row_aligned" for fact in relation_facts.values())


def test_click_object_center_change_records_color_change_fact() -> None:
    before = torch.tensor([[0, 0, 0], [0, 2, 0], [0, 0, 0]], dtype=torch.long)
    after = torch.tensor([[0, 0, 0], [0, 7, 0], [0, 0, 0]], dtype=torch.long)
    object_key, object_hash, object_color = _single_object_key_and_colors(before)

    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    memory.append_event(
        tick=1,
        observation={"grid": before},
        action="click:1:1",
        next_observation={"grid": after},
        metadata={"terminal": False, "score_delta": 1.0, "action_coordinate": (1, 1)},
    )

    click_key = f"click:action:click:1:1:object:{object_hash}:object_center"
    fact = memory.semantic_memory["click_facts"][click_key]
    compacted = memory.semantic_memory["facts"][click_key]

    assert fact["kind"] == "clicked_object_center_changes_color"
    assert fact["changed_count"] == 1
    assert fact["color_changed_count"] == 1
    assert fact["object_fact"] == object_key
    assert fact["last_before_color"] == object_color
    assert fact["last_after_color"] == 7
    assert compacted["kind"] == fact["kind"]


def test_click_on_empty_coordinate_creates_no_op_click_and_obstacle_facts() -> None:
    before = torch.tensor([[0, 0, 0], [0, 5, 0], [0, 0, 0]], dtype=torch.long)
    after = before.clone()

    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    memory.append_event(
        tick=1,
        observation={"grid": before},
        action="click:0:0",
        next_observation={"grid": after},
        metadata={"terminal": False, "score_delta": 0.0, "action_coordinate": (0, 0)},
    )

    fact = memory.semantic_memory["click_facts"]["click:action:click:0:0:empty"]
    obstacle = memory.semantic_memory["obstacle_facts"]["obstacle:color:5"]

    assert fact["kind"] == "click_empty_no_op"
    assert fact["no_op_count"] == 1
    assert fact["changed_count"] == 0
    assert obstacle["no_op_support"] == 1
    assert obstacle["support"] == 1
    assert obstacle["actions"] == {"click:0:0": 1}
    assert fact == memory.semantic_memory["facts"]["click:action:click:0:0:empty"]


def test_goal_candidate_semantic_facts_include_alignment_matching_and_height_equality() -> None:
    before = torch.zeros((4, 4), dtype=torch.long)
    after = torch.tensor(
        [
            [0, 1, 0, 1],
            [0, 2, 0, 2],
            [0, 0, 0, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    memory.append_event(
        tick=1,
        observation={"grid": before},
        action=1,
        next_observation={"grid": after},
        metadata={"terminal": False, "score_delta": 1.0},
    )

    goal_facts = memory.semantic_memory["goal_facts"]
    alignment_event = goal_facts["goal_candidate:alignment:{\"relation\":\"alignment\"}"]
    matching_event = goal_facts["goal_candidate:matching:{\"relation\":\"matching\"}"]
    height_equality = goal_facts["goal_candidate:height_equality:{\"height\":1}"]

    alignment_posterior = goal_facts["posterior_goal_candidate:goal|alignment"]
    matching_posterior = goal_facts["posterior_goal_candidate:goal|matching"]

    assert alignment_event["kind"] == "alignment"
    assert matching_event["kind"] == "matching"
    assert height_equality["kind"] == "height_equality"
    assert alignment_event["selector"] == {"relation": "alignment"}
    assert matching_event["selector"] == {"relation": "matching"}
    assert height_equality["support"] == 1
    assert alignment_posterior["support"] == memory.goal_posterior["goals"]["goal|alignment"]["support"] == 1
    assert matching_posterior["support"] == memory.goal_posterior["goals"]["goal|matching"]["support"] == 1
    assert alignment_posterior["posterior"] > 0.0
    assert matching_posterior["posterior"] > 0.0
