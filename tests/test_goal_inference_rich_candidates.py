from __future__ import annotations

import torch

from src.persistent_memory import PersistentMemoryState


def _goal_fact(memory: PersistentMemoryState, key: str) -> dict:
    return memory.semantic_memory["goal_facts"][key]


def _matches_kind(posterior: dict, scope: str, kind: str) -> bool:
    source = posterior["goals"] if scope == "goal" else posterior["subgoals"]
    return any(entry["kind"] == kind for entry in source.values())


def _candidate_entry(posterior: dict, scope: str, kind: str) -> dict | None:
    source = posterior["goals"] if scope == "goal" else posterior["subgoals"]
    for entry in source.values():
        if entry["kind"] == kind:
            return entry
    return None


def test_ordering_candidate_appears_for_ordered_layout_change() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = torch.tensor(
        [
            [0, 1, 0],
            [0, 0, 2],
            [3, 0, 0],
        ],
        dtype=torch.long,
    )
    after = torch.tensor(
        [
            [1, 2, 3],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=torch.long,
    )

    memory.append_event(
        tick=1,
        observation={"grid": before},
        action=0,
        next_observation={"grid": after},
        metadata={"score_delta": 0.0, "terminal": False},
    )

    candidate = _candidate_entry(memory.goal_posterior, "goal", "sorting")

    assert candidate is not None
    assert candidate["selector"] == {"axis": "horizontal", "key": "color", "direction": "ascending"}
    assert candidate["progress_test"] == {
        "ordered_sequence": {"axis": "horizontal", "key": "color", "direction": "ascending"}
    }
    fact = _goal_fact(memory, 'goal_candidate:sorting:{"axis":"horizontal","direction":"ascending","key":"color"}')
    assert fact["kind"] == "sorting"
    assert fact["selector"] == candidate["selector"]
    assert fact["support"] == 1


def test_containment_candidate_appears_when_object_nests_inside_another() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0],
            [0, 2, 0, 2, 0],
            [0, 2, 2, 2, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    after = torch.tensor(
        [
            [0, 0, 0, 0, 0],
            [0, 2, 2, 2, 0],
            [0, 2, 1, 2, 0],
            [0, 2, 2, 2, 0],
            [0, 0, 0, 0, 0],
        ],
        dtype=torch.long,
    )

    memory.append_event(
        tick=1,
        observation={"grid": before},
        action=0,
        next_observation={"grid": after},
        metadata={"score_delta": 0.0, "terminal": False},
    )

    assert _matches_kind(memory.goal_posterior, "goal", "containment")
    candidate = _candidate_entry(memory.goal_posterior, "goal", "containment")
    assert candidate is not None
    assert candidate["selector"] == {"relation": "contains"}
    assert candidate["progress_test"] == {"relation_present": "contains"}
    fact = _goal_fact(memory, 'goal_candidate:containment:{"relation":"contains"}')
    assert fact["kind"] == "containment"
    assert fact["progress_test"] == {"relation_present": "contains"}


def test_transformation_closure_candidate_appears_for_repetition_pattern_change() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = torch.tensor(
        [
            [0, 2, 2, 0],
            [0, 0, 0, 0],
        ],
        dtype=torch.long,
    )
    after = torch.tensor(
        [
            [0, 3, 3, 0],
            [0, 2, 2, 0],
        ],
        dtype=torch.long,
    )

    memory.append_event(
        tick=1,
        observation={"grid": before},
        action=0,
        next_observation={"grid": after},
        metadata={"score_delta": 0.0, "terminal": False},
    )

    assert _matches_kind(memory.goal_posterior, "goal", "transformation_closure")
    candidate = _candidate_entry(memory.goal_posterior, "goal", "transformation_closure")
    assert candidate is not None
    assert candidate["selector"] == {"mapping": {"2": 3}}
    assert candidate["progress_test"] == {"consistent_color_mapping": {"2": 3}}
    fact = _goal_fact(memory, 'goal_candidate:transformation_closure:{"mapping":{"2":3}}')
    assert fact["kind"] == "transformation_closure"
    assert fact["selector"] == {"mapping": {"2": 3}}


def test_goal_support_for_sorted_candidates_is_order_sensitive_and_deterministic() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = torch.tensor(
        [
            [0, 1, 0],
            [0, 0, 2],
            [3, 0, 0],
        ],
        dtype=torch.long,
    )
    after = torch.tensor(
        [
            [1, 2, 3],
            [0, 0, 0],
            [0, 0, 0],
        ],
        dtype=torch.long,
    )

    for tick in (1, 2):
        memory.append_event(
            tick=tick,
            observation={"grid": before if tick == 1 else after},
            action=0,
            next_observation={"grid": after},
            metadata={"score_delta": 0.0, "terminal": False},
        )

    candidate = _candidate_entry(memory.goal_posterior, "goal", "sorting")
    assert candidate is not None
    assert candidate["support"] == 2
    assert candidate["inconsistency"] == 0
    assert candidate["posterior"] >= 0.0
    top_goal_ids = [row["id"] for row in memory.goal_posterior["top_goals"]]
    assert candidate["id"] in top_goal_ids[:8]
