from __future__ import annotations

from pathlib import Path

import torch

from src.persistent_memory import PersistentMemoryState


def _container_counts(memory: PersistentMemoryState) -> dict[str, int]:
    transition_graph = memory.transition_graph
    semantic_memory = memory.semantic_memory
    plastic_memory = memory.plastic_memory
    hypothesis_posterior = memory.hypothesis_posterior
    active_theory_set = memory.active_theory_set
    goal_posterior = memory.goal_posterior
    coordinate_affordances = memory.coordinate_affordances
    macro_policy_library = memory.macro_policy_library
    controllability_model = memory.controllability_model
    progress_value_model = memory.progress_value_model
    return {
        "event_journal": len(memory.event_journal),
        "transition_nodes": len(transition_graph.get("nodes", {})),
        "transition_edges": len(transition_graph.get("edges", {})),
        "semantic_action_facts": len(semantic_memory.get("action_facts", {})),
        "semantic_invariants": len(semantic_memory.get("invariants", {})),
        "semantic_facts": len(semantic_memory.get("facts", {})),
        "plastic_updates": int(plastic_memory.get("updates", 0)),
        "plastic_key_values": len(plastic_memory.get("key_values", [])),
        "hypotheses": len(hypothesis_posterior.get("hypotheses", {})),
        "active_theories": len(active_theory_set.get("theories", [])),
        "goals": len(goal_posterior.get("goals", {})),
        "subgoals": len(goal_posterior.get("subgoals", {})),
        "coordinate_updates": int(coordinate_affordances.get("updates", 0)),
        "coordinate_candidates": len(coordinate_affordances.get("candidates", {})),
        "macro_policies": len(macro_policy_library.get("macros", {})),
        "macro_consolidations": int(macro_policy_library.get("consolidations", 0)),
        "controlled_objects": len(controllability_model.get("controlled_objects", {})),
        "controlled_variables": len(controllability_model.get("controlled_variables", {})),
        "progress_value_actions": len(progress_value_model.get("action_values", {})),
        "progress_value_sequences": len(progress_value_model.get("sequence_values", {})),
    }


def _sensor_grid(a: float, b: float, c: float, d: float) -> dict[str, torch.Tensor]:
    return {
        "grid": torch.tensor([[a, b], [c, d]], dtype=torch.long),
        "sensory": torch.tensor([float(a), float(b)], dtype=torch.float32),
    }


def test_fresh_memory_has_expected_context_state_schema() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    context = memory.context_state

    assert context["schema"] == "runtime_context_state_v1"
    assert context["game_id"] is None
    assert context["level_index"] == 0
    assert context["episode_index"] == 0
    assert context["phase"] == "running"
    assert context["uncertainty"] == 1.0
    assert set(context["confidence_gates"]) == {
        "boundary_count",
        "fragile_hypothesis_decay",
        "fragile_goal_decay",
        "verified_min_support",
    }
    assert context["boundaries"] == []
    assert context["last_terminal"] is False
    assert context["last_boundary"] is None


def test_append_event_without_boundary_keeps_context_running_and_preserves_containers() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = _sensor_grid(0.0, 0.0, 1.0, 1.0)
    after = _sensor_grid(0.0, 1.0, 1.0, 1.0)
    metadata = {"terminal": False, "available_action_mask": [True, True, True], "score_delta": 0.25}

    memory.append_event(
        tick=1,
        observation=before,
        action=1,
        next_observation=after,
        metadata=metadata,
    )

    baseline = _container_counts(memory)
    assert baseline["transition_nodes"] > 0
    assert baseline["transition_edges"] > 0
    assert baseline["semantic_action_facts"] > 0
    assert baseline["plastic_updates"] > 0

    next_observation = _sensor_grid(0.0, 1.0, 1.0, 0.0)
    memory.append_event(
        tick=2,
        observation=after,
        action=2,
        next_observation=next_observation,
        metadata=metadata,
    )
    after_counts = _container_counts(memory)

    assert memory.context_state["game_id"] is None
    assert memory.context_state["level_index"] == 0
    assert memory.context_state["episode_index"] == 0
    assert memory.context_state["phase"] == "running"
    assert memory.context_state["last_terminal"] is False
    assert memory.context_state["last_boundary"] is None
    assert memory.context_state["boundaries"] == []

    for key in baseline:
        assert after_counts[key] >= baseline[key] >= 0
    assert len(memory.event_journal) == 2


def test_append_event_with_episode_boundary_records_transform_and_preserved_counts() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = _sensor_grid(0.0, 0.0, 1.0, 0.0)
    mid = _sensor_grid(0.0, 1.0, 1.0, 0.0)
    terminal = _sensor_grid(1.0, 1.0, 1.0, 0.0)
    metadata = {"terminal": False, "available_action_mask": [True, False, True], "score_delta": 0.0}

    memory.append_event(
        tick=1,
        observation=before,
        action=0,
        next_observation=mid,
        metadata=metadata,
    )
    pre_boundary_counts = _container_counts(memory)
    pre_episode = memory.context_state["episode_index"]
    pre_level = memory.context_state["level_index"]

    boundary_metadata = {
        "terminal": True,
        "boundary": "episode",
        "game_id": "unit-test-game",
        "score_delta": 1.0,
        "available_action_mask": [True, True, True],
    }
    memory.append_event(
        tick=2,
        observation=mid,
        action=0,
        next_observation=terminal,
        metadata=boundary_metadata,
    )

    boundary_record = memory.context_state["boundaries"][-1]
    assert boundary_record["schema"] == "runtime_boundary_transform_v1"
    assert boundary_record["boundary"] == "episode"
    assert boundary_record["terminal"] is True
    assert boundary_record["before"]["episode_index"] == pre_episode
    assert boundary_record["after"]["episode_index"] == pre_episode + 1
    assert boundary_record["before"]["level_index"] == pre_level
    assert boundary_record["after"]["level_index"] == pre_level
    assert memory.context_state["episode_index"] == pre_episode + 1
    assert memory.context_state["level_index"] == pre_level
    assert memory.context_state["phase"] == "episode_boundary"
    assert memory.context_state["last_boundary"] == "episode"
    assert memory.context_state["last_terminal"] is True

    preserved = boundary_record["preserved_counts"]
    after_boundary_counts = _container_counts(memory)
    assert preserved == after_boundary_counts
    for key in pre_boundary_counts:
        assert preserved[key] >= pre_boundary_counts[key]
    assert len(memory.context_state["boundaries"]) == 1
    assert memory.context_state["boundaries"][0]["boundary"] == "episode"
    assert memory.context_state["game_id"] == "unit-test-game"


def test_level_boundary_with_win_increments_level_and_records_last_boundary() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    memory.append_event(
        tick=1,
        observation=_sensor_grid(0.0, 0.0, 1.0, 1.0),
        action=3,
        next_observation=_sensor_grid(1.0, 0.0, 1.0, 1.0),
        metadata={"terminal": False, "available_action_mask": [True, True], "score_delta": 0.4},
    )
    pre_level = memory.context_state["level_index"]
    pre_episode = memory.context_state["episode_index"]

    memory.append_event(
        tick=2,
        observation=_sensor_grid(1.0, 0.0, 1.0, 1.0),
        action=1,
        next_observation=_sensor_grid(1.0, 0.0, 0.0, 1.0),
        metadata={
            "terminal": True,
            "boundary": "win",
            "available_action_mask": [True, True],
            "score_delta": 1.0,
        },
    )

    boundary_record = memory.context_state["boundaries"][-1]
    assert boundary_record["boundary"] == "win"
    assert memory.context_state["level_index"] == pre_level + 1
    assert memory.context_state["episode_index"] == pre_episode
    assert memory.context_state["phase"] == "win"
    assert memory.context_state["last_boundary"] == "win"


def test_boundary_transform_lowers_fragile_confidence_without_deleting_verified_entries() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    memory.hypothesis_posterior["hypotheses"] = {
        "fragile-h": {
            "id": "fragile-h",
            "family": "manual",
            "action": "not-boundary-action",
            "support": 1,
            "counterexamples": 0,
            "description_length": 1.0,
            "score": 0.0,
            "posterior": 0.5,
            "context_confidence": 1.0,
        },
        "verified-h": {
            "id": "verified-h",
            "family": "manual",
            "action": "not-boundary-action",
            "support": 3,
            "counterexamples": 0,
            "description_length": 1.0,
            "score": 0.0,
            "posterior": 0.5,
            "context_confidence": 1.0,
        },
    }
    memory.goal_posterior["goals"] = {
        "fragile-g": {
            "id": "fragile-g",
            "scope": "goal",
            "kind": "manual",
            "selector": {},
            "progress_test": {},
            "support": 1,
            "inconsistency": 0,
            "description_length": 1.0,
            "score": 0.0,
            "posterior": 0.5,
            "context_confidence": 1.0,
        },
        "verified-g": {
            "id": "verified-g",
            "scope": "goal",
            "kind": "manual",
            "selector": {},
            "progress_test": {},
            "support": 3,
            "inconsistency": 0,
            "description_length": 1.0,
            "score": 0.0,
            "posterior": 0.5,
            "context_confidence": 1.0,
        },
    }

    memory.append_event(
        tick=1,
        observation=_sensor_grid(0.0, 0.0, 1.0, 1.0),
        action=2,
        next_observation=_sensor_grid(1.0, 0.0, 1.0, 1.0),
        metadata={"terminal": True, "boundary": "episode", "score_delta": 1.0},
    )

    boundary_record = memory.context_state["boundaries"][-1]
    hypotheses = memory.hypothesis_posterior["hypotheses"]
    goals = memory.goal_posterior["goals"]

    assert "fragile-h" in hypotheses
    assert "verified-h" in hypotheses
    assert "fragile-g" in goals
    assert "verified-g" in goals
    assert hypotheses["fragile-h"]["context_confidence"] < 1.0
    assert hypotheses["verified-h"]["context_confidence"] == 1.0
    assert goals["fragile-g"]["context_confidence"] < 1.0
    assert goals["verified-g"]["context_confidence"] == 1.0
    assert "fragile-h" in boundary_record["confidence_decay"]["hypotheses"]
    assert "verified-h" not in boundary_record["confidence_decay"]["hypotheses"]
    assert "fragile-g" in boundary_record["confidence_decay"]["goals"]
    assert "verified-g" not in boundary_record["confidence_decay"]["goals"]


def test_context_state_round_trips_through_save_load(tmp_path: Path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=8, batch_size=1, device="cpu")
    memory.append_event(
        tick=1,
        observation=_sensor_grid(0.0, 0.0, 1.0, 0.0),
        action=4,
        next_observation=_sensor_grid(0.0, 1.0, 1.0, 0.0),
        metadata={"terminal": False, "available_action_mask": [True, True], "score_delta": 0.2},
    )
    memory.append_event(
        tick=2,
        observation=_sensor_grid(0.0, 1.0, 1.0, 0.0),
        action=4,
        next_observation=_sensor_grid(1.0, 1.0, 1.0, 1.0),
        metadata={
            "terminal": True,
            "boundary": "episode",
            "game_id": "roundtrip-game",
            "score_delta": 2.0,
            "available_action_mask": [True, True],
        },
    )
    assert memory.context_state["phase"] == "episode_boundary"
    assert memory.context_state["last_boundary"] == "episode"
    assert memory.context_state["boundaries"]

    path = tmp_path / "context_memory.pt"
    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=8, batch_size=1, device="cpu")

    assert loaded.context_state == memory.context_state
    assert loaded.context_state["game_id"] == "roundtrip-game"
    assert loaded.context_state["last_boundary"] == "episode"
    assert loaded.context_state["boundaries"] == memory.context_state["boundaries"]
    assert loaded.context_state["phase"] == "episode_boundary"
