from __future__ import annotations

import pytest
import torch

from src.action_selection import select_experimental_action
from src.persistent_memory import PersistentMemoryState, canonical_observation_hash


class _FakeMemory:
    def __init__(
        self,
        transition_graph: dict[str, object] | None = None,
        semantic_memory: dict[str, object] | None = None,
        hypothesis_posterior: dict[str, object] | None = None,
        goal_posterior: dict[str, object] | None = None,
        macro_policy_library: dict[str, object] | None = None,
        progress_value_model: dict[str, object] | None = None,
    ) -> None:
        self.transition_graph = transition_graph or {}
        self.semantic_memory = semantic_memory or {}
        self.hypothesis_posterior = hypothesis_posterior or {}
        self.goal_posterior = goal_posterior or {}
        self.macro_policy_library = macro_policy_library or {}
        self.progress_value_model = progress_value_model or {}


def _make_observations() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    first = {"sensory": torch.tensor([0.2, 0.8]), "private": torch.tensor([1])}
    second = {"sensory": torch.tensor([0.8, 0.2]), "private": torch.tensor([1])}
    return first, second


def test_fresh_state_exposes_undo_tracking_fields() -> None:
    graph = PersistentMemoryState.fresh(hidden_dim=8, batch_size=1, device="cpu").transition_graph

    assert graph["schema"] == "runtime_transition_graph_v1"
    assert graph["undo_actions"] == {}
    assert graph["undo_edge_count"] == 0


def test_append_event_records_undo_action_7_evidence_and_reversibility() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    first, second = _make_observations()

    memory.append_event(
        tick=1,
        observation=first,
        action=2,
        next_observation=second,
        metadata={"available_action_mask": [True] * 8},
    )
    memory.append_event(
        tick=2,
        observation=second,
        action=7,
        next_observation=first,
        metadata={"available_action_mask": [True] * 8},
    )

    graph = memory.transition_graph
    first_hash = canonical_observation_hash(first)
    second_hash = canonical_observation_hash(second)
    forward_edge_id = f"{first_hash}|2|{second_hash}"
    undo_edge_id = f"{second_hash}|7|{first_hash}"

    assert graph["edges"][forward_edge_id]["reversible"] is True
    assert graph["edges"][undo_edge_id]["reversible"] is True
    assert graph["reversible_edge_count"] == 1

    undo_entry = graph["undo_actions"]["7"]
    assert undo_entry["action"] == "7"
    assert undo_entry["support"] == 1
    assert undo_entry["confidence"] == pytest.approx(0.5)
    assert undo_entry["undo_edges"] == [undo_edge_id]
    assert undo_entry["reverses_edges"] == [forward_edge_id]

    forward_evidence_key = f"{undo_edge_id}<-{forward_edge_id}"
    evidence = undo_entry["evidence"]
    assert forward_evidence_key in evidence
    assert evidence[forward_evidence_key]["undo_edge"] == undo_edge_id
    assert evidence[forward_evidence_key]["reverses_edge"] == forward_edge_id


def test_save_load_preserves_undo_action_evidence(tmp_path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    first, second = _make_observations()

    memory.append_event(
        tick=1,
        observation=first,
        action=2,
        next_observation=second,
        metadata={"available_action_mask": [True] * 8},
    )
    memory.append_event(
        tick=2,
        observation=second,
        action=7,
        next_observation=first,
        metadata={"available_action_mask": [True] * 8},
    )

    memory_file = tmp_path / "undo_memory.pt"
    memory.save(memory_file)
    loaded = PersistentMemoryState.load(memory_file, hidden_dim=4, batch_size=1, device="cpu")

    assert loaded.transition_graph["undo_actions"] == memory.transition_graph["undo_actions"]
    assert loaded.transition_graph["undo_edge_count"] == memory.transition_graph["undo_edge_count"]


def test_select_experimental_action_prefers_verified_undo_action_7_with_diagnostics() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    first, second = _make_observations()
    memory.append_event(
        tick=1,
        observation=first,
        action=2,
        next_observation=second,
        metadata={"available_action_mask": [True] * 8},
    )
    memory.append_event(
        tick=2,
        observation=second,
        action=7,
        next_observation=first,
        metadata={"available_action_mask": [True] * 8},
    )

    logits = torch.tensor([0.0] * 7 + [2.0], dtype=torch.float32)
    selected, diagnostics = select_experimental_action(
        logits,
        _FakeMemory(
            transition_graph=memory.transition_graph,
            semantic_memory={},
            hypothesis_posterior={},
            goal_posterior={},
            macro_policy_library={},
            progress_value_model={},
        ),
        second,
        available_action_mask=[True] * 8,
    )
    selected_row = next(item for item in diagnostics["components"] if item["action"] == selected)

    assert selected == 7
    assert diagnostics["candidate_count"] == 8
    assert diagnostics["state_hash"] == canonical_observation_hash(second)
    assert selected_row["observed_rank"] == "observed_undo"
    assert selected_row["undo_candidate"] is True
    assert selected_row["undo_confidence"] > 0.0
    assert selected_row["undo_probe"] is False
    assert selected_row["score"] == max(component["score"] for component in diagnostics["components"])
