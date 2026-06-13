from __future__ import annotations

from pathlib import Path

from src.persistent_memory import (
    PersistentMemoryState,
    canonical_observation_hash,
    shortest_action_path,
)
from src.run_unbroken import run_unbroken
from src.train import train_model
import torch


def test_fresh_state_initializes_transition_graph_fields() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=8, batch_size=1, device="cpu")
    graph = memory.transition_graph

    assert graph["schema"] == "runtime_transition_graph_v1"
    assert graph["nodes"] == {}
    assert graph["edges"] == {}
    assert graph["outgoing_edges"] == {}
    assert graph["outgoing_action_edges"] == {}
    assert graph["incoming_edges"] == {}
    assert graph["edge_index_count"] == 0
    assert graph["visited_sequence"] == []
    assert graph["last_node_hash"] is None
    assert graph["last_edge_id"] is None
    assert graph["observed_actions"] == {}
    assert graph["unexplored_actions"] == {}
    assert graph["progress_targets"] == {}
    assert graph["reversible_pairs"] == {}
    assert graph["undo_actions"] == {}
    assert graph["no_op_edge_count"] == 0
    assert graph["loop_edge_count"] == 0
    assert graph["reversible_edge_count"] == 0
    assert graph["undo_edge_count"] == 0


def test_append_event_records_exact_hashes_counts_and_unexplored_actions() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    observation = {"sensory": torch.tensor([0.1, 0.2]), "private": torch.tensor([3])}
    next_observation = {"sensory": torch.tensor([0.2, 0.1]), "private": torch.tensor([3])}
    metadata = {"available_action_mask": [True, False, True, False]}

    memory.append_event(
        tick=1,
        observation=observation,
        action=1,
        next_observation=next_observation,
        metadata=metadata,
    )

    graph = memory.transition_graph
    from_hash = canonical_observation_hash(observation)
    to_hash = canonical_observation_hash(next_observation)
    edge_id = f"{from_hash}|1|{to_hash}"

    assert from_hash in graph["nodes"]
    assert to_hash in graph["nodes"]
    assert graph["nodes"][from_hash]["hash"] == from_hash
    assert graph["nodes"][to_hash]["hash"] == to_hash
    assert graph["nodes"][from_hash]["visits"] == 1
    assert graph["nodes"][to_hash]["visits"] == 1

    edge = graph["edges"][edge_id]
    assert edge["from"] == from_hash
    assert edge["to"] == to_hash
    assert edge["action"] == "1"
    assert edge["count"] == 1
    assert edge["first_tick"] == 1
    assert edge["last_tick"] == 1
    assert edge["no_op"] is False
    assert edge["loop_observed"] is False
    assert edge["reversible"] is False
    assert graph["outgoing_edges"][from_hash] == [edge_id]
    assert graph["outgoing_action_edges"][from_hash]["1"] == [edge_id]
    assert graph["incoming_edges"][to_hash] == [edge_id]
    assert graph["edge_index_count"] == 1

    assert graph["observed_actions"][from_hash] == ["1"]
    assert graph["unexplored_actions"][from_hash] == ["0", "2"]


def test_transition_graph_indexes_progress_targets_from_score_and_terminal_edges() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    observation = {"sensory": torch.tensor([0.0])}
    next_observation = {"sensory": torch.tensor([1.0])}

    memory.append_event(
        tick=1,
        observation=observation,
        action=1,
        next_observation=next_observation,
        metadata={"score_delta": 2.0, "terminal": True, "available_action_mask": [True, True]},
    )

    graph = memory.transition_graph
    from_hash = canonical_observation_hash(observation)
    to_hash = canonical_observation_hash(next_observation)
    edge_id = f"{from_hash}|1|{to_hash}"

    assert graph["outgoing_edges"][from_hash] == [edge_id]
    assert graph["outgoing_action_edges"][from_hash]["1"] == [edge_id]
    assert graph["incoming_edges"][to_hash] == [edge_id]
    target = graph["progress_targets"][to_hash]
    assert target["state"] == to_hash
    assert target["source_edge"] == edge_id
    assert target["score_delta"] == 2.0
    assert target["terminal_rate"] == 1.0
    assert target["value"] == 3.25


def test_append_event_marks_no_op_edge_and_node_loop() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    observation = {"sensory": torch.tensor([0.5, 0.5]), "private": torch.tensor([1])}

    memory.append_event(
        tick=2,
        observation=observation,
        action=2,
        next_observation=observation,
        metadata={"available_action_mask": [True, True, True]},
    )

    graph = memory.transition_graph
    node_hash = canonical_observation_hash(observation)
    edge_id = f"{node_hash}|2|{node_hash}"
    edge = graph["edges"][edge_id]

    assert edge["no_op"] is True
    assert edge["count"] == 1
    assert graph["no_op_edge_count"] == 1
    assert graph["nodes"][node_hash]["visits"] == 2
    assert graph["loop_edge_count"] == 1


def test_append_event_marks_return_as_loop() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    first = {"sensory": torch.tensor([1.0, 0.0]), "private": torch.tensor([2])}
    second = {"sensory": torch.tensor([0.0, 1.0]), "private": torch.tensor([2])}

    memory.append_event(
        tick=1,
        observation=first,
        action=0,
        next_observation=second,
        metadata={"available_action_mask": [True, False, False]},
    )
    memory.append_event(
        tick=2,
        observation=second,
        action=1,
        next_observation=first,
        metadata={"available_action_mask": [True, False, False]},
    )

    graph = memory.transition_graph
    first_hash = canonical_observation_hash(first)
    second_hash = canonical_observation_hash(second)
    back_edge = graph["edges"][f"{second_hash}|1|{first_hash}"]

    assert back_edge["loop_observed"] is True
    assert graph["loop_edge_count"] == 1
    assert graph["nodes"][first_hash]["visits"] == 2
    assert graph["nodes"][second_hash]["visits"] == 1


def test_append_event_marks_reversible_edges() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    first = {"sensory": torch.tensor([0.3, 0.7]), "private": torch.tensor([0])}
    second = {"sensory": torch.tensor([0.7, 0.3]), "private": torch.tensor([0])}

    memory.append_event(
        tick=1,
        observation=first,
        action=3,
        next_observation=second,
        metadata={"available_action_mask": [True, True, True, True]},
    )
    memory.append_event(
        tick=2,
        observation=second,
        action=2,
        next_observation=first,
        metadata={"available_action_mask": [True, True, True, True]},
    )

    graph = memory.transition_graph
    first_hash = canonical_observation_hash(first)
    second_hash = canonical_observation_hash(second)
    forward_edge = graph["edges"][f"{first_hash}|3|{second_hash}"]
    reverse_edge = graph["edges"][f"{second_hash}|2|{first_hash}"]

    assert forward_edge["reversible"] is True
    assert reverse_edge["reversible"] is True
    assert graph["reversible_edge_count"] == 1
    assert len(graph["reversible_pairs"]) == 1
    assert graph["outgoing_edges"][first_hash] == [f"{first_hash}|3|{second_hash}"]
    assert graph["outgoing_edges"][second_hash] == [f"{second_hash}|2|{first_hash}"]
    assert graph["outgoing_action_edges"][first_hash]["3"] == [f"{first_hash}|3|{second_hash}"]
    assert graph["outgoing_action_edges"][second_hash]["2"] == [f"{second_hash}|2|{first_hash}"]
    assert graph["incoming_edges"][first_hash] == [f"{second_hash}|2|{first_hash}"]
    assert graph["incoming_edges"][second_hash] == [f"{first_hash}|3|{second_hash}"]
    assert graph["edge_index_count"] == 2


def test_transition_graph_finds_shortest_observed_action_path() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    first = {"sensory": torch.tensor([0.0, 0.0]), "private": torch.tensor([0])}
    second = {"sensory": torch.tensor([1.0, 0.0]), "private": torch.tensor([0])}
    third = {"sensory": torch.tensor([1.0, 1.0]), "private": torch.tensor([0])}

    memory.append_event(
        tick=1,
        observation=first,
        action=2,
        next_observation=second,
        metadata={"available_action_mask": [True, True, True]},
    )
    memory.append_event(
        tick=2,
        observation=second,
        action=1,
        next_observation=third,
        metadata={"available_action_mask": [True, True, True]},
    )
    memory.append_event(
        tick=3,
        observation=first,
        action=0,
        next_observation=third,
        metadata={"available_action_mask": [True, True, True]},
    )

    first_hash = canonical_observation_hash(first)
    second_hash = canonical_observation_hash(second)
    third_hash = canonical_observation_hash(third)

    assert shortest_action_path(memory.transition_graph, first_hash, second_hash) == ["2"]
    assert shortest_action_path(memory.transition_graph, first_hash, third_hash) == ["0"]
    assert shortest_action_path(memory.transition_graph, third_hash, first_hash) is None


def test_transition_graph_round_trips_with_memory_save_load(tmp_path: Path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    observation = {"sensory": torch.tensor([0.4, 0.6]), "private": torch.tensor([2])}
    next_observation = {"sensory": torch.tensor([0.6, 0.4]), "private": torch.tensor([2])}

    memory.append_event(
        tick=7,
        observation=observation,
        action=4,
        next_observation=next_observation,
        metadata={"available_action_mask": [True, True, True, True, True]},
    )

    path = tmp_path / "tmp_transition_memory.pt"
    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=4, batch_size=1, device="cpu")

    assert loaded.transition_graph == memory.transition_graph


def test_run_unbroken_persists_transition_graph_with_expected_node_visits(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runtime.ckpt"
    train_model("smoke", output=checkpoint, steps=1, device="cpu")
    memory_file = tmp_path / "runtime_memory.pt"
    max_ticks = 6

    stats = run_unbroken(checkpoint, max_ticks=max_ticks, log_every=0, device="cpu", memory_file=memory_file)
    payload = torch.load(memory_file, map_location="cpu")
    graph = payload["transition_graph"]

    assert stats["event_journal_length"] == float(max_ticks)
    assert graph["schema"] == "runtime_transition_graph_v1"
    assert len(graph["edges"]) >= 1
    assert len(graph["nodes"]) >= 2
    assert sum(node["visits"] for node in graph["nodes"].values()) == max_ticks + 1
    assert len(graph["visited_sequence"]) == max_ticks + 1
    assert graph["visited_sequence"][-1] == graph["last_node_hash"]
