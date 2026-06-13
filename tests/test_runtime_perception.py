from __future__ import annotations

from pathlib import Path

import torch

from src.grid_perception import raw_grid_hash
from src.persistent_memory import PersistentMemoryState


def test_append_event_uses_runtime_frame_perception_for_2d_grids() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    observation = {
        "grid": torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
    }
    next_observation = {
        "grid": torch.tensor(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 1, 0],
                [0, 0, 0, 0],
            ],
            dtype=torch.long,
        )
    }

    record = memory.append_event(
        tick=1,
        observation=observation,
        action="down",
        next_observation=next_observation,
    )

    perception = record["perception"]
    assert record["perception"]["schema"] == "runtime_frame_perception_v1"
    assert set(perception["current_frame"].keys()) >= {
        "raw_hash",
        "objects",
        "relation_graph",
        "salience_map",
        "edit_script",
    }
    assert set(perception["next_frame"].keys()) >= {
        "raw_hash",
        "objects",
        "relation_graph",
        "salience_map",
        "edit_script",
    }

    assert perception["current_frame"]["raw_hash"] == raw_grid_hash(observation["grid"])
    assert perception["next_frame"]["raw_hash"] == raw_grid_hash(next_observation["grid"])
    assert isinstance(perception["current_frame"]["objects"], list)
    assert isinstance(perception["next_frame"]["objects"], list)
    assert len(perception["next_frame"]["edit_script"]["moves"]) == 1
    assert perception["next_frame"]["edit_script"]["changed_count"] > 0
    assert perception["current_frame"]["edit_script"]["changed_count"] == 0

    move = perception["next_frame"]["edit_script"]["moves"][0]
    assert move["color"] == 1
    assert move["delta"] == [1.0, 0.0]
    assert move["to_centroid"][0] - move["from_centroid"][0] == 1.0
    assert move["to_centroid"][1] == move["from_centroid"][1]


def test_append_event_perception_payload_round_trips_with_memory_save_load(tmp_path: Path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    memory.append_event(
        tick=3,
        observation={"grid": torch.ones((3, 3), dtype=torch.long)},
        action="up",
        next_observation={"grid": torch.tensor([[0, 0, 0], [0, 2, 2], [2, 2, 2]], dtype=torch.long)},
    )

    path = tmp_path / "runtime_memory.pt"
    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=4, batch_size=1, device="cpu")

    assert loaded.event_journal == memory.event_journal
    loaded_perception = loaded.event_journal[0]["perception"]
    assert loaded_perception["current_frame"]["raw_hash"] == memory.event_journal[0]["perception"]["current_frame"]["raw_hash"]
    assert loaded_perception["next_frame"]["raw_hash"] == memory.event_journal[0]["perception"]["next_frame"]["raw_hash"]
    assert loaded_perception["next_frame"]["edit_script"]["changed_count"] == memory.event_journal[0]["perception"]["next_frame"]["edit_script"]["changed_count"]


def test_float_batch_sensory_vector_is_not_treated_as_grid_frame() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    observation = {"sensory": torch.zeros((1, 23), dtype=torch.float32)}
    next_observation = {"sensory": torch.ones((1, 23), dtype=torch.float32)}

    record = memory.append_event(
        tick=1,
        observation=observation,
        action=0,
        next_observation=next_observation,
    )

    assert "perception" not in record
    assert all("grid_hash" not in trace for trace in memory.controllability_model["intervention_traces"])
