from __future__ import annotations

import math
from pathlib import Path

import torch

from src.persistent_memory import PersistentMemoryState
from src.run_unbroken import run_unbroken
from src.train import train_model


def test_fresh_state_initializes_plastic_memory_schema_and_bounds() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=8, batch_size=1, device="cpu")
    plastic = memory.plastic_memory

    assert plastic["schema"] == "runtime_plastic_memory_v1"
    assert plastic["capacity"] > 0
    assert plastic["updates"] == 0
    assert plastic["key_values"] == []
    assert plastic["hebbian_traces"] == {}

    adapter = plastic["low_rank_adapter"]
    assert adapter["rank"] > 0
    assert adapter["rank"] <= plastic["capacity"]
    assert adapter["action_factors"] == {}
    assert adapter["field_factors"] == {}
    assert adapter["norm"] == 0.0


def test_append_event_updates_plastic_memory_without_mutating_latent_and_records_traces() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    latent_before = memory.latent.clone()
    private_before = memory.private_token.clone()

    observation = {"sensory": torch.tensor([1.0, 2.0]), "private": torch.tensor([1])}
    next_observation = {"sensory": torch.tensor([1.5, 2.0]), "private": torch.tensor([1])}

    record = memory.append_event(
        tick=1,
        observation=observation,
        action=3,
        next_observation=next_observation,
        metadata={"score_delta": 0.25, "terminal": False},
        prediction_error={"uncertainty": 0.0},
    )

    assert record["action"] == "3"
    assert memory.tick == 0
    assert torch.equal(memory.latent, latent_before)
    assert torch.equal(memory.private_token, private_before)

    plastic = memory.plastic_memory
    assert plastic["updates"] == 1
    assert len(plastic["key_values"]) == 1

    trace = plastic["key_values"][-1]
    assert trace["tick"] == 1
    assert trace["action"] == "3"
    assert trace["changed_fields"] == ["sensory"]
    assert math.isfinite(trace["neuromodulation"])
    assert trace["neuromodulation"] > 0.0

    key = "action:3|field:sensory"
    assert key in plastic["hebbian_traces"]
    assert plastic["hebbian_traces"][key] > 0.0

    adapter = plastic["low_rank_adapter"]
    assert "3" in adapter["action_factors"]
    assert "sensory" in adapter["field_factors"]
    assert adapter["norm"] > 0.0
    assert any(value != 0.0 for value in adapter["action_factors"]["3"])
    assert any(value != 0.0 for value in adapter["field_factors"]["sensory"])


def test_plastic_memory_key_value_trace_capacity_is_enforced() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    memory.plastic_memory["capacity"] = 3

    for tick in range(1, 7):
        value = float(tick)
        memory.append_event(
            tick=tick,
            observation={"sensory": torch.tensor([value, 2.0]), "private": torch.tensor([1])},
            action=tick % 2,
            next_observation={"sensory": torch.tensor([value + 1.0, 2.0]), "private": torch.tensor([1])},
            metadata={"score_delta": 0.0, "terminal": False},
        )

    plastic = memory.plastic_memory
    assert plastic["updates"] == 6
    assert len(plastic["key_values"]) == 3
    assert plastic["key_values"][0]["tick"] == 4
    assert plastic["key_values"][-1]["tick"] == 6


def test_neuromodulation_increases_with_prediction_error_or_score_delta() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")

    observation = {"sensory": torch.tensor([1.0, 2.0]), "private": torch.tensor([1])}
    memory.append_event(
        tick=1,
        observation=observation,
        action=1,
        next_observation=observation,
        metadata={"score_delta": 0.0, "terminal": False},
    )
    baseline = memory.plastic_memory["key_values"][-1]["neuromodulation"]

    memory.append_event(
        tick=2,
        observation=observation,
        action=1,
        next_observation={"sensory": torch.tensor([1.1, 2.0]), "private": torch.tensor([1])},
        metadata={"score_delta": 0.0, "terminal": False},
        prediction_error={"action_uncertainty": 0.5},
    )
    with_prediction_error = memory.plastic_memory["key_values"][-1]["neuromodulation"]

    memory.append_event(
        tick=3,
        observation={"sensory": torch.tensor([1.1, 2.0]), "private": torch.tensor([1])},
        action=1,
        next_observation={"sensory": torch.tensor([1.2, 2.0]), "private": torch.tensor([1])},
        metadata={"score_delta": 1.0, "terminal": False},
        prediction_error={"action_uncertainty": 0.0},
    )
    with_reward = memory.plastic_memory["key_values"][-1]["neuromodulation"]

    assert baseline > 0.0
    assert with_prediction_error > baseline
    assert with_reward > baseline


def test_plastic_memory_save_and_load_roundtrip(tmp_path: Path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=6, batch_size=1, device="cpu")
    memory.append_event(
        tick=1,
        observation={"sensory": torch.tensor([1.0, 2.0])},
        action=2,
        next_observation={"sensory": torch.tensor([1.1, 2.0])},
        metadata={"score_delta": 0.2, "terminal": False},
    )

    path = tmp_path / "plastic_memory.pt"
    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=6, batch_size=1, device="cpu")

    assert loaded.plastic_memory == memory.plastic_memory


def test_run_unbroken_roundtrip_persists_plastic_memory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runtime.ckpt"
    memory_file = tmp_path / "runtime_memory.pt"
    train_model("smoke", output=checkpoint, steps=1, device="cpu")

    max_ticks = 6
    stats = run_unbroken(checkpoint, max_ticks=max_ticks, log_every=0, device="cpu", memory_file=memory_file)

    assert stats["event_journal_length"] == float(max_ticks)

    payload = torch.load(memory_file, map_location="cpu")
    plastic = payload["plastic_memory"]
    assert plastic["schema"] == "runtime_plastic_memory_v1"
    assert plastic["updates"] == max_ticks
    assert len(plastic["key_values"]) == max_ticks
    assert len(plastic["hebbian_traces"]) > 0
    assert plastic["low_rank_adapter"]["norm"] > 0.0
