from __future__ import annotations

from pathlib import Path

import torch

from src.persistent_memory import PersistentMemoryState
from src.run_unbroken import run_unbroken
from src.train import train_model


def test_fresh_memory_has_empty_event_journal() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=12, batch_size=3, device="cpu")

    assert memory.event_journal == []
    assert memory.tick == 0


def test_append_event_records_json_like_payload() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    observation = {
        "sensory": torch.tensor([0.2, 0.4], dtype=torch.float32),
        "private": torch.tensor([1], dtype=torch.long),
    }
    next_observation = {
        "sensory": torch.tensor([0.5, 0.1], dtype=torch.float32),
        "private": torch.tensor([2], dtype=torch.long),
    }
    metadata = {"available_action_mask": [True, False, True, True, False], "score_delta": 1.25, "terminal": False}
    prediction_error = {"action_uncertainty": 0.42}

    record = memory.append_event(
        tick=7,
        observation=observation,
        action=3,
        next_observation=next_observation,
        metadata=metadata,
        prediction_error=prediction_error,
    )

    assert len(memory.event_journal) == 1
    assert memory.event_journal[-1] is record
    assert set(["observation", "action", "next_observation", "delta", "metadata", "prediction_error"]).issubset(record.keys())
    assert record["tick"] == 7
    assert record["action"] == "3"
    assert record["observation"]["sensory"]["dtype"] == "float32"
    assert record["next_observation"]["private"]["dtype"] == "int64"
    assert "sensory" in record["delta"]
    assert record["metadata"]["available_action_mask"] == [True, False, True, True, False]


def test_save_and_load_round_trip_includes_event_journal(tmp_path: Path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=8, batch_size=2, device="cpu")
    memory.update(
        latent=torch.tensor(
            [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8], [0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1]],
            dtype=torch.float32,
        ),
        private_token=torch.tensor([2, 3], dtype=torch.long),
        tick=17,
    )
    memory.append_event(
        tick=17,
        observation={"sensory": torch.tensor([1.0, 2.0])},
        action=1,
        next_observation={"sensory": torch.tensor([2.0, 4.0])},
        metadata={"terminal": False},
        prediction_error={"dummy": 0.1},
    )

    path = tmp_path / "tmp_memory.pt"
    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=8, batch_size=2, device="cpu")

    assert loaded.tick == 17
    assert torch.allclose(loaded.latent, memory.latent.to(loaded.latent.device))
    assert torch.equal(loaded.private_token, memory.private_token.to(loaded.private_token.device))
    assert loaded.event_journal == memory.event_journal


def test_run_unbroken_persists_event_journal_records(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runtime.ckpt"
    train_model("smoke", output=checkpoint, steps=1, device="cpu")
    memory_file = tmp_path / "runtime_memory.pt"

    stats = run_unbroken(checkpoint, max_ticks=8, log_every=0, device="cpu", memory_file=memory_file)
    assert stats["event_journal_length"] == 8.0

    payload = torch.load(memory_file, map_location="cpu")
    events = payload["event_journal"]
    assert len(events) == 8
    for idx, event in enumerate(events, start=1):
        assert event["tick"] == idx
        assert event["action"] in {"0", "1", "2", "3", "4"}
        metadata = event["metadata"]
        assert metadata["available_action_mask"] == [True, True, True, True, True]
        assert isinstance(metadata["score_delta"], float)
        assert isinstance(metadata["terminal"], bool)
        assert {"observation", "next_observation", "delta", "prediction_error"}.issubset(event.keys())
