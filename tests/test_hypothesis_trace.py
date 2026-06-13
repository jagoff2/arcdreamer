from __future__ import annotations

from pathlib import Path

import torch

from src.persistent_memory import PersistentMemoryState


def test_hypothesis_trace_attribution_for_change_then_no_op() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")

    record_change = memory.append_event(
        tick=1,
        observation={"x": torch.tensor([1.0])},
        action=0,
        next_observation={"x": torch.tensor([2.0])},
        metadata={"terminal": False, "score_delta": 0.0},
    )

    first_trace = record_change["hypothesis_trace"]
    assert first_trace["schema"] == "runtime_hypothesis_trace_v1"
    assert first_trace["changed_fields"] == ["x"]
    assert first_trace["invariant_fields"] == []
    assert [h["id"] for h in first_trace["predicted_hypotheses"]] == ["field_change|action:0|field:x"]

    record_no_op = memory.append_event(
        tick=2,
        observation={"x": torch.tensor([2.0])},
        action=0,
        next_observation={"x": torch.tensor([2.0])},
        metadata={"terminal": False, "score_delta": 0.0},
    )

    second_trace = record_no_op["hypothesis_trace"]
    assert second_trace["changed_fields"] == []
    assert second_trace["invariant_fields"] == ["x"]

    assert {item["id"] for item in second_trace["predicted_hypotheses"]} == {
        "no_op|action:0",
        "field_stable|action:0|field:x",
    }
    assert "field_change|action:0|field:x" in {
        item["id"] for item in second_trace["contradicted_hypotheses"]
    }


def test_hypothesis_trace_round_trips_with_save_load(tmp_path: Path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    memory.append_event(
        tick=1,
        observation={"x": torch.tensor([1.0])},
        action=0,
        next_observation={"x": torch.tensor([2.0])},
    )
    memory.append_event(
        tick=2,
        observation={"x": torch.tensor([2.0])},
        action=0,
        next_observation={"x": torch.tensor([2.0])},
    )

    path = tmp_path / "memory.pt"
    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=4, batch_size=1, device="cpu")

    assert loaded.event_journal == memory.event_journal
