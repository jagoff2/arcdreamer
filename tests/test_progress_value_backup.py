from __future__ import annotations

import pytest
import torch

from src.action_selection import select_experimental_action
from src.persistent_memory import PersistentMemoryState


def _obs(value: float) -> dict[str, torch.Tensor]:
    return {"sensory": torch.tensor([value], dtype=torch.float32)}


def _discount(model: dict[str, object]) -> float:
    return float(model.get("discount", 0.85))


def test_fresh_memory_has_progress_value_schema() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    model = memory.progress_value_model

    assert model["schema"] == "runtime_progress_value_model_v1"
    assert model["updates"] == 0
    assert model["backups"] == 0
    assert model["action_values"] == {}
    assert model["sequence_values"] == {}
    assert model["recent_progress"] == []
    assert model["top_actions"] == []
    assert model["top_sequences"] == []


def test_non_progress_event_does_not_back_up() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")

    memory.append_event(
        tick=1,
        observation=_obs(0.0),
        action=0,
        next_observation=_obs(0.1),
        metadata={"terminal": False, "score_delta": 0.0, "available_action_mask": [True, True]},
    )

    model = memory.progress_value_model
    assert model["updates"] == 1
    assert model["backups"] == 0
    assert model["action_values"] == {}
    assert model["sequence_values"] == {}
    assert model["recent_progress"] == []


def test_positive_score_backups_discounted_recent_actions_and_records_evidence() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")

    memory.append_event(
        tick=1,
        observation=_obs(0.0),
        action=0,
        next_observation=_obs(0.2),
        metadata={"terminal": False, "score_delta": 0.0, "available_action_mask": [True, True, True]},
    )
    memory.append_event(
        tick=2,
        observation=_obs(0.2),
        action=1,
        next_observation=_obs(0.4),
        metadata={"terminal": False, "score_delta": 0.0, "available_action_mask": [True, True, True]},
    )
    memory.append_event(
        tick=3,
        observation=_obs(0.4),
        action=2,
        next_observation=_obs(0.6),
        metadata={"terminal": False, "score_delta": 1.5, "available_action_mask": [True, True, True]},
    )

    model = memory.progress_value_model
    assert model["updates"] == 3
    assert model["backups"] == 1
    assert set(model["action_values"].keys()) == {"0", "1", "2"}
    assert len(model["sequence_values"]) == 1

    discount = _discount(model)
    expected = {
        "2": 1.5 * (discount ** 0),
        "1": 1.5 * (discount ** 1),
        "0": 1.5 * (discount ** 2),
    }

    for action, value in expected.items():
        entry = model["action_values"][action]
        assert entry["support"] == 1
        assert entry["mean_value"] == pytest.approx(value)
        assert entry["confidence"] == pytest.approx(max(value, 0.0) / 3.0)
        assert entry["evidence_ticks"] == [3]
        assert entry["signals"] == {"score_delta": 1}

    recent_progress = model["recent_progress"][-1]
    assert recent_progress["tick"] == 3
    assert recent_progress["signal"] == "score_delta"
    assert recent_progress["actions"] == ["0", "1", "2"]
    assert recent_progress["magnitude"] == pytest.approx(1.5)
    assert recent_progress["sequence_id"] in model["sequence_values"]

    sequence = model["sequence_values"][recent_progress["sequence_id"]]
    assert sequence["support"] == 1
    assert sequence["mean_value"] == pytest.approx(1.5)
    assert sequence["confidence"] == pytest.approx(0.5)
    assert sequence["evidence_ticks"] == [3]
    assert sequence["signals"] == {"score_delta": 1}


def test_terminal_win_metadata_triggers_progress_backup() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")

    memory.append_event(
        tick=1,
        observation=_obs(0.0),
        action=0,
        next_observation=_obs(0.2),
        metadata={"terminal": False, "score_delta": 0.0, "available_action_mask": [True, True]},
    )
    memory.append_event(
        tick=2,
        observation=_obs(0.2),
        action=1,
        next_observation=_obs(0.4),
        metadata={
            "terminal": True,
            "boundary": "win",
            "score_delta": 0.0,
            "available_action_mask": [True, True],
        },
    )

    model = memory.progress_value_model
    assert model["updates"] == 2
    assert model["backups"] == 1
    assert set(model["action_values"].keys()) == {"0", "1"}
    assert model["action_values"]["1"]["signals"] == {"event_progress": 1}
    assert model["action_values"]["0"]["signals"] == {"event_progress": 1}
    assert model["action_values"]["1"]["support"] == 1
    assert model["action_values"]["0"]["support"] == 1
    assert model["recent_progress"][-1]["tick"] == 2
    assert model["recent_progress"][-1]["actions"] == ["0", "1"]


def test_progress_value_model_round_trips_on_memory_save_load(tmp_path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")

    memory.append_event(
        tick=1,
        observation=_obs(0.0),
        action=0,
        next_observation=_obs(0.2),
        metadata={"terminal": False, "score_delta": 0.0, "available_action_mask": [True, True]},
    )
    memory.append_event(
        tick=2,
        observation=_obs(0.2),
        action=1,
        next_observation=_obs(0.4),
        metadata={"terminal": False, "score_delta": 2.0, "available_action_mask": [True, True]},
    )
    memory.append_event(
        tick=3,
        observation=_obs(0.4),
        action=0,
        next_observation=_obs(0.5),
        metadata={"terminal": True, "boundary": "win", "score_delta": 0.0, "available_action_mask": [True, True]},
    )

    path = tmp_path / "progress_value.ckpt"
    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=4, batch_size=1, device="cpu")

    assert loaded.progress_value_model == memory.progress_value_model


def test_select_experimental_action_prefers_progress_backed_action_over_unexplored() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    observation = _obs(0.0)
    next_observation = _obs(1.0)

    memory.append_event(
        tick=1,
        observation=observation,
        action=1,
        next_observation=next_observation,
        metadata={"terminal": False, "score_delta": 3.0, "available_action_mask": [True, True]},
    )

    selected, diagnostics = select_experimental_action(
        torch.tensor([0.0, 0.0], dtype=torch.float32),
        memory,
        observation,
        available_action_mask=[True, True],
    )

    assert selected == 1
    selected_row = next(item for item in diagnostics["components"] if item["action"] == 1)
    exploratory_row = next(item for item in diagnostics["components"] if item["action"] == 0)
    assert selected_row["expected_value"] > exploratory_row["expected_value"]
    assert selected_row["expected_value"] > 0.0
