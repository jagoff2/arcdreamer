from __future__ import annotations

from pathlib import Path
import math

import torch

from src.persistent_memory import PersistentMemoryState


def _grid_observation() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    observation = {
        "sensory": torch.tensor([0.25, -0.75, 0.5]),
        "grid": torch.tensor([[1, 1], [0, 0]], dtype=torch.int64),
    }
    next_observation = {
        "sensory": torch.tensor([0.30, -0.70, 0.40]),
        "grid": torch.tensor([[1, 1], [0, 0]], dtype=torch.int64),
    }
    return observation, next_observation


def test_fresh_state_initializes_self_supervised_adapter() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=8, batch_size=1, device="cpu")
    adapter = memory.plastic_memory["self_supervised_adapter"]

    assert adapter["schema"] == "runtime_self_supervised_adapter_v1"
    assert adapter["updates"] == 0
    assert isinstance(adapter.get("history_capacity"), int)
    assert adapter["history_capacity"] > 0
    assert adapter["loss_history"] == []
    assert len(adapter["loss_history"]) <= int(adapter["history_capacity"])
    assert set(adapter["heads"].keys()) == {
        "next_delta",
        "inverse_action",
        "no_op_change",
        "object_persistence",
    }
    for head in adapter["heads"].values():
        assert head["weight"] == []
        assert head["bias"] == []


def test_append_event_updates_self_supervised_adapter_and_records_finite_losses_and_metrics() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=8, batch_size=1, device="cpu")
    latent_before = memory.latent.clone()
    private_before = memory.private_token.clone()
    observation, next_observation = _grid_observation()

    memory.append_event(
        tick=1,
        observation=observation,
        action=2,
        next_observation=next_observation,
    )

    adapter = memory.plastic_memory["self_supervised_adapter"]
    assert adapter["updates"] == 1
    assert len(adapter["loss_history"]) == 1
    assert len(adapter["loss_history"]) <= int(adapter["history_capacity"])

    assert math.isfinite(adapter["latest_losses"]["next_delta"])
    assert math.isfinite(adapter["latest_losses"]["inverse_action"])
    assert math.isfinite(adapter["latest_losses"]["no_op_change"])
    assert math.isfinite(adapter["latest_losses"]["object_persistence"])
    assert math.isfinite(adapter["latest_losses"]["total"])

    metrics = adapter["latest_metrics"]
    assert metrics["action_bucket"] == 2
    assert math.isfinite(metrics["change_target"])
    assert 0.0 <= metrics["object_persistence_target"] <= 1.0
    assert math.isfinite(metrics["predicted_change_probability"])
    assert math.isfinite(metrics["predicted_object_persistence"])
    assert math.isfinite(metrics["delta_prediction_l1"])
    assert torch.equal(memory.latent, latent_before)
    assert torch.equal(memory.private_token, private_before)


def test_repeating_same_transition_helps_next_delta_loss_and_bounds_history() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=8, batch_size=1, device="cpu")
    adapter = memory.plastic_memory["self_supervised_adapter"]
    adapter["history_capacity"] = 4
    observation, next_observation = _grid_observation()

    first_loss = None
    last_loss = None
    for tick in range(1, 9):
        memory.append_event(
            tick=tick,
            observation=observation,
            action=2,
            next_observation=next_observation,
        )
        current = memory.plastic_memory["self_supervised_adapter"]["latest_losses"]["next_delta"]
        if first_loss is None:
            first_loss = current
        last_loss = current

    assert first_loss is not None
    assert last_loss is not None
    assert last_loss <= first_loss
    adapter = memory.plastic_memory["self_supervised_adapter"]
    assert adapter["updates"] == 8
    assert len(adapter["loss_history"]) == int(adapter["history_capacity"])


def test_save_and_load_self_supervised_adapter_roundtrip(tmp_path: Path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=8, batch_size=1, device="cpu")
    observation, next_observation = _grid_observation()

    for tick in range(1, 4):
        memory.append_event(
            tick=tick,
            observation=observation,
            action=tick % 2,
            next_observation=next_observation,
        )

    path = tmp_path / "self_supervised_memory.pt"
    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=8, batch_size=1, device="cpu")

    assert (
        loaded.plastic_memory["self_supervised_adapter"]
        == memory.plastic_memory["self_supervised_adapter"]
    )
