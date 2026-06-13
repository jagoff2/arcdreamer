from __future__ import annotations

from pathlib import Path

import torch

from src.adaptation import FAST_ADAPTATION_MODULES, model_state_fingerprint
from src.model import RecurrentLatentModel, load_checkpoint, save_checkpoint
from src.run_unbroken import run_unbroken


def test_save_checkpoint_separates_reusable_base_and_fast_adaptation_metadata(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    model = RecurrentLatentModel(device="cpu")
    save_checkpoint(checkpoint, model, train_config={})
    payload = torch.load(checkpoint, map_location="cpu")

    base_weights = payload["base_weights"]
    fast_adaptation = payload["fast_adaptation"]

    assert base_weights["schema"] == "reusable_base_weights_v1"
    assert base_weights["role"] == "reusable_base_weights"
    assert base_weights["sha256"] == model_state_fingerprint(model)
    assert base_weights["sha256"]
    assert isinstance(base_weights["parameter_count"], int)

    assert fast_adaptation["schema"] == "runtime_fast_adaptation_sidecar_v1"
    assert fast_adaptation["role"] == "runtime_fast_adaptation"
    assert fast_adaptation["stored_in_checkpoint"] is False
    assert fast_adaptation["modules"] == list(FAST_ADAPTATION_MODULES)

    assert "event_journal" not in payload
    assert "plastic_memory" not in payload


def test_run_unbroken_updates_fast_adaptation_counters_without_mutating_base_weights(tmp_path: Path) -> None:
    checkpoint = tmp_path / "checkpoint.pt"
    memory_file = tmp_path / "memory.pt"
    model = RecurrentLatentModel(device="cpu")
    save_checkpoint(checkpoint, model, train_config={})

    base_fingerprint_before = model_state_fingerprint(load_checkpoint(checkpoint, device="cpu"))
    stats = run_unbroken(
        checkpoint,
        max_ticks=4,
        log_every=0,
        device="cpu",
        memory_file=memory_file,
    )
    base_fingerprint_after = model_state_fingerprint(load_checkpoint(checkpoint, device="cpu"))

    payload = torch.load(memory_file, map_location="cpu")
    plastic_memory = payload["plastic_memory"]

    assert stats["base_weights_unchanged"] == 1.0
    assert base_fingerprint_before == base_fingerprint_after
    assert stats["fast_adaptation_updates"] == 4.0
    assert stats["self_supervised_adapter_updates"] == 4.0
    assert plastic_memory["updates"] == 4
    assert plastic_memory["self_supervised_adapter"]["updates"] == 4
