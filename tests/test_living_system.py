from __future__ import annotations

from pathlib import Path

import torch

from src.heldout_causal import FROZEN_CHECKPOINT
from src.living_eval import evaluate_living_system
from src.persistent_memory import PersistentMemoryState


def test_persistent_memory_file_round_trips_latent_and_private_token(tmp_path: Path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=8, batch_size=2)
    latent = torch.randn(2, 8)
    private = torch.tensor([3, 5], dtype=torch.long)
    memory.update(latent, private, tick=17)
    path = tmp_path / "memory.pt"
    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=8, batch_size=2)
    assert torch.allclose(loaded.latent, latent)
    assert torch.equal(loaded.private_token, private)
    assert loaded.tick == 17


def test_living_system_smoke_verdict_passes(tmp_path: Path) -> None:
    report = evaluate_living_system(
        FROZEN_CHECKPOINT,
        config_name="smoke",
        curriculum_output=tmp_path / "curriculum.pt",
    )
    assert report["verdict"]["passes"] is True
    assert report["durable_restart"]["memory_file_restart_final_memory_accuracy"] >= 0.85
    assert report["idle_mode"]["public_language_repetition_ratio"] < 0.40
    assert report["idle_mode"]["private_token_repetition_ratio"] < 0.40
    assert report["richer_dynamics"]["failed_action_no_move"] == 1.0
    assert report["private_internal_language"]["generated_private_unique_count"] >= 3.0
    assert report["curriculum_growth"]["accuracy_after"] >= 0.80
