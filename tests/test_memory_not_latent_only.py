from __future__ import annotations

from pathlib import Path

import torch

from src.model import RecurrentLatentModel, save_checkpoint
from src.persistent_memory import PersistentMemoryState
from src.run_unbroken import run_unbroken


def _public_transition() -> tuple[dict[str, torch.Tensor], dict[str, torch.Tensor]]:
    before = {"sensory": torch.tensor([0.0, 1.0, 0.0], dtype=torch.float32)}
    after = {"sensory": torch.tensor([1.0, 1.0, 0.0], dtype=torch.float32)}
    return before, after


def test_fresh_memory_state_has_explicit_non_latent_containers() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=8, batch_size=1, device="cpu")

    assert memory.latent.shape == (1, 8)
    assert memory.private_token.shape == (1,)
    assert memory.event_journal == []
    assert memory.transition_graph["schema"] == "runtime_transition_graph_v1"
    assert memory.semantic_memory["schema"] == "runtime_semantic_memory_v1"
    assert memory.plastic_memory["schema"] == "runtime_plastic_memory_v1"
    assert memory.hypothesis_posterior["schema"] == "runtime_causal_hypothesis_posterior_v1"
    assert memory.active_theory_set["schema"] == "runtime_active_theory_set_v1"
    assert memory.goal_posterior["schema"] == "runtime_goal_posterior_v1"
    assert memory.macro_policy_library["schema"] == "runtime_macro_policy_library_v1"
    assert memory.coordinate_affordances["schema"] == "runtime_coordinate_affordances_v1"
    assert memory.context_state["schema"] == "runtime_context_state_v1"
    assert memory.controllability_model["schema"] == "runtime_controllability_model_v1"
    assert memory.progress_value_model["schema"] == "runtime_progress_value_model_v1"


def test_append_event_updates_memory_beyond_latent_state() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=8, batch_size=1, device="cpu")
    before, after = _public_transition()

    record = memory.append_event(
        tick=1,
        observation=before,
        action=2,
        next_observation=after,
        metadata={"available_action_mask": [True, True, True], "score_delta": 1.0, "terminal": False},
        prediction_error={"action_uncertainty": 0.2},
    )

    assert record is memory.event_journal[-1]
    assert len(memory.event_journal) == 1
    assert len(memory.transition_graph["nodes"]) == 2
    assert len(memory.transition_graph["edges"]) == 1
    assert "action:2:effect" in memory.semantic_memory["facts"]
    assert memory.plastic_memory["updates"] == 1
    assert len(memory.plastic_memory["key_values"]) == 1
    assert memory.hypothesis_posterior["hypotheses"]
    assert memory.active_theory_set["theories"]
    assert memory.goal_posterior["goals"]
    assert memory.progress_value_model["action_values"]["2"]["support"] >= 1
    assert memory.controllability_model["controlled_variables"]
    assert memory.context_state["phase"] == "running"


def test_save_load_preserves_non_latent_memory_independently(tmp_path: Path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=8, batch_size=1, device="cpu")
    before, after = _public_transition()
    memory.append_event(
        tick=1,
        observation=before,
        action=2,
        next_observation=after,
        metadata={"available_action_mask": [True, True, True], "score_delta": 1.0},
    )
    memory.update(torch.ones_like(memory.latent), torch.tensor([3], dtype=torch.long), tick=9)
    path = tmp_path / "memory.pt"

    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=8, batch_size=1, device="cpu")

    assert torch.allclose(loaded.latent, torch.ones_like(loaded.latent))
    assert torch.equal(loaded.private_token, torch.tensor([3], dtype=torch.long))
    assert loaded.event_journal == memory.event_journal
    assert loaded.transition_graph == memory.transition_graph
    assert loaded.semantic_memory == memory.semantic_memory
    assert loaded.plastic_memory == memory.plastic_memory
    assert loaded.hypothesis_posterior == memory.hypothesis_posterior
    assert loaded.goal_posterior == memory.goal_posterior
    assert loaded.progress_value_model == memory.progress_value_model
    assert loaded.context_state == memory.context_state


def test_run_unbroken_persists_explicit_memory_file_sections(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runtime.pt"
    memory_file = tmp_path / "runtime_memory.pt"
    save_checkpoint(checkpoint, RecurrentLatentModel(device="cpu"), train_config={})

    stats = run_unbroken(
        checkpoint,
        max_ticks=4,
        log_every=0,
        device="cpu",
        memory_file=memory_file,
    )
    payload = torch.load(memory_file, map_location="cpu")

    assert stats["event_journal_length"] == 4.0
    assert len(payload["event_journal"]) == 4
    assert len(payload["transition_graph"]["edges"]) >= 1
    assert len(payload["semantic_memory"]["facts"]) >= 1
    assert payload["plastic_memory"]["updates"] == 4
    assert payload["hypothesis_posterior"]["hypotheses"]
    assert payload["context_state"]["schema"] == "runtime_context_state_v1"
