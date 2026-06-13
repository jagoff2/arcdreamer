from pathlib import Path

import torch

from src.persistent_memory import PersistentMemoryState
from src.run_unbroken import run_unbroken
from src.train import train_model


def test_fresh_memory_has_semantic_memory_schema_and_empty_sections() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")

    semantic_memory = memory.semantic_memory
    assert semantic_memory["schema"] == "runtime_semantic_memory_v1"
    assert semantic_memory["action_facts"] == {}
    assert semantic_memory["invariants"] == {}
    assert semantic_memory["counterexamples"] == {}
    assert semantic_memory["affordances"] == {}
    assert semantic_memory["facts"] == {}


def test_append_event_updates_action_facts_and_compressed_action_effect() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    observation = {
        "sensory": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "private": torch.tensor([1], dtype=torch.long),
    }
    next_observation = {
        "sensory": torch.tensor([2.0, 2.0], dtype=torch.float32),
        "private": torch.tensor([1], dtype=torch.long),
    }
    metadata = {"terminal": False, "score_delta": 1.5}

    memory.append_event(
        tick=1,
        observation=observation,
        action=2,
        next_observation=next_observation,
        metadata=metadata,
        prediction_error={},
    )

    action_facts = memory.semantic_memory["action_facts"]["2"]
    assert action_facts["count"] == 1
    assert action_facts["changed_count"] == 1
    assert action_facts["no_op_count"] == 0
    assert action_facts["terminal_count"] == 0
    assert action_facts["score_delta_sum"] == 1.5
    assert action_facts["changed_fields"] == {"sensory": 1}
    assert action_facts["stable_fields"] == {"private": 1}

    action_fact = memory.semantic_memory["facts"]["action:2:effect"]
    assert action_fact["action"] == "2"
    assert action_fact["support"] == 1
    assert action_fact["changed_fields"] == {"sensory": 1}
    assert action_fact["stable_fields"] == {"private": 1}
    assert action_fact["change_rate"] == 1.0
    assert action_fact["no_op_rate"] == 0.0
    assert action_fact["mean_score_delta"] == 1.5
    assert action_fact["confidence"] == 1.0


def test_no_change_transition_records_stable_field_invariant_and_no_op_affordance() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    observation = {
        "sensory": torch.tensor([1.0, 2.0], dtype=torch.float32),
        "private": torch.tensor([3], dtype=torch.long),
    }

    memory.append_event(
        tick=1,
        observation=observation,
        action=4,
        next_observation=observation,
        metadata={"terminal": False, "score_delta": 0.0},
        prediction_error={},
    )

    action_facts = memory.semantic_memory["action_facts"]["4"]
    assert action_facts["count"] == 1
    assert action_facts["changed_count"] == 0
    assert action_facts["no_op_count"] == 1
    assert action_facts["stable_fields"] == {"sensory": 1, "private": 1}

    invariant_key = "action:4:field:sensory:stable"
    invariant = memory.semantic_memory["invariants"][invariant_key]
    assert invariant["support"] == 1
    assert invariant["counterexamples"] == 0
    assert invariant["confidence"] == 1.0
    assert invariant["first_tick"] == 1
    assert invariant["last_tick"] == 1
    assert memory.semantic_memory["facts"][invariant_key] == invariant

    affordance = memory.semantic_memory["affordances"]["4"]
    assert affordance["trials"] == 1
    assert affordance["changed_trials"] == 0
    assert affordance["no_op_trials"] == 1
    assert affordance["change_rate"] == 0.0
    assert affordance["no_op_rate"] == 1.0


def test_stable_field_counterexample_records_reduced_confidence() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    observation = {"sensory": torch.tensor([1.0], dtype=torch.float32)}
    same_observation = {"sensory": torch.tensor([1.0], dtype=torch.float32)}
    changed_observation = {"sensory": torch.tensor([2.0], dtype=torch.float32)}

    memory.append_event(
        tick=1,
        observation=observation,
        action=7,
        next_observation=same_observation,
        metadata={"terminal": False, "score_delta": 0.0},
    )
    invariant_key = "action:7:field:sensory:stable"
    initial_confidence = memory.semantic_memory["invariants"][invariant_key]["confidence"]
    assert initial_confidence == 1.0

    memory.append_event(
        tick=2,
        observation=observation,
        action=7,
        next_observation=changed_observation,
        metadata={"terminal": True, "score_delta": 0.5},
    )

    action_facts = memory.semantic_memory["action_facts"]["7"]
    assert action_facts["count"] == 2
    assert action_facts["changed_count"] == 1
    assert action_facts["no_op_count"] == 1
    assert action_facts["terminal_count"] == 1
    assert action_facts["score_delta_sum"] == 0.5

    invariant = memory.semantic_memory["invariants"][invariant_key]
    assert invariant["support"] == 1
    assert invariant["counterexamples"] == 1
    assert invariant["confidence"] < initial_confidence
    assert memory.semantic_memory["counterexamples"][invariant_key][0]["tick"] == 2
    assert len(memory.semantic_memory["counterexamples"][invariant_key]) == 1


def test_save_and_load_roundtrip_preserves_semantic_memory(tmp_path: Path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=6, batch_size=1, device="cpu")
    memory.append_event(
        tick=1,
        observation={"sensory": torch.tensor([1.0, 2.0])},
        action=3,
        next_observation={"sensory": torch.tensor([2.0, 2.0])},
        metadata={"terminal": False, "score_delta": 0.25},
    )
    memory.append_event(
        tick=2,
        observation={"sensory": torch.tensor([2.0, 2.0])},
        action=3,
        next_observation={"sensory": torch.tensor([2.0, 2.0])},
        metadata={"terminal": False, "score_delta": 0.1},
    )

    memory_file = tmp_path / "semantic_memory.pt"
    memory.save(memory_file)
    loaded = PersistentMemoryState.load(memory_file, hidden_dim=6, batch_size=1, device="cpu")

    assert loaded.semantic_memory == memory.semantic_memory


def test_run_unbroken_with_memory_file_persists_semantic_memory(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runtime.ckpt"
    memory_file = tmp_path / "runtime_memory.pt"

    train_model("smoke", output=checkpoint, steps=1, device="cpu")
    stats = run_unbroken(checkpoint, max_ticks=6, log_every=0, device="cpu", memory_file=memory_file)

    assert stats["unbroken_ticks"] == 6.0
    assert stats["event_journal_length"] == 6.0

    payload = torch.load(memory_file, map_location="cpu")
    semantic_memory = payload["semantic_memory"]
    assert semantic_memory["schema"] == "runtime_semantic_memory_v1"
    assert len(semantic_memory["action_facts"]) > 0
    assert len(semantic_memory["facts"]) > 0
    assert len(payload["event_journal"]) == 6
    assert any(key.endswith(":effect") for key in semantic_memory["facts"])
