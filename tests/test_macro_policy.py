from __future__ import annotations

import torch

from src.persistent_memory import PersistentMemoryState


def _sensory(a: float, b: float) -> dict[str, torch.Tensor]:
    return {"sensory": torch.tensor([a, b], dtype=torch.float32)}


def _single_macro(library: dict[str, object]) -> tuple[str, dict[str, object]]:
    macros = library.get("macros", {})
    assert isinstance(macros, dict)
    assert len(macros) == 1
    (macro_id, macro), = macros.items()
    assert isinstance(macro, dict)
    return macro_id, macro


def test_fresh_memory_has_macro_policy_library_schema() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    library = memory.macro_policy_library

    assert library["schema"] == "runtime_macro_policy_library_v1"
    assert library["macros"] == {}
    assert library["updates"] == 0
    assert library["consolidations"] == 0
    assert library["last_consolidated_tick"] is None
    assert library["top_macros"] == []


def test_non_progress_append_event_does_not_create_macro() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")

    memory.append_event(
        tick=1,
        observation=_sensory(0.0, 0.0),
        action=1,
        next_observation=_sensory(0.0, 0.1),
        metadata={"terminal": False, "available_action_mask": [True, True], "score_delta": 0.0},
    )

    library = memory.macro_policy_library
    assert library["macros"] == {}
    assert library["consolidations"] == 0
    assert library["updates"] == 1
    assert library["top_macros"] == []


def test_progress_event_creates_and_records_macro_with_procedural_fact() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")

    memory.append_event(
        tick=1,
        observation=_sensory(0.0, 0.0),
        action=0,
        next_observation=_sensory(0.0, 1.0),
        metadata={"terminal": False, "score_delta": 0.0, "available_action_mask": [True, True]},
    )
    memory.append_event(
        tick=2,
        observation=_sensory(0.0, 1.0),
        action=1,
        next_observation=_sensory(1.0, 1.0),
        metadata={"terminal": False, "score_delta": 1.25, "available_action_mask": [True, True]},
    )

    library = memory.macro_policy_library
    assert library["consolidations"] == 1
    macro_id, macro = _single_macro(library)

    assert macro["objective"] == "positive_reward"
    assert macro["action_sequence"] == ["0", "1"]
    assert macro["support"] == 1
    assert macro["success_count"] == 1
    assert macro["failure_count"] == 0
    assert macro["source_event_ticks"] == [1, 2]

    procedural_facts = memory.semantic_memory.get("procedural_facts", {})
    facts = memory.semantic_memory.get("facts", {})
    fact_key = f"macro:{macro_id}"
    assert fact_key in procedural_facts
    assert fact_key in facts
    assert procedural_facts[fact_key]["macro_id"] == macro_id
    assert procedural_facts[fact_key]["objective"] == macro["objective"]
    assert procedural_facts[fact_key]["action_sequence"] == macro["action_sequence"]


def test_repeating_progress_sequence_after_episode_boundary_updates_support_and_contexts() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")

    memory.append_event(
        tick=1,
        observation=_sensory(0.0, 0.0),
        action=0,
        next_observation=_sensory(0.0, 1.0),
        metadata={"terminal": False, "score_delta": 0.0, "available_action_mask": [True, True]},
    )
    memory.append_event(
        tick=2,
        observation=_sensory(0.0, 1.0),
        action=1,
        next_observation=_sensory(1.0, 1.0),
        metadata={"terminal": False, "score_delta": 1.5, "available_action_mask": [True, True]},
    )
    library = memory.macro_policy_library
    assert library["consolidations"] == 1
    macro_id, first_macro = _single_macro(library)
    assert first_macro["support"] == 1
    assert first_macro["cross_context_support"] == 1
    assert first_macro["transfer_ready"] is False

    memory.append_event(
        tick=3,
        observation=_sensory(1.0, 1.0),
        action=0,
        next_observation=_sensory(1.0, 0.0),
        metadata={"terminal": True, "boundary": "episode", "score_delta": 0.0, "available_action_mask": [True, True]},
    )
    memory.append_event(
        tick=4,
        observation=_sensory(1.0, 0.0),
        action=0,
        next_observation=_sensory(1.0, 0.1),
        metadata={"terminal": False, "score_delta": 0.0, "available_action_mask": [True, True]},
    )
    memory.append_event(
        tick=5,
        observation=_sensory(1.0, 0.1),
        action=1,
        next_observation=_sensory(1.0, 0.2),
        metadata={"terminal": False, "score_delta": 1.5, "available_action_mask": [True, True]},
    )

    _, macro = _single_macro(memory.macro_policy_library)
    assert memory.macro_policy_library["consolidations"] == 2
    assert macro["support"] == 2
    assert macro["success_count"] == 2
    assert macro["source_event_ticks"] == [1, 2, 4, 5]
    assert macro["cross_context_support"] == 2
    assert macro["transfer_ready"] is True
    assert len(macro["context_keys"]) == 2
    assert len(macro["contexts"]) == 2
    assert macro_id == memory.macro_policy_library["top_macros"][0]["id"]


def test_macro_policy_library_round_trips_on_save_load(tmp_path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")

    memory.append_event(
        tick=1,
        observation=_sensory(0.0, 0.0),
        action=0,
        next_observation=_sensory(0.2, 0.0),
        metadata={"terminal": False, "score_delta": 0.0, "available_action_mask": [True, True]},
    )
    memory.append_event(
        tick=2,
        observation=_sensory(0.2, 0.0),
        action=1,
        next_observation=_sensory(0.2, 0.3),
        metadata={"terminal": False, "score_delta": 2.0, "available_action_mask": [True, True]},
    )
    memory.append_event(
        tick=3,
        observation=_sensory(0.2, 0.3),
        action=1,
        next_observation=_sensory(0.2, 0.4),
        metadata={"terminal": True, "boundary": "episode", "score_delta": 0.0, "available_action_mask": [True, True]},
    )

    path = tmp_path / "macro_memory.pt"
    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=4, batch_size=1, device="cpu")

    assert loaded.macro_policy_library == memory.macro_policy_library
