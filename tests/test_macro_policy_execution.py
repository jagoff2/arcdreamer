from __future__ import annotations

import torch

from src.action_selection import select_experimental_action
from src.persistent_memory import canonical_observation_hash


class _FakeMemory:
    def __init__(self, macro_policy_library: dict) -> None:
        self.macro_policy_library = macro_policy_library


def test_select_experimental_action_prefers_transfer_ready_macro_with_available_first_action() -> None:
    logits = torch.zeros(3, dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.4], dtype=torch.float32)}
    state_hash = canonical_observation_hash(observation)
    macro_id = "macro|transfer-ready-test"
    macro_library = {
        "schema": "runtime_macro_policy_library_v1",
        "macros": {
            macro_id: {
                "schema": "runtime_macro_policy_v1",
                "id": macro_id,
                "objective": "positive_reward",
                "action_sequence": ["0", "2"],
                "length": 2,
                "support": 2,
                "success_count": 2,
                "failure_count": 0,
                "terminal_count": 0,
                "boundary_count": 0,
                "score_delta_sum": 1.0,
                "mean_score_delta": 0.5,
                "confidence": 1.0,
                "transfer_ready": True,
                "cross_context_support": 2,
                "contexts": [
                    {"key": "game|1|1", "game_id": "game", "level_index": 1, "episode_index": 1, "phase": "running"},
                    {"key": "game|2|1", "game_id": "game", "level_index": 2, "episode_index": 1, "phase": "running"},
                ],
                "context_keys": ["game|1|1", "game|2|1"],
                "source_event_ticks": [1, 2],
                "first_tick": 1,
                "last_tick": 2,
            },
        },
        "updates": 3,
        "consolidations": 1,
        "last_consolidated_tick": 2,
        "top_macros": [
            {
                "id": macro_id,
                "objective": "positive_reward",
                "action_sequence": ["0", "2"],
                "support": 2,
                "confidence": 1.0,
                "transfer_ready": True,
            }
        ],
    }

    memory = _FakeMemory(macro_policy_library=macro_library)
    selected_action, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True, True],
    )

    assert selected_action == 0
    assert diagnostics["selected_macro_id"] == macro_id
    assert diagnostics["selected_macro_plan"]["macro_id"] == macro_id
    assert diagnostics["selected_macro_plan"]["source"] == "macro_policy_library"
    assert diagnostics["selected_macro_plan"]["next_action"] == 0
    assert diagnostics["selected_macro_plan"]["transfer_ready"] is True
    assert diagnostics["selected_macro_plan"]["source_event_ticks"] == [1, 2]
    assert diagnostics["state_hash"] == state_hash
    assert diagnostics["mode"] == "solution"
    assert diagnostics["selected_plan_status"] == "plan"
    assert diagnostics["selected_evidence_gate"] == "macro_policy_transfer"
    selected_row = next(item for item in diagnostics["components"] if item["action"] == selected_action)
    assert selected_row["macro_plan_value"] > 0.0
    assert selected_row["expected_value"] > selected_row["information_gain"]
    assert selected_row["dominant_future_source"] == "macro_policy"
    assert selected_row["high_uncertainty_dependency"] is False
    assert selected_row["macro_plan"]["macro_id"] == macro_id
    assert selected_row["macro_plan"]["source"] == "macro_policy_library"
    assert selected_row["macro_plan"]["objective"] == "positive_reward"


def test_macro_first_action_must_be_available_to_be_selected() -> None:
    logits = torch.tensor([0.0, 2.0, 4.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.9], dtype=torch.float32)}
    unavailable_macro_id = "macro|unavailable-first-action"
    macro_library = {
        "schema": "runtime_macro_policy_library_v1",
        "macros": {
            unavailable_macro_id: {
                "schema": "runtime_macro_policy_v1",
                "id": unavailable_macro_id,
                "objective": "positive_reward",
                "action_sequence": ["2", "1"],
                "length": 2,
                "support": 2,
                "success_count": 2,
                "failure_count": 0,
                "terminal_count": 0,
                "boundary_count": 0,
                "score_delta_sum": 1.2,
                "mean_score_delta": 0.6,
                "confidence": 1.0,
                "transfer_ready": True,
                "cross_context_support": 2,
                "context_keys": ["game|1|1", "game|2|1"],
                "source_event_ticks": [1, 2],
                "first_tick": 1,
                "last_tick": 2,
            },
        },
        "updates": 4,
        "consolidations": 2,
        "last_consolidated_tick": 2,
        "top_macros": [
            {
                "id": unavailable_macro_id,
                "objective": "positive_reward",
                "action_sequence": ["2", "1"],
                "support": 2,
                "confidence": 1.0,
                "transfer_ready": True,
            }
        ],
    }

    memory = _FakeMemory(macro_policy_library=macro_library)
    selected_action, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True, False],
    )

    assert selected_action == 1
    assert selected_action != 2
    assert diagnostics["selected_macro_id"] == ""
    assert set(action["action"] for action in diagnostics["components"]) == {0, 1}
    selected_row = next(item for item in diagnostics["components"] if item["action"] == selected_action)
    assert "macro_plan" not in selected_row
    assert selected_row["macro_plan_value"] == 0.0
