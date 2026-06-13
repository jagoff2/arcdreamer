from __future__ import annotations

from typing import Any

import pytest
import torch

from src.action_selection import select_experimental_action
from src.env import NUM_ACTIONS
from src.persistent_memory import canonical_observation_hash
from src.run_unbroken import run_unbroken
from src.train import train_model


class _FakeMemory:
    def __init__(
        self,
        transition_graph: dict[str, Any] | None = None,
        semantic_memory: dict[str, Any] | None = None,
        hypothesis_posterior: dict[str, Any] | None = None,
        goal_posterior: dict[str, Any] | None = None,
        macro_policy_library: dict[str, Any] | None = None,
        plastic_memory: dict[str, Any] | None = None,
    ) -> None:
        self.transition_graph = transition_graph or {}
        self.semantic_memory = semantic_memory or {}
        self.hypothesis_posterior = hypothesis_posterior or {}
        self.goal_posterior = goal_posterior or {}
        self.macro_policy_library = macro_policy_library or {}
        self.plastic_memory = plastic_memory or {}


def _softmax_value(logits: torch.Tensor, index: int) -> float:
    return float(torch.softmax(logits.float(), dim=-1)[index].item())


def test_select_experimental_action_diagnostics_include_all_components() -> None:
    logits = torch.tensor([0.2, 0.1, -0.1], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.1, 0.2], dtype=torch.float32)}
    state_hash = canonical_observation_hash(observation)
    memory = _FakeMemory(
        semantic_memory={
            "affordances": {
                "2": {
                    "mean_score_delta": 2.0,
                    "change_rate": 0.5,
                    "terminal_trials": 0,
                    "trials": 2,
                }
            }
        }
    )

    selected, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True, True],
    )

    assert selected == 2
    assert diagnostics["schema"] == "runtime_experimental_action_selection_v1"
    assert diagnostics["state_hash"] == state_hash
    assert diagnostics["candidate_count"] == 3
    assert diagnostics["weights"] == {
        "policy": 0.70,
        "expected_value": 1.05,
        "information_gain": 0.85,
        "risk": -0.70,
        "action_cost": -1.0,
        "future_risk_order": -0.12,
    }

    selected_row = next(item for item in diagnostics["components"] if item["action"] == selected)
    expected_value = 2.0 + 0.35 * 0.5
    expected_policy = _softmax_value(logits, 2)
    expected_information_gain = 0.75
    expected_risk = 0.25
    expected_cost = 0.01 + 0.002 * 2
    expected_score = (
        0.70 * expected_policy
        + 1.05 * expected_value
        + 0.85 * expected_information_gain
        - 0.70 * expected_risk
        - expected_cost
        - 0.12 * 3
    )

    assert selected_row["action"] == 2
    assert selected_row["policy"] == pytest.approx(expected_policy)
    assert selected_row["expected_value"] == pytest.approx(expected_value)
    assert selected_row["information_gain"] == pytest.approx(expected_information_gain)
    assert selected_row["risk"] == pytest.approx(expected_risk)
    assert selected_row["action_cost"] == pytest.approx(expected_cost)
    assert selected_row["score"] == pytest.approx(expected_score)


def test_select_experimental_action_prefers_unexplored_over_no_op_self_loop() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.3], dtype=torch.float32)}
    state_hash = canonical_observation_hash(observation)
    memory = _FakeMemory(
        transition_graph={
            "unexplored_actions": {state_hash: ["1"]},
            "edges": {
                f"{state_hash}|0|{state_hash}": {
                    "id": f"{state_hash}|0|{state_hash}",
                    "from": state_hash,
                    "to": state_hash,
                    "action": "0",
                    "count": 1,
                    "first_tick": 1,
                    "last_tick": 1,
                    "no_op": True,
                    "loop_observed": True,
                    "reversible": False,
                    "score_delta_sum": 0.0,
                    "terminal_count": 0,
                    "delta": {},
                }
            },
        }
    )

    selected, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True],
    )

    assert selected == 1
    selected_row = next(item for item in diagnostics["components"] if item["action"] == selected)
    loop_row = next(item for item in diagnostics["components"] if item["action"] == 0)
    assert selected_row["information_gain"] == pytest.approx(1.0)
    assert selected_row["risk"] == pytest.approx(0.25)
    assert loop_row["information_gain"] == pytest.approx(0.08)
    assert loop_row["risk"] == pytest.approx(1.10)


def test_select_experimental_action_prefers_high_value_observed_when_semantics_dominate_exploration() -> None:
    logits = torch.tensor([0.0, 1.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.7], dtype=torch.float32)}
    state_hash = canonical_observation_hash(observation)
    other_hash = "exploration-target-state"
    memory = _FakeMemory(
        transition_graph={
            "unexplored_actions": {state_hash: ["1"]},
            "edges": {
                f"{state_hash}|0|{other_hash}": {
                    "id": f"{state_hash}|0|{other_hash}",
                    "from": state_hash,
                    "to": other_hash,
                    "action": "0",
                    "count": 1,
                    "first_tick": 1,
                    "last_tick": 1,
                    "no_op": False,
                    "loop_observed": False,
                    "reversible": False,
                    "score_delta_sum": 0.0,
                    "terminal_count": 0,
                    "delta": {},
                }
            },
        },
        semantic_memory={
            "affordances": {
                "0": {
                    "mean_score_delta": 4.0,
                    "change_rate": 0.0,
                    "terminal_trials": 0,
                    "trials": 1,
                }
            }
        },
    )

    selected, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True],
    )

    assert selected == 0
    selected_row = next(item for item in diagnostics["components"] if item["action"] == selected)
    exploratory_row = next(item for item in diagnostics["components"] if item["action"] == 1)
    assert selected_row["expected_value"] > exploratory_row["expected_value"]
    assert selected_row["information_gain"] == pytest.approx(0.08)
    assert exploratory_row["information_gain"] == pytest.approx(1.0)


def test_select_experimental_action_prefers_reversible_over_risky_non_reversible() -> None:
    logits = torch.tensor([0.1, 0.1], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.9], dtype=torch.float32)}
    state_hash = canonical_observation_hash(observation)
    reverse_to_hash = "reversible-target"
    memory = _FakeMemory(
        transition_graph={
            "edges": {
                f"{state_hash}|0|{reverse_to_hash}": {
                    "id": f"{state_hash}|0|{reverse_to_hash}",
                    "from": state_hash,
                    "to": reverse_to_hash,
                    "action": "0",
                    "count": 1,
                    "first_tick": 1,
                    "last_tick": 1,
                    "no_op": False,
                    "loop_observed": False,
                    "reversible": True,
                    "score_delta_sum": 0.0,
                    "terminal_count": 0,
                    "delta": {},
                },
                f"{state_hash}|1|{state_hash}": {
                    "id": f"{state_hash}|1|{state_hash}",
                    "from": state_hash,
                    "to": state_hash,
                    "action": "1",
                    "count": 1,
                    "first_tick": 1,
                    "last_tick": 1,
                    "no_op": True,
                    "loop_observed": True,
                    "reversible": False,
                    "score_delta_sum": 0.0,
                    "terminal_count": 1,
                    "delta": {},
                },
            }
        }
    )

    selected, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True],
    )

    assert selected == 0
    reversible_row = next(item for item in diagnostics["components"] if item["action"] == 0)
    risky_row = next(item for item in diagnostics["components"] if item["action"] == 1)
    assert reversible_row["risk"] < risky_row["risk"]
    assert reversible_row["observed_rank"] == "observed_reversible"
    assert risky_row["observed_rank"] == "observed_no_op"


def test_imagined_futures_carry_uncertainty_for_observed_and_speculative_actions() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.5], dtype=torch.float32)}
    state_hash = canonical_observation_hash(observation)
    target_hash = "observed-target"
    memory = _FakeMemory(
        transition_graph={
            "edges": {
                f"{state_hash}|0|{target_hash}": {
                    "id": f"{state_hash}|0|{target_hash}",
                    "from": state_hash,
                    "to": target_hash,
                    "action": "0",
                    "count": 4,
                    "first_tick": 1,
                    "last_tick": 4,
                    "no_op": False,
                    "loop_observed": False,
                    "reversible": False,
                    "score_delta_sum": 0.0,
                    "terminal_count": 0,
                    "delta": {},
                }
            }
        }
    )

    _, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True],
    )

    observed_row = next(item for item in diagnostics["components"] if item["action"] == 0)
    speculative_row = next(item for item in diagnostics["components"] if item["action"] == 1)
    observed_future = observed_row["imagined_futures"][0]
    speculative_future = speculative_row["imagined_futures"][0]

    assert observed_future["source"] == "observed_graph"
    assert observed_future["target_state"] == target_hash
    assert observed_row["imagined_uncertainty"] < speculative_row["imagined_uncertainty"]
    assert observed_row["plan_status"] == "plan"
    assert speculative_future["source"] == "speculative_prior"
    assert speculative_row["plan_status"] == "hypothesis"


def test_imagined_futures_include_symbolic_hypothesis_posterior_mass() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.8], dtype=torch.float32)}
    hypothesis = {
        "id": "field_change|action:1|field:sensory",
        "family": "field_change",
        "action": "1",
        "selector": {"field": "sensory"},
        "transform": {"kind": "change"},
        "posterior": 0.75,
        "support": 3,
        "counterexamples": 0,
        "failed_predictions": 0,
    }
    memory = _FakeMemory(hypothesis_posterior={"hypotheses": {hypothesis["id"]: hypothesis}})

    _, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True],
    )

    row = next(item for item in diagnostics["components"] if item["action"] == 1)
    future = row["imagined_futures"][0]

    assert future["source"] == "verified_symbolic"
    assert future["hypothesis_id"] == hypothesis["id"]
    assert future["posterior_mass"] == pytest.approx(0.75)
    assert row["imagined_posterior_mass"] == pytest.approx(0.75)
    assert row["future_risk_order"] == 1
    assert row["future_source_priority"] == 2
    assert 0.0 < row["imagined_uncertainty"] < 1.0
    assert row["plan_status"] == "hypothesis"


def test_action_selection_exposes_posterior_world_model_mixture() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.6], dtype=torch.float32)}
    hypothesis = {
        "id": "field_change|action:1|field:sensory",
        "family": "field_change",
        "action": "1",
        "selector": {"field": "sensory"},
        "transform": {"kind": "change"},
        "goal_test": {"kind": "field_changed"},
        "posterior": 0.80,
        "support": 4,
        "counterexamples": 0,
        "failed_predictions": 0,
        "description_length": 1.85,
    }
    memory = _FakeMemory(hypothesis_posterior={"hypotheses": {hypothesis["id"]: hypothesis}})

    _, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True],
    )

    row = next(item for item in diagnostics["components"] if item["action"] == 1)
    mixture = row["world_model_mixture"]
    assert mixture["prediction_type"] == "causal_edit_program_mixture"
    assert mixture["pixel_prediction"] is False
    assert mixture["component_count"] == 1
    assert mixture["field_change_probability"]["sensory"] == pytest.approx(1.0)
    assert row["posterior_mixture_value"] > 0.0


def test_future_risk_order_prefers_observed_verified_neural_then_speculative() -> None:
    logits = torch.zeros(4, dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.4], dtype=torch.float32)}
    state_hash = canonical_observation_hash(observation)
    hypothesis = {
        "id": "field_change|action:1|field:sensory",
        "family": "field_change",
        "action": "1",
        "selector": {"field": "sensory"},
        "transform": {"kind": "change"},
        "posterior": 0.05,
        "support": 2,
        "counterexamples": 0,
        "failed_predictions": 0,
    }
    memory = _FakeMemory(
        transition_graph={
            "edges": {
                f"{state_hash}|0|observed-target": {
                    "id": f"{state_hash}|0|observed-target",
                    "from": state_hash,
                    "to": "observed-target",
                    "action": "0",
                    "count": 3,
                    "first_tick": 1,
                    "last_tick": 3,
                    "no_op": False,
                    "loop_observed": False,
                    "reversible": False,
                    "score_delta_sum": 0.0,
                    "terminal_count": 0,
                    "delta": {},
                }
            }
        },
        hypothesis_posterior={"hypotheses": {hypothesis["id"]: hypothesis}},
        plastic_memory={
            "self_supervised_adapter": {
                "updates": 8,
                "latest_metrics": {
                    "predicted_change_probability": 0.85,
                    "predicted_object_persistence": 0.80,
                },
                "latest_losses": {
                    "next_delta": 0.15,
                    "inverse_action": 0.20,
                    "no_op_change": 0.10,
                    "object_persistence": 0.12,
                },
                "neural_action_predictions": {"2": {"updates": 8}},
            }
        },
    )

    _, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True, True, True],
    )

    rows = {item["action"]: item for item in diagnostics["components"]}
    assert rows[0]["dominant_future_source"] == "observed_graph"
    assert rows[1]["dominant_future_source"] == "verified_symbolic"
    assert rows[2]["dominant_future_source"] == "ensemble_agreed_neural"
    assert rows[3]["dominant_future_source"] == "speculative_prior"
    assert [rows[action]["future_risk_order"] for action in range(4)] == [0, 1, 2, 3]
    assert [rows[action]["future_source_priority"] for action in range(4)] == [3, 2, 1, 0]


def test_high_value_speculative_plan_remains_hypothesis_and_experiment_mode() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.2], dtype=torch.float32)}
    memory = _FakeMemory(
        semantic_memory={
            "affordances": {
                "1": {
                    "mean_score_delta": 5.0,
                    "change_rate": 0.0,
                    "terminal_trials": 0,
                    "trials": 1,
                }
            }
        }
    )

    selected, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True],
    )

    selected_row = next(item for item in diagnostics["components"] if item["action"] == selected)
    assert selected == 1
    assert selected_row["dominant_future_source"] == "speculative_prior"
    assert selected_row["plan_status"] == "hypothesis"
    assert selected_row["high_uncertainty_dependency"] is True
    assert selected_row["hypothesis_only"] is True
    assert selected_row["evidence_gate"] == "uncertainty_limited_hypothesis"
    assert selected_row["expected_value"] > selected_row["information_gain"]
    assert diagnostics["mode"] == "experiment"
    assert diagnostics["selected_plan_status"] == "hypothesis"
    assert diagnostics["selected_high_uncertainty_dependency"] is True


def test_observed_low_uncertainty_plan_can_enter_solution_mode() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.9], dtype=torch.float32)}
    state_hash = canonical_observation_hash(observation)
    target_hash = "observed-low-uncertainty-target"
    edge_id = f"{state_hash}|0|{target_hash}"
    memory = _FakeMemory(
        transition_graph={
            "edges": {
                edge_id: {
                    "id": edge_id,
                    "from": state_hash,
                    "to": target_hash,
                    "action": "0",
                    "count": 5,
                    "first_tick": 1,
                    "last_tick": 5,
                    "no_op": False,
                    "loop_observed": False,
                    "reversible": False,
                    "score_delta_sum": 0.0,
                    "terminal_count": 0,
                    "delta": {},
                }
            }
        },
        semantic_memory={
            "affordances": {
                "0": {
                    "mean_score_delta": 2.0,
                    "change_rate": 0.0,
                    "terminal_trials": 0,
                    "trials": 1,
                }
            }
        },
    )

    selected, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True],
    )

    selected_row = next(item for item in diagnostics["components"] if item["action"] == selected)
    assert selected == 0
    assert selected_row["dominant_future_source"] == "observed_graph"
    assert selected_row["plan_status"] == "plan"
    assert selected_row["high_uncertainty_dependency"] is False
    assert selected_row["hypothesis_only"] is False
    assert selected_row["evidence_gate"] == "admissible_plan"
    assert diagnostics["mode"] == "solution"
    assert diagnostics["selected_plan_status"] == "plan"
    assert diagnostics["selected_evidence_gate"] == "admissible_plan"


def test_run_unbroken_persists_action_selection_in_event_metadata(tmp_path: Any) -> None:
    checkpoint = tmp_path / "runtime.ckpt"
    memory_file = tmp_path / "memory.pt"
    train_model("smoke", output=checkpoint, steps=1, device="cpu")

    run_unbroken(
        checkpoint,
        max_ticks=3,
        log_every=0,
        seed=900000,
        device="cpu",
        memory_file=memory_file,
    )

    payload = torch.load(memory_file, map_location="cpu")
    events = payload["event_journal"]
    assert len(events) == 3
    for event in events:
        metadata = event["metadata"]
        assert "action_selection" in metadata
        action_selection = metadata["action_selection"]
        assert action_selection["schema"] == "runtime_experimental_action_selection_v1"
        assert set(["selected_action", "candidate_count", "components", "weights", "available_actions"]).issubset(
            action_selection.keys()
        )
        assert action_selection["candidate_count"] == len(action_selection["components"])
        assert action_selection["available_actions"] == list(range(NUM_ACTIONS))
        selected = action_selection["selected_action"]
        assert any(component["action"] == selected for component in action_selection["components"])
        for component in action_selection["components"]:
            assert "imagined_futures" in component
            assert "imagined_uncertainty" in component
            assert "imagined_confidence" in component
            assert "plan_status" in component
            assert "future_risk_order" in component
            assert "future_source_priority" in component
            assert "evidence_gate" in component
            assert "high_uncertainty_dependency" in component
            assert "hypothesis_only" in component
