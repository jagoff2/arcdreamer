from __future__ import annotations

from typing import Any

import torch

from src.action_selection import select_experimental_action
from src.persistent_memory import canonical_observation_hash


class _FakeMemory:
    def __init__(
        self,
        transition_graph: dict[str, Any] | None = None,
        semantic_memory: dict[str, Any] | None = None,
        hypothesis_posterior: dict[str, Any] | None = None,
        plastic_memory: dict[str, Any] | None = None,
    ) -> None:
        self.transition_graph = transition_graph or {}
        self.semantic_memory = semantic_memory or {}
        self.hypothesis_posterior = hypothesis_posterior or {}
        self.goal_posterior = {}
        self.macro_policy_library = {}
        self.progress_value_model = {}
        self.plastic_memory = plastic_memory or {}


def _risk_order(component: dict[str, Any]) -> int:
    return int(component.get("risk_order", component.get("future_risk_order")))


def _source_priority(component: dict[str, Any]) -> int:
    return int(component.get("source_priority", component.get("future_source_priority")))


def _component(diagnostics: dict[str, Any], action: int) -> dict[str, Any]:
    return next(item for item in diagnostics["components"] if item["action"] == action)


def test_observed_graph_action_outranks_verified_symbolic_when_scores_are_close() -> None:
    logits = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.2], dtype=torch.float32)}
    state_hash = canonical_observation_hash(observation)
    graph = {
        "edges": {
            f"{state_hash}|0|observed-target": {
                "id": f"{state_hash}|0|observed-target",
                "from": state_hash,
                "to": "observed-target",
                "action": "0",
                "count": 1,
                "first_tick": 1,
                "last_tick": 2,
                "no_op": False,
                "loop_observed": False,
                "reversible": False,
                "score_delta_sum": 0.0,
                "terminal_count": 0,
                "delta": {},
            },
        },
        # no outgoing edges for action 1 despite placeholder graph entries
        "outgoing_action_edges": {
            state_hash: {
                "0": [f"{state_hash}|0|observed-target"],
                "1": [],
                "2": [],
            },
        },
        "unexplored_actions": {state_hash: ["1", "2"]},
    }
    hypothesis = {
        "id": "field_change|action:1|field:sensory",
        "family": "field_change",
        "action": "1",
        "selector": {"field": "sensory"},
        "transform": {"kind": "change"},
        "posterior": 0.75,
        "support": 2,
        "counterexamples": 0,
        "failed_predictions": 0,
    }
    memory = _FakeMemory(
        transition_graph=graph,
        semantic_memory={
            "affordances": {
                "0": {
                    "mean_score_delta": 2.5,
                    "change_rate": 0.0,
                    "terminal_trials": 0,
                    "trials": 1,
                }
            }
        },
        hypothesis_posterior={"hypotheses": {hypothesis["id"]: hypothesis}},
    )

    selected, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True, True],
    )

    observed_row = _component(diagnostics, 0)
    verified_row = _component(diagnostics, 1)
    speculative_row = _component(diagnostics, 2)

    assert selected == 0
    assert observed_row["dominant_future_source"] == "observed_graph"
    assert verified_row["dominant_future_source"] == "verified_symbolic"
    assert speculative_row["dominant_future_source"] == "speculative_prior"
    assert _risk_order(observed_row) < _risk_order(verified_row) < _risk_order(speculative_row)
    assert _source_priority(observed_row) > _source_priority(verified_row) > _source_priority(speculative_row)
    assert "plan_status" in observed_row
    assert "plan_status" in verified_row
    assert "plan_status" in speculative_row


def test_verified_symbolic_future_outranks_ensemble_neural_future() -> None:
    logits = torch.tensor([1.0, 1.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.4], dtype=torch.float32)}
    hypothesis = {
        "id": "field_change|action:0|field:sensory",
        "family": "field_change",
        "action": "0",
        "selector": {"field": "sensory"},
        "transform": {"kind": "change"},
        "posterior": 0.9,
        "support": 2,
        "counterexamples": 0,
        "failed_predictions": 0,
    }
    memory = _FakeMemory(
        transition_graph={"edges": {}, "unexplored_actions": {}},
        hypothesis_posterior={"hypotheses": {hypothesis["id"]: hypothesis}},
        plastic_memory={
            "self_supervised_adapter": {
                "updates": 8,
                "latest_metrics": {
                    "predicted_change_probability": 0.6,
                    "predicted_object_persistence": 0.5,
                },
                "latest_losses": {
                    "next_delta": 0.2,
                    "inverse_action": 0.2,
                    "no_op_change": 0.1,
                    "object_persistence": 0.2,
                },
            },
        },
    )

    selected, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True],
    )

    symbolic_row = _component(diagnostics, 0)
    neural_row = _component(diagnostics, 1)

    assert selected == 0
    assert symbolic_row["dominant_future_source"] == "verified_symbolic"
    assert neural_row["dominant_future_source"] == "ensemble_agreed_neural"
    assert _risk_order(symbolic_row) < _risk_order(neural_row)
    assert _source_priority(symbolic_row) > _source_priority(neural_row)


def test_ensemble_neural_outranks_speculative_prior_with_identical_policy_and_values() -> None:
    logits = torch.tensor([1.0, 1.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.6], dtype=torch.float32)}
    base_graph = {"edges": {}, "unexplored_actions": {}}
    base_memory_kwargs = {
        "transition_graph": base_graph,
        "semantic_memory": {
            "affordances": {
                "1": {
                    "mean_score_delta": 1.8,
                    "change_rate": 0.0,
                    "terminal_trials": 0,
                    "trials": 1,
                }
            }
        },
    }
    selected_speculative, diagnostics_speculative = select_experimental_action(
        logits,
        _FakeMemory(**base_memory_kwargs),
        observation,
        available_action_mask=[True, True],
    )
    selected_neural, diagnostics_neural = select_experimental_action(
        logits,
        _FakeMemory(
            **base_memory_kwargs,
            plastic_memory={
                "self_supervised_adapter": {
                    "updates": 8,
                    "latest_metrics": {
                        "predicted_change_probability": 0.6,
                        "predicted_object_persistence": 0.5,
                    },
                    "latest_losses": {
                        "next_delta": 0.2,
                        "inverse_action": 0.2,
                        "no_op_change": 0.1,
                        "object_persistence": 0.2,
                    },
                },
            },
        ),
        observation,
        available_action_mask=[True, True],
    )

    assert selected_speculative == 1
    assert selected_neural == 1

    speculative_row = _component(diagnostics_speculative, 1)
    neural_row = _component(diagnostics_neural, 1)

    assert speculative_row["dominant_future_source"] == "speculative_prior"
    assert neural_row["dominant_future_source"] == "ensemble_agreed_neural"
    assert _risk_order(neural_row) < _risk_order(speculative_row)
    assert _source_priority(neural_row) > _source_priority(speculative_row)
    assert neural_row["score"] > speculative_row["score"]
