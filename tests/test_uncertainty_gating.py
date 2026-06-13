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
        self.plastic_memory = plastic_memory or {}


def _row_for_action(components: list[dict[str, Any]], action: int) -> dict[str, Any]:
    return next(item for item in components if item["action"] == action)


def test_speculative_prior_rows_stay_hypothesis_and_block_solution_mode() -> None:
    logits = torch.tensor([1.2], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.25], dtype=torch.float32)}

    selected, diagnostics = select_experimental_action(
        logits,
        _FakeMemory(),
        observation,
        available_action_mask=[True],
    )

    assert selected == 0
    row = _row_for_action(diagnostics["components"], 0)
    assert row["dominant_future_source"] == "speculative_prior"
    assert row["plan_status"] == "hypothesis"
    assert row["high_uncertainty_dependency"] is True
    assert row["hypothesis_only"] is True
    assert row["evidence_gate"] == "uncertainty_limited_hypothesis"
    assert diagnostics["mode"] == "experiment"
    assert diagnostics["selected_plan_status"] == "hypothesis"
    assert diagnostics["selected_evidence_gate"] == "uncertainty_limited_hypothesis"
    assert diagnostics["selected_high_uncertainty_dependency"] is True


def test_ensemble_neural_rows_stay_hypothesis_and_block_solution_mode() -> None:
    logits = torch.tensor([1.2], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.25], dtype=torch.float32)}
    memory = _FakeMemory(
        plastic_memory={
            "self_supervised_adapter": {
                "updates": 1,
                "latest_metrics": {
                    "predicted_change_probability": 0.5,
                    "predicted_object_persistence": 0.5,
                },
                "latest_losses": {
                    "next_delta": 50.0,
                    "inverse_action": 50.0,
                    "no_op_change": 50.0,
                    "object_persistence": 50.0,
                },
                "neural_action_predictions": {"0": {"updates": 1}},
            }
        }
    )

    selected, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True],
    )

    assert selected == 0
    row = _row_for_action(diagnostics["components"], 0)
    assert row["dominant_future_source"] == "ensemble_agreed_neural"
    assert row["plan_status"] == "hypothesis"
    assert row["high_uncertainty_dependency"] is True
    assert row["hypothesis_only"] is True
    assert row["evidence_gate"] == "uncertainty_limited_hypothesis"
    assert diagnostics["mode"] == "experiment"
    assert diagnostics["selected_plan_status"] == "hypothesis"


def test_observed_low_uncertainty_row_can_be_plan_solution() -> None:
    logits = torch.tensor([0.8], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.4], dtype=torch.float32)}
    state_hash = canonical_observation_hash(observation)
    target_hash = "stable-observed-target"
    edge_id = f"{state_hash}|0|{target_hash}"

    memory = _FakeMemory(
        transition_graph={
            "edges": {
                edge_id: {
                    "id": edge_id,
                    "from": state_hash,
                    "to": target_hash,
                    "action": "0",
                    "count": 8,
                    "first_tick": 1,
                    "last_tick": 8,
                    "no_op": False,
                    "loop_observed": False,
                    "reversible": False,
                    "score_delta_sum": 3.5,
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
        available_action_mask=[True],
    )

    assert selected == 0
    row = _row_for_action(diagnostics["components"], 0)
    assert row["dominant_future_source"] == "observed_graph"
    assert row["plan_status"] == "plan"
    assert row["high_uncertainty_dependency"] is False
    assert row["hypothesis_only"] is False
    assert row["evidence_gate"] == "admissible_plan"
    assert diagnostics["mode"] == "solution"
    assert diagnostics["selected_plan_status"] == "plan"
    assert diagnostics["selected_evidence_gate"] == "admissible_plan"
    assert diagnostics["selected_high_uncertainty_dependency"] is False


def test_verified_symbolic_low_uncertainty_row_can_be_plan_solution() -> None:
    logits = torch.tensor([0.8], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.6], dtype=torch.float32)}
    hypothesis = {
        "id": "verified-change|action:0",
        "family": "field_change",
        "action": "0",
        "selector": {"field": "sensory"},
        "transform": {"kind": "change"},
        "posterior": 1.0,
        "support": 4,
        "counterexamples": 0,
        "failed_predictions": 0,
    }

    selected, diagnostics = select_experimental_action(
        logits,
        _FakeMemory(
            hypothesis_posterior={"hypotheses": {hypothesis["id"]: hypothesis}},
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
        ),
        observation,
        available_action_mask=[True],
    )

    assert selected == 0
    row = _row_for_action(diagnostics["components"], 0)
    assert row["dominant_future_source"] == "verified_symbolic"
    assert row["plan_status"] == "plan"
    assert row["high_uncertainty_dependency"] is False
    assert row["hypothesis_only"] is False
    assert row["evidence_gate"] == "admissible_plan"
    assert row["future_risk_order"] == 1
    assert diagnostics["mode"] == "solution"
    assert diagnostics["selected_plan_status"] == "plan"
