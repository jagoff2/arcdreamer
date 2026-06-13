from __future__ import annotations

from typing import Any

import pytest
import torch

from src.action_selection import select_experimental_action
from src.persistent_memory import canonical_observation_hash


class _FakeMemory:
    def __init__(
        self,
        transition_graph: dict[str, Any] | None = None,
        semantic_memory: dict[str, Any] | None = None,
        hypothesis_posterior: dict[str, Any] | None = None,
        goal_posterior: dict[str, Any] | None = None,
        macro_policy_library: dict[str, Any] | None = None,
    ) -> None:
        self.transition_graph = transition_graph or {}
        self.semantic_memory = semantic_memory or {}
        self.hypothesis_posterior = hypothesis_posterior or {}
        self.goal_posterior = goal_posterior or {}
        self.macro_policy_library = macro_policy_library or {}


def _value(row: dict[str, Any], *keys: str) -> float:
    for key in keys:
        if key in row:
            return float(row[key])
    raise AssertionError(f"Missing keys {keys} in diagnostics row")


def _component(diagnostics: dict[str, Any], action: int) -> dict[str, Any]:
    return next(component for component in diagnostics["components"] if component["action"] == action)


def test_imagined_futures_emit_aggregates_for_all_components() -> None:
    logits = torch.tensor([0.2, -0.1, 0.1], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.3], dtype=torch.float32)}

    _, diagnostics = select_experimental_action(
        logits,
        _FakeMemory(),
        observation,
        available_action_mask=[True, True, True],
    )

    for component in diagnostics["components"]:
        assert isinstance(component["imagined_futures"], list)
        assert component["imagined_futures"]
        assert 0.0 <= _value(component, "imagined_uncertainty") <= 1.0
        assert 0.0 <= _value(component, "imagined_confidence", "confidence") <= 1.0
        assert 0.0 <= _value(component, "imagined_posterior_mass", "imagined_posterior", "posterior_mass") <= 1.0
        assert component["plan_status"] in {"plan", "hypothesis", "speculative"}
        assert component["evidence_gate"] in {
            "admissible_plan",
            "uncertainty_limited_hypothesis",
            "pending_verification_hypothesis",
        }
        assert isinstance(component["high_uncertainty_dependency"], bool)
        assert isinstance(component["hypothesis_only"], bool)


def test_observed_graph_futures_have_lower_uncertainty_and_mark_plan_status() -> None:
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

    observed_row = _component(diagnostics, 0)
    speculative_row = _component(diagnostics, 1)
    observed_future = observed_row["imagined_futures"][0]
    speculative_future = speculative_row["imagined_futures"][0]

    assert observed_future["source"] == "observed_graph"
    assert observed_future["target_state"] == target_hash
    assert speculative_future["source"] == "speculative_prior"
    assert _value(observed_row, "imagined_uncertainty") < _value(speculative_row, "imagined_uncertainty")
    assert observed_row["plan_status"] == "plan"
    assert speculative_row["plan_status"] == "hypothesis"


def test_symbolic_hypothesis_future_carries_posterior_mass_and_id() -> None:
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

    row = _component(diagnostics, 1)
    future = row["imagined_futures"][0]

    assert future["source"] == "verified_symbolic"
    assert future["hypothesis_id"] == hypothesis["id"]
    assert future["posterior_mass"] == pytest.approx(0.75)
    assert _value(row, "imagined_posterior_mass", "imagined_posterior", "posterior_mass") == pytest.approx(0.75)
    assert row["future_risk_order"] == 1
    assert row["future_source_priority"] == 2
    assert row["plan_status"] in {"hypothesis", "speculative"}


def test_speculative_actions_are_high_uncertainty_and_hypothesis_like() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.1], dtype=torch.float32)
    }
    _, diagnostics = select_experimental_action(
        logits,
        _FakeMemory(),
        observation,
        available_action_mask=[True, True],
    )

    for row in diagnostics["components"]:
        future = row["imagined_futures"][0]
        assert future["source"] == "speculative_prior"
        assert _value(row, "imagined_uncertainty") > 0.5
        assert row["plan_status"] in {"hypothesis", "speculative"}
