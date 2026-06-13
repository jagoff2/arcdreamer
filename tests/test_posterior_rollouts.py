from __future__ import annotations

import pytest
import torch

from src.action_selection import select_experimental_action
from src.persistent_memory import canonical_observation_hash


class _FakeMemory:
    def __init__(
        self,
        *,
        transition_graph: dict | None = None,
        hypothesis_posterior: dict | None = None,
        semantic_memory: dict | None = None,
    ) -> None:
        self.transition_graph = transition_graph or {}
        self.hypothesis_posterior = hypothesis_posterior or {}
        self.semantic_memory = semantic_memory or {}
        self.goal_posterior = {}
        self.macro_policy_library = {}
        self.plastic_memory = {}


def _component(diagnostics: dict, action: int) -> dict:
    return next(component for component in diagnostics["components"] if component["action"] == action)


def test_multi_step_posterior_graph_rollout_is_preferred_over_one_step_scoring() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.2], dtype=torch.float32)}
    next_state = {"sensory": torch.tensor([0.3], dtype=torch.float32)}
    progress_state = {"sensory": torch.tensor([0.4], dtype=torch.float32)}

    current_state = canonical_observation_hash(observation)
    intermediate_state = canonical_observation_hash(next_state)
    target_state = canonical_observation_hash(progress_state)

    graph = {
        "edges": {
            f"{current_state}|0|{intermediate_state}": {
                "id": f"{current_state}|0|{intermediate_state}",
                "from": current_state,
                "to": intermediate_state,
                "action": "0",
                "count": 2,
                "first_tick": 1,
                "last_tick": 2,
                "no_op": False,
                "loop_observed": False,
                "reversible": False,
                "score_delta_sum": 0.0,
                "terminal_count": 0,
                "delta": {},
            },
            f"{intermediate_state}|2|{target_state}": {
                "id": f"{intermediate_state}|2|{target_state}",
                "from": intermediate_state,
                "to": target_state,
                "action": "2",
                "count": 3,
                "first_tick": 3,
                "last_tick": 5,
                "no_op": False,
                "loop_observed": False,
                "reversible": False,
                "score_delta_sum": 4.0,
                "terminal_count": 0,
                "delta": {},
            },
        }
    }

    hypothesis = {
        "id": "field_change|action:0|field:sensory",
        "family": "field_change",
        "action": "0",
        "selector": {"field": "sensory"},
        "transform": {"kind": "change"},
        "goal_test": {"kind": "field_changed"},
        "posterior": 0.8,
        "support": 4,
        "counterexamples": 0,
        "failed_predictions": 0,
        "description_length": 1.6,
    }
    memory = _FakeMemory(
        transition_graph=graph,
        hypothesis_posterior={"hypotheses": {hypothesis["id"]: hypothesis}},
    )

    selected, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True],
    )

    assert selected == 0
    selected_row = _component(diagnostics, selected)

    assert selected_row["world_model_mixture"]["prediction_type"] == "causal_edit_program_mixture"
    assert selected_row["posterior_mixture_value"] > 0.0
    assert selected_row["future_source_priority"] >= 2
    assert selected_row["dominant_future_source"] == "observed_graph"
    assert selected_row["graph_plan"]["target_state"] == target_state
    assert selected_row["graph_plan"]["path_length"] == 2
    assert selected_row["graph_plan"]["action_sequence"] == ["0", "2"]
    assert selected_row["graph_plan"]["path_length"] > 1
    assert selected_row["graph_plan_value"] > 0.0
    assert selected_row["internal_rollout_source"] == "observed_graph_rollout"
    assert selected_row["internal_rollout_depth"] == 2
    assert selected_row["internal_rollout_uncertainty"] < 0.20
    assert selected_row["internal_rollout_value"] == pytest.approx(selected_row["graph_plan_value"])
    graph_rollout = selected_row["internal_rollouts"][0]
    assert graph_rollout["source"] == "observed_graph_rollout"
    assert graph_rollout["path_source"] == "transition_graph"
    assert graph_rollout["action_sequence"] == ["0", "2"]
    assert graph_rollout["target_state"] == target_state
    posterior_rollouts = [
        rollout for rollout in selected_row["internal_rollouts"] if rollout["source"] == "posterior_ensemble_rollout"
    ]
    assert posterior_rollouts
    assert posterior_rollouts[0]["depth"] == 2
    assert posterior_rollouts[0]["path_source"] == "posterior_predictive_mixture"
    assert posterior_rollouts[0]["value"] > 0.0
    assert selected_row["posterior_rollout_value"] > 0.0
    assert selected_row["imagined_futures"][0]["source"] == "observed_graph"
    assert selected_row["imagined_futures"][0]["target_state"] == intermediate_state
    assert 0.0 <= selected_row["imagined_uncertainty"] < 0.5


def test_rollout_diagnostics_expose_source_path_uncertainty_and_value_fields() -> None:
    logits = torch.tensor([0.1, -0.2], dtype=torch.float32)
    observation = {"sensory": torch.tensor([1.1], dtype=torch.float32)}
    other = {"sensory": torch.tensor([1.2], dtype=torch.float32)}
    state = canonical_observation_hash(observation)
    other_state = canonical_observation_hash(other)

    graph = {
        "edges": {
            f"{state}|0|{other_state}": {
                "id": f"{state}|0|{other_state}",
                "from": state,
                "to": other_state,
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
        }
    }

    selected, diagnostics = select_experimental_action(
        logits,
        _FakeMemory(transition_graph=graph),
        observation,
        available_action_mask=[True, True],
    )

    assert diagnostics["schema"] == "runtime_experimental_action_selection_v1"
    assert "components" in diagnostics and len(diagnostics["components"]) == 2
    for action, row in [(0, _component(diagnostics, 0)), (1, _component(diagnostics, 1))]:
        assert isinstance(row["imagined_futures"], list)
        assert row["imagined_futures"], f"missing futures for action {action}"
        first_future = row["imagined_futures"][0]
        assert "source" in first_future and isinstance(first_future["source"], str)
        assert "uncertainty" in first_future
        assert 0.0 <= row["imagined_uncertainty"] <= 1.0
        assert 0.0 <= row["imagined_confidence"] <= 1.0
        assert "world_model_mixture" in row
        assert row["world_model_mixture"]["schema"] == "runtime_posterior_predictive_causal_mixture_v1"
        assert isinstance(row["internal_rollouts"], list)
        assert row["internal_rollouts"]
        rollout = row["internal_rollouts"][0]
        assert rollout["schema"] == "runtime_internal_rollout_v1"
        assert "source" in rollout and isinstance(rollout["source"], str)
        assert "path_source" in rollout
        assert "action_sequence" in rollout
        assert "value" in rollout
        assert 0.0 <= rollout["uncertainty"] <= 1.0

    no_plan_row = _component(diagnostics, 1)
    assert "graph_plan" not in no_plan_row
    assert no_plan_row["graph_plan_value"] == 0.0
    assert no_plan_row["posterior_rollout_value"] == 0.0
    plan_row = _component(diagnostics, 0)
    assert "graph_plan" not in plan_row or "action_sequence" in plan_row["graph_plan"]
    assert "expected_value" in plan_row
    assert isinstance(plan_row["expected_value"], float)


def test_high_uncertainty_speculative_rollout_stays_hypothesis_limited() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.2], dtype=torch.float32)}
    memory = _FakeMemory(
        semantic_memory={
            "affordances": {
                "1": {
                    "mean_score_delta": 4.5,
                    "change_rate": 0.0,
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
        available_action_mask=[True, True],
    )

    selected_row = _component(diagnostics, selected)

    assert selected == 1
    assert selected_row["dominant_future_source"] == "speculative_prior"
    assert selected_row["plan_status"] == "hypothesis"
    assert selected_row["high_uncertainty_dependency"] is True
    assert selected_row["hypothesis_only"] is True
    assert selected_row["evidence_gate"] == "uncertainty_limited_hypothesis"
    assert selected_row["internal_rollouts"][0]["source"] == "speculative_rollout"
    assert selected_row["internal_rollouts"][0]["hypothesis_limited"] is True
    assert selected_row["internal_rollout_plan_status"] == "hypothesis"
    assert selected_row["internal_rollout_high_uncertainty_dependency"] is True
    assert selected_row["posterior_rollout_value"] == 0.0
    assert diagnostics["mode"] == "experiment"
    assert diagnostics["selected_plan_status"] == "hypothesis"
