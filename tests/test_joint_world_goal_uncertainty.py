from __future__ import annotations

import math
from typing import Any

import pytest
import torch

from src.action_selection import select_experimental_action


class _FakeMemory:
    def __init__(
        self,
        transition_graph: dict[str, Any] | None = None,
        semantic_memory: dict[str, Any] | None = None,
        hypothesis_posterior: dict[str, Any] | None = None,
        goal_posterior: dict[str, Any] | None = None,
        macro_policy_library: dict[str, Any] | None = None,
        plastic_memory: dict[str, Any] | None = None,
        progress_value_model: dict[str, Any] | None = None,
    ) -> None:
        self.transition_graph = transition_graph or {}
        self.semantic_memory = semantic_memory or {}
        self.hypothesis_posterior = hypothesis_posterior or {}
        self.goal_posterior = goal_posterior or {}
        self.macro_policy_library = macro_policy_library or {}
        self.plastic_memory = plastic_memory or {}
        self.progress_value_model = progress_value_model or {}


def test_select_experimental_action_prefers_joint_world_goal_alignment() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.2], dtype=torch.float32)}

    memory = _FakeMemory(
        hypothesis_posterior={
            "hypotheses": {
                "h_goal": {
                    "id": "h_goal",
                    "family": "field_change",
                    "action": "1",
                    "selector": {"field": "x"},
                    "transform": {"kind": "change"},
                    "goal_test": {"kind": "field_changed"},
                    "posterior": 0.90,
                    "support": 4,
                    "counterexamples": 0,
                    "failed_predictions": 0,
                },
                "h_misaligned": {
                    "id": "h_misaligned",
                    "family": "move_color",
                    "action": "0",
                    "selector": {"field": "y"},
                    "transform": {"dx": 1, "dy": 0},
                    "goal_test": {"kind": "component_translation"},
                    "posterior": 0.10,
                    "support": 1,
                    "counterexamples": 0,
                    "failed_predictions": 0,
                },
            },
        },
        goal_posterior={
            "posterior_mass": 0.95,
            "goals": {
                "g_aligned": {
                    "id": "g_aligned",
                    "scope": "goal",
                    "kind": "field_changed",
                    "selector": {"field": "x"},
                    "support": 8,
                    "inconsistency": 0,
                    "posterior": 0.75,
                    "description_length": 1.0,
                },
                "g_other": {
                    "id": "g_other",
                    "scope": "goal",
                    "kind": "score_delta_gt",
                    "selector": {"field": "z"},
                    "support": 2,
                    "inconsistency": 1,
                    "posterior": 0.20,
                },
            },
            "top_goals": [{"id": "g_aligned", "score": 0.6}, {"id": "g_other", "score": 0.4}],
        },
    )

    selected, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True],
    )

    assert selected == 1
    assert diagnostics["selected_action"] == selected
    assert diagnostics["selected_joint_world_goal_plan"]["schema"] == "runtime_joint_world_goal_value_v1"
    assert diagnostics["selected_joint_world_goal_plan"]["action"] == 1

    row = next(item for item in diagnostics["components"] if item["action"] == selected)
    plan = row["joint_world_goal_plan"]

    assert plan["schema"] == "runtime_joint_world_goal_value_v1"
    assert row["joint_world_goal_value"] > 0.0
    assert row["joint_world_goal_value"] == pytest.approx(plan["value"])
    assert plan["hypothesis_goal_terms"]
    assert plan["hypothesis_marginal"] > 0.0
    assert plan["goal_marginal"] > 0.0
    assert plan["joint_uncertainty_propagated"] is True
    assert any(term["hypothesis_id"] and term["goal_id"] for term in plan["hypothesis_goal_terms"])


def test_select_experimental_action_joint_plan_terms_encode_joint_aggregation() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.6], dtype=torch.float32)}

    memory = _FakeMemory(
        hypothesis_posterior={
            "hypotheses": {
                "h_primary": {
                    "id": "h_primary",
                    "family": "field_change",
                    "action": "1",
                    "selector": {"field": "x"},
                    "goal_test": {"kind": "field_changed"},
                    "posterior": 0.60,
                    "support": 4,
                    "counterexamples": 0,
                    "failed_predictions": 0,
                },
                "h_secondary": {
                    "id": "h_secondary",
                    "family": "field_change",
                    "action": "1",
                    "selector": {"field": "x"},
                    "goal_test": {"kind": "field_changed"},
                    "posterior": 0.20,
                    "support": 2,
                    "counterexamples": 0,
                    "failed_predictions": 0,
                },
                "h_other": {
                    "id": "h_other",
                    "family": "move_color",
                    "action": "0",
                    "selector": {"field": "y"},
                    "goal_test": {"kind": "moved_object"},
                    "posterior": 0.20,
                    "support": 1,
                    "counterexamples": 0,
                    "failed_predictions": 0,
                },
            },
        },
        goal_posterior={
            "posterior_mass": 0.95,
            "goals": {
                "g_coverage": {
                    "id": "g_coverage",
                    "kind": "field_changed",
                    "selector": {"field": "x"},
                    "support": 6,
                    "inconsistency": 0,
                    "posterior": 0.70,
                },
                "g_trajectory": {
                    "id": "g_trajectory",
                    "kind": "component_translation",
                    "selector": {"color": "red"},
                    "support": 3,
                    "inconsistency": 0,
                    "posterior": 0.30,
                },
            },
            "top_goals": [{"id": "g_coverage", "score": 0.7}, {"id": "g_trajectory", "score": 0.3}],
        },
    )

    _, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True],
    )

    row = next(item for item in diagnostics["components"] if item["action"] == 1)
    plan = row["joint_world_goal_plan"]

    assert plan["hypothesis_marginal"] > 0.0
    assert plan["goal_marginal"] > 0.0
    assert plan["joint_mass"] > 0.0
    assert plan["joint_uncertainty_propagated"] is True
    assert len(plan["hypothesis_goal_terms"]) >= 2

    term_ids = {(term["hypothesis_id"], term["goal_id"]) for term in plan["hypothesis_goal_terms"]}
    assert term_ids
    assert any(
        math.isfinite(float(term["solution_value"])) and math.isfinite(float(term["information_value"]))
        for term in plan["hypothesis_goal_terms"]
    )

    assert plan["joint_mass"] == pytest.approx(sum(float(term["joint_weight"]) for term in plan["hypothesis_goal_terms"]))
    assert plan["expected_solution_value"] == pytest.approx(
        sum(float(term["weighted_solution_value"]) for term in plan["hypothesis_goal_terms"])
    )
    assert plan["expected_information_value"] == pytest.approx(
        sum(float(term["weighted_information_value"]) for term in plan["hypothesis_goal_terms"])
    )
    assert plan["value"] == pytest.approx(plan["expected_solution_value"] + 0.60 * plan["expected_information_value"])


@pytest.mark.parametrize(
    "hypotheses,goals,expected_selected",
    [
        (
            {
                "h1": {
                    "id": "h1",
                    "family": "field_change",
                    "action": "1",
                    "selector": {"field": "x"},
                    "goal_test": {"kind": "field_changed"},
                    "posterior": 0.60,
                    "support": 2,
                    "counterexamples": 0,
                    "failed_predictions": 0,
                },
            },
            {},
            1,
        ),
        (
            {},
            {
                "goals": {
                    "g_only": {
                        "id": "g_only",
                        "kind": "field_changed",
                        "selector": {"field": "x"},
                        "support": 2,
                        "inconsistency": 0,
                        "posterior": 1.0,
                    },
                },
                "posterior_mass": 0.80,
                "top_goals": [{"id": "g_only", "score": 1.0}],
            },
            0,
        ),
    ],
)
def test_joint_world_goal_plan_is_zero_without_partner_signals(
    hypotheses: dict[str, Any],
    goals: dict[str, Any],
    expected_selected: int,
) -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.4], dtype=torch.float32)}

    memory = _FakeMemory(
        hypothesis_posterior={"hypotheses": hypotheses} if hypotheses else {},
        goal_posterior=goals,
    )

    selected, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True],
    )

    assert selected == expected_selected
    for row in diagnostics["components"]:
        joint_plan = row["joint_world_goal_plan"]
        assert joint_plan["schema"] == "runtime_joint_world_goal_value_v1"
        assert joint_plan["joint_uncertainty_propagated"] is False
        assert row["joint_world_goal_value"] == 0.0
        assert joint_plan["joint_mass"] == 0.0
        assert joint_plan["goal_marginal"] == 0.0 or joint_plan["hypothesis_marginal"] == 0.0
