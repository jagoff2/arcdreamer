from __future__ import annotations

import pytest
import torch

from src.action_selection import select_experimental_action
from src.causal_hypotheses import posterior_predictive_mixture


class _FakeMemory:
    def __init__(
        self,
        *,
        hypothesis_posterior: dict[str, object] | None = None,
    ) -> None:
        self.transition_graph = {}
        self.semantic_memory = {}
        self.hypothesis_posterior = hypothesis_posterior or {}
        self.goal_posterior = {}
        self.macro_policy_library = {}
        self.plastic_memory = {}


def test_posterior_predictive_mixture_reports_causal_change_mass() -> None:
    posterior = {
        "hypotheses": {
            "field_change|action:3|field:counter": {
                "id": "field_change|action:3|field:counter",
                "family": "field_change",
                "action": "3",
                "selector": {"field": "counter"},
                "transform": {"kind": "change", "l1": 1.0},
                "goal_test": {"kind": "field_changed"},
                "posterior": 0.60,
            },
            "field_stable|action:3|field:counter": {
                "id": "field_stable|action:3|field:counter",
                "family": "field_stable",
                "action": "3",
                "selector": {"field": "counter"},
                "transform": {"kind": "identity"},
                "goal_test": {"kind": "field_unchanged"},
                "posterior": 0.20,
            },
            "move_color|action:3|color:2|dy:0.0|dx:1.0": {
                "id": "move_color|action:3|color:2|dy:0.0|dx:1.0",
                "family": "move_color",
                "action": "3",
                "selector": {"color": 2},
                "transform": {"kind": "translate", "dy": 0.0, "dx": 1.0},
                "goal_test": {"kind": "component_translation"},
                "posterior": 0.15,
            },
            "no_op|action:3": {
                "id": "no_op|action:3",
                "family": "no_op",
                "action": "3",
                "selector": {"scope": "observation"},
                "transform": {"kind": "identity"},
                "goal_test": {"kind": "no_visible_change"},
                "posterior": 0.05,
            },
        }
    }

    mixture = posterior_predictive_mixture(posterior, action=3)

    assert mixture["schema"] == "runtime_posterior_predictive_causal_mixture_v1"
    assert mixture["prediction_type"] == "causal_edit_program_mixture"
    assert mixture["pixel_prediction"] is False
    assert mixture["action"] == "3"
    assert mixture["component_count"] == 4
    assert mixture["mixture_available"] is True
    assert mixture["action_posterior_mass"] == pytest.approx(1.0)
    assert mixture["normalized_component_mass"] == pytest.approx(1.0)
    assert mixture["expected_change_probability"] == pytest.approx(0.75)
    assert mixture["no_op_probability"] == pytest.approx(0.05)
    assert mixture["field_change_probability"] == {"counter": pytest.approx(0.60)}
    assert mixture["field_stable_probability"] == {"counter": pytest.approx(0.20)}
    assert mixture["move_color_probability"] == {"color:2|dy:0.0|dx:1.0": pytest.approx(0.15)}
    assert mixture["family_mixture"]["field_change"] == pytest.approx(0.60)
    assert mixture["family_mixture"]["field_stable"] == pytest.approx(0.20)
    assert mixture["family_mixture"]["move_color"] == pytest.approx(0.15)
    assert mixture["family_mixture"]["no_op"] == pytest.approx(0.05)
    assert 0.0 <= mixture["uncertainty"] <= 1.0


def test_posterior_predictive_mixture_filters_by_action_and_family() -> None:
    posterior = {
        "hypotheses": {
            "field_change|action:1|field:x": {
                "id": "field_change|action:1|field:x",
                "family": "field_change",
                "action": "1",
                "selector": {"field": "x"},
                "transform": {"kind": "change"},
                "goal_test": {"kind": "field_changed"},
                "posterior": 0.40,
            },
            "field_change|action:2|field:y": {
                "id": "field_change|action:2|field:y",
                "family": "field_change",
                "action": "2",
                "selector": {"field": "y"},
                "transform": {"kind": "change"},
                "goal_test": {"kind": "field_changed"},
                "posterior": 0.25,
            },
            "exotic_family|action:1|field:z": {
                "id": "exotic_family|action:1|field:z",
                "family": "exotic_family",
                "action": "1",
                "selector": {"field": "z"},
                "transform": {"kind": "weird"},
                "goal_test": {"kind": "never"},
                "posterior": 0.50,
            },
            "no_op|action:1": {
                "id": "no_op|action:1",
                "family": "no_op",
                "action": "1",
                "selector": {"scope": "observation"},
                "transform": {"kind": "identity"},
                "goal_test": {"kind": "no_visible_change"},
                "posterior": 0.0,
            },
        }
    }

    mixture = posterior_predictive_mixture(posterior, action=1)

    assert mixture["component_count"] == 1
    assert mixture["action_posterior_mass"] == pytest.approx(0.40)
    assert {comp["id"] for comp in mixture["components"]} == {"field_change|action:1|field:x"}
    assert mixture["family_mixture"] == {"field_change": 1.0}


def test_action_selection_carries_posterior_world_model_mixture() -> None:
    logits = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([1.0]), "private": torch.tensor([3])}
    memory = _FakeMemory(
        hypothesis_posterior={
            "hypotheses": {
                "field_change|action:2|field:sensory": {
                    "id": "field_change|action:2|field:sensory",
                    "family": "field_change",
                    "action": "2",
                    "selector": {"field": "sensory"},
                    "transform": {"kind": "change"},
                    "goal_test": {"kind": "field_changed"},
                    "posterior": 0.72,
                    "support": 1,
                    "counterexamples": 0,
                    "failed_predictions": 0,
                }
            }
        }
    )

    _, diagnostics = select_experimental_action(logits, memory, observation, available_action_mask=[True, True, True])

    mixture_row = next(item for item in diagnostics["components"] if item["action"] == 2)
    no_mixture_row = next(item for item in diagnostics["components"] if item["action"] == 0)
    world_model_mixture = mixture_row["world_model_mixture"]
    absent_mixture = no_mixture_row["world_model_mixture"]

    assert world_model_mixture["prediction_type"] == "causal_edit_program_mixture"
    assert world_model_mixture["pixel_prediction"] is False
    assert world_model_mixture["mixture_available"] is True
    assert world_model_mixture["action"] == "2"
    assert world_model_mixture["expected_change_probability"] == pytest.approx(1.0)
    assert world_model_mixture["uncertainty"] < 1.0
    assert mixture_row["posterior_mixture_value"] > 0.0
    assert absent_mixture["mixture_available"] is False
    assert no_mixture_row["posterior_mixture_value"] == 0.0
