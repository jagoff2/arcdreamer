from __future__ import annotations

import math
from typing import Any

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


def _component(diagnostics: dict[str, Any], action: int) -> dict[str, Any]:
    return next(item for item in diagnostics["components"] if int(item["action"]) == int(action))


def _goal_entry(*, gid: str, posterior: float, support: int, inconsistency: int = 0) -> dict[str, Any]:
    return {
        "id": gid,
        "scope": "goal",
        "kind": "field_changed",
        "selector": {"field": "sensory"},
        "progress_test": {"changed_count_gt": 0, "field_changed": True, "move": "right"},
        "description_length": 2.0,
        "support": int(support),
        "inconsistency": int(inconsistency),
        "score": 1.6,
        "posterior": float(posterior),
    }


def _action_values_entries(action_values: Any, action: int) -> dict[str, Any]:
    key = str(action)
    if isinstance(action_values, dict):
        if key in action_values:
            value = action_values[key]
            if isinstance(value, dict):
                return value
        if action in action_values:
            value = action_values[action]
            if isinstance(value, dict):
                return value
        for value in action_values.values():
            if isinstance(value, dict) and int(value.get("action", -1)) == int(action):
                return value
        raise AssertionError(f"Action {action} not present in action_values")

    if isinstance(action_values, list):
        for value in action_values:
            if isinstance(value, dict) and int(value.get("action", -1)) == int(action):
                return value
        raise AssertionError(f"Action {action} not present in action_values")

    raise AssertionError(f"Unsupported action_values type: {type(action_values).__name__}")


def _assert_posterior_tree_value_finite(item: dict[str, Any], *, min_simulations: float = 1.0) -> dict[str, Any]:
    search = item["posterior_tree_search"]
    assert search["schema"] == "runtime_posterior_conditioned_pomcp_search_v1"
    assert search["horizon"] > 0
    for key in ["sampled_belief_particles", "selected_hypothesis_ids", "selected_goal_ids"]:
        assert key in search
    assert search["simulations"] >= min_simulations
    assert math.isfinite(float(item["posterior_tree_value"]))
    return search


def test_posterior_conditioned_search_can_select_high_posterior_action_when_logits_flat() -> None:
    logits = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.3], dtype=torch.float32)}

    hypothesis_massive = {
        "id": "field_change|action:1|field:sensory",
        "family": "field_change",
        "action": "1",
        "selector": {"field": "sensory"},
        "transform": {"kind": "change"},
        "goal_test": {"kind": "field_changed", "changed_count": 1},
        "posterior": 0.92,
        "support": 6,
        "counterexamples": 0,
        "failed_predictions": 0,
    }
    hypothesis_weak = {
        "id": "field_change|action:0|field:sensory",
        "family": "field_change",
        "action": "0",
        "selector": {"field": "sensory"},
        "transform": {"kind": "no_effect"},
        "goal_test": {"kind": "field_unchanged"},
        "posterior": 0.02,
        "support": 1,
        "counterexamples": 4,
        "failed_predictions": 2,
    }
    goal = _goal_entry(gid="goal|field_changed|field:sensory", posterior=0.90, support=4)

    memory = _FakeMemory(
        hypothesis_posterior={
            "hypotheses": {
                hypothesis_massive["id"]: hypothesis_massive,
                hypothesis_weak["id"]: hypothesis_weak,
            }
        },
        goal_posterior={
            "schema": "runtime_goal_posterior_v1",
            "updates": 2,
            "goals": {goal["id"]: dict(goal)},
            "subgoals": {
                "subgoal|move|move:right": {
                    "id": "subgoal|move|move:right",
                    "scope": "subgoal",
                    "kind": "movement_subgoal",
                    "selector": {"color": 3},
                    "progress_test": {"move": "right", "field_changed": True},
                    "description_length": 1.8,
                    "support": 3,
                    "inconsistency": 0,
                    "score": 1.2,
                    "posterior": 0.45,
                }
            },
            "posterior_mass": 0.9,
            "top_goals": [
                {
                    "id": goal["id"],
                    "kind": goal["kind"],
                    "posterior": goal["posterior"],
                    "support": goal["support"],
                    "inconsistency": 0,
                }
            ],
        },
    )

    selected, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True, True],
    )

    assert selected == 1
    selected_row = _component(diagnostics, selected)
    assert selected_row["action"] == 1
    assert float(selected_row["posterior_tree_value"]) > 0.0

    search = diagnostics["selected_posterior_tree_search"]
    assert search["selected_action"] == 1
    assert search["schema"] == "runtime_posterior_conditioned_pomcp_search_v1"
    assert selected_row["posterior_tree_search"]["selected_action"] == 1
    assert selected_row["posterior_tree_search"]["selected_action"] == search["selected_action"]
    assert search["selected_hypothesis_ids"]
    assert search["selected_goal_ids"]


def test_posterior_tree_search_exposes_visit_counts_by_action() -> None:
    logits = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.1], dtype=torch.float32)}

    hypotheses = {
        str(hid): {
            "id": str(hid),
            "family": "field_change",
            "action": str(action),
            "selector": {"field": "sensory"},
            "transform": {"kind": "change"},
            "goal_test": {"kind": "field_changed", "changed_count": int(changed_count)},
            "posterior": 0.4 + 0.2 * action,
            "support": 3 + action,
            "counterexamples": 0,
            "failed_predictions": 0,
        }
        for action, hid, changed_count in [
            (0, "field_change|action:0|field:sensory", 1),
            (1, "field_change|action:1|field:sensory", 2),
            (2, "field_change|action:2|field:sensory", 3),
        ]
    }

    memory = _FakeMemory(
        hypothesis_posterior={"hypotheses": hypotheses},
        goal_posterior={
            "schema": "runtime_goal_posterior_v1",
            "updates": 1,
            "goals": {},
            "subgoals": {},
            "posterior_mass": 0.0,
            "top_goals": [],
        },
    )

    _, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True, True],
    )

    search = diagnostics["selected_posterior_tree_search"]
    assert search["schema"] == "runtime_posterior_conditioned_pomcp_search_v1"
    assert isinstance(search["action_values"], (dict, list, tuple))

    for action in diagnostics["available_actions"]:
        action_stats = _action_values_entries(search["action_values"], action)
        simulations = action_stats.get("simulations", action_stats.get("visits", action_stats.get("visit_count", 0.0)))
        assert float(simulations) > 0.0
        value = action_stats.get("value", action_stats.get("mean_value", action_stats.get("q_value", float("nan"))))
        assert math.isfinite(float(value))

    for row in diagnostics["components"]:
        _assert_posterior_tree_value_finite(row)


def test_high_information_search_stays_hypothesis_mode_when_evidence_is_weak() -> None:
    logits = torch.tensor([0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.6], dtype=torch.float32)}

    weak_plan_hypothesis = {
        "id": "move_color|action:1|color:3|dy:0|dx:1",
        "family": "move_color",
        "action": "1",
        "selector": {"color": 3},
        "transform": {"kind": "move", "dx": 1.0, "dy": 0.0},
        "goal_test": {"kind": "move", "moved": 1, "field_changed": True, "changed_count": 1},
        "posterior": 0.88,
        "support": 1,
        "counterexamples": 2,
        "failed_predictions": 2,
    }

    memory = _FakeMemory(
        hypothesis_posterior={"hypotheses": {weak_plan_hypothesis["id"]: weak_plan_hypothesis}},
        goal_posterior={
            "schema": "runtime_goal_posterior_v1",
            "updates": 1,
            "goals": {
                "goal|field_changed|field:sensory": {
                    "id": "goal|field_changed|field:sensory",
                    "scope": "goal",
                    "kind": "field_changed",
                    "selector": {"field": "sensory"},
                    "progress_test": {"changed_count_gt": 0, "field_changed": True, "move": "right"},
                    "description_length": 2.0,
                    "support": 1,
                    "inconsistency": 1,
                    "score": 1.1,
                    "posterior": 0.30,
                }
            },
            "subgoals": {
                "subgoal|move|move:right": {
                    "id": "subgoal|move|move:right",
                    "scope": "subgoal",
                    "kind": "movement_subgoal",
                    "selector": {"color": 3},
                    "progress_test": {"move": "right", "field_changed": True},
                    "description_length": 1.9,
                    "support": 1,
                    "inconsistency": 1,
                    "score": 0.9,
                    "posterior": 0.25,
                }
            },
            "posterior_mass": 0.35,
            "top_goals": [
                {
                    "id": "goal|field_changed|field:sensory",
                    "kind": "field_changed",
                    "posterior": 0.35,
                    "support": 1,
                    "inconsistency": 1,
                }
            ],
        },
    )

    selected, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True],
    )

    assert selected == 1
    row = _component(diagnostics, selected)
    assert row["plan_status"] == "hypothesis"
    assert diagnostics["selected_plan_status"] == "hypothesis"
    assert diagnostics["mode"] == "experiment"
    assert row["high_uncertainty_dependency"] is True
    selected_search = row["posterior_tree_search"]
    assert selected_search["selected_action"] == 1
