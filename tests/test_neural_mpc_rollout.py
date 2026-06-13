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
    ) -> None:
        self.transition_graph = transition_graph or {}
        self.semantic_memory = semantic_memory or {}
        self.hypothesis_posterior = hypothesis_posterior or {}
        self.goal_posterior = goal_posterior or {}
        self.macro_policy_library = macro_policy_library or {}
        self.plastic_memory = plastic_memory or {}


def _assert_mpc_row_shape(row: dict[str, Any]) -> None:
    rollout = row["neural_mpc_rollout"]
    assert rollout["schema"] == "runtime_learned_neural_mpc_rollout_v1"
    assert isinstance(rollout["horizon"], int)
    assert rollout["horizon"] >= 2
    assert isinstance(rollout["action_sequence"], list)
    assert rollout["action_sequence"]
    assert isinstance(rollout["predicted_steps"], list)
    assert isinstance(rollout["value"], float)
    assert isinstance(rollout["uncertainty"], float)
    assert isinstance(rollout["confidence"], float)
    assert isinstance(rollout["plan_status"], str)
    assert isinstance(rollout["evidence_gate"], str)
    assert isinstance(rollout["high_uncertainty_dependency"], bool)
    assert isinstance(rollout["learned_model_used"], bool)


def _assert_predicted_step_schema(step: dict[str, Any]) -> None:
    assert set(["step", "action", "source", "confidence", "uncertainty", "value"]).issubset(step)
    assert math.isfinite(float(step["step"]))
    assert math.isfinite(float(step["action"]))
    assert math.isfinite(float(step["confidence"]))
    assert math.isfinite(float(step["uncertainty"]))
    assert math.isfinite(float(step["value"]))


def _build_plastic_memory(
    predictions: dict[str, dict[str, Any]],
    *,
    updates: int = 12,
) -> dict[str, Any]:
    base_losses = {
        "next_delta": 0.12,
        "inverse_action": 0.08,
        "no_op_change": 0.10,
        "object_persistence": 0.14,
    }
    return {
        "self_supervised_adapter": {
            "updates": updates,
            "latest_metrics": {
                "predicted_change_probability": 0.55,
                "predicted_object_persistence": 0.45,
                "predicted_progress_probability": 0.50,
                "predicted_score_delta": 0.2,
                "delta_prediction_l1": 0.2,
            },
            "latest_losses": dict(base_losses),
            "neural_action_predictions": predictions,
        }
    }


def test_select_experimental_action_prefers_strong_learned_neural_mpc_rollout() -> None:
    logits = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.25], dtype=torch.float32)}
    memory = _FakeMemory(
        transition_graph={},
        semantic_memory={},
        plastic_memory=_build_plastic_memory(
            {
                "1": {
                    "updates": 12,
                    "metrics": {
                        "predicted_change_probability": 0.92,
                        "predicted_object_persistence": 0.88,
                        "predicted_progress_probability": 0.94,
                        "predicted_score_delta": 2.2,
                        "delta_prediction_l1": 0.02,
                    },
                    "losses": {
                        "total": 0.10,
                        "next_delta": 0.10,
                        "inverse_action": 0.10,
                        "no_op_change": 0.10,
                        "object_persistence": 0.10,
                    },
                    "source": "learned_self_supervised_adapter",
                },
                "0": {
                    "updates": 2,
                    "metrics": {
                        "predicted_change_probability": 0.45,
                        "predicted_object_persistence": 0.50,
                        "predicted_progress_probability": 0.42,
                        "predicted_score_delta": 0.2,
                        "delta_prediction_l1": 0.4,
                    },
                    "losses": {
                        "total": 1.80,
                        "next_delta": 1.80,
                        "inverse_action": 1.80,
                        "no_op_change": 1.80,
                        "object_persistence": 1.80,
                    },
                    "source": "learned_self_supervised_adapter",
                },
                "2": {
                    "updates": 2,
                    "metrics": {
                        "predicted_change_probability": 0.40,
                        "predicted_object_persistence": 0.45,
                        "predicted_progress_probability": 0.30,
                        "predicted_score_delta": 0.1,
                        "delta_prediction_l1": 0.6,
                    },
                    "losses": {
                        "total": 1.20,
                        "next_delta": 1.20,
                        "inverse_action": 1.20,
                        "no_op_change": 1.20,
                        "object_persistence": 1.20,
                    },
                    "source": "learned_self_supervised_adapter",
                },
            },
            updates=12,
        ),
    )

    selected, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True, True],
    )

    assert selected == 1
    for row in diagnostics["components"]:
        assert "neural_mpc_rollout" in row
        assert "neural_mpc_value" in row

    selected_row = next(item for item in diagnostics["components"] if item["action"] == selected)
    rollout = selected_row["neural_mpc_rollout"]
    assert rollout["action"] == 1
    assert rollout["learned_model_used"] is True
    assert rollout["value"] > 0.0
    assert rollout["source"] == "learned_self_supervised_adapter_latent_dynamics"
    assert rollout["predicted_steps"]
    assert rollout["source"] in {"learned_self_supervised_adapter_latent_dynamics", "self_supervised_adapter"}
    _assert_mpc_row_shape(selected_row)

    top_level_rollout = diagnostics["selected_neural_mpc_rollout"]
    assert top_level_rollout["action"] == 1
    assert top_level_rollout["schema"] == "runtime_learned_neural_mpc_rollout_v1"


def test_select_experimental_action_exposes_multi_step_learned_neural_mpc_rollout() -> None:
    logits = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.35], dtype=torch.float32)}
    memory = _FakeMemory(
        plastic_memory=_build_plastic_memory(
            {
                "1": {
                    "updates": 12,
                    "metrics": {
                        "predicted_change_probability": 0.93,
                        "predicted_object_persistence": 0.86,
                        "predicted_progress_probability": 0.91,
                        "predicted_score_delta": 2.8,
                        "delta_prediction_l1": 0.01,
                    },
                    "losses": {
                        "total": 0.08,
                        "next_delta": 0.08,
                        "inverse_action": 0.06,
                        "no_op_change": 0.08,
                        "object_persistence": 0.08,
                    },
                },
                "0": {
                    "updates": 10,
                    "metrics": {
                        "predicted_change_probability": 0.61,
                        "predicted_object_persistence": 0.72,
                        "predicted_progress_probability": 0.61,
                        "predicted_score_delta": 0.4,
                        "delta_prediction_l1": 0.10,
                    },
                    "losses": {
                        "total": 0.45,
                        "next_delta": 0.45,
                        "inverse_action": 0.45,
                        "no_op_change": 0.45,
                        "object_persistence": 0.45,
                    },
                },
                "2": {
                    "updates": 10,
                    "metrics": {
                        "predicted_change_probability": 0.62,
                        "predicted_object_persistence": 0.75,
                        "predicted_progress_probability": 0.58,
                        "predicted_score_delta": 0.3,
                        "delta_prediction_l1": 0.12,
                    },
                    "losses": {
                        "total": 0.50,
                        "next_delta": 0.50,
                        "inverse_action": 0.50,
                        "no_op_change": 0.50,
                        "object_persistence": 0.50,
                    },
                },
            },
            updates=12,
        )
    )

    _, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True, True],
    )

    selected_row = next(item for item in diagnostics["components"] if item["action"] == 1)
    rollout = selected_row["neural_mpc_rollout"]

    assert rollout["horizon"] >= 2
    assert len(rollout["action_sequence"]) >= 2
    assert len(rollout["predicted_steps"]) >= 2
    assert rollout["predicted_steps"][0]["step"] == 1
    assert rollout["plan_status"] == "plan"

    for step in rollout["predicted_steps"]:
        _assert_predicted_step_schema(step)
        assert isinstance(step["action"], int)
        assert step["action"] in {0, 1, 2}
        assert step["source"] in {"self_supervised_adapter", "learned_self_supervised_adapter_latent_dynamics"}

    selected_neural_rollout = diagnostics["selected_neural_mpc_rollout"]
    assert selected_neural_rollout["action"] == 1
    assert selected_neural_rollout["schema"] == "runtime_learned_neural_mpc_rollout_v1"
    assert selected_neural_rollout["predicted_steps"]
    _assert_mpc_row_shape(selected_row)


def test_weak_learned_neural_predictions_remain_hypothesis_limited() -> None:
    logits = torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32)
    observation = {"sensory": torch.tensor([0.45], dtype=torch.float32)}
    memory = _FakeMemory(
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
        plastic_memory=_build_plastic_memory(
            {
                "1": {
                    "updates": 1,
                    "metrics": {
                        "predicted_change_probability": 0.50,
                        "predicted_object_persistence": 0.50,
                        "predicted_progress_probability": 0.51,
                        "predicted_score_delta": 0.3,
                        "delta_prediction_l1": 0.4,
                    },
                    "losses": {
                        "total": 2.50,
                        "next_delta": 2.50,
                        "inverse_action": 2.50,
                        "no_op_change": 2.50,
                        "object_persistence": 2.50,
                    },
                },
                "0": {
                    "updates": 1,
                    "metrics": {
                        "predicted_change_probability": 0.40,
                        "predicted_object_persistence": 0.40,
                        "predicted_progress_probability": 0.40,
                        "predicted_score_delta": 0.1,
                        "delta_prediction_l1": 0.8,
                    },
                    "losses": {
                        "total": 2.10,
                        "next_delta": 2.10,
                        "inverse_action": 2.10,
                        "no_op_change": 2.10,
                        "object_persistence": 2.10,
                    },
                },
            },
            updates=1,
        ),
    )

    selected, diagnostics = select_experimental_action(
        logits,
        memory,
        observation,
        available_action_mask=[True, True, True],
    )

    assert selected == 0
    selected_row = next(item for item in diagnostics["components"] if item["action"] == selected)
    weak_row = next(item for item in diagnostics["components"] if item["action"] == 1)
    weak_rollout = weak_row["neural_mpc_rollout"]

    assert weak_rollout["plan_status"] == "hypothesis"
    assert weak_rollout["high_uncertainty_dependency"] is True
    assert weak_rollout["evidence_gate"] in {"learned_neural_mpc_hypothesis", "uncertainty_limited_hypothesis"}
    assert weak_rollout["value"] == 0.0
    assert selected_row["expected_value"] >= weak_row["expected_value"]
