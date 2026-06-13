from __future__ import annotations

import torch

from src.metrics import categorical_head_stats, pass_fail


def _passing_base_metrics() -> dict[str, float]:
    return {
        "goal_action_success": 1.0,
        "delayed_memory_accuracy": 1.0,
        "object_permanence_accuracy": 1.0,
        "provenance_accuracy": 1.0,
        "grounded_language_accuracy": 1.0,
        "self_world_continuity_accuracy": 1.0,
        "latent_active_fraction": 1.0,
        "latent_effective_rank": 16.0,
        "latent_max_quantized_fraction": 0.0,
        "language_repetition_ratio": 0.0,
        "unbroken_ticks": 16.0,
        "requested_ticks": 16.0,
        "latent_dim": 64.0,
        "action_head_active_count": 10.0,
        "action_head_normalized_entropy": 0.01,
        "action_head_max_prediction_fraction": 0.50,
        "action_head_unique_prediction_fraction": 0.20,
        "language_head_active_count": 10.0,
        "language_head_normalized_entropy": 0.01,
        "language_head_max_prediction_fraction": 0.50,
        "language_head_unique_prediction_fraction": 0.20,
        "private_head_active_count": 10.0,
        "private_head_normalized_entropy": 0.01,
        "private_head_max_prediction_fraction": 0.50,
        "private_head_unique_prediction_fraction": 0.20,
    }


def test_categorical_head_stats_uses_masked_active_positions() -> None:
    logits = torch.tensor(
        [
            [[8.0, 0.0, 0.0], [0.0, 8.0, 0.0]],
            [[0.0, 0.0, 8.0], [8.0, 0.0, 0.0]],
        ]
    )
    mask = torch.tensor([[True, False], [True, False]])

    stats = categorical_head_stats(logits, prefix="action_head", mask=mask)

    assert stats["action_head_active_count"] == 2.0
    assert stats["action_head_unique_prediction_fraction"] == 2.0 / 3.0
    assert stats["action_head_max_prediction_fraction"] == 0.5
    assert 0.0 <= stats["action_head_normalized_entropy"] <= 1.0
    assert 0.0 <= stats["action_head_mean_confidence"] <= 1.0


def test_pass_fail_rejects_collapsed_action_head_distribution() -> None:
    metrics = _passing_base_metrics()
    metrics["action_head_normalized_entropy"] = 0.01
    metrics["action_head_max_prediction_fraction"] = 1.0
    metrics["action_head_unique_prediction_fraction"] = 0.05

    gates = pass_fail(metrics)

    assert gates["action_head_distribution"] is False
    assert all(value for key, value in gates.items() if key != "action_head_distribution")


def test_pass_fail_accepts_noncollapsed_head_distributions() -> None:
    gates = pass_fail(_passing_base_metrics())

    assert gates["action_head_distribution"] is True
    assert gates["language_head_distribution"] is True
    assert gates["private_head_distribution"] is True
    assert all(gates.values())
