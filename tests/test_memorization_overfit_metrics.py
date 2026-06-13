from __future__ import annotations

import pytest

from src.metrics import memorization_overfit_gates, memorization_overfit_stats, pass_fail


def test_memorization_overfit_gates_pass_case_with_close_accuracies_and_high_retention() -> None:
    stats = memorization_overfit_stats(
        train_accuracy=0.92,
        heldout_accuracy=0.90,
        perturbation_accuracy=0.88,
        adversarial_accuracy=0.75,
    )

    assert stats["memorization_train_heldout_gap"] == pytest.approx(0.02)
    assert stats["memorization_heldout_train_ratio"] == pytest.approx(0.90 / 0.92)
    assert stats["memorization_perturbation_retention"] == pytest.approx(0.88 / 0.90)
    assert stats["memorization_adversarial_retention"] == pytest.approx(0.75 / 0.90)

    gates = memorization_overfit_gates(stats)

    assert gates["memorization_train_heldout_gap"] is True
    assert gates["memorization_heldout_train_ratio"] is True
    assert gates["memorization_perturbation_retention"] is True
    assert gates["memorization_adversarial_retention"] is True
    assert gates["memorization_overfit"] is True


def test_memorization_overfit_gates_rejects_large_train_heldout_gap() -> None:
    stats = memorization_overfit_stats(
        train_accuracy=1.00,
        heldout_accuracy=0.35,
        perturbation_accuracy=0.40,
        adversarial_accuracy=0.40,
    )

    gates = memorization_overfit_gates(stats)

    assert gates["memorization_train_heldout_gap"] is False
    assert gates["memorization_heldout_train_ratio"] is False
    assert gates["memorization_overfit"] is False


def test_memorization_overfit_gates_rejects_low_perturbation_or_adversarial_retention() -> None:
    stats = memorization_overfit_stats(
        train_accuracy=0.95,
        heldout_accuracy=0.90,
        perturbation_accuracy=0.20,
        adversarial_accuracy=0.55,
    )

    gates = memorization_overfit_gates(stats)

    assert gates["memorization_perturbation_retention"] is False
    assert gates["memorization_adversarial_retention"] is False
    assert gates["memorization_overfit"] is False


def test_pass_fail_exposes_memorization_overfit_gate() -> None:
    gates = pass_fail({"memorization_overfit_pass": 0.0})

    assert gates["memorization_overfit"] is False
