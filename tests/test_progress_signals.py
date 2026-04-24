from __future__ import annotations

from arcagi.core.progress_signals import (
    transition_policy_supervision,
    transition_usefulness_target,
    visible_online_policy_supervision,
    visible_online_usefulness_target,
)


def test_decoy_reward_reset_is_not_treated_as_useful_progress() -> None:
    usefulness = transition_usefulness_target(
        "interact_right",
        0.08,
        "decoy_reward_reset",
        0.6,
    )

    assert usefulness < 0.0


def test_empty_interaction_gets_strong_negative_policy_supervision() -> None:
    supervision = transition_policy_supervision(
        "interact_right",
        -0.05,
        "empty_interaction",
        0.0,
    )

    assert supervision.target == 0.0
    assert supervision.weight >= 2.0


def test_move_supervision_is_soft_and_does_not_force_exact_path_imitation() -> None:
    supervision = transition_policy_supervision(
        "right",
        -0.01,
        "move",
        0.5,
    )

    assert 0.0 < supervision.target < 0.5
    assert 0.0 < supervision.sibling_move_target < supervision.target


def test_event_free_move_supervision_still_rewards_equivalent_moves() -> None:
    supervision = transition_policy_supervision(
        "right",
        -0.01,
        None,
        0.5,
    )

    assert 0.0 < supervision.target < 0.5
    assert 0.0 < supervision.sibling_move_target < supervision.target


def test_event_free_click_supervision_smooths_over_same_action_type() -> None:
    supervision = transition_policy_supervision(
        "click:2:3",
        0.0,
        None,
        0.35,
    )

    assert supervision.target > 0.0
    assert supervision.same_type_target > 0.0
    assert supervision.same_type_weight > 0.0


def test_correct_switch_gets_strong_positive_supervision() -> None:
    supervision = transition_policy_supervision(
        "interact_right",
        0.25,
        "correct_switch",
        0.3,
    )

    assert supervision.target == 1.0
    assert supervision.weight >= 1.5


def test_visible_online_selector_click_with_hidden_event_stripped_is_diagnostic_positive() -> None:
    usefulness = visible_online_usefulness_target(
        "click:2:3",
        0.0,
        0.089,
        prediction_error=0.2,
        predicted_uncertainty=0.05,
    )
    supervision = visible_online_policy_supervision(
        "click:2:3",
        0.0,
        0.089,
        prediction_error=0.2,
        predicted_uncertainty=0.05,
    )

    assert usefulness > 0.0
    assert supervision.target > 0.0
    assert supervision.weight > 0.0


def test_visible_online_small_move_delta_is_not_objective_progress() -> None:
    usefulness = visible_online_usefulness_target(
        "right",
        0.0,
        0.04,
        prediction_error=0.05,
        predicted_uncertainty=0.03,
    )
    supervision = visible_online_policy_supervision(
        "right",
        0.0,
        0.04,
        prediction_error=0.05,
        predicted_uncertainty=0.03,
    )

    assert usefulness <= 0.0
    assert supervision.target == 0.0
    assert supervision.sibling_move_target == 0.0


def test_visible_online_small_interaction_delta_is_diagnostic_positive() -> None:
    usefulness = visible_online_usefulness_target(
        "interact_right",
        0.0,
        0.04,
        prediction_error=0.05,
        predicted_uncertainty=0.03,
    )
    supervision = visible_online_policy_supervision(
        "interact_right",
        0.0,
        0.04,
        prediction_error=0.05,
        predicted_uncertainty=0.03,
    )

    assert usefulness > 0.0
    assert supervision.target > 0.0


def test_visible_online_reset_is_negative_even_when_display_changes() -> None:
    usefulness = visible_online_usefulness_target(
        "0",
        0.0,
        0.8,
        prediction_error=0.9,
        predicted_uncertainty=0.1,
    )
    supervision = visible_online_policy_supervision(
        "0",
        0.0,
        0.8,
        prediction_error=0.9,
        predicted_uncertainty=0.1,
    )

    assert usefulness < 0.0
    assert supervision.target == 0.0
    assert supervision.weight > 0.0
