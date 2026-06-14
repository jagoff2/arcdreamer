from __future__ import annotations

from src.goal_credit_patch_runtime import apply_goal_credit_patch_runtime
from src import jepa_attempt_memory as memory


def test_visible_control_without_goal_progress_is_not_useful() -> None:
    apply_goal_credit_patch_runtime()
    label = memory._classify_transition_usefulness(
        visible_effect=True,
        events=[],
        score_delta=0.0,
        terminal=False,
        no_effect_event=False,
        invalid_action=False,
        public_controllability_evidence=True,
        public_reachable_state_evidence=True,
        attached_goal_predicates=(),
        goal_entropy_drop=0.0,
        candidate_goal_reachability_improved=0.0,
    )
    assert label.visible_effect
    assert not label.useful_effect
    assert label.instrumental_evidence
    assert label.nuisance_effect


def test_attached_goal_distance_can_be_useful() -> None:
    apply_goal_credit_patch_runtime()
    label = memory._classify_transition_usefulness(
        visible_effect=True,
        events=[],
        score_delta=0.0,
        terminal=False,
        no_effect_event=False,
        invalid_action=False,
        public_controllability_evidence=True,
        attached_goal_predicates=("goal:reach_target",),
        goal_entropy_drop=0.0,
        candidate_goal_reachability_improved=memory.GOAL_REACHABILITY_THRESHOLD + 0.01,
    )
    assert label.useful_effect
    assert not label.nuisance_effect
    assert "attached_goal_distance_improved" in label.reasons


def test_terminal_nonprogress_credit_is_negative() -> None:
    apply_goal_credit_patch_runtime()
    label = memory._classify_transition_usefulness(
        visible_effect=True,
        events=["game_over"],
        score_delta=0.0,
        terminal=True,
        no_effect_event=False,
        invalid_action=False,
        public_controllability_evidence=True,
    )
    value = memory._transition_credit(
        useful=label,
        no_effect_event=False,
        invalid_action=False,
        score_delta=0.0,
        terminal=True,
    )
    assert not label.useful_effect
    assert value < 0.0
