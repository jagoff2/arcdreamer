from __future__ import annotations

import pytest

from arcagi.learned_online.object_event_bridge import (
    SelectedActionObservation,
    assert_no_forbidden_metadata_as_model_input,
    build_object_event_observation,
    forbidden_metadata_keys,
)
from arcagi.learned_online.object_event_curriculum import (
    ActiveOnlineObjectEventConfig,
    apply_synthetic_object_event_action,
    build_active_online_object_event_curriculum,
)


def test_bridge_builds_selected_action_transition_only() -> None:
    level = _level()
    example = level.example
    selected = int(example.metadata["blue_action_index"])
    result = apply_synthetic_object_event_action(level, selected)
    event = build_object_event_observation(
        SelectedActionObservation(
            before=example.state,
            selected_action=example.legal_actions[selected],
            after=example.next_state if result.success else example.state,
            reward=result.reward,
            terminated=False,
            legal_actions=example.legal_actions,
            info={"score_delta": result.reward},
        ),
        metadata={"correct_action_index": int(example.correct_action_index), "latent_rule": int(level.latent_rule)},
    )

    assert event.selected_action_index == selected
    assert event.transition_targets.actual_action_index == selected
    assert event.legal_action_count == len(example.legal_actions)


def test_bridge_preserves_full_legal_action_surface() -> None:
    level = _level()
    example = level.example
    event = _bridge_for(level, int(example.correct_action_index))

    assert len(example.legal_actions) == 68
    assert event.legal_action_count == 68
    assert event.action_tokens.numeric.shape[0] == 68
    assert int(event.action_tokens.mask.sum()) == 68


def test_bridge_rejects_selected_action_outside_legal_surface() -> None:
    level = _level()
    example = level.example

    with pytest.raises(ValueError):
        build_object_event_observation(
            SelectedActionObservation(
                before=example.state,
                selected_action="click:99:99",
                after=example.state,
                reward=0.0,
                terminated=False,
                legal_actions=example.legal_actions,
                info={},
            )
        )


def test_bridge_metadata_not_used_as_model_input() -> None:
    level = _level()
    event = _bridge_for(level, int(level.example.correct_action_index))
    diagnostic_metadata = {
        **dict(event.metadata),
        "game_id": "ar25-hidden",
        "trace_cursor": 37,
        "correct_action_index": int(level.example.correct_action_index),
    }

    assert "correct_action_index" not in event.model_input_metadata()
    assert_no_forbidden_metadata_as_model_input(event.model_input_metadata())
    assert set(forbidden_metadata_keys(diagnostic_metadata)) == {
        "correct_action_index",
        "game_id",
        "latent_rule",
        "trace_cursor",
    }
    with pytest.raises(AssertionError):
        assert_no_forbidden_metadata_as_model_input(diagnostic_metadata)


def test_synthetic_active_bridge_matches_apply_selected_action_result() -> None:
    level = _level()
    example = level.example
    for selected in (int(example.correct_action_index), int(example.metadata["red_action_index"])):
        result = apply_synthetic_object_event_action(level, selected)
        event = _bridge_for(level, selected)
        assert event.selected_action_index == result.selected_action_index
        assert event.transition_targets.actual_action_index == result.transition_targets.actual_action_index
        assert event.transition_targets.reward == result.transition_targets.reward
        assert event.legal_action_count == len(example.legal_actions)


def _level():
    return build_active_online_object_event_curriculum(
        ActiveOnlineObjectEventConfig(seed=101, train_sessions=1, heldout_sessions=0, levels_per_session=2, max_distractors=1)
    ).train[0].levels[0]


def _bridge_for(level, selected: int):
    example = level.example
    result = apply_synthetic_object_event_action(level, selected)
    return build_object_event_observation(
        SelectedActionObservation(
            before=example.state,
            selected_action=example.legal_actions[int(selected)],
            after=example.next_state if result.success else example.state,
            reward=result.reward,
            terminated=False,
            legal_actions=example.legal_actions,
            info={"score_delta": result.reward},
        ),
        metadata=result.metadata,
    )
