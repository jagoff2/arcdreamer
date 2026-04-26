from __future__ import annotations

import numpy as np
import pytest

from arcagi.learned_online.event_tokens import encode_action_tokens, encode_state_tokens
from arcagi.learned_online.object_event_bridge import (
    SelectedActionObservation,
    assert_no_forbidden_metadata_as_model_input,
    build_object_event_observation,
)
from arcagi.learned_online.object_event_curriculum import (
    ActiveOnlineObjectEventConfig,
    apply_synthetic_object_event_action_to_grid,
    build_active_online_object_event_curriculum,
    level_to_grid_observation,
    state_to_grid_observation,
)
from arcagi.learned_online.object_event_runtime_extraction import (
    ObjectEventRuntimeExtractor,
    extract_selected_transition_from_grid,
)


def test_runtime_extractor_preserves_dense_synthetic_action_surface() -> None:
    level = _level()
    observation = level_to_grid_observation(level)
    extractor = ObjectEventRuntimeExtractor()

    state = extractor.observe(observation)

    assert len(observation.available_actions) == 68
    assert len(state.affordances) == 68
    assert tuple(state.affordances) == tuple(level.example.legal_actions)


def test_runtime_extractor_selected_transition_matches_active_step_result() -> None:
    level = _level()
    selected = int(level.example.correct_action_index)
    before_observation, after_observation, result = apply_synthetic_object_event_action_to_grid(
        level,
        selected,
        before_step_index=0,
    )

    extracted = extract_selected_transition_from_grid(
        ObjectEventRuntimeExtractor(),
        before_observation=before_observation,
        selected_action=level.example.legal_actions[selected],
        after_observation=after_observation,
        reward=float(result.reward),
        terminated=False,
        info={"score_delta": float(result.reward), "level_boundary": bool(result.success)},
    )

    assert extracted.transition.action == level.example.legal_actions[selected]
    assert extracted.transition.reward == result.reward
    assert extracted.transition.state.affordances == level.example.legal_actions
    assert extracted.transition.next_state.affordances == level.example.legal_actions


def test_runtime_extractor_does_not_use_target_distractor_or_latent_metadata() -> None:
    observation = level_to_grid_observation(_level())
    text = repr(observation.extras).lower()

    for forbidden in ("target", "distractor", "latent_rule", "correct_action", "teacher", "trace", "state_hash"):
        assert forbidden not in text


def test_runtime_extractor_affordances_match_synthetic_level() -> None:
    level = _level()
    observation = level_to_grid_observation(level)
    state = ObjectEventRuntimeExtractor().observe(observation)

    assert tuple(state.affordances) == tuple(level.example.legal_actions)
    assert dict(state.action_roles) == dict(level.example.state.action_roles)


def test_runtime_extractor_bridge_selected_action_match() -> None:
    level = _level()
    selected = int(level.example.correct_action_index)
    before_observation, after_observation, result = apply_synthetic_object_event_action_to_grid(
        level,
        selected,
        before_step_index=0,
    )
    extracted = extract_selected_transition_from_grid(
        ObjectEventRuntimeExtractor(),
        before_observation=before_observation,
        selected_action=level.example.legal_actions[selected],
        after_observation=after_observation,
        reward=float(result.reward),
        terminated=False,
        info={"score_delta": float(result.reward), "level_boundary": bool(result.success)},
    )
    event = build_object_event_observation(
        SelectedActionObservation(
            before=extracted.before_state,
            selected_action=level.example.legal_actions[selected],
            after=extracted.after_state,
            reward=float(result.reward),
            terminated=False,
            legal_actions=level.example.legal_actions,
            info={"score_delta": float(result.reward)},
        ),
        metadata=result.metadata,
    )

    assert event.legal_action_count == 68
    assert event.selected_action_index == selected
    assert_no_forbidden_metadata_as_model_input(event.model_input_metadata())


def test_runtime_extractor_state_tokens_are_measured_against_structured_tokens() -> None:
    level = _level()
    observation = level_to_grid_observation(level)
    extracted = ObjectEventRuntimeExtractor().observe(observation)

    structured_state_tokens = level.example.state_tokens
    extracted_state_tokens = encode_state_tokens(extracted)
    structured_action_tokens = level.example.action_tokens
    extracted_action_tokens = encode_action_tokens(extracted, extracted.affordances)

    assert np.isfinite(np.linalg.norm(structured_state_tokens.numeric - extracted_state_tokens.numeric))
    assert np.isfinite(np.linalg.norm(structured_action_tokens.numeric - extracted_action_tokens.numeric))
    assert tuple(extracted.affordances) == tuple(level.example.legal_actions)


def test_runtime_extractor_no_policy_no_replay_no_graph_controller() -> None:
    extractor = ObjectEventRuntimeExtractor()

    with pytest.raises(RuntimeError):
        extractor.act(level_to_grid_observation(_level()))
    assert not hasattr(extractor, "planner")
    assert not hasattr(extractor, "runtime_rule_controller")
    assert not hasattr(extractor, "theory_manager")
    assert not hasattr(extractor, "trace_cursor")
    assert not hasattr(extractor, "action_sequence")


def test_state_to_grid_observation_keeps_public_cue_tags_only() -> None:
    level = _level()
    observation = state_to_grid_observation(level.example.state, level.example.legal_actions)
    cell_tags = observation.extras["cell_tags"]

    assert cell_tags
    assert any("agent" in tags for tags in cell_tags.values())
    assert all("target" not in tags for tags in cell_tags.values())
    assert all("distractor" not in tags for tags in cell_tags.values())


def _level():
    split = build_active_online_object_event_curriculum(
        ActiveOnlineObjectEventConfig(
            seed=901,
            train_sessions=1,
            heldout_sessions=0,
            levels_per_session=3,
            max_distractors=1,
            curriculum="latent_rule_variable_palette",
            palette_size=8,
            require_role_balanced_colors=True,
        )
    )
    return split.train[0].levels[0]
