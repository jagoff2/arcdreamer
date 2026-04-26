from __future__ import annotations

import numpy as np
import torch

from arcagi.core.action_schema import click_action_to_grid_cell
from arcagi.core.types import ObjectState, StructuredState
from arcagi.learned_online.event_tokens import ACTION_CLICK, OUT_NO_EFFECT_NONPROGRESS, encode_action_tokens
from arcagi.learned_online.object_event_bridge import (
    SelectedActionObservation,
    assert_no_forbidden_metadata_as_model_input,
    build_object_event_observation,
)
from arcagi.learned_online.object_event_curriculum import (
    ActiveOnlineObjectEventConfig,
    apply_synthetic_object_event_action,
    apply_synthetic_object_event_action_to_grid,
    build_active_online_object_event_curriculum,
    level_to_grid_observation,
    parametric_click_to_grid_cell,
)
from arcagi.learned_online.object_event_model import ObjectEventModel, ObjectEventModelConfig, policy_rank_logits_from_predictions
from arcagi.learned_online.object_event_runtime_extraction import ObjectEventRuntimeExtractor, extract_selected_transition_from_grid
from scripts.train_learned_online_object_event import _batch_to_tensors
from arcagi.learned_online.object_event_curriculum import collate_object_event_examples


def test_parametric_surface_preserves_requested_large_legal_count() -> None:
    level = _parametric_level(seed=201)
    example = level.example

    assert len(example.legal_actions) == 447
    assert int(example.action_tokens.mask.sum()) == 447
    assert example.metadata["action_surface_kind"] == "arc_scale_parametric"
    assert example.metadata["legal_action_count"] == 447


def test_parametric_surface_sets_interface_display_scale_for_extraction() -> None:
    state = _display_state()
    action = "click:31:61"
    tokens = encode_action_tokens(state, (action,))

    assert click_action_to_grid_cell(action, grid_shape=(8, 8), inventory=state.inventory_dict()) == (7, 3)
    assert parametric_click_to_grid_cell(action, grid_size=8, coordinate_grid_size=64) == (7, 3)
    assert tokens.action_type_ids[0] == ACTION_CLICK
    assert tokens.numeric[0, 4] == 1.0
    assert tokens.numeric[0, 8] == 1.0
    assert tokens.numeric[0, 9] == 1.0
    assert np.isclose(tokens.numeric[0, 10], 3.0 / 7.0)
    assert tokens.numeric[0, 11] == 1.0


def test_parametric_surface_has_many_empty_clicks_and_object_clicks() -> None:
    example = _parametric_level(seed=202).example

    empty_fraction = float(example.metadata["empty_action_count"]) / float(len(example.legal_actions))
    assert empty_fraction >= 0.55
    assert int(example.metadata["object_action_count"]) >= 4
    assert len(example.positive_action_indices) >= 2


def test_parametric_surface_positive_mask_can_have_multiple_actions() -> None:
    example = _parametric_level(seed=203).example

    assert example.positive_action_mask is not None
    assert int(np.asarray(example.positive_action_mask, dtype=bool).sum()) == len(example.positive_action_indices)
    assert len(example.positive_action_indices) >= 2
    assert all(example.candidate_targets.value[index] > 0.5 for index in example.positive_action_indices)


def test_parametric_active_step_success_uses_positive_mask_not_single_index() -> None:
    level = _parametric_level(seed=204)
    alternate = next(index for index in level.example.positive_action_indices if int(index) != int(level.example.correct_action_index))

    result = apply_synthetic_object_event_action(level, alternate)

    assert result.success is True
    assert result.reward == 1.0


def test_parametric_selected_no_effect_observation_is_selected_only() -> None:
    level = _parametric_level(seed=205)
    example = level.example
    selected = next(index for index, value in enumerate(example.candidate_targets.value) if float(value) == 0.0)
    result = apply_synthetic_object_event_action(level, selected)
    event = build_object_event_observation(
        SelectedActionObservation(
            before=example.state,
            selected_action=example.legal_actions[selected],
            after=example.state,
            reward=0.0,
            terminated=False,
            legal_actions=example.legal_actions,
            info={"score_delta": 0.0},
        ),
        metadata=result.metadata,
    )

    assert result.success is False
    assert result.no_effect is True
    assert event.selected_action_index == selected
    assert event.transition_targets.outcome[OUT_NO_EFFECT_NONPROGRESS] == 1.0


def test_parametric_bridge_preserves_large_legal_surface() -> None:
    level = _parametric_level(seed=206)
    selected = int(level.example.correct_action_index)
    result = apply_synthetic_object_event_action(level, selected)
    event = build_object_event_observation(
        SelectedActionObservation(
            before=level.example.state,
            selected_action=level.example.legal_actions[selected],
            after=level.example.next_state,
            reward=float(result.reward),
            terminated=False,
            legal_actions=level.example.legal_actions,
            info={"score_delta": float(result.reward)},
        ),
        metadata=result.metadata,
    )

    assert event.legal_action_count == 447
    assert event.selected_action_index == selected
    assert_no_forbidden_metadata_as_model_input(event.model_input_metadata())


def test_parametric_extracted_state_source_has_no_latent_or_role_leakage() -> None:
    level = _parametric_level(seed=207)
    observation = level_to_grid_observation(level)
    state = ObjectEventRuntimeExtractor().observe(observation)
    text = " ".join([repr(state.inventory).lower(), repr(state.flags).lower(), repr(state.action_roles).lower()])

    assert len(state.affordances) == 447
    assert state.inventory_dict()["interface_display_scale"] == "8"
    for forbidden in ("target", "distractor", "latent_rule", "correct_action", "teacher", "trace_cursor"):
        assert forbidden not in text


def test_parametric_runtime_extraction_preserves_selected_transition_surface() -> None:
    level = _parametric_level(seed=208)
    selected = int(level.example.correct_action_index)
    before, after, result = apply_synthetic_object_event_action_to_grid(level, selected, before_step_index=0)
    extracted = extract_selected_transition_from_grid(
        ObjectEventRuntimeExtractor(),
        before_observation=before,
        selected_action=level.example.legal_actions[selected],
        after_observation=after,
        reward=float(result.reward),
        terminated=False,
        info={"score_delta": float(result.reward), "level_boundary": bool(result.success)},
    )

    assert extracted.before_state.affordances == level.example.legal_actions
    assert extracted.after_state.affordances == level.example.legal_actions
    assert extracted.before_state.inventory_dict()["interface_display_scale"] == "8"


def test_failed_action_memory_changes_scores_without_hard_masking() -> None:
    level = _parametric_level(seed=209)
    example = level.example
    failed = next(index for index, value in enumerate(example.candidate_targets.value) if float(value) == 0.0)
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()
    with torch.no_grad():
        model.failed_action_memory_rank.rank_mlp[-1].weight.fill_(-0.25)
        model.failed_action_memory_rank.rank_mlp[-1].bias.zero_()
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    zero = torch.zeros((1, model.config.d_model), dtype=torch.float32)
    before = model(**tensors["inputs"], level_belief=zero)
    before_logits = policy_rank_logits_from_predictions(before, tensors["action_mask"])
    observed = dict(tensors)
    observed["inputs"] = dict(tensors["inputs"])
    observed["actual_action_index"] = torch.as_tensor([failed], dtype=torch.long)
    observed["target_outcome"] = tensors["candidate_outcome_targets"][:, failed]
    observed["target_delta"] = tensors["candidate_delta_targets"][:, failed]
    deltas = model.observed_event_belief_deltas(
        before,
        target_outcome=observed["target_outcome"],
        target_delta=observed["target_delta"],
        actual_action_index=observed["actual_action_index"],
        state_numeric=tensors["inputs"]["state_numeric"],
        state_type_ids=tensors["inputs"]["state_type_ids"],
        state_mask=tensors["inputs"]["state_mask"],
        action_numeric=tensors["inputs"]["action_numeric"],
    )
    after = model(**tensors["inputs"], level_belief=deltas.level_delta)
    after_logits = policy_rank_logits_from_predictions(after, tensors["action_mask"])

    assert before_logits.shape[-1] == 447
    assert after_logits.shape[-1] == 447
    assert torch.isfinite(after_logits[0, failed])
    assert after_logits[0, failed] != before_logits[0, failed]


def test_failed_action_memory_resets_on_level_reset() -> None:
    level = _parametric_level(seed=210)
    example = level.example
    failed = next(index for index, value in enumerate(example.candidate_targets.value) if float(value) == 0.0)
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()
    with torch.no_grad():
        model.failed_action_memory_rank.rank_mlp[-1].weight.fill_(-0.25)
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    zero = torch.zeros((1, model.config.d_model), dtype=torch.float32)
    before = model(**tensors["inputs"], level_belief=zero)
    observed_outcome = tensors["candidate_outcome_targets"][:, failed]
    observed_delta = tensors["candidate_delta_targets"][:, failed]
    deltas = model.observed_event_belief_deltas(
        before,
        target_outcome=observed_outcome,
        target_delta=observed_delta,
        actual_action_index=torch.as_tensor([failed], dtype=torch.long),
        state_numeric=tensors["inputs"]["state_numeric"],
        state_type_ids=tensors["inputs"]["state_type_ids"],
        state_mask=tensors["inputs"]["state_mask"],
        action_numeric=tensors["inputs"]["action_numeric"],
    )
    changed = policy_rank_logits_from_predictions(model(**tensors["inputs"], level_belief=deltas.level_delta), tensors["action_mask"])
    reset = policy_rank_logits_from_predictions(model(**tensors["inputs"], level_belief=zero), tensors["action_mask"])

    assert changed[0, failed] != reset[0, failed]


def _parametric_level(*, seed: int):
    split = build_active_online_object_event_curriculum(
        ActiveOnlineObjectEventConfig(
            seed=seed,
            train_sessions=1,
            heldout_sessions=0,
            levels_per_session=3,
            max_distractors=2,
            curriculum="latent_rule_variable_palette",
            palette_size=8,
            require_role_balanced_colors=True,
            action_surface="arc_scale_parametric",
            action_surface_size=447,
            coordinate_grid_size=64,
            empty_click_fraction=0.80,
            positive_region_radius=1,
        )
    )
    return split.train[0].levels[0]


def _display_state() -> StructuredState:
    grid = np.zeros((8, 8), dtype=np.int64)
    grid[7, 3] = 4
    obj = ObjectState("obj", 4, ((7, 3),), (7, 3, 7, 3), (7.0, 3.0), 1)
    return StructuredState(
        task_id="display",
        episode_id="0",
        step_index=0,
        grid_shape=(8, 8),
        grid_signature=tuple(int(value) for value in grid.reshape(-1)),
        objects=(obj,),
        relations=(),
        affordances=("click:31:61",),
        action_roles=(("click:31:61", "click"),),
        inventory=(
            ("interface_display_scale", "8"),
            ("interface_display_pad_x", "0"),
            ("interface_display_pad_y", "0"),
        ),
    )
