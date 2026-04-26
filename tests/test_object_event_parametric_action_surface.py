from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from arcagi.core.action_schema import click_action_to_grid_cell
from arcagi.core.types import ObjectState, StructuredState
from arcagi.learned_online.event_tokens import ACTION_CLICK, OUT_NO_EFFECT_NONPROGRESS, OUTCOME_DIM, encode_action_tokens
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
from scripts.train_learned_online_object_event import (
    _action_basis_known_noeffect_mask,
    _action_basis_posterior_diagnostic_targets,
    _batch_to_tensors,
    _far_click_action_indices,
    _family_assignment_regularization_losses,
    _family_known_noeffect_mask,
    _family_posterior_diagnostic_targets,
    _failure_contingent_action_diversity_metrics,
    _information_gain_from_hypothesis_success,
    _near_action_indices,
    _noeffect_contradiction_masks,
    _noeffect_contradiction_training_losses,
    _post_noeffect_window_metrics,
    _rank_trace_noeffect_contradiction_metrics,
    _selected_basis_overlap,
    _with_balanced_runtime_score,
)
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
    failed = _failed_click_index(example)
    model = ObjectEventModel(ObjectEventModelConfig(d_model=128, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
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
    failed = _failed_click_index(example)
    model = ObjectEventModel(ObjectEventModelConfig(d_model=128, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
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


def test_coordinate_noeffect_memory_writes_only_on_noeffect_click() -> None:
    level = _parametric_level(seed=211)
    example = level.example
    failed = _failed_click_index(example)
    success = int(example.positive_action_indices[0])
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    base = model(**tensors["inputs"])

    failed_deltas = model.observed_event_belief_deltas(
        base,
        target_outcome=tensors["candidate_outcome_targets"][:, failed],
        target_delta=tensors["candidate_delta_targets"][:, failed],
        actual_action_index=torch.as_tensor([failed], dtype=torch.long),
        state_numeric=tensors["inputs"]["state_numeric"],
        state_type_ids=tensors["inputs"]["state_type_ids"],
        state_mask=tensors["inputs"]["state_mask"],
        action_numeric=tensors["inputs"]["action_numeric"],
    )
    success_deltas = model.observed_event_belief_deltas(
        base,
        target_outcome=tensors["candidate_outcome_targets"][:, success],
        target_delta=tensors["candidate_delta_targets"][:, success],
        actual_action_index=torch.as_tensor([success], dtype=torch.long),
        state_numeric=tensors["inputs"]["state_numeric"],
        state_type_ids=tensors["inputs"]["state_type_ids"],
        state_mask=tensors["inputs"]["state_mask"],
        action_numeric=tensors["inputs"]["action_numeric"],
    )
    start = model.coordinate_noeffect_memory_rank.start
    stop = model.coordinate_noeffect_memory_rank.stop

    assert torch.linalg.vector_norm(failed_deltas.level_delta[:, start:stop]) > 0.0
    assert torch.linalg.vector_norm(failed_deltas.session_delta[:, start:stop]) == 0.0
    assert torch.linalg.vector_norm(success_deltas.level_delta[:, start:stop]) == 0.0


def test_coordinate_noeffect_rank_changes_nearby_click_scores_without_hard_masking() -> None:
    level = _parametric_level(seed=212)
    example = level.example
    failed = _failed_click_index(example)
    failed_action = example.legal_actions[failed]
    near = _near_action_indices(example, failed_action)
    far = _far_click_action_indices(example, failed_action)
    assert near
    assert far
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()
    with torch.no_grad():
        model.coordinate_noeffect_memory_rank.rank_mlp[-1].weight.fill_(-0.25)
        model.coordinate_noeffect_memory_rank.rank_mlp[-1].bias.zero_()
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    zero = torch.zeros((1, model.config.d_model), dtype=torch.float32)
    before = policy_rank_logits_from_predictions(model(**tensors["inputs"], level_belief=zero), tensors["action_mask"])
    base = model(**tensors["inputs"], level_belief=zero)
    deltas = model.observed_event_belief_deltas(
        base,
        target_outcome=tensors["candidate_outcome_targets"][:, failed],
        target_delta=tensors["candidate_delta_targets"][:, failed],
        actual_action_index=torch.as_tensor([failed], dtype=torch.long),
        state_numeric=tensors["inputs"]["state_numeric"],
        state_type_ids=tensors["inputs"]["state_type_ids"],
        state_mask=tensors["inputs"]["state_mask"],
        action_numeric=tensors["inputs"]["action_numeric"],
    )
    after = policy_rank_logits_from_predictions(model(**tensors["inputs"], level_belief=deltas.level_delta), tensors["action_mask"])

    assert before.shape[-1] == 447
    assert after.shape[-1] == 447
    assert torch.isfinite(after[0, failed])
    assert torch.isfinite(after[0, list(near)]).all()
    assert torch.isfinite(after[0, list(far)]).all()
    assert torch.mean(torch.abs(after[0, list(near)] - before[0, list(near)])) > 0.0
    reset = policy_rank_logits_from_predictions(model(**tensors["inputs"], level_belief=zero), tensors["action_mask"])
    assert torch.allclose(reset, before)


def test_axis_noeffect_memory_is_level_local_and_noeffect_only() -> None:
    level = _parametric_level(seed=213)
    example = level.example
    failed = _failed_click_index(example)
    success = int(example.positive_action_indices[0])
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    base = model(**tensors["inputs"])
    failed_deltas = model.observed_event_belief_deltas(
        base,
        target_outcome=tensors["candidate_outcome_targets"][:, failed],
        target_delta=tensors["candidate_delta_targets"][:, failed],
        actual_action_index=torch.as_tensor([failed], dtype=torch.long),
        state_numeric=tensors["inputs"]["state_numeric"],
        state_type_ids=tensors["inputs"]["state_type_ids"],
        state_mask=tensors["inputs"]["state_mask"],
        action_numeric=tensors["inputs"]["action_numeric"],
    )
    success_deltas = model.observed_event_belief_deltas(
        base,
        target_outcome=tensors["candidate_outcome_targets"][:, success],
        target_delta=tensors["candidate_delta_targets"][:, success],
        actual_action_index=torch.as_tensor([success], dtype=torch.long),
        state_numeric=tensors["inputs"]["state_numeric"],
        state_type_ids=tensors["inputs"]["state_type_ids"],
        state_mask=tensors["inputs"]["state_mask"],
        action_numeric=tensors["inputs"]["action_numeric"],
    )
    start = model.axis_noeffect_memory_rank.start
    stop = model.axis_noeffect_memory_rank.stop

    assert torch.linalg.vector_norm(failed_deltas.level_delta[:, start:stop]) > 0.0
    assert torch.linalg.vector_norm(failed_deltas.session_delta[:, start:stop]) == 0.0
    assert torch.linalg.vector_norm(success_deltas.level_delta[:, start:stop]) == 0.0


def test_axis_noeffect_rank_changes_same_column_scores_without_hard_mask() -> None:
    level = _parametric_level(seed=214)
    example = level.example
    failed_a, failed_b = _two_failed_clicks_same_x(example)
    failed_action = example.legal_actions[failed_a]
    same_x = tuple(index for index, action in enumerate(example.legal_actions) if action.startswith("click:") and action.split(":")[1] == failed_action.split(":")[1])
    far = tuple(index for index, action in enumerate(example.legal_actions) if action.startswith("click:") and action.split(":")[1] != failed_action.split(":")[1])
    assert same_x
    assert far
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()
    with torch.no_grad():
        model.axis_noeffect_memory_rank.rank_mlp[-1].weight.fill_(-0.25)
        model.axis_noeffect_memory_rank.rank_mlp[-1].bias.zero_()
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    zero = torch.zeros((1, model.config.d_model), dtype=torch.float32)
    before = policy_rank_logits_from_predictions(model(**tensors["inputs"], level_belief=zero), tensors["action_mask"])
    level_belief = zero.clone()
    for failed in (failed_a, failed_b):
        output = model(**tensors["inputs"], level_belief=level_belief)
        deltas = model.observed_event_belief_deltas(
            output,
            target_outcome=tensors["candidate_outcome_targets"][:, failed],
            target_delta=tensors["candidate_delta_targets"][:, failed],
            actual_action_index=torch.as_tensor([failed], dtype=torch.long),
            state_numeric=tensors["inputs"]["state_numeric"],
            state_type_ids=tensors["inputs"]["state_type_ids"],
            state_mask=tensors["inputs"]["state_mask"],
            action_numeric=tensors["inputs"]["action_numeric"],
        )
        level_belief = level_belief + deltas.level_delta
    after = policy_rank_logits_from_predictions(model(**tensors["inputs"], level_belief=level_belief), tensors["action_mask"])

    assert before.shape[-1] == 447
    assert after.shape[-1] == 447
    assert torch.isfinite(after[0, list(same_x)]).all()
    assert torch.isfinite(after[0, list(far)]).all()
    assert torch.mean(torch.abs(after[0, list(same_x)] - before[0, list(same_x)])) > 0.0
    reset = policy_rank_logits_from_predictions(model(**tensors["inputs"], level_belief=zero), tensors["action_mask"])
    assert torch.allclose(reset, before)


def test_relation_prior_scales_are_bounded() -> None:
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    relation = model.event_relation_memory_rank

    with torch.no_grad():
        relation.object_prior_scale_raw.fill_(100.0)
        relation.positive_prior_scale_raw.fill_(100.0)
        relation.negative_prior_scale_raw.fill_(100.0)
        relation.repeat_penalty_scale_raw.fill_(100.0)

    assert float((2.0 * torch.sigmoid(relation.object_prior_scale_raw)).detach()) <= 2.0
    assert float((4.0 * torch.sigmoid(relation.positive_prior_scale_raw)).detach()) <= 4.0
    assert float((4.0 * torch.sigmoid(relation.negative_prior_scale_raw)).detach()) <= 4.0
    assert float((6.0 * torch.sigmoid(relation.repeat_penalty_scale_raw)).detach()) <= 6.0


def test_rank_component_standardization_prevents_relation_swamping() -> None:
    level = _parametric_level(seed=215)
    tensors = _batch_to_tensors(collate_object_event_examples((level.example,)), device=torch.device("cpu"))
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()
    with torch.no_grad():
        model.event_relation_memory_rank.rank_mlp[-1].weight.fill_(50.0)
        model.event_relation_memory_rank.rank_mlp[-1].bias.fill_(50.0)
        model.raw_candidate_rank.rank_component_gates[1].fill_(10.0)

    output = model(**tensors["inputs"])
    components = output.rank_components
    assert components is not None
    valid = tensors["action_mask"].bool()
    relation = components.relation[valid]
    total = components.total[valid]

    assert output.rank_logits.shape[-1] == 447
    assert torch.isfinite(output.rank_logits[valid]).all()
    assert float(torch.abs(relation.mean()).detach()) < 1.0e-4
    assert float(relation.std(unbiased=False).detach()) <= 1.05
    assert float(total.std(unbiased=False).detach()) < 20.0


def test_contradiction_gate_increases_on_noeffect_object_evidence() -> None:
    level = _parametric_level(seed=216)
    example = level.example
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    failed = _failed_object_click_index(example, tensors)
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()
    zero = torch.zeros((1, model.config.d_model), dtype=torch.float32)
    before = model(**tensors["inputs"], level_belief=zero)
    before_components = before.rank_components
    assert before_components is not None
    deltas = model.observed_event_belief_deltas(
        before,
        target_outcome=tensors["candidate_outcome_targets"][:, failed],
        target_delta=tensors["candidate_delta_targets"][:, failed],
        actual_action_index=torch.as_tensor([failed], dtype=torch.long),
        state_numeric=tensors["inputs"]["state_numeric"],
        state_type_ids=tensors["inputs"]["state_type_ids"],
        state_mask=tensors["inputs"]["state_mask"],
        action_numeric=tensors["inputs"]["action_numeric"],
    )
    after = model(**tensors["inputs"], level_belief=deltas.level_delta)
    after_components = after.rank_components
    assert after_components is not None

    candidate_contains = tensors["inputs"]["action_numeric"][0, failed, 11] * (
        1.0 - tensors["inputs"]["action_numeric"][0, failed, 24]
    )
    ungated_object_prior = 2.0 * torch.sigmoid(model.event_relation_memory_rank.object_prior_scale_raw) * candidate_contains

    assert float(before_components.relation_contradiction_gate[0, failed].detach()) == 0.0
    assert float(after_components.relation_contradiction_gate[0, failed].detach()) > 0.0
    assert float(after_components.relation_object_prior[0, failed].detach()) < float(ungated_object_prior.detach())
    assert torch.isfinite(after.rank_logits[tensors["action_mask"]]).all()


def test_component_gating_keeps_all_actions_legal_and_finite_after_noeffect_column() -> None:
    level = _parametric_level(seed=217)
    example = level.example
    failed_a, failed_b = _two_failed_clicks_same_x(example)
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    level_belief = torch.zeros((1, model.config.d_model), dtype=torch.float32)
    for failed in (failed_a, failed_b):
        output = model(**tensors["inputs"], level_belief=level_belief)
        deltas = model.observed_event_belief_deltas(
            output,
            target_outcome=tensors["candidate_outcome_targets"][:, failed],
            target_delta=tensors["candidate_delta_targets"][:, failed],
            actual_action_index=torch.as_tensor([failed], dtype=torch.long),
            state_numeric=tensors["inputs"]["state_numeric"],
            state_type_ids=tensors["inputs"]["state_type_ids"],
            state_mask=tensors["inputs"]["state_mask"],
            action_numeric=tensors["inputs"]["action_numeric"],
        )
        level_belief = level_belief + deltas.level_delta
    after = model(**tensors["inputs"], level_belief=level_belief)
    logits = policy_rank_logits_from_predictions(after, tensors["action_mask"])
    components = after.rank_components

    assert components is not None
    assert logits.shape[-1] == 447
    assert torch.isfinite(logits[tensors["action_mask"]]).all()
    assert torch.isfinite(components.total[tensors["action_mask"]]).all()
    assert components.component_gates is not None
    assert torch.isfinite(components.component_gates).all()
    for forbidden in ("tried_actions", "blocked_actions", "action_blacklist", "sweep_index", "frontier", "coverage_queue"):
        assert not hasattr(model, forbidden)


def test_noeffect_contradiction_penalty_increases_after_noeffect_evidence_without_masking() -> None:
    level = _parametric_level(seed=228)
    example = level.example
    failed = _failed_click_index(example)
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    zero = torch.zeros((1, model.config.d_model), dtype=torch.float32)
    before = model(**tensors["inputs"], level_belief=zero)
    deltas = model.observed_event_belief_deltas(
        before,
        target_outcome=tensors["candidate_outcome_targets"][:, failed],
        target_delta=tensors["candidate_delta_targets"][:, failed],
        actual_action_index=torch.as_tensor([failed], dtype=torch.long),
        state_numeric=tensors["inputs"]["state_numeric"],
        state_type_ids=tensors["inputs"]["state_type_ids"],
        state_mask=tensors["inputs"]["state_mask"],
        action_numeric=tensors["inputs"]["action_numeric"],
    )
    after = model(**tensors["inputs"], level_belief=deltas.level_delta)

    assert before.noeffect_contradiction_gate is not None
    assert before.noeffect_contradiction_penalty is not None
    assert after.noeffect_contradiction_gate is not None
    assert after.noeffect_contradiction_penalty is not None
    assert after.noeffect_contradiction_penalty.shape[-1] == 447
    assert float(after.noeffect_contradiction_penalty[0, failed].detach()) > float(
        before.noeffect_contradiction_penalty[0, failed].detach()
    )
    logits = policy_rank_logits_from_predictions(after, tensors["action_mask"])
    assert torch.isfinite(logits[tensors["action_mask"]]).all()
    assert int(tensors["action_mask"].sum().detach()) == 447


def test_noeffect_contradiction_penalty_subtracts_from_rank_without_action_mask() -> None:
    level = _parametric_level(seed=229)
    tensors = _batch_to_tensors(collate_object_event_examples((level.example,)), device=torch.device("cpu"))
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()

    output = model(**tensors["inputs"])
    assert output.rank_components is not None
    assert output.noeffect_contradiction_penalty is not None
    raw_rank = model.rank_score_head(output.action_repr).squeeze(-1) + output.rank_components.total

    assert torch.allclose(output.rank_logits, raw_rank - output.noeffect_contradiction_penalty, atol=1.0e-5)
    assert output.rank_logits.shape[-1] == 447
    assert torch.isfinite(output.rank_logits[tensors["action_mask"]]).all()
    for forbidden in ("tried_actions", "blocked_actions", "action_blacklist", "avoid_column", "frontier", "coverage_queue"):
        assert not hasattr(model, forbidden)


def test_known_contradiction_mask_excludes_generic_noeffect_without_posterior_evidence() -> None:
    output, tensors, positive_mask, relation_failed_mask, coord_failure_mask = _toy_noeffect_contradiction_inputs(
        no_effect_prob=0.97,
        posterior_noeffect=0.0,
    )

    masks = _noeffect_contradiction_masks(
        output,
        tensors,
        positive_mask=positive_mask,
        relation_failed_mask=relation_failed_mask,
        coord_failure_mask=coord_failure_mask,
    )

    assert not bool(masks["known_mask"][0, 1])
    assert not bool(masks["broad_mask"][0, 1])


def test_known_contradiction_mask_includes_high_posterior_noeffect_candidate() -> None:
    output, tensors, positive_mask, relation_failed_mask, coord_failure_mask = _toy_noeffect_contradiction_inputs(
        no_effect_prob=0.97,
        posterior_noeffect=0.85,
    )

    masks = _noeffect_contradiction_masks(
        output,
        tensors,
        positive_mask=positive_mask,
        relation_failed_mask=relation_failed_mask,
        coord_failure_mask=coord_failure_mask,
    )

    assert bool(masks["known_mask"][0, 1])
    assert bool(masks["broad_mask"][0, 1])


def test_runtime_contradiction_metrics_use_synthetic_noeffect_target_when_prediction_is_low() -> None:
    level = _parametric_level(seed=230)
    example = level.example
    failed = _failed_click_index(example)
    positive = int(example.positive_action_indices[0])
    assert float(example.candidate_targets.outcome[failed, OUT_NO_EFFECT_NONPROGRESS]) > 0.5

    metrics = _rank_trace_noeffect_contradiction_metrics(
        example,
        failed_action=example.legal_actions[failed],
        records=(
            {
                "action": example.legal_actions[failed],
                "action_index": failed,
                "total_score": 3.0,
                "probability": 0.80,
                "out_no_effect_prob": 0.01,
                "basis_noeffect": 0.90,
                "family_noeffect": 0.0,
                "noeffect_contradiction_gate": 0.70,
                "noeffect_contradiction_penalty": 2.0,
            },
            {
                "action": example.legal_actions[positive],
                "action_index": positive,
                "total_score": 1.0,
                "probability": 0.20,
                "out_no_effect_prob": 0.01,
                "basis_noeffect": 0.0,
                "family_noeffect": 0.0,
                "noeffect_contradiction_gate": 0.0,
                "noeffect_contradiction_penalty": 0.0,
            },
        ),
    )

    assert metrics["counted"] is True
    assert metrics["known_available"] is True
    assert metrics["known_top1"] == 1.0
    assert metrics["broad_top1"] == 1.0
    assert metrics["known_policy_mass"] == 0.80


def test_known_contradiction_penalty_min_loss_rewards_larger_known_penalty() -> None:
    output, tensors, positive_mask, relation_failed_mask, coord_failure_mask = _toy_noeffect_contradiction_inputs(
        no_effect_prob=0.97,
        posterior_noeffect=0.85,
        known_penalty=0.25,
    )
    logits = torch.tensor([[2.0, 4.0, 0.0]], dtype=torch.float32)
    low_penalty = _noeffect_contradiction_training_losses(
        output,
        tensors,
        logits=logits,
        positive_mask=positive_mask,
        relation_failed_mask=relation_failed_mask,
        coord_failure_mask=coord_failure_mask,
    )
    output.noeffect_contradiction_penalty[0, 1] = 2.25
    high_penalty = _noeffect_contradiction_training_losses(
        output,
        tensors,
        logits=logits,
        positive_mask=positive_mask,
        relation_failed_mask=relation_failed_mask,
        coord_failure_mask=coord_failure_mask,
    )

    assert float(low_penalty["known_penalty_min_loss"]) > float(high_penalty["known_penalty_min_loss"])
    assert float(high_penalty["known_penalty_mean"]) > 1.75


def test_known_contradiction_policy_mass_loss_tracks_known_probability() -> None:
    output, tensors, positive_mask, relation_failed_mask, coord_failure_mask = _toy_noeffect_contradiction_inputs(
        no_effect_prob=0.97,
        posterior_noeffect=0.85,
    )
    high_known_logits = torch.tensor([[0.0, 4.0, 0.0]], dtype=torch.float32)
    low_known_logits = torch.tensor([[4.0, -4.0, 0.0]], dtype=torch.float32)

    high_mass = _noeffect_contradiction_training_losses(
        output,
        tensors,
        logits=high_known_logits,
        positive_mask=positive_mask,
        relation_failed_mask=relation_failed_mask,
        coord_failure_mask=coord_failure_mask,
    )
    low_mass = _noeffect_contradiction_training_losses(
        output,
        tensors,
        logits=low_known_logits,
        positive_mask=positive_mask,
        relation_failed_mask=relation_failed_mask,
        coord_failure_mask=coord_failure_mask,
    )

    assert float(high_mass["known_policy_mass"]) > float(low_mass["known_policy_mass"])
    assert float(high_mass["known_top1_rate"]) == 1.0
    assert float(low_mass["known_top1_rate"]) == 0.0


def test_known_contradiction_penalty_leaves_known_action_legal_and_finite_scored() -> None:
    output, tensors, positive_mask, relation_failed_mask, coord_failure_mask = _toy_noeffect_contradiction_inputs(
        no_effect_prob=0.97,
        posterior_noeffect=0.85,
        known_penalty=2.0,
    )
    logits = torch.tensor([[1.0, 3.0, -1.0]], dtype=torch.float32) - output.noeffect_contradiction_penalty

    losses = _noeffect_contradiction_training_losses(
        output,
        tensors,
        logits=logits,
        positive_mask=positive_mask,
        relation_failed_mask=relation_failed_mask,
        coord_failure_mask=coord_failure_mask,
    )

    assert bool(tensors["action_mask"][0, 1])
    assert torch.isfinite(logits[tensors["action_mask"]]).all()
    assert float(losses["known_penalty_mean"]) == 2.0


def test_diagnostic_utility_logits_present_and_full_surface() -> None:
    level = _parametric_level(seed=218)
    tensors = _batch_to_tensors(collate_object_event_examples((level.example,)), device=torch.device("cpu"))
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()

    output = model(**tensors["inputs"])

    assert output.diagnostic_utility_logits is not None
    assert output.diagnostic_utility_logits.shape == (1, 447)
    assert torch.isfinite(output.diagnostic_utility_logits[tensors["action_mask"]]).all()
    assert output.action_family_posterior_features is not None
    assert output.action_family_posterior_features.shape == (1, 447, 4)
    assert torch.isfinite(output.action_family_posterior_features[tensors["action_mask"]]).all()
    assert output.action_basis_posterior_features is not None
    assert output.action_basis_posterior_features.shape == (1, 447, 4)
    assert torch.isfinite(output.action_basis_posterior_features[tensors["action_mask"]]).all()


def test_action_family_belief_updates_selected_transition_level_only() -> None:
    level = _parametric_level(seed=222)
    example = level.example
    failed = _failed_click_index(example)
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()

    output = model(**tensors["inputs"])
    deltas = model.observed_event_belief_deltas(
        output,
        target_outcome=tensors["candidate_outcome_targets"][:, failed],
        target_delta=tensors["candidate_delta_targets"][:, failed],
        actual_action_index=torch.as_tensor([failed], dtype=torch.long),
        state_numeric=tensors["inputs"]["state_numeric"],
        state_type_ids=tensors["inputs"]["state_type_ids"],
        state_mask=tensors["inputs"]["state_mask"],
        action_numeric=tensors["inputs"]["action_numeric"],
    )
    start = model.action_family_belief.start
    stop = model.action_family_belief.stop

    assert stop > start
    assert torch.linalg.vector_norm(deltas.level_delta[:, start:stop]) > 0.0
    assert torch.linalg.vector_norm(deltas.session_delta[:, start:stop]) == 0.0
    for forbidden in ("visited_bins", "least_visited", "untried", "tried_families", "family_blacklist", "coverage_map", "frontier", "sweep_index", "probe_counter"):
        assert not hasattr(model, forbidden)


def test_action_basis_belief_updates_selected_transition_level_only() -> None:
    level = _parametric_level(seed=225)
    example = level.example
    failed = _failed_click_index(example)
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()

    output = model(**tensors["inputs"])
    deltas = model.observed_event_belief_deltas(
        output,
        target_outcome=tensors["candidate_outcome_targets"][:, failed],
        target_delta=tensors["candidate_delta_targets"][:, failed],
        actual_action_index=torch.as_tensor([failed], dtype=torch.long),
        state_numeric=tensors["inputs"]["state_numeric"],
        state_type_ids=tensors["inputs"]["state_type_ids"],
        state_mask=tensors["inputs"]["state_mask"],
        action_numeric=tensors["inputs"]["action_numeric"],
    )
    start = model.action_basis_belief.start
    stop = model.action_basis_belief.stop

    assert stop > start
    assert model.action_basis_belief.num_bases >= 8
    assert torch.linalg.vector_norm(deltas.level_delta[:, start:stop]) > 0.0
    assert torch.linalg.vector_norm(deltas.session_delta[:, start:stop]) == 0.0
    zero_level = torch.zeros((1, model.config.d_model), dtype=torch.float32)
    reset_output = model(**tensors["inputs"], level_belief=zero_level)
    assert reset_output.action_basis_posterior_features is not None
    assert float(reset_output.action_basis_posterior_features[..., 3].max().detach()) == 0.0
    for forbidden in ("visited_basis", "least_visited", "untried", "basis_blacklist", "coverage_map", "frontier", "sweep_index", "probe_counter"):
        assert not hasattr(model, forbidden)


def test_action_basis_representation_is_fixed_smooth_and_order_equivariant() -> None:
    level = _parametric_level(seed=226)
    example = level.example
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    config = ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0)
    model_a = ObjectEventModel(config)
    model_b = ObjectEventModel(config)
    action_numeric = tensors["inputs"]["action_numeric"]

    basis_a = model_a.action_basis_belief.basis(action_numeric)
    basis_b = model_b.action_basis_belief.basis(action_numeric)
    permutation = torch.randperm(action_numeric.shape[1])
    shuffled_basis = model_a.action_basis_belief.basis(action_numeric[:, permutation])

    assert not list(model_a.action_basis_belief.parameters())
    assert torch.allclose(basis_a, basis_b)
    assert torch.allclose(shuffled_basis, basis_a[:, permutation])
    assert torch.isfinite(basis_a[tensors["action_mask"]]).all()
    selected = int(example.metadata["correct_action_index"])
    near = _near_action_indices(example, example.legal_actions[selected])
    far = _far_click_action_indices(example, example.legal_actions[selected])
    if near and far:
        selected_basis = basis_a[0, selected]
        near_overlap = torch.max(torch.sum(basis_a[0, list(near)] * selected_basis[None, :], dim=-1))
        far_overlap = torch.mean(torch.sum(basis_a[0, list(far)] * selected_basis[None, :], dim=-1))
        assert float(near_overlap.detach()) > float(far_overlap.detach())


def test_family_posterior_target_prefers_uncertain_over_known_noeffect() -> None:
    action_mask = torch.ones((1, 3), dtype=torch.bool)
    features = torch.as_tensor(
        [[[0.5, 0.25, 0.1, 0.0], [0.2, 0.05, 1.0, 1.0], [0.5, 0.20, 0.1, 0.0]]],
        dtype=torch.float32,
    )

    output = type("Output", (), {})()
    output.action_family_posterior_features = features

    target = _family_posterior_diagnostic_targets(output, {"action_mask": action_mask})
    known_noeffect = _family_known_noeffect_mask(output, {"action_mask": action_mask})

    assert float(target[0, 0]) > float(target[0, 1])
    assert bool(known_noeffect[0, 1])
    assert not bool(known_noeffect[0, 0])


def test_family_posterior_target_penalizes_selected_failed_family_overlap() -> None:
    action_mask = torch.ones((1, 3), dtype=torch.bool)
    features = torch.as_tensor(
        [[[0.5, 0.25, 0.1, 0.0], [0.5, 0.25, 0.1, 0.0], [0.5, 0.20, 0.1, 0.0]]],
        dtype=torch.float32,
    )
    overlap = torch.as_tensor([[0.9, 0.1, 0.2]], dtype=torch.float32)
    output = type("Output", (), {})()
    output.action_family_posterior_features = features

    target = _family_posterior_diagnostic_targets(output, {"action_mask": action_mask}, selected_family_overlap=overlap)

    assert float(target[0, 1]) > float(target[0, 0])
    assert float(target[0, 0]) < 0.2


def test_action_basis_target_prefers_uncertain_nonoverlapping_basis_after_noeffect() -> None:
    action_mask = torch.ones((1, 3), dtype=torch.bool)
    features = torch.as_tensor(
        [[[0.5, 0.25, 0.1, 0.0], [0.5, 0.25, 0.1, 0.0], [0.5, 0.05, 1.0, 1.0]]],
        dtype=torch.float32,
    )
    overlap = torch.as_tensor([[0.9, 0.1, 0.1]], dtype=torch.float32)
    output = type("Output", (), {})()
    output.action_basis_posterior_features = features

    target = _action_basis_posterior_diagnostic_targets(
        output,
        {"action_mask": action_mask},
        selected_basis_overlap=overlap,
    )
    known_noeffect = _action_basis_known_noeffect_mask(output, {"action_mask": action_mask})

    assert float(target[0, 1]) > float(target[0, 0])
    assert float(target[0, 0]) < 0.2
    assert bool(known_noeffect[0, 2])
    assert not bool(known_noeffect[0, 1])


def test_selected_basis_overlap_keeps_same_basis_legal_and_finite() -> None:
    level = _parametric_level(seed=227)
    example = level.example
    failed = _failed_click_index(example)
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()

    output = model(**tensors["inputs"])
    deltas = model.observed_event_belief_deltas(
        output,
        target_outcome=tensors["candidate_outcome_targets"][:, failed],
        target_delta=tensors["candidate_delta_targets"][:, failed],
        actual_action_index=torch.as_tensor([failed], dtype=torch.long),
        state_numeric=tensors["inputs"]["state_numeric"],
        state_type_ids=tensors["inputs"]["state_type_ids"],
        state_mask=tensors["inputs"]["state_mask"],
        action_numeric=tensors["inputs"]["action_numeric"],
    )
    after = model(**tensors["inputs"], level_belief=deltas.level_delta)
    logits = policy_rank_logits_from_predictions(after, tensors["action_mask"])
    overlap = _selected_basis_overlap(model, tensors, tensors, torch.as_tensor([failed], dtype=torch.long))

    assert overlap.shape == tensors["action_mask"].shape
    assert float(overlap[0, failed].detach()) > 0.99
    assert torch.isfinite(logits[tensors["action_mask"]]).all()
    assert bool(tensors["action_mask"][0, failed])


def test_balanced_runtime_score_penalizes_diagnostic_mix_and_column_concentration() -> None:
    base = {
        "runtime_agent_act_path_active_success_within_5": 0.8,
        "runtime_agent_act_path_unique_action_count_mean": 3.0,
        "runtime_agent_act_path_selected_unique_mapped_col_count": 2.5,
        "runtime_agent_act_path_top_score_same_mapped_col_fraction": 0.55,
        "runtime_agent_act_path_runtime_diagnostic_mix_effective": 0.20,
    }
    worse = dict(base)
    worse["runtime_agent_act_path_top_score_same_mapped_col_fraction"] = 0.85
    worse["runtime_agent_act_path_runtime_diagnostic_mix_effective"] = 0.80

    assert _with_balanced_runtime_score(base)["runtime_agent_act_path_balanced_score"] > _with_balanced_runtime_score(worse)[
        "runtime_agent_act_path_balanced_score"
    ]


def test_failure_contingent_diversity_metrics_ignore_fast_success_levels() -> None:
    level = _parametric_level(seed=228)
    example = level.example
    failed_a, failed_b = _two_failed_clicks_same_x(example)
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()

    metrics = _failure_contingent_action_diversity_metrics(
        model=model,
        failed_level_records=[
            (
                example,
                (example.legal_actions[failed_a], example.legal_actions[failed_b], example.legal_actions[failed_a]),
                (0.8, 0.7, 0.6),
            )
        ],
        device=torch.device("cpu"),
        prefix="runtime_agent_act_path_",
    )

    assert metrics["runtime_agent_act_path_failed_level_count"] == 1.0
    assert metrics["runtime_agent_act_path_failed_level_unique_action_count_mean"] == 2.0
    assert metrics["runtime_agent_act_path_failed_level_max_same_action_streak_max"] == 1.0
    assert metrics["runtime_agent_act_path_failed_level_top_score_same_mapped_col_fraction"] == np.mean([0.8, 0.7, 0.6])


def test_post_noeffect_window_metrics_detect_same_column_repeats() -> None:
    level = _parametric_level(seed=229)
    example = level.example
    failed_a, failed_b = _two_failed_clicks_same_x(example)
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()

    metrics = _post_noeffect_window_metrics(
        model=model,
        records=[(example, example.legal_actions[failed_a], (example.legal_actions[failed_b],), (False,))],
        device=torch.device("cpu"),
        prefix="runtime_agent_act_path_",
    )

    assert metrics["runtime_agent_act_path_post_noeffect_window_count"] == 1.0
    assert metrics["runtime_agent_act_path_post_noeffect_next_action_same_x_rate"] == 1.0
    assert metrics["runtime_agent_act_path_post_noeffect_next_action_same_mapped_col_rate"] == 1.0
    assert metrics["runtime_agent_act_path_post_noeffect_rank_recovery_success_rate"] == 0.0


def test_post_noeffect_window_metrics_do_not_modify_actions_or_scores() -> None:
    level = _parametric_level(seed=230)
    example = level.example
    failed = _failed_click_index(example)
    positive = int(example.positive_action_indices[0])
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()
    before = {key: value.detach().clone() for key, value in model.state_dict().items()}

    metrics = _post_noeffect_window_metrics(
        model=model,
        records=[(example, example.legal_actions[failed], (example.legal_actions[positive],), (True,))],
        device=torch.device("cpu"),
        prefix="runtime_agent_act_path_",
    )

    assert metrics["runtime_agent_act_path_post_noeffect_rank_recovery_success_rate"] == 1.0
    for key, value in model.state_dict().items():
        assert torch.equal(value, before[key])


def test_balanced_runtime_score_can_use_failed_level_diversity_when_present() -> None:
    base = {
        "runtime_agent_act_path_active_success_within_5": 0.8,
        "runtime_agent_act_path_unique_action_count_mean": 1.0,
        "runtime_agent_act_path_selected_unique_mapped_col_count": 1.0,
        "runtime_agent_act_path_top_score_same_mapped_col_fraction": 0.90,
        "runtime_agent_act_path_failed_level_count": 3.0,
        "runtime_agent_act_path_failed_level_unique_action_count_mean": 3.0,
        "runtime_agent_act_path_failed_level_selected_unique_mapped_col_count": 2.5,
        "runtime_agent_act_path_failed_level_top_score_same_mapped_col_fraction": 0.50,
        "runtime_agent_act_path_runtime_diagnostic_mix_effective": 0.10,
    }
    without_failed = dict(base)
    without_failed["runtime_agent_act_path_failed_level_count"] = 0.0

    assert _with_balanced_runtime_score(base)["runtime_agent_act_path_balanced_score"] > _with_balanced_runtime_score(
        without_failed
    )["runtime_agent_act_path_balanced_score"]


def test_family_assignment_regularization_penalizes_single_family_collapse() -> None:
    level = _parametric_level(seed=223)
    tensors = _batch_to_tensors(collate_object_event_examples((level.example,)), device=torch.device("cpu"))
    action_count = tensors["action_mask"].shape[1]
    collapsed = torch.zeros((1, action_count, 4), dtype=torch.float32)
    collapsed[..., 0] = 1.0
    output = type("Output", (), {})()
    output.action_family_probs = collapsed

    losses = _family_assignment_regularization_losses(output, tensors)

    assert float(losses["balance_loss"]) > 0.1
    assert float(losses["effective_count"]) < 2.0


def test_family_assignment_uses_multiple_families_on_447_surface() -> None:
    level = _parametric_level(seed=224)
    tensors = _batch_to_tensors(collate_object_event_examples((level.example,)), device=torch.device("cpu"))
    model = ObjectEventModel(ObjectEventModelConfig(d_model=128, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()

    output = model(**tensors["inputs"])
    losses = _family_assignment_regularization_losses(output, tensors)

    assert output.action_family_probs is not None
    assert output.action_family_probs.shape == (1, 447, model.action_family_belief.num_families)
    assert model.action_family_belief.num_families >= 1
    if model.action_family_belief.num_families > 1:
        assert float(losses["effective_count"].detach()) > 1.0
    else:
        assert float(losses["effective_count"].detach()) == 1.0
    assert 0.0 <= float(losses["usage_min"].detach()) <= float(losses["usage_max"].detach()) <= 1.0
    assert torch.isfinite(output.action_family_probs[tensors["action_mask"]]).all()


def test_information_gain_target_prefers_hypothesis_disagreement_not_positive_only() -> None:
    success_by_hypothesis = torch.as_tensor([[[1.0, 1.0], [1.0, 0.0]]], dtype=torch.float32)

    target = _information_gain_from_hypothesis_success(success_by_hypothesis)

    assert target.shape == (1, 2)
    assert float(target[0, 1]) > float(target[0, 0])


def test_diagnostic_mix_is_low_without_evidence() -> None:
    level = _parametric_level(seed=220)
    tensors = _batch_to_tensors(collate_object_event_examples((level.example,)), device=torch.device("cpu"))
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()

    output = model(**tensors["inputs"])

    assert output.diagnostic_mix_logit is not None
    assert float(torch.sigmoid(output.diagnostic_mix_logit[0]).detach()) < 0.25


def test_diagnostic_utility_changes_after_online_noeffect_update() -> None:
    level = _parametric_level(seed=219)
    example = level.example
    failed = _failed_click_index(example)
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()
    with torch.no_grad():
        model.action_family_diagnostic_utility.utility[-1].weight.fill_(0.05)
        model.action_family_diagnostic_utility.utility[-1].bias.zero_()
    zero = torch.zeros((1, model.config.d_model), dtype=torch.float32)
    before = model(**tensors["inputs"], level_belief=zero)
    deltas = model.observed_event_belief_deltas(
        before,
        target_outcome=tensors["candidate_outcome_targets"][:, failed],
        target_delta=tensors["candidate_delta_targets"][:, failed],
        actual_action_index=torch.as_tensor([failed], dtype=torch.long),
        state_numeric=tensors["inputs"]["state_numeric"],
        state_type_ids=tensors["inputs"]["state_type_ids"],
        state_mask=tensors["inputs"]["state_mask"],
        action_numeric=tensors["inputs"]["action_numeric"],
    )
    after = model(**tensors["inputs"], level_belief=deltas.level_delta)

    assert before.diagnostic_utility_logits is not None
    assert after.diagnostic_utility_logits is not None
    assert torch.isfinite(after.diagnostic_utility_logits[tensors["action_mask"]]).all()
    assert torch.mean(torch.abs(after.diagnostic_utility_logits - before.diagnostic_utility_logits)) > 0.0


def test_diagnostic_mix_can_change_after_noeffect_evidence() -> None:
    level = _parametric_level(seed=221)
    example = level.example
    failed = _failed_click_index(example)
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    model = ObjectEventModel(ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0))
    model.eval()
    with torch.no_grad():
        model.diagnostic_mix_head[-1].weight.fill_(0.05)
    zero = torch.zeros((1, model.config.d_model), dtype=torch.float32)
    before = model(**tensors["inputs"], level_belief=zero)
    deltas = model.observed_event_belief_deltas(
        before,
        target_outcome=tensors["candidate_outcome_targets"][:, failed],
        target_delta=tensors["candidate_delta_targets"][:, failed],
        actual_action_index=torch.as_tensor([failed], dtype=torch.long),
        state_numeric=tensors["inputs"]["state_numeric"],
        state_type_ids=tensors["inputs"]["state_type_ids"],
        state_mask=tensors["inputs"]["state_mask"],
        action_numeric=tensors["inputs"]["action_numeric"],
    )
    after = model(**tensors["inputs"], level_belief=deltas.level_delta)

    assert before.diagnostic_mix_logit is not None
    assert after.diagnostic_mix_logit is not None
    assert float(torch.abs(after.diagnostic_mix_logit - before.diagnostic_mix_logit).detach()) > 0.0


def _toy_noeffect_contradiction_inputs(
    *,
    no_effect_prob: float,
    posterior_noeffect: float,
    known_gate: float = 0.8,
    known_penalty: float = 0.5,
):
    outcome_logits = torch.zeros((1, 3, OUTCOME_DIM), dtype=torch.float32)
    outcome_logits[..., OUT_NO_EFFECT_NONPROGRESS] = -4.0
    outcome_logits[0, 1, OUT_NO_EFFECT_NONPROGRESS] = torch.logit(
        torch.tensor(float(no_effect_prob), dtype=torch.float32).clamp(1.0e-4, 1.0 - 1.0e-4)
    )
    basis_features = torch.zeros((1, 3, 4), dtype=torch.float32)
    family_features = torch.zeros((1, 3, 4), dtype=torch.float32)
    basis_features[0, 1, 3] = float(posterior_noeffect)
    gate = torch.full((1, 3), 0.1, dtype=torch.float32)
    gate[0, 1] = float(known_gate)
    penalty = torch.zeros((1, 3), dtype=torch.float32)
    penalty[0, 1] = float(known_penalty)
    output = SimpleNamespace(
        outcome_logits=outcome_logits,
        action_basis_posterior_features=basis_features,
        action_family_posterior_features=family_features,
        noeffect_contradiction_gate=gate,
        noeffect_contradiction_penalty=penalty,
    )
    candidate_outcome_targets = torch.zeros((1, 3, OUTCOME_DIM), dtype=torch.float32)
    candidate_outcome_targets[0, 1, OUT_NO_EFFECT_NONPROGRESS] = 1.0
    tensors = {
        "action_mask": torch.ones((1, 3), dtype=torch.bool),
        "candidate_value_targets": torch.tensor([[1.0, 0.0, 0.0]], dtype=torch.float32),
        "candidate_outcome_targets": candidate_outcome_targets,
    }
    positive_mask = tensors["candidate_value_targets"] > 0.5
    relation_failed_mask = torch.zeros((1, 3), dtype=torch.bool)
    coord_failure_mask = torch.ones((1,), dtype=torch.float32)
    return output, tensors, positive_mask, relation_failed_mask, coord_failure_mask


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


def _failed_click_index(example) -> int:
    return next(
        index
        for index, value in enumerate(example.candidate_targets.value)
        if float(value) == 0.0 and example.legal_actions[index].startswith("click:")
    )


def _failed_object_click_index(example, tensors: dict[str, object]) -> int:
    action_numeric = tensors["inputs"]["action_numeric"][0]
    for index, value in enumerate(example.candidate_targets.value):
        is_object = float(action_numeric[index, 11]) > 0.5 and float(action_numeric[index, 24]) < 0.5
        if float(value) == 0.0 and example.legal_actions[index].startswith("click:") and is_object:
            return int(index)
    raise AssertionError("parametric test level has no failed object click")


def _two_failed_clicks_same_x(example) -> tuple[int, int]:
    by_x: dict[str, list[int]] = {}
    for index, value in enumerate(example.candidate_targets.value):
        action = example.legal_actions[index]
        if float(value) != 0.0 or not action.startswith("click:"):
            continue
        x = action.split(":")[1]
        by_x.setdefault(x, []).append(index)
    pair = next(indices for indices in by_x.values() if len(indices) >= 2)
    return int(pair[0]), int(pair[1])


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
