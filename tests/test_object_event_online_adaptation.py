from __future__ import annotations

import numpy as np
import torch

from arcagi.learned_online.event_tokens import OUT_OBJECTIVE_PROGRESS, OUT_REWARD_PROGRESS
from arcagi.learned_online.object_event_curriculum import (
    OnlineObjectEventCurriculumConfig,
    build_online_object_event_curriculum,
    collate_object_event_examples,
    make_latent_rule_color_click_example,
)
from arcagi.learned_online.object_event_model import (
    ObjectEventModel,
    ObjectEventModelConfig,
    policy_rank_logits_from_predictions,
)
from scripts.train_learned_online_object_event import (
    _batch_to_tensors,
    _forward_tensors,
    _observed_belief_delta,
    _with_observed_candidate_targets,
    _wrong_rule_support_examples,
)


def test_latent_rule_is_metadata_not_token_input() -> None:
    config = OnlineObjectEventCurriculumConfig(seed=10, train_sessions=1, heldout_sessions=0)
    first = make_latent_rule_color_click_example(
        config=config,
        geometry_seed=1234,
        cue_mode=0,
        latent_rule=0,
        split="diagnostic",
    )
    second = make_latent_rule_color_click_example(
        config=config,
        geometry_seed=1234,
        cue_mode=0,
        latent_rule=1,
        split="diagnostic",
    )

    assert first.metadata["latent_rule"] == 0
    assert second.metadata["latent_rule"] == 1
    assert np.allclose(first.state_tokens.numeric, second.state_tokens.numeric)
    assert np.array_equal(first.state_tokens.type_ids, second.state_tokens.type_ids)
    assert np.array_equal(first.state_tokens.mask, second.state_tokens.mask)
    assert np.allclose(first.action_tokens.numeric, second.action_tokens.numeric)
    assert np.array_equal(first.action_tokens.action_type_ids, second.action_tokens.action_type_ids)


def test_latent_rule_flips_candidate_labels_without_changing_action_rows() -> None:
    config = OnlineObjectEventCurriculumConfig(seed=11, train_sessions=1, heldout_sessions=0)
    normal = make_latent_rule_color_click_example(
        config=config,
        geometry_seed=4321,
        cue_mode=0,
        latent_rule=0,
        split="diagnostic",
    )
    inverted = make_latent_rule_color_click_example(
        config=config,
        geometry_seed=4321,
        cue_mode=0,
        latent_rule=1,
        split="diagnostic",
    )
    red_index = int(normal.metadata["red_action_index"])
    blue_index = int(normal.metadata["blue_action_index"])

    assert normal.legal_actions == inverted.legal_actions
    assert np.allclose(normal.action_tokens.numeric, inverted.action_tokens.numeric)
    assert normal.correct_action_index == red_index
    assert inverted.correct_action_index == blue_index
    assert normal.candidate_targets.value[red_index] == 1.0
    assert normal.candidate_targets.value[blue_index] == 0.0
    assert inverted.candidate_targets.value[red_index] == 0.0
    assert inverted.candidate_targets.value[blue_index] == 1.0


def test_online_sessions_preserve_dense_legal_surface() -> None:
    config = OnlineObjectEventCurriculumConfig(seed=12, train_sessions=3, heldout_sessions=2, levels_per_session=3)
    split = build_online_object_event_curriculum(config)
    expected_actions = config.grid_size * config.grid_size + 4

    for session in split.train + split.heldout:
        for level in session.levels:
            example = level.example
            assert len(example.legal_actions) == expected_actions
            assert example.action_tokens.numeric.shape[0] == expected_actions
            assert int(example.action_tokens.mask.sum()) == expected_actions
            assert example.metadata["legal_action_count"] == expected_actions


def test_online_session_levels_change_geometry_but_keep_rule() -> None:
    config = OnlineObjectEventCurriculumConfig(seed=13, train_sessions=4, heldout_sessions=0, levels_per_session=3)
    sessions = build_online_object_event_curriculum(config).train

    for session in sessions:
        geometry_seeds = {int(level.geometry_seed) for level in session.levels}
        cue_modes = {int(level.cue_mode) for level in session.levels}
        latent_rules = {int(level.latent_rule) for level in session.levels}
        grid_signatures = {level.example.state.grid_signature for level in session.levels}
        assert latent_rules == {int(session.latent_rule)}
        assert cue_modes == {0, 1}
        assert len(geometry_seeds) == len(session.levels)
        assert len(grid_signatures) == len(session.levels)


def test_online_candidate_targets_cover_full_surface() -> None:
    config = OnlineObjectEventCurriculumConfig(seed=14, train_sessions=2, heldout_sessions=0, levels_per_session=2)
    sessions = build_online_object_event_curriculum(config).train
    examples = tuple(level.example for session in sessions for level in session.levels)
    batch = collate_object_event_examples(examples)
    action_count = config.grid_size * config.grid_size + 4

    assert batch.candidate_outcome_targets.shape[:2] == (len(examples), action_count)
    assert batch.candidate_value_targets.shape == (len(examples), action_count)
    assert int(batch.action_mask.sum()) == len(examples) * action_count
    for row, example in enumerate(examples):
        assert batch.candidate_value_targets[row, example.correct_action_index] == 1.0
        assert batch.candidate_outcome_targets[row, example.correct_action_index, OUT_OBJECTIVE_PROGRESS] == 1.0
        assert batch.candidate_outcome_targets[row, example.correct_action_index, OUT_REWARD_PROGRESS] == 1.0
        assert int(batch.actual_action_index[row]) == example.correct_action_index


def test_positive_action_row_identifiable() -> None:
    config = OnlineObjectEventCurriculumConfig(seed=15, train_sessions=2, heldout_sessions=0, levels_per_session=3)
    sessions = build_online_object_event_curriculum(config).train

    for session in sessions:
        for level in session.levels:
            example = level.example
            positive_row = example.action_tokens.numeric[example.correct_action_index]
            indistinguishable = np.all(np.isclose(example.action_tokens.numeric, positive_row[None, :]), axis=1)
            assert np.flatnonzero(indistinguishable).tolist() == [example.correct_action_index]


def test_object_event_relation_memory_improves_heldout_support_query() -> None:
    torch.manual_seed(3)
    split = build_online_object_event_curriculum(
        OnlineObjectEventCurriculumConfig(
            seed=21,
            train_sessions=0,
            heldout_sessions=8,
            levels_per_session=3,
            max_objects=3,
            include_distractors=False,
        )
    )
    sessions = split.heldout
    support = tuple(session.levels[0].example for session in sessions)
    query = tuple(session.levels[1].example for session in sessions)
    model = ObjectEventModel(
        ObjectEventModelConfig(
            d_model=64,
            n_heads=4,
            state_layers=1,
            action_cross_layers=1,
            dropout=0.0,
            online_rank=4,
        )
    )
    support_tensors = _batch_to_tensors(collate_object_event_examples(support), device=torch.device("cpu"))
    query_tensors = _batch_to_tensors(collate_object_event_examples(query), device=torch.device("cpu"))
    wrong_support = _batch_to_tensors(
        collate_object_event_examples(_wrong_rule_support_examples(sessions)),
        device=torch.device("cpu"),
    )
    wrong_indices = torch.as_tensor(
        [
            int(example.metadata["blue_action_index"])
            if int(example.correct_action_index) == int(example.metadata["red_action_index"])
            else int(example.metadata["red_action_index"])
            for example in support
        ],
        dtype=torch.long,
    )
    failure_support = _with_observed_candidate_targets(support_tensors, wrong_indices)

    with torch.no_grad():
        pre_logits = policy_rank_logits_from_predictions(_forward_tensors(model, query_tensors), query_tensors["action_mask"])
        support_belief = _observed_belief_delta(model, support_tensors)
        post_logits = policy_rank_logits_from_predictions(
            _forward_tensors(model, query_tensors, session_belief=support_belief),
            query_tensors["action_mask"],
        )
        wrong_belief = _observed_belief_delta(model, wrong_support)
        wrong_logits = policy_rank_logits_from_predictions(
            _forward_tensors(model, query_tensors, session_belief=wrong_belief),
            query_tensors["action_mask"],
        )
        failure_belief = _observed_belief_delta(model, failure_support)
        failure_logits = policy_rank_logits_from_predictions(
            _forward_tensors(model, query_tensors, session_belief=failure_belief),
            query_tensors["action_mask"],
        )

    actual = query_tensors["actual_action_index"]
    pre = float((pre_logits.argmax(dim=-1) == actual).float().mean())
    post = float((post_logits.argmax(dim=-1) == actual).float().mean())
    wrong = float((wrong_logits.argmax(dim=-1) == actual).float().mean())
    failure = float((failure_logits.argmax(dim=-1) == actual).float().mean())

    assert {len(example.legal_actions) for example in query} == {68}
    assert pre <= 0.75
    assert post >= pre + 0.25
    assert post >= 0.875
    assert wrong <= post - 0.25
    assert failure >= 0.875
