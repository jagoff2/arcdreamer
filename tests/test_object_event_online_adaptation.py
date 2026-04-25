from __future__ import annotations

import numpy as np
import torch

from arcagi.learned_online.event_tokens import OUT_OBJECTIVE_PROGRESS, OUT_REWARD_PROGRESS
from arcagi.learned_online.object_event_curriculum import (
    ActiveOnlineObjectEventConfig,
    OnlineObjectEventCurriculumConfig,
    apply_synthetic_object_event_action,
    build_active_online_object_event_curriculum,
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


def test_active_rollout_preserves_dense_legal_surface() -> None:
    split = build_active_online_object_event_curriculum(
        ActiveOnlineObjectEventConfig(seed=50, train_sessions=2, heldout_sessions=2, levels_per_session=2, max_distractors=1)
    )
    expected_actions = 8 * 8 + 4

    for session in split.train + split.heldout:
        for level in session.levels:
            example = level.example
            assert len(example.legal_actions) == expected_actions
            assert int(example.action_tokens.mask.sum()) == expected_actions
            assert example.metadata["legal_action_count"] == expected_actions
            assert any(obj.object_id.startswith("distractor_") for obj in example.state.objects)


def test_active_rollout_uses_selected_action_not_oracle_support() -> None:
    level = build_active_online_object_event_curriculum(
        ActiveOnlineObjectEventConfig(seed=51, train_sessions=1, heldout_sessions=0, levels_per_session=2, max_distractors=1)
    ).train[0].levels[0]
    wrong_index = next(
        index
        for index in range(len(level.example.legal_actions))
        if index != int(level.example.correct_action_index)
    )

    wrong = apply_synthetic_object_event_action(level, wrong_index)
    correct = apply_synthetic_object_event_action(level, int(level.example.correct_action_index))

    assert wrong.selected_action_index == wrong_index
    assert wrong.transition_targets.actual_action_index == wrong_index
    assert wrong.success is False
    assert wrong.no_effect is True
    assert wrong.reward == 0.0
    assert wrong.metadata["oracle_support_used"] is False
    assert correct.success is True
    assert correct.transition_targets.actual_action_index == int(level.example.correct_action_index)


def test_distractor_role_is_not_token_leaked() -> None:
    level = build_active_online_object_event_curriculum(
        ActiveOnlineObjectEventConfig(seed=52, train_sessions=1, heldout_sessions=0, levels_per_session=2, max_distractors=1)
    ).train[0].levels[0]
    example = level.example
    distractor = next(obj for obj in example.state.objects if obj.object_id.startswith("distractor_"))
    row, col = distractor.cells[0]
    action_index = example.legal_actions.index(f"click:{col}:{row}")
    action_row = example.action_tokens.numeric[action_index]

    assert "distractor_action_index" not in example.metadata
    assert action_row[11] == 1.0
    assert np.isclose(action_row[13], float(distractor.color % 12) / 11.0)
    assert np.all(action_row[24:28] == 0.0)


def test_failed_distractor_transition_is_no_effect_and_observed_only() -> None:
    level = build_active_online_object_event_curriculum(
        ActiveOnlineObjectEventConfig(seed=53, train_sessions=1, heldout_sessions=0, levels_per_session=2, max_distractors=1)
    ).train[0].levels[0]
    example = level.example
    distractor = next(obj for obj in example.state.objects if obj.object_id.startswith("distractor_"))
    row, col = distractor.cells[0]
    distractor_index = example.legal_actions.index(f"click:{col}:{row}")
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    observed = _with_observed_candidate_targets(tensors, torch.as_tensor([distractor_index], dtype=torch.long))
    result = apply_synthetic_object_event_action(level, distractor_index)

    assert result.success is False
    assert result.no_effect is True
    assert int(observed["actual_action_index"][0]) == distractor_index
    assert float(observed["target_outcome"][0, OUT_OBJECTIVE_PROGRESS]) == 0.0
    assert float(observed["target_outcome"][0, OUT_REWARD_PROGRESS]) == 0.0
    assert int(tensors["actual_action_index"][0]) == int(example.correct_action_index)


def test_level_negative_memory_resets_but_session_relation_memory_persists() -> None:
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
    session = build_active_online_object_event_curriculum(
        ActiveOnlineObjectEventConfig(seed=54, train_sessions=1, heldout_sessions=0, levels_per_session=2, max_distractors=1)
    ).train[0]
    support = session.levels[0].example
    query = session.levels[1].example
    wrong_index = (
        int(support.metadata["blue_action_index"])
        if int(support.correct_action_index) == int(support.metadata["red_action_index"])
        else int(support.metadata["red_action_index"])
    )
    support_tensors = _batch_to_tensors(collate_object_event_examples((support,)), device=torch.device("cpu"))
    observed = _with_observed_candidate_targets(support_tensors, torch.as_tensor([wrong_index], dtype=torch.long))
    query_tensors = _batch_to_tensors(collate_object_event_examples((query,)), device=torch.device("cpu"))

    with torch.no_grad():
        deltas = model.observed_event_belief_deltas(
            _forward_tensors(model, observed),
            target_outcome=observed["target_outcome"],
            target_delta=observed["target_delta"],
            actual_action_index=observed["actual_action_index"],
            state_numeric=observed["inputs"]["state_numeric"],
            state_type_ids=observed["inputs"]["state_type_ids"],
            state_mask=observed["inputs"]["state_mask"],
            action_numeric=observed["inputs"]["action_numeric"],
        )
        with_session = policy_rank_logits_from_predictions(
            _forward_tensors(model, query_tensors, session_belief=deltas.session_delta),
            query_tensors["action_mask"],
        )
        with_reset_level = policy_rank_logits_from_predictions(
            _forward_tensors(
                model,
                query_tensors,
                session_belief=deltas.session_delta,
                level_belief=torch.zeros_like(deltas.level_delta),
            ),
            query_tensors["action_mask"],
        )

    assert int(torch.argmax(with_session[0]).detach().cpu()) == int(query.correct_action_index)
    assert int(torch.argmax(with_reset_level[0]).detach().cpu()) == int(query.correct_action_index)
    assert float(torch.linalg.vector_norm(deltas.session_delta)) > 0.0
    assert float(torch.linalg.vector_norm(deltas.level_delta)) > 0.0
