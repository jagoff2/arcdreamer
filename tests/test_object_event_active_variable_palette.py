from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

from arcagi.learned_online.object_event_curriculum import (
    ActiveOnlineObjectEventConfig,
    OnlineObjectEventCurriculumConfig,
    apply_synthetic_object_event_action,
    build_active_online_object_event_curriculum,
    collate_object_event_examples,
    make_latent_rule_variable_palette_example,
)
from arcagi.learned_online.object_event_model import ObjectEventModel, ObjectEventModelConfig
from scripts.train_learned_online_object_event import (
    _active_rollout_metrics,
    _batch_to_tensors,
    _online_selection_metric,
    _runtime_agent_active_metrics,
    _with_observed_candidate_targets,
)


def test_variable_palette_targets_and_distractors_share_color_distribution() -> None:
    split = build_active_online_object_event_curriculum(_config(seed=110, train_sessions=24, heldout_sessions=24))
    target_counts, distractor_counts = _role_counts(split.train + split.heldout)
    palette = set(next(iter(split.train)).levels[0].example.metadata["palette_colors"])

    assert palette.issubset(set(target_counts))
    assert palette.issubset(set(distractor_counts))
    assert min(target_counts[color] for color in palette) >= 1
    assert min(distractor_counts[color] for color in palette) >= 1


def test_variable_palette_no_target_or_distractor_role_token_leak() -> None:
    level = build_active_online_object_event_curriculum(_config(seed=111, train_sessions=1, heldout_sessions=0)).train[0].levels[0]
    ordinary = [obj for obj in level.example.state.objects if "agent" not in obj.tags]

    assert ordinary
    assert all(obj.object_id.startswith("object_") for obj in ordinary)
    assert all(not any(token in obj.object_id for token in ("red", "blue", "target", "distractor")) for obj in ordinary)
    assert all(obj.tags == () for obj in ordinary)
    assert "distractor_action_index" not in level.example.metadata


def test_variable_palette_latent_rule_still_metadata_only() -> None:
    config = OnlineObjectEventCurriculumConfig(
        seed=112,
        train_sessions=1,
        heldout_sessions=0,
        levels_per_session=2,
        max_objects=4,
        include_distractors=True,
        curriculum="latent_rule_variable_palette",
        palette_size=8,
    )
    first = make_latent_rule_variable_palette_example(
        config=config,
        geometry_seed=12345,
        cue_mode=0,
        latent_rule=0,
        split="train",
        session_index=0,
        level_index=0,
    )
    second = make_latent_rule_variable_palette_example(
        config=config,
        geometry_seed=12345,
        cue_mode=0,
        latent_rule=1,
        split="train",
        session_index=0,
        level_index=0,
    )

    assert first.metadata["latent_rule"] == 0
    assert second.metadata["latent_rule"] == 1
    assert np.allclose(first.state_tokens.numeric, second.state_tokens.numeric)
    assert np.array_equal(first.state_tokens.type_ids, second.state_tokens.type_ids)
    assert np.allclose(first.action_tokens.numeric, second.action_tokens.numeric)
    assert first.correct_action_index != second.correct_action_index


def test_variable_palette_dense_legal_surface_is_68() -> None:
    split = build_active_online_object_event_curriculum(_config(seed=113, train_sessions=2, heldout_sessions=2))

    for session in split.train + split.heldout:
        for level in session.levels:
            example = level.example
            assert len(example.legal_actions) == 68
            assert int(example.action_tokens.mask.sum()) == 68
            assert example.metadata["legal_action_count"] == 68


def test_variable_palette_selected_distractor_failure_is_observed_only() -> None:
    level = build_active_online_object_event_curriculum(_config(seed=114, train_sessions=1, heldout_sessions=0)).train[0].levels[0]
    example = level.example
    distractor_index = _distractor_action_index(example)
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    observed = _with_observed_candidate_targets(tensors, torch.as_tensor([distractor_index], dtype=torch.long))
    result = apply_synthetic_object_event_action(level, distractor_index)

    assert result.success is False
    assert result.no_effect is True
    assert int(observed["actual_action_index"][0]) == distractor_index
    assert int(tensors["actual_action_index"][0]) == int(example.correct_action_index)


def test_variable_palette_level0_can_include_distractor_first_actions_under_policy_eval() -> None:
    torch.manual_seed(115)
    split = build_active_online_object_event_curriculum(_config(seed=115, train_sessions=0, heldout_sessions=24))
    model = ObjectEventModel(
        ObjectEventModelConfig(d_model=64, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0, online_rank=4)
    )
    metrics = _active_rollout_metrics(model, split.heldout, device=torch.device("cpu"))

    assert metrics["object_event_bridge_legal_action_count_mean"] == 68.0
    assert metrics["object_event_bridge_selected_action_match_rate"] == 1.0
    assert metrics["metadata_model_input_forbidden_count"] == 0.0
    assert metrics["level0_distractor_first_click_rate"] > 0.0 or metrics["level0_target_first_click_rate"] < 1.0


def test_runtime_agent_metric_exercises_true_act_update_path() -> None:
    torch.manual_seed(116)
    split = build_active_online_object_event_curriculum(_config(seed=116, train_sessions=0, heldout_sessions=1))
    model = ObjectEventModel(
        ObjectEventModelConfig(d_model=32, n_heads=4, state_layers=1, action_cross_layers=1, dropout=0.0, online_rank=4)
    )
    metrics = _runtime_agent_active_metrics(model, split.heldout, device=torch.device("cpu"), max_steps_per_level=1)

    assert metrics["runtime_agent_rank_logits_used"] == 1.0
    assert metrics["runtime_agent_act_path_rank_logits_used"] == 1.0
    assert metrics["runtime_agent_act_path_num_legal_actions_mean"] == 68.0
    assert metrics["runtime_agent_act_path_bridge_selected_action_match_rate"] == 1.0
    assert metrics["runtime_agent_act_path_metadata_model_input_forbidden_count"] == 0.0
    assert metrics["runtime_agent_act_path_online_update_count_mean"] >= 1.0


def test_online_selection_metric_defaults_follow_active_mode() -> None:
    assert _online_selection_metric(SimpleNamespace(selection_metric=None), active=True) == "heldout_active_post_self_update_top1_acc"
    assert _online_selection_metric(SimpleNamespace(selection_metric=None), active=False) == "heldout_post_update_top1_acc"
    assert _online_selection_metric(SimpleNamespace(selection_metric="runtime_agent_rank_logits_used"), active=True) == "runtime_agent_rank_logits_used"


def _config(*, seed: int, train_sessions: int, heldout_sessions: int) -> ActiveOnlineObjectEventConfig:
    return ActiveOnlineObjectEventConfig(
        seed=seed,
        train_sessions=train_sessions,
        heldout_sessions=heldout_sessions,
        levels_per_session=3,
        max_distractors=1,
        curriculum="latent_rule_variable_palette",
        palette_size=8,
        require_role_balanced_colors=True,
    )


def _role_counts(sessions) -> tuple[dict[int, int], dict[int, int]]:
    target: dict[int, int] = {}
    distractor: dict[int, int] = {}
    for session in sessions:
        for level in session.levels:
            target_color = int(level.example.metadata["target_color"])
            target[target_color] = target.get(target_color, 0) + 1
            for color in level.example.metadata["distractor_colors"]:
                color = int(color)
                distractor[color] = distractor.get(color, 0) + 1
    return target, distractor


def _distractor_action_index(example) -> int:
    candidate_indices = set(int(index) for index in example.metadata["candidate_action_indices"])
    for index, action_row in enumerate(example.action_tokens.numeric):
        if action_row[11] > 0.5 and action_row[24] < 0.5 and index not in candidate_indices:
            return index
    raise AssertionError("example has no ordinary distractor action")
