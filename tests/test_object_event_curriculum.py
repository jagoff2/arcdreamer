from __future__ import annotations

import inspect

import numpy as np

import arcagi.learned_online.object_event_curriculum as curriculum_module
from arcagi.learned_online.object_event_curriculum import (
    ObjectEventCurriculumConfig,
    build_paired_color_click_curriculum,
    collate_object_event_examples,
)


def test_curriculum_preserves_dense_legal_surface() -> None:
    config = ObjectEventCurriculumConfig(seed=4, train_geometries=2, heldout_geometries=1, grid_size=8)
    split = build_paired_color_click_curriculum(config)
    expected_actions = config.grid_size * config.grid_size + 4

    for example in split.train + split.heldout:
        assert len(example.legal_actions) == expected_actions
        assert example.action_tokens.numeric.shape[0] == expected_actions
        assert int(example.action_tokens.mask.sum()) == expected_actions
        assert example.metadata["legal_action_count"] == expected_actions


def test_paired_cue_flip_changes_candidate_labels_not_action_rows() -> None:
    split = build_paired_color_click_curriculum(ObjectEventCurriculumConfig(seed=5, train_geometries=1, heldout_geometries=0))
    first, second = split.train

    assert first.metadata["geometry_seed"] == second.metadata["geometry_seed"]
    assert first.legal_actions == second.legal_actions
    assert not np.allclose(first.state_tokens.numeric, second.state_tokens.numeric)
    red_index = int(first.metadata["red_action_index"])
    blue_index = int(first.metadata["blue_action_index"])
    assert np.allclose(first.action_tokens.numeric[red_index], second.action_tokens.numeric[red_index])
    assert np.allclose(first.action_tokens.numeric[blue_index], second.action_tokens.numeric[blue_index])
    assert first.correct_action_index == red_index
    assert second.correct_action_index == blue_index
    assert first.candidate_targets.value[red_index] == 1.0
    assert first.candidate_targets.value[blue_index] == 0.0
    assert second.candidate_targets.value[red_index] == 0.0
    assert second.candidate_targets.value[blue_index] == 1.0
    assert np.allclose(first.transition_targets.outcome, second.transition_targets.outcome)
    assert np.allclose(first.transition_targets.delta, second.transition_targets.delta)


def test_train_heldout_geometry_splits_are_disjoint() -> None:
    split = build_paired_color_click_curriculum(
        ObjectEventCurriculumConfig(seed=6, train_geometries=5, heldout_geometries=4)
    )
    train_seeds = {int(example.metadata["geometry_seed"]) for example in split.train}
    heldout_seeds = {int(example.metadata["geometry_seed"]) for example in split.heldout}

    assert len(train_seeds) == 5
    assert len(heldout_seeds) == 4
    assert train_seeds.isdisjoint(heldout_seeds)


def test_curriculum_does_not_project_ids_hashes_or_action_strings() -> None:
    source = inspect.getsource(curriculum_module)
    forbidden = (
        "fingerprint(",
        "hash(",
        "state_hash",
        "teacher_action",
        "trace_cursor",
        "action_sequence",
        "stable_symbol_projection",
    )
    for fragment in forbidden:
        assert fragment not in source

    config = ObjectEventCurriculumConfig(seed=7, train_geometries=1, heldout_geometries=0)
    first = build_paired_color_click_curriculum(config).train[0]
    altered_objects = tuple(
        type(obj)(
            object_id=f"renamed_{index}",
            color=obj.color,
            cells=obj.cells,
            bbox=obj.bbox,
            centroid=obj.centroid,
            area=obj.area,
            tags=obj.tags,
        )
        for index, obj in enumerate(first.state.objects)
    )
    altered_state = type(first.state)(
        task_id="different_task_id",
        episode_id="different_episode_id",
        step_index=first.state.step_index,
        grid_shape=first.state.grid_shape,
        grid_signature=first.state.grid_signature,
        objects=altered_objects,
        relations=first.state.relations,
        affordances=first.state.affordances,
        action_roles=first.state.action_roles,
        inventory=first.state.inventory,
        flags=first.state.flags,
    )
    from arcagi.learned_online.event_tokens import encode_state_tokens

    assert np.allclose(first.state_tokens.numeric, encode_state_tokens(altered_state).numeric)


def test_candidate_targets_include_full_action_surface() -> None:
    split = build_paired_color_click_curriculum(ObjectEventCurriculumConfig(seed=8, train_geometries=2, heldout_geometries=0))
    batch = collate_object_event_examples(split.train)
    action_count = split.train[0].metadata["legal_action_count"]

    assert batch.candidate_outcome_targets.shape[:2] == (len(split.train), action_count)
    assert batch.candidate_value_targets.shape == (len(split.train), action_count)
    assert batch.action_mask.shape == (len(split.train), action_count)
    assert int(batch.action_mask.sum()) == len(split.train) * action_count
    for row, example in enumerate(split.train):
        assert batch.candidate_value_targets[row, example.correct_action_index] == 1.0
        assert batch.candidate_outcome_targets[row, example.correct_action_index, 1] == 1.0
        assert int(batch.actual_action_index[row]) == example.correct_action_index
        positive_row = example.action_tokens.numeric[example.correct_action_index]
        indistinguishable = np.all(np.isclose(example.action_tokens.numeric, positive_row[None, :]), axis=1)
        assert np.flatnonzero(indistinguishable).tolist() == [example.correct_action_index]
