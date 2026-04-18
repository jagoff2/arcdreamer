from __future__ import annotations

from collections import deque
import json

import torch

from arcagi.envs.synthetic import EMPTY, HiddenRuleEnv, SWITCH_RED
from arcagi.perception.object_encoder import extract_structured_state
from arcagi.training.synthetic import (
    SyntheticTrainingConfig,
    _bootstrap_schedule,
    _curriculum_stages,
    _dream_sequence_windows,
    _holdout_passed,
    _make_sample,
    _run_dream_phase,
    _sample_dream_sequences,
    _stage_family_weights,
    _teacher_episode_fraction,
    build_default_modules,
    collect_dataset,
)


def _episode_initial_signatures(dataset: list[dict[str, object]]) -> list[tuple[str, str]]:
    return [
        (sample["state"].episode_id, sample["state"].fingerprint())
        for sample in dataset
        if sample["state"].step_index == 0
    ]


def _bfs_path(grid, start: tuple[int, int], goal: tuple[int, int]) -> list[str]:
    queue = deque([(start, [])])
    visited = {start}
    deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    while queue:
        (y, x), path = queue.popleft()
        if (y, x) == goal:
            return path
        for action, (dy, dx) in deltas.items():
            ny, nx = y + dy, x + dx
            if (ny, nx) in visited:
                continue
            if int(grid[ny, nx]) != EMPTY and (ny, nx) != goal:
                continue
            visited.add((ny, nx))
            queue.append(((ny, nx), path + [action]))
    raise AssertionError("no path found")


def _reachable_interaction(grid, start: tuple[int, int], target: tuple[int, int]) -> tuple[list[str], str]:
    candidates = [
        ((target[0] - 1, target[1]), "interact_down"),
        ((target[0] + 1, target[1]), "interact_up"),
        ((target[0], target[1] - 1), "interact_right"),
        ((target[0], target[1] + 1), "interact_left"),
    ]
    for position, action in candidates:
        if int(grid[position]) != EMPTY:
            continue
        try:
            return _bfs_path(grid, start, position), action
        except AssertionError:
            continue
    raise AssertionError("no reachable interaction side")


def test_collect_dataset_uses_fresh_epoch_seed_within_same_curriculum_stage() -> None:
    config = SyntheticTrainingConfig(
        epochs=64,
        episodes_per_epoch=6,
        seed=11,
        log_every_episodes=0,
    )

    dataset_epoch_7, metrics_epoch_7 = collect_dataset(config, epoch_index=7)
    dataset_epoch_8, metrics_epoch_8 = collect_dataset(config, epoch_index=8)

    assert metrics_epoch_7["epoch_seed_base"] != metrics_epoch_8["epoch_seed_base"]
    assert _episode_initial_signatures(dataset_epoch_7) != _episode_initial_signatures(dataset_epoch_8)


def test_collect_dataset_logs_interval_family_counts_without_aliasing(capsys) -> None:
    config = SyntheticTrainingConfig(
        epochs=64,
        episodes_per_epoch=4,
        seed=11,
        log_every_episodes=2,
    )

    _dataset, _metrics = collect_dataset(config, epoch_index=0)
    log_lines = [json.loads(line) for line in capsys.readouterr().out.splitlines() if line.strip()]

    progress_lines = [line for line in log_lines if line["type"] == "episode_progress"]
    assert progress_lines
    assert any(
        {"switch_unlock", "order_collect"} <= set(line["interval_family_counts"].keys())
        for line in progress_lines
    )


def test_collect_dataset_balances_two_way_family_variants_within_epoch() -> None:
    config = SyntheticTrainingConfig(
        epochs=64,
        episodes_per_epoch=48,
        seed=11,
        log_every_episodes=0,
    )

    _dataset, metrics = collect_dataset(config, epoch_index=0, stage_index=0)
    observed_variants = {
        key
        for key in metrics["variant_breakdown"].keys()
        if key.startswith("order_collect/")
    }

    assert observed_variants == {
        "order_collect/red_then_blue",
        "order_collect/blue_then_red",
    }


def test_bootstrap_schedule_tracks_current_readiness_instead_of_latching_release() -> None:
    config = SyntheticTrainingConfig(
        oracle_imitation_epochs=2,
        oracle_bootstrap_steps=16,
        oracle_bootstrap_stride=1,
        oracle_bootstrap_min_steps=8,
        oracle_bootstrap_min_stride=2,
        oracle_bootstrap_full_epochs=4,
        oracle_bootstrap_decay_epochs=8,
        oracle_bootstrap_decay_success_threshold=0.9,
        oracle_bootstrap_decay_stability_epochs=2,
        teacher_guidance_holdout_success_threshold=0.2,
    )

    assert _bootstrap_schedule(config, 0) == (16, 1)
    assert _bootstrap_schedule(config, 1) == (16, 1)
    assert _bootstrap_schedule(config, 2) == (16, 1)

    stable_history = [
        {"collect_success_rate": 1.0, "holdout_evaluated": True, "holdout_frontier_success": 0.3},
        {"collect_success_rate": 0.97},
    ]
    assert _bootstrap_schedule(config, 2, stable_history) == (16, 1)
    long_stable_history = stable_history + [{"collect_success_rate": 0.96} for _ in range(6)]
    assert _bootstrap_schedule(config, 8, long_stable_history) == (13, 1)
    regressed_history = long_stable_history + [{"collect_success_rate": 0.2}]
    assert _bootstrap_schedule(config, 9, regressed_history) == (16, 1)


def test_teacher_episode_fraction_keeps_a_permanent_floor() -> None:
    config = SyntheticTrainingConfig(
        teacher_episode_fraction_initial=0.4,
        teacher_episode_fraction_floor=0.2,
        oracle_bootstrap_full_epochs=4,
        oracle_bootstrap_decay_epochs=8,
        oracle_bootstrap_decay_success_threshold=0.9,
        oracle_bootstrap_decay_stability_epochs=2,
    )
    assert _teacher_episode_fraction(config, []) == 0.4
    stable_history = [{"collect_success_rate": 0.95} for _ in range(16)]
    scheduled = _teacher_episode_fraction(config, stable_history)
    assert 0.2 <= scheduled <= 0.4
    assert round(scheduled, 4) == 0.2


def test_collect_dataset_reports_teacher_guidance_state_from_history() -> None:
    config = SyntheticTrainingConfig(
        episodes_per_epoch=8,
        log_every_episodes=0,
        oracle_bootstrap_decay_success_threshold=0.9,
        oracle_bootstrap_decay_stability_epochs=2,
        teacher_episode_fraction_initial=0.4,
        teacher_episode_fraction_floor=0.2,
    )
    history = [
        {"collect_success_rate": 1.0},
        {"collect_success_rate": 0.95},
    ]

    _dataset, metrics = collect_dataset(config, epoch_index=2, history=history)

    assert metrics["bootstrap_steps"] == 16.0
    assert metrics["bootstrap_stride"] == 1.0
    assert metrics["bootstrap_ready_streak"] == 2.0
    assert metrics["bootstrap_release_epoch"] == -1.0
    assert metrics["teacher_guidance_ready_streak"] == 2.0
    assert metrics["teacher_guidance_alpha"] == 0.0
    assert metrics["teacher_episode_fraction"] == 0.4
    assert metrics["teacher_step_fraction"] >= 0.0
    assert metrics["teacher_relabel_fraction"] >= 0.0


def test_language_targets_describe_current_state_not_post_action_state() -> None:
    env = HiddenRuleEnv(family_mode="switch_unlock", family_variant="red", seed=13)
    observation = env.reset(seed=13)

    switch = env._switch_positions[SWITCH_RED]
    path, interact_action = _reachable_interaction(env._grid, env._agent, switch)
    for action in path:
        observation = env.step(action).observation

    state = extract_structured_state(observation)
    result = env.step(interact_action)
    next_state = extract_structured_state(result.observation)
    sample = _make_sample(state, next_state, interact_action, result.reward, result.info)

    assert sample["belief_tokens"] == ("goal", "inactive")
    assert sample["question_tokens"] != ("move", "toward", "target")

    activated_sample = _make_sample(next_state, next_state, "wait", 0.0, {"event": "noop"})
    assert activated_sample["belief_tokens"] == ("goal", "active")
    assert activated_sample["question_tokens"] == ("move", "toward", "target")


def test_make_sample_preserves_teacher_action_labels_for_learner_states() -> None:
    env = HiddenRuleEnv(family_mode="switch_unlock", family_variant="red", seed=13)
    observation = env.reset(seed=13)
    state = extract_structured_state(observation)
    result = env.step("wait")
    next_state = extract_structured_state(result.observation)

    sample = _make_sample(
        state,
        next_state,
        "wait",
        result.reward,
        result.info,
        teacher_action="right",
        teacher_weight=0.8,
    )

    assert sample["teacher_action"] == "right"
    assert sample["teacher_weight"] == 0.8


def test_dream_sequence_windows_follow_contiguous_episode_order() -> None:
    config = SyntheticTrainingConfig(
        episodes_per_epoch=4,
        log_every_episodes=0,
    )
    dataset, _metrics = collect_dataset(config, epoch_index=0)

    windows = _dream_sequence_windows(dataset, horizon=3)

    assert windows
    first = windows[0]
    assert len(first) == 3
    for left, right in zip(first, first[1:]):
        assert dataset[left]["next_state"].episode_id == dataset[right]["state"].episode_id
        assert dataset[left]["next_state"].step_index == dataset[right]["state"].step_index


def test_sample_dream_sequences_respects_requested_horizon() -> None:
    config = SyntheticTrainingConfig(
        episodes_per_epoch=4,
        log_every_episodes=0,
    )
    dataset, _metrics = collect_dataset(config, epoch_index=0)

    windows = _sample_dream_sequences(dataset, horizon=2, total_sequences=3, seed=17)

    assert windows
    assert all(len(window) == 2 for window in windows)


def test_dream_phase_produces_grounded_rehearsal_metrics() -> None:
    config = SyntheticTrainingConfig(
        episodes_per_epoch=4,
        log_every_episodes=0,
        dream_batches_per_epoch=1,
        dream_batch_size=2,
        dream_horizon=2,
    )
    dataset, _metrics = collect_dataset(config, epoch_index=0)
    encoder, world_model, language_model, _planner = build_default_modules()
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(world_model.parameters()) + list(language_model.parameters()),
        lr=1e-4,
    )

    metrics = _run_dream_phase(
        config,
        dataset=dataset,
        epoch_index=0,
        encoder=encoder,
        world_model=world_model,
        language_model=language_model,
        optimizer=optimizer,
        device=torch.device("cpu"),
    )

    assert metrics["dream_sequences"] > 0.0
    assert metrics["dream_steps"] >= metrics["dream_sequences"]
    assert metrics["dream_loss"] >= 0.0


def test_stage_family_weights_replay_previous_frontier_families() -> None:
    config = SyntheticTrainingConfig(
        frontier_replay_weight=4,
        previous_stage_replay_weight=2,
    )
    stages = _curriculum_stages(config)

    weights = _stage_family_weights(config, stages, stage_index=1)

    assert weights["selector_unlock"] == 4
    assert weights["delayed_order_unlock"] == 4
    assert weights["switch_unlock"] == 2
    assert weights["order_collect"] == 2


def test_holdout_pass_requires_frontier_and_regression_thresholds() -> None:
    config = SyntheticTrainingConfig()
    stage = _curriculum_stages(config)[1]
    strong_frontier = {
        "first_episode_success": stage.first_episode_threshold + 0.1,
        "later_episode_success": stage.later_episode_threshold + 0.1,
        "avg_return": stage.avg_return_threshold + 0.1,
        "avg_interactions": stage.avg_interactions_cap - 1.0,
    }
    weak_regression = {
        "first_episode_success": stage.regression_first_success_floor - 0.1,
        "later_episode_success": stage.regression_later_success_floor,
    }

    assert not _holdout_passed(stage, strong_frontier, weak_regression)

    strong_regression = {
        "first_episode_success": stage.regression_first_success_floor,
        "later_episode_success": stage.regression_later_success_floor + 0.05,
    }

    assert _holdout_passed(stage, strong_frontier, strong_regression)
