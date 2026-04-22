from __future__ import annotations

from collections import deque
import json
from pathlib import Path

import torch

from arcagi.agents.graph_agent import GraphExplorerAgent
from arcagi.envs.synthetic import EMPTY, HiddenRuleEnv, SWITCH_RED
from arcagi.perception.object_encoder import extract_structured_state
from arcagi.training.synthetic import (
    SyntheticTrainingConfig,
    _apply_episode_hindsight,
    _build_checkpoint_snapshot,
    _build_secondary_training_runtime,
    _bootstrap_schedule,
    _curriculum_stages,
    _dataset_episode_sequences,
    _dream_sequence_windows,
    _evaluate_checkpoint_holdout,
    _evaluate_holdout,
    _finalize_holdout_result,
    _holdout_passed,
    _make_sample,
    _run_multidevice_dream_phase,
    _run_multidevice_replay_phase,
    _run_dream_phase,
    _sample_dream_sequences,
    _should_run_regression_holdout,
    _stage_family_weights,
    _teacher_episode_fraction,
    _trainable_parameters,
    build_default_modules,
    collect_dataset,
    train_synthetic,
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


def test_collect_dataset_uses_hybrid_collector_for_mixed_policy_when_modules_are_available() -> None:
    config = SyntheticTrainingConfig(
        episodes_per_epoch=4,
        log_every_episodes=0,
        behavior_policy="mixed",
    )
    encoder, world_model, language_model, _planner = build_default_modules(device=torch.device("cpu"))

    _dataset, metrics = collect_dataset(
        config,
        epoch_index=0,
        encoder=encoder,
        world_model=world_model,
        language_model=language_model,
        device=torch.device("cpu"),
    )

    assert metrics["collector_agent"] == "hybrid"


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
    assert metrics["dense_teacher_supervision"] is False
    assert metrics["teacher_episode_fraction"] == 0.4
    assert metrics["teacher_step_fraction"] >= 0.0
    assert metrics["teacher_relabel_fraction"] >= 0.0


def test_mixed_warm_start_uses_dense_supervision_without_full_teacher_ownership() -> None:
    config = SyntheticTrainingConfig(
        episodes_per_epoch=32,
        log_every_episodes=0,
        behavior_policy="mixed",
        oracle_imitation_epochs=2,
        teacher_episode_fraction_initial=0.25,
        teacher_episode_fraction_floor=0.25,
        teacher_takeover_prob_initial=0.5,
        teacher_takeover_prob_floor=0.5,
    )
    encoder, world_model, language_model, _planner = build_default_modules(device=torch.device("cpu"))

    _dataset, metrics = collect_dataset(
        config,
        epoch_index=0,
        encoder=encoder,
        world_model=world_model,
        language_model=language_model,
        device=torch.device("cpu"),
    )

    assert metrics["dense_teacher_supervision"] is True
    assert metrics["teacher_episode_fraction"] == 0.25
    assert metrics["teacher_episode_count"] < metrics["episodes"]
    assert metrics["teacher_step_fraction"] < 1.0
    assert metrics["teacher_relabel_fraction"] > 0.0


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

    assert sample["belief_tokens"][0] == "belief"
    assert "inactive" in sample["belief_tokens"]
    assert sample["question_tokens"][0] == "question"
    assert "move" not in sample["question_tokens"]
    assert sample["plan_tokens"][0] == "plan"
    assert "interact" in sample["plan_tokens"]

    activated_sample = _make_sample(next_state, next_state, "wait", 0.0, {"event": "noop"})
    assert activated_sample["belief_tokens"][0] == "belief"
    assert "active" in activated_sample["belief_tokens"]
    assert activated_sample["question_tokens"][0] == "question"
    assert "move" in activated_sample["question_tokens"]
    assert "target" in activated_sample["question_tokens"]


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
    assert sample["plan_tokens"][:5] == ("plan", "action", "move", "direction", "right")


def test_episode_hindsight_boosts_early_credit_for_late_positive_outcome() -> None:
    config = SyntheticTrainingConfig()
    episode_samples = [
        {
            "action": "right",
            "teacher_action": "",
            "teacher_weight": 0.0,
            "reward": 0.0,
            "usefulness": 0.08,
            "policy_target": 0.4,
            "policy_weight": 0.75,
            "sibling_move_target": 0.2,
            "sibling_move_weight": 0.35,
            "same_type_target": 0.0,
            "same_type_weight": 0.0,
        },
        {
            "action": "interact",
            "teacher_action": "",
            "teacher_weight": 0.0,
            "reward": 1.0,
            "usefulness": 0.85,
            "policy_target": 1.0,
            "policy_weight": 1.6,
            "sibling_move_target": 0.0,
            "sibling_move_weight": 0.0,
            "same_type_target": 0.0,
            "same_type_weight": 0.0,
        },
    ]

    _apply_episode_hindsight(config, episode_samples)

    first = episode_samples[0]
    assert first["discounted_return"] > 0.0
    assert first["usefulness"] > 0.08
    assert first["policy_target"] > 0.4
    assert first["replay_weight"] > 1.0


def test_episode_hindsight_upweights_teacher_disagreement_after_bad_future_outcome() -> None:
    config = SyntheticTrainingConfig()
    episode_samples = [
        {
            "action": "wait",
            "teacher_action": "interact",
            "teacher_weight": 0.8,
            "reward": 0.0,
            "usefulness": 0.05,
            "policy_target": 0.2,
            "policy_weight": 0.8,
            "sibling_move_target": 0.0,
            "sibling_move_weight": 0.0,
            "same_type_target": 0.0,
            "same_type_weight": 0.0,
        },
        {
            "action": "interact",
            "teacher_action": "",
            "teacher_weight": 0.0,
            "reward": -1.0,
            "usefulness": -0.45,
            "policy_target": 0.0,
            "policy_weight": 1.8,
            "sibling_move_target": 0.0,
            "sibling_move_weight": 0.0,
            "same_type_target": 0.0,
            "same_type_weight": 0.0,
        },
    ]

    _apply_episode_hindsight(config, episode_samples)

    first = episode_samples[0]
    assert first["teacher_weight"] > 0.8
    assert first["future_setback"] > 0.0
    assert first["outcome_signal"] < 0.0
    assert first["replay_weight"] > 1.0


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


def test_dataset_episode_sequences_preserve_episode_boundaries() -> None:
    config = SyntheticTrainingConfig(
        episodes_per_epoch=4,
        log_every_episodes=0,
    )
    dataset, _metrics = collect_dataset(config, epoch_index=0)

    sequences = _dataset_episode_sequences(dataset)

    assert len(sequences) == 4
    assert sum(len(sequence) for sequence in sequences) == len(dataset)
    assert all(
        len({sample["state"].episode_id for sample in sequence}) == 1
        for sequence in sequences
    )


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
    assert metrics["dream_plan_loss"] >= 0.0


def test_multidevice_replay_phase_uses_secondary_episode_shard_on_cpu() -> None:
    config = SyntheticTrainingConfig(
        episodes_per_epoch=4,
        log_every_episodes=0,
    )
    dataset, _metrics = collect_dataset(config, epoch_index=0)
    encoder, world_model, language_model, _planner = build_default_modules(device=torch.device("cpu"))
    secondary_runtime = _build_secondary_training_runtime(
        encoder,
        world_model,
        language_model,
        device=torch.device("cpu"),
    )
    optimizer = torch.optim.Adam(
        _trainable_parameters(encoder, world_model, language_model),
        lr=1e-4,
    )
    metric_lists = {
        "epoch_losses": [],
        "epoch_uncertainty": [],
        "epoch_world_total_losses": [],
        "epoch_latent_losses": [],
        "epoch_reward_losses": [],
        "epoch_delta_losses": [],
        "epoch_usefulness_losses": [],
        "epoch_belief_losses": [],
        "epoch_question_losses": [],
        "epoch_plan_losses": [],
        "epoch_policy_losses": [],
        "epoch_positive_policy_losses": [],
        "epoch_negative_policy_losses": [],
        "epoch_gate_losses": [],
    }

    metrics = _run_multidevice_replay_phase(
        config,
        dataset=dataset,
        encoder=encoder,
        world_model=world_model,
        language_model=language_model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        secondary_runtime=secondary_runtime,
        metric_lists=metric_lists,
    )

    assert len(metric_lists["epoch_losses"]) == len(dataset)
    assert metrics["secondary_replay_samples"] > 0.0
    assert metrics["secondary_device"] == "cpu"


def test_multidevice_dream_phase_reports_secondary_sequences_on_cpu() -> None:
    config = SyntheticTrainingConfig(
        episodes_per_epoch=4,
        log_every_episodes=0,
        dream_batches_per_epoch=1,
        dream_batch_size=4,
        dream_horizon=2,
    )
    dataset, _metrics = collect_dataset(config, epoch_index=0)
    encoder, world_model, language_model, _planner = build_default_modules(device=torch.device("cpu"))
    secondary_runtime = _build_secondary_training_runtime(
        encoder,
        world_model,
        language_model,
        device=torch.device("cpu"),
    )
    optimizer = torch.optim.Adam(
        _trainable_parameters(encoder, world_model, language_model),
        lr=1e-4,
    )

    metrics = _run_multidevice_dream_phase(
        config,
        dataset=dataset,
        epoch_index=0,
        encoder=encoder,
        world_model=world_model,
        language_model=language_model,
        optimizer=optimizer,
        device=torch.device("cpu"),
        secondary_runtime=secondary_runtime,
    )

    assert metrics["dream_sequences"] > 0.0
    assert metrics["secondary_dream_sequences"] > 0.0
    assert metrics["secondary_device"] == "cpu"


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


def test_regression_holdout_cadence_uses_cached_snapshot_between_due_evals() -> None:
    config = SyntheticTrainingConfig(regression_holdout_every_evals=3)

    assert _should_run_regression_holdout(
        config,
        holdout_eval_index=1,
        cached_regression_metrics=None,
    )
    assert not _should_run_regression_holdout(
        config,
        holdout_eval_index=2,
        cached_regression_metrics={"success_rate": 0.5},
    )
    assert _should_run_regression_holdout(
        config,
        holdout_eval_index=3,
        cached_regression_metrics={"success_rate": 0.5},
    )


def test_single_episode_holdout_does_not_force_zero_later_success() -> None:
    config = SyntheticTrainingConfig(
        holdout_episodes_per_variant=1,
        holdout_failure_examples=1,
        holdout_trace_steps=4,
    )

    result = _evaluate_holdout(
        config,
        GraphExplorerAgent(),
        family_modes=("switch_unlock",),
        size_options=(7,),
        seed_base=17,
    )

    assert result["later_episode_count"] == 0.0
    assert result["later_episode_success"] == result["first_episode_success"]


def test_checkpoint_holdout_evaluation_runs_from_saved_checkpoint(tmp_path: Path) -> None:
    config = SyntheticTrainingConfig(
        checkpoint_path=str(tmp_path / "synthetic.pt"),
        holdout_episodes_per_variant=1,
        holdout_failure_examples=1,
        holdout_trace_steps=4,
    )
    encoder, world_model, language_model, _planner = build_default_modules(device=torch.device("cpu"))
    snapshot = _build_checkpoint_snapshot(
        config,
        encoder,
        world_model,
        language_model,
        history=[],
        holdout_history=[],
        training_state={},
        last_holdout_result=None,
    )
    checkpoint_path = tmp_path / "synthetic.pt"
    torch.save(snapshot, checkpoint_path)

    result = _evaluate_checkpoint_holdout(
        str(checkpoint_path),
        config,
        epoch=0,
        stage_index=0,
        stage_epoch_count=1,
        holdout_eval_index=1,
        device=torch.device("cpu"),
    )

    assert result["epoch"] == 0
    assert result["stage_index"] == 0
    assert result["checkpoint_path"] == str(checkpoint_path)
    assert result["frontier"]["family_modes"] == ("switch_unlock", "order_collect")
    assert result["holdout_seconds"] >= 0.0


def test_resume_checkpoint_restores_stage_and_epoch_progress(tmp_path: Path) -> None:
    resume_path = tmp_path / "resume.pt"
    output_path = tmp_path / "continued.pt"
    encoder, world_model, language_model, _planner = build_default_modules(device=torch.device("cpu"))
    resume_config = SyntheticTrainingConfig(
        checkpoint_path=str(resume_path),
        behavior_policy="graph",
        holdout_eval_every_epochs=0,
        dream_batches_per_epoch=0,
    )
    snapshot = _build_checkpoint_snapshot(
        resume_config,
        encoder,
        world_model,
        language_model,
        history=[
            {"epoch": 0.0, "stage_index": 1.0, "stage_name": "hidden_modes"},
            {"epoch": 1.0, "stage_index": 1.0, "stage_name": "hidden_modes"},
            {"epoch": 2.0, "stage_index": 1.0, "stage_name": "hidden_modes"},
        ],
        holdout_history=[],
        training_state={
            "completed_epochs": 3,
            "stage_index": 1,
            "stage_name": "hidden_modes",
            "stage_epoch_count": 2,
            "consecutive_holdout_passes": 1,
        },
        last_holdout_result=None,
    )
    torch.save(snapshot, resume_path)

    metrics = train_synthetic(
        SyntheticTrainingConfig(
            epochs=1,
            episodes_per_epoch=2,
            checkpoint_path=str(output_path),
            resume_checkpoint_path=str(resume_path),
            behavior_policy="graph",
            holdout_eval_every_epochs=0,
            dream_batches_per_epoch=0,
            log_every_episodes=0,
        ),
        device=torch.device("cpu"),
    )

    assert metrics["epochs_completed"] == 4
    assert metrics["stage_index"] == 1
    assert metrics["last_epoch_checkpoint_path"].endswith("continued.epoch_0003.pt")
    assert Path(metrics["last_epoch_checkpoint_path"]).exists()


def test_weights_only_init_rejects_tracked_training_checkpoint_by_default(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "tracked.pt"
    encoder, world_model, language_model, _planner = build_default_modules(device=torch.device("cpu"))
    snapshot = _build_checkpoint_snapshot(
        SyntheticTrainingConfig(checkpoint_path=str(checkpoint_path)),
        encoder,
        world_model,
        language_model,
        history=[{"epoch": 0.0, "stage_index": 1.0, "stage_name": "hidden_modes"}],
        holdout_history=[],
        training_state={"completed_epochs": 1, "stage_index": 1, "stage_name": "hidden_modes", "stage_epoch_count": 1},
        last_holdout_result=None,
    )
    torch.save(snapshot, checkpoint_path)

    try:
        train_synthetic(
            SyntheticTrainingConfig(
                epochs=1,
                episodes_per_epoch=2,
                checkpoint_path=str(tmp_path / "out.pt"),
                init_checkpoint_path=str(checkpoint_path),
                behavior_policy="graph",
                holdout_eval_every_epochs=0,
                dream_batches_per_epoch=0,
                log_every_episodes=0,
            ),
            device=torch.device("cpu"),
        )
    except ValueError as exc:
        assert "use resume_checkpoint_path" in str(exc)
    else:
        raise AssertionError("expected tracked checkpoint weights-only init to be rejected")


def test_weights_only_init_can_be_explicitly_allowed_for_tracked_checkpoint(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "tracked.pt"
    encoder, world_model, language_model, _planner = build_default_modules(device=torch.device("cpu"))
    snapshot = _build_checkpoint_snapshot(
        SyntheticTrainingConfig(checkpoint_path=str(checkpoint_path)),
        encoder,
        world_model,
        language_model,
        history=[{"epoch": 0.0, "stage_index": 1.0, "stage_name": "hidden_modes"}],
        holdout_history=[],
        training_state={"completed_epochs": 1, "stage_index": 1, "stage_name": "hidden_modes", "stage_epoch_count": 1},
        last_holdout_result=None,
    )
    torch.save(snapshot, checkpoint_path)

    metrics = train_synthetic(
        SyntheticTrainingConfig(
            epochs=1,
            episodes_per_epoch=2,
            checkpoint_path=str(tmp_path / "out.pt"),
            init_checkpoint_path=str(checkpoint_path),
            allow_weights_only_init_from_training_checkpoint=True,
            behavior_policy="graph",
            holdout_eval_every_epochs=0,
            dream_batches_per_epoch=0,
            log_every_episodes=0,
        ),
        device=torch.device("cpu"),
    )

    assert metrics["epochs_completed"] == 1
    assert metrics["stage_index"] == 0


def test_invalid_behavior_policy_is_rejected() -> None:
    config = SyntheticTrainingConfig(behavior_policy="not_a_policy")

    try:
        collect_dataset(config, epoch_index=0)
    except ValueError as exc:
        assert "unsupported behavior_policy" in str(exc)
    else:
        raise AssertionError("expected invalid behavior policy to raise")


def test_finalize_holdout_result_preserves_current_stage_counter_for_stale_results() -> None:
    config = SyntheticTrainingConfig(holdout_eval_every_epochs=1, promotion_consecutive_evals=2)
    raw_result = {
        "epoch": 4,
        "stage_index": 0,
        "stage_name": "foundation",
        "stage_epoch_count": 4,
        "holdout_eval_index": 2,
        "frontier": {"success_rate": 1.0},
        "regression": None,
        "threshold_passed": True,
        "failure_reasons": [],
        "holdout_seconds": 1.2,
        "frontier_holdout_seconds": 0.8,
        "regression_holdout_seconds": 0.0,
        "regression_reference": "none",
        "checkpoint_path": "artifacts/test.pt",
    }

    holdout_result, next_consecutive = _finalize_holdout_result(
        config,
        raw_result,
        current_stage_index=1,
        consecutive_holdout_passes=1,
    )

    assert holdout_result["stale_for_stage"] is True
    assert holdout_result["passed"] is True
    assert next_consecutive == 1
