from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from src.arcagi3_adapter import ArcAGI3Observation, ArcAGI3StepResult
from src.attempt_buffer import AttemptBuffer, tensors_from_attempts
import src.jepa_arc_eval as jepa_arc_eval
import src.jepa_attempt_memory as jepa_attempt_memory
from src.jepa_attempt_memory import JEPAAttemptMemory
from src.jepa_train import synthetic_attempts, train_jepa
from src.video_jepa import VideoJEPA, jepa_loss, null_future_loss


def _obs(step: int, y: int, x: int) -> ArcAGI3Observation:
    grid = np.zeros((8, 8), dtype=np.int64)
    grid[y, x] = 2
    return _obs_grid(step, grid, ("up", "down", "wait"))


def _obs_grid(step: int, grid: np.ndarray, actions: tuple[str, ...]) -> ArcAGI3Observation:
    return ArcAGI3Observation(
        task_id="test",
        episode_id="episode",
        step_index=step,
        grid=grid,
        available_actions=actions,
        extras={},
    )


def _sidecar_controller(
    *,
    mode: str = "propose_only",
    disable_direct: bool = True,
    max_bias: float = 0.05,
) -> jepa_arc_eval.JEPAAugmentedController:
    controller = object.__new__(jepa_arc_eval.JEPAAugmentedController)
    controller.jepa_action_mode = mode
    controller.jepa_max_action_bias = max_bias
    controller.disable_jepa_direct_override = disable_direct
    controller.sidecar_override_ledger = []
    controller.pending_sidecar_override = None
    controller.memory = JEPAAttemptMemory(use_jepa_tokens=True)
    return controller


def test_focused_official_scope_uses_first_manifest_game_and_excludes_noop_controls() -> None:
    assert jepa_arc_eval.manifest_game_ids(limit=1) == ["ar25-0c556536"]
    assert jepa_arc_eval.focused_variant_ids([]) == ["jepa_plus_attempt_memory"]
    assert jepa_arc_eval.focused_causality_variant_ids([]) == [
        "attempt_memory_no_jepa",
        "jepa_random_init",
        "jepa_trained_shuffled",
        "jepa_plus_attempt_memory",
    ]
    assert not set(jepa_arc_eval.focused_variant_ids([])).intersection(jepa_arc_eval.NO_OP_CONTROL_VARIANT_IDS)
    with pytest.raises(ValueError):
        jepa_arc_eval.focused_variant_ids(["jepa_random_init"])


def test_relation_scoring_action_bound_keeps_non_click_actions() -> None:
    actions = ["1", "2", "3", "4", "5", *[f"click:{index}:0" for index in range(200)]]
    bounded = jepa_attempt_memory._bounded_relation_scoring_actions(actions)
    assert len(bounded) == jepa_attempt_memory.RELATION_SCORING_ACTION_LIMIT
    assert bounded[:5] == ["1", "2", "3", "4", "5"]
    assert bounded[5] == "click:0:0"
    assert bounded[-1].startswith("click:")


def test_attempt_memory_numeric_action_families_match_official_controls() -> None:
    assert jepa_attempt_memory._action_family("1") == "move"
    assert jepa_attempt_memory._action_family("2") == "move"
    assert jepa_attempt_memory._action_family("3") == "move"
    assert jepa_attempt_memory._action_family("4") == "move"
    assert jepa_attempt_memory._action_family("5") == "other"
    assert jepa_attempt_memory._action_family("7") == "wait"


def test_observation_key_ignores_public_score_counters() -> None:
    grid = np.zeros((8, 8), dtype=np.int64)
    actions = ("1", "click:1:1")
    left = ArcAGI3Observation("test", "episode", 0, grid, actions, {"score": 0.0, "normalized_score": 0.0})
    right = ArcAGI3Observation("test", "episode", 1, grid, actions, {"score": -0.01, "normalized_score": 0.0})

    assert jepa_attempt_memory._observation_key(left) == jepa_attempt_memory._observation_key(right)


def _clear_non_relation_chain_memory(memory: JEPAAttemptMemory) -> None:
    for entry in memory.entries:
        entry.next_attempt_plan.clear()
    memory.transition_counts.clear()
    memory.transition_values.clear()
    memory.transition_effects.clear()
    memory.transition_failures.clear()
    memory.transition_events.clear()
    memory.state_seen_actions.clear()
    memory.action_counts.clear()
    memory.action_values.clear()
    memory.action_failures.clear()
    memory.action_events.clear()
    memory.family_counts.clear()
    memory.family_values.clear()
    memory.region_values.clear()
    memory.family_region_values.clear()
    memory.color_values.clear()
    memory.component_relation_values.clear()
    memory.family_component_values.clear()
    memory.component_goal_relations.clear()
    memory.component_value_scores.clear()
    memory.component_prediction_values.clear()
    memory.component_prediction_goal_values.clear()
    memory.family_component_prediction_values.clear()
    memory.component_prediction_contradictions.clear()
    memory.component_chain_counts.clear()
    memory.component_chain_values.clear()
    memory.component_chain_goal_values.clear()
    memory.component_chain_failures.clear()
    memory.component_chain_contradictions.clear()
    memory.component_chain_expectations.clear()
    memory.component_chain_edges_by_state.clear()
    memory.component_goal_state_values.clear()
    memory.component_relation_delta_counts.clear()
    memory.component_relation_delta_values.clear()
    memory.component_relation_delta_goal_values.clear()
    memory.component_relation_delta_failures.clear()
    memory.component_relation_delta_contradictions.clear()
    memory.component_relation_delta_tokens_by_scope.clear()
    memory.sequence_candidates.clear()
    memory.active_sequence.clear()
    memory.active_sequence_expectations.clear()
    memory.sequence_cursor = 0
    memory.active_sequence_source = ""


def _clear_non_relation_delta_memory(memory: JEPAAttemptMemory) -> None:
    for entry in memory.entries:
        entry.next_attempt_plan.clear()
    memory.transition_counts.clear()
    memory.transition_values.clear()
    memory.transition_effects.clear()
    memory.transition_failures.clear()
    memory.transition_events.clear()
    memory.state_seen_actions.clear()
    memory.action_counts.clear()
    memory.action_values.clear()
    memory.action_failures.clear()
    memory.action_events.clear()
    memory.family_counts.clear()
    memory.family_values.clear()
    memory.region_values.clear()
    memory.family_region_values.clear()
    memory.color_values.clear()
    memory.component_relation_values.clear()
    memory.family_component_values.clear()
    memory.component_goal_relations.clear()
    memory.component_value_scores.clear()
    memory.component_prediction_values.clear()
    memory.component_prediction_goal_values.clear()
    memory.family_component_prediction_values.clear()
    memory.component_prediction_contradictions.clear()
    memory.component_chain_counts.clear()
    memory.component_chain_values.clear()
    memory.component_chain_goal_values.clear()
    memory.component_chain_failures.clear()
    memory.component_chain_contradictions.clear()
    memory.component_chain_expectations.clear()
    memory.component_chain_edges_by_state.clear()
    memory.component_goal_state_values.clear()
    memory.component_relation_chain_counts.clear()
    memory.component_relation_chain_values.clear()
    memory.component_relation_chain_goal_values.clear()
    memory.component_relation_chain_failures.clear()
    memory.component_relation_chain_contradictions.clear()
    memory.component_relation_chain_expectations.clear()
    memory.component_relation_chain_edges_by_state.clear()
    memory.component_relation_goal_state_values.clear()
    memory.sequence_candidates.clear()
    memory.active_sequence.clear()
    memory.active_sequence_expectations.clear()
    memory.sequence_cursor = 0
    memory.active_sequence_source = ""


def _clear_non_sequence_candidate_memory(memory: JEPAAttemptMemory) -> None:
    for entry in memory.entries:
        entry.next_attempt_plan.clear()
    memory.transition_counts.clear()
    memory.transition_values.clear()
    memory.transition_effects.clear()
    memory.transition_failures.clear()
    memory.transition_events.clear()
    memory.state_seen_actions.clear()
    memory.action_counts.clear()
    memory.action_values.clear()
    memory.action_failures.clear()
    memory.action_events.clear()
    memory.family_counts.clear()
    memory.family_values.clear()
    memory.region_values.clear()
    memory.family_region_values.clear()
    memory.color_values.clear()
    memory.component_relation_values.clear()
    memory.family_component_values.clear()
    memory.component_goal_relations.clear()
    memory.component_value_scores.clear()
    memory.component_prediction_values.clear()
    memory.component_prediction_goal_values.clear()
    memory.family_component_prediction_values.clear()
    memory.component_prediction_contradictions.clear()
    memory.component_chain_counts.clear()
    memory.component_chain_values.clear()
    memory.component_chain_goal_values.clear()
    memory.component_chain_failures.clear()
    memory.component_chain_contradictions.clear()
    memory.component_chain_expectations.clear()
    memory.component_chain_edges_by_state.clear()
    memory.component_goal_state_values.clear()
    memory.component_relation_chain_counts.clear()
    memory.component_relation_chain_values.clear()
    memory.component_relation_chain_goal_values.clear()
    memory.component_relation_chain_failures.clear()
    memory.component_relation_chain_contradictions.clear()
    memory.component_relation_chain_expectations.clear()
    memory.component_relation_chain_edges_by_state.clear()
    memory.component_relation_goal_state_values.clear()
    memory.component_relation_delta_counts.clear()
    memory.component_relation_delta_values.clear()
    memory.component_relation_delta_goal_values.clear()
    memory.component_relation_delta_failures.clear()
    memory.component_relation_delta_contradictions.clear()
    memory.component_relation_delta_tokens_by_scope.clear()
    memory.active_sequence.clear()
    memory.active_sequence_expectations.clear()
    memory.sequence_cursor = 0
    memory.active_sequence_source = ""


def test_attempt_buffer_stores_full_attempt_timeline() -> None:
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=1, attempt_index=2)
    before = _obs(0, 2, 2)
    after = _obs(1, 3, 2)
    result = ArcAGI3StepResult(after, 0.25, False, False, {"events": ["positive_reward"]})
    buffer.append_transition(before, "down", result)
    record = buffer.to_record()
    payload = record.to_dict()
    assert payload["attempt_index"] == 2
    assert payload["steps"][0]["frame"][2][2] == 2
    assert payload["steps"][0]["action"] == "down"
    assert payload["steps"][0]["legal_actions"] == ["up", "down", "wait"]
    assert payload["steps"][0]["score_delta"] == 0.25
    assert payload["steps"][0]["event_delta"] == ["positive_reward"]
    assert payload["steps"][0]["next_frame"][3][2] == 2
    tensors = tensors_from_attempts([record])
    assert tensors["frames"].shape[0] == 1
    assert tensors["action_features"].shape[-1] == 16
    assert tensors["legal_masks"].sum().item() == 3


def test_video_jepa_predicts_latent_future_without_text() -> None:
    records = synthetic_attempts(12, seed=44)
    batch = tensors_from_attempts(records, max_steps=24)
    model = VideoJEPA()
    output = model(batch["frames"], batch["action_ids"], batch["legal_counts"], batch["valid"])
    assert output["predicted_future"].shape == output["target_future"].shape
    assert model.emits_text is False
    loss = jepa_loss(output)
    null = null_future_loss(output)
    assert torch.isfinite(loss)
    assert torch.isfinite(null)


def test_tiny_training_beats_null_on_generated_attempts() -> None:
    records = synthetic_attempts(32, seed=45)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, metrics = train_jepa(records, device=device, seed=123)
    assert metrics["jepa_beats_null"] is True
    assert metrics["predicts_actions"] is False
    assert metrics["emits_text"] is False


def test_attempt_memory_bias_uses_prior_attempts_not_direct_jepa_action() -> None:
    records = synthetic_attempts(4, seed=46)
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    memory.ingest_attempt(records[0])
    summary = memory.summary()
    assert summary["direct_action_source"] is False
    assert summary["emits_text"] is False
    legal_actions = records[0].steps[0].legal_actions
    scores = {action: memory.score_action(action) for action in legal_actions}
    assert all(isinstance(value, float) for value in scores.values())
    assert summary["causal_chain"] == [
        "attempt_video_action_history",
        "jepa_temporal_representation",
        "attempt_memory",
        "rule_causal_hypothesis_update",
        "sidecar_action_prior_proposals",
        "symbolic_causal_controller_approval",
    ]


def test_attempt_memory_separates_visible_effect_from_failed_action() -> None:
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=2, attempt_index=1)
    before = _obs(0, 2, 2)
    after = _obs(1, 3, 2)
    buffer.append_transition(before, "down", ArcAGI3StepResult(after, -0.001, False, False, {"events": ["move"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    entry = memory.ingest_attempt(buffer.to_record())

    assert "down" not in entry.failed_actions
    assert entry.effect_actions["down"] == 1
    assert entry.causal_hypotheses["no_effect_rate"] == 0.0
    assert entry.causal_hypotheses["visible_effect_rate"] == 1.0
    assert entry.causal_hypotheses["useful_effect_rate"] == 1.0
    assert entry.causal_hypotheses["nuisance_effect_rate"] == 0.0
    assert entry.causal_hypotheses["controllability_effect_rate"] == 1.0
    assert entry.causal_hypotheses["reachable_state_class_effect_rate"] == 1.0
    assert entry.transition_graph_summary["useful_edges"] == 1
    assert entry.transition_graph_summary["nuisance_edges"] == 0
    assert memory.score_action("down") == 0.0
    assert entry.next_attempt_plan.get("down", 0.0) == 0.0


def test_scoreless_visible_action_return_penalizes_exact_repeat() -> None:
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=23, attempt_index=1)
    actions = ("1", "2", "3", "4")
    for step in range(8):
        before_grid = np.zeros((8, 8), dtype=np.int64)
        after_grid = np.zeros((8, 8), dtype=np.int64)
        before_grid[2, min(2 + step, 7)] = 2
        after_grid[2, min(3 + step, 7)] = 2
        before = _obs_grid(step, before_grid, actions)
        after = _obs_grid(step + 1, after_grid, actions)
        buffer.append_transition(before, "4", ArcAGI3StepResult(after, -0.001, False, False, {"events": ["move"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    entry = memory.ingest_attempt(buffer.to_record())
    scores = memory.plan_scores_for_observation(_obs_grid(20, np.zeros((8, 8), dtype=np.int64), actions))

    assert entry.transition_graph_summary["observed_actions"] == 1
    assert entry.transition_graph_summary["negative_actions"] == 1
    assert scores["4"] < scores["1"]


def test_attempt_memory_penalizes_repeated_no_effect_actions() -> None:
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=3, attempt_index=1)
    for step in range(6):
        before = _obs(step, 2, 2)
        after = _obs(step + 1, 2, 2)
        buffer.append_transition(before, "wait", ArcAGI3StepResult(after, -0.001, False, False, {"events": ["wait"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    entry = memory.ingest_attempt(buffer.to_record())

    assert entry.failed_actions["wait"] == 6
    assert entry.repeated_actions["wait"] == 1.0
    assert entry.causal_hypotheses["no_effect_rate"] == 1.0
    assert memory.score_action("wait") < memory.score_action("down")


def test_transition_graph_planner_penalizes_public_no_effect_edge() -> None:
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=4, attempt_index=1)
    before = _obs(0, 2, 2)
    after = _obs(1, 2, 2)
    buffer.append_transition(before, "wait", ArcAGI3StepResult(after, -0.001, False, False, {"events": ["no_effect"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    entry = memory.ingest_attempt(buffer.to_record())
    scores = memory.plan_scores_for_observation(before)

    assert entry.transition_graph_summary["observed_edges"] == 1
    assert entry.transition_graph_summary["no_effect_edges"] == 1
    assert memory.summary()["transition_graph"]["observed_states"] == 1
    assert scores["wait"] <= -0.5
    assert scores["wait"] < scores["down"]


def test_live_no_effect_transition_updates_public_edge_penalty() -> None:
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    before = _obs(0, 2, 2)
    after = _obs(1, 2, 2)
    result = ArcAGI3StepResult(after, -0.001, False, False, {"events": ["no_effect"]})

    memory.observe_live_transition(before, "wait", result)
    scores = memory.plan_scores_for_observation(before)

    assert memory.transition_graph_summary()["observed_edges"] == 1
    assert memory.transition_graph_summary()["no_effect_edges"] == 1
    assert scores["wait"] <= -0.5
    assert scores["wait"] < scores["down"]


def test_jepa_bridge_support_decays_after_public_no_effect_evidence() -> None:
    memory = JEPAAttemptMemory(use_jepa_tokens=True)
    before = _obs(0, 2, 2)
    after = _obs(1, 2, 2)
    no_effect = ArcAGI3StepResult(after, -0.001, False, False, {"events": ["no_effect"]})

    prior = memory.bridge_public_support_weight("click:1:1")
    for _ in range(4):
        memory.observe_live_transition(before, "click:1:1", no_effect)
    decayed = memory.bridge_public_support_weight("click:1:1")
    positive = ArcAGI3StepResult(after, 1.0, False, False, {"events": ["positive_reward"]})
    memory.observe_live_transition(before, "click:2:2", positive)

    assert prior <= 0.20
    assert decayed < prior
    assert memory.bridge_public_support_weight("click:2:2") == 1.0


def test_visible_delta_without_progress_is_not_macro() -> None:
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=93, attempt_index=1)
    actions = ("4", "5", "wait")
    for step in range(10):
        before_grid = np.zeros((8, 8), dtype=np.int64)
        after_grid = np.zeros((8, 8), dtype=np.int64)
        before_grid[2, 2] = 4
        after_grid[2, 2] = 5
        before = _obs_grid(step, before_grid, actions)
        after = _obs_grid(step + 1, after_grid, actions)
        buffer.append_transition(before, "4", ArcAGI3StepResult(after, -0.001, False, False, {"events": ["toggle"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    entry = memory.ingest_attempt(buffer.to_record())
    memory.start_attempt()
    scores = memory.plan_scores_for_observation(_obs_grid(12, np.zeros((8, 8), dtype=np.int64), actions))

    assert entry.causal_hypotheses["visible_effect_rate"] == 1.0
    assert entry.causal_hypotheses["useful_effect_rate"] == 0.0
    assert entry.causal_hypotheses["positive_event_rate"] == 0.0
    assert entry.causal_hypotheses["nuisance_effect_rate"] == 1.0
    assert entry.causal_hypotheses["transition_graph_useful_edges"] == 0.0
    assert entry.causal_hypotheses["transition_graph_delayed_credit_edges"] == 0.0
    assert entry.causal_hypotheses["component_goal_relation_count"] == 0.0
    assert entry.causal_hypotheses["component_relation_delta_goal_count"] == 0.0
    assert entry.transition_graph_summary["positive_edges"] == 0
    assert entry.transition_graph_summary["nuisance_edges"] >= 1
    assert entry.sequence_plan_summary["sequence_candidate_count"] == 0
    assert memory.sequence_plan_action(actions, observation=_obs_grid(13, np.zeros((8, 8), dtype=np.int64), actions)) is None
    assert all(value <= 0.0 for value in scores.values())
    assert scores["4"] < scores["wait"]


def test_jepa_override_requires_posthoc_usefulness() -> None:
    memory = JEPAAttemptMemory(use_jepa_tokens=True)
    actions = ("click:2:2", "click:6:6", "wait")
    before_grid = np.zeros((8, 8), dtype=np.int64)
    after_grid = np.zeros((8, 8), dtype=np.int64)
    after_grid[2, 2] = 5
    before = _obs_grid(0, before_grid, actions)
    after = _obs_grid(1, after_grid, actions)
    nuisance = ArcAGI3StepResult(after, -0.001, False, False, {"events": ["toggle"]})

    prior = memory.bridge_public_support_weight("click:2:2")
    for _ in range(8):
        memory.observe_live_transition(before, "click:2:2", nuisance)
    summary = memory.transition_graph_summary()

    assert prior <= 0.20
    assert summary["effect_edges"] == 1
    assert summary["useful_edges"] == 0
    assert summary["positive_edges"] == 0
    assert summary["nuisance_edges"] == 1
    assert memory.action_events.get("click:2:2", 0) == 0
    assert memory.bridge_public_support_weight("click:2:2") < prior


def test_no_effect_family_evidence_downweights_unseen_bridge_actions() -> None:
    memory = JEPAAttemptMemory(use_jepa_tokens=True)
    before = _obs_grid(0, np.zeros((8, 8), dtype=np.int64), ("1", "click:7:7"))
    after = _obs_grid(1, np.zeros((8, 8), dtype=np.int64), ("1", "click:7:7"))
    no_effect = ArcAGI3StepResult(after, -0.001, False, False, {"events": ["no_effect"]})

    for index in range(48):
        memory.observe_live_transition(before, f"click:{index % 8}:{index // 8}", no_effect)
    scores = memory.plan_scores_for_observation(before)

    assert memory.bridge_public_support_weight("click:7:7") <= 0.10
    assert scores["click:7:7"] < scores["1"]


def test_public_no_effect_family_can_suppress_contact_actions() -> None:
    memory = JEPAAttemptMemory(use_jepa_tokens=True)
    before = _obs_grid(0, np.zeros((8, 8), dtype=np.int64), ("1", "click:7:7"))
    after = _obs_grid(1, np.zeros((8, 8), dtype=np.int64), ("1", "click:7:7"))
    no_effect = ArcAGI3StepResult(after, -0.001, False, False, {"events": ["no_effect"]})

    for index in range(56):
        memory.observe_live_transition(before, f"click:{index % 8}:{index // 8}", no_effect)

    assert memory.public_no_effect_suppresses_action("click:7:7") is True
    assert memory.public_no_effect_suppresses_action("1") is False


def test_transition_graph_planner_credits_delayed_public_event_predecessor() -> None:
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=5, attempt_index=1)
    start = _obs(0, 2, 2)
    moved = _obs(1, 3, 2)
    after_event = _obs(2, 3, 2)
    buffer.append_transition(start, "down", ArcAGI3StepResult(moved, -0.001, False, False, {"events": ["move"]}))
    buffer.append_transition(moved, "up", ArcAGI3StepResult(after_event, 1.0, False, False, {"events": ["positive_reward"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    entry = memory.ingest_attempt(buffer.to_record())
    scores = memory.plan_scores_for_observation(start)

    assert entry.causal_hypotheses["transition_graph_delayed_credit_edges"] == 1.0
    assert entry.transition_graph_summary["positive_edges"] == 1
    assert entry.transition_graph_summary["effect_edges"] == 1
    assert scores["down"] > scores["wait"]
    assert scores["down"] > 0.0


def test_object_causal_hypothesis_detects_public_movement() -> None:
    before_grid = np.zeros((8, 8), dtype=np.int64)
    before_grid[2, 2] = 4
    after_grid = np.zeros((8, 8), dtype=np.int64)
    after_grid[2, 3] = 4
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=6, attempt_index=1)
    before = _obs_grid(0, before_grid, ("right", "left", "wait"))
    after = _obs_grid(1, after_grid, ("right", "left", "wait"))
    buffer.append_transition(before, "right", ArcAGI3StepResult(after, -0.001, False, False, {"events": ["move"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    entry = memory.ingest_attempt(buffer.to_record())

    assert entry.object_causal_hypotheses
    assert entry.object_causal_hypotheses[0]["mechanism"] == "movement"
    assert entry.object_causal_hypotheses[0]["movement_colors"] == [4]
    assert entry.causal_hypotheses["object_changed_region_count"] >= 1.0
    assert memory.summary()["object_memory"]["observed_colors"] == 1


def test_public_movement_creates_controllability_usefulness_without_progress() -> None:
    before_grid = np.zeros((8, 8), dtype=np.int64)
    before_grid[2, 2] = 4
    after_grid = np.zeros((8, 8), dtype=np.int64)
    after_grid[2, 3] = 4
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    memory.start_attempt()
    before = _obs_grid(0, before_grid, ("right", "left", "wait"))
    after = _obs_grid(1, after_grid, ("right", "left", "wait"))
    experiment = memory.experiment_for_action(before, "right")

    assert experiment is not None
    memory.begin_experiment(experiment)
    memory.observe_live_transition(before, "right", ArcAGI3StepResult(after, -0.001, False, False, {"events": ["move"]}))
    summary = memory.transition_graph_summary()
    recent = memory.summary()["experiment_protocol"]["recent_results"][-1]

    assert summary["useful_edges"] == 1
    assert summary["nuisance_edges"] == 0
    assert recent["classification"] == "controllability"
    assert recent["posthoc_useful"] is True
    assert recent["retired"] is False


def test_object_region_memory_scores_contact_clicks_from_public_frame_diff() -> None:
    before_grid = np.zeros((8, 8), dtype=np.int64)
    after_grid = np.zeros((8, 8), dtype=np.int64)
    after_grid[2, 2] = 5
    actions = ("click:20:20", "click:56:56", "wait")
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=7, attempt_index=1)
    before = _obs_grid(0, before_grid, actions)
    after = _obs_grid(1, after_grid, actions)
    buffer.append_transition(before, "click:20:20", ArcAGI3StepResult(after, 0.5, False, False, {"events": ["useful_click"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    entry = memory.ingest_attempt(buffer.to_record())
    scores = memory.plan_scores_for_observation(before)

    assert entry.object_causal_hypotheses[0]["mechanism"] == "spawn"
    assert entry.object_causal_hypotheses[0]["click_contacts_change"] is True
    assert entry.causal_hypotheses["object_delayed_region_links"] == 0.0
    assert scores["click:20:20"] > scores["click:56:56"]
    assert scores["click:20:20"] > scores["wait"]


def test_sequence_plan_replays_prior_public_event_window_when_legal() -> None:
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=8, attempt_index=1)
    start = _obs(0, 2, 2)
    moved = _obs(1, 3, 2)
    after_event = _obs(2, 3, 2)
    buffer.append_transition(start, "down", ArcAGI3StepResult(moved, -0.001, False, False, {"events": ["move"]}))
    buffer.append_transition(moved, "up", ArcAGI3StepResult(after_event, 1.0, False, False, {"events": ["positive_reward"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    entry = memory.ingest_attempt(buffer.to_record())

    assert entry.sequence_plan_summary["sequence_candidate_count"] >= 1
    memory.start_attempt()
    first_scores = memory.plan_scores_for_observation(start)
    assert first_scores["down"] > first_scores["wait"]
    memory.advance_sequence("down")
    second_scores = memory.plan_scores_for_observation(moved)
    assert second_scores["up"] > second_scores["wait"]


def test_noncontact_positive_prefix_replays_before_speculative_chain() -> None:
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=82, attempt_index=1)
    start = _obs(0, 2, 2)
    first = _obs(1, 3, 2)
    second = _obs(2, 3, 3)
    event = _obs(3, 3, 3)
    buffer.append_transition(start, "down", ArcAGI3StepResult(first, -0.001, False, False, {"events": ["move"]}))
    buffer.append_transition(first, "right", ArcAGI3StepResult(second, -0.001, False, False, {"events": ["move"]}))
    buffer.append_transition(second, "up", ArcAGI3StepResult(event, 1.0, False, False, {"events": ["positive_reward"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    memory.ingest_attempt(buffer.to_record())
    memory.start_attempt()

    assert memory.summary()["object_memory"]["best_event_prefix_length"] == 3
    assert memory.summary()["object_memory"]["active_sequence_source"] == "positive_public_control_prefix"
    assert memory.sequence_plan_action(("up", "down", "right", "wait"), observation=start) == "down"
    activated = memory._activate_component_chain_plan(
        {"source": "component_transition_goal_chain", "actions": ["wait"], "component_expectations": [{}], "value": 9.0}
    )
    assert activated is False
    assert memory.sequence_plan_action(("up", "down", "right", "wait"), observation=start) == "down"
    memory.observe_live_transition(start, "down", ArcAGI3StepResult(first, -0.001, False, False, {"events": ["move"]}))
    memory.observe_live_transition(first, "right", ArcAGI3StepResult(second, -0.001, False, False, {"events": ["move"]}))
    memory.observe_live_transition(second, "up", ArcAGI3StepResult(event, 1.0, False, False, {"events": ["positive_reward"]}))
    assert memory.summary()["object_memory"]["positive_prefix_completed"] is True


def test_level_completion_retires_literal_replay_but_preserves_affordance_memory() -> None:
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=83, attempt_index=1)
    start = _obs(0, 2, 2)
    first = _obs(1, 3, 2)
    event = _obs(2, 3, 3)
    buffer.append_transition(start, "down", ArcAGI3StepResult(first, -0.001, False, False, {"events": ["move"]}))
    buffer.append_transition(first, "right", ArcAGI3StepResult(event, 1.0, False, False, {"events": ["level_completed"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    memory.ingest_attempt(buffer.to_record())
    memory.start_attempt()

    assert memory.sequence_plan_action(("down", "right", "click:3:3", "wait"), observation=start) == "down"
    memory.observe_live_transition(start, "down", ArcAGI3StepResult(first, -0.001, False, False, {"events": ["move"]}))
    memory.observe_live_transition(first, "right", ArcAGI3StepResult(event, 1.0, False, False, {"events": ["level_completed"]}))

    summary = memory.summary()["object_memory"]
    assert summary["discovery_phase"] == 1
    assert summary["active_sequence_remaining"] == 0
    assert summary["active_sequence_source"] == "post_level_discovery"
    assert memory.action_events["right"] >= 1
    assert memory.best_event_prefix == ["down", "right"]


def test_post_level_discovery_candidates_include_new_legal_family() -> None:
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    memory.positive_prefix_completed = True
    memory.discovery_phase = 1
    observation = _obs_grid(
        3,
        np.asarray([[0, 0, 0], [0, 2, 5], [0, 0, 0]], dtype=np.int64),
        ("down", "right", "click:2:1", "wait"),
    )
    candidates = memory.discovery_exploration_candidates(
        observation.available_actions,
        observation=observation,
        scores={"down": 0.5, "right": 0.4, "click:2:1": 0.3, "wait": 9.0},
    )

    assert "click:2:1" in candidates
    assert "wait" not in candidates


def test_controller_uses_post_level_discovery_after_prefix_source_is_retired() -> None:
    class DummyBase:
        def choose_action(self, observation: ArcAGI3Observation):
            del observation
            return (
                "1",
                {
                    "action_scores": {
                        "1": 2.0,
                        "2": 1.5,
                        "5": 0.8,
                        "click:2:1": 0.1,
                        "7": -1.0,
                    },
                    "policy": {},
                },
            )

        def summary(self):
            return {}

    controller = object.__new__(jepa_arc_eval.JEPAAugmentedController)
    controller.variant = jepa_arc_eval.JEPAVariant("unit", use_memory=True)
    controller.base = DummyBase()
    controller.memory = JEPAAttemptMemory(use_jepa_tokens=False)
    controller.memory.positive_prefix_completed = True
    controller.memory.discovery_phase = 1
    controller.memory.active_sequence_source = "post_level_discovery"
    controller.memory.phase_family_counts[(1, "move")] = 4
    controller.memory.phase_family_counts[(1, "object")] = 2
    controller.jepa_model = None
    controller.device = torch.device("cpu")
    controller.changed_actions = 0
    controller.frames = 0
    controller.last_base_action = None
    controller.last_observation = None
    controller.bridge_history_frames = []
    controller.bridge_history_action_ids = []
    controller.bridge_history_action_features = []
    controller.bridge_history_legal_counts = []
    controller.rng = jepa_arc_eval.random.Random(0)
    controller.post_prefix_rng = jepa_arc_eval.random.Random(0)

    observation = _obs_grid(
        4,
        np.asarray([[0, 0, 0], [0, 2, 5], [0, 0, 0]], dtype=np.int64),
        ("1", "2", "5", "click:2:1", "7"),
    )
    action, diagnostics = controller.choose_action(observation)

    assert action == "click:2:1"
    assert diagnostics["jepa_policy"]["public_post_prefix_exploration"] is False
    assert diagnostics["jepa_policy"]["experiment_protocol"]["active"] is True
    assert diagnostics["jepa_policy"]["experiment_protocol"]["experiment"]["action"] == "click:2:1"


def test_action_selection_reports_experiment_prediction_before_acting() -> None:
    class DummyBase:
        def choose_action(self, observation: ArcAGI3Observation):
            return "wait", {"action_scores": {action: 0.0 for action in observation.available_actions}, "policy": {}}

        def observe_transition(self, action: str, result: ArcAGI3StepResult) -> None:
            del action, result

        def summary(self) -> dict[str, object]:
            return {}

    controller = object.__new__(jepa_arc_eval.JEPAAugmentedController)
    controller.variant = jepa_arc_eval.JEPAVariant("unit", use_memory=True)
    controller.base = DummyBase()
    controller.memory = JEPAAttemptMemory(use_jepa_tokens=False)
    controller.memory.start_attempt()
    controller.jepa_model = None
    controller.device = torch.device("cpu")
    controller.changed_actions = 0
    controller.frames = 0
    controller.last_base_action = None
    controller.last_observation = None
    controller.bridge_history_frames = []
    controller.bridge_history_action_ids = []
    controller.bridge_history_action_features = []
    controller.bridge_history_legal_counts = []
    controller.rng = jepa_arc_eval.random.Random(0)
    controller.post_prefix_rng = jepa_arc_eval.random.Random(0)
    controller.anti_attractor = jepa_arc_eval.HardAntiAttractorGate()
    observation = _obs_grid(0, np.zeros((8, 8), dtype=np.int64), ("wait", "1", "2", "click:2:1"))

    action, diagnostics = controller.choose_action(observation)
    experiment = diagnostics["jepa_policy"]["experiment_protocol"]

    assert action == "1"
    assert experiment["active"] is True
    assert experiment["selected"] is True
    assert experiment["executed"] is True
    assert experiment["experiment"]["action"] == action
    assert experiment["experiment"]["hypothesis_ids"]
    assert "h_progress" in experiment["experiment"]["predicted_outcomes"]
    assert "h_no_change" in experiment["experiment"]["predicted_outcomes"]
    assert "score_delta > 0" in experiment["experiment"]["useful_if"]
    assert controller.memory.pending_experiment is not None
    assert controller.memory.pending_experiment["predicted_outcomes"] == experiment["experiment"]["predicted_outcomes"]


def test_hard_veto_fallback_reports_experiment_for_executed_action() -> None:
    class DummyBase:
        def choose_action(self, observation: ArcAGI3Observation):
            return (
                "click:2:1",
                {
                    "action_scores": {
                        "click:2:1": 9.0,
                        "click:5:1": 8.0,
                        "5": 1.0,
                    },
                    "policy": {},
                },
            )

        def observe_transition(self, action: str, result: ArcAGI3StepResult) -> None:
            del action, result

        def summary(self) -> dict[str, object]:
            return {}

    controller = object.__new__(jepa_arc_eval.JEPAAugmentedController)
    controller.variant = jepa_arc_eval.JEPAVariant("unit", use_memory=True)
    controller.base = DummyBase()
    controller.memory = JEPAAttemptMemory(use_jepa_tokens=False)
    controller.memory.start_attempt()
    controller.jepa_model = None
    controller.device = torch.device("cpu")
    controller.changed_actions = 0
    controller.frames = 0
    controller.last_base_action = None
    controller.last_observation = None
    controller.bridge_history_frames = []
    controller.bridge_history_action_ids = []
    controller.bridge_history_action_features = []
    controller.bridge_history_legal_counts = []
    controller.rng = jepa_arc_eval.random.Random(0)
    controller.post_prefix_rng = jepa_arc_eval.random.Random(0)
    controller.anti_attractor = jepa_arc_eval.HardAntiAttractorGate()
    grid = np.zeros((8, 8), dtype=np.int64)
    grid[1, 2] = 5
    grid[1, 5] = 6
    observation = _obs_grid(0, grid, ("click:2:1", "click:5:1", "5"))
    controller.anti_attractor.observe_transition(
        observation,
        "click:2:1",
        ArcAGI3StepResult(observation, -0.001, False, False, {"events": []}),
    )

    action, diagnostics = controller.choose_action(observation)
    experiment = diagnostics["jepa_policy"]["experiment_protocol"]

    assert action == "click:5:1"
    assert diagnostics["jepa_policy"]["pre_veto_chosen_action"] == "click:2:1"
    assert experiment["active"] is True
    assert experiment["executed"] is True
    assert experiment["hard_veto_replacement"] is True
    assert experiment["experiment"]["action"] == "click:5:1"
    assert experiment["pre_veto_experiment"]["action"] == "click:2:1"
    assert controller.memory.pending_experiment is not None
    assert controller.memory.pending_experiment["action"] == "click:5:1"
    assert controller.memory.pending_experiment["predicted_outcomes"] == experiment["experiment"]["predicted_outcomes"]


def test_no_experiment_repeated_after_nuisance_classification() -> None:
    class DummyBase:
        def choose_action(self, observation: ArcAGI3Observation):
            return "wait", {"action_scores": {action: 0.0 for action in observation.available_actions}, "policy": {}}

        def observe_transition(self, action: str, result: ArcAGI3StepResult) -> None:
            del action, result

        def summary(self) -> dict[str, object]:
            return {}

    controller = object.__new__(jepa_arc_eval.JEPAAugmentedController)
    controller.variant = jepa_arc_eval.JEPAVariant("unit", use_memory=True)
    controller.base = DummyBase()
    controller.memory = JEPAAttemptMemory(use_jepa_tokens=False)
    controller.memory.start_attempt()
    controller.memory.action_counts["1"] = 1
    controller.memory.family_counts["move"] = 8
    controller.memory.phase_action_counts[(controller.memory.discovery_phase, "2")] = 1
    controller.jepa_model = None
    controller.device = torch.device("cpu")
    controller.changed_actions = 0
    controller.frames = 0
    controller.last_base_action = None
    controller.last_observation = None
    controller.bridge_history_frames = []
    controller.bridge_history_action_ids = []
    controller.bridge_history_action_features = []
    controller.bridge_history_legal_counts = []
    controller.rng = jepa_arc_eval.random.Random(0)
    controller.post_prefix_rng = jepa_arc_eval.random.Random(0)
    controller.anti_attractor = jepa_arc_eval.HardAntiAttractorGate()
    before_grid = np.zeros((8, 8), dtype=np.int64)
    before_grid[2, 2] = 4
    after_grid = np.zeros((8, 8), dtype=np.int64)
    after_grid[2, 3] = 4
    observation = _obs_grid(0, before_grid, ("1", "2", "wait"))
    after = _obs_grid(1, after_grid, ("1", "2", "wait"))

    first_action, first_diagnostics = controller.choose_action(observation)
    controller.observe_transition(first_action, ArcAGI3StepResult(after, -0.001, False, False, {"events": ["toggle"]}))
    second_action, second_diagnostics = controller.choose_action(observation)

    assert first_action == "1"
    assert first_diagnostics["jepa_policy"]["experiment_protocol"]["experiment"]["action"] == "1"
    assert controller.memory.summary()["experiment_protocol"]["recent_results"][-1]["classification"] == "nuisance"
    assert controller.memory.summary()["experiment_protocol"]["recent_results"][-1]["retired"] is True
    assert second_action == "2"
    assert second_diagnostics["jepa_policy"]["experiment_protocol"]["experiment"]["action"] == "2"


def test_experiment_equivalence_class_retires_across_state_changes() -> None:
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    memory.start_attempt()

    grid_a = np.zeros((8, 8), dtype=np.int64)
    grid_a[1:3, 1:3] = 5
    grid_b = grid_a.copy()
    grid_b[7, 7] = 6
    grid_c = grid_a.copy()
    grid_c[6:8, 6:8] = 6
    actions = ("click:8:8", "click:40:40")

    for step, grid in enumerate((grid_a, grid_b)):
        observation = _obs_grid(step, grid, actions)
        experiment = memory.experiment_for_action(observation, "click:8:8")
        assert experiment is not None
        memory.begin_experiment(experiment)
        memory.observe_live_transition(
            observation,
            "click:8:8",
            ArcAGI3StepResult(observation, -0.001, False, False, {"events": []}),
        )

    candidate = memory.experiment_for_action(_obs_grid(2, grid_c, actions), "click:8:8")
    selected = memory.select_experiment(_obs_grid(2, grid_c, actions), actions)
    summary = memory.summary()["experiment_protocol"]

    assert candidate is None
    assert selected is not None
    assert selected.action == "click:40:40"
    assert summary["retired_experiment_class_count"] == 1
    assert memory.experiment_history[-1]["class_retired"] is True


def test_select_experiment_uses_scores_for_contact_class_representative() -> None:
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    memory.start_attempt()
    grid = np.zeros((8, 8), dtype=np.int64)
    grid[1:3, 1:3] = 5
    grid[5, 5] = 6
    actions = ("click:8:8", "click:15:15", "click:40:40")
    observation = _obs_grid(0, grid, actions)

    low = memory.experiment_for_action(observation, "click:8:8")
    high = memory.experiment_for_action(observation, "click:15:15")
    other = memory.experiment_for_action(observation, "click:40:40")
    selected = memory.select_experiment(
        observation,
        actions,
        scores={"click:8:8": 0.1, "click:15:15": 0.9, "click:40:40": 0.2},
    )

    assert low is not None and high is not None and other is not None
    assert low.coordinate_equiv_class == high.coordinate_equiv_class
    assert other.coordinate_equiv_class != high.coordinate_equiv_class
    assert selected is not None
    assert selected.action == "click:15:15"
    assert selected.coordinate_equiv_class == high.coordinate_equiv_class


def test_coordinate_equiv_class_collapses_raw_clicks_but_separates_component_cells() -> None:
    grid = np.zeros((8, 8), dtype=np.int64)
    grid[1:5, 1:5] = 5

    same_cell_a = jepa_attempt_memory._experiment_coordinate_equiv_class("click:8:8", grid)
    same_cell_b = jepa_attempt_memory._experiment_coordinate_equiv_class("click:15:15", grid)
    adjacent_cell = jepa_attempt_memory._experiment_coordinate_equiv_class("click:16:8", grid)
    lower_cell = jepa_attempt_memory._experiment_coordinate_equiv_class("click:8:16", grid)

    assert same_cell_a == same_cell_b
    assert same_cell_a != adjacent_cell
    assert same_cell_a != lower_cell
    assert ":component_cell:0:0" in same_cell_a
    assert ":component_cell:0:1" in adjacent_cell
    assert ":component_cell:1:0" in lower_cell


def test_jepa_sidecar_is_read_only_proposer_by_default() -> None:
    controller = _sidecar_controller()

    ledger, applied = controller._evaluate_sidecar_override(
        base_action="1",
        bridge_scores={"1": 0.0, "2": 0.8},
        bridge_public_support={"1": 0.0, "2": 1.0},
    )

    assert ledger["schema"] == "sidecar_override_ledger_v1"
    assert ledger["active"] is True
    assert ledger["proposed_action"] == "2"
    assert ledger["base_action"] == "1"
    assert ledger["approved"] is False
    assert ledger["reason"] == "propose_only_read_only"
    assert ledger["mode"] == "propose_only"
    assert ledger["posthoc_useful"] is None
    assert all(value == 0.0 for value in applied.values())


def test_approved_jepa_sidecar_bias_is_capped_and_posthoc_ledgered() -> None:
    controller = _sidecar_controller(mode="approved", disable_direct=False, max_bias=0.05)

    ledger, applied = controller._evaluate_sidecar_override(
        base_action="1",
        bridge_scores={"1": 0.0, "2": 0.8},
        bridge_public_support={"1": 0.0, "2": 0.9},
    )
    finalized = controller._finalize_sidecar_override_for_selection(
        ledger,
        chosen_action="2",
        pre_veto_chosen_action="2",
    )
    controller.pending_sidecar_override = finalized
    after = _obs_grid(1, np.zeros((8, 8), dtype=np.int64), ("1", "2"))

    controller._complete_pending_sidecar_override(
        "2",
        ArcAGI3StepResult(after, 1.0, False, False, {"events": ["level_completed"]}),
    )

    assert ledger["approved"] is True
    assert ledger["reason"] == "approved_by_public_usefulness"
    assert applied["2"] == pytest.approx(0.05)
    assert controller.sidecar_override_ledger[-1]["posthoc_useful"] is True
    assert controller.sidecar_override_ledger[-1]["posthoc_progress"] is True
    assert controller.sidecar_override_ledger[-1]["posthoc_entropy_drop"] == 0.0


def test_jepa_sidecar_quarantines_unuseful_approved_overrides() -> None:
    controller = _sidecar_controller(mode="approved", disable_direct=False, max_bias=0.05)
    controller.sidecar_override_ledger = [
        {
            "approved": True,
            "executed": True,
            "posthoc_useful": False,
        }
        for _ in range(jepa_arc_eval.SIDECAR_OVERRIDE_THRESHOLD + 1)
    ]

    ledger, applied = controller._evaluate_sidecar_override(
        base_action="1",
        bridge_scores={"1": 0.0, "2": 0.8},
        bridge_public_support={"1": 0.0, "2": 1.0},
    )

    assert ledger["approved"] is False
    assert ledger["reason"] == "sidecar_quarantined"
    assert ledger["quarantined"] is True
    assert ledger["sidecar_can_override"] is False
    assert ledger["action_weight"] == pytest.approx(0.005)
    assert all(value == 0.0 for value in applied.values())


def _write_behavior_trace(path: Path, actions: list[str], *, progress_indices: set[int] | None = None) -> None:
    progress_indices = set(progress_indices or set())
    steps = []
    for index, action in enumerate(actions):
        progress = index in progress_indices
        steps.append(
            {
                "step_index": index,
                "action": action,
                "score_delta": 1.0 if progress else -0.001,
                "event_delta": ["level_completed"] if progress else [],
                "diagnostics": {"jepa_policy": {}},
            }
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"attempt": {"steps": steps}}), encoding="utf-8")


def test_behavioral_offline_gates_detect_zero_useful_button_loops(tmp_path: Path) -> None:
    trace = tmp_path / "loop.json"
    _write_behavior_trace(trace, ["5", "7"] * 10)

    report = jepa_arc_eval.behavioral_offline_gates(
        [
            {
                "variant": "jepa_plus_attempt_memory",
                "useful_events": 0,
                "trace_path": str(trace),
            }
        ]
    )

    assert report["schema"] == "arc_offline_behavioral_gates_v1"
    assert report["passes"] is False
    assert report["gates"]["loop_attempt_rate"] is False
    assert report["gates"]["button_loop_attempt_rate"] is False
    assert report["gates"]["mean_useful_events_per_attempt"] is False
    assert report["gates"]["attempts_with_zero_useful_events"] is False
    assert report["primary_metrics"]["loop_attempt_rate"] > 0.05


def test_behavioral_offline_gates_pass_clean_progress_over_baseline(tmp_path: Path) -> None:
    baseline_trace = tmp_path / "baseline.json"
    primary_trace = tmp_path / "primary.json"
    _write_behavior_trace(baseline_trace, ["1", "2", "3", "4"])
    _write_behavior_trace(primary_trace, ["1", "2", "3", "4"], progress_indices={2})

    report = jepa_arc_eval.behavioral_offline_gates(
        [
            {
                "variant": "attempt_memory_no_jepa",
                "useful_events": 0,
                "trace_path": str(baseline_trace),
            },
            {
                "variant": "jepa_plus_attempt_memory",
                "useful_events": 1,
                "trace_path": str(primary_trace),
            },
        ]
    )

    assert report["passes"] is True
    assert all(report["gates"].values())
    assert report["baseline_available"] is True
    assert report["primary_metrics"]["mean_useful_events_per_attempt"] == pytest.approx(1.0)
    assert report["primary_metrics"]["progress_discovery_rate"] > report["baseline_metrics"]["progress_discovery_rate"]


def test_run_attempt_counts_event_and_level_progress_without_positive_reward(tmp_path: Path) -> None:
    class DummyController:
        def reset_attempt(self, seed: int) -> None:
            self.seed = seed

        def choose_action(self, observation: ArcAGI3Observation):
            del observation
            return "1", {"policy": {}, "jepa_policy": {}}

        def observe_transition(self, action: str, result: ArcAGI3StepResult) -> None:
            del action, result

        def finish_attempt(self, record) -> None:
            self.record = record

        def summary(self) -> dict[str, object]:
            return {}

    class ProgressEnv:
        max_steps = 2
        task_id = "event-progress"
        score = 0.0

        def __init__(self) -> None:
            self.step_index = 0

        def reset(self, seed: int) -> ArcAGI3Observation:
            del seed
            self.step_index = 0
            return _obs_grid(
                0,
                np.zeros((8, 8), dtype=np.int64),
                ("1",),
            )

        def step(self, action: str) -> ArcAGI3StepResult:
            del action
            self.step_index += 1
            if self.step_index == 1:
                after = _obs_grid(
                    1,
                    np.ones((8, 8), dtype=np.int64),
                    ("1",),
                )
                return ArcAGI3StepResult(after, 0.0, False, False, {"events": ["resource_collected"]})
            after = ArcAGI3Observation(
                task_id="test",
                episode_id="episode",
                step_index=2,
                grid=np.full((8, 8), 2, dtype=np.int64),
                available_actions=("1",),
                extras={"total_levels_completed": 1},
            )
            return ArcAGI3StepResult(after, -0.001, True, False, {"events": []})

        def close(self) -> dict[str, object]:
            return {}

        def normalized_score(self) -> float:
            return 0.0

    controller = DummyController()
    row = jepa_arc_eval.run_attempt(
        ProgressEnv(),
        controller,
        suite_id="unit",
        variant_id="jepa_plus_attempt_memory",
        split="sealed_eval",
        seed=3,
        attempt_index=1,
        trace_path=tmp_path / "trace.json",
    )
    trace = json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))
    steps = trace["attempt"]["steps"]

    assert row["useful_events"] == 2
    assert trace["summary"]["useful_events"] == 2
    assert jepa_arc_eval._step_has_progress_event(steps[0]) is True
    assert steps[0]["score_delta"] == 0.0


def test_run_attempt_counts_posthoc_public_usefulness_without_reward_progress(tmp_path: Path) -> None:
    class UsefulController:
        def reset_attempt(self, seed: int) -> None:
            self.seed = seed

        def choose_action(self, observation: ArcAGI3Observation):
            del observation
            return "right", {"policy": {}, "jepa_policy": {}}

        def observe_transition(self, action: str, result: ArcAGI3StepResult) -> dict[str, object]:
            del action, result
            return {
                "visible_effect": True,
                "useful_effect": True,
                "nuisance_effect": False,
                "progress_effect": False,
                "terminal_win": False,
                "controllability_effect": True,
                "reachable_state_class_effect": True,
                "reasons": ["publicly_supported_controllability"],
            }

        def finish_attempt(self, record) -> None:
            self.record = record

        def summary(self) -> dict[str, object]:
            return {}

    class UsefulEnv:
        max_steps = 1
        task_id = "public-useful"
        score = 0.0

        def reset(self, seed: int) -> ArcAGI3Observation:
            del seed
            grid = np.zeros((8, 8), dtype=np.int64)
            grid[2, 2] = 4
            return _obs_grid(0, grid, ("right",))

        def step(self, action: str) -> ArcAGI3StepResult:
            del action
            grid = np.zeros((8, 8), dtype=np.int64)
            grid[2, 3] = 4
            return ArcAGI3StepResult(_obs_grid(1, grid, ("right",)), -0.001, False, False, {"events": ["move"]})

        def close(self) -> dict[str, object]:
            return {}

        def normalized_score(self) -> float:
            return 0.0

    row = jepa_arc_eval.run_attempt(
        UsefulEnv(),
        UsefulController(),
        suite_id="unit",
        variant_id="jepa_plus_attempt_memory",
        split="sealed_eval",
        seed=5,
        attempt_index=1,
        trace_path=tmp_path / "trace.json",
    )
    trace = json.loads((tmp_path / "trace.json").read_text(encoding="utf-8"))
    step = trace["attempt"]["steps"][0]
    policy = step["diagnostics"]["jepa_policy"]

    assert row["useful_events"] == 1
    assert trace["summary"]["useful_events"] == 1
    assert policy["posthoc_useful"] is True
    assert policy["posthoc_progress"] is False
    assert jepa_arc_eval._step_has_progress_event(step) is False
    assert jepa_arc_eval._step_has_useful_event(step) is True


def test_component_causal_graph_detects_public_component_movement() -> None:
    before_grid = np.zeros((8, 8), dtype=np.int64)
    before_grid[2, 2:4] = 4
    after_grid = np.zeros((8, 8), dtype=np.int64)
    after_grid[2, 3:5] = 4
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=9, attempt_index=1)
    before = _obs_grid(0, before_grid, ("right", "left", "wait"))
    after = _obs_grid(1, after_grid, ("right", "left", "wait"))
    buffer.append_transition(before, "right", ArcAGI3StepResult(after, -0.001, False, False, {"events": ["move"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    entry = memory.ingest_attempt(buffer.to_record())

    assert entry.component_causal_hypotheses
    assert entry.component_causal_hypotheses[0]["mechanism"] == "component_movement"
    assert entry.component_causal_hypotheses[0]["moved_components"][0]["value"] == 4
    assert entry.causal_hypotheses["component_relation_count"] >= 1.0
    assert memory.summary()["object_memory"]["component_values_seen"] == 1


def test_component_relation_memory_scores_component_clicks() -> None:
    before_grid = np.zeros((8, 8), dtype=np.int64)
    before_grid[2, 2:4] = 5
    after_grid = before_grid.copy()
    after_grid[3, 2:4] = 6
    actions = ("click:20:20", "click:56:56", "wait")
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=10, attempt_index=1)
    before = _obs_grid(0, before_grid, actions)
    after = _obs_grid(1, after_grid, actions)
    buffer.append_transition(before, "click:20:20", ArcAGI3StepResult(after, 1.0, False, False, {"events": ["useful_click"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    entry = memory.ingest_attempt(buffer.to_record())
    scores = memory.plan_scores_for_observation(before)

    assert entry.component_causal_hypotheses[0]["target_relation"]["kind"] == "on_component"
    assert entry.causal_hypotheses["component_goal_relation_count"] >= 1.0
    assert scores["click:20:20"] > scores["click:56:56"]
    assert scores["click:20:20"] > scores["wait"]


def test_component_transition_prediction_scores_expected_public_transform() -> None:
    before_grid = np.zeros((8, 8), dtype=np.int64)
    before_grid[2, 2:4] = 5
    after_grid = np.zeros((8, 8), dtype=np.int64)
    after_grid[2, 2:4] = 6
    actions = ("click:20:20", "click:56:56", "wait")
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=12, attempt_index=1)
    before = _obs_grid(0, before_grid, actions)
    after = _obs_grid(1, after_grid, actions)
    buffer.append_transition(before, "click:20:20", ArcAGI3StepResult(after, 1.0, False, False, {"events": ["useful_click"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    entry = memory.ingest_attempt(buffer.to_record())

    for prior_entry in memory.entries:
        prior_entry.next_attempt_plan.clear()
    memory.region_values.clear()
    memory.family_region_values.clear()
    memory.color_values.clear()
    memory.component_relation_values.clear()
    memory.family_component_values.clear()
    memory.component_goal_relations.clear()
    memory.component_value_scores.clear()
    scores = memory.plan_scores_for_observation(before)

    assert entry.causal_hypotheses["component_transition_prediction_count"] >= 1.0
    assert memory.summary()["object_memory"]["component_transition_prediction_count"] >= 1
    assert "predicted_component_transition_planner" in memory.summary()["planner_chain"]
    assert scores["click:20:20"] > scores["click:56:56"]
    assert scores["click:20:20"] > scores["wait"]


def test_component_transition_goal_chain_activates_state_grounded_plan() -> None:
    start_grid = np.zeros((8, 8), dtype=np.int64)
    start_grid[2, 2] = 5
    moved_grid = np.zeros((8, 8), dtype=np.int64)
    moved_grid[2, 3] = 5
    event_grid = np.zeros((8, 8), dtype=np.int64)
    event_grid[2, 3] = 6
    actions = ("right", "click:28:20", "wait")
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=13, attempt_index=1)
    start = _obs_grid(0, start_grid, actions)
    moved = _obs_grid(1, moved_grid, actions)
    event_after = _obs_grid(2, event_grid, actions)
    buffer.append_transition(start, "right", ArcAGI3StepResult(moved, -0.001, False, False, {"events": ["move"]}))
    buffer.append_transition(moved, "click:28:20", ArcAGI3StepResult(event_after, 1.0, False, False, {"events": ["useful_click"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    entry = memory.ingest_attempt(buffer.to_record())

    scores = memory.plan_scores_for_observation(start)
    summary = memory.summary()

    assert entry.causal_hypotheses["component_chain_edge_count"] >= 2.0
    assert entry.causal_hypotheses["component_chain_goal_edge_count"] >= 2.0
    assert summary["object_memory"]["component_chain_edge_count"] >= 2
    assert summary["object_memory"]["active_sequence_source"] == "component_transition_goal_chain"
    assert "component_transition_goal_chain_search" in summary["planner_chain"]
    assert scores["right"] > scores["wait"]
    assert memory.sequence_plan_action(actions) == "right"
    memory.observe_live_transition(
        start,
        "right",
        ArcAGI3StepResult(moved, -0.001, False, False, {"events": ["move"]}),
    )
    assert memory.sequence_plan_action(actions) == "click:28:20"
    moved_scores = memory.plan_scores_for_observation(moved)
    assert moved_scores["click:28:20"] > moved_scores["wait"]


def test_component_transition_goal_chain_aborts_on_postcondition_contradiction() -> None:
    start_grid = np.zeros((8, 8), dtype=np.int64)
    start_grid[2, 2] = 5
    moved_grid = np.zeros((8, 8), dtype=np.int64)
    moved_grid[2, 3] = 5
    event_grid = np.zeros((8, 8), dtype=np.int64)
    event_grid[2, 3] = 6
    actions = ("right", "click:28:20", "wait")
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=14, attempt_index=1)
    start = _obs_grid(0, start_grid, actions)
    moved = _obs_grid(1, moved_grid, actions)
    event_after = _obs_grid(2, event_grid, actions)
    buffer.append_transition(start, "right", ArcAGI3StepResult(moved, -0.001, False, False, {"events": ["move"]}))
    buffer.append_transition(moved, "click:28:20", ArcAGI3StepResult(event_after, 1.0, False, False, {"events": ["useful_click"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    memory.ingest_attempt(buffer.to_record())
    scores = memory.plan_scores_for_observation(start)

    assert scores["right"] > scores["wait"]
    assert memory.sequence_plan_action(actions) == "right"
    memory.observe_live_transition(
        start,
        "right",
        ArcAGI3StepResult(start, -0.001, False, False, {"events": ["no_effect"]}),
    )
    summary = memory.summary()["object_memory"]
    assert summary["sequence_contradictions"] == 1
    assert summary["component_chain_contradictions"] >= 1
    assert summary["active_sequence_remaining"] == 0


def test_component_relation_goal_chain_generalizes_contact_across_positions() -> None:
    train_grid = np.zeros((8, 8), dtype=np.int64)
    train_grid[2, 2] = 5
    train_after_grid = np.zeros((8, 8), dtype=np.int64)
    train_after_grid[2, 2] = 6
    train_actions = ("click:20:20", "click:4:4", "wait")
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=15, attempt_index=1)
    train_before = _obs_grid(0, train_grid, train_actions)
    train_after = _obs_grid(1, train_after_grid, train_actions)
    buffer.append_transition(
        train_before,
        "click:20:20",
        ArcAGI3StepResult(train_after, 1.0, False, False, {"events": ["useful_click"]}),
    )
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    entry = memory.ingest_attempt(buffer.to_record())
    _clear_non_relation_chain_memory(memory)

    test_grid = np.zeros((8, 8), dtype=np.int64)
    test_grid[5, 5] = 5
    test_actions = ("click:44:44", "click:4:4", "wait")
    test_observation = _obs_grid(0, test_grid, test_actions)
    scores = memory.plan_scores_for_observation(test_observation)
    summary = memory.summary()

    assert entry.causal_hypotheses["component_relation_chain_edge_count"] >= 1.0
    assert summary["object_memory"]["component_relation_chain_edge_count"] >= 1
    assert summary["object_memory"]["active_sequence_source"] == "component_relation_goal_chain"
    assert summary["planner_activation"]["relation_chain"]["activations"] >= 1
    assert summary["planner_activation"]["relation_chain"]["legal_resolutions"] >= 1
    assert "component_relation_goal_chain_search" in summary["planner_chain"]
    assert scores["click:44:44"] > scores["click:4:4"]
    assert scores["click:44:44"] > scores["wait"]


def test_component_relation_goal_chain_penalizes_relation_postcondition_contradiction() -> None:
    train_grid = np.zeros((8, 8), dtype=np.int64)
    train_grid[2, 2] = 5
    train_after_grid = np.zeros((8, 8), dtype=np.int64)
    train_after_grid[2, 2] = 6
    train_actions = ("click:20:20", "wait")
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=16, attempt_index=1)
    train_before = _obs_grid(0, train_grid, train_actions)
    train_after = _obs_grid(1, train_after_grid, train_actions)
    buffer.append_transition(
        train_before,
        "click:20:20",
        ArcAGI3StepResult(train_after, 1.0, False, False, {"events": ["useful_click"]}),
    )
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    memory.ingest_attempt(buffer.to_record())
    _clear_non_relation_chain_memory(memory)

    test_grid = np.zeros((8, 8), dtype=np.int64)
    test_grid[5, 5] = 5
    test_actions = ("click:44:44", "click:4:4", "wait")
    test_observation = _obs_grid(0, test_grid, test_actions)
    scores = memory.plan_scores_for_observation(test_observation)

    assert scores["click:44:44"] > scores["wait"]
    assert memory.sequence_plan_action(test_actions) == "click:44:44"
    memory.observe_live_transition(
        test_observation,
        "click:44:44",
        ArcAGI3StepResult(test_observation, -0.001, False, False, {"events": ["no_effect"]}),
    )
    summary = memory.summary()["object_memory"]
    assert summary["sequence_contradictions"] == 1
    assert summary["component_relation_chain_contradictions"] >= 1
    assert summary["planner_activation"]["relation_chain"]["contradiction_aborts"] >= 1
    assert summary["planner_activation"]["relation_chain"]["stale_penalties"] >= 1
    assert summary["active_sequence_remaining"] == 0


def test_component_relation_delta_event_miner_scores_shifted_movement() -> None:
    train_grid = np.zeros((8, 8), dtype=np.int64)
    train_grid[2, 2] = 5
    train_after_grid = np.zeros((8, 8), dtype=np.int64)
    train_after_grid[2, 3] = 5
    train_actions = ("click:28:20", "click:4:4", "wait")
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=17, attempt_index=1)
    train_before = _obs_grid(0, train_grid, train_actions)
    train_after = _obs_grid(1, train_after_grid, train_actions)
    buffer.append_transition(
        train_before,
        "click:28:20",
        ArcAGI3StepResult(train_after, 1.0, False, False, {"events": ["useful_click"]}),
    )
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    entry = memory.ingest_attempt(buffer.to_record())
    _clear_non_relation_delta_memory(memory)

    test_grid = np.zeros((8, 8), dtype=np.int64)
    test_grid[5, 5] = 5
    test_actions = ("click:52:44", "click:4:4", "wait")
    test_observation = _obs_grid(0, test_grid, test_actions)
    scores = memory.plan_scores_for_observation(test_observation)
    summary = memory.summary()

    assert entry.causal_hypotheses["component_relation_delta_count"] >= 1.0
    assert entry.causal_hypotheses["component_relation_delta_goal_count"] >= 1.0
    assert summary["object_memory"]["component_relation_delta_count"] >= 1
    assert summary["planner_activation"]["relation_delta"]["score_hits"] >= 1
    assert summary["planner_activation"]["relation_delta"]["legal_resolutions"] >= 1
    assert "component_relation_delta_event_miner" in summary["planner_chain"]
    assert scores["click:52:44"] > scores["click:4:4"]
    assert scores["click:52:44"] > scores["wait"]


def test_component_relation_delta_contradiction_records_failed_postcondition() -> None:
    train_grid = np.zeros((8, 8), dtype=np.int64)
    train_grid[2, 2] = 5
    train_after_grid = np.zeros((8, 8), dtype=np.int64)
    train_after_grid[2, 2] = 6
    train_actions = ("click:20:20", "wait")
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=18, attempt_index=1)
    train_before = _obs_grid(0, train_grid, train_actions)
    train_after = _obs_grid(1, train_after_grid, train_actions)
    buffer.append_transition(
        train_before,
        "click:20:20",
        ArcAGI3StepResult(train_after, 1.0, False, False, {"events": ["useful_click"]}),
    )
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    memory.ingest_attempt(buffer.to_record())

    test_grid = np.zeros((8, 8), dtype=np.int64)
    test_grid[5, 5] = 5
    test_actions = ("click:44:44", "click:4:4", "wait")
    test_observation = _obs_grid(0, test_grid, test_actions)
    scores = memory.plan_scores_for_observation(test_observation)

    assert scores["click:44:44"] > scores["wait"]
    assert memory.sequence_plan_action(test_actions) == "click:44:44"
    memory.observe_live_transition(
        test_observation,
        "click:44:44",
        ArcAGI3StepResult(test_observation, -0.001, False, False, {"events": ["no_effect"]}),
    )
    summary = memory.summary()["object_memory"]
    assert summary["sequence_contradictions"] == 1
    assert summary["component_relation_delta_contradictions"] >= 1
    assert summary["active_sequence_remaining"] == 0


def test_relation_delta_sequence_resolves_shifted_multistep_plan() -> None:
    train_start_grid = np.zeros((8, 8), dtype=np.int64)
    train_start_grid[2, 2] = 5
    train_moved_grid = np.zeros((8, 8), dtype=np.int64)
    train_moved_grid[2, 3] = 5
    train_event_grid = np.zeros((8, 8), dtype=np.int64)
    train_event_grid[2, 3] = 6
    train_actions = ("click:28:20", "click:4:4", "wait")
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=19, attempt_index=1)
    train_start = _obs_grid(0, train_start_grid, train_actions)
    train_moved = _obs_grid(1, train_moved_grid, train_actions)
    train_event = _obs_grid(2, train_event_grid, train_actions)
    buffer.append_transition(
        train_start,
        "click:28:20",
        ArcAGI3StepResult(train_moved, -0.001, False, False, {"events": ["move"]}),
    )
    buffer.append_transition(
        train_moved,
        "click:28:20",
        ArcAGI3StepResult(train_event, 1.0, False, False, {"events": ["useful_click"]}),
    )
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    entry = memory.ingest_attempt(buffer.to_record())
    _clear_non_sequence_candidate_memory(memory)
    memory.start_attempt()

    shifted_start_grid = np.zeros((8, 8), dtype=np.int64)
    shifted_start_grid[5, 5] = 5
    shifted_moved_grid = np.zeros((8, 8), dtype=np.int64)
    shifted_moved_grid[5, 6] = 5
    shifted_actions = ("click:52:44", "click:4:4", "wait")
    shifted_start = _obs_grid(0, shifted_start_grid, shifted_actions)
    shifted_moved = _obs_grid(1, shifted_moved_grid, shifted_actions)
    scores = memory.plan_scores_for_observation(shifted_start)
    summary = memory.summary()["object_memory"]

    assert entry.sequence_plan_summary["relation_delta_sequence_candidate_count"] >= 1
    assert summary["active_sequence_source"] == "positive_public_relation_delta_sequence"
    assert memory.summary()["planner_activation"]["relation_delta_sequence"]["activations"] >= 1
    assert memory.summary()["planner_activation"]["relation_delta_sequence"]["legal_resolutions"] >= 1
    assert "relation_delta_sequence_planner" in memory.summary()["planner_chain"]
    assert memory.sequence_plan_action(shifted_actions, observation=shifted_start) == "click:52:44"
    assert scores["click:52:44"] > scores["click:4:4"]
    assert scores["click:52:44"] > scores["wait"]
    memory.observe_live_transition(
        shifted_start,
        "click:52:44",
        ArcAGI3StepResult(shifted_moved, -0.001, False, False, {"events": ["move"]}),
    )
    assert memory.summary()["planner_activation"]["relation_delta_sequence"]["visible_changes"] >= 1
    assert memory.sequence_plan_action(shifted_actions, observation=shifted_moved) == "click:52:44"
    moved_scores = memory.plan_scores_for_observation(shifted_moved)
    assert moved_scores["click:52:44"] > moved_scores["wait"]


def test_relation_delta_sequence_aborts_shifted_plan_on_delta_contradiction() -> None:
    train_start_grid = np.zeros((8, 8), dtype=np.int64)
    train_start_grid[2, 2] = 5
    train_moved_grid = np.zeros((8, 8), dtype=np.int64)
    train_moved_grid[2, 3] = 5
    train_event_grid = np.zeros((8, 8), dtype=np.int64)
    train_event_grid[2, 3] = 6
    train_actions = ("click:28:20", "click:4:4", "wait")
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=20, attempt_index=1)
    train_start = _obs_grid(0, train_start_grid, train_actions)
    train_moved = _obs_grid(1, train_moved_grid, train_actions)
    train_event = _obs_grid(2, train_event_grid, train_actions)
    buffer.append_transition(
        train_start,
        "click:28:20",
        ArcAGI3StepResult(train_moved, -0.001, False, False, {"events": ["move"]}),
    )
    buffer.append_transition(
        train_moved,
        "click:28:20",
        ArcAGI3StepResult(train_event, 1.0, False, False, {"events": ["useful_click"]}),
    )
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    memory.ingest_attempt(buffer.to_record())
    _clear_non_sequence_candidate_memory(memory)
    memory.start_attempt()

    shifted_start_grid = np.zeros((8, 8), dtype=np.int64)
    shifted_start_grid[5, 5] = 5
    shifted_actions = ("click:52:44", "click:4:4", "wait")
    shifted_start = _obs_grid(0, shifted_start_grid, shifted_actions)
    scores = memory.plan_scores_for_observation(shifted_start)

    assert scores["click:52:44"] > scores["wait"]
    memory.observe_live_transition(
        shifted_start,
        "click:52:44",
        ArcAGI3StepResult(shifted_start, -0.001, False, False, {"events": ["no_effect"]}),
    )
    summary = memory.summary()["object_memory"]
    assert summary["sequence_contradictions"] == 1
    assert summary["component_relation_delta_contradictions"] >= 1
    assert summary["planner_activation"]["relation_delta_sequence"]["contradiction_aborts"] >= 1
    assert summary["planner_activation"]["relation_delta_sequence"]["stale_penalties"] >= 1
    assert summary["active_sequence_remaining"] == 0


def test_component_grounded_sequence_aborts_on_public_contradiction() -> None:
    before_grid = np.zeros((8, 8), dtype=np.int64)
    before_grid[2, 2] = 5
    event_grid = before_grid.copy()
    event_grid[2, 3] = 6
    actions = ("click:20:20", "click:56:56", "wait")
    buffer = AttemptBuffer(suite_id="suite", task_id="task", variant="v", split="dev", seed=11, attempt_index=1)
    before = _obs_grid(0, before_grid, actions)
    event_after = _obs_grid(1, event_grid, actions)
    buffer.append_transition(before, "click:20:20", ArcAGI3StepResult(event_after, 1.0, False, False, {"events": ["useful_click"]}))
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    memory.ingest_attempt(buffer.to_record())
    memory.start_attempt()

    assert memory.sequence_plan_action(actions) == "click:20:20"
    no_change = _obs_grid(1, before_grid.copy(), actions)
    memory.observe_live_transition(
        before,
        "click:20:20",
        ArcAGI3StepResult(no_change, -0.001, False, False, {"events": ["no_effect"]}),
    )
    summary = memory.summary()["object_memory"]
    assert summary["sequence_contradictions"] == 1
    assert summary["component_transition_contradictions"] >= 1
    assert summary["active_sequence_remaining"] == 0


def test_component_scoring_caches_public_components_per_observation(monkeypatch) -> None:
    grid = np.zeros((8, 8), dtype=np.int64)
    grid[1, 1] = 2
    grid[5, 5] = 3
    actions = ("click:8:8", "click:16:16", "click:40:40", "click:56:56", "wait")
    observation = _obs_grid(0, grid, actions)
    memory = JEPAAttemptMemory(use_jepa_tokens=False)
    original = jepa_attempt_memory._frame_components
    calls = {"count": 0}

    def counted_components(frame: np.ndarray):
        calls["count"] += 1
        return original(frame)

    monkeypatch.setattr(jepa_attempt_memory, "_frame_components", counted_components)
    scores = memory.plan_scores_for_observation(observation)

    assert set(scores) == set(actions)
    assert calls["count"] == 1


def test_jepa_temporal_representation_records_read_only_sidecar_priors() -> None:
    records = synthetic_attempts(8, seed=50)
    model = VideoJEPA()
    memory_jepa = JEPAAttemptMemory(use_jepa_tokens=True)
    memory_no_jepa = JEPAAttemptMemory(use_jepa_tokens=False)
    jepa_entry = memory_jepa.ingest_attempt(records[0], model=model)
    no_jepa_entry = memory_no_jepa.ingest_attempt(records[0], model=None)
    legal_actions = tuple(records[0].steps[0].legal_actions)
    jepa_distribution = memory_jepa.action_distribution(legal_actions)
    no_jepa_distribution = memory_no_jepa.action_distribution(legal_actions)
    distribution_l1 = sum(abs(jepa_distribution[action] - no_jepa_distribution[action]) for action in legal_actions)
    plan_l1 = sum(
        abs(float(jepa_entry.next_attempt_plan.get(action, 0.0)) - float(no_jepa_entry.next_attempt_plan.get(action, 0.0)))
        for action in set(jepa_entry.next_attempt_plan) | set(no_jepa_entry.next_attempt_plan)
    )
    proposal_l1 = sum(abs(float(value)) for value in jepa_entry.sidecar_action_priors.values())

    assert jepa_entry.causal_substrate_active is True
    assert jepa_entry.token_mean
    assert jepa_entry.jepa_action_evidence
    assert jepa_entry.causal_hypotheses["jepa_causal_substrate_active"] == 1.0
    assert jepa_entry.causal_hypotheses["jepa_action_effect_mean"] > 0.0
    assert proposal_l1 > 0.0
    assert plan_l1 <= 1.0e-9
    assert distribution_l1 <= 1.0e-9
    assert jepa_entry.jepa_planner_state["planner_role"] == "proposer_read_only"
    assert jepa_entry.jepa_planner_state["consumed_by_planner"] is False
    assert jepa_entry.jepa_planner_state["action_priors"]
    assert memory_jepa.summary()["causal_substrate_active"] is True
    assert memory_jepa.summary()["direct_action_source"] is False


def test_shuffled_jepa_tokens_have_explicit_control_state() -> None:
    records = synthetic_attempts(8, seed=51)
    model = VideoJEPA()
    memory_trained = JEPAAttemptMemory(use_jepa_tokens=True)
    memory_shuffled = JEPAAttemptMemory(use_jepa_tokens=True, jepa_token_mode="shuffled")
    trained_entry = memory_trained.ingest_attempt(records[0], model=model)
    shuffled_entry = memory_shuffled.ingest_attempt(records[0], model=model)

    assert trained_entry.jepa_planner_state["mode"] == "normal"
    assert shuffled_entry.jepa_planner_state["mode"] == "shuffled"
    assert trained_entry.jepa_planner_state["consumed_by_planner"] is False
    assert shuffled_entry.jepa_planner_state["consumed_by_planner"] is False
    assert trained_entry.jepa_planner_state["action_priors"]
    assert shuffled_entry.jepa_planner_state["action_priors"]
    assert shuffled_entry.jepa_action_evidence
    assert shuffled_entry.jepa_planner_state != trained_entry.jepa_planner_state
    assert memory_shuffled.summary()["jepa_token_mode"] == "shuffled"


def test_minimal_report_schema_for_audit(tmp_path: Path) -> None:
    report = {
        "terminal_outcome": "NO IMPROVEMENT FOUND",
        "variant_ids": [
            "baseline_core",
            "attempt_memory_no_jepa",
            "jepa_random_init",
            "jepa_pretrained_frozen_if_available",
            "jepa_trained_dev",
            "jepa_plus_attempt_memory",
            "null_control",
        ],
        "gates": {"jepa_beats_null_on_dev": True, "non_arc_drop_within_limit": True},
        "no_hack_proof": {"passes": True, "jepa_emits_text": False},
        "official": {"attempt_table": []},
        "non_arc": {"aggregate_by_variant": {}},
        "trace_paths": [],
    }
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    assert json.loads(path.read_text())["terminal_outcome"] == "NO IMPROVEMENT FOUND"
