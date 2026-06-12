from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.arcagi3_adapter import ArcAGI3Observation, ArcAGI3StepResult
from src.attempt_buffer import AttemptBuffer, tensors_from_attempts
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
        "changed_next_attempt_action_distribution",
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
    assert memory.score_action("down") > 0.0


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
    assert scores["wait"] < scores["down"]


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

    assert entry.sequence_plan_summary["sequence_candidate_count"] == 1
    memory.start_attempt()
    first_scores = memory.plan_scores_for_observation(start)
    assert first_scores["down"] > first_scores["wait"]
    memory.advance_sequence("down")
    second_scores = memory.plan_scores_for_observation(moved)
    assert second_scores["up"] > second_scores["wait"]


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


def test_jepa_temporal_representation_changes_next_attempt_distribution() -> None:
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

    assert jepa_entry.causal_substrate_active is True
    assert jepa_entry.token_mean
    assert jepa_entry.jepa_action_evidence
    assert jepa_entry.causal_hypotheses["jepa_causal_substrate_active"] == 1.0
    assert jepa_entry.causal_hypotheses["jepa_action_effect_mean"] > 0.0
    assert plan_l1 > 0.0
    assert distribution_l1 > 0.0
    assert memory_jepa.summary()["causal_substrate_active"] is True
    assert memory_jepa.summary()["direct_action_source"] is False


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
