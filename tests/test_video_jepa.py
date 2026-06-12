from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch

from src.arcagi3_adapter import ArcAGI3Observation, ArcAGI3StepResult
from src.attempt_buffer import AttemptBuffer, tensors_from_attempts
from src.jepa_attempt_memory import JEPAAttemptMemory
from src.jepa_train import synthetic_attempts, train_jepa
from src.video_jepa import VideoJEPA, jepa_loss, null_future_loss


def _obs(step: int, y: int, x: int) -> ArcAGI3Observation:
    grid = np.zeros((8, 8), dtype=np.int64)
    grid[y, x] = 2
    return ArcAGI3Observation(
        task_id="test",
        episode_id="episode",
        step_index=step,
        grid=grid,
        available_actions=("up", "down", "wait"),
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
