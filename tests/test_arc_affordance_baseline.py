from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.arc_affordance_baseline import AFFORDANCE_VARIANTS, ArcAffordancePolicy, build_affordance_variants
from src.arc_affordance_report import build_report, no_hack_proof
from src.arc_affordance_search import changed_regions, component_summary
from src.arcagi3_adapter import ArcAGI3Observation, ArcAGI3StepResult


def make_observation(step: int = 0) -> ArcAGI3Observation:
    grid = np.zeros((8, 8), dtype=np.int64)
    grid[4, 4] = 2
    grid[1:3, 1:3] = 7
    grid[6, 6] = 5
    return ArcAGI3Observation(
        task_id="unit/affordance",
        episode_id="unit/affordance/0",
        step_index=step,
        grid=grid,
        available_actions=("1", "2", "3", "4", "click:1:1", "click:6:6", "wait"),
        extras={"score": 0.0},
    )


def test_component_summary_extracts_public_regions() -> None:
    summary = component_summary(make_observation())
    assert summary["component_count"] >= 2
    assert summary["largest"][0]["area"] >= 1
    assert "obs_hash" in summary


def test_affordance_variants_choose_legal_actions() -> None:
    obs = make_observation()
    policies = build_affordance_variants(include_ablations=True)
    assert {policy.name for policy in policies}.issuperset({variant.variant_id for variant in AFFORDANCE_VARIANTS})
    for policy in policies:
        policy.reset(0)
        action, diagnostics = policy.choose_action(obs)
        assert action in obs.available_actions
        assert diagnostics["policy"]["forced_cycle"] is False
        assert diagnostics["policy"]["choice_reason"]["enabled_features"]


def test_effect_memory_records_change_and_noop_avoidance() -> None:
    before = make_observation()
    after_grid = before.grid.copy()
    after_grid[1, 1] = 0
    after = ArcAGI3Observation(
        task_id=before.task_id,
        episode_id=before.episode_id,
        step_index=1,
        grid=after_grid,
        available_actions=before.available_actions,
        extras={"score": 1.0},
    )
    result = ArcAGI3StepResult(after, 1.0, False, False, {"events": ["level_completed"], "score": 1.0})
    policy = ArcAffordancePolicy("combined_affordance_search")
    policy.reset(0)
    action, _ = policy.choose_action(before)
    policy.observe(before, action, result)
    assert changed_regions(before, after)["pixel_count"] == 1
    snapshot = policy.search.effect.snapshot()
    assert snapshot["observed_actions"] == 1
    assert snapshot["action_effect_hit_rate"] >= 0.0


def test_report_gate_and_no_hack_proof(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(json.dumps({"steps": []}), encoding="utf-8")
    rows = []
    for variant, score, useful in [
        ("coverage_graph_exploration", 0.004, 0.08),
        ("old_explorer", 0.0, 0.0),
        ("combined_affordance_search", 0.01, 0.17),
        ("ablation_no_component_memory", 0.0, 0.0),
    ]:
        rows.append(
            {
                "game": "unit-game",
                "variant": variant,
                "solved": False,
                "score": score,
                "normalized_score": score,
                "useful_events": useful,
                "steps": 10,
                "action_entropy": 1.0,
                "repeat_collapse": 0.4,
                "invalid_action_rate": 0.0,
                "unique_states": 5,
                "trace_path": str(trace),
            }
        )
    report = build_report(
        rows=rows,
        trace_paths=[str(trace)],
        runtime_status={"official_runtime_available": False},
        trace_dir=tmp_path,
        config="unit",
    )
    assert report["terminal_outcome"] == "AFFORDANCE BASELINE SIGNAL FOUND"
    assert report["improvement_gate"]["passes"] is True
    proof = no_hack_proof()
    assert proof["passes"] is True
    assert proof["no_neural_training"] is True
