from __future__ import annotations

import json
from pathlib import Path

from src.external_eval import build_claim_registry, supported_claim_has_evidence
from src.external_registry import CLAIMS, claim_registry_template, discover_external_suites
from src.generalization_audit import claim_discipline_checks, trace_checks


def test_external_registry_contains_required_claims_and_arc_suite() -> None:
    claims = claim_registry_template()
    assert {row["claim_id"] for row in claims} == {claim.claim_id for claim in CLAIMS}
    assert all(row["external_prediction"] == row["external_behavioral_prediction"] for row in claims)
    suites = discover_external_suites()
    assert any(suite.suite_id == "official_arcagi3" for suite in suites)
    assert all(suite.generated_by_repo is False for suite in suites)


def test_supported_claim_requires_external_margin_or_ablation_drop() -> None:
    unsupported = {
        "claim_id": "memory",
        "status": "supported",
        "result": {"best_external_margin": 0.01, "max_external_ablation_drop": 0.05},
    }
    supported = {
        "claim_id": "memory",
        "status": "supported",
        "result": {"best_external_margin": -0.01, "max_external_ablation_drop": 0.12},
    }
    assert supported_claim_has_evidence(unsupported) is False
    assert supported_claim_has_evidence(supported) is True


def test_claim_registry_preserves_unsupported_negative_results() -> None:
    suite_report = {
        "suite": {"suite_id": "external_suite", "split": "sealed_eval"},
        "aggregate_by_controller": {
            "explorer": {"mean_normalized_score": 0.0},
            "random_legal": {"mean_normalized_score": 0.2},
            "repeat_last_action": {"mean_normalized_score": 0.1},
            "coverage_graph_exploration": {"mean_normalized_score": 0.3},
            "novelty_first": {"mean_normalized_score": 0.2},
            "greedy_observable_score_delta": {"mean_normalized_score": 0.0},
            "oracle_free_observed_graph_bfs": {"mean_normalized_score": 0.2},
            "ablation_corrupt_memory": {"mean_normalized_score": 0.0},
            "ablation_corrupt_drive": {"mean_normalized_score": 0.0},
            "ablation_disable_planner_imagination": {"mean_normalized_score": 0.0},
        },
    }
    rows = build_claim_registry([suite_report])
    assert rows
    assert all(row["status"] == "unsupported" for row in rows)
    checks = claim_discipline_checks({"claim_registry": rows})
    assert checks["claim_registry_complete"] is True
    assert checks["unsupported_claims_preserved"] is True


def test_trace_checks_require_external_trace_schema(tmp_path: Path) -> None:
    trace = tmp_path / "trace.json"
    trace.write_text(
        json.dumps(
            {
                "steps": [
                    {
                        "legal_actions": ["1"],
                        "chosen_action": "1",
                        "baseline_actions": {"random_legal": "1"},
                        "score_delta": 0.0,
                        "event_delta": [],
                        "memory_drive_hypothesis": {},
                        "failure_class": "no_external_reward",
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    checks = trace_checks({"trace_paths": [str(trace)]})
    assert checks["passes"] is True
