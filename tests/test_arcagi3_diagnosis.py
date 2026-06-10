from __future__ import annotations

import json
from pathlib import Path

from src.arcagi3_failure_taxonomy import classify_failure, no_hack_audit
from src.arcagi3_trace_analysis import enrich_trace, required_trace_fields_present


def _step(index: int, action: str = "3") -> dict[str, object]:
    obs = {
        "task_id": "arcagi3-official/test-game",
        "episode_id": "arcagi3-official/test-game/episode_0",
        "step_index": index,
        "grid": [".G.", ".A.", ".R."],
        "available_actions": ["1", "2", "3", "4", "click:1:0"],
        "extras": {"fixture_id": "test-game", "score": 0.0, "normalized_score": 0.0, "game_state": "NOT_FINISHED"},
    }
    next_obs = dict(obs)
    next_obs["step_index"] = index + 1
    return {
        "step": index,
        "obs": obs,
        "action": action,
        "next_obs": next_obs,
        "event_delta": [],
        "score_delta": -0.001,
        "score": 0.0,
        "normalized_score": 0.0,
        "memory_recall": {"visited_cells": index + 1, "goal": [[0, 1]], "resource": [[2, 1]]},
        "drive": {"novelty": 1.0, "hazard_level": 0, "resource_level": 1, "body_energy": 1.0},
        "hypothesis_state": {"semantic_action": 5, "target": [0, 1], "confidence": 0.95},
        "policy": {"semantic_action": 5},
    }


def _trace() -> dict[str, object]:
    return {
        "metadata": {
            "controller": "explorer_official_normal",
            "task_id": "arcagi3-official/test-game",
            "episode_id": "arcagi3-official/test-game/episode_0",
            "fixture_source": "official_arcagi3_runtime",
            "game_id": "test-game",
        },
        "summary": {
            "solved": False,
            "score": 0.0,
            "normalized_score": 0.0,
            "steps": 4,
            "invalid_action_rate": 0.0,
            "unique_states": 1,
            "useful_events": 0,
            "action_entropy": 0.0,
            "repeat_collapse": 1.0,
            "official_state": "NOT_FINISHED",
            "levels_completed": 0,
            "win_levels": 3,
            "official_scorecard": {"completed": False, "levels_completed": 0},
        },
        "steps": [_step(i) for i in range(4)],
    }


def test_trace_enrichment_adds_required_diagnosis_fields() -> None:
    enriched = enrich_trace(_trace(), seed=0)
    assert enriched["diagnosis"]["required_trace_fields_present"] is True
    assert required_trace_fields_present(enriched["steps"]) is True
    first = enriched["steps"][0]["diagnosis"]
    assert first["baseline_actions"]["random_legal"] in first["legal_actions"]
    assert first["extracted_object_entity_features"]["cell_counts"]["agent"] == 1
    assert "z_memory_drive_hypothesis_summary" in first


def test_taxonomy_classifies_repeated_trace_as_cycle() -> None:
    enriched = enrich_trace(_trace(), seed=0)
    game = {
        "summary": enriched["summary"],
        "audits": enriched["diagnosis"]["audits"],
        "tags": ["keyboard_click"],
        "trace_loaded": True,
    }
    result = classify_failure(game, {})
    assert result["primary_failure_class"] == "C"
    assert result["diagnosis_confidence"] == "high"


def test_no_hack_audit_has_no_findings_for_diagnosis_sources() -> None:
    audit = no_hack_audit(["test-game"])
    assert audit["passes"] is True
    assert audit["findings"] == []


def test_enriched_trace_can_be_written_and_read(tmp_path: Path) -> None:
    path = tmp_path / "test-game.official.normal.json"
    enriched = enrich_trace(_trace(), seed=0)
    path.write_text(json.dumps(enriched), encoding="utf-8")
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert loaded["metadata"]["game_id"] == "test-game"
    assert loaded["diagnosis"]["audits"]["action_audit"]["invalid_mapping_count"] == 0

