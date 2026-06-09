from __future__ import annotations

from src.evidence_dossier import (
    FROZEN_CHECKPOINT,
    generate_evidence_dossier,
    observation_schema_dump,
)


def test_observation_schema_excludes_supervision_and_hidden_state() -> None:
    schema = observation_schema_dump()
    assert schema["model_input_keys"] == ["sensory", "lang_in"]
    assert schema["disallowed_supervision_keys_in_model_input"] == []
    assert "provenance_target" in schema["supervision_only_keys"]
    assert "world_pos_target" in schema["supervision_only_keys"]
    assert "memory_color_target" in schema["supervision_only_keys"]
    assert schema["post_occlusion_unknown_sentinel_check"]["tick_0_visible_flag_mean"] == 1.0
    assert schema["post_occlusion_unknown_sentinel_check"]["tick_6_visible_flag_mean"] == 0.0


def test_evidence_dossier_smoke_covers_required_evidence_families() -> None:
    report = generate_evidence_dossier(FROZEN_CHECKPOINT, config_name="smoke")
    assert report["verdict"]["passes"] is True

    rows = report["heldout_causal"]["per_subtest_rows"]
    methods = {row["method"] for row in rows}
    suites = {row["suite"] for row in rows}
    assert "trained_reset_recurrent" in methods
    assert "trained_feedforward_capacity" in methods
    assert "contradictory_source_chain" in suites
    assert "variable_delays_23_49_77_103" in suites
    assert "larger_world_9_projected" in suites

    assert len(report["randomized_post_freeze_templates"]) >= 1
    assert "changed_grammar_aliases" in report["ood_coverage"]
    for item in report["randomized_post_freeze_templates"].values():
        assert item["template"]["alias_map"]["ASK_COLOR_alias"] == "ASK_GOAL"
        assert item["template"]["distractor_ticks"]

    trajectories = report["example_trajectories"]
    assert trajectories["successes"][0]["rows"]
    assert trajectories["failures"][0]["rows"]

    probe = report["latent_causal_probe"]
    assert probe["world_pos_shift_rate"] > 0.0 or probe["action_changed_rate"] > 0.0
    assert probe["examples"]

    restart = report["restart_consolidation"]
    assert restart["restart_tick"] > 0.0

    idle = report["idle_mode"]
    assert idle["final_memory_accuracy"] >= 0.85
    assert idle["final_object_pos_accuracy"] >= 0.85
    assert idle["world_pos_entropy_range"] > 0.001
