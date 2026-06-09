from __future__ import annotations

from src.heldout_causal import FROZEN_CHECKPOINT, evaluate_heldout_causal


def test_heldout_causal_smoke_report_shape() -> None:
    report = evaluate_heldout_causal(FROZEN_CHECKPOINT, config_name="smoke")
    assert report["checkpoint"] == FROZEN_CHECKPOINT
    assert report["summary"]["passes"] is True
    assert min(report["summary"]["margins"].values()) >= report["summary"]["required_margin"]
    for name in [
        "larger_world_9_projected",
        "distractor_objects_untrained_ticks",
        "variable_delays_23_49_77_103",
        "contradictory_source_chain",
        "multi_step_goal_schedule",
        "blank_continuation_after_16",
        "energy_pressure_late",
    ]:
        assert name in report["report"]


def test_heldout_causal_includes_required_baselines() -> None:
    report = evaluate_heldout_causal(FROZEN_CHECKPOINT, config_name="smoke")
    for baseline in [
        "feedforward",
        "zero_z",
        "shuffled_z",
        "no_language",
        "no_provenance",
        "no_blank_continuation",
        "internal_language_mask",
    ]:
        assert baseline in report["baselines"]
        assert baseline in report["summary"]["baseline_means"]
    assert report["summary"]["internal_language_delta"] >= report["summary"]["required_margin"]
