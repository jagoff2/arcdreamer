from __future__ import annotations

import pytest
import torch

import arcagi.evaluation.harness as harness
from arcagi.evaluation.harness import build_agent
from arcagi.envs.arc_adapter import require_dense_arc_action_surface


def test_clean_eval_agents_do_not_construct_runtime_controller() -> None:
    device = torch.device("cpu")

    recurrent = build_agent("recurrent", device=device)
    language = build_agent("language", device=device)
    language_memory = build_agent("online_language_memory", device=device)
    hybrid = build_agent("hybrid", device=device)

    assert recurrent.config.use_runtime_controller is False
    assert recurrent.runtime_rule_controller is None
    assert recurrent.config.use_theory_manager is False
    assert recurrent.theory_manager is None
    assert language.config.use_runtime_controller is False
    assert language.runtime_rule_controller is None
    assert language.config.use_theory_manager is False
    assert language.theory_manager is None
    assert language_memory.config.use_memory is True
    assert language_memory.episodic_memory is not None
    assert language_memory.config.use_runtime_controller is False
    assert language_memory.runtime_rule_controller is None
    assert language_memory.config.use_theory_manager is False
    assert language_memory.theory_manager is None
    assert hybrid.config.use_runtime_controller is False
    assert hybrid.runtime_rule_controller is None
    assert hybrid.config.use_theory_manager is True
    assert hybrid.theory_manager is not None


def test_arc_eval_rejects_sparse_click_surface_without_smoke_flag(monkeypatch) -> None:
    monkeypatch.setenv("ARCAGI_SPARSE_CLICKS_BASELINE", "1")
    monkeypatch.setattr(harness, "arc_toolkit_available", lambda: True)

    with pytest.raises(RuntimeError, match="hides legal ARC click parameters"):
        harness.evaluate_arc("random", game_id="game-0")


def test_sparse_click_surface_is_explicitly_labeled_when_allowed(monkeypatch) -> None:
    monkeypatch.setenv("ARCAGI_SPARSE_CLICKS_BASELINE", "1")

    metadata = require_dense_arc_action_surface(context="test", allow_sparse_click_smoke=True)

    assert metadata["dense_action_surface"] is False
    assert metadata["sparse_click_baseline"] is True
    assert metadata["allow_sparse_click_smoke"] is True
