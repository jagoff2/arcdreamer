from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from arcagi.core.types import GridObservation
from arcagi.evaluation.harness import build_agent


def test_recurrent_agent_scores_full_dense_surface_and_updates_hidden() -> None:
    agent = build_agent("learned_online_recurrent")
    actions = tuple(f"click:{x}:{y}" for y in range(6) for x in range(6))
    observation = GridObservation("recurrent_dense", "0", 0, np.zeros((6, 6), dtype=np.int64), actions)

    chosen = agent.act(observation)
    before_hidden = float(agent.diagnostics()["hidden_norm"])
    agent.update_after_step(observation, reward=0.0, terminated=False, info={})
    after_hidden = float(agent.diagnostics()["hidden_norm"])

    assert chosen in actions
    assert agent.diagnostics()["last_scored_action_count"] == len(actions)
    assert agent.diagnostics()["last_legal_action_count"] == len(actions)
    assert after_hidden != before_hidden


def test_recurrent_agent_does_not_construct_forbidden_controllers() -> None:
    agent = build_agent("learned_online_recurrent")

    assert agent.controller_kind == "learned_online_recurrent_v1"
    assert agent.claim_eligible_arc_controller is True
    assert agent.arc_competence_validated is False
    assert not hasattr(agent, "runtime_rule_controller")
    assert not hasattr(agent, "theory_manager")
    assert not hasattr(agent, "spotlight")
    assert not hasattr(agent, "planner")
    assert not hasattr(agent, "rule_inducer")


def test_recurrent_import_boundary_is_clean() -> None:
    root = Path(__file__).resolve().parents[1]
    checked = [
        root / "arcagi" / "agents" / "learned_online_recurrent_agent.py",
        root / "arcagi" / "learned_online" / "recurrent_model.py",
        root / "arcagi" / "learned_online" / "recurrent_policy.py",
        root / "arcagi" / "learned_online" / "sequence.py",
    ]
    forbidden = (
        "RuntimeRuleController",
        "HybridPlanner",
        "EpisodeRuleInducer",
        "EpisodeTheoryManager",
        "ActionSpotlight",
        "SpotlightScientistAgent",
        "GraphExplorerAgent",
    )
    for path in checked:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{token} leaked into {path}"


def test_recurrent_source_has_no_diagnostic_binding_probe_dependency() -> None:
    import arcagi.agents.learned_online_recurrent_agent as mod

    source = inspect.getsource(mod)
    assert "diagnostic_binding_probe" not in source
