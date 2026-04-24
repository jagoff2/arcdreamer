from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np

from arcagi.core.types import GridObservation, StepResult
from arcagi.evaluation.harness import build_agent, run_episode
from arcagi.memory.graph import StateGraph
from arcagi.scientist.boundary import CLAIM_ELIGIBLE_ARC_CONTROLLER, SCIENTIST_STACK_ROLE


def test_spotlight_stack_is_not_claim_eligible() -> None:
    agent = build_agent("spotlight")

    assert SCIENTIST_STACK_ROLE == "instrumented_spotlight_baseline"
    assert CLAIM_ELIGIBLE_ARC_CONTROLLER is False
    assert agent.diagnostics()["controller_kind"] == "instrumented_spotlight_baseline"
    assert agent.diagnostics()["claim_eligible"] is False


def test_learned_online_minimal_does_not_construct_forbidden_controllers() -> None:
    agent = build_agent("learned_online_minimal")

    assert agent.controller_kind == "learned_online_minimal"
    assert agent.claim_eligible_arc_controller is True
    assert not hasattr(agent, "runtime_rule_controller")
    assert not hasattr(agent, "theory_manager")
    assert not hasattr(agent, "spotlight")
    assert not hasattr(agent, "planner")
    assert not hasattr(agent, "rule_inducer")


def test_learned_online_minimal_import_boundary_is_clean() -> None:
    root = Path(__file__).resolve().parents[1]
    checked = list((root / "arcagi" / "learned_online").glob("*.py"))
    checked.append(root / "arcagi" / "agents" / "learned_online_minimal_agent.py")
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


def test_learned_online_minimal_scores_full_dense_surface() -> None:
    agent = build_agent("learned_online_minimal")
    actions = tuple(f"click:{x}:{y}" for y in range(7) for x in range(9))
    observation = GridObservation(
        task_id="dense",
        episode_id="0",
        step_index=0,
        grid=np.zeros((7, 9), dtype=np.int64),
        available_actions=actions,
    )

    action = agent.act(observation)
    diagnostics = agent.diagnostics()

    assert action in actions
    assert diagnostics["last_legal_action_count"] == len(actions)
    assert diagnostics["last_scored_action_count"] == len(actions)


def test_learned_online_minimal_does_not_call_graph_frontier_methods(monkeypatch) -> None:
    def fail(*_args, **_kwargs):
        raise AssertionError("clean learned online agent must not use graph frontier control")

    monkeypatch.setattr(StateGraph, "frontier_actions", fail, raising=False)
    monkeypatch.setattr(StateGraph, "best_frontier_action", fail, raising=False)
    monkeypatch.setattr(StateGraph, "shortest_path_to_frontier", fail, raising=False)

    class Env:
        family_id = "graph_boundary"

        def reset(self, seed: int):
            return GridObservation("graph_boundary", str(seed), 0, np.zeros((2, 2), dtype=np.int64), ("a", "b"))

        def step(self, action: str):
            return StepResult(
                GridObservation("graph_boundary", "0", 1, np.zeros((2, 2), dtype=np.int64), ("a", "b")),
                0.0,
                False,
                False,
                {},
            )

    result = run_episode(build_agent("learned_online_minimal"), Env(), seed=0, max_steps=1)

    assert result["steps"] == 1
    assert result["claim_eligible"] is True


def test_clean_learned_agent_source_has_no_spotlight_controller_dependency() -> None:
    import arcagi.agents.learned_online_minimal_agent as mod

    source = inspect.getsource(mod)
    forbidden = (
        "ActionSpotlight",
        "EpisodeTheoryManager",
        "RuntimeRuleController",
        "HybridPlanner",
        "diagnostic_binding_probe",
    )
    for token in forbidden:
        assert token not in source


def test_learned_online_minimal_checkpoint_roundtrip(tmp_path) -> None:
    agent = build_agent("learned_online_minimal")
    agent.model.biases["reward"] = 0.25
    path = tmp_path / "learned_online.pkl"
    agent.save_checkpoint(path)

    restored = build_agent("learned_online_minimal", checkpoint_path=str(path))

    assert restored.model.biases["reward"] == 0.25
    assert restored.controller_kind == "learned_online_minimal"


def test_learned_online_minimal_is_scaffold_not_validated_arc_competence() -> None:
    agent = build_agent("learned_online_minimal")

    assert agent.claim_eligible_arc_controller is True
    assert agent.arc_competence_validated is False
    assert agent.role == "falsification_gate_scaffold"
    assert agent.diagnostics()["arc_competence_validated"] is False
