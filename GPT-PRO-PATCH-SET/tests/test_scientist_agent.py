from __future__ import annotations

import numpy as np

from arcagi.scientist import HiddenRuleGridEnv, ScientistAgent, SyntheticConfig, run_episode
from arcagi.scientist.hypotheses import HypothesisEngine
from arcagi.scientist.perception import compare_states, extract_state
from arcagi.scientist.types import GridFrame, combined_progress_signal


def test_perception_segments_non_background_objects() -> None:
    grid = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    state = extract_state(GridFrame("t", "e", 0, grid, ("up",)))
    colors = sorted(obj.color for obj in state.objects)
    assert colors == [1, 2]
    assert state.abstract_fingerprint
    assert len(state.relations) >= 1


def test_hypothesis_engine_induces_movement_rule() -> None:
    before = GridFrame("t", "e", 0, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), ("right",))
    after = GridFrame("t", "e", 1, np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]), ("right",))
    record = compare_states(extract_state(before), extract_state(after), action="right")
    engine = HypothesisEngine()
    engine.observe_transition(record)
    kinds = {hyp.kind for hyp in engine.hypotheses.values()}
    assert "action_moves_object" in kinds


def test_combined_progress_signal_does_not_double_count_score_delta() -> None:
    assert combined_progress_signal(0.1, 0.1) == 0.1
    assert combined_progress_signal(0.1, 0.0) == 0.1
    assert combined_progress_signal(0.1, 0.2) == 0.30000000000000004


def test_scientist_agent_runs_updates_and_solves_hidden_rule_smoke_test() -> None:
    env = HiddenRuleGridEnv(SyntheticConfig(size=7, requires_key=True, seed=3, max_steps=80))
    agent = ScientistAgent()
    result = run_episode(env, agent, max_steps=80, seed=3)
    assert result.steps > 0
    assert result.diagnostics["transitions_observed"] == result.steps
    assert result.diagnostics["hypothesis_count"] > 0
    assert result.diagnostics["memory_items"] > 0
    assert agent.trace
    assert result.total_reward >= 1.0
    assert result.terminated
