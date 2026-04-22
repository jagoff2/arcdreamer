from __future__ import annotations

import numpy as np

from arcagi.evaluation.harness import build_agent
from arcagi.agents.spotlight_scientist_agent import SpotlightScientistAgent
from arcagi.scientist.perception import compare_states, extract_state
from arcagi.scientist.spotlight import ActionSpotlight
from arcagi.scientist.types import GridFrame


def test_spotlight_scientist_agent_exposes_spotlight_diagnostics() -> None:
    agent = SpotlightScientistAgent()
    diagnostics = agent.diagnostics()
    assert "spotlight" in diagnostics


def test_harness_supports_scientist_and_spotlight_aliases() -> None:
    for name in ("scientist", "spotlight"):
        agent = build_agent(name)
        diagnostics = agent.diagnostics()
        assert "spotlight" in diagnostics


def test_binding_action_creates_pending_probe_and_probe_clears_on_support() -> None:
    spotlight = ActionSpotlight()

    before_bind = extract_state(
        GridFrame("task", "episode", 0, np.array([[0, 1], [0, 0]], dtype=np.int64), ("action5", "right"))
    )
    after_bind = extract_state(
        GridFrame("task", "episode", 1, np.array([[0, 1], [0, 0]], dtype=np.int64), ("action5", "right"))
    )
    bind_record = compare_states(before_bind, after_bind, action="action5")
    spotlight.notify_transition(record=bind_record)

    diagnostics = spotlight.diagnostics()
    assert diagnostics["pending_binder_probe"] is not None
    assert diagnostics["pending_binder_probe"]["binder_action"] == "action5"

    after_probe = extract_state(
        GridFrame("task", "episode", 2, np.array([[0, 0], [0, 1]], dtype=np.int64), ("action5", "right"))
    )
    probe_record = compare_states(after_bind, after_probe, action="right")
    spotlight.notify_transition(record=probe_record)

    diagnostics = spotlight.diagnostics()
    assert diagnostics["pending_binder_probe"] is None
    assert diagnostics["binding_success_total"] >= 1
