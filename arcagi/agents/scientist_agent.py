"""Compatibility wrapper for the existing ``arcagi.agents`` namespace."""

from __future__ import annotations

from typing import Any

from arcagi.scientist import ScientistAgent, ScientistAgentConfig

try:  # Existing repository compatibility; tests do not require this base class.
    from arcagi.agents.base import BaseAgent  # type: ignore
except Exception:  # pragma: no cover
    BaseAgent = object  # type: ignore


class HyperGeneralizingScientistAgent(ScientistAgent, BaseAgent):  # type: ignore[misc]
    """ARC-facing online learner built from the scientist-agent components."""

    def __init__(self, config: ScientistAgentConfig | None = None, **_: Any) -> None:
        ScientistAgent.__init__(self, config=config)


def make_agent(config: ScientistAgentConfig | None = None) -> HyperGeneralizingScientistAgent:
    return HyperGeneralizingScientistAgent(config=config)
