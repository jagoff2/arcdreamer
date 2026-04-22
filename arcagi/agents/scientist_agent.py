"""Compatibility wrapper for the public scientist agent entry point."""

from __future__ import annotations

from typing import Any

from arcagi.agents.spotlight_scientist_agent import (
    SpotlightScientistAgent,
    SpotlightScientistConfig,
    load_spotlight_scientist_checkpoint,
    make_agent as make_spotlight_agent,
    save_spotlight_scientist_checkpoint,
)

try:  # Existing repository compatibility; tests do not require this base class.
    from arcagi.agents.base import BaseAgent  # type: ignore
except Exception:  # pragma: no cover
    BaseAgent = object  # type: ignore


class HyperGeneralizingScientistAgent(SpotlightScientistAgent, BaseAgent):  # type: ignore[misc]
    """ARC-facing online learner with action-level spotlight control."""

    def __init__(self, config: SpotlightScientistConfig | None = None, **kwargs: Any) -> None:
        SpotlightScientistAgent.__init__(self, config=config, **kwargs)


def make_agent(config: SpotlightScientistConfig | None = None) -> HyperGeneralizingScientistAgent:
    return HyperGeneralizingScientistAgent(config=config)


__all__ = [
    "SpotlightScientistAgent",
    "SpotlightScientistConfig",
    "HyperGeneralizingScientistAgent",
    "make_agent",
    "make_spotlight_agent",
    "load_spotlight_scientist_checkpoint",
    "save_spotlight_scientist_checkpoint",
]
