"""Hypothesis-driven online experimental scientist agent for ARC-AGI-3."""

from .agent import ScientistAgent, ScientistAgentConfig
from .runtime import EpisodeResult, run_episode
from .synthetic_env import HiddenRuleGridEnv, SyntheticConfig

__all__ = [
    "ScientistAgent",
    "ScientistAgentConfig",
    "EpisodeResult",
    "run_episode",
    "HiddenRuleGridEnv",
    "SyntheticConfig",
]
