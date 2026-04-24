"""Hypothesis-driven online experimental scientist agent for ARC-AGI-3."""

from .agent import (
    ScientistAgent,
    ScientistAgentConfig,
    load_scientist_checkpoint,
    save_scientist_checkpoint,
)
from .boundary import CLAIM_ELIGIBLE_ARC_CONTROLLER, SCIENTIST_STACK_ROLE
from .runtime import EpisodeResult, run_episode
from .synthetic_env import HiddenRuleGridEnv, SyntheticConfig

__all__ = [
    "SCIENTIST_STACK_ROLE",
    "CLAIM_ELIGIBLE_ARC_CONTROLLER",
    "ScientistAgent",
    "ScientistAgentConfig",
    "save_scientist_checkpoint",
    "load_scientist_checkpoint",
    "EpisodeResult",
    "run_episode",
    "HiddenRuleGridEnv",
    "SyntheticConfig",
]
