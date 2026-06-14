"""Compact runtime package for the recurrent latent scientist agent.

Startup hooks are applied during import.
"""

from __future__ import annotations

try:
    from .goal_credit_patch_runtime import apply_goal_credit_patch_runtime

    apply_goal_credit_patch_runtime()
except Exception:
    pass

REQUIRED_RUNTIME_MODULES: tuple[str, ...] = (
    "env",
    "model",
    "train",
    "evaluate",
    "metrics",
    "run_unbroken",
)

__all__ = ["REQUIRED_RUNTIME_MODULES"]
