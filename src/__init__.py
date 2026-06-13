"""Compact runtime package for the recurrent latent scientist agent."""

from __future__ import annotations

REQUIRED_RUNTIME_MODULES: tuple[str, ...] = (
    "env",
    "model",
    "train",
    "evaluate",
    "metrics",
    "run_unbroken",
)

__all__ = ["REQUIRED_RUNTIME_MODULES"]
