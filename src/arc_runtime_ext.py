"""ARC runtime extension loader."""
from __future__ import annotations


def enable() -> None:
    from .plan_cognition_runtime import apply_plan_cognition_patch

    apply_plan_cognition_patch()
