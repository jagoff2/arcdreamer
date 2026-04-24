"""Generic effect and budget helpers shared across scientist subsystems."""

from __future__ import annotations

from typing import Any, Mapping

from .types import StructuredState, TransitionRecord, action_family

_INFRASTRUCTURE_KEYS = {
    "adapter",
    "raw_observation_type",
    "game_state",
    "levels_completed",
    "background_color",
    "raw_available_actions",
    "action_roles",
    "cell_tags",
    "arc_step_index",
    "camera_grid_shape",
    "camera_origin",
    "display_scale",
    "display_padding",
    "score",
    "score_delta",
    "reward",
}
_INFRASTRUCTURE_PREFIXES = ("session_", "camera_", "display_", "arc_")


def state_numeric_channels(state: StructuredState) -> dict[str, float]:
    extras = state.frame.extras if isinstance(state.frame.extras, Mapping) else {}
    channels: dict[str, float] = {}
    for key, value in _iter_numeric_mapping(extras, prefix="extra::"):
        channels[key] = value
    inventory = extras.get("inventory")
    if isinstance(inventory, Mapping):
        for key, value in _iter_numeric_mapping(inventory, prefix="inventory::"):
            channels[key] = value
    flags = extras.get("flags")
    if isinstance(flags, Mapping):
        for key, value in _iter_numeric_mapping(flags, prefix="flag::"):
            channels[key] = value
    return channels


def transition_numeric_deltas(record: TransitionRecord) -> dict[str, float]:
    before = state_numeric_channels(record.before)
    after = state_numeric_channels(record.after)
    deltas: dict[str, float] = {}
    for key in set(before) | set(after):
        delta = float(after.get(key, 0.0) - before.get(key, 0.0))
        if abs(delta) > 1e-9:
            deltas[key] = delta
    return deltas


def numeric_delta_tags(record: TransitionRecord) -> frozenset[str]:
    deltas = transition_numeric_deltas(record)
    if not deltas:
        return frozenset()
    tags: set[str] = set()
    if any(delta > 0.0 for delta in deltas.values()):
        tags.add("effect:numeric_increase")
    if any(delta < 0.0 for delta in deltas.values()):
        tags.add("effect:numeric_decrease")
    if len(deltas) >= 2 or (tags and len(tags) == 2):
        tags.add("effect:numeric_rewrite")
    return frozenset(tags)


def state_schema_tokens(state: StructuredState, *, limit: int = 32) -> frozenset[str]:
    tokens = set(state.token_set())
    for action in state.available_actions[:32]:
        tokens.add(f"avail::{action_family(action)}")
    for key, value in state_numeric_channels(state).items():
        tokens.add(f"{key}:bin:{_bucket_numeric(value)}")
    ordered = sorted(tokens)
    return frozenset(ordered[:limit])


def _iter_numeric_mapping(values: Mapping[str, Any], *, prefix: str) -> tuple[tuple[str, float], ...]:
    items: list[tuple[str, float]] = []
    for raw_key, raw_value in values.items():
        key = str(raw_key)
        if _skip_key(key):
            continue
        number = _coerce_numeric(raw_value)
        if number is None:
            continue
        items.append((f"{prefix}{key}", number))
    return tuple(items)


def _skip_key(key: str) -> bool:
    if key in _INFRASTRUCTURE_KEYS:
        return True
    return any(key.startswith(prefix) for prefix in _INFRASTRUCTURE_PREFIXES)


def _coerce_numeric(value: Any) -> float | None:
    if isinstance(value, bool):
        return float(value)
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if not text:
        return None
    lowered = text.lower()
    if lowered in {"true", "false"}:
        return 1.0 if lowered == "true" else 0.0
    try:
        return float(text)
    except Exception:
        return None


def _bucket_numeric(value: float) -> int:
    magnitude = abs(float(value))
    if magnitude < 0.5:
        bucket = 0
    elif magnitude < 1.5:
        bucket = 1
    elif magnitude < 3.5:
        bucket = 2
    elif magnitude < 7.5:
        bucket = 3
    else:
        bucket = 4
    return -bucket if value < 0.0 else bucket
