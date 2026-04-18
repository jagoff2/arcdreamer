from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

from arcagi.core.types import ActionName

_DIRECTION_TO_VECTOR = {
    "up": (-1.0, 0.0),
    "down": (1.0, 0.0),
    "left": (0.0, -1.0),
    "right": (0.0, 1.0),
}


@dataclass(frozen=True)
class ActionSchemaContext:
    max_click_x: float = 1.0
    max_click_y: float = 1.0
    max_raw_action: float = 1.0
    affordance_count: float = 1.0
    action_roles: dict[ActionName, str] | None = None


@dataclass(frozen=True)
class ActionSchema:
    action: ActionName
    action_type: str
    role: str
    direction: str | None
    parts: tuple[str, ...]
    click: tuple[int, int] | None
    raw_action: int | None
    normalized_click: tuple[float, float] | None
    coarse_bin: tuple[int, int] | None
    family: str


def build_action_schema_context(
    affordances: Sequence[ActionName] = (),
    action_roles: Mapping[ActionName, str] | None = None,
) -> ActionSchemaContext:
    max_click_x = 1.0
    max_click_y = 1.0
    max_raw_action = 1.0
    for affordance in affordances:
        click = parse_click_action(affordance)
        if click is not None:
            click_x, click_y = click
            max_click_x = max(max_click_x, float(click_x))
            max_click_y = max(max_click_y, float(click_y))
        elif affordance.isdigit():
            max_raw_action = max(max_raw_action, float(int(affordance)))
    return ActionSchemaContext(
        max_click_x=max_click_x,
        max_click_y=max_click_y,
        max_raw_action=max_raw_action,
        affordance_count=max(float(len(affordances)), 1.0),
        action_roles=dict(action_roles or {}),
    )


def action_parts(action: ActionName) -> tuple[str, ...]:
    if action.startswith("click:"):
        return tuple(part for part in action.split(":") if part)
    return tuple(part for part in action.replace(":", "_").split("_") if part)


def parse_click_action(action: ActionName) -> tuple[int, int] | None:
    if not action.startswith("click:"):
        return None
    _, x_str, y_str, *_rest = action.split(":") + ["", ""]
    if not x_str.isdigit() or not y_str.isdigit():
        return None
    return int(x_str), int(y_str)


def extract_direction(parts: tuple[str, ...]) -> str | None:
    for part in reversed(parts):
        if part in _DIRECTION_TO_VECTOR:
            return part
    return None


def direction_vector(direction: str | None) -> tuple[float, float]:
    return _DIRECTION_TO_VECTOR.get(direction or "", (0.0, 0.0))


def infer_action_type(action: ActionName, parts: tuple[str, ...], role: str) -> str:
    if action in _DIRECTION_TO_VECTOR:
        return "move"
    if action.startswith("interact_"):
        return "interact"
    if action.startswith("click:"):
        return "click"
    if action in {"wait", "noop"} or role == "wait":
        return "wait"
    if action in {"cycle", "select"} or role in {"cycle", "select", "select_cycle"}:
        return "select"
    if action == "undo" or role == "undo":
        return "undo"
    if role.startswith("move"):
        return "move"
    if role == "click":
        return "click"
    if action.isdigit():
        return "raw"
    if "click" in parts or role == "click":
        return "click"
    if "interact" in parts or role == "interact":
        return "interact"
    return "other"


def build_action_schema(action: ActionName, context: ActionSchemaContext) -> ActionSchema:
    parts = action_parts(action)
    role = (context.action_roles or {}).get(action, "unknown")
    direction = extract_direction(parts)
    click = parse_click_action(action)
    raw_action = int(action) if action.isdigit() else None
    action_type = infer_action_type(action, parts, role)
    normalized_click = None
    coarse_bin = None
    if click is not None:
        click_x, click_y = click
        normalized_click = (
            float(click_x) / max(context.max_click_x, 1.0),
            float(click_y) / max(context.max_click_y, 1.0),
        )
        coarse_bin = (
            min(int(normalized_click[0] * 3.0), 2),
            min(int(normalized_click[1] * 3.0), 2),
        )
    family = f"{action_type}:{role}:{direction or 'none'}"
    if click is not None:
        family = f"{action_type}:{role}:parametric"
    elif raw_action is not None:
        family = f"{action_type}:{role}:{raw_action}"
    return ActionSchema(
        action=action,
        action_type=action_type,
        role=role,
        direction=direction,
        parts=parts,
        click=click,
        raw_action=raw_action,
        normalized_click=normalized_click,
        coarse_bin=coarse_bin,
        family=family,
    )
