from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Sequence

import numpy as np

from arcagi.core.action_schema import (
    ActionSchema,
    build_action_schema,
    build_action_schema_context,
    click_action_to_grid_cell,
)
from arcagi.core.types import ActionName, ObjectState, StructuredState

ACTION_TYPES: tuple[str, ...] = ("click", "move", "interact", "select", "reset", "undo", "wait", "raw", "other")
DIRECTIONS: tuple[str, ...] = ("none", "up", "down", "left", "right")
ACTION_FEATURE_DIM = 40


@dataclass(frozen=True)
class ActionFeatureBatch:
    actions: tuple[ActionName, ...]
    features: np.ndarray
    legal_action_count: int
    scored_action_count: int


@dataclass(frozen=True)
class ActionFeatureConfig:
    include_exact_action_projection: bool = False
    include_role_projection: bool = True
    include_family_projection: bool = True

    def to_dict(self) -> dict[str, bool]:
        return {
            "include_exact_action_projection": bool(self.include_exact_action_projection),
            "include_role_projection": bool(self.include_role_projection),
            "include_family_projection": bool(self.include_family_projection),
        }


DEFAULT_ACTION_FEATURE_CONFIG = ActionFeatureConfig()


def encode_action_candidates(
    state: StructuredState,
    legal_actions: Sequence[ActionName],
    *,
    config: ActionFeatureConfig = DEFAULT_ACTION_FEATURE_CONFIG,
) -> ActionFeatureBatch:
    actions = tuple(str(action) for action in legal_actions)
    context = build_action_schema_context(actions, dict(state.action_roles))
    rows = [_encode_one_action(state, build_action_schema(action, context), config=config) for action in actions]
    features = np.asarray(rows, dtype=np.float32)
    if features.size == 0:
        features = np.zeros((0, ACTION_FEATURE_DIM), dtype=np.float32)
    assert features.shape[1] == ACTION_FEATURE_DIM
    return ActionFeatureBatch(
        actions=actions,
        features=features,
        legal_action_count=len(actions),
        scored_action_count=len(actions),
    )


def action_context_key(state: StructuredState, action: ActionName) -> str:
    context = build_action_schema_context(state.affordances, dict(state.action_roles))
    schema = build_action_schema(str(action), context)
    obj = _target_object(state, str(action))
    if obj is None:
        obj_key = "none"
    else:
        obj_key = f"c{int(obj.color) % 12}:a{_bucket(float(obj.area), (1.0, 4.0, 9.0, 16.0))}"
    click_bin = "none" if schema.coarse_bin is None else f"{schema.coarse_bin[0]}:{schema.coarse_bin[1]}"
    return f"{schema.action_type}:{schema.direction or 'none'}:{click_bin}:{obj_key}"


def _encode_one_action(state: StructuredState, schema: ActionSchema, *, config: ActionFeatureConfig) -> list[float]:
    height, width = state.grid_shape
    obj = _target_object(state, schema.action)
    grid_cell = click_action_to_grid_cell(
        schema.action,
        grid_shape=state.grid_shape,
        inventory=state.inventory_dict(),
    )
    row: list[float] = [1.0]
    row.extend(1.0 if schema.action_type == item else 0.0 for item in ACTION_TYPES)
    direction = schema.direction or "none"
    row.extend(1.0 if direction == item else 0.0 for item in DIRECTIONS)
    click_x, click_y = schema.click if schema.click is not None else (0, 0)
    norm_click = schema.normalized_click or (0.0, 0.0)
    coarse = schema.coarse_bin or (0, 0)
    row.extend(
        [
            float(schema.click is not None),
            float(norm_click[0]),
            float(norm_click[1]),
            float(coarse[0]) / 2.0,
            float(coarse[1]) / 2.0,
            float(grid_cell is not None),
            float(click_y) / max(float(height), 1.0),
            float(click_x) / max(float(width), 1.0),
            float(schema.raw_action or 0) / 16.0,
            _stable_projection(schema.action, salt=11) if config.include_exact_action_projection else 0.0,
            _stable_projection(schema.role, salt=3) if config.include_role_projection else 0.0,
            _stable_projection(schema.family, salt=7) if config.include_family_projection else 0.0,
        ]
    )
    row.extend(_object_features(state, obj))
    row.extend(
        [
            math.tanh(float(len(state.objects)) / 8.0),
            math.tanh(float(len(state.affordances)) / 256.0),
            float(height) / 64.0,
            float(width) / 64.0,
        ]
    )
    if len(row) < ACTION_FEATURE_DIM:
        row.extend([0.0] * (ACTION_FEATURE_DIM - len(row)))
    return row[:ACTION_FEATURE_DIM]


def _object_features(state: StructuredState, obj: ObjectState | None) -> list[float]:
    if obj is None:
        return [0.0] * 9
    height, width = state.grid_shape
    y0, x0, y1, x1 = obj.bbox
    tags = set(obj.tags)
    total_cells = max(float(height * width), 1.0)
    return [
        1.0,
        float(int(obj.color) % 12) / 11.0,
        float(obj.area) / total_cells,
        float(obj.centroid[0]) / max(float(height), 1.0),
        float(obj.centroid[1]) / max(float(width), 1.0),
        float(max(0, y1 - y0 + 1)) / max(float(height), 1.0),
        float(max(0, x1 - x0 + 1)) / max(float(width), 1.0),
        float("clickable" in tags or "interface_target" in tags),
        float("interactable" in tags or "blocking" in tags),
    ]


def _target_object(state: StructuredState, action: ActionName) -> ObjectState | None:
    cell = click_action_to_grid_cell(action, grid_shape=state.grid_shape, inventory=state.inventory_dict())
    if cell is None:
        return None
    for obj in state.objects:
        if cell in obj.cells:
            return obj
    return None


def _bucket(value: float, thresholds: tuple[float, ...]) -> int:
    for index, threshold in enumerate(thresholds):
        if value <= threshold:
            return index
    return len(thresholds)


def _stable_projection(token: str, *, salt: int) -> float:
    text = f"{salt}|{token}"
    state = 2166136261
    for character in text:
        state ^= ord(character)
        state = (state * 16777619) & 0xFFFFFFFF
    return float((state / 0xFFFFFFFF) * 2.0 - 1.0)
