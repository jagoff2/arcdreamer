from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

Position = tuple[int, int]
BoundingBox = tuple[int, int, int, int]
ActionName = str
COLOR_BUCKETS = 12

ACTION_ORDER: tuple[ActionName, ...] = (
    "up",
    "down",
    "left",
    "right",
    "interact_up",
    "interact_down",
    "interact_left",
    "interact_right",
    "wait",
)

_DIRECTION_VALUES: tuple[str, ...] = ("none", "up", "down", "left", "right")


@dataclass(frozen=True)
class GridObservation:
    task_id: str
    episode_id: str
    step_index: int
    grid: np.ndarray
    available_actions: tuple[ActionName, ...] = ACTION_ORDER
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ObjectState:
    object_id: str
    color: int
    cells: tuple[Position, ...]
    bbox: BoundingBox
    centroid: tuple[float, float]
    area: int
    tags: tuple[str, ...] = ()


@dataclass(frozen=True)
class Relation:
    relation_type: str
    source_id: str
    target_id: str
    value: float


@dataclass(frozen=True)
class StructuredState:
    task_id: str
    episode_id: str
    step_index: int
    grid_shape: tuple[int, int]
    grid_signature: tuple[int, ...]
    objects: tuple[ObjectState, ...]
    relations: tuple[Relation, ...]
    affordances: tuple[ActionName, ...]
    action_roles: tuple[tuple[ActionName, str], ...] = ()
    inventory: tuple[tuple[str, str], ...] = ()
    flags: tuple[tuple[str, str], ...] = ()

    def fingerprint(self) -> str:
        object_keys = {obj.object_id: _object_fingerprint_key(obj) for obj in self.objects}
        object_parts = []
        for obj in sorted(self.objects, key=_object_fingerprint_sort_key):
            object_parts.append(object_keys[obj.object_id])
        relation_parts = [
            f"{rel.relation_type}:{object_keys.get(rel.source_id, rel.source_id)}:{object_keys.get(rel.target_id, rel.target_id)}:{rel.value:.3f}"
            for rel in sorted(
                self.relations,
                key=lambda item: (
                    item.relation_type,
                    object_keys.get(item.source_id, item.source_id),
                    object_keys.get(item.target_id, item.target_id),
                    item.value,
                ),
            )
        ]
        return "|".join(
            [
                f"shape={self.grid_shape}",
                f"grid={','.join(map(str, self.grid_signature))}",
                f"objects={';'.join(object_parts)}",
                f"relations={';'.join(relation_parts)}",
                f"inventory={';'.join(f'{k}={v}' for k, v in self.inventory)}",
                f"flags={';'.join(f'{k}={v}' for k, v in self.flags)}",
            ]
        )

    def object_feature_rows(self, max_objects: int = 16) -> np.ndarray:
        rows: list[list[float]] = []
        height, width = self.grid_shape
        for obj in self.objects[:max_objects]:
            y0, x0, y1, x1 = obj.bbox
            rows.append(
                [
                    float(_color_bucket(obj.color)) / max(COLOR_BUCKETS - 1, 1),
                    float(obj.area),
                    obj.centroid[0] / max(height, 1),
                    obj.centroid[1] / max(width, 1),
                    y0 / max(height, 1),
                    x0 / max(width, 1),
                    y1 / max(height, 1),
                    x1 / max(width, 1),
                    float("agent" in obj.tags),
                    float("target" in obj.tags),
                    float("interactable" in obj.tags),
                    float("blocking" in obj.tags),
                ]
            )
        while len(rows) < max_objects:
            rows.append([0.0] * 12)
        return np.asarray(rows, dtype=np.float32)

    def summary_vector(self) -> np.ndarray:
        counts_by_color: dict[int, int] = {}
        agent_y = 0.0
        agent_x = 0.0
        target_y = 0.0
        target_x = 0.0
        for obj in self.objects:
            bucket = _color_bucket(obj.color)
            counts_by_color[bucket] = counts_by_color.get(bucket, 0) + obj.area
            if "agent" in obj.tags:
                agent_y, agent_x = obj.centroid
            if "target" in obj.tags:
                target_y, target_x = obj.centroid
        height, width = self.grid_shape
        palette = [float(counts_by_color.get(color, 0)) for color in range(COLOR_BUCKETS)]
        state_features = self._symbolic_state_features(width=8)
        return np.asarray(
            [
                float(len(self.objects)),
                agent_y / max(height, 1),
                agent_x / max(width, 1),
                target_y / max(height, 1),
                target_x / max(width, 1),
                *palette,
                *state_features,
            ],
            dtype=np.float32,
        )

    def transition_vector(self, max_objects: int = 4) -> np.ndarray:
        height, width = self.grid_shape
        total_cells = max(float(height * width), 1.0)
        objects = sorted(
            self.objects,
            key=lambda obj: (-obj.area, obj.color, obj.centroid[0], obj.centroid[1], obj.object_id),
        )
        total_area = float(sum(obj.area for obj in objects))
        if total_area > 0.0:
            global_y = sum(obj.centroid[0] * obj.area for obj in objects) / total_area
            global_x = sum(obj.centroid[1] * obj.area for obj in objects) / total_area
        else:
            global_y = 0.0
            global_x = 0.0
        features: list[float] = [
            float(len(objects)),
            total_area / total_cells,
            global_y / max(height, 1),
            global_x / max(width, 1),
        ]
        for obj in objects[:max_objects]:
            features.extend(
                [
                    float(_color_bucket(obj.color)) / max(COLOR_BUCKETS, 1),
                    float(obj.area) / total_cells,
                    obj.centroid[0] / max(height, 1),
                    obj.centroid[1] / max(width, 1),
                ]
            )
        while len(features) < 4 + (max_objects * 4):
            features.extend([0.0] * 4)
        features.extend(self._symbolic_state_features(width=5))
        return np.asarray(features[:25], dtype=np.float32)

    def as_grid(self) -> np.ndarray:
        return np.asarray(self.grid_signature, dtype=np.int64).reshape(self.grid_shape)

    def inventory_dict(self) -> dict[str, str]:
        return dict(self.inventory)

    def flags_dict(self) -> dict[str, str]:
        return dict(self.flags)

    def spatial_vector(self) -> np.ndarray:
        inventory = self.inventory_dict()
        flags = self.flags_dict()
        height, width = self.grid_shape
        total_cells = max(float(height * width), 1.0)
        visited_cells = _parse_symbolic_scalar(inventory.get("belief_visited_cells", "0")) or 0.0
        tested_sites = _parse_symbolic_scalar(inventory.get("belief_tested_sites", "0")) or 0.0
        effect_sites = _parse_symbolic_scalar(inventory.get("belief_effect_sites", "0")) or 0.0
        contradiction_sites = _parse_symbolic_scalar(inventory.get("belief_contradiction_sites", "0")) or 0.0
        frontier_distance = _distance_bucket_value(inventory.get("belief_frontier_distance", "none"))
        anchor_distance = _distance_bucket_value(inventory.get("belief_nearest_anchor_distance", "none"))
        tested_distance = _distance_bucket_value(inventory.get("belief_nearest_tested_distance", "none"))
        contradiction_distance = _distance_bucket_value(inventory.get("belief_nearest_contradiction_distance", "none"))
        frontier_direction = _direction_vector(inventory.get("belief_frontier_direction", "none"))
        anchor_direction = _direction_vector(inventory.get("belief_anchor_direction", "none"))
        contradiction_direction = _direction_vector(inventory.get("belief_contradiction_direction", "none"))
        return np.asarray(
            [
                visited_cells / total_cells,
                float(np.tanh(tested_sites / 4.0)),
                float(np.tanh(effect_sites / 4.0)),
                float(np.tanh(contradiction_sites / 4.0)),
                frontier_distance,
                anchor_distance,
                tested_distance,
                contradiction_distance,
                float(_is_truthy_symbol(flags.get("belief_has_spatial_anchor", "0"))),
                float(_is_truthy_symbol(flags.get("belief_near_spatial_anchor", "0"))),
                float(_is_truthy_symbol(flags.get("belief_near_tested_site", "0"))),
                float(_is_truthy_symbol(flags.get("belief_has_contradiction_hotspot", "0"))),
                frontier_direction[0],
                frontier_direction[1],
                anchor_direction[0],
                anchor_direction[1],
                contradiction_direction[0],
                contradiction_direction[1],
            ],
            dtype=np.float32,
        )

    def _symbolic_state_features(self, width: int) -> list[float]:
        pairs = tuple(sorted(self.inventory + self.flags))
        if width <= 0:
            return []
        if not pairs:
            return [0.0] * width
        pair_count = float(min(len(pairs), 8)) / 8.0
        binary_on = sum(1.0 for _, value in pairs if _is_truthy_symbol(value))
        binary_ratio = binary_on / max(float(len(pairs)), 1.0)
        numeric_values = [_parse_symbolic_scalar(value) for _, value in pairs]
        numeric_present = [value for value in numeric_values if value is not None]
        numeric_mean = (
            float(np.tanh(sum(numeric_present) / max(len(numeric_present), 1) / 4.0))
            if numeric_present
            else 0.0
        )
        numeric_max = (
            float(np.tanh(max(abs(value) for value in numeric_present) / 4.0))
            if numeric_present
            else 0.0
        )
        projections = [
            sum(_stable_symbol_projection(f"{key}={value}", salt=index) for key, value in pairs) / float(len(pairs))
            for index in range(max(width - 4, 0))
        ]
        features = [pair_count, binary_ratio, numeric_mean, numeric_max, *projections]
        if len(features) < width:
            features.extend([0.0] * (width - len(features)))
        return features[:width]


@dataclass(frozen=True)
class StepResult:
    observation: GridObservation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Transition:
    state: StructuredState
    action: ActionName
    reward: float
    next_state: StructuredState
    terminated: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class LanguageTrace:
    belief_tokens: tuple[str, ...] = ()
    question_tokens: tuple[str, ...] = ()
    plan_tokens: tuple[str, ...] = ()


@dataclass(frozen=True)
class ActionThought:
    action: ActionName
    value: float = 0.0
    uncertainty: float = 0.0
    policy: float = 0.0
    policy_weight: float = 0.0
    predicted_reward: float = 0.0
    usefulness: float = 0.0
    selector_followup: float = 0.0
    next_latent: Any | None = None
    next_hidden: Any | None = None
    next_state_proxy: Any | None = None


@dataclass(frozen=True)
class StructuredClaim:
    claim_type: str
    subject: str
    relation: str
    object: str
    confidence: float = 0.0
    evidence: float = 0.0
    salience: float = 0.0

    def as_tokens(self) -> tuple[str, ...]:
        confidence_bucket = "high" if self.confidence >= 0.75 else "mid" if self.confidence >= 0.5 else "low"
        return (self.claim_type, self.subject, self.relation, self.object, confidence_bucket)


@dataclass(frozen=True)
class RuntimeThought:
    belief_tokens: tuple[str, ...] = ()
    question_tokens: tuple[str, ...] = ()
    plan_tokens: tuple[str, ...] = ()
    actions: tuple[ActionThought, ...] = ()
    claims: tuple[StructuredClaim, ...] = ()
    world_model_calls: int = 0

    def for_action(self, action: ActionName) -> ActionThought | None:
        for candidate in self.actions:
            if candidate.action == action:
                return candidate
        return None

    def value_for(self, action: ActionName) -> float:
        thought = self.for_action(action)
        return 0.0 if thought is None else thought.value

    def uncertainty_for(self, action: ActionName) -> float:
        thought = self.for_action(action)
        return 0.0 if thought is None else thought.uncertainty

    def policy_for(self, action: ActionName) -> float:
        thought = self.for_action(action)
        return 0.0 if thought is None else thought.policy

    def selector_followup_for(self, action: ActionName) -> float:
        thought = self.for_action(action)
        return 0.0 if thought is None else thought.selector_followup

    def claim_tokens(self, limit: int = 2) -> tuple[str, ...]:
        tokens: list[str] = []
        for claim in self.claims[:limit]:
            tokens.extend(claim.as_tokens())
        return tuple(tokens)


@dataclass(frozen=True)
class PlanOutput:
    action: ActionName
    scores: dict[str, float]
    language: LanguageTrace = field(default_factory=LanguageTrace)
    search_path: tuple[ActionName, ...] = ()


def _color_bucket(color: int) -> int:
    return int(color) % COLOR_BUCKETS


def _is_truthy_symbol(value: str) -> bool:
    normalized = str(value).strip().lower()
    return normalized in {"1", "true", "yes", "on", "active"}


def _parse_symbolic_scalar(value: str) -> float | None:
    normalized = str(value).strip().lower()
    if normalized in {"", "none"}:
        return None
    if normalized in {"true", "yes", "on", "active"}:
        return 1.0
    if normalized in {"false", "no", "off", "inactive"}:
        return 0.0
    try:
        return float(normalized)
    except ValueError:
        return None


def _stable_symbol_projection(token: str, *, salt: int) -> float:
    mixed = f"{salt}|{token}"
    state = 2166136261
    for character in mixed:
        state ^= ord(character)
        state = (state * 16777619) & 0xFFFFFFFF
    return ((state / 0xFFFFFFFF) * 2.0) - 1.0


def _distance_bucket_value(value: str) -> float:
    normalized = str(value).strip().lower()
    if normalized == "near":
        return 1.0 / 3.0
    if normalized == "mid":
        return 2.0 / 3.0
    if normalized == "far":
        return 1.0
    return 0.0


def _direction_vector(value: str) -> tuple[float, float]:
    normalized = str(value).strip().lower()
    if normalized == "up":
        return (-1.0, 0.0)
    if normalized == "down":
        return (1.0, 0.0)
    if normalized == "left":
        return (0.0, -1.0)
    if normalized == "right":
        return (0.0, 1.0)
    return (0.0, 0.0)


def _object_fingerprint_key(obj: ObjectState) -> str:
    cells_key = ",".join(f"{y}:{x}" for y, x in sorted(obj.cells))
    return f"{obj.color}:{obj.area}:{obj.bbox}:{obj.centroid}:{','.join(obj.tags)}:{cells_key}"


def _object_fingerprint_sort_key(obj: ObjectState) -> tuple[object, ...]:
    return (
        obj.color,
        obj.area,
        obj.bbox,
        tuple(sorted(obj.tags)),
        tuple(sorted(obj.cells)),
    )
