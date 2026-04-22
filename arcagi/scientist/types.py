"""Core datatypes for the ARC-AGI-3 hypothesis-driven scientist agent.

The objects in this file deliberately avoid any dependency on ARC internals.  The
agent only assumes a turn-based environment that returns grid-like observations,
a finite action set, and scalar feedback.  The same types can therefore wrap the
existing ``arcagi.core.types.GridObservation`` objects, raw numpy grids, or the
official ARC toolkit wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Any, Mapping, Sequence

import numpy as np

ActionName = str
GridArray = np.ndarray

MOVE_DELTAS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "move_up": (-1, 0),
    "move_down": (1, 0),
    "move_left": (0, -1),
    "move_right": (0, 1),
    "action1": (-1, 0),
    "action2": (1, 0),
    "action3": (0, -1),
    "action4": (0, 1),
}

TARGETED_FAMILIES = {
    "click",
    "interact_at",
    "touch",
    "probe",
    "action6",
    "select_at",
}

SELECTOR_FAMILIES = {
    "select",
    "select_cycle",
    "switch",
    "mode",
    "action5",
}

INTERACT_FAMILIES = {
    "interact",
    "use",
    "activate",
    "push",
    "pickup",
    "action6",
    "click",
}


def _stable_digest(text: str, *, size: int = 16) -> str:
    return blake2b(text.encode("utf-8"), digest_size=size).hexdigest()


def normalize_action(action: Any) -> ActionName:
    """Return the repository-wide string representation for an action."""

    if isinstance(action, str):
        return action
    name = getattr(action, "name", None)
    if isinstance(name, str):
        return name.lower()
    value = getattr(action, "value", None)
    if value is not None:
        return str(value).lower()
    return str(action).lower()


def action_family(action: ActionName) -> str:
    """Return a canonical family name, stripping coordinates and punctuation.

    Examples:
        ``click:4:7`` -> ``click``
        ``ACTION6@4,7`` -> ``action6``
        ``move_up`` -> ``up``
    """

    raw = normalize_action(action).strip().lower()
    if ":" in raw:
        raw = raw.split(":", 1)[0]
    if "@" in raw:
        raw = raw.split("@", 1)[0]
    raw = raw.replace("-", "_").replace(" ", "_")
    aliases = {
        "moveup": "up",
        "movedown": "down",
        "moveleft": "left",
        "moveright": "right",
        "north": "up",
        "south": "down",
        "west": "left",
        "east": "right",
        "a1": "action1",
        "a2": "action2",
        "a3": "action3",
        "a4": "action4",
        "a5": "action5",
        "a6": "action6",
        "a7": "action7",
    }
    raw = aliases.get(raw, raw)
    if raw in {"move_up", "action1"}:
        return "up"
    if raw in {"move_down", "action2"}:
        return "down"
    if raw in {"move_left", "action3"}:
        return "left"
    if raw in {"move_right", "action4"}:
        return "right"
    return raw


def is_move_action(action: ActionName) -> bool:
    return action_family(action) in {"up", "down", "left", "right"}


def action_delta(action: ActionName) -> tuple[int, int] | None:
    family = action_family(action)
    return {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }.get(family)


def is_selector_action(action: ActionName) -> bool:
    return action_family(action) in SELECTOR_FAMILIES


def is_interact_action(action: ActionName) -> bool:
    return action_family(action) in INTERACT_FAMILIES


def is_targeted_action(action: ActionName) -> bool:
    family = action_family(action)
    return family in TARGETED_FAMILIES or parse_action_target(action) is not None


def parse_action_target(action: ActionName) -> tuple[int, int] | None:
    """Parse row/column coordinates from an action string if present."""

    raw = normalize_action(action).strip().lower()
    coord_part = ""
    if ":" in raw:
        parts = raw.split(":")
        if len(parts) >= 3:
            coord_part = ",".join(parts[-2:])
    elif "@" in raw:
        coord_part = raw.split("@", 1)[1]
    if not coord_part:
        return None
    coord_part = coord_part.replace(";", ",").replace(" ", ",")
    pieces = [p for p in coord_part.split(",") if p != ""]
    if len(pieces) < 2:
        return None
    try:
        row = int(float(pieces[0]))
        col = int(float(pieces[1]))
    except ValueError:
        return None
    return row, col


def make_targeted_action(base_action: ActionName, row: int, col: int) -> ActionName:
    family = action_family(base_action)
    if family in {"action6", "click"}:
        return f"click:{int(row)}:{int(col)}"
    return f"{normalize_action(base_action)}:{int(row)}:{int(col)}"


def grid_cell_to_action_coordinates(row: int, col: int, extras: Mapping[str, Any] | None = None) -> tuple[int, int]:
    """Map a logical grid cell to click coordinates when adapter metadata exists."""

    meta = (extras or {}).get("camera_meta") if isinstance(extras, Mapping) else None
    if not isinstance(meta, Mapping):
        return int(row), int(col)
    try:
        scale = int(meta.get("scale", 1) or 1)
        pad_x = int(meta.get("pad_x", 0) or 0)
        pad_y = int(meta.get("pad_y", 0) or 0)
        camera_x = int(meta.get("x", 0) or 0)
        camera_y = int(meta.get("y", 0) or 0)
    except Exception:
        return int(row), int(col)
    display_x = int(pad_x + ((int(col) - camera_x) * scale) + (scale // 2))
    display_y = int(pad_y + ((int(row) - camera_y) * scale) + (scale // 2))
    return display_x, display_y


def action_target_to_grid_cell(action: ActionName, extras: Mapping[str, Any] | None = None) -> tuple[int, int] | None:
    """Parse an action target and map adapter click coordinates back to grid cells."""

    target = parse_action_target(action)
    if target is None:
        return None
    family = action_family(action)
    meta = (extras or {}).get("camera_meta") if isinstance(extras, Mapping) else None
    if family not in {"click", "action6"} or not isinstance(meta, Mapping):
        return target
    try:
        x, y = target
        scale = max(int(meta.get("scale", 1) or 1), 1)
        pad_x = int(meta.get("pad_x", 0) or 0)
        pad_y = int(meta.get("pad_y", 0) or 0)
        camera_x = int(meta.get("x", 0) or 0)
        camera_y = int(meta.get("y", 0) or 0)
        grid_col = int(round((int(x) - pad_x - (scale // 2)) / scale + camera_x))
        grid_row = int(round((int(y) - pad_y - (scale // 2)) / scale + camera_y))
        return grid_row, grid_col
    except Exception:
        return target


@dataclass(frozen=True)
class GridFrame:
    task_id: str
    episode_id: str
    step_index: int
    grid: GridArray
    available_actions: tuple[ActionName, ...] = ()
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", np.asarray(self.grid, dtype=np.int64))
        object.__setattr__(self, "available_actions", tuple(normalize_action(a) for a in self.available_actions))

    @property
    def shape(self) -> tuple[int, int]:
        if self.grid.ndim != 2:
            raise ValueError(f"GridFrame expects a rank-2 grid, got shape {self.grid.shape!r}")
        return int(self.grid.shape[0]), int(self.grid.shape[1])

    @property
    def fingerprint(self) -> str:
        shape = "x".join(map(str, self.grid.shape))
        payload = self.grid.astype(np.int64, copy=False).tobytes()
        return _stable_digest(shape + ":" + blake2b(payload, digest_size=16).hexdigest())


@dataclass(frozen=True)
class ObjectToken:
    object_id: str
    color: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    cells: tuple[tuple[int, int], ...]
    shape_hash: str
    role_tags: tuple[str, ...] = ()

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def signature(self) -> str:
        """Object identity without absolute position.

        Absolute positions are intentionally excluded.  The signature is used for
        online rule transfer within a game family: "the same color/shape kind"
        rather than "the same instance at row 3, col 4".
        """

        tags = ",".join(sorted(self.role_tags))
        return f"c{self.color}:a{self.area}:h{self.height}:w{self.width}:s{self.shape_hash}:t{tags}"

    @property
    def center_cell(self) -> tuple[int, int]:
        return int(round(self.centroid[0])), int(round(self.centroid[1]))

    def distance_to(self, other: "ObjectToken") -> float:
        return abs(self.centroid[0] - other.centroid[0]) + abs(self.centroid[1] - other.centroid[1])


@dataclass(frozen=True)
class RelationToken:
    relation: str
    subject_id: str
    object_id: str
    value: float = 1.0

    def as_key(self) -> str:
        return f"{self.relation}:{self.subject_id}:{self.object_id}:{round(float(self.value), 3)}"


@dataclass(frozen=True)
class StructuredState:
    frame: GridFrame
    objects: tuple[ObjectToken, ...]
    relations: tuple[RelationToken, ...]
    dominant_color: int
    abstract_fingerprint: str
    exact_fingerprint: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def grid(self) -> GridArray:
        return self.frame.grid

    @property
    def available_actions(self) -> tuple[ActionName, ...]:
        return self.frame.available_actions

    @property
    def step_index(self) -> int:
        return self.frame.step_index

    def object_by_id(self, object_id: str) -> ObjectToken | None:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def objects_by_color(self, color: int) -> tuple[ObjectToken, ...]:
        return tuple(obj for obj in self.objects if obj.color == color)

    def role_objects(self, tag: str) -> tuple[ObjectToken, ...]:
        return tuple(obj for obj in self.objects if tag in obj.role_tags)

    def token_set(self) -> frozenset[str]:
        tokens: set[str] = {
            f"shape:{self.grid.shape[0]}x{self.grid.shape[1]}",
            f"dominant:c{self.dominant_color}",
            f"objects:{len(self.objects)}",
        }
        for obj in self.objects:
            tokens.add(f"object:c{obj.color}:area_bin:{min(obj.area // 2, 12)}")
            tokens.add(f"shape_hash:{obj.shape_hash}")
            for tag in obj.role_tags:
                tokens.add(f"role:{tag}:c{obj.color}")
        for rel in self.relations[:128]:
            tokens.add(f"rel:{rel.relation}")
        return frozenset(tokens)


@dataclass(frozen=True)
class ObjectMotion:
    before_id: str
    after_id: str
    before_signature: str
    after_signature: str
    color: int
    delta: tuple[int, int]
    distance: float


@dataclass(frozen=True)
class TransitionDelta:
    action: ActionName
    changed_cells: int
    changed_fraction: float
    moved_objects: tuple[ObjectMotion, ...]
    appeared: tuple[str, ...]
    disappeared: tuple[str, ...]
    touched_objects: tuple[str, ...]
    reward: float
    score_delta: float
    terminated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_visible_effect(self) -> bool:
        return self.changed_cells > 0 or bool(self.moved_objects or self.appeared or self.disappeared)

    @property
    def is_positive(self) -> bool:
        return self.reward > 0.0 or self.score_delta > 0.0


@dataclass(frozen=True)
class TransitionRecord:
    before: StructuredState
    after: StructuredState
    delta: TransitionDelta

    @property
    def action(self) -> ActionName:
        return self.delta.action

    @property
    def reward(self) -> float:
        return self.delta.reward

    @property
    def step_index(self) -> int:
        return self.before.step_index


@dataclass(frozen=True)
class ActionDecision:
    action: ActionName
    score: float
    components: Mapping[str, float]
    language: tuple[str, ...]
    candidate_count: int
    chosen_reason: str




def combined_progress_signal(reward: float, score_delta: float) -> float:
    """Combine environment reward and score delta without double-counting.

    Many wrappers expose the same progress twice: once as ``reward`` and once as
    ``info["score_delta"]``.  When the two signals are numerically identical,
    treating them as independent evidence doubles the reward and corrupts online
    calibration.  When they differ, preserve both because ARC-like wrappers may
    use shaped reward plus score progress.
    """

    reward_f = float(reward)
    score_f = float(score_delta)
    if abs(reward_f) > 1e-12 and abs(score_f) > 1e-12 and abs(reward_f - score_f) <= 1e-9:
        return reward_f
    return reward_f + score_f


def coerce_grid_frame(
    observation: Any,
    *,
    task_id: str | None = None,
    episode_id: str | None = None,
    step_index: int | None = None,
    available_actions: Sequence[Any] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> GridFrame:
    """Convert common ARC-like observation objects into ``GridFrame``.

    The function is deliberately permissive so the agent can be called from the
    current ``arcagi`` harness, the official toolkit, or a raw unit-test grid.
    """

    if isinstance(observation, GridFrame):
        return observation

    if isinstance(observation, np.ndarray) or isinstance(observation, list):
        grid = np.asarray(observation, dtype=np.int64)
        obs_task_id = task_id or "raw_grid"
        obs_episode_id = episode_id or f"{obs_task_id}/episode"
        obs_step_index = 0 if step_index is None else int(step_index)
        actions = tuple(normalize_action(a) for a in (available_actions or ()))
        return GridFrame(obs_task_id, obs_episode_id, obs_step_index, grid, actions, extras or {})

    grid = getattr(observation, "grid", None)
    if grid is None:
        grid = getattr(observation, "frame", None)
    if grid is None:
        grid = getattr(observation, "observation", None)
    if grid is None:
        raise TypeError(f"Cannot coerce observation of type {type(observation)!r} into GridFrame")

    obs_task_id = task_id or str(getattr(observation, "task_id", "arc_task"))
    obs_episode_id = episode_id or str(getattr(observation, "episode_id", f"{obs_task_id}/episode"))
    obs_step_index = int(step_index if step_index is not None else getattr(observation, "step_index", 0))

    if available_actions is None:
        available_actions = getattr(observation, "available_actions", ()) or getattr(observation, "actions", ()) or ()
    obs_extras = dict(getattr(observation, "extras", {}) or {})
    if extras:
        obs_extras.update(extras)
    return GridFrame(
        task_id=obs_task_id,
        episode_id=obs_episode_id,
        step_index=obs_step_index,
        grid=np.asarray(grid, dtype=np.int64),
        available_actions=tuple(normalize_action(a) for a in available_actions),
        extras=obs_extras,
    )
