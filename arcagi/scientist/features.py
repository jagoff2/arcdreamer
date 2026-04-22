"""Hashed numeric features used by online memory and world modeling."""

from __future__ import annotations

from hashlib import blake2b
from typing import Iterable

import numpy as np

from .types import ActionName, StructuredState, TransitionRecord, action_family, combined_progress_signal


def stable_index(text: str, modulo: int) -> int:
    digest = blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % modulo


def add_hash_feature(vec: np.ndarray, token: str, value: float = 1.0) -> None:
    idx = stable_index(token, int(vec.shape[0]))
    sign = 1.0 if stable_index("sign:" + token, 2) == 0 else -1.0
    vec[idx] += sign * float(value)


def normalize(vec: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return vec.astype(np.float32, copy=False)
    return (vec / norm).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-8) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= eps:
        return 0.0
    return float(np.dot(a, b) / denom)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb)) / float(len(sa | sb))


def state_features(state: StructuredState, *, dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    rows, cols = state.grid.shape
    vec[0] = float(rows) / 64.0
    vec[1] = float(cols) / 64.0
    vec[2] = float(len(state.objects)) / 32.0
    vec[3] = float(len(state.relations)) / 256.0
    vec[4] = float(state.dominant_color) / 32.0
    color_counts: dict[int, int] = {}
    for obj in state.objects:
        color_counts[obj.color] = color_counts.get(obj.color, 0) + 1
        add_hash_feature(vec, f"obj_color:c{obj.color}", 0.20)
        add_hash_feature(vec, f"obj_shape:{obj.shape_hash}", 0.10)
        add_hash_feature(vec, f"obj_area_bin:{min(obj.area // 2, 16)}", 0.15)
        add_hash_feature(vec, f"obj_bbox:{min(obj.height,16)}x{min(obj.width,16)}", 0.12)
        for tag in obj.role_tags:
            add_hash_feature(vec, f"role:{tag}:c{obj.color}", 0.15)
    for color, count in color_counts.items():
        add_hash_feature(vec, f"color_count:c{color}:{min(count, 8)}", 0.25)
    for rel in state.relations[:128]:
        add_hash_feature(vec, f"rel:{rel.relation}", 0.08)
    return normalize(vec)


def action_features(action: ActionName, *, dim: int = 64) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    family = action_family(action)
    add_hash_feature(vec, f"action:{family}", 1.0)
    add_hash_feature(vec, f"raw:{action.lower()}", 0.35)
    return normalize(vec)


def state_action_features(state: StructuredState, action: ActionName, *, dim: int = 320) -> np.ndarray:
    sdim = dim - 64
    return np.concatenate([state_features(state, dim=sdim), action_features(action, dim=64)]).astype(np.float32)


def transition_target_features(record: TransitionRecord) -> tuple[float, float]:
    """Return normalized online learning targets: reward and visible change."""

    reward = float(combined_progress_signal(record.delta.reward, record.delta.score_delta))
    change = min(1.0, float(record.delta.changed_fraction) * 4.0 + 0.1 * len(record.delta.moved_objects))
    return reward, change
