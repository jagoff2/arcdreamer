from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .arcagi3_adapter import (
    MOVE_DELTAS,
    ArcAGI3Observation,
    ArcAGI3StepResult,
    action_family,
    find_agent,
    in_bounds,
    parse_click,
)


Cell = tuple[int, int]


@dataclass(frozen=True)
class Region:
    region_id: str
    value: int
    cells: tuple[Cell, ...]
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    changed_count: int
    track_id: int | None = None
    stable_frames: int = 1

    @property
    def area(self) -> int:
        return len(self.cells)

    def compact(self) -> dict[str, Any]:
        return {
            "region_id": self.region_id,
            "track_id": self.track_id,
            "value": self.value,
            "area": self.area,
            "bbox": list(self.bbox),
            "centroid": [round(float(self.centroid[0]), 3), round(float(self.centroid[1]), 3)],
            "changed_count": self.changed_count,
            "stable_frames": self.stable_frames,
        }


@dataclass
class PerceptionFrame:
    shape: tuple[int, int]
    background: int
    regions: list[Region]
    changed_count: int
    changed_regions: int
    click_regions: list[dict[str, Any]]

    def region_at(self, y: int, x: int) -> Region | None:
        for region in self.regions:
            if (int(y), int(x)) in region.cells:
                return region
        return None

    def compact(self) -> dict[str, Any]:
        stable = sum(1 for region in self.regions if region.stable_frames >= 3)
        return {
            "shape": list(self.shape),
            "background": self.background,
            "component_count": len(self.regions),
            "stable_tracks": stable,
            "click_regions": len(self.click_regions),
            "changed_regions": self.changed_regions,
            "changed_count": self.changed_count,
            "regions_sample": [region.compact() for region in self.regions[:8]],
            "click_regions_sample": self.click_regions[:8],
        }


@dataclass
class Track:
    track_id: int
    value: int
    cells: frozenset[Cell]
    centroid: tuple[float, float]
    stable_frames: int = 1
    seen_frames: int = 1


@dataclass
class EffectStats:
    count: int = 0
    change_count: int = 0
    reward_sum: float = 0.0
    no_effect_count: int = 0
    terminal_count: int = 0
    value_ewma: float = 0.0
    change_ewma: float = 0.0
    alpha: float = 0.35

    def update(self, *, changed: bool, reward: float, no_effect: bool, terminal: bool) -> None:
        value = float(reward) + (0.06 if changed else 0.0) - (0.08 if no_effect else 0.0)
        self.count += 1
        self.change_count += int(changed)
        self.reward_sum += float(reward)
        self.no_effect_count += int(no_effect)
        self.terminal_count += int(terminal)
        self.value_ewma = (1.0 - self.alpha) * self.value_ewma + self.alpha * value
        self.change_ewma = (1.0 - self.alpha) * self.change_ewma + self.alpha * float(changed)

    def score(self) -> float:
        if self.count == 0:
            return 0.0
        reward_rate = self.reward_sum / max(self.count, 1)
        no_effect_rate = self.no_effect_count / max(self.count, 1)
        return float(0.70 * self.value_ewma + 0.20 * self.change_ewma + 0.10 * reward_rate - 0.08 * no_effect_rate)

    def compact(self) -> dict[str, Any]:
        return {
            "count": self.count,
            "change_rate": self.change_count / max(self.count, 1),
            "mean_reward": self.reward_sum / max(self.count, 1),
            "no_effect_rate": self.no_effect_count / max(self.count, 1),
            "terminal_count": self.terminal_count,
            "value_ewma": round(float(self.value_ewma), 6),
            "change_ewma": round(float(self.change_ewma), 6),
            "score": round(float(self.score()), 6),
        }


def background_value(grid: np.ndarray) -> int:
    counts = Counter(int(item) for item in np.asarray(grid, dtype=np.int64).reshape(-1))
    return int(counts.most_common(1)[0][0]) if counts else 0


def region_digest(value: int, cells: tuple[Cell, ...]) -> str:
    payload = {"value": int(value), "cells": cells}
    return hashlib.sha256(json.dumps(payload, separators=(",", ":")).encode("utf-8")).hexdigest()[:12]


def connected_regions(grid: np.ndarray, changed: np.ndarray | None = None) -> list[Region]:
    arr = np.asarray(grid, dtype=np.int64)
    bg = background_value(arr)
    seen = np.zeros(arr.shape, dtype=bool)
    changed_mask = np.zeros(arr.shape, dtype=bool) if changed is None else np.asarray(changed, dtype=bool)
    regions: list[Region] = []
    height, width = arr.shape
    for y0 in range(height):
        for x0 in range(width):
            if seen[y0, x0] or int(arr[y0, x0]) == bg:
                continue
            value = int(arr[y0, x0])
            stack = [(y0, x0)]
            seen[y0, x0] = True
            cells: list[Cell] = []
            while stack:
                y, x = stack.pop()
                cells.append((int(y), int(x)))
                for dy, dx in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
                    ny, nx = y + dy, x + dx
                    if in_bounds(arr, ny, nx) and not seen[ny, nx] and int(arr[ny, nx]) == value:
                        seen[ny, nx] = True
                        stack.append((ny, nx))
            cells_tuple = tuple(sorted(cells))
            ys = [cell[0] for cell in cells_tuple]
            xs = [cell[1] for cell in cells_tuple]
            regions.append(
                Region(
                    region_id=region_digest(value, cells_tuple),
                    value=value,
                    cells=cells_tuple,
                    bbox=(min(ys), min(xs), max(ys), max(xs)),
                    centroid=(float(sum(ys) / len(ys)), float(sum(xs) / len(xs))),
                    changed_count=int(sum(1 for y, x in cells_tuple if bool(changed_mask[y, x]))),
                )
            )
    return regions


def extract_regions(observation: ArcAGI3Observation, previous_grid: np.ndarray | None = None) -> PerceptionFrame:
    grid = np.asarray(observation.grid, dtype=np.int64)
    changed = np.zeros(grid.shape, dtype=bool)
    if previous_grid is not None and np.asarray(previous_grid).shape == grid.shape:
        changed = np.asarray(previous_grid, dtype=np.int64) != grid
    regions = connected_regions(grid, changed)
    click_regions: list[dict[str, Any]] = []
    for action in observation.available_actions:
        click = parse_click(action)
        if click is None:
            continue
        x, y = click
        if not in_bounds(grid, y, x):
            continue
        region = next((item for item in regions if (int(y), int(x)) in item.cells), None)
        click_regions.append(
            {
                "action": action,
                "y": int(y),
                "x": int(x),
                "value": int(grid[y, x]),
                "region_id": region.region_id if region else None,
                "changed": bool(changed[y, x]),
            }
        )
    return PerceptionFrame(
        shape=tuple(int(item) for item in grid.shape),
        background=background_value(grid),
        regions=regions,
        changed_count=int(np.count_nonzero(changed)),
        changed_regions=int(sum(region.changed_count > 0 for region in regions)),
        click_regions=click_regions,
    )


def target_cell_for_action(observation: ArcAGI3Observation, action: str) -> Cell | None:
    grid = np.asarray(observation.grid, dtype=np.int64)
    click = parse_click(action)
    if click is not None:
        x, y = click
        return (int(y), int(x)) if in_bounds(grid, y, x) else None
    agent = find_agent(grid)
    if action in MOVE_DELTAS and agent is not None:
        dy, dx = MOVE_DELTAS[action]
        y, x = agent[0] + dy, agent[1] + dx
        return (int(y), int(x)) if in_bounds(grid, y, x) else None
    return None


def grid_changed(before: ArcAGI3Observation, after: ArcAGI3Observation) -> bool:
    return not np.array_equal(np.asarray(before.grid, dtype=np.int64), np.asarray(after.grid, dtype=np.int64))


@dataclass
class OnlinePerceptualAffordance:
    effect_alpha: float = 0.35
    tracks: dict[int, Track] = field(default_factory=dict)
    effects: dict[tuple[str, str], EffectStats] = field(default_factory=dict)
    next_track_id: int = 1
    last_grid: np.ndarray | None = None
    current_frame: PerceptionFrame | None = None
    frames_seen: int = 0
    prediction_trials: int = 0
    prediction_correct: int = 0
    null_prediction_correct: int = 0
    perturbation_trials: int = 0
    perturbation_changed: int = 0

    def reset(self) -> None:
        self.tracks.clear()
        self.effects.clear()
        self.next_track_id = 1
        self.last_grid = None
        self.current_frame = None
        self.frames_seen = 0
        self.prediction_trials = 0
        self.prediction_correct = 0
        self.null_prediction_correct = 0
        self.perturbation_trials = 0
        self.perturbation_changed = 0

    def perceive(self, observation: ArcAGI3Observation) -> PerceptionFrame:
        frame = extract_regions(observation, self.last_grid)
        tracked = self._assign_tracks(frame.regions)
        frame = PerceptionFrame(
            shape=frame.shape,
            background=frame.background,
            regions=tracked,
            changed_count=frame.changed_count,
            changed_regions=frame.changed_regions,
            click_regions=frame.click_regions,
        )
        self.current_frame = frame
        self.last_grid = np.asarray(observation.grid, dtype=np.int64).copy()
        self.frames_seen += 1
        return frame

    def action_region_key(self, observation: ArcAGI3Observation, action: str, frame: PerceptionFrame | None = None) -> str:
        frame = frame or self.current_frame or self.perceive(observation)
        grid = np.asarray(observation.grid, dtype=np.int64)
        target = target_cell_for_action(observation, action)
        family = action_family(action)
        if target is None:
            return f"{family}:no_target"
        y, x = target
        value = int(grid[y, x])
        region = frame.region_at(y, x)
        if region is not None:
            stable_bucket = min(4, int(region.stable_frames))
            return f"{family}:region:{region.value}:area:{min(region.area, 16)}:stable:{stable_bucket}"
        role = "background" if value == frame.background else "cell"
        return f"{family}:{role}:{value}"

    def score_action(self, observation: ArcAGI3Observation, action: str, enabled: set[str]) -> dict[str, float]:
        frame = self.current_frame or self.perceive(observation)
        target = target_cell_for_action(observation, action)
        component = 0.0
        temporal = 0.0
        effect = 0.0
        prediction = 0.0
        if target is not None:
            y, x = target
            region = frame.region_at(y, x)
            if region is not None:
                component += 0.10 + min(region.area, 12) * 0.004
                component += 0.04 if region.changed_count > 0 else 0.0
                temporal += min(region.stable_frames, 5) * 0.025
            else:
                grid = np.asarray(observation.grid, dtype=np.int64)
                component += -0.015 if int(grid[y, x]) == frame.background else 0.025
        key = self.action_region_key(observation, action, frame)
        stats = self.effects.get((action_family(action), key)) or self.effects.get(("*", key))
        if stats is not None:
            effect = stats.score()
            prediction = 0.10 * stats.change_ewma
        return {
            "component": component if "component" in enabled else 0.0,
            "temporal": temporal if "temporal" in enabled else 0.0,
            "effect": effect if "effect" in enabled else 0.0,
            "prediction": prediction if "prediction" in enabled else 0.0,
        }

    def observe_transition(self, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult) -> dict[str, Any]:
        before_frame = extract_regions(before, None)
        key = self.action_region_key(before, action, before_frame)
        family = action_family(action)
        stats_key = (family, key)
        stats = self.effects.get(stats_key)
        changed = grid_changed(before, result.observation)
        predicted_changed = bool(stats is not None and stats.change_ewma >= 0.5)
        self.prediction_trials += 1
        self.prediction_correct += int(predicted_changed == changed)
        self.null_prediction_correct += int(not changed)
        events = list(result.info.get("events", []))
        no_effect = float(result.reward) <= 0.0 and not changed and not events
        terminal = bool(result.terminated or result.truncated)
        stats = self.effects.setdefault(stats_key, EffectStats(alpha=self.effect_alpha))
        stats.update(changed=changed, reward=float(result.reward), no_effect=no_effect, terminal=terminal)
        wildcard = self.effects.setdefault(("*", key), EffectStats(alpha=self.effect_alpha))
        wildcard.update(changed=changed, reward=float(result.reward), no_effect=no_effect, terminal=terminal)
        return {
            "action": action,
            "effect_key": f"{stats_key[0]}|{stats_key[1]}",
            "changed": changed,
            "reward": float(result.reward),
            "no_effect": no_effect,
            "predicted_changed": predicted_changed,
            "prediction_correct": predicted_changed == changed,
            "stats": stats.compact(),
        }

    def record_perturbation(self, changed: bool) -> None:
        self.perturbation_trials += 1
        self.perturbation_changed += int(changed)

    def diagnostics(self) -> dict[str, Any]:
        frame = self.current_frame
        stable_tracks = sum(1 for track in self.tracks.values() if track.stable_frames >= 3)
        return {
            "frames_seen": self.frames_seen,
            "track_count": len(self.tracks),
            "stable_tracks": stable_tracks,
            "component_count": len(frame.regions) if frame else 0,
            "click_regions": len(frame.click_regions) if frame else 0,
            "changed_regions": frame.changed_regions if frame else 0,
            "changed_count": frame.changed_count if frame else 0,
            "effect_entries": len(self.effects),
            "prediction": self.prediction_summary(),
            "affordance_causality": self.perturbation_summary(),
            "top_effects": self.top_effects(),
        }

    def prediction_summary(self) -> dict[str, Any]:
        trials = max(self.prediction_trials, 1)
        model_accuracy = self.prediction_correct / trials
        null_accuracy = self.null_prediction_correct / trials
        return {
            "trials": self.prediction_trials,
            "model_correct": self.prediction_correct,
            "null_correct": self.null_prediction_correct,
            "model_accuracy": model_accuracy,
            "null_accuracy": null_accuracy,
            "above_null": model_accuracy > null_accuracy,
        }

    def perturbation_summary(self) -> dict[str, Any]:
        trials = max(self.perturbation_trials, 1)
        return {
            "trials": self.perturbation_trials,
            "changed_actions": self.perturbation_changed,
            "changed_rate": self.perturbation_changed / trials,
        }

    def top_effects(self, limit: int = 12) -> list[dict[str, Any]]:
        rows = []
        for key, stats in self.effects.items():
            if key[0] == "*":
                continue
            rows.append({"key": f"{key[0]}|{key[1]}", **stats.compact()})
        rows.sort(key=lambda item: (abs(float(item["score"])), item["count"]), reverse=True)
        return rows[:limit]

    def _assign_tracks(self, regions: list[Region]) -> list[Region]:
        assigned_tracks: set[int] = set()
        tracked_regions: list[Region] = []
        updated: dict[int, Track] = {}
        for region in regions:
            best_id: int | None = None
            best_score = -1.0
            region_cells = frozenset(region.cells)
            for track_id, track in self.tracks.items():
                if track_id in assigned_tracks or track.value != region.value:
                    continue
                overlap = len(region_cells & track.cells) / max(len(region_cells | track.cells), 1)
                distance = math.dist(region.centroid, track.centroid)
                score = overlap + max(0.0, 1.5 - distance) * 0.15
                if score > best_score and (overlap > 0.0 or distance <= 2.0):
                    best_id = track_id
                    best_score = score
            if best_id is None:
                best_id = self.next_track_id
                self.next_track_id += 1
                stable = 1
                seen = 1
            else:
                assigned_tracks.add(best_id)
                old = self.tracks[best_id]
                stable = old.stable_frames + 1
                seen = old.seen_frames + 1
            updated[best_id] = Track(
                track_id=best_id,
                value=region.value,
                cells=region_cells,
                centroid=region.centroid,
                stable_frames=stable,
                seen_frames=seen,
            )
            tracked_regions.append(
                Region(
                    region_id=region.region_id,
                    value=region.value,
                    cells=region.cells,
                    bbox=region.bbox,
                    centroid=region.centroid,
                    changed_count=region.changed_count,
                    track_id=best_id,
                    stable_frames=stable,
                )
            )
        self.tracks = updated
        return tracked_regions
