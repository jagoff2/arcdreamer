from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from .arcagi3_adapter import (
    CELL_AGENT,
    CELL_EMPTY,
    MOVE_DELTAS,
    ArcAGI3Observation,
    ArcAGI3StepResult,
    action_family,
    parse_click,
)


def stable_hash(value: Any) -> str:
    encoded = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    return repr(value)


def observation_hash(observation: ArcAGI3Observation) -> str:
    grid = np.asarray(observation.grid, dtype=np.int64)
    return stable_hash({"grid": grid, "actions": observation.available_actions})


def visual_hash(observation: ArcAGI3Observation) -> str:
    grid = np.asarray(observation.grid, dtype=np.int64)
    return stable_hash({"grid": grid})


@dataclass(frozen=True)
class Component:
    component_id: int
    value: int
    area: int
    centroid_y: float
    centroid_x: float
    min_y: int
    min_x: int
    max_y: int
    max_x: int

    def center_yx(self) -> tuple[float, float]:
        return self.centroid_y, self.centroid_x

    def border_points(self) -> list[tuple[int, int]]:
        return [
            (self.min_y, self.min_x),
            (self.min_y, self.max_x),
            (self.max_y, self.min_x),
            (self.max_y, self.max_x),
            ((self.min_y + self.max_y) // 2, self.min_x),
            ((self.min_y + self.max_y) // 2, self.max_x),
            (self.min_y, (self.min_x + self.max_x) // 2),
            (self.max_y, (self.min_x + self.max_x) // 2),
        ]

    def compact(self) -> dict[str, Any]:
        return {
            "id": self.component_id,
            "value": self.value,
            "area": self.area,
            "centroid": [round(self.centroid_y, 3), round(self.centroid_x, 3)],
            "bbox": [self.min_y, self.min_x, self.max_y, self.max_x],
        }


def extract_components(observation: ArcAGI3Observation) -> list[Component]:
    grid = np.asarray(observation.grid, dtype=np.int64)
    mask = (grid != CELL_EMPTY) & (grid != CELL_AGENT)
    visited = np.zeros(grid.shape, dtype=bool)
    components: list[Component] = []
    height, width = grid.shape
    component_id = 0
    for y0, x0 in np.argwhere(mask):
        y = int(y0)
        x = int(x0)
        if visited[y, x]:
            continue
        value = int(grid[y, x])
        stack = [(y, x)]
        visited[y, x] = True
        cells: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            cells.append((cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width and not visited[ny, nx] and mask[ny, nx] and int(grid[ny, nx]) == value:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        ys = [item[0] for item in cells]
        xs = [item[1] for item in cells]
        components.append(
            Component(
                component_id=component_id,
                value=value,
                area=len(cells),
                centroid_y=float(sum(ys) / max(len(ys), 1)),
                centroid_x=float(sum(xs) / max(len(xs), 1)),
                min_y=min(ys),
                min_x=min(xs),
                max_y=max(ys),
                max_x=max(xs),
            )
        )
        component_id += 1
    components.sort(key=lambda item: (-item.area, item.value, item.component_id))
    return components


def component_summary(observation: ArcAGI3Observation, *, limit: int = 12) -> dict[str, Any]:
    components = extract_components(observation)
    values = Counter(int(item.value) for item in components)
    return {
        "obs_hash": observation_hash(observation),
        "component_count": len(components),
        "value_counts": {str(key): int(value) for key, value in sorted(values.items())},
        "largest": [item.compact() for item in components[:limit]],
    }


def changed_regions(before: ArcAGI3Observation, after: ArcAGI3Observation, *, limit: int = 8) -> dict[str, Any]:
    old = np.asarray(before.grid, dtype=np.int64)
    new = np.asarray(after.grid, dtype=np.int64)
    if old.shape != new.shape:
        return {"pixel_count": int(new.size), "regions": [{"bbox": [0, 0, int(new.shape[0] - 1), int(new.shape[1] - 1)]}]}
    diff = old != new
    if not np.any(diff):
        return {"pixel_count": 0, "regions": []}
    visited = np.zeros(diff.shape, dtype=bool)
    regions: list[dict[str, Any]] = []
    height, width = diff.shape
    for y0, x0 in np.argwhere(diff):
        y = int(y0)
        x = int(x0)
        if visited[y, x]:
            continue
        stack = [(y, x)]
        visited[y, x] = True
        cells: list[tuple[int, int]] = []
        while stack:
            cy, cx = stack.pop()
            cells.append((cy, cx))
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width and diff[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
        ys = [item[0] for item in cells]
        xs = [item[1] for item in cells]
        regions.append(
            {
                "area": len(cells),
                "centroid": [round(float(sum(ys) / len(ys)), 3), round(float(sum(xs) / len(xs)), 3)],
                "bbox": [min(ys), min(xs), max(ys), max(xs)],
            }
        )
    regions.sort(key=lambda item: -int(item["area"]))
    return {"pixel_count": int(np.count_nonzero(diff)), "regions": regions[:limit]}


def action_target(action: str) -> tuple[int, int] | None:
    click = parse_click(action)
    if click is None:
        return None
    x, y = click
    return int(y), int(x)


def distance_score(point: tuple[int | float, int | float], target: tuple[int | float, int | float], scale: float) -> float:
    dy = float(point[0]) - float(target[0])
    dx = float(point[1]) - float(target[1])
    return 1.0 / (1.0 + math.sqrt(dy * dy + dx * dx) / max(scale, 1.0))


@dataclass
class EffectMemory:
    action_counts: dict[str, int] = field(default_factory=dict)
    action_change: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    action_reward: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    action_events: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    action_no_effect: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    family_reward: dict[str, float] = field(default_factory=lambda: defaultdict(float))
    family_events: dict[str, int] = field(default_factory=lambda: defaultdict(int))
    state_action_next: dict[tuple[str, str], str] = field(default_factory=dict)
    last_changed_regions: dict[str, Any] = field(default_factory=lambda: {"pixel_count": 0, "regions": []})
    afforded_positive_events: int = 0
    positive_events: int = 0
    no_effect_avoidance_checks: int = 0
    no_effect_avoidance_hits: int = 0
    last_choice_was_affordance: bool = False

    def score(self, observation: ArcAGI3Observation, action: str) -> tuple[float, dict[str, Any]]:
        family = action_family(action)
        count = self.action_counts.get(action, 0)
        total_reward = self.action_reward.get(action, 0.0) + 0.35 * self.family_reward.get(family, 0.0)
        event_bonus = float(self.action_events.get(action, 0) + self.family_events.get(family, 0)) * 0.55
        change_bonus = min(self.action_change.get(action, 0.0), 128.0) / 128.0
        no_effect = self.action_no_effect.get(action, 0)
        state_key = visual_hash(observation)
        known_next = self.state_action_next.get((state_key, action))
        loop_penalty = -0.70 if known_next == state_key else 0.0
        novelty = 0.20 / (1.0 + count)
        score = total_reward + event_bonus + 0.40 * change_bonus - 0.22 * no_effect + loop_penalty + novelty
        return score, {
            "family": family,
            "count": count,
            "reward_total": round(total_reward, 6),
            "event_bonus": round(event_bonus, 6),
            "change_bonus": round(change_bonus, 6),
            "no_effect_count": no_effect,
            "known_loop": known_next == state_key,
        }

    def observe(self, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult, *, choice_was_affordance: bool) -> dict[str, Any]:
        after = result.observation
        regions = changed_regions(before, after)
        change_pixels = int(regions["pixel_count"])
        event_count = len(result.info.get("events", []))
        reward = float(result.reward)
        self.action_counts[action] = self.action_counts.get(action, 0) + 1
        self.action_change[action] += float(change_pixels)
        self.action_reward[action] += reward
        self.action_events[action] += event_count
        family = action_family(action)
        self.family_reward[family] += reward
        self.family_events[family] += event_count
        no_effect = change_pixels == 0 and reward <= 0.0 and event_count == 0
        if no_effect:
            self.action_no_effect[action] += 1
        else:
            self.action_no_effect[action] = 0
        self.state_action_next[(visual_hash(before), action)] = visual_hash(after)
        self.last_changed_regions = regions
        positive = reward > 0.0 or event_count > 0
        if positive:
            self.positive_events += 1
            if choice_was_affordance:
                self.afforded_positive_events += 1
        self.last_choice_was_affordance = bool(choice_was_affordance)
        return regions

    def record_choice(self, observation: ArcAGI3Observation, action: str) -> None:
        no_effect_actions = {item for item, count in self.action_no_effect.items() if count > 0 and item in observation.available_actions}
        if no_effect_actions:
            self.no_effect_avoidance_checks += 1
            if action not in no_effect_actions:
                self.no_effect_avoidance_hits += 1

    def snapshot(self, limit: int = 8) -> dict[str, Any]:
        ranked = sorted(
            self.action_counts,
            key=lambda action: (
                self.action_events.get(action, 0),
                self.action_reward.get(action, 0.0),
                self.action_change.get(action, 0.0),
                -self.action_no_effect.get(action, 0),
            ),
            reverse=True,
        )
        return {
            "observed_actions": len(self.action_counts),
            "top_actions": [
                {
                    "action": action,
                    "count": self.action_counts.get(action, 0),
                    "reward": round(float(self.action_reward.get(action, 0.0)), 6),
                    "events": self.action_events.get(action, 0),
                    "change": round(float(self.action_change.get(action, 0.0)), 3),
                    "no_effect": self.action_no_effect.get(action, 0),
                }
                for action in ranked[:limit]
            ],
            "last_changed_regions": self.last_changed_regions,
            "action_effect_hit_rate": self.action_effect_hit_rate(),
            "no_op_avoidance_rate": self.no_op_avoidance_rate(),
        }

    def action_effect_hit_rate(self) -> float:
        return float(self.afforded_positive_events / max(self.positive_events, 1))

    def no_op_avoidance_rate(self) -> float:
        return float(self.no_effect_avoidance_hits / max(self.no_effect_avoidance_checks, 1))


@dataclass
class StateGraphMemory:
    seen_states: set[str] = field(default_factory=set)
    edges: dict[tuple[str, str], str] = field(default_factory=dict)
    state_action_counts: dict[tuple[str, str], int] = field(default_factory=lambda: defaultdict(int))

    def score(self, observation: ArcAGI3Observation, action: str) -> tuple[float, dict[str, Any]]:
        state = visual_hash(observation)
        self.seen_states.add(state)
        count = self.state_action_counts.get((state, action), 0)
        known_next = self.edges.get((state, action))
        score = 0.45 / (1.0 + count)
        if known_next and known_next not in self.seen_states:
            score += 0.65
        if known_next == state:
            score -= 0.75
        return score, {"state_action_count": count, "known_next_seen": known_next in self.seen_states if known_next else None}

    def observe(self, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult) -> None:
        state = visual_hash(before)
        next_state = visual_hash(result.observation)
        self.seen_states.add(state)
        self.seen_states.add(next_state)
        self.edges[(state, action)] = next_state
        self.state_action_counts[(state, action)] += 1

    def cycle_stats(self, observation: ArcAGI3Observation) -> dict[str, Any]:
        state = visual_hash(observation)
        outgoing = [(action, dst) for (src, action), dst in self.edges.items() if src == state]
        loops = [action for action, dst in outgoing if dst == state]
        return {"seen_states": len(self.seen_states), "known_outgoing": len(outgoing), "known_self_loops": loops[:8]}


@dataclass
class ObjectMemory:
    previous_components: list[Component] = field(default_factory=list)
    persistent_targets: deque[tuple[float, float]] = field(default_factory=lambda: deque(maxlen=32))

    def observe_frame(self, observation: ArcAGI3Observation) -> None:
        components = extract_components(observation)
        previous = self.previous_components
        for component in components:
            for old in previous:
                if component.value == old.value and abs(component.area - old.area) <= max(3, 0.25 * old.area):
                    distance = math.sqrt((component.centroid_y - old.centroid_y) ** 2 + (component.centroid_x - old.centroid_x) ** 2)
                    if distance <= 8.0:
                        self.persistent_targets.append(component.center_yx())
                        break
        self.previous_components = components

    def score(self, action: str) -> tuple[float, dict[str, Any]]:
        target = action_target(action)
        if target is None or not self.persistent_targets:
            return 0.0, {"persistent_target_count": len(self.persistent_targets)}
        best = max(distance_score(target, item, 10.0) for item in self.persistent_targets)
        return 0.40 * best, {"persistent_target_count": len(self.persistent_targets), "best_target_score": round(best, 6)}


@dataclass
class SearchWeights:
    component: float = 1.00
    change: float = 0.90
    graph: float = 0.85
    event: float = 1.15
    object: float = 0.70


@dataclass
class AffordanceSearchState:
    effect: EffectMemory = field(default_factory=EffectMemory)
    graph: StateGraphMemory = field(default_factory=StateGraphMemory)
    objects: ObjectMemory = field(default_factory=ObjectMemory)
    action_history: list[str] = field(default_factory=list)
    last_choice_reason: dict[str, Any] = field(default_factory=dict)

    def reset(self) -> None:
        self.effect = EffectMemory()
        self.graph = StateGraphMemory()
        self.objects = ObjectMemory()
        self.action_history = []
        self.last_choice_reason = {}

    def score_action(
        self,
        observation: ArcAGI3Observation,
        action: str,
        *,
        enable_component: bool,
        enable_change: bool,
        enable_graph: bool,
        enable_event: bool,
        enable_object: bool,
        weights: SearchWeights,
    ) -> tuple[float, dict[str, Any]]:
        legal = tuple(observation.available_actions)
        legal_rank = legal.index(action) if action in legal else len(legal)
        score = -0.0001 * legal_rank
        detail: dict[str, Any] = {"legal_rank_bias": round(-0.0001 * legal_rank, 6)}
        if enable_component:
            component_score, component_detail = self.component_score(observation, action)
            score += weights.component * component_score
            detail["component"] = component_detail
        if enable_change:
            change_score, change_detail = self.change_score(action)
            score += weights.change * change_score
            detail["change"] = change_detail
        if enable_graph:
            graph_score, graph_detail = self.graph.score(observation, action)
            score += weights.graph * graph_score
            detail["graph"] = graph_detail
        if enable_event:
            event_score, event_detail = self.effect.score(observation, action)
            score += weights.event * event_score
            detail["event"] = event_detail
        if enable_object:
            object_score, object_detail = self.objects.score(action)
            score += weights.object * object_score
            detail["object"] = object_detail
        if action in self.action_history[-3:]:
            score -= 0.35
            detail["recent_repeat_penalty"] = True
        return score, detail

    def component_score(self, observation: ArcAGI3Observation, action: str) -> tuple[float, dict[str, Any]]:
        target = action_target(action)
        components = extract_components(observation)
        if target is None:
            movement_bonus = 0.0
            if action in MOVE_DELTAS:
                movement_bonus = 0.18
            return movement_bonus, {"component_count": len(components), "movement_probe": bool(movement_bonus)}
        if not components:
            return 0.0, {"component_count": 0}
        height, width = np.asarray(observation.grid).shape
        scale = max(height, width) / 5.0
        best = 0.0
        best_id = None
        best_kind = "none"
        for component in components:
            candidates: list[tuple[str, tuple[float, float]]] = [("centroid", component.center_yx())]
            candidates.extend(("border", item) for item in component.border_points())
            for kind, point in candidates:
                value = distance_score(target, point, scale)
                size_bonus = min(component.area, 512) / 512.0
                adjusted = value + 0.20 * size_bonus
                if adjusted > best:
                    best = adjusted
                    best_id = component.component_id
                    best_kind = kind
        return best, {"component_count": len(components), "target_component": best_id, "target_kind": best_kind, "raw_score": round(best, 6)}

    def change_score(self, action: str) -> tuple[float, dict[str, Any]]:
        target = action_target(action)
        regions = self.effect.last_changed_regions.get("regions", [])
        if target is None or not regions:
            return 0.0, {"changed_region_count": len(regions)}
        best = 0.0
        for region in regions:
            centroid = region.get("centroid", [0.0, 0.0])
            best = max(best, distance_score(target, (float(centroid[0]), float(centroid[1])), 8.0))
        return best, {"changed_region_count": len(regions), "best_changed_region_score": round(best, 6)}

    def observe(self, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult, *, choice_was_affordance: bool) -> dict[str, Any]:
        self.action_history.append(action)
        regions = self.effect.observe(before, action, result, choice_was_affordance=choice_was_affordance)
        self.graph.observe(before, action, result)
        self.objects.observe_frame(result.observation)
        return regions

    def diagnostics(self, observation: ArcAGI3Observation) -> dict[str, Any]:
        return {
            "component_summary": component_summary(observation),
            "action_effect_memory": self.effect.snapshot(),
            "repeat_cycle_stats": {
                **self.graph.cycle_stats(observation),
                "recent_actions": self.action_history[-12:],
                "recent_repeat_fraction": repeat_fraction(self.action_history[-32:]),
            },
        }


def repeat_fraction(actions: list[str]) -> float:
    if not actions:
        return 0.0
    counts = Counter(actions)
    return float(max(counts.values()) / len(actions))
