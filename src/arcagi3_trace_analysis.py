from __future__ import annotations

import hashlib
import json
import math
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .arcagi3_adapter import (
    CELL_AGENT,
    CELL_DOOR,
    CELL_EMPTY,
    CELL_GOAL,
    CELL_HAZARD,
    CELL_KEY,
    CELL_RESOURCE,
    CELL_UNKNOWN,
    CELL_WALL,
    CHAR_TO_CELL,
    ARCAGI3Adapter,
    ArcAGI3Observation,
    ArcAGI3StepResult,
    action_family,
    find_agent,
)
from .arcagi3_baselines import build_baselines
from .arcagi3_trace import action_entropy, repeat_collapse


USEFUL_EVENTS = {
    "key_collected",
    "door_opened",
    "goal_reached",
    "goal_clicked",
    "resource_collected",
    "useful_click",
    "level_completed",
}


def load_trace(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_trace(path: str | Path, payload: dict[str, Any]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(payload, indent=2), encoding="utf-8")


def rows_to_grid(rows: Iterable[str]) -> np.ndarray:
    parsed: list[list[int]] = []
    for row in rows:
        parsed.append([CHAR_TO_CELL.get(char, CELL_UNKNOWN) for char in str(row)])
    if not parsed:
        return np.zeros((0, 0), dtype=np.int64)
    width = max(len(row) for row in parsed)
    return np.asarray([row + [CELL_UNKNOWN] * (width - len(row)) for row in parsed], dtype=np.int64)


def observation_from_summary(summary: dict[str, Any]) -> ArcAGI3Observation:
    return ArcAGI3Observation(
        task_id=str(summary.get("task_id", "")),
        episode_id=str(summary.get("episode_id", "")),
        step_index=int(summary.get("step_index", 0)),
        grid=rows_to_grid(summary.get("grid", [])),
        available_actions=tuple(str(item) for item in summary.get("available_actions", [])),
        extras=dict(summary.get("extras", {})),
    )


def result_from_step(step: dict[str, Any]) -> ArcAGI3StepResult:
    next_obs = observation_from_summary(step.get("next_obs", {}))
    events = list(step.get("event_delta", []))
    summary_terminal = any(item in {"goal_reached", "goal_clicked", "game_over", "step_limit"} for item in events)
    info = {
        "events": events,
        "score": float(step.get("score", 0.0)),
        "normalized_score": float(step.get("normalized_score", 0.0)),
        "game_state": next_obs.extras.get("game_state"),
    }
    return ArcAGI3StepResult(
        observation=next_obs,
        reward=float(step.get("score_delta", 0.0)),
        terminated=bool("goal_reached" in events or "goal_clicked" in events),
        truncated=bool(summary_terminal and not ("goal_reached" in events or "goal_clicked" in events)),
        info=info,
    )


def grid_hash(grid: np.ndarray) -> str:
    digest = hashlib.sha256()
    arr = np.asarray(grid, dtype=np.int16)
    digest.update(str(arr.shape).encode("ascii"))
    digest.update(arr.tobytes())
    return digest.hexdigest()[:16]


def cell_counts(grid: np.ndarray) -> dict[str, int]:
    labels = {
        CELL_EMPTY: "empty",
        CELL_WALL: "wall",
        CELL_AGENT: "agent",
        CELL_KEY: "key",
        CELL_DOOR: "door",
        CELL_GOAL: "goal_proxy",
        CELL_HAZARD: "hazard",
        CELL_RESOURCE: "resource_proxy",
        CELL_UNKNOWN: "unknown",
    }
    counts = Counter(int(value) for value in np.asarray(grid).reshape(-1))
    return {name: int(counts.get(code, 0)) for code, name in labels.items()}


def component_count(mask: np.ndarray) -> int:
    mask = np.asarray(mask, dtype=bool)
    if mask.size == 0:
        return 0
    visited = np.zeros(mask.shape, dtype=bool)
    total = 0
    height, width = mask.shape
    for y0, x0 in np.argwhere(mask):
        y = int(y0)
        x = int(x0)
        if visited[y, x]:
            continue
        total += 1
        stack = [(y, x)]
        visited[y, x] = True
        while stack:
            cy, cx = stack.pop()
            for dy, dx in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                ny, nx = cy + dy, cx + dx
                if 0 <= ny < height and 0 <= nx < width and mask[ny, nx] and not visited[ny, nx]:
                    visited[ny, nx] = True
                    stack.append((ny, nx))
    return total


def action_surface(actions: Iterable[str]) -> dict[str, Any]:
    values = [str(action) for action in actions]
    families = Counter(action_family(action) for action in values)
    click_actions = [action for action in values if action.startswith("click:")]
    numeric_actions = [action for action in values if action.isdigit()]
    return {
        "count": len(values),
        "families": dict(families),
        "click_count": len(click_actions),
        "numeric_count": len(numeric_actions),
        "has_click": bool(click_actions),
        "has_keyboard": bool(numeric_actions),
        "sample": values[:16],
    }


def extracted_features(observation: ArcAGI3Observation) -> dict[str, Any]:
    grid = np.asarray(observation.grid, dtype=np.int64)
    counts = cell_counts(grid)
    non_empty = int(grid.size - counts.get("empty", 0))
    foreground = np.isin(grid, [CELL_AGENT, CELL_KEY, CELL_DOOR, CELL_GOAL, CELL_HAZARD, CELL_RESOURCE, CELL_UNKNOWN])
    agent = find_agent(grid)
    return {
        "shape": list(grid.shape),
        "hash": grid_hash(grid),
        "cell_counts": counts,
        "non_empty_cells": non_empty,
        "foreground_fraction": float(non_empty / max(int(grid.size), 1)),
        "component_count": component_count(foreground),
        "goal_components": component_count(grid == CELL_GOAL),
        "resource_components": component_count(grid == CELL_RESOURCE),
        "agent_position": list(agent) if agent is not None else None,
        "action_surface": action_surface(observation.available_actions),
    }


def tensor_stats(tensor: torch.Tensor) -> dict[str, Any]:
    value = tensor.detach().float().cpu()
    flat = value.reshape(-1)
    if flat.numel() == 0:
        return {"shape": list(value.shape), "numel": 0}
    return {
        "shape": list(value.shape),
        "numel": int(flat.numel()),
        "mean": float(flat.mean().item()),
        "std": float(flat.std(unbiased=False).item()),
        "l2": float(torch.linalg.vector_norm(flat).item()),
        "nonzero_fraction": float((flat.abs() > 1.0e-8).float().mean().item()),
        "min": float(flat.min().item()),
        "max": float(flat.max().item()),
    }


def summarize_tensors(tensors: dict[str, torch.Tensor]) -> dict[str, Any]:
    return {key: tensor_stats(value) for key, value in tensors.items()}


def longest_repeated_run(actions: list[str]) -> int:
    best = 0
    current = 0
    previous: str | None = None
    for action in actions:
        if action == previous:
            current += 1
        else:
            previous = action
            current = 1
        best = max(best, current)
    return best


def cycle_stats(steps: list[dict[str, Any]]) -> dict[str, Any]:
    state_hashes = [extracted_features(observation_from_summary(step.get("obs", {})))["hash"] for step in steps]
    counts = Counter(state_hashes)
    actions = [str(step.get("action", "")) for step in steps]
    two_cycle = 0
    for idx in range(2, len(actions)):
        if actions[idx] == actions[idx - 2] and actions[idx] != actions[idx - 1]:
            two_cycle += 1
    return {
        "unique_observation_states": len(counts),
        "duplicate_state_visits": int(sum(count - 1 for count in counts.values() if count > 1)),
        "max_state_visit_count": int(max(counts.values(), default=0)),
        "longest_repeated_action_run": longest_repeated_run(actions),
        "two_action_cycle_rate": float(two_cycle / max(len(actions) - 2, 1)),
    }


def first_useful_event(steps: list[dict[str, Any]]) -> dict[str, Any] | None:
    for step in steps:
        events = [str(item) for item in step.get("event_delta", [])]
        useful = [item for item in events if item in USEFUL_EVENTS]
        if useful:
            return {
                "step": int(step.get("step", 0)),
                "events": useful,
                "action": str(step.get("action", "")),
                "score_delta": float(step.get("score_delta", 0.0)),
                "normalized_score": float(step.get("normalized_score", 0.0)),
            }
    return None


def valid_action_mapping(action: str, observation: ArcAGI3Observation) -> bool:
    if action in {"wait", "noop"}:
        return True
    if action.isdigit():
        return 0 <= int(action) <= 7
    if action.startswith("click:"):
        parts = action.split(":")
        if len(parts) != 3:
            return False
        if not parts[1].lstrip("-").isdigit() or not parts[2].lstrip("-").isdigit():
            return False
        x = int(parts[1])
        y = int(parts[2])
        height, width = observation.grid.shape
        return 0 <= y < height and 0 <= x < width
    return False


def action_audit(steps: list[dict[str, Any]], summary: dict[str, Any]) -> dict[str, Any]:
    legal_seen: set[str] = set()
    invalid_mapping: list[dict[str, Any]] = []
    executed_invalid = 0
    eventless = 0
    executed_no_progress = 0
    for step in steps:
        obs = observation_from_summary(step.get("obs", {}))
        legal = tuple(obs.available_actions)
        legal_seen.update(legal)
        for action in legal:
            if not valid_action_mapping(action, obs):
                invalid_mapping.append({"step": int(step.get("step", 0)), "action": action})
        chosen = str(step.get("action", ""))
        if chosen not in legal or "invalid_action" in step.get("event_delta", []):
            executed_invalid += 1
        no_event = len(step.get("event_delta", [])) == 0
        if no_event:
            eventless += 1
        if no_event and float(step.get("score_delta", 0.0)) <= 0.0:
            executed_no_progress += 1
    step_count = len(steps)
    surface = action_surface(sorted(legal_seen))
    return {
        "reachable_states": int(summary.get("unique_states", 0)),
        "legal_actions_observed": sorted(legal_seen)[:512],
        "legal_actions_observed_count": len(legal_seen),
        "action_surface": surface,
        "mapping_checks": sum(len(observation_from_summary(step.get("obs", {})).available_actions) for step in steps),
        "invalid_mapping_count": len(invalid_mapping),
        "invalid_mapping_examples": invalid_mapping[:8],
        "executed_invalid_count": executed_invalid,
        "executed_invalid_rate": float(executed_invalid / max(step_count, 1)),
        "eventless_step_rate": float(eventless / max(step_count, 1)),
        "no_progress_step_rate": float(executed_no_progress / max(step_count, 1)),
        "no_op_dominance": bool(executed_no_progress / max(step_count, 1) >= 0.85),
    }


def observation_audit(steps: list[dict[str, Any]]) -> dict[str, Any]:
    features = [extracted_features(observation_from_summary(step.get("obs", {}))) for step in steps]
    if not features:
        return {"features_match_visible_state": False, "missing_affordances": ["trace_has_no_steps"]}
    agent_seen = any(item["cell_counts"]["agent"] > 0 for item in features)
    movement_legal = any(item["action_surface"]["has_keyboard"] for item in features)
    max_goal_proxy = max(item["cell_counts"]["goal_proxy"] for item in features)
    max_resource_proxy = max(item["cell_counts"]["resource_proxy"] for item in features)
    max_click = max(item["action_surface"]["click_count"] for item in features)
    foreground = [float(item["foreground_fraction"]) for item in features]
    missing: list[str] = []
    if movement_legal and not agent_seen:
        missing.append("movement_actions_without_agent_marker")
    if max_click > 64:
        missing.append("large_click_surface_without_object_semantics")
    if max_goal_proxy + max_resource_proxy > 512:
        missing.append("dense_pixel_proxy_not_semantic_objects")
    if max(foreground) > 0.25:
        missing.append("foreground_too_dense_for_object_level_parse")
    return {
        "features_match_visible_state": True,
        "steps_checked": len(features),
        "agent_seen": agent_seen,
        "movement_legal": movement_legal,
        "max_goal_proxy_cells": int(max_goal_proxy),
        "max_resource_proxy_cells": int(max_resource_proxy),
        "max_click_actions": int(max_click),
        "mean_foreground_fraction": float(sum(foreground) / max(len(foreground), 1)),
        "missing_affordances": missing,
        "first_step_features": features[0],
    }


def replay_baseline_actions(
    steps: list[dict[str, Any]],
    *,
    seed: int,
) -> list[dict[str, str]]:
    baselines = build_baselines()
    for baseline in baselines:
        baseline.reset(seed)
    rows: list[dict[str, str]] = []
    for step in steps:
        obs = observation_from_summary(step.get("obs", {}))
        result = result_from_step(step)
        baseline_actions: dict[str, str] = {}
        for baseline in baselines:
            action = baseline.choose(obs)
            baseline_actions[baseline.name] = action
            baseline.observe(obs, action, result)
        rows.append(baseline_actions)
    return rows


def replay_adapter_sensitivity(
    steps: list[dict[str, Any]],
    normal_adapter: ARCAGI3Adapter | None,
    corrupt_adapter: ARCAGI3Adapter | None,
) -> dict[str, Any]:
    if normal_adapter is None or corrupt_adapter is None:
        return {"available": False, "reason": "adapter_not_loaded"}
    normal_adapter.reset()
    corrupt_adapter.reset()
    normal_actions: list[str] = []
    corrupt_actions: list[str] = []
    matching_trace = 0
    tensor_samples: list[dict[str, Any]] = []
    for index, step in enumerate(steps):
        obs = observation_from_summary(step.get("obs", {}))
        result = result_from_step(step)
        normal_action, normal_diag = normal_adapter.choose_action(obs)
        corrupt_action, _ = corrupt_adapter.choose_action(obs)
        normal_actions.append(normal_action)
        corrupt_actions.append(corrupt_action)
        if normal_action == str(step.get("action", "")):
            matching_trace += 1
        if index < 4 or index == len(steps) - 1:
            tensors, feature_state = normal_adapter.tensorize(obs)
            tensor_samples.append(
                {
                    "step": int(step.get("step", index)),
                    "tensor_summary": summarize_tensors(tensors),
                    "feature_state": {
                        "semantic_action": feature_state.get("semantic_action"),
                        "target": feature_state.get("target"),
                        "confidence": feature_state.get("confidence"),
                        "drive": feature_state.get("drive"),
                        "hypothesis_state": feature_state.get("hypothesis_state"),
                    },
                    "normal_replay_action": normal_action,
                    "trace_action": str(step.get("action", "")),
                    "trace_policy": step.get("policy", {}),
                    "normal_policy": normal_diag.get("policy", {}),
                }
            )
        normal_adapter.observe_transition(str(step.get("action", "")), result)
        corrupt_adapter.observe_transition(str(step.get("action", "")), result)
    shifts = sum(1 for left, right in zip(normal_actions, corrupt_actions) if left != right)
    return {
        "available": True,
        "steps_replayed": len(steps),
        "normal_trace_match_rate": float(matching_trace / max(len(steps), 1)),
        "corrupt_memory_action_shift_rate": float(shifts / max(len(steps), 1)),
        "normal_action_sample": normal_actions[:24],
        "corrupt_action_sample": corrupt_actions[:24],
        "tensor_samples": tensor_samples,
    }


def memory_goal_planner_audits(
    steps: list[dict[str, Any]],
    sensitivity: dict[str, Any],
) -> dict[str, Any]:
    useful = first_useful_event(steps)
    actions = [str(step.get("action", "")) for step in steps]
    before_after: dict[str, Any] = {"available": False}
    if useful is not None:
        pivot = int(useful["step"])
        before = actions[max(0, pivot - 8) : pivot]
        after = actions[pivot + 1 : pivot + 9]
        before_after = {
            "available": True,
            "pivot_step": pivot,
            "before_actions": before,
            "after_actions": after,
            "action_changed_after_useful_event": bool(after and (not before or Counter(after).most_common(1)[0][0] != Counter(before).most_common(1)[0][0])),
        }
    event_steps = [step for step in steps if step.get("event_delta")]
    memory_records_events = False
    if useful is not None:
        for step in steps[int(useful["step"]) + 1 : int(useful["step"]) + 8]:
            recall = step.get("memory_recall", {})
            if recall.get("resource") or recall.get("goal") or recall.get("key") or recall.get("visited_cells", 0) > 1:
                memory_records_events = True
                break
    planner = {
        "action_entropy": action_entropy(actions),
        "repeat_collapse": repeat_collapse(actions),
        "cycle_stats": cycle_stats(steps),
        "sequence_failure": bool(repeat_collapse(actions) >= 0.85 or (action_entropy(actions) < 0.75 and not event_steps)),
    }
    return {
        "memory_audit": {
            "score_event_delta_steps": len(event_steps),
            "first_useful_event": useful,
            "useful_event_stored_in_memory_window": memory_records_events,
            "corrupt_memory_action_shift_rate": sensitivity.get("corrupt_memory_action_shift_rate"),
        },
        "goal_audit": {
            "first_useful_event": useful,
            "post_useful_event_policy_change": before_after,
            "useful_event_changed_future_policy": bool(before_after.get("action_changed_after_useful_event", False)),
        },
        "planner_audit": planner,
    }


def enrich_trace(
    payload: dict[str, Any],
    *,
    seed: int,
    normal_adapter: ARCAGI3Adapter | None = None,
    corrupt_adapter: ARCAGI3Adapter | None = None,
) -> dict[str, Any]:
    steps = list(payload.get("steps", []))
    baseline_rows = replay_baseline_actions(steps, seed=seed)
    sensitivity = replay_adapter_sensitivity(steps, normal_adapter, corrupt_adapter)
    audits = {
        "observation_audit": observation_audit(steps),
        "action_audit": action_audit(steps, dict(payload.get("summary", {}))),
        **memory_goal_planner_audits(steps, sensitivity),
        "adapter_sensitivity": sensitivity,
    }
    enriched_steps = []
    step_feature_cache = [extracted_features(observation_from_summary(step.get("obs", {}))) for step in steps]
    seen_hashes: set[str] = set()
    unique_so_far: list[int] = []
    for features in step_feature_cache:
        seen_hashes.add(str(features["hash"]))
        unique_so_far.append(len(seen_hashes))
    for index, step in enumerate(steps):
        obs = observation_from_summary(step.get("obs", {}))
        is_last = index == len(steps) - 1
        events = list(step.get("event_delta", []))
        enriched = dict(step)
        enriched["diagnosis"] = {
            "schema": "arcagi3_failure_trace_v1",
            "game_id": payload.get("metadata", {}).get("game_id"),
            "seed": seed,
            "observations_present": "obs" in step and "next_obs" in step,
            "legal_actions": list(obs.available_actions),
            "chosen_action": str(step.get("action", "")),
            "baseline_actions": baseline_rows[index] if index < len(baseline_rows) else {},
            "score_delta": float(step.get("score_delta", 0.0)),
            "event_delta": events,
            "terminal_flag": bool(is_last and payload.get("summary", {}).get("official_state") in {"WIN", "GAME_OVER"}),
            "invalid_action": bool(str(step.get("action", "")) not in obs.available_actions or "invalid_action" in events),
            "unique_states_so_far": unique_so_far[index] if index < len(unique_so_far) else 0,
            "extracted_object_entity_features": step_feature_cache[index] if index < len(step_feature_cache) else extracted_features(obs),
            "adapter_tensor_summary": next(
                (
                    sample["tensor_summary"]
                    for sample in sensitivity.get("tensor_samples", [])
                    if int(sample["step"]) == int(step.get("step", index))
                ),
                {},
            ),
            "z_memory_drive_hypothesis_summary": {
                "memory_recall": step.get("memory_recall", {}),
                "drive": step.get("drive", {}),
                "hypothesis_state": step.get("hypothesis_state", {}),
                "policy": step.get("policy", {}),
            },
        }
        enriched_steps.append(enriched)
    actions = [str(step.get("action", "")) for step in steps]
    payload = dict(payload)
    payload["steps"] = enriched_steps
    payload["diagnosis"] = {
        "schema": "arcagi3_failure_trace_v1",
        "required_trace_fields_present": required_trace_fields_present(enriched_steps),
        "seed": seed,
        "action_entropy": action_entropy(actions),
        "repeat_collapse": repeat_collapse(actions),
        "cycle_stats": cycle_stats(steps),
        "first_useful_event": first_useful_event(steps),
        "audits": audits,
    }
    return payload


def required_trace_fields_present(steps: list[dict[str, Any]]) -> bool:
    if not steps:
        return False
    required = {
        "game_id",
        "seed",
        "observations_present",
        "legal_actions",
        "chosen_action",
        "baseline_actions",
        "score_delta",
        "event_delta",
        "terminal_flag",
        "invalid_action",
        "unique_states_so_far",
        "extracted_object_entity_features",
        "adapter_tensor_summary",
        "z_memory_drive_hypothesis_summary",
    }
    for step in steps:
        diag = step.get("diagnosis", {})
        if not required.issubset(diag):
            return False
    return True


def trace_seed(payload: dict[str, Any], fallback: int = 0) -> int:
    episode = str(payload.get("metadata", {}).get("episode_id", ""))
    if "episode_" in episode:
        tail = episode.rsplit("episode_", 1)[-1]
        if tail.isdigit():
            return int(tail)
    return int(fallback)


def representative_excerpt(payload: dict[str, Any]) -> dict[str, Any]:
    steps = list(payload.get("steps", []))
    if not steps:
        return {"game_id": payload.get("metadata", {}).get("game_id"), "steps": []}
    useful = first_useful_event(steps)
    if useful is not None:
        center = int(useful["step"])
    else:
        center = min(len(steps) - 1, max(0, len(steps) // 2))
    start = max(0, center - 1)
    selected = steps[start : min(len(steps), start + 3)]
    return {
        "game_id": payload.get("metadata", {}).get("game_id"),
        "summary": payload.get("summary", {}),
        "steps": [
            {
                "step": step.get("step"),
                "action": step.get("action"),
                "event_delta": step.get("event_delta"),
                "score_delta": step.get("score_delta"),
                "baseline_actions": step.get("diagnosis", {}).get("baseline_actions", {}),
                "features": step.get("diagnosis", {}).get("extracted_object_entity_features", {}),
                "memory_drive_hypothesis": step.get("diagnosis", {}).get("z_memory_drive_hypothesis_summary", {}),
            }
            for step in selected
        ],
    }


def aggregate_failure_counts(rows: list[dict[str, Any]]) -> dict[str, int]:
    counts: Counter[str] = Counter()
    for row in rows:
        counts[str(row.get("primary_failure_class", "unclassified"))] += 1
        for item in row.get("secondary_failure_classes", []):
            counts[str(item)] += 1
    return dict(sorted(counts.items()))


def mean(values: Iterable[float]) -> float:
    items = list(values)
    return float(sum(items) / max(len(items), 1))
