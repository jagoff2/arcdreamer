from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .arcagi3_adapter import ArcAGI3Observation, ArcAGI3StepResult
from .base_world_model import GRID_SIZE, grid_to_fixed
from .external_eval import json_safe


ACTION_BUCKETS = 512


def stable_hash(value: Any) -> str:
    payload = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()[:16]


def action_id(action: str, buckets: int = ACTION_BUCKETS) -> int:
    digest = hashlib.sha256(str(action).encode("utf-8")).digest()
    return int.from_bytes(digest[:4], "little") % int(buckets)


def observation_frame(observation: ArcAGI3Observation, size: int = GRID_SIZE) -> np.ndarray:
    return grid_to_fixed(np.asarray(observation.grid, dtype=np.int64), size=size)


def legal_action_mask(actions: list[str] | tuple[str, ...], buckets: int = ACTION_BUCKETS) -> list[int]:
    mask = [0] * int(buckets)
    for action in actions:
        mask[action_id(action, buckets)] = 1
    return mask


@dataclass
class AttemptStep:
    step_index: int
    frame: list[list[int]]
    action: str
    legal_actions: list[str]
    score_delta: float
    event_delta: list[str]
    terminal: bool
    obs_hash: str
    next_obs_hash: str
    invalid_action: bool = False
    diagnostics: dict[str, Any] = field(default_factory=dict)


@dataclass
class AttemptRecord:
    suite_id: str
    task_id: str
    variant: str
    split: str
    seed: int
    attempt_index: int
    steps: list[AttemptStep]

    def to_dict(self, *, include_diagnostics: bool = True) -> dict[str, Any]:
        payload = asdict(self)
        if not include_diagnostics:
            for step in payload["steps"]:
                step.pop("diagnostics", None)
        payload["timeline_schema"] = {
            "frame": "fixed 8x8 public observation grid before action",
            "action": "chosen legal action string",
            "legal_actions": "public legal actions before action",
            "score_delta": "public reward or score delta from transition",
            "event_delta": "public event strings from transition",
            "terminal": "terminated or truncated flag after transition",
        }
        return payload

    def save(self, path: str | Path, *, include_diagnostics: bool = True) -> None:
        output = Path(path)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(self.to_dict(include_diagnostics=include_diagnostics), indent=2), encoding="utf-8")


class AttemptBuffer:
    def __init__(
        self,
        *,
        suite_id: str,
        task_id: str,
        variant: str,
        split: str,
        seed: int,
        attempt_index: int,
    ) -> None:
        self.suite_id = suite_id
        self.task_id = task_id
        self.variant = variant
        self.split = split
        self.seed = int(seed)
        self.attempt_index = int(attempt_index)
        self.steps: list[AttemptStep] = []

    def append_transition(
        self,
        before: ArcAGI3Observation,
        action: str,
        result: ArcAGI3StepResult,
        *,
        invalid_action: bool = False,
        diagnostics: dict[str, Any] | None = None,
    ) -> None:
        events = [str(item) for item in result.info.get("events", [])]
        self.steps.append(
            AttemptStep(
                step_index=int(before.step_index),
                frame=observation_frame(before).astype(int).tolist(),
                action=str(action),
                legal_actions=[str(item) for item in before.available_actions],
                score_delta=float(result.reward),
                event_delta=events,
                terminal=bool(result.terminated or result.truncated),
                obs_hash=stable_hash(
                    {
                        "grid": np.asarray(before.grid, dtype=np.int64),
                        "extras": before.extras,
                        "actions": before.available_actions,
                    }
                ),
                next_obs_hash=stable_hash(
                    {
                        "grid": np.asarray(result.observation.grid, dtype=np.int64),
                        "extras": result.observation.extras,
                        "actions": result.observation.available_actions,
                    }
                ),
                invalid_action=bool(invalid_action),
                diagnostics=json_safe(diagnostics or {}),
            )
        )

    def to_record(self) -> AttemptRecord:
        return AttemptRecord(
            suite_id=self.suite_id,
            task_id=self.task_id,
            variant=self.variant,
            split=self.split,
            seed=self.seed,
            attempt_index=self.attempt_index,
            steps=list(self.steps),
        )


def tensors_from_attempts(
    records: list[AttemptRecord],
    *,
    max_steps: int | None = None,
    buckets: int = ACTION_BUCKETS,
    device: torch.device | str | None = None,
) -> dict[str, torch.Tensor]:
    if not records:
        raise ValueError("at least one attempt record is required")
    limit = int(max_steps or max(len(record.steps) for record in records))
    frame_dim = GRID_SIZE * GRID_SIZE
    frames = torch.zeros(len(records), limit, frame_dim, dtype=torch.float32, device=device)
    action_ids = torch.zeros(len(records), limit, dtype=torch.long, device=device)
    legal_counts = torch.zeros(len(records), limit, 1, dtype=torch.float32, device=device)
    legal_masks = torch.zeros(len(records), limit, buckets, dtype=torch.float32, device=device)
    terminals = torch.zeros(len(records), limit, 1, dtype=torch.float32, device=device)
    valid = torch.zeros(len(records), limit, 1, dtype=torch.float32, device=device)
    score_deltas = torch.zeros(len(records), limit, 1, dtype=torch.float32, device=device)
    for row, record in enumerate(records):
        for col, step in enumerate(record.steps[:limit]):
            frame = np.asarray(step.frame, dtype=np.float32).reshape(-1)
            frames[row, col, : min(frame_dim, frame.size)] = torch.as_tensor(frame[:frame_dim], dtype=torch.float32, device=device)
            action_ids[row, col] = action_id(step.action, buckets)
            legal_counts[row, col, 0] = float(len(step.legal_actions))
            legal_masks[row, col] = torch.as_tensor(legal_action_mask(step.legal_actions, buckets), dtype=torch.float32, device=device)
            terminals[row, col, 0] = float(step.terminal)
            score_deltas[row, col, 0] = float(step.score_delta)
            valid[row, col, 0] = 1.0
    return {
        "frames": frames,
        "action_ids": action_ids,
        "legal_counts": legal_counts,
        "legal_masks": legal_masks,
        "terminals": terminals,
        "score_deltas": score_deltas,
        "valid": valid,
    }


def attempt_from_trace(path: str | Path) -> AttemptRecord:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if "timeline_schema" in payload and "steps" in payload:
        return AttemptRecord(
            suite_id=str(payload["suite_id"]),
            task_id=str(payload["task_id"]),
            variant=str(payload["variant"]),
            split=str(payload["split"]),
            seed=int(payload["seed"]),
            attempt_index=int(payload["attempt_index"]),
            steps=[AttemptStep(**step) for step in payload["steps"]],
        )
    meta = payload.get("metadata", {})
    steps: list[AttemptStep] = []
    for index, step in enumerate(payload.get("steps", [])):
        summary = step.get("obs", {}).get("grid_summary", {})
        shape = summary.get("shape", [GRID_SIZE, GRID_SIZE])
        frame = np.zeros((GRID_SIZE, GRID_SIZE), dtype=np.int64)
        counts = summary.get("counts", {})
        cursor = 0
        for key, count in sorted(counts.items(), key=lambda item: int(item[0])):
            for _ in range(int(count)):
                if cursor < GRID_SIZE * GRID_SIZE:
                    frame.reshape(-1)[cursor] = int(key)
                    cursor += 1
        if shape and int(shape[0]) > GRID_SIZE:
            frame[-1, -1] = min(int(shape[0]), 99)
        steps.append(
            AttemptStep(
                step_index=int(step.get("step", index)),
                frame=frame.tolist(),
                action=str(step.get("chosen_action", "")),
                legal_actions=[str(item) for item in step.get("legal_actions", [])],
                score_delta=float(step.get("score_delta", 0.0)),
                event_delta=[str(item) for item in step.get("event_delta", [])],
                terminal=bool(step.get("terminal", False)),
                obs_hash=str(step.get("obs", {}).get("obs_hash", "")),
                next_obs_hash=str(step.get("next_obs_hash", "")),
                invalid_action=bool(step.get("invalid_action", False)),
                diagnostics=json_safe(step.get("memory_drive_hypothesis", {})),
            )
        )
    return AttemptRecord(
        suite_id=str(meta.get("suite_id", "")),
        task_id=str(meta.get("task_id", "")),
        variant=str(meta.get("variant", "")),
        split=str(meta.get("split", "")),
        seed=int(meta.get("seed", 0)),
        attempt_index=int(meta.get("attempt_index", 1)),
        steps=steps,
    )
