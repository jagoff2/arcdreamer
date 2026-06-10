from __future__ import annotations

import json
import math
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .arcagi3_adapter import ArcAGI3Observation, ArcAGI3StepResult, grid_to_rows


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [json_safe(item) for item in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    return repr(value)


def observation_summary(observation: ArcAGI3Observation) -> dict[str, Any]:
    grid = np.asarray(observation.grid, dtype=np.int64)
    return {
        "task_id": observation.task_id,
        "episode_id": observation.episode_id,
        "step_index": int(observation.step_index),
        "grid": grid_to_rows(grid),
        "available_actions": list(observation.available_actions),
        "extras": {
            key: observation.extras.get(key)
            for key in ["fixture_id", "score", "normalized_score", "have_key", "door_open", "game_state"]
            if key in observation.extras
        },
    }


class ArcTraceRecorder:
    def __init__(self, path: str | Path, *, metadata: dict[str, Any]) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.rows: list[dict[str, Any]] = []
        self.metadata = dict(metadata)

    def add_step(
        self,
        before: ArcAGI3Observation,
        action: str,
        result: ArcAGI3StepResult,
        diagnostics: dict[str, Any],
    ) -> None:
        self.rows.append(
            {
                "step": len(self.rows),
                "obs": observation_summary(before),
                "action": action,
                "next_obs": observation_summary(result.observation),
                "event_delta": list(result.info.get("events", [])),
                "score_delta": float(result.reward),
                "score": float(result.info.get("score", 0.0)),
                "normalized_score": float(result.info.get("normalized_score", 0.0)),
                "memory_recall": diagnostics.get("memory_recall", {}),
                "drive": diagnostics.get("drive", {}),
                "hypothesis_state": diagnostics.get("hypothesis_state", {}),
                "policy": {
                    "semantic_action": diagnostics.get("semantic_action"),
                    "behavior": diagnostics.get("behavior"),
                    "causal_trace": diagnostics.get("causal_trace"),
                },
            }
        )

    def write(self, summary: dict[str, Any]) -> Path:
        payload = {"metadata": self.metadata, "summary": summary, "steps": self.rows}
        self.path.write_text(json.dumps(json_safe(payload), indent=2), encoding="utf-8")
        return self.path


def action_entropy(actions: list[str]) -> float:
    if not actions:
        return 0.0
    counts = Counter(actions)
    total = float(len(actions))
    return float(-sum((count / total) * math.log(max(count / total, 1.0e-12), 2) for count in counts.values()))


def repeat_collapse(actions: list[str]) -> float:
    if not actions:
        return 0.0
    counts = Counter(actions)
    return float(max(counts.values()) / len(actions))

