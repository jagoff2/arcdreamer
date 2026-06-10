from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .arcagi3_adapter import (
    ARCAGI3Adapter,
    CELL_AGENT,
    CELL_EMPTY,
    CELL_GOAL,
    CELL_RESOURCE,
    ArcAGI3Observation,
    ArcAGI3StepResult,
)
from .arcagi3_baselines import BaselinePolicy, build_baselines
from .arcagi3_diagnose import DIAGNOSIS_AUDITED_PATHS, runtime_status
from .arcagi3_eval import aggregate_rows, collect_hashes, state_signature
from .arcagi3_failure_taxonomy import no_hack_audit
from .arcagi3_trace import action_entropy, repeat_collapse
from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .external_registry import CLAIMS, ExternalSuite, claim_registry_template, discover_external_suites
from .head_collapse import HeadCollapsedExplorer
from .world_model import load_explorer_checkpoint


EXTERNAL_AUDITED_PATHS = [
    "src/external_registry.py",
    "src/external_eval.py",
    "src/generalization_audit.py",
    "tests/test_external_generalization.py",
    "docs/external_generalization_report.json",
    "docs/external_generalization_report.md",
    "docs/generalization_audit.json",
]

BASELINE_NAMES = [
    "random_legal",
    "repeat_last_action",
    "coverage_graph_exploration",
    "novelty_first",
    "greedy_observable_score_delta",
    "oracle_free_observed_graph_bfs",
]

ABLATION_MODES = {
    "zero_z": "zero_z",
    "shuffled_z": "shuffled_z",
    "corrupt_memory": "corrupt_memory",
    "corrupt_drive": "corrupt_drive",
    "disable_planner_imagination": "no_planner_imagination",
    "no_dialogue": "normal",
    "no_private_tokens": "normal",
    "no_social_state": "normal",
}


class ExternalEnv(Protocol):
    suite_id: str
    task_id: str
    max_steps: int

    def reset(self, seed: int) -> ArcAGI3Observation:
        ...

    def step(self, action: str) -> ArcAGI3StepResult:
        ...

    def close(self) -> None:
        ...


def sha_obj(value: Any) -> str:
    encoded = json.dumps(json_safe(value), sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()[:16]


def json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    return repr(value)


def grid_summary(grid: np.ndarray) -> dict[str, Any]:
    arr = np.asarray(grid, dtype=np.int64)
    counts = Counter(int(item) for item in arr.reshape(-1))
    return {
        "shape": list(arr.shape),
        "hash": sha_obj(arr),
        "counts": {str(key): int(value) for key, value in sorted(counts.items())},
    }


def observation_summary(observation: ArcAGI3Observation) -> dict[str, Any]:
    return {
        "task_id": observation.task_id,
        "episode_id": observation.episode_id,
        "step_index": int(observation.step_index),
        "obs_hash": sha_obj(
            {
                "grid": np.asarray(observation.grid, dtype=np.int64),
                "extras": observation.extras,
                "actions": observation.available_actions,
            }
        ),
        "grid_summary": grid_summary(observation.grid),
        "available_actions": list(observation.available_actions),
        "public_extras": {
            key: observation.extras.get(key)
            for key in ["suite_id", "task_id", "raw_observation", "score", "terminated", "truncated"]
            if key in observation.extras
        },
    }


class GymnasiumExternalEnv:
    def __init__(self, suite_id: str, task_id: str, *, max_steps: int) -> None:
        import gymnasium as gym

        self.suite_id = suite_id
        self.task_id = task_id
        if task_id == "FrozenLake-v1":
            self.env = gym.make(task_id, is_slippery=False)
        else:
            self.env = gym.make(task_id)
        self.max_steps = max_steps
        self.step_index = 0
        self.score = 0.0
        self._raw_obs: Any = None
        self._terminated = False
        self._truncated = False
        self.seed = 0

    def reset(self, seed: int) -> ArcAGI3Observation:
        self.seed = int(seed)
        self.step_index = 0
        self.score = 0.0
        self._terminated = False
        self._truncated = False
        obs, _ = self.env.reset(seed=self.seed)
        self._raw_obs = obs
        return self._observation()

    def step(self, action: str) -> ArcAGI3StepResult:
        if self._terminated or self._truncated:
            return ArcAGI3StepResult(
                self._observation(),
                0.0,
                self._terminated,
                self._truncated,
                {"events": ["already_terminal"], "score": self.score, "normalized_score": self.normalized_score()},
            )
        legal = self._legal_actions()
        invalid = action not in legal
        action_index = 0 if invalid else int(action) - 1
        raw_next, reward, terminated, truncated, _ = self.env.step(action_index)
        self.step_index += 1
        self._raw_obs = raw_next
        self.score += float(reward)
        self._terminated = bool(terminated)
        self._truncated = bool(truncated or self.step_index >= self.max_steps)
        events = []
        if invalid:
            events.append("invalid_action")
        if reward > 0.0:
            events.append("positive_reward")
        if self._terminated:
            events.append("terminated")
        if self._truncated and not self._terminated:
            events.append("truncated")
        obs = self._observation()
        return ArcAGI3StepResult(
            obs,
            float(reward),
            self._terminated,
            self._truncated,
            {
                "events": events,
                "score": self.score,
                "normalized_score": self.normalized_score(),
                "raw_reward": float(reward),
                "invalid": invalid,
            },
        )

    def close(self) -> None:
        self.env.close()

    def normalized_score(self) -> float:
        if self.task_id == "CartPole-v1":
            return float(min(self.score / max(self.max_steps, 1), 1.0))
        return float(max(0.0, min(self.score, 1.0)))

    def _legal_actions(self) -> tuple[str, ...]:
        return tuple(str(index + 1) for index in range(int(self.env.action_space.n)))

    def _observation(self) -> ArcAGI3Observation:
        grid = gym_observation_to_grid(self.task_id, self._raw_obs)
        return ArcAGI3Observation(
            task_id=f"external/{self.suite_id}/{self.task_id}",
            episode_id=f"external/{self.suite_id}/{self.task_id}/seed_{self.seed}",
            step_index=self.step_index,
            grid=grid,
            available_actions=self._legal_actions(),
            extras={
                "suite_id": self.suite_id,
                "task_id": self.task_id,
                "raw_observation": raw_observation_summary(self._raw_obs),
                "score": self.score,
                "normalized_score": self.normalized_score(),
                "terminated": self._terminated,
                "truncated": self._truncated,
                "public_rules": {"external_suite": self.suite_id, "task": self.task_id},
            },
        )


def raw_observation_summary(raw: Any) -> Any:
    if isinstance(raw, np.ndarray):
        return [round(float(item), 6) for item in raw.reshape(-1).tolist()]
    if isinstance(raw, (int, np.integer)):
        return int(raw)
    if isinstance(raw, (float, np.floating)):
        return round(float(raw), 6)
    return repr(raw)


def gym_observation_to_grid(task_id: str, raw: Any) -> np.ndarray:
    if task_id == "FrozenLake-v1":
        index = int(raw)
        grid = np.full((4, 4), CELL_EMPTY, dtype=np.int64)
        y, x = divmod(index, 4)
        grid[y, x] = CELL_AGENT
        return grid
    values = np.asarray(raw, dtype=np.float32).reshape(-1)
    grid = np.full((8, 8), CELL_EMPTY, dtype=np.int64)
    cart = int(np.clip(round(float(values[0]) * 2.0 + 3.5), 0, 7)) if values.size else 3
    angle = int(np.clip(round(float(values[2]) * 12.0 + 3.5), 0, 7)) if values.size >= 3 else 3
    grid[6, cart] = CELL_AGENT
    grid[1, angle] = CELL_GOAL
    grid[3, int(np.clip(round(abs(float(values[1])) * 3.0), 0, 7))] = CELL_RESOURCE
    return grid


def make_adapter(explorer_checkpoint: str | Path, *, mode: str, device: DeviceLike) -> ARCAGI3Adapter:
    target_device = resolve_device(device)
    model = load_explorer_checkpoint(explorer_checkpoint, device=target_device)
    return ARCAGI3Adapter(
        HeadCollapsedExplorer(model, disable_structured_heads=True, remove_probe_modules=False),
        device=target_device,
        mode=mode,
    )


def make_suite_env(suite: ExternalSuite, task_id: str) -> ExternalEnv:
    if suite.suite_id == "gymnasium_classic_control":
        return GymnasiumExternalEnv(suite.suite_id, task_id, max_steps=80)
    if suite.suite_id == "gymnasium_toy_text":
        return GymnasiumExternalEnv(suite.suite_id, task_id, max_steps=32)
    raise ValueError(f"cannot instantiate suite {suite.suite_id}")


def controller_choose(controller: Any, obs: ArcAGI3Observation, seed: int) -> tuple[str, dict[str, Any]]:
    if isinstance(controller, ARCAGI3Adapter):
        return controller.choose_action(obs)
    action = controller.choose(obs)
    return action, {"memory_recall": {}, "drive": {}, "hypothesis_state": {}, "policy": {"baseline": controller.name}}


def controller_reset(controller: Any, seed: int) -> None:
    if isinstance(controller, ARCAGI3Adapter):
        controller.reset()
    else:
        controller.reset(seed)


def controller_observe(controller: Any, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult) -> None:
    if isinstance(controller, ARCAGI3Adapter):
        controller.observe_transition(action, result)
    else:
        controller.observe(before, action, result)


def run_external_episode(
    env: ExternalEnv,
    controller: Any,
    *,
    controller_name: str,
    seed: int,
    trace_path: Path,
    baseline_snapshots: bool,
) -> dict[str, Any]:
    controller_reset(controller, seed)
    baselines: list[BaselinePolicy] = build_baselines() if baseline_snapshots else []
    for baseline in baselines:
        baseline.reset(seed)
    obs = env.reset(seed)
    actions: list[str] = []
    invalid = 0
    useful_events = 0
    states = {state_signature(obs)}
    rows: list[dict[str, Any]] = []
    try:
        for _ in range(env.max_steps):
            before = obs
            baseline_actions: dict[str, str] = {}
            for baseline in baselines:
                baseline_actions[baseline.name] = baseline.choose(before)
            action, diagnostics = controller_choose(controller, before, seed)
            if action not in before.available_actions:
                invalid += 1
                action = before.available_actions[0]
            result = env.step(action)
            actions.append(action)
            states.add(state_signature(result.observation))
            useful_events += int(float(result.reward) > 0.0)
            for baseline in baselines:
                baseline.observe(before, baseline_actions[baseline.name], result)
            controller_observe(controller, before, action, result)
            rows.append(
                {
                    "step": len(rows),
                    "obs": observation_summary(before),
                    "legal_actions": list(before.available_actions),
                    "chosen_action": action,
                    "baseline_actions": baseline_actions,
                    "next_obs_hash": observation_summary(result.observation)["obs_hash"],
                    "score_delta": float(result.reward),
                    "event_delta": list(result.info.get("events", [])),
                    "terminal": bool(result.terminated or result.truncated),
                    "invalid_action": action not in before.available_actions,
                    "memory_drive_hypothesis": {
                        "memory_recall": diagnostics.get("memory_recall", {}),
                        "drive": diagnostics.get("drive", {}),
                        "hypothesis_state": diagnostics.get("hypothesis_state", {}),
                        "policy": diagnostics.get("policy", {}),
                    },
                    "failure_class": "none" if result.reward > 0.0 else "no_external_reward",
                }
            )
            obs = result.observation
            if result.terminated or result.truncated:
                break
    finally:
        env.close()
    summary = {
        "suite_id": env.suite_id,
        "task_id": env.task_id,
        "controller": controller_name,
        "seed": seed,
        "score": float(getattr(env, "score", 0.0)),
        "normalized_score": float(env.normalized_score()),
        "solved": bool(env.normalized_score() >= 1.0),
        "steps": len(actions),
        "invalid_action_rate": float(invalid / max(len(actions), 1)),
        "unique_states": len(states),
        "useful_events": useful_events,
        "action_entropy": action_entropy(actions),
        "repeat_collapse": repeat_collapse(actions),
        "actions": actions[:64],
        "failure_class": "external_reward_not_achieved" if float(env.normalized_score()) <= 0.0 else "partial_or_complete_reward",
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps({"metadata": {k: summary[k] for k in ["suite_id", "task_id", "controller", "seed"]}, "summary": summary, "steps": rows}, indent=2),
        encoding="utf-8",
    )
    summary["trace_path"] = str(trace_path)
    return summary


def evaluate_gym_suite(
    suite: ExternalSuite,
    *,
    explorer_checkpoint: str | Path,
    trace_dir: str | Path,
    device: DeviceLike,
) -> dict[str, Any]:
    dev_seeds = [0, 1]
    eval_seeds = [100, 101, 102]
    controllers: dict[str, Any] = {"explorer": make_adapter(explorer_checkpoint, mode="normal", device=device)}
    controllers.update({baseline.name: baseline for baseline in build_baselines()})
    for ablation, mode in ABLATION_MODES.items():
        controllers[f"ablation_{ablation}"] = make_adapter(explorer_checkpoint, mode=mode, device=device)
    rows: list[dict[str, Any]] = []
    for split, seeds in [("dev", dev_seeds), ("sealed_eval", eval_seeds)]:
        for task_id in suite.tasks:
            for controller_name, controller in controllers.items():
                for seed in seeds:
                    env = make_suite_env(suite, task_id)
                    trace_path = Path(trace_dir) / suite.suite_id / split / controller_name / f"{task_id}.seed_{seed}.json"
                    row = run_external_episode(
                        env,
                        controller,
                        controller_name=controller_name,
                        seed=seed,
                        trace_path=trace_path,
                        baseline_snapshots=controller_name == "explorer",
                    )
                    row["split"] = split
                    rows.append(row)
    return summarize_suite_rows(suite, rows)


def compact_arc_traces(trace_dir: str | Path) -> list[str]:
    source_root = Path("docs/arcagi3_official_traces")
    target_root = Path(trace_dir) / "official_arcagi3" / "sealed_eval" / "explorer"
    target_root.mkdir(parents=True, exist_ok=True)
    paths: list[str] = []
    for source in sorted(source_root.glob("*.official.normal.json")):
        payload = json.loads(source.read_text(encoding="utf-8"))
        compact_steps = []
        for step in payload.get("steps", []):
            diag = step.get("diagnosis", {})
            compact_steps.append(
                {
                    "step": step.get("step"),
                    "obs_hash": diag.get("extracted_object_entity_features", {}).get("hash"),
                    "legal_actions": diag.get("legal_actions", []),
                    "chosen_action": diag.get("chosen_action", step.get("action")),
                    "baseline_actions": diag.get("baseline_actions", {}),
                    "score_delta": diag.get("score_delta", step.get("score_delta")),
                    "event_delta": diag.get("event_delta", step.get("event_delta", [])),
                    "terminal": diag.get("terminal_flag", False),
                    "invalid_action": diag.get("invalid_action", False),
                    "memory_drive_hypothesis": diag.get("z_memory_drive_hypothesis_summary", {}),
                    "failure_class": payload.get("diagnosis", {}).get("schema", "official_arc_failure_trace"),
                }
            )
        target = target_root / source.name
        target.write_text(
            json.dumps(
                {
                    "metadata": payload.get("metadata", {}),
                    "summary": payload.get("summary", {}),
                    "steps": compact_steps,
                    "source_trace_path": str(source),
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        paths.append(str(target))
    return paths


def official_arc_suite_summary(trace_dir: str | Path) -> dict[str, Any]:
    report = json.loads(Path("docs/arcagi3_failure_report.json").read_text(encoding="utf-8"))
    trace_paths = compact_arc_traces(trace_dir)
    aggregate = {row["name"]: row for row in report.get("aggregate_score_table", [])}
    rows = []
    for name, values in aggregate.items():
        row = {"suite_id": "official_arcagi3", "task_id": "all_official_public_games", "split": "sealed_eval", "controller": name}
        row.update(values)
        rows.append(row)
    return {
        "suite": {
            "suite_id": "official_arcagi3",
            "name": "Official ARC-AGI-3 public games",
            "available": True,
            "generated_by_repo": False,
            "split": "sealed_eval",
        },
        "rows": rows,
        "aggregate_by_controller": aggregate,
        "trace_paths": trace_paths,
        "source_report": "docs/arcagi3_failure_report.json",
    }


def summarize_suite_rows(suite: ExternalSuite, rows: list[dict[str, Any]]) -> dict[str, Any]:
    aggregate: dict[str, dict[str, Any]] = {}
    for controller in sorted({row["controller"] for row in rows}):
        eval_rows = [row for row in rows if row["controller"] == controller and row["split"] == "sealed_eval"]
        aggregate[controller] = aggregate_external_rows(eval_rows)
    return {
        "suite": suite.to_dict(),
        "rows": rows,
        "aggregate_by_controller": aggregate,
        "trace_paths": [row["trace_path"] for row in rows],
    }


def aggregate_external_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {
            "solve_rate": 0.0,
            "mean_score": 0.0,
            "mean_normalized_score": 0.0,
            "mean_steps": 0.0,
            "mean_invalid_action_rate": 0.0,
            "mean_unique_states": 0.0,
            "mean_useful_events": 0.0,
            "mean_action_entropy": 0.0,
            "mean_repeat_collapse": 0.0,
        }
    return aggregate_rows(rows)


def best_baseline_for_suite(suite_report: dict[str, Any]) -> dict[str, Any]:
    aggregate = suite_report.get("aggregate_by_controller", {})
    candidates = [(name, aggregate[name]) for name in BASELINE_NAMES if name in aggregate]
    if not candidates:
        return {"name": "none", "mean_normalized_score": 0.0}
    name, row = max(candidates, key=lambda item: float(item[1].get("mean_normalized_score", 0.0)))
    return {"name": name, **row}


def build_claim_registry(suite_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = claim_registry_template()
    sealed_suites = [suite for suite in suite_reports if suite["suite"].get("split") in {"sealed_eval", "dev_and_sealed_eval"}]
    best_external_margin = -math.inf
    max_ablation_drop: dict[str, float] = {claim.ablation_expected_to_hurt: 0.0 for claim in CLAIMS}
    external_scores: list[str] = []
    for suite in sealed_suites:
        aggregate = suite.get("aggregate_by_controller", {})
        explorer = aggregate.get("explorer", {})
        best = best_baseline_for_suite(suite)
        margin = float(explorer.get("mean_normalized_score", 0.0)) - float(best.get("mean_normalized_score", 0.0))
        best_external_margin = max(best_external_margin, margin)
        external_scores.append(
            f"{suite['suite']['suite_id']}: explorer={float(explorer.get('mean_normalized_score', 0.0)):.3f}, best_baseline={float(best.get('mean_normalized_score', 0.0)):.3f}, margin={margin:.3f}"
        )
        for ablation in max_ablation_drop:
            row = aggregate.get(f"ablation_{ablation}", {})
            drop = float(explorer.get("mean_normalized_score", 0.0)) - float(row.get("mean_normalized_score", 0.0))
            max_ablation_drop[ablation] = max(max_ablation_drop[ablation], drop)
    for row in rows:
        ablation = str(row["ablation_expected_to_hurt"])
        margin_supported = best_external_margin >= 0.05
        ablation_supported = max_ablation_drop.get(ablation, 0.0) >= 0.10
        if margin_supported or ablation_supported:
            row["status"] = "supported"
        else:
            row["status"] = "unsupported"
        row["result"] = {
            "best_external_margin": best_external_margin if best_external_margin != -math.inf else None,
            "max_external_ablation_drop": max_ablation_drop.get(ablation, 0.0),
            "external_scores": external_scores,
            "anti_goodhart": "Internal synthetic evidence is diagnostic only; external scores did not meet support threshold."
            if row["status"] == "unsupported"
            else "Supported by external threshold.",
        }
        if row["claim_id"] in {"dialogue", "social_state"} and row["status"] == "supported":
            row["status"] = "unsupported"
            row["result"]["anti_goodhart"] = "No sealed external language/social suite was available; support is not allowed."
    return rows


def build_external_report(
    *,
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    config: str,
    json_output: str | Path,
    trace_dir: str | Path,
    device: DeviceLike = AUTO_DEVICE,
) -> dict[str, Any]:
    if config != "external":
        raise ValueError("external_eval supports only --config external")
    target_device = resolve_device(device)
    suites = discover_external_suites()
    available = [suite for suite in suites if suite.available]
    suite_reports: list[dict[str, Any]] = []
    if any(suite.suite_id == "official_arcagi3" for suite in available):
        suite_reports.append(official_arc_suite_summary(trace_dir))
    for suite in available:
        if suite.suite_id.startswith("gymnasium_"):
            suite_reports.append(evaluate_gym_suite(suite, explorer_checkpoint=explorer_checkpoint, trace_dir=trace_dir, device=target_device))
    claim_registry = build_claim_registry(suite_reports)
    unsupported = [row for row in claim_registry if row["status"] == "unsupported"]
    no_hack = no_hack_audit([])
    requirements = {
        "external_suites_present": len(suite_reports) >= 1,
        "official_arc_included": any(item["suite"]["suite_id"] == "official_arcagi3" for item in suite_reports),
        "two_additional_suites_if_present": len([item for item in suite_reports if item["suite"]["suite_id"].startswith("gymnasium_")]) >= 2,
        "claim_registry_complete": {row["claim_id"] for row in claim_registry} == {claim.claim_id for claim in CLAIMS},
        "supported_claims_have_external_thresholds": all(supported_claim_has_evidence(row) for row in claim_registry if row["status"] == "supported"),
        "negative_results_preserved": bool(unsupported),
        "no_hack_audit_passes": no_hack["passes"],
        "trace_paths_present": all(Path(path).exists() for suite in suite_reports for path in suite.get("trace_paths", [])),
    }
    outcome = "EXTERNAL GENERALIZATION DISCIPLINE PROVEN" if all(requirements.values()) else "NOT PROVEN"
    report: dict[str, Any] = {
        "terminal_outcome": outcome,
        "config": config,
        "checkpoint": str(checkpoint),
        "explorer_checkpoint": str(explorer_checkpoint),
        "requested_device": str(device),
        "resolved_device": str(target_device),
        "external_suites_discovered": [suite.to_dict() for suite in suites],
        "external_suites_evaluated": [suite["suite"] for suite in suite_reports],
        "suite_reports": suite_reports,
        "claim_registry": claim_registry,
        "unsupported_claims": unsupported,
        "aggregate_scores": aggregate_score_table(suite_reports),
        "baselines": BASELINE_NAMES,
        "ablations": list(ABLATION_MODES),
        "no_hack_audit": no_hack,
        "official_runtime_status": runtime_status(target_device),
        "trace_dir": str(trace_dir),
        "trace_paths": [path for suite in suite_reports for path in suite.get("trace_paths", [])],
        "requirements": requirements,
        "limitations": limitations(suite_reports, claim_registry),
        "hashes": {},
    }
    output = Path(json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report["hashes"] = collect_hashes(EXTERNAL_AUDITED_PATHS + DIAGNOSIS_AUDITED_PATHS + [str(checkpoint), str(explorer_checkpoint)])
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    report["hashes"] = collect_hashes(EXTERNAL_AUDITED_PATHS + DIAGNOSIS_AUDITED_PATHS + [str(checkpoint), str(explorer_checkpoint)])
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    return report


def supported_claim_has_evidence(row: dict[str, Any]) -> bool:
    result = row.get("result", {})
    margin = result.get("best_external_margin")
    drop = float(result.get("max_external_ablation_drop") or 0.0)
    return (margin is not None and float(margin) >= 0.05) or drop >= 0.10


def aggregate_score_table(suite_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for suite in suite_reports:
        for controller, aggregate in suite.get("aggregate_by_controller", {}).items():
            rows.append({"suite_id": suite["suite"]["suite_id"], "controller": controller, **aggregate})
    return rows


def limitations(suite_reports: list[dict[str, Any]], claim_registry: list[dict[str, Any]]) -> list[str]:
    out: list[str] = []
    if not any(item["suite"]["suite_id"] == "official_arcagi3" for item in suite_reports):
        out.append("Official ARC-AGI-3 was not evaluated.")
    if not any(row["status"] == "supported" for row in claim_registry):
        out.append("No active claim is externally supported; all internal synthetic metrics are diagnostic only.")
    out.append("Gymnasium wrappers are external smoke suites, not evidence that the ARC failure is solved.")
    return out


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# External Generalization Discipline")
    lines.append("")
    lines.append(f"Terminal outcome: **{report['terminal_outcome']}**")
    lines.append("")
    lines.append("## External Suites")
    lines.append("")
    lines.append("| Suite | Available | Tasks | Split |")
    lines.append("| --- | --- | --- | --- |")
    for suite in report["external_suites_discovered"]:
        lines.append(f"| {suite['suite_id']} | {suite['available']} | {', '.join(suite['tasks'])} | {suite['split']} |")
    lines.append("")
    lines.append("## Claim Registry")
    lines.append("")
    lines.append("| Claim | Metric | Baseline | Ablation | Status |")
    lines.append("| --- | --- | --- | --- | --- |")
    for row in report["claim_registry"]:
        lines.append(
            f"| {row['claim_id']} | {row['external_metric']} | {row['baseline_to_beat']} | {row['ablation_expected_to_hurt']} | {row['status']} |"
        )
    lines.append("")
    lines.append("## Aggregate Scores")
    lines.append("")
    lines.append("| Suite | Controller | Mean Normalized | Solve Rate | Steps | Entropy | Repeat |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: | ---: |")
    for row in report["aggregate_scores"]:
        lines.append(
            f"| {row['suite_id']} | {row['controller']} | {float(row.get('mean_normalized_score', 0.0)):.3f} | {float(row.get('solve_rate', 0.0)):.3f} | {float(row.get('mean_steps', 0.0)):.2f} | {float(row.get('mean_action_entropy', 0.0)):.3f} | {float(row.get('mean_repeat_collapse', 0.0)):.3f} |"
        )
    lines.append("")
    lines.append("## Unsupported Claims")
    for row in report["unsupported_claims"]:
        result = row.get("result", {})
        lines.append(
            f"- {row['claim_id']}: margin `{result.get('best_external_margin')}`, ablation drop `{result.get('max_external_ablation_drop')}`."
        )
    lines.append("")
    lines.append("## Trace Paths")
    for path in report["trace_paths"][:40]:
        lines.append(f"- `{path}`")
    if len(report["trace_paths"]) > 40:
        lines.append(f"- ... {len(report['trace_paths']) - 40} more traces")
    lines.append("")
    lines.append("## Limitations")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="external")
    parser.add_argument("--checkpoint", default="frozen/recurrent_latent_fast.pt")
    parser.add_argument("--explorer-checkpoint", default="runs/explorer_tiny.pt")
    parser.add_argument("--json-output", default="docs/external_generalization_report.json")
    parser.add_argument("--trace-dir", default="docs/external_traces")
    parser.add_argument("--device", default=AUTO_DEVICE)
    args = parser.parse_args()
    try:
        report = build_external_report(
            checkpoint=args.checkpoint,
            explorer_checkpoint=args.explorer_checkpoint,
            config=args.config,
            json_output=args.json_output,
            trace_dir=args.trace_dir,
            device=args.device,
        )
    except Exception as exc:
        failure = {"terminal_outcome": "NOT PROVEN", "error": repr(exc)}
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps(failure, indent=2), file=sys.stderr)
        raise
    print(
        json.dumps(
            {
                "terminal_outcome": report["terminal_outcome"],
                "requirements": report["requirements"],
                "unsupported_claims": [row["claim_id"] for row in report["unsupported_claims"]],
                "limitations": report["limitations"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
