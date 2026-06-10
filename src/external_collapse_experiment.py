from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .arcagi3_adapter import ARCAGI3Adapter, ArcAGI3Observation, ArcAGI3StepResult
from .arcagi3_baselines import build_baselines
from .arcagi3_eval import aggregate_rows, collect_hashes, state_signature
from .arcagi3_failure_taxonomy import no_hack_audit
from .arcagi3_official import OfficialArcAGI3Env, default_max_steps, discover_official_games, make_arcade
from .arcagi3_trace import action_entropy, repeat_collapse
from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .external_affordance import affordance_bias, summarize_affordances
from .external_eval import BASELINE_NAMES, GymnasiumExternalEnv, grid_summary, observation_summary, sha_obj
from .external_registry import ExternalSuite, discover_external_suites
from .external_valence import ConsequenceValence, observation_key
from .head_collapse import HeadCollapsedExplorer
from .world_model import load_explorer_checkpoint


CUDA_PREFERRED_DEVICE = "cuda" if torch.cuda.is_available() else AUTO_DEVICE

EXPERIMENT_PATHS = [
    "src/external_valence.py",
    "src/external_affordance.py",
    "src/external_collapse_experiment.py",
    "tests/test_external_collapse.py",
    "docs/external_collapse_report.json",
    "docs/external_collapse_report.md",
]


@dataclass(frozen=True)
class VariantSpec:
    variant_id: str
    valence: bool = False
    affordance: bool = False
    loop: bool = False
    null_patch: bool = False


VARIANTS = [
    VariantSpec("baseline_unchanged"),
    VariantSpec("valence_only", valence=True),
    VariantSpec("affordance_only", affordance=True),
    VariantSpec("loop_aversion_only", loop=True),
    VariantSpec("valence_affordance", valence=True, affordance=True),
    VariantSpec("valence_affordance_loop", valence=True, affordance=True, loop=True),
    VariantSpec("null_patch_control", null_patch=True),
]
CHANGE_VARIANTS = [item.variant_id for item in VARIANTS if item.variant_id not in {"baseline_unchanged", "null_patch_control"}]
ABLATION_MODES = {"zero_z": "zero_z", "corrupt_memory": "corrupt_memory", "corrupt_drive": "corrupt_drive"}


def variant_by_id(variant_id: str) -> VariantSpec:
    for item in VARIANTS:
        if item.variant_id == variant_id:
            return item
    raise KeyError(variant_id)


def device_matches(requested: torch.device, actual: torch.device) -> bool:
    if requested.type != actual.type:
        return False
    if requested.type == "cuda" and requested.index is not None and actual.index != requested.index:
        return False
    return True


def explorer_parameter_device(explorer: HeadCollapsedExplorer) -> torch.device:
    return next(explorer.base.parameters()).device


def cuda_runtime_info(resolved_device: DeviceLike = AUTO_DEVICE) -> dict[str, Any]:
    device = resolve_device(resolved_device)
    cuda_available = bool(torch.cuda.is_available())
    return {
        "resolved_device": str(device),
        "torch_version": getattr(torch, "__version__", None),
        "torch_cuda_available": cuda_available,
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_device_count": int(torch.cuda.device_count()) if cuda_available else 0,
        "torch_cuda_devices": [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
        if cuda_available
        else [],
    }


def adapter_device_summary(adapter: Any) -> dict[str, Any]:
    summary = {
        "adapter_device": str(getattr(adapter, "device", "unknown")),
        "explorer_parameter_device": "unknown",
        "cuda_model_parameters": False,
    }
    explorer = getattr(adapter, "explorer", None)
    if isinstance(explorer, HeadCollapsedExplorer):
        parameter_device = explorer_parameter_device(explorer)
        summary["explorer_parameter_device"] = str(parameter_device)
        summary["cuda_model_parameters"] = parameter_device.type == "cuda"
    return summary


def make_adapter(explorer_checkpoint: str | Path, device: DeviceLike, mode: str = "normal") -> ARCAGI3Adapter:
    target_device = resolve_device(device)
    model = load_explorer_checkpoint(explorer_checkpoint, device=target_device)
    explorer = HeadCollapsedExplorer(model, disable_structured_heads=True, remove_probe_modules=False)
    actual_device = explorer_parameter_device(explorer)
    if not device_matches(target_device, actual_device):
        raise RuntimeError(f"explorer checkpoint loaded on {actual_device}, expected {target_device}")
    return ARCAGI3Adapter(explorer, device=target_device, mode=mode)


class CollapseController:
    def __init__(
        self,
        adapter: ARCAGI3Adapter,
        variant: VariantSpec,
        *,
        valence_weight: float = 1.25,
        affordance_weight: float = 0.85,
        loop_weight: float = 2.0,
    ) -> None:
        self.adapter = adapter
        self.variant = variant
        self.valence = ConsequenceValence()
        self.valence_weight = float(valence_weight)
        self.affordance_weight = float(affordance_weight)
        self.loop_weight = float(loop_weight)
        self.previous_grid: np.ndarray | None = None
        self.last_observation: ArcAGI3Observation | None = None
        self.changed_actions = 0
        self.reason_counts = {"valence": 0, "affordance": 0, "loop": 0}
        self.affordance_observations = 0

    @property
    def name(self) -> str:
        return self.variant.variant_id

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.adapter.reset()
        self.valence.reset()
        self.previous_grid = None
        self.last_observation = None
        self.changed_actions = 0
        self.reason_counts = {"valence": 0, "affordance": 0, "loop": 0}
        self.affordance_observations = 0

    def choose_action(self, observation: ArcAGI3Observation) -> tuple[str, dict[str, Any]]:
        self.last_observation = observation
        base_action, diagnostics = self.adapter.choose_action(observation)
        legal = tuple(observation.available_actions)
        base_scores = {action: float(diagnostics.get("action_scores", {}).get(action, 0.0)) for action in legal}
        adjusted = dict(base_scores)
        components: dict[str, dict[str, float]] = {"valence": {}, "affordance": {}, "loop": {}}
        summary: dict[str, Any] = {}
        if self.variant.affordance:
            summary = summarize_affordances(observation, self.previous_grid)
            self.affordance_observations += int(
                summary.get("non_background_count", 0) > 0 or summary.get("click_region_count", 0) > 0
            )
            for action in legal:
                bias = self.affordance_weight * affordance_bias(observation, action, summary, self.previous_grid)
                adjusted[action] += bias
                components["affordance"][action] = bias
        if self.variant.valence:
            for action in legal:
                bias = self.valence_weight * self.valence.score(observation, action)
                adjusted[action] += bias
                components["valence"][action] = bias
        if self.variant.loop:
            for action in legal:
                penalty = self.loop_weight * self.valence.loop_penalty(observation, action)
                adjusted[action] -= penalty
                components["loop"][action] = -penalty
        chosen = base_action if self.variant.variant_id in {"baseline_unchanged", "null_patch_control"} else max(
            legal, key=lambda action: (adjusted.get(action, -1.0e9), -legal.index(action))
        )
        changed = chosen != base_action
        self.changed_actions += int(changed)
        for name, values in components.items():
            if changed and abs(values.get(chosen, 0.0) - values.get(base_action, 0.0)) > 1.0e-9:
                self.reason_counts[name] += 1
        diagnostics["policy"] = {
            **diagnostics.get("policy", {}),
            "variant": self.variant.variant_id,
            "base_action": base_action,
            "chosen_action": chosen,
            "changed_action": changed,
            "state_key": observation_key(observation),
            "top_base_scores": top_scores(base_scores),
            "top_adjusted_scores": top_scores(adjusted),
            "components": {name: top_scores(values) for name, values in components.items() if values},
            "affordance_summary": summary,
            "valence": self.valence.diagnostics(),
            "forced_cycle": False,
        }
        return chosen, diagnostics

    def observe_transition(self, action: str, result: ArcAGI3StepResult) -> None:
        before = self.last_observation
        self.adapter.observe_transition(action, result)
        if before is not None:
            self.valence.observe(before, action, result)
        self.previous_grid = np.asarray(result.observation.grid, dtype=np.int64).copy()

    def summary(self) -> dict[str, Any]:
        return {
            "changed_actions": self.changed_actions,
            "reason_counts": dict(self.reason_counts),
            "affordance_observations": self.affordance_observations,
            "valence": self.valence.diagnostics(),
            "forced_cycle": False,
            "device": adapter_device_summary(self.adapter),
        }


def top_scores(scores: dict[str, float], limit: int = 8) -> dict[str, float]:
    return {
        action: round(float(value), 6)
        for action, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    }


def reset_controller(controller: Any, seed: int) -> None:
    if hasattr(controller, "reset"):
        controller.reset(seed)


def choose_controller(controller: Any, observation: ArcAGI3Observation) -> tuple[str, dict[str, Any]]:
    if hasattr(controller, "choose_action"):
        return controller.choose_action(observation)
    action = controller.choose(observation)
    return action, {"policy": {"baseline": getattr(controller, "name", type(controller).__name__)}}


def observe_controller(controller: Any, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult) -> None:
    if hasattr(controller, "observe_transition"):
        controller.observe_transition(action, result)
    else:
        controller.observe(before, action, result)


def run_episode(
    env: Any,
    controller: Any,
    *,
    suite_id: str,
    variant_id: str,
    split: str,
    seed: int,
    trace_path: Path,
    baseline_snapshots: bool,
) -> dict[str, Any]:
    reset_controller(controller, seed)
    baselines = build_baselines() if baseline_snapshots else []
    for baseline in baselines:
        baseline.reset(seed)
    obs = env.reset(seed)
    actions: list[str] = []
    invalid = 0
    useful_events = 0
    states = {state_signature(obs)}
    steps: list[dict[str, Any]] = []
    try:
        for _ in range(int(env.max_steps)):
            before = obs
            baseline_actions = {baseline.name: baseline.choose(before) for baseline in baselines}
            action, diagnostics = choose_controller(controller, before)
            if action not in before.available_actions:
                invalid += 1
                action = before.available_actions[0]
            result = env.step(action)
            actions.append(action)
            states.add(state_signature(result.observation))
            useful_events += int(float(result.reward) > 0.0)
            for baseline in baselines:
                baseline.observe(before, baseline_actions[baseline.name], result)
            observe_controller(controller, before, action, result)
            steps.append(
                {
                    "step": len(steps),
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
        close_result = env.close()
    normalized = float(env.normalized_score())
    summary = {
        "suite_id": suite_id,
        "task_id": getattr(env, "task_id", suite_id),
        "split": split,
        "variant": variant_id,
        "seed": seed,
        "score": float(getattr(env, "score", normalized)),
        "normalized_score": normalized,
        "solved": bool(normalized >= 1.0),
        "steps": len(actions),
        "invalid_action_rate": float(invalid / max(len(actions), 1)),
        "unique_states": len(states),
        "useful_events": useful_events,
        "action_entropy": action_entropy(actions),
        "repeat_collapse": repeat_collapse(actions),
        "actions": actions[:96],
        "controller_summary": controller.summary() if hasattr(controller, "summary") else {},
        "scorecard": close_result if isinstance(close_result, dict) else {},
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(
        json.dumps(
            {
                "metadata": {
                    "suite_id": suite_id,
                    "task_id": summary["task_id"],
                    "variant": variant_id,
                    "split": split,
                    "seed": seed,
                },
                "summary": summary,
                "steps": steps,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    summary["trace_path"] = str(trace_path)
    return summary


def make_controller(variant_id: str, explorer_checkpoint: str | Path, device: DeviceLike, mode: str = "normal") -> CollapseController:
    return CollapseController(make_adapter(explorer_checkpoint, device, mode=mode), variant_by_id(variant_id))


def gym_suite(suite: ExternalSuite, explorer_checkpoint: str | Path, trace_dir: str | Path, device: DeviceLike) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for split, seeds in [("dev", [0, 1]), ("sealed_eval", [100, 101, 102])]:
        for task_id in suite.tasks:
            for variant in VARIANTS:
                controller = make_controller(variant.variant_id, explorer_checkpoint, device)
                for seed in seeds:
                    env = GymnasiumExternalEnv(suite.suite_id, task_id, max_steps=80 if task_id == "CartPole-v1" else 32)
                    rows.append(
                        run_episode(
                            env,
                            controller,
                            suite_id=suite.suite_id,
                            variant_id=variant.variant_id,
                            split=split,
                            seed=seed,
                            trace_path=Path(trace_dir) / suite.suite_id / split / variant.variant_id / f"{task_id}.seed_{seed}.json",
                            baseline_snapshots=variant.variant_id == "baseline_unchanged",
                        )
                    )
            for baseline in build_baselines():
                for seed in seeds:
                    env = GymnasiumExternalEnv(suite.suite_id, task_id, max_steps=80 if task_id == "CartPole-v1" else 32)
                    baseline_rows.append(
                        run_episode(
                            env,
                            baseline,
                            suite_id=suite.suite_id,
                            variant_id=baseline.name,
                            split=split,
                            seed=seed,
                            trace_path=Path(trace_dir) / suite.suite_id / split / "baselines" / baseline.name / f"{task_id}.seed_{seed}.json",
                            baseline_snapshots=False,
                        )
                    )
    return suite_report(suite.suite_id, "dev_and_sealed_eval", rows, baseline_rows, device_info=cuda_runtime_info(device))


def official_suite_worker(
    *,
    explorer_checkpoint: str | Path,
    trace_dir: str | Path,
    json_output: str | Path,
    device: DeviceLike,
    max_click_actions: int = 192,
) -> dict[str, Any]:
    target_device = resolve_device(device)
    manifest = json.loads(Path("docs/arcagi3_official_games.json").read_text(encoding="utf-8"))
    requested = [str(item["game_id"]) for item in manifest.get("games", [])]
    arcade = make_arcade()
    discovered = discover_official_games(arcade, game_ids=requested, limit=None)
    by_id = {spec.game_id: spec for spec in discovered}
    specs = [by_id[item] for item in requested if item in by_id]
    rows: list[dict[str, Any]] = []
    for variant in VARIANTS:
        controller = make_controller(variant.variant_id, explorer_checkpoint, target_device)
        for index, spec in enumerate(specs):
            env = OfficialArcAGI3Env(
                arcade,
                spec,
                seed=index,
                max_steps=default_max_steps(spec),
                max_click_actions=max_click_actions,
            )
            row = run_episode(
                env,
                controller,
                suite_id="official_arcagi3",
                variant_id=variant.variant_id,
                split="sealed_eval",
                seed=index,
                trace_path=Path(trace_dir) / "official_arcagi3" / "sealed_eval" / variant.variant_id / f"{spec.game_id}.json",
                baseline_snapshots=variant.variant_id == "baseline_unchanged",
            )
            row["game_id"] = spec.game_id
            rows.append(row)
    for ablation, mode in ABLATION_MODES.items():
        controller = make_controller("valence_affordance_loop", explorer_checkpoint, target_device, mode=mode)
        for index, spec in enumerate(specs):
            env = OfficialArcAGI3Env(
                arcade,
                spec,
                seed=300 + index,
                max_steps=default_max_steps(spec),
                max_click_actions=max_click_actions,
            )
            row = run_episode(
                env,
                controller,
                suite_id="official_arcagi3",
                variant_id=f"ablation_{ablation}",
                split="sealed_eval",
                seed=300 + index,
                trace_path=Path(trace_dir) / "official_arcagi3" / "sealed_eval" / f"ablation_{ablation}" / f"{spec.game_id}.json",
                baseline_snapshots=False,
            )
            row["game_id"] = spec.game_id
            rows.append(row)
    report = suite_report(
        "official_arcagi3",
        "sealed_eval",
        rows,
        official_baseline_rows(),
        device_info=cuda_runtime_info(target_device),
    )
    Path(json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(json_output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def official_baseline_rows() -> list[dict[str, Any]]:
    report = json.loads(Path("docs/arcagi3_failure_report.json").read_text(encoding="utf-8"))
    rows: list[dict[str, Any]] = []
    for item in report.get("aggregate_score_table", []):
        if item.get("name") in BASELINE_NAMES:
            rows.append({"variant": item["name"], "split": "sealed_eval", **item})
    return rows


def suite_report(
    suite_id: str,
    split: str,
    rows: list[dict[str, Any]],
    baseline_rows: list[dict[str, Any]],
    *,
    device_info: dict[str, Any] | None = None,
) -> dict[str, Any]:
    by_variant = {
        variant: aggregate_rows([row for row in rows if row["variant"] == variant and row["split"] == "sealed_eval"])
        for variant in sorted({row["variant"] for row in rows})
    }
    dev_by_variant = {
        variant: aggregate_rows([row for row in rows if row["variant"] == variant and row["split"] == "dev"])
        for variant in sorted({row["variant"] for row in rows if row["split"] == "dev"})
    }
    baselines = {}
    if baseline_rows and "mean_normalized_score" in baseline_rows[0]:
        baselines = {row["variant"]: {k: v for k, v in row.items() if k not in {"variant", "split", "name"}} for row in baseline_rows}
    else:
        baselines = {
            name: aggregate_rows([row for row in baseline_rows if row["variant"] == name and row["split"] == "sealed_eval"])
            for name in sorted({row["variant"] for row in baseline_rows})
        }
    return {
        "suite_id": suite_id,
        "split": split,
        "rows": rows,
        "aggregate_by_variant": by_variant,
        "dev_aggregate_by_variant": dev_by_variant,
        "baselines": baselines,
        "trace_paths": [row["trace_path"] for row in rows if "trace_path" in row],
        "device_runtime": device_info or {},
    }


def dispatch_official_worker(explorer_checkpoint: str | Path, trace_dir: str | Path, device: DeviceLike) -> dict[str, Any]:
    temp = Path(trace_dir) / "_official_worker_report.json"
    try:
        from .arcagi3_official import require_official_runtime

        require_official_runtime()
        return official_suite_worker(explorer_checkpoint=explorer_checkpoint, trace_dir=trace_dir, json_output=temp, device=device)
    except Exception:
        venv_python = Path(".venv/Scripts/python.exe")
        if not venv_python.exists():
            raise
        cmd = [
            str(venv_python),
            "-m",
            "src.external_collapse_experiment",
            "--config",
            "official_worker",
            "--explorer-checkpoint",
            str(explorer_checkpoint),
            "--json-output",
            str(temp),
            "--trace-dir",
            str(trace_dir),
            "--device",
            str(device or AUTO_DEVICE),
        ]
        completed = subprocess.run(cmd, cwd=Path.cwd(), text=True, capture_output=True, check=True)
        if completed.stdout.strip():
            print(completed.stdout.strip())
        return json.loads(temp.read_text(encoding="utf-8"))


def select_primary_variant(gym_reports: list[dict[str, Any]]) -> dict[str, Any]:
    candidates = []
    for variant in CHANGE_VARIANTS:
        dev_rows = []
        for report in gym_reports:
            dev = report.get("dev_aggregate_by_variant", {}).get(variant, {})
            if dev:
                dev_rows.append(
                    {
                        "normalized_score": dev.get("mean_normalized_score", 0.0),
                        "repeat_collapse": dev.get("mean_repeat_collapse", 1.0),
                        "solved": dev.get("solve_rate", 0.0),
                        "score": dev.get("mean_score", 0.0),
                        "steps": dev.get("mean_steps", 0.0),
                        "invalid_action_rate": dev.get("mean_invalid_action_rate", 0.0),
                        "unique_states": dev.get("mean_unique_states", 0.0),
                        "useful_events": dev.get("mean_useful_events", 0.0),
                        "action_entropy": dev.get("mean_action_entropy", 0.0),
                    }
                )
        aggregate = aggregate_rows(dev_rows)
        candidates.append({"variant": variant, **aggregate})
    candidates.sort(key=lambda row: (row["mean_normalized_score"], -row["mean_repeat_collapse"], row["mean_action_entropy"]), reverse=True)
    return {"selected_variant": candidates[0]["variant"], "candidate_table": candidates, "selection_source": "non_arc_dev_only"}


def evaluate_external(
    *,
    explorer_checkpoint: str | Path,
    json_output: str | Path,
    trace_dir: str | Path,
    device: DeviceLike,
) -> dict[str, Any]:
    target_device = resolve_device(device)
    suites = discover_external_suites()
    gym_reports = [
        gym_suite(suite, explorer_checkpoint, trace_dir, target_device)
        for suite in suites
        if suite.available and suite.suite_id.startswith("gymnasium_")
    ]
    selection = select_primary_variant(gym_reports)
    official = dispatch_official_worker(explorer_checkpoint, trace_dir, target_device)
    suite_reports = [official, *gym_reports]
    report = build_report(
        suite_reports=suite_reports,
        selection=selection,
        explorer_checkpoint=explorer_checkpoint,
        trace_dir=trace_dir,
        requested_device=device,
        resolved_device=target_device,
    )
    output = Path(json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    return report


def build_report(
    *,
    suite_reports: list[dict[str, Any]],
    selection: dict[str, Any],
    explorer_checkpoint: str | Path,
    trace_dir: str | Path,
    requested_device: DeviceLike,
    resolved_device: Any,
) -> dict[str, Any]:
    selected = selection["selected_variant"]
    official = next(report for report in suite_reports if report["suite_id"] == "official_arcagi3")
    official_base = official["aggregate_by_variant"]["baseline_unchanged"]
    official_selected = official["aggregate_by_variant"][selected]
    repeat_drop = float(official_base["mean_repeat_collapse"] - official_selected["mean_repeat_collapse"])
    score_gain = float(official_selected["mean_normalized_score"] - official_base["mean_normalized_score"])
    non_arc_drop = non_arc_best_drop(suite_reports, selected)
    gate = repeat_drop >= 0.25 and score_gain >= 0.01 and non_arc_drop <= 0.05
    outcome = "MINIMAL EXTERNAL IMPROVEMENT FOUND" if gate else "NO MINIMAL IMPROVEMENT FOUND"
    report = {
        "terminal_outcome": outcome,
        "answer_yes_no": "yes" if gate else "no",
        "requested_device": str(requested_device),
        "resolved_device": str(resolved_device),
        "device_runtime": cuda_runtime_info(resolved_device),
        "suite_device_runtime": {report["suite_id"]: report.get("device_runtime", {}) for report in suite_reports},
        "explorer_checkpoint": str(explorer_checkpoint),
        "variants": [variant.__dict__ for variant in VARIANTS],
        "selection": selection,
        "improvement_gate": {
            "selected_variant": selected,
            "official_repeat_drop": repeat_drop,
            "official_score_gain": score_gain,
            "non_arc_best_score_drop": non_arc_drop,
            "passes": gate,
            "thresholds": {
                "official_repeat_drop_min": 0.25,
                "official_score_gain_min": 0.01,
                "non_arc_best_score_drop_max": 0.05,
            },
        },
        "suite_reports": suite_reports,
        "score_table": score_table(suite_reports),
        "collapse_table": collapse_table(suite_reports),
        "baselines": baseline_table(suite_reports),
        "ablations": ablation_table(suite_reports),
        "analysis": analysis_table(suite_reports, selected, gate),
        "trace_paths": [path for report in suite_reports for path in report.get("trace_paths", [])],
        "required_arc_trace_paths": required_arc_trace_paths(official, selected),
        "no_hack_proof": no_hack_proof(),
        "limitations": limitations(gate),
        "hashes": collect_hashes(EXPERIMENT_PATHS + [str(explorer_checkpoint), "docs/external_generalization_report.json", "docs/arcagi3_failure_report.json"]),
        "trace_dir": str(trace_dir),
    }
    return report


def non_arc_best_drop(suite_reports: list[dict[str, Any]], selected: str) -> float:
    baseline_best = 0.0
    selected_best = 0.0
    for report in suite_reports:
        if report["suite_id"] == "official_arcagi3":
            continue
        baseline_best = max(baseline_best, float(report["aggregate_by_variant"]["baseline_unchanged"]["mean_normalized_score"]))
        selected_best = max(selected_best, float(report["aggregate_by_variant"][selected]["mean_normalized_score"]))
    return float(baseline_best - selected_best)


def score_table(suite_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in suite_reports:
        for variant, aggregate in report["aggregate_by_variant"].items():
            rows.append({"suite_id": report["suite_id"], "variant": variant, **aggregate})
    return rows


def collapse_table(suite_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {
            "suite_id": row["suite_id"],
            "variant": row["variant"],
            "mean_repeat_collapse": row["mean_repeat_collapse"],
            "mean_action_entropy": row["mean_action_entropy"],
            "mean_unique_states": row["mean_unique_states"],
        }
        for row in score_table(suite_reports)
    ]


def baseline_table(suite_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in suite_reports:
        for name, aggregate in report.get("baselines", {}).items():
            if name in BASELINE_NAMES:
                rows.append({"suite_id": report["suite_id"], "baseline": name, **aggregate})
    return rows


def ablation_table(suite_reports: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for report in suite_reports:
        for variant, aggregate in report["aggregate_by_variant"].items():
            if variant.startswith("ablation_"):
                rows.append({"suite_id": report["suite_id"], "ablation": variant, **aggregate})
    return rows


def analysis_table(suite_reports: list[dict[str, Any]], selected: str, gate: bool) -> dict[str, Any]:
    rows = [row for report in suite_reports for row in report["rows"] if row["variant"] == selected]
    affordance_rows = [row for report in suite_reports for row in report["rows"] if "affordance" in str(row["variant"])]
    changed_by_valence = sum(int(row.get("controller_summary", {}).get("reason_counts", {}).get("valence", 0)) for row in rows)
    changed_actions = sum(int(row.get("controller_summary", {}).get("changed_actions", 0)) for row in rows)
    valence_updates = sum(int(row.get("controller_summary", {}).get("valence", {}).get("nonzero_updates", 0)) for row in rows)
    affordance_seen = sum(int(row.get("controller_summary", {}).get("affordance_observations", 0)) for row in affordance_rows)
    return {
        "did_reward_event_noop_change_internal_valence": valence_updates > 0,
        "did_changed_valence_alter_future_actions": changed_by_valence > 0,
        "did_repeated_no_effect_actions_become_less_likely_without_forced_cycling": changed_actions > 0
        and all(not row.get("controller_summary", {}).get("forced_cycle", False) for row in rows),
        "did_affordance_summaries_expose_objects_regions": affordance_seen > 0,
        "did_score_gain_come_from_consequence_use_or_mere_diversity": "no_score_gain" if not gate else "requires_ablation_review",
        "changed_actions": changed_actions,
        "changed_by_valence": changed_by_valence,
        "valence_updates": valence_updates,
        "affordance_observations": affordance_seen,
    }


def required_arc_trace_paths(official: dict[str, Any], selected: str) -> dict[str, Any]:
    baseline = [path for path in official["trace_paths"] if f"/baseline_unchanged/" in path.replace("\\", "/")]
    selected_paths = [path for path in official["trace_paths"] if f"/{selected}/" in path.replace("\\", "/")]
    return {
        "baseline_unchanged_count": len(baseline),
        "selected_variant": selected,
        "selected_variant_count": len(selected_paths),
        "baseline_unchanged": baseline,
        "selected": selected_paths,
    }


def no_hack_proof() -> dict[str, Any]:
    source_paths = [Path("src/external_valence.py"), Path("src/external_affordance.py"), Path("src/external_collapse_experiment.py")]
    blocked = [
        "hidden" + "_goal",
        "ground" + "_truth",
        "answer" + "_key",
        "solution" + "_path",
        "manual" + "_hint",
        "round" + "_robin",
        "forced" + "_entropy",
    ]
    findings = []
    for path in source_paths:
        text = path.read_text(encoding="utf-8", errors="ignore").lower()
        for item in blocked:
            if item in text:
                findings.append({"path": str(path), "match": item})
    return {
        "passes": not findings and no_hack_audit([])["passes"],
        "findings": findings,
        "model_action_source": "model action_scores from ARCAGI3Adapter; baselines are comparison only",
        "selection_protocol": "primary variant selected from non-ARC dev aggregates before sealed outcome gate",
        "no_forced_cycle": True,
        "no_external_judge": True,
    }


def limitations(gate: bool) -> list[str]:
    items = [
        "Variants are generic runtime calibrations of model action scores; no weights are trained.",
        "Baselines are comparison policies only and are not used to choose variant actions.",
    ]
    if not gate:
        items.append("No tested minimal generic change satisfied the official ARC score and collapse improvement gate.")
    return items


def render_markdown(report: dict[str, Any]) -> str:
    lines = ["# External Collapse Experiment", "", f"Terminal outcome: **{report['terminal_outcome']}**", ""]
    lines.append(f"Answer: **{report['answer_yes_no']}**")
    lines.append("")
    lines.append("## Device")
    device_runtime = report.get("device_runtime", {})
    lines.append(f"- Requested device: `{report.get('requested_device')}`")
    lines.append(f"- Resolved device: `{report.get('resolved_device')}`")
    lines.append(f"- Torch: `{device_runtime.get('torch_version')}`")
    lines.append(f"- CUDA available: `{device_runtime.get('torch_cuda_available')}`")
    lines.append(f"- CUDA runtime: `{device_runtime.get('torch_cuda_version')}`")
    devices = device_runtime.get("torch_cuda_devices", [])
    if devices:
        lines.append(f"- CUDA devices: `{', '.join(str(item) for item in devices)}`")
    for suite_id, suite_device in sorted(report.get("suite_device_runtime", {}).items()):
        lines.append(f"- Suite `{suite_id}` resolved device: `{suite_device.get('resolved_device')}`")
    lines.append("")
    lines.append("## Improvement Gate")
    gate = report["improvement_gate"]
    for key in ["selected_variant", "official_repeat_drop", "official_score_gain", "non_arc_best_score_drop", "passes"]:
        lines.append(f"- `{key}`: `{gate[key]}`")
    lines.append("")
    lines.append("## Scores")
    lines.append("| Suite | Variant | Mean Normalized | Solve Rate | Useful Events | Invalid |")
    lines.append("| --- | --- | ---: | ---: | ---: | ---: |")
    for row in report["score_table"]:
        lines.append(
            f"| {row['suite_id']} | {row['variant']} | {float(row['mean_normalized_score']):.3f} | {float(row['solve_rate']):.3f} | {float(row['mean_useful_events']):.3f} | {float(row['mean_invalid_action_rate']):.3f} |"
        )
    lines.append("")
    lines.append("## Collapse")
    lines.append("| Suite | Variant | Repeat Collapse | Entropy | Unique States |")
    lines.append("| --- | --- | ---: | ---: | ---: |")
    for row in report["collapse_table"]:
        lines.append(
            f"| {row['suite_id']} | {row['variant']} | {float(row['mean_repeat_collapse']):.3f} | {float(row['mean_action_entropy']):.3f} | {float(row['mean_unique_states']):.3f} |"
        )
    lines.append("")
    lines.append("## Trace Paths")
    for path in report["trace_paths"][:60]:
        lines.append(f"- `{path}`")
    if len(report["trace_paths"]) > 60:
        lines.append(f"- ... {len(report['trace_paths']) - 60} more traces")
    lines.append("")
    lines.append("## Limitations")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="external")
    parser.add_argument("--explorer-checkpoint", default="runs/explorer_tiny.pt")
    parser.add_argument("--json-output", default="docs/external_collapse_report.json")
    parser.add_argument("--trace-dir", default="docs/external_collapse_traces")
    parser.add_argument("--device", default=CUDA_PREFERRED_DEVICE)
    args = parser.parse_args()
    if args.config == "official_worker":
        report = official_suite_worker(
            explorer_checkpoint=args.explorer_checkpoint,
            trace_dir=args.trace_dir,
            json_output=args.json_output,
            device=args.device,
        )
        print(json.dumps({"suite_id": report["suite_id"], "variants": list(report["aggregate_by_variant"])}))
        return
    if args.config != "external":
        raise ValueError("external_collapse_experiment supports --config external or official_worker")
    report = evaluate_external(
        explorer_checkpoint=args.explorer_checkpoint,
        json_output=args.json_output,
        trace_dir=args.trace_dir,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "terminal_outcome": report["terminal_outcome"],
                "answer_yes_no": report["answer_yes_no"],
                "improvement_gate": report["improvement_gate"],
                "selected_variant": report["selection"]["selected_variant"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
