from __future__ import annotations

import argparse
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .arcagi3_adapter import ARCAGI3Adapter, ArcAGI3Observation, ArcAGI3StepResult
from .arcagi3_official import OfficialArcAGI3Env, default_max_steps, discover_official_games, make_arcade
from .base_world_model import ExternalBaseWorldModel, load_external_arm
from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .external_collapse_experiment import (
    CUDA_PREFERRED_DEVICE,
    adapter_device_summary,
    cuda_runtime_info,
    make_adapter,
    official_baseline_rows,
    run_episode,
    suite_report,
)
from .external_eval import GymnasiumExternalEnv
from .external_registry import ExternalSuite, discover_external_suites
from .arcagi3_eval import aggregate_rows
from .arcagi3_baselines import build_baselines


@dataclass(frozen=True)
class BaseArmSpec:
    arm_id: str
    checkpoint_arm: str | None = None
    uses_external_base: bool = False
    null_control: bool = False


ARM_SPECS = [
    BaseArmSpec("old_base_unchanged"),
    BaseArmSpec("old_base_world_model", checkpoint_arm="old_base_world_model", uses_external_base=True),
    BaseArmSpec("old_base_finetuned", checkpoint_arm="old_base_finetuned", uses_external_base=True),
    BaseArmSpec("from_scratch_external_base", checkpoint_arm="from_scratch_external_base", uses_external_base=True),
    BaseArmSpec("null_training_control", checkpoint_arm="null_training_control", uses_external_base=True, null_control=True),
]
CHANGE_ARMS = ["old_base_world_model", "old_base_finetuned", "from_scratch_external_base"]
ABLATIONS = {
    "no_external_base": {"disable_external_base": True},
    "no_world_model": {"disable_world_model": True},
    "no_memory": {"disable_memory": True},
    "no_affordance": {"disable_affordance": True},
}


def arm_by_id(arm_id: str) -> BaseArmSpec:
    for arm in ARM_SPECS:
        if arm.arm_id == arm_id:
            return arm
    raise KeyError(arm_id)


class ExternalBaseController:
    def __init__(
        self,
        adapter: ARCAGI3Adapter,
        arm: BaseArmSpec,
        *,
        model: ExternalBaseWorldModel | None = None,
        disable_external_base: bool = False,
        disable_world_model: bool = False,
        disable_memory: bool = False,
        disable_affordance: bool = False,
        base_weight: float = 0.35,
        external_weight: float = 0.85,
    ) -> None:
        self.adapter = adapter
        self.arm = arm
        self.model = model
        self.disable_external_base = disable_external_base
        self.disable_world_model = disable_world_model
        self.disable_memory = disable_memory
        self.disable_affordance = disable_affordance
        self.base_weight = float(base_weight)
        self.external_weight = float(external_weight)
        self.changed_actions = 0
        self.frames = 0
        self.reason_counts = {"world_model": 0, "memory": 0, "affordance": 0}
        self.memory: torch.Tensor | None = None

    @property
    def name(self) -> str:
        return self.arm.arm_id

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.adapter.reset()
        self.changed_actions = 0
        self.frames = 0
        self.reason_counts = {"world_model": 0, "memory": 0, "affordance": 0}
        self.memory = None

    def choose_action(self, observation: ArcAGI3Observation) -> tuple[str, dict[str, Any]]:
        base_action, diagnostics = self.adapter.choose_action(observation)
        legal = tuple(observation.available_actions)
        base_scores = {action: float(diagnostics.get("action_scores", {}).get(action, 0.0)) for action in legal}
        use_learned_calibration = self.model is not None and not self.disable_external_base
        adjusted = (
            {action: self.base_weight * value for action, value in normalized_scores(base_scores).items()}
            if use_learned_calibration
            else dict(base_scores)
        )
        external_scores = {action: 0.0 for action in legal}
        external_calibrated = {action: 0.0 for action in legal}
        external_diag: dict[str, Any] = {}
        candidate_memory: torch.Tensor | None = None
        if self.model is not None and not self.disable_external_base:
            external_scores, external_diag, candidate_memory = self.model.score_actions(
                observation,
                legal,
                memory=self.memory,
                disable_memory=self.disable_memory,
                disable_world_model=self.disable_world_model,
                disable_affordance=self.disable_affordance,
            )
            external_calibrated = normalized_scores(external_scores)
            for action, score in external_scores.items():
                adjusted[action] = adjusted.get(action, 0.0) + self.external_weight * float(external_calibrated.get(action, score))
        if self.arm.arm_id == "old_base_unchanged" or self.arm.null_control or self.disable_external_base:
            chosen = base_action
        else:
            chosen = max(legal, key=lambda action: (adjusted.get(action, -1.0e9), -legal.index(action)))
        if candidate_memory is not None:
            try:
                chosen_index = list(legal).index(chosen)
            except ValueError:
                chosen_index = 0
            if candidate_memory.ndim >= 2 and candidate_memory.shape[0] == len(legal):
                self.memory = candidate_memory[chosen_index : chosen_index + 1].detach()
            else:
                self.memory = candidate_memory[:1].detach()
        changed = chosen != base_action
        self.changed_actions += int(changed)
        self.frames += 1
        if changed:
            if not self.disable_world_model:
                self.reason_counts["world_model"] += 1
            if not self.disable_memory:
                self.reason_counts["memory"] += 1
            if not self.disable_affordance:
                self.reason_counts["affordance"] += 1
        diagnostics["adapter_action_scores"] = base_scores
        diagnostics["action_scores"] = adjusted
        diagnostics["policy"] = {
            **diagnostics.get("policy", {}),
            "variant": self.arm.arm_id,
            "base_action": base_action,
            "chosen_action": chosen,
            "changed_action": changed,
            "top_base_scores": top_scores(base_scores),
            "top_adjusted_scores": top_scores(adjusted),
            "external_scores": top_scores(external_scores),
            "external_calibrated_scores": top_scores(external_calibrated),
            "external_diagnostics": external_diag,
            "score_weights": {
                "base": self.base_weight if use_learned_calibration else 1.0,
                "external": self.external_weight if use_learned_calibration else 0.0,
            },
            "ablation": {
                "disable_external_base": self.disable_external_base,
                "disable_world_model": self.disable_world_model,
                "disable_memory": self.disable_memory,
                "disable_affordance": self.disable_affordance,
            },
            "forced_cycle": False,
        }
        return chosen, diagnostics

    def observe_transition(self, action: str, result: ArcAGI3StepResult) -> None:
        self.adapter.observe_transition(action, result)

    def summary(self) -> dict[str, Any]:
        return {
            "arm": self.arm.arm_id,
            "changed_actions": self.changed_actions,
            "frames": self.frames,
            "reason_counts": dict(self.reason_counts),
            "forced_cycle": False,
            "device": adapter_device_summary(self.adapter),
            "external_base_enabled": self.model is not None and not self.disable_external_base,
            "ablation": {
                "disable_external_base": self.disable_external_base,
                "disable_world_model": self.disable_world_model,
                "disable_memory": self.disable_memory,
                "disable_affordance": self.disable_affordance,
            },
        }


def top_scores(scores: dict[str, float], limit: int = 8) -> dict[str, float]:
    return {
        action: round(float(value), 6)
        for action, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    }


def normalized_scores(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    values = [float(value) for value in scores.values()]
    mean = sum(values) / len(values)
    span = max(values) - min(values)
    if span <= 1.0e-9:
        return {action: 0.0 for action in scores}
    return {
        action: max(min((float(value) - mean) / span * 2.0, 1.0), -1.0)
        for action, value in scores.items()
    }


def make_base_controller(
    *,
    arm_id: str,
    external_checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    device: DeviceLike,
    ablation: str | None = None,
) -> ExternalBaseController:
    spec = arm_by_id(arm_id)
    target_device = resolve_device(device)
    model = None
    if spec.checkpoint_arm is not None:
        model = load_external_arm(external_checkpoint, spec.checkpoint_arm, device=target_device)
    base_weight = 0.35
    external_weight = 0.85
    manifest = getattr(model, "external_manifest", {}) if model is not None else {}
    if isinstance(manifest, dict) and manifest.get("format") == "real_arc_external_base_manifest_v1":
        base_weight = 0.05
        external_weight = 1.45
    flags = dict(ABLATIONS.get(ablation or "", {}))
    return ExternalBaseController(
        make_adapter(explorer_checkpoint, target_device),
        spec,
        model=model,
        base_weight=base_weight,
        external_weight=external_weight,
        **flags,
    )


def gym_suite(
    suite: ExternalSuite,
    *,
    external_checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    trace_dir: str | Path,
    device: DeviceLike,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    baseline_rows: list[dict[str, Any]] = []
    for split, seeds in [("dev", [0, 1]), ("sealed_eval", [100, 101, 102])]:
        for task_id in suite.tasks:
            for arm in ARM_SPECS:
                for seed in seeds:
                    controller = make_base_controller(
                        arm_id=arm.arm_id,
                        external_checkpoint=external_checkpoint,
                        explorer_checkpoint=explorer_checkpoint,
                        device=device,
                    )
                    env = GymnasiumExternalEnv(suite.suite_id, task_id, max_steps=80 if task_id == "CartPole-v1" else 32)
                    rows.append(
                        run_episode(
                            env,
                            controller,
                            suite_id=suite.suite_id,
                            variant_id=arm.arm_id,
                            split=split,
                            seed=seed,
                            trace_path=Path(trace_dir) / suite.suite_id / split / arm.arm_id / f"{task_id}.seed_{seed}.json",
                            baseline_snapshots=arm.arm_id == "old_base_unchanged",
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


def select_primary_arm(gym_reports: list[dict[str, Any]]) -> dict[str, Any]:
    priority = {"old_base_finetuned": 3, "old_base_world_model": 2, "from_scratch_external_base": 1}
    candidates = []
    for arm in CHANGE_ARMS:
        dev_rows = []
        for report in gym_reports:
            dev = report.get("dev_aggregate_by_variant", {}).get(arm, {})
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
        candidates.append({"variant": arm, "priority": priority[arm], **aggregate})
    candidates.sort(
        key=lambda row: (
            row.get("mean_normalized_score", 0.0),
            row.get("mean_useful_events", 0.0),
            -row.get("mean_repeat_collapse", 1.0),
            row.get("priority", 0),
        ),
        reverse=True,
    )
    return {
        "selected_variant": candidates[0]["variant"],
        "candidate_table": candidates,
        "selection_source": "non_arc_dev_only_before_official_sealed_eval",
    }


def official_suite_worker(
    *,
    external_checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    trace_dir: str | Path,
    json_output: str | Path,
    selected_arm: str,
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
    for arm in ARM_SPECS:
        controller = make_base_controller(
            arm_id=arm.arm_id,
            external_checkpoint=external_checkpoint,
            explorer_checkpoint=explorer_checkpoint,
            device=target_device,
        )
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
                variant_id=arm.arm_id,
                split="sealed_eval",
                seed=index,
                trace_path=Path(trace_dir) / "official_arcagi3" / "sealed_eval" / arm.arm_id / f"{spec.game_id}.json",
                baseline_snapshots=arm.arm_id == "old_base_unchanged",
            )
            row["game_id"] = spec.game_id
            rows.append(row)
    for ablation in ABLATIONS:
        variant_id = f"ablation_{ablation}"
        controller = make_base_controller(
            arm_id=selected_arm,
            external_checkpoint=external_checkpoint,
            explorer_checkpoint=explorer_checkpoint,
            device=target_device,
            ablation=ablation,
        )
        for index, spec in enumerate(specs):
            env = OfficialArcAGI3Env(
                arcade,
                spec,
                seed=500 + index,
                max_steps=default_max_steps(spec),
                max_click_actions=max_click_actions,
            )
            row = run_episode(
                env,
                controller,
                suite_id="official_arcagi3",
                variant_id=variant_id,
                split="sealed_eval",
                seed=500 + index,
                trace_path=Path(trace_dir) / "official_arcagi3" / "sealed_eval" / variant_id / f"{spec.game_id}.json",
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


def dispatch_official_worker(
    *,
    external_checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    trace_dir: str | Path,
    selected_arm: str,
    device: DeviceLike,
) -> dict[str, Any]:
    temp = Path(trace_dir) / "_official_base_worker_report.json"
    try:
        from .arcagi3_official import require_official_runtime

        require_official_runtime()
        return official_suite_worker(
            external_checkpoint=external_checkpoint,
            explorer_checkpoint=explorer_checkpoint,
            trace_dir=trace_dir,
            json_output=temp,
            selected_arm=selected_arm,
            device=device,
        )
    except Exception:
        venv_python = Path(".venv/Scripts/python.exe")
        if not venv_python.exists():
            raise
        cmd = [
            str(venv_python),
            "-m",
            "src.base_retrain_experiment",
            "--config",
            "official_worker",
            "--external-checkpoint",
            str(external_checkpoint),
            "--explorer-checkpoint",
            str(explorer_checkpoint),
            "--json-output",
            str(temp),
            "--trace-dir",
            str(trace_dir),
            "--selected-arm",
            selected_arm,
            "--device",
            str(device or AUTO_DEVICE),
        ]
        completed = subprocess.run(cmd, cwd=Path.cwd(), text=True, capture_output=True, check=True)
        if completed.stdout.strip():
            print(completed.stdout.strip())
        return json.loads(temp.read_text(encoding="utf-8"))


def evaluate_suites(
    *,
    external_checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    trace_dir: str | Path,
    device: DeviceLike,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    target_device = resolve_device(device)
    gym_reports = [
        gym_suite(
            suite,
            external_checkpoint=external_checkpoint,
            explorer_checkpoint=explorer_checkpoint,
            trace_dir=trace_dir,
            device=target_device,
        )
        for suite in discover_external_suites()
        if suite.available and suite.suite_id.startswith("gymnasium_")
    ]
    selection = select_primary_arm(gym_reports)
    official = dispatch_official_worker(
        external_checkpoint=external_checkpoint,
        explorer_checkpoint=explorer_checkpoint,
        trace_dir=trace_dir,
        selected_arm=selection["selected_variant"],
        device=target_device,
    )
    return [official, *gym_reports], selection


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["official_worker"], default="official_worker")
    parser.add_argument("--external-checkpoint", default="frozen/external_base_v1.pt")
    parser.add_argument("--explorer-checkpoint", default="runs/explorer_tiny.pt")
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--selected-arm", required=True)
    parser.add_argument("--device", default=CUDA_PREFERRED_DEVICE)
    args = parser.parse_args()
    del args.config
    report = official_suite_worker(
        external_checkpoint=args.external_checkpoint,
        explorer_checkpoint=args.explorer_checkpoint,
        trace_dir=args.trace_dir,
        json_output=args.json_output,
        selected_arm=args.selected_arm,
        device=args.device,
    )
    print(json.dumps({"suite_id": report["suite_id"], "variants": sorted(report["aggregate_by_variant"])}, indent=2))


if __name__ == "__main__":
    main()
