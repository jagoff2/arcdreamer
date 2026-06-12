from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

from .arc_affordance_baseline import AFFORDANCE_ABLATIONS, AFFORDANCE_VARIANTS, ArcAffordancePolicy
from .arc_affordance_report import build_report, render_markdown
from .arc_affordance_search import changed_regions, component_summary, observation_hash, visual_hash
from .arcagi3_adapter import ArcAGI3Observation, ArcAGI3StepResult
from .arcagi3_baselines import build_baselines
from .arcagi3_trace import action_entropy, repeat_collapse
from .arcagi3_official import (
    OfficialArcAGI3Env,
    default_max_steps,
    discover_official_games,
    make_arcade,
    official_runtime_versions,
    require_official_runtime,
)
from .device import resolve_device
from .external_collapse_experiment import adapter_device_summary, cuda_runtime_info, make_adapter


COMPARISON_NAMES = [
    "random_legal",
    "repeat_last_action",
    "coverage_graph_exploration",
    "novelty_first",
    "greedy_observable_score_delta",
    "oracle_free_observed_graph_bfs",
    "old_explorer",
]


class OldExplorerPolicy:
    name = "old_explorer"

    def __init__(self, explorer_checkpoint: str | Path, device: str) -> None:
        self.requested_device = str(device)
        self.adapter = make_adapter(explorer_checkpoint, device=device)
        self.device_summary = adapter_device_summary(self.adapter)

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.adapter.reset()

    def choose_action(self, observation: ArcAGI3Observation) -> tuple[str, dict[str, Any]]:
        action, diagnostics = self.adapter.choose_action(observation)
        diagnostics["policy"] = {**diagnostics.get("policy", {}), "variant": self.name}
        return action, diagnostics

    def observe_transition(self, action: str, result: ArcAGI3StepResult, *, before: ArcAGI3Observation | None = None) -> None:
        del before
        self.adapter.observe_transition(action, result)

    def summary(self) -> dict[str, Any]:
        return {
            "variant": self.name,
            "comparison_only": True,
            "requested_device": self.requested_device,
            "device": self.device_summary,
        }


def build_controllers(*, explorer_checkpoint: str | Path, device: str, include_ablations: bool = True) -> list[Any]:
    controllers: list[Any] = [ArcAffordancePolicy(variant) for variant in AFFORDANCE_VARIANTS]
    if include_ablations:
        controllers.extend(ArcAffordancePolicy(variant) for variant in AFFORDANCE_ABLATIONS)
    controllers.extend(build_baselines())
    controllers.append(OldExplorerPolicy(explorer_checkpoint, device))
    return controllers


def reset_controller(controller: Any, seed: int) -> None:
    if hasattr(controller, "reset"):
        controller.reset(seed)


def choose_controller(controller: Any, observation: ArcAGI3Observation) -> tuple[str, dict[str, Any]]:
    if hasattr(controller, "choose_action"):
        return controller.choose_action(observation)
    action = controller.choose(observation)
    return action, {"policy": {"variant": getattr(controller, "name", type(controller).__name__)}}


def observe_controller(controller: Any, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult) -> None:
    if hasattr(controller, "observe_transition"):
        controller.observe_transition(action, result, before=before)
    else:
        controller.observe(before, action, result)


def baseline_action_snapshot(observation: ArcAGI3Observation, seed: int, step: int) -> dict[str, str]:
    snapshot: dict[str, str] = {}
    for baseline in build_baselines():
        baseline.reset(seed + step * 7919)
        try:
            snapshot[baseline.name] = baseline.choose(observation)
        except Exception as exc:
            snapshot[baseline.name] = f"error:{type(exc).__name__}"
    return snapshot


def run_episode(
    env: OfficialArcAGI3Env,
    controller: Any,
    *,
    variant: str,
    game: str,
    seed: int,
    trace_path: str | Path,
) -> dict[str, Any]:
    reset_controller(controller, seed)
    obs = env.reset(seed)
    actions: list[str] = []
    invalid = 0
    useful_events = 0
    unique_states = {visual_hash(obs)}
    trace_steps: list[dict[str, Any]] = []
    last_before: ArcAGI3Observation | None = None
    try:
        for step in range(int(env.max_steps)):
            before = obs
            baseline_actions = baseline_action_snapshot(before, seed, step)
            action, diagnostics = choose_controller(controller, before)
            invalid_action = action not in before.available_actions
            if invalid_action:
                invalid += 1
                action = before.available_actions[0] if before.available_actions else "wait"
            result = env.step(action)
            actions.append(action)
            useful = float(result.reward) > 0.0
            useful_events += int(useful)
            unique_states.add(visual_hash(result.observation))
            regions = changed_regions(before, result.observation)
            observe_controller(controller, before, action, result)
            last_before = before
            trace_steps.append(
                {
                    "game": game,
                    "step": step,
                    "obs_hash": observation_hash(before),
                    "legal_action_count": len(before.available_actions),
                    "chosen_action": action,
                    "baseline_actions": baseline_actions,
                    "component_summary": compact_component_summary(before),
                    "changed_regions": regions,
                    "action_effect_memory": action_effect_snapshot(controller),
                    "score_delta": float(result.reward),
                    "event_delta": list(result.info.get("events", [])),
                    "terminal": bool(result.terminated or result.truncated),
                    "invalid_flag": bool(invalid_action),
                    "repeat_cycle_stats": repeat_cycle_snapshot(controller, actions, result.observation),
                    "choice_reason": diagnostics.get("policy", {}).get("choice_reason", diagnostics.get("policy", {})),
                }
            )
            obs = result.observation
            if result.terminated or result.truncated:
                break
    finally:
        scorecard = env.close()
    summary = {
        "game": game,
        "variant": variant,
        "seed": seed,
        "solved": bool(env.terminated),
        "score": float(env.score),
        "normalized_score": float(env.normalized_score()),
        "useful_events": useful_events,
        "steps": len(actions),
        "action_entropy": action_entropy(actions),
        "repeat_collapse": repeat_collapse(actions),
        "invalid_action_rate": float(invalid / max(len(actions), 1)),
        "unique_states": len(unique_states),
        "actions": actions[:96],
        "controller_summary": controller.summary() if hasattr(controller, "summary") else {},
        "scorecard": compact_scorecard(scorecard),
        "action_effect_hit_rate": controller.search.effect.action_effect_hit_rate() if isinstance(controller, ArcAffordancePolicy) else 0.0,
        "no_op_avoidance_rate": controller.search.effect.no_op_avoidance_rate() if isinstance(controller, ArcAffordancePolicy) else 0.0,
    }
    del last_before
    output = Path(trace_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        json.dumps(
            {
                "format": "arc_affordance_trace_v1",
                "metadata": {
                    "game": game,
                    "variant": variant,
                    "seed": seed,
                    "public_inputs_only": True,
                    "no_neural_training": True,
                },
                "summary": summary,
                "steps": trace_steps,
            },
            separators=(",", ":"),
        ),
        encoding="utf-8",
    )
    summary["trace_path"] = str(output)
    return summary


def compact_component_summary(observation: ArcAGI3Observation) -> dict[str, Any]:
    summary = component_summary(observation, limit=8)
    return {
        "obs_hash": summary["obs_hash"],
        "component_count": summary["component_count"],
        "value_counts": summary["value_counts"],
        "largest": summary["largest"][:8],
    }


def action_effect_snapshot(controller: Any) -> dict[str, Any]:
    if isinstance(controller, ArcAffordancePolicy):
        return controller.search.effect.snapshot(limit=6)
    return {"observed_actions": 0, "top_actions": []}


def repeat_cycle_snapshot(controller: Any, actions: list[str], observation: ArcAGI3Observation) -> dict[str, Any]:
    if isinstance(controller, ArcAffordancePolicy):
        graph = controller.search.graph.cycle_stats(observation)
    else:
        graph = {"seen_states": 0, "known_outgoing": 0, "known_self_loops": []}
    return {
        **graph,
        "recent_actions": actions[-12:],
        "repeat_collapse_so_far": repeat_collapse(actions),
        "action_entropy_so_far": action_entropy(actions),
    }


def compact_scorecard(scorecard: Any) -> dict[str, Any]:
    if not isinstance(scorecard, dict):
        return {}
    return {
        key: scorecard.get(key)
        for key in ["id", "score", "state", "levels_completed", "win_levels"]
        if key in scorecard
    }


def evaluate_official(
    *,
    json_output: str | Path,
    trace_dir: str | Path,
    explorer_checkpoint: str | Path,
    device: str,
    include_ablations: bool = True,
) -> dict[str, Any]:
    require_official_runtime()
    resolved_device = resolve_device(device)
    manifest = json.loads(Path("docs/arcagi3_official_games.json").read_text(encoding="utf-8"))
    requested = [str(item["game_id"]) for item in manifest.get("games", [])]
    arcade = make_arcade()
    discovered = discover_official_games(arcade, game_ids=requested, limit=None)
    by_id = {spec.game_id: spec for spec in discovered}
    specs = [by_id[item] for item in requested if item in by_id]
    rows: list[dict[str, Any]] = []
    trace_paths: list[str] = []
    controllers = build_controllers(
        explorer_checkpoint=explorer_checkpoint,
        device=str(resolved_device),
        include_ablations=include_ablations,
    )
    old_explorer_device = {}
    for controller in controllers:
        if isinstance(controller, OldExplorerPolicy):
            old_explorer_device = controller.summary().get("device", {})
            break
    for controller in controllers:
        variant = str(getattr(controller, "name", type(controller).__name__))
        for index, spec in enumerate(specs):
            env = OfficialArcAGI3Env(
                arcade,
                spec,
                seed=index,
                max_steps=default_max_steps(spec),
                max_click_actions=192,
            )
            trace_path = Path(trace_dir) / "official_arcagi3" / "sealed_eval" / variant / f"{spec.game_id}.json"
            row = run_episode(env, controller, variant=variant, game=spec.game_id, seed=index, trace_path=trace_path)
            rows.append(row)
            trace_paths.append(str(trace_path))
    runtime_status = {
        "official_runtime_available": True,
        "runtime_versions": official_runtime_versions(),
        "requested_device": str(device),
        "resolved_device": str(resolved_device),
        "device_runtime": cuda_runtime_info(resolved_device),
        "old_explorer_device": old_explorer_device,
    }
    report = build_report(
        rows=rows,
        trace_paths=trace_paths,
        runtime_status=runtime_status,
        trace_dir=trace_dir,
        config="official",
    )
    output = Path(json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    return report


def dispatch_official(
    *,
    json_output: str | Path,
    trace_dir: str | Path,
    explorer_checkpoint: str | Path,
    device: str,
    include_ablations: bool,
) -> dict[str, Any]:
    try:
        require_official_runtime()
    except Exception:
        venv_python = Path(".venv/Scripts/python.exe")
        if not venv_python.exists() or Path(sys.executable).resolve() == venv_python.resolve():
            raise
        cmd = [
            str(venv_python),
            "-m",
            "src.arc_affordance_eval",
            "--config",
            "official_worker",
            "--json-output",
            str(json_output),
            "--trace-dir",
            str(trace_dir),
            "--explorer-checkpoint",
            str(explorer_checkpoint),
            "--device",
            str(device),
        ]
        if not include_ablations:
            cmd.append("--no-ablations")
        completed = subprocess.run(cmd, cwd=Path.cwd(), text=True, capture_output=True, check=True)
        if completed.stdout.strip():
            print(completed.stdout.strip())
        return json.loads(Path(json_output).read_text(encoding="utf-8"))
    return evaluate_official(
        json_output=json_output,
        trace_dir=trace_dir,
        explorer_checkpoint=explorer_checkpoint,
        device=device,
        include_ablations=include_ablations,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["official", "official_worker"], default="official")
    parser.add_argument("--json-output", default="docs/arc_affordance_report.json")
    parser.add_argument("--trace-dir", default="docs/arc_affordance_traces")
    parser.add_argument("--explorer-checkpoint", default="runs/explorer_tiny.pt")
    parser.add_argument("--device", default="cuda", help="Device for old explorer comparison adapter.")
    parser.add_argument("--no-ablations", action="store_true")
    args = parser.parse_args()
    include_ablations = not bool(args.no_ablations)
    if args.config == "official_worker":
        report = evaluate_official(
            json_output=args.json_output,
            trace_dir=args.trace_dir,
            explorer_checkpoint=args.explorer_checkpoint,
            device=args.device,
            include_ablations=include_ablations,
        )
    else:
        report = dispatch_official(
            json_output=args.json_output,
            trace_dir=args.trace_dir,
            explorer_checkpoint=args.explorer_checkpoint,
            device=args.device,
            include_ablations=include_ablations,
        )
    print(
        json.dumps(
            {
                "terminal_outcome": report["terminal_outcome"],
                "answer_yes_no": report["answer_yes_no"],
                "selected_variant": report["selected_variant"],
                "improvement_gate": report["improvement_gate"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
