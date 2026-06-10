from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from .arcagi3_adapter import ARCAGI3Adapter
from .arcagi3_baselines import BaselinePolicy, build_baselines
from .arcagi3_eval import aggregate_rows, collect_hashes, state_signature
from .arcagi3_official import (
    DEFAULT_ENVIRONMENTS_DIR,
    DEFAULT_RECORDINGS_DIR,
    OfficialArcAGI3Env,
    OfficialGameSpec,
    default_max_steps,
    discover_official_games,
    make_arcade,
    official_runtime_versions,
    write_official_game_manifest,
)
from .arcagi3_trace import ArcTraceRecorder, action_entropy, repeat_collapse
from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .head_collapse import HeadCollapsedExplorer
from .world_model import load_explorer_checkpoint


OFFICIAL_AUDITED_PATHS = [
    "src/arcagi3_official.py",
    "src/arcagi3_official_eval.py",
    "docs/arcagi3_official_report.json",
    "docs/arcagi3_official_games.json",
]


def run_official_episode(
    env: OfficialArcAGI3Env,
    controller: ARCAGI3Adapter | BaselinePolicy,
    *,
    seed: int,
    trace_path: Path | None = None,
    controller_name: str = "explorer",
) -> dict[str, Any]:
    if hasattr(controller, "reset"):
        controller.reset() if isinstance(controller, ARCAGI3Adapter) else controller.reset(seed)
    obs = env.reset(seed=seed)
    actions: list[str] = []
    invalid = 0
    useful_events = 0
    states = {state_signature(obs)}
    trace = (
        ArcTraceRecorder(
            trace_path,
            metadata={
                "controller": controller_name,
                "task_id": obs.task_id,
                "episode_id": obs.episode_id,
                "fixture_source": "official_arcagi3_runtime",
                "game_id": env.spec.game_id,
            },
        )
        if trace_path is not None
        else None
    )
    try:
        for _ in range(env.max_steps):
            before = obs
            if isinstance(controller, ARCAGI3Adapter):
                action, diagnostics = controller.choose_action(obs)
            else:
                action = controller.choose(obs)
                diagnostics = {"memory_recall": {}, "drive": {}, "hypothesis_state": {}, "semantic_action": None}
            if action not in obs.available_actions:
                invalid += 1
                action = obs.available_actions[0]
            result = env.step(action)
            actions.append(action)
            useful_events += sum(
                1
                for item in result.info.get("events", [])
                if item
                in {
                    "key_collected",
                    "door_opened",
                    "goal_reached",
                    "goal_clicked",
                    "resource_collected",
                    "useful_click",
                    "level_completed",
                }
            )
            states.add(state_signature(result.observation))
            if isinstance(controller, ARCAGI3Adapter):
                controller.observe_transition(action, result)
            else:
                controller.observe(before, action, result)
            if trace is not None:
                trace.add_step(before, action, result, diagnostics)
            obs = result.observation
            if result.terminated or result.truncated:
                break
    finally:
        official_scorecard = env.close()
    summary = {
        "solved": bool(env.terminated),
        "score": float(env.score),
        "normalized_score": float(env.normalized_score()),
        "steps": len(actions),
        "invalid_action_rate": float(invalid / max(len(actions), 1)),
        "unique_states": len(states),
        "useful_events": useful_events,
        "action_entropy": action_entropy(actions),
        "repeat_collapse": repeat_collapse(actions),
        "actions": actions[:128],
        "actions_truncated": len(actions) > 128,
        "official_state": env._game_state_name,
        "levels_completed": int(env.levels_completed),
        "win_levels": int(env.win_levels),
        "official_scorecard": scorecard_summary(official_scorecard, env.spec.game_id),
    }
    if trace is not None:
        summary["trace_path"] = str(trace.write(summary))
    return summary


def scorecard_summary(scorecard: dict[str, Any] | None, game_id: str) -> dict[str, Any]:
    if not scorecard:
        return {}
    for environment in scorecard.get("environments", []):
        if str(environment.get("id", "")).startswith(game_id.split("-", 1)[0]):
            runs = environment.get("runs", [])
            best = runs[0] if runs else {}
            return {
                "score": float(environment.get("score", 0.0)),
                "actions": int(environment.get("actions", 0)),
                "levels_completed": int(environment.get("levels_completed", 0)),
                "completed": bool(environment.get("completed", False)),
                "level_count": int(environment.get("level_count", 0)),
                "resets": int(environment.get("resets", 0)),
                "best_run_state": best.get("state"),
                "best_run_level_scores": best.get("level_scores"),
                "best_run_level_actions": best.get("level_actions"),
                "best_run_level_baseline_actions": best.get("level_baseline_actions"),
            }
    return {
        "score": float(scorecard.get("score", 0.0)),
        "total_levels_completed": int(scorecard.get("total_levels_completed", 0)),
        "total_levels": int(scorecard.get("total_levels", 0)),
        "total_actions": int(scorecard.get("total_actions", 0)),
    }


def run_explorer_suite(
    arcade: Any,
    specs: list[OfficialGameSpec],
    *,
    explorer_checkpoint: str | Path,
    trace_dir: str | Path,
    max_steps: int | None,
    max_click_actions: int,
    device: DeviceLike = AUTO_DEVICE,
) -> list[dict[str, Any]]:
    target_device = resolve_device(device)
    model = load_explorer_checkpoint(explorer_checkpoint, device=target_device)
    explorer = HeadCollapsedExplorer(model, disable_structured_heads=True, remove_probe_modules=False)
    adapter = ARCAGI3Adapter(explorer, device=target_device, mode="normal")
    trace_root = Path(trace_dir)
    trace_root.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for idx, spec in enumerate(specs):
        env = OfficialArcAGI3Env(
            arcade,
            spec,
            seed=idx,
            max_steps=max_steps or default_max_steps(spec),
            max_click_actions=max_click_actions,
        )
        trace_path = trace_root / f"{spec.game_id}.official.normal.json"
        row = run_official_episode(
            env,
            adapter,
            seed=idx,
            trace_path=trace_path,
            controller_name="explorer_official_normal",
        )
        row["game_id"] = spec.game_id
        row["title"] = spec.title
        row["tags"] = list(spec.tags)
        row["official_baseline_actions"] = list(spec.baseline_actions)
        rows.append(row)
    return rows


def run_baseline_suite(
    arcade: Any,
    specs: list[OfficialGameSpec],
    *,
    max_steps: int | None,
    max_click_actions: int,
) -> dict[str, dict[str, Any]]:
    results: dict[str, list[dict[str, Any]]] = {baseline.name: [] for baseline in build_baselines()}
    for game_index, spec in enumerate(specs):
        for baseline in build_baselines():
            env = OfficialArcAGI3Env(
                arcade,
                spec,
                seed=100 + game_index,
                max_steps=max_steps or default_max_steps(spec),
                max_click_actions=max_click_actions,
            )
            row = run_official_episode(
                env,
                baseline,
                seed=100 + game_index,
                controller_name=baseline.name,
            )
            row["game_id"] = spec.game_id
            row["title"] = spec.title
            row["tags"] = list(spec.tags)
            results[baseline.name].append(row)
    return {name: aggregate_rows(rows) for name, rows in results.items()}


def prepare_local_games(arcade: Any, specs: list[OfficialGameSpec]) -> dict[str, Any]:
    prepared: list[str] = []
    failed: list[dict[str, str]] = []
    for idx, spec in enumerate(specs):
        scorecard_id: str | None = None
        try:
            scorecard_id = arcade.create_scorecard(tags=["official-runtime-prepare"])
            wrapper = arcade.make(
                spec.game_id,
                seed=idx,
                scorecard_id=scorecard_id,
                save_recording=False,
                include_frame_data=False,
            )
            if wrapper is None:
                failed.append({"game_id": spec.game_id, "error": "arcade.make returned None"})
            else:
                prepared.append(spec.game_id)
        except Exception as exc:
            failed.append({"game_id": spec.game_id, "error": repr(exc)})
        finally:
            if scorecard_id is not None:
                try:
                    arcade.close_scorecard(scorecard_id)
                except Exception:
                    pass
    return {"prepared": prepared, "failed": failed}


def prior_fixture_comparison(path: str | Path = "docs/arcagi3_report.json") -> dict[str, Any]:
    item = Path(path)
    if not item.exists():
        return {"available": False}
    report = json.loads(item.read_text(encoding="utf-8"))
    return {
        "available": True,
        "source": report.get("arc_source"),
        "terminal_outcome": report.get("terminal_outcome"),
        "aggregate": report.get("aggregate"),
        "best_baseline": report.get("best_baseline"),
        "baseline_margin": report.get("baseline_margin"),
    }


def build_report(
    *,
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    json_output: str | Path,
    trace_dir: str | Path,
    environments_dir: str | Path,
    recordings_dir: str | Path,
    operation_mode: str,
    game_ids: list[str] | None,
    limit: int | None,
    max_steps: int | None,
    max_click_actions: int,
    device: DeviceLike = AUTO_DEVICE,
) -> dict[str, Any]:
    arcade = make_arcade(
        operation_mode=operation_mode,
        environments_dir=environments_dir,
        recordings_dir=recordings_dir,
    )
    specs = discover_official_games(arcade, game_ids=game_ids, limit=limit)
    if not specs:
        raise RuntimeError("official ARC-AGI-3 runtime returned no games for the requested selection")
    requested_order = [spec.game_id for spec in specs]
    preparation = {"prepared": [], "failed": [], "used": False}
    run_operation_mode = operation_mode
    if operation_mode == "normal":
        preparation = prepare_local_games(arcade, specs)
        preparation["used"] = True
        arcade = make_arcade(
            operation_mode="offline",
            environments_dir=environments_dir,
            recordings_dir=recordings_dir,
        )
        local_specs = discover_official_games(arcade, game_ids=requested_order, limit=None)
        local_by_id = {spec.game_id: spec for spec in local_specs}
        specs = [local_by_id[game_id] for game_id in requested_order if game_id in local_by_id]
        if not specs:
            raise RuntimeError("official games were not available offline after download preparation")
        run_operation_mode = "offline_after_download"
    write_official_game_manifest("docs/arcagi3_official_games.json", specs)
    per_game = run_explorer_suite(
        arcade,
        specs,
        explorer_checkpoint=explorer_checkpoint,
        trace_dir=trace_dir,
        max_steps=max_steps,
        max_click_actions=max_click_actions,
        device=device,
    )
    baselines = run_baseline_suite(
        arcade,
        specs,
        max_steps=max_steps,
        max_click_actions=max_click_actions,
    )
    aggregate = aggregate_rows(per_game)
    best_baseline_name, best_baseline = max(
        baselines.items(), key=lambda item: float(item[1]["mean_normalized_score"])
    )
    baseline_margin = float(aggregate["mean_normalized_score"] - best_baseline["mean_normalized_score"])
    report: dict[str, Any] = {
        "terminal_outcome": "ARC-AGI-3 OFFICIAL RUNTIME EVALUATED",
        "arc_source": {
            "kind": "official_arcagi3_runtime",
            "package": "arc-agi",
            "operation_mode_requested": operation_mode,
            "operation_mode_run": run_operation_mode,
            "environments_dir": str(environments_dir),
            "recordings_dir": str(recordings_dir),
            "runtime_versions": official_runtime_versions(),
            "preparation": preparation,
        },
        "no_retraining": True,
        "checkpoint": str(checkpoint),
        "explorer_checkpoint": str(explorer_checkpoint),
        "game_count": len(specs),
        "games": [spec.__dict__ for spec in specs],
        "max_steps": max_steps,
        "max_click_actions": max_click_actions,
        "per_game": per_game,
        "aggregate": aggregate,
        "baselines": baselines,
        "best_baseline": {"name": best_baseline_name, **best_baseline},
        "baseline_margin": baseline_margin,
        "prior_committed_fixture_comparison": prior_fixture_comparison(),
        "trace_dir": str(trace_dir),
        "trace_paths": [row.get("trace_path", "") for row in per_game],
        "hashes": collect_hashes(
            OFFICIAL_AUDITED_PATHS
            + [str(checkpoint), str(explorer_checkpoint), "docs/arcagi3_report.json"]
        ),
        "limitations": official_limitations(aggregate, best_baseline, baseline_margin),
    }
    output = Path(json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["hashes"] = collect_hashes(
        OFFICIAL_AUDITED_PATHS + [str(checkpoint), str(explorer_checkpoint), "docs/arcagi3_report.json"]
    )
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def official_limitations(
    aggregate: dict[str, Any],
    best_baseline: dict[str, Any],
    baseline_margin: float,
) -> list[str]:
    limitations: list[str] = []
    if float(aggregate["solve_rate"]) == 0.0:
        limitations.append("The unchanged adapter solved zero official runtime games in this run.")
    if baseline_margin <= 0.0:
        limitations.append("The unchanged adapter did not beat the best same-baseline aggregate on official runtime games.")
    if float(best_baseline["solve_rate"]) == 0.0:
        limitations.append("All same baselines also solved zero official runtime games in this run.")
    limitations.append(
        "The bridge uses only public official frames and legal action IDs; it does not inspect downloaded game source for policy logic."
    )
    return limitations


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--explorer-checkpoint", required=True)
    parser.add_argument("--json-output", default="docs/arcagi3_official_report.json")
    parser.add_argument("--trace-dir", default="docs/arcagi3_official_traces")
    parser.add_argument("--environments-dir", default=DEFAULT_ENVIRONMENTS_DIR)
    parser.add_argument("--recordings-dir", default=DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--operation-mode", default="normal", choices=["normal", "online", "offline", "competition"])
    parser.add_argument("--game-id", action="append", default=None)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--max-steps", type=int, default=None)
    parser.add_argument("--max-click-actions", type=int, default=192)
    parser.add_argument("--device", default=AUTO_DEVICE)
    args = parser.parse_args()
    try:
        report = build_report(
            checkpoint=args.checkpoint,
            explorer_checkpoint=args.explorer_checkpoint,
            json_output=args.json_output,
            trace_dir=args.trace_dir,
            environments_dir=args.environments_dir,
            recordings_dir=args.recordings_dir,
            operation_mode=args.operation_mode,
            game_ids=args.game_id,
            limit=args.limit,
            max_steps=args.max_steps,
            max_click_actions=args.max_click_actions,
            device=args.device,
        )
    except Exception as exc:
        failure = {
            "terminal_outcome": "ARC-AGI-3 OFFICIAL RUNTIME EVALUATION FAILED",
            "error": repr(exc),
            "runtime_versions": official_runtime_versions(),
        }
        Path(args.json_output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.json_output).write_text(json.dumps(failure, indent=2), encoding="utf-8")
        print(json.dumps(failure, indent=2), file=sys.stderr)
        raise
    print(
        json.dumps(
            {
                "terminal_outcome": report["terminal_outcome"],
                "game_count": report["game_count"],
                "aggregate": report["aggregate"],
                "best_baseline": report["best_baseline"],
                "baseline_margin": report["baseline_margin"],
                "limitations": report["limitations"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
