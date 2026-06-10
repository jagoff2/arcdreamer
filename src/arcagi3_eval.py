from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from audit.leakage_scan import run_scan

from .arcagi3_adapter import (
    ARCAGI3Adapter,
    ArcAGI3Fixture,
    ArcAGI3FixtureEnv,
    FIXTURE_PATH,
    load_public_fixtures,
)
from .arcagi3_baselines import BaselinePolicy, build_baselines
from .arcagi3_trace import ArcTraceRecorder, action_entropy, repeat_collapse
from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .head_collapse import HeadCollapsedExplorer
from .world_model import load_explorer_checkpoint


ARC_AUDITED_PATHS = [
    "src/arcagi3_adapter.py",
    "src/arcagi3_baselines.py",
    "src/arcagi3_trace.py",
    "src/arcagi3_eval.py",
    "tests/test_arcagi3_adapter.py",
    "docs/arcagi3_public_fixtures.json",
    "docs/arcagi3_report.json",
    "docs/audit_after_arcagi3.json",
]


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def collect_hashes(paths: list[str]) -> dict[str, dict[str, Any]]:
    hashes: dict[str, dict[str, Any]] = {}
    for item in paths:
        path = Path(item)
        if path.exists():
            hashes[item] = {"sha256": sha256(path), "size_bytes": path.stat().st_size}
        else:
            hashes[item] = {"missing": True}
    return hashes


def run_episode(
    env: ArcAGI3FixtureEnv,
    controller: ARCAGI3Adapter | BaselinePolicy,
    *,
    seed: int,
    max_steps: int | None = None,
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
                "fixture_source": "docs/arcagi3_public_fixtures.json",
            },
        )
        if trace_path is not None
        else None
    )
    limit = max_steps if max_steps is not None else env.fixture.max_steps
    for _ in range(limit):
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
        useful_events += sum(1 for item in result.info.get("events", []) if item in {"key_collected", "door_opened", "goal_reached", "goal_clicked", "resource_collected", "useful_click"})
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
        "actions": actions,
    }
    if trace is not None:
        summary["trace_path"] = str(trace.write(summary))
    return summary


def state_signature(observation: Any) -> str:
    grid = observation.grid.reshape(-1)
    return f"{observation.step_index}:{','.join(str(int(v)) for v in grid)}"


def run_explorer_suite(
    fixtures: list[ArcAGI3Fixture],
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    trace_dir: str | Path,
    device: DeviceLike = AUTO_DEVICE,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    del checkpoint
    target_device = resolve_device(device)
    model = load_explorer_checkpoint(explorer_checkpoint, device=target_device)
    explorer = HeadCollapsedExplorer(model, disable_structured_heads=True, remove_probe_modules=False)
    per_game: list[dict[str, Any]] = []
    ablations: dict[str, dict[str, Any]] = {}
    trace_root = Path(trace_dir)
    trace_root.mkdir(parents=True, exist_ok=True)
    modes = [
        "normal",
        "zero_z",
        "shuffled_z",
        "corrupt_memory",
        "corrupt_drive",
        "no_hypothesis_memory",
        "no_planner_imagination",
        "no_novelty_drive",
    ]
    for mode in modes:
        rows = []
        for idx, fixture in enumerate(fixtures):
            env = ArcAGI3FixtureEnv(fixture, seed=idx)
            adapter = ARCAGI3Adapter(explorer, device=target_device, mode=mode)
            trace_path = trace_root / f"{fixture.fixture_id}.{mode}.json"
            row = run_episode(
                env,
                adapter,
                seed=idx,
                trace_path=trace_path if mode == "normal" else None,
                controller_name=f"explorer_{mode}",
            )
            row["game_id"] = fixture.fixture_id
            row["title"] = fixture.title
            rows.append(row)
        if mode == "normal":
            per_game = rows
        ablations[mode] = aggregate_rows(rows)
    return per_game, ablations


def run_baseline_suite(fixtures: list[ArcAGI3Fixture]) -> dict[str, dict[str, Any]]:
    results: dict[str, list[dict[str, Any]]] = {baseline.name: [] for baseline in build_baselines()}
    for fixture_index, fixture in enumerate(fixtures):
        for baseline in build_baselines():
            env = ArcAGI3FixtureEnv(fixture, seed=100 + fixture_index)
            row = run_episode(env, baseline, seed=100 + fixture_index, controller_name=baseline.name)
            row["game_id"] = fixture.fixture_id
            results[baseline.name].append(row)
    return {name: aggregate_rows(rows) for name, rows in results.items()}


def aggregate_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
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
    return {
        "solve_rate": mean(float(row["solved"]) for row in rows),
        "mean_score": mean(float(row["score"]) for row in rows),
        "mean_normalized_score": mean(float(row["normalized_score"]) for row in rows),
        "mean_steps": mean(float(row["steps"]) for row in rows),
        "mean_invalid_action_rate": mean(float(row["invalid_action_rate"]) for row in rows),
        "mean_unique_states": mean(float(row["unique_states"]) for row in rows),
        "mean_useful_events": mean(float(row["useful_events"]) for row in rows),
        "mean_action_entropy": mean(float(row["action_entropy"]) for row in rows),
        "mean_repeat_collapse": mean(float(row["repeat_collapse"]) for row in rows),
    }


def mean(values: Any) -> float:
    items = list(values)
    return float(sum(items) / len(items)) if items else 0.0


def prior_properties() -> dict[str, Any]:
    def read(path: str) -> dict[str, Any]:
        item = Path(path)
        if not item.exists():
            return {}
        return json.loads(item.read_text(encoding="utf-8"))

    head = read("docs/head_collapse_report.json")
    audit = read("docs/audit_after_head_collapse.json")
    explorer = read("docs/explorer_report_after_head_collapse.json")
    prior = {
        "head_collapse_passes": head.get("terminal_outcome") == "HEAD-COLLAPSE PROVEN",
        "audit_passes": audit.get("terminal_outcome") == "AUDIT PROVEN",
        "explorer_passes": explorer.get("terminal_outcome") == "EXPLORER CORE PROVEN",
        "retention_eval_passes": bool(head.get("prior_properties", {}).get("retention_eval_passes", False)),
        "memory_eval_passes": bool(head.get("prior_properties", {}).get("memory_eval_passes", False)),
        "dialogue_eval_passes": bool(head.get("prior_properties", {}).get("dialogue_eval_passes", False)),
        "living_eval_passes": bool(head.get("prior_properties", {}).get("living_eval_passes", False)),
        "leakage_scan_passes": bool(head.get("prior_properties", {}).get("leakage_scan_passes", False)),
        "hidden_target_canary_zero": bool(head.get("prior_properties", {}).get("hidden_target_canary_zero", False)),
        "hidden_target_canary_diff": float(head.get("prior_properties", {}).get("hidden_target_canary_diff", 1.0)),
        "independent_verify_passes": bool(head.get("prior_properties", {}).get("independent_verify_passes", False)),
        "no_text_as_state_path": bool(head.get("prior_properties", {}).get("no_text_as_state_path", False)),
    }
    prior["passes"] = all(
        bool(prior[key])
        for key in [
            "head_collapse_passes",
            "audit_passes",
            "explorer_passes",
            "retention_eval_passes",
            "memory_eval_passes",
            "dialogue_eval_passes",
            "living_eval_passes",
            "leakage_scan_passes",
            "hidden_target_canary_zero",
            "independent_verify_passes",
            "no_text_as_state_path",
        ]
    ) and prior["hidden_target_canary_diff"] == 0.0
    return prior


def no_hack_audit() -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    source_paths = [Path("src/arcagi3_adapter.py"), Path("src/arcagi3_eval.py"), Path("src/arcagi3_baselines.py")]
    for path in source_paths:
        text = path.read_text(encoding="utf-8")
        tree = ast.parse(text)
        excluded = _audit_excluded_ranges(tree)
        control_text = _text_outside_ranges(text, excluded)
        for node in ast.walk(tree):
            line = getattr(node, "lineno", 0) or 0
            if _line_in_ranges(line, excluded):
                continue
            if isinstance(node, (ast.If, ast.Match)):
                source = ast.get_source_segment(text, node) or ""
                lowered = source.lower()
                if "fixture_id" in lowered and ("==" in lowered or " in " in lowered):
                    findings.append({"path": str(path), "line": getattr(node, "lineno", None), "kind": "fixture_id_branch"})
                if "game_id" in lowered and ("==" in lowered or " in " in lowered):
                    findings.append({"path": str(path), "line": getattr(node, "lineno", None), "kind": "game_id_branch"})
        lowered = control_text.lower()
        for pattern in ["solution", "answer_key", "private_label", "fixed_script", "trajectory_replay"]:
            if pattern in lowered:
                findings.append({"path": str(path), "line": None, "kind": "blocked_text", "match": pattern})
    fixtures = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    blocked_fixture_fields = []
    for game in fixtures.get("games", []):
        for key in game:
            if str(key).lower() in {"solution", "answers", "answer_key", "policy", "script"}:
                blocked_fixture_fields.append({"game": game.get("id"), "field": key})
    if blocked_fixture_fields:
        findings.append({"path": str(FIXTURE_PATH), "kind": "blocked_fixture_fields", "fields": blocked_fixture_fields})
    return {
        "passes": not findings,
        "findings": findings,
        "scanned": [str(path) for path in source_paths] + [str(FIXTURE_PATH)],
        "copy_boundary": {
            "g_arcagi_used_for": [
                "reset_step_legal_actions_contract",
                "grid_observation_shape",
                "raw_and_click_action_naming",
                "trace_and_scorecard_field_names",
                "offline_public_fixture_layout",
            ],
            "g_arcagi_not_used_for": [
                "solver_logic",
                "policies",
                "scripts",
                "weights",
                "environment_source_semantics",
                "public_game_solutions",
            ],
        },
    }


def _audit_excluded_ranges(tree: ast.AST) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name == "no_hack_audit":
            start = int(getattr(node, "lineno", 0) or 0)
            end = int(getattr(node, "end_lineno", start) or start)
            ranges.append((start, end))
    return ranges


def _line_in_ranges(line: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= line <= end for start, end in ranges)


def _text_outside_ranges(text: str, ranges: list[tuple[int, int]]) -> str:
    out: list[str] = []
    for index, line in enumerate(text.splitlines(), start=1):
        if not _line_in_ranges(index, ranges):
            out.append(line)
    return "\n".join(out)


def build_report(
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    config: str,
    json_output: str | Path,
    trace_dir: str | Path,
    device: DeviceLike = AUTO_DEVICE,
) -> dict[str, Any]:
    fixtures = load_public_fixtures()
    per_game, ablations = run_explorer_suite(fixtures, checkpoint, explorer_checkpoint, trace_dir, device=device)
    baselines = run_baseline_suite(fixtures)
    aggregate = aggregate_rows(per_game)
    best_baseline_name, best_baseline = max(
        baselines.items(), key=lambda item: float(item[1]["mean_normalized_score"])
    )
    baseline_margin = float(aggregate["mean_normalized_score"] - best_baseline["mean_normalized_score"])
    ablation_deltas = {
        "zero_z": aggregate["mean_normalized_score"] - ablations["zero_z"]["mean_normalized_score"],
        "shuffled_z": aggregate["mean_normalized_score"] - ablations["shuffled_z"]["mean_normalized_score"],
        "corrupt_memory": aggregate["mean_normalized_score"] - ablations["corrupt_memory"]["mean_normalized_score"],
        "corrupt_drive": aggregate["mean_normalized_score"] - ablations["corrupt_drive"]["mean_normalized_score"],
        "no_hypothesis_memory": aggregate["mean_normalized_score"] - ablations["no_hypothesis_memory"]["mean_normalized_score"],
        "no_planner_imagination": aggregate["mean_normalized_score"] - ablations["no_planner_imagination"]["mean_normalized_score"],
        "no_novelty_drive": aggregate["mean_normalized_score"] - ablations["no_novelty_drive"]["mean_normalized_score"],
    }
    nohack = no_hack_audit()
    prior = prior_properties()
    leakage = run_scan()
    gate_checks = {
        "fixtures_present": len(fixtures) >= 3,
        "traces_present": all(Path(str(row.get("trace_path", ""))).exists() for row in per_game),
        "baseline_margin_ge_0_10": baseline_margin >= 0.10,
        "z_or_memory_degrades_ge_0_20": max(ablation_deltas["zero_z"], ablation_deltas["corrupt_memory"]) >= 0.20,
        "drive_or_novelty_degrades_ge_0_10": max(ablation_deltas["corrupt_drive"], ablation_deltas["no_novelty_drive"]) >= 0.10,
        "planner_degrades_ge_0_10": ablation_deltas["no_planner_imagination"] >= 0.10,
        "no_hack_audit_passes": nohack["passes"],
        "leakage_scan_passes": leakage["passes"],
        "prior_properties_pass": prior["passes"],
    }
    limitations: list[str] = []
    if not gate_checks["baseline_margin_ge_0_10"]:
        limitations.append("Explorer normalized score does not beat the best baseline by >=0.10.")
    if not gate_checks["z_or_memory_degrades_ge_0_20"]:
        limitations.append("z or memory ablation did not degrade normalized score by >=0.20.")
    if not gate_checks["drive_or_novelty_degrades_ge_0_10"]:
        limitations.append("drive or novelty ablation did not degrade normalized score by >=0.10.")
    if not gate_checks["planner_degrades_ge_0_10"]:
        limitations.append("planner/imagination ablation did not degrade normalized score by >=0.10.")
    if not nohack["passes"]:
        limitations.append("No-hack audit found blocked patterns.")
    if not prior["passes"]:
        limitations.append("Prior proof properties did not all pass from committed reports.")
    terminal = "ARC-AGI-3 ADAPTER PROVEN" if all(gate_checks.values()) else "NOT PROVEN"
    report: dict[str, Any] = {
        "terminal_outcome": terminal,
        "arc_source": {
            "kind": "committed_public_fixtures",
            "path": str(FIXTURE_PATH),
            "reason": "arc_agi and arcengine are unavailable in the active Python 3.10 runtime",
            "toolkit_available": False,
        },
        "config": config,
        "checkpoint": str(checkpoint),
        "explorer_checkpoint": str(explorer_checkpoint),
        "g_arcagi_findings": nohack["copy_boundary"],
        "per_game": per_game,
        "aggregate": aggregate,
        "baselines": baselines,
        "best_baseline": {"name": best_baseline_name, **best_baseline},
        "baseline_margin": baseline_margin,
        "ablations": ablations,
        "ablation_deltas": ablation_deltas,
        "trace_dir": str(trace_dir),
        "trace_paths": [row.get("trace_path", "") for row in per_game],
        "no_hack_audit": nohack,
        "prior_properties": prior,
        "leakage_scan": {"passes": leakage["passes"], "findings": leakage["findings"]},
        "gate_checks": gate_checks,
        "hashes": collect_hashes(ARC_AUDITED_PATHS + [str(checkpoint), str(explorer_checkpoint), "docs/head_collapse_report.json", "docs/audit_after_head_collapse.json"]),
        "limitations": limitations,
    }
    output = Path(json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    report["hashes"] = collect_hashes(ARC_AUDITED_PATHS + [str(checkpoint), str(explorer_checkpoint), "docs/head_collapse_report.json", "docs/audit_after_head_collapse.json"])
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--explorer-checkpoint", required=True)
    parser.add_argument("--config", default="public", choices=["public"])
    parser.add_argument("--json-output", default="docs/arcagi3_report.json")
    parser.add_argument("--trace-dir", default="docs/arcagi3_traces")
    parser.add_argument("--device", default=AUTO_DEVICE)
    args = parser.parse_args()
    report = build_report(
        checkpoint=args.checkpoint,
        explorer_checkpoint=args.explorer_checkpoint,
        config=args.config,
        json_output=args.json_output,
        trace_dir=args.trace_dir,
        device=args.device,
    )
    print(json.dumps({"terminal_outcome": report["terminal_outcome"], "baseline_margin": report["baseline_margin"], "limitations": report["limitations"]}, indent=2))


if __name__ == "__main__":
    main()
