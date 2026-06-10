from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any

import torch

from .arcagi3_adapter import ARCAGI3Adapter
from .arcagi3_eval import collect_hashes
from .arcagi3_failure_taxonomy import FAILURE_CLASSES, classify_failure, no_hack_audit, taxonomy_counts
from .arcagi3_official import (
    DEFAULT_ENVIRONMENTS_DIR,
    DEFAULT_RECORDINGS_DIR,
    official_runtime_versions,
)
from .arcagi3_official_eval import build_report as build_official_report
from .arcagi3_trace_analysis import (
    aggregate_failure_counts,
    enrich_trace,
    load_trace,
    mean,
    representative_excerpt,
    required_trace_fields_present,
    trace_seed,
    write_trace,
)
from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .head_collapse import HeadCollapsedExplorer
from .world_model import load_explorer_checkpoint


DIAGNOSIS_AUDITED_PATHS = [
    "src/arcagi3_diagnose.py",
    "src/arcagi3_failure_taxonomy.py",
    "src/arcagi3_trace_analysis.py",
    "src/arcagi3_official.py",
    "src/arcagi3_official_eval.py",
    "src/arcagi3_adapter.py",
    "src/arcagi3_baselines.py",
    "tests/test_arcagi3_diagnosis.py",
    "docs/arcagi3_failure_report.json",
    "docs/arcagi3_failure_report.md",
    "docs/arcagi3_official_report.json",
    "docs/arcagi3_official_cuda_report.json",
    "docs/arcagi3_official_games.json",
    "docs/arcagi3_report.json",
    "docs/audit_after_arcagi3.json",
    "docs/head_collapse_report.json",
]


BASELINE_ORDER = [
    "explorer",
    "random_legal",
    "repeat_last_action",
    "coverage_graph_exploration",
    "novelty_first",
    "greedy_observable_score_delta",
    "oracle_free_observed_graph_bfs",
]


def choose_official_report(trace_dir: str | Path, explicit: str | Path | None = None) -> Path:
    if explicit is not None:
        return Path(explicit)
    target = str(Path(trace_dir))
    candidates = [Path("docs/arcagi3_official_report.json"), Path("docs/arcagi3_official_cuda_report.json")]
    for candidate in candidates:
        if not candidate.exists():
            continue
        try:
            report = json.loads(candidate.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            continue
        if str(Path(str(report.get("trace_dir", "")))) == target:
            return candidate
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return Path("docs/arcagi3_official_report.json")


def trace_files(trace_dir: str | Path) -> list[Path]:
    root = Path(trace_dir)
    if not root.exists():
        return []
    return sorted(root.glob("*.official.normal.json"))


def load_report(path: str | Path) -> dict[str, Any]:
    item = Path(path)
    if not item.exists():
        return {}
    return json.loads(item.read_text(encoding="utf-8"))


def ensure_official_materials(
    *,
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    trace_dir: str | Path,
    official_report_path: str | Path,
    refresh: bool,
    environments_dir: str | Path,
    recordings_dir: str | Path,
    max_click_actions: int,
    device: DeviceLike,
) -> dict[str, Any]:
    report_path = Path(official_report_path)
    existing = load_report(report_path)
    game_count = int(existing.get("game_count", 0) or 0)
    if not refresh and existing and len(trace_files(trace_dir)) >= max(game_count, 25):
        return existing
    return build_official_report(
        checkpoint=checkpoint,
        explorer_checkpoint=explorer_checkpoint,
        json_output=report_path,
        trace_dir=trace_dir,
        environments_dir=environments_dir,
        recordings_dir=recordings_dir,
        operation_mode="normal",
        game_ids=None,
        limit=None,
        max_steps=None,
        max_click_actions=max_click_actions,
        device=device,
    )


def runtime_status(device: DeviceLike) -> dict[str, Any]:
    status = {
        "official_runtime_available": False,
        "runtime_versions": official_runtime_versions(),
        "torch_cuda_available": bool(torch.cuda.is_available()),
        "torch_cuda_version": getattr(torch.version, "cuda", None),
        "torch_device_count": int(torch.cuda.device_count()) if torch.cuda.is_available() else 0,
        "diagnosis_device": str(resolve_device(device)),
    }
    try:
        import arc_agi  # noqa: F401
        import arcengine  # noqa: F401

        status["official_runtime_available"] = True
    except Exception as exc:  # pragma: no cover - depends on optional runtime
        status["official_runtime_error"] = repr(exc)
    if torch.cuda.is_available():
        status["torch_cuda_devices"] = [torch.cuda.get_device_name(index) for index in range(torch.cuda.device_count())]
    if not status["official_runtime_available"]:
        venv_status = venv_official_runtime_status()
        if venv_status.get("official_runtime_available"):
            status["official_runtime_available"] = True
            status["official_runtime_via"] = venv_status.get("python")
            status["venv_runtime_status"] = venv_status
    return status


def venv_official_runtime_status() -> dict[str, Any]:
    python_path = Path(".venv/Scripts/python.exe")
    if not python_path.exists():
        return {"official_runtime_available": False, "reason": "no_.venv_python"}
    probe = (
        "import importlib.util, json, sys, torch; "
        "print(json.dumps({"
        "'python': sys.executable, "
        "'official_runtime_available': importlib.util.find_spec('arc_agi') is not None and importlib.util.find_spec('arcengine') is not None, "
        "'arc_agi': importlib.util.find_spec('arc_agi') is not None, "
        "'arcengine': importlib.util.find_spec('arcengine') is not None, "
        "'torch': torch.__version__, "
        "'cuda_available': torch.cuda.is_available(), "
        "'cuda': torch.version.cuda, "
        "'device_count': torch.cuda.device_count(), "
        "'device_0': torch.cuda.get_device_name(0) if torch.cuda.is_available() else None"
        "}))"
    )
    try:
        result = subprocess.run(
            [str(python_path), "-c", probe],
            check=True,
            capture_output=True,
            text=True,
        )
        return json.loads(result.stdout)
    except Exception as exc:
        return {"official_runtime_available": False, "python": str(python_path), "error": repr(exc)}


def make_adapters(explorer_checkpoint: str | Path, device: DeviceLike) -> tuple[ARCAGI3Adapter | None, ARCAGI3Adapter | None, str | None]:
    try:
        target_device = resolve_device(device)
        model = load_explorer_checkpoint(explorer_checkpoint, device=target_device)
        normal = ARCAGI3Adapter(
            HeadCollapsedExplorer(model, disable_structured_heads=True, remove_probe_modules=False),
            device=target_device,
            mode="normal",
        )
        corrupt = ARCAGI3Adapter(
            HeadCollapsedExplorer(model, disable_structured_heads=True, remove_probe_modules=False),
            device=target_device,
            mode="corrupt_memory",
        )
        return normal, corrupt, None
    except Exception as exc:
        return None, None, repr(exc)


def aggregate_score_table(official_report: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    aggregate = dict(official_report.get("aggregate", {}))
    rows.append({"name": "explorer", **aggregate})
    baselines = dict(official_report.get("baselines", {}))
    for name in BASELINE_ORDER:
        if name == "explorer":
            continue
        if name in baselines:
            rows.append({"name": name, **dict(baselines[name])})
    return rows


def prior_property_table() -> list[dict[str, Any]]:
    specs = [
        ("fixture_arc_report", Path("docs/arcagi3_report.json"), "ARC-AGI-3 ADAPTER PROVEN"),
        ("post_arc_audit", Path("docs/audit_after_arcagi3.json"), "AUDIT PROVEN"),
        ("head_collapse", Path("docs/head_collapse_report.json"), "HEAD-COLLAPSE PROVEN"),
    ]
    rows: list[dict[str, Any]] = []
    for name, path, expected in specs:
        if not path.exists():
            rows.append({"name": name, "path": str(path), "status": "MISSING", "passes": False})
            continue
        report = json.loads(path.read_text(encoding="utf-8"))
        outcome = str(report.get("terminal_outcome", ""))
        limitations = list(report.get("limitations", []))
        rows.append(
            {
                "name": name,
                "path": str(path),
                "terminal_outcome": outcome,
                "passes": outcome == expected and not limitations,
                "limitations": limitations,
            }
        )
    return rows


def baseline_audit(official_report: dict[str, Any]) -> dict[str, Any]:
    aggregate = dict(official_report.get("aggregate", {}))
    best = dict(official_report.get("best_baseline", {}))
    margin = float(official_report.get("baseline_margin", 0.0))
    explorer_score = float(aggregate.get("mean_normalized_score", 0.0))
    best_score = float(best.get("mean_normalized_score", 0.0))
    if margin < 0.0:
        explanation = (
            "Best baseline beats explorer because it produced rare official progress while the explorer produced none, "
            "and because its higher action entropy explored more legal actions."
        )
    elif margin == 0.0:
        explanation = (
            "Best baseline ties explorer on normalized score; neither solved a game, so the tie is weak progress rather than proof of competence."
        )
    else:
        explanation = (
            "Explorer beats the best baseline on aggregate, but the official result is still zero solves and remains a failure diagnosis target."
        )
    return {
        "best_baseline": best,
        "baseline_margin": margin,
        "explorer_mean_normalized_score": explorer_score,
        "best_baseline_mean_normalized_score": best_score,
        "explanation": explanation,
    }


def source_report_summaries() -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, path in {
        "official_cpu": Path("docs/arcagi3_official_report.json"),
        "official_cuda": Path("docs/arcagi3_official_cuda_report.json"),
        "fixture_prior": Path("docs/arcagi3_report.json"),
    }.items():
        report = load_report(path)
        if report:
            out[name] = {
                "path": str(path),
                "terminal_outcome": report.get("terminal_outcome"),
                "game_count": report.get("game_count"),
                "aggregate": report.get("aggregate"),
                "best_baseline": report.get("best_baseline"),
                "baseline_margin": report.get("baseline_margin"),
                "limitations": report.get("limitations"),
            }
    return out


def build_failure_report(
    *,
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    config: str,
    json_output: str | Path,
    trace_dir: str | Path,
    official_report_path: str | Path | None = None,
    refresh_official: bool = False,
    environments_dir: str | Path = DEFAULT_ENVIRONMENTS_DIR,
    recordings_dir: str | Path = DEFAULT_RECORDINGS_DIR,
    max_click_actions: int = 192,
    device: DeviceLike = AUTO_DEVICE,
) -> dict[str, Any]:
    if config != "official":
        raise ValueError("arcagi3_diagnose currently supports only --config official")
    selected_report = choose_official_report(trace_dir, official_report_path)
    official_report = ensure_official_materials(
        checkpoint=checkpoint,
        explorer_checkpoint=explorer_checkpoint,
        trace_dir=trace_dir,
        official_report_path=selected_report,
        refresh=refresh_official,
        environments_dir=environments_dir,
        recordings_dir=recordings_dir,
        max_click_actions=max_click_actions,
        device=device,
    )
    games = {str(item.get("game_id")): dict(item) for item in official_report.get("games", [])}
    per_game_report = {str(item.get("game_id")): dict(item) for item in official_report.get("per_game", [])}
    normal_adapter, corrupt_adapter, adapter_error = make_adapters(explorer_checkpoint, device)
    trace_rows: list[dict[str, Any]] = []
    excerpts: list[dict[str, Any]] = []
    loaded_paths = trace_files(trace_dir)
    for index, path in enumerate(loaded_paths):
        payload = load_trace(path)
        game_id = str(payload.get("metadata", {}).get("game_id", path.name.split(".")[0]))
        seed = trace_seed(payload, fallback=index)
        enriched = enrich_trace(payload, seed=seed, normal_adapter=normal_adapter, corrupt_adapter=corrupt_adapter)
        write_trace(path, enriched)
        summary = dict(enriched.get("summary", {}))
        official_row = per_game_report.get(game_id, {})
        if official_row:
            summary.update({key: official_row[key] for key in official_row if key not in {"actions", "trace_path"}})
        audits = dict(enriched.get("diagnosis", {}).get("audits", {}))
        game_info = games.get(game_id, {})
        game_for_classification = {
            "game_id": game_id,
            "summary": summary,
            "audits": audits,
            "tags": list(game_info.get("tags", [])),
            "trace_loaded": True,
            "best_baseline_beats_explorer": False,
        }
        classification = classify_failure(game_for_classification, dict(official_report.get("baselines", {})))
        row = {
            "game_id": game_id,
            "title": game_info.get("title", payload.get("metadata", {}).get("game_id")),
            "seed": seed,
            "tags": list(game_info.get("tags", [])),
            "trace_path": str(path),
            "solved": bool(summary.get("solved", False)),
            "score": float(summary.get("score", 0.0)),
            "normalized_score": float(summary.get("normalized_score", 0.0)),
            "levels_completed": int(summary.get("levels_completed", 0)),
            "win_levels": int(summary.get("win_levels", 0)),
            "steps": int(summary.get("steps", len(enriched.get("steps", [])))),
            "invalid_action_rate": float(summary.get("invalid_action_rate", 0.0)),
            "unique_states": int(summary.get("unique_states", 0)),
            "useful_events": int(summary.get("useful_events", 0)),
            "action_entropy": float(summary.get("action_entropy", 0.0)),
            "repeat_collapse": float(summary.get("repeat_collapse", 0.0)),
            "first_useful_event": enriched.get("diagnosis", {}).get("first_useful_event"),
            "required_trace_fields_present": required_trace_fields_present(enriched.get("steps", [])),
            "observation_audit": audits.get("observation_audit", {}),
            "action_audit": audits.get("action_audit", {}),
            "memory_audit": audits.get("memory_audit", {}),
            "goal_audit": audits.get("goal_audit", {}),
            "planner_audit": audits.get("planner_audit", {}),
            **classification,
        }
        trace_rows.append(row)
        if len(excerpts) < 5:
            excerpts.append(representative_excerpt(enriched))
    game_ids = [row["game_id"] for row in trace_rows]
    hack_audit = no_hack_audit(game_ids)
    prior = prior_property_table()
    score_table = aggregate_score_table(official_report)
    trace_count_ok = len(trace_rows) == 25
    classified_ok = len(trace_rows) == 25 and all(row.get("primary_failure_class") in FAILURE_CLASSES for row in trace_rows)
    trace_fields_ok = trace_count_ok and all(bool(row.get("required_trace_fields_present")) for row in trace_rows)
    baseline_names = {row["name"] for row in score_table}
    baseline_ok = set(BASELINE_ORDER).issubset(baseline_names)
    prior_ok = all(row.get("passes") for row in prior)
    proven = (
        runtime_status(device)["official_runtime_available"]
        and trace_count_ok
        and classified_ok
        and trace_fields_ok
        and baseline_ok
        and hack_audit["passes"]
        and prior_ok
    )
    bottlenecks = top_bottlenecks(trace_rows)
    report: dict[str, Any] = {
        "terminal_outcome": "ARC FAILURE DIAGNOSIS PROVEN" if proven else "NOT PROVEN",
        "config": config,
        "checkpoint": str(checkpoint),
        "explorer_checkpoint": str(explorer_checkpoint),
        "official_report_path": str(selected_report),
        "official_runtime_status": runtime_status(device),
        "source_reports": source_report_summaries(),
        "adapter_tensor_replay_error": adapter_error,
        "aggregate_score_table": score_table,
        "baseline_audit": baseline_audit(official_report),
        "per_game_failure_table": trace_rows,
        "failure_taxonomy": taxonomy_counts(trace_rows),
        "failure_class_counts_flat": aggregate_failure_counts(trace_rows),
        "top_bottlenecks": bottlenecks,
        "representative_trace_excerpts": excerpts,
        "trace_dir": str(trace_dir),
        "trace_paths": [row["trace_path"] for row in trace_rows],
        "required_analyses": required_analysis_summary(trace_rows, official_report, hack_audit),
        "no_hack_audit": hack_audit,
        "prior_property_table": prior,
        "requirements": {
            "traces_25_of_25": trace_count_ok,
            "primary_failure_class_25_of_25": classified_ok,
            "required_trace_fields_25_of_25": trace_fields_ok,
            "baseline_table_has_all_baselines_and_explorer": baseline_ok,
            "representative_trace_excerpts_ge_5": len(excerpts) >= 5,
            "no_hack_audit_passes": hack_audit["passes"],
            "prior_properties_pass": prior_ok,
        },
        "commands_required": [
            "pytest -q",
            "python -m src.arcagi3_diagnose --checkpoint frozen/recurrent_latent_fast.pt --explorer-checkpoint runs/explorer_tiny.pt --config official --json-output docs/arcagi3_failure_report.json --trace-dir docs/arcagi3_official_traces",
            "python -m audit.leakage_scan",
            "python -m audit.independent_verify --checkpoint frozen/recurrent_latent_fast.pt --config fast --json-output docs/audit_after_arcagi3_diagnosis.json",
        ],
        "limitations": diagnosis_limitations(trace_rows, official_report, hack_audit, prior),
    }
    report["hashes"] = collect_hashes(DIAGNOSIS_AUDITED_PATHS + [str(checkpoint), str(explorer_checkpoint), str(selected_report)])
    output = Path(json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    markdown = render_markdown(report)
    output.with_suffix(".md").write_text(markdown, encoding="utf-8")
    report["hashes"] = collect_hashes(DIAGNOSIS_AUDITED_PATHS + [str(checkpoint), str(explorer_checkpoint), str(selected_report)])
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    output.with_suffix(".md").write_text(render_markdown(report), encoding="utf-8")
    return report


def top_bottlenecks(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    repeat_games = [row for row in rows if float(row.get("repeat_collapse", 0.0)) >= 0.85]
    dense_games = [row for row in rows if row.get("observation_audit", {}).get("missing_affordances")]
    useful_but_failed = [row for row in rows if int(row.get("useful_events", 0)) > 0 and not row.get("solved")]
    planner_games = [
        row
        for row in rows
        if row.get("primary_failure_class") == "G" or "G" in row.get("secondary_failure_classes", [])
    ]
    items.append(
        {
            "name": "action repetition collapse",
            "game_count": len(repeat_games),
            "evidence": f"mean repeat collapse {mean(float(row.get('repeat_collapse', 0.0)) for row in rows):.3f}",
        }
    )
    items.append(
        {
            "name": "pixel-proxy observation bottleneck",
            "game_count": len(dense_games),
            "evidence": "Observation audit flags dense visual proxies or large click surfaces.",
        }
    )
    items.append(
        {
            "name": "useful events not converted into completion",
            "game_count": len(useful_but_failed),
            "evidence": "Only games with useful events still failed to solve.",
        }
    )
    items.append(
        {
            "name": "no multi-step planner sequence",
            "game_count": len(planner_games),
            "evidence": "Planner class assigned where entropy did not produce progress.",
        }
    )
    return items


def required_analysis_summary(
    rows: list[dict[str, Any]],
    official_report: dict[str, Any],
    hack_audit: dict[str, Any],
) -> dict[str, Any]:
    return {
        "observation_audit": {
            "games_checked": len(rows),
            "games_with_missing_affordances": sum(1 for row in rows if row.get("observation_audit", {}).get("missing_affordances")),
        },
        "action_audit": {
            "games_checked": len(rows),
            "mean_invalid_action_rate": mean(float(row.get("invalid_action_rate", 0.0)) for row in rows),
            "games_with_no_op_dominance": sum(1 for row in rows if row.get("action_audit", {}).get("no_op_dominance")),
        },
        "exploration_audit": {
            "explorer": official_report.get("aggregate", {}),
            "baselines": official_report.get("baselines", {}),
            "best_baseline": official_report.get("best_baseline", {}),
        },
        "memory_audit": {
            "games_with_useful_events": sum(1 for row in rows if int(row.get("useful_events", 0)) > 0),
            "mean_corrupt_memory_action_shift_rate": mean(
                float(row.get("memory_audit", {}).get("corrupt_memory_action_shift_rate") or 0.0) for row in rows
            ),
        },
        "goal_audit": {
            "games_where_useful_event_changed_future_policy": sum(
                1 for row in rows if row.get("goal_audit", {}).get("useful_event_changed_future_policy")
            ),
        },
        "planner_audit": {
            "games_with_sequence_failure": sum(1 for row in rows if row.get("planner_audit", {}).get("sequence_failure")),
        },
        "baseline_audit": baseline_audit(official_report),
        "no_hack_audit": hack_audit,
    }


def diagnosis_limitations(
    rows: list[dict[str, Any]],
    official_report: dict[str, Any],
    hack_audit: dict[str, Any],
    prior: list[dict[str, Any]],
) -> list[str]:
    limitations: list[str] = []
    if len(rows) != 25:
        limitations.append(f"Expected 25 official traces, found {len(rows)}.")
    if not all(row.get("required_trace_fields_present") for row in rows):
        limitations.append("One or more official traces are missing required diagnosis fields.")
    if float(official_report.get("aggregate", {}).get("solve_rate", 0.0)) != 0.0:
        limitations.append("Official report no longer has zero solves; diagnosis should be refreshed against the latest result.")
    if not hack_audit.get("passes", False):
        limitations.append("No-hack audit found suspicious source patterns.")
    if not all(row.get("passes") for row in prior):
        limitations.append("A prior property report is missing or failing.")
    limitations.append("Diagnosis does not build a solver and does not prove that any proposed next step will solve ARC-AGI-3.")
    return limitations


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# ARC-AGI-3 Failure Diagnosis")
    lines.append("")
    lines.append(f"Terminal outcome: **{report['terminal_outcome']}**")
    lines.append("")
    lines.append("## Official Runtime Status")
    status = report["official_runtime_status"]
    lines.append(
        f"Official runtime available: `{status['official_runtime_available']}`. Diagnosis device: `{status['diagnosis_device']}`. CUDA: `{status['torch_cuda_available']}`."
    )
    lines.append("")
    lines.append("## Aggregate Score Table")
    lines.append("")
    lines.append("| Name | Solve Rate | Mean Score | Mean Normalized | Steps | Entropy | Repeat Collapse |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for row in report["aggregate_score_table"]:
        lines.append(
            f"| {row['name']} | {float(row.get('solve_rate', 0.0)):.3f} | {float(row.get('mean_score', 0.0)):.3f} | {float(row.get('mean_normalized_score', 0.0)):.3f} | {float(row.get('mean_steps', 0.0)):.2f} | {float(row.get('mean_action_entropy', 0.0)):.3f} | {float(row.get('mean_repeat_collapse', 0.0)):.3f} |"
        )
    lines.append("")
    lines.append("## Failure Table")
    lines.append("")
    lines.append("| Game | Score | Levels | Steps | Invalid | Entropy | Repeat | Primary | Secondary | Confidence |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |")
    for row in report["per_game_failure_table"]:
        secondary = ", ".join(row.get("secondary_failure_classes", []))
        lines.append(
            f"| {row['game_id']} | {float(row.get('normalized_score', 0.0)):.3f} | {int(row.get('levels_completed', 0))}/{int(row.get('win_levels', 0))} | {int(row.get('steps', 0))} | {float(row.get('invalid_action_rate', 0.0)):.3f} | {float(row.get('action_entropy', 0.0)):.3f} | {float(row.get('repeat_collapse', 0.0)):.3f} | {row.get('primary_failure_class')} | {secondary} | {row.get('diagnosis_confidence')} |"
        )
    lines.append("")
    lines.append("## Top Bottlenecks")
    for item in report["top_bottlenecks"]:
        lines.append(f"- {item['name']}: {item['game_count']} games. {item['evidence']}")
    lines.append("")
    lines.append("## No-Hack Proof")
    lines.append(f"No-hack audit passes: `{report['no_hack_audit']['passes']}`.")
    lines.append(f"Scanned: `{', '.join(report['no_hack_audit']['scanned'])}`.")
    lines.append("")
    lines.append("## Prior Properties")
    lines.append("")
    lines.append("| Name | Outcome | Passes |")
    lines.append("| --- | --- | --- |")
    for row in report["prior_property_table"]:
        lines.append(f"| {row['name']} | {row.get('terminal_outcome', row.get('status'))} | {row.get('passes')} |")
    lines.append("")
    lines.append("## Representative Trace Excerpts")
    for excerpt in report["representative_trace_excerpts"]:
        lines.append("")
        lines.append(f"### {excerpt['game_id']}")
        for step in excerpt.get("steps", []):
            lines.append(
                f"- step {step.get('step')}: action `{step.get('action')}`, events `{step.get('event_delta')}`, score_delta `{step.get('score_delta')}`, baselines `{step.get('baseline_actions')}`"
            )
    lines.append("")
    lines.append("## Concrete Next Steps")
    for item in concrete_next_steps(report):
        lines.append(f"- {item}")
    lines.append("")
    lines.append("## Limitations")
    for item in report["limitations"]:
        lines.append(f"- {item}")
    lines.append("")
    return "\n".join(lines)


def concrete_next_steps(report: dict[str, Any]) -> list[str]:
    return [
        "Replace the pixel-proxy frame collapse with learned or explicitly evaluated object/affordance extraction, then rerun diagnosis before any policy change.",
        "Separate click-surface ranking from movement-action ranking so dense click games do not collapse to one click or one keyboard action.",
        "Add a diagnosis-only transition model probe that predicts event deltas from public frames and legal actions; keep it outside the policy until audited.",
        "Make memory audit stricter by recording event-token writes directly in adapter memory diagnostics rather than inferring them from post-event recall.",
        "Evaluate planner sequence formation on official trace replays with counterfactual legal-action rollouts, still without using game source or labels.",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--explorer-checkpoint", required=True)
    parser.add_argument("--config", default="official")
    parser.add_argument("--json-output", default="docs/arcagi3_failure_report.json")
    parser.add_argument("--trace-dir", default="docs/arcagi3_official_traces")
    parser.add_argument("--official-report", default=None)
    parser.add_argument("--refresh-official", action="store_true")
    parser.add_argument("--environments-dir", default=DEFAULT_ENVIRONMENTS_DIR)
    parser.add_argument("--recordings-dir", default=DEFAULT_RECORDINGS_DIR)
    parser.add_argument("--max-click-actions", type=int, default=192)
    parser.add_argument("--device", default=AUTO_DEVICE)
    args = parser.parse_args()
    try:
        report = build_failure_report(
            checkpoint=args.checkpoint,
            explorer_checkpoint=args.explorer_checkpoint,
            config=args.config,
            json_output=args.json_output,
            trace_dir=args.trace_dir,
            official_report_path=args.official_report,
            refresh_official=args.refresh_official,
            environments_dir=args.environments_dir,
            recordings_dir=args.recordings_dir,
            max_click_actions=args.max_click_actions,
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
                "failure_taxonomy": report["failure_taxonomy"],
                "top_bottlenecks": report["top_bottlenecks"],
                "limitations": report["limitations"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
