from __future__ import annotations

import ast
from collections import Counter
from pathlib import Path
from typing import Any


FAILURE_CLASSES = {
    "A": "observation parse failure",
    "B": "action mapping failure",
    "C": "exploration stuck/cycle",
    "D": "useful event observed but not stored",
    "E": "useful event stored but not exploited",
    "F": "wrong/absent goal inference",
    "G": "planner cannot form sequence",
    "H": "unsupported mechanic",
    "I": "scoring/terminal mismatch",
    "J": "adapter/runtime bug",
    "K": "baseline-only artifact",
}


def class_label(code: str) -> str:
    return f"{code} {FAILURE_CLASSES[code]}"


def classify_failure(game: dict[str, Any], baselines: dict[str, Any] | None = None) -> dict[str, Any]:
    summary = dict(game.get("summary", {}))
    audits = dict(game.get("audits", {}))
    observation = dict(audits.get("observation_audit", {}))
    action = dict(audits.get("action_audit", {}))
    memory = dict(audits.get("memory_audit", {}))
    goal = dict(audits.get("goal_audit", {}))
    planner = dict(audits.get("planner_audit", {}))
    scorecard = dict(summary.get("official_scorecard", {}))
    solved = bool(summary.get("solved", False))
    useful_events = int(summary.get("useful_events", 0))
    repeat = float(summary.get("repeat_collapse", 0.0))
    entropy = float(summary.get("action_entropy", 0.0))
    normalized = float(summary.get("normalized_score", 0.0))
    primary = "G"
    secondary: list[str] = []
    evidence: list[str] = []

    if not game.get("trace_loaded", True):
        primary = "J"
        evidence.append("Trace file could not be loaded.")
    elif not observation.get("features_match_visible_state", False):
        primary = "A"
        evidence.append("Observation feature extraction did not cover the visible trace state.")
    elif int(action.get("invalid_mapping_count", 0)) > 0 or float(summary.get("invalid_action_rate", 0.0)) > 0.0:
        primary = "B"
        evidence.append("Legal-action mapping produced invalid entries or executed invalid actions.")
    elif useful_events > 0 and not solved:
        stored = bool(memory.get("useful_event_stored_in_memory_window", False))
        if stored:
            primary = "E"
            evidence.append("A useful event was observed and memory changed, but subsequent policy did not finish the game.")
        else:
            primary = "D"
            evidence.append("A useful event was observed, but the memory window did not show a durable stored cue.")
    elif repeat >= 0.85 or (bool(action.get("no_op_dominance", False)) and entropy < 0.75):
        primary = "C"
        evidence.append(f"Repeated-action collapse was {repeat:.3f} with no solve.")
    elif observation.get("missing_affordances"):
        primary = "F"
        evidence.append("Visible frame features were converted to generic proxies without reliable goal affordances.")
    else:
        primary = "G"
        evidence.append(f"Entropy {entropy:.3f} did not produce a successful multi-step sequence.")

    if repeat >= 0.85 or bool(action.get("no_op_dominance", False)):
        secondary.append("C")
    if observation.get("missing_affordances"):
        secondary.append("F")
    if action.get("action_surface", {}).get("click_count", 0) > 64 or "click" in game.get("tags", []):
        secondary.append("H")
    if bool(planner.get("sequence_failure", False)) or (entropy < 0.75 and not solved):
        secondary.append("G")
    if useful_events > 0 and not solved:
        secondary.append("E" if memory.get("useful_event_stored_in_memory_window") else "D")
    if scorecard and (
        int(scorecard.get("levels_completed", summary.get("levels_completed", 0))) != int(summary.get("levels_completed", 0))
        or bool(scorecard.get("completed", False)) != solved
    ):
        secondary.append("I")
        evidence.append("Trace summary and official scorecard disagree on completion or levels.")
    if baselines and game.get("best_baseline_beats_explorer", False):
        secondary.append("K")
        evidence.append("A baseline scored higher than the explorer on this game.")

    secondary = [code for code in dict.fromkeys(secondary) if code != primary]
    confidence = diagnosis_confidence(primary, summary, audits)
    return {
        "primary_failure_class": primary,
        "primary_failure_label": class_label(primary),
        "secondary_failure_classes": secondary,
        "secondary_failure_labels": [class_label(code) for code in secondary],
        "diagnosis_confidence": confidence["level"],
        "confidence_evidence": confidence["evidence"],
        "evidence": evidence,
    }


def diagnosis_confidence(primary: str, summary: dict[str, Any], audits: dict[str, Any]) -> dict[str, Any]:
    evidence: list[str] = []
    level = "medium"
    if primary == "C" and float(summary.get("repeat_collapse", 0.0)) >= 0.90:
        level = "high"
        evidence.append("Repeated-action collapse exceeds 0.90.")
    if primary in {"D", "E"} and int(summary.get("useful_events", 0)) > 0:
        level = "high"
        evidence.append("Useful event and post-event behavior are directly visible in trace.")
    if primary in {"F", "H"} and audits.get("observation_audit", {}).get("missing_affordances"):
        evidence.append("Observation audit flags missing affordances.")
    if primary == "G" and audits.get("planner_audit", {}).get("cycle_stats", {}).get("unique_observation_states", 0) > 0:
        evidence.append("Planner audit has transition and action evidence.")
    if not evidence:
        evidence.append("Classification is based on aggregate trace statistics.")
    return {"level": level, "evidence": evidence}


def taxonomy_counts(rows: list[dict[str, Any]]) -> dict[str, Any]:
    primary = Counter(row.get("primary_failure_class", "unknown") for row in rows)
    secondary = Counter()
    for row in rows:
        secondary.update(row.get("secondary_failure_classes", []))
    return {
        "primary": {class_label(code): count for code, count in sorted(primary.items()) if code in FAILURE_CLASSES},
        "secondary": {class_label(code): count for code, count in sorted(secondary.items()) if code in FAILURE_CLASSES},
    }


def _function_line_ranges(tree: ast.AST, names: set[str]) -> list[tuple[int, int]]:
    ranges: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and node.name in names:
            start = int(getattr(node, "lineno", 0))
            end = int(getattr(node, "end_lineno", start))
            ranges.append((start, end))
    return ranges


def _line_is_in_ranges(line: int, ranges: list[tuple[int, int]]) -> bool:
    return any(start <= line <= end for start, end in ranges)


def no_hack_audit(game_ids: list[str] | None = None) -> dict[str, Any]:
    scanned = [
        Path("src/arcagi3_adapter.py"),
        Path("src/arcagi3_baselines.py"),
        Path("src/arcagi3_official.py"),
        Path("src/arcagi3_official_eval.py"),
        Path("src/arcagi3_trace_analysis.py"),
        Path("src/arcagi3_failure_taxonomy.py"),
        Path("src/arcagi3_diagnose.py"),
    ]
    control_branch_files = {Path("src/arcagi3_adapter.py"), Path("src/arcagi3_baselines.py")}
    blocked_literals = [
        "solution",
        "replay_path",
        "hidden_goal",
        "ground_truth",
    ]
    findings: list[dict[str, Any]] = []
    ids = [item for item in (game_ids or []) if item]
    for path in scanned:
        if not path.exists():
            findings.append({"path": str(path), "line": None, "kind": "missing_source"})
            continue
        text = path.read_text(encoding="utf-8", errors="ignore")
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            findings.append({"path": str(path), "line": exc.lineno, "kind": "parse_error", "match": repr(exc)})
            continue
        excluded = _function_line_ranges(tree, {"no_hack_audit"})
        lines = text.splitlines()
        for idx, line in enumerate(lines, start=1):
            if _line_is_in_ranges(idx, excluded):
                continue
            lowered = line.lower()
            for game_id in ids:
                if game_id.lower() in lowered:
                    findings.append({"path": str(path), "line": idx, "kind": "game_id_literal", "match": game_id})
            for literal in blocked_literals:
                if literal in lowered:
                    findings.append({"path": str(path), "line": idx, "kind": "blocked_literal", "match": literal})
        if path in control_branch_files:
            for node in ast.walk(tree):
                if _line_is_in_ranges(int(getattr(node, "lineno", 0)), excluded):
                    continue
                if isinstance(node, ast.If) and "game_id" in ast.unparse(node.test):
                    findings.append(
                        {
                            "path": str(path),
                            "line": int(getattr(node, "lineno", 0)),
                            "kind": "game_id_branch",
                            "match": ast.unparse(node.test),
                        }
                    )
    return {
        "passes": len(findings) == 0,
        "findings": findings,
        "scanned": [str(path) for path in scanned],
        "blocked_checks": [
            "game_id_branches",
            "fixed_replay_paths",
            "hidden_or_private_solution_fields",
            "copied_level_action_paths",
        ],
    }
