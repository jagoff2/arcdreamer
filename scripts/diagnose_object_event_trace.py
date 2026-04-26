from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import json
from pathlib import Path
from statistics import mean
from typing import Any


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def _trace(row: dict[str, Any]) -> dict[str, Any]:
    trace = row.get("rank_trace", {})
    return trace if isinstance(trace, dict) else {}


def _selected(row: dict[str, Any]) -> dict[str, Any]:
    selected = _trace(row).get("selected_rank", {})
    return selected if isinstance(selected, dict) else {}


def _token(record: dict[str, Any]) -> dict[str, Any]:
    token = record.get("action_token", {})
    return token if isinstance(token, dict) else {}


def _float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return float(default)


def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
    selected_actions = [str(row.get("selected_action", "")) for row in rows]
    selected_records = [_selected(row) for row in rows]
    selected_tokens = [_token(record) for record in selected_records]
    repeated = Counter(selected_actions)
    mapped_cells = Counter(
        tuple(token.get("mapped_grid_cell_from_token") or ())
        for token in selected_tokens
        if token.get("mapped_grid_cell_from_token") is not None
    )
    mapping_errors = [
        {
            "step": int(row.get("step", 0) or 0),
            "action": row.get("selected_action", ""),
            "from_action": token.get("mapped_grid_cell_from_action"),
            "from_token": token.get("mapped_grid_cell_from_token"),
        }
        for row, token in zip(rows, selected_tokens)
        if token and not bool(token.get("mapping_match", False))
    ]
    component_names = (
        "component_base",
        "component_relation",
        "component_failed_action",
        "component_coordinate_noeffect",
        "component_axis_noeffect",
        "component_total",
        "relation_object_prior",
        "relation_positive_prior",
        "relation_repeat_penalty",
        "relation_contradiction_gate",
        "noeffect_contradiction_gate",
        "noeffect_contradiction_penalty",
        "basis_noeffect",
        "basis_uncertainty",
        "family_noeffect",
        "out_no_effect_prob",
        "total_score",
        "policy_rank",
        "diagnostic_utility",
    )
    component_values: dict[str, list[float]] = defaultdict(list)
    for record in selected_records:
        for name in component_names:
            component_values[name].append(_float(record.get(name, 0.0)))
    repeated_action = repeated.most_common(1)[0][0] if repeated else ""
    repeated_records = [record for action, record in zip(selected_actions, selected_records) if action == repeated_action]
    repeated_values: dict[str, list[float]] = defaultdict(list)
    for record in repeated_records:
        for name in component_names:
            repeated_values[name].append(_float(record.get(name, 0.0)))
    return {
        "step_count": len(rows),
        "selected_action_counts": dict(repeated.most_common(12)),
        "selected_mapped_cell_counts": {str(key): value for key, value in mapped_cells.most_common(12)},
        "mapping_error_count": len(mapping_errors),
        "mapping_errors_first_12": mapping_errors[:12],
        "selected_component_means": {name: mean(values) if values else 0.0 for name, values in component_values.items()},
        "repeated_action": repeated_action,
        "repeated_action_count": int(repeated.get(repeated_action, 0)),
        "repeated_component_means": {name: mean(values) if values else 0.0 for name, values in repeated_values.items()},
        "diagnostic_only": all(bool(_trace(row).get("runtime_rank_trace_diagnostics_only", False)) for row in rows),
        "controls_action": any(bool(_trace(row).get("runtime_rank_trace_controls_action", False)) for row in rows),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace", required=True, type=Path)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    rows = _load_jsonl(args.trace)
    print(json.dumps(summarize(rows), indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
