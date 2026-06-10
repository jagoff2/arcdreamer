from __future__ import annotations

from collections import defaultdict
from typing import Any


def aggregate_perception_diagnostics(suite_reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [row for report in suite_reports for row in report.get("rows", [])]
    by_variant: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_variant[str(row.get("variant"))].append(row.get("controller_summary", {}).get("perception", {}))
    variants = {}
    for variant, items in sorted(by_variant.items()):
        count = max(len(items), 1)
        prediction_trials = sum(int(item.get("prediction", {}).get("trials", 0)) for item in items)
        model_correct = sum(int(item.get("prediction", {}).get("model_correct", 0)) for item in items)
        null_correct = sum(int(item.get("prediction", {}).get("null_correct", 0)) for item in items)
        perturb_trials = sum(int(item.get("affordance_causality", {}).get("trials", 0)) for item in items)
        perturb_changed = sum(int(item.get("affordance_causality", {}).get("changed_actions", 0)) for item in items)
        variants[variant] = {
            "mean_component_count": sum(float(item.get("component_count", 0.0)) for item in items) / count,
            "mean_stable_tracks": sum(float(item.get("stable_tracks", 0.0)) for item in items) / count,
            "mean_click_regions": sum(float(item.get("click_regions", 0.0)) for item in items) / count,
            "mean_changed_regions": sum(float(item.get("changed_regions", 0.0)) for item in items) / count,
            "effect_entries": sum(int(item.get("effect_entries", 0)) for item in items),
            "prediction_trials": prediction_trials,
            "prediction_accuracy": model_correct / max(prediction_trials, 1),
            "null_prediction_accuracy": null_correct / max(prediction_trials, 1),
            "prediction_above_null": model_correct > null_correct,
            "affordance_perturbation_trials": perturb_trials,
            "affordance_perturbation_changed_rate": perturb_changed / max(perturb_trials, 1),
        }
    return variants


def action_effect_tables(suite_reports: list[dict[str, Any]], limit: int = 20) -> dict[str, list[dict[str, Any]]]:
    tables: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for report in suite_reports:
        for row in report.get("rows", []):
            variant = str(row.get("variant"))
            perception = row.get("controller_summary", {}).get("perception", {})
            for item in perception.get("top_effects", [])[:limit]:
                tables[variant].append({"suite_id": row.get("suite_id"), "task_id": row.get("task_id"), **item})
    out = {}
    for variant, rows in tables.items():
        rows.sort(key=lambda item: (abs(float(item.get("score", 0.0))), int(item.get("count", 0))), reverse=True)
        out[variant] = rows[:limit]
    return out


def dev_prediction_summary(suite_reports: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        row
        for report in suite_reports
        for row in report.get("rows", [])
        if row.get("split") == "dev" and str(row.get("variant")) not in {"baseline_unchanged", "null_patch_control"}
    ]
    trials = sum(int(row.get("controller_summary", {}).get("perception", {}).get("prediction", {}).get("trials", 0)) for row in rows)
    model_correct = sum(
        int(row.get("controller_summary", {}).get("perception", {}).get("prediction", {}).get("model_correct", 0)) for row in rows
    )
    null_correct = sum(
        int(row.get("controller_summary", {}).get("perception", {}).get("prediction", {}).get("null_correct", 0)) for row in rows
    )
    return {
        "trials": trials,
        "model_accuracy": model_correct / max(trials, 1),
        "null_accuracy": null_correct / max(trials, 1),
        "above_null": model_correct > null_correct,
    }


def selected_diagnostics(suite_reports: list[dict[str, Any]], selected: str) -> dict[str, Any]:
    selected_rows = [row for report in suite_reports for row in report.get("rows", []) if row.get("variant") == selected]
    all_diag = aggregate_perception_diagnostics(suite_reports)
    selected_diag = all_diag.get(selected, {})
    return {
        "selected_variant": selected,
        "selected": selected_diag,
        "dev_prediction": dev_prediction_summary(suite_reports),
        "any_components_extracted": any(value.get("mean_component_count", 0.0) > 0.0 for value in all_diag.values()),
        "any_stable_tracks": any(value.get("mean_stable_tracks", 0.0) > 0.0 for value in all_diag.values()),
        "any_action_effects": any(value.get("effect_entries", 0) > 0 for value in all_diag.values()),
        "affordance_perturbation_changed": selected_diag.get("affordance_perturbation_changed_rate", 0.0) > 0.0,
        "selected_row_count": len(selected_rows),
    }
