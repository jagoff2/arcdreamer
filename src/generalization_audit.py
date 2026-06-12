from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

import torch

from audit.leakage_scan import run_scan
from .arcagi3_eval import collect_hashes
from .external_eval import EXTERNAL_AUDITED_PATHS, supported_claim_has_evidence
from .external_registry import CLAIMS


AUDIT_PATHS = [
    "src/external_registry.py",
    "src/external_eval.py",
    "src/generalization_audit.py",
    "src/external_valence.py",
    "src/external_affordance.py",
    "src/external_collapse_experiment.py",
    "src/perceptual_affordance.py",
    "src/perception_train.py",
    "src/perception_eval.py",
    "src/external_perception_experiment.py",
    "src/trace_collect.py",
    "src/base_world_model.py",
    "src/base_pretrain.py",
    "src/base_eval.py",
    "src/base_retrain_experiment.py",
    "src/arc_affordance_baseline.py",
    "src/arc_affordance_search.py",
    "src/arc_affordance_eval.py",
    "src/arc_affordance_report.py",
    "tests/test_external_generalization.py",
    "tests/test_external_collapse.py",
    "tests/test_perceptual_affordance.py",
    "tests/test_external_base_training.py",
    "tests/test_arc_affordance_baseline.py",
    "docs/external_generalization_report.json",
    "docs/external_generalization_report.md",
    "docs/generalization_audit.json",
    "docs/external_collapse_report.json",
    "docs/external_collapse_report.md",
    "docs/generalization_audit_after_collapse.json",
    "docs/perceptual_affordance_report.json",
    "docs/perceptual_affordance_report.md",
    "docs/generalization_audit_after_perception.json",
    "docs/external_base_report.json",
    "docs/external_base_report.md",
    "docs/generalization_audit_after_base_retrain.json",
    "docs/audit_after_external_base.json",
    "docs/arc_affordance_report.json",
    "docs/arc_affordance_report.md",
    "docs/generalization_audit_after_affordance_baseline.json",
    "frozen/recurrent_latent_fast.pt",
    "frozen/external_base_v1.pt",
    "frozen/external_base_manifest_v1.json",
    "data/external_traces_manifest.json",
    "runs/explorer_tiny.pt",
]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def external_no_hack_scan(report: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    paths = [
        Path("src/external_eval.py"),
        Path("src/external_registry.py"),
        Path("src/generalization_audit.py"),
        Path("src/external_valence.py"),
        Path("src/external_affordance.py"),
        Path("src/external_collapse_experiment.py"),
        Path("src/perceptual_affordance.py"),
        Path("src/perception_train.py"),
        Path("src/perception_eval.py"),
        Path("src/external_perception_experiment.py"),
        Path("src/trace_collect.py"),
        Path("src/base_world_model.py"),
        Path("src/base_pretrain.py"),
        Path("src/base_eval.py"),
        Path("src/base_retrain_experiment.py"),
        Path("src/arc_affordance_baseline.py"),
        Path("src/arc_affordance_search.py"),
        Path("src/arc_affordance_eval.py"),
        Path("src/arc_affordance_report.py"),
    ]
    forbidden_literals = [
        "hidden" + "_goal",
        "ground" + "_truth",
        "answer" + "_key",
        "solution" + "_path",
        "manual" + "_hint",
    ]
    task_ids = [
        task
        for suite in report.get("external_suites_discovered", [])
        for task in suite.get("tasks", [])
        if task not in {"CartPole-v1", "FrozenLake-v1"}
    ]
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        for literal in forbidden_literals:
            if literal in lowered:
                findings.append({"path": str(path), "kind": "forbidden_literal", "match": literal})
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            findings.append({"path": str(path), "line": exc.lineno, "kind": "parse_error", "match": repr(exc)})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = ast.unparse(node.test)
                if any(task_id in test for task_id in task_ids):
                    findings.append(
                        {"path": str(path), "line": getattr(node, "lineno", None), "kind": "task_specific_branch", "match": test}
                    )
    return {
        "passes": len(findings) == 0,
        "findings": findings,
        "scanned": [str(path) for path in paths],
        "allowed_task_wrappers": ["CartPole-v1", "FrozenLake-v1"],
    }


def no_text_as_state_check() -> dict[str, Any]:
    runtime = Path("src/run_unbroken.py").read_text(encoding="utf-8")
    external = Path("src/external_eval.py").read_text(encoding="utf-8")
    collapse = "\n".join(
        path.read_text(encoding="utf-8", errors="ignore")
        for path in [
            Path("src/external_valence.py"),
            Path("src/external_affordance.py"),
            Path("src/external_collapse_experiment.py"),
            Path("src/perceptual_affordance.py"),
            Path("src/perception_train.py"),
            Path("src/perception_eval.py"),
            Path("src/external_perception_experiment.py"),
            Path("src/trace_collect.py"),
            Path("src/base_world_model.py"),
            Path("src/base_pretrain.py"),
            Path("src/base_eval.py"),
            Path("src/base_retrain_experiment.py"),
            Path("src/arc_affordance_baseline.py"),
            Path("src/arc_affordance_search.py"),
            Path("src/arc_affordance_eval.py"),
            Path("src/arc_affordance_report.py"),
        ]
        if path.exists()
    )
    blocked = ["world.step(language", "lang_in = language", "dialogue_history", "transcript"]
    findings = []
    for item in blocked:
        if item.lower() in runtime.lower() or item.lower() in external.lower() or item.lower() in collapse.lower():
            findings.append(item)
    return {
        "passes": not findings and "model.step(observation, z)" in runtime,
        "findings": findings,
        "runtime_z_loop_present": "model.step(observation, z)" in runtime,
    }


def claim_discipline_checks(report: dict[str, Any]) -> dict[str, Any]:
    claims = report.get("claim_registry", [])
    claim_ids = {row.get("claim_id") for row in claims}
    expected = {claim.claim_id for claim in CLAIMS}
    supported = [row for row in claims if row.get("status") == "supported"]
    dev_only_supported = [
        row
        for row in supported
        if "dev" in json.dumps(row.get("result", {})).lower() and "sealed_eval" not in json.dumps(row.get("result", {})).lower()
    ]
    return {
        "claim_registry_complete": claim_ids == expected,
        "supported_claims_have_external_thresholds": all(supported_claim_has_evidence(row) for row in supported),
        "no_dev_only_supported_claims": not dev_only_supported,
        "unsupported_claims_preserved": bool([row for row in claims if row.get("status") == "unsupported"]),
        "supported_claims": [row.get("claim_id") for row in supported],
        "dev_only_supported_claims": [row.get("claim_id") for row in dev_only_supported],
    }


def trace_checks(report: dict[str, Any]) -> dict[str, Any]:
    paths = [Path(path) for path in report.get("trace_paths", [])]
    missing = [str(path) for path in paths if not path.exists()]
    sample_failures: list[str] = []
    for path in paths[:20]:
        if not path.exists():
            continue
        payload = load_json(path)
        for step in payload.get("steps", [])[:3]:
            required = {"legal_actions", "chosen_action", "baseline_actions", "score_delta", "event_delta", "memory_drive_hypothesis", "failure_class"}
            if not required.issubset(step):
                sample_failures.append(str(path))
                break
    return {
        "trace_count": len(paths),
        "missing_traces": missing,
        "sample_schema_failures": sample_failures,
        "passes": bool(paths) and not missing and not sample_failures,
    }


def hidden_target_canary(report_path: str | Path = "docs/audit_after_arcagi3_diagnosis.json") -> dict[str, Any]:
    path = Path(report_path)
    if not path.exists():
        return {"passes": False, "missing": str(path)}
    report = load_json(path)
    diff = float(report.get("anti_leakage", {}).get("hidden_target_canary_max_abs_diff", 1.0))
    return {"passes": diff == 0.0, "hidden_target_canary_max_abs_diff": diff, "source": str(path)}


def collapse_report_checks(report_path: str | Path = "docs/external_collapse_report.json") -> dict[str, Any]:
    path = Path(report_path)
    if not path.exists():
        return {"present": False, "passes": True}
    report = load_json(path)
    variants = {item.get("variant_id") for item in report.get("variants", [])}
    suites = {item.get("suite_id") for item in report.get("suite_reports", [])}
    baselines = {item.get("baseline") for item in report.get("baselines", [])}
    ablations = {str(item.get("ablation", "")).replace("ablation_", "") for item in report.get("ablations", [])}
    trace_paths = [Path(item) for item in report.get("trace_paths", [])]
    missing = [str(item) for item in trace_paths if not item.exists()]
    required_arc = report.get("required_arc_trace_paths", {})
    gate = report.get("improvement_gate", {})
    outcome = report.get("terminal_outcome")
    cuda_available = bool(torch.cuda.is_available())
    device_runtime = report.get("device_runtime", {})
    suite_device_runtime = report.get("suite_device_runtime", {})
    gate_consistent = (
        (bool(gate.get("passes")) and outcome == "MINIMAL EXTERNAL IMPROVEMENT FOUND")
        or ((not bool(gate.get("passes"))) and outcome == "NO MINIMAL IMPROVEMENT FOUND")
    )
    required_variants = {
        "baseline_unchanged",
        "valence_only",
        "affordance_only",
        "loop_aversion_only",
        "valence_affordance",
        "valence_affordance_loop",
        "null_patch_control",
    }
    checks = {
        "collapse_report_present": True,
        "collapse_outcome_valid": outcome in {"MINIMAL EXTERNAL IMPROVEMENT FOUND", "NO MINIMAL IMPROVEMENT FOUND"},
        "collapse_gate_consistent": gate_consistent,
        "collapse_variants_complete": required_variants.issubset(variants),
        "collapse_external_suites_complete": {"official_arcagi3", "gymnasium_classic_control", "gymnasium_toy_text"}.issubset(suites),
        "collapse_baselines_complete": {
            "random_legal",
            "repeat_last_action",
            "coverage_graph_exploration",
            "novelty_first",
            "greedy_observable_score_delta",
            "oracle_free_observed_graph_bfs",
        }.issubset(baselines),
        "collapse_ablations_complete": {"zero_z", "corrupt_memory", "corrupt_drive"}.issubset(ablations),
        "collapse_traces_exist": bool(trace_paths) and not missing,
        "collapse_required_arc_traces": int(required_arc.get("baseline_unchanged_count", 0)) == 25
        and int(required_arc.get("selected_variant_count", 0)) == 25,
        "collapse_no_hack_passes": bool(report.get("no_hack_proof", {}).get("passes")),
        "collapse_no_external_judge": bool(report.get("no_hack_proof", {}).get("no_external_judge")),
        "collapse_no_forced_cycle": bool(report.get("no_hack_proof", {}).get("no_forced_cycle")),
        "collapse_cuda_available_recorded": bool(device_runtime.get("torch_cuda_available")) == cuda_available,
        "collapse_cuda_main_runtime": (not cuda_available) or str(device_runtime.get("resolved_device", "")).startswith("cuda"),
        "collapse_cuda_suite_runtimes": (not cuda_available)
        or (
            {"official_arcagi3", "gymnasium_classic_control", "gymnasium_toy_text"}.issubset(suite_device_runtime)
            and all(str(item.get("resolved_device", "")).startswith("cuda") for item in suite_device_runtime.values())
        ),
    }
    return {
        "present": True,
        "passes": all(checks.values()),
        "checks": checks,
        "missing_traces": missing[:20],
        "outcome": outcome,
        "selected_variant": gate.get("selected_variant"),
    }


def perception_report_checks(report_path: str | Path = "docs/perceptual_affordance_report.json") -> dict[str, Any]:
    path = Path(report_path)
    if not path.exists():
        return {"present": False, "passes": True}
    report = load_json(path)
    variants = {item.get("variant_id") for item in report.get("variants", [])}
    suites = {item.get("suite_id") for item in report.get("suite_reports", [])}
    baselines = {item.get("baseline") for item in report.get("baselines", [])}
    trace_paths = [Path(item) for item in report.get("trace_paths", [])]
    missing = [str(item) for item in trace_paths if not item.exists()]
    required_arc = report.get("required_arc_trace_paths", {})
    gate = report.get("improvement_gate", {})
    outcome = report.get("terminal_outcome")
    cuda_available = bool(torch.cuda.is_available())
    device_runtime = report.get("device_runtime", {})
    suite_device_runtime = report.get("suite_device_runtime", {})
    diagnostics = report.get("selected_diagnostics", {})
    analysis = report.get("analysis", {})
    gate_consistent = (
        (bool(gate.get("passes")) and outcome == "PERCEPTUAL AFFORDANCE IMPROVEMENT FOUND")
        or ((not bool(gate.get("passes"))) and outcome == "NO IMPROVEMENT FOUND")
    )
    required_variants = {
        "baseline_unchanged",
        "component_only",
        "temporal_slots",
        "action_effect_memory",
        "predictive_object_model",
        "full_perceptual_affordance",
        "null_patch_control",
    }
    checks = {
        "perception_report_present": True,
        "perception_outcome_valid": outcome in {"PERCEPTUAL AFFORDANCE IMPROVEMENT FOUND", "NO IMPROVEMENT FOUND"},
        "perception_gate_consistent": gate_consistent,
        "perception_variants_complete": required_variants.issubset(variants),
        "perception_external_suites_complete": {"official_arcagi3", "gymnasium_classic_control", "gymnasium_toy_text"}.issubset(suites),
        "perception_baselines_complete": {
            "random_legal",
            "repeat_last_action",
            "coverage_graph_exploration",
            "novelty_first",
            "greedy_observable_score_delta",
            "oracle_free_observed_graph_bfs",
        }.issubset(baselines),
        "perception_ablations_present": bool(report.get("ablations")),
        "perception_traces_exist": bool(trace_paths) and not missing,
        "perception_required_arc_traces": int(required_arc.get("baseline_unchanged_count", 0)) == 25
        and int(required_arc.get("selected_variant_count", 0)) == 25,
        "perception_no_hack_passes": bool(report.get("no_hack_proof", {}).get("passes")),
        "perception_no_external_judge": bool(report.get("no_hack_proof", {}).get("no_external_judge")),
        "perception_no_forced_cycle": bool(report.get("no_hack_proof", {}).get("no_forced_cycle")),
        "perception_cuda_available_recorded": bool(device_runtime.get("torch_cuda_available")) == cuda_available,
        "perception_cuda_main_runtime": (not cuda_available) or str(device_runtime.get("resolved_device", "")).startswith("cuda"),
        "perception_cuda_suite_runtimes": (not cuda_available)
        or (
            {"official_arcagi3", "gymnasium_classic_control", "gymnasium_toy_text"}.issubset(suite_device_runtime)
            and all(str(item.get("resolved_device", "")).startswith("cuda") for item in suite_device_runtime.values())
        ),
        "perception_components_extracted": bool(analysis.get("did_components_extract_regions")),
        "perception_action_effects_recorded": bool(analysis.get("did_action_effect_memory_update")),
        "perception_selected_diagnostics_present": bool(diagnostics),
    }
    return {
        "present": True,
        "passes": all(checks.values()),
        "checks": checks,
        "missing_traces": missing[:20],
        "outcome": outcome,
        "selected_variant": gate.get("selected_variant"),
    }


def external_base_report_checks(report_path: str | Path = "docs/external_base_report.json") -> dict[str, Any]:
    path = Path(report_path)
    if not path.exists():
        return {"present": False, "passes": True}
    report = load_json(path)
    arms = {item.get("arm_id") for item in report.get("model_arms", [])}
    suites = {item.get("suite_id") for item in report.get("suite_reports", [])}
    baselines = {item.get("baseline") for item in report.get("baselines", [])}
    ablations = {str(item.get("ablation", "")).replace("ablation_", "") for item in report.get("ablations", [])}
    trace_paths = [Path(item) for item in report.get("trace_paths", [])]
    missing = [str(item) for item in trace_paths if not item.exists()]
    required_arc = report.get("required_arc_trace_paths", {})
    gate = report.get("improvement_gate", {})
    outcome = report.get("terminal_outcome")
    cuda_available = bool(torch.cuda.is_available())
    device_runtime = report.get("device_runtime", {})
    suite_device_runtime = report.get("suite_device_runtime", {})
    trace_manifest = report.get("trace_manifest", {})
    frozen_manifest = report.get("frozen_manifest", {})
    no_hack = report.get("no_hack_proof", {})
    gate_consistent = (
        (bool(gate.get("passes")) and outcome == "EXTERNAL BASE IMPROVEMENT FOUND")
        or ((not bool(gate.get("passes"))) and outcome == "NO IMPROVEMENT FOUND")
    )
    required_arms = {
        "old_base_unchanged",
        "old_base_world_model",
        "old_base_finetuned",
        "from_scratch_external_base",
        "null_training_control",
    }
    required_ablations = {"no_external_base", "no_world_model", "no_memory", "no_affordance"}
    old_hash = "D36D59ED56A5BF4DC79835CB04D8B10F46E59FB00B2FE95DBF5AED30D1DBEFBD"
    current_old_hash = collect_hashes(["frozen/recurrent_latent_fast.pt"])["frozen/recurrent_latent_fast.pt"].get("sha256")
    checks = {
        "external_base_report_present": True,
        "external_base_outcome_valid": outcome in {"EXTERNAL BASE IMPROVEMENT FOUND", "NO IMPROVEMENT FOUND"},
        "external_base_gate_consistent": gate_consistent,
        "external_base_arms_complete": required_arms.issubset(arms),
        "external_base_external_suites_complete": {"official_arcagi3", "gymnasium_classic_control", "gymnasium_toy_text"}.issubset(suites),
        "external_base_baselines_complete": {
            "random_legal",
            "repeat_last_action",
            "coverage_graph_exploration",
            "novelty_first",
            "greedy_observable_score_delta",
            "oracle_free_observed_graph_bfs",
        }.issubset(baselines),
        "external_base_ablations_complete": required_ablations.issubset(ablations),
        "external_base_traces_exist": bool(trace_paths) and not missing,
        "external_base_required_arc_traces": int(required_arc.get("old_base_unchanged_count", 0)) == 25
        and int(required_arc.get("selected_variant_count", 0)) == 25,
        "external_base_trace_manifest_present": bool(trace_manifest)
        and Path(str(report.get("frozen_manifest", {}).get("trace_manifest", "data/external_traces_manifest.json"))).exists(),
        "external_base_trace_count_first_pass": int(trace_manifest.get("transition_count", 0)) >= 1_000_000,
        "external_base_checkpoint_exists": Path(str(report.get("external_base_checkpoint", ""))).exists(),
        "external_base_manifest_exists": Path(str(report.get("external_base_manifest", ""))).exists(),
        "external_base_old_checkpoint_preserved": current_old_hash == old_hash
        and frozen_manifest.get("old_recurrent_checkpoint_sha256") == old_hash,
        "external_base_training_diagnostics_present": bool(report.get("training_losses")),
        "external_base_negative_results_preserved": bool(report.get("negative_result_preservation")),
        "external_base_no_hack_passes": bool(no_hack.get("passes")),
        "external_base_no_external_judge": bool(no_hack.get("no_external_judge")),
        "external_base_no_forced_cycle": bool(no_hack.get("no_forced_cycle")),
        "external_base_no_public_text_as_state": bool(no_hack.get("no_public_text_as_state")),
        "external_base_cuda_available_recorded": bool(device_runtime.get("torch_cuda_available")) == cuda_available,
        "external_base_cuda_main_runtime": (not cuda_available) or str(device_runtime.get("resolved_device", "")).startswith("cuda"),
        "external_base_cuda_suite_runtimes": (not cuda_available)
        or (
            {"official_arcagi3", "gymnasium_classic_control", "gymnasium_toy_text"}.issubset(suite_device_runtime)
            and all(str(item.get("resolved_device", "")).startswith("cuda") for item in suite_device_runtime.values())
        ),
    }
    return {
        "present": True,
        "passes": all(checks.values()),
        "checks": checks,
        "missing_traces": missing[:20],
        "outcome": outcome,
        "selected_variant": gate.get("selected_variant"),
    }


def arc_affordance_report_checks(report_path: str | Path = "docs/arc_affordance_report.json") -> dict[str, Any]:
    path = Path(report_path)
    if not path.exists():
        return {"present": False, "passes": True}
    report = load_json(path)
    declared_variants = {item.get("variant_id") for item in report.get("declared_variants", [])}
    variant_rows = {item.get("variant") for item in report.get("variant_table", [])}
    comparison_rows = {item.get("variant") for item in report.get("comparison_table", [])}
    ablation_rows = {item.get("variant") for item in report.get("ablation_table", [])}
    trace_paths = [Path(item) for item in report.get("trace_paths", [])]
    missing = [str(item) for item in trace_paths if not item.exists()]
    sample_failures: list[str] = []
    required_step_fields = {
        "game",
        "step",
        "obs_hash",
        "legal_action_count",
        "chosen_action",
        "baseline_actions",
        "component_summary",
        "changed_regions",
        "action_effect_memory",
        "score_delta",
        "event_delta",
        "terminal",
        "invalid_flag",
        "repeat_cycle_stats",
        "choice_reason",
    }
    for trace_path in trace_paths[:20]:
        if not trace_path.exists():
            continue
        payload = load_json(trace_path)
        steps = payload.get("steps", [])
        if not steps:
            sample_failures.append(str(trace_path))
            continue
        if not required_step_fields.issubset(steps[0]):
            sample_failures.append(str(trace_path))
    gate = report.get("improvement_gate", {})
    selected = str(report.get("selected_variant", ""))
    selected_row = next((row for row in report.get("aggregate_table", []) if row.get("variant") == selected), {})
    outcome = report.get("terminal_outcome")
    no_hack = report.get("no_hack_proof", {})
    runtime = report.get("runtime_status", {})
    device_runtime = runtime.get("device_runtime", {})
    old_explorer_device = runtime.get("old_explorer_device", {})
    cuda_available = bool(torch.cuda.is_available())
    gate_consistent = (
        (bool(gate.get("passes")) and outcome == "AFFORDANCE BASELINE SIGNAL FOUND")
        or ((not bool(gate.get("passes"))) and outcome == "NO SIGNAL FOUND")
    )
    repeat = float(selected_row.get("mean_repeat_collapse", 1.0))
    repeat_explained = repeat <= 0.50 or bool(report.get("repeat_collapse_explanation"))
    required_variants = {
        "component_click_search",
        "change_memory_search",
        "state_graph_affordance",
        "event_linked_ranking",
        "object_persistence_search",
        "combined_affordance_search",
    }
    required_ablations = {
        "ablation_no_component_memory",
        "ablation_no_change_memory",
        "ablation_no_event_memory",
    }
    required_comparisons = {
        "random_legal",
        "repeat_last_action",
        "coverage_graph_exploration",
        "novelty_first",
        "greedy_observable_score_delta",
        "oracle_free_observed_graph_bfs",
        "old_explorer",
    }
    allowed_inputs = set(report.get("source_data", {}).get("allowed_inputs", []))
    disallowed_sources = set(report.get("source_data", {}).get("not_used", []))
    checks = {
        "arc_affordance_report_present": True,
        "arc_affordance_outcome_valid": outcome in {"AFFORDANCE BASELINE SIGNAL FOUND", "NO SIGNAL FOUND"},
        "arc_affordance_gate_consistent": gate_consistent,
        "arc_affordance_declared_variants_complete": required_variants.issubset(declared_variants),
        "arc_affordance_variant_rows_complete": required_variants.issubset(variant_rows),
        "arc_affordance_comparisons_complete": required_comparisons.issubset(comparison_rows)
        and required_comparisons.issubset(set(report.get("comparison_baselines", []))),
        "arc_affordance_ablations_complete": required_ablations.issubset(ablation_rows),
        "arc_affordance_traces_exist": bool(trace_paths) and not missing,
        "arc_affordance_required_official_traces": int(report.get("required_official_trace_count", 0)) >= 25,
        "arc_affordance_trace_schema": not sample_failures,
        "arc_affordance_no_hack_passes": bool(no_hack.get("passes")),
        "arc_affordance_no_external_judge": bool(no_hack.get("no_external_judge")),
        "arc_affordance_no_game_specific_branches": bool(no_hack.get("no_game_specific_branches")),
        "arc_affordance_no_forced_cycle": bool(no_hack.get("no_forced_cycle")),
        "arc_affordance_invalid_action_rate_zero": float(gate.get("official_invalid_action_rate", 1.0)) == 0.0,
        "arc_affordance_repeat_ok_or_explained": repeat_explained,
        "arc_affordance_source_data_public_only": {
            "official public observations",
            "legal actions",
            "public reward and score deltas",
            "public event deltas",
            "terminal flags",
        }.issubset(allowed_inputs)
        and {
            "neural training",
            "pretrained models",
            "web data",
            "game source inspection",
            "game-specific branches",
            "manual hints",
        }.issubset(disallowed_sources),
        "arc_affordance_cuda_available_recorded": bool(device_runtime.get("torch_cuda_available")) == cuda_available,
        "arc_affordance_cuda_runtime": (not cuda_available) or str(runtime.get("resolved_device", "")).startswith("cuda"),
        "arc_affordance_old_explorer_cuda_parameters": (not cuda_available)
        or bool(old_explorer_device.get("cuda_model_parameters")),
        "arc_affordance_action_effect_metrics_present": "action_effect_hit_rate" in report
        and "no_op_avoidance_after_no_effect_evidence" in report,
    }
    return {
        "present": True,
        "passes": all(checks.values()),
        "checks": checks,
        "missing_traces": missing[:20],
        "sample_schema_failures": sample_failures[:20],
        "outcome": outcome,
        "selected_variant": selected,
    }


def build_audit(
    *,
    report_path: str | Path,
    json_output: str | Path,
) -> dict[str, Any]:
    report = load_json(report_path)
    leakage = run_scan()
    no_hack = external_no_hack_scan(report)
    text_state = no_text_as_state_check()
    claims = claim_discipline_checks(report)
    traces = trace_checks(report)
    canary = hidden_target_canary()
    collapse = collapse_report_checks()
    perception = perception_report_checks()
    external_base = external_base_report_checks()
    arc_affordance = arc_affordance_report_checks()
    hashes = collect_hashes(AUDIT_PATHS + EXTERNAL_AUDITED_PATHS)
    frozen_ok = (
        hashes["frozen/recurrent_latent_fast.pt"].get("sha256") == "D36D59ED56A5BF4DC79835CB04D8B10F46E59FB00B2FE95DBF5AED30D1DBEFBD"
    )
    checks = {
        "external_report_proven": report.get("terminal_outcome") == "EXTERNAL GENERALIZATION DISCIPLINE PROVEN",
        "leakage_scan_passes": bool(leakage.get("passes")),
        "external_no_hack_passes": no_hack["passes"],
        "no_text_as_state": text_state["passes"],
        "hidden_target_canary_diff_zero": canary["passes"],
        "frozen_hash_recorded_and_unchanged": frozen_ok,
        "traceability_passes": traces["passes"],
        **claims,
    }
    if collapse["present"]:
        checks.update(collapse.get("checks", {}))
    if perception["present"]:
        checks.update(perception.get("checks", {}))
    if external_base["present"]:
        checks.update(external_base.get("checks", {}))
    if arc_affordance["present"]:
        checks.update(arc_affordance.get("checks", {}))
    audit = {
        "terminal_outcome": "EXTERNAL GENERALIZATION AUDIT PROVEN" if all(value is True or not isinstance(value, bool) for value in checks.values()) else "NOT PROVEN",
        "external_report": str(report_path),
        "checks": checks,
        "leakage_scan": leakage,
        "external_no_hack_scan": no_hack,
        "no_text_as_state": text_state,
        "trace_checks": traces,
        "hidden_target_canary": canary,
        "collapse_report_checks": collapse,
        "perception_report_checks": perception,
        "external_base_report_checks": external_base,
        "arc_affordance_report_checks": arc_affordance,
        "hashes": hashes,
        "limitations": [],
    }
    if not any(row.get("status") == "supported" for row in report.get("claim_registry", [])):
        audit["limitations"].append("No active capability claim is externally supported; this is acceptable discipline, not performance success.")
    output = Path(json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(audit, indent=2), encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", default="docs/external_generalization_report.json")
    parser.add_argument("--json-output", default="docs/generalization_audit.json")
    args = parser.parse_args()
    audit = build_audit(report_path=args.report, json_output=args.json_output)
    print(json.dumps({"terminal_outcome": audit["terminal_outcome"], "checks": audit["checks"], "limitations": audit["limitations"]}, indent=2))


if __name__ == "__main__":
    main()
