from __future__ import annotations

import argparse
import ast
import json
from pathlib import Path
from typing import Any

from audit.leakage_scan import run_scan
from .arcagi3_eval import collect_hashes
from .external_eval import EXTERNAL_AUDITED_PATHS, supported_claim_has_evidence
from .external_registry import CLAIMS


AUDIT_PATHS = [
    "src/external_registry.py",
    "src/external_eval.py",
    "src/generalization_audit.py",
    "tests/test_external_generalization.py",
    "docs/external_generalization_report.json",
    "docs/external_generalization_report.md",
    "docs/generalization_audit.json",
    "frozen/recurrent_latent_fast.pt",
    "runs/explorer_tiny.pt",
]


def load_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def external_no_hack_scan(report: dict[str, Any]) -> dict[str, Any]:
    findings: list[dict[str, Any]] = []
    paths = [Path("src/external_eval.py"), Path("src/external_registry.py"), Path("src/generalization_audit.py")]
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
    blocked = ["world.step(language", "lang_in = language", "dialogue_history", "transcript"]
    findings = []
    for item in blocked:
        if item.lower() in runtime.lower() or item.lower() in external.lower():
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
    audit = {
        "terminal_outcome": "EXTERNAL GENERALIZATION AUDIT PROVEN" if all(value is True or not isinstance(value, bool) for value in checks.values()) else "NOT PROVEN",
        "external_report": str(report_path),
        "checks": checks,
        "leakage_scan": leakage,
        "external_no_hack_scan": no_hack,
        "no_text_as_state": text_state,
        "trace_checks": traces,
        "hidden_target_canary": canary,
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
