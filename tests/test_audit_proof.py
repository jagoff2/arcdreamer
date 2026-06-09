from __future__ import annotations

import json
from pathlib import Path

from audit.independent_verify import AUDITED_PATHS, collect_hashes, manifest_status, source_snippets
from audit.leakage_scan import run_scan


def test_audit_leakage_scan_passes_static_checks() -> None:
    report = run_scan()
    assert report["passes"], report["findings"]


def test_audit_hash_manifest_check_matches_frozen_artifacts() -> None:
    hashes = collect_hashes(AUDITED_PATHS)
    status = manifest_status(hashes)
    assert status["model_hash_matches"]
    assert status["checkpoint_hash_matches"]
    assert status["checkpoint_size_matches"]


def test_audit_snippets_cover_required_code_paths() -> None:
    snippets = source_snippets()
    required = {
        "model_step_update",
        "runtime_z_loop",
        "memory_load_save",
        "model_input_construction",
        "target_input_separation",
        "private_token_generation_use",
        "curriculum_update",
        "metric_computation",
        "leakage_scan_logic",
    }
    assert required <= set(snippets)
    assert "model.step" in snippets["runtime_z_loop"]["text"]
    assert "private_in" in snippets["private_token_generation_use"]["text"]


def test_existing_living_report_is_json_with_verdict() -> None:
    report = json.loads(Path("docs/living_system_report.json").read_text(encoding="utf-8"))
    assert report["verdict"]["passes"] is True
