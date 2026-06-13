from __future__ import annotations

from pathlib import Path

from audit.hardcoding_scan import run_scan, scan_file_for_hardcoding
from src.arc_affordance_report import no_hack_proof as affordance_no_hack_proof
from src.arcagi3_eval import no_hack_audit
from src.jepa_arc_eval import no_hack_proof as jepa_no_hack_proof


def _write_case(tmp_path: Path, name: str, source: str) -> Path:
    path = tmp_path / name
    path.write_text(source, encoding="utf-8")
    return path


def test_source_tree_has_no_hardcoded_policy_or_answer_tables() -> None:
    report = run_scan("src")
    assert report["passes"], report["findings"]
    scanned = set(report["files_scanned"])
    for required in {
        "src\\action_selection.py",
        "src\\causal_hypotheses.py",
        "src\\goal_inference.py",
        "src\\persistent_memory.py",
        "src\\run_unbroken.py",
    }:
        assert required in scanned


def test_scanner_catches_solution_map_keyed_by_puzzle_id(tmp_path: Path) -> None:
    path = _write_case(
        tmp_path,
        "bad_solution_map.py",
        """
SOLUTIONS_BY_GAME = {
    "game-0001": [1, 2, 3],
    "puzzle_abcd12": ["left", "right"],
}
""",
    )
    findings = scan_file_for_hardcoding(path)
    assert {item["kind"] for item in findings} >= {"hardcoded_solution_map"}


def test_scanner_catches_scripted_id_branch_returning_actions(tmp_path: Path) -> None:
    path = _write_case(
        tmp_path,
        "bad_scripted_branch.py",
        """
def choose_action(game_id, observation):
    if game_id == "game-0001":
        return [1, 2, 3]
    return []
""",
    )
    findings = scan_file_for_hardcoding(path)
    assert {item["kind"] for item in findings} >= {"scripted_policy_by_id"}


def test_scanner_catches_external_symbolic_memory_database_load(tmp_path: Path) -> None:
    path = _write_case(
        tmp_path,
        "bad_symbolic_db.py",
        """
import json

solver_database = json.load(open("solver_db.json"))
""",
    )
    findings = scan_file_for_hardcoding(path)
    assert {item["kind"] for item in findings} >= {"external_symbolic_memory_db"}


def test_scanner_allows_domain_general_grammar_tables(tmp_path: Path) -> None:
    path = _write_case(
        tmp_path,
        "generic_grammar.py",
        """
ACTION_VOCAB = {"move": 0, "click": 1, "wait": 2}
PROGRAM_GRAMMAR = {
    "selectors": ["color", "shape", "relation"],
    "transforms": ["move", "copy", "recolor"],
}
""",
    )
    assert scan_file_for_hardcoding(path) == []


def test_existing_scoped_no_hack_audits_still_pass() -> None:
    reports = [
        no_hack_audit(),
        affordance_no_hack_proof(),
        jepa_no_hack_proof(),
    ]
    for report in reports:
        assert report["passes"], report["findings"]
