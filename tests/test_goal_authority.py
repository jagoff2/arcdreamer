from __future__ import annotations

from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
GOAL_PATH = ROOT / "GOAL.md"
TRACKER_PATH = ROOT / "GOAL_IMPLEMENTATION_TRACKER.md"


def _read_text(path: Path) -> str:
    return path.read_text(encoding="utf-8")


def test_goal_file_exists_and_has_current_arc_failure_acceptance_content() -> None:
    assert GOAL_PATH.exists(), f"Missing {GOAL_PATH}"
    content = _read_text(GOAL_PATH)

    assert "goal-forming intelligence" in content
    assert "goal-hypothesis search over candidate terminal predicates" in content
    assert "progress_hypothesis_bank" in content
    assert "goal_predicate_generator" in content
    assert "counterfactual_goal_progress_score" in content
    assert "goal_entropy_reduction" in content
    assert "useful_effect" in content
    assert "instrumental_evidence" in content


def test_tracker_declares_goal_markdown_as_source_of_truth() -> None:
    assert TRACKER_PATH.exists(), f"Missing {TRACKER_PATH}"
    content = _read_text(TRACKER_PATH)

    assert "Source of truth: `GOAL.md`." in content


def test_tracker_only_tracks_progress_and_does_not_replace_goal_md() -> None:
    content = _read_text(TRACKER_PATH)

    assert "This file is a progress ledger only." in content
    assert "must not replace" in content
    assert "or reinterpret `GOAL.md`." in content


def test_tracker_has_c0_1_contract_row() -> None:
    content = _read_text(TRACKER_PATH).splitlines()
    row = next((line for line in content if line.startswith("| C0.1 |")), "")
    assert row, "Missing C0.1 row in tracker"
    assert "`GOAL.md` remains the authoritative acceptance contract." in row
    assert "This tracker references, but does not replace, `GOAL.md`." in row
