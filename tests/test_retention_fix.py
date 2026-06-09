from __future__ import annotations

from pathlib import Path

from src.continual_learning import PersistentConceptMemory, learn_concept_sequence, recall_accuracy
from src.heldout_causal import FROZEN_CHECKPOINT
from src.retention_eval import evaluate_retention_fix


def test_concept_memory_learns_sequence_and_corrupt_degrades(tmp_path: Path) -> None:
    concept_ids = list(range(5))
    path = tmp_path / "concepts.pt"
    report = learn_concept_sequence(concept_ids, path)
    rows = report["concept_rows"]
    assert len(rows) == 5
    assert all(row["accuracy_before"] <= 0.20 for row in rows)
    assert all(row["accuracy_after"] >= 0.85 for row in rows)
    loaded = PersistentConceptMemory.load(path)
    assert recall_accuracy(loaded, concept_ids) >= 0.85
    assert recall_accuracy(loaded.zeroed(), concept_ids) <= 0.20
    assert recall_accuracy(loaded.corrupted(), concept_ids) <= 0.20


def test_retention_fix_smoke_verdict_passes(tmp_path: Path) -> None:
    report = evaluate_retention_fix(
        FROZEN_CHECKPOINT,
        config_name="smoke",
        json_output=tmp_path / "retention.json",
    )
    assert report["terminal_outcome"] == "RETENTION FIX PROVEN"
    assert report["new_concepts"]["passes"] is True
    assert report["old_task_retention"]["passes"] is True
    assert report["restart"]["checks"]["zero_memory_degrades_new_recall"] is True
    assert report["prior_living"]["passes"] is True
    assert report["audit_preservation"]["passes"] is True
