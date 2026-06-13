from __future__ import annotations

from pathlib import Path

import torch

from src.persistent_memory import PersistentMemoryState


def _seed_revisable_hypotheses() -> tuple[PersistentMemoryState, str, str]:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    observation = {"x": torch.tensor([0.0]), "private": torch.tensor([1])}
    for tick in range(1, 3):
        next_observation = {"x": torch.tensor([float(tick)]), "private": torch.tensor([1])}
        memory.append_event(
            tick=tick,
            observation=observation,
            action=5,
            next_observation=next_observation,
        )
        observation = next_observation

    change_id = "field_change|action:5|field:x"
    stable_id = "field_stable|action:5|field:private"
    return memory, change_id, stable_id


def test_failed_prediction_revision_tracks_patch_loss_posterior_and_retains_hypothesis() -> None:
    memory, change_id, _ = _seed_revisable_hypotheses()
    hypotheses = memory.hypothesis_posterior["hypotheses"]
    target = hypotheses[change_id]
    before_posterior = float(target["posterior"])
    before_failed = int(target["failed_predictions"])
    before_loss = float(target["prediction_loss"])

    failed_event_observation = {"x": torch.tensor([2.0]), "private": torch.tensor([1])}
    failed_event_next = {"x": torch.tensor([3.0]), "private": torch.tensor([1])}
    memory.append_event(
        tick=3,
        observation=failed_event_observation,
        action=5,
        next_observation=failed_event_next,
        prediction_error={
            "failed_hypotheses": [change_id],
            "expected": "x=3",
            "observed": "x=4",
            "loss": 2.5,
        },
    )

    target = memory.hypothesis_posterior["hypotheses"][change_id]
    assert target["failed_predictions"] == before_failed + 1
    assert target["prediction_loss"] == before_loss + 2.5
    assert target["posterior"] < before_posterior
    assert len(target["patches"]) == 1
    assert target["patches"][0]["kind"] == "prediction_failure"
    assert target["patches"][0]["failure"]["id"] == change_id
    assert target["patches"][0]["failure"]["loss"] == 2.5
    assert change_id in memory.hypothesis_posterior["hypotheses"]

    for extra_failure in range(18):
        memory.append_event(
            tick=4 + extra_failure,
            observation={"x": torch.tensor([float(3 + extra_failure)]), "private": torch.tensor([1])},
            action=5,
            next_observation={"x": torch.tensor([float(4 + extra_failure)]), "private": torch.tensor([1])},
            prediction_error={
                "failed_hypotheses": [change_id],
                "expected": "x",
                "observed": "x",
                "loss": 2.5,
            },
        )

    revisited = memory.hypothesis_posterior["hypotheses"][change_id]
    assert revisited["failed_predictions"] == before_failed + 1 + 18
    assert len(revisited["patches"]) == 16


def test_failure_records_match_by_family_action_field_selector_when_id_missing() -> None:
    memory, change_id, stable_id = _seed_revisable_hypotheses()
    change_before = int(memory.hypothesis_posterior["hypotheses"][change_id]["failed_predictions"])
    stable_before = int(memory.hypothesis_posterior["hypotheses"][stable_id]["failed_predictions"])

    memory.append_event(
        tick=3,
        observation={"x": torch.tensor([2.0]), "private": torch.tensor([1])},
        action=5,
        next_observation={"x": torch.tensor([3.0]), "private": torch.tensor([1])},
        prediction_error={
            "prediction_failed": True,
            "expected": "private unchanged",
            "observed": "private changed",
            "loss": 2.5,
            "family": "field_stable",
            "action": "5",
            "field": "private",
        },
    )

    changed = memory.hypothesis_posterior["hypotheses"][change_id]
    stable = memory.hypothesis_posterior["hypotheses"][stable_id]

    assert changed["failed_predictions"] == change_before
    assert stable["failed_predictions"] == stable_before + 1
    assert stable["prediction_loss"] == 2.5
    assert stable["patches"][0]["failure"]["family"] == "field_stable"
    assert stable["patches"][0]["failure"]["action"] == "5"
    assert stable["patches"][0]["failure"]["field"] == "private"


def test_failed_revision_evidence_survives_save_and_load(tmp_path: Path) -> None:
    memory, change_id, _ = _seed_revisable_hypotheses()
    memory.append_event(
        tick=3,
        observation={"x": torch.tensor([2.0]), "private": torch.tensor([1])},
        action=5,
        next_observation={"x": torch.tensor([3.0]), "private": torch.tensor([1])},
        prediction_error={
            "failed_hypotheses": [change_id],
            "expected": "x=3",
            "observed": "x=4",
            "loss": 2.5,
        },
    )

    path = tmp_path / "memory_failure_revision.pt"
    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=4, batch_size=1, device="cpu")

    loaded_hypotheses = loaded.hypothesis_posterior["hypotheses"]
    loaded_target = loaded_hypotheses[change_id]

    assert loaded_target["failed_predictions"] == 1
    assert loaded_target["prediction_loss"] == 2.5
    assert loaded_target["patches"][0]["failure"]["id"] == change_id
    assert loaded_target["patches"][0]["loss"] == 2.5
    assert len(loaded_target["patches"]) == 1
