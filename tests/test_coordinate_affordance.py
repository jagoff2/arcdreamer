from __future__ import annotations

import numpy as np
import pytest
import torch

import src.coordinate_affordance as coordinate_affordance
from src.coordinate_affordance import CoordinateAffordanceModel, propose_coordinate_candidates
from src.grid_perception import parse_grid
from src.persistent_memory import PersistentMemoryState


def _build_sparse_64_grid() -> tuple[np.ndarray, np.ndarray]:
    before = np.zeros((64, 64), dtype=np.int64)
    after = np.zeros((64, 64), dtype=np.int64)
    after[5:9, 5:9] = 4
    after[5:9, 20:24] = 4
    after[5:9, 35:39] = 4
    after[30:34, 10:14] = 7
    before[0, 0] = 1
    return before, after


def test_coordinate_affordance_model_tracks_change_and_no_op_rates() -> None:
    model = CoordinateAffordanceModel.fresh((4, 4))

    assert model.shape == (4, 4)
    assert tuple(np.asarray(model.heatmap()).shape) == (4, 4)
    assert float(np.asarray(model.heatmap()).sum()) == 0.0

    before = np.array([[0, 1, 0, 0], [0, 0, 0, 2], [0, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64)
    changed_after = before.copy()
    changed_after[1, 1] = 3
    noop_after = before.copy()
    noop_after[1, 1] = 3

    changed_report = model.update(before, changed_after, (1, 1), score_delta=2.0)
    _ = model.update(changed_after, noop_after, (2, 2), score_delta=0.5)
    noop_report = model.update(changed_after, changed_after, (2, 2), score_delta=0.5)

    assert changed_report["coordinate"] == [1, 1]
    assert changed_report["changed"] is True
    assert model.trials["1,1"] == 1
    assert model.changed["1,1"] == 1
    assert model.no_op["1,1"] == 0
    assert model.trials["2,2"] == 2
    assert model.changed["2,2"] == 0
    assert model.no_op["2,2"] == 2
    assert model.p_change(1, 1) > model.p_change(2, 2)
    assert model.p_change(1, 1) == 1.0
    assert model.p_change(2, 2) == 0.0
    assert changed_report["p_change"] == model.p_change(1, 1)
    assert changed_report["mean_score_delta"] == 2.0
    assert noop_report["mean_score_delta"] == 0.5


def test_coordinate_affordance_model_serializes_and_restores() -> None:
    model = CoordinateAffordanceModel.fresh((3, 3))
    before = np.array([[0, 1, 0], [0, 2, 0], [0, 0, 0]], dtype=np.int64)
    after = np.array([[0, 1, 0], [0, 0, 0], [0, 0, 3]], dtype=np.int64)

    report = model.update(before, after, (2, 2), score_delta=1.25)
    assert report["changed"] is True

    payload = model.to_dict()
    loaded = CoordinateAffordanceModel.from_dict(payload)

    assert loaded.shape == model.shape
    assert loaded.trials == model.trials
    assert loaded.changed == model.changed
    assert loaded.no_op == model.no_op
    assert np.asarray(loaded.heatmap()).tolist() == np.asarray(model.heatmap()).tolist()
    assert loaded.p_change(2, 2) == 1.0


def test_propose_coordinate_candidates_collects_sparse_sources_and_remains_deterministic() -> None:
    before = np.array(
        [
            [2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    after = np.array(
        [
            [2, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 5, 0],
            [0, 3, 3, 0, 0, 0],
            [0, 3, 3, 0, 0, 0],
            [0, 5, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int64,
    )

    scene = parse_grid(after, previous_grid=before, action="probe", time_index=11)
    candidates_a = propose_coordinate_candidates(scene, max_candidates=16)
    candidates_b = propose_coordinate_candidates(scene, max_candidates=16)

    assert candidates_a == candidates_b
    assert len(candidates_a) == 16
    assert len(candidates_a) <= 16
    assert len(candidates_a) < 6 * 6

    seen_reasons: list[str] = []
    original_add = coordinate_affordance._add_candidate

    def track_add_candidate(
        candidates: dict[tuple[int, int], object],
        y: int,
        x: int,
        *,
        reason: str,
        score: float,
        shape: tuple[int, int],
    ) -> None:
        seen_reasons.append(reason)
        original_add(candidates, y, x, reason=reason, score=score, shape=shape)

    coordinate_affordance._add_candidate = track_add_candidate
    try:
        propose_coordinate_candidates(scene, max_candidates=16)
    finally:
        coordinate_affordance._add_candidate = original_add

    required_reasons = {
        "object_center",
        "object_corner",
        "object_boundary",
        "adjacent_empty",
        "repeated_pattern",
        "high_salience",
    }
    assert required_reasons.issubset(set(seen_reasons))


def test_propose_coordinate_candidates_large_grid_is_sparse_and_truncated() -> None:
    before, after = _build_sparse_64_grid()
    scene = parse_grid(after, previous_grid=before, action="click", time_index=4)

    sparse = propose_coordinate_candidates(scene, max_candidates=8)
    all_candidates = propose_coordinate_candidates(scene, max_candidates=4096)

    assert len(sparse) == 8
    assert len(sparse) < 64 * 64
    assert len(all_candidates) > len(sparse)
    assert sparse == all_candidates[:8]


@pytest.mark.parametrize("coordinate_key,coordinate", [("action_coordinate", [1, 1]), ("coordinate_action", {"y": 1, "x": 1})])
def test_append_event_populates_coordinate_affordances_via_metadata_key(
    coordinate_key: str,
    coordinate: object,
    tmp_path,
) -> None:
    before = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=torch.long,
    )
    after = torch.tensor(
        [[0, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
        dtype=torch.long,
    )

    memory = PersistentMemoryState.fresh(hidden_dim=8, batch_size=1, device="cpu")
    memory.append_event(
        tick=1,
        observation={"grid": before},
        action=1,
        next_observation={"grid": after},
        metadata={coordinate_key: coordinate},
    )

    assert memory.coordinate_affordances["schema"] == "runtime_coordinate_affordances_v1"
    assert len(memory.coordinate_affordances["candidate_history"]) == 1
    assert memory.coordinate_affordances["updates"] == 1

    model_payload = memory.coordinate_affordances["models"]["4x4"]
    assert model_payload["shape"] == [4, 4]
    assert model_payload["trials"]["1,1"] == 1
    assert model_payload["changed"]["1,1"] == 1
    assert model_payload["no_op"]["1,1"] == 0

    path = tmp_path / "coordinate_affordance_state.pt"
    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=8, batch_size=1, device="cpu")
    assert loaded.coordinate_affordances == memory.coordinate_affordances
