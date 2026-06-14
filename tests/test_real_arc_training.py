from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from src.real_arc_train import build_real_arc_splits


def _write_trace(path: Path, game_id: str, action: str = "1") -> None:
    payload = {
        "metadata": {
            "suite_id": "official_arcagi3",
            "task_id": f"arcagi3-official/{game_id}",
            "variant": "fixture",
            "split": "sealed_eval",
            "seed": 0,
            "attempt_index": 1,
        },
        "attempt": {
            "suite_id": "official_arcagi3",
            "task_id": f"arcagi3-official/{game_id}",
            "variant": "fixture",
            "split": "sealed_eval",
            "seed": 0,
            "attempt_index": 1,
            "steps": [
                {
                    "step_index": 0,
                    "frame": [[0, 2], [0, 7]],
                    "next_frame": [[0, 0], [2, 7]],
                    "action": action,
                    "legal_actions": ["1", "2", "7"],
                    "score_delta": 0.25,
                    "event_delta": ["level_completed"],
                    "terminal": False,
                    "obs_hash": "a",
                    "next_obs_hash": "b",
                }
            ],
        },
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_real_arc_split_excludes_holdout_games_before_tensorization(tmp_path: Path) -> None:
    game_manifest = tmp_path / "games.json"
    game_manifest.write_text(
        json.dumps(
            {
                "games": [
                    {"game_id": "train-aa"},
                    {"game_id": "train-bb"},
                    {"game_id": "holdout-cc"},
                ]
            }
        ),
        encoding="utf-8",
    )
    root = tmp_path / "traces"
    _write_trace(root / "official_arcagi3" / "sealed_eval" / "x" / "train-aa.json", "train-aa")
    _write_trace(root / "official_arcagi3" / "sealed_eval" / "x" / "train-bb.json", "train-bb", action="2")
    _write_trace(root / "official_arcagi3" / "sealed_eval" / "x" / "holdout-cc.json", "holdout-cc")

    train_arrays, holdout_arrays, manifest = build_real_arc_splits(
        roots=[root],
        game_manifest=game_manifest,
        holdout_game_ids=["holdout-cc"],
        train_game_ids=None,
        max_traces_per_game=4,
        train_data_output=tmp_path / "train.npz",
        holdout_data_output=tmp_path / "holdout.npz",
        manifest_output=tmp_path / "manifest.json",
    )

    assert manifest["train_game_ids"] == ["train-aa", "train-bb"]
    assert manifest["holdout_game_ids"] == ["holdout-cc"]
    assert manifest["rules"]["split_by_game_id_before_tensorization"] is True
    assert manifest["rules"]["official_arcagi3_used_for_training"] is True
    assert manifest["rules"]["holdout_official_arcagi3_used_for_training"] is False
    assert train_arrays["obs"].shape[0] == 2
    assert holdout_arrays["obs"].shape[0] == 1
    assert np.all(train_arrays["source_id"] == 10)
