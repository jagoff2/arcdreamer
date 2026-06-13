from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from src.arcagi3_adapter import ARCAGI3Adapter, ArcAGI3FixtureEnv, load_public_fixtures
from src.arcagi3_baselines import build_baselines
from src.arcagi3_eval import no_hack_audit, run_episode
from src.arcagi3_official import official_frame_to_grid
from src.explore_env import HYPOTHESIS_DIM, OBS_DIM, Z_DIM
from src.head_collapse import HeadCollapsedExplorer
from src.world_model import ExplorerCore


def test_public_fixtures_load_and_env_contract() -> None:
    fixtures = load_public_fixtures()
    assert len(fixtures) >= 3
    env = ArcAGI3FixtureEnv(fixtures[0], seed=5)
    obs = env.reset(seed=5)
    assert obs.grid.ndim == 2
    assert obs.available_actions
    result = env.step(obs.available_actions[0])
    assert result.observation.step_index == 1
    assert isinstance(result.reward, float)
    assert isinstance(result.info, dict)


def test_adapter_tensorizes_arc_observation_without_text_state() -> None:
    fixture = load_public_fixtures()[0]
    env = ArcAGI3FixtureEnv(fixture, seed=7)
    obs = env.reset(seed=7)
    explorer = HeadCollapsedExplorer(ExplorerCore())
    adapter = ARCAGI3Adapter(explorer)
    tensors, state = adapter.tensorize(obs)
    assert tensors["obs"].shape[-1] == OBS_DIM
    assert tensors["z"].shape[-1] == Z_DIM
    assert tensors["hypothesis"].shape[-1] == HYPOTHESIS_DIM
    assert "target_texts" not in tensors
    assert state["hypothesis_state"]["semantic_action"] == state["semantic_action"]


def test_official_frame_grid_preserves_raw_public_foreground_buckets() -> None:
    class Response:
        frame = [
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 11, 11, 0],
                    [0, 22, 22, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int64,
            )
        ]

    grid, stats = official_frame_to_grid(Response(), legal_ids=(1, 6), max_click_actions=16)
    foreground = set(int(value) for value in grid.reshape(-1) if int(value) != 0)
    assert 2 in foreground
    assert len(foreground) >= 3
    assert stats["raw_value_bucket_count"] == 2


def test_baselines_and_trace_run_on_fixture(tmp_path: Path) -> None:
    fixture = load_public_fixtures()[0]
    baseline = build_baselines()[0]
    env = ArcAGI3FixtureEnv(fixture, seed=3)
    row = run_episode(env, baseline, seed=3, trace_path=tmp_path / "trace.json", controller_name=baseline.name)
    assert 0.0 <= row["normalized_score"] <= 1.0
    assert Path(row["trace_path"]).exists()


def test_no_hack_audit_passes_for_arc_sources() -> None:
    audit = no_hack_audit()
    assert audit["passes"] is True
    assert audit["findings"] == []
