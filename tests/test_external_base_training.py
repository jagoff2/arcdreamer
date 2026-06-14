from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import torch

from src.arcagi3_adapter import ArcAGI3Observation
from src.base_eval import BaseArmSpec, ExternalBaseController
from src.base_pretrain import train_arm, train_external_base
from src.base_retrain_experiment import no_hack_proof
from src.base_world_model import (
    ExternalBaseConfig,
    ExternalBaseWorldModel,
    action_family,
    action_hash_features,
    action_to_features,
    grid_to_fixed,
)
from src.model import load_checkpoint
from src.trace_collect import generated_arc_like_arrays


def test_generated_trace_arrays_have_required_schema() -> None:
    arrays = generated_arc_like_arrays(64, seed=11)
    expected = {"obs", "next_obs", "action_features", "action_id", "family", "legal_mask", "reward", "terminal", "source_id"}
    assert expected.issubset(arrays)
    assert arrays["obs"].shape == (64, 64)
    assert arrays["next_obs"].shape == (64, 64)
    assert arrays["action_features"].shape == (64, 16)
    assert arrays["legal_mask"].shape == (64, 8)
    assert np.any(np.abs(arrays["next_obs"] - arrays["obs"]).sum(axis=1) > 0.0)


def test_fixed_grid_projection_preserves_large_public_frame_content() -> None:
    grid = np.zeros((64, 64), dtype=np.int64)
    grid[32, 32] = 2
    grid[40, 40] = 7
    grid[63, 63] = 5
    fixed = grid_to_fixed(grid)
    assert fixed.shape == (8, 8)
    assert fixed.sum() > 0
    assert fixed[4, 4] == 2
    assert fixed[5, 5] == 7
    assert fixed[7, 7] == 5


def test_fixed_grid_projection_keeps_agent_over_goal_cluster() -> None:
    grid = np.zeros((64, 64), dtype=np.int64)
    grid[32:36, 32:36] = 5
    grid[32, 32] = 2
    fixed = grid_to_fixed(grid)
    assert fixed[4, 4] == 2


def test_action_features_use_actual_public_grid_coordinates() -> None:
    observation = ArcAGI3Observation(
        task_id="external/test",
        episode_id="episode",
        step_index=0,
        grid=np.zeros((64, 64), dtype=np.int64),
        available_actions=("click:63:31",),
        extras={},
    )
    features = action_to_features(observation, "click:63:31", legal_index=0, legal_count=1)
    assert features[1] == 1.0
    assert np.isclose(features[5], 31.0 / 63.0)
    assert np.isclose(features[6], 1.0)


def test_official_numeric_action_features_match_runtime_families() -> None:
    observation = ArcAGI3Observation(
        task_id="external/test",
        episode_id="episode",
        step_index=0,
        grid=np.zeros((8, 8), dtype=np.int64),
        available_actions=("1", "2", "3", "4", "5", "7"),
        extras={},
    )
    assert action_family("1") == "move"
    assert action_family("2") == "move"
    assert action_family("3") == "move"
    assert action_family("4") == "move"
    assert action_family("5") == "other"
    assert action_family("7") == "wait"
    assert action_to_features(observation, "1", legal_index=0, legal_count=6)[0] == 1.0
    assert action_to_features(observation, "5", legal_index=4, legal_count=6)[4] == 1.0
    assert action_to_features(observation, "7", legal_index=5, legal_count=6)[2] == 1.0


def test_external_base_affordance_diagnostic_uses_sigmoid_scale() -> None:
    observation = ArcAGI3Observation(
        task_id="external/test",
        episode_id="episode",
        step_index=0,
        grid=np.zeros((8, 8), dtype=np.int64),
        available_actions=("1", "5", "click:2:2", "7"),
        extras={},
    )
    model = ExternalBaseWorldModel(ExternalBaseConfig(hidden_dim=32, latent_dim=16), device="cpu")
    _, diagnostics, _ = model.score_actions(observation, observation.available_actions)

    assert all(0.0 <= row["affordance"] <= 1.0 for row in diagnostics.values())
    assert all(0.0 <= row["progress"] <= 1.0 for row in diagnostics.values())
    assert all(0.0 <= row["future_progress"] <= 1.0 for row in diagnostics.values())
    assert all(-1.0 <= row["action_prior"] <= 1.0 for row in diagnostics.values())


def test_generated_pretraining_balances_clicks_with_controls() -> None:
    arrays = generated_arc_like_arrays(8192, seed=14)
    action_id = arrays["action_id"]
    reward = arrays["reward"]
    move_positive = np.mean(reward[np.isin(action_id, [0, 1, 2, 3])] > 0.0)
    use_positive = np.mean(reward[action_id == 4] > 0.0)
    target_click_positive = np.mean(reward[action_id == 5] > 0.0)
    decoy_click_positive = np.mean(reward[action_id == 6] > 0.0)
    assert move_positive > 0.20
    assert use_positive > 0.02
    assert 0.10 < target_click_positive < 0.45
    assert decoy_click_positive == 0.0
    assert np.allclose(arrays["action_features"][action_id == 0][0, 10:], action_hash_features("1"))
    assert np.allclose(arrays["action_features"][action_id == 4][0, 10:], action_hash_features("5"))
    assert np.allclose(arrays["action_features"][action_id == 7][0, 10:], action_hash_features("7"))


def test_generated_pretraining_contains_viewport_navigation_value() -> None:
    arrays = generated_arc_like_arrays(16384, seed=19)
    obs = np.rint(arrays["obs"].reshape(-1, 8, 8) * 8.0).astype(np.int64)
    next_obs = np.rint(arrays["next_obs"].reshape(-1, 8, 8) * 8.0).astype(np.int64)
    action_id = arrays["action_id"]
    move = np.isin(action_id, [0, 1, 2, 3])
    centered_agent = (obs[:, 4, 4] == 2) & (next_obs[:, 4, 4] == 2)
    visible_shift = np.any(obs != next_obs, axis=(1, 2))
    positive_navigation = move & centered_agent & visible_shift & (arrays["reward"] > 0.0)
    assert np.mean(positive_navigation) > 0.02


def test_train_arm_reports_all_self_supervised_objectives() -> None:
    arrays = generated_arc_like_arrays(256, seed=12)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    _, metrics = train_arm(
        arm="from_scratch_external_base",
        arrays=arrays,
        config=ExternalBaseConfig(hidden_dim=32, latent_dim=24),
        steps=4,
        batch_size=32,
        device=device,
        seed=101,
    )
    required = {
        "next_observation",
        "change_mask",
        "reward_event_noop",
        "progress_event",
        "future_progress",
        "action_policy_prior",
        "inverse_dynamics",
        "action_affordance",
        "temporal_object_persistence",
        "latent_rollout_consistency",
        "memory_utility",
    }
    assert required.issubset(metrics["final_losses"])
    assert math.isfinite(metrics["final_losses"]["total"])
    assert metrics["training_losses_are_diagnostic_only"] is True


def test_external_base_checkpoint_remains_recurrent_loader_compatible(tmp_path: Path) -> None:
    arrays = generated_arc_like_arrays(96, seed=13)
    data_path = tmp_path / "trace_tensors.npz"
    np.savez_compressed(data_path, **arrays)
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "data_path": str(data_path),
                "data_sha256": "test",
                "transition_count": 96,
                "source_counts": {"generated_arc_like_pretrain": 96},
            }
        ),
        encoding="utf-8",
    )
    checkpoint = tmp_path / "external_base_v1.pt"
    manifest_out = tmp_path / "external_base_manifest_v1.json"
    train_external_base(
        manifest_path=manifest_path,
        checkpoint_output=checkpoint,
        manifest_output=manifest_out,
        recurrent_checkpoint="frozen/recurrent_latent_fast.pt",
        steps=1,
        batch_size=16,
        device="cuda" if torch.cuda.is_available() else "cpu",
    )
    loaded = load_checkpoint(checkpoint, device="cpu")
    assert loaded.config.hidden_dim > 0
    payload = torch.load(checkpoint, map_location="cpu")
    assert payload["external_base_format"] == "external_action_conditioned_world_model_v1"
    assert set(payload["external_arms"]) >= {
        "old_base_world_model",
        "old_base_finetuned",
        "from_scratch_external_base",
        "null_training_control",
    }


class DummyAdapter:
    def reset(self) -> None:
        return None

    def choose_action(self, observation: ArcAGI3Observation):
        first, second = observation.available_actions[:2]
        return (
            first,
            {
                "action_scores": {first: 1.0, second: 0.95},
                "policy": {},
                "memory_recall": {},
                "drive": {},
                "hypothesis_state": {},
            },
        )

    def observe_transition(self, action, result) -> None:
        return None


class RewardingFakeModel:
    def score_actions(self, observation, legal_actions, **kwargs):
        del observation, kwargs
        scores = {action: 0.0 for action in legal_actions}
        scores[legal_actions[1]] = 1.0
        diagnostics = {action: {"score": score} for action, score in scores.items()}
        return scores, diagnostics, torch.zeros(1, 4)


class MemoryFakeModel:
    def score_actions(self, observation, legal_actions, **kwargs):
        del observation, kwargs
        scores = {action: float(index * 2) for index, action in enumerate(legal_actions)}
        diagnostics = {action: {"score": score} for action, score in scores.items()}
        memories = torch.arange(len(legal_actions), dtype=torch.float32).view(-1, 1).repeat(1, 4)
        return scores, diagnostics, memories


def test_external_base_controller_can_change_action_without_forced_cycle() -> None:
    observation = ArcAGI3Observation(
        task_id="external/test",
        episode_id="episode",
        step_index=0,
        grid=np.zeros((2, 2), dtype=np.int64),
        available_actions=("1", "2"),
        extras={},
    )
    controller = ExternalBaseController(
        DummyAdapter(),
        BaseArmSpec("from_scratch_external_base", checkpoint_arm="from_scratch_external_base", uses_external_base=True),
        model=RewardingFakeModel(),
    )
    action, diagnostics = controller.choose_action(observation)
    assert action == "2"
    assert diagnostics["policy"]["changed_action"] is True
    assert diagnostics["policy"]["forced_cycle"] is False
    assert diagnostics["adapter_action_scores"]["1"] > diagnostics["adapter_action_scores"]["2"]
    assert diagnostics["action_scores"]["2"] > diagnostics["action_scores"]["1"]


def test_external_base_memory_commits_chosen_candidate_state() -> None:
    observation = ArcAGI3Observation(
        task_id="external/test",
        episode_id="episode",
        step_index=0,
        grid=np.zeros((2, 2), dtype=np.int64),
        available_actions=("1", "2", "3"),
        extras={},
    )
    controller = ExternalBaseController(
        DummyAdapter(),
        BaseArmSpec("from_scratch_external_base", checkpoint_arm="from_scratch_external_base", uses_external_base=True),
        model=MemoryFakeModel(),
    )
    action, _ = controller.choose_action(observation)
    assert action == "3"
    assert controller.memory is not None
    assert torch.allclose(controller.memory, torch.full((1, 4), 2.0))


def test_external_base_no_hack_source_scan_passes() -> None:
    assert no_hack_proof()["passes"] is True
