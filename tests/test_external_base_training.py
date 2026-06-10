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
from src.base_world_model import ExternalBaseConfig
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


def test_external_base_no_hack_source_scan_passes() -> None:
    assert no_hack_proof()["passes"] is True
