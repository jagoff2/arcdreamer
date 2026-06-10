from __future__ import annotations

from pathlib import Path

import torch

from src.explore_env import (
    NUM_EXPLORER_ACTIONS,
    NUM_MIND_ACTS,
    OBS_DIM,
    build_explorer_dataset,
)
from src.explorer_eval import evaluate_explorer
from src.explorer_train import train_explorer
from src.heldout_causal import FROZEN_CHECKPOINT
from src.world_model import ExplorerCore, explorer_forward


def test_explorer_dataset_has_required_open_world_masks() -> None:
    data = build_explorer_dataset("smoke", "heldout", record_count=256, checkpoint=FROZEN_CHECKPOINT)
    tensors = data.tensors
    assert tensors["obs"].shape[-1] == OBS_DIM
    assert tensors["z"].shape[-1] == 64
    assert tensors["informative_mask"].any()
    assert tensors["noise_mask"].any()
    assert tensors["planning_mask"].any()
    assert tensors["counterfactual_mask"].any()
    assert tensors["transfer_mask"].any()
    assert tensors["restart_mask"].any()
    assert tensors["social_uncertainty_mask"].any()
    assert tensors["conflict_mask"].any()
    assert tensors["partner_restart_mask"].any()
    assert set(data.metadata["open_world_features"]) >= {
        "objects",
        "tools",
        "doors",
        "hazards",
        "resources",
        "agents",
        "changing_rules",
    }


def test_explorer_core_forward_shapes_and_no_target_inputs() -> None:
    data = build_explorer_dataset("smoke", "heldout", record_count=64, checkpoint=FROZEN_CHECKPOINT)
    model = ExplorerCore()
    tensors = data.tensors
    with torch.no_grad():
        out = explorer_forward(model, tensors)
        no_skill = explorer_forward(model, tensors, skill_enabled=False)
        no_intrinsic = explorer_forward(model, tensors, intrinsic_enabled=False)
    assert out["action_logits"].shape == (64, NUM_EXPLORER_ACTIONS)
    assert out["mind_logits"].shape == (64, NUM_MIND_ACTS)
    assert not torch.allclose(out["skill_logits"], no_skill["skill_logits"])
    assert not torch.allclose(out["next_state_logits"], no_intrinsic["next_state_logits"])
    target_keys = {key for key in tensors if key.endswith("_target")}
    assert target_keys
    assert target_keys.isdisjoint({"obs", "z", "hypothesis", "skill_memory", "project_state", "social_state", "intrinsic"})
    assert {"input_texts", "target_texts", "conversation_history", "transcript"}.isdisjoint(set(vars(model)))


def test_explorer_smoke_train_and_eval_runs(tmp_path: Path) -> None:
    output = tmp_path / "explorer_smoke.pt"
    summary = train_explorer("smoke", output=output, checkpoint=FROZEN_CHECKPOINT)
    assert output.exists()
    assert summary["dataset"]["records"] > 0

    report = evaluate_explorer(
        FROZEN_CHECKPOINT,
        output,
        config_name="smoke",
        json_output=tmp_path / "explorer_report.json",
        include_prior=False,
    )
    assert report["config"] == "smoke"
    assert "exploration" in report
    assert "mindlike" in report
    assert report["gate_checks"]["hidden_target_canary_zero"] is True
    assert Path(tmp_path / "explorer_report.json").exists()
