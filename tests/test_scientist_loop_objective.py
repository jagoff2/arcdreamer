from __future__ import annotations

from pathlib import Path

import torch

from src.curriculum import generate_scientist_curriculum_batch
from src.model import RecurrentLatentModel
from src.train import SCIENTIST_LOOP_STAGES, compute_losses, train_model


def test_scientist_loop_stages_and_compute_losses_are_valid() -> None:
    torch.manual_seed(1234)
    batch = generate_scientist_curriculum_batch(
        batch_size=4,
        seq_len=16,
        base_seed=17,
        device="cpu",
    )
    model = RecurrentLatentModel(device="cpu")
    outputs = model(
        batch["sensory"],
        batch["lang_in"],
        batch["private_in"],
        batch["prev_action"],
        batch["prev_delta"],
        batch["dt"],
    )
    losses = compute_losses(outputs, batch)

    assert SCIENTIST_LOOP_STAGES == (
        "perceive",
        "perturb",
        "infer",
        "compress",
        "plan",
        "test",
        "consolidate",
    )

    for stage in SCIENTIST_LOOP_STAGES:
        key = f"loop_{stage}"
        value = losses[key]
        assert isinstance(value, torch.Tensor)
        assert value.ndim == 0
        assert torch.isfinite(value).all()
        assert (value >= 0).all()


def test_train_model_smoke_reports_scientist_loop_losses(tmp_path: Path) -> None:
    checkpoint = tmp_path / "scientist_loop_smoke.pt"
    summary = train_model("smoke", steps=1, output=checkpoint)

    assert checkpoint.exists()
    assert summary["scientist_loop_stage_count"] == 7.0
    for stage in SCIENTIST_LOOP_STAGES:
        assert f"loss_loop_{stage}" in summary
