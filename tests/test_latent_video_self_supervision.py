from __future__ import annotations

import torch

from src.video_jepa import (
    VideoJEPA,
    jepa_loss,
    jepa_self_supervision_manifest,
    synthetic_latent_video_batch,
    train_local_latent_jepa_smoke,
)


def _assert_batch_compatibility(batch: dict[str, torch.Tensor], config: VideoJEPA) -> None:
    assert torch.is_tensor(batch["frames"])
    assert torch.is_tensor(batch["action_ids"])
    assert torch.is_tensor(batch["legal_counts"])
    assert torch.is_tensor(batch["valid"])
    assert torch.is_tensor(batch["action_features"])

    batch_size = batch["frames"].shape[0]
    seq_len = batch["frames"].shape[1]
    frame_dim = batch["frames"].shape[2]
    latent = config.encode_frames(batch["frames"][:, :1].flatten(0, 1)).reshape(batch_size, 1, -1)

    output = config(
        batch["frames"],
        batch["action_ids"],
        batch["legal_counts"],
        batch["valid"],
        batch["action_features"],
    )
    assert output["predicted_future"].shape[:2] == (batch_size, seq_len - 1)
    assert output["target_future"].shape == output["predicted_future"].shape
    assert output["predicted_future"].shape[-1] == latent.shape[-1]


def test_synthetic_latent_video_batch_is_deterministic_and_compatible() -> None:
    batch_a = synthetic_latent_video_batch(
        batch_size=4,
        seq_len=9,
        seed=2026,
        device="cpu",
    )
    batch_b = synthetic_latent_video_batch(
        batch_size=4,
        seq_len=9,
        seed=2026,
        device="cpu",
    )

    for key in ("frames", "action_ids", "legal_counts", "valid", "action_features"):
        assert torch.equal(batch_a[key], batch_b[key])

    model = VideoJEPA()
    _assert_batch_compatibility(batch_a, model)
    loss = jepa_loss(
        model(
            batch_a["frames"],
            batch_a["action_ids"],
            batch_a["legal_counts"],
            batch_a["valid"],
            batch_a["action_features"],
        )
    )
    assert torch.isfinite(loss)


def test_train_local_latent_jepa_smoke_flags_local_and_improving() -> None:
    _, metrics = train_local_latent_jepa_smoke(
        steps=6,
        batch_size=6,
        seq_len=12,
        seed=2026,
        device="cpu",
    )

    assert isinstance(metrics["initial_loss"], float)
    assert isinstance(metrics["final_loss"], float)
    assert isinstance(metrics["null_loss"], float)
    assert metrics["initial_loss"] > 0.0
    assert torch.isfinite(torch.tensor(metrics["final_loss"]))
    assert torch.isfinite(torch.tensor(metrics["null_loss"]))
    assert metrics["uses_pretrained_backbone"] is False
    assert metrics["emits_text"] is False
    assert metrics["predicts_actions"] is False
    assert metrics["loss_improved"] or metrics["jepa_beats_null"]


def test_jepa_self_supervision_manifest_is_local_future_latent_and_non_action_text() -> None:
    _, metrics = train_local_latent_jepa_smoke(steps=4, batch_size=3, seq_len=8, seed=77, device="cpu")
    manifest = jepa_self_supervision_manifest(model=VideoJEPA(), metrics=metrics)

    assert manifest["format"] == "local_latent_video_jepa_self_supervision_v1"
    assert manifest["training_source"] == "synthetic_public_frame_sequences"
    assert manifest["initialization"] == "local_random_init"
    assert manifest["uses_pretrained_backbone"] is False
    assert manifest["predicts"] == "future latent visual regions"
    assert manifest["emits_text"] is False
    assert "actions" in manifest["does_not_predict"]
    assert "text" in manifest["does_not_predict"]
    assert "labels" in manifest["does_not_predict"]
