from __future__ import annotations

from pathlib import Path

from arcagi.envs.synthetic import DEFAULT_SYNTHETIC_FAMILY_MODES, family_variants_for_mode
from arcagi.evaluation.harness import evaluate_synthetic
from arcagi.training.synthetic import SyntheticTrainingConfig, train_synthetic


def test_train_and_eval_smoke(tmp_path: Path) -> None:
    checkpoint = tmp_path / "synthetic.pt"
    metrics = train_synthetic(
        SyntheticTrainingConfig(
            epochs=1,
            episodes_per_epoch=4,
            checkpoint_path=str(checkpoint),
            seed=11,
            holdout_eval_every_epochs=0,
        )
    )
    assert checkpoint.exists()
    assert metrics["loss_last_epoch"] >= 0.0

    result = evaluate_synthetic(
        agent_name="hybrid",
        checkpoint_path=str(checkpoint),
        episodes_per_family=1,
        seed=19,
    )
    assert "success_rate" in result
    expected_family_count = sum(len(family_variants_for_mode(mode)) for mode in DEFAULT_SYNTHETIC_FAMILY_MODES)
    assert len(result["families"]) == expected_family_count
