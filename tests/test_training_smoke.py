from pathlib import Path

from src.evaluate import evaluate_checkpoint
from src.train import train_model


def test_training_smoke_creates_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "smoke.pt"
    summary = train_model("smoke", output=checkpoint, steps=2)
    assert checkpoint.exists()
    assert summary["loss_total"] > 0.0


def test_smoke_checkpoint_evaluates(tmp_path: Path) -> None:
    checkpoint = tmp_path / "smoke_eval.pt"
    train_model("smoke", output=checkpoint, steps=1)
    metrics = evaluate_checkpoint(checkpoint, config_name="smoke", runtime_ticks=64)
    assert metrics["unbroken_ticks"] == 64.0
    assert "goal_action_success" in metrics
