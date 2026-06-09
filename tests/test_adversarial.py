from pathlib import Path

from src.adversarial import evaluate_adversarial
from src.train import train_model


def test_adversarial_smoke_report_contains_required_sections(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adv.pt"
    train_model("smoke", output=checkpoint, steps=1)
    report = evaluate_adversarial(str(checkpoint), config_name="smoke")
    for key in [
        "counterbalanced_provenance",
        "latent_ablations",
        "baselines",
        "language_causal",
        "false_belief_conflict",
        "blank_input_continuation",
        "multi_event_episodic_probe",
        "ood",
        "terminal_a_adversarial_verdict",
    ]:
        assert key in report


def test_adversarial_report_includes_required_baselines(tmp_path: Path) -> None:
    checkpoint = tmp_path / "adv_baselines.pt"
    train_model("smoke", output=checkpoint, steps=1)
    report = evaluate_adversarial(str(checkpoint), config_name="smoke")
    baselines = report["baselines"]
    for key in ["feedforward_only", "shuffled_z", "no_language", "no_provenance", "no_occlusion"]:
        assert key in baselines
    for key in ["freeze_z", "shuffle_z", "zero_z", "perturb_z"]:
        assert key in report["latent_ablations"]
