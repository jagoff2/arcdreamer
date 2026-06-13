from pathlib import Path

import torch

from src.env import TinyWorldRuntime
from src.model import RecurrentLatentModel, load_checkpoint, save_checkpoint
from src.run_unbroken import run_unbroken
from src.train import train_model


def test_unbroken_runtime_keeps_ticking_and_latent_changes(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runtime.pt"
    train_model("smoke", output=checkpoint, steps=1)
    stats = run_unbroken(checkpoint, max_ticks=128, log_every=0)
    assert stats["unbroken_ticks"] == 128.0
    assert stats["latent_active_fraction"] > 0.0
    assert stats["latent_max_quantized_fraction"] < 1.0


def test_runtime_reuses_previous_latent_each_tick(tmp_path: Path) -> None:
    checkpoint = tmp_path / "reuse.pt"
    train_model("smoke", output=checkpoint, steps=1)
    model = load_checkpoint(checkpoint)
    world = TinyWorldRuntime(seed=123)
    z = model.initial_state(1)
    norms = []

    with torch.no_grad():
        for _ in range(32):
            previous = z
            observation = world.observation()
            output, z = model.step(observation, previous)
            assert z.shape == previous.shape
            assert z.data_ptr() != previous.data_ptr()
            norms.append(float((z - previous).norm().item()))
            world.step(int(output["action_logits"].argmax(dim=-1).item()))

    assert sum(value > 1e-6 for value in norms) == len(norms)


def test_run_unbroken_persists_neural_program_proposals(tmp_path: Path) -> None:
    checkpoint = tmp_path / "proposal_runtime.pt"
    memory_file = tmp_path / "proposal_memory.pt"
    save_checkpoint(checkpoint, RecurrentLatentModel(device="cpu"), train_config={})

    stats = run_unbroken(
        checkpoint,
        max_ticks=2,
        log_every=0,
        device="cpu",
        memory_file=memory_file,
    )
    payload = torch.load(memory_file, map_location="cpu")
    event = payload["event_journal"][0]
    proposals = event["prediction_error"]["neural_program_proposals"]

    assert stats["event_journal_length"] == 2.0
    assert proposals
    assert proposals[0]["schema"] == "neural_causal_program_proposal_v1"
    assert proposals[0]["source"] == "neural_proposal"
    assert payload["hypothesis_posterior"]["hypotheses"]
