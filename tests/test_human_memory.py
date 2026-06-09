from __future__ import annotations

from pathlib import Path

from src.heldout_causal import FROZEN_CHECKPOINT
from src.human_memory import HumanAnalogueMemory, HumanMemoryConfig, deterministic_vector
from src.memory_eval import evaluate_human_memory


def test_sparse_engram_memory_completes_partial_cue(tmp_path: Path) -> None:
    memory = HumanAnalogueMemory(HumanMemoryConfig())
    content = deterministic_vector(1, memory.config.content_dim)
    cue = deterministic_vector(2, memory.config.cue_dim)
    latent = deterministic_vector(3, memory.config.latent_dim)
    body = deterministic_vector(4, memory.config.body_dim)
    affect = deterministic_vector(5, memory.config.affect_dim)
    action = deterministic_vector(6, memory.config.action_dim)
    private = deterministic_vector(7, memory.config.private_dim)
    memory.write(cue, latent, body, affect, action, 0, private, content, 0.0)
    partial = cue.clone()
    partial[::3] = 0.0
    recalled = memory.recall(partial)
    assert recalled.accepted
    assert float((recalled.content * content).sum().item()) > 0.85
    path = tmp_path / "human_memory.pt"
    memory.save(path)
    loaded = HumanAnalogueMemory.load(path)
    assert loaded.recall(partial).accepted


def test_human_memory_smoke_report_passes(tmp_path: Path) -> None:
    report = evaluate_human_memory(
        FROZEN_CHECKPOINT,
        config_name="smoke",
        json_output=tmp_path / "human_memory.json",
    )
    assert report["terminal_outcome"] == "HUMAN MEMORY PROVEN"
    assert report["content_recall"]["partial_cue_accuracy"] >= 0.85
    assert report["causal_memory"]["targeted_trace_corruption_recall_action_degrade"] >= 0.40
    assert report["prior_properties"]["leakage_scan_passes"] is True
