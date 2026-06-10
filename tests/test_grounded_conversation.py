from __future__ import annotations

from pathlib import Path

import torch

from src.conversation_env import CONVERSATION_CONFIGS, CONV_LABELS, build_conversation_dataset, conversation_config
from src.conversation_eval import evaluate_conversation
from src.conversation_train import train_conversation
from src.free_text_decoder import FreeTextConversationDecoder, tokenizers
from src.heldout_causal import FROZEN_CHECKPOINT


def test_free_text_decoder_forward_and_generation() -> None:
    data = build_conversation_dataset(FROZEN_CHECKPOINT, "smoke", "heldout", session_count=2)
    cfg = conversation_config()
    model = FreeTextConversationDecoder(cfg)
    tensors = data.tensors
    with torch.no_grad():
        out = model(
            tensors["input_ids"],
            tensors["word_ids"],
            tensors["z"],
            tensors["memory"],
            tensors["private_tokens"],
            tensors["turn_ids"],
            tensors["target_ids"],
        )
    assert out["answer_logits"].shape[-1] == len(CONV_LABELS)
    assert out["updated_z"].shape == tensors["z"].shape
    assert out["updated_memory"].shape == tensors["memory"].shape
    assert not torch.allclose(out["updated_z"], tensors["z"])
    tokenizer, _ = tokenizers(cfg)
    assert model.generate_text(tokenizer, out)
    assert {"input_texts", "target_texts", "conversation_history", "transcript"}.isdisjoint(set(vars(model)))


def test_conversation_dataset_has_long_sessions_and_split_separation() -> None:
    train = build_conversation_dataset(FROZEN_CHECKPOINT, "smoke", "train", session_count=4)
    heldout = build_conversation_dataset(FROZEN_CHECKPOINT, "smoke", "heldout", session_count=4)
    assert train.tensors["turn_ids"].max().item() == CONVERSATION_CONFIGS["smoke"].turns - 1
    assert train.synthetic_tokens > 0
    assert set(train.forms).isdisjoint(set(heldout.forms))
    assert train.tensors["requires_memory"].any()
    assert train.tensors["requires_z"].any()
    assert train.tensors["requires_private"].any()
    assert train.tensors["is_refusal"].any()
    assert train.tensors["is_silence"].any()


def test_conversation_smoke_train_and_eval_runs(tmp_path: Path) -> None:
    output = tmp_path / "conversation_smoke.pt"
    summary = train_conversation("smoke", output=output, checkpoint=FROZEN_CHECKPOINT)
    assert output.exists()
    assert summary["records"] == CONVERSATION_CONFIGS["smoke"].train_sessions * CONVERSATION_CONFIGS["smoke"].turns

    report = evaluate_conversation(
        FROZEN_CHECKPOINT,
        "runs/dialogue_tiny.pt",
        output,
        config_name="smoke",
        json_output=tmp_path / "conversation_report.json",
        include_prior=False,
    )
    assert report["config"] == "smoke"
    assert report["gate_checks"]["copy_rate_le_0_03"] is True
    assert report["gate_checks"]["random_labels_no_effect"] is True
    assert Path(tmp_path / "conversation_report.json").exists()
