from __future__ import annotations

from pathlib import Path

import torch

from src.dialogue_env import ANSWER_LABELS, DIALOGUE_CONFIGS, answer_config, build_dialogue_dataset
from src.dialogue_eval import evaluate_dialogue
from src.dialogue_train import train_dialogue
from src.heldout_causal import FROZEN_CHECKPOINT
from src.language_organ import DialogueOrgan, TinyCharTokenizer


def test_tiny_char_tokenizer_is_local_and_round_trips() -> None:
    tokenizer = TinyCharTokenizer()
    text = "trace 7: color red."
    encoded = tokenizer.encode(text, max_len=32)
    assert encoded.shape == (32,)
    assert tokenizer.decode(encoded) == text
    assert tokenizer.pad_id == 0


def test_dialogue_organ_updates_tensor_state_without_text_state() -> None:
    data = build_dialogue_dataset(FROZEN_CHECKPOINT, "smoke", "heldout", count=16)
    organ = DialogueOrgan(answer_config())
    tensors = data.tensors
    with torch.no_grad():
        out = organ(tensors["input_ids"], tensors["z"], tensors["memory"], tensors["private_tokens"])

    assert out["answer_logits"].shape == (16, len(ANSWER_LABELS))
    assert out["updated_z"].shape == tensors["z"].shape
    assert out["updated_memory"].shape == tensors["memory"].shape
    assert not torch.allclose(out["updated_z"], tensors["z"])
    assert not torch.allclose(out["updated_memory"], tensors["memory"])
    assert organ.generate_text(TinyCharTokenizer(), out)

    forbidden_runtime_attrs = {"input_texts", "target_texts", "dialogue_history", "transcript"}
    assert forbidden_runtime_attrs.isdisjoint(set(vars(organ)))
    assert all(isinstance(value, torch.Tensor) for value in organ.state_dict().values())


def test_dialogue_dataset_has_grounded_split_separation() -> None:
    train = build_dialogue_dataset(FROZEN_CHECKPOINT, "smoke", "train", count=64)
    heldout = build_dialogue_dataset(FROZEN_CHECKPOINT, "smoke", "heldout", count=64)
    assert train.synthetic_tokens > 0
    assert set(train.forms).isdisjoint(set(heldout.forms))
    assert train.tensors["requires_memory"].any()
    assert train.tensors["requires_z"].any()
    assert train.tensors["requires_private"].any()
    assert train.tensors["is_wrong"].any()


def test_dialogue_smoke_train_and_eval_runs(tmp_path: Path) -> None:
    output = tmp_path / "dialogue_smoke.pt"
    summary = train_dialogue("smoke", output=output, checkpoint=FROZEN_CHECKPOINT)
    assert output.exists()
    assert summary["records"] == DIALOGUE_CONFIGS["smoke"].train_records

    report = evaluate_dialogue(
        FROZEN_CHECKPOINT,
        output,
        config_name="smoke",
        json_output=tmp_path / "dialogue_report.json",
        include_prior=False,
    )
    assert report["config"] == "smoke"
    assert report["gate_checks"]["copy_rate_le_0_05"] is True
    assert report["gate_checks"]["random_labels_no_effect"] is True
    assert Path(tmp_path / "dialogue_report.json").exists()
