from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .device import AUTO_DEVICE, DeviceLike, make_generator, resolve_device
from .dialogue_env import DIALOGUE_CONFIGS, build_dialogue_dataset, dataset_metadata, answer_config
from .heldout_causal import FROZEN_CHECKPOINT
from .language_organ import DialogueOrgan, TinyCharTokenizer, masked_char_ce, save_dialogue_checkpoint


def _take(tensors: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in tensors.items()}


def dialogue_losses(organ: DialogueOrgan, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = organ(batch["input_ids"], batch["z"], batch["memory"], batch["private_tokens"], batch.get("word_ids"))
    char_loss = masked_char_ce(out["char_logits"], batch["target_ids"])
    answer_loss = F.cross_entropy(out["answer_logits"], batch["answer_target"])
    act_loss = F.cross_entropy(out["speech_act_logits"], batch["act_target"])
    source_loss = F.cross_entropy(out["source_logits"], batch["source_target"])
    action_loss = F.cross_entropy(out["action_delta_logits"], batch["action_target"])
    kind_loss = F.cross_entropy(out["kind_logits"], batch["kind"])
    private_loss = F.cross_entropy(out["private_logits"], batch["private_tokens"])
    z_delta = (out["updated_z"] - batch["z"]).pow(2).mean()
    memory_delta = (out["updated_memory"] - batch["memory"]).pow(2).mean()
    total = (
        1.80 * answer_loss
        + 1.10 * act_loss
        + 0.90 * source_loss
        + 0.85 * action_loss
        + 1.00 * kind_loss
        + 0.55 * private_loss
        + 0.65 * char_loss
        + 0.01 * z_delta
        + 0.01 * memory_delta
    )
    return {
        "total": total,
        "answer": answer_loss.detach(),
        "act": act_loss.detach(),
        "source": source_loss.detach(),
        "action": action_loss.detach(),
        "kind": kind_loss.detach(),
        "private": private_loss.detach(),
        "char": char_loss.detach(),
    }


def train_dialogue(
    config_name: str = "tiny",
    output: str | Path = "runs/dialogue_tiny.pt",
    checkpoint: str | Path = FROZEN_CHECKPOINT,
    device: DeviceLike = AUTO_DEVICE,
) -> dict[str, Any]:
    target_device = resolve_device(device)
    cfg = DIALOGUE_CONFIGS[config_name]
    train = build_dialogue_dataset(checkpoint, config_name, "train", device=target_device)
    if train.synthetic_tokens < 1_000_000 and config_name != "smoke":
        raise ValueError(f"dialogue training data too small: {train.synthetic_tokens} synthetic tokens")
    torch.manual_seed(cfg.seed)
    organ = DialogueOrgan(answer_config(), device=target_device)
    optimizer = torch.optim.AdamW(organ.parameters(), lr=cfg.lr, weight_decay=1e-4)
    generator = make_generator(cfg.seed + 99, target_device)
    n = train.tensors["input_ids"].shape[0]
    last: dict[str, torch.Tensor] = {}
    organ.train()
    for step in range(1, cfg.steps + 1):
        indices = torch.randint(0, n, (cfg.batch_size,), generator=generator, device=target_device)
        batch = _take(train.tensors, indices)
        losses = dialogue_losses(organ, batch)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(organ.parameters(), 1.0)
        optimizer.step()
        last = losses
        if step == 1 or step == cfg.steps or step % max(1, cfg.steps // 5) == 0:
            printable = {key: round(float(value.item()), 5) for key, value in losses.items()}
            print(json.dumps({"step": step, **printable}))
    organ.eval()
    tokenizer = TinyCharTokenizer()
    with torch.no_grad():
        sample = _take(train.tensors, torch.arange(min(8, n), device=target_device))
        out = organ(sample["input_ids"], sample["z"], sample["memory"], sample["private_tokens"])
        generated = organ.generate_text(tokenizer, out)
        train_answer_acc = float((out["answer_logits"].argmax(dim=-1) == sample["answer_target"]).float().mean().item())
    summary = {
        "config": config_name,
        "checkpoint": str(checkpoint),
        "output": str(output),
        "synthetic_tokens": int(train.synthetic_tokens),
        "records": int(n),
        "losses": {key: float(value.item()) for key, value in last.items()},
        "sample_answer_accuracy": train_answer_acc,
        "sample_generated_text": generated[:3],
        "dataset": dataset_metadata(train),
    }
    save_dialogue_checkpoint(output, organ, summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(DIALOGUE_CONFIGS), default="tiny")
    parser.add_argument("--output", default="runs/dialogue_tiny.pt")
    parser.add_argument("--checkpoint", default=FROZEN_CHECKPOINT)
    parser.add_argument("--device", default=AUTO_DEVICE)
    args = parser.parse_args()
    train_dialogue(args.config, args.output, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
