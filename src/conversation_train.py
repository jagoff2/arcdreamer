from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .conversation_env import (
    ACT_REFUSAL,
    ACT_SILENCE,
    ACT_UNCERTAIN,
    CLARIFY_CLASS,
    CONVERSATION_CONFIGS,
    REFUSE_CLASS,
    SILENCE_CLASS,
    build_conversation_dataset,
    conversation_config,
    dataset_metadata,
)
from .device import AUTO_DEVICE, DeviceLike, make_generator, resolve_device
from .free_text_decoder import FreeTextConversationDecoder, masked_sequence_ce, save_conversation_checkpoint
from .heldout_causal import FROZEN_CHECKPOINT


def _take(tensors: dict[str, torch.Tensor], indices: torch.Tensor) -> dict[str, torch.Tensor]:
    return {key: value[indices] for key, value in tensors.items()}


def _masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return logits.sum() * 0.0
    return F.cross_entropy(logits[mask], target[mask])


def conversation_losses(model: FreeTextConversationDecoder, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    out = model(
        batch["input_ids"],
        batch["word_ids"],
        batch["z"],
        batch["memory"],
        batch["private_tokens"],
        batch["turn_ids"],
        batch["target_ids"],
    )
    char_loss = masked_sequence_ce(out["char_logits"], batch["target_ids"])
    answer_weights = torch.ones(out["answer_logits"].shape[-1], device=out["answer_logits"].device)
    answer_weights[CLARIFY_CLASS] = 4.0
    answer_weights[REFUSE_CLASS] = 2.0
    answer_weights[SILENCE_CLASS] = 4.0
    act_weights = torch.ones(out["speech_act_logits"].shape[-1], device=out["speech_act_logits"].device)
    act_weights[ACT_UNCERTAIN] = 4.0
    act_weights[ACT_REFUSAL] = 2.0
    act_weights[ACT_SILENCE] = 4.0
    answer_loss = F.cross_entropy(out["answer_logits"], batch["answer_target"], weight=answer_weights)
    act_loss = F.cross_entropy(out["speech_act_logits"], batch["act_target"], weight=act_weights)
    source_loss = F.cross_entropy(out["source_logits"], batch["source_target"])
    action_loss = F.cross_entropy(out["action_logits"], batch["action_target"])
    kind_loss = F.cross_entropy(out["kind_logits"], batch["kind"])
    private_loss = F.cross_entropy(out["private_logits"], batch["private_tokens"])
    memory_answer_loss = _masked_ce(out["answer_logits"], batch["answer_target"], batch["requires_memory"])
    conflict_answer_loss = _masked_ce(out["answer_logits"], batch["answer_target"], batch["is_conflict"])
    missing_answer_loss = _masked_ce(out["answer_logits"], batch["answer_target"], batch["is_missing"])
    total = (
        1.90 * answer_loss
        + 1.15 * act_loss
        + 0.90 * source_loss
        + 0.90 * action_loss
        + 1.05 * kind_loss
        + 0.55 * private_loss
        + 1.20 * char_loss
        + 0.60 * memory_answer_loss
        + 1.00 * conflict_answer_loss
        + 1.00 * missing_answer_loss
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
        "memory_answer": memory_answer_loss.detach(),
        "conflict_answer": conflict_answer_loss.detach(),
        "missing_answer": missing_answer_loss.detach(),
    }


def train_conversation(
    config_name: str = "small",
    output: str | Path = "runs/conversation_small.pt",
    checkpoint: str | Path = FROZEN_CHECKPOINT,
    device: DeviceLike = AUTO_DEVICE,
) -> dict[str, Any]:
    target_device = resolve_device(device)
    cfg = CONVERSATION_CONFIGS[config_name]
    data = build_conversation_dataset(checkpoint, config_name, "train", device=target_device)
    if config_name != "smoke" and data.synthetic_tokens < 10_000_000:
        raise ValueError(f"conversation data too small: {data.synthetic_tokens} synthetic tokens")
    torch.manual_seed(cfg.seed)
    model = FreeTextConversationDecoder(conversation_config(), device=target_device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)
    generator = make_generator(cfg.seed + 99, target_device)
    n = int(data.tensors["input_ids"].shape[0])
    model.train()
    last: dict[str, torch.Tensor] = {}
    for step in range(1, cfg.steps + 1):
        indices = torch.randint(0, n, (cfg.batch_size,), generator=generator, device=target_device)
        batch = _take(data.tensors, indices)
        losses = conversation_losses(model, batch)
        optimizer.zero_grad(set_to_none=True)
        losses["total"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        last = losses
        if step == 1 or step == cfg.steps or step % max(1, cfg.steps // 5) == 0:
            printable = {key: round(float(value.item()), 5) for key, value in losses.items()}
            print(json.dumps({"step": step, **printable}))
    model.eval()
    with torch.no_grad():
        sample = _take(data.tensors, torch.arange(min(16, n), device=target_device))
        out = model(
            sample["input_ids"],
            sample["word_ids"],
            sample["z"],
            sample["memory"],
            sample["private_tokens"],
            sample["turn_ids"],
            sample["target_ids"],
        )
        sample_accuracy = float((out["answer_logits"].argmax(dim=-1) == sample["answer_target"]).float().mean().item())
    summary = {
        "config": config_name,
        "checkpoint": str(checkpoint),
        "output": str(output),
        "records": n,
        "synthetic_tokens": int(data.synthetic_tokens),
        "losses": {key: float(value.item()) for key, value in last.items()},
        "sample_answer_accuracy": sample_accuracy,
        "dataset": dataset_metadata(data),
    }
    save_conversation_checkpoint(output, model, summary)
    print(json.dumps(summary, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=sorted(CONVERSATION_CONFIGS), default="small")
    parser.add_argument("--output", default="runs/conversation_small.pt")
    parser.add_argument("--checkpoint", default=FROZEN_CHECKPOINT)
    parser.add_argument("--device", default=AUTO_DEVICE)
    args = parser.parse_args()
    train_conversation(args.config, args.output, args.checkpoint, args.device)


if __name__ == "__main__":
    main()
