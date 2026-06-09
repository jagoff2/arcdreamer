from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from .env import (
    ANS_CURRICULUM,
    NUM_CURRICULUM_CONCEPTS,
    TOK_CURRICULUM_ALIAS,
    generate_batch,
    private_target,
)
from .model import load_checkpoint, save_checkpoint


def curriculum_batch(
    batch_size: int,
    seq_len: int,
    concept_id: int,
    base_seed: int,
    device: str = "cpu",
) -> Dict[str, torch.Tensor]:
    concept = concept_id % NUM_CURRICULUM_CONCEPTS
    batch = generate_batch(batch_size, seq_len, base_seed=base_seed, device=device)
    query_start = max(8, seq_len // 3)
    batch["lang_in"][:, query_start:] = TOK_CURRICULUM_ALIAS
    batch["language_target"][:, query_start:] = ANS_CURRICULUM + concept
    for tick in range(query_start, seq_len):
        energy = batch["sensory"][:, tick, 7]
        damage = batch["sensory"][:, tick, 9]
        batch["private_target"][:, tick] = private_target(
            tick,
            TOK_CURRICULUM_ALIAS,
            batch["world_color_target"][:, tick],
            batch["world_pos_target"][:, tick],
            energy,
            damage,
        )
    batch["private_in"].zero_()
    batch["private_in"][:, 1:] = batch["private_target"][:, :-1]
    batch["curriculum_mask"] = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    batch["curriculum_mask"][:, query_start:] = True
    return batch


def curriculum_accuracy(model, concept_id: int, seed: int, device: str = "cpu") -> float:
    batch = curriculum_batch(64, 48, concept_id, seed, device=device)
    with torch.no_grad():
        outputs = model(batch["sensory"], batch["lang_in"], batch["private_in"])
    pred = outputs["language_logits"].argmax(dim=-1)
    mask = batch["curriculum_mask"]
    target = batch["language_target"]
    return float((pred[mask] == target[mask]).float().mean().item())


def acquire_new_concept(
    checkpoint: str | Path,
    output: str | Path,
    concept_id: int = 1,
    steps: int = 80,
    batch_size: int = 32,
    seq_len: int = 48,
    device: str = "cpu",
) -> Dict[str, float]:
    torch.manual_seed(8100 + concept_id)
    model = load_checkpoint(checkpoint, device=device)
    before = curriculum_accuracy(model, concept_id, seed=8200 + concept_id, device=device)
    model.train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-4)
    final_loss = 0.0
    for step in range(1, steps + 1):
        batch = curriculum_batch(batch_size, seq_len, concept_id, base_seed=9000 + concept_id * 100 + step, device=device)
        outputs = model(batch["sensory"], batch["lang_in"], batch["private_in"])
        mask = batch["curriculum_mask"]
        language_loss = F.cross_entropy(outputs["language_logits"][mask], batch["language_target"][mask])
        private_loss = F.cross_entropy(
            outputs["private_logits"].reshape(-1, outputs["private_logits"].shape[-1]),
            batch["private_target"].reshape(-1),
        )
        retention = F.cross_entropy(
            outputs["world_pos_logits"].reshape(-1, outputs["world_pos_logits"].shape[-1]),
            batch["world_pos_target"].reshape(-1),
        )
        loss = language_loss + 0.25 * private_loss + 0.05 * retention
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        final_loss = float(loss.detach().item())
    model.eval()
    after = curriculum_accuracy(model, concept_id, seed=8300 + concept_id, device=device)
    save_checkpoint(output, model, {"curriculum_concept_id": concept_id, "steps": steps}, {"curriculum_after": after})
    return {
        "concept_id": float(concept_id),
        "steps": float(steps),
        "accuracy_before": before,
        "accuracy_after": after,
        "accuracy_delta": after - before,
        "final_loss": final_loss,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="runs/curriculum_latest.pt")
    parser.add_argument("--concept-id", type=int, default=1)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    print(
        json.dumps(
            acquire_new_concept(
                checkpoint=args.checkpoint,
                output=args.output,
                concept_id=args.concept_id,
                steps=args.steps,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
