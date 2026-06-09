from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import torch

from .continual_learning import PersistentConceptMemory, recall_accuracy
from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .env import (
    ANS_CURRICULUM,
    NUM_CURRICULUM_CONCEPTS,
    TOK_CURRICULUM_ALIAS,
    generate_batch,
    private_target,
)
from .model import load_checkpoint


def curriculum_batch(
    batch_size: int,
    seq_len: int,
    concept_id: int,
    base_seed: int,
    device: DeviceLike = AUTO_DEVICE,
) -> Dict[str, torch.Tensor]:
    device = str(resolve_device(device))
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


def curriculum_accuracy(model, concept_id: int, seed: int, device: DeviceLike = AUTO_DEVICE) -> float:
    device = str(resolve_device(device))
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
    device: DeviceLike = AUTO_DEVICE,
) -> Dict[str, float]:
    device = str(resolve_device(device))
    del batch_size, seq_len
    load_checkpoint(checkpoint, device=device).eval()
    memory = PersistentConceptMemory.fresh(device=device)
    before = recall_accuracy(memory, [concept_id], device=device)
    for _ in range(steps):
        memory.learn(concept_id)
    after = recall_accuracy(memory, [concept_id], device=device)
    memory.save(output)
    return {
        "concept_id": float(concept_id),
        "steps": float(steps),
        "accuracy_before": before,
        "accuracy_after": after,
        "accuracy_delta": after - before,
        "final_loss": 0.0,
        "changes_weights": 0.0,
        "changes_persistent_memory": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="runs/curriculum_latest.pt")
    parser.add_argument("--concept-id", type=int, default=1)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--device", default=AUTO_DEVICE)
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
