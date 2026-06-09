from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import torch

from src.env import GRID_SIZE, NUM_BODY_SCALARS, NUM_COLORS, generate_batch
from src.living_eval import run_autoregressive_private
from src.model import load_checkpoint
from src.train import blank_training_batch

from audit.independent_verify import ACTION_NAMES, TOKEN_NAMES


VISIBLE_FLAG_INDEX = GRID_SIZE + 2 + NUM_BODY_SCALARS + (NUM_COLORS + 1)


def build_examples(checkpoint: str | Path, device: str = "cpu") -> list[dict[str, Any]]:
    model = load_checkpoint(checkpoint, device=device)
    model.eval()
    base = generate_batch(3, 112, 87200, device=device)
    idle = blank_training_batch(base, blank_after=16)
    with torch.no_grad():
        outputs = run_autoregressive_private(model, idle)
    ticks = [0, 3, 8, 16, 32, 64, 95, 111]
    examples: list[dict[str, Any]] = []
    for sample in range(3):
        rows = []
        for tick in ticks:
            action = int(outputs["action_logits"][sample, tick].argmax().item())
            memory = int(outputs["memory_color_logits"][sample, tick].argmax().item())
            world_pos = int(outputs["world_pos_logits"][sample, tick].argmax().item())
            language = int(outputs["language_logits"][sample, tick].argmax().item())
            target_memory = int(base["memory_color_target"][sample, tick].item())
            target_pos = int(base["world_pos_target"][sample, tick].item())
            rows.append(
                {
                    "tick": tick,
                    "token": TOKEN_NAMES.get(int(idle["lang_in"][sample, tick].item()), str(int(idle["lang_in"][sample, tick].item()))),
                    "visible": bool(idle["sensory"][sample, tick, VISIBLE_FLAG_INDEX].item() > 0.5),
                    "private_in": int(outputs["generated_private"][sample, max(tick - 1, 0)].item()) if tick > 0 else 0,
                    "generated_private": int(outputs["generated_private"][sample, tick].item()),
                    "action": ACTION_NAMES.get(action, str(action)),
                    "language_token": language,
                    "memory": f"{memory}/{target_memory}",
                    "object_pos": f"{world_pos}/{target_pos}",
                    "pass": memory == target_memory and world_pos == target_pos,
                }
            )
        examples.append({"sample": sample, "rows": rows})
    return examples


def render(examples: list[dict[str, Any]]) -> str:
    lines = ["# Audit Probe Examples", ""]
    for example in examples:
        lines.append(f"## Sample {example['sample']}")
        lines.append("")
        lines.append("| Tick | Token | Visible | Private In | Generated Private | Action | Language | Memory | Object Pos | Pass |")
        lines.append("| ---: | --- | --- | ---: | ---: | --- | ---: | --- | --- | --- |")
        for row in example["rows"]:
            lines.append(
                f"| {row['tick']} | {row['token']} | {row['visible']} | {row['private_in']} | {row['generated_private']} | {row['action']} | {row['language_token']} | {row['memory']} | {row['object_pos']} | {row['pass']} |"
            )
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()
    examples = build_examples(args.checkpoint, args.device)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render(examples), encoding="utf-8")
    print(f"wrote {output}")


if __name__ == "__main__":
    main()

