from __future__ import annotations

import argparse
import json
from typing import Dict, List, Tuple

import torch

from .adversarial import (
    SENSOR_BODY,
    SENSOR_COLOR,
    SENSOR_OBJECT_POS,
    SENSOR_POS,
    SENSOR_VISIBLE,
    aggregate_core,
    blank_continuation_batch,
    clone_batch,
    no_language_batch,
    no_provenance_batch,
    run_sequence,
    score_outputs,
)
from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .env import (
    ACTION_STAY,
    BODY_DAMAGE,
    BODY_ENERGY,
    BODY_FATIGUE,
    GRID_SIZE,
    NUM_COLORS,
    PROV_IMAGINED,
    PROV_OBSERVED,
    PROV_REMEMBERED,
    PROV_TOLD,
    TOK_ASK_ACTION,
    TOK_ASK_COLOR,
    TOK_ASK_GOAL,
    TOK_ASK_OBJECT_POS,
    TOK_IMAGINE,
    TOK_INFER_OBJECT,
    TOK_NONE,
    TOK_OBSERVE_OBJECT,
    TOK_TOLD_GOAL,
    generate_batch,
    shortest_action,
)
from .metrics import masked_accuracy
from .model import RecurrentLatentModel, load_checkpoint


FROZEN_CHECKPOINT = "frozen/recurrent_latent_fast.pt"

HELDOUT_CONFIGS = {
    "smoke": {"batch_size": 24, "seq_len": 104, "seed": 4400, "margin": 0.20},
    "fast": {"batch_size": 192, "seq_len": 112, "seed": 14400, "margin": 0.25},
}


def set_visible_object(batch: Dict[str, torch.Tensor], tick: int, pos: torch.Tensor, color: torch.Tensor) -> None:
    batch["sensory"][:, tick, SENSOR_COLOR] = torch.nn.functional.one_hot(color, NUM_COLORS + 1).float()
    batch["sensory"][:, tick, SENSOR_VISIBLE] = 1.0
    batch["sensory"][:, tick, SENSOR_OBJECT_POS] = torch.nn.functional.one_hot(pos, GRID_SIZE + 1).float()


def recompute_actions(batch: Dict[str, torch.Tensor], start_tick: int = 0) -> None:
    current_pos = batch["sensory"][:, :, SENSOR_POS].argmax(dim=-1)
    for tick in range(start_tick, batch["sensory"].shape[1]):
        batch["action_target"][:, tick] = shortest_action(current_pos[:, tick], batch["world_pos_target"][:, tick])


def larger_world_batch(batch: Dict[str, torch.Tensor], virtual_size: int = 9) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    current = altered["sensory"][:, :, SENSOR_POS].argmax(dim=-1)
    virtual_offset = (torch.arange(altered["sensory"].shape[0], device=current.device).view(-1, 1) * 3) % virtual_size
    virtual_pos = (current + virtual_offset) % virtual_size
    projected_pos = virtual_pos % GRID_SIZE
    virtual_target = (altered["world_pos_target"] + virtual_offset) % virtual_size
    projected_target = virtual_target % GRID_SIZE
    altered["sensory"][:, :, SENSOR_POS] = torch.nn.functional.one_hot(projected_pos, GRID_SIZE).float()
    altered["world_pos_target"] = projected_target.long()
    visible = altered["sensory"][:, :, SENSOR_VISIBLE] > 0.5
    altered["sensory"][:, :, SENSOR_OBJECT_POS] = torch.nn.functional.one_hot(
        torch.where(visible, projected_target, torch.full_like(projected_target, GRID_SIZE)),
        GRID_SIZE + 1,
    ).float()
    recompute_actions(altered)
    return altered


def heldout_distractor_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    batch_size = altered["sensory"].shape[0]
    device = altered["sensory"].device
    base = torch.arange(batch_size, device=device)
    for idx, tick in enumerate((9, 22, 39, 58, 91)):
        distractor_color = (altered["world_color_target"][:, tick] + idx + 1) % NUM_COLORS
        distractor_pos = (altered["world_pos_target"][:, tick] + base + idx + 1) % GRID_SIZE
        set_visible_object(altered, tick, distractor_pos, distractor_color)
        altered["lang_in"][:, tick] = TOK_IMAGINE if idx % 2 else TOK_INFER_OBJECT
        altered["provenance_target"][:, tick] = PROV_IMAGINED if idx % 2 else PROV_TOLD
    return altered


def variable_delay_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    altered["grounded_language_mask"].fill_(False)
    altered["delayed_memory_mask"].fill_(False)
    altered["action_mask"].fill_(False)
    delay_tokens = [TOK_ASK_COLOR, TOK_ASK_OBJECT_POS, TOK_ASK_GOAL, TOK_ASK_ACTION]
    for i, tick in enumerate((23, 49, 77, 103)):
        if tick >= altered["lang_in"].shape[1]:
            continue
        token = delay_tokens[i % len(delay_tokens)]
        altered["lang_in"][:, tick] = token
        if token == TOK_ASK_ACTION:
            altered["action_mask"][:, tick] = True
        else:
            altered["grounded_language_mask"][:, tick] = True
            altered["delayed_memory_mask"][:, tick] = tick >= 49
    return altered


def contradictory_source_chain_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    truth_color = altered["world_color_target"][:, 0]
    truth_pos = altered["world_pos_target"][:, 0]
    false_color = (truth_color + 1) % NUM_COLORS
    false_pos = (truth_pos + 2) % GRID_SIZE
    imagined_color = (truth_color + 2) % NUM_COLORS
    imagined_pos = (truth_pos + 3) % GRID_SIZE
    chain_events = [
        (19, TOK_TOLD_GOAL, false_pos, false_color, PROV_TOLD),
        (37, TOK_IMAGINE, imagined_pos, imagined_color, PROV_IMAGINED),
        (61, TOK_INFER_OBJECT, truth_pos, truth_color, PROV_TOLD),
    ]
    for tick, token, pos, color, provenance in chain_events:
        altered["lang_in"][:, tick] = token
        altered["provenance_target"][:, tick] = provenance
        set_visible_object(altered, tick, pos, color)
    altered["lang_in"][:, 88] = TOK_ASK_COLOR
    altered["grounded_language_mask"][:, 88] = True
    altered["delayed_memory_mask"][:, 88] = True
    return altered


def multi_step_goal_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    first = (altered["world_pos_target"][:, 0] + 1) % GRID_SIZE
    second = (altered["world_pos_target"][:, 0] + 3) % GRID_SIZE
    final = altered["world_pos_target"][:, 0]
    schedule = [(28, first), (55, second), (82, final)]
    for tick, pos in schedule:
        altered["world_pos_target"][:, tick:] = pos.view(-1, 1)
        altered["lang_in"][:, tick] = TOK_TOLD_GOAL
        set_visible_object(altered, tick, pos, altered["world_color_target"][:, tick])
        altered["grounded_language_mask"][:, tick] = True
    recompute_actions(altered, start_tick=28)
    altered["action_mask"][:, 84:] = True
    return altered


def blank_heldout_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = blank_continuation_batch(batch, blank_after=16)
    altered["grounded_language_mask"][:, -1] = True
    altered["delayed_memory_mask"][:, -1] = True
    altered["action_mask"][:, -1] = True
    return altered


def energy_pressure_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    altered["sensory"][:, 70:, SENSOR_BODY.start + BODY_ENERGY] = 0.01
    altered["sensory"][:, 70:, SENSOR_BODY.start + BODY_FATIGUE] = 0.99
    altered["sensory"][:, 70:, SENSOR_BODY.start + BODY_DAMAGE] = 0.90
    altered["action_target"][:, 90:] = ACTION_STAY
    altered["action_mask"][:, 90:] = True
    return altered


def step_with_optional_language_mask(
    model: RecurrentLatentModel,
    sensory: torch.Tensor,
    lang_in: torch.Tensor,
    z_prev: torch.Tensor,
    mask_language_embedding: bool,
    private_in: torch.Tensor | None = None,
) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
    if not mask_language_embedding:
        observation = {"sensory": sensory, "lang_in": lang_in}
        if private_in is not None:
            observation["private_in"] = private_in
        return model.step(observation, z_prev)
    if private_in is None:
        private_in = torch.zeros_like(lang_in)
    sensor_features = model.sensor_encoder(sensory)
    token_features = torch.zeros(
        lang_in.shape[0],
        model.config.embed_dim,
        device=sensory.device,
        dtype=sensory.dtype,
    )
    private_features = model.private_embedding(private_in)
    mixed = model.input_mixer(torch.cat([sensor_features, token_features, private_features], dim=-1))
    z_next = model.core(mixed, z_prev)
    z_view = model.norm(z_next)
    output = {
        "action_logits": model.action_head(z_view),
        "language_logits": model.language_head(z_view),
        "private_logits": model.private_head(z_view),
        "provenance_logits": model.provenance_head(z_view),
        "world_color_logits": model.world_color_head(z_view),
        "world_pos_logits": model.world_pos_head(z_view),
        "memory_color_logits": model.memory_color_head(z_view),
        "self_start_logits": model.self_start_head(z_view),
    }
    return output, z_next


def run_sequence_heldout(
    model: RecurrentLatentModel,
    batch: Dict[str, torch.Tensor],
    baseline: str = "recurrent",
    ablation_start: int = 16,
) -> Dict[str, torch.Tensor]:
    if baseline == "feedforward":
        return run_sequence(model, batch, mode="feedforward_only")
    if baseline == "zero_z":
        return run_sequence(model, batch, mode="zero_z", ablation_start=ablation_start)
    if baseline == "shuffled_z":
        return run_sequence(model, batch, mode="shuffle_z", ablation_start=ablation_start)
    if baseline == "no_language":
        return run_sequence(model, no_language_batch(batch))
    if baseline == "no_provenance":
        return run_sequence(model, no_provenance_batch(batch))
    if baseline == "no_blank_continuation":
        blank = blank_heldout_batch(batch)
        return run_sequence(model, blank, mode="zero_z", ablation_start=16)

    batch_size, seq_len, _ = batch["sensory"].shape
    z = model.initial_state(batch_size, device=batch["sensory"].device)
    private_in_batch = batch.get("private_in")
    outputs: List[Dict[str, torch.Tensor]] = []
    latents: List[torch.Tensor] = []
    mask_embedding = baseline == "internal_language_mask"
    for tick in range(seq_len):
        lang_in = batch["lang_in"][:, tick]
        if private_in_batch is not None:
            private_in = private_in_batch[:, tick]
        else:
            private_in = torch.zeros_like(lang_in)
        output, z = step_with_optional_language_mask(
            model,
            batch["sensory"][:, tick],
            lang_in,
            z,
            mask_language_embedding=mask_embedding,
            private_in=private_in,
        )
        outputs.append(output)
        latents.append(z)
    stacked: Dict[str, torch.Tensor] = {}
    for key in outputs[0]:
        stacked[key] = torch.stack([item[key] for item in outputs], dim=1)
    stacked["latents"] = torch.stack(latents, dim=1)
    return stacked


def heldout_score(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    metrics = score_outputs(outputs, batch)
    metrics["core_score"] = aggregate_core(metrics)
    final_mask = torch.zeros_like(batch["delayed_memory_mask"])
    final_mask[:, -1] = True
    metrics["final_memory"] = masked_accuracy(outputs["memory_color_logits"], batch["memory_color_target"], final_mask)
    metrics["final_object_pos"] = masked_accuracy(outputs["world_pos_logits"], batch["world_pos_target"], final_mask)
    return metrics


def build_heldout_suites(batch: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, torch.Tensor]]:
    return {
        "larger_world_9_projected": larger_world_batch(batch, virtual_size=9),
        "distractor_objects_untrained_ticks": heldout_distractor_batch(batch),
        "variable_delays_23_49_77_103": variable_delay_batch(batch),
        "contradictory_source_chain": contradictory_source_chain_batch(batch),
        "multi_step_goal_schedule": multi_step_goal_batch(batch),
        "blank_continuation_after_16": blank_heldout_batch(batch),
        "energy_pressure_late": energy_pressure_batch(batch),
    }


def evaluate_suite(
    model: RecurrentLatentModel,
    suite_batch: Dict[str, torch.Tensor],
    baselines: List[str],
) -> Dict[str, Dict[str, float]]:
    report = {}
    for baseline in baselines:
        outputs = run_sequence_heldout(model, suite_batch, baseline=baseline)
        report[baseline] = heldout_score(outputs, suite_batch)
    return report


def summarize(report: Dict[str, Dict[str, Dict[str, float]]], baselines: List[str], margin: float) -> Dict[str, object]:
    recurrent_scores = [suite["recurrent"]["core_score"] for suite in report.values()]
    baseline_scores = {
        baseline: sum(suite[baseline]["core_score"] for suite in report.values()) / len(report)
        for baseline in baselines
        if baseline != "recurrent"
    }
    recurrent_mean = sum(recurrent_scores) / len(recurrent_scores)
    margins = {baseline: recurrent_mean - score for baseline, score in baseline_scores.items()}
    suite_passes = {
        name: suite["recurrent"]["core_score"] >= max(suite[baseline]["core_score"] for baseline in baselines if baseline != "recurrent") + margin
        for name, suite in report.items()
    }
    internal_language_delta = recurrent_mean - baseline_scores["internal_language_mask"]
    return {
        "recurrent_mean": recurrent_mean,
        "baseline_means": baseline_scores,
        "margins": margins,
        "required_margin": margin,
        "suite_passes": suite_passes,
        "internal_language_delta": internal_language_delta,
        "passes": all(value >= margin for value in margins.values())
        and all(suite_passes.values())
        and internal_language_delta >= margin,
    }


def evaluate_heldout_causal(
    checkpoint: str = FROZEN_CHECKPOINT,
    config_name: str = "fast",
    device: DeviceLike = AUTO_DEVICE,
) -> Dict[str, object]:
    device = str(resolve_device(device))
    cfg = HELDOUT_CONFIGS[config_name]
    model = load_checkpoint(checkpoint, device=device)
    model.eval()
    batch = generate_batch(cfg["batch_size"], cfg["seq_len"], cfg["seed"], device=device)
    suites = build_heldout_suites(batch)
    baselines = [
        "recurrent",
        "feedforward",
        "zero_z",
        "shuffled_z",
        "no_language",
        "no_provenance",
        "no_blank_continuation",
        "internal_language_mask",
    ]
    with torch.no_grad():
        suite_reports = {name: evaluate_suite(model, suite, baselines) for name, suite in suites.items()}
    summary = summarize(suite_reports, baselines, margin=float(cfg["margin"]))
    report: Dict[str, object] = {
        "checkpoint": checkpoint,
        "config": cfg,
        "task_structures": list(suites.keys()),
        "baselines": baselines,
        "report": suite_reports,
        "summary": summary,
        "notes": {
            "architecture_frozen": "src/model.py is not modified by this evaluator.",
            "checkpoint_frozen": "Default checkpoint is frozen/recurrent_latent_fast.pt.",
            "internal_language_ablation": "internal_language_mask zeros the model token embedding path at evaluation time; this model has no generated language tokens fed back into state.",
        },
    }
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=FROZEN_CHECKPOINT)
    parser.add_argument("--config", choices=sorted(HELDOUT_CONFIGS), default="fast")
    parser.add_argument("--device", default=AUTO_DEVICE)
    args = parser.parse_args()
    evaluate_heldout_causal(args.checkpoint, args.config, args.device)


if __name__ == "__main__":
    main()
