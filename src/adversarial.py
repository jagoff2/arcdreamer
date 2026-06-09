from __future__ import annotations

import argparse
import json
from copy import deepcopy
from typing import Callable, Dict, Iterable, List, Tuple

import torch

from .device import AUTO_DEVICE, DeviceLike, make_generator, resolve_device
from .env import (
    ACTION_STAY,
    ANS_COLOR,
    ANS_IMAGINED,
    BODY_DAMAGE,
    BODY_ENERGY,
    BODY_FATIGUE,
    BODY_RESOURCE,
    GRID_SIZE,
    NUM_BODY_SCALARS,
    NUM_ACTIONS,
    NUM_COLORS,
    PROV_IMAGINED,
    PROV_OBSERVED,
    PROV_REMEMBERED,
    PROV_TOLD,
    SENSOR_DIM,
    TOK_ASK_ACTION,
    TOK_ASK_COLOR,
    TOK_IMAGINE,
    TOK_INFER_OBJECT,
    TOK_NONE,
    TOK_OBSERVE_OBJECT,
    TOK_TOLD_GOAL,
    build_sensory,
    generate_batch,
    shortest_action,
    token_for_tick,
)
from .metrics import latent_noncollapse_stats, masked_accuracy
from .model import RecurrentLatentModel, load_checkpoint


SENSOR_POS = slice(0, GRID_SIZE)
SENSOR_ORIENT = slice(GRID_SIZE, GRID_SIZE + 2)
SENSOR_BODY = slice(GRID_SIZE + 2, GRID_SIZE + 2 + NUM_BODY_SCALARS)
SENSOR_COLOR = slice(GRID_SIZE + 2 + NUM_BODY_SCALARS, GRID_SIZE + 2 + NUM_BODY_SCALARS + NUM_COLORS + 1)
SENSOR_VISIBLE = GRID_SIZE + 2 + NUM_BODY_SCALARS + NUM_COLORS + 1
SENSOR_OBJECT_POS = slice(SENSOR_VISIBLE + 1, SENSOR_DIM)


ADVERSARIAL_CONFIGS = {
    "smoke": {"batch_size": 16, "seq_len": 80, "seed": 3300, "probe_steps": 30},
    "fast": {"batch_size": 256, "seq_len": 80, "seed": 9300, "probe_steps": 120},
}


def clone_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    return {key: value.clone() for key, value in batch.items()}


def event_mask(batch: Dict[str, torch.Tensor], token: int, min_tick: int = 0) -> torch.Tensor:
    ticks = torch.arange(batch["lang_in"].shape[1], device=batch["lang_in"].device).view(1, -1)
    return (batch["lang_in"] == token) & (ticks >= min_tick)


def run_sequence(
    model: RecurrentLatentModel,
    batch: Dict[str, torch.Tensor],
    mode: str = "recurrent",
    ablation_start: int = 64,
    noise_scale: float = 0.75,
) -> Dict[str, torch.Tensor]:
    batch_size, seq_len, _ = batch["sensory"].shape
    device = batch["sensory"].device
    z = model.initial_state(batch_size, device=device)
    frozen_z: torch.Tensor | None = None
    outputs: List[Dict[str, torch.Tensor]] = []
    latents: List[torch.Tensor] = []

    for tick in range(seq_len):
        sensory = batch["sensory"][:, tick]
        lang_in = batch["lang_in"][:, tick]
        private_in = batch.get("private_in")
        if mode == "feedforward_only":
            z = model.initial_state(batch_size, device=device)
        elif mode == "no_memory" and tick >= 4:
            z = model.initial_state(batch_size, device=device)
        elif mode == "zero_z" and tick >= ablation_start:
            z = torch.zeros_like(z)
        elif mode == "freeze_z" and tick >= ablation_start:
            if frozen_z is None:
                frozen_z = z.detach().clone()
            z = frozen_z
        elif mode == "shuffle_z" and tick >= ablation_start:
            z = z[torch.randperm(batch_size, device=device)]
        elif mode == "perturb_z" and tick >= ablation_start:
            generator = make_generator(99173 + tick, device)
            noise = torch.randn(z.shape, generator=generator, device=device)
            z = z + noise_scale * noise

        observation = {"sensory": sensory, "lang_in": lang_in}
        if private_in is not None:
            observation["private_in"] = private_in[:, tick]
        output, z_next = model.step(observation, z)
        outputs.append(output)
        z = z_next
        if mode == "freeze_z" and tick >= ablation_start and frozen_z is not None:
            z = frozen_z
        latents.append(z)

    stacked: Dict[str, torch.Tensor] = {}
    for key in outputs[0]:
        stacked[key] = torch.stack([item[key] for item in outputs], dim=1)
    stacked["latents"] = torch.stack(latents, dim=1)
    return stacked


def no_language_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    altered["lang_in"].fill_(TOK_NONE)
    return altered


def no_provenance_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    source_tokens = torch.zeros_like(altered["lang_in"], dtype=torch.bool)
    for token in (TOK_OBSERVE_OBJECT, TOK_TOLD_GOAL, TOK_IMAGINE, TOK_INFER_OBJECT):
        source_tokens |= altered["lang_in"] == token
    altered["lang_in"] = torch.where(source_tokens, torch.full_like(altered["lang_in"], TOK_NONE), altered["lang_in"])
    return altered


def no_occlusion_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    color = batch["world_color_target"]
    pos = batch["world_pos_target"]
    altered["sensory"][:, :, SENSOR_COLOR] = torch.nn.functional.one_hot(color, NUM_COLORS + 1).float()
    altered["sensory"][:, :, SENSOR_VISIBLE] = 1.0
    altered["sensory"][:, :, SENSOR_OBJECT_POS] = torch.nn.functional.one_hot(pos, GRID_SIZE + 1).float()
    return altered


def blank_continuation_batch(batch: Dict[str, torch.Tensor], blank_after: int = 4) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    altered["sensory"][:, blank_after:, :] = 0.0
    altered["lang_in"][:, blank_after:] = TOK_NONE
    return altered


def remove_told_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    told_mask = altered["lang_in"] == TOK_TOLD_GOAL
    altered["lang_in"] = torch.where(
        told_mask,
        torch.full_like(altered["lang_in"], TOK_NONE),
        altered["lang_in"],
    )
    altered["sensory"][told_mask] = 0.0
    return altered


def remove_questions_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    query_mask = altered["grounded_language_mask"] | (altered["lang_in"] == TOK_ASK_ACTION)
    altered["lang_in"] = torch.where(query_mask, torch.full_like(altered["lang_in"], TOK_NONE), altered["lang_in"])
    return altered


def false_told_conflict_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    false_color = (batch["world_color_target"][:, 8] + 1) % NUM_COLORS
    false_pos = (batch["world_pos_target"][:, 8] + 1) % GRID_SIZE
    altered["sensory"][:, 8, SENSOR_COLOR] = torch.nn.functional.one_hot(false_color, NUM_COLORS + 1).float()
    altered["sensory"][:, 8, SENSOR_VISIBLE] = 1.0
    altered["sensory"][:, 8, SENSOR_OBJECT_POS] = torch.nn.functional.one_hot(false_pos, GRID_SIZE + 1).float()
    return altered


def told_only_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    altered["sensory"][:, :4, SENSOR_COLOR] = torch.nn.functional.one_hot(
        torch.full_like(batch["world_color_target"][:, :4], NUM_COLORS), NUM_COLORS + 1
    ).float()
    altered["sensory"][:, :4, SENSOR_VISIBLE] = 0.0
    altered["sensory"][:, :4, SENSOR_OBJECT_POS] = torch.nn.functional.one_hot(
        torch.full_like(batch["world_pos_target"][:, :4], GRID_SIZE), GRID_SIZE + 1
    ).float()
    altered["sensory"][:, 8, SENSOR_COLOR] = torch.nn.functional.one_hot(
        batch["world_color_target"][:, 8], NUM_COLORS + 1
    ).float()
    altered["sensory"][:, 8, SENSOR_VISIBLE] = 1.0
    altered["sensory"][:, 8, SENSOR_OBJECT_POS] = torch.nn.functional.one_hot(
        batch["world_pos_target"][:, 8], GRID_SIZE + 1
    ).float()
    altered["lang_in"][:, 8] = TOK_TOLD_GOAL
    return altered


def distractor_batch(batch: Dict[str, torch.Tensor], seed: int = 123) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    generator = make_generator(seed, batch["sensory"].device)
    batch_size = batch["sensory"].shape[0]
    distractor_color = torch.randint(
        0, NUM_COLORS, (batch_size,), generator=generator, device=batch["sensory"].device
    )
    distractor_pos = torch.randint(0, GRID_SIZE, (batch_size,), generator=generator, device=batch["sensory"].device)
    for tick in (12, 28, 44):
        altered["sensory"][:, tick, SENSOR_COLOR] = torch.nn.functional.one_hot(
            distractor_color, NUM_COLORS + 1
        ).float()
        altered["sensory"][:, tick, SENSOR_VISIBLE] = 1.0
        altered["sensory"][:, tick, SENSOR_OBJECT_POS] = torch.nn.functional.one_hot(
            distractor_pos, GRID_SIZE + 1
        ).float()
    return altered


def smaller_map_batch(batch: Dict[str, torch.Tensor], map_size: int = 3) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    current_pos = batch["sensory"][:, :, SENSOR_POS].argmax(dim=-1) % map_size
    target_pos = batch["world_pos_target"] % map_size
    color = batch["world_color_target"]
    energy = batch["sensory"][:, :, SENSOR_BODY.start + BODY_ENERGY]
    fatigue = batch["sensory"][:, :, SENSOR_BODY.start + BODY_FATIGUE]
    damage = batch["sensory"][:, :, SENSOR_BODY.start + BODY_DAMAGE]
    resource = batch["sensory"][:, :, SENSOR_BODY.start + BODY_RESOURCE]
    orientation = batch["sensory"][:, :, SENSOR_ORIENT].argmax(dim=-1)
    visible = batch["sensory"][:, :, SENSOR_VISIBLE] > 0.5
    visible_color = torch.where(visible, color, torch.full_like(color, NUM_COLORS))
    visible_pos = torch.where(visible, target_pos, torch.full_like(target_pos, GRID_SIZE))
    for tick in range(batch["sensory"].shape[1]):
        altered["sensory"][:, tick, :] = build_sensory(
            current_pos[:, tick],
            orientation[:, tick],
            energy[:, tick],
            fatigue[:, tick],
            damage[:, tick],
            resource[:, tick],
            visible_color[:, tick],
            visible_pos[:, tick],
        )
        altered["world_pos_target"][:, tick] = target_pos[:, tick]
        altered["action_target"][:, tick] = shortest_action(current_pos[:, tick], target_pos[:, tick])
    return altered


def delayed_goal_change_batch(batch: Dict[str, torch.Tensor], change_tick: int = 56) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    new_pos = (batch["world_pos_target"][:, change_tick:] + 2) % GRID_SIZE
    altered["world_pos_target"][:, change_tick:] = new_pos
    current_pos = altered["sensory"][:, :, SENSOR_POS].argmax(dim=-1)
    for tick in range(change_tick, batch["sensory"].shape[1]):
        altered["action_target"][:, tick] = shortest_action(current_pos[:, tick], altered["world_pos_target"][:, tick])
    altered["sensory"][:, change_tick : change_tick + 4, SENSOR_OBJECT_POS] = torch.nn.functional.one_hot(
        altered["world_pos_target"][:, change_tick : change_tick + 4], GRID_SIZE + 1
    ).float()
    altered["sensory"][:, change_tick : change_tick + 4, SENSOR_VISIBLE] = 1.0
    return altered


def energy_constraint_batch(batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    altered = clone_batch(batch)
    altered["sensory"][:, 60:, SENSOR_BODY.start + BODY_ENERGY] = 0.02
    altered["sensory"][:, 60:, SENSOR_BODY.start + BODY_FATIGUE] = 0.98
    altered["sensory"][:, 60:, SENSOR_BODY.start + BODY_DAMAGE] = 0.90
    altered["action_target"][:, 60:] = ACTION_STAY
    altered["language_target"][:, 60:] = batch["language_target"][:, 60:]
    return altered


def score_outputs(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    delayed_mask = batch["delayed_memory_mask"]
    object_mask = batch["object_mask"]
    grounded_mask = batch["grounded_language_mask"]
    self_mask = batch["self_mask"]
    return {
        "action": masked_accuracy(outputs["action_logits"], batch["action_target"], batch["action_mask"]),
        "delayed_memory": masked_accuracy(outputs["memory_color_logits"], batch["memory_color_target"], delayed_mask),
        "object_color": masked_accuracy(outputs["world_color_logits"], batch["world_color_target"], object_mask),
        "object_pos": masked_accuracy(outputs["world_pos_logits"], batch["world_pos_target"], object_mask),
        "provenance": float(
            (outputs["provenance_logits"].argmax(dim=-1) == batch["provenance_target"]).float().mean().item()
        ),
        "grounded_language": masked_accuracy(outputs["language_logits"], batch["language_target"], grounded_mask),
        "self_world": masked_accuracy(outputs["self_start_logits"], batch["self_start_target"], self_mask),
    }


def aggregate_core(metrics: Dict[str, float]) -> float:
    keys = ["action", "delayed_memory", "object_pos", "provenance", "grounded_language", "self_world"]
    return float(sum(metrics[key] for key in keys) / len(keys))


def counterbalanced_provenance(model: RecurrentLatentModel, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    outputs = run_sequence(model, batch)
    scenarios = {
        "observed": (event_mask(batch, TOK_OBSERVE_OBJECT), PROV_OBSERVED),
        "remembered": (event_mask(batch, TOK_ASK_COLOR, min_tick=64), PROV_REMEMBERED),
        "imagined": (event_mask(batch, TOK_IMAGINE), PROV_IMAGINED),
        "inferred": (event_mask(batch, TOK_INFER_OBJECT), PROV_TOLD),
        "told": (event_mask(batch, TOK_TOLD_GOAL), PROV_TOLD),
    }
    pred = outputs["provenance_logits"].argmax(dim=-1)
    result: Dict[str, float] = {}
    for name, (mask, expected) in scenarios.items():
        if int(mask.sum().item()) == 0:
            result[name] = 0.0
        else:
            result[name] = float((pred[mask] == expected).float().mean().item())
    result["same_fact_colors_covered"] = float(batch["world_color_target"][:, 0].unique().numel())
    result["mean"] = float(sum(result[key] for key in scenarios) / len(scenarios))
    return result


def latent_ablation_report(model: RecurrentLatentModel, batch: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    modes = ["recurrent", "feedforward_only", "no_memory", "freeze_z", "shuffle_z", "zero_z", "perturb_z"]
    report = {}
    for mode in modes:
        start = 0 if mode == "freeze_z" else 4
        outputs = run_sequence(model, batch, mode=mode, ablation_start=start)
        metrics = score_outputs(outputs, batch)
        metrics["core_score"] = aggregate_core(metrics)
        report[mode] = metrics
    recurrent = report["recurrent"]["core_score"]
    destructive = [report[mode]["core_score"] for mode in modes if mode != "recurrent"]
    report["summary"] = {
        "recurrent_core_score": recurrent,
        "best_destructive_baseline": max(destructive),
        "margin_vs_best_destructive": recurrent - max(destructive),
    }
    return report


def baseline_report(model: RecurrentLatentModel, batch: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    variants: Dict[str, Tuple[Dict[str, torch.Tensor], str]] = {
        "recurrent": (batch, "recurrent"),
        "feedforward_only": (batch, "feedforward_only"),
        "shuffled_z": (batch, "shuffle_z"),
        "no_language": (no_language_batch(batch), "recurrent"),
        "no_provenance": (no_provenance_batch(batch), "recurrent"),
        "no_occlusion": (no_occlusion_batch(batch), "recurrent"),
    }
    report = {}
    for name, (variant, mode) in variants.items():
        outputs = run_sequence(model, variant, mode=mode, ablation_start=4)
        metrics = score_outputs(outputs, variant)
        metrics["core_score"] = aggregate_core(metrics)
        report[name] = metrics
    recurrent = report["recurrent"]["core_score"]
    adversarial_baselines = ["feedforward_only", "shuffled_z", "no_language", "no_provenance"]
    best = max(report[name]["core_score"] for name in adversarial_baselines)
    report["summary"] = {
        "recurrent_core_score": recurrent,
        "best_adversarial_baseline": best,
        "margin_vs_best_adversarial_baseline": recurrent - best,
        "no_occlusion_control_score": report["no_occlusion"]["core_score"],
    }
    return report


def language_causal_tests(model: RecurrentLatentModel, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    normal = run_sequence(model, batch)
    no_lang = run_sequence(model, no_language_batch(batch))
    no_questions = run_sequence(model, remove_questions_batch(batch))
    told_only = told_only_batch(batch)
    told_normal = run_sequence(model, told_only)
    no_told = run_sequence(model, remove_told_batch(told_only))
    action_mask = batch["action_mask"] & event_mask(batch, TOK_ASK_ACTION, min_tick=64)
    normal_action = normal["action_logits"].argmax(dim=-1)
    no_lang_action = no_lang["action_logits"].argmax(dim=-1)
    told_normal_action = told_normal["action_logits"].argmax(dim=-1)
    no_told_action = no_told["action_logits"].argmax(dim=-1)
    no_question_action = no_questions["action_logits"].argmax(dim=-1)
    action_change = float((normal_action[action_mask] != no_lang_action[action_mask]).float().mean().item())
    told_action_change = float((told_normal_action[action_mask] != no_told_action[action_mask]).float().mean().item())
    question_action_change = float(
        (normal_action[action_mask] != no_question_action[action_mask]).float().mean().item()
    )
    logit_shift = float(
        (normal["action_logits"][action_mask] - no_lang["action_logits"][action_mask]).abs().mean().item()
    )
    told_logit_shift = float(
        (told_normal["action_logits"][action_mask] - no_told["action_logits"][action_mask]).abs().mean().item()
    )
    question_logit_shift = float(
        (normal["action_logits"][action_mask] - no_questions["action_logits"][action_mask]).abs().mean().item()
    )
    normal_acc = masked_accuracy(normal["action_logits"], batch["action_target"], action_mask)
    no_lang_acc = masked_accuracy(no_lang["action_logits"], batch["action_target"], action_mask)
    told_normal_acc = masked_accuracy(told_normal["action_logits"], batch["action_target"], action_mask)
    no_told_acc = masked_accuracy(no_told["action_logits"], batch["action_target"], action_mask)
    no_question_acc = masked_accuracy(no_questions["action_logits"], batch["action_target"], action_mask)
    return {
        "delayed_question_action_change_rate": action_change,
        "delayed_question_action_logit_l1": logit_shift,
        "told_fact_removal_action_change_rate": told_action_change,
        "told_fact_removal_action_logit_l1": told_logit_shift,
        "question_removal_action_change_rate": question_action_change,
        "question_removal_action_logit_l1": question_logit_shift,
        "normal_delayed_action_accuracy": normal_acc,
        "no_language_delayed_action_accuracy": no_lang_acc,
        "told_only_delayed_action_accuracy": told_normal_acc,
        "no_told_delayed_action_accuracy": no_told_acc,
        "no_question_delayed_action_accuracy": no_question_acc,
        "accuracy_delta_vs_no_language": normal_acc - no_lang_acc,
        "accuracy_delta_vs_no_told": told_normal_acc - no_told_acc,
        "accuracy_delta_vs_no_question": normal_acc - no_question_acc,
    }


def false_belief_conflict_tests(model: RecurrentLatentModel, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    outputs = run_sequence(model, batch)
    conflict = false_told_conflict_batch(batch)
    conflict_outputs = run_sequence(model, conflict)
    language_pred = outputs["language_logits"].argmax(dim=-1)
    provenance_pred = outputs["provenance_logits"].argmax(dim=-1)
    observed_mask = event_mask(batch, TOK_OBSERVE_OBJECT)
    imagined_mask = event_mask(batch, TOK_IMAGINE)
    told_mask = event_mask(batch, TOK_TOLD_GOAL)
    inferred_mask = event_mask(batch, TOK_INFER_OBJECT)
    observed_content = float(
        (language_pred[observed_mask] == batch["language_target"][observed_mask]).float().mean().item()
    )
    imagined_content = float(
        (language_pred[imagined_mask] == batch["language_target"][imagined_mask]).float().mean().item()
    )
    told_source = float((provenance_pred[told_mask] == PROV_TOLD).float().mean().item())
    inferred_source = float((provenance_pred[inferred_mask] == PROV_TOLD).float().mean().item())
    imagination_not_observed = float(
        (language_pred[imagined_mask] >= ANS_IMAGINED).float().mean().item()
    )
    delayed_mask = batch["delayed_memory_mask"]
    observed_over_false_told = masked_accuracy(
        conflict_outputs["memory_color_logits"], batch["memory_color_target"], delayed_mask
    )
    conflict_world_pos = masked_accuracy(
        conflict_outputs["world_pos_logits"], batch["world_pos_target"], batch["object_mask"]
    )
    return {
        "observed_fact_content_accuracy": observed_content,
        "imagined_alternative_content_accuracy": imagined_content,
        "told_source_accuracy": told_source,
        "inferred_source_accuracy": inferred_source,
        "imagined_not_observed_rate": imagination_not_observed,
        "observed_over_false_told_memory_accuracy": observed_over_false_told,
        "observed_over_false_told_world_pos_accuracy": conflict_world_pos,
        "mean": (
            observed_content
            + imagined_content
            + told_source
            + inferred_source
            + imagination_not_observed
            + observed_over_false_told
            + conflict_world_pos
        )
        / 7.0,
    }


def blank_input_continuation(model: RecurrentLatentModel, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
    blank = blank_continuation_batch(batch)
    outputs = run_sequence(model, blank)
    final_mask = torch.zeros_like(batch["delayed_memory_mask"])
    final_mask[:, -1] = True
    stats = latent_noncollapse_stats(outputs["latents"], outputs["language_logits"].argmax(dim=-1))
    return {
        "final_memory_accuracy": masked_accuracy(
            outputs["memory_color_logits"], batch["memory_color_target"], final_mask
        ),
        "final_object_pos_accuracy": masked_accuracy(outputs["world_pos_logits"], batch["world_pos_target"], final_mask),
        "latent_active_fraction": stats["latent_active_fraction"],
        "latent_effective_rank": stats["latent_effective_rank"],
        "latent_max_quantized_fraction": stats["latent_max_quantized_fraction"],
        "language_repetition_ratio": stats["language_repetition_ratio"],
    }


def episodic_probe_tests(
    model: RecurrentLatentModel,
    train_batch: Dict[str, torch.Tensor],
    test_batch: Dict[str, torch.Tensor],
    steps: int,
) -> Dict[str, float]:
    event_ticks = [0, 8, 11, 17, 64]

    with torch.no_grad():
        train_outputs = run_sequence(model, train_batch)
        test_outputs = run_sequence(model, test_batch)

    def samples(batch: Dict[str, torch.Tensor], outputs: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        latents = outputs["latents"][:, event_ticks, :].reshape(-1, outputs["latents"].shape[-1]).detach()
        event_index = torch.arange(len(event_ticks), device=latents.device).repeat(batch["sensory"].shape[0])
        content = batch["language_target"][:, event_ticks].reshape(-1)
        source = batch["provenance_target"][:, event_ticks].reshape(-1)
        time_bucket = event_index.clone()
        return latents, {
            "event_index": event_index,
            "content": content,
            "source": source,
            "time": time_bucket,
        }

    x_train, y_train = samples(train_batch, train_outputs)
    x_test, y_test = samples(test_batch, test_outputs)
    heads = {
        "event_index": torch.nn.Linear(x_train.shape[-1], len(event_ticks)).to(x_train.device),
        "content": torch.nn.Linear(x_train.shape[-1], int(train_batch["language_target"].max().item()) + 1).to(x_train.device),
        "source": torch.nn.Linear(x_train.shape[-1], 4).to(x_train.device),
        "time": torch.nn.Linear(x_train.shape[-1], len(event_ticks)).to(x_train.device),
    }
    params: List[torch.nn.Parameter] = []
    for head in heads.values():
        params.extend(list(head.parameters()))
    optimizer = torch.optim.AdamW(params, lr=0.03)
    for _ in range(steps):
        loss = sum(torch.nn.functional.cross_entropy(head(x_train), y_train[name]) for name, head in heads.items())
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

    result = {}
    with torch.no_grad():
        for name, head in heads.items():
            result[f"{name}_probe_accuracy"] = float((head(x_test).argmax(dim=-1) == y_test[name]).float().mean().item())
    result["mean"] = float(sum(result.values()) / len(result))
    return result


def ood_tests(model: RecurrentLatentModel, batch: Dict[str, torch.Tensor]) -> Dict[str, Dict[str, float]]:
    variants: Dict[str, Dict[str, torch.Tensor]] = {
        "smaller_map_size_3": smaller_map_batch(batch, map_size=3),
        "distractor_objects": distractor_batch(batch),
        "delayed_goal_change": delayed_goal_change_batch(batch),
        "energy_constraints": energy_constraint_batch(batch),
    }
    report = {}
    for name, variant in variants.items():
        outputs = run_sequence(model, variant)
        metrics = score_outputs(outputs, variant)
        metrics["core_score"] = aggregate_core(metrics)
        report[name] = metrics
    return report


def terminal_a_adversarial_verdict(report: Dict[str, object]) -> Dict[str, object]:
    latent_summary = report["latent_ablations"]["summary"]  # type: ignore[index]
    baseline_summary = report["baselines"]["summary"]  # type: ignore[index]
    counter = report["counterbalanced_provenance"]  # type: ignore[assignment]
    blank = report["blank_input_continuation"]  # type: ignore[assignment]
    episodic = report["multi_event_episodic_probe"]  # type: ignore[assignment]
    language = report["language_causal"]  # type: ignore[assignment]
    conflict = report["false_belief_conflict"]  # type: ignore[assignment]
    checks = {
        "latent_margin_ge_0_20": latent_summary["margin_vs_best_destructive"] >= 0.20,
        "baseline_margin_ge_0_15": baseline_summary["margin_vs_best_adversarial_baseline"] >= 0.15,
        "counterbalanced_provenance_ge_0_85": counter["mean"] >= 0.85,
        "blank_memory_ge_0_85": blank["final_memory_accuracy"] >= 0.85,
        "blank_object_pos_ge_0_85": blank["final_object_pos_accuracy"] >= 0.85,
        "blank_latent_rank_ge_8": blank["latent_effective_rank"] >= 8.0,
        "told_fact_action_change_ge_0_20": language["told_fact_removal_action_change_rate"] >= 0.20,
        "told_fact_accuracy_delta_ge_0_20": language["accuracy_delta_vs_no_told"] >= 0.20,
        "question_removal_action_logit_l1_ge_0_50": language["question_removal_action_logit_l1"] >= 0.50,
        "false_belief_conflict_mean_ge_0_85": conflict["mean"] >= 0.85,
        "observed_over_false_told_memory_ge_0_85": conflict["observed_over_false_told_memory_accuracy"] >= 0.85,
        "observed_over_false_told_pos_ge_0_85": conflict["observed_over_false_told_world_pos_accuracy"] >= 0.85,
        "episodic_probe_mean_ge_0_85": episodic["mean"] >= 0.85,
    }
    return {"passes": all(checks.values()), "checks": checks}


def evaluate_adversarial(
    checkpoint: str,
    config_name: str = "fast",
    device: DeviceLike = AUTO_DEVICE,
) -> Dict[str, object]:
    device = str(resolve_device(device))
    cfg = ADVERSARIAL_CONFIGS[config_name]
    model = load_checkpoint(checkpoint, device=device)
    model.eval()
    batch = generate_batch(cfg["batch_size"], cfg["seq_len"], cfg["seed"], device=device)
    probe_train = generate_batch(cfg["batch_size"], cfg["seq_len"], cfg["seed"] + 1000, device=device)
    probe_test = generate_batch(cfg["batch_size"], cfg["seq_len"], cfg["seed"] + 2000, device=device)
    with torch.no_grad():
        report: Dict[str, object] = {
            "config": cfg,
            "counterbalanced_provenance": counterbalanced_provenance(model, batch),
            "latent_ablations": latent_ablation_report(model, batch),
            "baselines": baseline_report(model, batch),
            "language_causal": language_causal_tests(model, batch),
            "false_belief_conflict": false_belief_conflict_tests(model, batch),
            "blank_input_continuation": blank_input_continuation(model, batch),
            "ood": ood_tests(model, batch),
        }
    report["multi_event_episodic_probe"] = episodic_probe_tests(
        model, probe_train, probe_test, steps=int(cfg["probe_steps"])
    )
    report["terminal_a_adversarial_verdict"] = terminal_a_adversarial_verdict(report)
    print(json.dumps(report, indent=2))
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", choices=sorted(ADVERSARIAL_CONFIGS), default="fast")
    parser.add_argument("--device", default=AUTO_DEVICE)
    args = parser.parse_args()
    evaluate_adversarial(args.checkpoint, args.config, args.device)


if __name__ == "__main__":
    main()
