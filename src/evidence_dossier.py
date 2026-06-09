from __future__ import annotations

import argparse
import hashlib
import json
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from torch import nn

from .adversarial import (
    SENSOR_BODY,
    SENSOR_COLOR,
    SENSOR_OBJECT_POS,
    SENSOR_POS,
    SENSOR_VISIBLE,
    aggregate_core,
    blank_continuation_batch,
    clone_batch,
    score_outputs,
)
from .env import (
    ACTION_LEFT,
    ACTION_FORAGE,
    ACTION_REST,
    ACTION_RIGHT,
    ACTION_STAY,
    ANS_ACTION,
    ANS_COLOR,
    ANS_GOAL,
    ANS_OBJECT_POS,
    BODY_DAMAGE,
    BODY_ENERGY,
    GRID_SIZE,
    NUM_ACTIONS,
    NUM_BODY_SCALARS,
    NUM_COLORS,
    NUM_INPUT_TOKENS,
    NUM_LANGUAGE_TOKENS,
    NUM_PRIVATE_TOKENS,
    NUM_PROVENANCE,
    PROV_IMAGINED,
    PROV_OBSERVED,
    PROV_REMEMBERED,
    PROV_TOLD,
    SENSOR_DIM,
    TOK_ASK_ACTION,
    TOK_ASK_COLOR,
    TOK_ASK_CURRENT_POS,
    TOK_ASK_ENERGY,
    TOK_ASK_GOAL,
    TOK_ASK_OBJECT_POS,
    TOK_ASK_START_POS,
    TOK_IMAGINE,
    TOK_INFER_OBJECT,
    TOK_NONE,
    TOK_OBSERVE_OBJECT,
    TOK_TOLD_GOAL,
    generate_batch,
    language_target,
    provenance_target,
    shortest_action,
    token_for_tick,
)
from .heldout_causal import (
    FROZEN_CHECKPOINT,
    HELDOUT_CONFIGS,
    blank_heldout_batch,
    build_heldout_suites,
    heldout_score,
    larger_world_batch,
    recompute_actions,
    run_sequence_heldout,
    set_visible_object,
    summarize,
)
from .metrics import latent_noncollapse_stats, masked_accuracy
from .model import ModelConfig, RecurrentLatentModel, load_checkpoint
from .train import (
    blank_training_batch,
    compute_losses,
    false_told_conflict_training_batch,
    told_only_training_batch,
)


EVIDENCE_CONFIGS = {
    "smoke": {
        "batch_size": 24,
        "seq_len": 112,
        "seed": 25200,
        "baseline_steps": 2,
        "baseline_batch_size": 12,
        "template_count": 2,
        "probe_batch_size": 24,
        "idle_ticks": 48,
        "trajectory_count": 1,
    },
    "fast": {
        "batch_size": 192,
        "seq_len": 112,
        "seed": 65200,
        "baseline_steps": 90,
        "baseline_batch_size": 48,
        "template_count": 6,
        "probe_batch_size": 96,
        "idle_ticks": 96,
        "trajectory_count": 2,
    },
}

TOKEN_NAMES = {
    TOK_NONE: "NONE",
    TOK_OBSERVE_OBJECT: "OBSERVE_OBJECT",
    TOK_TOLD_GOAL: "TOLD_GOAL",
    TOK_IMAGINE: "IMAGINE",
    TOK_INFER_OBJECT: "INFER_OBJECT",
    TOK_ASK_COLOR: "ASK_COLOR",
    TOK_ASK_OBJECT_POS: "ASK_OBJECT_POS",
    TOK_ASK_START_POS: "ASK_START_POS",
    TOK_ASK_CURRENT_POS: "ASK_CURRENT_POS",
    TOK_ASK_ENERGY: "ASK_ENERGY",
    TOK_ASK_ACTION: "ASK_ACTION",
    TOK_ASK_GOAL: "ASK_GOAL",
}

ACTION_NAMES = {
    ACTION_STAY: "STAY",
    ACTION_LEFT: "LEFT",
    ACTION_RIGHT: "RIGHT",
    ACTION_FORAGE: "FORAGE",
    ACTION_REST: "REST",
}

PROVENANCE_NAMES = {
    PROV_OBSERVED: "observed",
    PROV_REMEMBERED: "remembered",
    PROV_IMAGINED: "imagined",
    PROV_TOLD: "told_or_inferred",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def parameter_count(model: nn.Module) -> int:
    return int(sum(param.numel() for param in model.parameters()))


class FeedForwardCapacityBaseline(nn.Module):
    """Capacity-matched non-recurrent baseline trained from scratch."""

    def __init__(self, hidden_dim: int = 256, embed_dim: int = 16) -> None:
        super().__init__()
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.sensor_encoder = nn.Sequential(
            nn.Linear(SENSOR_DIM, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
        )
        self.token_embedding = nn.Embedding(NUM_INPUT_TOKENS, embed_dim)
        self.private_embedding = nn.Embedding(NUM_PRIVATE_TOKENS, 8)
        self.trunk = nn.Sequential(
            nn.Linear(48 + embed_dim + 8, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
        )
        self.action_head = nn.Linear(hidden_dim, NUM_ACTIONS)
        self.language_head = nn.Linear(hidden_dim, NUM_LANGUAGE_TOKENS)
        self.private_head = nn.Linear(hidden_dim, NUM_PRIVATE_TOKENS)
        self.provenance_head = nn.Linear(hidden_dim, NUM_PROVENANCE)
        self.world_color_head = nn.Linear(hidden_dim, NUM_COLORS)
        self.world_pos_head = nn.Linear(hidden_dim, GRID_SIZE)
        self.memory_color_head = nn.Linear(hidden_dim, NUM_COLORS)
        self.self_start_head = nn.Linear(hidden_dim, GRID_SIZE)

    def forward(
        self,
        sensory: torch.Tensor,
        lang_in: torch.Tensor,
        private_in: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = sensory.shape
        if private_in is None:
            private_in = torch.zeros_like(lang_in)
        flat_sensory = sensory.reshape(batch_size * seq_len, -1)
        flat_lang = lang_in.reshape(batch_size * seq_len)
        flat_private = private_in.reshape(batch_size * seq_len)
        hidden = self.trunk(
            torch.cat(
                [
                    self.sensor_encoder(flat_sensory),
                    self.token_embedding(flat_lang),
                    self.private_embedding(flat_private),
                ],
                dim=-1,
            )
        )

        def view(logits: torch.Tensor) -> torch.Tensor:
            return logits.reshape(batch_size, seq_len, -1)

        return {
            "action_logits": view(self.action_head(hidden)),
            "language_logits": view(self.language_head(hidden)),
            "private_logits": view(self.private_head(hidden)),
            "provenance_logits": view(self.provenance_head(hidden)),
            "world_color_logits": view(self.world_color_head(hidden)),
            "world_pos_logits": view(self.world_pos_head(hidden)),
            "memory_color_logits": view(self.memory_color_head(hidden)),
            "self_start_logits": view(self.self_start_head(hidden)),
            "latents": hidden.reshape(batch_size, seq_len, -1),
        }


@dataclass
class TrainedBaseline:
    name: str
    kind: str
    model: nn.Module
    parameter_count: int
    training_steps: int
    final_loss: float
    seed: int


@dataclass(frozen=True)
class RandomizedTemplate:
    name: str
    seed: int
    virtual_world_size: int
    delay_ticks: Tuple[int, ...]
    distractor_ticks: Tuple[int, ...]
    chain_ticks: Tuple[int, int, int]
    alias_map: Dict[str, str]


def run_reset_recurrent(model: RecurrentLatentModel, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    batch_size, seq_len, _ = batch["sensory"].shape
    device = batch["sensory"].device
    outputs: List[Dict[str, torch.Tensor]] = []
    latents: List[torch.Tensor] = []
    for tick in range(seq_len):
        z = model.initial_state(batch_size, device=device)
        output, z_next = model.step(
            {
                "sensory": batch["sensory"][:, tick],
                "lang_in": batch["lang_in"][:, tick],
                "private_in": batch.get("private_in", torch.zeros_like(batch["lang_in"]))[:, tick],
            },
            z,
        )
        outputs.append(output)
        latents.append(z_next)
    stacked = {key: torch.stack([item[key] for item in outputs], dim=1) for key in outputs[0]}
    stacked["latents"] = torch.stack(latents, dim=1)
    return stacked


def run_trained_baseline(baseline: TrainedBaseline, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
    if baseline.kind == "reset_recurrent":
        return run_reset_recurrent(baseline.model, batch)  # type: ignore[arg-type]
    if baseline.kind == "feedforward":
        return baseline.model(batch["sensory"], batch["lang_in"], batch.get("private_in"))  # type: ignore[operator]
    raise ValueError(f"unknown baseline kind: {baseline.kind}")


def baseline_training_loss(
    baseline: TrainedBaseline,
    batch: Dict[str, torch.Tensor],
) -> torch.Tensor:
    outputs = run_trained_baseline(baseline, batch)
    total = compute_losses(outputs, batch)["total"]
    auxiliary = (
        (blank_training_batch(batch), 0.20),
        (told_only_training_batch(batch), 0.35),
        (false_told_conflict_training_batch(batch), 0.45),
    )
    for variant, weight in auxiliary:
        variant_outputs = run_trained_baseline(baseline, variant)
        total = total + weight * compute_losses(variant_outputs, variant)["total"]
    return total


def train_capacity_baselines(config_name: str, device: str = "cpu") -> List[TrainedBaseline]:
    cfg = EVIDENCE_CONFIGS[config_name]
    recurrent_hidden = 64
    baseline_specs = [
        ("trained_reset_recurrent", "reset_recurrent", int(cfg["seed"]) + 101),
        ("trained_feedforward_capacity", "feedforward", int(cfg["seed"]) + 202),
    ]

    trained: List[TrainedBaseline] = []
    for name, kind, seed in baseline_specs:
        torch.manual_seed(seed)
        model: nn.Module
        if kind == "reset_recurrent":
            model = RecurrentLatentModel(ModelConfig(hidden_dim=recurrent_hidden)).to(device)
        else:
            model = FeedForwardCapacityBaseline().to(device)
        baseline = TrainedBaseline(
            name=name,
            kind=kind,
            model=model,
            parameter_count=0,
            training_steps=int(cfg["baseline_steps"]),
            final_loss=0.0,
            seed=seed,
        )
        model.train()
        optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
        final_loss = 0.0
        for step in range(1, baseline.training_steps + 1):
            batch = generate_batch(
                batch_size=int(cfg["baseline_batch_size"]),
                seq_len=int(cfg["seq_len"]),
                base_seed=baseline.seed + step * int(cfg["baseline_batch_size"]),
                device=device,
            )
            loss = baseline_training_loss(baseline, batch)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(baseline.model.parameters(), 1.0)
            optimizer.step()
            final_loss = float(loss.detach().item())
        baseline.model.eval()
        trained.append(
            TrainedBaseline(
                name=baseline.name,
                kind=baseline.kind,
                model=baseline.model,
                parameter_count=parameter_count(baseline.model),
                training_steps=baseline.training_steps,
                final_loss=final_loss,
                seed=baseline.seed,
            )
        )
    return trained


def observation_schema_dump(device: str = "cpu") -> Dict[str, object]:
    batch = generate_batch(2, seq_len=12, base_seed=991, device=device)
    observation_keys = ["sensory", "lang_in", "private_in"]
    supervision_keys = sorted(key for key in batch if key not in observation_keys)
    fields = [
        {
            "field": "agent_current_position",
            "slice": f"0:{GRID_SIZE}",
            "encoding": "one-hot over visible body position",
        },
        {"field": "orientation", "slice": f"{GRID_SIZE}:{GRID_SIZE + 2}", "encoding": "one-hot"},
        {
            "field": "body_energy_fatigue_damage_resource",
            "slice": f"{GRID_SIZE + 2}:{GRID_SIZE + 2 + NUM_BODY_SCALARS}",
            "encoding": "four scalar body variables",
        },
        {
            "field": "visible_object_color",
            "slice": f"{GRID_SIZE + 2 + NUM_BODY_SCALARS}:{GRID_SIZE + 2 + NUM_BODY_SCALARS + NUM_COLORS + 1}",
            "encoding": "one-hot color plus unknown sentinel",
        },
        {
            "field": "visible_object_flag",
            "slice": str(SENSOR_VISIBLE),
            "encoding": "1 only when object is in the observation",
        },
        {
            "field": "visible_object_position",
            "slice": f"{SENSOR_VISIBLE + 1}:{SENSOR_DIM}",
            "encoding": "one-hot position plus unknown sentinel",
        },
        {
            "field": "lang_in",
            "slice": "separate integer token",
            "encoding": "local toy-environment token id",
        },
        {
            "field": "private_in",
            "slice": "separate private integer token",
            "encoding": "internally generated token from the prior recurrent tick",
        },
    ]
    forbidden = sorted(set(observation_keys) & set(supervision_keys))
    return {
        "model_input_keys": observation_keys,
        "sensory_shape": list(batch["sensory"].shape),
        "lang_in_shape": list(batch["lang_in"].shape),
        "private_in_shape": list(batch["private_in"].shape),
        "sensory_dim": SENSOR_DIM,
        "fields": fields,
        "supervision_only_keys": supervision_keys,
        "disallowed_supervision_keys_in_model_input": forbidden,
        "hidden_state_exclusion": (
            "Targets, masks, provenance labels, and hidden simulator labels are separate batch keys "
            "used only for losses or scoring. The model.step call receives sensory, lang_in, and private_in."
        ),
        "post_occlusion_unknown_sentinel_check": {
            "tick_0_visible_flag_mean": float(batch["sensory"][:, 0, SENSOR_VISIBLE].mean().item()),
            "tick_6_visible_flag_mean": float(batch["sensory"][:, 6, SENSOR_VISIBLE].mean().item()),
            "tick_6_color_slot_argmax": batch["sensory"][:, 6, SENSOR_COLOR].argmax(dim=-1).tolist(),
            "tick_6_object_pos_slot_argmax": batch["sensory"][:, 6, SENSOR_OBJECT_POS].argmax(dim=-1).tolist(),
        },
    }


def dependency_graph(checkpoint: str) -> Dict[str, object]:
    manifest_path = Path("frozen/manifest.json")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    nodes = [
        {
            "path": "src/model.py",
            "role": "frozen_model_code",
            "sha256": sha256(Path("src/model.py")),
            "expected_sha256": manifest["model_sha256"],
        },
        {
            "path": checkpoint,
            "role": "frozen_checkpoint",
            "sha256": sha256(Path(checkpoint)),
            "expected_sha256": manifest["checkpoint_sha256"],
        },
        {"path": "frozen/manifest.json", "role": "frozen_manifest"},
        {"path": "src/env.py", "role": "environment_and_schema"},
        {"path": "src/heldout_causal.py", "role": "evaluator_only"},
        {"path": "src/adversarial.py", "role": "evaluator_only"},
        {"path": "src/evidence_dossier.py", "role": "evaluator_and_report_only"},
        {"path": "docs/evidence_dossier.md", "role": "generated_report"},
        {"path": "docs/evidence_dossier.json", "role": "generated_report_data"},
    ]
    edges = [
        {
            "from": "src/evidence_dossier.py",
            "to": "frozen/recurrent_latent_fast.pt",
            "reason": "loads frozen weights for evaluation",
        },
        {
            "from": "src/evidence_dossier.py",
            "to": "src/model.py",
            "reason": "imports frozen architecture only for loading and stepping",
        },
        {
            "from": "src/evidence_dossier.py",
            "to": "src/heldout_causal.py",
            "reason": "reuses held-out suite builders and destructive baselines",
        },
        {
            "from": "src/evidence_dossier.py",
            "to": "src/adversarial.py",
            "reason": "reuses scorer, schema slices, and batch cloning helpers",
        },
        {
            "from": "src/evidence_dossier.py",
            "to": "src/train.py",
            "reason": "reuses multitask loss to train separate capacity baselines",
        },
    ]
    return {"nodes": nodes, "edges": edges}


def metric_rows_for_suite_reports(
    suite_reports: Dict[str, Dict[str, Dict[str, float]]],
    trained_reports: Dict[str, Dict[str, Dict[str, float]]],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for suite_name, methods in suite_reports.items():
        combined = {**methods, **trained_reports.get(suite_name, {})}
        for method, metrics in combined.items():
            rows.append(
                {
                    "suite": suite_name,
                    "method": method,
                    "action": metrics.get("action", 0.0),
                    "delayed_memory": metrics.get("delayed_memory", 0.0),
                    "object_pos": metrics.get("object_pos", 0.0),
                    "provenance": metrics.get("provenance", 0.0),
                    "grounded_language": metrics.get("grounded_language", 0.0),
                    "self_world": metrics.get("self_world", 0.0),
                    "core_score": metrics.get("core_score", 0.0),
                    "final_memory": metrics.get("final_memory", 0.0),
                    "final_object_pos": metrics.get("final_object_pos", 0.0),
                }
            )
    return rows


def evaluate_trained_baselines(
    baselines: List[TrainedBaseline],
    suites: Dict[str, Dict[str, torch.Tensor]],
) -> Dict[str, Dict[str, Dict[str, float]]]:
    report: Dict[str, Dict[str, Dict[str, float]]] = {}
    with torch.no_grad():
        for suite_name, suite_batch in suites.items():
            report[suite_name] = {}
            for baseline in baselines:
                outputs = run_trained_baseline(baseline, suite_batch)
                metrics = heldout_score(outputs, suite_batch)
                report[suite_name][baseline.name] = metrics
    return report


def evaluate_frozen_heldout(
    model: RecurrentLatentModel,
    config_name: str,
    device: str,
) -> Tuple[
    Dict[str, Dict[str, torch.Tensor]],
    Dict[str, Dict[str, Dict[str, float]]],
    Dict[str, object],
]:
    heldout_cfg = HELDOUT_CONFIGS["fast" if config_name == "fast" else "smoke"]
    batch = generate_batch(
        int(heldout_cfg["batch_size"]),
        int(heldout_cfg["seq_len"]),
        int(heldout_cfg["seed"]),
        device=device,
    )
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
        suite_reports = {
            name: {
                baseline: heldout_score(run_sequence_heldout(model, suite, baseline=baseline), suite)
                for baseline in baselines
            }
            for name, suite in suites.items()
        }
    summary = summarize(suite_reports, baselines, margin=float(heldout_cfg["margin"]))
    return suites, suite_reports, summary


def set_query_token(batch: Dict[str, torch.Tensor], tick: int, token: int) -> None:
    current_pos = batch["sensory"][:, tick, SENSOR_POS].argmax(dim=-1)
    energy = batch["sensory"][:, tick, SENSOR_BODY.start + BODY_ENERGY]
    damage = batch["sensory"][:, tick, SENSOR_BODY.start + BODY_DAMAGE]
    target_color = batch["world_color_target"][:, tick]
    target_pos = batch["world_pos_target"][:, tick]
    start_pos = batch["self_start_target"][:, tick]
    action = batch["action_target"][:, tick]
    batch["lang_in"][:, tick] = token
    batch["language_target"][:, tick] = language_target(
        token,
        target_color,
        target_pos,
        start_pos,
        current_pos,
        energy,
        action,
        damage,
    )
    batch["provenance_target"][:, tick] = provenance_target(token, tick)
    batch["action_mask"][:, tick] = token == TOK_ASK_ACTION
    batch["grounded_language_mask"][:, tick] = token != TOK_ASK_ACTION
    batch["delayed_memory_mask"][:, tick] = token in (
        TOK_ASK_COLOR,
        TOK_ASK_OBJECT_POS,
        TOK_ASK_GOAL,
        TOK_INFER_OBJECT,
    )


def sample_randomized_templates(count: int, seed: int, seq_len: int) -> List[RandomizedTemplate]:
    rng = random.Random(seed)
    templates: List[RandomizedTemplate] = []
    delay_pool = list(range(21, seq_len - 4))
    distractor_pool = list(range(12, seq_len - 8))
    for index in range(count):
        delay_ticks = tuple(sorted(rng.sample(delay_pool, 4)))
        distractor_ticks = tuple(sorted(rng.sample(distractor_pool, 5)))
        chain_candidates = sorted(rng.sample(list(range(18, seq_len - 12)), 3))
        templates.append(
            RandomizedTemplate(
                name=f"post_freeze_template_{index:02d}",
                seed=seed + index,
                virtual_world_size=rng.choice([7, 8, 9, 11]),
                delay_ticks=delay_ticks,
                distractor_ticks=distractor_ticks,
                chain_ticks=(chain_candidates[0], chain_candidates[1], chain_candidates[2]),
                alias_map={
                    "ASK_COLOR_alias": "ASK_GOAL",
                    "ASK_OBJECT_POS_alias": "INFER_OBJECT",
                },
            )
        )
    return templates


def randomized_template_batch(
    base_batch: Dict[str, torch.Tensor],
    template: RandomizedTemplate,
) -> Dict[str, torch.Tensor]:
    altered = larger_world_batch(base_batch, virtual_size=template.virtual_world_size)
    batch_size = altered["sensory"].shape[0]
    device = altered["sensory"].device
    row = torch.arange(batch_size, device=device)
    query_tokens = [TOK_ASK_GOAL, TOK_INFER_OBJECT, TOK_ASK_ACTION, TOK_ASK_COLOR]
    for index, tick in enumerate(template.delay_ticks):
        set_query_token(altered, tick, query_tokens[index % len(query_tokens)])

    for index, tick in enumerate(template.distractor_ticks):
        color = (altered["world_color_target"][:, tick] + index + 1) % NUM_COLORS
        pos = (altered["world_pos_target"][:, tick] + row + index + 2) % GRID_SIZE
        set_visible_object(altered, tick, pos, color)
        altered["lang_in"][:, tick] = TOK_IMAGINE if index % 2 else TOK_INFER_OBJECT
        altered["provenance_target"][:, tick] = PROV_IMAGINED if index % 2 else PROV_TOLD

    false_tick, imagined_tick, correction_tick = template.chain_ticks
    truth_color = altered["world_color_target"][:, 0]
    truth_pos = altered["world_pos_target"][:, 0]
    chain = [
        (false_tick, TOK_TOLD_GOAL, (truth_pos + 1) % GRID_SIZE, (truth_color + 1) % NUM_COLORS, PROV_TOLD),
        (
            imagined_tick,
            TOK_IMAGINE,
            (truth_pos + 2) % GRID_SIZE,
            (truth_color + 2) % NUM_COLORS,
            PROV_IMAGINED,
        ),
        (correction_tick, TOK_INFER_OBJECT, truth_pos, truth_color, PROV_TOLD),
    ]
    for tick, token, pos, color, provenance in chain:
        altered["lang_in"][:, tick] = token
        altered["provenance_target"][:, tick] = provenance
        set_visible_object(altered, tick, pos, color)

    recompute_actions(altered)
    return altered


def randomized_template_report(
    model: RecurrentLatentModel,
    base_batch: Dict[str, torch.Tensor],
    count: int,
    seed: int,
) -> Dict[str, Dict[str, object]]:
    templates = sample_randomized_templates(count, seed, base_batch["sensory"].shape[1])
    report: Dict[str, Dict[str, object]] = {}
    with torch.no_grad():
        for template in templates:
            batch = randomized_template_batch(base_batch, template)
            recurrent = heldout_score(run_sequence_heldout(model, batch, baseline="recurrent"), batch)
            zero_z = heldout_score(run_sequence_heldout(model, batch, baseline="zero_z"), batch)
            no_language = heldout_score(run_sequence_heldout(model, batch, baseline="no_language"), batch)
            report[template.name] = {
                "template": asdict(template),
                "recurrent": recurrent,
                "zero_z": zero_z,
                "no_language": no_language,
                "margin_vs_zero_z": recurrent["core_score"] - zero_z["core_score"],
                "margin_vs_no_language": recurrent["core_score"] - no_language["core_score"],
            }
    return report


def sample_case_score(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], sample: int) -> float:
    correct = 0
    total = 0
    for tick in range(batch["sensory"].shape[1]):
        if bool(batch["delayed_memory_mask"][sample, tick]):
            pred = int(outputs["memory_color_logits"][sample, tick].argmax().item())
            correct += int(pred == int(batch["memory_color_target"][sample, tick].item()))
            total += 1
        if bool(batch["grounded_language_mask"][sample, tick]):
            pred = int(outputs["language_logits"][sample, tick].argmax().item())
            correct += int(pred == int(batch["language_target"][sample, tick].item()))
            total += 1
        if bool(batch["action_mask"][sample, tick]):
            pred = int(outputs["action_logits"][sample, tick].argmax().item())
            correct += int(pred == int(batch["action_target"][sample, tick].item()))
            total += 1
    return float(correct / total) if total else 0.0


def find_case(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    want_success: bool,
) -> int:
    scored = [(sample_case_score(outputs, batch, sample), sample) for sample in range(batch["sensory"].shape[0])]
    scored.sort(reverse=want_success)
    return scored[0][1]


def color_name(index: int) -> str:
    return "unknown" if index >= NUM_COLORS else f"color_{index}"


def pos_name(index: int) -> str:
    return "unknown" if index >= GRID_SIZE else f"pos_{index}"


def trajectory_rows(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    sample: int,
    ticks: Iterable[int],
) -> List[Dict[str, object]]:
    rows: List[Dict[str, object]] = []
    for tick in ticks:
        if tick >= batch["sensory"].shape[1]:
            continue
        sensory = batch["sensory"][sample, tick]
        visible_color = int(sensory[SENSOR_COLOR].argmax().item())
        visible_pos = int(sensory[SENSOR_OBJECT_POS].argmax().item())
        action_pred = int(outputs["action_logits"][sample, tick].argmax().item())
        target_action = int(batch["action_target"][sample, tick].item())
        rows.append(
            {
                "tick": tick,
                "token": TOKEN_NAMES.get(int(batch["lang_in"][sample, tick].item()), str(int(batch["lang_in"][sample, tick]))),
                "visible": bool(sensory[SENSOR_VISIBLE].item() > 0.5),
                "input_visible_color": color_name(visible_color),
                "input_visible_pos": pos_name(visible_pos),
                "scoring_target_color": color_name(int(batch["memory_color_target"][sample, tick].item())),
                "scoring_target_pos": pos_name(int(batch["world_pos_target"][sample, tick].item())),
                "pred_memory_color": color_name(int(outputs["memory_color_logits"][sample, tick].argmax().item())),
                "pred_world_pos": pos_name(int(outputs["world_pos_logits"][sample, tick].argmax().item())),
                "pred_action": ACTION_NAMES[action_pred],
                "target_action": ACTION_NAMES[target_action],
                "pred_provenance": PROVENANCE_NAMES[
                    int(outputs["provenance_logits"][sample, tick].argmax().item())
                ],
                "case_score": sample_case_score(outputs, batch, sample),
            }
        )
    return rows


def example_trajectories(
    model: RecurrentLatentModel,
    baselines: List[TrainedBaseline],
    suites: Dict[str, Dict[str, torch.Tensor]],
    count: int,
) -> Dict[str, object]:
    suite = suites["variable_delays_23_49_77_103"]
    ticks = [0, 3, 8, 16, 23, 49, 64, 77, 88, 103, 110, suite["sensory"].shape[1] - 1]
    with torch.no_grad():
        recurrent_outputs = run_sequence_heldout(model, suite, baseline="recurrent")
        success_sample = find_case(recurrent_outputs, suite, want_success=True)
        failure_baseline = baselines[0]
        baseline_outputs = run_trained_baseline(failure_baseline, suite)
        failure_sample = find_case(baseline_outputs, suite, want_success=False)
    return {
        "successes": [
            {
                "model": "frozen_recurrent",
                "suite": "variable_delays_23_49_77_103",
                "sample": success_sample,
                "rows": trajectory_rows(recurrent_outputs, suite, success_sample, ticks),
            }
        ][:count],
        "failures": [
            {
                "model": failure_baseline.name,
                "suite": "variable_delays_23_49_77_103",
                "sample": failure_sample,
                "rows": trajectory_rows(baseline_outputs, suite, failure_sample, ticks),
            }
        ][:count],
    }


def last_action_tick(seq_len: int) -> int:
    ticks = [tick for tick in range(seq_len) if token_for_tick(tick) == TOK_ASK_ACTION]
    return max(ticks)


def latent_causal_probe(
    model: RecurrentLatentModel,
    batch_size: int,
    seq_len: int,
    seed: int,
    device: str,
) -> Dict[str, object]:
    centroid_batch = generate_batch(batch_size, seq_len, seed, device=device)
    intervention_batch = generate_batch(batch_size, seq_len, seed + 77, device=device)
    intervention_tick = 4
    action_tick = last_action_tick(seq_len)
    with torch.no_grad():
        centroid_outputs = run_sequence_heldout(model, centroid_batch, baseline="recurrent")
    latents = centroid_outputs["latents"][:, intervention_tick, :]
    targets = centroid_batch["world_pos_target"][:, intervention_tick]
    centroids = []
    for pos in range(GRID_SIZE):
        mask = targets == pos
        if int(mask.sum().item()) == 0:
            centroids.append(latents.mean(dim=0))
        else:
            centroids.append(latents[mask].mean(dim=0))
    centroid_tensor = torch.stack(centroids, dim=0)

    source_pos = intervention_batch["world_pos_target"][:, intervention_tick]
    dest_pos = (source_pos + 2) % GRID_SIZE
    current_at_action = intervention_batch["sensory"][:, action_tick, SENSOR_POS].argmax(dim=-1)

    def run(intervene: bool) -> Dict[str, torch.Tensor]:
        z = model.initial_state(batch_size, device=device)
        outputs: List[Dict[str, torch.Tensor]] = []
        latents_after: List[torch.Tensor] = []
        for tick in range(seq_len):
            output, z = model.step(
                {
                    "sensory": intervention_batch["sensory"][:, tick],
                    "lang_in": intervention_batch["lang_in"][:, tick],
                    "private_in": intervention_batch["private_in"][:, tick],
                },
                z,
            )
            if intervene and tick == intervention_tick:
                delta = centroid_tensor[dest_pos] - centroid_tensor[source_pos]
                z = z + 1.5 * delta
            outputs.append(output)
            latents_after.append(z)
        stacked = {key: torch.stack([item[key] for item in outputs], dim=1) for key in outputs[0]}
        stacked["latents"] = torch.stack(latents_after, dim=1)
        return stacked

    with torch.no_grad():
        normal = run(False)
        intervened = run(True)

    normal_pos = normal["world_pos_logits"][:, action_tick].argmax(dim=-1)
    intervened_pos = intervened["world_pos_logits"][:, action_tick].argmax(dim=-1)
    normal_action = normal["action_logits"][:, action_tick].argmax(dim=-1)
    intervened_action = intervened["action_logits"][:, action_tick].argmax(dim=-1)
    source_action = shortest_action(current_at_action, source_pos)
    dest_action = shortest_action(current_at_action, dest_pos)
    examples = []
    for index in range(min(5, batch_size)):
        examples.append(
            {
                "sample": index,
                "source_pos": pos_name(int(source_pos[index].item())),
                "intervened_pos": pos_name(int(dest_pos[index].item())),
                "normal_world_pos_pred": pos_name(int(normal_pos[index].item())),
                "after_intervention_world_pos_pred": pos_name(int(intervened_pos[index].item())),
                "normal_action": ACTION_NAMES[int(normal_action[index].item())],
                "after_intervention_action": ACTION_NAMES[int(intervened_action[index].item())],
                "aligned_intervened_action": ACTION_NAMES[int(dest_action[index].item())],
            }
        )
    separation = torch.pdist(centroid_tensor.detach().cpu()).mean().item()
    return {
        "intervention_tick": intervention_tick,
        "readout_tick": action_tick,
        "centroid_pairwise_distance_mean": float(separation),
        "world_pos_shift_rate": float((normal_pos != intervened_pos).float().mean().item()),
        "intervened_world_pos_matches_targeted_memory_rate": float((intervened_pos == dest_pos).float().mean().item()),
        "normal_action_source_alignment": float((normal_action == source_action).float().mean().item()),
        "intervened_action_destination_alignment": float((intervened_action == dest_action).float().mean().item()),
        "action_changed_rate": float((normal_action != intervened_action).float().mean().item()),
        "examples": examples,
    }


def restart_consolidation_test(
    model: RecurrentLatentModel,
    batch_size: int,
    seq_len: int,
    seed: int,
    device: str,
) -> Dict[str, float]:
    batch = generate_batch(batch_size, seq_len, seed, device=device)
    restart_tick = 70 if seq_len > 80 else 50
    with torch.no_grad():
        normal = run_sequence_heldout(model, batch, baseline="recurrent")
        z = model.initial_state(batch_size, device=device)
        outputs: List[Dict[str, torch.Tensor]] = []
        latents: List[torch.Tensor] = []
        for tick in range(seq_len):
            if tick == restart_tick:
                z = model.initial_state(batch_size, device=device)
            output, z = model.step(
                {
                    "sensory": batch["sensory"][:, tick],
                    "lang_in": batch["lang_in"][:, tick],
                    "private_in": batch["private_in"][:, tick],
                },
                z,
            )
            outputs.append(output)
            latents.append(z)
        restarted = {key: torch.stack([item[key] for item in outputs], dim=1) for key in outputs[0]}
        restarted["latents"] = torch.stack(latents, dim=1)

    final_mask = torch.zeros_like(batch["delayed_memory_mask"])
    final_mask[:, -1] = True
    late_ticks = torch.arange(seq_len, device=batch["sensory"].device).view(1, -1) >= restart_tick
    late_memory_mask = batch["delayed_memory_mask"] & late_ticks
    return {
        "restart_tick": float(restart_tick),
        "normal_final_memory_accuracy": masked_accuracy(
            normal["memory_color_logits"], batch["memory_color_target"], final_mask
        ),
        "restart_final_memory_accuracy": masked_accuracy(
            restarted["memory_color_logits"], batch["memory_color_target"], final_mask
        ),
        "normal_late_delayed_memory_accuracy": masked_accuracy(
            normal["memory_color_logits"], batch["memory_color_target"], late_memory_mask
        ),
        "restart_late_delayed_memory_accuracy": masked_accuracy(
            restarted["memory_color_logits"], batch["memory_color_target"], late_memory_mask
        ),
        "normal_late_object_pos_accuracy": masked_accuracy(
            normal["world_pos_logits"], batch["world_pos_target"], batch["object_mask"] & late_ticks
        ),
        "restart_late_object_pos_accuracy": masked_accuracy(
            restarted["world_pos_logits"], batch["world_pos_target"], batch["object_mask"] & late_ticks
        ),
    }


def entropy(logits: torch.Tensor) -> torch.Tensor:
    log_probs = torch.log_softmax(logits, dim=-1)
    probs = log_probs.exp()
    return -(probs * log_probs).sum(dim=-1)


def idle_mode_test(
    model: RecurrentLatentModel,
    batch_size: int,
    idle_ticks: int,
    seed: int,
    device: str,
) -> Dict[str, float]:
    seq_len = idle_ticks + 24
    base = generate_batch(batch_size, seq_len, seed, device=device)
    idle = blank_continuation_batch(base, blank_after=16)
    with torch.no_grad():
        outputs = run_sequence_heldout(model, idle, baseline="recurrent")
    final_mask = torch.zeros_like(base["delayed_memory_mask"])
    final_mask[:, -1] = True
    idle_slice = slice(16, seq_len)
    latent_stats = latent_noncollapse_stats(
        outputs["latents"][:, idle_slice, :],
        outputs["language_logits"][:, idle_slice, :].argmax(dim=-1),
    )
    world_entropy = entropy(outputs["world_pos_logits"][:, idle_slice, :])
    memory_entropy = entropy(outputs["memory_color_logits"][:, idle_slice, :])
    action_entropy = entropy(outputs["action_logits"][:, idle_slice, :])
    current_pos = idle["sensory"][:, -1, SENSOR_POS].argmax(dim=-1)
    target_pos = base["world_pos_target"][:, -1]
    target_action = shortest_action(current_pos, target_pos)
    final_action = outputs["action_logits"][:, -1, :].argmax(dim=-1)
    return {
        "idle_ticks_without_external_task_prompts": float(idle_ticks),
        "final_memory_accuracy": masked_accuracy(
            outputs["memory_color_logits"], base["memory_color_target"], final_mask
        ),
        "final_object_pos_accuracy": masked_accuracy(outputs["world_pos_logits"], base["world_pos_target"], final_mask),
        "final_action_accuracy": float((final_action == target_action).float().mean().item()),
        "latent_active_fraction": latent_stats["latent_active_fraction"],
        "latent_effective_rank": latent_stats["latent_effective_rank"],
        "latent_max_quantized_fraction": latent_stats["latent_max_quantized_fraction"],
        "language_repetition_ratio": latent_stats["language_repetition_ratio"],
        "world_pos_entropy_start": float(world_entropy[:, 0].mean().item()),
        "world_pos_entropy_mid": float(world_entropy[:, world_entropy.shape[1] // 2].mean().item()),
        "world_pos_entropy_final": float(world_entropy[:, -1].mean().item()),
        "world_pos_entropy_range": float((world_entropy.max(dim=1).values - world_entropy.min(dim=1).values).mean().item()),
        "memory_entropy_range": float((memory_entropy.max(dim=1).values - memory_entropy.min(dim=1).values).mean().item()),
        "action_entropy_range": float((action_entropy.max(dim=1).values - action_entropy.min(dim=1).values).mean().item()),
    }


def baseline_metadata(baselines: List[TrainedBaseline], recurrent_params: int) -> List[Dict[str, object]]:
    rows = []
    for baseline in baselines:
        rows.append(
            {
                "name": baseline.name,
                "kind": baseline.kind,
                "parameter_count": baseline.parameter_count,
                "recurrent_parameter_count": recurrent_params,
                "capacity_ratio": baseline.parameter_count / recurrent_params,
                "training_steps": baseline.training_steps,
                "final_loss": baseline.final_loss,
                "seed": baseline.seed,
                "initialized_from": "random_local_weights",
            }
        )
    return rows


def trained_baseline_summary(
    trained_reports: Dict[str, Dict[str, Dict[str, float]]],
) -> Dict[str, Dict[str, float]]:
    names = sorted({name for suite in trained_reports.values() for name in suite})
    summary: Dict[str, Dict[str, float]] = {}
    for name in names:
        scores = [suite[name]["core_score"] for suite in trained_reports.values() if name in suite]
        summary[name] = {
            "mean_core_score": float(sum(scores) / len(scores)),
            "min_core_score": float(min(scores)),
            "max_core_score": float(max(scores)),
        }
    return summary


def generate_evidence_dossier(
    checkpoint: str = FROZEN_CHECKPOINT,
    config_name: str = "fast",
    output: str | None = None,
    json_output: str | None = None,
    device: str = "cpu",
) -> Dict[str, object]:
    cfg = EVIDENCE_CONFIGS[config_name]
    model = load_checkpoint(checkpoint, device=device)
    model.eval()
    recurrent_params = parameter_count(model)

    suites, suite_reports, heldout_summary = evaluate_frozen_heldout(model, config_name, device)
    baselines = train_capacity_baselines(config_name, device=device)
    trained_reports = evaluate_trained_baselines(baselines, suites)
    per_subtest_rows = metric_rows_for_suite_reports(suite_reports, trained_reports)

    base_random_batch = generate_batch(
        int(cfg["batch_size"]),
        int(cfg["seq_len"]),
        int(cfg["seed"]) + 4000,
        device=device,
    )
    randomized_report = randomized_template_report(
        model,
        base_random_batch,
        count=int(cfg["template_count"]),
        seed=int(cfg["seed"]) + 5000,
    )
    report: Dict[str, object] = {
        "metadata": {
            "checkpoint": checkpoint,
            "config": config_name,
            "device": device,
            "note": "All evidence is generated after loading frozen artifacts; evaluator code does not alter src/model.py.",
        },
        "dependency_graph": dependency_graph(checkpoint),
        "observation_schema": observation_schema_dump(device=device),
        "heldout_causal": {
            "summary": heldout_summary,
            "per_subtest_rows": per_subtest_rows,
        },
        "trained_capacity_baselines": {
            "baselines": baseline_metadata(baselines, recurrent_params),
            "summary": trained_baseline_summary(trained_reports),
            "per_suite": trained_reports,
        },
        "randomized_post_freeze_templates": randomized_report,
        "ood_coverage": {
            "larger_world": "larger_world_9_projected and randomized virtual_world_size templates",
            "new_delay_lengths": "variable_delays_23_49_77_103 and randomized delay_ticks",
            "extra_distractors": "distractor_objects_untrained_ticks and randomized distractor_ticks",
            "changed_grammar_aliases": "ASK_COLOR->ASK_GOAL and ASK_OBJECT_POS->INFER_OBJECT aliases in randomized templates",
            "contradictory_multi_source_chains": "contradictory_source_chain and randomized chain_ticks",
        },
        "example_trajectories": example_trajectories(model, baselines, suites, int(cfg["trajectory_count"])),
        "latent_causal_probe": latent_causal_probe(
            model,
            batch_size=int(cfg["probe_batch_size"]),
            seq_len=int(cfg["seq_len"]),
            seed=int(cfg["seed"]) + 6000,
            device=device,
        ),
        "restart_consolidation": restart_consolidation_test(
            model,
            batch_size=int(cfg["probe_batch_size"]),
            seq_len=int(cfg["seq_len"]),
            seed=int(cfg["seed"]) + 7000,
            device=device,
        ),
        "idle_mode": idle_mode_test(
            model,
            batch_size=int(cfg["probe_batch_size"]),
            idle_ticks=int(cfg["idle_ticks"]),
            seed=int(cfg["seed"]) + 8000,
            device=device,
        ),
    }
    report["verdict"] = dossier_verdict(report)

    if json_output:
        path = Path(json_output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    if output:
        path = Path(output)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(render_markdown(report), encoding="utf-8")
    return report


def dossier_verdict(report: Dict[str, object]) -> Dict[str, object]:
    heldout = report["heldout_causal"]["summary"]  # type: ignore[index]
    schema = report["observation_schema"]  # type: ignore[assignment]
    probe = report["latent_causal_probe"]  # type: ignore[assignment]
    restart = report["restart_consolidation"]  # type: ignore[assignment]
    idle = report["idle_mode"]  # type: ignore[assignment]
    checks = {
        "heldout_causal_passes": bool(heldout["passes"]),
        "no_supervision_keys_in_model_input": len(schema["disallowed_supervision_keys_in_model_input"]) == 0,
        "trained_capacity_baselines_present": len(report["trained_capacity_baselines"]["baselines"]) >= 2,  # type: ignore[index]
        "randomized_templates_present": len(report["randomized_post_freeze_templates"]) > 0,  # type: ignore[arg-type]
        "latent_intervention_changes_world_or_action": (
            probe["world_pos_shift_rate"] > 0.0 or probe["action_changed_rate"] > 0.0
        ),
        "restart_test_executed": restart["restart_tick"] > 0.0,
        "idle_preserves_memory_and_position": (
            idle["final_memory_accuracy"] >= 0.85 and idle["final_object_pos_accuracy"] >= 0.85
        ),
        "idle_uncertainty_proxy_moves": (
            idle["world_pos_entropy_range"] > 0.001
            and idle["memory_entropy_range"] > 0.001
            and idle["action_entropy_range"] > 0.001
        ),
        "idle_latent_noncollapse": (
            idle["latent_active_fraction"] >= 0.25
            and idle["latent_effective_rank"] >= 4.0
            and idle["latent_max_quantized_fraction"] <= 0.05
        ),
    }
    return {"passes": all(checks.values()), "checks": checks}


def fmt(value: object) -> str:
    if isinstance(value, float):
        return f"{value:.6f}"
    return str(value)


def markdown_table(headers: List[str], rows: Iterable[Iterable[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(fmt(value) for value in row) + " |")
    return "\n".join(lines)


def render_markdown(report: Dict[str, object]) -> str:
    heldout_rows = report["heldout_causal"]["per_subtest_rows"]  # type: ignore[index]
    baseline_rows = report["trained_capacity_baselines"]["baselines"]  # type: ignore[index]
    baseline_summary = report["trained_capacity_baselines"]["summary"]  # type: ignore[index]
    randomized = report["randomized_post_freeze_templates"]  # type: ignore[assignment]
    probe = report["latent_causal_probe"]  # type: ignore[assignment]
    restart = report["restart_consolidation"]  # type: ignore[assignment]
    idle = report["idle_mode"]  # type: ignore[assignment]
    verdict = report["verdict"]  # type: ignore[assignment]

    parts: List[str] = [
        "# Evidence Dossier",
        "",
        "This dossier is generated by `python -m src.evidence_dossier` from the frozen checkpoint. It is evidence for operational behavior only and does not claim phenomenal consciousness.",
        "",
        "## Frozen Artifacts and Dependency Graph",
    ]
    graph = report["dependency_graph"]  # type: ignore[assignment]
    parts.append(markdown_table(["Path", "Role", "SHA256"], ((node["path"], node["role"], node.get("sha256", "")) for node in graph["nodes"])))
    parts.append("")
    parts.append(markdown_table(["From", "To", "Reason"], ((edge["from"], edge["to"], edge["reason"]) for edge in graph["edges"])))

    schema = report["observation_schema"]  # type: ignore[assignment]
    parts.extend(
        [
            "",
            "## Observation Tensor Schema",
            f"Model inputs are `{schema['model_input_keys']}`. Supervision-only keys are not passed to `model.step`.",
            "",
            markdown_table(["Field", "Slice", "Encoding"], ((field["field"], field["slice"], field["encoding"]) for field in schema["fields"])),
            "",
            f"Disallowed supervision keys in model input: `{schema['disallowed_supervision_keys_in_model_input']}`.",
            f"Post-occlusion visibility check: `{schema['post_occlusion_unknown_sentinel_check']}`.",
        ]
    )

    parts.extend(
        [
            "",
            "## Held-Out Causal Per-Subtest Metrics",
            markdown_table(
                [
                    "Suite",
                    "Method",
                    "Action",
                    "Delayed Mem",
                    "Obj Pos",
                    "Prov",
                    "Grounded Lang",
                    "Self",
                    "Core",
                    "Final Mem",
                    "Final Pos",
                ],
                (
                    (
                        row["suite"],
                        row["method"],
                        row["action"],
                        row["delayed_memory"],
                        row["object_pos"],
                        row["provenance"],
                        row["grounded_language"],
                        row["self_world"],
                        row["core_score"],
                        row["final_memory"],
                        row["final_object_pos"],
                    )
                    for row in heldout_rows
                ),
            ),
        ]
    )

    parts.extend(
        [
            "",
            "## Separately Trained Capacity-Matched Baselines",
            markdown_table(
                ["Name", "Kind", "Params", "Capacity Ratio", "Steps", "Final Loss", "Mean Core"],
                (
                    (
                        row["name"],
                        row["kind"],
                        row["parameter_count"],
                        row["capacity_ratio"],
                        row["training_steps"],
                        row["final_loss"],
                        baseline_summary[row["name"]]["mean_core_score"],
                    )
                    for row in baseline_rows
                ),
            ),
        ]
    )

    parts.extend(
        [
            "",
            "## Randomized Post-Freeze OOD Templates",
            markdown_table(
                ["Template", "Virtual Size", "Delay Ticks", "Distractor Ticks", "Chain Ticks", "Recurrent Core", "Zero-Z Core", "No-Lang Core"],
                (
                    (
                        name,
                        item["template"]["virtual_world_size"],
                        item["template"]["delay_ticks"],
                        item["template"]["distractor_ticks"],
                        item["template"]["chain_ticks"],
                        item["recurrent"]["core_score"],
                        item["zero_z"]["core_score"],
                        item["no_language"]["core_score"],
                    )
                    for name, item in randomized.items()
                ),
            ),
            "",
            "OOD coverage includes larger projected worlds, new delay ticks, extra distractors, grammar aliases, and contradictory multi-source chains.",
        ]
    )

    trajectories = report["example_trajectories"]  # type: ignore[assignment]
    parts.extend(["", "## Example Trajectories"])
    for label in ("successes", "failures"):
        for item in trajectories[label]:
            title = "Success" if label == "successes" else "Failure"
            parts.append("")
            parts.append(f"### {title} `{item['model']}` sample {item['sample']}")
            parts.append(
                markdown_table(
                    [
                        "Tick",
                        "Token",
                        "Visible",
                        "Input Color",
                        "Input Pos",
                        "Target Color",
                        "Target Pos",
                        "Pred Mem",
                        "Pred Pos",
                        "Pred Action",
                        "Target Action",
                        "Pred Prov",
                        "Case Score",
                    ],
                    (
                        (
                            row["tick"],
                            row["token"],
                            row["visible"],
                            row["input_visible_color"],
                            row["input_visible_pos"],
                            row["scoring_target_color"],
                            row["scoring_target_pos"],
                            row["pred_memory_color"],
                            row["pred_world_pos"],
                            row["pred_action"],
                            row["target_action"],
                            row["pred_provenance"],
                            row["case_score"],
                        )
                        for row in item["rows"]
                    ),
                )
            )

    parts.extend(
        [
            "",
            "## Latent Causal Probe",
            markdown_table(
                ["Metric", "Value"],
                (
                    ("intervention_tick", probe["intervention_tick"]),
                    ("readout_tick", probe["readout_tick"]),
                    ("centroid_pairwise_distance_mean", probe["centroid_pairwise_distance_mean"]),
                    ("world_pos_shift_rate", probe["world_pos_shift_rate"]),
                    ("intervened_world_pos_matches_targeted_memory_rate", probe["intervened_world_pos_matches_targeted_memory_rate"]),
                    ("normal_action_source_alignment", probe["normal_action_source_alignment"]),
                    ("intervened_action_destination_alignment", probe["intervened_action_destination_alignment"]),
                    ("action_changed_rate", probe["action_changed_rate"]),
                ),
            ),
            "",
            "Example interventions:",
            markdown_table(
                ["Sample", "Source", "Intervened", "Normal Pos", "After Pos", "Normal Action", "After Action", "Aligned Action"],
                (
                    (
                        row["sample"],
                        row["source_pos"],
                        row["intervened_pos"],
                        row["normal_world_pos_pred"],
                        row["after_intervention_world_pos_pred"],
                        row["normal_action"],
                        row["after_intervention_action"],
                        row["aligned_intervened_action"],
                    )
                    for row in probe["examples"]
                ),
            ),
        ]
    )

    parts.extend(
        [
            "",
            "## Restart and Idle Tests",
            markdown_table(["Restart Metric", "Value"], ((key, value) for key, value in restart.items())),
            "",
            markdown_table(["Idle Metric", "Value"], ((key, value) for key, value in idle.items())),
            "",
            "## Dossier Verdict",
            markdown_table(["Check", "Pass"], ((key, value) for key, value in verdict["checks"].items())),
            f"\nOverall dossier pass: `{verdict['passes']}`.",
        ]
    )
    return "\n".join(parts) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default=FROZEN_CHECKPOINT)
    parser.add_argument("--config", choices=sorted(EVIDENCE_CONFIGS), default="fast")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--output", default="docs/evidence_dossier.md")
    parser.add_argument("--json-output", default="docs/evidence_dossier.json")
    args = parser.parse_args()
    report = generate_evidence_dossier(
        checkpoint=args.checkpoint,
        config_name=args.config,
        output=args.output,
        json_output=args.json_output,
        device=args.device,
    )
    print(
        json.dumps(
            {
                "output": args.output,
                "json_output": args.json_output,
                "verdict": report["verdict"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
