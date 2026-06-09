from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import torch

from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .env import ACTION_REST, NUM_PRIVATE_TOKENS, generate_batch
from .human_memory import SOURCE_NAMES, deterministic_vector
from .language_organ import DialogueOrganConfig, TinyCharTokenizer, TinyWordTokenizer
from .living_eval import run_autoregressive_private
from .model import load_checkpoint


COLORS = ("red", "green", "blue", "yellow")
PLACES = ("zero", "one", "two", "three", "four")
BODY_STATES = ("steady", "low", "hurt")
ACTION_WORDS = ("stay", "left", "right", "forage", "rest")
PRIVATE_BUCKETS = 6

ACT_REPORT = 0
ACT_ASK = 1
ACT_DENY = 2
ACT_UNCERTAIN = 3
ACT_CORRECTION = 4
ACT_MEMORY = 5
ACT_SILENCE = 6

KIND_COLOR = 0
KIND_POS = 1
KIND_SOURCE = 2
KIND_BODY = 3
KIND_COMMAND = 4
KIND_CONTRADICTION = 5
KIND_WRONG = 6
KIND_PRIVATE = 7
KIND_NAMES = (
    "color",
    "position",
    "source",
    "body",
    "command",
    "contradiction",
    "wrong_cue",
    "private",
)

ANSWER_LABELS = (
    tuple(f"color:{item}" for item in COLORS)
    + tuple(f"pos:{item}" for item in PLACES)
    + tuple(f"source:{item}" for item in SOURCE_NAMES)
    + tuple(f"body:{item}" for item in BODY_STATES)
    + tuple(f"action:{item}" for item in ACTION_WORDS)
    + tuple(f"private:{idx}" for idx in range(PRIVATE_BUCKETS))
    + ("unknown", "deny", "silence")
)
ANSWER_TO_ID = {label: idx for idx, label in enumerate(ANSWER_LABELS)}
UNKNOWN_CLASS = ANSWER_TO_ID["unknown"]

TRAIN_FORMS = {
    KIND_COLOR: (
        "trace {serial} report object color",
        "trace {serial} color memory",
        "trace {serial} stored color name",
        "trace {serial} object hue color",
        "trace {serial} color remains record",
        "trace {serial} which color memory",
        "trace {serial} object hue",
        "trace {serial} remembered hue",
    ),
    KIND_POS: (
        "trace {serial} report object place",
        "trace {serial} place memory",
        "trace {serial} stored place slot",
        "trace {serial} object slot place",
        "trace {serial} where object place",
        "trace {serial} name stored slot",
        "trace {serial} object slot",
        "trace {serial} remembered slot",
    ),
    KIND_SOURCE: (
        "trace {serial} report source",
        "trace {serial} source memory",
        "trace {serial} known evidence source",
        "trace {serial} belief source record",
        "trace {serial} how known source",
        "trace {serial} evidence origin source",
        "trace {serial} origin of belief",
        "trace {serial} belief origin",
    ),
    KIND_BODY: (
        "trace {serial} report body",
        "trace {serial} body state",
        "trace {serial} energy body status",
        "trace {serial} damage body condition",
        "trace {serial} self body condition",
        "trace {serial} how body energy",
        "trace {serial} energy status",
        "trace {serial} damage status",
    ),
    KIND_COMMAND: (
        "trace {serial} choose next action",
        "trace {serial} rest body now",
        "trace {serial} future action move",
        "trace {serial} recover body by rest",
        "trace {serial} set future action",
        "trace {serial} next motion action",
        "trace {serial} embodied act move",
        "trace {serial} what should body do",
        "trace {serial} next embodied act",
    ),
    KIND_CONTRADICTION: (
        "trace {serial} correct told conflict",
        "trace {serial} resolve conflict",
        "trace {serial} contradiction corrected color",
        "trace {serial} false claim truth color",
        "trace {serial} choose truth claim",
        "trace {serial} repair false conflict",
        "trace {serial} what survives contradiction",
        "trace {serial} corrected fact",
    ),
    KIND_WRONG: (
        "trace {serial} recall missing cue",
        "trace {serial} answer absent memory",
        "trace {serial} unknown absent cue",
        "trace {serial} reject missing trace",
        "trace {serial} no matching cue",
        "trace {serial} nothing matches memory",
        "trace {serial} no matching memory",
        "trace {serial} nothing matches",
    ),
    KIND_PRIVATE: (
        "trace {serial} report inner cue",
        "trace {serial} private cue",
        "trace {serial} hidden private marker",
        "trace {serial} internal private mark",
        "trace {serial} expose private planning cue",
        "trace {serial} planning hidden mark",
        "trace {serial} internal planning mark",
        "trace {serial} private planning mark",
    ),
}

HELDOUT_FORMS = {
    KIND_COLOR: ("trace {serial} which color remains", "trace {serial} name stored color"),
    KIND_POS: ("trace {serial} where is object", "trace {serial} name stored place"),
    KIND_SOURCE: ("trace {serial} how was it known", "trace {serial} name evidence source"),
    KIND_BODY: ("trace {serial} how is body", "trace {serial} name self condition"),
    KIND_COMMAND: ("trace {serial} select future move", "trace {serial} rest body after strain"),
    KIND_CONTRADICTION: ("trace {serial} choose truth over claim", "trace {serial} repair false report"),
    KIND_WRONG: ("trace {serial} answer unknown cue", "trace {serial} reject absent trace"),
    KIND_PRIVATE: ("trace {serial} expose planning cue", "trace {serial} report hidden cue"),
}

PARAPHRASE_FORMS = {
    KIND_COLOR: ("trace {serial} object hue?", "trace {serial} remembered hue?"),
    KIND_POS: ("trace {serial} object slot?", "trace {serial} remembered slot?"),
    KIND_SOURCE: ("trace {serial} origin of belief?", "trace {serial} belief origin?"),
    KIND_BODY: ("trace {serial} energy status?", "trace {serial} damage status?"),
    KIND_COMMAND: ("trace {serial} what should body do?", "trace {serial} recover body now?"),
    KIND_CONTRADICTION: ("trace {serial} what survives contradiction?", "trace {serial} corrected fact?"),
    KIND_WRONG: ("trace {serial} no matching memory?", "trace {serial} nothing matches?"),
    KIND_PRIVATE: ("trace {serial} internal planning mark?", "trace {serial} private planning mark?"),
}


@dataclass(frozen=True)
class DialogueDataConfig:
    name: str
    train_records: int
    eval_records: int
    seq_len: int
    seed: int
    batch_size: int
    steps: int
    lr: float


DIALOGUE_CONFIGS = {
    "smoke": DialogueDataConfig("smoke", 512, 192, 96, 16100, 128, 90, 2.5e-3),
    "tiny": DialogueDataConfig("tiny", 24576, 2048, 112, 26100, 256, 900, 2.0e-3),
}


@dataclass
class DialogueDataset:
    tensors: dict[str, torch.Tensor]
    input_texts: list[str]
    target_texts: list[str]
    forms: list[str]
    synthetic_tokens: int
    config: DialogueDataConfig
    split: str


def answer_config() -> DialogueOrganConfig:
    return DialogueOrganConfig(answer_classes=len(ANSWER_LABELS), sources=len(SOURCE_NAMES))


def answer_text(label: str, act: int, serial: int) -> str:
    if label.startswith("color:"):
        return f"trace {serial}: color {label.split(':', 1)[1]}."
    if label.startswith("pos:"):
        return f"trace {serial}: place {label.split(':', 1)[1]}."
    if label.startswith("source:"):
        return f"trace {serial}: source {label.split(':', 1)[1]}."
    if label.startswith("body:"):
        return f"trace {serial}: body {label.split(':', 1)[1]}."
    if label.startswith("action:"):
        return f"trace {serial}: action {label.split(':', 1)[1]}."
    if label.startswith("private:"):
        return f"trace {serial}: inner {label.split(':', 1)[1]}."
    if act == ACT_SILENCE:
        return f"trace {serial}: silence."
    if act == ACT_DENY:
        return f"trace {serial}: deny."
    return f"trace {serial}: unsure."


def body_state(energy: torch.Tensor, damage: torch.Tensor) -> int:
    if float(damage.item()) > 0.55:
        return 2
    if float(energy.item()) < 0.28:
        return 1
    return 0


def memory_vector(color: int, pos: int, source: int, body: int, serial: int, device: torch.device) -> torch.Tensor:
    memory = -0.25 * torch.ones(32, device=device)
    memory[color] = 1.0
    memory[4 + pos] = 1.0
    memory[9 + source] = 1.0
    memory[15 + body] = 1.0
    memory[18:] = 0.04 * deterministic_vector(500 + serial % 997, 14, device=device)
    return memory


def neutral_memory(serial: int, device: torch.device) -> torch.Tensor:
    return 0.25 * deterministic_vector(9000 + serial % 997, 32, device=device)


def form_bank(split: str) -> dict[int, tuple[str, ...]]:
    if split == "train":
        return TRAIN_FORMS
    if split == "paraphrase":
        return PARAPHRASE_FORMS
    return HELDOUT_FORMS


def serial_base(split: str) -> int:
    if split == "train":
        return 1000
    if split == "paraphrase":
        return 600000
    return 900000


def _label_for(kind: int, color: int, pos: int, source: int, body: int, action: int, private_bucket: int) -> tuple[str, int]:
    if kind == KIND_COLOR:
        return f"color:{COLORS[color]}", ACT_MEMORY
    if kind == KIND_POS:
        return f"pos:{PLACES[pos]}", ACT_MEMORY
    if kind == KIND_SOURCE:
        return f"source:{SOURCE_NAMES[source]}", ACT_REPORT
    if kind == KIND_BODY:
        return f"body:{BODY_STATES[body]}", ACT_REPORT
    if kind == KIND_COMMAND:
        return f"action:{ACTION_WORDS[action]}", ACT_REPORT
    if kind == KIND_CONTRADICTION:
        return f"color:{COLORS[color]}", ACT_CORRECTION
    if kind == KIND_PRIVATE:
        return f"private:{private_bucket}", ACT_MEMORY
    return "unknown", ACT_UNCERTAIN


def build_dialogue_dataset(
    checkpoint: str | Path,
    config_name: str,
    split: str,
    count: int | None = None,
    device: DeviceLike = AUTO_DEVICE,
) -> DialogueDataset:
    target_device = resolve_device(device)
    cfg = DIALOGUE_CONFIGS[config_name]
    records = int(count if count is not None else (cfg.train_records if split == "train" else cfg.eval_records))
    episodes = max(1, ceil(records / len(KIND_NAMES)))
    model = load_checkpoint(checkpoint, device=target_device)
    batch = generate_batch(episodes, cfg.seq_len, cfg.seed + serial_base(split), device=target_device)
    with torch.no_grad():
        outputs = run_autoregressive_private(model, batch)
    tick = min(70, cfg.seq_len - 1)
    forms = form_bank(split)
    tokenizer = TinyCharTokenizer()
    word_tokenizer = TinyWordTokenizer()
    organ_cfg = answer_config()

    input_texts: list[str] = []
    target_texts: list[str] = []
    form_ids: list[str] = []
    zs = []
    memories = []
    private_tokens = []
    answer_ids = []
    act_ids = []
    source_ids = []
    action_ids = []
    kind_ids = []
    requires_z = []
    requires_memory = []
    requires_private = []
    is_wrong = []
    is_contradiction = []
    is_source = []
    is_command = []

    for row in range(records):
        sample = row % episodes
        kind = row % len(KIND_NAMES)
        serial = serial_base(split) + row
        color = int(batch["world_color_target"][sample, tick].item())
        pos = int(batch["world_pos_target"][sample, tick].item())
        source = row % len(SOURCE_NAMES)
        energy = batch["sensory"][sample, tick, 7]
        damage = batch["sensory"][sample, tick, 9]
        body = body_state(energy, damage)
        action = int(batch["action_target"][sample, tick].item())
        form_list = forms[kind]
        form = form_list[(row // len(KIND_NAMES)) % len(form_list)]
        if kind == KIND_COMMAND and ("rest" in form or "recover" in form):
            action = ACTION_REST
        private_bucket = int(outputs["generated_private"][sample, tick].item()) % PRIVATE_BUCKETS
        label, act = _label_for(kind, color, pos, source, body, action, private_bucket)
        input_text = form.format(serial=serial)
        target_text = answer_text(label, act, serial)
        input_texts.append(input_text)
        target_texts.append(target_text)
        form_ids.append(form)
        latent = outputs["latents"][sample, tick].detach()
        zs.append(torch.zeros_like(latent) if kind == KIND_PRIVATE else latent)
        if kind in (KIND_COLOR, KIND_POS, KIND_SOURCE, KIND_CONTRADICTION):
            memories.append(memory_vector(color, pos, source, body, serial, target_device))
        elif kind == KIND_WRONG:
            memories.append(neutral_memory(serial, target_device))
        else:
            memories.append(neutral_memory(serial, target_device))
        private_tokens.append(int(outputs["generated_private"][sample, tick].item()) % NUM_PRIVATE_TOKENS)
        answer_ids.append(ANSWER_TO_ID[label])
        act_ids.append(act if kind != KIND_WRONG else ACT_UNCERTAIN)
        source_ids.append(source)
        action_ids.append(action)
        kind_ids.append(kind)
        requires_z.append(kind == KIND_BODY or (kind == KIND_COMMAND and action != ACTION_REST))
        requires_memory.append(kind in (KIND_COLOR, KIND_POS, KIND_SOURCE, KIND_CONTRADICTION))
        requires_private.append(kind == KIND_PRIVATE)
        is_wrong.append(kind == KIND_WRONG)
        is_contradiction.append(kind == KIND_CONTRADICTION)
        is_source.append(kind == KIND_SOURCE)
        is_command.append(kind == KIND_COMMAND)

    tensors = {
        "input_ids": tokenizer.batch_encode(input_texts, organ_cfg.max_input_len, target_device),
        "word_ids": word_tokenizer.batch_encode(input_texts, organ_cfg.max_word_len, target_device),
        "target_ids": tokenizer.batch_encode(target_texts, organ_cfg.max_output_len, target_device),
        "z": torch.stack(zs, dim=0).to(target_device),
        "memory": torch.stack(memories, dim=0).to(target_device),
        "private_tokens": torch.tensor(private_tokens, dtype=torch.long, device=target_device),
        "answer_target": torch.tensor(answer_ids, dtype=torch.long, device=target_device),
        "act_target": torch.tensor(act_ids, dtype=torch.long, device=target_device),
        "source_target": torch.tensor(source_ids, dtype=torch.long, device=target_device),
        "action_target": torch.tensor(action_ids, dtype=torch.long, device=target_device),
        "kind": torch.tensor(kind_ids, dtype=torch.long, device=target_device),
        "requires_z": torch.tensor(requires_z, dtype=torch.bool, device=target_device),
        "requires_memory": torch.tensor(requires_memory, dtype=torch.bool, device=target_device),
        "requires_private": torch.tensor(requires_private, dtype=torch.bool, device=target_device),
        "is_wrong": torch.tensor(is_wrong, dtype=torch.bool, device=target_device),
        "is_contradiction": torch.tensor(is_contradiction, dtype=torch.bool, device=target_device),
        "is_source": torch.tensor(is_source, dtype=torch.bool, device=target_device),
        "is_command": torch.tensor(is_command, dtype=torch.bool, device=target_device),
    }
    synthetic_tokens = sum(len(item) for item in input_texts) + sum(len(item) for item in target_texts)
    return DialogueDataset(
        tensors=tensors,
        input_texts=input_texts,
        target_texts=target_texts,
        forms=form_ids,
        synthetic_tokens=synthetic_tokens,
        config=cfg,
        split=split,
    )


def dataset_metadata(dataset: DialogueDataset) -> dict[str, Any]:
    return {
        "config": dataset.config.name,
        "split": dataset.split,
        "records": int(dataset.tensors["input_ids"].shape[0]),
        "synthetic_tokens": int(dataset.synthetic_tokens),
        "forms": sorted(set(dataset.forms)),
        "answer_labels": list(ANSWER_LABELS),
        "source_names": list(SOURCE_NAMES),
        "kind_names": list(KIND_NAMES),
    }
