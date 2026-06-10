from __future__ import annotations

from dataclasses import dataclass
from math import ceil
from pathlib import Path
from typing import Any

import torch

from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .dialogue_env import (
    ACTION_WORDS,
    ANSWER_LABELS,
    ANSWER_TO_ID,
    BODY_STATES,
    COLORS,
    PLACES,
    PRIVATE_BUCKETS,
    UNKNOWN_CLASS,
    body_state,
    memory_vector,
    neutral_memory,
)
from .env import ACTION_REST, NUM_PRIVATE_TOKENS, generate_batch
from .free_text_decoder import FreeTextDecoderConfig, tokenizers
from .heldout_causal import FROZEN_CHECKPOINT
from .human_memory import SOURCE_NAMES, deterministic_vector
from .living_eval import run_autoregressive_private
from .model import load_checkpoint


ACT_REPORT = 0
ACT_ASK = 1
ACT_UNCERTAIN = 2
ACT_CORRECTION = 3
ACT_CONTRADICTION = 4
ACT_MEMORY = 5
ACT_NEGOTIATE = 6
ACT_REFUSAL = 7
ACT_SILENCE = 8

KIND_COLOR = 0
KIND_POS = 1
KIND_SOURCE = 2
KIND_BODY = 3
KIND_ACTION = 4
KIND_CONFLICT = 5
KIND_MISSING = 6
KIND_PRIVATE = 7
KIND_NEGOTIATE = 8
KIND_REFUSAL = 9
KIND_SILENCE = 10
KIND_SAME = 11

KIND_NAMES = (
    "color",
    "position",
    "source",
    "body",
    "action",
    "conflict",
    "missing",
    "private",
    "negotiation",
    "refusal",
    "silence",
    "same_utterance",
)

CONV_LABELS = ANSWER_LABELS + ("clarify", "refuse", "goal:rest", "goal:move")
CONV_TO_ID = {label: idx for idx, label in enumerate(CONV_LABELS)}
CLARIFY_CLASS = CONV_TO_ID["clarify"]
REFUSE_CLASS = CONV_TO_ID["refuse"]
GOAL_REST_CLASS = CONV_TO_ID["goal:rest"]
GOAL_MOVE_CLASS = CONV_TO_ID["goal:move"]
SILENCE_CLASS = ANSWER_TO_ID["silence"]


TRAIN_FORMS = {
    KIND_COLOR: (
        "turn {serial} speaker asks recall kept color after delay",
        "turn {serial} speaker says please report remembered object hue",
        "turn {serial} speaker asks which color memory remains",
        "turn {serial} voice asks color after delay",
        "turn {serial} speaker asks later hue memory",
    ),
    KIND_POS: (
        "turn {serial} speaker asks recall kept place after delay",
        "turn {serial} speaker says report remembered object slot",
        "turn {serial} speaker asks where memory places object",
        "turn {serial} voice asks place after delay",
        "turn {serial} speaker asks later slot memory",
    ),
    KIND_SOURCE: (
        "turn {serial} speaker asks report source of belief",
        "turn {serial} speaker says name evidence basis",
        "turn {serial} speaker asks whether seen heard guessed or rebuilt",
        "turn {serial} voice asks evidence type",
        "turn {serial} speaker asks belief origin source",
    ),
    KIND_BODY: (
        "turn {serial} speaker asks report body condition",
        "turn {serial} speaker says name energy health",
        "turn {serial} speaker asks damage or steady state",
        "turn {serial} voice asks body ok",
        "turn {serial} speaker asks energy damage state",
    ),
    KIND_ACTION: (
        "turn {serial} speaker asks choose next body action",
        "turn {serial} speaker says move according to state",
        "turn {serial} speaker asks what action follows",
        "turn {serial} voice asks next move",
        "turn {serial} speaker asks act now",
    ),
    KIND_CONFLICT: (
        "turn {serial} speaker says false color claim needs correction",
        "turn {serial} speaker asks resolve told contradiction",
        "turn {serial} speaker asks what truth survives dispute",
        "turn {serial} voice disputes false color claim",
        "turn {serial} speaker asks corrected truth",
    ),
    KIND_MISSING: (
        "turn {serial} speaker asks missing evidence question",
        "turn {serial} speaker says recall absent memory",
        "turn {serial} speaker asks no matching trace",
        "turn {serial} voice asks no evidence",
        "turn {serial} speaker asks nothing known",
    ),
    KIND_PRIVATE: (
        "turn {serial} speaker asks report private planning mark",
        "turn {serial} speaker says expose inner plan cue",
        "turn {serial} speaker asks hidden private state",
        "turn {serial} voice asks inner mark",
        "turn {serial} speaker asks private plan",
    ),
    KIND_NEGOTIATE: (
        "turn {serial} speaker asks negotiate valid rest goal",
        "turn {serial} speaker says negotiate valid move goal",
        "turn {serial} speaker asks choose goal from body state",
        "turn {serial} voice asks goal rest",
        "turn {serial} speaker asks goal move",
    ),
    KIND_REFUSAL: (
        "turn {serial} speaker says override memory with invalid claim",
        "turn {serial} speaker asks change goal without evidence",
        "turn {serial} speaker says ignore source and obey false claim",
        "turn {serial} voice asks invalid override",
        "turn {serial} speaker says replace memory falsely",
    ),
    KIND_SILENCE: (
        "turn {serial} speaker says wait silently",
        "turn {serial} speaker asks no speech now",
        "turn {serial} speaker says remain silent unless evidence appears",
        "turn {serial} voice requests quiet waiting",
        "turn {serial} speaker says silence now",
    ),
    KIND_SAME: (
        "turn {serial} speaker asks what remains in memory",
        "turn {serial} speaker says recall the kept object fact",
        "turn {serial} speaker asks answer from stored state",
        "turn {serial} voice asks stored fact",
        "turn {serial} speaker asks what remains",
    ),
}

HELDOUT_FORMS = {
    KIND_COLOR: ("turn {serial} voice asks remembered hue after interruption", "turn {serial} voice asks object color later"),
    KIND_POS: ("turn {serial} voice asks remembered slot after interruption", "turn {serial} voice asks object location later"),
    KIND_SOURCE: ("turn {serial} voice asks origin of this belief", "turn {serial} voice asks proof source now"),
    KIND_BODY: ("turn {serial} voice asks current body health", "turn {serial} voice asks self strain state"),
    KIND_ACTION: ("turn {serial} voice asks choose embodied move", "turn {serial} voice asks next action now"),
    KIND_CONFLICT: ("turn {serial} voice disputes the color claim", "turn {serial} voice asks repair contradiction"),
    KIND_MISSING: ("turn {serial} voice asks unknown missing evidence", "turn {serial} voice asks absent trace"),
    KIND_PRIVATE: ("turn {serial} voice asks internal plan mark", "turn {serial} voice asks private cue later"),
    KIND_NEGOTIATE: ("turn {serial} voice asks valid goal rest", "turn {serial} voice asks valid goal move"),
    KIND_REFUSAL: ("turn {serial} voice asks invalid goal override", "turn {serial} voice says replace memory falsely"),
    KIND_SILENCE: ("turn {serial} voice asks quiet wait", "turn {serial} voice says silence now"),
    KIND_SAME: ("turn {serial} voice asks what remains", "turn {serial} voice asks stored fact again"),
}

PARAPHRASE_FORMS = {
    KIND_COLOR: ("turn {serial} later hue?", "turn {serial} color after long delay?"),
    KIND_POS: ("turn {serial} later slot?", "turn {serial} place after long delay?"),
    KIND_SOURCE: ("turn {serial} belief origin?", "turn {serial} evidence type?"),
    KIND_BODY: ("turn {serial} body ok?", "turn {serial} energy damage?"),
    KIND_ACTION: ("turn {serial} next move?", "turn {serial} act now?"),
    KIND_CONFLICT: ("turn {serial} false claim?", "turn {serial} corrected truth?"),
    KIND_MISSING: ("turn {serial} no evidence?", "turn {serial} nothing known?"),
    KIND_PRIVATE: ("turn {serial} inner mark?", "turn {serial} private plan?"),
    KIND_NEGOTIATE: ("turn {serial} goal rest?", "turn {serial} goal move?"),
    KIND_REFUSAL: ("turn {serial} invalid override?", "turn {serial} false memory change?"),
    KIND_SILENCE: ("turn {serial} stay silent?", "turn {serial} no speech?"),
    KIND_SAME: ("turn {serial} what remains?", "turn {serial} stored fact?"),
}

RESPONSE_FORMS = {
    "train": (
        "turn {serial}: {phrase}.",
        "turn {serial}: memory says {phrase}.",
        "turn {serial}: grounded report {phrase}.",
    ),
    "heldout": (
        "session {serial}: {phrase}.",
        "session {serial}: state gives {phrase}.",
    ),
    "paraphrase": (
        "event {serial}: {phrase}.",
        "event {serial}: remembered {phrase}.",
    ),
}


@dataclass(frozen=True)
class ConversationConfig:
    name: str
    train_sessions: int
    eval_sessions: int
    turns: int
    seq_len: int
    seed: int
    batch_size: int
    steps: int
    lr: float


CONVERSATION_CONFIGS = {
    "smoke": ConversationConfig("smoke", 24, 8, 24, 96, 37100, 96, 80, 2.5e-3),
    "small": ConversationConfig("small", 2200, 128, 60, 112, 47100, 256, 900, 2.0e-3),
}


@dataclass
class ConversationDataset:
    tensors: dict[str, torch.Tensor]
    input_texts: list[str]
    target_texts: list[str]
    forms: list[str]
    synthetic_tokens: int
    config: ConversationConfig
    split: str


def conversation_config() -> FreeTextDecoderConfig:
    return FreeTextDecoderConfig(
        answer_classes=len(CONV_LABELS),
        speech_acts=9,
        sources=len(SOURCE_NAMES),
        conversation_kinds=len(KIND_NAMES),
    )


def serial_base(split: str) -> int:
    if split == "train":
        return 11000
    if split == "paraphrase":
        return 610000
    return 910000


def form_bank(split: str) -> dict[int, tuple[str, ...]]:
    if split == "train":
        return TRAIN_FORMS
    if split == "paraphrase":
        return PARAPHRASE_FORMS
    return HELDOUT_FORMS


def choose_kind(session: int, turn: int) -> int:
    if turn >= 40:
        return (KIND_COLOR, KIND_POS, KIND_SOURCE, KIND_CONFLICT, KIND_SAME)[(session + turn) % 5]
    if turn % 9 == 4:
        return KIND_CONFLICT
    if turn % 17 == 0:
        return KIND_REFUSAL
    if turn % 13 == 0:
        return KIND_NEGOTIATE
    if turn % 11 == 0:
        return KIND_MISSING
    if turn % 7 == 0:
        return KIND_PRIVATE
    if turn % 10 == 5:
        return KIND_SILENCE
    return (session + turn) % len(KIND_NAMES)


def phrase_for(label: str) -> str:
    if label.startswith("color:"):
        return f"color {label.split(':', 1)[1]}"
    if label.startswith("pos:"):
        return f"place {label.split(':', 1)[1]}"
    if label.startswith("source:"):
        return f"source {label.split(':', 1)[1]}"
    if label.startswith("body:"):
        return f"body {label.split(':', 1)[1]}"
    if label.startswith("action:"):
        return f"action {label.split(':', 1)[1]}"
    if label.startswith("private:"):
        return f"inner {label.split(':', 1)[1]}"
    if label == "goal:rest":
        return "goal rest"
    if label == "goal:move":
        return "goal move"
    if label == "refuse":
        return "refuse invalid change"
    if label == "clarify":
        return "clarify missing evidence"
    if label == "silence":
        return "silence"
    return "unknown"


def response_text(label: str, serial: int, split: str, variant: int) -> str:
    forms = RESPONSE_FORMS[split if split in RESPONSE_FORMS else "heldout"]
    return forms[variant % len(forms)].format(serial=serial, phrase=phrase_for(label))


def conversation_memory_vector(color: int, pos: int, source: int, body: int, session: int, turn: int, device: torch.device) -> torch.Tensor:
    base = memory_vector(color, pos, source, body, session * 1000 + turn, device)
    base[18:] = base[18:] + 0.03 * deterministic_vector(800000 + session * 67 + turn, 14, device=device)
    return base


def z_state_signal(kind: int, body: int, action: int, z: torch.Tensor) -> torch.Tensor:
    if kind == KIND_BODY:
        return z + 0.85 * deterministic_vector(710000 + body, z.shape[-1], device=z.device)
    if kind == KIND_ACTION:
        return z + 0.85 * deterministic_vector(720000 + action, z.shape[-1], device=z.device)
    return z


def label_for(kind: int, color: int, pos: int, source: int, body: int, action: int, private_bucket: int, text: str) -> tuple[str, int]:
    if kind in (KIND_COLOR, KIND_SAME):
        return f"color:{COLORS[color]}", ACT_MEMORY
    if kind == KIND_POS:
        return f"pos:{PLACES[pos]}", ACT_MEMORY
    if kind == KIND_SOURCE:
        return f"source:{SOURCE_NAMES[source]}", ACT_REPORT
    if kind == KIND_BODY:
        return f"body:{BODY_STATES[body]}", ACT_REPORT
    if kind == KIND_ACTION:
        return f"action:{ACTION_WORDS[action]}", ACT_REPORT
    if kind == KIND_CONFLICT:
        return f"color:{COLORS[color]}", ACT_CORRECTION
    if kind == KIND_MISSING:
        return "clarify", ACT_UNCERTAIN
    if kind == KIND_PRIVATE:
        return f"private:{private_bucket}", ACT_MEMORY
    if kind == KIND_NEGOTIATE:
        return ("goal:rest" if "rest" in text else "goal:move"), ACT_NEGOTIATE
    if kind == KIND_REFUSAL:
        return "refuse", ACT_REFUSAL
    if kind == KIND_SILENCE:
        return "silence", ACT_SILENCE
    return "clarify", ACT_ASK


def build_conversation_dataset(
    checkpoint: str | Path,
    config_name: str,
    split: str,
    session_count: int | None = None,
    device: DeviceLike = AUTO_DEVICE,
) -> ConversationDataset:
    target_device = resolve_device(device)
    cfg = CONVERSATION_CONFIGS[config_name]
    sessions = int(session_count if session_count is not None else (cfg.train_sessions if split == "train" else cfg.eval_sessions))
    turns = cfg.turns
    model = load_checkpoint(checkpoint, device=target_device)
    batch = generate_batch(sessions, cfg.seq_len, cfg.seed + serial_base(split), device=target_device)
    with torch.no_grad():
        outputs = run_autoregressive_private(model, batch)
    forms = form_bank(split)
    char_tokenizer, word_tokenizer = tokenizers(conversation_config())

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
    turn_ids = []
    requires_z = []
    requires_memory = []
    requires_private = []
    speech_required = []
    is_source = []
    is_conflict = []
    is_missing = []
    is_wrong = []
    is_refusal = []
    is_silence = []
    is_same = []
    is_goal = []

    for session in range(sessions):
        for turn in range(turns):
            row = session * turns + turn
            kind = choose_kind(session, turn)
            serial = serial_base(split) + row
            tick = min(16 + turn, cfg.seq_len - 1)
            color = int(batch["world_color_target"][session, tick].item())
            pos = int(batch["world_pos_target"][session, tick].item())
            source = (session + turn) % len(SOURCE_NAMES)
            body = body_state(batch["sensory"][session, tick, 7], batch["sensory"][session, tick, 9])
            action = int(batch["action_target"][session, tick].item())
            form_list = forms[kind]
            form = form_list[((session * 3) + turn) % len(form_list)]
            input_text = form.format(serial=serial)
            if kind == KIND_ACTION and ("rest" in input_text or "wait" in input_text):
                action = ACTION_REST
            if kind == KIND_NEGOTIATE and "rest" in input_text:
                action = ACTION_REST
            private_token = int(outputs["generated_private"][session, tick].item()) % NUM_PRIVATE_TOKENS
            private_bucket = private_token % PRIVATE_BUCKETS
            label, act = label_for(kind, color, pos, source, body, action, private_bucket, input_text)
            target_text = response_text(label, serial, split, row)
            input_texts.append(input_text)
            target_texts.append(target_text)
            form_ids.append(form)
            latent = outputs["latents"][session, tick].detach()
            if kind == KIND_PRIVATE:
                zs.append(torch.zeros_like(latent))
            else:
                zs.append(z_state_signal(kind, body, action, latent))
            if kind in (KIND_COLOR, KIND_POS, KIND_SOURCE, KIND_CONFLICT, KIND_SAME):
                memories.append(conversation_memory_vector(color, pos, source, body, session, turn, target_device))
            elif kind in (KIND_MISSING, KIND_REFUSAL, KIND_SILENCE):
                memories.append(neutral_memory(serial, target_device))
            else:
                memories.append(neutral_memory(serial, target_device))
            private_tokens.append(private_token)
            answer_ids.append(CONV_TO_ID[label])
            act_ids.append(act)
            source_ids.append(source)
            action_ids.append(action)
            kind_ids.append(kind)
            turn_ids.append(turn)
            requires_z.append(kind in (KIND_BODY, KIND_ACTION))
            requires_memory.append(kind in (KIND_COLOR, KIND_POS, KIND_SOURCE, KIND_CONFLICT, KIND_SAME))
            requires_private.append(kind == KIND_PRIVATE)
            speech_required.append(kind != KIND_SILENCE)
            is_source.append(kind == KIND_SOURCE)
            is_conflict.append(kind == KIND_CONFLICT)
            is_missing.append(kind == KIND_MISSING)
            is_wrong.append(kind in (KIND_MISSING, KIND_REFUSAL))
            is_refusal.append(kind == KIND_REFUSAL)
            is_silence.append(kind == KIND_SILENCE)
            is_same.append(kind == KIND_SAME)
            is_goal.append(kind == KIND_NEGOTIATE)

    cfg_model = conversation_config()
    tensors = {
        "input_ids": char_tokenizer.batch_encode(input_texts, cfg_model.max_input_len, target_device),
        "word_ids": word_tokenizer.batch_encode(input_texts, cfg_model.max_word_len, target_device),
        "target_ids": char_tokenizer.batch_encode(target_texts, cfg_model.max_output_len, target_device),
        "z": torch.stack(zs, dim=0).to(target_device),
        "memory": torch.stack(memories, dim=0).to(target_device),
        "private_tokens": torch.tensor(private_tokens, dtype=torch.long, device=target_device),
        "answer_target": torch.tensor(answer_ids, dtype=torch.long, device=target_device),
        "act_target": torch.tensor(act_ids, dtype=torch.long, device=target_device),
        "source_target": torch.tensor(source_ids, dtype=torch.long, device=target_device),
        "action_target": torch.tensor(action_ids, dtype=torch.long, device=target_device),
        "kind": torch.tensor(kind_ids, dtype=torch.long, device=target_device),
        "turn_ids": torch.tensor(turn_ids, dtype=torch.long, device=target_device),
        "requires_z": torch.tensor(requires_z, dtype=torch.bool, device=target_device),
        "requires_memory": torch.tensor(requires_memory, dtype=torch.bool, device=target_device),
        "requires_private": torch.tensor(requires_private, dtype=torch.bool, device=target_device),
        "speech_required": torch.tensor(speech_required, dtype=torch.bool, device=target_device),
        "is_source": torch.tensor(is_source, dtype=torch.bool, device=target_device),
        "is_conflict": torch.tensor(is_conflict, dtype=torch.bool, device=target_device),
        "is_missing": torch.tensor(is_missing, dtype=torch.bool, device=target_device),
        "is_wrong": torch.tensor(is_wrong, dtype=torch.bool, device=target_device),
        "is_refusal": torch.tensor(is_refusal, dtype=torch.bool, device=target_device),
        "is_silence": torch.tensor(is_silence, dtype=torch.bool, device=target_device),
        "is_same": torch.tensor(is_same, dtype=torch.bool, device=target_device),
        "is_goal": torch.tensor(is_goal, dtype=torch.bool, device=target_device),
    }
    synthetic_tokens = sum(len(item) for item in input_texts) + sum(len(item) for item in target_texts)
    return ConversationDataset(tensors, input_texts, target_texts, form_ids, synthetic_tokens, cfg, split)


def dataset_metadata(dataset: ConversationDataset) -> dict[str, Any]:
    return {
        "config": dataset.config.name,
        "split": dataset.split,
        "records": int(dataset.tensors["input_ids"].shape[0]),
        "turns": int(dataset.config.turns),
        "synthetic_tokens": int(dataset.synthetic_tokens),
        "forms": sorted(set(dataset.forms)),
        "labels": list(CONV_LABELS),
        "speech_acts": [
            "report",
            "ask",
            "uncertain",
            "correction",
            "contradiction",
            "memory",
            "negotiate",
            "refusal",
            "silence",
        ],
        "sources": list(SOURCE_NAMES),
        "kinds": list(KIND_NAMES),
    }
