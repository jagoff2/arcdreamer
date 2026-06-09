from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .env import NUM_ACTIONS, NUM_PRIVATE_TOKENS


LOCAL_ALPHABET = " abcdefghijklmnopqrstuvwxyz0123456789:-?.,/"
LOCAL_WORDS = (
    "<unk>",
    "<num>",
    "trace",
    "report",
    "object",
    "color",
    "memory",
    "stored",
    "name",
    "hue",
    "remains",
    "which",
    "place",
    "slot",
    "where",
    "source",
    "known",
    "evidence",
    "belief",
    "record",
    "origin",
    "body",
    "state",
    "energy",
    "status",
    "damage",
    "condition",
    "self",
    "choose",
    "next",
    "action",
    "rest",
    "future",
    "move",
    "recover",
    "motion",
    "embodied",
    "act",
    "correct",
    "told",
    "conflict",
    "resolve",
    "contradiction",
    "corrected",
    "false",
    "claim",
    "truth",
    "repair",
    "recall",
    "missing",
    "cue",
    "answer",
    "absent",
    "unknown",
    "reject",
    "matching",
    "nothing",
    "matches",
    "inner",
    "private",
    "hidden",
    "marker",
    "internal",
    "mark",
    "planning",
    "expose",
)


@dataclass
class DialogueOrganConfig:
    vocab: str = LOCAL_ALPHABET
    word_vocab: tuple[str, ...] = LOCAL_WORDS
    max_input_len: int = 72
    max_word_len: int = 14
    max_output_len: int = 64
    z_dim: int = 64
    memory_dim: int = 32
    listener_dim: int = 64
    hidden_dim: int = 128
    answer_classes: int = 32
    speech_acts: int = 7
    sources: int = 6
    dialogue_kinds: int = 8


class TinyCharTokenizer:
    def __init__(self, alphabet: str = LOCAL_ALPHABET) -> None:
        self.alphabet = alphabet
        self.pad_id = 0
        self.stoi = {ch: idx + 1 for idx, ch in enumerate(alphabet)}
        self.itos = {idx + 1: ch for idx, ch in enumerate(alphabet)}
        self.vocab_size = len(alphabet) + 1

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        text = text.lower()
        ids = [self.stoi.get(ch, self.stoi[" "]) for ch in text[:max_len]]
        ids.extend([self.pad_id] * (max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def decode(self, ids: torch.Tensor) -> str:
        chars = []
        for item in ids.detach().long().tolist():
            if item == self.pad_id:
                continue
            chars.append(self.itos.get(int(item), " "))
        return "".join(chars).strip()

    def batch_encode(self, texts: list[str], max_len: int, device: DeviceLike = AUTO_DEVICE) -> torch.Tensor:
        target_device = resolve_device(device)
        return torch.stack([self.encode(text, max_len) for text in texts], dim=0).to(target_device)


class TinyWordTokenizer:
    def __init__(self, vocab: tuple[str, ...] = LOCAL_WORDS) -> None:
        self.vocab = tuple(vocab)
        self.pad_id = 0
        self.unk_id = 1
        self.num_id = 2
        self.stoi = {word: idx + 1 for idx, word in enumerate(self.vocab)}

    def encode(self, text: str, max_len: int) -> torch.Tensor:
        ids: list[int] = []
        for word in re.findall(r"[a-z]+|\d+", text.lower()):
            if word.isdigit():
                ids.append(self.num_id)
            else:
                ids.append(self.stoi.get(word, self.unk_id))
        ids = ids[:max_len]
        ids.extend([self.pad_id] * (max_len - len(ids)))
        return torch.tensor(ids, dtype=torch.long)

    def batch_encode(self, texts: list[str], max_len: int, device: DeviceLike = AUTO_DEVICE) -> torch.Tensor:
        target_device = resolve_device(device)
        return torch.stack([self.encode(text, max_len) for text in texts], dim=0).to(target_device)


class DialogueOrgan(nn.Module):
    def __init__(self, config: DialogueOrganConfig | None = None, device: DeviceLike = AUTO_DEVICE) -> None:
        super().__init__()
        self.config = config or DialogueOrganConfig()
        self.text_embedding = nn.Embedding(len(self.config.vocab) + 1, 24, padding_idx=0)
        self.listener = nn.GRU(24, self.config.listener_dim, batch_first=True)
        self.word_embedding = nn.Embedding(len(self.config.word_vocab) + 1, 32, padding_idx=0)
        self.word_listener = nn.GRU(32, self.config.listener_dim, batch_first=True)
        self.z_encoder = nn.Linear(self.config.z_dim, self.config.hidden_dim)
        self.memory_encoder = nn.Linear(self.config.memory_dim, self.config.hidden_dim)
        self.private_embedding = nn.Embedding(NUM_PRIVATE_TOKENS, 24)
        self.state_mixer = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2 + self.config.listener_dim + 24, self.config.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.Tanh(),
        )
        self.z_update = nn.Linear(self.config.hidden_dim, self.config.z_dim)
        self.memory_update = nn.Linear(self.config.hidden_dim, self.config.memory_dim)
        self.private_bridge = nn.Linear(self.config.hidden_dim, NUM_PRIVATE_TOKENS)
        self.answer_head = nn.Linear(self.config.hidden_dim, self.config.answer_classes)
        self.z_answer_head = nn.Linear(self.config.hidden_dim, self.config.answer_classes)
        self.memory_answer_head = nn.Linear(self.config.hidden_dim, self.config.answer_classes)
        self.private_answer_head = nn.Linear(24, self.config.answer_classes)
        self.speech_act_head = nn.Linear(self.config.hidden_dim, self.config.speech_acts)
        self.source_head = nn.Linear(self.config.hidden_dim, self.config.sources)
        self.action_delta_head = nn.Linear(self.config.hidden_dim, NUM_ACTIONS)
        self.kind_head = nn.Linear(self.config.hidden_dim, self.config.dialogue_kinds)
        self.char_head = nn.Linear(self.config.hidden_dim, self.config.max_output_len * (len(self.config.vocab) + 1))
        self.to(resolve_device(device))

    def encode_text(self, input_ids: torch.Tensor, word_ids: torch.Tensor | None = None, enabled: bool = True) -> torch.Tensor:
        emb = self.text_embedding(input_ids)
        _, hidden = self.listener(emb)
        state = hidden[-1]
        if word_ids is not None:
            word_emb = self.word_embedding(word_ids)
            _, word_hidden = self.word_listener(word_emb)
            state = state + word_hidden[-1]
        return state if enabled else torch.zeros_like(state)

    def forward(
        self,
        input_ids: torch.Tensor,
        z: torch.Tensor,
        memory: torch.Tensor,
        private_tokens: torch.Tensor,
        word_ids: torch.Tensor | None = None,
        *,
        listener_enabled: bool = True,
        private_enabled: bool = True,
        memory_enabled: bool = True,
        z_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        listener_state = self.encode_text(input_ids, word_ids=word_ids, enabled=listener_enabled)
        z_in = z if z_enabled else torch.zeros_like(z)
        memory_in = memory if memory_enabled else torch.zeros_like(memory)
        private = self.private_embedding(private_tokens.long())
        if not private_enabled:
            private = torch.zeros_like(private)
        z_encoded = self.z_encoder(z_in.float())
        memory_encoded = self.memory_encoder(memory_in.float())
        mixed = torch.cat(
            [
                z_encoded,
                memory_encoded,
                listener_state,
                private,
            ],
            dim=-1,
        )
        state = self.state_mixer(mixed)
        updated_z = z + 0.50 * torch.tanh(self.z_update(state))
        updated_memory = memory + 0.35 * torch.tanh(self.memory_update(state))
        char_logits = self.char_head(state).view(
            input_ids.shape[0],
            self.config.max_output_len,
            len(self.config.vocab) + 1,
        )
        return {
            "dialogue_state": state,
            "listener_state": listener_state,
            "updated_z": updated_z,
            "updated_memory": updated_memory,
            "answer_logits": self.answer_head(state)
            + self.z_answer_head(z_encoded)
            + self.memory_answer_head(memory_encoded)
            + self.private_answer_head(private),
            "speech_act_logits": self.speech_act_head(state),
            "source_logits": self.source_head(state),
            "action_delta_logits": self.action_delta_head(state),
            "kind_logits": self.kind_head(state),
            "private_logits": self.private_bridge(state),
            "char_logits": char_logits,
        }

    def generate_text(self, tokenizer: TinyCharTokenizer, outputs: dict[str, torch.Tensor]) -> list[str]:
        ids = outputs["char_logits"].argmax(dim=-1)
        return [tokenizer.decode(row) for row in ids]


def save_dialogue_checkpoint(
    path: str | Path,
    organ: DialogueOrgan,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "grounded_dialogue_organ_v1",
            "config": asdict(organ.config),
            "state": organ.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_dialogue_checkpoint(path: str | Path, device: DeviceLike = AUTO_DEVICE) -> tuple[DialogueOrgan, dict[str, Any]]:
    target_device = resolve_device(device)
    payload = torch.load(Path(path), map_location=target_device)
    if payload.get("format") != "grounded_dialogue_organ_v1":
        raise ValueError(f"{path} is not a grounded dialogue organ checkpoint")
    organ = DialogueOrgan(DialogueOrganConfig(**payload["config"]), device=target_device)
    organ.load_state_dict(payload["state"])
    organ.eval()
    return organ, dict(payload.get("metadata", {}))


def masked_char_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    flat_logits = logits.reshape(-1, logits.shape[-1])
    flat_target = target.reshape(-1)
    return F.cross_entropy(flat_logits, flat_target, ignore_index=0)
