from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .env import NUM_ACTIONS, NUM_PRIVATE_TOKENS
from .language_organ import LOCAL_ALPHABET, LOCAL_WORDS, TinyCharTokenizer, TinyWordTokenizer


CONVERSATION_WORDS = tuple(
    dict.fromkeys(
        LOCAL_WORDS
        + (
            "speaker",
            "voice",
            "says",
            "asks",
            "please",
            "now",
            "later",
            "after",
            "interruption",
            "again",
            "earlier",
            "turn",
            "session",
            "kept",
            "remain",
            "remain?",
            "remember",
            "recall",
            "position",
            "location",
            "proof",
            "basis",
            "seen",
            "heard",
            "guess",
            "rebuild",
            "body",
            "health",
            "strain",
            "wait",
            "move",
            "left",
            "right",
            "forage",
            "stay",
            "rest",
            "goal",
            "change",
            "override",
            "replace",
            "falsely",
            "valid",
            "invalid",
            "cannot",
            "refuse",
            "clarify",
            "silent",
            "silence",
            "maybe",
            "missing",
            "evidence",
            "dispute",
            "disputes",
            "correction",
            "correct",
            "false",
            "claim",
            "truth",
            "inner",
            "private",
            "plan",
            "mark",
            "steady",
            "ok",
            "low",
            "hurt",
            "quiet",
            "known",
            "type",
            "red",
            "green",
            "blue",
            "yellow",
            "zero",
            "one",
            "two",
            "three",
            "four",
            "observed",
            "told",
            "imagined",
            "inferred",
            "replayed",
            "reconstructed",
        )
    )
)


@dataclass
class FreeTextDecoderConfig:
    vocab: str = LOCAL_ALPHABET
    word_vocab: tuple[str, ...] = CONVERSATION_WORDS
    max_input_len: int = 128
    max_word_len: int = 24
    max_output_len: int = 96
    z_dim: int = 64
    memory_dim: int = 32
    listener_dim: int = 96
    hidden_dim: int = 160
    char_dim: int = 32
    answer_classes: int = 36
    speech_acts: int = 9
    sources: int = 6
    conversation_kinds: int = 12
    max_turns: int = 64


class FreeTextConversationDecoder(nn.Module):
    def __init__(self, config: FreeTextDecoderConfig | None = None, device: DeviceLike = AUTO_DEVICE) -> None:
        super().__init__()
        self.config = config or FreeTextDecoderConfig()
        vocab_size = len(self.config.vocab) + 1
        word_size = len(self.config.word_vocab) + 1
        self.char_embedding = nn.Embedding(vocab_size, self.config.char_dim, padding_idx=0)
        self.char_listener = nn.GRU(self.config.char_dim, self.config.listener_dim, batch_first=True)
        self.word_embedding = nn.Embedding(word_size, 40, padding_idx=0)
        self.word_listener = nn.GRU(40, self.config.listener_dim, batch_first=True)
        self.private_embedding = nn.Embedding(NUM_PRIVATE_TOKENS, 32)
        self.turn_embedding = nn.Embedding(self.config.max_turns + 1, 24)
        self.z_encoder = nn.Linear(self.config.z_dim, self.config.hidden_dim)
        self.memory_encoder = nn.Linear(self.config.memory_dim, self.config.hidden_dim)
        self.mixer = nn.Sequential(
            nn.Linear(self.config.hidden_dim * 2 + self.config.listener_dim + 32 + 24, self.config.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
            nn.Tanh(),
        )
        self.z_update = nn.Linear(self.config.hidden_dim, self.config.z_dim)
        self.memory_update = nn.Linear(self.config.hidden_dim, self.config.memory_dim)
        self.listener_z_update = nn.Linear(self.config.listener_dim, self.config.z_dim)
        self.answer_head = nn.Linear(self.config.hidden_dim, self.config.answer_classes)
        self.z_answer_head = nn.Linear(self.config.hidden_dim, self.config.answer_classes)
        self.memory_answer_head = nn.Linear(self.config.hidden_dim, self.config.answer_classes)
        self.private_answer_head = nn.Linear(32, self.config.answer_classes)
        self.act_head = nn.Linear(self.config.hidden_dim, self.config.speech_acts)
        self.source_head = nn.Linear(self.config.hidden_dim, self.config.sources)
        self.action_head = nn.Linear(self.config.hidden_dim, NUM_ACTIONS)
        self.kind_head = nn.Linear(self.config.hidden_dim, self.config.conversation_kinds)
        self.private_head = nn.Linear(self.config.hidden_dim, NUM_PRIVATE_TOKENS)
        self.decoder_cell = nn.GRUCell(self.config.char_dim + self.config.hidden_dim, self.config.hidden_dim)
        self.decoder_head = nn.Linear(self.config.hidden_dim, vocab_size)
        self.to(resolve_device(device))

    def encode_listener(
        self,
        input_ids: torch.Tensor,
        word_ids: torch.Tensor,
        enabled: bool = True,
    ) -> torch.Tensor:
        char_emb = self.char_embedding(input_ids)
        _, char_hidden = self.char_listener(char_emb)
        word_emb = self.word_embedding(word_ids)
        _, word_hidden = self.word_listener(word_emb)
        state = char_hidden[-1] + word_hidden[-1]
        return state if enabled else torch.zeros_like(state)

    def condition(
        self,
        input_ids: torch.Tensor,
        word_ids: torch.Tensor,
        z: torch.Tensor,
        memory: torch.Tensor,
        private_tokens: torch.Tensor,
        turn_ids: torch.Tensor,
        *,
        listener_enabled: bool = True,
        private_enabled: bool = True,
        memory_enabled: bool = True,
        z_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        listener_state = self.encode_listener(input_ids, word_ids, enabled=listener_enabled)
        z_in = z if z_enabled else torch.zeros_like(z)
        memory_in = memory if memory_enabled else torch.zeros_like(memory)
        private = self.private_embedding(private_tokens.long())
        if not private_enabled:
            private = torch.zeros_like(private)
        turns = self.turn_embedding(turn_ids.clamp(0, self.config.max_turns).long())
        z_encoded = self.z_encoder(z_in.float())
        memory_encoded = self.memory_encoder(memory_in.float())
        state = self.mixer(torch.cat([z_encoded, memory_encoded, listener_state, private, turns], dim=-1))
        return {
            "conversation_state": state,
            "listener_state": listener_state,
            "z_encoded": z_encoded,
            "memory_encoded": memory_encoded,
            "private_encoded": private,
            "updated_z": z + 1.25 * torch.tanh(self.z_update(state) + 0.20 * self.listener_z_update(listener_state)),
            "updated_memory": memory + 0.45 * torch.tanh(self.memory_update(state)),
        }

    def decode_logits(self, state: torch.Tensor, target_ids: torch.Tensor | None = None) -> torch.Tensor:
        batch = state.shape[0]
        steps = self.config.max_output_len if target_ids is None else target_ids.shape[1]
        hidden = state
        prev = torch.zeros(batch, dtype=torch.long, device=state.device)
        logits: list[torch.Tensor] = []
        for idx in range(steps):
            emb = self.char_embedding(prev)
            hidden = self.decoder_cell(torch.cat([emb, state], dim=-1), hidden)
            step_logits = self.decoder_head(hidden)
            logits.append(step_logits)
            if target_ids is None:
                prev = step_logits.argmax(dim=-1)
            else:
                prev = target_ids[:, idx].long()
        return torch.stack(logits, dim=1)

    def forward(
        self,
        input_ids: torch.Tensor,
        word_ids: torch.Tensor,
        z: torch.Tensor,
        memory: torch.Tensor,
        private_tokens: torch.Tensor,
        turn_ids: torch.Tensor,
        target_ids: torch.Tensor | None = None,
        *,
        listener_enabled: bool = True,
        private_enabled: bool = True,
        memory_enabled: bool = True,
        z_enabled: bool = True,
    ) -> dict[str, torch.Tensor]:
        conditioned = self.condition(
            input_ids,
            word_ids,
            z,
            memory,
            private_tokens,
            turn_ids,
            listener_enabled=listener_enabled,
            private_enabled=private_enabled,
            memory_enabled=memory_enabled,
            z_enabled=z_enabled,
        )
        state = conditioned["conversation_state"]
        z_encoded = conditioned["z_encoded"]
        memory_encoded = conditioned["memory_encoded"]
        private = conditioned["private_encoded"]
        char_logits = self.decode_logits(state, target_ids)
        return {
            **conditioned,
            "answer_logits": self.answer_head(state)
            + self.z_answer_head(z_encoded)
            + self.memory_answer_head(memory_encoded)
            + self.private_answer_head(private),
            "speech_act_logits": self.act_head(state),
            "source_logits": self.source_head(state),
            "action_logits": self.action_head(state),
            "kind_logits": self.kind_head(state),
            "private_logits": self.private_head(state),
            "char_logits": char_logits,
        }

    def generate_text(self, tokenizer: TinyCharTokenizer, outputs: dict[str, torch.Tensor]) -> list[str]:
        ids = outputs["char_logits"].argmax(dim=-1)
        return [tokenizer.decode(row) for row in ids]


def masked_sequence_ce(logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), ignore_index=0)


def save_conversation_checkpoint(
    path: str | Path,
    model: FreeTextConversationDecoder,
    metadata: dict[str, Any],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "format": "grounded_conversation_decoder_v1",
            "config": asdict(model.config),
            "state": model.state_dict(),
            "metadata": metadata,
        },
        path,
    )


def load_conversation_checkpoint(
    path: str | Path,
    device: DeviceLike = AUTO_DEVICE,
) -> tuple[FreeTextConversationDecoder, dict[str, Any]]:
    target_device = resolve_device(device)
    payload = torch.load(Path(path), map_location=target_device)
    if payload.get("format") != "grounded_conversation_decoder_v1":
        raise ValueError(f"{path} is not a grounded conversation checkpoint")
    model = FreeTextConversationDecoder(FreeTextDecoderConfig(**payload["config"]), device=target_device)
    model.load_state_dict(payload["state"])
    model.eval()
    return model, dict(payload.get("metadata", {}))


def tokenizers(config: FreeTextDecoderConfig | None = None) -> tuple[TinyCharTokenizer, TinyWordTokenizer]:
    cfg = config or FreeTextDecoderConfig()
    return TinyCharTokenizer(cfg.vocab), TinyWordTokenizer(cfg.word_vocab)
