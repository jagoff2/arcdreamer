from __future__ import annotations

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


def _default_vocab_tokens() -> tuple[str, ...]:
    groups = (
        (
            "<pad>",
            "<bos>",
            "<eos>",
            "unknown",
        ),
        (
            "belief",
            "question",
            "plan",
            "theory",
            "diagnostic",
            "goal",
            "target",
            "state",
            "mode",
            "progress",
            "focus",
            "contradiction",
            "competition",
            "support",
            "effect",
            "direction",
            "action",
            "family",
            "because",
            "step",
            "color",
        ),
        (
            "need",
            "test",
            "explore",
            "probe",
            "confirm",
            "commit",
            "move",
            "interact",
            "click",
            "select",
            "wait",
            "toward",
            "return",
            "anchor",
            "frontier",
            "hotspot",
            "adjacent",
        ),
        (
            "active",
            "inactive",
            "uncertain",
            "present",
            "absent",
            "visible",
            "hidden",
            "positive",
            "negative",
            "none",
            "high",
            "low",
            "near",
            "mid",
            "far",
        ),
        (
            "rule",
            "collect",
            "unlock",
            "selector",
            "switch",
            "order",
            "delayed",
            "sequence",
            "recent",
            "agent",
            "interactable",
            "blocking",
        ),
        (
            "up",
            "down",
            "left",
            "right",
            "red",
            "blue",
            "green",
            "yellow",
            "gray",
            "orange",
            "purple",
            "cyan",
        ),
        tuple(f"c{index}" for index in range(12)),
        tuple(f"p{index}" for index in range(6)),
        tuple(f"n{index}" for index in range(6)),
    )
    ordered: list[str] = []
    seen: set[str] = set()
    for group in groups:
        for token in group:
            if token in seen:
                continue
            seen.add(token)
            ordered.append(token)
    return tuple(ordered)


DEFAULT_VOCAB: tuple[str, ...] = _default_vocab_tokens()

MODE_TO_ID = {"belief": 0, "question": 1, "plan": 2, "theory": 3, "diagnostic": 4}


@dataclass(frozen=True)
class TokenVocabulary:
    token_to_id: dict[str, int]
    id_to_token: tuple[str, ...]

    @classmethod
    def from_tokens(cls, tokens: tuple[str, ...] = DEFAULT_VOCAB) -> "TokenVocabulary":
        return cls(token_to_id={token: idx for idx, token in enumerate(tokens)}, id_to_token=tokens)

    def encode(self, tokens: tuple[str, ...]) -> list[int]:
        ids = [self.token_to_id["<bos>"]]
        ids.extend(self.token_to_id.get(token, self.token_to_id["unknown"]) for token in tokens)
        ids.append(self.token_to_id["<eos>"])
        return ids

    def decode(self, ids: list[int]) -> tuple[str, ...]:
        tokens: list[str] = []
        for idx in ids:
            token = self.id_to_token[idx]
            if token in ("<bos>", "<pad>"):
                continue
            if token == "<eos>":
                break
            tokens.append(token)
        return tuple(tokens)


class GroundedLanguageModel(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        vocab: TokenVocabulary | None = None,
    ) -> None:
        super().__init__()
        self.vocab = vocab or TokenVocabulary.from_tokens()
        self.embedding = nn.Embedding(len(self.vocab.id_to_token), hidden_dim)
        self.mode_embedding = nn.Embedding(len(MODE_TO_ID), hidden_dim)
        self.init_hidden = nn.Linear(latent_dim, hidden_dim)
        self.gru = nn.GRU(hidden_dim, hidden_dim, batch_first=True)
        self.output = nn.Linear(hidden_dim, len(self.vocab.id_to_token))

    def embed_tokens(self, token_lists: list[tuple[str, ...]], device: torch.device | None = None) -> torch.Tensor:
        max_len = max((len(tokens) for tokens in token_lists), default=1)
        ids = []
        for tokens in token_lists:
            encoded = [self.vocab.token_to_id.get(token, self.vocab.token_to_id["unknown"]) for token in tokens]
            encoded = encoded[:max_len]
            encoded += [self.vocab.token_to_id["<pad>"]] * (max_len - len(encoded))
            ids.append(encoded)
        token_tensor = torch.tensor(ids, dtype=torch.long, device=device)
        embedded = self.embedding(token_tensor)
        mask = (token_tensor != self.vocab.token_to_id["<pad>"]).float().unsqueeze(-1)
        denom = mask.sum(dim=1).clamp(min=1.0)
        return (embedded * mask).sum(dim=1) / denom

    def teacher_forcing_loss(
        self,
        latent: torch.Tensor,
        token_lists: list[tuple[str, ...]],
        mode: str = "belief",
    ) -> torch.Tensor:
        device = latent.device
        encoded = [self.vocab.encode(tokens) for tokens in token_lists]
        max_len = max(len(tokens) for tokens in encoded)
        padded = [tokens + [self.vocab.token_to_id["<pad>"]] * (max_len - len(tokens)) for tokens in encoded]
        token_tensor = torch.tensor(padded, dtype=torch.long, device=device)
        inputs = token_tensor[:, :-1]
        targets = token_tensor[:, 1:]
        input_embeds = self.embedding(inputs)
        mode_embed = self.mode_embedding(
            torch.full((latent.shape[0],), MODE_TO_ID[mode], dtype=torch.long, device=device)
        ).unsqueeze(1)
        conditioned_inputs = input_embeds + mode_embed
        hidden0 = self.init_hidden(latent).unsqueeze(0)
        outputs, _ = self.gru(conditioned_inputs, hidden0)
        logits = self.output(outputs)
        return F.cross_entropy(
            logits.reshape(-1, logits.shape[-1]),
            targets.reshape(-1),
            ignore_index=self.vocab.token_to_id["<pad>"],
        )

    @torch.no_grad()
    def decode(
        self,
        latent: torch.Tensor,
        mode: str = "belief",
        max_length: int = 12,
    ) -> tuple[str, ...]:
        if latent.ndim == 1:
            latent = latent.unsqueeze(0)
        device = latent.device
        token_id = torch.tensor([[self.vocab.token_to_id["<bos>"]]], dtype=torch.long, device=device)
        hidden = self.init_hidden(latent).unsqueeze(0)
        mode_embed = self.mode_embedding(
            torch.full((latent.shape[0],), MODE_TO_ID[mode], dtype=torch.long, device=device)
        ).unsqueeze(1)
        generated: list[int] = []
        for _ in range(max_length):
            embedded = self.embedding(token_id[:, -1:]) + mode_embed
            output, hidden = self.gru(embedded, hidden)
            logits = self.output(output[:, -1])
            next_id = int(logits.argmax(dim=-1)[0].item())
            if next_id == self.vocab.token_to_id["<eos>"]:
                break
            generated.append(next_id)
            token_id = torch.cat(
                [token_id, torch.tensor([[next_id]], dtype=torch.long, device=device)],
                dim=1,
            )
        return self.vocab.decode(generated)
