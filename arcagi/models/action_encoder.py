from __future__ import annotations

import hashlib
from typing import Sequence

import torch
from torch import nn

from arcagi.core.action_schema import (
    ActionSchemaContext,
    build_action_schema,
    build_action_schema_context,
    direction_vector,
)
from arcagi.core.types import ActionName, StructuredState

NUMERIC_ACTION_FEATURES = 16


def _stable_bucket(token: str, bucket_count: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % bucket_count


def action_to_tokens_and_features(
    action: ActionName,
    context: ActionSchemaContext,
) -> tuple[tuple[str, ...], list[float]]:
    schema = build_action_schema(action, context)
    dy, dx = direction_vector(schema.direction)
    coord_x = 0.0
    coord_y = 0.0
    if schema.normalized_click is not None:
        coord_x, coord_y = schema.normalized_click
    raw_action = schema.raw_action or 0
    raw_value = float(raw_action) / max(context.max_raw_action, 1.0)
    coarse_x = schema.coarse_bin[0] if schema.coarse_bin is not None else "none"
    coarse_y = schema.coarse_bin[1] if schema.coarse_bin is not None else "none"
    tokens = [
        f"type:{schema.action_type}",
        f"role:{schema.role}",
        f"dir:{schema.direction or 'none'}",
        f"family:{schema.family}",
        f"raw:{raw_action if raw_action else 'none'}",
        f"coarse_x:{coarse_x}",
        f"coarse_y:{coarse_y}",
    ]
    tokens.extend(f"part:{part}" for part in schema.parts[:4])
    features = [
        float(schema.action_type == "move"),
        float(schema.action_type == "interact"),
        float(schema.action_type == "click"),
        float(schema.action_type == "wait"),
        float(schema.action_type == "undo"),
        float(schema.action_type == "select"),
        float(schema.action_type == "raw"),
        float(schema.role in {"move", "interact", "click", "select", "undo", "wait", "select_cycle"}),
        dy,
        dx,
        coord_y,
        coord_x,
        float(schema.click is not None),
        raw_value,
        float(len(schema.parts)) / 4.0,
        min(context.affordance_count, 32.0) / 32.0,
    ]
    return tuple(tokens), features


def build_action_context(
    affordances: Sequence[ActionName] = (),
    action_roles: dict[ActionName, str] | None = None,
) -> ActionSchemaContext:
    return build_action_schema_context(affordances, action_roles)


def state_action_context(state: StructuredState | None) -> ActionSchemaContext:
    if state is None:
        return build_action_schema_context()
    return build_action_context(state.affordances, dict(state.action_roles))


class ActionEncoder(nn.Module):
    def __init__(
        self,
        embedding_dim: int = 32,
        token_bucket_count: int = 256,
        hidden_dim: int = 64,
    ) -> None:
        super().__init__()
        self.token_bucket_count = token_bucket_count
        self.token_embedding = nn.EmbeddingBag(token_bucket_count, embedding_dim, mode="mean")
        self.numeric_projection = nn.Sequential(
            nn.Linear(NUMERIC_ACTION_FEATURES, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, embedding_dim),
        )
        self.output = nn.Sequential(
            nn.Linear(embedding_dim * 2, embedding_dim),
            nn.ReLU(),
            nn.Linear(embedding_dim, embedding_dim),
        )

    def encode(
        self,
        actions: Sequence[ActionName],
        state: StructuredState | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        context = state_action_context(state)
        token_ids: list[int] = []
        offsets: list[int] = []
        numeric_rows: list[list[float]] = []
        for action in actions:
            offsets.append(len(token_ids))
            tokens, features = action_to_tokens_and_features(action, context)
            token_ids.extend(_stable_bucket(token, self.token_bucket_count) for token in tokens)
            numeric_rows.append(features)
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        offset_tensor = torch.tensor(offsets, dtype=torch.long, device=device)
        numeric_tensor = torch.tensor(numeric_rows, dtype=torch.float32, device=device)
        token_embed = self.token_embedding(token_tensor, offset_tensor)
        numeric_embed = self.numeric_projection(numeric_tensor)
        return self.output(torch.cat([token_embed, numeric_embed], dim=-1))
