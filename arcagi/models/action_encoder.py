from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn

from arcagi.core.action_schema import (
    ActionSchemaContext,
    build_action_schema,
    build_action_schema_context,
    click_to_grid_cell,
    direction_vector,
)
from arcagi.core.types import COLOR_BUCKETS, ActionName, ObjectState, StructuredState

NUMERIC_ACTION_FEATURES = 28


@dataclass(frozen=True)
class ActionTargetLookup:
    by_cell: dict[tuple[int, int], ObjectState]
    agent_y: float = 0.0
    agent_x: float = 0.0


def _stable_bucket(token: str, bucket_count: int) -> int:
    digest = hashlib.blake2b(token.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False) % bucket_count


def action_to_tokens_and_features(
    action: ActionName,
    context: ActionSchemaContext,
    state: StructuredState | None = None,
    target_lookup: ActionTargetLookup | None = None,
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
    target_obj, target_y, target_x, agent_y, agent_x = _target_object_for_action(
        schema,
        state,
        target_lookup=target_lookup,
    )
    target_color = -1
    target_tags: tuple[str, ...] = ()
    target_area = 0.0
    if target_obj is not None:
        target_color = int(target_obj.color) % COLOR_BUCKETS
        target_tags = tuple(target_obj.tags)
        total_cells = 1.0
        if state is not None:
            total_cells = max(float(state.grid_shape[0] * state.grid_shape[1]), 1.0)
        target_area = float(target_obj.area) / total_cells
    tokens = [
        f"type:{schema.action_type}",
        f"role:{schema.role}",
        f"dir:{schema.direction or 'none'}",
        f"family:{schema.family}",
        f"raw:{raw_action if raw_action else 'none'}",
        f"coarse_x:{coarse_x}",
        f"coarse_y:{coarse_y}",
    ]
    if target_obj is not None:
        tokens.append(f"target_color:{target_color}")
        tokens.extend(f"target_tag:{tag}" for tag in target_tags[:4])
    tokens.extend(f"part:{part}" for part in schema.parts[:4])
    target_present = float(target_obj is not None)
    rel_y = 0.0
    rel_x = 0.0
    if state is not None and target_y is not None and target_x is not None:
        height, width = state.grid_shape
        rel_y = (float(target_y) - float(agent_y)) / max(height, 1)
        rel_x = (float(target_x) - float(agent_x)) / max(width, 1)
    target_y_norm = 0.0
    target_x_norm = 0.0
    if state is not None and target_y is not None and target_x is not None:
        height, width = state.grid_shape
        target_y_norm = float(target_y) / max(height, 1)
        target_x_norm = float(target_x) / max(width, 1)
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
        target_present,
        float(target_color) / max(COLOR_BUCKETS - 1, 1) if target_color >= 0 else 0.0,
        target_area,
        float("agent" in target_tags),
        float("target" in target_tags),
        float("interactable" in target_tags),
        float("selector" in target_tags),
        float("blocking" in target_tags),
        target_y_norm,
        target_x_norm,
        rel_y,
        rel_x,
    ]
    if len(features) != NUMERIC_ACTION_FEATURES:
        raise ValueError(
            f"action feature width mismatch: got {len(features)}, expected {NUMERIC_ACTION_FEATURES}"
        )
    return tuple(tokens), features


def _target_object_for_action(
    schema: object,
    state: StructuredState | None,
    *,
    target_lookup: ActionTargetLookup | None = None,
) -> tuple[ObjectState | None, int | None, int | None, float, float]:
    agent_y = 0.0
    agent_x = 0.0
    if not isinstance(state, StructuredState):
        return None, None, None, agent_y, agent_x
    if target_lookup is not None:
        agent_y, agent_x = target_lookup.agent_y, target_lookup.agent_x
    else:
        agent = next((obj for obj in state.objects if "agent" in obj.tags), None)
        if agent is not None:
            agent_y, agent_x = agent.centroid
    target_y: int | None = None
    target_x: int | None = None
    click = getattr(schema, "click", None)
    if click is not None:
        grid_cell = click_to_grid_cell(click, grid_shape=state.grid_shape, inventory=state.inventory_dict())
        if grid_cell is not None:
            target_y, target_x = grid_cell
    else:
        direction = getattr(schema, "direction", "")
        if direction:
            dy, dx = direction_vector(direction)
            target_y = int(round(agent_y + dy))
            target_x = int(round(agent_x + dx))
    if target_y is None or target_x is None:
        return None, None, None, agent_y, agent_x
    target = (target_y, target_x)
    if target_lookup is not None:
        target_obj = target_lookup.by_cell.get(target)
    else:
        target_obj = next((obj for obj in state.objects if target in obj.cells), None)
    return target_obj, target_y, target_x, agent_y, agent_x


def _build_action_target_lookup(state: StructuredState | None) -> ActionTargetLookup | None:
    if not isinstance(state, StructuredState):
        return None
    by_cell: dict[tuple[int, int], ObjectState] = {}
    agent_y = 0.0
    agent_x = 0.0
    for obj in state.objects:
        if "agent" in obj.tags:
            agent_y, agent_x = obj.centroid
        for cell in obj.cells:
            by_cell[tuple(cell)] = obj
    return ActionTargetLookup(by_cell=by_cell, agent_y=float(agent_y), agent_x=float(agent_x))


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
        target_lookup = _build_action_target_lookup(state)
        token_ids: list[int] = []
        offsets: list[int] = []
        numeric_rows: list[list[float]] = []
        for action in actions:
            offsets.append(len(token_ids))
            tokens, features = action_to_tokens_and_features(action, context, state, target_lookup=target_lookup)
            token_ids.extend(_stable_bucket(token, self.token_bucket_count) for token in tokens)
            numeric_rows.append(features)
        token_tensor = torch.tensor(token_ids, dtype=torch.long, device=device)
        offset_tensor = torch.tensor(offsets, dtype=torch.long, device=device)
        numeric_tensor = torch.tensor(numeric_rows, dtype=torch.float32, device=device)
        token_embed = self.token_embedding(token_tensor, offset_tensor)
        numeric_embed = self.numeric_projection(numeric_tensor)
        return torch.tanh(self.output(torch.cat([token_embed, numeric_embed], dim=-1)))
