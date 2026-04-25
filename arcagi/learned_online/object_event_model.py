from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from arcagi.learned_online.event_tokens import (
    ACTION_NUMERIC_DIM,
    ACTION_OTHER,
    DIR_RIGHT,
    OUT_MECHANIC_CHANGE,
    OUT_OBJECTIVE_PROGRESS,
    OUT_NO_EFFECT_NONPROGRESS,
    OUT_REWARD_PROGRESS,
    OUTCOME_DIM,
    STATE_DELTA_DIM,
    STATE_NUMERIC_DIM,
)

DEFAULT_OBJECT_EVENT_LOSS_WEIGHTS = {
    "outcome": 1.0,
    "rank": 1.0,
    "inverse": 0.5,
    "value": 0.25,
    "delta": 0.05,
}


@dataclass(frozen=True)
class ObjectEventModelConfig:
    d_model: int = 128
    n_heads: int = 4
    state_layers: int = 2
    action_cross_layers: int = 2
    ff_mult: int = 4
    dropout: float = 0.05
    online_rank: int = 8
    outcome_dim: int = OUTCOME_DIM
    delta_dim: int = STATE_DELTA_DIM

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectEventModelConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class ObjectEventModelOutput:
    outcome_logits: torch.Tensor
    delta_pred: torch.Tensor
    value_logits: torch.Tensor
    action_repr: torch.Tensor
    encoded_state: torch.Tensor


def policy_rank_logits_from_predictions(
    predictions: ObjectEventModelOutput,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    logits = (
        predictions.outcome_logits[..., OUT_OBJECTIVE_PROGRESS]
        + predictions.outcome_logits[..., OUT_REWARD_PROGRESS]
        - predictions.outcome_logits[..., OUT_NO_EFFECT_NONPROGRESS]
        + predictions.value_logits
    )
    if action_mask is not None:
        logits = logits.masked_fill(~action_mask.bool(), -1.0e9)
    return logits


class LowRankOnlineAdapter(nn.Module):
    def __init__(self, d_model: int, rank: int) -> None:
        super().__init__()
        self.down = nn.Linear(d_model, rank, bias=False)
        self.up = nn.Linear(rank, d_model, bias=False)
        nn.init.normal_(self.down.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.up.weight)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.up(torch.tanh(self.down(values)))


class _CrossBlock(nn.Module):
    def __init__(self, d_model: int, n_heads: int, ff_mult: int, dropout: float) -> None:
        super().__init__()
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout, batch_first=True)
        self.norm_attn = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * ff_mult),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * ff_mult, d_model),
        )
        self.norm_ff = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, actions: torch.Tensor, context: torch.Tensor, context_mask: torch.Tensor) -> torch.Tensor:
        attended, _weights = self.attn(
            actions,
            context,
            context,
            key_padding_mask=~context_mask,
            need_weights=False,
        )
        actions = self.norm_attn(actions + self.dropout(attended))
        actions = self.norm_ff(actions + self.dropout(self.ff(actions)))
        return actions


class ObjectEventModel(nn.Module):
    def __init__(self, config: ObjectEventModelConfig | None = None) -> None:
        super().__init__()
        self.config = config or ObjectEventModelConfig()
        d_model = int(self.config.d_model)
        self.state_numeric = nn.Linear(STATE_NUMERIC_DIM, d_model)
        self.state_type = nn.Embedding(7, d_model)
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=int(self.config.n_heads),
            dim_feedforward=d_model * int(self.config.ff_mult),
            dropout=float(self.config.dropout),
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.state_encoder = nn.TransformerEncoder(encoder_layer, num_layers=int(self.config.state_layers))
        self.action_numeric = nn.Linear(ACTION_NUMERIC_DIM, d_model)
        self.action_type = nn.Embedding(ACTION_OTHER + 1, d_model)
        self.direction = nn.Embedding(DIR_RIGHT + 1, d_model)
        self.state_action_film = nn.Sequential(
            nn.LayerNorm(d_model),
            nn.Linear(d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 2 * d_model),
        )
        nn.init.zeros_(self.state_action_film[-1].weight)
        nn.init.zeros_(self.state_action_film[-1].bias)
        self.state_action_pair_residual = nn.Sequential(
            nn.LayerNorm(3 * d_model),
            nn.Linear(3 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )
        nn.init.zeros_(self.state_action_pair_residual[-1].weight)
        nn.init.zeros_(self.state_action_pair_residual[-1].bias)
        self.cross_blocks = nn.ModuleList(
            [
                _CrossBlock(d_model, int(self.config.n_heads), int(self.config.ff_mult), float(self.config.dropout))
                for _ in range(int(self.config.action_cross_layers))
            ]
        )
        self.online_adapter = LowRankOnlineAdapter(d_model, int(self.config.online_rank))
        self.outcome_head = nn.Linear(d_model, int(self.config.outcome_dim))
        self.delta_head = nn.Linear(d_model, int(self.config.delta_dim))
        self.value_head = nn.Linear(d_model, 1)
        self.fast_outcome_head = nn.Linear(d_model, int(self.config.outcome_dim))
        self.fast_delta_head = nn.Linear(d_model, int(self.config.delta_dim))
        self.fast_value_head = nn.Linear(d_model, 1)
        nn.init.zeros_(self.fast_outcome_head.weight)
        nn.init.zeros_(self.fast_outcome_head.bias)
        nn.init.zeros_(self.fast_delta_head.weight)
        nn.init.zeros_(self.fast_delta_head.bias)
        nn.init.zeros_(self.fast_value_head.weight)
        nn.init.zeros_(self.fast_value_head.bias)
        self.event_encoder = nn.Sequential(
            nn.Linear(int(self.config.outcome_dim) + int(self.config.delta_dim), d_model),
            nn.GELU(),
            nn.Linear(d_model, d_model),
        )

    def forward(
        self,
        *,
        state_numeric: torch.Tensor,
        state_type_ids: torch.Tensor,
        state_mask: torch.Tensor,
        action_numeric: torch.Tensor,
        action_type_ids: torch.Tensor,
        direction_ids: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        session_belief: torch.Tensor | None = None,
        level_belief: torch.Tensor | None = None,
    ) -> ObjectEventModelOutput:
        del action_mask
        state_type_ids = state_type_ids.clamp(min=0, max=6)
        encoded_state = self.state_numeric(state_numeric) + self.state_type(state_type_ids)
        encoded_state = self.state_encoder(encoded_state, src_key_padding_mask=~state_mask.bool())
        batch_size = state_numeric.shape[0]
        d_model = encoded_state.shape[-1]
        if session_belief is None:
            session_belief = torch.zeros((batch_size, d_model), dtype=encoded_state.dtype, device=encoded_state.device)
        if level_belief is None:
            level_belief = torch.zeros((batch_size, d_model), dtype=encoded_state.dtype, device=encoded_state.device)
        belief_context = torch.stack([session_belief, level_belief], dim=1)
        context = torch.cat([belief_context, encoded_state], dim=1)
        belief_mask = torch.ones((batch_size, 2), dtype=torch.bool, device=encoded_state.device)
        context_mask = torch.cat([belief_mask, state_mask.bool()], dim=1)
        action_type_ids = action_type_ids.clamp(min=0, max=ACTION_OTHER)
        direction_ids = direction_ids.clamp(min=0, max=DIR_RIGHT)
        action_repr = (
            self.action_numeric(action_numeric)
            + self.action_type(action_type_ids)
            + self.direction(direction_ids)
        )
        state_weights = state_mask.bool().unsqueeze(-1).to(dtype=encoded_state.dtype)
        state_summary = (encoded_state * state_weights).sum(dim=1) / state_weights.sum(dim=1).clamp_min(1.0)
        gamma, beta = self.state_action_film(state_summary).chunk(2, dim=-1)
        gamma = 0.1 * torch.tanh(gamma).unsqueeze(1)
        beta = 0.1 * torch.tanh(beta).unsqueeze(1)
        action_repr = action_repr * (1.0 + gamma) + beta
        expanded_state = state_summary.unsqueeze(1).expand_as(action_repr)
        pair_input = torch.cat([action_repr, expanded_state, action_repr * expanded_state], dim=-1)
        action_repr = action_repr + 0.1 * self.state_action_pair_residual(pair_input)
        for block in self.cross_blocks:
            action_repr = block(action_repr, context, context_mask)
        adapted = action_repr + self.online_adapter(action_repr)
        outcome_logits = self.outcome_head(adapted) + self.fast_outcome_head(adapted)
        delta_pred = self.delta_head(adapted) + self.fast_delta_head(adapted)
        value_logits = (self.value_head(adapted) + self.fast_value_head(adapted)).squeeze(-1)
        return ObjectEventModelOutput(
            outcome_logits=outcome_logits,
            delta_pred=delta_pred,
            value_logits=value_logits,
            action_repr=adapted,
            encoded_state=encoded_state,
        )

    def inverse_logits(
        self,
        action_repr: torch.Tensor,
        *,
        target_outcome: torch.Tensor,
        target_delta: torch.Tensor,
        action_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        event = torch.cat([target_outcome, target_delta], dim=-1)
        event_repr = self.event_encoder(event)
        logits = torch.einsum("bad,bd->ba", action_repr, event_repr) / max(float(action_repr.shape[-1]) ** 0.5, 1.0)
        if action_mask is not None:
            logits = logits.masked_fill(~action_mask.bool(), -1.0e9)
        return logits

    def loss(
        self,
        output: ObjectEventModelOutput,
        *,
        target_outcome: torch.Tensor,
        target_delta: torch.Tensor,
        actual_action_index: torch.Tensor,
        action_mask: torch.Tensor,
        value_targets: torch.Tensor | None = None,
        candidate_outcome_targets: torch.Tensor | None = None,
        candidate_value_targets: torch.Tensor | None = None,
        candidate_delta_targets: torch.Tensor | None = None,
        candidate_mask: torch.Tensor | None = None,
        loss_weights: dict[str, float] | None = None,
        anchor_weight: float = 1.0e-4,
    ) -> dict[str, torch.Tensor]:
        batch = torch.arange(output.outcome_logits.shape[0], device=output.outcome_logits.device)
        chosen_outcome_logits = output.outcome_logits[batch, actual_action_index]
        chosen_delta = output.delta_pred[batch, actual_action_index]
        valid_mask = action_mask.bool() if candidate_mask is None else (action_mask.bool() & candidate_mask.bool())
        if candidate_outcome_targets is None:
            outcome_loss = F.binary_cross_entropy_with_logits(chosen_outcome_logits, target_outcome)
        else:
            outcome_loss = F.binary_cross_entropy_with_logits(
                output.outcome_logits[valid_mask],
                candidate_outcome_targets[valid_mask],
            )
        if candidate_delta_targets is None:
            delta_loss = F.smooth_l1_loss(chosen_delta, target_delta)
        else:
            delta_loss = F.smooth_l1_loss(output.delta_pred[valid_mask], candidate_delta_targets[valid_mask])
        inverse_logits = self.inverse_logits(
            output.action_repr,
            target_outcome=target_outcome,
            target_delta=target_delta,
            action_mask=action_mask,
        )
        inverse_loss = F.cross_entropy(inverse_logits, actual_action_index)
        candidate_values = candidate_value_targets if candidate_value_targets is not None else value_targets
        if candidate_values is None:
            value_target = torch.clamp(
                target_outcome[:, OUT_OBJECTIVE_PROGRESS] + 0.5 * target_outcome[:, OUT_MECHANIC_CHANGE],
                min=0.0,
                max=1.0,
            )
            chosen_value = output.value_logits[batch, actual_action_index]
            value_loss = F.binary_cross_entropy_with_logits(chosen_value, value_target)
        else:
            masked_targets = candidate_values[valid_mask]
            masked_logits = output.value_logits[valid_mask]
            positives = torch.clamp(masked_targets.sum(), min=1.0)
            negatives = torch.clamp((1.0 - masked_targets).sum(), min=1.0)
            pos_weight = (negatives / positives).detach()
            value_loss = F.binary_cross_entropy_with_logits(masked_logits, masked_targets, pos_weight=pos_weight)
        rank_logits = policy_rank_logits_from_predictions(output, action_mask)
        rank_loss = F.cross_entropy(rank_logits, actual_action_index)
        anchor = self.online_adapter.up.weight.square().mean()
        anchor = anchor + self.fast_outcome_head.weight.square().mean()
        anchor = anchor + self.fast_delta_head.weight.square().mean()
        anchor = anchor + self.fast_value_head.weight.square().mean()
        weights = dict(DEFAULT_OBJECT_EVENT_LOSS_WEIGHTS)
        if loss_weights:
            weights.update({key: float(value) for key, value in loss_weights.items()})
        total = (
            weights["outcome"] * outcome_loss
            + weights["rank"] * rank_loss
            + weights["inverse"] * inverse_loss
            + weights["value"] * value_loss
            + weights["delta"] * delta_loss
            + anchor_weight * anchor
        )
        return {
            "loss": total,
            "outcome_loss": outcome_loss.detach(),
            "delta_loss": delta_loss.detach(),
            "inverse_loss": inverse_loss.detach(),
            "rank_loss": rank_loss.detach(),
            "value_loss": value_loss.detach(),
            "anchor_loss": anchor.detach(),
        }

    def online_parameters(self) -> list[nn.Parameter]:
        return [
            *self.online_adapter.parameters(),
            *self.fast_outcome_head.parameters(),
            *self.fast_delta_head.parameters(),
            *self.fast_value_head.parameters(),
        ]
