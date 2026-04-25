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
    OBJECT,
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
    rank_logits: torch.Tensor | None = None
    candidate_state_attn: torch.Tensor | None = None


def policy_rank_logits_from_predictions(
    predictions: ObjectEventModelOutput,
    action_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    semantic_logits = (
        predictions.outcome_logits[..., OUT_OBJECTIVE_PROGRESS]
        + predictions.outcome_logits[..., OUT_REWARD_PROGRESS]
        - predictions.outcome_logits[..., OUT_NO_EFFECT_NONPROGRESS]
        + predictions.value_logits
    )
    logits = semantic_logits
    if predictions.rank_logits is not None:
        logits = predictions.rank_logits + 0.25 * semantic_logits
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


class CandidateStateCompatibility(nn.Module):
    def __init__(self, d_model: int, dropout: float) -> None:
        super().__init__()
        self.query = nn.Linear(d_model, d_model)
        self.key = nn.Linear(d_model, d_model)
        self.value = nn.Linear(d_model, d_model)
        self.pair_norm = nn.LayerNorm(8 * d_model)
        self.pair_mlp = nn.Sequential(
            nn.Linear(8 * d_model, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
        )
        self.out_norm = nn.LayerNorm(d_model)
        nn.init.normal_(self.pair_mlp[-1].weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.pair_mlp[-1].bias)

    def forward(
        self,
        *,
        action_hidden: torch.Tensor,
        state_hidden: torch.Tensor,
        state_mask: torch.Tensor,
        state_type_ids: torch.Tensor,
        state_numeric: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = self.query(action_hidden)
        key = self.key(state_hidden)
        value = self.value(state_hidden)
        scale = max(float(query.shape[-1]) ** 0.5, 1.0)
        attention_logits = torch.matmul(query, key.transpose(-1, -2)) / scale
        attention_logits = attention_logits.masked_fill(~state_mask.bool().unsqueeze(1), -1.0e9)
        attention = torch.softmax(attention_logits, dim=-1)
        context = torch.matmul(attention, value)
        object_mask = state_mask.bool() & (state_type_ids == OBJECT)
        agent_object_mask = object_mask & (state_numeric[..., 9] > 0.5)
        object_summary = _masked_mean(state_hidden, object_mask, fallback_mask=state_mask.bool())
        agent_summary = _masked_mean(state_hidden, agent_object_mask, fallback_mask=object_mask)
        object_summary = object_summary.unsqueeze(1).expand_as(action_hidden)
        agent_summary = agent_summary.unsqueeze(1).expand_as(action_hidden)
        pair_features = torch.cat(
            [
                action_hidden,
                context,
                object_summary,
                agent_summary,
                action_hidden * context,
                action_hidden * agent_summary,
                torch.abs(action_hidden - context),
                torch.abs(action_hidden - agent_summary),
            ],
            dim=-1,
        )
        delta = self.pair_mlp(self.pair_norm(pair_features))
        return self.out_norm(action_hidden + delta), attention


class EventRelationMemoryRank(nn.Module):
    def __init__(
        self,
        *,
        cue_feature_dim: int,
        target_feature_dim: int,
        outcome_dim: int,
        belief_dim: int,
        key_dim: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.key_dim = int(key_dim)
        self.belief_dim = int(belief_dim)
        self.cue_key = nn.Sequential(
            nn.LayerNorm(cue_feature_dim),
            nn.Linear(cue_feature_dim, self.key_dim),
            nn.Tanh(),
        )
        self.target_key = nn.Sequential(
            nn.LayerNorm(target_feature_dim),
            nn.Linear(target_feature_dim, self.key_dim),
            nn.Tanh(),
        )
        self.outcome_key = nn.Sequential(
            nn.LayerNorm(outcome_dim),
            nn.Linear(outcome_dim, self.key_dim),
            nn.Tanh(),
        )
        relation_dim = 15 * self.key_dim
        self.rank_mlp = nn.Sequential(
            nn.LayerNorm(relation_dim),
            nn.Linear(relation_dim, 4 * self.key_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(4 * self.key_dim, 1),
        )

    def cue_key_features(self, cue_features: torch.Tensor) -> torch.Tensor:
        key = self.cue_key(cue_features)
        direct = torch.zeros_like(key)
        if self.key_dim > 0:
            direct[..., 0] = cue_features[..., 0]
        return torch.tanh(0.1 * key + direct)

    def target_key_features(self, target_features: torch.Tensor) -> torch.Tensor:
        key = self.target_key(target_features)
        direct = torch.zeros_like(key)
        if self.key_dim > 0:
            direct[..., 0] = target_features[..., 13]
        if self.key_dim > 1:
            direct[..., 1] = target_features[..., 11]
        return torch.tanh(0.1 * key + direct)

    def observed_delta(
        self,
        *,
        cue_features: torch.Tensor,
        target_features: torch.Tensor,
        outcome_targets: torch.Tensor,
    ) -> torch.Tensor:
        cue = self.cue_key_features(cue_features)
        target = self.target_key_features(target_features)
        outcome = self.outcome_key(outcome_targets)
        success_gate = torch.clamp(
            outcome_targets[:, OUT_OBJECTIVE_PROGRESS] + outcome_targets[:, OUT_REWARD_PROGRESS],
            min=0.0,
            max=1.0,
        ).unsqueeze(-1)
        no_effect_gate = outcome_targets[:, OUT_NO_EFFECT_NONPROGRESS].unsqueeze(-1)
        key_dim = self.key_dim
        delta = cue.new_zeros((cue.shape[0], self.belief_dim))
        delta[:, 0:key_dim] = success_gate * cue
        delta[:, key_dim : 2 * key_dim] = success_gate * target
        delta[:, 2 * key_dim : 3 * key_dim] = no_effect_gate * cue
        delta[:, 3 * key_dim : 4 * key_dim] = no_effect_gate * target
        delta[:, 4 * key_dim : 5 * key_dim] = outcome
        return delta

    def rank_delta(
        self,
        *,
        query_cue_features: torch.Tensor,
        candidate_target_features: torch.Tensor,
        session_belief: torch.Tensor,
        level_belief: torch.Tensor,
    ) -> torch.Tensor:
        belief = session_belief + level_belief
        key_dim = self.key_dim
        pos_cue = belief[:, 0:key_dim]
        pos_target = belief[:, key_dim : 2 * key_dim]
        neg_cue = belief[:, 2 * key_dim : 3 * key_dim]
        neg_target = belief[:, 3 * key_dim : 4 * key_dim]
        outcome = belief[:, 4 * key_dim : 5 * key_dim]
        query_cue = self.cue_key_features(query_cue_features)
        candidate_target = self.target_key_features(candidate_target_features)
        query_cue_expanded = query_cue.unsqueeze(1).expand_as(candidate_target)
        pos_cue_expanded = pos_cue.unsqueeze(1).expand_as(candidate_target)
        pos_target_expanded = pos_target.unsqueeze(1).expand_as(candidate_target)
        neg_cue_expanded = neg_cue.unsqueeze(1).expand_as(candidate_target)
        neg_target_expanded = neg_target.unsqueeze(1).expand_as(candidate_target)
        outcome_expanded = outcome.unsqueeze(1).expand_as(candidate_target)
        pos_same_cue = _soft_same(query_cue_expanded, pos_cue_expanded)
        pos_same_target = _soft_same(candidate_target, pos_target_expanded)
        neg_same_cue = _soft_same(query_cue_expanded, neg_cue_expanded)
        neg_same_target = _soft_same(candidate_target, neg_target_expanded)
        pos_cue_scalar = pos_same_cue[..., 0]
        pos_target_scalar = pos_same_target[..., 0]
        neg_cue_scalar = neg_same_cue[..., 0]
        neg_target_scalar = neg_same_target[..., 0]
        pos_strength = torch.clamp(pos_cue[:, :1].abs(), 0.0, 1.0)
        neg_strength = torch.clamp(neg_cue[:, :1].abs(), 0.0, 1.0)
        same_relation = pos_cue_scalar * pos_target_scalar + (1.0 - pos_cue_scalar) * (1.0 - pos_target_scalar)
        different_relation = neg_cue_scalar * (1.0 - neg_target_scalar) + (1.0 - neg_cue_scalar) * neg_target_scalar
        candidate_contains = candidate_target_features[..., 11] * (1.0 - candidate_target_features[..., 24])
        object_target_prior = 12.0 * candidate_contains
        relation_prior = object_target_prior + 40.0 * candidate_contains * (
            pos_strength * same_relation + neg_strength * different_relation
        )
        features = torch.cat(
            [
                query_cue_expanded,
                candidate_target,
                pos_same_cue,
                pos_same_target,
                pos_same_cue * pos_same_target,
                torch.abs(query_cue_expanded - pos_cue_expanded),
                torch.abs(candidate_target - pos_target_expanded),
                candidate_target * pos_target_expanded,
                neg_same_cue,
                neg_same_target,
                neg_same_cue * neg_same_target,
                torch.abs(query_cue_expanded - neg_cue_expanded),
                torch.abs(candidate_target - neg_target_expanded),
                candidate_target * neg_target_expanded,
                outcome_expanded,
            ],
            dim=-1,
        )
        return self.rank_mlp(features).squeeze(-1) + relation_prior


class RawCandidateStateRank(nn.Module):
    def __init__(self, d_model: int, dropout: float, relation_memory: EventRelationMemoryRank) -> None:
        super().__init__()
        self.relation_memory = relation_memory
        input_dim = ACTION_NUMERIC_DIM + (4 * STATE_NUMERIC_DIM) + (2 * STATE_NUMERIC_DIM) + (4 * d_model)
        self.rank_mlp = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, 2 * d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * d_model, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )

    def forward(
        self,
        *,
        action_numeric: torch.Tensor,
        state_numeric: torch.Tensor,
        state_type_ids: torch.Tensor,
        state_mask: torch.Tensor,
        session_belief: torch.Tensor,
        level_belief: torch.Tensor,
    ) -> torch.Tensor:
        object_mask = state_mask.bool() & (state_type_ids == OBJECT)
        agent_object_mask = object_mask & (state_numeric[..., 9] > 0.5)
        meta_mask = state_mask.bool() & (state_type_ids == 1)
        object_summary = _masked_mean(state_numeric, object_mask, fallback_mask=state_mask.bool())
        agent_summary = _masked_mean(state_numeric, agent_object_mask, fallback_mask=object_mask)
        meta_summary = _masked_mean(state_numeric, meta_mask, fallback_mask=state_mask.bool())
        global_summary = _masked_mean(state_numeric, state_mask.bool(), fallback_mask=state_mask.bool())
        action_head = action_numeric[..., :STATE_NUMERIC_DIM]
        expanded_agent = agent_summary.unsqueeze(1).expand(-1, action_numeric.shape[1], -1)
        expanded_session = session_belief.unsqueeze(1).expand(-1, action_numeric.shape[1], -1)
        expanded_level = level_belief.unsqueeze(1).expand(-1, action_numeric.shape[1], -1)
        action_hidden_proxy = action_numeric[..., : session_belief.shape[-1]]
        if action_hidden_proxy.shape[-1] < session_belief.shape[-1]:
            padding = torch.zeros(
                (
                    action_hidden_proxy.shape[0],
                    action_hidden_proxy.shape[1],
                    session_belief.shape[-1] - action_hidden_proxy.shape[-1],
                ),
                dtype=action_hidden_proxy.dtype,
                device=action_hidden_proxy.device,
            )
            action_hidden_proxy = torch.cat([action_hidden_proxy, padding], dim=-1)
        features = torch.cat(
            [
                action_numeric,
                object_summary.unsqueeze(1).expand(-1, action_numeric.shape[1], -1),
                agent_summary.unsqueeze(1).expand(-1, action_numeric.shape[1], -1),
                meta_summary.unsqueeze(1).expand(-1, action_numeric.shape[1], -1),
                global_summary.unsqueeze(1).expand(-1, action_numeric.shape[1], -1),
                action_head * expanded_agent,
                torch.abs(action_head - expanded_agent),
                expanded_session,
                expanded_level,
                action_hidden_proxy * expanded_session,
                torch.abs(action_hidden_proxy - expanded_session),
            ],
            dim=-1,
        )
        relation_rank = self.relation_memory.rank_delta(
            query_cue_features=agent_summary,
            candidate_target_features=action_numeric,
            session_belief=session_belief,
            level_belief=level_belief,
        )
        return self.rank_mlp(features).squeeze(-1) + relation_rank


def _soft_same(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.exp(-20.0 * torch.abs(left - right))


def _masked_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    *,
    fallback_mask: torch.Tensor,
) -> torch.Tensor:
    mask = mask.bool()
    fallback_mask = fallback_mask.bool()
    active = torch.where(mask.any(dim=1, keepdim=True), mask, fallback_mask)
    weights = active.unsqueeze(-1).to(dtype=values.dtype)
    return (values * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


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
        self.event_relation_memory_rank = EventRelationMemoryRank(
            cue_feature_dim=STATE_NUMERIC_DIM,
            target_feature_dim=ACTION_NUMERIC_DIM,
            outcome_dim=int(self.config.outcome_dim),
            belief_dim=d_model,
            key_dim=max(4, min(16, d_model // 5)),
            dropout=float(self.config.dropout),
        )
        self.candidate_state_compat = CandidateStateCompatibility(d_model, float(self.config.dropout))
        self.raw_candidate_rank = RawCandidateStateRank(
            d_model,
            float(self.config.dropout),
            self.event_relation_memory_rank,
        )
        self.rank_score_head = nn.Linear(d_model, 1)
        nn.init.normal_(self.rank_score_head.weight, mean=0.0, std=0.02)
        nn.init.zeros_(self.rank_score_head.bias)
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
        raw_belief_dim = ACTION_NUMERIC_DIM + (2 * STATE_NUMERIC_DIM) + (2 * STATE_NUMERIC_DIM)
        self.observed_event_belief_encoder = nn.Sequential(
            nn.LayerNorm(raw_belief_dim + int(self.config.outcome_dim) + int(self.config.delta_dim)),
            nn.Linear(raw_belief_dim + int(self.config.outcome_dim) + int(self.config.delta_dim), 2 * d_model),
            nn.GELU(),
            nn.Linear(2 * d_model, d_model),
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
        action_repr, candidate_state_attn = self.candidate_state_compat(
            action_hidden=action_repr,
            state_hidden=encoded_state,
            state_mask=state_mask.bool(),
            state_type_ids=state_type_ids,
            state_numeric=state_numeric,
        )
        adapted = action_repr + self.online_adapter(action_repr)
        outcome_logits = self.outcome_head(adapted) + self.fast_outcome_head(adapted)
        delta_pred = self.delta_head(adapted) + self.fast_delta_head(adapted)
        value_logits = (self.value_head(adapted) + self.fast_value_head(adapted)).squeeze(-1)
        rank_logits = self.rank_score_head(adapted).squeeze(-1) + self.raw_candidate_rank(
            action_numeric=action_numeric,
            state_numeric=state_numeric,
            state_type_ids=state_type_ids,
            state_mask=state_mask.bool(),
            session_belief=session_belief,
            level_belief=level_belief,
        )
        return ObjectEventModelOutput(
            outcome_logits=outcome_logits,
            delta_pred=delta_pred,
            value_logits=value_logits,
            action_repr=adapted,
            encoded_state=encoded_state,
            rank_logits=rank_logits,
            candidate_state_attn=candidate_state_attn,
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

    def observed_event_belief_delta(
        self,
        output: ObjectEventModelOutput,
        *,
        target_outcome: torch.Tensor,
        target_delta: torch.Tensor,
        actual_action_index: torch.Tensor,
        state_numeric: torch.Tensor | None = None,
        state_type_ids: torch.Tensor | None = None,
        state_mask: torch.Tensor | None = None,
        action_numeric: torch.Tensor | None = None,
    ) -> torch.Tensor:
        batch = torch.arange(output.action_repr.shape[0], device=output.action_repr.device)
        if (
            state_numeric is not None
            and state_type_ids is not None
            and state_mask is not None
            and action_numeric is not None
        ):
            chosen_action = action_numeric[batch, actual_action_index]
            object_mask = state_mask.bool() & (state_type_ids == OBJECT)
            agent_object_mask = object_mask & (state_numeric[..., 9] > 0.5)
            agent_summary = _masked_mean(state_numeric, agent_object_mask, fallback_mask=object_mask)
            global_summary = _masked_mean(state_numeric, state_mask.bool(), fallback_mask=state_mask.bool())
            action_head = chosen_action[..., :STATE_NUMERIC_DIM]
            raw_context = torch.cat(
                [
                    chosen_action,
                    agent_summary,
                    global_summary,
                    action_head * agent_summary,
                    torch.abs(action_head - agent_summary),
                ],
                dim=-1,
            )
        else:
            chosen_action = output.action_repr[batch, actual_action_index]
            raw_context = chosen_action.new_zeros(
                (
                    chosen_action.shape[0],
                    ACTION_NUMERIC_DIM + (4 * STATE_NUMERIC_DIM),
                )
            )
        event = torch.cat([target_outcome, target_delta], dim=-1)
        if (
            state_numeric is not None
            and state_type_ids is not None
            and state_mask is not None
            and action_numeric is not None
        ):
            direct = self.event_relation_memory_rank.observed_delta(
                cue_features=agent_summary,
                target_features=chosen_action,
                outcome_targets=target_outcome,
            )
        else:
            direct = raw_context.new_zeros((raw_context.shape[0], self.config.d_model))
        learned = self.observed_event_belief_encoder(torch.cat([raw_context, event], dim=-1))
        learned = 0.05 * learned
        relation_width = min(self.event_relation_memory_rank.key_dim * 5, learned.shape[1])
        if relation_width > 0:
            learned = learned.clone()
            learned[:, :relation_width] = 0.0
        return torch.tanh(direct + learned)

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
