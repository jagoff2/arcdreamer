from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from arcagi.learned_online.event_tokens import (
    ACTION_FEATURE_CLICK_X,
    ACTION_FEATURE_CLICK_Y,
    ACTION_FEATURE_GRID_COL,
    ACTION_FEATURE_GRID_ROW,
    ACTION_FEATURE_HAS_GRID_CELL,
    ACTION_FEATURE_IS_CLICK,
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
    relation_key_dim: int = 8
    failed_action_key_dim: int = 12
    coordinate_noeffect_key_dim: int = 8
    axis_noeffect_key_dim: int = 12
    action_family_count: int = 8
    action_family_key_dim: int = 16
    action_family_temperature: float = 0.25
    outcome_dim: int = OUTCOME_DIM
    delta_dim: int = STATE_DELTA_DIM

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ObjectEventModelConfig":
        known = {field.name for field in cls.__dataclass_fields__.values()}
        return cls(**{key: value for key, value in data.items() if key in known})

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(frozen=True)
class RelationComponentOutput:
    learned: torch.Tensor
    object_prior: torch.Tensor
    positive_prior: torch.Tensor
    negative_prior: torch.Tensor
    repeat_penalty: torch.Tensor
    contradiction_gate: torch.Tensor
    total: torch.Tensor


@dataclass(frozen=True)
class RankComponentOutput:
    base: torch.Tensor
    relation: torch.Tensor
    failed_action: torch.Tensor
    coordinate_noeffect: torch.Tensor
    axis_noeffect: torch.Tensor
    total: torch.Tensor
    base_raw: torch.Tensor | None = None
    relation_raw: torch.Tensor | None = None
    failed_action_raw: torch.Tensor | None = None
    coordinate_noeffect_raw: torch.Tensor | None = None
    axis_noeffect_raw: torch.Tensor | None = None
    relation_object_prior: torch.Tensor | None = None
    relation_positive_prior: torch.Tensor | None = None
    relation_negative_prior: torch.Tensor | None = None
    relation_repeat_penalty: torch.Tensor | None = None
    relation_contradiction_gate: torch.Tensor | None = None
    component_gates: torch.Tensor | None = None
    noeffect_boost: torch.Tensor | None = None


@dataclass(frozen=True)
class ObjectEventModelOutput:
    outcome_logits: torch.Tensor
    delta_pred: torch.Tensor
    value_logits: torch.Tensor
    action_repr: torch.Tensor
    encoded_state: torch.Tensor
    rank_logits: torch.Tensor | None = None
    diagnostic_utility_logits: torch.Tensor | None = None
    diagnostic_mix_logit: torch.Tensor | None = None
    action_family_logits: torch.Tensor | None = None
    action_family_probs: torch.Tensor | None = None
    action_family_posterior_features: torch.Tensor | None = None
    candidate_state_attn: torch.Tensor | None = None
    rank_components: RankComponentOutput | None = None


@dataclass(frozen=True)
class ObservedEventBeliefDelta:
    session_delta: torch.Tensor
    level_delta: torch.Tensor


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
        self.object_prior_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.positive_prior_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.negative_prior_scale_raw = nn.Parameter(torch.tensor(0.0))
        self.repeat_penalty_scale_raw = nn.Parameter(torch.tensor(1.0))
        self.contradiction_gate = nn.Sequential(
            nn.LayerNorm(8),
            nn.Linear(8, 16),
            nn.GELU(),
            nn.Linear(16, 1),
        )
        nn.init.zeros_(self.contradiction_gate[-1].weight)
        nn.init.zeros_(self.contradiction_gate[-1].bias)

    def cue_key_features(self, cue_features: torch.Tensor) -> torch.Tensor:
        key = self.cue_key(cue_features)
        direct = torch.zeros_like(key)
        if self.key_dim > 0:
            direct[..., 0] = cue_features[..., 0]
        if self.key_dim > 1:
            direct[..., 1] = cue_features[..., 9]
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
        return self.components(
            query_cue_features=query_cue_features,
            candidate_target_features=candidate_target_features,
            session_belief=session_belief,
            level_belief=level_belief,
        ).total

    def components(
        self,
        *,
        query_cue_features: torch.Tensor,
        candidate_target_features: torch.Tensor,
        session_belief: torch.Tensor,
        level_belief: torch.Tensor,
    ) -> RelationComponentOutput:
        belief = session_belief + level_belief
        key_dim = self.key_dim
        pos_cue = belief[:, 0:key_dim]
        pos_target = belief[:, key_dim : 2 * key_dim]
        neg_cue = belief[:, 2 * key_dim : 3 * key_dim]
        neg_target = belief[:, 3 * key_dim : 4 * key_dim]
        outcome = belief[:, 4 * key_dim : 5 * key_dim]
        level_neg_cue = level_belief[:, 2 * key_dim : 3 * key_dim]
        level_neg_target = level_belief[:, 3 * key_dim : 4 * key_dim]
        query_cue = self.cue_key_features(query_cue_features)
        candidate_target = self.target_key_features(candidate_target_features)
        query_cue_expanded = query_cue.unsqueeze(1).expand_as(candidate_target)
        pos_cue_expanded = pos_cue.unsqueeze(1).expand_as(candidate_target)
        pos_target_expanded = pos_target.unsqueeze(1).expand_as(candidate_target)
        neg_cue_expanded = neg_cue.unsqueeze(1).expand_as(candidate_target)
        neg_target_expanded = neg_target.unsqueeze(1).expand_as(candidate_target)
        outcome_expanded = outcome.unsqueeze(1).expand_as(candidate_target)
        level_neg_target_expanded = level_neg_target.unsqueeze(1).expand_as(candidate_target)
        pos_same_cue = _soft_same(query_cue_expanded, pos_cue_expanded)
        pos_same_target = _soft_same(candidate_target, pos_target_expanded)
        neg_same_cue = _soft_same(query_cue_expanded, neg_cue_expanded)
        neg_same_target = _soft_same(candidate_target, neg_target_expanded)
        level_neg_same_target = _soft_same(candidate_target, level_neg_target_expanded)
        pos_cue_scalar = pos_same_cue[..., 0]
        pos_target_scalar = pos_same_target[..., 0]
        neg_cue_scalar = neg_same_cue[..., 0]
        neg_target_scalar = neg_same_target[..., 0]
        level_neg_target_scalar = level_neg_same_target[..., 0]
        pos_strength = torch.clamp(pos_cue[:, :1].abs(), 0.0, 1.0)
        neg_strength = torch.clamp(neg_cue[:, : min(2, key_dim)].abs().max(dim=-1, keepdim=True).values, 0.0, 1.0)
        level_neg_strength = torch.clamp(level_neg_cue[:, : min(2, key_dim)].abs().max(dim=-1, keepdim=True).values, 0.0, 1.0)
        same_relation = pos_cue_scalar * pos_target_scalar + (1.0 - pos_cue_scalar) * (1.0 - pos_target_scalar)
        different_relation = neg_cue_scalar * (1.0 - neg_target_scalar) + (1.0 - neg_cue_scalar) * neg_target_scalar
        candidate_contains = candidate_target_features[..., 11] * (1.0 - candidate_target_features[..., 24])
        object_scale = _bounded_scale(self.object_prior_scale_raw, 2.0)
        positive_scale = _bounded_scale(self.positive_prior_scale_raw, 4.0)
        negative_scale = _bounded_scale(self.negative_prior_scale_raw, 4.0)
        repeat_scale = _bounded_scale(self.repeat_penalty_scale_raw, 6.0)
        object_prior = object_scale * candidate_contains
        positive_prior = positive_scale * candidate_contains * pos_strength * same_relation
        negative_prior = negative_scale * candidate_contains * neg_strength * different_relation
        repeat_penalty = repeat_scale * candidate_contains * level_neg_strength * level_neg_target_scalar
        contradiction_evidence = (level_neg_strength * level_neg_target_scalar).clamp(0.0, 1.0)
        contradiction_features = torch.stack(
            [
                candidate_contains,
                pos_strength.expand_as(candidate_contains),
                neg_strength.expand_as(candidate_contains),
                level_neg_strength.expand_as(candidate_contains),
                pos_target_scalar,
                neg_target_scalar,
                level_neg_target_scalar,
                (repeat_penalty / 6.0).clamp(0.0, 1.0),
            ],
            dim=-1,
        )
        contradiction_gate = contradiction_evidence * torch.sigmoid(
            self.contradiction_gate(contradiction_features).squeeze(-1)
        )
        object_prior = object_prior * (1.0 - 0.5 * contradiction_gate)
        positive_prior = positive_prior * (1.0 - contradiction_gate)
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
        learned = self.rank_mlp(features).squeeze(-1)
        total = learned + object_prior + positive_prior + negative_prior - repeat_penalty
        return RelationComponentOutput(
            learned=learned,
            object_prior=object_prior,
            positive_prior=positive_prior,
            negative_prior=negative_prior,
            repeat_penalty=repeat_penalty,
            contradiction_gate=contradiction_gate,
            total=total,
        )


class FailedActionMemoryRank(nn.Module):
    def __init__(
        self,
        *,
        action_feature_dim: int,
        belief_dim: int,
        key_dim: int,
        start: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.key_dim = int(key_dim)
        self.start = int(start)
        self.belief_dim = int(belief_dim)
        self.action_key = nn.Sequential(
            nn.LayerNorm(action_feature_dim),
            nn.Linear(action_feature_dim, self.key_dim),
            nn.Tanh(),
        )
        feature_dim = 5 * self.key_dim
        self.rank_mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 2 * self.key_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(2 * self.key_dim, 1),
        )
        nn.init.zeros_(self.rank_mlp[-1].weight)
        nn.init.zeros_(self.rank_mlp[-1].bias)

    @property
    def stop(self) -> int:
        return int(self.start + self.key_dim)

    def observed_delta(
        self,
        *,
        selected_action_numeric: torch.Tensor,
        no_effect_gate: torch.Tensor,
    ) -> torch.Tensor:
        key = self.action_key(selected_action_numeric)
        delta = selected_action_numeric.new_zeros((selected_action_numeric.shape[0], self.belief_dim))
        delta[:, self.start : self.stop] = no_effect_gate.reshape(-1, 1) * key
        return delta

    def rank_delta(
        self,
        *,
        candidate_action_numeric: torch.Tensor,
        level_belief: torch.Tensor,
    ) -> torch.Tensor:
        candidate = self.action_key(candidate_action_numeric)
        failed = level_belief[:, None, self.start : self.stop]
        same = _soft_same(candidate, failed.expand_as(candidate))
        features = torch.cat(
            [
                candidate,
                failed.expand_as(candidate),
                same,
                candidate * failed,
                torch.abs(candidate - failed),
            ],
            dim=-1,
        )
        return self.rank_mlp(features).squeeze(-1)


class CoordinateNoEffectMemoryRank(nn.Module):
    def __init__(
        self,
        *,
        belief_dim: int,
        start: int,
        width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.start = int(start)
        self.width = max(int(width), 0)
        self.belief_dim = int(belief_dim)
        feature_dim = 12
        self.rank_mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 32),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(32, 1),
        )
        nn.init.zeros_(self.rank_mlp[-1].weight)
        nn.init.zeros_(self.rank_mlp[-1].bias)

    @property
    def stop(self) -> int:
        return int(self.start + self.width)

    def _coord_features(self, action_numeric: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_click = action_numeric[..., ACTION_FEATURE_IS_CLICK : ACTION_FEATURE_IS_CLICK + 1].clamp(0.0, 1.0)
        screen_xy = action_numeric[..., ACTION_FEATURE_CLICK_X : ACTION_FEATURE_CLICK_Y + 1].clamp(0.0, 1.0)
        has_grid = action_numeric[..., ACTION_FEATURE_HAS_GRID_CELL : ACTION_FEATURE_HAS_GRID_CELL + 1].clamp(0.0, 1.0)
        mapped_rc = action_numeric[..., ACTION_FEATURE_GRID_ROW : ACTION_FEATURE_GRID_COL + 1].clamp(0.0, 1.0) * has_grid
        return is_click, screen_xy, mapped_rc

    def observed_delta(
        self,
        *,
        selected_action_numeric: torch.Tensor,
        no_effect_gate: torch.Tensor,
    ) -> torch.Tensor:
        delta = selected_action_numeric.new_zeros((selected_action_numeric.shape[0], self.belief_dim))
        if self.width <= 0 or self.start >= self.belief_dim:
            return delta
        is_click, screen_xy, mapped_rc = self._coord_features(selected_action_numeric)
        gate = no_effect_gate.reshape(-1, 1).clamp(0.0, 1.0) * is_click
        values = torch.cat(
            [
                gate,
                gate * screen_xy[:, 0:1],
                gate * screen_xy[:, 1:2],
                gate * screen_xy[:, 0:1] * screen_xy[:, 0:1],
                gate * screen_xy[:, 1:2] * screen_xy[:, 1:2],
                gate * mapped_rc[:, 0:1],
                gate * mapped_rc[:, 1:2],
                gate,
            ],
            dim=-1,
        )
        writable = min(self.width, values.shape[-1], max(self.belief_dim - self.start, 0))
        if writable > 0:
            delta[:, self.start : self.start + writable] = values[:, :writable]
        return delta

    def rank_delta(
        self,
        *,
        candidate_action_numeric: torch.Tensor,
        level_belief: torch.Tensor,
    ) -> torch.Tensor:
        if self.width <= 0 or self.start >= level_belief.shape[-1]:
            return candidate_action_numeric.new_zeros(candidate_action_numeric.shape[:2])
        is_click, screen_xy, mapped_rc = self._coord_features(candidate_action_numeric)
        memory = level_belief[:, self.start : min(self.stop, level_belief.shape[-1])]
        if memory.shape[-1] < 8:
            padding = memory.new_zeros((memory.shape[0], 8 - memory.shape[-1]))
            memory = torch.cat([memory, padding], dim=-1)
        count = memory[:, 0:1].clamp_min(0.0)
        denom = count.clamp_min(1.0)
        mean_screen = torch.cat([memory[:, 1:2] / denom, memory[:, 2:3] / denom], dim=-1)[:, None, :]
        mean_grid = torch.cat([memory[:, 5:6] / denom, memory[:, 6:7] / denom], dim=-1)[:, None, :]
        strength = torch.tanh(memory[:, 7:8]).clamp_min(0.0)[:, None, :]
        screen_delta = screen_xy - mean_screen
        grid_delta = mapped_rc - mean_grid
        screen_d2 = (screen_delta * screen_delta).sum(dim=-1, keepdim=True)
        grid_d2 = (grid_delta * grid_delta).sum(dim=-1, keepdim=True)
        radial = torch.cat(
            [
                torch.exp(-screen_d2 / 0.0025),
                torch.exp(-screen_d2 / 0.0100),
                torch.exp(-grid_d2 / 0.0200),
                torch.exp(-grid_d2 / 0.0800),
            ],
            dim=-1,
        )
        features = torch.cat(
            [
                screen_xy,
                mapped_rc,
                radial * strength,
                torch.abs(screen_delta),
                torch.abs(grid_delta),
            ],
            dim=-1,
        )
        return is_click.squeeze(-1) * self.rank_mlp(features).squeeze(-1)


class AxisNoEffectMemoryRank(nn.Module):
    def __init__(
        self,
        *,
        belief_dim: int,
        start: int,
        width: int,
        dropout: float,
    ) -> None:
        super().__init__()
        self.start = int(start)
        self.width = max(int(width), 0)
        self.belief_dim = int(belief_dim)
        feature_dim = 26
        self.rank_mlp = nn.Sequential(
            nn.LayerNorm(feature_dim),
            nn.Linear(feature_dim, 48),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(48, 1),
        )
        nn.init.zeros_(self.rank_mlp[-1].weight)
        nn.init.zeros_(self.rank_mlp[-1].bias)

    @property
    def stop(self) -> int:
        return int(self.start + self.width)

    def _coord_features(self, action_numeric: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        is_click = action_numeric[..., ACTION_FEATURE_IS_CLICK : ACTION_FEATURE_IS_CLICK + 1].clamp(0.0, 1.0)
        screen_xy = action_numeric[..., ACTION_FEATURE_CLICK_X : ACTION_FEATURE_CLICK_Y + 1].clamp(0.0, 1.0)
        has_grid = action_numeric[..., ACTION_FEATURE_HAS_GRID_CELL : ACTION_FEATURE_HAS_GRID_CELL + 1].clamp(0.0, 1.0)
        mapped_rc = action_numeric[..., ACTION_FEATURE_GRID_ROW : ACTION_FEATURE_GRID_COL + 1].clamp(0.0, 1.0) * has_grid
        return is_click, screen_xy, mapped_rc

    def observed_delta(
        self,
        *,
        selected_action_numeric: torch.Tensor,
        no_effect_gate: torch.Tensor,
    ) -> torch.Tensor:
        delta = selected_action_numeric.new_zeros((selected_action_numeric.shape[0], self.belief_dim))
        if self.width <= 0 or self.start >= self.belief_dim:
            return delta
        is_click, screen_xy, mapped_rc = self._coord_features(selected_action_numeric)
        gate = no_effect_gate.reshape(-1, 1).clamp(0.0, 1.0) * is_click
        x = screen_xy[:, 0:1]
        y = screen_xy[:, 1:2]
        row = mapped_rc[:, 0:1]
        col = mapped_rc[:, 1:2]
        values = torch.cat(
            [
                gate,
                gate * x,
                gate * y,
                gate * x * x,
                gate * y * y,
                gate * row,
                gate * col,
                gate * row * row,
                gate * col * col,
                gate,
                gate,
                gate,
            ],
            dim=-1,
        )
        writable = min(self.width, values.shape[-1], max(self.belief_dim - self.start, 0))
        if writable > 0:
            delta[:, self.start : self.start + writable] = values[:, :writable]
        return delta

    def rank_delta(
        self,
        *,
        candidate_action_numeric: torch.Tensor,
        level_belief: torch.Tensor,
    ) -> torch.Tensor:
        if self.width <= 0 or self.start >= level_belief.shape[-1]:
            return candidate_action_numeric.new_zeros(candidate_action_numeric.shape[:2])
        is_click, screen_xy, mapped_rc = self._coord_features(candidate_action_numeric)
        memory = level_belief[:, self.start : min(self.stop, level_belief.shape[-1])]
        if memory.shape[-1] < 12:
            padding = memory.new_zeros((memory.shape[0], 12 - memory.shape[-1]))
            memory = torch.cat([memory, padding], dim=-1)
        count = memory[:, 0:1].clamp_min(0.0)
        denom = count.clamp_min(1.0)
        mean_x = memory[:, 1:2] / denom
        mean_y = memory[:, 2:3] / denom
        mean_x2 = memory[:, 3:4] / denom
        mean_y2 = memory[:, 4:5] / denom
        var_x = (mean_x2 - mean_x * mean_x).clamp_min(0.0)
        var_y = (mean_y2 - mean_y * mean_y).clamp_min(0.0)
        mean_r = memory[:, 5:6] / denom
        mean_c = memory[:, 6:7] / denom
        mean_r2 = memory[:, 7:8] / denom
        mean_c2 = memory[:, 8:9] / denom
        var_r = (mean_r2 - mean_r * mean_r).clamp_min(0.0)
        var_c = (mean_c2 - mean_c * mean_c).clamp_min(0.0)
        strength = torch.tanh(memory[:, 9:10]).clamp_min(0.0)
        x = screen_xy[..., 0:1]
        y = screen_xy[..., 1:2]
        row = mapped_rc[..., 0:1]
        col = mapped_rc[..., 1:2]
        dx2 = (x - mean_x[:, None, :]) ** 2
        dy2 = (y - mean_y[:, None, :]) ** 2
        dr2 = (row - mean_r[:, None, :]) ** 2
        dc2 = (col - mean_c[:, None, :]) ** 2
        same_x_tight = torch.exp(-dx2 / 0.0008)
        same_x_wide = torch.exp(-dx2 / 0.0040)
        same_y_tight = torch.exp(-dy2 / 0.0008)
        same_y_wide = torch.exp(-dy2 / 0.0040)
        same_c_tight = torch.exp(-dc2 / 0.0100)
        same_c_wide = torch.exp(-dc2 / 0.0400)
        same_r_tight = torch.exp(-dr2 / 0.0100)
        same_r_wide = torch.exp(-dr2 / 0.0400)
        strength_expanded = strength[:, None, :].expand_as(x)
        count_expanded = count[:, None, :].clamp(max=20.0).expand_as(x) / 20.0
        column_like = torch.sigmoid((var_y - var_x) * 50.0)[:, None, :] * strength_expanded
        row_like = torch.sigmoid((var_x - var_y) * 50.0)[:, None, :] * strength_expanded
        grid_column_like = torch.sigmoid((var_r - var_c) * 50.0)[:, None, :] * strength_expanded
        grid_row_like = torch.sigmoid((var_c - var_r) * 50.0)[:, None, :] * strength_expanded
        features = torch.cat(
            [
                screen_xy,
                mapped_rc,
                strength_expanded,
                count_expanded,
                var_x[:, None, :].expand_as(x),
                var_y[:, None, :].expand_as(x),
                var_r[:, None, :].expand_as(x),
                var_c[:, None, :].expand_as(x),
                same_x_tight,
                same_x_wide,
                same_y_tight,
                same_y_wide,
                same_c_tight,
                same_c_wide,
                same_r_tight,
                same_r_wide,
                same_x_wide * column_like,
                same_c_wide * grid_column_like,
                same_y_wide * row_like,
                same_r_wide * grid_row_like,
                torch.abs(x - mean_x[:, None, :]),
                torch.abs(y - mean_y[:, None, :]),
                torch.abs(row - mean_r[:, None, :]),
                torch.abs(col - mean_c[:, None, :]),
            ],
            dim=-1,
        )
        return is_click.squeeze(-1) * self.rank_mlp(features).squeeze(-1)


class ActionFamilyBelief(nn.Module):
    def __init__(
        self,
        *,
        action_feature_dim: int,
        belief_dim: int,
        start: int,
        num_families: int = 8,
        key_dim: int = 16,
        temperature: float = 0.25,
    ) -> None:
        super().__init__()
        self.start = int(start)
        self.num_families = max(int(num_families), 0)
        self.width = 3 * self.num_families
        self.stop = self.start + self.width
        self.belief_dim = int(belief_dim)
        self.temperature = max(float(temperature), 1.0e-6)
        hidden = max(int(key_dim), 2)
        self.action_key = nn.Sequential(
            nn.LayerNorm(int(action_feature_dim)),
            nn.Linear(int(action_feature_dim), hidden),
            nn.Tanh(),
        )
        self.prototype = nn.Parameter(torch.randn(self.num_families, hidden) * 0.05)

    def family_logits(self, action_numeric: torch.Tensor) -> torch.Tensor:
        if self.num_families <= 0:
            return action_numeric.new_zeros((*action_numeric.shape[:-1], 0))
        key = self.action_key(action_numeric)
        return torch.einsum("...d,kd->...k", key, self.prototype) / self.temperature

    def family_probs(self, action_numeric: torch.Tensor) -> torch.Tensor:
        logits = self.family_logits(action_numeric)
        if logits.shape[-1] <= 0:
            return logits
        return torch.softmax(logits, dim=-1)

    def observed_delta(
        self,
        *,
        selected_action_numeric: torch.Tensor,
        target_outcome: torch.Tensor,
    ) -> torch.Tensor:
        delta = selected_action_numeric.new_zeros((selected_action_numeric.shape[0], self.belief_dim))
        if self.num_families <= 0 or self.width <= 0:
            return delta
        probs = self.family_probs(selected_action_numeric)
        success = (
            target_outcome[:, OUT_OBJECTIVE_PROGRESS] + target_outcome[:, OUT_REWARD_PROGRESS]
        ).clamp(0.0, 1.0).unsqueeze(-1)
        no_effect = target_outcome[:, OUT_NO_EFFECT_NONPROGRESS].clamp(0.0, 1.0).unsqueeze(-1)
        k = self.num_families
        delta[:, self.start : self.start + k] = probs
        delta[:, self.start + k : self.start + 2 * k] = probs * success
        delta[:, self.start + 2 * k : self.start + 3 * k] = probs * no_effect
        return delta

    def posterior_features(
        self,
        *,
        candidate_action_numeric: torch.Tensor,
        level_belief: torch.Tensor,
    ) -> torch.Tensor:
        shape = (*candidate_action_numeric.shape[:2], 4)
        if self.num_families <= 0 or self.width <= 0:
            return candidate_action_numeric.new_zeros(shape)
        probs = self.family_probs(candidate_action_numeric)
        k = self.num_families
        counts = level_belief[:, self.start : self.start + k].clamp_min(0.0)
        successes = level_belief[:, self.start + k : self.start + 2 * k].clamp_min(0.0)
        no_effects = level_belief[:, self.start + 2 * k : self.start + 3 * k].clamp_min(0.0)
        alpha = 1.0 + successes
        beta = 1.0 + no_effects
        total = (alpha + beta).clamp_min(1.0e-6)
        mean = alpha / total
        variance = (alpha * beta) / ((total * total) * (total + 1.0)).clamp_min(1.0e-6)
        uncertainty = torch.sqrt(variance.clamp_min(0.0))
        candidate_mean = torch.einsum("bak,bk->ba", probs, mean)
        candidate_uncertainty = torch.einsum("bak,bk->ba", probs, uncertainty)
        candidate_count = torch.einsum("bak,bk->ba", probs, counts)
        candidate_noeffect = torch.einsum("bak,bk->ba", probs, no_effects)
        return torch.stack(
            [
                candidate_mean,
                candidate_uncertainty,
                torch.tanh(candidate_count),
                torch.tanh(candidate_noeffect),
            ],
            dim=-1,
        )


class ActionFamilyDiagnosticUtility(nn.Module):
    def __init__(
        self,
        *,
        action_feature_dim: int,
        belief_dim: int,
        d_model: int,
        dropout: float,
    ) -> None:
        super().__init__()
        input_dim = int(action_feature_dim) + (2 * int(belief_dim)) + 16 + 4
        hidden = max(int(d_model), 16)
        self.utility = nn.Sequential(
            nn.LayerNorm(input_dim),
            nn.Linear(input_dim, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, max(hidden // 2, 8)),
            nn.GELU(),
            nn.Linear(max(hidden // 2, 8), 1),
        )
        nn.init.zeros_(self.utility[-1].weight)
        nn.init.zeros_(self.utility[-1].bias)

    def forward(
        self,
        *,
        action_numeric: torch.Tensor,
        session_belief: torch.Tensor,
        level_belief: torch.Tensor,
        family_posterior_features: torch.Tensor | None = None,
    ) -> torch.Tensor:
        family_features = _action_family_features(action_numeric)
        expanded_session = session_belief[:, None, :].expand(-1, action_numeric.shape[1], -1)
        expanded_level = level_belief[:, None, :].expand(-1, action_numeric.shape[1], -1)
        if family_posterior_features is None:
            family_posterior_features = action_numeric.new_zeros((*action_numeric.shape[:2], 4))
        features = torch.cat(
            [action_numeric, expanded_session, expanded_level, family_features, family_posterior_features],
            dim=-1,
        )
        return self.utility(features).squeeze(-1)


class RawCandidateStateRank(nn.Module):
    def __init__(
        self,
        d_model: int,
        dropout: float,
        relation_memory: EventRelationMemoryRank,
        failed_action_memory: FailedActionMemoryRank,
        coordinate_noeffect_memory: CoordinateNoEffectMemoryRank,
        axis_noeffect_memory: AxisNoEffectMemoryRank,
    ) -> None:
        super().__init__()
        self.relation_memory = relation_memory
        self.failed_action_memory = failed_action_memory
        self.coordinate_noeffect_memory = coordinate_noeffect_memory
        self.axis_noeffect_memory = axis_noeffect_memory
        self.rank_component_gates = nn.Parameter(torch.tensor([1.0, 2.0, 1.0, 1.0, 1.0], dtype=torch.float32))
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
        action_mask: torch.Tensor | None = None,
        session_belief: torch.Tensor,
        level_belief: torch.Tensor,
    ) -> torch.Tensor:
        return self.components(
            action_numeric=action_numeric,
            state_numeric=state_numeric,
            state_type_ids=state_type_ids,
            state_mask=state_mask,
            action_mask=action_mask,
            session_belief=session_belief,
            level_belief=level_belief,
        ).total

    def components(
        self,
        *,
        action_numeric: torch.Tensor,
        state_numeric: torch.Tensor,
        state_type_ids: torch.Tensor,
        state_mask: torch.Tensor,
        action_mask: torch.Tensor | None = None,
        session_belief: torch.Tensor,
        level_belief: torch.Tensor,
    ) -> RankComponentOutput:
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
        base_raw = self.rank_mlp(features).squeeze(-1)
        relation_components = self.relation_memory.components(
            query_cue_features=agent_summary,
            candidate_target_features=action_numeric,
            session_belief=session_belief,
            level_belief=level_belief,
        )
        relation_raw = relation_components.total
        failed_action_raw = self.failed_action_memory.rank_delta(
            candidate_action_numeric=action_numeric,
            level_belief=level_belief,
        )
        coordinate_noeffect_raw = self.coordinate_noeffect_memory.rank_delta(
            candidate_action_numeric=action_numeric,
            level_belief=level_belief,
        )
        axis_noeffect_raw = self.axis_noeffect_memory.rank_delta(
            candidate_action_numeric=action_numeric,
            level_belief=level_belief,
        )
        base_rank = _standardize_component(base_raw, action_mask)
        relation_rank = _standardize_component(relation_raw, action_mask)
        failed_action_rank = _standardize_component(failed_action_raw, action_mask)
        coordinate_noeffect_rank = _standardize_component(coordinate_noeffect_raw, action_mask)
        axis_noeffect_rank = _standardize_component(axis_noeffect_raw, action_mask)
        gates = F.softplus(self.rank_component_gates)
        gates = gates / gates.mean().clamp_min(1.0e-6)
        noeffect_boost = self._noeffect_boost(level_belief)
        total = (
            gates[0] * base_rank
            + gates[1] * relation_rank
            + gates[2] * failed_action_rank * noeffect_boost
            + gates[3] * coordinate_noeffect_rank * noeffect_boost
            + gates[4] * axis_noeffect_rank * noeffect_boost
        )
        return RankComponentOutput(
            base=base_rank,
            relation=relation_rank,
            failed_action=failed_action_rank,
            coordinate_noeffect=coordinate_noeffect_rank,
            axis_noeffect=axis_noeffect_rank,
            total=total,
            base_raw=base_raw,
            relation_raw=relation_raw,
            failed_action_raw=failed_action_raw,
            coordinate_noeffect_raw=coordinate_noeffect_raw,
            axis_noeffect_raw=axis_noeffect_raw,
            relation_object_prior=relation_components.object_prior,
            relation_positive_prior=relation_components.positive_prior,
            relation_negative_prior=relation_components.negative_prior,
            relation_repeat_penalty=relation_components.repeat_penalty,
            relation_contradiction_gate=relation_components.contradiction_gate,
            component_gates=gates,
            noeffect_boost=noeffect_boost,
        )

    def _noeffect_boost(self, level_belief: torch.Tensor) -> torch.Tensor:
        counts: list[torch.Tensor] = []
        for memory in (self.coordinate_noeffect_memory, self.axis_noeffect_memory):
            start = int(getattr(memory, "start", 0))
            width = int(getattr(memory, "width", 0))
            if width > 0 and start < level_belief.shape[-1]:
                counts.append(level_belief[:, start : start + 1].clamp_min(0.0))
        if not counts:
            return level_belief.new_ones((level_belief.shape[0], 1))
        strength = torch.tanh(torch.stack(counts, dim=0).sum(dim=0))
        return 1.0 + torch.clamp(strength, 0.0, 2.0)

    def coordinate_noeffect_rank_delta(
        self,
        *,
        action_numeric: torch.Tensor,
        level_belief: torch.Tensor,
    ) -> torch.Tensor:
        return self.coordinate_noeffect_memory.rank_delta(
            candidate_action_numeric=action_numeric,
            level_belief=level_belief,
        )

    def axis_noeffect_rank_delta(
        self,
        *,
        action_numeric: torch.Tensor,
        level_belief: torch.Tensor,
    ) -> torch.Tensor:
        return self.axis_noeffect_memory.rank_delta(
            candidate_action_numeric=action_numeric,
            level_belief=level_belief,
        )


def _soft_same(left: torch.Tensor, right: torch.Tensor) -> torch.Tensor:
    return torch.exp(-20.0 * torch.abs(left - right))


def _bounded_scale(raw: torch.Tensor, max_value: float) -> torch.Tensor:
    return torch.as_tensor(float(max_value), dtype=raw.dtype, device=raw.device) * torch.sigmoid(raw)


def _standardize_component(
    values: torch.Tensor,
    action_mask: torch.Tensor | None,
    *,
    clamp: float = 8.0,
) -> torch.Tensor:
    if action_mask is None:
        valid = torch.ones_like(values, dtype=torch.bool)
    else:
        valid = action_mask.bool()
    active = values.masked_fill(~valid, 0.0)
    count = valid.to(dtype=values.dtype).sum(dim=1, keepdim=True).clamp_min(1.0)
    mean = active.sum(dim=1, keepdim=True) / count
    centered = torch.where(valid, values - mean, torch.zeros_like(values))
    var = (centered * centered).sum(dim=1, keepdim=True) / count
    standardized = torch.clamp(centered / torch.sqrt(var.clamp_min(1.0e-6)), -float(clamp), float(clamp))
    return standardized.masked_fill(~valid, -1.0e9)


def _action_family_features(action_numeric: torch.Tensor) -> torch.Tensor:
    is_click = action_numeric[..., ACTION_FEATURE_IS_CLICK : ACTION_FEATURE_IS_CLICK + 1].clamp(0.0, 1.0)
    screen_xy = action_numeric[..., ACTION_FEATURE_CLICK_X : ACTION_FEATURE_CLICK_Y + 1].clamp(0.0, 1.0)
    has_grid = action_numeric[..., ACTION_FEATURE_HAS_GRID_CELL : ACTION_FEATURE_HAS_GRID_CELL + 1].clamp(0.0, 1.0)
    mapped_rc = action_numeric[..., ACTION_FEATURE_GRID_ROW : ACTION_FEATURE_GRID_COL + 1].clamp(0.0, 1.0) * has_grid
    contains_object = action_numeric[..., 11:12].clamp(0.0, 1.0)
    contains_agent = action_numeric[..., 24:25].clamp(0.0, 1.0)
    contains_nonagent = contains_object * (1.0 - contains_agent)
    x = screen_xy[..., 0:1]
    y = screen_xy[..., 1:2]
    row = mapped_rc[..., 0:1]
    col = mapped_rc[..., 1:2]
    edge_x = torch.minimum(x, 1.0 - x)
    edge_y = torch.minimum(y, 1.0 - y)
    center_dx = torch.abs(x - 0.5)
    center_dy = torch.abs(y - 0.5)
    return torch.cat(
        [
            is_click,
            x,
            y,
            has_grid,
            row,
            col,
            contains_object,
            contains_agent,
            contains_nonagent,
            x * x,
            y * y,
            row * row,
            col * col,
            edge_x,
            edge_y,
            torch.sqrt((center_dx * center_dx + center_dy * center_dy).clamp_min(0.0)),
        ],
        dim=-1,
    )


def _multi_positive_rank_ce(
    logits: torch.Tensor,
    positive_mask: torch.Tensor,
    action_mask: torch.Tensor,
) -> torch.Tensor:
    valid = action_mask.bool()
    positives = positive_mask.bool() & valid
    safe_logits = logits.masked_fill(~valid, -1.0e9)
    log_probs = torch.log_softmax(safe_logits, dim=-1)
    pos_log_probs = log_probs.masked_fill(~positives, -1.0e9)
    return -torch.logsumexp(pos_log_probs, dim=-1).mean()


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
        requested_family_count = max(int(self.config.action_family_count), 0)
        family_count = min(requested_family_count, max((d_model - 32) // 3, 0))
        family_reserved_width = 3 * family_count
        pre_family_budget = max(d_model - family_reserved_width, 0)
        relation_key_dim = max(
            2,
            min(
                int(self.config.relation_key_dim),
                max((pre_family_budget - 16) // 5, 2),
            ),
        )
        self.event_relation_memory_rank = EventRelationMemoryRank(
            cue_feature_dim=STATE_NUMERIC_DIM,
            target_feature_dim=ACTION_NUMERIC_DIM,
            outcome_dim=int(self.config.outcome_dim),
            belief_dim=d_model,
            key_dim=relation_key_dim,
            dropout=float(self.config.dropout),
        )
        relation_width = int(self.event_relation_memory_rank.key_dim * 5)
        available_after_relation = max(pre_family_budget - relation_width, 0)
        axis_width = min(
            max(int(self.config.axis_noeffect_key_dim), 0),
            max(available_after_relation - 4, 0),
        )
        available_after_axis = max(available_after_relation - axis_width, 0)
        coord_width = min(
            max(int(self.config.coordinate_noeffect_key_dim), 0),
            max(available_after_axis - 2, 0),
        )
        failed_key_dim = max(
            2,
            min(int(self.config.failed_action_key_dim), max(available_after_axis - coord_width, 2)),
        )
        self.failed_action_memory_rank = FailedActionMemoryRank(
            action_feature_dim=ACTION_NUMERIC_DIM,
            belief_dim=d_model,
            key_dim=failed_key_dim,
            start=relation_width,
            dropout=float(self.config.dropout),
        )
        self.coordinate_noeffect_memory_rank = CoordinateNoEffectMemoryRank(
            belief_dim=d_model,
            start=relation_width + failed_key_dim,
            width=coord_width,
            dropout=float(self.config.dropout),
        )
        self.axis_noeffect_memory_rank = AxisNoEffectMemoryRank(
            belief_dim=d_model,
            start=relation_width + failed_key_dim + coord_width,
            width=axis_width,
            dropout=float(self.config.dropout),
        )
        family_start = relation_width + failed_key_dim + coord_width + axis_width
        family_count = min(family_count, max((d_model - family_start) // 3, 0))
        self.action_family_belief = ActionFamilyBelief(
            action_feature_dim=ACTION_NUMERIC_DIM,
            belief_dim=d_model,
            start=family_start,
            num_families=family_count,
            key_dim=int(self.config.action_family_key_dim),
            temperature=float(self.config.action_family_temperature),
        )
        self.reserved_belief_width = min(int(family_start + self.action_family_belief.width), d_model)
        self.candidate_state_compat = CandidateStateCompatibility(d_model, float(self.config.dropout))
        self.raw_candidate_rank = RawCandidateStateRank(
            d_model,
            float(self.config.dropout),
            self.event_relation_memory_rank,
            self.failed_action_memory_rank,
            self.coordinate_noeffect_memory_rank,
            self.axis_noeffect_memory_rank,
        )
        self.action_family_diagnostic_utility = ActionFamilyDiagnosticUtility(
            action_feature_dim=ACTION_NUMERIC_DIM,
            belief_dim=d_model,
            d_model=d_model,
            dropout=float(self.config.dropout),
        )
        self.diagnostic_mix_head = nn.Sequential(
            nn.LayerNorm((2 * d_model) + 8),
            nn.Linear((2 * d_model) + 8, d_model),
            nn.GELU(),
            nn.Linear(d_model, 1),
        )
        nn.init.zeros_(self.diagnostic_mix_head[-1].weight)
        nn.init.constant_(self.diagnostic_mix_head[-1].bias, -1.5)
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
        rank_components = self.raw_candidate_rank.components(
            action_numeric=action_numeric,
            state_numeric=state_numeric,
            state_type_ids=state_type_ids,
            state_mask=state_mask.bool(),
            action_mask=action_mask,
            session_belief=session_belief,
            level_belief=level_belief,
        )
        action_family_logits = self.action_family_belief.family_logits(action_numeric)
        action_family_probs = torch.softmax(action_family_logits, dim=-1) if action_family_logits.shape[-1] > 0 else action_family_logits
        action_family_posterior_features = self.action_family_belief.posterior_features(
            candidate_action_numeric=action_numeric,
            level_belief=level_belief,
        )
        diagnostic_utility_logits = self.action_family_diagnostic_utility(
            action_numeric=action_numeric,
            session_belief=session_belief,
            level_belief=level_belief,
            family_posterior_features=action_family_posterior_features,
        )
        diagnostic_mix_logit = self.diagnostic_mix_head(
            self._diagnostic_mix_features(
                session_belief=session_belief,
                level_belief=level_belief,
                rank_components=rank_components,
                action_mask=action_mask,
            )
        ).squeeze(-1)
        rank_logits = self.rank_score_head(adapted).squeeze(-1) + rank_components.total
        return ObjectEventModelOutput(
            outcome_logits=outcome_logits,
            delta_pred=delta_pred,
            value_logits=value_logits,
            action_repr=adapted,
            encoded_state=encoded_state,
            rank_logits=rank_logits,
            diagnostic_utility_logits=diagnostic_utility_logits,
            diagnostic_mix_logit=diagnostic_mix_logit,
            action_family_logits=action_family_logits,
            action_family_probs=action_family_probs,
            action_family_posterior_features=action_family_posterior_features,
            candidate_state_attn=candidate_state_attn,
            rank_components=rank_components,
        )

    def _diagnostic_mix_features(
        self,
        *,
        session_belief: torch.Tensor,
        level_belief: torch.Tensor,
        rank_components: RankComponentOutput,
        action_mask: torch.Tensor | None,
    ) -> torch.Tensor:
        valid = action_mask.bool() if action_mask is not None else torch.ones_like(rank_components.total, dtype=torch.bool)

        def std(values: torch.Tensor) -> torch.Tensor:
            active = values.masked_fill(~valid, 0.0)
            count = valid.to(dtype=values.dtype).sum(dim=1).clamp_min(1.0)
            mean = active.sum(dim=1) / count
            centered = torch.where(valid, values - mean[:, None], torch.zeros_like(values))
            return torch.sqrt(((centered * centered).sum(dim=1) / count).clamp_min(1.0e-6))

        def top_margin(values: torch.Tensor) -> torch.Tensor:
            masked = values.masked_fill(~valid, -1.0e9)
            topk = torch.topk(masked, k=min(2, masked.shape[1]), dim=1).values
            if topk.shape[1] < 2:
                return torch.zeros((values.shape[0],), dtype=values.dtype, device=values.device)
            return topk[:, 0] - topk[:, 1]

        coord_start = int(getattr(self.coordinate_noeffect_memory_rank, "start", 0))
        coord_stop = int(getattr(self.coordinate_noeffect_memory_rank, "stop", coord_start))
        axis_start = int(getattr(self.axis_noeffect_memory_rank, "start", 0))
        axis_stop = int(getattr(self.axis_noeffect_memory_rank, "stop", axis_start))
        coord_memory = level_belief[:, coord_start:coord_stop]
        axis_memory = level_belief[:, axis_start:axis_stop]
        coord_count = coord_memory[:, 0] if coord_memory.shape[1] > 0 else level_belief.new_zeros((level_belief.shape[0],))
        axis_count = axis_memory[:, 0] if axis_memory.shape[1] > 0 else level_belief.new_zeros((level_belief.shape[0],))
        coord_norm = torch.linalg.vector_norm(coord_memory, dim=1) if coord_memory.numel() else level_belief.new_zeros((level_belief.shape[0],))
        axis_norm = torch.linalg.vector_norm(axis_memory, dim=1) if axis_memory.numel() else level_belief.new_zeros((level_belief.shape[0],))
        stats = torch.stack(
            [
                torch.linalg.vector_norm(session_belief, dim=1),
                torch.linalg.vector_norm(level_belief, dim=1),
                torch.tanh(coord_count.clamp_min(0.0)),
                torch.tanh(axis_count.clamp_min(0.0)),
                torch.tanh(coord_norm),
                torch.tanh(axis_norm),
                std(rank_components.total),
                top_margin(rank_components.total),
            ],
            dim=-1,
        )
        return torch.cat([session_belief, level_belief, stats], dim=-1)

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
        return self.observed_event_belief_deltas(
            output,
            target_outcome=target_outcome,
            target_delta=target_delta,
            actual_action_index=actual_action_index,
            state_numeric=state_numeric,
            state_type_ids=state_type_ids,
            state_mask=state_mask,
            action_numeric=action_numeric,
        ).session_delta

    def observed_event_belief_deltas(
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
    ) -> ObservedEventBeliefDelta:
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
            failed_action_delta = self.failed_action_memory_rank.observed_delta(
                selected_action_numeric=chosen_action,
                no_effect_gate=target_outcome[:, OUT_NO_EFFECT_NONPROGRESS],
            )
            coordinate_noeffect_delta = self.coordinate_noeffect_memory_rank.observed_delta(
                selected_action_numeric=chosen_action,
                no_effect_gate=target_outcome[:, OUT_NO_EFFECT_NONPROGRESS],
            )
            axis_noeffect_delta = self.axis_noeffect_memory_rank.observed_delta(
                selected_action_numeric=chosen_action,
                no_effect_gate=target_outcome[:, OUT_NO_EFFECT_NONPROGRESS],
            )
            action_family_delta = self.action_family_belief.observed_delta(
                selected_action_numeric=chosen_action,
                target_outcome=target_outcome,
            )
        else:
            direct = raw_context.new_zeros((raw_context.shape[0], self.config.d_model))
            failed_action_delta = raw_context.new_zeros((raw_context.shape[0], self.config.d_model))
            coordinate_noeffect_delta = raw_context.new_zeros((raw_context.shape[0], self.config.d_model))
            axis_noeffect_delta = raw_context.new_zeros((raw_context.shape[0], self.config.d_model))
            action_family_delta = raw_context.new_zeros((raw_context.shape[0], self.config.d_model))
        learned = self.observed_event_belief_encoder(torch.cat([raw_context, event], dim=-1))
        learned = 0.05 * learned
        reserved_width = min(int(self.reserved_belief_width), learned.shape[1])
        if reserved_width > 0:
            learned = learned.clone()
            learned[:, :reserved_width] = 0.0
        session_delta = torch.tanh(direct + learned)
        no_effect_gate = target_outcome[:, OUT_NO_EFFECT_NONPROGRESS].reshape(-1, 1)
        level_delta = torch.tanh(
            direct
            + failed_action_delta
            + coordinate_noeffect_delta
            + axis_noeffect_delta
            + action_family_delta
            + learned * (0.5 + no_effect_gate)
        )
        return ObservedEventBeliefDelta(session_delta=session_delta, level_delta=level_delta)

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
        if candidate_value_targets is not None:
            rank_loss = _multi_positive_rank_ce(rank_logits, candidate_value_targets > 0.5, action_mask)
        else:
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
