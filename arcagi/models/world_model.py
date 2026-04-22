from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
from torch import nn
from torch.nn import functional as F

from arcagi.core.types import ActionName, StructuredState
from arcagi.models.action_encoder import ActionEncoder

EFFECT_KINDS: tuple[str, ...] = ("reward_gain", "setback", "state_change", "latent_shift", "no_effect")


@dataclass(frozen=True)
class WorldModelPrediction:
    hidden: torch.Tensor
    next_latent_mean: torch.Tensor
    next_latent_std: torch.Tensor
    reward: torch.Tensor
    return_value: torch.Tensor
    usefulness: torch.Tensor
    policy: torch.Tensor
    delta: torch.Tensor
    causal_value: torch.Tensor
    diagnostic_value: torch.Tensor
    effect_logits: torch.Tensor
    uncertainty: torch.Tensor


class _EnsembleHead(nn.Module):
    def __init__(self, hidden_dim: int, latent_dim: int, summary_dim: int) -> None:
        super().__init__()
        self.next_latent = nn.Linear(hidden_dim, latent_dim)
        self.reward = nn.Linear(hidden_dim, 1)
        self.return_value = nn.Linear(hidden_dim, 1)
        self.usefulness = nn.Linear(hidden_dim, 1)
        self.policy = nn.Linear(hidden_dim, 1)
        self.delta = nn.Linear(hidden_dim, summary_dim)
        self.causal_value = nn.Linear(hidden_dim, 1)
        self.diagnostic_value = nn.Linear(hidden_dim, 1)
        self.effect_logits = nn.Linear(hidden_dim, len(EFFECT_KINDS))

    def forward(self, hidden: torch.Tensor) -> dict[str, torch.Tensor]:
        return {
            "next_latent": self.next_latent(hidden),
            "reward": self.reward(hidden).squeeze(-1),
            "return_value": self.return_value(hidden).squeeze(-1),
            "usefulness": self.usefulness(hidden).squeeze(-1),
            "policy": self.policy(hidden).squeeze(-1),
            "delta": self.delta(hidden),
            "causal_value": self.causal_value(hidden).squeeze(-1),
            "diagnostic_value": self.diagnostic_value(hidden).squeeze(-1),
            "effect_logits": self.effect_logits(hidden),
        }


class RecurrentWorldModel(nn.Module):
    def __init__(
        self,
        latent_dim: int = 128,
        hidden_dim: int = 128,
        action_dim: int = 32,
        summary_dim: int = 25,
        ensemble_size: int = 3,
    ) -> None:
        super().__init__()
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.summary_dim = summary_dim
        self.action_encoder = ActionEncoder(embedding_dim=action_dim)
        self.gru = nn.GRUCell(latent_dim + action_dim, hidden_dim)
        self.heads = nn.ModuleList(
            [_EnsembleHead(hidden_dim, latent_dim, summary_dim) for _ in range(ensemble_size)]
        )

    def encode_actions(
        self,
        actions: Sequence[ActionName],
        state: StructuredState | None = None,
        device: torch.device | None = None,
    ) -> torch.Tensor:
        return self.action_encoder.encode(actions, state=state, device=device)

    def initial_hidden(self, batch_size: int, device: torch.device | None = None) -> torch.Tensor:
        return torch.zeros(batch_size, self.hidden_dim, device=device)

    def step(
        self,
        latent: torch.Tensor,
        actions: Sequence[ActionName] | None = None,
        state: StructuredState | None = None,
        action_embeddings: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> WorldModelPrediction:
        (
            next_hidden,
            next_latents,
            rewards,
            return_values,
            usefulness,
            policies,
            deltas,
            causal_values,
            diagnostic_values,
            effect_logits,
        ) = self._predict_ensemble(
            latent,
            actions=actions,
            state=state,
            action_embeddings=action_embeddings,
            hidden=hidden,
        )
        next_latent_mean = next_latents.mean(dim=0)
        next_latent_std = next_latents.std(dim=0)
        return WorldModelPrediction(
            hidden=next_hidden,
            next_latent_mean=next_latent_mean,
            next_latent_std=next_latent_std,
            reward=rewards.mean(dim=0),
            return_value=return_values.mean(dim=0),
            usefulness=usefulness.mean(dim=0),
            policy=policies.mean(dim=0),
            delta=deltas.mean(dim=0),
            causal_value=causal_values.mean(dim=0),
            diagnostic_value=diagnostic_values.mean(dim=0),
            effect_logits=effect_logits.mean(dim=0),
            uncertainty=next_latent_std.mean(dim=-1),
        )

    def _predict_ensemble(
        self,
        latent: torch.Tensor,
        actions: Sequence[ActionName] | None = None,
        state: StructuredState | None = None,
        action_embeddings: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        if hidden is None:
            hidden = self.initial_hidden(latent.shape[0], device=latent.device)
        if action_embeddings is None:
            if actions is None:
                raise ValueError("actions or action_embeddings must be provided")
            action_embeddings = self.encode_actions(actions, state=state, device=latent.device)
        action_embed = action_embeddings
        next_hidden = self.gru(torch.cat([latent, action_embed], dim=-1), hidden)
        predictions = [head(next_hidden) for head in self.heads]
        next_latents = torch.stack([prediction["next_latent"] for prediction in predictions], dim=0)
        rewards = torch.stack([prediction["reward"] for prediction in predictions], dim=0)
        return_values = torch.stack([prediction["return_value"] for prediction in predictions], dim=0)
        usefulness = torch.stack([prediction["usefulness"] for prediction in predictions], dim=0)
        policies = torch.stack([prediction["policy"] for prediction in predictions], dim=0)
        deltas = torch.stack([prediction["delta"] for prediction in predictions], dim=0)
        causal_values = torch.stack([prediction["causal_value"] for prediction in predictions], dim=0)
        diagnostic_values = torch.stack([prediction["diagnostic_value"] for prediction in predictions], dim=0)
        effect_logits = torch.stack([prediction["effect_logits"] for prediction in predictions], dim=0)
        return (
            next_hidden,
            next_latents,
            rewards,
            return_values,
            usefulness,
            policies,
            deltas,
            causal_values,
            diagnostic_values,
            effect_logits,
        )

    def loss(
        self,
        latent: torch.Tensor,
        actions: Sequence[ActionName],
        state: StructuredState | None,
        hidden: torch.Tensor | None,
        next_latent_target: torch.Tensor,
        reward_target: torch.Tensor,
        return_target: torch.Tensor,
        delta_target: torch.Tensor,
        usefulness_target: torch.Tensor,
        effect_target: torch.Tensor | None = None,
        causal_target: torch.Tensor | None = None,
        diagnostic_target: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        (
            next_hidden,
            next_latents,
            rewards,
            return_values,
            usefulness,
            _policies,
            deltas,
            causal_values,
            diagnostic_values,
            effect_logits,
        ) = self._predict_ensemble(
            latent,
            actions=actions,
            state=state,
            hidden=hidden,
        )
        del next_hidden
        expanded_next_latent_target = next_latent_target.unsqueeze(0).expand_as(next_latents)
        expanded_reward_target = reward_target.unsqueeze(0).expand_as(rewards)
        expanded_return_target = return_target.unsqueeze(0).expand_as(return_values)
        expanded_delta_target = delta_target.unsqueeze(0).expand_as(deltas)
        expanded_usefulness_target = usefulness_target.unsqueeze(0).expand_as(usefulness)
        latent_loss = F.mse_loss(next_latents, expanded_next_latent_target)
        reward_loss = F.mse_loss(rewards, expanded_reward_target)
        return_loss = F.mse_loss(return_values, expanded_return_target)
        delta_loss = F.mse_loss(deltas, expanded_delta_target)
        usefulness_loss = F.mse_loss(usefulness, expanded_usefulness_target)
        if causal_target is None:
            causal_loss = torch.zeros((), dtype=latent.dtype, device=latent.device)
        else:
            expanded_causal_target = causal_target.unsqueeze(0).expand_as(causal_values)
            causal_loss = F.mse_loss(causal_values, expanded_causal_target)
        if diagnostic_target is None:
            diagnostic_loss = torch.zeros((), dtype=latent.dtype, device=latent.device)
        else:
            expanded_diagnostic_target = diagnostic_target.unsqueeze(0).expand_as(diagnostic_values)
            diagnostic_loss = F.mse_loss(diagnostic_values, expanded_diagnostic_target)
        if effect_target is None:
            effect_loss = torch.zeros((), dtype=latent.dtype, device=latent.device)
        else:
            repeated_targets = effect_target.unsqueeze(0).expand(effect_logits.shape[0], -1).reshape(-1)
            effect_loss = F.cross_entropy(
                effect_logits.reshape(-1, effect_logits.shape[-1]),
                repeated_targets,
            )
        next_latent_std = next_latents.std(dim=0)
        total = (
            latent_loss
            + reward_loss
            + (0.35 * return_loss)
            + (0.5 * delta_loss)
            + (0.5 * usefulness_loss)
            + (0.35 * causal_loss)
            + (0.3 * diagnostic_loss)
            + (0.35 * effect_loss)
        )
        metrics = {
            "loss_total": float(total.detach().cpu()),
            "loss_latent": float(latent_loss.detach().cpu()),
            "loss_reward": float(reward_loss.detach().cpu()),
            "loss_return": float(return_loss.detach().cpu()),
            "loss_delta": float(delta_loss.detach().cpu()),
            "loss_usefulness": float(usefulness_loss.detach().cpu()),
            "loss_causal": float(causal_loss.detach().cpu()),
            "loss_diagnostic": float(diagnostic_loss.detach().cpu()),
            "loss_effect": float(effect_loss.detach().cpu()),
            "uncertainty": float(next_latent_std.mean(dim=-1).mean().detach().cpu()),
        }
        return total, metrics
