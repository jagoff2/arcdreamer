from __future__ import annotations

import torch

from arcagi.models.world_model import RecurrentWorldModel


class _MeanMatchingButWrongEnsembleWorldModel(RecurrentWorldModel):
    def __init__(self) -> None:
        super().__init__(latent_dim=2, hidden_dim=2, action_dim=2, summary_dim=3, ensemble_size=3)

    def _predict_ensemble(
        self,
        latent: torch.Tensor,
        actions=None,
        state=None,
        action_embeddings: torch.Tensor | None = None,
        hidden: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        batch_size = latent.shape[0]
        device = latent.device
        next_hidden = torch.zeros(batch_size, self.hidden_dim, device=device)
        next_latents = torch.tensor(
            [
                [[1.0, 0.0]],
                [[-1.0, 0.0]],
                [[0.0, 0.0]],
            ],
            dtype=torch.float32,
            device=device,
        )
        rewards = torch.tensor(
            [
                [1.0],
                [-1.0],
                [0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        return_values = torch.tensor(
            [
                [1.5],
                [-1.5],
                [0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        usefulness = torch.tensor(
            [
                [0.8],
                [-0.8],
                [0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        policies = torch.zeros(3, batch_size, dtype=torch.float32, device=device)
        deltas = torch.tensor(
            [
                [[1.0, 0.0, 0.0]],
                [[-1.0, 0.0, 0.0]],
                [[0.0, 0.0, 0.0]],
            ],
            dtype=torch.float32,
            device=device,
        )
        causal_values = torch.tensor(
            [
                [0.7],
                [-0.7],
                [0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        diagnostic_values = torch.tensor(
            [
                [0.6],
                [0.1],
                [0.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        effect_logits = torch.tensor(
            [
                [[2.0, -1.0, 0.0, 0.0, -0.5]],
                [[-1.0, 2.0, 0.0, 0.0, -0.5]],
                [[0.0, 0.0, 0.0, 0.0, 2.0]],
            ],
            dtype=torch.float32,
            device=device,
        )
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


def test_world_model_loss_penalizes_wrong_heads_even_if_ensemble_mean_matches() -> None:
    world_model = _MeanMatchingButWrongEnsembleWorldModel()
    latent = torch.zeros((1, 2), dtype=torch.float32)
    next_latent_target = torch.zeros((1, 2), dtype=torch.float32)
    reward_target = torch.zeros((1,), dtype=torch.float32)
    return_target = torch.zeros((1,), dtype=torch.float32)
    delta_target = torch.zeros((1, 3), dtype=torch.float32)
    usefulness_target = torch.zeros((1,), dtype=torch.float32)
    effect_target = torch.tensor([4], dtype=torch.long)
    causal_target = torch.zeros((1,), dtype=torch.float32)
    diagnostic_target = torch.zeros((1,), dtype=torch.float32)

    loss, metrics = world_model.loss(
        latent=latent,
        actions=["wait"],
        state=None,
        hidden=None,
        next_latent_target=next_latent_target,
        reward_target=reward_target,
        return_target=return_target,
        delta_target=delta_target,
        usefulness_target=usefulness_target,
        effect_target=effect_target,
        causal_target=causal_target,
        diagnostic_target=diagnostic_target,
    )

    assert float(loss.item()) > 0.5
    assert metrics["loss_latent"] > 0.0
    assert metrics["loss_reward"] > 0.0
    assert metrics["loss_return"] > 0.0
    assert metrics["loss_delta"] > 0.0
    assert metrics["loss_usefulness"] > 0.0
    assert metrics["loss_causal"] > 0.0
    assert metrics["loss_diagnostic"] > 0.0
    assert metrics["loss_effect"] > 0.0


def test_world_model_reports_disagreement_without_penalizing_it_in_total_loss() -> None:
    world_model = _MeanMatchingButWrongEnsembleWorldModel()
    latent = torch.zeros((1, 2), dtype=torch.float32)
    next_latent_target = torch.zeros((1, 2), dtype=torch.float32)
    reward_target = torch.zeros((1,), dtype=torch.float32)
    return_target = torch.zeros((1,), dtype=torch.float32)
    delta_target = torch.zeros((1, 3), dtype=torch.float32)
    usefulness_target = torch.zeros((1,), dtype=torch.float32)
    effect_target = torch.tensor([4], dtype=torch.long)
    causal_target = torch.zeros((1,), dtype=torch.float32)
    diagnostic_target = torch.zeros((1,), dtype=torch.float32)

    loss, metrics = world_model.loss(
        latent=latent,
        actions=["wait"],
        state=None,
        hidden=None,
        next_latent_target=next_latent_target,
        reward_target=reward_target,
        return_target=return_target,
        delta_target=delta_target,
        usefulness_target=usefulness_target,
        effect_target=effect_target,
        causal_target=causal_target,
        diagnostic_target=diagnostic_target,
    )

    expected_total = (
        metrics["loss_latent"]
        + metrics["loss_reward"]
        + (0.35 * metrics["loss_return"])
        + (0.5 * metrics["loss_delta"])
        + (0.5 * metrics["loss_usefulness"])
        + (0.35 * metrics["loss_causal"])
        + (0.3 * metrics["loss_diagnostic"])
        + (0.35 * metrics["loss_effect"])
    )
    assert abs(metrics["loss_total"] - expected_total) < 1e-6
    assert metrics["uncertainty"] > 0.0
    assert float(loss.item()) == metrics["loss_total"]
