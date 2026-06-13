from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F


def masked_accuracy(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> float:
    active = int(mask.sum().item())
    if active == 0:
        return 0.0
    pred = logits.argmax(dim=-1)
    return float(((pred == target) & mask).sum().item() / active)


def plain_accuracy(logits: torch.Tensor, target: torch.Tensor) -> float:
    pred = logits.argmax(dim=-1)
    return float((pred == target).float().mean().item())


def masked_ce(logits: torch.Tensor, target: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    if int(mask.sum().item()) == 0:
        return logits.sum() * 0.0
    loss = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), target.reshape(-1), reduction="none")
    return (loss * mask.reshape(-1).float()).sum() / mask.sum().float()


def latent_noncollapse_stats(latents: torch.Tensor, language_tokens: torch.Tensor) -> Dict[str, float]:
    flat = latents.reshape(-1, latents.shape[-1]).detach().float()
    tokens = language_tokens.reshape(-1).detach()
    if flat.shape[0] < 2:
        return {
            "latent_active_fraction": 0.0,
            "latent_effective_rank": 0.0,
            "latent_max_quantized_fraction": 1.0,
            "language_repetition_ratio": 1.0,
        }

    std = flat.std(dim=0)
    active_fraction = float((std > 0.01).float().mean().item())

    centered = flat - flat.mean(dim=0, keepdim=True)
    cov = centered.T @ centered / max(1, flat.shape[0] - 1)
    eig = torch.linalg.eigvalsh(cov).clamp_min(1e-12)
    probs = eig / eig.sum()
    effective_rank = float(torch.exp(-(probs * probs.log()).sum()).item())

    quantized = torch.round(flat * 100.0).to(torch.int16)
    _, counts = torch.unique(quantized, dim=0, return_counts=True)
    max_quantized_fraction = float(counts.max().item() / flat.shape[0])

    if tokens.numel() < 2:
        repetition = 0.0
    else:
        repetition = float((tokens[1:] == tokens[:-1]).float().mean().item())

    return {
        "latent_active_fraction": active_fraction,
        "latent_effective_rank": effective_rank,
        "latent_max_quantized_fraction": max_quantized_fraction,
        "language_repetition_ratio": repetition,
    }


def categorical_head_stats(
    logits: torch.Tensor,
    *,
    prefix: str,
    mask: torch.Tensor | None = None,
) -> Dict[str, float]:
    flat_logits = logits.detach().float().reshape(-1, logits.shape[-1])
    if mask is not None:
        flat_mask = mask.detach().bool().reshape(-1)
        flat_logits = flat_logits[flat_mask]
    if flat_logits.shape[0] == 0:
        return {
            f"{prefix}_normalized_entropy": 0.0,
            f"{prefix}_mean_confidence": 1.0,
            f"{prefix}_max_prediction_fraction": 1.0,
            f"{prefix}_unique_prediction_fraction": 0.0,
            f"{prefix}_active_count": 0.0,
        }

    probs = torch.softmax(flat_logits, dim=-1)
    entropy = -(probs * probs.clamp_min(1.0e-12).log()).sum(dim=-1)
    normalized_entropy = entropy / max(math.log(float(flat_logits.shape[-1])), 1.0e-12)
    confidence = probs.max(dim=-1).values
    predictions = probs.argmax(dim=-1)
    _, counts = torch.unique(predictions, return_counts=True)
    max_prediction_fraction = float(counts.max().item() / max(int(predictions.numel()), 1))
    unique_prediction_fraction = float(counts.numel() / max(int(flat_logits.shape[-1]), 1))
    return {
        f"{prefix}_normalized_entropy": float(normalized_entropy.mean().item()),
        f"{prefix}_mean_confidence": float(confidence.mean().item()),
        f"{prefix}_max_prediction_fraction": max_prediction_fraction,
        f"{prefix}_unique_prediction_fraction": unique_prediction_fraction,
        f"{prefix}_active_count": float(flat_logits.shape[0]),
    }


def memorization_overfit_stats(
    *,
    train_accuracy: float,
    heldout_accuracy: float,
    perturbation_accuracy: float,
    adversarial_accuracy: float,
) -> Dict[str, float]:
    eps = 1.0e-6
    train = float(train_accuracy)
    heldout = float(heldout_accuracy)
    perturbation = float(perturbation_accuracy)
    adversarial = float(adversarial_accuracy)
    return {
        "memorization_train_seed_accuracy": train,
        "memorization_heldout_seed_accuracy": heldout,
        "memorization_perturbation_accuracy": perturbation,
        "memorization_adversarial_seed_accuracy": adversarial,
        "memorization_train_heldout_gap": max(0.0, train - heldout),
        "memorization_heldout_train_ratio": heldout / max(train, eps),
        "memorization_perturbation_retention": perturbation / max(heldout, eps),
        "memorization_adversarial_retention": adversarial / max(heldout, eps),
    }


def memorization_overfit_gates(
    stats: Dict[str, float],
    *,
    max_train_heldout_gap: float = 0.20,
    min_heldout_train_ratio: float = 0.75,
    min_perturbation_retention: float = 0.90,
    min_adversarial_retention: float = 0.75,
) -> Dict[str, bool]:
    gates = {
        "memorization_train_heldout_gap": stats.get("memorization_train_heldout_gap", 1.0)
        <= max_train_heldout_gap,
        "memorization_heldout_train_ratio": stats.get("memorization_heldout_train_ratio", 0.0)
        >= min_heldout_train_ratio,
        "memorization_perturbation_retention": stats.get("memorization_perturbation_retention", 0.0)
        >= min_perturbation_retention,
        "memorization_adversarial_retention": stats.get("memorization_adversarial_retention", 0.0)
        >= min_adversarial_retention,
    }
    gates["memorization_overfit"] = all(gates.values())
    return gates


def pass_fail(metrics: Dict[str, float]) -> Dict[str, bool]:
    rank_floor = min(8.0, 0.10 * metrics.get("latent_dim", 64.0))
    return {
        "goal_action_success": metrics.get("goal_action_success", 0.0) >= 0.80,
        "delayed_memory_accuracy": metrics.get("delayed_memory_accuracy", 0.0) >= 0.85,
        "object_permanence_accuracy": metrics.get("object_permanence_accuracy", 0.0) >= 0.85,
        "provenance_accuracy": metrics.get("provenance_accuracy", 0.0) >= 0.85,
        "grounded_language_accuracy": metrics.get("grounded_language_accuracy", 0.0) >= 0.85,
        "self_world_continuity_accuracy": metrics.get("self_world_continuity_accuracy", 0.0) >= 0.90,
        "latent_active_fraction": metrics.get("latent_active_fraction", 0.0) >= 0.25,
        "latent_effective_rank": metrics.get("latent_effective_rank", 0.0) >= rank_floor,
        "latent_max_quantized_fraction": metrics.get("latent_max_quantized_fraction", 1.0) <= 0.05,
        "language_repetition_ratio": metrics.get("language_repetition_ratio", 1.0) < 0.40,
        "action_head_distribution": (
            metrics.get("action_head_active_count", 0.0) > 0.0
            and metrics.get("action_head_max_prediction_fraction", 1.0) <= 0.98
            and metrics.get("action_head_unique_prediction_fraction", 0.0) >= 0.10
        ),
        "language_head_distribution": (
            metrics.get("language_head_active_count", 0.0) > 0.0
            and metrics.get("language_head_max_prediction_fraction", 1.0) <= 0.98
            and metrics.get("language_head_unique_prediction_fraction", 0.0) >= 0.10
        ),
        "private_head_distribution": (
            metrics.get("private_head_active_count", 0.0) > 0.0
            and metrics.get("private_head_max_prediction_fraction", 1.0) <= 0.98
            and metrics.get("private_head_unique_prediction_fraction", 0.0) >= 0.10
        ),
        "memorization_overfit": metrics.get("memorization_overfit_pass", 1.0) >= 1.0,
        "unbroken_ticks": metrics.get("unbroken_ticks", 0.0) >= metrics.get("requested_ticks", 0.0),
    }
