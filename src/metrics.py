from __future__ import annotations

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
        "unbroken_ticks": metrics.get("unbroken_ticks", 0.0) >= metrics.get("requested_ticks", 0.0),
    }
