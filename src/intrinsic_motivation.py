from __future__ import annotations

import torch


INTRINSIC_DIM = 12


def learning_progress(previous_error: torch.Tensor, current_error: torch.Tensor) -> torch.Tensor:
    """Positive normalized improvement signal for prediction errors."""
    denom = previous_error.abs().clamp_min(1.0e-4)
    return torch.clamp((previous_error - current_error) / denom, 0.0, 1.0)


def uncertainty_drive(confidence: torch.Tensor, contradiction: torch.Tensor) -> torch.Tensor:
    return torch.clamp((1.0 - confidence) + 0.35 * contradiction.float(), 0.0, 1.0)


def novelty_drive(novelty: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
    return torch.clamp(novelty.float() * (1.0 - noise.float()), 0.0, 1.0)


def preservation_drive(hazard_level: torch.Tensor, resource_level: torch.Tensor, body_energy: torch.Tensor) -> torch.Tensor:
    hazard = torch.clamp(hazard_level.float() / 3.0, 0.0, 1.0)
    resource_need = torch.clamp(1.0 - resource_level.float() / 3.0, 0.0, 1.0)
    fatigue = torch.clamp(1.0 - body_energy.float(), 0.0, 1.0)
    return torch.clamp(0.45 * hazard + 0.30 * resource_need + 0.25 * fatigue, 0.0, 1.0)


def project_drive(project_step: torch.Tensor, restart: torch.Tensor) -> torch.Tensor:
    step = torch.clamp(project_step.float() / 3.0, 0.0, 1.0)
    return torch.clamp(0.65 * step + 0.35 * restart.float(), 0.0, 1.0)


def social_gap_drive(partner_uncertain: torch.Tensor, conflict: torch.Tensor) -> torch.Tensor:
    return torch.clamp(0.55 * partner_uncertain.float() + 0.45 * conflict.float(), 0.0, 1.0)


def build_intrinsic_vector(
    *,
    confidence: torch.Tensor,
    contradiction: torch.Tensor,
    novelty: torch.Tensor,
    noise: torch.Tensor,
    hazard_level: torch.Tensor,
    resource_level: torch.Tensor,
    body_energy: torch.Tensor,
    project_step: torch.Tensor,
    restart: torch.Tensor,
    partner_uncertain: torch.Tensor,
    state_hint: torch.Tensor,
) -> torch.Tensor:
    uncertainty = uncertainty_drive(confidence, contradiction)
    novelty_clean = novelty_drive(novelty, noise)
    preservation = preservation_drive(hazard_level, resource_level, body_energy)
    project = project_drive(project_step, restart)
    social = social_gap_drive(partner_uncertain, contradiction)
    progress = learning_progress(0.55 + uncertainty, 0.20 + 0.25 * noise.float())
    state_oh = torch.nn.functional.one_hot(state_hint.long(), 4).float()
    return torch.cat(
        [
            progress[:, None],
            uncertainty[:, None],
            novelty_clean[:, None],
            preservation[:, None],
            project[:, None],
            social[:, None],
            body_energy.float().clamp(0.0, 1.0)[:, None],
            state_oh,
            noise.float()[:, None],
        ],
        dim=-1,
    )
