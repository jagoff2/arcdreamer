from __future__ import annotations

from dataclasses import asdict

import torch

from src.env import SENSOR_DIM
from src.model import RecurrentLatentModel, load_checkpoint


def _legacy_payload(state: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    return {
        key: value
        for key, value in state.items()
        if not (
            key.startswith("action_impulse_embedding.")
            or key.startswith("delta_impulse_encoder.")
            or key.startswith("liquid_time_constant.")
            or key.startswith("liquid_drift.")
        )
    }


def test_model_exposes_trainable_state_dependent_liquid_dynamics() -> None:
    model = RecurrentLatentModel(device="cpu")

    module_names = {name for name, _ in model.named_modules()}
    assert "liquid_time_constant" in module_names
    assert "liquid_drift" in module_names

    liquid_params = [
        param
        for name, param in model.named_parameters()
        if name.startswith("liquid_time_constant.") or name.startswith("liquid_drift.")
    ]
    assert liquid_params
    assert all(param.requires_grad for param in liquid_params)

    hidden = model.config.hidden_dim
    mixed = torch.zeros(1, hidden, dtype=torch.float32)
    obs = {"dt": torch.ones(1, dtype=torch.float32)}
    z_zero = torch.zeros(1, hidden, dtype=torch.float32)
    z_one = torch.ones(1, hidden, dtype=torch.float32)
    z_candidate = torch.zeros(1, hidden, dtype=torch.float32)

    with torch.no_grad():
        model.liquid_time_constant.weight.zero_()
        model.liquid_time_constant.bias.zero_()
        model.liquid_time_constant.weight[:, hidden:] = torch.eye(
            hidden,
            dtype=model.liquid_time_constant.weight.dtype,
            device=model.liquid_time_constant.weight.device,
        )
        model.liquid_drift[0].weight.zero_()
        model.liquid_drift[0].bias.zero_()
        model.liquid_drift[0].weight[:, hidden:] = torch.eye(
            hidden,
            dtype=model.liquid_drift[0].weight.dtype,
            device=model.liquid_drift[0].weight.device,
        )
        model.liquid_drift[2].weight.zero_()
        model.liquid_drift[2].bias.zero_()
        model.liquid_drift[2].weight[: hidden, :hidden] = torch.eye(
            hidden,
            dtype=model.liquid_drift[2].weight.dtype,
            device=model.liquid_drift[2].weight.device,
        )

    z_next_from_zero = model._liquid_integrate(
        mixed=mixed, z_prev=z_zero, z_candidate=z_candidate, observation=obs
    )
    z_next_from_one = model._liquid_integrate(
        mixed=mixed, z_prev=z_one, z_candidate=z_candidate, observation=obs
    )

    assert not torch.allclose(z_next_from_zero, z_next_from_one)


def test_step_respects_optional_dt_and_changes_z_next_with_residual_enabled() -> None:
    model = RecurrentLatentModel(device="cpu")
    z0 = model.initial_state(1)
    observation = {
        "sensory": torch.zeros(1, SENSOR_DIM, dtype=torch.float32),
        "lang_in": torch.zeros(1, dtype=torch.long),
    }

    with torch.no_grad():
        model.liquid_time_constant.weight.zero_()
        model.liquid_time_constant.bias.zero_()
        model.liquid_drift[0].weight.zero_()
        model.liquid_drift[0].bias.zero_()
        model.liquid_drift[2].weight.zero_()
        model.liquid_drift[2].bias.fill_(0.5)

    _, z_default = model.step(observation, z0)
    _, z_t0 = model.step({**observation, "dt": torch.zeros(1, dtype=torch.float32)}, z0)
    _, z_t2 = model.step({**observation, "dt": torch.full((1,), 2.0, dtype=torch.float32)}, z0)

    assert not torch.allclose(z_default, z_t0)
    assert not torch.allclose(z_t0, z_t2)
    assert not torch.allclose(z_default, z_t2)


def test_load_checkpoint_accepts_legacy_state_without_liquid_or_impulse_keys(tmp_path) -> None:
    model = RecurrentLatentModel(device="cpu")
    checkpoint = tmp_path / "legacy_state.pt"
    torch.save(
        {
            "model_config": asdict(model.config),
            "model_state": _legacy_payload(model.state_dict()),
            "train_config": {},
            "metrics": {},
        },
        checkpoint,
    )

    loaded = load_checkpoint(checkpoint, device="cpu")

    assert isinstance(loaded, RecurrentLatentModel)
    assert "liquid_time_constant.weight" in loaded.state_dict()
    assert "liquid_drift.0.weight" in loaded.state_dict()
    assert "action_impulse_embedding.weight" in loaded.state_dict()
    assert "delta_impulse_encoder.weight" in loaded.state_dict()
