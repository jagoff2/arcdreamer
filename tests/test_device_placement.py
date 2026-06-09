from __future__ import annotations

import torch

from src.device import resolve_device
from src.env import generate_batch
from src.model import RecurrentLatentModel
from src.persistent_memory import PersistentMemoryState


def test_auto_device_prefers_cuda_when_available() -> None:
    expected = "cuda" if torch.cuda.is_available() else "cpu"
    assert resolve_device().type == expected


def test_default_tensors_follow_auto_device() -> None:
    device = resolve_device()
    batch = generate_batch(2, seq_len=4, base_seed=123)
    model = RecurrentLatentModel()
    memory = PersistentMemoryState.fresh(hidden_dim=model.config.hidden_dim, batch_size=2)

    assert batch["sensory"].device.type == device.type
    assert next(model.parameters()).device.type == device.type
    assert memory.latent.device.type == device.type
