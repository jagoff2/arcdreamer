from __future__ import annotations

from typing import TypeAlias

import torch


DeviceLike: TypeAlias = str | torch.device | None
AUTO_DEVICE = "auto"


def resolve_device(device: DeviceLike = AUTO_DEVICE) -> torch.device:
    """Return the target compute device, preferring CUDA for auto placement."""
    if isinstance(device, torch.device):
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError(f"requested CUDA device {device} but CUDA is not available")
        return device
    if device is None:
        device = AUTO_DEVICE
    name = str(device).strip().lower()
    if name in {"", AUTO_DEVICE, "gpu", "cuda_if_available"}:
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    resolved = torch.device(name)
    if resolved.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError(f"requested CUDA device {resolved} but CUDA is not available")
    return resolved


def resolve_device_name(device: DeviceLike = AUTO_DEVICE) -> str:
    return str(resolve_device(device))


def make_generator(seed: int | None = None, device: DeviceLike = AUTO_DEVICE) -> torch.Generator:
    generator = torch.Generator(device=resolve_device(device))
    if seed is not None:
        generator.manual_seed(int(seed))
    return generator
