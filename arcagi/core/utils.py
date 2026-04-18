from __future__ import annotations

import math
import random

import numpy as np
import torch


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def softmax(values: list[float], temperature: float = 1.0) -> list[float]:
    if not values:
        return []
    temperature = max(temperature, 1e-6)
    max_value = max(values)
    exps = [math.exp((value - max_value) / temperature) for value in values]
    total = sum(exps)
    if total == 0.0:
        return [1.0 / len(values)] * len(values)
    return [value / total for value in exps]


def cosine_similarity(left: np.ndarray, right: np.ndarray) -> float:
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom == 0.0:
        return 0.0
    return float(np.dot(left, right) / denom)

