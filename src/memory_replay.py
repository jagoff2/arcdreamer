from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F

from .human_memory import HumanAnalogueMemory, _unit


@dataclass
class ReplayReport:
    samples_replayed: int
    adapter_rank: int
    ridge: float


def replay_consolidate(memory: HumanAnalogueMemory, ridge: float = 1.0e-3) -> ReplayReport:
    if memory.trace_count == 0:
        raise ValueError("cannot replay an empty memory")
    cues = memory._bank(memory._cues)
    contents = memory._bank(memory._contents)
    x = torch.cat([cues, torch.ones(cues.shape[0], 1, device=cues.device)], dim=-1)
    eye = torch.eye(x.shape[1], device=x.device)
    weight = torch.linalg.solve(x.T @ x + ridge * eye, x.T @ contents)
    memory.semantic_weight = weight
    rank = int(torch.linalg.matrix_rank(x).item())
    return ReplayReport(samples_replayed=memory.trace_count, adapter_rank=rank, ridge=float(ridge))


def semantic_predict(memory: HumanAnalogueMemory, cue: torch.Tensor) -> torch.Tensor:
    if memory.semantic_weight is None:
        return torch.zeros(memory.config.content_dim, device=memory.device)
    cue = cue.to(memory.device).float()
    x = torch.cat([cue, torch.ones(1, device=memory.device)], dim=0)
    return _unit(x @ memory.semantic_weight)


def semantic_accuracy(memory: HumanAnalogueMemory, cues: list[torch.Tensor], targets: list[torch.Tensor], bank: torch.Tensor) -> float:
    if not cues:
        return 0.0
    bank = _unit(bank.to(memory.device).float())
    correct = 0
    for cue, target in zip(cues, targets):
        pred = semantic_predict(memory, cue)
        pred_idx = int((pred @ bank.T).argmax().item())
        target_idx = int((_unit(target.to(memory.device).float()) @ bank.T).argmax().item())
        correct += int(pred_idx == target_idx)
    return float(correct / len(cues))
