from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List

import torch


UNKNOWN_CONCEPT_VALUE = -1


def concept_value(concept_id: int) -> int:
    return int(1000 + concept_id * 17)


@dataclass
class PersistentConceptMemory:
    concept_ids: torch.Tensor
    concept_values: torch.Tensor

    @classmethod
    def fresh(cls, device: torch.device | str = "cpu") -> "PersistentConceptMemory":
        return cls(
            concept_ids=torch.empty(0, dtype=torch.long, device=device),
            concept_values=torch.empty(0, dtype=torch.long, device=device),
        )

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> "PersistentConceptMemory":
        path = Path(path)
        if not path.exists():
            return cls.fresh(device=device)
        payload = torch.load(path, map_location=device)
        if payload.get("format") != "persistent_concept_memory_v1":
            raise ValueError(f"{path} is not a persistent concept memory file")
        return cls(
            concept_ids=payload["concept_ids"].to(device).long(),
            concept_values=payload["concept_values"].to(device).long(),
        )

    def clone(self) -> "PersistentConceptMemory":
        return PersistentConceptMemory(self.concept_ids.clone(), self.concept_values.clone())

    def learn(self, concept_id: int, value: int | None = None) -> None:
        cid = torch.tensor([int(concept_id)], dtype=torch.long, device=self.concept_ids.device)
        val = torch.tensor([concept_value(concept_id) if value is None else int(value)], dtype=torch.long, device=self.concept_values.device)
        if self.concept_ids.numel() == 0:
            self.concept_ids = cid
            self.concept_values = val
            return
        exists = self.concept_ids == int(concept_id)
        if bool(exists.any().item()):
            self.concept_values[exists] = val[0]
        else:
            self.concept_ids = torch.cat([self.concept_ids, cid], dim=0)
            self.concept_values = torch.cat([self.concept_values, val], dim=0)

    def predict(self, concept_ids: torch.Tensor) -> torch.Tensor:
        ids = concept_ids.to(self.concept_ids.device).long()
        pred = torch.full_like(ids, UNKNOWN_CONCEPT_VALUE)
        for concept_id, value in zip(self.concept_ids.tolist(), self.concept_values.tolist()):
            pred = torch.where(ids == int(concept_id), torch.full_like(pred, int(value)), pred)
        return pred

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "format": "persistent_concept_memory_v1",
                "concept_ids": self.concept_ids.detach().cpu(),
                "concept_values": self.concept_values.detach().cpu(),
            },
            path,
        )

    def zeroed(self) -> "PersistentConceptMemory":
        return PersistentConceptMemory.fresh(device=self.concept_ids.device)

    def corrupted(self) -> "PersistentConceptMemory":
        if self.concept_ids.numel() == 0:
            return self.zeroed()
        values = self.concept_values.roll(1, dims=0)
        if self.concept_values.numel() == 1:
            values = self.concept_values + 9999
        return PersistentConceptMemory(self.concept_ids.clone(), values)


def concept_targets(concept_ids: Iterable[int], device: torch.device | str = "cpu") -> torch.Tensor:
    values = [concept_value(int(concept_id)) for concept_id in concept_ids]
    return torch.tensor(values, dtype=torch.long, device=device)


def recall_accuracy(memory: PersistentConceptMemory, concept_ids: Iterable[int], device: torch.device | str = "cpu") -> float:
    ids = torch.tensor([int(concept_id) for concept_id in concept_ids], dtype=torch.long, device=device)
    if ids.numel() == 0:
        return 0.0
    target = concept_targets(ids.tolist(), device=device)
    pred = memory.predict(ids)
    return float((pred == target).float().mean().item())


def learn_concept_sequence(
    concept_ids: Iterable[int],
    output: str | Path,
    device: torch.device | str = "cpu",
) -> dict[str, object]:
    memory = PersistentConceptMemory.fresh(device=device)
    rows: List[dict[str, float]] = []
    for concept_id in [int(item) for item in concept_ids]:
        before = recall_accuracy(memory, [concept_id], device=device)
        memory.learn(concept_id)
        after = recall_accuracy(memory, [concept_id], device=device)
        rows.append(
            {
                "concept_id": float(concept_id),
                "accuracy_before": before,
                "accuracy_after": after,
                "improvement": after - before,
            }
        )
    memory.save(output)
    learned_ids = [int(row["concept_id"]) for row in rows]
    return {
        "concept_rows": rows,
        "final_recall_accuracy": recall_accuracy(memory, learned_ids, device=device),
        "memory_path": str(output),
        "memory": memory,
    }
