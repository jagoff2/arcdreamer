from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import torch
import torch.nn.functional as F


SOURCE_NAMES = ("observed", "told", "imagined", "inferred", "replayed", "reconstructed")
NUM_SOURCES = len(SOURCE_NAMES)


@dataclass(frozen=True)
class HumanMemoryConfig:
    cue_dim: int = 48
    latent_dim: int = 64
    content_dim: int = 32
    body_dim: int = 4
    affect_dim: int = 3
    action_dim: int = 5
    private_dim: int = 18
    language_dim: int = 46
    key_dim: int = 64
    sparse_dim: int = 192
    sparse_k: int = 12
    rejection_threshold: float = 0.45
    seed: int = 99123


@dataclass
class RecallResult:
    accepted: bool
    trace_index: int
    confidence: float
    content: torch.Tensor
    latent: torch.Tensor
    body: torch.Tensor
    affect: torch.Tensor
    action_logits: torch.Tensor
    private_logits: torch.Tensor
    language_logits: torch.Tensor
    source_distribution: torch.Tensor
    source_history: torch.Tensor


def _unit(x: torch.Tensor) -> torch.Tensor:
    if x.ndim == 1:
        return F.normalize(x.float(), dim=0)
    return F.normalize(x.float(), dim=-1)


def deterministic_vector(index: int, dim: int, seed: int = 40000) -> torch.Tensor:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed + int(index) * 104729)
    return _unit(torch.randn(dim, generator=generator))


def one_hot(index: int, dim: int) -> torch.Tensor:
    out = torch.zeros(dim)
    out[int(index) % dim] = 1.0
    return out


class HumanAnalogueMemory:
    def __init__(self, config: HumanMemoryConfig | None = None, device: torch.device | str = "cpu") -> None:
        self.config = config or HumanMemoryConfig()
        self.device = torch.device(device)
        generator = torch.Generator(device="cpu")
        generator.manual_seed(self.config.seed)
        self.cue_projection = _unit(torch.randn(self.config.cue_dim, self.config.key_dim, generator=generator)).to(self.device)
        self.latent_projection = _unit(torch.randn(self.config.latent_dim, self.config.key_dim, generator=generator)).to(self.device)
        self.body_projection = _unit(torch.randn(self.config.body_dim, self.config.key_dim, generator=generator)).to(self.device)
        self.source_projection = _unit(torch.randn(NUM_SOURCES, self.config.key_dim, generator=generator)).to(self.device)
        self.separation_projection = _unit(torch.randn(self.config.key_dim, self.config.sparse_dim, generator=generator)).to(self.device)
        self.action_projection = _unit(torch.randn(self.config.content_dim, self.config.action_dim, generator=generator)).to(self.device)
        self.private_projection = _unit(torch.randn(self.config.content_dim, self.config.private_dim, generator=generator)).to(self.device)
        self.language_projection = _unit(torch.randn(self.config.content_dim, self.config.language_dim, generator=generator)).to(self.device)
        self.semantic_weight: torch.Tensor | None = None
        self._keys: list[torch.Tensor] = []
        self._sparse: list[torch.Tensor] = []
        self._cues: list[torch.Tensor] = []
        self._contents: list[torch.Tensor] = []
        self._latents: list[torch.Tensor] = []
        self._bodies: list[torch.Tensor] = []
        self._affects: list[torch.Tensor] = []
        self._actions: list[torch.Tensor] = []
        self._private: list[torch.Tensor] = []
        self._sources: list[torch.Tensor] = []
        self._source_history: list[torch.Tensor] = []
        self._times: list[float] = []
        self._confidence: list[float] = []

    @property
    def trace_count(self) -> int:
        return len(self._keys)

    def clone(self) -> "HumanAnalogueMemory":
        out = HumanAnalogueMemory(self.config, self.device)
        for name in (
            "_keys",
            "_sparse",
            "_cues",
            "_contents",
            "_latents",
            "_bodies",
            "_affects",
            "_actions",
            "_private",
            "_sources",
            "_source_history",
        ):
            setattr(out, name, [item.clone() for item in getattr(self, name)])
        out._times = list(self._times)
        out._confidence = list(self._confidence)
        out.semantic_weight = None if self.semantic_weight is None else self.semantic_weight.clone()
        return out

    def _key(
        self,
        cue: torch.Tensor,
        latent: torch.Tensor | None = None,
        body: torch.Tensor | None = None,
        source_id: int | None = None,
    ) -> torch.Tensor:
        cue = cue.to(self.device).float()
        key = cue @ self.cue_projection
        if latent is not None:
            key = key + 0.25 * latent.to(self.device).float() @ self.latent_projection
        if body is not None:
            key = key + 0.10 * body.to(self.device).float() @ self.body_projection
        if source_id is not None:
            key = key + 0.05 * self.source_projection[int(source_id) % NUM_SOURCES]
        return _unit(key)

    def _sparse_code(self, key: torch.Tensor) -> torch.Tensor:
        activations = key @ self.separation_projection
        topk = torch.topk(activations, k=self.config.sparse_k).indices
        code = torch.zeros(self.config.sparse_dim, device=self.device)
        code[topk] = 1.0
        return code

    def write(
        self,
        cue: torch.Tensor,
        latent: torch.Tensor,
        body: torch.Tensor,
        affect: torch.Tensor,
        action_context: torch.Tensor,
        source_id: int,
        private_state: torch.Tensor,
        content: torch.Tensor,
        time_index: float,
    ) -> int:
        key = self._key(cue, latent=latent, body=body, source_id=source_id)
        sparse = self._sparse_code(key)
        source = one_hot(source_id, NUM_SOURCES).to(self.device)
        self._keys.append(key)
        self._sparse.append(sparse)
        self._cues.append(cue.to(self.device).float().clone())
        self._contents.append(_unit(content.to(self.device).float()))
        self._latents.append(latent.to(self.device).float().clone())
        self._bodies.append(body.to(self.device).float().clone())
        self._affects.append(affect.to(self.device).float().clone())
        self._actions.append(action_context.to(self.device).float().clone())
        self._private.append(private_state.to(self.device).float().clone())
        self._sources.append(source)
        self._source_history.append(source.clone())
        self._times.append(float(time_index))
        self._confidence.append(1.0)
        return len(self._keys) - 1

    def _bank(self, items: list[torch.Tensor]) -> torch.Tensor:
        if not items:
            raise ValueError("memory has no traces")
        return torch.stack(items, dim=0)

    def attention(self, cue: torch.Tensor, latent: torch.Tensor | None = None, body: torch.Tensor | None = None) -> torch.Tensor:
        query_key = self._key(cue, latent=latent, body=body)
        query_sparse = self._sparse_code(query_key)
        keys = self._bank(self._keys)
        sparse = self._bank(self._sparse)
        cosine = keys @ query_key
        overlap = (sparse @ query_sparse) / float(self.config.sparse_k)
        logits = 9.0 * cosine + 2.0 * overlap
        return F.softmax(logits, dim=0)

    def recall(
        self,
        cue: torch.Tensor,
        latent: torch.Tensor | None = None,
        body: torch.Tensor | None = None,
    ) -> RecallResult:
        if self.trace_count == 0:
            return self._empty_recall()
        attn = self.attention(cue, latent=latent, body=body)
        trace_index = int(attn.argmax().item())
        query_key = self._key(cue, latent=latent, body=body)
        query_sparse = self._sparse_code(query_key)
        cosine = self._bank(self._keys) @ query_key
        overlap = (self._bank(self._sparse) @ query_sparse) / float(self.config.sparse_k)
        confidence = float((0.65 * cosine + 0.35 * overlap).max().item())
        if confidence < self.config.rejection_threshold:
            return self._empty_recall(trace_index=trace_index, confidence=confidence)
        content = attn @ self._bank(self._contents)
        latent_out = attn @ self._bank(self._latents)
        body_out = attn @ self._bank(self._bodies)
        affect_out = attn @ self._bank(self._affects)
        source = attn @ self._bank(self._sources)
        history = attn @ self._bank(self._source_history)
        action = content @ self.action_projection + 0.10 * (attn @ self._bank(self._actions))
        private = content @ self.private_projection + 0.10 * (attn @ self._bank(self._private))
        language = content @ self.language_projection
        return RecallResult(
            accepted=True,
            trace_index=trace_index,
            confidence=confidence,
            content=_unit(content),
            latent=latent_out,
            body=body_out,
            affect=affect_out,
            action_logits=action,
            private_logits=private,
            language_logits=language,
            source_distribution=F.normalize(source.clamp_min(0.0), p=1, dim=0),
            source_history=history.clamp_min(0.0),
        )

    def _empty_recall(self, trace_index: int = -1, confidence: float = 0.0) -> RecallResult:
        return RecallResult(
            accepted=False,
            trace_index=trace_index,
            confidence=confidence,
            content=torch.zeros(self.config.content_dim, device=self.device),
            latent=torch.zeros(self.config.latent_dim, device=self.device),
            body=torch.zeros(self.config.body_dim, device=self.device),
            affect=torch.zeros(self.config.affect_dim, device=self.device),
            action_logits=torch.zeros(self.config.action_dim, device=self.device),
            private_logits=torch.zeros(self.config.private_dim, device=self.device),
            language_logits=torch.zeros(self.config.language_dim, device=self.device),
            source_distribution=torch.zeros(NUM_SOURCES, device=self.device),
            source_history=torch.zeros(NUM_SOURCES, device=self.device),
        )

    def corrupt_trace(self, trace_index: int, strength: float = 1.0) -> None:
        idx = int(trace_index)
        self._contents[idx] = _unit((1.0 - strength) * self._contents[idx])
        self._actions[idx] = (1.0 - strength) * self._actions[idx]
        self._private[idx] = (1.0 - strength) * self._private[idx]
        self._confidence[idx] *= max(0.0, 1.0 - strength)

    def perturb_trace(self, trace_index: int, noise_scale: float = 0.85) -> None:
        idx = int(trace_index)
        noise = deterministic_vector(9000 + idx, self.config.content_dim).to(self.device)
        self._contents[idx] = _unit(self._contents[idx] + noise_scale * noise)

    def reconsolidate(
        self,
        trace_index: int,
        new_content: torch.Tensor,
        new_source_id: int,
        affect_delta: torch.Tensor | None = None,
    ) -> None:
        idx = int(trace_index)
        self._contents[idx] = _unit(new_content.to(self.device).float())
        source = one_hot(new_source_id, NUM_SOURCES).to(self.device)
        self._sources[idx] = source
        self._source_history[idx] = torch.clamp(self._source_history[idx] + source, 0.0, 1.0)
        if affect_delta is not None:
            self._affects[idx] = self._affects[idx] + affect_delta.to(self.device).float()
        self._confidence[idx] = min(1.0, self._confidence[idx] + 0.05)

    def zeroed(self) -> "HumanAnalogueMemory":
        return HumanAnalogueMemory(self.config, self.device)

    def corrupted(self) -> "HumanAnalogueMemory":
        out = self.clone()
        for idx in range(out.trace_count):
            out.corrupt_trace(idx)
        return out

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "format": "human_analogue_memory_v1",
            "config": self.config.__dict__,
            "semantic_weight": None if self.semantic_weight is None else self.semantic_weight.detach().cpu(),
            "times": torch.tensor(self._times, dtype=torch.float32),
            "confidence": torch.tensor(self._confidence, dtype=torch.float32),
        }
        for name in (
            "keys",
            "sparse",
            "cues",
            "contents",
            "latents",
            "bodies",
            "affects",
            "actions",
            "private",
            "sources",
            "source_history",
        ):
            payload[name] = self._bank(getattr(self, f"_{name}")).detach().cpu() if self.trace_count else torch.empty(0)
        torch.save(payload, path)

    @classmethod
    def load(cls, path: str | Path, device: torch.device | str = "cpu") -> "HumanAnalogueMemory":
        payload = torch.load(Path(path), map_location=device)
        if payload.get("format") != "human_analogue_memory_v1":
            raise ValueError(f"{path} is not a human analogue memory file")
        memory = cls(HumanMemoryConfig(**payload["config"]), device=device)
        for name in (
            "keys",
            "sparse",
            "cues",
            "contents",
            "latents",
            "bodies",
            "affects",
            "actions",
            "private",
            "sources",
            "source_history",
        ):
            tensor = payload[name].to(device).float()
            setattr(memory, f"_{name}", [row.clone() for row in tensor] if tensor.numel() else [])
        memory._times = [float(item) for item in payload["times"].tolist()]
        memory._confidence = [float(item) for item in payload["confidence"].tolist()]
        semantic = payload.get("semantic_weight")
        memory.semantic_weight = None if semantic is None else semantic.to(device).float()
        return memory


def nearest_accuracy(predictions: Iterable[torch.Tensor], targets: Iterable[torch.Tensor], bank: torch.Tensor) -> float:
    correct = 0
    total = 0
    bank = _unit(bank)
    for pred, target in zip(predictions, targets):
        pred_idx = int((_unit(pred.to(bank.device).float()) @ bank.T).argmax().item())
        target_idx = int((_unit(target.to(bank.device).float()) @ bank.T).argmax().item())
        correct += int(pred_idx == target_idx)
        total += 1
    return float(correct / total) if total else 0.0
