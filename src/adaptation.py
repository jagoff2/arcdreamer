from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import Any

import torch


FAST_ADAPTATION_MODULES = (
    "PersistentMemoryState.latent",
    "PersistentMemoryState.private_token",
    "transition_graph",
    "semantic_memory",
    "plastic_memory.low_rank_adapter",
    "plastic_memory.self_supervised_adapter",
    "hypothesis_posterior",
    "goal_posterior",
    "macro_policy_library",
    "controllability_model",
    "progress_value_model",
    "coordinate_affordances",
    "context_state",
)


def _as_state_dict(model_or_state: Any) -> Mapping[str, torch.Tensor]:
    if isinstance(model_or_state, Mapping):
        return model_or_state
    if hasattr(model_or_state, "state_dict"):
        return model_or_state.state_dict()
    raise TypeError("expected a model with state_dict() or a mapping of tensors")


def model_state_fingerprint(model_or_state: Any) -> str:
    digest = hashlib.sha256()
    try:
        state = _as_state_dict(model_or_state)
    except TypeError:
        digest.update(type(model_or_state).__qualname__.encode("utf-8"))
        config = getattr(model_or_state, "config", None)
        digest.update(repr(getattr(config, "__dict__", config)).encode("utf-8"))
        return digest.hexdigest().upper()
    for key in sorted(state):
        value = state[key]
        if not torch.is_tensor(value):
            continue
        tensor = value.detach().cpu().contiguous()
        digest.update(key.encode("utf-8"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest().upper()


def parameter_count(model_or_state: Any) -> int:
    total = 0
    state = _as_state_dict(model_or_state)
    for value in state.values():
        if torch.is_tensor(value):
            total += int(value.numel())
    return total


def base_checkpoint_contract(model_or_state: Any) -> dict[str, Any]:
    return {
        "schema": "reusable_base_weights_v1",
        "role": "reusable_base_weights",
        "sha256": model_state_fingerprint(model_or_state),
        "parameter_count": parameter_count(model_or_state),
        "trained_from": "local_random_initialization",
        "mutable_during_evaluation": False,
    }


def fast_adaptation_contract() -> dict[str, Any]:
    return {
        "schema": "runtime_fast_adaptation_sidecar_v1",
        "role": "runtime_fast_adaptation",
        "stored_in_checkpoint": False,
        "mutates_base_weights": False,
        "modules": list(FAST_ADAPTATION_MODULES),
    }


def fast_adaptation_summary(memory_like: Mapping[str, Any]) -> dict[str, float]:
    plastic = memory_like.get("plastic_memory", {})
    if not isinstance(plastic, Mapping):
        plastic = {}
    self_adapter = plastic.get("self_supervised_adapter", {})
    if not isinstance(self_adapter, Mapping):
        self_adapter = {}
    low_rank = plastic.get("low_rank_adapter", {})
    if not isinstance(low_rank, Mapping):
        low_rank = {}
    return {
        "fast_adaptation_updates": float(plastic.get("updates", 0)),
        "self_supervised_adapter_updates": float(self_adapter.get("updates", 0)),
        "low_rank_adapter_norm": float(low_rank.get("norm", 0.0)),
    }
