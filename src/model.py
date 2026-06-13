from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
from pathlib import Path
from typing import Any, Dict, Mapping, Tuple

import torch
from torch import nn

from .adaptation import base_checkpoint_contract, fast_adaptation_contract
from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .env import (
    GRID_SIZE,
    NUM_ACTIONS,
    NUM_COLORS,
    NUM_INPUT_TOKENS,
    NUM_LANGUAGE_TOKENS,
    NUM_PRIVATE_TOKENS,
    NUM_PROVENANCE,
    SENSOR_DIM,
)

PROGRAM_FAMILIES: tuple[str, ...] = ("no_op", "field_change", "field_stable", "move_color")
PROGRAM_FIELDS: tuple[str, ...] = ("sensory", "grid", "lang_in", "private_in")
PROGRAM_TRANSFORMS: tuple[str, ...] = (
    "identity",
    "change",
    "translate_up",
    "translate_down",
    "translate_left",
    "translate_right",
)
OBJECT_FEATURE_DIM = 16
RELATION_FEATURE_DIM = 16
RELATION_TYPES: tuple[str, ...] = (
    "same_color",
    "adjacent",
    "contains",
    "row_aligned",
    "col_aligned",
    "repetition",
    "symmetry",
    "has_hole",
)


@dataclass
class ModelConfig:
    sensor_dim: int = SENSOR_DIM
    input_tokens: int = NUM_INPUT_TOKENS
    private_tokens: int = NUM_PRIVATE_TOKENS
    language_tokens: int = NUM_LANGUAGE_TOKENS
    hidden_dim: int = 64
    embed_dim: int = 16
    private_embed_dim: int = 8
    object_feature_dim: int = OBJECT_FEATURE_DIM
    relation_feature_dim: int = RELATION_FEATURE_DIM


def _field(item: Any, name: str, default: Any = None) -> Any:
    if isinstance(item, Mapping):
        return item.get(name, default)
    return getattr(item, name, default)


def _hash_fraction(value: Any) -> float:
    digest = hashlib.sha256(str(value).encode("utf-8")).digest()
    bucket = int.from_bytes(digest[:2], "big")
    return float(bucket / 65535.0)


def _as_tensor_features(
    value: Any,
    *,
    feature_dim: int,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if value is None:
        return torch.zeros(batch_size, 0, feature_dim, dtype=torch.float32, device=device)
    tensor = torch.as_tensor(value, dtype=torch.float32, device=device)
    if tensor.ndim == 1:
        tensor = tensor.view(1, 1, -1)
    elif tensor.ndim == 2:
        tensor = tensor.unsqueeze(0)
    elif tensor.ndim != 3:
        raise ValueError(f"expected 1D, 2D, or 3D feature tensor, got shape={tuple(tensor.shape)}")
    if tensor.shape[0] == 1 and batch_size > 1:
        tensor = tensor.expand(batch_size, -1, -1)
    if tensor.shape[0] != batch_size:
        raise ValueError(f"feature batch {tensor.shape[0]} does not match batch_size={batch_size}")
    if tensor.shape[-1] < feature_dim:
        pad = torch.zeros(*tensor.shape[:-1], feature_dim - tensor.shape[-1], dtype=tensor.dtype, device=device)
        tensor = torch.cat([tensor, pad], dim=-1)
    elif tensor.shape[-1] > feature_dim:
        tensor = tensor[..., :feature_dim]
    return tensor


class ObjectRelationEncoder(nn.Module):
    def __init__(
        self,
        hidden_dim: int = 64,
        object_feature_dim: int = OBJECT_FEATURE_DIM,
        relation_feature_dim: int = RELATION_FEATURE_DIM,
    ) -> None:
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.object_feature_dim = int(object_feature_dim)
        self.relation_feature_dim = int(relation_feature_dim)
        self.object_encoder = nn.Sequential(
            nn.Linear(self.object_feature_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.relation_encoder = nn.Sequential(
            nn.Linear(self.relation_feature_dim, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.Tanh(),
        )
        self.summary_encoder = nn.Sequential(
            nn.Linear(self.hidden_dim * 4 + 4, self.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.hidden_dim, self.hidden_dim),
        )

    @staticmethod
    def object_features_from_scene(scene: Any, *, device: DeviceLike = None) -> torch.Tensor:
        target_device = resolve_device(device)
        objects = _field(scene, "objects", ()) or ()
        rows: list[list[float]] = []
        for obj in objects:
            kind = str(_field(obj, "kind", "other"))
            bbox = tuple(float(item) for item in (_field(obj, "bbox", (0, 0, 0, 0)) or (0, 0, 0, 0)))
            if len(bbox) != 4:
                bbox = (0.0, 0.0, 0.0, 0.0)
            centroid = tuple(float(item) for item in (_field(obj, "centroid", (0.0, 0.0)) or (0.0, 0.0)))
            if len(centroid) != 2:
                centroid = (0.0, 0.0)
            symmetry = _field(obj, "symmetry", {}) or {}
            area = max(float(_field(obj, "area", 0.0)), 0.0)
            boundary = _field(obj, "boundary_cells", ()) or ()
            changed = max(float(_field(obj, "changed_count", 0.0)), 0.0)
            row = [
                1.0 if kind == "component" else 0.0,
                1.0 if kind == "color_region" else 0.0,
                1.0 if kind not in {"component", "color_region"} else 0.0,
                float(_field(obj, "color", 0.0)) / max(NUM_COLORS - 1, 1),
                min(area / float(64 * 64), 1.0),
                max(bbox[0], 0.0) / 63.0,
                max(bbox[1], 0.0) / 63.0,
                max(bbox[2], 0.0) / 63.0,
                max(bbox[3], 0.0) / 63.0,
                max(centroid[0], 0.0) / 63.0,
                max(centroid[1], 0.0) / 63.0,
                min(float(_field(obj, "holes", 0.0)) / 16.0, 1.0),
                min(changed / max(area, 1.0), 1.0),
                min(float(len(boundary)) / max(area, 1.0), 1.0),
                1.0 if bool(_field(symmetry, "horizontal", False)) else 0.0,
                1.0 if bool(_field(symmetry, "vertical", False)) else 0.0,
            ]
            rows.append(row)
        if not rows:
            return torch.zeros(0, OBJECT_FEATURE_DIM, dtype=torch.float32, device=target_device)
        return torch.tensor(rows, dtype=torch.float32, device=target_device)

    @staticmethod
    def relation_features_from_scene(scene: Any, *, device: DeviceLike = None) -> torch.Tensor:
        target_device = resolve_device(device)
        relations = _field(scene, "relation_graph", ()) or ()
        rows: list[list[float]] = []
        for relation in relations:
            relation_name = str(_field(relation, "relation", "other"))
            detail = _field(relation, "detail", {}) or {}
            type_features = [1.0 if relation_name == name else 0.0 for name in RELATION_TYPES]
            other = 0.0 if relation_name in RELATION_TYPES else 1.0
            delta = detail.get("delta", [0.0, 0.0]) if isinstance(detail, Mapping) else [0.0, 0.0]
            if not isinstance(delta, (list, tuple)) or len(delta) != 2:
                delta = [0.0, 0.0]
            color = float(detail.get("color", 0.0)) if isinstance(detail, Mapping) else 0.0
            count = float(detail.get("count", 0.0)) if isinstance(detail, Mapping) else 0.0
            source = str(_field(relation, "source", ""))
            target = str(_field(relation, "target", ""))
            row = [
                *type_features,
                other,
                color / max(NUM_COLORS - 1, 1),
                max(min(float(delta[0]) / 64.0, 1.0), -1.0),
                max(min(float(delta[1]) / 64.0, 1.0), -1.0),
                min(max(count, 0.0) / 16.0, 1.0),
                1.0 if source == target else 0.0,
                _hash_fraction(f"{source}|{target}"),
                _hash_fraction(json_like(detail)),
            ]
            rows.append(row[:RELATION_FEATURE_DIM])
        if not rows:
            return torch.zeros(0, RELATION_FEATURE_DIM, dtype=torch.float32, device=target_device)
        return torch.tensor(rows, dtype=torch.float32, device=target_device)

    @staticmethod
    def scene_to_features(scene: Any, *, device: DeviceLike = None) -> tuple[torch.Tensor, torch.Tensor]:
        return (
            ObjectRelationEncoder.object_features_from_scene(scene, device=device),
            ObjectRelationEncoder.relation_features_from_scene(scene, device=device),
        )

    def _features_from_scenes(
        self,
        scenes: Any,
        *,
        batch_size: int | None,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if scenes is None:
            count = 1 if batch_size is None else int(batch_size)
            return (
                torch.zeros(count, 0, self.object_feature_dim, dtype=torch.float32, device=device),
                torch.zeros(count, 0, self.relation_feature_dim, dtype=torch.float32, device=device),
            )
        if not isinstance(scenes, (list, tuple)):
            scenes = [scenes]
        if batch_size is not None and len(scenes) == 1 and int(batch_size) > 1:
            scenes = list(scenes) * int(batch_size)
        object_rows = [self.object_features_from_scene(scene, device=device) for scene in scenes]
        relation_rows = [self.relation_features_from_scene(scene, device=device) for scene in scenes]
        max_objects = max((tensor.shape[0] for tensor in object_rows), default=0)
        max_relations = max((tensor.shape[0] for tensor in relation_rows), default=0)
        object_batch = torch.zeros(len(scenes), max_objects, self.object_feature_dim, dtype=torch.float32, device=device)
        relation_batch = torch.zeros(
            len(scenes), max_relations, self.relation_feature_dim, dtype=torch.float32, device=device
        )
        for index, tensor in enumerate(object_rows):
            if tensor.numel() > 0:
                object_batch[index, : tensor.shape[0]] = tensor[:, : self.object_feature_dim]
        for index, tensor in enumerate(relation_rows):
            if tensor.numel() > 0:
                relation_batch[index, : tensor.shape[0]] = tensor[:, : self.relation_feature_dim]
        return object_batch, relation_batch

    def forward(
        self,
        scenes: Any = None,
        *,
        object_features: Any = None,
        relation_features: Any = None,
        batch_size: int | None = None,
        device: DeviceLike = None,
    ) -> torch.Tensor:
        target_device = next(self.parameters()).device if device is None else resolve_device(device)
        if object_features is None and relation_features is None:
            object_tensor, relation_tensor = self._features_from_scenes(
                scenes, batch_size=batch_size, device=target_device
            )
        else:
            inferred_batch = int(batch_size or 0)
            if inferred_batch <= 0:
                sample = object_features if object_features is not None else relation_features
                sample_tensor = torch.as_tensor(sample)
                inferred_batch = int(sample_tensor.shape[0]) if sample_tensor.ndim == 3 else 1
            object_tensor = _as_tensor_features(
                object_features,
                feature_dim=self.object_feature_dim,
                batch_size=inferred_batch,
                device=target_device,
            )
            relation_tensor = _as_tensor_features(
                relation_features,
                feature_dim=self.relation_feature_dim,
                batch_size=inferred_batch,
                device=target_device,
            )
        object_count = (object_tensor.abs().sum(dim=-1) > 0.0).float().sum(dim=1, keepdim=True)
        relation_count = (relation_tensor.abs().sum(dim=-1) > 0.0).float().sum(dim=1, keepdim=True)
        if object_tensor.shape[1] == 0:
            object_mean = object_max = torch.zeros(object_tensor.shape[0], self.hidden_dim, device=target_device)
        else:
            object_encoded = self.object_encoder(object_tensor)
            object_mask = (object_tensor.abs().sum(dim=-1, keepdim=True) > 0.0).float()
            object_mean = (object_encoded * object_mask).sum(dim=1) / object_count.clamp_min(1.0)
            object_max = object_encoded.masked_fill(object_mask == 0.0, -1.0e4).max(dim=1).values
            object_max = torch.where(object_count > 0.0, object_max, torch.zeros_like(object_max))
        if relation_tensor.shape[1] == 0:
            relation_mean = relation_max = torch.zeros(relation_tensor.shape[0], self.hidden_dim, device=target_device)
        else:
            relation_encoded = self.relation_encoder(relation_tensor)
            relation_mask = (relation_tensor.abs().sum(dim=-1, keepdim=True) > 0.0).float()
            relation_mean = (relation_encoded * relation_mask).sum(dim=1) / relation_count.clamp_min(1.0)
            relation_max = relation_encoded.masked_fill(relation_mask == 0.0, -1.0e4).max(dim=1).values
            relation_max = torch.where(relation_count > 0.0, relation_max, torch.zeros_like(relation_max))
        count_features = torch.cat(
            [
                torch.clamp(object_count / 32.0, 0.0, 1.0),
                torch.clamp(relation_count / 64.0, 0.0, 1.0),
                torch.clamp(object_count / relation_count.clamp_min(1.0), 0.0, 1.0),
                torch.clamp(relation_count / object_count.clamp_min(1.0), 0.0, 1.0),
            ],
            dim=-1,
        )
        return self.summary_encoder(torch.cat([object_mean, object_max, relation_mean, relation_max, count_features], dim=-1))


def json_like(value: Any) -> str:
    if isinstance(value, Mapping):
        return "|".join(f"{key}:{json_like(value[key])}" for key in sorted(value))
    if isinstance(value, (list, tuple)):
        return "[" + ",".join(json_like(item) for item in value) + "]"
    return str(value)


class RecurrentLatentModel(nn.Module):
    def __init__(self, config: ModelConfig | None = None, device: DeviceLike = AUTO_DEVICE) -> None:
        super().__init__()
        self.config = config or ModelConfig()
        self.sensor_encoder = nn.Sequential(
            nn.Linear(self.config.sensor_dim, 48),
            nn.Tanh(),
            nn.Linear(48, 48),
            nn.Tanh(),
        )
        self.token_embedding = nn.Embedding(self.config.input_tokens, self.config.embed_dim)
        self.private_embedding = nn.Embedding(self.config.private_tokens, self.config.private_embed_dim)
        self.input_mixer = nn.Sequential(
            nn.Linear(48 + self.config.embed_dim + self.config.private_embed_dim, self.config.hidden_dim),
            nn.Tanh(),
        )
        self.object_relation_encoder = ObjectRelationEncoder(
            hidden_dim=self.config.hidden_dim,
            object_feature_dim=self.config.object_feature_dim,
            relation_feature_dim=self.config.relation_feature_dim,
        )
        self.action_impulse_embedding = nn.Embedding(NUM_ACTIONS + 1, self.config.hidden_dim)
        self.delta_impulse_encoder = nn.Linear(self.config.sensor_dim, self.config.hidden_dim)
        self.core = nn.GRUCell(self.config.hidden_dim, self.config.hidden_dim)
        liquid_input_dim = self.config.hidden_dim * 2
        self.liquid_time_constant = nn.Linear(liquid_input_dim, self.config.hidden_dim)
        self.liquid_drift = nn.Sequential(
            nn.Linear(liquid_input_dim, self.config.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dim, self.config.hidden_dim),
        )
        self.norm = nn.LayerNorm(self.config.hidden_dim)
        self.action_head = nn.Linear(self.config.hidden_dim, NUM_ACTIONS)
        self.action_context_head = nn.Sequential(
            nn.Linear(self.config.hidden_dim + self.config.sensor_dim + GRID_SIZE, self.config.hidden_dim),
            nn.Tanh(),
            nn.Linear(self.config.hidden_dim, NUM_ACTIONS),
        )
        self.language_head = nn.Linear(self.config.hidden_dim, self.config.language_tokens)
        self.private_head = nn.Linear(self.config.hidden_dim, self.config.private_tokens)
        self.provenance_head = nn.Linear(self.config.hidden_dim, NUM_PROVENANCE)
        self.world_color_head = nn.Linear(self.config.hidden_dim, NUM_COLORS)
        self.world_pos_head = nn.Linear(self.config.hidden_dim, GRID_SIZE)
        self.memory_color_head = nn.Linear(self.config.hidden_dim, NUM_COLORS)
        self.self_start_head = nn.Linear(self.config.hidden_dim, GRID_SIZE)
        self.program_family_head = nn.Linear(self.config.hidden_dim, len(PROGRAM_FAMILIES))
        self.program_field_head = nn.Linear(self.config.hidden_dim, len(PROGRAM_FIELDS))
        self.program_transform_head = nn.Linear(self.config.hidden_dim, len(PROGRAM_TRANSFORMS))
        self.program_color_head = nn.Linear(self.config.hidden_dim, NUM_COLORS)
        self._initialize_impulse_paths()
        self._initialize_liquid_dynamics()
        self._initialize_action_context_head()
        self._initialize_program_proposal_heads()
        self.to(resolve_device(device))

    def _initialize_impulse_paths(self) -> None:
        nn.init.zeros_(self.action_impulse_embedding.weight)
        nn.init.zeros_(self.delta_impulse_encoder.weight)
        nn.init.zeros_(self.delta_impulse_encoder.bias)

    def _initialize_liquid_dynamics(self) -> None:
        nn.init.zeros_(self.liquid_drift[-1].weight)
        nn.init.zeros_(self.liquid_drift[-1].bias)

    def _initialize_action_context_head(self) -> None:
        nn.init.zeros_(self.action_context_head[-1].weight)
        nn.init.zeros_(self.action_context_head[-1].bias)

    def _initialize_program_proposal_heads(self) -> None:
        for head in (
            self.program_family_head,
            self.program_field_head,
            self.program_transform_head,
            self.program_color_head,
        ):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)

    def initial_state(self, batch_size: int, device: DeviceLike = None) -> torch.Tensor:
        target_device = next(self.parameters()).device if device is None else resolve_device(device)
        return torch.zeros(batch_size, self.config.hidden_dim, device=target_device)

    def _impulse_input(
        self,
        observation: Dict[str, torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        prev_action = observation.get("prev_action")
        if prev_action is None:
            prev_action = torch.full((batch_size,), NUM_ACTIONS, dtype=torch.long, device=device)
        else:
            prev_action = prev_action.to(device=device, dtype=torch.long).view(batch_size).clamp(0, NUM_ACTIONS)

        prev_delta = observation.get("prev_delta")
        if prev_delta is None:
            prev_delta = torch.zeros(batch_size, self.config.sensor_dim, dtype=torch.float32, device=device)
        else:
            prev_delta = prev_delta.to(device=device, dtype=torch.float32).view(batch_size, self.config.sensor_dim)

        return self.action_impulse_embedding(prev_action) + torch.tanh(self.delta_impulse_encoder(prev_delta))

    def _object_relation_input(
        self,
        observation: Dict[str, Any],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        direct_embedding = observation.get("object_relation_embedding")
        if direct_embedding is not None:
            embedding = torch.as_tensor(direct_embedding, dtype=torch.float32, device=device)
            if embedding.ndim == 1:
                embedding = embedding.view(1, -1)
            if embedding.shape[0] == 1 and batch_size > 1:
                embedding = embedding.expand(batch_size, -1)
            if embedding.shape[0] != batch_size:
                raise ValueError(f"object_relation_embedding batch {embedding.shape[0]} does not match {batch_size}")
            if embedding.shape[-1] < self.config.hidden_dim:
                pad = torch.zeros(batch_size, self.config.hidden_dim - embedding.shape[-1], device=device)
                embedding = torch.cat([embedding, pad], dim=-1)
            elif embedding.shape[-1] > self.config.hidden_dim:
                embedding = embedding[:, : self.config.hidden_dim]
            return embedding

        scenes = (
            observation.get("perception_scenes")
            or observation.get("perception_scene")
            or observation.get("perception")
        )
        object_features = observation.get("object_features")
        relation_features = observation.get("relation_features")
        if scenes is None and object_features is None and relation_features is None:
            return torch.zeros(batch_size, self.config.hidden_dim, dtype=torch.float32, device=device)
        return self.object_relation_encoder(
            scenes,
            object_features=object_features,
            relation_features=relation_features,
            batch_size=batch_size,
            device=device,
        )

    def _dt_input(
        self,
        observation: Dict[str, torch.Tensor],
        *,
        batch_size: int,
        device: torch.device,
    ) -> torch.Tensor:
        dt = observation.get("dt")
        if dt is None:
            return torch.ones(batch_size, 1, dtype=torch.float32, device=device)
        dt_tensor = torch.as_tensor(dt, dtype=torch.float32, device=device).reshape(-1)
        if dt_tensor.numel() == 1:
            dt_tensor = dt_tensor.expand(batch_size)
        else:
            dt_tensor = dt_tensor.view(batch_size)
        return dt_tensor.clamp_min(0.0).view(batch_size, 1)

    def _liquid_integrate(
        self,
        *,
        mixed: torch.Tensor,
        z_prev: torch.Tensor,
        z_candidate: torch.Tensor,
        observation: Dict[str, torch.Tensor],
    ) -> torch.Tensor:
        liquid_input = torch.cat([mixed, z_prev], dim=-1)
        tau = torch.nn.functional.softplus(self.liquid_time_constant(liquid_input)) + 1.0e-3
        dt = self._dt_input(observation, batch_size=z_prev.shape[0], device=z_prev.device)
        alpha = 1.0 - torch.exp(-dt / tau)
        drift = self.liquid_drift(liquid_input)
        return z_candidate + alpha * drift

    def step(
        self, observation: Dict[str, Any], z_prev: torch.Tensor
    ) -> Tuple[Dict[str, torch.Tensor], torch.Tensor]:
        sensory = observation["sensory"]
        lang_in = observation["lang_in"]
        private_in = observation.get("private_in")
        if private_in is None:
            private_in = torch.zeros_like(lang_in)
        sensor_features = self.sensor_encoder(sensory)
        token_features = self.token_embedding(lang_in)
        private_features = self.private_embedding(private_in)
        mixed = self.input_mixer(torch.cat([sensor_features, token_features, private_features], dim=-1))
        mixed = mixed + self._impulse_input(observation, batch_size=sensory.shape[0], device=sensory.device)
        mixed = mixed + self._object_relation_input(observation, batch_size=sensory.shape[0], device=sensory.device)
        z_candidate = self.core(mixed, z_prev)
        z_next = self._liquid_integrate(
            mixed=mixed,
            z_prev=z_prev,
            z_candidate=z_candidate,
            observation=observation,
        )
        z_view = self.norm(z_next)
        world_pos_logits = self.world_pos_head(z_view)
        action_context = torch.cat([z_view, sensory.to(dtype=z_view.dtype), world_pos_logits], dim=-1)
        output = {
            "action_logits": self.action_head(z_view) + self.action_context_head(action_context),
            "language_logits": self.language_head(z_view),
            "private_logits": self.private_head(z_view),
            "provenance_logits": self.provenance_head(z_view),
            "world_color_logits": self.world_color_head(z_view),
            "world_pos_logits": world_pos_logits,
            "memory_color_logits": self.memory_color_head(z_view),
            "self_start_logits": self.self_start_head(z_view),
            "program_family_logits": self.program_family_head(z_view),
            "program_field_logits": self.program_field_head(z_view),
            "program_transform_logits": self.program_transform_head(z_view),
            "program_color_logits": self.program_color_head(z_view),
        }
        return output, z_next

    def forward(
        self,
        sensory: torch.Tensor,
        lang_in: torch.Tensor,
        private_in: torch.Tensor | None = None,
        prev_action: torch.Tensor | None = None,
        prev_delta: torch.Tensor | None = None,
        dt: torch.Tensor | None = None,
        object_features: torch.Tensor | None = None,
        relation_features: torch.Tensor | None = None,
    ) -> Dict[str, torch.Tensor]:
        batch_size, seq_len, _ = sensory.shape
        z = self.initial_state(batch_size, sensory.device)
        if private_in is None:
            private_in = torch.zeros_like(lang_in)
        outputs = []
        latents = []
        for tick in range(seq_len):
            output, z = self.step(
                {
                    "sensory": sensory[:, tick],
                    "lang_in": lang_in[:, tick],
                    "private_in": private_in[:, tick],
                    "prev_action": prev_action[:, tick] if prev_action is not None else None,
                    "prev_delta": prev_delta[:, tick] if prev_delta is not None else None,
                    "dt": dt[:, tick] if dt is not None else None,
                    "object_features": object_features[:, tick] if object_features is not None else None,
                    "relation_features": relation_features[:, tick] if relation_features is not None else None,
                },
                z,
            )
            outputs.append(output)
            latents.append(z)
        stacked: Dict[str, torch.Tensor] = {}
        for key in outputs[0]:
            stacked[key] = torch.stack([item[key] for item in outputs], dim=1)
        stacked["latents"] = torch.stack(latents, dim=1)
        return stacked


def save_checkpoint(
    path: str | Path,
    model: RecurrentLatentModel,
    train_config: Dict[str, object],
    metrics: Dict[str, float] | None = None,
) -> None:
    payload = {
        "format": "recurrent_latent_base_checkpoint_v2",
        "model_config": asdict(model.config),
        "model_state": model.state_dict(),
        "train_config": train_config,
        "metrics": metrics or {},
        "base_weights": base_checkpoint_contract(model),
        "fast_adaptation": fast_adaptation_contract(),
    }
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, device: DeviceLike = AUTO_DEVICE) -> RecurrentLatentModel:
    target_device = resolve_device(device)
    payload = torch.load(Path(path), map_location=target_device)
    config = ModelConfig(**payload["model_config"])
    model = RecurrentLatentModel(config, device=target_device)
    result = model.load_state_dict(payload["model_state"], strict=False)
    allowed_missing = {
        "action_impulse_embedding.weight",
        "delta_impulse_encoder.weight",
        "delta_impulse_encoder.bias",
        "liquid_time_constant.weight",
        "liquid_time_constant.bias",
        "liquid_drift.0.weight",
        "liquid_drift.0.bias",
        "liquid_drift.2.weight",
        "liquid_drift.2.bias",
        "program_family_head.weight",
        "program_family_head.bias",
        "program_field_head.weight",
        "program_field_head.bias",
        "program_transform_head.weight",
        "program_transform_head.bias",
        "program_color_head.weight",
        "program_color_head.bias",
    }
    allowed_missing_prefixes = ("object_relation_encoder.",)
    allowed_missing_prefixes = (*allowed_missing_prefixes, "action_context_head.")
    unexpected = list(result.unexpected_keys)
    missing = [
        key
        for key in result.missing_keys
        if key not in allowed_missing and not any(key.startswith(prefix) for prefix in allowed_missing_prefixes)
    ]
    if unexpected or missing:
        raise RuntimeError(
            f"checkpoint schema mismatch: missing={missing}, unexpected={unexpected}"
        )
    model.eval()
    return model


def _translate_transform(label: str) -> dict[str, object]:
    if label == "translate_up":
        return {"kind": "translate", "dy": -1.0, "dx": 0.0}
    if label == "translate_down":
        return {"kind": "translate", "dy": 1.0, "dx": 0.0}
    if label == "translate_left":
        return {"kind": "translate", "dy": 0.0, "dx": -1.0}
    if label == "translate_right":
        return {"kind": "translate", "dy": 0.0, "dx": 1.0}
    return {"kind": "translate", "dy": 0.0, "dx": 0.0}


def _first_logit_row(logits: torch.Tensor) -> torch.Tensor:
    tensor = logits.detach().float()
    if tensor.ndim == 0:
        return tensor.view(1)
    if tensor.ndim == 1:
        return tensor.reshape(-1)
    return tensor.reshape(-1, tensor.shape[-1])[0]


def decode_program_proposals(
    output: Dict[str, torch.Tensor],
    *,
    action: int | str,
    top_k: int = 3,
) -> list[dict[str, object]]:
    required = {
        "program_family_logits",
        "program_field_logits",
        "program_transform_logits",
        "program_color_logits",
    }
    if not required.issubset(output):
        return []

    family_logits = _first_logit_row(output["program_family_logits"])
    field_logits = _first_logit_row(output["program_field_logits"])
    transform_logits = _first_logit_row(output["program_transform_logits"])
    color_logits = _first_logit_row(output["program_color_logits"])
    if (
        family_logits.numel() != len(PROGRAM_FAMILIES)
        or field_logits.numel() != len(PROGRAM_FIELDS)
        or transform_logits.numel() != len(PROGRAM_TRANSFORMS)
        or color_logits.numel() != NUM_COLORS
    ):
        return []

    family_probs = torch.softmax(family_logits, dim=-1)
    field_probs = torch.softmax(field_logits, dim=-1)
    transform_probs = torch.softmax(transform_logits, dim=-1)
    color_probs = torch.softmax(color_logits, dim=-1)
    family_count = min(max(int(top_k), 1), len(PROGRAM_FAMILIES))
    family_indices = torch.topk(family_probs, k=family_count).indices.tolist()
    field_index = int(torch.argmax(field_probs).item())
    transform_index = int(torch.argmax(transform_probs).item())
    color_index = int(torch.argmax(color_probs).item())
    field = PROGRAM_FIELDS[field_index]
    transform_label = PROGRAM_TRANSFORMS[transform_index]
    proposals: list[dict[str, object]] = []
    for family_index in family_indices:
        family = PROGRAM_FAMILIES[int(family_index)]
        confidence = float(
            family_probs[family_index].item()
            * max(
                float(field_probs[field_index].item()),
                float(transform_probs[transform_index].item()),
                float(color_probs[color_index].item()),
            )
        )
        if family == "no_op":
            selector = {"scope": "observation"}
            transform = {"kind": "identity"}
            goal_test = {"kind": "no_visible_change"}
        elif family == "field_change":
            selector = {"field": field}
            transform = {"kind": "change", "source": "neural_program_head"}
            goal_test = {"kind": "field_changed"}
        elif family == "field_stable":
            selector = {"field": field}
            transform = {"kind": "identity"}
            goal_test = {"kind": "field_unchanged"}
        else:
            selector = {"color": color_index}
            transform = _translate_transform(transform_label)
            goal_test = {"kind": "component_translation"}
        proposals.append(
            {
                "schema": "neural_causal_program_proposal_v1",
                "source": "neural_proposal",
                "family": family,
                "action": str(action),
                "selector": selector,
                "transform": transform,
                "goal_test": goal_test,
                "confidence": confidence,
                "family_probability": float(family_probs[family_index].item()),
                "field_probability": float(field_probs[field_index].item()),
                "transform_probability": float(transform_probs[transform_index].item()),
                "color_probability": float(color_probs[color_index].item()),
                "family_index": int(family_index),
                "field_index": field_index,
                "transform_index": transform_index,
                "color_index": color_index,
            }
        )
    return proposals
