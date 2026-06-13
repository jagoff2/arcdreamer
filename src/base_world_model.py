from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch import nn

from .arcagi3_adapter import (
    CELL_AGENT,
    CELL_DOOR,
    CELL_GOAL,
    CELL_HAZARD,
    CELL_KEY,
    CELL_RESOURCE,
    CELL_UNKNOWN,
    CELL_WALL,
    ArcAGI3Observation,
    MOVE_DELTAS,
    find_agent,
    in_bounds,
    parse_click,
)
from .device import AUTO_DEVICE, DeviceLike, resolve_device


GRID_SIZE = 8
NUM_CELLS = 9
ACTION_FEATURE_DIM = 16
FAMILY_TO_INDEX = {"move": 0, "click": 1, "wait": 2, "reset": 3, "other": 4}
POOL_PRIORITY = np.zeros(NUM_CELLS, dtype=np.int64)
for _value, _priority in {
    CELL_WALL: 1,
    CELL_UNKNOWN: 1,
    CELL_RESOURCE: 2,
    CELL_KEY: 3,
    CELL_DOOR: 3,
    CELL_GOAL: 4,
    CELL_HAZARD: 4,
    CELL_AGENT: 5,
}.items():
    if 0 <= int(_value) < len(POOL_PRIORITY):
        POOL_PRIORITY[int(_value)] = int(_priority)


@dataclass
class ExternalBaseConfig:
    grid_size: int = GRID_SIZE
    action_feature_dim: int = ACTION_FEATURE_DIM
    hidden_dim: int = 96
    latent_dim: int = 64
    max_cell_value: int = NUM_CELLS - 1


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def grid_to_fixed(grid: np.ndarray, size: int = GRID_SIZE) -> np.ndarray:
    arr = np.asarray(grid, dtype=np.int64)
    out = np.zeros((size, size), dtype=np.int64)
    if arr.ndim != 2 or arr.size == 0:
        return out
    clipped = np.clip(arr, 0, NUM_CELLS - 1)
    if clipped.shape[0] <= size and clipped.shape[1] <= size:
        height = min(size, clipped.shape[0])
        width = min(size, clipped.shape[1])
        out[:height, :width] = clipped[:height, :width]
        return out
    row_bins = np.array_split(np.arange(clipped.shape[0]), size)
    col_bins = np.array_split(np.arange(clipped.shape[1]), size)
    for row_index, rows in enumerate(row_bins):
        if rows.size == 0:
            continue
        for col_index, cols in enumerate(col_bins):
            if cols.size == 0:
                continue
            block = clipped[np.ix_(rows, cols)].reshape(-1)
            if block.size == 0:
                continue
            non_empty = block[block != 0]
            if non_empty.size == 0:
                continue
            values, counts = np.unique(non_empty, return_counts=True)
            priorities = POOL_PRIORITY[values]
            order = np.lexsort((values, counts, priorities))
            out[row_index, col_index] = int(values[order[-1]])
    return out


def grid_to_features(grid: np.ndarray, size: int = GRID_SIZE) -> np.ndarray:
    return grid_to_fixed(grid, size).astype(np.float32).reshape(-1) / float(NUM_CELLS - 1)


def action_family(action: str) -> str:
    if action in MOVE_DELTAS:
        return "move"
    if action.startswith("click:"):
        return "click"
    if action in {"7", "wait", "noop"}:
        return "wait"
    if action == "0":
        return "reset"
    return "other"


def family_index(action: str) -> int:
    return FAMILY_TO_INDEX[action_family(action)]


def _hash_features(action: str, width: int = 6) -> list[float]:
    digest = hashlib.sha256(action.encode("utf-8")).digest()
    return [((digest[index] / 255.0) * 2.0) - 1.0 for index in range(width)]


def action_hash_features(action: str, width: int = 6) -> np.ndarray:
    return np.asarray(_hash_features(str(action), width=width), dtype=np.float32)


def action_to_features(
    observation: ArcAGI3Observation,
    action: str,
    *,
    legal_index: int = 0,
    legal_count: int | None = None,
    size: int = GRID_SIZE,
) -> np.ndarray:
    family = action_family(action)
    family_vec = [0.0] * len(FAMILY_TO_INDEX)
    family_vec[FAMILY_TO_INDEX[family]] = 1.0
    grid = np.asarray(observation.grid, dtype=np.int64)
    height = int(grid.shape[0]) if grid.ndim >= 2 else int(size)
    width = int(grid.shape[1]) if grid.ndim >= 2 else int(size)
    target_y = 0.0
    target_x = 0.0
    click = parse_click(action)
    if click is not None:
        x, y = click
        target_y = float(np.clip(y / max(height - 1, 1), 0.0, 1.0))
        target_x = float(np.clip(x / max(width - 1, 1), 0.0, 1.0))
    elif action in MOVE_DELTAS:
        agent = find_agent(grid)
        if agent is not None:
            dy, dx = MOVE_DELTAS[action]
            y, x = agent[0] + dy, agent[1] + dx
            target_y = float(np.clip(y / max(height - 1, 1), 0.0, 1.0))
            target_x = float(np.clip(x / max(width - 1, 1), 0.0, 1.0))
    legal_total = max(int(legal_count if legal_count is not None else len(observation.available_actions)), 1)
    legal_features = [float(legal_index / max(legal_total - 1, 1)), float(legal_total / 256.0), 1.0]
    features = family_vec + [target_y, target_x] + legal_features + _hash_features(action)
    return np.asarray(features[:ACTION_FEATURE_DIM], dtype=np.float32)


def fixed_action_features(action_id: int, size: int = GRID_SIZE) -> np.ndarray:
    actions = ("1", "2", "3", "4", "click:0:0", "click:0:7", "click:7:0", "click:7:7")
    action = actions[int(action_id) % len(actions)]
    dummy = ArcAGI3Observation(
        task_id="generated/arc_like",
        episode_id="generated",
        step_index=0,
        grid=np.zeros((size, size), dtype=np.int64),
        available_actions=actions,
        extras={},
    )
    return action_to_features(dummy, action, legal_index=int(action_id) % len(actions), legal_count=len(actions), size=size)


class ExternalBaseWorldModel(nn.Module):
    def __init__(self, config: ExternalBaseConfig | None = None, device: DeviceLike = AUTO_DEVICE) -> None:
        super().__init__()
        self.config = config or ExternalBaseConfig()
        obs_dim = self.config.grid_size * self.config.grid_size
        self.obs_encoder = nn.Sequential(
            nn.Linear(obs_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_dim),
        )
        self.action_encoder = nn.Sequential(
            nn.Linear(self.config.action_feature_dim, self.config.hidden_dim // 2),
            nn.GELU(),
        )
        self.memory_cell = nn.GRUCell(self.config.hidden_dim + self.config.hidden_dim // 2, self.config.latent_dim)
        self.joint = nn.Sequential(
            nn.Linear(self.config.hidden_dim + self.config.hidden_dim // 2 + self.config.latent_dim, self.config.hidden_dim),
            nn.GELU(),
            nn.LayerNorm(self.config.hidden_dim),
        )
        obs_dim = self.config.grid_size * self.config.grid_size
        self.next_obs_head = nn.Linear(self.config.hidden_dim, obs_dim)
        self.change_head = nn.Linear(self.config.hidden_dim, obs_dim)
        self.reward_head = nn.Linear(self.config.hidden_dim, 1)
        self.noop_head = nn.Linear(self.config.hidden_dim, 1)
        self.inverse_head = nn.Linear(self.config.hidden_dim, len(FAMILY_TO_INDEX))
        self.affordance_head = nn.Linear(self.config.hidden_dim, 1)
        self.rollout_head = nn.Linear(self.config.latent_dim, self.config.latent_dim)
        self.memory_obs_head = nn.Linear(self.config.latent_dim, obs_dim)
        self.to(resolve_device(device))

    def initial_memory(self, batch_size: int, device: DeviceLike | None = None) -> torch.Tensor:
        target_device = next(self.parameters()).device if device is None else resolve_device(device)
        return torch.zeros(batch_size, self.config.latent_dim, device=target_device)

    def encode(self, obs: torch.Tensor, action_features: torch.Tensor, memory: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        if memory is None:
            memory = self.initial_memory(obs.shape[0], obs.device)
        obs_state = self.obs_encoder(obs)
        action_state = self.action_encoder(action_features)
        memory_next = self.memory_cell(torch.cat([obs_state, action_state], dim=-1), memory)
        state = self.joint(torch.cat([obs_state, action_state, memory_next], dim=-1))
        return {"state": state, "memory": memory_next, "obs_state": obs_state, "action_state": action_state}

    def forward(self, obs: torch.Tensor, action_features: torch.Tensor, memory: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        encoded = self.encode(obs, action_features, memory)
        state = encoded["state"]
        memory_next = encoded["memory"]
        return {
            "next_obs": torch.sigmoid(self.next_obs_head(state)),
            "change_logits": self.change_head(state),
            "reward": self.reward_head(state).squeeze(-1),
            "noop_logits": self.noop_head(state).squeeze(-1),
            "inverse_logits": self.inverse_head(state),
            "affordance": self.affordance_head(state).squeeze(-1),
            "rollout_latent": self.rollout_head(memory_next),
            "memory_next_obs": torch.sigmoid(self.memory_obs_head(memory_next)),
            "memory": memory_next,
        }

    @torch.no_grad()
    def score_actions(
        self,
        observation: ArcAGI3Observation,
        legal_actions: tuple[str, ...] | list[str],
        *,
        memory: torch.Tensor | None = None,
        disable_memory: bool = False,
        disable_world_model: bool = False,
        disable_affordance: bool = False,
    ) -> tuple[dict[str, float], dict[str, Any], torch.Tensor]:
        device = next(self.parameters()).device
        actions = tuple(legal_actions)
        obs_np = np.stack([grid_to_features(observation.grid, self.config.grid_size) for _ in actions], axis=0)
        feat_np = np.stack(
            [
                action_to_features(observation, action, legal_index=index, legal_count=len(actions), size=self.config.grid_size)
                for index, action in enumerate(actions)
            ],
            axis=0,
        )
        obs = torch.as_tensor(obs_np, dtype=torch.float32, device=device)
        action_features = torch.as_tensor(feat_np, dtype=torch.float32, device=device)
        base_memory = memory if memory is not None else self.initial_memory(len(actions), device=device)
        if base_memory.shape[0] == 1 and len(actions) > 1:
            base_memory = base_memory.repeat(len(actions), 1)
        if disable_memory:
            base_memory = torch.zeros_like(base_memory)
        output = self(obs, action_features, base_memory)
        change = torch.sigmoid(output["change_logits"]).mean(dim=-1)
        reward = output["reward"].tanh()
        noop = torch.sigmoid(output["noop_logits"])
        affordance = torch.sigmoid(output["affordance"])
        scores = {}
        diagnostics = {}
        for index, action in enumerate(actions):
            world_score = (
                float((1.6 * reward[index] + 0.35 * change[index] - 0.45 * noop[index]).item())
                if not disable_world_model
                else 0.0
            )
            affordance_score = float(0.35 * affordance[index].item()) if not disable_affordance else 0.0
            scores[action] = world_score + affordance_score
            diagnostics[action] = {
                "reward": round(float(reward[index].item()), 6),
                "change": round(float(change[index].item()), 6),
                "noop": round(float(noop[index].item()), 6),
                "affordance": round(float(affordance[index].item()), 6),
                "score": round(float(scores[action]), 6),
            }
        next_memory = output["memory"].detach()
        return scores, diagnostics, next_memory


def training_targets(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    obs = batch["obs"]
    next_obs = batch["next_obs"]
    change_mask = (torch.abs(next_obs - obs) > 1.0e-6).float()
    noop = (change_mask.mean(dim=-1) <= 1.0e-6).float()
    reward = torch.clamp(batch["reward"], -1.0, 1.0)
    affordance = torch.clamp(change_mask.mean(dim=-1) + torch.relu(reward), 0.0, 2.0) / 2.0
    return {
        "next_obs": next_obs,
        "change_mask": change_mask,
        "noop": noop,
        "reward": reward,
        "family": batch["family"].long(),
        "affordance": affordance,
    }


def save_external_base_checkpoint(
    *,
    recurrent_checkpoint: str | Path,
    output_path: str | Path,
    external_config: ExternalBaseConfig,
    arm_states: dict[str, dict[str, Any]],
    manifest: dict[str, Any],
    metrics: dict[str, Any],
) -> None:
    payload = torch.load(Path(recurrent_checkpoint), map_location="cpu")
    payload = dict(payload)
    payload["external_base_format"] = "external_action_conditioned_world_model_v1"
    payload["external_config"] = asdict(external_config)
    payload["external_arms"] = arm_states
    payload["external_manifest"] = manifest
    payload["metrics"] = {**dict(payload.get("metrics", {})), "external_base": metrics}
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output)


def load_external_base_payload(path: str | Path, device: DeviceLike = AUTO_DEVICE) -> dict[str, Any]:
    target_device = resolve_device(device)
    payload = torch.load(Path(path), map_location=target_device)
    if payload.get("external_base_format") != "external_action_conditioned_world_model_v1":
        raise ValueError(f"{path} is not an external base checkpoint")
    return payload


def load_external_arm(path: str | Path, arm: str, device: DeviceLike = AUTO_DEVICE) -> ExternalBaseWorldModel:
    payload = load_external_base_payload(path, device=device)
    config = ExternalBaseConfig(**payload["external_config"])
    model = ExternalBaseWorldModel(config, device=device)
    state = payload["external_arms"][arm]["state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def write_external_manifest(path: str | Path, payload: dict[str, Any]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, indent=2), encoding="utf-8")
