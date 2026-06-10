from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .arcagi3_adapter import ArcAGI3Observation
from .arcagi3_baselines import build_baselines
from .device import AUTO_DEVICE, DeviceLike
from .external_collapse_experiment import make_adapter
from .external_eval import GymnasiumExternalEnv
from .external_registry import discover_external_suites
from .base_world_model import GRID_SIZE, action_to_features, family_index, grid_to_features, sha256_file


ACTION_DIM = 8
SOURCE_TO_ID = {"generated_arc_like_pretrain": 0, "gymnasium_dev_policy": 1, "old_explorer_dev": 2}


@dataclass
class TraceArrays:
    obs: list[np.ndarray]
    next_obs: list[np.ndarray]
    action_features: list[np.ndarray]
    action_id: list[int]
    family: list[int]
    legal_mask: list[np.ndarray]
    reward: list[float]
    terminal: list[bool]
    source_id: list[int]

    @classmethod
    def empty(cls) -> "TraceArrays":
        return cls([], [], [], [], [], [], [], [], [])

    def append(
        self,
        *,
        before: ArcAGI3Observation,
        action: str,
        action_index: int,
        after: ArcAGI3Observation,
        reward: float,
        terminal: bool,
        source: str,
    ) -> None:
        legal = np.zeros((ACTION_DIM,), dtype=np.float32)
        legal[: min(len(before.available_actions), ACTION_DIM)] = 1.0
        self.obs.append(grid_to_features(before.grid))
        self.next_obs.append(grid_to_features(after.grid))
        self.action_features.append(
            action_to_features(before, action, legal_index=action_index, legal_count=len(before.available_actions))
        )
        self.action_id.append(int(action_index % ACTION_DIM))
        self.family.append(family_index(action))
        self.legal_mask.append(legal)
        self.reward.append(float(reward))
        self.terminal.append(bool(terminal))
        self.source_id.append(SOURCE_TO_ID[source])

    def as_np(self) -> dict[str, np.ndarray]:
        if not self.obs:
            return {}
        return {
            "obs": np.asarray(self.obs, dtype=np.float32),
            "next_obs": np.asarray(self.next_obs, dtype=np.float32),
            "action_features": np.asarray(self.action_features, dtype=np.float32),
            "action_id": np.asarray(self.action_id, dtype=np.int64),
            "family": np.asarray(self.family, dtype=np.int64),
            "legal_mask": np.asarray(self.legal_mask, dtype=np.float32),
            "reward": np.asarray(self.reward, dtype=np.float32),
            "terminal": np.asarray(self.terminal, dtype=np.bool_),
            "source_id": np.asarray(self.source_id, dtype=np.int64),
        }


def generated_arc_like_arrays(count: int, *, seed: int = 7001, size: int = GRID_SIZE) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    obs_grid = np.zeros((count, size, size), dtype=np.uint8)
    next_grid = np.zeros_like(obs_grid)
    agent_y = rng.integers(0, size, size=count)
    agent_x = rng.integers(0, size, size=count)
    object_y = rng.integers(0, size, size=count)
    object_x = rng.integers(0, size, size=count)
    same = (agent_y == object_y) & (agent_x == object_x)
    object_x[same] = (object_x[same] + 1) % size
    obs_grid[np.arange(count), agent_y, agent_x] = 2
    obs_grid[np.arange(count), object_y, object_x] = 7
    next_grid[:] = obs_grid
    action_id = rng.integers(0, ACTION_DIM, size=count)
    reward = np.zeros((count,), dtype=np.float32)
    terminal = np.zeros((count,), dtype=np.bool_)
    family = np.zeros((count,), dtype=np.int64)
    action_features = np.zeros((count, 16), dtype=np.float32)
    for action in range(ACTION_DIM):
        idx = np.where(action_id == action)[0]
        if idx.size == 0:
            continue
        features = action_features[idx]
        features[:, 10:] = _generated_hash_features(action)
        features[:, 7] = float(action) / float(ACTION_DIM - 1)
        features[:, 8] = ACTION_DIM / 256.0
        features[:, 9] = 1.0
        if action < 4:
            family[idx] = 0
            features[:, 0] = 1.0
            dy, dx = [(-1, 0), (1, 0), (0, -1), (0, 1)][action]
            ny = np.clip(agent_y[idx] + dy, 0, size - 1)
            nx = np.clip(agent_x[idx] + dx, 0, size - 1)
            features[:, 5] = ny / max(size - 1, 1)
            features[:, 6] = nx / max(size - 1, 1)
            old_dist = np.abs(agent_y[idx] - object_y[idx]) + np.abs(agent_x[idx] - object_x[idx])
            new_dist = np.abs(ny - object_y[idx]) + np.abs(nx - object_x[idx])
            next_grid[idx, agent_y[idx], agent_x[idx]] = 0
            next_grid[idx, ny, nx] = 2
            reward[idx] = (new_dist < old_dist).astype(np.float32) * 0.05
        elif action == 4:
            family[idx] = 1
            features[:, 1] = 1.0
            features[:, 5] = object_y[idx] / max(size - 1, 1)
            features[:, 6] = object_x[idx] / max(size - 1, 1)
            next_grid[idx, object_y[idx], object_x[idx]] = 0
            reward[idx] = 1.0
            terminal[idx] = True
        elif action == 5:
            family[idx] = 1
            features[:, 1] = 1.0
            ty = rng.integers(0, size, size=idx.size)
            tx = rng.integers(0, size, size=idx.size)
            features[:, 5] = ty / max(size - 1, 1)
            features[:, 6] = tx / max(size - 1, 1)
            hit = (ty == object_y[idx]) & (tx == object_x[idx])
            next_grid[idx[hit], object_y[idx[hit]], object_x[idx[hit]]] = 0
            reward[idx[hit]] = 1.0
            terminal[idx[hit]] = True
        elif action == 6:
            family[idx] = 2
            features[:, 2] = 1.0
            reward[idx] = -0.01
        else:
            family[idx] = 3
            features[:, 3] = 1.0
        action_features[idx] = features
    legal_mask = np.ones((count, ACTION_DIM), dtype=np.float32)
    return {
        "obs": obs_grid.reshape(count, -1).astype(np.float32) / 8.0,
        "next_obs": next_grid.reshape(count, -1).astype(np.float32) / 8.0,
        "action_features": action_features,
        "action_id": action_id.astype(np.int64),
        "family": family.astype(np.int64),
        "legal_mask": legal_mask,
        "reward": reward,
        "terminal": terminal,
        "source_id": np.full((count,), SOURCE_TO_ID["generated_arc_like_pretrain"], dtype=np.int64),
    }


def _generated_hash_features(action: int) -> np.ndarray:
    rng = np.random.default_rng(9100 + int(action))
    return rng.uniform(-1.0, 1.0, size=(6,)).astype(np.float32)


class AdapterPolicy:
    name = "old_explorer"

    def __init__(self, checkpoint: str | Path, device: DeviceLike) -> None:
        self.adapter = make_adapter(checkpoint, device)

    def reset(self, seed: int) -> None:
        del seed
        self.adapter.reset()

    def choose(self, observation: ArcAGI3Observation) -> str:
        action, _ = self.adapter.choose_action(observation)
        return action

    def observe(self, before: ArcAGI3Observation, action: str, result: Any) -> None:
        del before
        self.adapter.observe_transition(action, result)


def collect_external_dev_arrays(
    *,
    seeds: list[int],
    explorer_checkpoint: str | Path,
    device: DeviceLike,
    sample_path: str | Path,
    sample_limit: int = 256,
) -> tuple[dict[str, np.ndarray], list[dict[str, Any]], list[dict[str, Any]]]:
    arrays = TraceArrays.empty()
    sample_output = Path(sample_path)
    sample_output.parent.mkdir(parents=True, exist_ok=True)
    samples_written = 0
    sources: list[dict[str, Any]] = []
    suites = [suite for suite in discover_external_suites() if suite.available and suite.suite_id.startswith("gymnasium_")]
    with sample_output.open("w", encoding="utf-8") as sample_file:
        for suite in suites:
            for task_id in suite.tasks:
                policies = [*build_baselines(), AdapterPolicy(explorer_checkpoint, device)]
                for policy in policies:
                    source_count = 0
                    for seed in seeds:
                        env = GymnasiumExternalEnv(
                            suite.suite_id,
                            task_id,
                            max_steps=80 if task_id == "CartPole-v1" else 32,
                        )
                        policy.reset(seed)
                        obs = env.reset(seed)
                        try:
                            for _ in range(env.max_steps):
                                before = obs
                                action = policy.choose(before)
                                action_index = list(before.available_actions).index(action) if action in before.available_actions else 0
                                result = env.step(action)
                                terminal = bool(result.terminated or result.truncated)
                                arrays.append(
                                    before=before,
                                    action=action,
                                    action_index=action_index,
                                    after=result.observation,
                                    reward=float(result.reward),
                                    terminal=terminal,
                                    source="old_explorer_dev" if policy.name == "old_explorer" else "gymnasium_dev_policy",
                                )
                                source_count += 1
                                if samples_written < sample_limit:
                                    sample_file.write(json.dumps(sample_record(before, action, result, policy.name)) + "\n")
                                    samples_written += 1
                                policy.observe(before, action, result)
                                obs = result.observation
                                if terminal:
                                    break
                        finally:
                            env.close()
                    sources.append(
                        {
                            "source": "old_explorer_dev" if policy.name == "old_explorer" else "gymnasium_dev_policy",
                            "suite_id": suite.suite_id,
                            "task_id": task_id,
                            "policy": policy.name,
                            "split": "dev",
                            "transition_count": source_count,
                        }
                    )
    return arrays.as_np(), sources, [{"path": str(sample_output), "sha256": sha256_file(sample_output)}]


def sample_record(before: ArcAGI3Observation, action: str, result: Any, policy: str) -> dict[str, Any]:
    return {
        "policy": policy,
        "obs_t": np.asarray(before.grid, dtype=np.int64).tolist(),
        "legal_actions_t": list(before.available_actions),
        "action_t": action,
        "obs_t_plus_1": np.asarray(result.observation.grid, dtype=np.int64).tolist(),
        "reward_delta": float(result.reward),
        "score_delta": float(result.info.get("score", result.reward)),
        "terminal": bool(result.terminated or result.truncated),
        "info_events_if_public": list(result.info.get("events", [])),
    }


def concatenate_arrays(parts: list[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    keys = [key for key in parts[0] if key in parts[0]]
    return {key: np.concatenate([part[key] for part in parts if key in part and part[key].size], axis=0) for key in keys}


def build_manifest(
    *,
    output: str | Path,
    data_path: str | Path,
    sample_files: list[dict[str, Any]],
    source_rows: list[dict[str, Any]],
    arrays: dict[str, np.ndarray],
    generated_count: int,
) -> dict[str, Any]:
    output_path = Path(output)
    data = {
        "format": "external_interaction_trace_manifest_v1",
        "schema": {
            "obs": "float32 fixed 8x8 visible grid, flattened and scaled, from obs_t",
            "legal_mask": "float32 legal action mask over the compact training action set",
            "action_features": "float32 generic action family/target/hash features from action_t",
            "action_id": "int compact action index used only for inverse-dynamics diagnostics",
            "family": "int generic action family target for inverse dynamics",
            "next_obs": "float32 fixed 8x8 visible grid, flattened and scaled, from obs_t+1",
            "reward": "float32 public reward/score delta clipped during training",
            "terminal": "bool public terminal/truncated flag",
            "source_id": SOURCE_TO_ID,
        },
        "rules": {
            "private_target_state_absent": True,
            "no_solver_labels": True,
            "no_game_id_features": True,
            "no_public_text_as_state": True,
            "sealed_official_arcagi3_used_for_training": False,
        },
        "data_path": str(data_path),
        "data_sha256": sha256_file(data_path),
        "sample_records": sample_files,
        "source_summary": source_rows,
        "generated_arc_like_pretrain_count": generated_count,
        "transition_count": int(arrays["obs"].shape[0]),
        "array_shapes": {key: list(value.shape) for key, value in arrays.items()},
        "source_counts": {
            name: int((arrays["source_id"] == source_id).sum())
            for name, source_id in SOURCE_TO_ID.items()
        },
    }
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    data["manifest_sha256"] = sha256_file(output_path)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return data


def collect_traces(
    *,
    output: str | Path,
    data_output: str | Path,
    sample_output: str | Path,
    generated_count: int,
    external_seeds: list[int],
    explorer_checkpoint: str | Path,
    device: DeviceLike,
) -> dict[str, Any]:
    generated = generated_arc_like_arrays(generated_count)
    external, source_rows, sample_files = collect_external_dev_arrays(
        seeds=external_seeds,
        explorer_checkpoint=explorer_checkpoint,
        device=device,
        sample_path=sample_output,
    )
    arrays = concatenate_arrays([generated, external])
    data_path = Path(data_output)
    data_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(data_path, **arrays)
    source_rows.insert(
        0,
        {
            "source": "generated_arc_like_pretrain",
            "split": "pretrain_only",
            "policy": "random_synthetic_dynamics",
            "transition_count": generated_count,
        },
    )
    return build_manifest(
        output=output,
        data_path=data_path,
        sample_files=sample_files,
        source_rows=source_rows,
        arrays=arrays,
        generated_count=generated_count,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["external_dev"], default="external_dev")
    parser.add_argument("--output", default="data/external_traces_manifest.json")
    parser.add_argument("--data-output", default="data/external_trace_tensors.npz")
    parser.add_argument("--sample-output", default="data/external_trace_samples.jsonl")
    parser.add_argument("--generated-count", type=int, default=1_000_000)
    parser.add_argument("--external-seeds", default="0,1,2")
    parser.add_argument("--explorer-checkpoint", default="runs/explorer_tiny.pt")
    parser.add_argument("--device", default=AUTO_DEVICE)
    args = parser.parse_args()
    del args.config
    seeds = [int(item) for item in args.external_seeds.split(",") if item.strip()]
    manifest = collect_traces(
        output=args.output,
        data_output=args.data_output,
        sample_output=args.sample_output,
        generated_count=args.generated_count,
        external_seeds=seeds,
        explorer_checkpoint=args.explorer_checkpoint,
        device=args.device,
    )
    print(json.dumps({"manifest": args.output, "transition_count": manifest["transition_count"], "source_counts": manifest["source_counts"]}, indent=2))


if __name__ == "__main__":
    main()
