from __future__ import annotations

import argparse
import json
import random
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch

from .arcagi3_adapter import ArcAGI3Observation
from .attempt_buffer import action_id, legal_action_mask
from .base_pretrain import (
    evaluate_losses,
    memory_utility_diagnostic,
    objective_losses,
    objective_losses_from_output,
    sample_batch,
    set_seed,
    shuffle_targets,
    train_arm,
    weighted_loss,
)
from .base_world_model import (
    ACTION_FEATURE_DIM,
    ExternalBaseConfig,
    action_to_features,
    family_index,
    grid_to_features,
    save_external_base_checkpoint,
    sha256_file,
    write_external_manifest,
)
from .device import AUTO_DEVICE, DeviceLike, resolve_device


DEFAULT_HOLDOUT_GAME_IDS = (
    "ar25-0c556536",
    "lf52-271a04aa",
    "sk48-d8078629",
    "vc33-5430563c",
    "wa30-ee6fef47",
)
REAL_ARC_SOURCE_ID = 10
POSITIVE_REAL_ARC_EVENTS = {
    "positive_reward",
    "resource_collected",
    "level_completed",
    "goal_reached",
    "goal_clicked",
    "useful_click",
}
TRAINED_ARMS = (
    "old_base_world_model",
    "old_base_finetuned",
    "from_scratch_external_base",
    "null_training_control",
)


def load_manifest_game_ids(path: str | Path) -> list[str]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return [str(item["game_id"]) for item in payload.get("games", [])]


def parse_csv(value: str | Iterable[str]) -> list[str]:
    if isinstance(value, str):
        parts = value.split(",")
    else:
        parts = list(value)
    return [str(item).strip() for item in parts if str(item).strip()]


def split_game_ids(
    all_game_ids: list[str],
    *,
    train_game_ids: Iterable[str] | None,
    holdout_game_ids: Iterable[str],
) -> tuple[list[str], list[str]]:
    known = set(all_game_ids)
    requested_train = [item for item in (train_game_ids or []) if item in known]
    requested_holdout = [str(item) for item in holdout_game_ids]
    if requested_train:
        if any(item.lower() == "all_except_train" for item in requested_holdout) or not requested_holdout:
            holdout = [item for item in all_game_ids if item not in set(requested_train)]
        else:
            holdout = [item for item in requested_holdout if item in known and item not in set(requested_train)]
        return requested_train, holdout
    holdout = [item for item in requested_holdout if item in known]
    if not holdout:
        raise ValueError("holdout split must contain at least one known game id")
    return [item for item in all_game_ids if item not in set(holdout)], holdout


def trace_game_id(payload: dict[str, Any], path: Path) -> str:
    meta = payload.get("metadata", {}) if isinstance(payload, dict) else {}
    attempt = payload.get("attempt", {}) if isinstance(payload, dict) else {}
    for source in (meta, attempt, payload.get("summary", {}) if isinstance(payload, dict) else {}):
        for key in ("game_id", "task_id"):
            value = str(source.get(key, "") or "")
            if value:
                return value.split("/")[-1]
    return path.stem.split(".")[0]


def trace_attempt_payload(payload: dict[str, Any]) -> dict[str, Any] | None:
    attempt = payload.get("attempt") if isinstance(payload, dict) else None
    if isinstance(attempt, dict) and isinstance(attempt.get("steps"), list):
        return attempt
    if isinstance(payload, dict) and isinstance(payload.get("steps"), list):
        return payload
    return None


def usable_real_arc_trace(path: Path, allowed_game_ids: set[str]) -> tuple[str, dict[str, Any]] | None:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None
    attempt = trace_attempt_payload(payload)
    if attempt is None:
        return None
    suite = str(attempt.get("suite_id", payload.get("metadata", {}).get("suite_id", "")))
    if suite != "official_arcagi3":
        return None
    game_id = trace_game_id(payload, path)
    if allowed_game_ids and game_id not in allowed_game_ids:
        return None
    steps = attempt.get("steps") or []
    if not steps or not isinstance(steps[0], dict):
        return None
    first = steps[0]
    if "frame" not in first or "next_frame" not in first or "legal_actions" not in first:
        return None
    return game_id, attempt


def index_real_arc_traces(
    *,
    roots: Iterable[str | Path],
    game_ids: Iterable[str],
    max_traces_per_game: int,
) -> dict[str, list[tuple[Path, dict[str, Any]]]]:
    allowed = {str(item) for item in game_ids}
    indexed: dict[str, list[tuple[Path, dict[str, Any]]]] = defaultdict(list)
    max_per_game = int(max_traces_per_game)
    unlimited = max_per_game <= 0
    for root in roots:
        root_path = Path(root)
        if not root_path.exists():
            continue
        for path in sorted(root_path.rglob("*.json")):
            if "official_arcagi3" not in path.as_posix().replace("\\", "/"):
                continue
            parsed = usable_real_arc_trace(path, allowed)
            if parsed is None:
                continue
            game_id, attempt = parsed
            if not unlimited and len(indexed[game_id]) >= max_per_game:
                continue
            indexed[game_id].append((path, attempt))
    return dict(indexed)


def empty_arrays() -> dict[str, list[Any]]:
    return {
        "obs": [],
        "next_obs": [],
        "action_features": [],
        "action_id": [],
        "family": [],
        "legal_mask": [],
        "reward": [],
        "future_progress": [],
        "terminal": [],
        "source_id": [],
        "trace_id": [],
        "trace_step": [],
    }


def append_step(
    arrays: dict[str, list[Any]],
    *,
    task_id: str,
    step: dict[str, Any],
    future_progress: float = 0.0,
    trace_id: int = 0,
    trace_step: int = 0,
) -> bool:
    next_frame = step.get("next_frame")
    if next_frame is None:
        return False
    legal_actions = [str(item) for item in step.get("legal_actions", [])]
    action = str(step.get("action", step.get("chosen_action", "")))
    if not legal_actions or not action:
        return False
    frame = np.asarray(step.get("frame"), dtype=np.int64)
    next_arr = np.asarray(next_frame, dtype=np.int64)
    if frame.ndim != 2 or next_arr.ndim != 2:
        return False
    try:
        legal_index = legal_actions.index(action)
    except ValueError:
        legal_index = 0
    observation = ArcAGI3Observation(
        task_id=task_id,
        episode_id=f"{task_id}/real_arc_trace",
        step_index=int(step.get("step_index", step.get("step", 0)) or 0),
        grid=frame,
        available_actions=tuple(legal_actions),
        extras={},
    )
    arrays["obs"].append(grid_to_features(frame))
    arrays["next_obs"].append(grid_to_features(next_arr))
    arrays["action_features"].append(
        action_to_features(
            observation,
            action,
            legal_index=legal_index,
            legal_count=len(legal_actions),
        )
    )
    arrays["action_id"].append(action_id(action))
    arrays["family"].append(family_index(action))
    arrays["legal_mask"].append(np.asarray(legal_action_mask(legal_actions), dtype=np.float32))
    arrays["reward"].append(float(step.get("score_delta", 0.0)))
    arrays["future_progress"].append(float(np.clip(future_progress, 0.0, 1.0)))
    arrays["terminal"].append(bool(step.get("terminal", False)))
    arrays["source_id"].append(REAL_ARC_SOURCE_ID)
    arrays["trace_id"].append(int(trace_id))
    arrays["trace_step"].append(int(trace_step))
    return True


def finalize_arrays(arrays: dict[str, list[Any]]) -> dict[str, np.ndarray]:
    if not arrays["obs"]:
        raise ValueError("no usable real ARC transitions found")
    return {
        "obs": np.asarray(arrays["obs"], dtype=np.float32),
        "next_obs": np.asarray(arrays["next_obs"], dtype=np.float32),
        "action_features": np.asarray(arrays["action_features"], dtype=np.float32).reshape(-1, ACTION_FEATURE_DIM),
        "action_id": np.asarray(arrays["action_id"], dtype=np.int64),
        "family": np.asarray(arrays["family"], dtype=np.int64),
        "legal_mask": np.asarray(arrays["legal_mask"], dtype=np.float32),
        "reward": np.asarray(arrays["reward"], dtype=np.float32),
        "future_progress": np.asarray(arrays["future_progress"], dtype=np.float32),
        "terminal": np.asarray(arrays["terminal"], dtype=np.bool_),
        "source_id": np.asarray(arrays["source_id"], dtype=np.int64),
        "trace_id": np.asarray(arrays["trace_id"], dtype=np.int64),
        "trace_step": np.asarray(arrays["trace_step"], dtype=np.int64),
    }


def real_arc_positive_step(step: dict[str, Any]) -> bool:
    if float(step.get("score_delta", 0.0) or 0.0) > 0.0:
        return True
    events = {str(item) for item in step.get("event_delta", []) if str(item)}
    return bool(POSITIVE_REAL_ARC_EVENTS.intersection(events))


def trace_future_progress(steps: list[dict[str, Any]], discount: float = 0.995) -> list[float]:
    values = [0.0] * len(steps)
    running = 0.0
    for index in range(len(steps) - 1, -1, -1):
        current = 1.0 if real_arc_positive_step(steps[index]) else 0.0
        running = max(current, float(discount) * running)
        values[index] = float(min(max(running, 0.0), 1.0))
    return values


def arrays_from_index(
    indexed: dict[str, list[tuple[Path, dict[str, Any]]]],
    selected_game_ids: Iterable[str],
) -> tuple[dict[str, np.ndarray], dict[str, int], list[str]]:
    arrays = empty_arrays()
    transitions_by_game: dict[str, int] = {}
    trace_paths: list[str] = []
    trace_id = 0
    for game_id in selected_game_ids:
        count = 0
        for path, attempt in indexed.get(game_id, []):
            task_id = str(attempt.get("task_id", f"arcagi3-official/{game_id}"))
            before = len(arrays["obs"])
            steps = [step for step in attempt.get("steps", []) if isinstance(step, dict)]
            future_progress = trace_future_progress(steps)
            for trace_step, (step, future_value) in enumerate(zip(steps, future_progress)):
                if isinstance(step, dict):
                    append_step(
                        arrays,
                        task_id=task_id,
                        step=step,
                        future_progress=future_value,
                        trace_id=trace_id,
                        trace_step=trace_step,
                    )
            added = len(arrays["obs"]) - before
            if added:
                trace_paths.append(str(path))
                count += added
                trace_id += 1
        if count:
            transitions_by_game[str(game_id)] = count
    return finalize_arrays(arrays), transitions_by_game, trace_paths


def write_split_data(
    *,
    arrays: dict[str, np.ndarray],
    output: str | Path,
) -> str:
    output_path = Path(output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output_path, **arrays)
    return sha256_file(output_path)


def positive_transition_count(arrays: dict[str, np.ndarray]) -> int:
    return int(np.sum(np.asarray(arrays["reward"]) > 0.0))


def future_positive_transition_count(arrays: dict[str, np.ndarray]) -> int:
    if "future_progress" not in arrays:
        return positive_transition_count(arrays)
    return int(np.sum(np.asarray(arrays["future_progress"]) > 0.0))


def sequence_start_indices(arrays: dict[str, np.ndarray], sequence_length: int) -> np.ndarray:
    if "trace_id" not in arrays:
        return np.asarray([], dtype=np.int64)
    trace_ids = np.asarray(arrays["trace_id"], dtype=np.int64)
    sequence_length = max(int(sequence_length), 2)
    starts: list[int] = []
    start = 0
    count = int(trace_ids.shape[0])
    while start < count:
        end = start + 1
        while end < count and int(trace_ids[end]) == int(trace_ids[start]):
            end += 1
        if end - start >= sequence_length:
            starts.extend(range(start, end - sequence_length + 1))
        start = end
    return np.asarray(starts, dtype=np.int64)


def sample_sequence_batch(
    arrays: dict[str, np.ndarray],
    *,
    starts: np.ndarray,
    positive_starts: np.ndarray,
    sequence_length: int,
    sequence_batch_size: int,
    device: torch.device,
    rng: np.random.Generator,
    positive_fraction: float,
) -> dict[str, torch.Tensor]:
    if starts.size == 0:
        raise ValueError("no valid real ARC sequence windows found")
    sequence_length = max(int(sequence_length), 2)
    sequence_batch_size = max(int(sequence_batch_size), 1)
    positive_fraction = max(min(float(positive_fraction), 1.0), 0.0)
    if positive_starts.size and positive_fraction > 0.0:
        positive_count = min(sequence_batch_size, max(1, int(round(float(sequence_batch_size) * positive_fraction))))
        rest_count = sequence_batch_size - positive_count
        positive_sample = rng.choice(positive_starts, size=positive_count, replace=positive_starts.size < positive_count)
        rest_sample = rng.choice(starts, size=rest_count, replace=starts.size < rest_count) if rest_count else np.asarray([], dtype=np.int64)
        chosen_starts = np.concatenate([positive_sample, rest_sample]).astype(np.int64, copy=False)
        rng.shuffle(chosen_starts)
    else:
        chosen_starts = rng.choice(starts, size=sequence_batch_size, replace=starts.size < sequence_batch_size)
    offsets = np.arange(sequence_length, dtype=np.int64)
    idx = chosen_starts[:, None] + offsets[None, :]
    batch = {
        "obs": torch.as_tensor(arrays["obs"][idx], dtype=torch.float32, device=device),
        "next_obs": torch.as_tensor(arrays["next_obs"][idx], dtype=torch.float32, device=device),
        "action_features": torch.as_tensor(arrays["action_features"][idx], dtype=torch.float32, device=device),
        "reward": torch.as_tensor(arrays["reward"][idx], dtype=torch.float32, device=device),
        "family": torch.as_tensor(arrays["family"][idx], dtype=torch.long, device=device),
    }
    if "action_id" in arrays:
        batch["action_id"] = torch.as_tensor(arrays["action_id"][idx], dtype=torch.long, device=device)
    if "future_progress" in arrays:
        batch["future_progress"] = torch.as_tensor(arrays["future_progress"][idx], dtype=torch.float32, device=device)
    return batch


def sequence_objective_losses(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    obs = batch["obs"]
    batch_size, sequence_length = int(obs.shape[0]), int(obs.shape[1])
    memory = model.initial_memory(batch_size, obs.device)
    outputs: dict[str, list[torch.Tensor]] = {
        "next_obs": [],
        "change_logits": [],
        "reward": [],
        "progress_logits": [],
        "future_progress_logits": [],
        "action_prior_logits": [],
        "noop_logits": [],
        "inverse_logits": [],
        "affordance": [],
        "rollout_latent": [],
        "memory_next_obs": [],
        "memory": [],
    }
    for step in range(sequence_length):
        output = model(batch["obs"][:, step], batch["action_features"][:, step], memory)
        memory = output["memory"]
        for key in outputs:
            outputs[key].append(output[key])
    flat_output = {}
    for key, values in outputs.items():
        stacked = torch.stack(values, dim=1)
        flat_output[key] = stacked.reshape(batch_size * sequence_length, *stacked.shape[2:])
    flat_batch = {
        key: value.reshape(batch_size * sequence_length, *value.shape[2:])
        if value.ndim > 2
        else value.reshape(batch_size * sequence_length)
        for key, value in batch.items()
    }
    return objective_losses_from_output(model, flat_output, flat_batch)


@torch.no_grad()
def evaluate_sequence_losses(model: torch.nn.Module, batch: dict[str, torch.Tensor]) -> dict[str, float]:
    losses = sequence_objective_losses(model, batch)
    out = {name: float(value.detach().item()) for name, value in losses.items()}
    out["total"] = float(weighted_loss(losses).detach().item())
    return out


def build_real_arc_splits(
    *,
    roots: Iterable[str | Path],
    game_manifest: str | Path,
    holdout_game_ids: Iterable[str],
    train_game_ids: Iterable[str] | None,
    max_traces_per_game: int,
    train_data_output: str | Path,
    holdout_data_output: str | Path,
    manifest_output: str | Path,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    all_game_ids = load_manifest_game_ids(game_manifest)
    train_games, holdout = split_game_ids(
        all_game_ids,
        train_game_ids=train_game_ids,
        holdout_game_ids=holdout_game_ids,
    )
    indexed = index_real_arc_traces(
        roots=roots,
        game_ids=all_game_ids,
        max_traces_per_game=max_traces_per_game,
    )
    train_arrays, train_counts, train_paths = arrays_from_index(indexed, train_games)
    holdout_arrays, holdout_counts, holdout_paths = arrays_from_index(indexed, holdout)
    train_sha = write_split_data(arrays=train_arrays, output=train_data_output)
    holdout_sha = write_split_data(arrays=holdout_arrays, output=holdout_data_output)
    train_positive_count = positive_transition_count(train_arrays)
    holdout_positive_count = positive_transition_count(holdout_arrays)
    train_future_positive_count = future_positive_transition_count(train_arrays)
    holdout_future_positive_count = future_positive_transition_count(holdout_arrays)
    manifest = {
        "format": "real_arc_interaction_trace_manifest_v1",
        "source": "local_offline_arcagi3_public_traces",
        "rules": {
            "public_frames_only": True,
            "used_hidden_labels": False,
            "used_action_advice": False,
            "no_game_id_features": True,
            "split_by_game_id_before_tensorization": True,
            "official_arcagi3_used_for_training": True,
            "holdout_official_arcagi3_used_for_training": False,
        },
        "train_game_ids": train_games,
        "holdout_game_ids": holdout,
        "max_traces_per_game": int(max_traces_per_game),
        "max_traces_per_game_unlimited": int(max_traces_per_game) <= 0,
        "train_data_path": str(train_data_output),
        "train_data_sha256": train_sha,
        "holdout_data_path": str(holdout_data_output),
        "holdout_data_sha256": holdout_sha,
        "train_transition_count": int(train_arrays["obs"].shape[0]),
        "holdout_transition_count": int(holdout_arrays["obs"].shape[0]),
        "train_positive_transition_count": train_positive_count,
        "holdout_positive_transition_count": holdout_positive_count,
        "train_future_positive_transition_count": train_future_positive_count,
        "holdout_future_positive_transition_count": holdout_future_positive_count,
        "train_positive_transition_rate": float(train_positive_count / max(int(train_arrays["obs"].shape[0]), 1)),
        "holdout_positive_transition_rate": float(holdout_positive_count / max(int(holdout_arrays["obs"].shape[0]), 1)),
        "train_future_positive_transition_rate": float(
            train_future_positive_count / max(int(train_arrays["obs"].shape[0]), 1)
        ),
        "holdout_future_positive_transition_rate": float(
            holdout_future_positive_count / max(int(holdout_arrays["obs"].shape[0]), 1)
        ),
        "train_transitions_by_game": train_counts,
        "holdout_transitions_by_game": holdout_counts,
        "train_trace_count": len(train_paths),
        "holdout_trace_count": len(holdout_paths),
        "train_trace_path_sample": train_paths[:40],
        "holdout_trace_path_sample": holdout_paths[:40],
        "source_counts": {
            "real_official_arcagi3_train": int(train_arrays["obs"].shape[0]),
            "real_official_arcagi3_holdout": int(holdout_arrays["obs"].shape[0]),
        },
        "array_shapes": {
            "train": {key: list(value.shape) for key, value in train_arrays.items()},
            "holdout": {key: list(value.shape) for key, value in holdout_arrays.items()},
        },
    }
    output = Path(manifest_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    manifest["manifest_sha256"] = sha256_file(output)
    output.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return train_arrays, holdout_arrays, manifest


def train_arm_until_plateau(
    *,
    arm: str,
    arrays: dict[str, np.ndarray],
    config: ExternalBaseConfig,
    max_steps: int,
    min_steps: int,
    batch_size: int,
    device: torch.device,
    seed: int,
    check_interval: int,
    patience: int,
    min_delta: float,
    grok_loss: float,
    positive_fraction: float = 0.0,
    sequence_length: int = 1,
) -> tuple[torch.nn.Module, dict[str, Any]]:
    set_seed(seed)
    from .base_world_model import ExternalBaseWorldModel

    model = ExternalBaseWorldModel(config, device=device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=2.0e-3 if arm != "old_base_finetuned" else 1.0e-3,
        weight_decay=1.0e-4,
    )
    rng = np.random.default_rng(seed + 17)
    torch_rng = torch.Generator(device=device).manual_seed(seed + 29)
    flat_eval_batch = sample_batch(
        arrays,
        min(max(int(batch_size), 1) * 2, int(arrays["obs"].shape[0])),
        device,
        rng,
        positive_fraction=0.0,
    )
    sequence_length = max(int(sequence_length), 1)
    sequence_starts = sequence_start_indices(arrays, sequence_length) if arm != "null_training_control" else np.asarray([], dtype=np.int64)
    positive_sequence_starts = np.asarray([], dtype=np.int64)
    if sequence_starts.size and "future_progress" in arrays:
        future_progress = np.asarray(arrays["future_progress"], dtype=np.float32)
        positive_sequence_starts = np.asarray(
            [
                int(start)
                for start in sequence_starts
                if float(np.max(future_progress[int(start) : int(start) + sequence_length])) > 0.0
            ],
            dtype=np.int64,
        )
    use_sequence_training = bool(sequence_starts.size and sequence_length > 1)
    sequence_batch_size = max(1, int(batch_size) // max(sequence_length, 1))
    if use_sequence_training:
        eval_batch = sample_sequence_batch(
            arrays,
            starts=sequence_starts,
            positive_starts=positive_sequence_starts,
            sequence_length=sequence_length,
            sequence_batch_size=sequence_batch_size,
            device=device,
            rng=rng,
            positive_fraction=positive_fraction,
        )
        initial = evaluate_sequence_losses(model, eval_batch)
    else:
        eval_batch = sample_batch(
            arrays,
            min(max(int(batch_size), 1) * 2, int(arrays["obs"].shape[0])),
            device,
            rng,
            positive_fraction=positive_fraction,
        )
        if arm == "null_training_control":
            eval_batch = shuffle_targets(eval_batch, torch_rng)
        initial = evaluate_losses(model, eval_batch)
    best_total = float(initial["total"])
    best_step = 0
    stale_checks = 0
    best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    history = [{"step": 0, "total": best_total}]
    tail_losses: list[float] = []
    stop_reason = "max_steps"
    check_interval = max(int(check_interval), 1)
    patience = max(int(patience), 1)
    min_steps = max(int(min_steps), 0)
    max_steps = max(int(max_steps), 1)
    model.train()
    for step in range(1, max_steps + 1):
        if use_sequence_training:
            batch = sample_sequence_batch(
                arrays,
                starts=sequence_starts,
                positive_starts=positive_sequence_starts,
                sequence_length=sequence_length,
                sequence_batch_size=sequence_batch_size,
                device=device,
                rng=rng,
                positive_fraction=positive_fraction,
            )
            losses = sequence_objective_losses(model, batch)
        else:
            batch = sample_batch(arrays, batch_size, device, rng, positive_fraction=positive_fraction)
            if arm == "null_training_control":
                batch = shuffle_targets(batch, torch_rng)
            losses = objective_losses(model, batch)
        loss = weighted_loss(losses)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 2.0)
        optimizer.step()
        tail_losses.append(float(loss.detach().item()))
        if step % check_interval != 0 and step != max_steps:
            continue
        model.eval()
        current = evaluate_sequence_losses(model, eval_batch) if use_sequence_training else evaluate_losses(model, eval_batch)
        model.train()
        current_total = float(current["total"])
        history.append({"step": step, "total": current_total})
        if current_total < best_total - float(min_delta):
            best_total = current_total
            best_step = step
            stale_checks = 0
            best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
        else:
            stale_checks += 1
        if step >= min_steps and current_total <= float(grok_loss):
            stop_reason = "grok_loss_threshold"
            break
        if step >= min_steps and stale_checks >= patience:
            stop_reason = "loss_plateau"
            break
    model.load_state_dict(best_state)
    model.eval()
    final = evaluate_sequence_losses(model, eval_batch) if use_sequence_training else evaluate_losses(model, eval_batch)
    utility = memory_utility_diagnostic(model, flat_eval_batch)
    return model, {
        "arm": arm,
        "seed": seed,
        "steps": int(history[-1]["step"]),
        "best_step": int(best_step),
        "batch_size": batch_size,
        "positive_fraction": float(max(min(float(positive_fraction), 1.0), 0.0)),
        "sequence_training": {
            "enabled": use_sequence_training,
            "sequence_length": int(sequence_length),
            "sequence_batch_size": int(sequence_batch_size),
            "sequence_window_count": int(sequence_starts.size),
            "positive_sequence_window_count": int(positive_sequence_starts.size),
        },
        "initial_losses": initial,
        "final_losses": final,
        "recent_total_loss_mean": float(np.mean(tail_losses[-min(len(tail_losses), 20) :])) if tail_losses else final["total"],
        "loss_delta": initial["total"] - final["total"],
        "training_losses_are_diagnostic_only": True,
        "memory_utility": utility,
        "plateau": {
            "enabled": True,
            "stop_reason": stop_reason,
            "check_interval": check_interval,
            "patience": patience,
            "min_delta": float(min_delta),
            "min_steps": int(min_steps),
            "max_steps": int(max_steps),
            "grok_loss": float(grok_loss),
            "stale_checks": int(stale_checks),
            "best_eval_total": float(best_total),
            "history_tail": history[-20:],
        },
    }


@torch.no_grad()
def evaluate_on_holdout(
    model: torch.nn.Module,
    arrays: dict[str, np.ndarray],
    *,
    batch_size: int,
    device: torch.device,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    batch = sample_batch(arrays, min(max(int(batch_size), 1) * 2, int(arrays["obs"].shape[0])), device, rng)
    losses = evaluate_losses(model, batch)
    utility = memory_utility_diagnostic(model, batch)
    return {"losses": losses, "memory_utility": utility}


def train_real_arc_external_base(
    *,
    roots: Iterable[str | Path],
    game_manifest: str | Path,
    train_game_ids: Iterable[str] | None,
    holdout_game_ids: Iterable[str],
    arms: Iterable[str],
    max_traces_per_game: int,
    train_data_output: str | Path,
    holdout_data_output: str | Path,
    trace_manifest_output: str | Path,
    checkpoint_output: str | Path,
    manifest_output: str | Path,
    recurrent_checkpoint: str | Path,
    steps: int,
    plateau: bool,
    min_steps: int,
    plateau_check_interval: int,
    plateau_patience: int,
    plateau_min_delta: float,
    grok_loss: float,
    sequence_length: int,
    batch_size: int,
    device: DeviceLike,
    seed: int,
) -> dict[str, Any]:
    target_device = resolve_device(device)
    train_arrays, holdout_arrays, trace_manifest = build_real_arc_splits(
        roots=roots,
        game_manifest=game_manifest,
        train_game_ids=train_game_ids,
        holdout_game_ids=holdout_game_ids,
        max_traces_per_game=max_traces_per_game,
        train_data_output=train_data_output,
        holdout_data_output=holdout_data_output,
        manifest_output=trace_manifest_output,
    )
    config = ExternalBaseConfig()
    arm_states: dict[str, dict[str, Any]] = {}
    arm_metrics: dict[str, Any] = {}
    seed_by_arm = {
        "old_base_world_model": int(seed) + 101,
        "old_base_finetuned": int(seed) + 102,
        "from_scratch_external_base": int(seed) + 103,
        "null_training_control": int(seed) + 104,
    }
    selected_arms = [str(item) for item in arms if str(item) in set(TRAINED_ARMS)]
    if not selected_arms:
        raise ValueError(f"at least one training arm is required from: {', '.join(TRAINED_ARMS)}")
    train_future_positive_count = future_positive_transition_count(train_arrays)
    positive_fraction = 0.25 if train_future_positive_count > 0 else 0.0
    for arm in selected_arms:
        set_seed(seed_by_arm[arm])
        if plateau:
            model, metrics = train_arm_until_plateau(
                arm=arm,
                arrays=train_arrays,
                config=config,
                max_steps=steps,
                min_steps=min_steps,
                batch_size=batch_size,
                device=target_device,
                seed=seed_by_arm[arm],
                check_interval=plateau_check_interval,
                patience=plateau_patience,
                min_delta=plateau_min_delta,
                grok_loss=grok_loss,
                positive_fraction=positive_fraction,
                sequence_length=sequence_length,
            )
        else:
            model, metrics = train_arm(
                arm=arm,
                arrays=train_arrays,
                config=config,
                steps=steps,
                batch_size=batch_size,
                device=target_device,
                seed=seed_by_arm[arm],
                positive_fraction=positive_fraction,
            )
        metrics["real_arc_holdout"] = evaluate_on_holdout(
            model,
            holdout_arrays,
            batch_size=batch_size,
            device=target_device,
            seed=seed_by_arm[arm] + 1000,
        )
        arm_metrics[arm] = metrics
        arm_states[arm] = {
            "state_dict": {key: value.detach().cpu() for key, value in model.state_dict().items()},
            "metrics": metrics,
        }
    manifest = {
        "format": "real_arc_external_base_manifest_v1",
        "checkpoint": str(checkpoint_output),
        "old_recurrent_checkpoint": str(recurrent_checkpoint),
        "old_recurrent_checkpoint_sha256": sha256_file(recurrent_checkpoint),
        "trace_manifest": str(trace_manifest_output),
        "trace_manifest_sha256": sha256_file(trace_manifest_output),
        "trace_data_sha256": trace_manifest.get("train_data_sha256"),
        "holdout_data_sha256": trace_manifest.get("holdout_data_sha256"),
        "trace_transition_count": trace_manifest.get("train_transition_count"),
        "holdout_transition_count": trace_manifest.get("holdout_transition_count"),
        "train_positive_transition_count": trace_manifest.get("train_positive_transition_count"),
        "holdout_positive_transition_count": trace_manifest.get("holdout_positive_transition_count"),
        "train_future_positive_transition_count": trace_manifest.get("train_future_positive_transition_count"),
        "holdout_future_positive_transition_count": trace_manifest.get("holdout_future_positive_transition_count"),
        "train_positive_transition_rate": trace_manifest.get("train_positive_transition_rate"),
        "holdout_positive_transition_rate": trace_manifest.get("holdout_positive_transition_rate"),
        "train_future_positive_transition_rate": trace_manifest.get("train_future_positive_transition_rate"),
        "holdout_future_positive_transition_rate": trace_manifest.get("holdout_future_positive_transition_rate"),
        "source_counts": trace_manifest.get("source_counts"),
        "sequence_length": int(sequence_length),
        "train_game_ids": trace_manifest.get("train_game_ids", []),
        "holdout_game_ids": trace_manifest.get("holdout_game_ids", []),
        "model_arms": ["old_base_unchanged", *selected_arms],
        "frozen_before_holdout_eval": True,
        "no_sealed_official_training": False,
        "holdout_official_arcagi3_used_for_training": False,
        "training_objectives": [
            "next observation/change prediction",
            "reward/event/no-op prediction",
            "positive progress event classification with class weighting",
            "discounted future progress classification from successful trace prefixes",
            "future-positive recurrent action prior",
            "inverse dynamics",
            "action-affordance prediction",
            "temporal object/region persistence",
            "latent rollout consistency",
            "memory improves delayed/partial prediction",
        ],
        "arm_metrics": arm_metrics,
    }
    save_external_base_checkpoint(
        recurrent_checkpoint=recurrent_checkpoint,
        output_path=checkpoint_output,
        external_config=config,
        arm_states=arm_states,
        manifest=manifest,
        metrics=arm_metrics,
    )
    manifest["checkpoint_sha256"] = sha256_file(checkpoint_output)
    write_external_manifest(manifest_output, manifest)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", choices=["external_base"], default="external_base")
    parser.add_argument("--trace-roots", default="runs,docs")
    parser.add_argument("--game-manifest", default="docs/arcagi3_official_games.json")
    parser.add_argument("--train-game-ids", default="")
    parser.add_argument("--holdout-game-ids", default=",".join(DEFAULT_HOLDOUT_GAME_IDS))
    parser.add_argument("--arms", default=",".join(TRAINED_ARMS))
    parser.add_argument("--max-traces-per-game", type=int, default=8)
    parser.add_argument("--train-data-output", default="runs/real_arc_train_tensors.npz")
    parser.add_argument("--holdout-data-output", default="runs/real_arc_holdout_tensors.npz")
    parser.add_argument("--trace-manifest-output", default="runs/real_arc_trace_manifest.json")
    parser.add_argument("--checkpoint-output", default="runs/real_arc_external_base.pt")
    parser.add_argument("--manifest-output", default="runs/real_arc_external_base_manifest.json")
    parser.add_argument("--recurrent-checkpoint", default="frozen/recurrent_latent_fast.pt")
    parser.add_argument("--steps", type=int, default=120)
    parser.add_argument("--plateau", action="store_true", default=False)
    parser.add_argument("--min-steps", type=int, default=400)
    parser.add_argument("--plateau-check-interval", type=int, default=50)
    parser.add_argument("--plateau-patience", type=int, default=8)
    parser.add_argument("--plateau-min-delta", type=float, default=5.0e-4)
    parser.add_argument("--grok-loss", type=float, default=2.0e-2)
    parser.add_argument("--sequence-length", type=int, default=32)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--device", default=AUTO_DEVICE)
    parser.add_argument("--seed", type=int, default=20260613)
    args = parser.parse_args()
    del args.config
    manifest = train_real_arc_external_base(
        roots=parse_csv(args.trace_roots),
        game_manifest=args.game_manifest,
        train_game_ids=parse_csv(args.train_game_ids),
        holdout_game_ids=parse_csv(args.holdout_game_ids),
        arms=parse_csv(args.arms),
        max_traces_per_game=args.max_traces_per_game,
        train_data_output=args.train_data_output,
        holdout_data_output=args.holdout_data_output,
        trace_manifest_output=args.trace_manifest_output,
        checkpoint_output=args.checkpoint_output,
        manifest_output=args.manifest_output,
        recurrent_checkpoint=args.recurrent_checkpoint,
        steps=args.steps,
        plateau=args.plateau,
        min_steps=args.min_steps,
        plateau_check_interval=args.plateau_check_interval,
        plateau_patience=args.plateau_patience,
        plateau_min_delta=args.plateau_min_delta,
        grok_loss=args.grok_loss,
        sequence_length=args.sequence_length,
        batch_size=args.batch_size,
        device=args.device,
        seed=args.seed,
    )
    print(
        json.dumps(
            {
                "checkpoint": manifest["checkpoint"],
                "checkpoint_sha256": manifest["checkpoint_sha256"],
                "train_games": len(manifest["train_game_ids"]),
                "holdout_games": manifest["holdout_game_ids"],
                "train_transition_count": manifest["trace_transition_count"],
                "holdout_transition_count": manifest["holdout_transition_count"],
                "train_positive_transition_count": manifest["train_positive_transition_count"],
                "holdout_positive_transition_count": manifest["holdout_positive_transition_count"],
                "train_future_positive_transition_count": manifest["train_future_positive_transition_count"],
                "holdout_future_positive_transition_count": manifest["holdout_future_positive_transition_count"],
                "sequence_length": manifest["sequence_length"],
                "no_sealed_official_training": manifest["no_sealed_official_training"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
