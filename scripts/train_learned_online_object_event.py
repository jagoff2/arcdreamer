from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle

import numpy as np
import torch

from arcagi.agents.learned_online_object_event_agent import LearnedOnlineObjectEventAgent
from arcagi.core.types import ActionName, ObjectState, StructuredState, Transition
from arcagi.learned_online.event_tokens import (
    build_transition_targets,
    encode_action_tokens,
    encode_state_tokens,
)
from arcagi.learned_online.object_event_model import ObjectEventModel, ObjectEventModelConfig


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the learned online object-event scaffold.")
    parser.add_argument("--source", choices=("synthetic_object_event", "trace_jsonl"), default="synthetic_object_event")
    parser.add_argument("--trace-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=Path("artifacts/learned_online_object_event_v1.pkl"))
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--state-layers", type=int, default=1)
    parser.add_argument("--action-cross-layers", type=int, default=1)
    args = parser.parse_args()

    config = ObjectEventModelConfig(
        d_model=int(args.d_model),
        state_layers=int(args.state_layers),
        action_cross_layers=int(args.action_cross_layers),
        dropout=0.0,
    )
    device = torch.device(args.device)
    model = ObjectEventModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3)
    rng = np.random.default_rng(int(args.seed))

    if args.source == "trace_jsonl":
        if args.trace_path is None:
            raise SystemExit("--trace-path is required for --source trace_jsonl")
        raise SystemExit("trace_jsonl transition bootstrap is intentionally deferred until object-event scaffold tests pass")

    losses: list[float] = []
    for step in range(max(int(args.steps), 1)):
        transition = _synthetic_transition(rng)
        actions = transition.state.affordances
        state_tokens = encode_state_tokens(transition.state)
        action_tokens = encode_action_tokens(transition.state, actions)
        targets = build_transition_targets(transition, actions=actions)
        output = model(
            state_numeric=torch.as_tensor(state_tokens.numeric[None, :, :], dtype=torch.float32, device=device),
            state_type_ids=torch.as_tensor(state_tokens.type_ids[None, :], dtype=torch.long, device=device),
            state_mask=torch.as_tensor(state_tokens.mask[None, :], dtype=torch.bool, device=device),
            action_numeric=torch.as_tensor(action_tokens.numeric[None, :, :], dtype=torch.float32, device=device),
            action_type_ids=torch.as_tensor(action_tokens.action_type_ids[None, :], dtype=torch.long, device=device),
            direction_ids=torch.as_tensor(action_tokens.direction_ids[None, :], dtype=torch.long, device=device),
            action_mask=torch.as_tensor(action_tokens.mask[None, :], dtype=torch.bool, device=device),
        )
        loss_parts = model.loss(
            output,
            target_outcome=torch.as_tensor(targets.outcome[None, :], dtype=torch.float32, device=device),
            target_delta=torch.as_tensor(targets.delta[None, :], dtype=torch.float32, device=device),
            actual_action_index=torch.as_tensor([targets.actual_action_index], dtype=torch.long, device=device),
            action_mask=torch.as_tensor(action_tokens.mask[None, :], dtype=torch.bool, device=device),
        )
        optimizer.zero_grad(set_to_none=True)
        loss_parts["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        losses.append(float(loss_parts["loss"].detach().cpu()))

    metadata = LearnedOnlineObjectEventAgent.checkpoint_metadata()
    checkpoint = {
        "seed": int(args.seed),
        "config": config.to_dict(),
        "model_state": model.state_dict(),
        "metadata": metadata,
        "training_summary": {
            "source": args.source,
            "steps": int(args.steps),
            "avg_loss_last_50": float(np.mean(losses[-50:])) if losses else 0.0,
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("wb") as handle:
        pickle.dump(checkpoint, handle)
    print(json.dumps({"checkpoint_path": str(args.output), **checkpoint["training_summary"]}, sort_keys=True))


def _synthetic_transition(rng: np.random.Generator) -> Transition:
    state, correct_action = _synthetic_state(rng)
    selected = _clicked_object_for_action(state, correct_action)
    if selected is None:
        next_state = state
    else:
        remaining = tuple(obj for obj in state.objects if obj is not selected)
        grid = np.asarray(state.as_grid()).copy()
        for row, col in selected.cells:
            grid[row, col] = 0
        next_state = StructuredState(
            task_id=state.task_id,
            episode_id=state.episode_id,
            step_index=state.step_index + 1,
            grid_shape=state.grid_shape,
            grid_signature=tuple(int(value) for value in grid.reshape(-1)),
            objects=remaining,
            relations=(),
            affordances=state.affordances,
            action_roles=state.action_roles,
            inventory=state.inventory,
            flags=(("objective_progress", "1"),),
        )
    return Transition(state=state, action=correct_action, reward=1.0, next_state=next_state, terminated=False, info={"score_delta": 1.0})


def _synthetic_state(rng: np.random.Generator) -> tuple[StructuredState, ActionName]:
    grid = np.zeros((8, 8), dtype=np.int64)
    cue_mode = int(rng.integers(2))
    cue_color = 3 if cue_mode == 0 else 7
    red_pos = _random_free_cell(rng, {(0, 0)})
    blue_pos = _random_free_cell(rng, {(0, 0), red_pos})
    cue = _object("cue", cue_color, (0, 0), tags=("interactable",))
    red = _object("red", 2, red_pos)
    blue = _object("blue", 5, blue_pos)
    for obj in (cue, red, blue):
        for row, col in obj.cells:
            grid[row, col] = obj.color
    actions = [f"click:{red_pos[1]}:{red_pos[0]}", f"click:{blue_pos[1]}:{blue_pos[0]}"]
    occupied = {(0, 0), red_pos, blue_pos}
    while len(actions) < 12:
        row, col = _random_free_cell(rng, occupied)
        occupied.add((row, col))
        actions.append(f"click:{col}:{row}")
    actions.extend(["0", "undo", "up", "down"])
    rng.shuffle(actions)
    correct = f"click:{red_pos[1]}:{red_pos[0]}" if cue_mode == 0 else f"click:{blue_pos[1]}:{blue_pos[0]}"
    roles = {action: "click" for action in actions if action.startswith("click:")}
    roles.update({"0": "reset_level", "undo": "undo", "up": "move_up", "down": "move_down"})
    state = StructuredState(
        task_id="synthetic_object_event",
        episode_id="0",
        step_index=0,
        grid_shape=grid.shape,
        grid_signature=tuple(int(value) for value in grid.reshape(-1)),
        objects=(cue, red, blue),
        relations=(),
        affordances=tuple(actions),
        action_roles=tuple(sorted(roles.items())),
    )
    return state, correct


def _object(name: str, color: int, cell: tuple[int, int], *, tags: tuple[str, ...] = ()) -> ObjectState:
    row, col = cell
    return ObjectState(
        object_id=name,
        color=color,
        cells=(cell,),
        bbox=(row, col, row, col),
        centroid=(float(row), float(col)),
        area=1,
        tags=tags,
    )


def _random_free_cell(rng: np.random.Generator, occupied: set[tuple[int, int]]) -> tuple[int, int]:
    while True:
        cell = (int(rng.integers(1, 8)), int(rng.integers(0, 8)))
        if cell not in occupied:
            return cell


def _clicked_object_for_action(state: StructuredState, action: ActionName) -> ObjectState | None:
    parts = action.split(":")
    if len(parts) != 3 or parts[0] != "click":
        return None
    col = int(parts[1])
    row = int(parts[2])
    for obj in state.objects:
        if (row, col) in obj.cells:
            return obj
    return None


if __name__ == "__main__":
    main()
