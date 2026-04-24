from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from arcagi.agents.learned_online_recurrent_agent import LearnedOnlineRecurrentAgent
from arcagi.envs.arc_adapter import Arcade, ArcToolkitEnv, arc_operation_mode
from arcagi.learned_online.questions import select_question


def finetune(
    *,
    trace_path: Path,
    output: Path,
    checkpoint_path: Path | None,
    game_id: str,
    mode: str,
    epochs: int,
    max_steps: int,
) -> dict[str, Any]:
    actions = _trace_actions(trace_path)
    if checkpoint_path is not None and checkpoint_path.exists():
        agent = LearnedOnlineRecurrentAgent.from_checkpoint(checkpoint_path)
    else:
        agent = LearnedOnlineRecurrentAgent()
    operation_mode = arc_operation_mode(mode)
    arcade = None if Arcade is None else Arcade(operation_mode=operation_mode)
    returns: list[float] = []
    levels_completed: list[int] = []
    try:
        for epoch in range(int(epochs)):
            env = ArcToolkitEnv(game_id, operation_mode=operation_mode, arcade=arcade)
            try:
                result = _run_teacher_epoch(
                    agent=agent,
                    env=env,
                    actions=actions,
                    max_steps=max_steps,
                    seed=epoch,
                )
            finally:
                env.close()
            returns.append(float(result["return"]))
            levels_completed.append(int(result["levels_completed"]))
    finally:
        if arcade is not None:
            close_scorecard = getattr(arcade, "close_scorecard", None)
            if callable(close_scorecard):
                close_scorecard()
    agent.save_checkpoint(output)
    return {
        "checkpoint": str(output),
        "source_checkpoint": str(checkpoint_path) if checkpoint_path is not None else "",
        "trace_path": str(trace_path),
        "game_id": str(game_id),
        "mode": str(mode),
        "epochs": int(epochs),
        "teacher_steps": min(int(max_steps), len(actions)),
        "avg_return": sum(returns) / max(float(len(returns)), 1.0),
        "success_rate": sum(1.0 for value in returns if value > 0.0) / max(float(len(returns)), 1.0),
        "max_levels_completed": max(levels_completed) if levels_completed else 0,
        "model_updates": int(agent.model.updates),
        "pretrain_updates": int(agent.model.pretrain_updates),
    }


def _trace_actions(trace_path: Path) -> tuple[str, ...]:
    actions: list[str] = []
    for line in trace_path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if row.get("event") == "step":
            actions.append(str(row["action"]))
    if not actions:
        raise ValueError(f"trace at {trace_path!s} contains no step actions")
    return tuple(actions)


def _run_teacher_epoch(
    *,
    agent: LearnedOnlineRecurrentAgent,
    env: Any,
    actions: tuple[str, ...],
    max_steps: int,
    seed: int,
) -> dict[str, Any]:
    observation = env.reset(seed=seed)
    agent.reset_episode()
    total_return = 0.0
    levels_completed = 0
    steps = 0
    for action in actions[: max(int(max_steps), 0)]:
        state = agent.observe(observation)
        question = select_question(agent.belief)
        hidden_before = agent.model.hidden.copy()
        actions = tuple(str(item) for item in state.affordances)
        if str(action) in actions:
            features = agent.policy.build_feature_matrix(state, actions, question=question)
            positive_index = actions.index(str(action))
            if len(actions) > 1:
                negative_features = features[[index for index in range(len(actions)) if index != positive_index]]
                agent.model.imitation_update(
                    features[positive_index],
                    negative_features,
                    hidden=hidden_before,
                )
        agent.current_question = question
        agent.last_state = state
        agent.last_action = str(action)
        agent.last_question = question
        result = env.step(str(action))
        observation = result.observation
        agent.update_after_step(
            next_observation=result.observation,
            reward=float(result.reward),
            terminated=bool(result.terminated or result.truncated),
            info=dict(result.info or {}),
        )
        total_return += float(result.reward)
        levels_completed = max(levels_completed, _levels_completed(observation))
        steps += 1
    return {
        "return": float(total_return),
        "levels_completed": int(levels_completed),
        "steps": int(steps),
    }


def _levels_completed(observation: Any) -> int:
    extras = getattr(observation, "extras", {})
    if isinstance(extras, dict):
        try:
            return int(extras.get("levels_completed", 0) or 0)
        except Exception:
            return 0
    return 0


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint-path", type=Path, default=Path(""))
    parser.add_argument("--game-id", type=str, default="ar25-0c556536")
    parser.add_argument("--mode", type=str, default="offline")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=320)
    args = parser.parse_args()
    checkpoint_path = args.checkpoint_path if str(args.checkpoint_path) else None
    result = finetune(
        trace_path=args.trace_path,
        output=args.output,
        checkpoint_path=checkpoint_path,
        game_id=args.game_id,
        mode=args.mode,
        epochs=args.epochs,
        max_steps=args.max_steps,
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
