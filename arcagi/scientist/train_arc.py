"""Repeated-session offline ARC training for the public scientist agent.

This path trains the same ARC-facing scientist agent exposed through the shared
evaluation harness. It replays a cached local offline ARC game multiple times
and persists the agent checkpoint between prize-shaped sessions so online world
model updates and spotlight priors can carry forward.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Any

from arcagi.agents.scientist_agent import (
    HyperGeneralizingScientistAgent,
    load_spotlight_scientist_checkpoint,
    save_spotlight_scientist_checkpoint,
)
from arcagi.envs.arc_adapter import (
    ArcToolkitEnv,
    arc_operation_mode,
    arc_toolkit_available,
    list_arc_games,
    require_dense_arc_action_surface,
)

from .runtime import run_episode


@dataclass(frozen=True)
class ScientistArcTrainingConfig:
    game_id: str = "ar25-0c556536"
    mode: str = "offline"
    seed: int = 0
    sessions: int = 24
    eval_every: int = 6
    eval_sessions: int = 2
    max_steps: int = 256
    checkpoint_path: str = "artifacts/scientist_arc_offline_best.pkl"
    latest_checkpoint_path: str = "artifacts/scientist_arc_offline_latest.pkl"
    init_checkpoint_path: str = ""
    allow_sparse_click_smoke: bool = False


def train_arc_offline(config: ScientistArcTrainingConfig) -> dict[str, object]:
    if not arc_toolkit_available():
        raise RuntimeError("ARC toolkit is not installed in this environment")
    action_surface = require_dense_arc_action_surface(
        context="scientist ARC training",
        allow_sparse_click_smoke=config.allow_sparse_click_smoke,
    )
    operation_mode = arc_operation_mode(config.mode)
    available_games = list_arc_games(operation_mode=operation_mode)
    if config.game_id not in available_games:
        raise RuntimeError(
            f"ARC game {config.game_id!r} not available in mode={config.mode!r}; available={available_games!r}"
        )

    agent = (
        load_spotlight_scientist_checkpoint(config.init_checkpoint_path)
        if config.init_checkpoint_path and Path(config.init_checkpoint_path).exists()
        else HyperGeneralizingScientistAgent()
    )

    train_rows: list[dict[str, object]] = []
    best_key: tuple[float, float, float, float] | None = None
    for session_idx in range(config.sessions):
        seed = config.seed + session_idx
        result = _run_arc_episode(
            config.game_id,
            agent=agent,
            operation_mode=operation_mode,
            seed=seed,
            max_steps=config.max_steps,
        )
        row = {
            "session": session_idx + 1,
            "seed": seed,
            "return": float(result.total_reward),
            "steps": int(result.steps),
            "terminated": bool(result.terminated),
            "won": bool(result.won),
            "levels_completed": int(result.levels_completed),
            "reset_steps": int(result.reset_steps),
        }
        train_rows.append(row)

        should_eval = ((session_idx + 1) % max(config.eval_every, 1) == 0) or (session_idx + 1 == config.sessions)
        if not should_eval:
            continue

        holdout = evaluate_arc_offline(agent, config=config, operation_mode=operation_mode)
        result_key = _result_key(holdout)
        payload = {
            "event": "scientist_arc_train_eval",
            "session": session_idx + 1,
            "seed": seed,
            "spotlight_feature_schema_version": _spotlight_feature_schema_version(agent),
            "recent_session_win_rate": _mean_bool(item["won"] for item in train_rows[-config.eval_every :]),
            "recent_avg_levels_completed": mean(float(item["levels_completed"]) for item in train_rows[-config.eval_every :]),
            "recent_avg_return": mean(float(item["return"]) for item in train_rows[-config.eval_every :]),
            "holdout": holdout,
            "checkpoint_path": config.checkpoint_path,
            "latest_checkpoint_path": config.latest_checkpoint_path,
        }
        save_spotlight_scientist_checkpoint(agent, config.latest_checkpoint_path)
        if best_key is None or result_key > best_key:
            best_key = result_key
            save_spotlight_scientist_checkpoint(agent, config.checkpoint_path)
            payload["best_updated"] = True
        else:
            payload["best_updated"] = False
        print(json.dumps(payload, sort_keys=True))

    final_holdout = evaluate_arc_offline(agent, config=config, operation_mode=operation_mode)
    save_spotlight_scientist_checkpoint(agent, config.latest_checkpoint_path)
    summary = {
        "config": asdict(config),
        **action_surface,
        "spotlight_feature_schema_version": _spotlight_feature_schema_version(agent),
        "sessions": config.sessions,
        "train_session_win_rate": _mean_bool(item["won"] for item in train_rows),
        "train_avg_levels_completed": mean(float(item["levels_completed"]) for item in train_rows) if train_rows else 0.0,
        "train_avg_return": mean(float(item["return"]) for item in train_rows) if train_rows else 0.0,
        "final_holdout": final_holdout,
        "best_checkpoint_path": config.checkpoint_path,
        "latest_checkpoint_path": config.latest_checkpoint_path,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def _spotlight_feature_schema_version(agent: HyperGeneralizingScientistAgent) -> int:
    diagnostics = agent.diagnostics()
    spotlight = diagnostics.get("spotlight", {}) if isinstance(diagnostics, dict) else {}
    if not isinstance(spotlight, dict):
        return 0
    try:
        return int(spotlight.get("feature_schema_version", 0) or 0)
    except Exception:
        return 0


def evaluate_arc_offline(
    agent: HyperGeneralizingScientistAgent,
    *,
    config: ScientistArcTrainingConfig,
    operation_mode: Any | None = None,
) -> dict[str, object]:
    action_surface = require_dense_arc_action_surface(
        context="scientist ARC holdout evaluation",
        allow_sparse_click_smoke=config.allow_sparse_click_smoke,
    )
    snapshot = agent.state_dict()
    returns: list[float] = []
    session_wins: list[bool] = []
    levels_completed: list[int] = []
    steps: list[int] = []
    diagnostics: dict[str, object] = {}
    for idx in range(config.eval_sessions):
        eval_agent = HyperGeneralizingScientistAgent()
        eval_agent.load_state_dict(snapshot)
        seed = config.seed + 10_000 + idx
        result = _run_arc_episode(
            config.game_id,
            agent=eval_agent,
            operation_mode=operation_mode if operation_mode is not None else arc_operation_mode(config.mode),
            seed=seed,
            max_steps=config.max_steps,
        )
        returns.append(float(result.total_reward))
        session_wins.append(bool(result.won))
        levels_completed.append(int(result.levels_completed))
        steps.append(int(result.steps))
        diagnostics = dict(result.diagnostics)
    return {
        **action_surface,
        "session_win_rate": _mean_bool(session_wins),
        "success_rate": _mean_bool(session_wins),
        "avg_levels_completed": mean(float(value) for value in levels_completed) if levels_completed else 0.0,
        "avg_return": mean(returns) if returns else 0.0,
        "avg_steps": mean(steps) if steps else 0.0,
        "diagnostics": diagnostics,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Train the scientist agent by repeating local offline ARC prize-shaped sessions and carrying learned state across sessions."
    )
    parser.add_argument("--game-id", type=str, default="ar25-0c556536")
    parser.add_argument("--mode", type=str, default="offline")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sessions", "--episodes", dest="sessions", type=int, default=24)
    parser.add_argument("--eval-every", type=int, default=6)
    parser.add_argument("--eval-sessions", "--eval-episodes", dest="eval_sessions", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=256)
    parser.add_argument("--checkpoint-path", type=str, default="artifacts/scientist_arc_offline_best.pkl")
    parser.add_argument("--latest-checkpoint-path", type=str, default="artifacts/scientist_arc_offline_latest.pkl")
    parser.add_argument("--init-checkpoint-path", type=str, default="")
    parser.add_argument(
        "--allow-sparse-click-smoke",
        action="store_true",
        help="allow ARCAGI_SPARSE_CLICKS_BASELINE=1 for explicitly invalid smoke/debug runs",
    )
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    train_arc_offline(
        ScientistArcTrainingConfig(
            game_id=args.game_id,
            mode=args.mode,
            seed=args.seed,
            sessions=args.sessions,
            eval_every=args.eval_every,
            eval_sessions=args.eval_sessions,
            max_steps=args.max_steps,
            checkpoint_path=args.checkpoint_path,
            latest_checkpoint_path=args.latest_checkpoint_path,
            init_checkpoint_path=args.init_checkpoint_path,
            allow_sparse_click_smoke=bool(args.allow_sparse_click_smoke),
        )
    )


def _run_arc_episode(
    game_id: str,
    *,
    agent: HyperGeneralizingScientistAgent,
    operation_mode: Any,
    seed: int,
    max_steps: int,
):
    env = ArcToolkitEnv(game_id, operation_mode=operation_mode)
    try:
        return run_episode(env, agent, max_steps=max_steps, seed=seed)
    finally:
        env.close()


def _mean_bool(values) -> float:
    values = list(values)
    return mean(float(v) for v in values) if values else 0.0


def _result_key(result: dict[str, object]) -> tuple[float, float, float, float]:
    success = float(result.get("session_win_rate", result.get("success_rate", 0.0)))
    avg_levels_completed = float(result.get("avg_levels_completed", 0.0))
    avg_return = float(result.get("avg_return", 0.0))
    avg_steps = float(result.get("avg_steps", 0.0))
    return success, avg_levels_completed, avg_return, -avg_steps


if __name__ == "__main__":
    main()
