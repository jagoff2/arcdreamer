from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from arcagi.agents.graph_agent import GraphExplorerAgent
from arcagi.agents.random_agent import RandomHeuristicAgent
from arcagi.envs.arc_adapter import Arcade, ArcToolkitEnv, arc_operation_mode, arc_toolkit_available, list_arc_games
from arcagi.envs.synthetic import DEFAULT_SYNTHETIC_FAMILY_MODES, HiddenRuleEnv, family_variants_for_mode


def build_agent(agent_name: str, checkpoint_path: str | None = None, device: Any = None):
    normalized = agent_name.strip().lower()
    if normalized == "random":
        return RandomHeuristicAgent()
    if normalized == "graph":
        return GraphExplorerAgent()
    if normalized in {"scientist", "hyper_scientist", "hyper-generalizing-scientist", "spotlight", "spotlight_scientist"}:
        from arcagi.agents.scientist_agent import (
            HyperGeneralizingScientistAgent,
            load_spotlight_scientist_checkpoint,
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            return load_spotlight_scientist_checkpoint(checkpoint_path)
        return HyperGeneralizingScientistAgent()

    import torch

    from arcagi.agents.learned_agent import HybridAgent, LanguageNoMemoryAgent, RecurrentAblationAgent
    from arcagi.memory.episodic import EpisodicMemory
    from arcagi.planning.planner import HybridPlanner, PlannerConfig
    from arcagi.training.synthetic import build_default_modules, load_checkpoint

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path and Path(checkpoint_path).exists():
        encoder, world_model, language_model = load_checkpoint(checkpoint_path, device=device)
    else:
        encoder, world_model, language_model, _planner = build_default_modules(device=device)

    planner = HybridPlanner(
        PlannerConfig(search_depth=2, search_root_width=2, search_branch_width=1, max_world_model_calls=48)
        if device.type == "cpu"
        else PlannerConfig()
    )
    if normalized == "recurrent":
        return RecurrentAblationAgent(encoder=encoder, world_model=world_model, planner=planner, device=device)
    if normalized == "language":
        return LanguageNoMemoryAgent(
            encoder=encoder,
            world_model=world_model,
            planner=planner,
            language_model=language_model,
            device=device,
        )
    if normalized == "hybrid":
        return HybridAgent(
            encoder=encoder,
            world_model=world_model,
            planner=planner,
            language_model=language_model,
            episodic_memory=EpisodicMemory(),
            device=device,
        )
    raise ValueError(f"unknown agent: {agent_name}")


def _reset_agent_for_episode(agent: Any) -> None:
    reset_episode = getattr(agent, "reset_episode", None)
    if callable(reset_episode):
        reset_episode()


def _reset_agent_for_family(agent: Any) -> None:
    reset_all = getattr(agent, "reset_all", None)
    if callable(reset_all):
        reset_all()
        return
    _reset_agent_for_episode(agent)


def _observe_step(agent: Any, *, action: str, before: Any, result: Any) -> None:
    update_after_step = getattr(agent, "update_after_step", None)
    if callable(update_after_step):
        update_after_step(
            next_observation=result.observation,
            reward=result.reward,
            terminated=result.terminated or result.truncated,
            info=result.info,
        )
        return
    observe_result = getattr(agent, "observe_result", None)
    if callable(observe_result):
        observe_result(
            action=action,
            before_observation=before,
            after_observation=result.observation,
            reward=result.reward,
            terminated=result.terminated or result.truncated,
            info=result.info,
        )


def run_episode(
    agent: Any,
    env: Any,
    seed: int,
    max_steps: int | None = None,
    stop_on_positive_reward: bool = False,
    progress_every: int = 0,
) -> dict[str, object]:
    observation = env.reset(seed=seed)
    _reset_agent_for_episode(agent)
    steps = 0
    success = False
    won = False
    rewards: list[float] = []
    interaction_steps = 0
    reset_steps = 0
    levels_completed = _levels_completed(observation)
    language_traces: list[str] = []
    done = False
    while not done and (max_steps is None or steps < max_steps):
        before = observation
        action = agent.act(observation)
        action_text = str(action)
        if action_text.startswith("interact") or action_text.startswith("click:"):
            interaction_steps += 1
        if action_text == "0":
            reset_steps += 1
        result = env.step(action)
        _observe_step(agent, action=action_text, before=before, result=result)
        observation = result.observation
        steps += 1
        rewards.append(float(result.reward))
        levels_completed = max(levels_completed, _levels_completed(observation))
        won = won or _is_win_state(observation)
        if result.reward > 0.9:
            success = True
            if stop_on_positive_reward:
                done = True
        if won:
            success = True
            done = True
        if not done:
            done = bool(result.terminated or result.truncated) and not _should_continue_after_terminal(observation)
        if getattr(agent, "latest_language", ()):
            language_traces.append(" ".join(str(x) for x in agent.latest_language))
        if progress_every > 0 and steps % progress_every == 0:
            print(
                json.dumps(
                    {
                        "event": "arc_runtime_progress",
                        "family_id": getattr(env, "family_id", getattr(env, "task_id", "unknown")),
                        "steps": steps,
                        "return": float(sum(rewards)),
                        "interaction_steps": interaction_steps,
                        "reset_steps": reset_steps,
                        "levels_completed": levels_completed,
                        "game_state": _game_state(observation),
                        "last_action": action_text,
                    },
                    sort_keys=True,
                )
            )
    return {
        "success": success,
        "won": won,
        "return": float(sum(rewards)),
        "steps": steps,
        "interaction_steps": interaction_steps,
        "reset_steps": reset_steps,
        "levels_completed": levels_completed,
        "game_state": _game_state(observation),
        "language": language_traces[-1] if language_traces else "",
        "family_id": getattr(env, "family_id", getattr(env, "task_id", "unknown")),
        "diagnostics": getattr(agent, "diagnostics", lambda: {})(),
    }


def evaluate_synthetic(
    agent_name: str,
    checkpoint_path: str | None = None,
    episodes_per_family: int = 3,
    seed: int = 17,
) -> dict[str, object]:
    agent = build_agent(agent_name, checkpoint_path=checkpoint_path)
    families: list[dict[str, object]] = []
    seed_cursor = seed
    for family_mode in DEFAULT_SYNTHETIC_FAMILY_MODES:
        for variant in family_variants_for_mode(family_mode):
            _reset_agent_for_family(agent)
            episode_metrics = []
            for episode_idx in range(episodes_per_family):
                env = HiddenRuleEnv(
                    family_mode=family_mode,
                    family_variant=variant,
                    seed=seed_cursor + episode_idx,
                )
                episode_metrics.append(run_episode(agent, env, seed=seed_cursor + episode_idx))
            seed_cursor += episodes_per_family
            families.append(
                {
                    "family_mode": family_mode,
                    "family_variant": variant,
                    "episodes": episode_metrics,
                }
            )
    all_episodes = [episode for family in families for episode in family["episodes"]]
    first_episode_success = [family["episodes"][0]["success"] for family in families]
    later_episode_success = [episode["success"] for family in families for episode in family["episodes"][1:]]
    return {
        "agent": agent_name,
        "success_rate": mean(float(item["success"]) for item in all_episodes) if all_episodes else 0.0,
        "avg_return": mean(float(item["return"]) for item in all_episodes) if all_episodes else 0.0,
        "avg_steps": mean(float(item["steps"]) for item in all_episodes) if all_episodes else 0.0,
        "avg_interactions": mean(float(item["interaction_steps"]) for item in all_episodes) if all_episodes else 0.0,
        "first_episode_success": mean(float(value) for value in first_episode_success) if first_episode_success else 0.0,
        "later_episode_success": mean(float(value) for value in later_episode_success) if later_episode_success else 0.0,
        "families": families,
    }


def evaluate_arc(
    agent_name: str,
    checkpoint_path: str | None = None,
    game_limit: int = 3,
    mode: str = "offline",
    game_id: str | None = None,
    max_steps: int | None = 256,
    progress_every: int = 0,
) -> dict[str, object]:
    if not arc_toolkit_available():
        return {"agent": agent_name, "skipped": True, "reason": "ARC toolkit not installed"}
    operation_mode = arc_operation_mode(mode)
    if game_id:
        games = [game_id]
    else:
        games = list_arc_games(operation_mode=operation_mode)[:game_limit]
    agent = build_agent(agent_name, checkpoint_path=checkpoint_path)
    results = []
    shared_arcade = None if Arcade is None else Arcade(operation_mode=operation_mode)
    for index, game_id in enumerate(games):
        _reset_agent_for_family(agent)
        env = ArcToolkitEnv(game_id, operation_mode=operation_mode, arcade=shared_arcade)
        try:
            episode = run_episode(
                agent,
                env,
                seed=index,
                max_steps=max_steps,
                stop_on_positive_reward=False,
                progress_every=progress_every,
            )
        finally:
            env.close()
        results.append({"game_id": game_id, **episode})
    if shared_arcade is not None:
        close_scorecard = getattr(shared_arcade, "close_scorecard", None)
        if callable(close_scorecard):
            try:
                close_scorecard()
            except Exception:
                pass
    return {"agent": agent_name, "mode": mode, "games": results}


def _observation_extras(observation: Any) -> dict[str, Any]:
    extras = getattr(observation, "extras", None)
    return dict(extras) if isinstance(extras, dict) else {}


def _levels_completed(observation: Any) -> int:
    extras = _observation_extras(observation)
    try:
        return int(extras.get("levels_completed", getattr(observation, "levels_completed", 0)) or 0)
    except Exception:
        return 0


def _game_state(observation: Any) -> str:
    extras = _observation_extras(observation)
    return str(extras.get("game_state", getattr(observation, "state", "")) or "")


def _is_win_state(observation: Any) -> bool:
    return _game_state(observation).strip().upper().endswith("WIN")


def _should_continue_after_terminal(observation: Any) -> bool:
    available_actions = tuple(str(item) for item in getattr(observation, "available_actions", ()) or ())
    game_state = _game_state(observation).strip().upper()
    if "0" not in available_actions:
        return False
    return game_state.endswith("GAME_OVER") or game_state.endswith("SESSION_ENDED")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m arcagi.evaluation.harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthetic_parser = subparsers.add_parser("synthetic")
    synthetic_parser.add_argument("--agent", type=str, default="graph")
    synthetic_parser.add_argument("--checkpoint-path", type=str, default="")
    synthetic_parser.add_argument("--episodes-per-family", type=int, default=3)
    synthetic_parser.add_argument("--seed", type=int, default=17)
    arc_parser = subparsers.add_parser("arc")
    arc_parser.add_argument("--agent", type=str, default="graph")
    arc_parser.add_argument("--checkpoint-path", type=str, default="")
    arc_parser.add_argument("--game-limit", type=int, default=3)
    arc_parser.add_argument("--game-id", type=str, default="")
    arc_parser.add_argument("--max-steps", type=int, default=256)
    arc_parser.add_argument("--progress-every", type=int, default=0)
    arc_parser.add_argument("--mode", type=str, default="offline")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "synthetic":
        result = evaluate_synthetic(
            agent_name=args.agent,
            checkpoint_path=args.checkpoint_path or None,
            episodes_per_family=args.episodes_per_family,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "arc":
        result = evaluate_arc(
            agent_name=args.agent,
            checkpoint_path=args.checkpoint_path or None,
            game_limit=args.game_limit,
            game_id=args.game_id or None,
            max_steps=None if int(args.max_steps) <= 0 else int(args.max_steps),
            progress_every=max(0, int(args.progress_every)),
            mode=args.mode,
        )
        print(json.dumps(result, indent=2))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
