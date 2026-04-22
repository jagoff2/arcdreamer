"""Synthetic curriculum training for the scientist agent.

The scientist stack does not use a large offline neural trainer. Its persistent
learned state is the online world model, which can be improved across episodes
by repeatedly solving synthetic environments while carrying world-model weights
forward. This module makes that process reproducible and checkpointable.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from arcagi.envs.synthetic import DEFAULT_SYNTHETIC_FAMILY_MODES, HiddenRuleEnv, family_variants_for_mode

from . import HiddenRuleGridEnv, ScientistAgent, SyntheticConfig, load_scientist_checkpoint, run_episode, save_scientist_checkpoint


@dataclass(frozen=True)
class ScientistTrainingConfig:
    seed: int = 0
    stage1_episodes: int = 96
    stage2_episodes: int = 192
    eval_every: int = 32
    simple_max_steps: int = 80
    rich_max_steps: int = 48
    holdout_simple_episodes: int = 12
    checkpoint_path: str = "artifacts/scientist_curriculum_best.pkl"
    latest_checkpoint_path: str = "artifacts/scientist_curriculum_latest.pkl"
    init_checkpoint_path: str = ""


def train_scientist(config: ScientistTrainingConfig) -> dict[str, object]:
    agent = (
        load_scientist_checkpoint(config.init_checkpoint_path)
        if config.init_checkpoint_path and Path(config.init_checkpoint_path).exists()
        else ScientistAgent()
    )
    rich_variants = _rich_variants()
    best_score = float("-inf")
    training_rows: list[dict[str, object]] = []

    total_episodes = config.stage1_episodes + config.stage2_episodes
    for episode_idx in range(total_episodes):
        seed = config.seed + episode_idx
        if episode_idx < config.stage1_episodes:
            stage = "simple"
            env = HiddenRuleGridEnv(
                SyntheticConfig(
                    size=7 + (episode_idx % 3),
                    requires_key=(episode_idx % 4) != 1,
                    seed=seed,
                    max_steps=config.simple_max_steps,
                )
            )
            result = run_episode(env, agent, max_steps=config.simple_max_steps, seed=seed)
        else:
            stage = "rich"
            mode, variant = rich_variants[(episode_idx - config.stage1_episodes) % len(rich_variants)]
            env = HiddenRuleEnv(
                family_mode=mode,
                family_variant=variant,
                seed=seed,
                max_steps=config.rich_max_steps,
            )
            result = run_episode(env, agent, max_steps=config.rich_max_steps, seed=seed)

        row = {
            "episode": episode_idx + 1,
            "stage": stage,
            "seed": seed,
            "return": float(result.total_reward),
            "steps": int(result.steps),
            "terminated": bool(result.terminated),
            "success": _episode_success(result.total_reward),
        }
        training_rows.append(row)

        should_eval = ((episode_idx + 1) % max(config.eval_every, 1) == 0) or (episode_idx + 1 == total_episodes)
        if not should_eval:
            continue

        holdout = evaluate_scientist(agent, config=config, rich_variants=rich_variants)
        latest_payload = {
            "event": "scientist_train_eval",
            "episode": episode_idx + 1,
            "stage": stage,
            "recent_success_rate": _mean_bool(item["success"] for item in training_rows[-config.eval_every :]),
            "recent_return": mean(float(item["return"]) for item in training_rows[-config.eval_every :]),
            "holdout": holdout,
            "checkpoint_path": config.checkpoint_path,
            "latest_checkpoint_path": config.latest_checkpoint_path,
        }
        save_scientist_checkpoint(agent, config.latest_checkpoint_path)
        if float(holdout["combined_score"]) >= best_score:
            best_score = float(holdout["combined_score"])
            save_scientist_checkpoint(agent, config.checkpoint_path)
            latest_payload["best_updated"] = True
        else:
            latest_payload["best_updated"] = False
        print(json.dumps(latest_payload, sort_keys=True))

    final_holdout = evaluate_scientist(agent, config=config, rich_variants=rich_variants)
    save_scientist_checkpoint(agent, config.latest_checkpoint_path)
    summary = {
        "config": asdict(config),
        "episodes": total_episodes,
        "train_success_rate": _mean_bool(item["success"] for item in training_rows),
        "train_avg_return": mean(float(item["return"]) for item in training_rows) if training_rows else 0.0,
        "final_holdout": final_holdout,
        "best_checkpoint_path": config.checkpoint_path,
        "latest_checkpoint_path": config.latest_checkpoint_path,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def evaluate_scientist(
    agent: ScientistAgent,
    *,
    config: ScientistTrainingConfig,
    rich_variants: tuple[tuple[str, str], ...] | None = None,
) -> dict[str, object]:
    snapshot = agent.state_dict()
    variants = rich_variants or _rich_variants()

    simple_returns: list[float] = []
    simple_successes: list[bool] = []
    for idx in range(config.holdout_simple_episodes):
        seed = config.seed + 10_000 + idx
        eval_agent = ScientistAgent()
        eval_agent.load_state_dict(snapshot)
        env = HiddenRuleGridEnv(
            SyntheticConfig(
                size=7 + (idx % 3),
                requires_key=(idx % 4) != 1,
                seed=seed,
                max_steps=config.simple_max_steps,
            )
        )
        result = run_episode(env, eval_agent, max_steps=config.simple_max_steps, seed=seed)
        simple_returns.append(float(result.total_reward))
        simple_successes.append(_episode_success(result.total_reward))

    rich_returns: list[float] = []
    rich_successes: list[bool] = []
    for idx, (mode, variant) in enumerate(variants):
        seed = config.seed + 20_000 + idx
        eval_agent = ScientistAgent()
        eval_agent.load_state_dict(snapshot)
        env = HiddenRuleEnv(
            family_mode=mode,
            family_variant=variant,
            seed=seed,
            max_steps=config.rich_max_steps,
        )
        result = run_episode(env, eval_agent, max_steps=config.rich_max_steps, seed=seed)
        rich_returns.append(float(result.total_reward))
        rich_successes.append(_episode_success(result.total_reward))

    simple_success = _mean_bool(simple_successes)
    rich_success = _mean_bool(rich_successes)
    simple_return = mean(simple_returns) if simple_returns else 0.0
    rich_return = mean(rich_returns) if rich_returns else 0.0
    combined_score = (20.0 * simple_success) + (80.0 * rich_success) + simple_return + rich_return
    return {
        "simple_success_rate": simple_success,
        "simple_avg_return": simple_return,
        "rich_success_rate": rich_success,
        "rich_avg_return": rich_return,
        "combined_score": combined_score,
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the scientist agent via repeated synthetic online-learning episodes.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage1-episodes", type=int, default=96)
    parser.add_argument("--stage2-episodes", type=int, default=192)
    parser.add_argument("--eval-every", type=int, default=32)
    parser.add_argument("--simple-max-steps", type=int, default=80)
    parser.add_argument("--rich-max-steps", type=int, default=48)
    parser.add_argument("--holdout-simple-episodes", type=int, default=12)
    parser.add_argument("--checkpoint-path", type=str, default="artifacts/scientist_curriculum_best.pkl")
    parser.add_argument("--latest-checkpoint-path", type=str, default="artifacts/scientist_curriculum_latest.pkl")
    parser.add_argument("--init-checkpoint-path", type=str, default="")
    return parser


def main(argv: list[str] | None = None) -> None:
    parser = build_parser()
    args = parser.parse_args(argv)
    train_scientist(
        ScientistTrainingConfig(
            seed=args.seed,
            stage1_episodes=args.stage1_episodes,
            stage2_episodes=args.stage2_episodes,
            eval_every=args.eval_every,
            simple_max_steps=args.simple_max_steps,
            rich_max_steps=args.rich_max_steps,
            holdout_simple_episodes=args.holdout_simple_episodes,
            checkpoint_path=args.checkpoint_path,
            latest_checkpoint_path=args.latest_checkpoint_path,
            init_checkpoint_path=args.init_checkpoint_path,
        )
    )


def _episode_success(total_reward: float) -> bool:
    return float(total_reward) >= 0.95


def _mean_bool(values) -> float:
    values = list(values)
    return mean(float(v) for v in values) if values else 0.0


def _rich_variants() -> tuple[tuple[str, str], ...]:
    pairs: list[tuple[str, str]] = []
    for mode in DEFAULT_SYNTHETIC_FAMILY_MODES:
        for variant in family_variants_for_mode(mode):
            pairs.append((mode, variant))
    return tuple(pairs)


if __name__ == "__main__":
    main()
