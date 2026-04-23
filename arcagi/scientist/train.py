"""Synthetic curriculum training for the spotlight scientist agent.

The explicit executive protocol stays hand-designed. The action scorer inside
that protocol is trained through repeated synthetic sessions that require hidden
binder discovery, fail/reset/retry loops, and cross-level carryover.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean

from arcagi.agents.scientist_agent import (
    HyperGeneralizingScientistAgent,
    load_spotlight_scientist_checkpoint,
    save_spotlight_scientist_checkpoint,
)
from arcagi.core.types import GridObservation, StepResult
from arcagi.envs.session import PersistentLevelSessionEnv
from arcagi.envs.synthetic import HiddenRuleEnv, family_variants_for_mode
from arcagi.training.synthetic_oracle import teacher_action as synthetic_teacher_action

from . import HiddenRuleGridEnv, SyntheticConfig, run_episode
from .runtime import EpisodeResult


@dataclass(frozen=True)
class ScientistTrainingConfig:
    seed: int = 0
    stage1_sessions: int = 64
    stage2_sessions: int = 128
    eval_every: int = 24
    simple_max_steps: int = 80
    rich_level_max_steps: int = 40
    rich_session_max_steps: int = 192
    rich_levels_per_session: int = 2
    holdout_simple_sessions: int = 8
    holdout_rich_sessions: int = 6
    checkpoint_path: str = "artifacts/scientist_curriculum_best.pkl"
    latest_checkpoint_path: str = "artifacts/scientist_curriculum_latest.pkl"
    init_checkpoint_path: str = ""


def train_scientist(config: ScientistTrainingConfig) -> dict[str, object]:
    agent = (
        load_spotlight_scientist_checkpoint(config.init_checkpoint_path)
        if config.init_checkpoint_path and Path(config.init_checkpoint_path).exists()
        else HyperGeneralizingScientistAgent()
    )
    rich_templates = _rich_session_templates()
    best_key = tuple(float("-inf") for _ in range(7))
    training_rows: list[dict[str, object]] = []

    total_sessions = config.stage1_sessions + config.stage2_sessions
    for session_idx in range(total_sessions):
        seed = config.seed + session_idx
        if session_idx < config.stage1_sessions:
            stage = "simple"
            env = _build_simple_session_env(
                seed=seed,
                config=config,
                size=7 + (session_idx % 3),
                requires_key=(session_idx % 4) != 1,
            )
            result = _run_teacher_session(env, agent, max_steps=_simple_session_max_steps(config), seed=seed)
            close = getattr(env, "close", None)
            if callable(close):
                close()
        else:
            stage = "rich_session"
            template = rich_templates[(session_idx - config.stage1_sessions) % len(rich_templates)]
            env = _build_rich_session_env(template=template, seed=seed, config=config)
            result = _run_teacher_session(env, agent, max_steps=config.rich_session_max_steps, seed=seed)
            close = getattr(env, "close", None)
            if callable(close):
                close()

        row = {
            "session": session_idx + 1,
            "stage": stage,
            "seed": seed,
            "return": float(result.total_reward),
            "steps": int(result.steps),
            "terminated": bool(result.terminated),
            "won": bool(result.won),
            "levels_completed": int(result.levels_completed),
            "reset_steps": int(result.reset_steps),
            "attempt_improvement": float(_spotlight_metric(result, "avg_attempt_improvement")),
        }
        training_rows.append(row)

        should_eval = ((session_idx + 1) % max(config.eval_every, 1) == 0) or (session_idx + 1 == total_sessions)
        if not should_eval:
            continue

        holdout = evaluate_scientist(agent, config=config, rich_templates=rich_templates)
        latest_payload = {
            "event": "scientist_train_eval",
            "session": session_idx + 1,
            "stage": stage,
            "recent_session_win_rate": _mean_bool(item["won"] for item in training_rows[-config.eval_every :]),
            "recent_avg_levels_completed": mean(float(item["levels_completed"]) for item in training_rows[-config.eval_every :]),
            "recent_avg_return": mean(float(item["return"]) for item in training_rows[-config.eval_every :]),
            "recent_avg_attempt_improvement": mean(float(item["attempt_improvement"]) for item in training_rows[-config.eval_every :]),
            "holdout": holdout,
            "checkpoint_path": config.checkpoint_path,
            "latest_checkpoint_path": config.latest_checkpoint_path,
        }
        save_spotlight_scientist_checkpoint(agent, config.latest_checkpoint_path)
        key = _result_key(holdout)
        if key >= best_key:
            best_key = key
            save_spotlight_scientist_checkpoint(agent, config.checkpoint_path)
            latest_payload["best_updated"] = True
        else:
            latest_payload["best_updated"] = False
        print(json.dumps(latest_payload, sort_keys=True))

    final_holdout = evaluate_scientist(agent, config=config, rich_templates=rich_templates)
    save_spotlight_scientist_checkpoint(agent, config.latest_checkpoint_path)
    summary = {
        "config": asdict(config),
        "sessions": total_sessions,
        "train_session_win_rate": _mean_bool(item["won"] for item in training_rows),
        "train_avg_levels_completed": mean(float(item["levels_completed"]) for item in training_rows) if training_rows else 0.0,
        "train_avg_return": mean(float(item["return"]) for item in training_rows) if training_rows else 0.0,
        "train_avg_attempt_improvement": mean(float(item["attempt_improvement"]) for item in training_rows) if training_rows else 0.0,
        "final_holdout": final_holdout,
        "best_checkpoint_path": config.checkpoint_path,
        "latest_checkpoint_path": config.latest_checkpoint_path,
    }
    print(json.dumps(summary, indent=2, sort_keys=True))
    return summary


def evaluate_scientist(
    agent: HyperGeneralizingScientistAgent,
    *,
    config: ScientistTrainingConfig,
    rich_templates: tuple[tuple[tuple[str, str], ...], ...] | None = None,
) -> dict[str, object]:
    snapshot = agent.state_dict()
    templates = rich_templates or _rich_session_templates()

    simple_returns: list[float] = []
    simple_wins: list[bool] = []
    simple_attempt_improvements: list[float] = []
    for idx in range(config.holdout_simple_sessions):
        seed = config.seed + 10_000 + idx
        eval_agent = HyperGeneralizingScientistAgent()
        eval_agent.load_state_dict(snapshot)
        env = _build_simple_session_env(
            seed=seed,
            config=config,
            size=7 + (idx % 3),
            requires_key=(idx % 4) != 1,
        )
        result = run_episode(env, eval_agent, max_steps=_simple_session_max_steps(config), seed=seed)
        close = getattr(env, "close", None)
        if callable(close):
            close()
        simple_returns.append(float(result.total_reward))
        simple_wins.append(bool(result.won or result.total_reward >= 0.95))
        simple_attempt_improvements.append(float(_spotlight_metric(result, "avg_attempt_improvement")))

    rich_returns: list[float] = []
    rich_wins: list[bool] = []
    rich_levels_completed: list[int] = []
    rich_reset_steps: list[int] = []
    rich_attempt_improvements: list[float] = []
    for idx, template in enumerate(templates[: config.holdout_rich_sessions]):
        seed = config.seed + 20_000 + idx
        eval_agent = HyperGeneralizingScientistAgent()
        eval_agent.load_state_dict(snapshot)
        env = _build_rich_session_env(template=template, seed=seed, config=config)
        result = run_episode(env, eval_agent, max_steps=config.rich_session_max_steps, seed=seed)
        close = getattr(env, "close", None)
        if callable(close):
            close()
        rich_returns.append(float(result.total_reward))
        rich_wins.append(bool(result.won))
        rich_levels_completed.append(int(result.levels_completed))
        rich_reset_steps.append(int(result.reset_steps))
        rich_attempt_improvements.append(float(_spotlight_metric(result, "avg_attempt_improvement")))

    simple_success = _mean_bool(simple_wins)
    rich_session_win = _mean_bool(rich_wins)
    simple_return = mean(simple_returns) if simple_returns else 0.0
    rich_return = mean(rich_returns) if rich_returns else 0.0
    simple_attempt_improvement = mean(simple_attempt_improvements) if simple_attempt_improvements else 0.0
    rich_attempt_improvement = mean(rich_attempt_improvements) if rich_attempt_improvements else 0.0
    avg_levels_completed = mean(float(item) for item in rich_levels_completed) if rich_levels_completed else 0.0
    avg_reset_steps = mean(float(item) for item in rich_reset_steps) if rich_reset_steps else 0.0
    ranking_key = _result_key(
        {
            "rich_session_win_rate": rich_session_win,
            "rich_avg_levels_completed": avg_levels_completed,
            "rich_avg_attempt_improvement": rich_attempt_improvement,
            "simple_success_rate": simple_success,
            "simple_avg_attempt_improvement": simple_attempt_improvement,
            "rich_avg_return": rich_return,
            "simple_avg_return": simple_return,
        }
    )
    return {
        "simple_success_rate": simple_success,
        "simple_avg_return": simple_return,
        "simple_avg_attempt_improvement": simple_attempt_improvement,
        "rich_session_win_rate": rich_session_win,
        "rich_avg_levels_completed": avg_levels_completed,
        "rich_avg_attempt_improvement": rich_attempt_improvement,
        "rich_avg_return": rich_return,
        "rich_avg_reset_steps": avg_reset_steps,
        "ranking_key": list(ranking_key),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train the spotlight scientist via synthetic retry/carryover sessions.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--stage1-sessions", "--stage1-episodes", type=int, default=64, dest="stage1_sessions")
    parser.add_argument("--stage2-sessions", "--stage2-episodes", type=int, default=128, dest="stage2_sessions")
    parser.add_argument("--eval-every", type=int, default=24)
    parser.add_argument("--simple-max-steps", type=int, default=80)
    parser.add_argument("--rich-level-max-steps", type=int, default=40)
    parser.add_argument("--rich-session-max-steps", type=int, default=192)
    parser.add_argument("--rich-levels-per-session", type=int, default=2)
    parser.add_argument("--holdout-simple-sessions", "--holdout-simple-episodes", type=int, default=8, dest="holdout_simple_sessions")
    parser.add_argument("--holdout-rich-sessions", type=int, default=6)
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
            stage1_sessions=args.stage1_sessions,
            stage2_sessions=args.stage2_sessions,
            eval_every=args.eval_every,
            simple_max_steps=args.simple_max_steps,
            rich_level_max_steps=args.rich_level_max_steps,
            rich_session_max_steps=args.rich_session_max_steps,
            rich_levels_per_session=args.rich_levels_per_session,
            holdout_simple_sessions=args.holdout_simple_sessions,
            holdout_rich_sessions=args.holdout_rich_sessions,
            checkpoint_path=args.checkpoint_path,
            latest_checkpoint_path=args.latest_checkpoint_path,
            init_checkpoint_path=args.init_checkpoint_path,
        )
    )


def _run_teacher_session(
    env,
    agent: HyperGeneralizingScientistAgent,
    *,
    max_steps: int,
    seed: int | None,
) -> EpisodeResult:
    agent.reset_episode()
    try:
        observation = env.reset(seed=seed)
    except TypeError:
        observation = env.reset()
    if isinstance(observation, tuple):
        observation = observation[0]

    total_reward = 0.0
    terminated = False
    won = False
    reset_steps = 0
    levels_completed = _levels_completed(observation)
    final_info: dict[str, object] = {}
    steps = 0

    for step in range(max_steps):
        action = agent.act(observation)
        teacher = synthetic_teacher_action(env, observation=observation)
        if teacher and teacher != "wait":
            agent.observe_teacher_action(teacher)
            if agent.trace:
                agent.trace[-1]["teacher_action"] = str(teacher)
                agent.trace[-1]["teacher_agree"] = bool(action == teacher)
        if str(action) == "0":
            reset_steps += 1
        next_observation, reward, terminated, info = _unpack_step_result(env.step(action))
        agent.observe_result(
            action=action,
            before_observation=observation,
            after_observation=next_observation,
            reward=reward,
            terminated=terminated,
            info=info,
        )
        total_reward += float(reward)
        observation = next_observation
        steps = step + 1
        final_info = dict(info)
        levels_completed = max(levels_completed, _levels_completed(next_observation))
        won = won or _is_win_state(next_observation)
        if won:
            terminated = True
            break
        if terminated and not _should_continue_after_terminal(next_observation):
            break

    return EpisodeResult(
        steps=steps,
        total_reward=total_reward,
        terminated=terminated,
        won=won,
        reset_steps=reset_steps,
        levels_completed=levels_completed,
        final_info=final_info,
        diagnostics=agent.diagnostics(),
    )


def _mean_bool(values) -> float:
    values = list(values)
    return mean(float(v) for v in values) if values else 0.0


def _unpack_step_result(result) -> tuple[object, float, bool, dict[str, object]]:
    if hasattr(result, "observation") or hasattr(result, "next_observation"):
        observation = getattr(result, "observation", None)
        if observation is None:
            observation = getattr(result, "next_observation")
        reward = float(getattr(result, "reward", 0.0))
        terminated = bool(getattr(result, "terminated", False) or getattr(result, "done", False))
        info = getattr(result, "info", None) or getattr(result, "extras", None) or {}
        return observation, reward, terminated, dict(info)
    if isinstance(result, tuple) and len(result) == 5:
        observation, reward, terminated, truncated, info = result
        return observation, float(reward), bool(terminated or truncated), dict(info or {})
    if isinstance(result, tuple) and len(result) == 4:
        observation, reward, done, info = result
        return observation, float(reward), bool(done), dict(info or {})
    raise TypeError(f"unsupported env.step result: {type(result)!r}")


def _levels_completed(observation) -> int:
    extras = getattr(observation, "extras", None)
    if isinstance(extras, dict):
        try:
            return int(extras.get("levels_completed", 0) or 0)
        except Exception:
            return 0
    return 0


def _is_win_state(observation) -> bool:
    extras = getattr(observation, "extras", None)
    if isinstance(extras, dict):
        return str(extras.get("game_state", "") or "").strip().upper().endswith("WIN")
    return False


def _should_continue_after_terminal(observation) -> bool:
    available_actions = tuple(str(item) for item in getattr(observation, "available_actions", ()) or ())
    extras = getattr(observation, "extras", None)
    if "0" not in available_actions:
        return False
    if not isinstance(extras, dict):
        return False
    game_state = str(extras.get("game_state", "") or "").strip().upper()
    return game_state.endswith("GAME_OVER") or game_state.endswith("SESSION_ENDED")


def _result_key(metrics: dict[str, float]) -> tuple[float, float, float, float, float, float, float]:
    return (
        float(metrics.get("rich_session_win_rate", 0.0)),
        float(metrics.get("rich_avg_levels_completed", 0.0)),
        float(metrics.get("rich_avg_attempt_improvement", 0.0)),
        float(metrics.get("simple_success_rate", 0.0)),
        float(metrics.get("simple_avg_attempt_improvement", 0.0)),
        float(metrics.get("rich_avg_return", 0.0)),
        float(metrics.get("simple_avg_return", 0.0)),
    )


def _spotlight_metric(result: EpisodeResult, key: str) -> float:
    diagnostics = result.diagnostics if isinstance(result.diagnostics, dict) else {}
    spotlight = diagnostics.get("spotlight", {}) if isinstance(diagnostics, dict) else {}
    try:
        return float(spotlight.get(key, 0.0))
    except Exception:
        return 0.0


def _simple_session_max_steps(config: ScientistTrainingConfig) -> int:
    return max(config.simple_max_steps, 2 * config.simple_max_steps)


class _SimpleSessionLevel:
    def __init__(self, env: HiddenRuleGridEnv) -> None:
        self.teacher_env = env

    def legal_actions(self):
        return tuple(str(item) for item in self.teacher_env.available_actions)

    def reset(self, seed=None):
        observation = self.teacher_env.reset(seed=seed)
        return self._to_observation(observation)

    def step(self, action):
        observation, reward, done, info = self.teacher_env.step(str(action))
        return StepResult(
            observation=self._to_observation(observation),
            reward=float(reward),
            terminated=bool(done),
            truncated=False,
            info=dict(info or {}),
        )

    @staticmethod
    def _to_observation(frame) -> GridObservation:
        return GridObservation(
            task_id=frame.task_id,
            episode_id=frame.episode_id,
            step_index=int(frame.step_index),
            grid=frame.grid,
            available_actions=tuple(str(item) for item in frame.available_actions),
            extras=dict(frame.extras),
        )


def _build_simple_session_env(
    *,
    seed: int,
    config: ScientistTrainingConfig,
    size: int,
    requires_key: bool,
) -> PersistentLevelSessionEnv:
    return PersistentLevelSessionEnv(
        level_builders=(
            lambda level_seed: _SimpleSessionLevel(
                HiddenRuleGridEnv(
                    SyntheticConfig(
                        size=size,
                        requires_key=requires_key,
                        seed=level_seed,
                        max_steps=config.simple_max_steps,
                    )
                )
            ),
        ),
        task_id="synthetic/simple_session",
        family_id=f"synthetic/simple_session/{size}_{int(requires_key)}",
        seed=seed,
    )


def _build_rich_session_env(
    *,
    template: tuple[tuple[str, str], ...],
    seed: int,
    config: ScientistTrainingConfig,
) -> PersistentLevelSessionEnv:
    builders = []
    for level_mode, level_variant in template[: config.rich_levels_per_session]:
        builders.append(
            lambda level_seed, mode=level_mode, variant=level_variant: HiddenRuleEnv(
                family_mode=mode,
                family_variant=variant,
                seed=level_seed,
                max_steps=config.rich_level_max_steps,
            )
        )
    family = "__".join(f"{mode}:{variant}" for mode, variant in template[: config.rich_levels_per_session])
    return PersistentLevelSessionEnv(
        level_builders=tuple(builders),
        task_id="synthetic/spotlight_session",
        family_id=f"synthetic_session/{family}",
        seed=seed,
    )


def _rich_session_templates() -> tuple[tuple[tuple[str, str], ...], ...]:
    templates: list[tuple[tuple[str, str], ...]] = []
    for gate in family_variants_for_mode("selector_unlock"):
        for order in family_variants_for_mode("delayed_order_unlock"):
            templates.append(
                (
                    ("selector_unlock", gate),
                    ("selector_sequence_unlock", f"{gate}__{order}"),
                )
            )
        templates.append(
            (
                ("selector_unlock", gate),
                ("selector_unlock", gate),
            )
        )
    return tuple(templates)


if __name__ == "__main__":
    main()
