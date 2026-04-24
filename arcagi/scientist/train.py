"""Synthetic curriculum training for the spotlight scientist agent.

The agent executes its own actions during collection.  Synthetic oracle access is
training instrumentation only: by default it is sampled sparsely as a relabeling
signal for learner-visited states, then model selection is gated by teacher-free
holdout solve metrics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from statistics import mean
from typing import Literal

import numpy as np

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
    allow_unproven_best_checkpoint: bool = False
    teacher_feedback_mode: Literal["none", "sparse_disagreement", "dense"] = "sparse_disagreement"
    teacher_query_probability: float = 0.35
    teacher_feedback_probability: float = 0.05
    teacher_disagreement_probability: float = 0.45
    teacher_feedback_weight: float = 0.35


def train_scientist(config: ScientistTrainingConfig) -> dict[str, object]:
    agent = (
        load_spotlight_scientist_checkpoint(config.init_checkpoint_path)
        if config.init_checkpoint_path and Path(config.init_checkpoint_path).exists()
        else HyperGeneralizingScientistAgent()
    )
    rich_templates = _rich_session_templates()
    best_key = tuple(float("-inf") for _ in range(8))
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
            result = _run_training_session(env, agent, max_steps=_simple_session_max_steps(config), seed=seed, config=config)
            close = getattr(env, "close", None)
            if callable(close):
                close()
        else:
            stage = "rich_session"
            template = rich_templates[(session_idx - config.stage1_sessions) % len(rich_templates)]
            env = _build_rich_session_env(template=template, seed=seed, config=config)
            result = _run_training_session(env, agent, max_steps=config.rich_session_max_steps, seed=seed, config=config)
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
            "teacher_queries": float(_diagnostic_metric(result, "training_teacher_queries")),
            "teacher_feedback_steps": float(_diagnostic_metric(result, "training_teacher_feedback_steps")),
            "teacher_agreement_rate": float(_diagnostic_metric(result, "training_teacher_agreement_rate")),
        }
        training_rows.append(row)

        should_eval = ((session_idx + 1) % max(config.eval_every, 1) == 0) or (session_idx + 1 == total_sessions)
        if not should_eval:
            continue

        holdout = evaluate_scientist(agent, config=config, rich_templates=rich_templates)
        train_recent = {
            "learner_owned_session_win_rate": _mean_bool(item["won"] for item in training_rows[-config.eval_every :]),
            "learner_owned_avg_levels_completed": mean(
                float(item["levels_completed"]) for item in training_rows[-config.eval_every :]
            ),
            "learner_owned_avg_return": mean(float(item["return"]) for item in training_rows[-config.eval_every :]),
            "learner_owned_avg_reset_steps": mean(float(item["reset_steps"]) for item in training_rows[-config.eval_every :]),
            "learner_owned_avg_attempt_improvement": mean(
                float(item["attempt_improvement"]) for item in training_rows[-config.eval_every :]
            ),
            "teacher_feedback_fraction": _feedback_fraction(training_rows[-config.eval_every :]),
            "teacher_query_fraction": _query_fraction(training_rows[-config.eval_every :]),
            "teacher_agreement_rate": _weighted_agreement(training_rows[-config.eval_every :]),
        }
        promotion_eligible = _holdout_has_autonomous_solve(holdout)
        checkpoint_metadata = _checkpoint_metadata(
            config=config,
            session=session_idx + 1,
            stage=stage,
            holdout=holdout,
            train_recent=train_recent,
            promotion_eligible=promotion_eligible,
            promoted=False,
            agent=agent,
        )
        latest_payload = {
            "event": "scientist_train_eval",
            "session": session_idx + 1,
            "stage": stage,
            "spotlight_feature_schema_version": _spotlight_feature_schema_version(agent),
            **train_recent,
            "autonomous_holdout": holdout,
            "promotion_eligible": promotion_eligible,
            "allow_unproven_best_checkpoint": bool(config.allow_unproven_best_checkpoint),
            "checkpoint_path": config.checkpoint_path,
            "latest_checkpoint_path": config.latest_checkpoint_path,
        }
        save_spotlight_scientist_checkpoint(agent, config.latest_checkpoint_path, metadata=checkpoint_metadata)
        key = _result_key(holdout)
        can_promote = promotion_eligible or bool(config.allow_unproven_best_checkpoint)
        if can_promote and key > best_key:
            best_key = key
            promoted_metadata = dict(checkpoint_metadata)
            promoted_metadata["promoted"] = True
            save_spotlight_scientist_checkpoint(agent, config.checkpoint_path, metadata=promoted_metadata)
            latest_payload["best_updated"] = True
        else:
            latest_payload["best_updated"] = False
            if not promotion_eligible:
                _write_unpromoted_sidecar(config.checkpoint_path, checkpoint_metadata)
        print(json.dumps(latest_payload, sort_keys=True))

    final_holdout = evaluate_scientist(agent, config=config, rich_templates=rich_templates)
    final_train_summary = {
        "learner_owned_session_win_rate": _mean_bool(item["won"] for item in training_rows),
        "learner_owned_avg_levels_completed": mean(float(item["levels_completed"]) for item in training_rows) if training_rows else 0.0,
        "learner_owned_avg_return": mean(float(item["return"]) for item in training_rows) if training_rows else 0.0,
        "learner_owned_avg_reset_steps": mean(float(item["reset_steps"]) for item in training_rows) if training_rows else 0.0,
        "learner_owned_avg_attempt_improvement": mean(float(item["attempt_improvement"]) for item in training_rows) if training_rows else 0.0,
        "teacher_feedback_fraction": _feedback_fraction(training_rows),
        "teacher_query_fraction": _query_fraction(training_rows),
        "teacher_agreement_rate": _weighted_agreement(training_rows),
    }
    final_promotion_eligible = _holdout_has_autonomous_solve(final_holdout)
    final_metadata = _checkpoint_metadata(
        config=config,
        session=total_sessions,
        stage="final",
        holdout=final_holdout,
        train_recent=final_train_summary,
        promotion_eligible=final_promotion_eligible,
        promoted=False,
        agent=agent,
    )
    save_spotlight_scientist_checkpoint(agent, config.latest_checkpoint_path, metadata=final_metadata)
    summary = {
        "config": asdict(config),
        "spotlight_feature_schema_version": _spotlight_feature_schema_version(agent),
        "sessions": total_sessions,
        **final_train_summary,
        "final_holdout": final_holdout,
        "final_promotion_eligible": final_promotion_eligible,
        "best_checkpoint_valid": bool(best_key > tuple(float("-inf") for _ in range(8))),
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
            "rich_avg_reset_steps": avg_reset_steps,
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
    parser.add_argument(
        "--allow-unproven-best-checkpoint",
        action="store_true",
        help="Write a best checkpoint even when autonomous holdout solve metrics are all zero. Intended only for debugging.",
    )
    parser.add_argument(
        "--teacher-feedback-mode",
        choices=("none", "sparse_disagreement", "dense"),
        default=ScientistTrainingConfig.teacher_feedback_mode,
        help="Oracle relabeling mode. The agent always executes its own actions; dense reproduces old per-step habit labels.",
    )
    parser.add_argument("--teacher-query-probability", type=float, default=ScientistTrainingConfig.teacher_query_probability)
    parser.add_argument("--teacher-feedback-probability", type=float, default=ScientistTrainingConfig.teacher_feedback_probability)
    parser.add_argument("--teacher-disagreement-probability", type=float, default=ScientistTrainingConfig.teacher_disagreement_probability)
    parser.add_argument("--teacher-feedback-weight", type=float, default=ScientistTrainingConfig.teacher_feedback_weight)
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
            allow_unproven_best_checkpoint=args.allow_unproven_best_checkpoint,
            teacher_feedback_mode=args.teacher_feedback_mode,
            teacher_query_probability=args.teacher_query_probability,
            teacher_feedback_probability=args.teacher_feedback_probability,
            teacher_disagreement_probability=args.teacher_disagreement_probability,
            teacher_feedback_weight=args.teacher_feedback_weight,
        )
    )


def _run_training_session(
    env,
    agent: HyperGeneralizingScientistAgent,
    *,
    max_steps: int,
    seed: int | None,
    config: ScientistTrainingConfig,
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
    rng = np.random.default_rng(seed)
    teacher_queries = 0
    teacher_feedback_steps = 0
    teacher_agreements = 0

    for step in range(max_steps):
        action = agent.act(observation)
        teacher = ""
        query_teacher = _should_query_teacher(
            mode=config.teacher_feedback_mode,
            rng=rng,
            query_probability=config.teacher_query_probability,
        )
        if query_teacher:
            teacher_queries += 1
            teacher = str(synthetic_teacher_action(env, observation=observation) or "")
        if teacher and teacher != "wait":
            agree = bool(action == teacher)
            teacher_agreements += int(agree)
            should_apply = _should_apply_teacher_feedback(
                mode=config.teacher_feedback_mode,
                agree=agree,
                rng=rng,
                base_probability=config.teacher_feedback_probability,
                disagreement_probability=config.teacher_disagreement_probability,
            )
            if should_apply:
                teacher_feedback_steps += 1
                agent.observe_teacher_action(teacher, weight=config.teacher_feedback_weight)
            if agent.trace:
                agent.trace[-1]["teacher_action"] = str(teacher)
                agent.trace[-1]["teacher_agree"] = agree
                agent.trace[-1]["teacher_feedback_applied"] = bool(should_apply)
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

    diagnostics = agent.diagnostics()
    diagnostics["training_teacher_queries"] = float(teacher_queries)
    diagnostics["training_teacher_feedback_steps"] = float(teacher_feedback_steps)
    diagnostics["training_teacher_agreements"] = float(teacher_agreements)
    diagnostics["training_teacher_query_fraction"] = float(teacher_queries / max(steps, 1))
    diagnostics["training_teacher_feedback_fraction"] = float(teacher_feedback_steps / max(steps, 1))
    diagnostics["training_teacher_agreement_rate"] = float(teacher_agreements / max(teacher_queries, 1))

    return EpisodeResult(
        steps=steps,
        total_reward=total_reward,
        terminated=terminated,
        won=won,
        reset_steps=reset_steps,
        levels_completed=levels_completed,
        final_info=final_info,
        diagnostics=diagnostics,
    )


def _mean_bool(values) -> float:
    values = list(values)
    return mean(float(v) for v in values) if values else 0.0


def _diagnostic_metric(result: EpisodeResult, key: str) -> float:
    diagnostics = result.diagnostics if isinstance(result.diagnostics, dict) else {}
    try:
        return float(diagnostics.get(key, 0.0))
    except Exception:
        return 0.0


def _feedback_fraction(rows: list[dict[str, object]]) -> float:
    steps = sum(float(row.get("steps", 0.0) or 0.0) for row in rows)
    feedback = sum(float(row.get("teacher_feedback_steps", 0.0) or 0.0) for row in rows)
    return float(feedback / steps) if steps > 0.0 else 0.0


def _query_fraction(rows: list[dict[str, object]]) -> float:
    steps = sum(float(row.get("steps", 0.0) or 0.0) for row in rows)
    queries = sum(float(row.get("teacher_queries", 0.0) or 0.0) for row in rows)
    return float(queries / steps) if steps > 0.0 else 0.0


def _weighted_agreement(rows: list[dict[str, object]]) -> float:
    queries = sum(float(row.get("teacher_queries", 0.0) or 0.0) for row in rows)
    if queries <= 0.0:
        return 0.0
    weighted = sum(
        float(row.get("teacher_agreement_rate", 0.0) or 0.0) * float(row.get("teacher_queries", 0.0) or 0.0)
        for row in rows
    )
    return float(weighted / queries)


def _should_apply_teacher_feedback(
    *,
    mode: str,
    agree: bool,
    rng: np.random.Generator,
    base_probability: float,
    disagreement_probability: float,
) -> bool:
    if mode == "none":
        return False
    if mode == "dense":
        return True
    base = max(0.0, min(1.0, float(base_probability)))
    disagreement = max(0.0, min(1.0, float(disagreement_probability)))
    probability = base if agree else max(base, disagreement)
    return bool(rng.random() < probability)


def _should_query_teacher(
    *,
    mode: str,
    rng: np.random.Generator,
    query_probability: float,
) -> bool:
    if mode == "none":
        return False
    if mode == "dense":
        return True
    probability = max(0.0, min(1.0, float(query_probability)))
    return bool(rng.random() < probability)


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


def _result_key(metrics: dict[str, float]) -> tuple[float, float, float, float, float, float, float, float]:
    rich_win = float(metrics.get("rich_session_win_rate", 0.0))
    rich_levels = float(metrics.get("rich_avg_levels_completed", 0.0))
    simple_success = float(metrics.get("simple_success_rate", 0.0))
    if max(rich_win, rich_levels, simple_success) <= 0.0:
        return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
    return (
        rich_win,
        rich_levels,
        simple_success,
        -float(metrics.get("rich_avg_reset_steps", 0.0)),
        float(metrics.get("rich_avg_attempt_improvement", 0.0)),
        float(metrics.get("simple_avg_attempt_improvement", 0.0)),
        float(metrics.get("rich_avg_return", 0.0)),
        float(metrics.get("simple_avg_return", 0.0)),
    )


def _holdout_has_autonomous_solve(metrics: dict[str, object]) -> bool:
    try:
        rich_win = float(metrics.get("rich_session_win_rate", 0.0))
        rich_levels = float(metrics.get("rich_avg_levels_completed", 0.0))
        simple_success = float(metrics.get("simple_success_rate", 0.0))
    except Exception:
        return False
    return max(rich_win, rich_levels, simple_success) > 0.0


def _checkpoint_metadata(
    *,
    config: ScientistTrainingConfig,
    session: int,
    stage: str,
    holdout: dict[str, object],
    train_recent: dict[str, float],
    promotion_eligible: bool,
    promoted: bool,
    agent: HyperGeneralizingScientistAgent,
) -> dict[str, object]:
    return {
        "artifact_schema_version": 2,
        "artifact_kind": "spotlight_scientist",
        "training_mode": "learner_owned_sparse_oracle_relabel_autonomous_holdout",
        "teacher_feedback_mode": str(config.teacher_feedback_mode),
        "teacher_query_probability": float(config.teacher_query_probability),
        "teacher_feedback_probability": float(config.teacher_feedback_probability),
        "teacher_disagreement_probability": float(config.teacher_disagreement_probability),
        "teacher_feedback_weight": float(config.teacher_feedback_weight),
        "session": int(session),
        "stage": str(stage),
        "spotlight_feature_schema_version": _spotlight_feature_schema_version(agent),
        "world_model": agent.diagnostics().get("world_model", {}),
        "learner_owned_recent": dict(train_recent),
        "autonomous_holdout": dict(holdout),
        "promotion_eligible": bool(promotion_eligible),
        "promoted": bool(promoted),
        "allow_unproven_best_checkpoint": bool(config.allow_unproven_best_checkpoint),
        "note": (
            "The agent executes its own actions during collection. Synthetic teacher labels are sparse bootstrap instrumentation only. "
            "Promotion eligibility is based only on teacher-free holdout solve metrics."
        ),
    }


def _write_unpromoted_sidecar(checkpoint_path: str, metadata: dict[str, object]) -> None:
    target = Path(checkpoint_path).with_suffix(Path(checkpoint_path).suffix + ".unpromoted.json")
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(metadata, indent=2, sort_keys=True), encoding="utf-8")


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
