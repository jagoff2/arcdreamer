from __future__ import annotations

import argparse
import json
from pathlib import Path
import pickle
import sys
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from arcagi.agents.learned_online_object_event_agent import LearnedOnlineObjectEventAgent
from arcagi.learned_online.object_event_curriculum import (
    ActiveOnlineObjectEventConfig,
    OnlineObjectEventCurriculumConfig,
    OnlineObjectEventSession,
    ObjectEventBatch,
    ObjectEventCurriculumConfig,
    ObjectEventExample,
    apply_synthetic_object_event_action,
    build_active_online_object_event_curriculum,
    build_online_object_event_curriculum,
    build_paired_color_click_curriculum,
    collate_object_event_examples,
)
from arcagi.learned_online.event_tokens import (
    OUT_NO_EFFECT_NONPROGRESS,
    OUT_OBJECTIVE_PROGRESS,
    OUT_REWARD_PROGRESS,
)
from arcagi.learned_online.object_event_model import (
    ObjectEventModel,
    ObjectEventModelConfig,
    policy_rank_logits_from_predictions,
)

LOSS_WEIGHTS = {"outcome": 1.0, "rank": 5.0, "inverse": 0.1, "value": 0.5, "delta": 0.0}


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate the learned online object-event scaffold.")
    parser.add_argument(
        "--mode",
        choices=("synthetic_object_event", "synthetic_online_object_event", "synthetic_active_online_object_event", "trace_jsonl"),
        default="synthetic_object_event",
    )
    parser.add_argument("--source", choices=("synthetic_object_event", "trace_jsonl"), default=None)
    parser.add_argument("--curriculum", choices=("paired_color_click", "latent_rule_color_click"), default="paired_color_click")
    parser.add_argument("--trace-path", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-geometries", type=int, default=16)
    parser.add_argument("--heldout-geometries", type=int, default=16)
    parser.add_argument("--train-sessions", type=int, default=32)
    parser.add_argument("--heldout-sessions", type=int, default=32)
    parser.add_argument("--levels-per-session", type=int, default=3)
    parser.add_argument("--support-updates", type=int, default=1)
    parser.add_argument("--max-distractors", type=int, default=1)
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--state-layers", type=int, default=1)
    parser.add_argument("--action-cross-layers", type=int, default=1)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Legacy alias for --save-path.")
    parser.add_argument("--no-save", action="store_true")
    args = parser.parse_args()

    mode = args.source or args.mode
    if mode == "trace_jsonl":
        if args.trace_path is None:
            raise SystemExit("--trace-path is required for --mode trace_jsonl")
        raise SystemExit("trace_jsonl transition bootstrap is intentionally deferred until synthetic object-event metrics are stable")

    config = ObjectEventModelConfig(
        d_model=int(args.d_model),
        state_layers=int(args.state_layers),
        action_cross_layers=int(args.action_cross_layers),
        dropout=0.0,
    )
    device = torch.device(args.device)
    torch.manual_seed(int(args.seed))
    rng = np.random.default_rng(int(args.seed))
    if mode in {"synthetic_online_object_event", "synthetic_active_online_object_event"}:
        _run_synthetic_online_object_event(args=args, config=config, device=device, rng=rng, active=mode == "synthetic_active_online_object_event")
        return

    curriculum = build_paired_color_click_curriculum(
        ObjectEventCurriculumConfig(
            seed=int(args.seed),
            train_geometries=int(args.train_geometries),
            heldout_geometries=int(args.heldout_geometries),
            grid_size=8,
            require_full_dense_actions=True,
        )
    )
    model = ObjectEventModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)

    last_loss = 0.0
    eval_steps = {max(int(args.steps), 1)}
    eval_every = max(int(args.eval_every), 1)
    eval_steps.update(step for step in range(eval_every, max(int(args.steps), 1) + 1, eval_every))
    for step in range(1, max(int(args.steps), 1) + 1):
        batch_examples = _sample_examples(curriculum.train, batch_size=max(int(args.batch_size), 1), rng=rng)
        batch = collate_object_event_examples(batch_examples)
        tensors = _batch_to_tensors(batch, device=device)
        output = model(**tensors["inputs"])
        losses = model.loss(
            output,
            target_outcome=tensors["target_outcome"],
            target_delta=tensors["target_delta"],
            actual_action_index=tensors["actual_action_index"],
            action_mask=tensors["action_mask"],
            candidate_outcome_targets=tensors["candidate_outcome_targets"],
            candidate_value_targets=tensors["candidate_value_targets"],
            loss_weights=LOSS_WEIGHTS,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = float(losses["loss"].detach().cpu())
        if step in eval_steps:
            metrics = _training_metrics(
                model,
                train=curriculum.train,
                heldout=curriculum.heldout,
                device=device,
                step=step,
                train_loss=last_loss,
            )
            print(json.dumps(metrics, sort_keys=True), flush=True)

    save_path = args.save_path or args.output
    if save_path is not None and not bool(args.no_save):
        checkpoint = {
            "seed": int(args.seed),
            "config": config.to_dict(),
            "model_state": model.state_dict(),
            "metadata": LearnedOnlineObjectEventAgent.checkpoint_metadata(),
            "training_summary": _training_metrics(
                model,
                train=curriculum.train,
                heldout=curriculum.heldout,
                device=device,
                step=max(int(args.steps), 1),
                train_loss=last_loss,
            ),
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as handle:
            pickle.dump(checkpoint, handle)


def _run_synthetic_online_object_event(
    *,
    args: argparse.Namespace,
    config: ObjectEventModelConfig,
    device: torch.device,
    rng: np.random.Generator,
    active: bool,
) -> None:
    if active:
        curriculum = build_active_online_object_event_curriculum(
            ActiveOnlineObjectEventConfig(
                seed=int(args.seed),
                train_sessions=int(args.train_sessions),
                heldout_sessions=int(args.heldout_sessions),
                levels_per_session=int(args.levels_per_session),
                max_distractors=int(args.max_distractors),
                include_distractors=True,
                grid_size=8,
                max_steps_per_level=3,
                curriculum="latent_rule_color_click",
            )
        )
    else:
        curriculum = build_online_object_event_curriculum(
            OnlineObjectEventCurriculumConfig(
            seed=int(args.seed),
            train_sessions=int(args.train_sessions),
            heldout_sessions=int(args.heldout_sessions),
            levels_per_session=int(args.levels_per_session),
            max_objects=3,
            grid_size=8,
            include_distractors=False,
            require_full_dense_actions=True,
            curriculum="latent_rule_color_click",
            )
        )
    model = ObjectEventModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    last_loss = 0.0
    eval_steps = {max(int(args.steps), 1)}
    eval_every = max(int(args.eval_every), 1)
    eval_steps.update(step for step in range(eval_every, max(int(args.steps), 1) + 1, eval_every))
    for step in range(1, max(int(args.steps), 1) + 1):
        sessions = _sample_sessions(curriculum.train, batch_size=max(int(args.batch_size), 1), rng=rng)
        support_examples = tuple(session.levels[0].example for session in sessions)
        query_examples = tuple(session.levels[1].example for session in sessions)
        next_examples = tuple(session.levels[min(2, len(session.levels) - 1)].example for session in sessions)
        support_tensors = _batch_to_tensors(collate_object_event_examples(support_examples), device=device)
        query_tensors = _batch_to_tensors(collate_object_event_examples(query_examples), device=device)
        next_tensors = _batch_to_tensors(collate_object_event_examples(next_examples), device=device)
        session_belief = _observed_belief_delta(model, support_tensors)
        query_output = _forward_tensors(model, query_tensors, session_belief=session_belief)
        query_logits = policy_rank_logits_from_predictions(query_output, query_tensors["action_mask"])
        query_loss = F.cross_entropy(query_logits, query_tensors["actual_action_index"])
        next_output = _forward_tensors(model, next_tensors, session_belief=session_belief)
        next_logits = policy_rank_logits_from_predictions(next_output, next_tensors["action_mask"])
        next_loss = F.cross_entropy(next_logits, next_tensors["actual_action_index"])
        support_output = _forward_tensors(model, support_tensors)
        support_logits = policy_rank_logits_from_predictions(support_output, support_tensors["action_mask"])
        plausible_loss = _plausible_red_blue_loss(support_logits, support_examples)
        loss = query_loss + next_loss + 0.2 * plausible_loss
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        if step in eval_steps:
            print(
                json.dumps(
                    _online_training_metrics(
                        model,
                        train=curriculum.train,
                        heldout=curriculum.heldout,
                        device=device,
                        step=step,
                        train_loss=last_loss,
                        active=active,
                    ),
                    sort_keys=True,
                ),
                flush=True,
            )

    save_path = args.save_path or args.output
    if save_path is not None and not bool(args.no_save):
        checkpoint = {
            "seed": int(args.seed),
            "config": config.to_dict(),
            "model_state": model.state_dict(),
            "metadata": LearnedOnlineObjectEventAgent.checkpoint_metadata(),
            "training_summary": _online_training_metrics(
                model,
                train=curriculum.train,
                heldout=curriculum.heldout,
                device=device,
                step=max(int(args.steps), 1),
                train_loss=last_loss,
                active=active,
            ),
        }
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with save_path.open("wb") as handle:
            pickle.dump(checkpoint, handle)


def _sample_examples(
    examples: Sequence[ObjectEventExample],
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[ObjectEventExample, ...]:
    if not examples:
        raise ValueError("cannot train on an empty object-event curriculum split")
    indices = rng.integers(0, len(examples), size=int(batch_size))
    return tuple(examples[int(index)] for index in indices)


def _sample_sessions(
    sessions: Sequence[OnlineObjectEventSession],
    *,
    batch_size: int,
    rng: np.random.Generator,
) -> tuple[OnlineObjectEventSession, ...]:
    if not sessions:
        raise ValueError("cannot train on an empty online object-event session split")
    indices = rng.integers(0, len(sessions), size=int(batch_size))
    return tuple(sessions[int(index)] for index in indices)


def _forward_tensors(
    model: ObjectEventModel,
    tensors: dict[str, Any],
    *,
    session_belief: torch.Tensor | None = None,
    level_belief: torch.Tensor | None = None,
):
    inputs = dict(tensors["inputs"])
    if session_belief is not None:
        inputs["session_belief"] = session_belief
    if level_belief is not None:
        inputs["level_belief"] = level_belief
    return model(**inputs)


def _observed_belief_delta(model: ObjectEventModel, tensors: dict[str, Any]) -> torch.Tensor:
    return _observed_belief_deltas(model, tensors).session_delta


def _observed_belief_deltas(model: ObjectEventModel, tensors: dict[str, Any]):
    output = _forward_tensors(model, tensors)
    return model.observed_event_belief_deltas(
        output,
        target_outcome=tensors["target_outcome"],
        target_delta=tensors["target_delta"],
        actual_action_index=tensors["actual_action_index"],
        state_numeric=tensors["inputs"]["state_numeric"],
        state_type_ids=tensors["inputs"]["state_type_ids"],
        state_mask=tensors["inputs"]["state_mask"],
        action_numeric=tensors["inputs"]["action_numeric"],
    )


def _sample_red_blue_indices(
    examples: Sequence[ObjectEventExample],
    *,
    rng: np.random.Generator,
) -> torch.Tensor:
    indices: list[int] = []
    for example in examples:
        candidates = (int(example.metadata["red_action_index"]), int(example.metadata["blue_action_index"]))
        indices.append(candidates[int(rng.integers(0, 2))])
    return torch.as_tensor(indices, dtype=torch.long)


def _with_observed_candidate_targets(tensors: dict[str, Any], observed_indices: torch.Tensor) -> dict[str, Any]:
    device = tensors["actual_action_index"].device
    observed = observed_indices.to(device=device, dtype=torch.long)
    row = torch.arange(observed.shape[0], device=device)
    updated = {
        key: value
        for key, value in tensors.items()
        if key != "inputs"
    }
    updated["inputs"] = dict(tensors["inputs"])
    updated["actual_action_index"] = observed
    updated["target_outcome"] = tensors["candidate_outcome_targets"][row, observed]
    updated["target_delta"] = tensors["candidate_delta_targets"][row, observed]
    return updated


def _plausible_red_blue_loss(rank_logits: torch.Tensor, examples: Sequence[ObjectEventExample]) -> torch.Tensor:
    mask = torch.zeros_like(rank_logits, dtype=torch.bool)
    for row, example in enumerate(examples):
        mask[row, int(example.metadata["red_action_index"])] = True
        mask[row, int(example.metadata["blue_action_index"])] = True
    plausible = torch.logsumexp(rank_logits.masked_fill(~mask, -1.0e9), dim=-1)
    all_actions = torch.logsumexp(rank_logits, dim=-1)
    return torch.mean(all_actions - plausible)


def _online_training_metrics(
    model: ObjectEventModel,
    *,
    train: Sequence[OnlineObjectEventSession],
    heldout: Sequence[OnlineObjectEventSession],
    device: torch.device,
    step: int,
    train_loss: float,
    active: bool = False,
) -> dict[str, Any]:
    train_metrics = _evaluate_online_sessions(model, train, device=device)
    heldout_metrics = _evaluate_online_sessions(model, heldout, device=device)
    failure_metrics = _evaluate_online_sessions(model, heldout, device=device, support_mode="failure")
    wrong_metrics = _evaluate_online_sessions(model, heldout, device=device, support_mode="wrong_rule")
    zero_metrics = _evaluate_online_sessions(model, heldout, device=device, ablation="zero_state")
    shuffled_metrics = _evaluate_online_sessions(model, heldout, device=device, ablation="shuffled_state")
    reset_metrics = _evaluate_online_sessions(model, heldout, device=device, support_mode="session_reset")
    same_cue = tuple(session for session in heldout if session.levels[0].cue_mode == session.levels[1].cue_mode)
    different_cue = tuple(session for session in heldout if session.levels[0].cue_mode != session.levels[1].cue_mode)
    same_cue_metrics = _evaluate_online_sessions(model, same_cue, device=device)
    different_cue_metrics = _evaluate_online_sessions(model, different_cue, device=device)
    closed = _closed_loop_online_metrics(model, heldout, device=device)
    active_closed = _active_rollout_metrics(model, heldout, device=device) if active else {}
    metrics = {
        "step": int(step),
        "train_loss": float(train_loss),
        "train_pre_update_top1_acc": train_metrics["pre_top1"],
        "train_post_update_top1_acc": train_metrics["post_top1"],
        "heldout_pre_update_top1_acc": heldout_metrics["pre_top1"],
        "heldout_post_update_top1_acc": heldout_metrics["post_top1"],
        "heldout_no_update_top1_acc": heldout_metrics["pre_top1"],
        "heldout_wrong_session_update_top1_acc": wrong_metrics["post_top1"],
        "heldout_zero_state_update_top1_acc": zero_metrics["post_top1"],
        "heldout_shuffled_state_update_top1_acc": shuffled_metrics["post_top1"],
        "heldout_post_update_success_support_top1_acc": heldout_metrics["post_top1"],
        "heldout_post_update_failure_support_top1_acc": failure_metrics["post_top1"],
        "heldout_post_update_same_cue_top1_acc": same_cue_metrics["post_top1"],
        "heldout_post_update_different_cue_top1_acc": different_cue_metrics["post_top1"],
        "heldout_relation_rank_top1_acc": heldout_metrics["post_top1"],
        "heldout_base_rank_top1_acc": heldout_metrics["pre_top1"],
        "heldout_next_level_first_try_acc": heldout_metrics["next_level_top1"],
        "heldout_next_level_with_session_reset_acc": reset_metrics["next_level_top1"],
        "online_delta_top1_acc": heldout_metrics["post_top1"] - heldout_metrics["pre_top1"],
        "mean_steps_to_first_success": closed["mean_steps_to_first_success"],
        "success_within_1": closed["success_within_1"],
        "success_within_2": closed["success_within_2"],
        "success_within_3": closed["success_within_3"],
        "rank_ce": heldout_metrics["pre_rank_ce"],
        "post_update_rank_ce": heldout_metrics["post_rank_ce"],
        "mean_score_entropy_pre": heldout_metrics["pre_entropy"],
        "mean_score_entropy_post": heldout_metrics["post_entropy"],
        "num_legal_actions_mean": heldout_metrics["num_legal_actions_mean"],
        "session_belief_norm_pre": 0.0,
        "session_belief_norm_post": heldout_metrics["session_belief_norm"],
        "level_belief_norm_pre": 0.0,
        "level_belief_norm_post": 0.0,
    }
    if active:
        metrics.update(
            {
                "oracle_support_used": 0,
                "heldout_active_success_within_1": active_closed["success_within_1"],
                "heldout_active_success_within_2": active_closed["success_within_2"],
                "heldout_active_success_within_3": active_closed["success_within_3"],
                "heldout_active_next_level_first_try_acc": heldout_metrics["next_level_top1"],
                "heldout_active_next_level_with_session_reset_acc": reset_metrics["next_level_top1"],
                "heldout_active_pre_update_top1_acc": heldout_metrics["pre_top1"],
                "heldout_active_post_self_update_top1_acc": active_closed["post_self_update_top1_acc"],
                "heldout_active_no_update_top1_acc": heldout_metrics["pre_top1"],
                "heldout_active_wrong_session_update_top1_acc": wrong_metrics["post_top1"],
                "heldout_active_zero_state_update_top1_acc": zero_metrics["post_top1"],
                "heldout_active_shuffled_state_update_top1_acc": shuffled_metrics["post_top1"],
                "failed_candidate_repeat_top1_rate": active_closed["failed_candidate_repeat_top1_rate"],
                "failed_candidate_score_delta": active_closed["failed_candidate_score_delta"],
                "ordinary_object_click_rate": active_closed["ordinary_object_click_rate"],
                "agent_or_cue_click_rate": active_closed["agent_or_cue_click_rate"],
                "distractor_first_click_rate": active_closed["distractor_first_click_rate"],
                "target_first_click_rate": active_closed["target_first_click_rate"],
                "level_negative_memory_norm": active_closed["level_negative_memory_norm"],
                "session_relation_memory_norm": heldout_metrics["session_belief_norm"],
            }
        )
    return metrics


def _evaluate_online_sessions(
    model: ObjectEventModel,
    sessions: Sequence[OnlineObjectEventSession],
    *,
    device: torch.device,
    support_mode: str = "observed",
    ablation: str = "none",
) -> dict[str, float]:
    if not sessions:
        return {
            "pre_top1": 0.0,
            "post_top1": 0.0,
            "next_level_top1": 0.0,
            "pre_rank_ce": 0.0,
            "post_rank_ce": 0.0,
            "pre_entropy": 0.0,
            "post_entropy": 0.0,
            "num_legal_actions_mean": 0.0,
            "session_belief_norm": 0.0,
        }
    support_examples = tuple(session.levels[0].example for session in sessions)
    query_examples = tuple(session.levels[1].example for session in sessions)
    next_examples = tuple(session.levels[min(2, len(session.levels) - 1)].example for session in sessions)
    support_tensors = _batch_to_tensors(collate_object_event_examples(support_examples), device=device, ablation=ablation)
    query_tensors = _batch_to_tensors(collate_object_event_examples(query_examples), device=device, ablation=ablation)
    next_tensors = _batch_to_tensors(collate_object_event_examples(next_examples), device=device, ablation=ablation)
    if support_mode == "wrong_rule":
        wrong_support = _wrong_rule_support_examples(sessions)
        support_tensors = _batch_to_tensors(collate_object_event_examples(wrong_support), device=device, ablation=ablation)
    if support_mode == "failure":
        wrong_indices = torch.as_tensor(
            [
                int(example.metadata["blue_action_index"])
                if int(example.correct_action_index) == int(example.metadata["red_action_index"])
                else int(example.metadata["red_action_index"])
                for example in support_examples
            ],
            dtype=torch.long,
            device=device,
        )
        support_tensors = _with_observed_candidate_targets(support_tensors, wrong_indices)
    with torch.no_grad():
        pre_output = _forward_tensors(model, query_tensors)
        pre_logits = policy_rank_logits_from_predictions(pre_output, query_tensors["action_mask"])
        if support_mode == "session_reset":
            session_belief = torch.zeros(
                (len(sessions), model.config.d_model),
                dtype=pre_logits.dtype,
                device=device,
            )
        else:
            session_belief = _observed_belief_delta(model, support_tensors)
        post_output = _forward_tensors(model, query_tensors, session_belief=session_belief)
        post_logits = policy_rank_logits_from_predictions(post_output, query_tensors["action_mask"])
        next_output = _forward_tensors(model, next_tensors, session_belief=session_belief)
        next_logits = policy_rank_logits_from_predictions(next_output, next_tensors["action_mask"])
        pre_entropy = _mean_entropy(pre_logits)
        post_entropy = _mean_entropy(post_logits)
        legal_counts = torch.sum(query_tensors["action_mask"].float(), dim=-1)
    return {
        "pre_top1": _top1(pre_logits, query_tensors["actual_action_index"]),
        "post_top1": _top1(post_logits, query_tensors["actual_action_index"]),
        "next_level_top1": _top1(next_logits, next_tensors["actual_action_index"]),
        "pre_rank_ce": float(F.cross_entropy(pre_logits, query_tensors["actual_action_index"]).detach().cpu()),
        "post_rank_ce": float(F.cross_entropy(post_logits, query_tensors["actual_action_index"]).detach().cpu()),
        "pre_entropy": float(pre_entropy.detach().cpu()),
        "post_entropy": float(post_entropy.detach().cpu()),
        "num_legal_actions_mean": float(torch.mean(legal_counts).detach().cpu()),
        "session_belief_norm": float(torch.linalg.vector_norm(session_belief, dim=-1).mean().detach().cpu()),
    }


def _wrong_rule_support_examples(sessions: Sequence[OnlineObjectEventSession]) -> tuple[ObjectEventExample, ...]:
    by_rule: dict[int, list[ObjectEventExample]] = {0: [], 1: []}
    for session in sessions:
        by_rule[int(session.latent_rule)].append(session.levels[0].example)
    result: list[ObjectEventExample] = []
    for session in sessions:
        opposite = by_rule[1 - int(session.latent_rule)] or by_rule[int(session.latent_rule)]
        result.append(opposite[len(result) % len(opposite)])
    return tuple(result)


def _closed_loop_online_metrics(
    model: ObjectEventModel,
    sessions: Sequence[OnlineObjectEventSession],
    *,
    device: torch.device,
) -> dict[str, float]:
    if not sessions:
        return {"mean_steps_to_first_success": 0.0, "success_within_1": 0.0, "success_within_2": 0.0, "success_within_3": 0.0}
    steps: list[int] = []
    for session in sessions:
        support = session.levels[0].example
        support_tensors = _batch_to_tensors(collate_object_event_examples((support,)), device=device)
        with torch.no_grad():
            output = _forward_tensors(model, support_tensors)
            logits = policy_rank_logits_from_predictions(output, support_tensors["action_mask"])
            first = int(torch.argmax(logits[0]).detach().cpu())
            if first == int(support.correct_action_index):
                steps.append(1)
                continue
            observed = _with_observed_candidate_targets(support_tensors, torch.as_tensor([first], dtype=torch.long))
            session_belief = _observed_belief_delta(model, observed)
            retry_output = _forward_tensors(model, support_tensors, session_belief=session_belief)
            retry_logits = policy_rank_logits_from_predictions(retry_output, support_tensors["action_mask"])
            second = int(torch.argmax(retry_logits[0]).detach().cpu())
            steps.append(2 if second == int(support.correct_action_index) else 4)
    values = np.asarray(steps, dtype=np.float32)
    return {
        "mean_steps_to_first_success": float(np.mean(np.minimum(values, 4.0))),
        "success_within_1": float(np.mean(values <= 1.0)),
        "success_within_2": float(np.mean(values <= 2.0)),
        "success_within_3": float(np.mean(values <= 3.0)),
    }


def _active_rollout_metrics(
    model: ObjectEventModel,
    sessions: Sequence[OnlineObjectEventSession],
    *,
    device: torch.device,
    max_steps_per_level: int = 3,
) -> dict[str, float]:
    if not sessions:
        return {
            "success_within_1": 0.0,
            "success_within_2": 0.0,
            "success_within_3": 0.0,
            "post_self_update_top1_acc": 0.0,
            "failed_candidate_repeat_top1_rate": 0.0,
            "failed_candidate_score_delta": 0.0,
            "ordinary_object_click_rate": 0.0,
            "agent_or_cue_click_rate": 0.0,
            "distractor_first_click_rate": 0.0,
            "target_first_click_rate": 0.0,
            "level_negative_memory_norm": 0.0,
        }
    success_steps: list[int] = []
    post_update_hits: list[float] = []
    repeat_hits: list[float] = []
    failed_deltas: list[float] = []
    ordinary_clicks = 0
    agent_clicks = 0
    target_first = 0
    distractor_first = 0
    first_clicks = 0
    total_actions = 0
    level_norms: list[float] = []
    for session in sessions:
        session_belief = torch.zeros((1, model.config.d_model), dtype=torch.float32, device=device)
        for level in session.levels[:1]:
            level_belief = torch.zeros_like(session_belief)
            step_success = max_steps_per_level + 1
            failed_index: int | None = None
            failed_score_before = 0.0
            for step in range(1, max_steps_per_level + 1):
                tensors = _batch_to_tensors(collate_object_event_examples((level.example,)), device=device)
                with torch.no_grad():
                    output = _forward_tensors(model, tensors, session_belief=session_belief, level_belief=level_belief)
                    logits = policy_rank_logits_from_predictions(output, tensors["action_mask"])
                    selected = int(torch.argmax(logits[0]).detach().cpu())
                    action_row = tensors["inputs"]["action_numeric"][0, selected]
                    is_ordinary = float(action_row[11].detach().cpu()) > 0.5 and float(action_row[24].detach().cpu()) < 0.5
                    is_agent = float(action_row[24].detach().cpu()) > 0.5
                    total_actions += 1
                    ordinary_clicks += int(is_ordinary)
                    agent_clicks += int(is_agent)
                    if step == 1:
                        first_clicks += 1
                        target_indices = {int(level.example.metadata["red_action_index"]), int(level.example.metadata["blue_action_index"])}
                        target_first += int(selected in target_indices)
                        distractor_first += int(is_ordinary and selected not in target_indices)
                    result = apply_synthetic_object_event_action(level, selected)
                    observed = _with_observed_candidate_targets(tensors, torch.as_tensor([selected], dtype=torch.long, device=device))
                    deltas = _observed_belief_deltas(model, observed)
                    if result.success:
                        session_belief = session_belief + deltas.session_delta
                        level_belief = level_belief + deltas.level_delta
                        level_norms.append(float(torch.linalg.vector_norm(level_belief).detach().cpu()))
                        step_success = step
                        break
                    level_belief = level_belief + deltas.level_delta
                    level_norms.append(float(torch.linalg.vector_norm(level_belief).detach().cpu()))
                    if failed_index is None:
                        failed_index = selected
                        failed_score_before = float(logits[0, selected].detach().cpu())
                        retry_output = _forward_tensors(model, tensors, session_belief=session_belief, level_belief=level_belief)
                        retry_logits = policy_rank_logits_from_predictions(retry_output, tensors["action_mask"])
                        post_update_hits.append(float(int(torch.argmax(retry_logits[0]).detach().cpu()) == int(level.example.correct_action_index)))
                        repeat_hits.append(float(int(torch.argmax(retry_logits[0]).detach().cpu()) == selected))
                        failed_deltas.append(float(retry_logits[0, selected].detach().cpu()) - failed_score_before)
            success_steps.append(step_success)
    values = np.asarray(success_steps, dtype=np.float32)
    total_clicks = max(total_actions, 1)
    return {
        "success_within_1": float(np.mean(values <= 1.0)),
        "success_within_2": float(np.mean(values <= 2.0)),
        "success_within_3": float(np.mean(values <= 3.0)),
        "post_self_update_top1_acc": float(np.mean(post_update_hits)) if post_update_hits else 1.0,
        "failed_candidate_repeat_top1_rate": float(np.mean(repeat_hits)) if repeat_hits else 0.0,
        "failed_candidate_score_delta": float(np.mean(failed_deltas)) if failed_deltas else 0.0,
        "ordinary_object_click_rate": float(ordinary_clicks / total_clicks),
        "agent_or_cue_click_rate": float(agent_clicks / total_clicks),
        "distractor_first_click_rate": float(distractor_first / max(first_clicks, 1)),
        "target_first_click_rate": float(target_first / max(first_clicks, 1)),
        "level_negative_memory_norm": float(np.mean(level_norms)) if level_norms else 0.0,
    }


def _top1(logits: torch.Tensor, actual: torch.Tensor) -> float:
    return float(torch.mean((torch.argmax(logits, dim=-1) == actual).float()).detach().cpu())


def _mean_entropy(logits: torch.Tensor) -> torch.Tensor:
    probs = torch.softmax(logits, dim=-1)
    return -torch.sum(probs * torch.log(torch.clamp(probs, min=1.0e-8)), dim=-1).mean()


def _training_metrics(
    model: ObjectEventModel,
    *,
    train: Sequence[ObjectEventExample],
    heldout: Sequence[ObjectEventExample],
    device: torch.device,
    step: int,
    train_loss: float,
) -> dict[str, Any]:
    train_metrics = _evaluate_split(model, train, device=device)
    heldout_metrics = _evaluate_split(model, heldout, device=device)
    zero_metrics = _evaluate_split(model, train, device=device, ablation="zero_state")
    shuffled_metrics = _evaluate_split(model, train, device=device, ablation="shuffled_state")
    action_only_metrics = _evaluate_split(model, train, device=device, ablation="action_only")
    return {
        "step": int(step),
        "train_loss": float(train_loss),
        "train_rank_acc": train_metrics["top1_acc"],
        "train_top1_acc": train_metrics["top1_acc"],
        "rank_ce": train_metrics["rank_ce"],
        "semantic_rank_top1_acc": train_metrics["semantic_top1_acc"],
        "compat_rank_top1_acc": train_metrics["compat_top1_acc"],
        "combined_rank_top1_acc": train_metrics["combined_top1_acc"],
        "heldout_rank_acc": heldout_metrics["top1_acc"],
        "heldout_top1_acc": heldout_metrics["top1_acc"],
        "zero_state_top1_acc": zero_metrics["top1_acc"],
        "shuffled_state_top1_acc": shuffled_metrics["top1_acc"],
        "action_only_top1_acc": action_only_metrics["top1_acc"],
        "candidate_ce": train_metrics["candidate_ce"],
        "outcome_loss": train_metrics["outcome_loss"],
        "delta_loss": train_metrics["delta_loss"],
        "value_loss": train_metrics["value_loss"],
        "mean_score_entropy": train_metrics["mean_score_entropy"],
        "num_legal_actions_mean": train_metrics["num_legal_actions_mean"],
        "heldout_candidate_ce": heldout_metrics["candidate_ce"],
        "heldout_mean_score_entropy": heldout_metrics["mean_score_entropy"],
    }


def _evaluate_split(
    model: ObjectEventModel,
    examples: Sequence[ObjectEventExample],
    *,
    device: torch.device,
    ablation: str = "none",
) -> dict[str, float]:
    if not examples:
        return {
            "top1_acc": 0.0,
            "rank_ce": 0.0,
            "semantic_top1_acc": 0.0,
            "compat_top1_acc": 0.0,
            "combined_top1_acc": 0.0,
            "candidate_ce": 0.0,
            "outcome_loss": 0.0,
            "delta_loss": 0.0,
            "value_loss": 0.0,
            "mean_score_entropy": 0.0,
            "num_legal_actions_mean": 0.0,
        }
    model.eval()
    batch = collate_object_event_examples(examples)
    tensors = _batch_to_tensors(batch, device=device, ablation=ablation)
    with torch.no_grad():
        output = model(**tensors["inputs"])
        combined_logits = policy_rank_logits_from_predictions(output, tensors["action_mask"])
        semantic_logits = (
            output.outcome_logits[..., OUT_OBJECTIVE_PROGRESS]
            + output.outcome_logits[..., OUT_REWARD_PROGRESS]
            - output.outcome_logits[..., OUT_NO_EFFECT_NONPROGRESS]
            + output.value_logits
        ).masked_fill(~tensors["action_mask"].bool(), -1.0e9)
        compat_logits = output.rank_logits if output.rank_logits is not None else semantic_logits
        compat_logits = compat_logits.masked_fill(~tensors["action_mask"].bool(), -1.0e9)
        combined_predictions = torch.argmax(combined_logits, dim=-1)
        semantic_predictions = torch.argmax(semantic_logits, dim=-1)
        compat_predictions = torch.argmax(compat_logits, dim=-1)
        top1_acc = torch.mean((combined_predictions == tensors["actual_action_index"]).float())
        semantic_top1 = torch.mean((semantic_predictions == tensors["actual_action_index"]).float())
        compat_top1 = torch.mean((compat_predictions == tensors["actual_action_index"]).float())
        rank_ce = F.cross_entropy(combined_logits, tensors["actual_action_index"])
        candidate_ce = F.binary_cross_entropy_with_logits(
            output.outcome_logits[tensors["action_mask"]],
            tensors["candidate_outcome_targets"][tensors["action_mask"]],
        )
        row = torch.arange(output.outcome_logits.shape[0], device=device)
        chosen_delta = output.delta_pred[row, tensors["actual_action_index"]]
        delta_loss = F.smooth_l1_loss(chosen_delta, tensors["target_delta"])
        value_loss = F.binary_cross_entropy_with_logits(
            output.value_logits[tensors["action_mask"]],
            tensors["candidate_value_targets"][tensors["action_mask"]],
        )
        probs = torch.softmax(combined_logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(torch.clamp(probs, min=1.0e-8)), dim=-1)
        legal_counts = torch.sum(tensors["action_mask"].float(), dim=-1)
    model.train()
    return {
        "top1_acc": float(top1_acc.detach().cpu()),
        "rank_ce": float(rank_ce.detach().cpu()),
        "semantic_top1_acc": float(semantic_top1.detach().cpu()),
        "compat_top1_acc": float(compat_top1.detach().cpu()),
        "combined_top1_acc": float(top1_acc.detach().cpu()),
        "candidate_ce": float(candidate_ce.detach().cpu()),
        "outcome_loss": float(candidate_ce.detach().cpu()),
        "delta_loss": float(delta_loss.detach().cpu()),
        "value_loss": float(value_loss.detach().cpu()),
        "mean_score_entropy": float(torch.mean(entropy).detach().cpu()),
        "num_legal_actions_mean": float(torch.mean(legal_counts).detach().cpu()),
    }


def _batch_to_tensors(
    batch: ObjectEventBatch,
    *,
    device: torch.device,
    ablation: str = "none",
) -> dict[str, Any]:
    state_numeric = torch.as_tensor(batch.state_numeric, dtype=torch.float32, device=device)
    state_type_ids = torch.as_tensor(batch.state_type_ids, dtype=torch.long, device=device)
    state_mask = torch.as_tensor(batch.state_mask, dtype=torch.bool, device=device)
    if ablation in {"zero_state", "action_only"}:
        state_numeric = torch.zeros_like(state_numeric)
    if ablation == "action_only":
        state_type_ids = torch.zeros_like(state_type_ids)
        state_mask = torch.ones_like(state_mask)
    if ablation == "shuffled_state" and state_numeric.shape[0] > 1:
        permutation = torch.roll(torch.arange(state_numeric.shape[0], device=device), shifts=1)
        state_numeric = state_numeric[permutation]
        state_type_ids = state_type_ids[permutation]
        state_mask = state_mask[permutation]
    action_mask = torch.as_tensor(batch.action_mask, dtype=torch.bool, device=device)
    return {
        "inputs": {
            "state_numeric": state_numeric,
            "state_type_ids": state_type_ids,
            "state_mask": state_mask,
            "action_numeric": torch.as_tensor(batch.action_numeric, dtype=torch.float32, device=device),
            "action_type_ids": torch.as_tensor(batch.action_type_ids, dtype=torch.long, device=device),
            "direction_ids": torch.as_tensor(batch.direction_ids, dtype=torch.long, device=device),
            "action_mask": action_mask,
        },
        "target_outcome": torch.as_tensor(batch.target_outcome, dtype=torch.float32, device=device),
        "target_delta": torch.as_tensor(batch.target_delta, dtype=torch.float32, device=device),
        "actual_action_index": torch.as_tensor(batch.actual_action_index, dtype=torch.long, device=device),
        "action_mask": action_mask,
        "candidate_outcome_targets": torch.as_tensor(batch.candidate_outcome_targets, dtype=torch.float32, device=device),
        "candidate_value_targets": torch.as_tensor(batch.candidate_value_targets, dtype=torch.float32, device=device),
        "candidate_delta_targets": torch.as_tensor(batch.candidate_delta_targets, dtype=torch.float32, device=device),
    }


if __name__ == "__main__":
    main()
