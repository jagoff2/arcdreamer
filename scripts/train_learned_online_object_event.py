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
    ObjectEventBatch,
    ObjectEventCurriculumConfig,
    ObjectEventExample,
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
    parser.add_argument("--mode", choices=("synthetic_object_event", "trace_jsonl"), default="synthetic_object_event")
    parser.add_argument("--source", choices=("synthetic_object_event", "trace_jsonl"), default=None)
    parser.add_argument("--curriculum", choices=("paired_color_click",), default="paired_color_click")
    parser.add_argument("--trace-path", type=Path, default=None)
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--train-geometries", type=int, default=16)
    parser.add_argument("--heldout-geometries", type=int, default=16)
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
        permutation = torch.arange(state_numeric.shape[0] - 1, -1, -1, device=device)
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
