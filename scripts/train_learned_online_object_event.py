from __future__ import annotations

import argparse
import json
import math
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
from arcagi.core.action_schema import parse_click_action
from arcagi.core.types import GridObservation, StructuredState, Transition
from arcagi.learned_online.object_event_bridge import (
    SelectedActionObservation,
    assert_no_forbidden_metadata_as_model_input,
    build_object_event_observation,
)
from arcagi.learned_online.object_event_curriculum import (
    ActiveOnlineObjectEventConfig,
    OnlineObjectEventCurriculumConfig,
    OnlineObjectEventCurriculumSplit,
    OnlineObjectEventSession,
    OnlineObjectEventLevel,
    ObjectEventBatch,
    ObjectEventCurriculumConfig,
    ObjectEventExample,
    apply_synthetic_object_event_action,
    apply_synthetic_object_event_action_to_grid,
    build_active_online_object_event_curriculum,
    build_online_object_event_curriculum,
    build_paired_color_click_curriculum,
    collate_object_event_examples,
    rebuild_object_event_example_with_states,
    state_to_grid_observation,
)
from arcagi.learned_online.event_tokens import (
    ACTION_FEATURE_CLICK_X,
    ACTION_FEATURE_CLICK_Y,
    ACTION_FEATURE_GRID_COL,
    ACTION_FEATURE_GRID_ROW,
    ACTION_FEATURE_HAS_GRID_CELL,
    ACTION_FEATURE_IS_CLICK,
    OUT_NO_EFFECT_NONPROGRESS,
    OUT_OBJECTIVE_PROGRESS,
    OUT_REWARD_PROGRESS,
)
from arcagi.learned_online.object_event_model import (
    ObjectEventModel,
    ObjectEventModelConfig,
    policy_rank_logits_from_predictions,
)
from arcagi.learned_online.object_event_runtime_extraction import (
    ObjectEventRuntimeExtractor,
    extract_selected_transition_from_grid,
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
    parser.add_argument(
        "--curriculum",
        choices=("paired_color_click", "latent_rule_color_click", "latent_rule_variable_palette"),
        default="paired_color_click",
    )
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
    parser.add_argument("--palette-size", type=int, default=8)
    parser.add_argument("--require-role-balanced-colors", action="store_true")
    parser.add_argument("--active-eval-level0-stratified", action="store_true")
    parser.add_argument("--eval-every", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--d-model", type=int, default=64)
    parser.add_argument("--state-layers", type=int, default=1)
    parser.add_argument("--action-cross-layers", type=int, default=1)
    parser.add_argument("--save-path", type=Path, default=None)
    parser.add_argument("--output", type=Path, default=None, help="Legacy alias for --save-path.")
    parser.add_argument("--no-save", action="store_true")
    parser.add_argument("--eval-runtime-agent", action="store_true")
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--selection-metric", default=None)
    parser.add_argument("--state-source", choices=("structured", "extracted", "both"), default="structured")
    parser.add_argument("--action-surface", choices=("dense_8x8", "arc_scale_parametric"), default="dense_8x8")
    parser.add_argument("--action-surface-size", type=int, default=68)
    parser.add_argument("--coordinate-grid-size", type=int, default=64)
    parser.add_argument("--empty-click-fraction", type=float, default=0.80)
    parser.add_argument("--positive-region-radius", type=int, default=1)
    parser.add_argument("--max-steps-per-level", type=int, default=3)
    parser.add_argument("--axis-noeffect-cases", type=float, default=0.0)
    parser.add_argument("--relation-contradiction-cases", type=float, default=0.0)
    parser.add_argument("--diagnostic-utility-cases", type=float, default=0.0)
    parser.add_argument("--diagnostic-utility-weight", type=float, default=0.75)
    parser.add_argument("--family-diagnostic-cases", type=float, default=0.0)
    parser.add_argument("--family-balance-loss-weight", type=float, default=0.0)
    parser.add_argument("--family-contrastive-loss-weight", type=float, default=0.0)
    parser.add_argument("--family-sharpness-loss-weight", type=float, default=0.0)
    parser.add_argument("--family-postfailure-margin-weight", type=float, default=0.0)
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
        if str(args.curriculum) == "paired_color_click":
            args.curriculum = "latent_rule_color_click"
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
                max_steps_per_level=int(args.max_steps_per_level),
                curriculum=str(args.curriculum),
                palette_size=int(args.palette_size),
                require_role_balanced_colors=bool(args.require_role_balanced_colors),
                action_surface=str(args.action_surface),
                action_surface_size=int(args.action_surface_size),
                coordinate_grid_size=int(args.coordinate_grid_size),
                empty_click_fraction=float(args.empty_click_fraction),
                positive_region_radius=int(args.positive_region_radius),
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
                action_surface=str(args.action_surface),
                action_surface_size=int(args.action_surface_size),
                coordinate_grid_size=int(args.coordinate_grid_size),
                empty_click_fraction=float(args.empty_click_fraction),
                positive_region_radius=int(args.positive_region_radius),
            )
        )
    model = ObjectEventModel(config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    state_source = str(args.state_source)
    extraction_metrics: dict[str, float | str] = {"state_source": state_source}
    if active and state_source in {"extracted", "both"}:
        structured_curriculum = curriculum
        extracted_curriculum = _runtime_extracted_curriculum(curriculum)
        extraction_metrics.update(_runtime_extraction_diagnostics(structured_curriculum, extracted_curriculum))
        if state_source == "extracted":
            curriculum = extracted_curriculum
        else:
            curriculum = OnlineObjectEventCurriculumSplit(
                train=tuple(structured_curriculum.train) + tuple(extracted_curriculum.train),
                heldout=tuple(extracted_curriculum.heldout),
            )
    last_loss = 0.0
    last_diagnostic_utility_loss = 0.0
    last_diagnostic_margin_loss = 0.0
    last_diagnostic_good_minus_failed_family = 0.0
    last_diagnostic_failed_family_top1_rate = 0.0
    last_diagnostic_ig_top1_rate = 0.0
    last_diagnostic_mix_before_evidence = 0.0
    last_diagnostic_mix_after_noeffect = 0.0
    last_diagnostic_family_uncertainty_top1_rate = 0.0
    last_diagnostic_known_noeffect_family_top1_rate = 0.0
    last_diagnostic_uncertain_minus_noeffect = 0.0
    last_family_balance_loss = 0.0
    last_family_sharpness_loss = 0.0
    last_family_contrastive_loss = 0.0
    last_family_postfailure_margin_loss = 0.0
    last_diagnostic_different_uncertain_family_top1_rate = 0.0
    last_diagnostic_same_failed_family_top1_rate = 0.0
    last_diagnostic_different_minus_same_failed_family = 0.0
    last_family_assignment_effective_count = 0.0
    last_family_assignment_usage_min = 0.0
    last_family_assignment_usage_max = 0.0
    last_family_assignment_per_action_entropy = 0.0
    last_family_contrastive_positive_score = 0.0
    last_family_contrastive_negative_score = 0.0
    best_metric_value = float("-inf")
    best_step = 0
    best_state: dict[str, torch.Tensor] | None = None
    best_summary: dict[str, Any] | None = None
    selection_metric = _online_selection_metric(args, active=active)
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
        query_loss = _positive_rank_loss(query_logits, query_tensors)
        next_output = _forward_tensors(model, next_tensors, session_belief=session_belief)
        next_logits = policy_rank_logits_from_predictions(next_output, next_tensors["action_mask"])
        next_loss = _positive_rank_loss(next_logits, next_tensors)
        wrong_indices = _wrong_candidate_indices(support_examples, device=device)
        failure_support_tensors = _with_observed_candidate_targets(support_tensors, wrong_indices)
        failure_deltas = _observed_belief_deltas(model, failure_support_tensors)
        failure_belief = failure_deltas.session_delta
        failure_query_output = _forward_tensors(model, query_tensors, session_belief=failure_belief)
        failure_query_logits = policy_rank_logits_from_predictions(failure_query_output, query_tensors["action_mask"])
        failure_loss = _positive_rank_loss(failure_query_logits, query_tensors)
        retry_examples = _post_failure_retry_examples(
            tuple(session.levels[0] for session in sessions),
            tuple(int(index) for index in wrong_indices.detach().cpu().tolist()),
        )
        retry_tensors = _batch_to_tensors(collate_object_event_examples(retry_examples), device=device)
        retry_output = _forward_tensors(
            model,
            retry_tensors,
            session_belief=failure_deltas.session_delta,
            level_belief=failure_deltas.level_delta,
        )
        retry_logits = policy_rank_logits_from_predictions(retry_output, retry_tensors["action_mask"])
        retry_loss = _positive_rank_loss(retry_logits, retry_tensors)
        support_output = _forward_tensors(model, support_tensors)
        support_logits = policy_rank_logits_from_predictions(support_output, support_tensors["action_mask"])
        family_regularization = _family_assignment_regularization_losses(support_output, support_tensors)
        self_selected = torch.argmax(support_logits.detach(), dim=-1)
        self_failure_mask = (~_selected_is_positive(support_tensors, self_selected)).float()
        self_observed_tensors = _with_observed_candidate_targets(support_tensors, self_selected)
        self_deltas = _observed_belief_deltas(model, self_observed_tensors)
        self_retry_examples = _post_failure_retry_examples(
            tuple(session.levels[0] for session in sessions),
            tuple(int(index) for index in self_selected.detach().cpu().tolist()),
        )
        self_retry_tensors = _batch_to_tensors(collate_object_event_examples(self_retry_examples), device=device)
        self_retry_output = _forward_tensors(
            model,
            self_retry_tensors,
            session_belief=self_deltas.session_delta,
            level_belief=self_deltas.level_delta,
        )
        self_retry_logits = policy_rank_logits_from_predictions(self_retry_output, self_retry_tensors["action_mask"])
        self_retry_ce = _positive_rank_loss_per_row(self_retry_logits, self_retry_tensors)
        self_retry_loss = torch.sum(self_retry_ce * self_failure_mask) / torch.clamp(torch.sum(self_failure_mask), min=1.0)
        selected_before = support_logits.gather(1, self_selected[:, None]).squeeze(1)
        selected_after = self_retry_logits.gather(1, self_selected[:, None]).squeeze(1)
        positive_mask = _positive_mask_from_tensors(self_retry_tensors)
        positive_after = torch.logsumexp(self_retry_logits.masked_fill(~positive_mask, -1.0e9), dim=-1)
        exact_failed_mask = F.one_hot(self_selected, num_classes=support_logits.shape[1]).bool() & support_tensors["action_mask"].bool()
        near_failed_mask = _near_failed_mask_from_tensors(support_tensors, self_selected) & ~positive_mask
        axis_failed_masks = {
            key: mask & ~positive_mask
            for key, mask in _axis_failed_masks_from_tensors(support_tensors, self_selected).items()
        }
        self_noeffect_click_mask = _selected_noeffect_click_mask(support_tensors, self_selected).float()
        coord_failure_mask = self_failure_mask * self_noeffect_click_mask
        near_after = torch.logsumexp(self_retry_logits.masked_fill(~near_failed_mask, -1.0e9), dim=-1)
        near_before = torch.logsumexp(support_logits.masked_fill(~near_failed_mask, -1.0e9), dim=-1)
        exact_after = torch.logsumexp(self_retry_logits.masked_fill(~exact_failed_mask, -1.0e9), dim=-1)
        exact_before = torch.logsumexp(support_logits.masked_fill(~exact_failed_mask, -1.0e9), dim=-1)
        repeat_margin_loss = _masked_mean_loss(
            F.relu(selected_after - positive_after + 1.0),
            self_failure_mask,
        )
        score_drop_loss = _masked_mean_loss(
            F.relu(selected_after - selected_before + 0.5),
            self_failure_mask,
        )
        near_margin_loss = _masked_mean_loss(
            F.relu(near_after - positive_after + 0.5),
            coord_failure_mask,
        )
        exact_coord_drop_loss = _masked_mean_loss(
            F.relu(exact_after - exact_before + 0.5),
            coord_failure_mask,
        )
        near_drop_loss = _masked_mean_loss(
            F.relu(near_after - near_before + 0.25),
            coord_failure_mask,
        )
        axis_margin_terms = []
        axis_drop_terms = []
        for axis_mask in axis_failed_masks.values():
            axis_after = torch.logsumexp(self_retry_logits.masked_fill(~axis_mask, -1.0e9), dim=-1)
            axis_before = torch.logsumexp(support_logits.masked_fill(~axis_mask, -1.0e9), dim=-1)
            axis_margin_terms.append(_masked_mean_loss(F.relu(axis_after - positive_after + 0.35), coord_failure_mask))
            axis_drop_terms.append(_masked_mean_loss(F.relu(axis_after - axis_before + 0.20), coord_failure_mask))
        axis_margin_loss = torch.stack(axis_margin_terms).mean() if axis_margin_terms else support_logits.new_tensor(0.0)
        axis_drop_loss = torch.stack(axis_drop_terms).mean() if axis_drop_terms else support_logits.new_tensor(0.0)
        axis_loss_scale = 1.0 + max(float(args.axis_noeffect_cases), 0.0)
        relation_failed_mask = (
            axis_failed_masks.get("same_x", torch.zeros_like(exact_failed_mask))
            | axis_failed_masks.get("same_mapped_col", torch.zeros_like(exact_failed_mask))
        ) & ~positive_mask
        contradiction_scale = max(float(args.relation_contradiction_cases), 0.0)
        if self_retry_output.rank_components is not None:
            relation_column_after = torch.logsumexp(self_retry_logits.masked_fill(~relation_failed_mask, -1.0e9), dim=-1)
            contradiction_margin_loss = _masked_mean_loss(
                F.relu(relation_column_after - positive_after + 0.5),
                coord_failure_mask,
            )
            components = self_retry_output.rank_components
            relation_minus_noeffect = components.relation - (
                components.failed_action + components.coordinate_noeffect + components.axis_noeffect
            )
            relation_balance = _masked_action_mean(relation_minus_noeffect, relation_failed_mask)
            component_balance_loss = _masked_mean_loss(F.relu(relation_balance - 2.0), coord_failure_mask)
        else:
            contradiction_margin_loss = support_logits.new_tensor(0.0)
            component_balance_loss = support_logits.new_tensor(0.0)
        diagnostic_scale = max(float(args.diagnostic_utility_cases), float(args.family_diagnostic_cases), 0.0)
        if self_retry_output.diagnostic_utility_logits is not None:
            diag_logits = self_retry_output.diagnostic_utility_logits
            selected_family_overlap = _selected_family_overlap(support_output, self_retry_output, self_selected)
            hypothesis_target = _diagnostic_information_gain_targets(support_examples, self_retry_tensors, self_selected).to(
                dtype=diag_logits.dtype,
                device=diag_logits.device,
            )
            family_target = _family_posterior_diagnostic_targets(
                self_retry_output,
                self_retry_tensors,
                selected_family_overlap=selected_family_overlap,
            ).to(
                dtype=diag_logits.dtype,
                device=diag_logits.device,
            )
            family_weight = 0.7 if float(args.family_diagnostic_cases) > 0.0 else 0.0
            diag_target = (family_weight * family_target) + ((1.0 - family_weight) * hypothesis_target)
            diag_target = diag_target.masked_fill(~self_retry_tensors["action_mask"].bool(), 0.0)
            target_max = diag_target.max(dim=1, keepdim=True).values.clamp_min(1.0e-6)
            diag_target = (diag_target / target_max).clamp(0.0, 1.0)
            valid_weight = self_retry_tensors["action_mask"].to(dtype=diag_logits.dtype)
            diag_bce = F.binary_cross_entropy_with_logits(diag_logits, diag_target, reduction="none")
            diagnostic_utility_loss = torch.sum(diag_bce * valid_weight) / torch.clamp(torch.sum(valid_weight), min=1.0)
            info_good_mask = (diag_target >= 0.6) & self_retry_tensors["action_mask"].bool()
            safe_info_good = torch.where(info_good_mask.any(dim=1, keepdim=True), info_good_mask, positive_mask)
            diag_good = torch.logsumexp(diag_logits.masked_fill(~safe_info_good, -1.0e9), dim=-1)
            diag_failed = torch.logsumexp(diag_logits.masked_fill(~relation_failed_mask, -1.0e9), dim=-1)
            diagnostic_margin_loss = _masked_mean_loss(F.relu(diag_failed - diag_good + 0.5), coord_failure_mask)
            diagnostic_good_minus_failed = _masked_mean_loss(diag_good - diag_failed, coord_failure_mask)
            diag_selected = torch.argmax(diag_logits.masked_fill(~self_retry_tensors["action_mask"].bool(), -1.0e9), dim=-1)
            family_uncertain_mask = (family_target >= 0.6) & self_retry_tensors["action_mask"].bool()
            family_known_noeffect_mask = _family_known_noeffect_mask(self_retry_output, self_retry_tensors)
            safe_uncertain = torch.where(
                family_uncertain_mask.any(dim=1, keepdim=True),
                family_uncertain_mask,
                info_good_mask,
            )
            family_uncertain_logit = _masked_action_mean(diag_logits, safe_uncertain)
            family_noeffect_logit = _masked_action_mean(diag_logits, family_known_noeffect_mask)
            diagnostic_uncertain_minus_noeffect = _masked_mean_loss(
                family_uncertain_logit - family_noeffect_logit,
                coord_failure_mask,
            )
            diagnostic_family_uncertainty_top1 = torch.mean(
                family_uncertain_mask.gather(1, diag_selected[:, None]).squeeze(1).to(dtype=diag_logits.dtype)
            )
            diagnostic_known_noeffect_family_top1 = torch.mean(
                family_known_noeffect_mask.gather(1, diag_selected[:, None]).squeeze(1).to(dtype=diag_logits.dtype)
            )
            diagnostic_failed_family_top1 = torch.mean(
                relation_failed_mask.gather(1, diag_selected[:, None]).squeeze(1).to(dtype=diag_logits.dtype)
            )
            diagnostic_ig_top1 = torch.mean(
                info_good_mask.gather(1, diag_selected[:, None]).squeeze(1).to(dtype=diag_logits.dtype)
            )
            same_failed_family_mask = (
                (selected_family_overlap > 0.50)
                & self_retry_tensors["action_mask"].bool()
                & ~positive_mask
            )
            different_uncertain_family_mask = (
                (selected_family_overlap < 0.25)
                & (family_target >= 0.60)
                & self_retry_tensors["action_mask"].bool()
            )
            safe_different_uncertain = torch.where(
                different_uncertain_family_mask.any(dim=1, keepdim=True),
                different_uncertain_family_mask,
                safe_info_good,
            )
            same_failed_family_logit = torch.logsumexp(diag_logits.masked_fill(~same_failed_family_mask, -1.0e9), dim=-1)
            different_uncertain_family_logit = torch.logsumexp(
                diag_logits.masked_fill(~safe_different_uncertain, -1.0e9),
                dim=-1,
            )
            family_margin_rows = (
                coord_failure_mask
                * same_failed_family_mask.any(dim=1).to(dtype=diag_logits.dtype)
                * safe_different_uncertain.any(dim=1).to(dtype=diag_logits.dtype)
            )
            family_postfailure_margin_loss = _masked_mean_loss(
                F.relu(same_failed_family_logit - different_uncertain_family_logit + 0.5),
                family_margin_rows,
            )
            diagnostic_different_minus_same_failed = _masked_mean_loss(
                different_uncertain_family_logit - same_failed_family_logit,
                family_margin_rows,
            )
            diagnostic_different_uncertain_family_top1 = torch.mean(
                different_uncertain_family_mask.gather(1, diag_selected[:, None]).squeeze(1).to(dtype=diag_logits.dtype)
            )
            diagnostic_same_failed_family_top1 = torch.mean(
                same_failed_family_mask.gather(1, diag_selected[:, None]).squeeze(1).to(dtype=diag_logits.dtype)
            )
            if self_retry_output.diagnostic_mix_logit is not None and support_output.diagnostic_mix_logit is not None:
                ig_available = (diag_target.max(dim=1).values > 0.5).to(dtype=diag_logits.dtype)
                mix_target = (coord_failure_mask * ig_available).clamp(0.0, 1.0)
                diagnostic_mix_after_loss = F.binary_cross_entropy_with_logits(
                    self_retry_output.diagnostic_mix_logit,
                    mix_target,
                )
                diagnostic_mix_before_loss = F.binary_cross_entropy_with_logits(
                    support_output.diagnostic_mix_logit,
                    torch.zeros_like(support_output.diagnostic_mix_logit),
                )
                diagnostic_mix_loss = diagnostic_mix_after_loss + 0.5 * diagnostic_mix_before_loss
                diagnostic_mix_after = torch.sigmoid(self_retry_output.diagnostic_mix_logit)
                diagnostic_mix_before = torch.sigmoid(support_output.diagnostic_mix_logit)
            else:
                diagnostic_mix_loss = support_logits.new_tensor(0.0)
                diagnostic_mix_after = support_logits.new_zeros((support_logits.shape[0],))
                diagnostic_mix_before = support_logits.new_zeros((support_logits.shape[0],))
        else:
            diagnostic_utility_loss = support_logits.new_tensor(0.0)
            diagnostic_margin_loss = support_logits.new_tensor(0.0)
            diagnostic_good_minus_failed = support_logits.new_tensor(0.0)
            diagnostic_failed_family_top1 = support_logits.new_tensor(0.0)
            diagnostic_ig_top1 = support_logits.new_tensor(0.0)
            diagnostic_family_uncertainty_top1 = support_logits.new_tensor(0.0)
            diagnostic_known_noeffect_family_top1 = support_logits.new_tensor(0.0)
            diagnostic_uncertain_minus_noeffect = support_logits.new_tensor(0.0)
            family_postfailure_margin_loss = support_logits.new_tensor(0.0)
            diagnostic_different_uncertain_family_top1 = support_logits.new_tensor(0.0)
            diagnostic_same_failed_family_top1 = support_logits.new_tensor(0.0)
            diagnostic_different_minus_same_failed = support_logits.new_tensor(0.0)
            diagnostic_mix_loss = support_logits.new_tensor(0.0)
            diagnostic_mix_after = support_logits.new_zeros((support_logits.shape[0],))
            diagnostic_mix_before = support_logits.new_zeros((support_logits.shape[0],))
        plausible_loss = _plausible_red_blue_loss(support_logits, support_examples)
        loss = (
            query_loss
            + next_loss
            + 0.5 * failure_loss
            + 0.75 * retry_loss
            + 0.75 * self_retry_loss
            + 0.5 * repeat_margin_loss
            + 0.25 * score_drop_loss
            + 0.35 * near_margin_loss
            + 0.25 * exact_coord_drop_loss
            + 0.15 * near_drop_loss
            + axis_loss_scale * 0.25 * axis_margin_loss
            + axis_loss_scale * 0.15 * axis_drop_loss
            + contradiction_scale * 0.25 * contradiction_margin_loss
            + contradiction_scale * 0.10 * component_balance_loss
            + diagnostic_scale * max(float(args.diagnostic_utility_weight), 0.0) * 0.30 * diagnostic_utility_loss
            + diagnostic_scale * 0.20 * diagnostic_margin_loss
            + diagnostic_scale * 0.15 * F.relu(0.25 - diagnostic_uncertain_minus_noeffect)
            + diagnostic_scale * 0.10 * diagnostic_mix_loss
            + max(float(args.family_balance_loss_weight), 0.0) * family_regularization["balance_loss"]
            + max(float(args.family_sharpness_loss_weight), 0.0) * family_regularization["sharpness_loss"]
            + max(float(args.family_contrastive_loss_weight), 0.0) * family_regularization["contrastive_loss"]
            + max(float(args.family_postfailure_margin_weight), 0.0) * family_postfailure_margin_loss
            + 0.2 * plausible_loss
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        last_loss = float(loss.detach().cpu())
        last_diagnostic_utility_loss = float(diagnostic_utility_loss.detach().cpu())
        last_diagnostic_margin_loss = float(diagnostic_margin_loss.detach().cpu())
        last_diagnostic_good_minus_failed_family = float(diagnostic_good_minus_failed.detach().cpu())
        last_diagnostic_failed_family_top1_rate = float(diagnostic_failed_family_top1.detach().cpu())
        last_diagnostic_ig_top1_rate = float(diagnostic_ig_top1.detach().cpu())
        last_diagnostic_mix_before_evidence = float(torch.mean(diagnostic_mix_before.detach()).cpu())
        last_diagnostic_mix_after_noeffect = float(torch.mean(diagnostic_mix_after.detach()).cpu())
        last_diagnostic_family_uncertainty_top1_rate = float(diagnostic_family_uncertainty_top1.detach().cpu())
        last_diagnostic_known_noeffect_family_top1_rate = float(diagnostic_known_noeffect_family_top1.detach().cpu())
        last_diagnostic_uncertain_minus_noeffect = float(diagnostic_uncertain_minus_noeffect.detach().cpu())
        last_family_balance_loss = float(family_regularization["balance_loss"].detach().cpu())
        last_family_sharpness_loss = float(family_regularization["sharpness_loss"].detach().cpu())
        last_family_contrastive_loss = float(family_regularization["contrastive_loss"].detach().cpu())
        last_family_postfailure_margin_loss = float(family_postfailure_margin_loss.detach().cpu())
        last_diagnostic_different_uncertain_family_top1_rate = float(diagnostic_different_uncertain_family_top1.detach().cpu())
        last_diagnostic_same_failed_family_top1_rate = float(diagnostic_same_failed_family_top1.detach().cpu())
        last_diagnostic_different_minus_same_failed_family = float(diagnostic_different_minus_same_failed.detach().cpu())
        last_family_assignment_effective_count = float(family_regularization["effective_count"].detach().cpu())
        last_family_assignment_usage_min = float(family_regularization["usage_min"].detach().cpu())
        last_family_assignment_usage_max = float(family_regularization["usage_max"].detach().cpu())
        last_family_assignment_per_action_entropy = float(family_regularization["per_action_entropy"].detach().cpu())
        last_family_contrastive_positive_score = float(family_regularization["positive_score_mean"].detach().cpu())
        last_family_contrastive_negative_score = float(family_regularization["negative_score_mean"].detach().cpu())
        if step in eval_steps:
            eval_extra_metrics = dict(extraction_metrics)
            eval_extra_metrics.update(
                {
                    "diagnostic_utility_loss": last_diagnostic_utility_loss,
                    "diagnostic_margin_loss": last_diagnostic_margin_loss,
                    "diagnostic_good_minus_failed_family_mean": last_diagnostic_good_minus_failed_family,
                    "diagnostic_failed_family_top1_rate": last_diagnostic_failed_family_top1_rate,
                    "diagnostic_ig_top1_rate": last_diagnostic_ig_top1_rate,
                    "diagnostic_family_uncertainty_top1_rate": last_diagnostic_family_uncertainty_top1_rate,
                    "diagnostic_known_noeffect_family_top1_rate": last_diagnostic_known_noeffect_family_top1_rate,
                    "diagnostic_uncertain_minus_noeffect_mean": last_diagnostic_uncertain_minus_noeffect,
                    "diagnostic_different_uncertain_family_top1_rate": last_diagnostic_different_uncertain_family_top1_rate,
                    "diagnostic_same_failed_family_top1_rate": last_diagnostic_same_failed_family_top1_rate,
                    "diagnostic_different_minus_same_failed_family_mean": last_diagnostic_different_minus_same_failed_family,
                    "family_balance_loss": last_family_balance_loss,
                    "family_sharpness_loss": last_family_sharpness_loss,
                    "family_contrastive_loss": last_family_contrastive_loss,
                    "family_postfailure_margin_loss": last_family_postfailure_margin_loss,
                    "family_assignment_effective_count": last_family_assignment_effective_count,
                    "family_assignment_usage_min": last_family_assignment_usage_min,
                    "family_assignment_usage_max": last_family_assignment_usage_max,
                    "family_assignment_per_action_entropy": last_family_assignment_per_action_entropy,
                    "family_contrastive_positive_score_mean": last_family_contrastive_positive_score,
                    "family_contrastive_negative_score_mean": last_family_contrastive_negative_score,
                    "diagnostic_mix_before_evidence_mean": last_diagnostic_mix_before_evidence,
                    "diagnostic_mix_after_noeffect_mean": last_diagnostic_mix_after_noeffect,
                }
            )
            metrics = _online_training_metrics(
                model,
                train=curriculum.train,
                heldout=curriculum.heldout,
                device=device,
                step=step,
                train_loss=last_loss,
                active=active,
                eval_runtime_agent=bool(args.eval_runtime_agent),
                extra_metrics=eval_extra_metrics,
                max_steps_per_level=int(args.max_steps_per_level),
            )
            if bool(args.save_best):
                if selection_metric not in metrics:
                    available = ", ".join(sorted(str(key) for key in metrics))
                    raise KeyError(f"--selection-metric {selection_metric!r} is not present in metrics; available: {available}")
                metric_value = float(metrics[selection_metric])
            else:
                metric_value = float("-inf")
            if bool(args.save_best) and metric_value >= best_metric_value:
                best_metric_value = metric_value
                best_step = int(step)
                best_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
                best_summary = dict(metrics)
            print(
                json.dumps(
                    metrics,
                    sort_keys=True,
                ),
                flush=True,
            )

    save_path = args.save_path or args.output
    if save_path is not None and not bool(args.no_save):
        save_state = best_state if bool(args.save_best) and best_state is not None else model.state_dict()
        summary = best_summary if bool(args.save_best) and best_summary is not None else _online_training_metrics(
            model,
            train=curriculum.train,
            heldout=curriculum.heldout,
            device=device,
            step=max(int(args.steps), 1),
            train_loss=last_loss,
            active=active,
            eval_runtime_agent=bool(args.eval_runtime_agent),
            extra_metrics=extraction_metrics,
            max_steps_per_level=int(args.max_steps_per_level),
        )
        if bool(args.save_best):
            summary = {
                **dict(summary),
                "checkpoint_best_step": int(best_step),
                "checkpoint_selection_metric": str(selection_metric),
                "checkpoint_selection_metric_value": float(best_metric_value),
            }
        checkpoint = {
            "seed": int(args.seed),
            "config": config.to_dict(),
            "model_state": save_state,
            "metadata": LearnedOnlineObjectEventAgent.checkpoint_metadata(),
            "training_summary": summary,
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


def _online_selection_metric(args: argparse.Namespace, *, active: bool) -> str:
    if args.selection_metric:
        return str(args.selection_metric)
    return "heldout_active_post_self_update_top1_acc" if active else "heldout_post_update_top1_acc"


def _runtime_extracted_curriculum(split: OnlineObjectEventCurriculumSplit) -> OnlineObjectEventCurriculumSplit:
    return OnlineObjectEventCurriculumSplit(
        train=_runtime_extracted_sessions(split.train),
        heldout=_runtime_extracted_sessions(split.heldout),
    )


def _runtime_extracted_sessions(sessions: Sequence[OnlineObjectEventSession]) -> tuple[OnlineObjectEventSession, ...]:
    return tuple(_runtime_extracted_session(session) for session in sessions)


def _runtime_extracted_session(session: OnlineObjectEventSession) -> OnlineObjectEventSession:
    levels = tuple(_runtime_extracted_level(level) for level in session.levels)
    return OnlineObjectEventSession(
        session_index=int(session.session_index),
        latent_rule=int(session.latent_rule),
        levels=levels,
    )


def _runtime_extracted_level(level: OnlineObjectEventLevel) -> OnlineObjectEventLevel:
    example = level.example
    selected = int(example.correct_action_index)
    before_observation, after_observation, result = apply_synthetic_object_event_action_to_grid(
        level,
        selected,
        before_step_index=int(example.state.step_index),
    )
    extractor = ObjectEventRuntimeExtractor()
    extracted = extract_selected_transition_from_grid(
        extractor,
        before_observation=before_observation,
        selected_action=example.legal_actions[selected],
        after_observation=after_observation,
        reward=float(result.reward),
        terminated=False,
        info={
            "score_delta": float(result.reward),
            "level_boundary": bool(result.success),
            "levels_completed_before": 0,
            "levels_completed_after": int(result.levels_completed),
        },
    )
    extracted_example = rebuild_object_event_example_with_states(
        example,
        state=extracted.before_state,
        next_state=extracted.after_state,
        metadata_extra={"state_source": "extracted"},
    )
    return OnlineObjectEventLevel(
        session_index=int(level.session_index),
        level_index=int(level.level_index),
        geometry_seed=int(level.geometry_seed),
        cue_mode=int(level.cue_mode),
        latent_rule=int(level.latent_rule),
        example=extracted_example,
    )


def _runtime_extraction_diagnostics(
    structured: OnlineObjectEventCurriculumSplit,
    extracted: OnlineObjectEventCurriculumSplit,
) -> dict[str, float]:
    structured_examples = tuple(level.example for session in structured.heldout for level in session.levels)
    extracted_examples = tuple(level.example for session in extracted.heldout for level in session.levels)
    if not structured_examples or not extracted_examples:
        return {
            "structured_state_object_count_mean": 0.0,
            "extracted_state_object_count_mean": 0.0,
            "structured_extracted_object_count_abs_delta_mean": 0.0,
            "structured_extracted_state_token_l2_mean": 0.0,
            "structured_extracted_action_token_l2_mean": 0.0,
            "structured_extracted_affordance_match_rate": 0.0,
            "structured_extracted_action_role_match_rate": 0.0,
            "extracted_num_legal_actions_mean": 0.0,
            "extracted_bridge_selected_action_match_rate": 0.0,
            "extracted_metadata_model_input_forbidden_count": 0.0,
        }
    object_counts_structured: list[float] = []
    object_counts_extracted: list[float] = []
    object_count_delta: list[float] = []
    state_l2: list[float] = []
    action_l2: list[float] = []
    affordance_matches: list[float] = []
    role_matches: list[float] = []
    legal_counts: list[float] = []
    bridge_matches: list[float] = []
    forbidden_count = 0
    for structured_example, extracted_example in zip(structured_examples, extracted_examples):
        object_counts_structured.append(float(len(structured_example.state.objects)))
        object_counts_extracted.append(float(len(extracted_example.state.objects)))
        object_count_delta.append(float(abs(len(structured_example.state.objects) - len(extracted_example.state.objects))))
        state_l2.append(float(np.linalg.norm(structured_example.state_tokens.numeric - extracted_example.state_tokens.numeric)))
        action_l2.append(float(np.linalg.norm(structured_example.action_tokens.numeric - extracted_example.action_tokens.numeric)))
        affordance_matches.append(float(tuple(structured_example.legal_actions) == tuple(extracted_example.legal_actions)))
        role_matches.append(float(dict(structured_example.state.action_roles) == dict(extracted_example.state.action_roles)))
        selected = int(extracted_example.correct_action_index)
        result = apply_synthetic_object_event_action(
            OnlineObjectEventLevel(
                session_index=0,
                level_index=0,
                geometry_seed=0,
                cue_mode=0,
                latent_rule=0,
                example=extracted_example,
            ),
            selected,
        )
        bridge = _synthetic_bridge_observation(extracted_example, selected, result)
        legal_counts.append(float(bridge.legal_action_count))
        bridge_matches.append(float(int(bridge.selected_action_index) == selected))
        try:
            assert_no_forbidden_metadata_as_model_input(bridge.model_input_metadata())
        except AssertionError:
            forbidden_count += 1
    return {
        "structured_state_object_count_mean": float(np.mean(object_counts_structured)),
        "extracted_state_object_count_mean": float(np.mean(object_counts_extracted)),
        "structured_extracted_object_count_abs_delta_mean": float(np.mean(object_count_delta)),
        "structured_extracted_state_token_l2_mean": float(np.mean(state_l2)),
        "structured_extracted_action_token_l2_mean": float(np.mean(action_l2)),
        "structured_extracted_affordance_match_rate": float(np.mean(affordance_matches)),
        "structured_extracted_action_role_match_rate": float(np.mean(role_matches)),
        "extracted_num_legal_actions_mean": float(np.mean(legal_counts)),
        "extracted_bridge_selected_action_match_rate": float(np.mean(bridge_matches)),
        "extracted_metadata_model_input_forbidden_count": float(forbidden_count),
    }


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


def _positive_mask_from_tensors(tensors: dict[str, Any]) -> torch.Tensor:
    return tensors["candidate_value_targets"].to(dtype=torch.float32) > 0.5


def _positive_rank_loss(logits: torch.Tensor, tensors: dict[str, Any]) -> torch.Tensor:
    return _positive_rank_loss_per_row(logits, tensors).mean()


def _positive_rank_loss_per_row(logits: torch.Tensor, tensors: dict[str, Any]) -> torch.Tensor:
    valid = tensors["action_mask"].bool()
    positives = _positive_mask_from_tensors(tensors) & valid
    safe_logits = logits.masked_fill(~valid, -1.0e9)
    log_probs = torch.log_softmax(safe_logits, dim=-1)
    positive_log_probs = log_probs.masked_fill(~positives, -1.0e9)
    return -torch.logsumexp(positive_log_probs, dim=-1)


def _selected_is_positive(tensors: dict[str, Any], selected_indices: torch.Tensor) -> torch.Tensor:
    row = torch.arange(selected_indices.shape[0], device=selected_indices.device)
    return _positive_mask_from_tensors(tensors)[row, selected_indices]


def _selected_noeffect_click_mask(tensors: dict[str, Any], selected_indices: torch.Tensor) -> torch.Tensor:
    row = torch.arange(selected_indices.shape[0], device=selected_indices.device)
    selected_outcome = tensors["candidate_outcome_targets"][row, selected_indices]
    selected_action = tensors["inputs"]["action_numeric"][row, selected_indices]
    no_effect = selected_outcome[:, OUT_NO_EFFECT_NONPROGRESS] > 0.5
    is_click = selected_action[:, ACTION_FEATURE_IS_CLICK] > 0.5
    return no_effect & is_click


def _near_failed_mask_from_tensors(tensors: dict[str, Any], selected_indices: torch.Tensor) -> torch.Tensor:
    actions = tensors["inputs"]["action_numeric"]
    valid = tensors["action_mask"].bool()
    row = torch.arange(selected_indices.shape[0], device=selected_indices.device)
    selected = actions[row, selected_indices]
    selected_is_click = selected[:, ACTION_FEATURE_IS_CLICK] > 0.5
    is_click = actions[..., ACTION_FEATURE_IS_CLICK] > 0.5
    screen_xy = actions[..., ACTION_FEATURE_CLICK_X : ACTION_FEATURE_CLICK_Y + 1]
    selected_xy = selected[:, ACTION_FEATURE_CLICK_X : ACTION_FEATURE_CLICK_Y + 1].unsqueeze(1)
    screen_l1 = torch.abs(screen_xy - selected_xy).sum(dim=-1)
    mapped_rc = actions[..., ACTION_FEATURE_GRID_ROW : ACTION_FEATURE_GRID_COL + 1]
    selected_rc = selected[:, ACTION_FEATURE_GRID_ROW : ACTION_FEATURE_GRID_COL + 1].unsqueeze(1)
    grid_l1 = torch.abs(mapped_rc - selected_rc).sum(dim=-1)
    has_grid = (actions[..., ACTION_FEATURE_HAS_GRID_CELL] > 0.5) & (
        selected[:, ACTION_FEATURE_HAS_GRID_CELL].unsqueeze(1) > 0.5
    )
    # One display cell on the 64-to-8 parametric surface is about 8/63 in
    # normalized screen coordinates. The threshold is only a training feature
    # target; runtime action selection still scores all legal actions.
    near_screen = screen_l1 <= 0.15
    near_grid = has_grid & (grid_l1 <= 0.15)
    selected_click = selected_is_click.unsqueeze(1)
    return valid & is_click & selected_click & (near_screen | near_grid)


def _axis_failed_masks_from_tensors(tensors: dict[str, Any], selected_indices: torch.Tensor) -> dict[str, torch.Tensor]:
    actions = tensors["inputs"]["action_numeric"]
    valid = tensors["action_mask"].bool()
    row = torch.arange(selected_indices.shape[0], device=selected_indices.device)
    selected = actions[row, selected_indices]
    selected_is_click = selected[:, ACTION_FEATURE_IS_CLICK] > 0.5
    is_click = actions[..., ACTION_FEATURE_IS_CLICK] > 0.5
    selected_click = selected_is_click.unsqueeze(1)
    screen_xy = actions[..., ACTION_FEATURE_CLICK_X : ACTION_FEATURE_CLICK_Y + 1]
    selected_xy = selected[:, ACTION_FEATURE_CLICK_X : ACTION_FEATURE_CLICK_Y + 1].unsqueeze(1)
    mapped_rc = actions[..., ACTION_FEATURE_GRID_ROW : ACTION_FEATURE_GRID_COL + 1]
    selected_rc = selected[:, ACTION_FEATURE_GRID_ROW : ACTION_FEATURE_GRID_COL + 1].unsqueeze(1)
    has_grid = (actions[..., ACTION_FEATURE_HAS_GRID_CELL] > 0.5) & (
        selected[:, ACTION_FEATURE_HAS_GRID_CELL].unsqueeze(1) > 0.5
    )
    base = valid & is_click & selected_click
    return {
        "same_x": base & (torch.abs(screen_xy[..., 0] - selected_xy[..., 0]) <= 0.02),
        "same_y": base & (torch.abs(screen_xy[..., 1] - selected_xy[..., 1]) <= 0.02),
        "same_mapped_col": base & has_grid & (torch.abs(mapped_rc[..., 1] - selected_rc[..., 1]) <= 0.15),
        "same_mapped_row": base & has_grid & (torch.abs(mapped_rc[..., 0] - selected_rc[..., 0]) <= 0.15),
    }


def _information_gain_from_hypothesis_success(
    success_by_hypothesis: torch.Tensor,
    hypothesis_prior: torch.Tensor | None = None,
) -> torch.Tensor:
    success = success_by_hypothesis.to(dtype=torch.float32)
    if success.ndim != 3:
        raise ValueError("success_by_hypothesis must have shape [batch, hypotheses, actions]")
    if hypothesis_prior is None:
        prior = torch.full(
            success.shape[:2],
            1.0 / max(int(success.shape[1]), 1),
            dtype=success.dtype,
            device=success.device,
        )
    else:
        prior = hypothesis_prior.to(dtype=success.dtype, device=success.device)
        prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
    p_success = torch.sum(prior[:, :, None] * success, dim=1)
    p_success = p_success.clamp(1.0e-6, 1.0 - 1.0e-6)
    entropy = -(p_success * torch.log(p_success) + (1.0 - p_success) * torch.log(1.0 - p_success))
    return entropy / torch.log(torch.as_tensor(2.0, dtype=success.dtype, device=success.device))


def _diagnostic_information_gain_targets(
    examples: Sequence[ObjectEventExample],
    tensors: dict[str, Any],
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    action_mask = tensors["action_mask"].bool()
    targets = torch.zeros_like(action_mask, dtype=torch.float32)
    device = action_mask.device
    for row, example in enumerate(examples):
        hypothesis_masks: list[torch.Tensor] = []
        for key in ("red_action_index", "blue_action_index"):
            if key not in example.metadata:
                continue
            index = int(example.metadata[key])
            if index < 0 or index >= len(example.legal_actions):
                continue
            mask = torch.zeros((action_mask.shape[1],), dtype=torch.float32, device=device)
            indices = {index}
            indices.update(int(value) for value in _near_action_indices(example, example.legal_actions[index]))
            for candidate in indices:
                if 0 <= int(candidate) < action_mask.shape[1] and bool(action_mask[row, int(candidate)]):
                    mask[int(candidate)] = 1.0
            if float(mask.sum().detach().cpu()) > 0.0:
                hypothesis_masks.append(mask)
        if len(hypothesis_masks) < 2:
            positive = _positive_mask_from_tensors(tensors)[row].to(dtype=torch.float32)
            if float(positive.sum().detach().cpu()) > 0.0:
                targets[row] = positive
            continue
        success = torch.stack(hypothesis_masks, dim=0)[None, :, :]
        prior = torch.ones((1, success.shape[1]), dtype=torch.float32, device=device)
        selected = int(selected_indices[row].detach().cpu())
        likelihood = 1.0 - success[0, :, selected].clamp(0.0, 1.0)
        if float(likelihood.sum().detach().cpu()) > 1.0e-6:
            prior = prior * likelihood[None, :]
        prior = prior / prior.sum(dim=1, keepdim=True).clamp_min(1.0e-6)
        targets[row] = _information_gain_from_hypothesis_success(success, prior)[0] * action_mask[row].to(dtype=torch.float32)
    return targets


def _family_posterior_diagnostic_targets(
    output: Any,
    tensors: dict[str, Any],
    *,
    selected_family_overlap: torch.Tensor | None = None,
) -> torch.Tensor:
    action_mask = tensors["action_mask"].bool()
    features = getattr(output, "action_family_posterior_features", None)
    if features is None:
        return torch.zeros_like(action_mask, dtype=torch.float32)
    posterior = features.to(dtype=torch.float32)
    uncertainty = posterior[..., 1].clamp_min(0.0)
    noeffect_count = posterior[..., 3].clamp_min(0.0)
    if selected_family_overlap is None:
        overlap_penalty = 1.0
    else:
        overlap_penalty = (1.0 - selected_family_overlap.to(dtype=torch.float32)).clamp(0.0, 1.0)
    targets = uncertainty * torch.exp(-0.75 * noeffect_count) * overlap_penalty
    targets = targets.masked_fill(~action_mask, 0.0)
    max_value = targets.max(dim=1, keepdim=True).values.clamp_min(1.0e-6)
    return (targets / max_value).clamp(0.0, 1.0)


def _family_known_noeffect_mask(
    output: Any,
    tensors: dict[str, Any],
) -> torch.Tensor:
    action_mask = tensors["action_mask"].bool()
    features = getattr(output, "action_family_posterior_features", None)
    if features is None:
        return torch.zeros_like(action_mask, dtype=torch.bool)
    posterior = features.to(dtype=torch.float32)
    return action_mask & (posterior[..., 3] >= 0.20)


def _selected_family_overlap(
    before_output: Any,
    after_output: Any,
    selected_indices: torch.Tensor,
) -> torch.Tensor:
    before_probs = getattr(before_output, "action_family_probs", None)
    after_probs = getattr(after_output, "action_family_probs", None)
    if before_probs is None or after_probs is None or before_probs.shape[-1] <= 0:
        action_count = after_output.outcome_logits.shape[1]
        return after_output.outcome_logits.new_zeros((after_output.outcome_logits.shape[0], action_count))
    family_count = before_probs.shape[-1]
    selected = before_probs.gather(
        1,
        selected_indices[:, None, None].expand(-1, 1, family_count),
    ).squeeze(1)
    return torch.sum(after_probs * selected[:, None, :], dim=-1).clamp(0.0, 1.0)


def _family_assignment_regularization_losses(
    output: Any,
    tensors: dict[str, Any],
) -> dict[str, torch.Tensor]:
    probs = getattr(output, "action_family_probs", None)
    if probs is None or probs.shape[-1] <= 0:
        zero = tensors["action_mask"].to(dtype=torch.float32).sum() * 0.0
        return {
            "balance_loss": zero,
            "sharpness_loss": zero,
            "contrastive_loss": zero,
            "positive_score_mean": zero,
            "negative_score_mean": zero,
            "effective_count": zero,
            "usage_min": zero,
            "usage_max": zero,
            "per_action_entropy": zero,
        }
    action_mask = tensors["action_mask"].bool()
    valid_weight = action_mask.to(dtype=probs.dtype)
    valid_count = valid_weight.sum(dim=1, keepdim=True).clamp_min(1.0)
    usage = (probs * valid_weight[..., None]).sum(dim=1) / valid_count
    usage = usage / usage.sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
    uniform = torch.full_like(usage, 1.0 / max(int(usage.shape[-1]), 1))
    balance_loss = F.kl_div((usage + 1.0e-6).log(), uniform, reduction="batchmean")
    per_action_entropy = -(probs.clamp_min(1.0e-6) * probs.clamp_min(1.0e-6).log()).sum(dim=-1)
    entropy_mean = torch.sum(per_action_entropy * valid_weight) / valid_weight.sum().clamp_min(1.0)
    target_entropy = 0.35 * math.log(max(int(probs.shape[-1]), 2))
    sharpness_loss = (entropy_mean - probs.new_tensor(target_entropy)).pow(2)
    surface_entropy = -(usage.clamp_min(1.0e-6) * usage.clamp_min(1.0e-6).log()).sum(dim=-1)
    effective_count = torch.exp(surface_entropy).mean()
    usage_min = usage.min(dim=-1).values.mean()
    usage_max = usage.max(dim=-1).values.mean()
    contrastive = _family_contrastive_loss(output, tensors, probs=probs)
    return {
        "balance_loss": balance_loss,
        "sharpness_loss": sharpness_loss,
        "contrastive_loss": contrastive["loss"],
        "positive_score_mean": contrastive["positive_score_mean"],
        "negative_score_mean": contrastive["negative_score_mean"],
        "effective_count": effective_count,
        "usage_min": usage_min,
        "usage_max": usage_max,
        "per_action_entropy": entropy_mean,
    }


def _family_contrastive_loss(
    output: Any,
    tensors: dict[str, Any],
    *,
    probs: torch.Tensor,
) -> dict[str, torch.Tensor]:
    del output
    actions = tensors["inputs"]["action_numeric"]
    valid = tensors["action_mask"].bool()
    value = tensors["candidate_value_targets"].to(dtype=torch.float32) > 0.5
    is_click = actions[..., ACTION_FEATURE_IS_CLICK] > 0.5
    screen_xy = actions[..., ACTION_FEATURE_CLICK_X : ACTION_FEATURE_CLICK_Y + 1]
    mapped_rc = actions[..., ACTION_FEATURE_GRID_ROW : ACTION_FEATURE_GRID_COL + 1]
    has_grid = actions[..., ACTION_FEATURE_HAS_GRID_CELL] > 0.5
    positive_scores: list[torch.Tensor] = []
    negative_scores: list[torch.Tensor] = []
    for row in range(actions.shape[0]):
        row_valid = valid[row] & is_click[row]
        if int(row_valid.sum().detach().cpu()) < 2:
            continue
        indices = torch.nonzero(row_valid, as_tuple=False).squeeze(1)
        p = probs[row, indices]
        similarity = torch.matmul(p, p.t()).clamp(0.0, 1.0)
        xy = screen_xy[row, indices]
        rc = mapped_rc[row, indices]
        grid = has_grid[row, indices]
        val = value[row, indices]
        screen_l1 = torch.abs(xy[:, None, :] - xy[None, :, :]).sum(dim=-1)
        grid_l1 = torch.abs(rc[:, None, :] - rc[None, :, :]).sum(dim=-1)
        both_grid = grid[:, None] & grid[None, :]
        near = (screen_l1 <= 0.15) | (both_grid & (grid_l1 <= 0.15))
        far = (screen_l1 >= 0.35) | (both_grid & (grid_l1 >= 0.35))
        same_outcome = val[:, None] == val[None, :]
        different_outcome = ~same_outcome
        upper = torch.triu(torch.ones_like(similarity, dtype=torch.bool), diagonal=1)
        positive_mask = upper & near & same_outcome
        negative_mask = upper & far & different_outcome
        if bool(positive_mask.any()):
            positive_scores.append(similarity[positive_mask])
        if bool(negative_mask.any()):
            negative_scores.append(similarity[negative_mask])
    zero = probs.sum() * 0.0
    positive_values = torch.cat(positive_scores) if positive_scores else zero.reshape(1)[:0]
    negative_values = torch.cat(negative_scores) if negative_scores else zero.reshape(1)[:0]
    positive_loss = F.relu(0.75 - positive_values).mean() if positive_values.numel() else zero
    negative_loss = F.relu(negative_values - 0.25).mean() if negative_values.numel() else zero
    positive_mean = positive_values.mean() if positive_values.numel() else zero
    negative_mean = negative_values.mean() if negative_values.numel() else zero
    return {
        "loss": positive_loss + negative_loss,
        "positive_score_mean": positive_mean,
        "negative_score_mean": negative_mean,
    }


def _masked_mean_loss(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return torch.sum(values * weights) / torch.clamp(torch.sum(weights), min=1.0)


def _masked_action_mean(values: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(dtype=values.dtype)
    return torch.sum(values * weights, dim=-1) / torch.clamp(torch.sum(weights, dim=-1), min=1.0)


def _is_positive_index(example: ObjectEventExample, index: int) -> bool:
    if example.positive_action_mask is not None:
        return bool(np.asarray(example.positive_action_mask, dtype=bool)[int(index)])
    return int(index) == int(example.correct_action_index)


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


def _wrong_candidate_indices(
    examples: Sequence[ObjectEventExample],
    *,
    device: torch.device,
) -> torch.Tensor:
    indices: list[int] = []
    for example in examples:
        raw_wrong = example.metadata.get("wrong_action_indices")
        if isinstance(raw_wrong, (list, tuple)) and raw_wrong:
            wrong = [int(index) for index in raw_wrong if not _is_positive_index(example, int(index))]
        else:
            candidates = _candidate_action_indices(example)
            wrong = [int(index) for index in candidates if not _is_positive_index(example, int(index))]
        if not wrong:
            wrong = [
                int(index)
                for index, value in enumerate(example.candidate_targets.value)
                if float(value) <= 0.5 and bool(example.candidate_targets.action_mask[index])
            ]
        candidates = _candidate_action_indices(example)
        indices.append(int(wrong[0] if wrong else candidates[0]))
    return torch.as_tensor(indices, dtype=torch.long, device=device)


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


def _post_failure_retry_examples(
    levels: Sequence[OnlineObjectEventLevel],
    observed_indices: Sequence[int],
) -> tuple[ObjectEventExample, ...]:
    examples: list[ObjectEventExample] = []
    for level, observed_index in zip(levels, observed_indices):
        example = level.example
        result = apply_synthetic_object_event_action(level, int(observed_index))
        before_observation = state_to_grid_observation(
            example.state,
            example.legal_actions,
            step_index=int(example.state.step_index),
        )
        after_state = example.next_state if bool(result.success) else example.state
        after_observation = state_to_grid_observation(
            after_state,
            example.legal_actions,
            step_index=int(example.state.step_index) + 1,
        )
        extracted = extract_selected_transition_from_grid(
            ObjectEventRuntimeExtractor(),
            before_observation=before_observation,
            selected_action=example.legal_actions[int(observed_index)],
            after_observation=after_observation,
            reward=float(result.reward),
            terminated=False,
            info={"score_delta": float(result.reward), "level_boundary": bool(result.success)},
        )
        examples.append(
            rebuild_object_event_example_with_states(
                example,
                state=extracted.after_state,
                next_state=example.next_state,
                metadata_extra={
                    "state_source": str(example.metadata.get("state_source", "structured")),
                    "retry_after_selected_action_index": int(observed_index),
                },
            )
        )
    return tuple(examples)


def _plausible_red_blue_loss(rank_logits: torch.Tensor, examples: Sequence[ObjectEventExample]) -> torch.Tensor:
    mask = torch.zeros_like(rank_logits, dtype=torch.bool)
    for row, example in enumerate(examples):
        for index in _candidate_action_indices(example):
            if 0 <= int(index) < rank_logits.shape[1]:
                mask[row, int(index)] = True
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
    eval_runtime_agent: bool = False,
    extra_metrics: dict[str, Any] | None = None,
    max_steps_per_level: int = 3,
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
    active_closed = _active_rollout_metrics(model, heldout, device=device, max_steps_per_level=int(max_steps_per_level)) if active else {}
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
                "heldout_active_success_within_5": active_closed["success_within_5"],
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
                "actual_target_first_click_rate": active_closed["actual_target_first_click_rate"],
                "level_negative_memory_norm": active_closed["level_negative_memory_norm"],
                "session_relation_memory_norm": heldout_metrics["session_belief_norm"],
                "object_event_bridge_legal_action_count_mean": active_closed["object_event_bridge_legal_action_count_mean"],
                "object_event_bridge_selected_action_match_rate": active_closed["object_event_bridge_selected_action_match_rate"],
                "metadata_model_input_forbidden_count": active_closed["metadata_model_input_forbidden_count"],
                "observed_distractor_failure_count": active_closed["observed_distractor_failure_count"],
                "post_failure_candidate_switch_rate": active_closed["post_failure_candidate_switch_rate"],
                "post_failure_repeat_top1_rate": active_closed["post_failure_repeat_top1_rate"],
                "level0_target_first_click_rate": active_closed["level0_target_first_click_rate"],
                "level0_distractor_first_click_rate": active_closed["level0_distractor_first_click_rate"],
                "level0_success_within_2": active_closed["level0_success_within_2"],
                "level0_success_within_3": active_closed["level0_success_within_3"],
                "post_session_target_first_click_rate": active_closed["post_session_target_first_click_rate"],
                "post_session_distractor_first_click_rate": active_closed["post_session_distractor_first_click_rate"],
                "post_session_next_level_first_try_acc": active_closed["post_session_next_level_first_try_acc"],
                "self_selected_no_effect_count": active_closed["self_selected_no_effect_count"],
                "failed_exact_repeat_top1_rate": active_closed["failed_exact_repeat_top1_rate"],
                "failed_near_repeat_top1_rate": active_closed["failed_near_repeat_top1_rate"],
                "failed_action_score_delta_mean": active_closed["failed_action_score_delta_mean"],
                "max_same_action_streak_mean": active_closed["max_same_action_streak_mean"],
                "max_same_action_streak_max": active_closed["max_same_action_streak_max"],
                "unique_action_count_mean": active_closed["unique_action_count_mean"],
                "object_event_action_surface_capped": False,
                "trace_replay_used": False,
                "graph_controller_used": False,
            }
        )
        metrics.update(_action_surface_metrics(train=train, heldout=heldout))
        metrics.update(_color_role_balance_metrics(train=train, heldout=heldout))
        if eval_runtime_agent:
            metrics.update(_runtime_agent_active_metrics(model, heldout, device=device, max_steps_per_level=int(max_steps_per_level)))
        if "runtime_agent_active_success_within_3" in metrics and "runtime_agent_act_path_active_success_within_3" in metrics:
            metrics["structured_vs_act_path_success_gap"] = (
                float(metrics["runtime_agent_active_success_within_3"])
                - float(metrics["runtime_agent_act_path_active_success_within_3"])
            )
    if extra_metrics:
        metrics.update(extra_metrics)
    return metrics


def _action_surface_metrics(
    *,
    train: Sequence[OnlineObjectEventSession],
    heldout: Sequence[OnlineObjectEventSession],
) -> dict[str, Any]:
    examples = tuple(level.example for session in heldout for level in session.levels)
    if not examples:
        examples = tuple(level.example for session in train for level in session.levels)
    if not examples:
        return {
            "action_surface_kind": "unknown",
            "parametric_action_count_mean": 0.0,
            "parametric_action_count_min": 0.0,
            "parametric_action_count_max": 0.0,
            "parametric_coordinate_grid_size": 0.0,
            "empty_click_fraction_observed": 0.0,
            "object_click_fraction_observed": 0.0,
            "positive_action_count_mean": 0.0,
            "positive_action_count_min": 0.0,
            "positive_action_count_max": 0.0,
        }
    kinds = [str(example.metadata.get("action_surface_kind", "dense_8x8")) for example in examples]
    legal_counts = np.asarray([float(len(example.legal_actions)) for example in examples], dtype=np.float32)
    empty_counts = np.asarray([float(example.metadata.get("empty_action_count", 0.0) or 0.0) for example in examples], dtype=np.float32)
    object_counts = np.asarray([float(example.metadata.get("object_action_count", 0.0) or 0.0) for example in examples], dtype=np.float32)
    positive_counts = np.asarray([float(len(example.positive_action_indices) if example.positive_action_indices else 1.0) for example in examples], dtype=np.float32)
    coord_sizes = np.asarray([float(example.metadata.get("coordinate_grid_size", 0.0) or 0.0) for example in examples], dtype=np.float32)
    dominant_kind = max(sorted(set(kinds)), key=kinds.count)
    return {
        "action_surface_kind": dominant_kind,
        "parametric_action_count_mean": float(np.mean(legal_counts)),
        "parametric_action_count_min": float(np.min(legal_counts)),
        "parametric_action_count_max": float(np.max(legal_counts)),
        "parametric_coordinate_grid_size": float(np.mean(coord_sizes)),
        "empty_click_fraction_observed": float(np.mean(empty_counts / np.maximum(legal_counts, 1.0))),
        "object_click_fraction_observed": float(np.mean(object_counts / np.maximum(legal_counts, 1.0))),
        "positive_action_count_mean": float(np.mean(positive_counts)),
        "positive_action_count_min": float(np.min(positive_counts)),
        "positive_action_count_max": float(np.max(positive_counts)),
    }


def _color_role_balance_metrics(
    *,
    train: Sequence[OnlineObjectEventSession],
    heldout: Sequence[OnlineObjectEventSession],
) -> dict[str, Any]:
    train_target, train_distractor = _color_role_counts(train)
    heldout_target, heldout_distractor = _color_role_counts(heldout)
    colors = set(train_target) | set(train_distractor) | set(heldout_target) | set(heldout_distractor)
    target_min = min((heldout_target.get(color, 0) for color in colors), default=0)
    distractor_min = min((heldout_distractor.get(color, 0) for color in colors), default=0)
    return {
        "train_target_color_count_by_color": {str(color): int(train_target.get(color, 0)) for color in sorted(colors)},
        "train_distractor_color_count_by_color": {str(color): int(train_distractor.get(color, 0)) for color in sorted(colors)},
        "heldout_target_color_count_by_color": {str(color): int(heldout_target.get(color, 0)) for color in sorted(colors)},
        "heldout_distractor_color_count_by_color": {str(color): int(heldout_distractor.get(color, 0)) for color in sorted(colors)},
        "target_color_role_balance_min": float(target_min),
        "distractor_color_role_balance_min": float(distractor_min),
    }


def _color_role_counts(
    sessions: Sequence[OnlineObjectEventSession],
) -> tuple[dict[int, int], dict[int, int]]:
    target: dict[int, int] = {}
    distractor: dict[int, int] = {}
    for session in sessions:
        for level in session.levels:
            metadata = level.example.metadata
            if "target_color" in metadata:
                color = int(metadata["target_color"])
                target[color] = target.get(color, 0) + 1
            raw_distractors = metadata.get("distractor_colors", ())
            if isinstance(raw_distractors, (list, tuple)):
                for raw_color in raw_distractors:
                    color = int(raw_color)
                    distractor[color] = distractor.get(color, 0) + 1
    return target, distractor


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
        wrong_indices = _wrong_candidate_indices(support_examples, device=device)
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
        "pre_top1": _top1_positive(pre_logits, query_tensors),
        "post_top1": _top1_positive(post_logits, query_tensors),
        "next_level_top1": _top1_positive(next_logits, next_tensors),
        "pre_rank_ce": float(_positive_rank_loss(pre_logits, query_tensors).detach().cpu()),
        "post_rank_ce": float(_positive_rank_loss(post_logits, query_tensors).detach().cpu()),
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
            if _is_positive_index(support, first):
                steps.append(1)
                continue
            observed = _with_observed_candidate_targets(support_tensors, torch.as_tensor([first], dtype=torch.long))
            session_belief = _observed_belief_delta(model, observed)
            retry_output = _forward_tensors(model, support_tensors, session_belief=session_belief)
            retry_logits = policy_rank_logits_from_predictions(retry_output, support_tensors["action_mask"])
            second = int(torch.argmax(retry_logits[0]).detach().cpu())
            steps.append(2 if _is_positive_index(support, second) else 4)
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
            "success_within_5": 0.0,
            "post_self_update_top1_acc": 0.0,
            "failed_candidate_repeat_top1_rate": 0.0,
            "failed_candidate_score_delta": 0.0,
            "self_selected_no_effect_count": 0.0,
            "failed_exact_repeat_top1_rate": 0.0,
            "failed_near_repeat_top1_rate": 0.0,
            "failed_action_score_delta_mean": 0.0,
            "failed_near_score_delta_mean": 0.0,
            "failed_far_score_delta_mean": 0.0,
            "failed_neighborhood_top1_rate": 0.0,
            "failed_same_mapped_cell_top1_rate": 0.0,
            "failed_same_x_top1_rate": 0.0,
            "failed_same_y_top1_rate": 0.0,
            "failed_same_x_score_delta_mean": 0.0,
            "failed_same_y_score_delta_mean": 0.0,
            "coordinate_noeffect_memory_norm": 0.0,
            "coordinate_noeffect_count_mean": 0.0,
            "axis_noeffect_memory_norm": 0.0,
            "axis_noeffect_count_mean": 0.0,
            "max_same_action_streak_mean": 0.0,
            "max_same_action_streak_max": 0.0,
            "unique_action_count_mean": 0.0,
            "ordinary_object_click_rate": 0.0,
            "agent_or_cue_click_rate": 0.0,
            "distractor_first_click_rate": 0.0,
            "target_first_click_rate": 0.0,
            "level_negative_memory_norm": 0.0,
            "object_event_bridge_legal_action_count_mean": 0.0,
            "object_event_bridge_selected_action_match_rate": 0.0,
            "metadata_model_input_forbidden_count": 0.0,
            "observed_distractor_failure_count": 0.0,
            "post_failure_candidate_switch_rate": 0.0,
            "post_failure_repeat_top1_rate": 0.0,
            "level0_target_first_click_rate": 0.0,
            "level0_distractor_first_click_rate": 0.0,
            "level0_success_within_2": 0.0,
            "level0_success_within_3": 0.0,
            "post_session_target_first_click_rate": 0.0,
            "post_session_distractor_first_click_rate": 0.0,
            "post_session_next_level_first_try_acc": 0.0,
        }
    success_steps: list[int] = []
    post_update_hits: list[float] = []
    repeat_hits: list[float] = []
    near_repeat_hits: list[float] = []
    failed_deltas: list[float] = []
    failed_near_deltas: list[float] = []
    failed_far_deltas: list[float] = []
    failed_neighborhood_top1_hits: list[float] = []
    failed_same_mapped_cell_top1_hits: list[float] = []
    failed_same_x_hits: list[float] = []
    failed_same_y_hits: list[float] = []
    failed_same_x_deltas: list[float] = []
    failed_same_y_deltas: list[float] = []
    same_streaks: list[float] = []
    unique_counts: list[float] = []
    no_effect_count = 0
    bridge_legal_counts: list[float] = []
    bridge_selected_matches: list[float] = []
    model_input_forbidden_count = 0
    observed_distractor_failures = 0
    ordinary_clicks = 0
    agent_clicks = 0
    target_set_first = 0
    actual_target_first = 0
    distractor_first = 0
    first_clicks = 0
    post_session_target_first = 0
    post_session_distractor_first = 0
    post_session_first_clicks = 0
    post_session_hits: list[float] = []
    total_actions = 0
    level_norms: list[float] = []
    coordinate_norms: list[float] = []
    coordinate_counts: list[float] = []
    axis_norms: list[float] = []
    axis_counts: list[float] = []
    for session in sessions:
        session_belief = torch.zeros((1, model.config.d_model), dtype=torch.float32, device=device)
        for level in session.levels[:1]:
            level_belief = torch.zeros_like(session_belief)
            step_success = max_steps_per_level + 1
            failed_index: int | None = None
            failed_score_before = 0.0
            selected_actions_for_level: list[str] = []
            for step in range(1, max_steps_per_level + 1):
                tensors = _batch_to_tensors(collate_object_event_examples((level.example,)), device=device)
                with torch.no_grad():
                    output = _forward_tensors(model, tensors, session_belief=session_belief, level_belief=level_belief)
                    logits = policy_rank_logits_from_predictions(output, tensors["action_mask"])
                    selected = int(torch.argmax(logits[0]).detach().cpu())
                    selected_actions_for_level.append(level.example.legal_actions[selected])
                    action_row = tensors["inputs"]["action_numeric"][0, selected]
                    is_ordinary = float(action_row[11].detach().cpu()) > 0.5 and float(action_row[24].detach().cpu()) < 0.5
                    is_agent = float(action_row[24].detach().cpu()) > 0.5
                    total_actions += 1
                    ordinary_clicks += int(is_ordinary)
                    agent_clicks += int(is_agent)
                    if step == 1:
                        first_clicks += 1
                        target_indices = set(_candidate_action_indices(level.example))
                        target_set_first += int(selected in target_indices)
                        actual_target_first += int(_is_positive_index(level.example, selected))
                        distractor_first += int(_is_distractor_action(level.example, selected, action_row=action_row))
                    result = apply_synthetic_object_event_action(level, selected)
                    bridge = _synthetic_bridge_observation(level.example, selected, result)
                    bridge_legal_counts.append(float(bridge.legal_action_count))
                    bridge_selected_matches.append(float(int(bridge.selected_action_index) == int(result.selected_action_index)))
                    try:
                        assert_no_forbidden_metadata_as_model_input(bridge.model_input_metadata())
                    except AssertionError:
                        model_input_forbidden_count += 1
                    observed = _with_observed_candidate_targets(tensors, torch.as_tensor([selected], dtype=torch.long, device=device))
                    deltas = _observed_belief_deltas(model, observed)
                    if result.success:
                        session_belief = session_belief + deltas.session_delta
                        level_belief = level_belief + deltas.level_delta
                        level_norms.append(float(torch.linalg.vector_norm(level_belief).detach().cpu()))
                        coord_start = int(model.coordinate_noeffect_memory_rank.start)
                        coord_stop = int(model.coordinate_noeffect_memory_rank.stop)
                        coord_memory = level_belief[:, coord_start:coord_stop]
                        coordinate_norms.append(float(torch.linalg.vector_norm(coord_memory).detach().cpu()))
                        coordinate_counts.append(float(coord_memory[0, 0].detach().cpu()) if coord_memory.numel() else 0.0)
                        axis_start = int(model.axis_noeffect_memory_rank.start)
                        axis_stop = int(model.axis_noeffect_memory_rank.stop)
                        axis_memory = level_belief[:, axis_start:axis_stop]
                        axis_norms.append(float(torch.linalg.vector_norm(axis_memory).detach().cpu()))
                        axis_counts.append(float(axis_memory[0, 0].detach().cpu()) if axis_memory.numel() else 0.0)
                        step_success = step
                        break
                    level_belief = level_belief + deltas.level_delta
                    level_norms.append(float(torch.linalg.vector_norm(level_belief).detach().cpu()))
                    coord_start = int(model.coordinate_noeffect_memory_rank.start)
                    coord_stop = int(model.coordinate_noeffect_memory_rank.stop)
                    coord_memory = level_belief[:, coord_start:coord_stop]
                    coordinate_norms.append(float(torch.linalg.vector_norm(coord_memory).detach().cpu()))
                    coordinate_counts.append(float(coord_memory[0, 0].detach().cpu()) if coord_memory.numel() else 0.0)
                    axis_start = int(model.axis_noeffect_memory_rank.start)
                    axis_stop = int(model.axis_noeffect_memory_rank.stop)
                    axis_memory = level_belief[:, axis_start:axis_stop]
                    axis_norms.append(float(torch.linalg.vector_norm(axis_memory).detach().cpu()))
                    axis_counts.append(float(axis_memory[0, 0].detach().cpu()) if axis_memory.numel() else 0.0)
                    if failed_index is None:
                        no_effect_count += 1
                        failed_index = selected
                        failed_score_before = float(logits[0, selected].detach().cpu())
                        retry_output = _forward_tensors(model, tensors, session_belief=session_belief, level_belief=level_belief)
                        retry_logits = policy_rank_logits_from_predictions(retry_output, tensors["action_mask"])
                        retry_selected = int(torch.argmax(retry_logits[0]).detach().cpu())
                        post_update_hits.append(float(_is_positive_index(level.example, retry_selected)))
                        repeat = float(retry_selected == selected)
                        repeat_hits.append(repeat)
                        near_repeat_hits.append(
                            float(_is_near_repeat_action(level.example.legal_actions[selected], level.example.legal_actions[retry_selected], level.example))
                        )
                        failed_deltas.append(float(retry_logits[0, selected].detach().cpu()) - failed_score_before)
                        near_indices = _near_action_indices(level.example, level.example.legal_actions[selected])
                        far_indices = _far_click_action_indices(level.example, level.example.legal_actions[selected])
                        failed_near_deltas.append(_score_delta_for_indices(logits[0], retry_logits[0], near_indices))
                        failed_far_deltas.append(_score_delta_for_indices(logits[0], retry_logits[0], far_indices))
                        failed_neighborhood_top1_hits.append(float(retry_selected in near_indices))
                        failed_same_mapped_cell_top1_hits.append(
                            float(_is_same_mapped_cell_action(level.example.legal_actions[selected], level.example.legal_actions[retry_selected], level.example))
                        )
                        same_x_indices = _same_x_action_indices(level.example, level.example.legal_actions[selected])
                        same_y_indices = _same_y_action_indices(level.example, level.example.legal_actions[selected])
                        failed_same_x_hits.append(float(retry_selected in same_x_indices))
                        failed_same_y_hits.append(float(retry_selected in same_y_indices))
                        failed_same_x_deltas.append(_score_delta_for_indices(logits[0], retry_logits[0], same_x_indices))
                        failed_same_y_deltas.append(_score_delta_for_indices(logits[0], retry_logits[0], same_y_indices))
                        if _is_distractor_action(level.example, selected, action_row=action_row):
                            observed_distractor_failures += 1
            same_streaks.append(float(_max_same_action_streak(selected_actions_for_level)))
            unique_counts.append(float(len(set(selected_actions_for_level))))
            success_steps.append(step_success)
        if len(session.levels) > 1:
            query_level = session.levels[1]
            query = query_level.example
            query_tensors = _batch_to_tensors(collate_object_event_examples((query,)), device=device)
            with torch.no_grad():
                query_output = _forward_tensors(model, query_tensors, session_belief=session_belief)
                query_logits = policy_rank_logits_from_predictions(query_output, query_tensors["action_mask"])
                query_selected = int(torch.argmax(query_logits[0]).detach().cpu())
                query_action_row = query_tensors["inputs"]["action_numeric"][0, query_selected]
            post_session_first_clicks += 1
            post_session_target_first += int(_is_positive_index(query, query_selected))
            post_session_distractor_first += int(_is_distractor_action(query, query_selected, action_row=query_action_row))
            post_session_hits.append(float(_is_positive_index(query, query_selected)))
            if not _is_positive_index(query, query_selected):
                with torch.no_grad():
                    query_result = apply_synthetic_object_event_action(query_level, query_selected)
                    query_observed = _with_observed_candidate_targets(
                        query_tensors,
                        torch.as_tensor([query_selected], dtype=torch.long, device=device),
                    )
                    query_deltas = _observed_belief_deltas(model, query_observed)
                    query_level_belief = query_deltas.level_delta if not query_result.success else query_deltas.session_delta
                    query_retry_output = _forward_tensors(
                        model,
                        query_tensors,
                        session_belief=session_belief,
                        level_belief=query_level_belief,
                    )
                    query_retry_logits = policy_rank_logits_from_predictions(query_retry_output, query_tensors["action_mask"])
                    query_retry_selected = int(torch.argmax(query_retry_logits[0]).detach().cpu())
                    post_update_hits.append(float(_is_positive_index(query, query_retry_selected)))
                    repeat = float(query_retry_selected == query_selected)
                    repeat_hits.append(repeat)
                    near_repeat_hits.append(
                        float(_is_near_repeat_action(query.legal_actions[query_selected], query.legal_actions[query_retry_selected], query))
                    )
                    failed_deltas.append(
                        float(query_retry_logits[0, query_selected].detach().cpu())
                        - float(query_logits[0, query_selected].detach().cpu())
                    )
                    near_indices = _near_action_indices(query, query.legal_actions[query_selected])
                    far_indices = _far_click_action_indices(query, query.legal_actions[query_selected])
                    failed_near_deltas.append(_score_delta_for_indices(query_logits[0], query_retry_logits[0], near_indices))
                    failed_far_deltas.append(_score_delta_for_indices(query_logits[0], query_retry_logits[0], far_indices))
                    failed_neighborhood_top1_hits.append(float(query_retry_selected in near_indices))
                    failed_same_mapped_cell_top1_hits.append(
                        float(_is_same_mapped_cell_action(query.legal_actions[query_selected], query.legal_actions[query_retry_selected], query))
                    )
                    same_x_indices = _same_x_action_indices(query, query.legal_actions[query_selected])
                    same_y_indices = _same_y_action_indices(query, query.legal_actions[query_selected])
                    failed_same_x_hits.append(float(query_retry_selected in same_x_indices))
                    failed_same_y_hits.append(float(query_retry_selected in same_y_indices))
                    failed_same_x_deltas.append(_score_delta_for_indices(query_logits[0], query_retry_logits[0], same_x_indices))
                    failed_same_y_deltas.append(_score_delta_for_indices(query_logits[0], query_retry_logits[0], same_y_indices))
    values = np.asarray(success_steps, dtype=np.float32)
    total_clicks = max(total_actions, 1)
    return {
        "success_within_1": float(np.mean(values <= 1.0)),
        "success_within_2": float(np.mean(values <= 2.0)),
        "success_within_3": float(np.mean(values <= 3.0)),
        "success_within_5": float(np.mean(values <= 5.0)),
        "post_self_update_top1_acc": float(np.mean(post_update_hits)) if post_update_hits else 1.0,
        "failed_candidate_repeat_top1_rate": float(np.mean(repeat_hits)) if repeat_hits else 0.0,
        "failed_candidate_score_delta": float(np.mean(failed_deltas)) if failed_deltas else 0.0,
        "self_selected_no_effect_count": float(no_effect_count),
        "failed_exact_repeat_top1_rate": float(np.mean(repeat_hits)) if repeat_hits else 0.0,
        "failed_near_repeat_top1_rate": float(np.mean(near_repeat_hits)) if near_repeat_hits else 0.0,
        "failed_action_score_delta_mean": float(np.mean(failed_deltas)) if failed_deltas else 0.0,
        "failed_near_score_delta_mean": float(np.mean(failed_near_deltas)) if failed_near_deltas else 0.0,
        "failed_far_score_delta_mean": float(np.mean(failed_far_deltas)) if failed_far_deltas else 0.0,
        "failed_neighborhood_top1_rate": float(np.mean(failed_neighborhood_top1_hits)) if failed_neighborhood_top1_hits else 0.0,
        "failed_same_mapped_cell_top1_rate": float(np.mean(failed_same_mapped_cell_top1_hits)) if failed_same_mapped_cell_top1_hits else 0.0,
        "failed_same_x_top1_rate": float(np.mean(failed_same_x_hits)) if failed_same_x_hits else 0.0,
        "failed_same_y_top1_rate": float(np.mean(failed_same_y_hits)) if failed_same_y_hits else 0.0,
        "failed_same_x_score_delta_mean": float(np.mean(failed_same_x_deltas)) if failed_same_x_deltas else 0.0,
        "failed_same_y_score_delta_mean": float(np.mean(failed_same_y_deltas)) if failed_same_y_deltas else 0.0,
        "coordinate_noeffect_memory_norm": float(np.mean(coordinate_norms)) if coordinate_norms else 0.0,
        "coordinate_noeffect_count_mean": float(np.mean(coordinate_counts)) if coordinate_counts else 0.0,
        "axis_noeffect_memory_norm": float(np.mean(axis_norms)) if axis_norms else 0.0,
        "axis_noeffect_count_mean": float(np.mean(axis_counts)) if axis_counts else 0.0,
        "max_same_action_streak_mean": float(np.mean(same_streaks)) if same_streaks else 0.0,
        "max_same_action_streak_max": float(np.max(same_streaks)) if same_streaks else 0.0,
        "unique_action_count_mean": float(np.mean(unique_counts)) if unique_counts else 0.0,
        "ordinary_object_click_rate": float(ordinary_clicks / total_clicks),
        "agent_or_cue_click_rate": float(agent_clicks / total_clicks),
        "distractor_first_click_rate": float(distractor_first / max(first_clicks, 1)),
        "target_first_click_rate": float(target_set_first / max(first_clicks, 1)),
        "actual_target_first_click_rate": float(actual_target_first / max(first_clicks, 1)),
        "level_negative_memory_norm": float(np.mean(level_norms)) if level_norms else 0.0,
        "object_event_bridge_legal_action_count_mean": float(np.mean(bridge_legal_counts)) if bridge_legal_counts else 0.0,
        "object_event_bridge_selected_action_match_rate": float(np.mean(bridge_selected_matches)) if bridge_selected_matches else 0.0,
        "metadata_model_input_forbidden_count": float(model_input_forbidden_count),
        "observed_distractor_failure_count": float(observed_distractor_failures),
        "post_failure_candidate_switch_rate": float(1.0 - np.mean(repeat_hits)) if repeat_hits else 0.0,
        "post_failure_repeat_top1_rate": float(np.mean(repeat_hits)) if repeat_hits else 0.0,
        "level0_target_first_click_rate": float(actual_target_first / max(first_clicks, 1)),
        "level0_distractor_first_click_rate": float(distractor_first / max(first_clicks, 1)),
        "level0_success_within_2": float(np.mean(values <= 2.0)),
        "level0_success_within_3": float(np.mean(values <= 3.0)),
        "post_session_target_first_click_rate": float(post_session_target_first / max(post_session_first_clicks, 1)),
        "post_session_distractor_first_click_rate": float(post_session_distractor_first / max(post_session_first_clicks, 1)),
        "post_session_next_level_first_try_acc": float(np.mean(post_session_hits)) if post_session_hits else 0.0,
    }


def _candidate_action_indices(example: ObjectEventExample) -> tuple[int, ...]:
    raw = example.metadata.get("candidate_action_indices")
    if isinstance(raw, (list, tuple)):
        return tuple(int(item) for item in raw)
    return (int(example.metadata["red_action_index"]), int(example.metadata["blue_action_index"]))


def _max_same_action_streak(actions: Sequence[str]) -> int:
    best = 0
    current = 0
    previous: str | None = None
    for action in actions:
        if action == previous:
            current += 1
        else:
            previous = action
            current = 1
        best = max(best, current)
    return int(best)


def _selected_action_diversity(
    *,
    model: ObjectEventModel,
    example: ObjectEventExample,
    selected_actions: Sequence[str],
    device: torch.device,
) -> dict[str, float]:
    if not selected_actions:
        return {
            "selected_unique_x_count": 0.0,
            "selected_unique_mapped_col_count": 0.0,
            "selected_unique_family_count": 0.0,
            "selected_family_entropy": 0.0,
        }
    action_to_index = {str(action): index for index, action in enumerate(example.legal_actions)}
    selected_indices = tuple(action_to_index[str(action)] for action in selected_actions if str(action) in action_to_index)
    click_x_values: set[int] = set()
    mapped_cols: set[float] = set()
    family_ids: list[int] = []
    for action in selected_actions:
        click = parse_click_action(str(action))
        if click is not None:
            click_x_values.add(int(click[0]))
    if selected_indices:
        numeric_np = example.action_tokens.numeric[list(selected_indices)]
        for row in numeric_np:
            if float(row[ACTION_FEATURE_HAS_GRID_CELL]) > 0.5:
                mapped_cols.add(round(float(row[ACTION_FEATURE_GRID_COL]), 4))
        numeric = torch.as_tensor(numeric_np[None, :, :], dtype=torch.float32, device=device)
        with torch.no_grad():
            probs = model.action_family_belief.family_probs(numeric)
        if probs.numel() > 0:
            family_ids = [int(value) for value in torch.argmax(probs[0], dim=-1).detach().cpu().tolist()]
    if family_ids:
        counts = np.asarray([family_ids.count(family) for family in sorted(set(family_ids))], dtype=np.float64)
        probabilities = counts / max(float(np.sum(counts)), 1.0)
        entropy = -float(np.sum(probabilities * np.log(np.clip(probabilities, 1.0e-12, 1.0))))
    else:
        entropy = 0.0
    return {
        "selected_unique_x_count": float(len(click_x_values)),
        "selected_unique_mapped_col_count": float(len(mapped_cols)),
        "selected_unique_family_count": float(len(set(family_ids))),
        "selected_family_entropy": float(entropy),
    }


def _is_near_repeat_action(first: str, second: str, example: ObjectEventExample) -> bool:
    if first == second:
        return True
    first_click = parse_click_action(first)
    second_click = parse_click_action(second)
    if first_click is None or second_click is None:
        return False
    scale = int(example.metadata.get("interface_display_scale", 1) or 1)
    if abs(int(first_click[0]) - int(second_click[0])) + abs(int(first_click[1]) - int(second_click[1])) <= max(scale, 1):
        return True
    inventory = example.state.inventory_dict()
    from arcagi.core.action_schema import click_action_to_grid_cell

    first_cell = click_action_to_grid_cell(first, grid_shape=example.state.grid_shape, inventory=inventory)
    second_cell = click_action_to_grid_cell(second, grid_shape=example.state.grid_shape, inventory=inventory)
    return first_cell is not None and first_cell == second_cell


def _is_same_mapped_cell_action(first: str, second: str, example: ObjectEventExample) -> bool:
    first_click = parse_click_action(first)
    second_click = parse_click_action(second)
    if first_click is None or second_click is None:
        return False
    from arcagi.core.action_schema import click_action_to_grid_cell

    inventory = example.state.inventory_dict()
    first_cell = click_action_to_grid_cell(first, grid_shape=example.state.grid_shape, inventory=inventory)
    second_cell = click_action_to_grid_cell(second, grid_shape=example.state.grid_shape, inventory=inventory)
    return first_cell is not None and first_cell == second_cell


def _is_same_click_x_action(first: str, second: str) -> bool:
    first_click = parse_click_action(first)
    second_click = parse_click_action(second)
    return first_click is not None and second_click is not None and int(first_click[0]) == int(second_click[0])


def _is_same_click_y_action(first: str, second: str) -> bool:
    first_click = parse_click_action(first)
    second_click = parse_click_action(second)
    return first_click is not None and second_click is not None and int(first_click[1]) == int(second_click[1])


def _same_x_action_indices(example: ObjectEventExample, selected_action: str) -> tuple[int, ...]:
    return tuple(
        index
        for index, action in enumerate(example.legal_actions)
        if _is_same_click_x_action(selected_action, action)
    )


def _same_y_action_indices(example: ObjectEventExample, selected_action: str) -> tuple[int, ...]:
    return tuple(
        index
        for index, action in enumerate(example.legal_actions)
        if _is_same_click_y_action(selected_action, action)
    )


def _near_action_indices(example: ObjectEventExample, selected_action: str) -> tuple[int, ...]:
    return tuple(
        index
        for index, action in enumerate(example.legal_actions)
        if _is_near_repeat_action(selected_action, action, example)
    )


def _far_click_action_indices(example: ObjectEventExample, selected_action: str) -> tuple[int, ...]:
    return tuple(
        index
        for index, action in enumerate(example.legal_actions)
        if parse_click_action(action) is not None and not _is_near_repeat_action(selected_action, action, example)
    )


def _score_delta_for_indices(before: torch.Tensor, after: torch.Tensor, indices: Sequence[int]) -> float:
    if not indices:
        return 0.0
    index_tensor = torch.as_tensor(tuple(int(index) for index in indices), dtype=torch.long, device=before.device)
    return float(torch.mean(after[index_tensor] - before[index_tensor]).detach().cpu())


_RANK_DIAGNOSTIC_FLOAT_KEYS = (
    "rank_component_relation_raw_std",
    "rank_component_relation_std",
    "rank_component_axis_noeffect_raw_std",
    "rank_component_axis_noeffect_std",
    "rank_component_coordinate_noeffect_raw_std",
    "rank_component_coordinate_noeffect_std",
    "rank_component_failed_action_raw_std",
    "rank_component_failed_action_std",
    "rank_component_gate_base",
    "rank_component_gate_relation",
    "rank_component_gate_failed_action",
    "rank_component_gate_coordinate_noeffect",
    "rank_component_gate_axis_noeffect",
    "rank_component_noeffect_boost_mean",
    "relation_object_prior_scale",
    "relation_positive_prior_scale",
    "relation_negative_prior_scale",
    "relation_repeat_penalty_scale",
    "relation_contradiction_gate_mean",
    "relation_contradiction_gate_failed_column_mean",
    "relation_object_prior_failed_column_mean",
    "relation_positive_prior_failed_column_mean",
    "relation_prior_after_contradiction_failed_column_mean",
    "rank_component_relation_minus_axis_failed_column_mean",
    "rank_component_relation_minus_noeffect_failed_column_mean",
    "top_score_same_x_fraction",
    "top_score_same_mapped_col_fraction",
    "runtime_learned_diagnostic_utility_std",
    "runtime_entropy_tiebreak_std",
    "runtime_diagnostic_mix",
    "learned_diagnostic_utility_raw_std",
    "learned_diagnostic_utility_std",
    "diagnostic_mix_model_value",
    "runtime_rank_diag_top_action_agreement",
    "action_family_belief_mean_effect_mean",
    "action_family_belief_uncertainty_mean",
    "action_family_belief_count_mean",
    "action_family_belief_noeffect_mean",
    "action_family_belief_uncertainty_top_value",
    "action_family_belief_noeffect_top_value",
    "family_assignment_surface_entropy",
    "family_assignment_per_action_entropy",
    "family_assignment_usage_min",
    "family_assignment_usage_max",
    "family_assignment_effective_count",
)


def _new_rank_diagnostic_series() -> dict[str, list[float]]:
    return {key: [] for key in _RANK_DIAGNOSTIC_FLOAT_KEYS}


def _append_rank_diagnostics(series: dict[str, list[float]], diagnostics: dict[str, Any]) -> None:
    for key in _RANK_DIAGNOSTIC_FLOAT_KEYS:
        value = diagnostics.get(key)
        if isinstance(value, (int, float, np.integer, np.floating)) and np.isfinite(float(value)):
            series[key].append(float(value))


def _mean_rank_diagnostic_metrics(prefix: str, series: dict[str, list[float]]) -> dict[str, float]:
    return {
        f"{prefix}{key}": float(np.mean(values)) if values else 0.0
        for key, values in series.items()
    }


def _is_distractor_action(
    example: ObjectEventExample,
    selected: int,
    *,
    action_row: torch.Tensor,
) -> bool:
    is_ordinary = float(action_row[11].detach().cpu()) > 0.5 and float(action_row[24].detach().cpu()) < 0.5
    return bool(is_ordinary and int(selected) not in set(_candidate_action_indices(example)))


def _synthetic_bridge_observation(example: ObjectEventExample, selected: int, result: Any):
    after = example.next_state if bool(result.success) else example.state
    bridge = build_object_event_observation(
        SelectedActionObservation(
            before=example.state,
            selected_action=example.legal_actions[int(selected)],
            after=after,
            reward=float(result.reward),
            terminated=False,
            legal_actions=example.legal_actions,
            info={"score_delta": float(result.reward)},
        ),
        metadata=result.metadata,
    )
    return bridge


def _runtime_agent_active_metrics(
    model: ObjectEventModel,
    sessions: Sequence[OnlineObjectEventSession],
    *,
    device: torch.device,
    max_steps_per_level: int = 3,
) -> dict[str, float]:
    if not sessions:
        return {
            "runtime_agent_rank_logits_used": 0.0,
            "runtime_agent_num_legal_actions_mean": 0.0,
            "runtime_agent_top1_matches_model_policy_top1_rate": 0.0,
            "runtime_agent_active_success_within_1": 0.0,
            "runtime_agent_active_success_within_2": 0.0,
            "runtime_agent_active_success_within_3": 0.0,
            "runtime_agent_active_success_within_5": 0.0,
            "runtime_agent_next_level_first_try_acc": 0.0,
            "runtime_agent_next_level_with_session_reset_acc": 0.0,
            "runtime_agent_post_failure_repeat_top1_rate": 0.0,
            "runtime_agent_bridge_selected_action_match_rate": 0.0,
            "runtime_agent_metadata_model_input_forbidden_count": 0.0,
            "runtime_agent_act_path_rank_logits_used": 0.0,
            "runtime_agent_act_path_num_legal_actions_mean": 0.0,
            "runtime_agent_act_path_active_success_within_1": 0.0,
            "runtime_agent_act_path_active_success_within_2": 0.0,
            "runtime_agent_act_path_active_success_within_3": 0.0,
            "runtime_agent_act_path_active_success_within_5": 0.0,
            "runtime_agent_act_path_next_level_first_try_acc": 0.0,
            "runtime_agent_act_path_next_level_with_session_reset_acc": 0.0,
            "runtime_agent_act_path_bridge_selected_action_match_rate": 0.0,
            "runtime_agent_act_path_metadata_model_input_forbidden_count": 0.0,
            "runtime_agent_act_path_online_update_count_mean": 0.0,
            "runtime_agent_act_path_failed_near_score_delta_mean": 0.0,
            "runtime_agent_act_path_failed_far_score_delta_mean": 0.0,
            "runtime_agent_act_path_failed_neighborhood_top1_rate": 0.0,
            "runtime_agent_act_path_failed_same_mapped_cell_top1_rate": 0.0,
            "runtime_agent_act_path_selected_unique_x_count": 0.0,
            "runtime_agent_act_path_selected_unique_mapped_col_count": 0.0,
            "runtime_agent_act_path_selected_unique_family_count": 0.0,
            "runtime_agent_act_path_selected_family_entropy": 0.0,
        }
    base_state = {key: value.detach().cpu().clone() for key, value in model.state_dict().items()}
    success_steps: list[int] = []
    next_hits: list[float] = []
    reset_hits: list[float] = []
    rank_used: list[float] = []
    learned_diag_used: list[float] = []
    legal_counts: list[float] = []
    axis_component_stds: list[float] = []
    relation_component_stds: list[float] = []
    diag_series = _new_rank_diagnostic_series()
    match_hits: list[float] = []
    repeat_hits: list[float] = []
    bridge_matches: list[float] = []
    forbidden_count = 0
    for session in sessions:
        agent = LearnedOnlineObjectEventAgent(
            seed=0,
            config=model.config,
            device=device,
            temperature=0.01,
            epsilon_floor=0.0,
        )
        agent.model.load_state_dict(base_state)
        agent.reset_episode()
        first_level = session.levels[0]
        first_example = first_level.example
        step_success = max_steps_per_level + 1
        first_failed_action: str | None = None
        for step in range(1, max_steps_per_level + 1):
            decisions = agent.score_actions_for_state(first_example.state, first_example.legal_actions)
            selected_action = max(decisions.values(), key=lambda decision: decision.score).action
            selected = int(first_example.legal_actions.index(selected_action))
            diagnostics = agent.diagnostics()
            rank_used.append(float(bool(diagnostics.get("runtime_rank_logits_used", False))))
            learned_diag_used.append(float(bool(diagnostics.get("runtime_learned_diagnostic_utility_used", False))))
            legal_counts.append(float(diagnostics.get("legal_action_count", 0) or 0))
            axis_component_stds.append(float(diagnostics.get("rank_component_axis_noeffect_std", 0.0) or 0.0))
            relation_component_stds.append(float(diagnostics.get("rank_component_relation_std", 0.0) or 0.0))
            _append_rank_diagnostics(diag_series, diagnostics)
            if step == 1:
                model_top1 = _model_policy_top1(model, first_example, device=device)
                match_hits.append(float(selected == model_top1))
            result = apply_synthetic_object_event_action(first_level, selected)
            bridge = _synthetic_bridge_observation(first_example, selected, result)
            bridge_matches.append(float(int(bridge.selected_action_index) == selected))
            try:
                assert_no_forbidden_metadata_as_model_input(bridge.model_input_metadata())
            except AssertionError:
                forbidden_count += 1
            transition = _synthetic_transition(first_example, selected_action, result)
            agent.on_transition(transition)
            if result.success:
                step_success = step
                break
            if first_failed_action is None:
                first_failed_action = selected_action
                retry_scores = agent.score_actions_for_state(first_example.state, first_example.legal_actions)
                retry_selected = max(retry_scores.values(), key=lambda decision: decision.score).action
                repeat_hits.append(float(retry_selected == first_failed_action))
        success_steps.append(step_success)
        if len(session.levels) > 1:
            query = session.levels[1].example
            decisions = agent.score_actions_for_state(query.state, query.legal_actions)
            selected_action = max(decisions.values(), key=lambda decision: decision.score).action
            next_hits.append(float(_is_positive_index(query, int(query.legal_actions.index(selected_action)))))
            reset_agent = LearnedOnlineObjectEventAgent(
                seed=0,
                config=model.config,
                device=device,
                temperature=0.01,
                epsilon_floor=0.0,
            )
            reset_agent.model.load_state_dict(base_state)
            reset_decisions = reset_agent.score_actions_for_state(query.state, query.legal_actions)
            reset_action = max(reset_decisions.values(), key=lambda decision: decision.score).action
            reset_hits.append(float(_is_positive_index(query, int(query.legal_actions.index(reset_action)))))
    values = np.asarray(success_steps, dtype=np.float32)
    metrics = {
        "runtime_agent_rank_logits_used": float(np.mean(rank_used)) if rank_used else 0.0,
        "runtime_agent_learned_diagnostic_utility_used": float(np.mean(learned_diag_used)) if learned_diag_used else 0.0,
        "runtime_agent_num_legal_actions_mean": float(np.mean(legal_counts)) if legal_counts else 0.0,
        "runtime_agent_top1_matches_model_policy_top1_rate": float(np.mean(match_hits)) if match_hits else 0.0,
        "runtime_agent_active_success_within_1": float(np.mean(values <= 1.0)),
        "runtime_agent_active_success_within_2": float(np.mean(values <= 2.0)),
        "runtime_agent_active_success_within_3": float(np.mean(values <= 3.0)),
        "runtime_agent_active_success_within_5": float(np.mean(values <= 5.0)),
        "runtime_agent_next_level_first_try_acc": float(np.mean(next_hits)) if next_hits else 0.0,
        "runtime_agent_next_level_with_session_reset_acc": float(np.mean(reset_hits)) if reset_hits else 0.0,
        "runtime_agent_post_failure_repeat_top1_rate": float(np.mean(repeat_hits)) if repeat_hits else 0.0,
        "runtime_agent_bridge_selected_action_match_rate": float(np.mean(bridge_matches)) if bridge_matches else 0.0,
        "runtime_agent_metadata_model_input_forbidden_count": float(forbidden_count),
        "runtime_agent_rank_component_axis_noeffect_std": float(np.mean(axis_component_stds)) if axis_component_stds else 0.0,
        "runtime_agent_rank_component_relation_std": float(np.mean(relation_component_stds)) if relation_component_stds else 0.0,
    }
    metrics.update(_mean_rank_diagnostic_metrics("runtime_agent_", diag_series))
    metrics.update(
        _runtime_agent_act_path_metrics(
            model_state=base_state,
            config=model.config,
            sessions=sessions,
            device=device,
            max_steps_per_level=max_steps_per_level,
        )
    )
    return metrics


def _runtime_agent_act_path_metrics(
    *,
    model_state: dict[str, torch.Tensor],
    config: ObjectEventModelConfig,
    sessions: Sequence[OnlineObjectEventSession],
    device: torch.device,
    max_steps_per_level: int,
) -> dict[str, float]:
    success_steps: list[int] = []
    next_hits: list[float] = []
    reset_hits: list[float] = []
    rank_used: list[float] = []
    learned_diag_used: list[float] = []
    legal_counts: list[float] = []
    bridge_matches: list[float] = []
    online_counts: list[float] = []
    coordinate_norms: list[float] = []
    coordinate_counts: list[float] = []
    axis_norms: list[float] = []
    axis_counts: list[float] = []
    axis_component_stds: list[float] = []
    relation_component_stds: list[float] = []
    diag_series = _new_rank_diagnostic_series()
    exact_repeat_hits: list[float] = []
    near_repeat_hits: list[float] = []
    failed_deltas: list[float] = []
    failed_near_deltas: list[float] = []
    failed_far_deltas: list[float] = []
    failed_neighborhood_top1_hits: list[float] = []
    failed_same_mapped_cell_top1_hits: list[float] = []
    failed_same_x_hits: list[float] = []
    failed_same_y_hits: list[float] = []
    failed_same_x_deltas: list[float] = []
    failed_same_y_deltas: list[float] = []
    same_streaks: list[float] = []
    unique_counts: list[float] = []
    unique_x_counts: list[float] = []
    unique_mapped_col_counts: list[float] = []
    unique_family_counts: list[float] = []
    family_entropies: list[float] = []
    no_effect_count = 0
    forbidden_count = 0
    for session_index, session in enumerate(sessions):
        agent = LearnedOnlineObjectEventAgent(
            seed=session_index,
            config=config,
            device=device,
            temperature=0.01,
            epsilon_floor=0.0,
        )
        agent.model.load_state_dict(model_state)
        agent.reset_episode()
        first_level = session.levels[0]
        first_example = first_level.example
        step_success = max_steps_per_level + 1
        selected_actions_for_level: list[str] = []
        first_failed_action: str | None = None
        first_failed_score = 0.0
        for step in range(1, max_steps_per_level + 1):
            observation = _grid_observation_from_state(first_example.state, first_example.legal_actions, step_index=step - 1)
            selected_action = agent.act(observation)
            selected = int(first_example.legal_actions.index(selected_action))
            selected_actions_for_level.append(selected_action)
            diagnostics = agent.diagnostics()
            rank_used.append(float(bool(diagnostics.get("runtime_rank_logits_used", False))))
            learned_diag_used.append(float(bool(diagnostics.get("runtime_learned_diagnostic_utility_used", False))))
            legal_counts.append(float(diagnostics.get("legal_action_count", 0) or 0))
            axis_component_stds.append(float(diagnostics.get("rank_component_axis_noeffect_std", 0.0) or 0.0))
            relation_component_stds.append(float(diagnostics.get("rank_component_relation_std", 0.0) or 0.0))
            _append_rank_diagnostics(diag_series, diagnostics)
            result = apply_synthetic_object_event_action(first_level, selected)
            bridge = _synthetic_bridge_observation(first_example, selected, result)
            bridge_matches.append(float(int(bridge.selected_action_index) == selected))
            try:
                assert_no_forbidden_metadata_as_model_input(bridge.model_input_metadata())
            except AssertionError:
                forbidden_count += 1
            next_state = first_example.next_state if bool(result.success) else first_example.state
            next_observation = _grid_observation_from_state(next_state, first_example.legal_actions, step_index=step)
            agent.update_after_step(
                next_observation,
                reward=float(result.reward),
                terminated=False,
                info={"score_delta": float(result.reward), "level_boundary": bool(result.success)},
            )
            post_update_diag = agent.diagnostics()
            coordinate_norms.append(float(post_update_diag.get("coordinate_noeffect_memory_norm", 0.0) or 0.0))
            coordinate_counts.append(float(post_update_diag.get("coordinate_noeffect_count", 0.0) or 0.0))
            axis_norms.append(float(post_update_diag.get("axis_noeffect_memory_norm", 0.0) or 0.0))
            axis_counts.append(float(post_update_diag.get("axis_noeffect_count", 0.0) or 0.0))
            if result.success:
                step_success = step
                break
            if first_failed_action is None:
                no_effect_count += 1
                first_failed_action = selected_action
                first_failed_score = float(agent.last_scores.get(selected_action, 0.0))
                before_score_snapshot = dict(agent.last_scores)
                retry_scores = agent.score_actions_for_state(first_example.state, first_example.legal_actions)
                _append_rank_diagnostics(diag_series, agent.diagnostics())
                retry_selected_action = max(retry_scores.values(), key=lambda decision: decision.score).action
                exact_repeat_hits.append(float(retry_selected_action == first_failed_action))
                near_repeat_hits.append(float(_is_near_repeat_action(first_failed_action, retry_selected_action, first_example)))
                failed_deltas.append(float(retry_scores[first_failed_action].score) - first_failed_score)
                near_actions = tuple(first_example.legal_actions[index] for index in _near_action_indices(first_example, first_failed_action))
                far_actions = tuple(first_example.legal_actions[index] for index in _far_click_action_indices(first_example, first_failed_action))
                if near_actions:
                    failed_near_deltas.append(
                        float(np.mean([retry_scores[action].score - before_score_snapshot.get(action, 0.0) for action in near_actions]))
                    )
                if far_actions:
                    failed_far_deltas.append(
                        float(np.mean([retry_scores[action].score - before_score_snapshot.get(action, 0.0) for action in far_actions]))
                    )
                failed_neighborhood_top1_hits.append(float(retry_selected_action in near_actions))
                failed_same_mapped_cell_top1_hits.append(
                    float(_is_same_mapped_cell_action(first_failed_action, retry_selected_action, first_example))
                )
                same_x_actions = tuple(first_example.legal_actions[index] for index in _same_x_action_indices(first_example, first_failed_action))
                same_y_actions = tuple(first_example.legal_actions[index] for index in _same_y_action_indices(first_example, first_failed_action))
                failed_same_x_hits.append(float(retry_selected_action in same_x_actions))
                failed_same_y_hits.append(float(retry_selected_action in same_y_actions))
                if same_x_actions:
                    failed_same_x_deltas.append(
                        float(np.mean([retry_scores[action].score - before_score_snapshot.get(action, 0.0) for action in same_x_actions]))
                    )
                if same_y_actions:
                    failed_same_y_deltas.append(
                        float(np.mean([retry_scores[action].score - before_score_snapshot.get(action, 0.0) for action in same_y_actions]))
                    )
        success_steps.append(step_success)
        same_streaks.append(float(_max_same_action_streak(selected_actions_for_level)))
        unique_counts.append(float(len(set(selected_actions_for_level))))
        diversity = _selected_action_diversity(
            model=agent.model,
            example=first_example,
            selected_actions=selected_actions_for_level,
            device=device,
        )
        unique_x_counts.append(float(diversity["selected_unique_x_count"]))
        unique_mapped_col_counts.append(float(diversity["selected_unique_mapped_col_count"]))
        unique_family_counts.append(float(diversity["selected_unique_family_count"]))
        family_entropies.append(float(diversity["selected_family_entropy"]))
        online_counts.append(float(agent.online_update_count))
        if len(session.levels) > 1:
            query = session.levels[1].example
            query_observation = _grid_observation_from_state(query.state, query.legal_actions, step_index=0)
            selected_action = agent.act(query_observation)
            next_hits.append(float(_is_positive_index(query, int(query.legal_actions.index(selected_action)))))
            reset_agent = LearnedOnlineObjectEventAgent(
                seed=session_index,
                config=config,
                device=device,
                temperature=0.01,
                epsilon_floor=0.0,
            )
            reset_agent.model.load_state_dict(model_state)
            reset_action = reset_agent.act(query_observation)
            reset_hits.append(float(_is_positive_index(query, int(query.legal_actions.index(reset_action)))))
    values = np.asarray(success_steps, dtype=np.float32)
    metrics = {
        "runtime_agent_act_path_rank_logits_used": float(np.mean(rank_used)) if rank_used else 0.0,
        "runtime_agent_act_path_learned_diagnostic_utility_used": float(np.mean(learned_diag_used)) if learned_diag_used else 0.0,
        "runtime_agent_act_path_num_legal_actions_mean": float(np.mean(legal_counts)) if legal_counts else 0.0,
        "runtime_agent_act_path_active_success_within_1": float(np.mean(values <= 1.0)),
        "runtime_agent_act_path_active_success_within_2": float(np.mean(values <= 2.0)),
        "runtime_agent_act_path_active_success_within_3": float(np.mean(values <= 3.0)),
        "runtime_agent_act_path_active_success_within_5": float(np.mean(values <= 5.0)),
        "runtime_agent_act_path_next_level_first_try_acc": float(np.mean(next_hits)) if next_hits else 0.0,
        "runtime_agent_act_path_next_level_with_session_reset_acc": float(np.mean(reset_hits)) if reset_hits else 0.0,
        "runtime_agent_act_path_bridge_selected_action_match_rate": float(np.mean(bridge_matches)) if bridge_matches else 0.0,
        "runtime_agent_act_path_metadata_model_input_forbidden_count": float(forbidden_count),
        "runtime_agent_act_path_online_update_count_mean": float(np.mean(online_counts)) if online_counts else 0.0,
        "runtime_agent_act_path_coordinate_noeffect_memory_norm": float(np.mean(coordinate_norms)) if coordinate_norms else 0.0,
        "runtime_agent_act_path_coordinate_noeffect_count_mean": float(np.mean(coordinate_counts)) if coordinate_counts else 0.0,
        "runtime_agent_act_path_axis_noeffect_memory_norm": float(np.mean(axis_norms)) if axis_norms else 0.0,
        "runtime_agent_act_path_axis_noeffect_count_mean": float(np.mean(axis_counts)) if axis_counts else 0.0,
        "runtime_agent_act_path_rank_component_axis_noeffect_std": float(np.mean(axis_component_stds)) if axis_component_stds else 0.0,
        "runtime_agent_act_path_rank_component_relation_std": float(np.mean(relation_component_stds)) if relation_component_stds else 0.0,
        "runtime_agent_act_path_self_selected_no_effect_count": float(no_effect_count),
        "runtime_agent_act_path_failed_exact_repeat_top1_rate": float(np.mean(exact_repeat_hits)) if exact_repeat_hits else 0.0,
        "runtime_agent_act_path_failed_near_repeat_top1_rate": float(np.mean(near_repeat_hits)) if near_repeat_hits else 0.0,
        "runtime_agent_act_path_failed_action_score_delta_mean": float(np.mean(failed_deltas)) if failed_deltas else 0.0,
        "runtime_agent_act_path_failed_near_score_delta_mean": float(np.mean(failed_near_deltas)) if failed_near_deltas else 0.0,
        "runtime_agent_act_path_failed_far_score_delta_mean": float(np.mean(failed_far_deltas)) if failed_far_deltas else 0.0,
        "runtime_agent_act_path_failed_neighborhood_top1_rate": float(np.mean(failed_neighborhood_top1_hits)) if failed_neighborhood_top1_hits else 0.0,
        "runtime_agent_act_path_failed_same_mapped_cell_top1_rate": float(np.mean(failed_same_mapped_cell_top1_hits)) if failed_same_mapped_cell_top1_hits else 0.0,
        "runtime_agent_act_path_failed_same_x_top1_rate": float(np.mean(failed_same_x_hits)) if failed_same_x_hits else 0.0,
        "runtime_agent_act_path_failed_same_y_top1_rate": float(np.mean(failed_same_y_hits)) if failed_same_y_hits else 0.0,
        "runtime_agent_act_path_failed_same_x_score_delta_mean": float(np.mean(failed_same_x_deltas)) if failed_same_x_deltas else 0.0,
        "runtime_agent_act_path_failed_same_y_score_delta_mean": float(np.mean(failed_same_y_deltas)) if failed_same_y_deltas else 0.0,
        "runtime_agent_act_path_post_failure_candidate_switch_rate": float(1.0 - np.mean(exact_repeat_hits)) if exact_repeat_hits else 0.0,
        "runtime_agent_act_path_max_same_action_streak_mean": float(np.mean(same_streaks)) if same_streaks else 0.0,
        "runtime_agent_act_path_max_same_action_streak_max": float(np.max(same_streaks)) if same_streaks else 0.0,
        "runtime_agent_act_path_unique_action_count_mean": float(np.mean(unique_counts)) if unique_counts else 0.0,
        "runtime_agent_act_path_selected_unique_x_count": float(np.mean(unique_x_counts)) if unique_x_counts else 0.0,
        "runtime_agent_act_path_selected_unique_mapped_col_count": float(np.mean(unique_mapped_col_counts)) if unique_mapped_col_counts else 0.0,
        "runtime_agent_act_path_selected_unique_family_count": float(np.mean(unique_family_counts)) if unique_family_counts else 0.0,
        "runtime_agent_act_path_selected_family_entropy": float(np.mean(family_entropies)) if family_entropies else 0.0,
    }
    metrics.update(_mean_rank_diagnostic_metrics("runtime_agent_act_path_", diag_series))
    return metrics


def _grid_observation_from_state(
    state: StructuredState,
    actions: Sequence[str],
    *,
    step_index: int,
) -> GridObservation:
    return state_to_grid_observation(state, actions, step_index=int(step_index))


def _model_policy_top1(model: ObjectEventModel, example: ObjectEventExample, *, device: torch.device) -> int:
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=device)
    with torch.no_grad():
        logits = policy_rank_logits_from_predictions(_forward_tensors(model, tensors), tensors["action_mask"])
    return int(torch.argmax(logits[0]).detach().cpu())


def _synthetic_transition(example: ObjectEventExample, action: str, result: Any) -> Transition:
    return Transition(
        state=example.state,
        action=action,
        reward=float(result.reward),
        next_state=example.next_state if bool(result.success) else example.state,
        terminated=False,
        info={"score_delta": float(result.reward), "level_boundary": bool(result.success)},
    )


def _top1(logits: torch.Tensor, actual: torch.Tensor) -> float:
    return float(torch.mean((torch.argmax(logits, dim=-1) == actual).float()).detach().cpu())


def _top1_positive(logits: torch.Tensor, tensors: dict[str, Any]) -> float:
    selected = torch.argmax(logits, dim=-1)
    return float(torch.mean(_selected_is_positive(tensors, selected).float()).detach().cpu())


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
