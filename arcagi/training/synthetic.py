from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
import logging
from pathlib import Path
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from arcagi.agents.graph_agent import GraphExplorerAgent
from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.progress_signals import (
    PolicySupervision,
    action_family as _shared_action_family,
    transition_policy_supervision,
    transition_usefulness_target,
)
from arcagi.core.types import StructuredState
from arcagi.core.utils import seed_everything
from arcagi.envs.synthetic import DEFAULT_SYNTHETIC_FAMILY_MODES, HiddenRuleEnv, family_variants_for_mode
from arcagi.models.encoder import StructuredStateEncoder
from arcagi.models.language import GroundedLanguageModel
from arcagi.models.world_model import RecurrentWorldModel
from arcagi.perception.object_encoder import extract_structured_state
from arcagi.planning.planner import HybridPlanner
from arcagi.training.synthetic_oracle import oracle_action

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class SyntheticTrainingConfig:
    family_modes: tuple[str, ...] = DEFAULT_SYNTHETIC_FAMILY_MODES
    episodes_per_epoch: int = 96
    epochs: int = 8
    max_steps: int = 48
    learning_rate: float = 3e-4
    seed: int = 7
    checkpoint_path: str = "artifacts/synthetic_hybrid.pt"
    behavior_policy: str = "explore"
    size_options: tuple[int, ...] = (7, 8, 9)
    init_checkpoint_path: str = ""
    curriculum: str = "staged"
    log_every_episodes: int = 16
    holdout_eval_every_epochs: int = 4
    holdout_episodes_per_variant: int = 2
    promotion_consecutive_evals: int = 2
    frontier_replay_weight: int = 3
    previous_stage_replay_weight: int = 1
    oracle_imitation_epochs: int = 2
    oracle_bootstrap_steps: int = 16
    oracle_bootstrap_stride: int = 1
    oracle_bootstrap_min_steps: int = 8
    oracle_bootstrap_min_stride: int = 2
    oracle_bootstrap_full_epochs: int = 4
    oracle_bootstrap_decay_epochs: int = 8
    oracle_bootstrap_decay_success_threshold: float = 0.9
    oracle_bootstrap_decay_stability_epochs: int = 2
    teacher_guidance_holdout_success_threshold: float = 0.2
    teacher_episode_fraction_initial: float = 0.2
    teacher_episode_fraction_floor: float = 0.2
    teacher_takeover_prob_initial: float = 0.5
    teacher_takeover_prob_floor: float = 0.2
    teacher_relabel_weight: float = 0.8
    dream_batches_per_epoch: int = 8
    dream_batch_size: int = 8
    dream_horizon: int = 3
    dream_loss_weight: float = 0.35
    dream_belief_weight: float = 0.2
    dream_question_weight: float = 0.2
    holdout_failure_examples: int = 5
    holdout_trace_steps: int = 48
    holdout_trace_top_actions: int = 3
    enhancement_gate_target: float = 0.3
    enhancement_gate_weight: float = 0.05


@dataclass(frozen=True)
class CurriculumStage:
    name: str
    frontier_families: tuple[str, ...]
    train_size_options: tuple[int, ...]
    holdout_size_options: tuple[int, ...]
    first_episode_threshold: float
    later_episode_threshold: float
    avg_return_threshold: float
    avg_interactions_cap: float
    regression_first_success_floor: float
    regression_later_success_floor: float


def build_default_modules(device: torch.device | None = None) -> tuple[
    StructuredStateEncoder,
    RecurrentWorldModel,
    GroundedLanguageModel,
    HybridPlanner,
]:
    device = device or torch.device("cpu")
    encoder = StructuredStateEncoder().to(device)
    world_model = RecurrentWorldModel().to(device)
    language_model = GroundedLanguageModel().to(device)
    planner = HybridPlanner()
    return encoder, world_model, language_model, planner


def _load_compatible_state_dict(
    module: torch.nn.Module,
    state_dict: dict[str, torch.Tensor],
    *,
    module_name: str,
) -> None:
    current_state = module.state_dict()
    filtered_state: dict[str, torch.Tensor] = {}
    unexpected_keys: list[str] = []
    mismatched_keys: list[str] = []
    for key, value in state_dict.items():
        if key not in current_state:
            unexpected_keys.append(key)
            continue
        if current_state[key].shape != value.shape:
            mismatched_keys.append(f"{key}: checkpoint={tuple(value.shape)} model={tuple(current_state[key].shape)}")
            continue
        filtered_state[key] = value
    load_result = module.load_state_dict(filtered_state, strict=False)
    missing_keys = sorted(set(load_result.missing_keys))
    if missing_keys or unexpected_keys or mismatched_keys:
        logger.warning(
            "Compatibility load for %s skipped or missed parameters: missing=%s unexpected=%s mismatched=%s",
            module_name,
            missing_keys,
            unexpected_keys,
            mismatched_keys,
        )


def _make_sample(
    state: StructuredState,
    next_state: StructuredState,
    action: str,
    reward: float,
    info: dict[str, object],
    *,
    teacher_action: str = "",
    teacher_weight: float = 0.0,
) -> dict[str, object]:
    delta = next_state.transition_vector() - state.transition_vector()
    delta_norm = float(np.linalg.norm(delta))
    usefulness = transition_usefulness_target(
        action,
        reward,
        None,
        delta_norm,
    )
    policy_supervision = transition_policy_supervision(
        action,
        reward,
        None,
        delta_norm,
    )
    return {
        "state": state,
        "next_state": next_state,
        "action": action,
        "teacher_action": teacher_action,
        "teacher_weight": float(teacher_weight),
        "available_actions": state.affordances,
        "reward": reward,
        "event": str(info.get("event", "")),
        "delta": delta.astype(np.float32),
        "delta_norm": delta_norm,
        "usefulness": usefulness,
        "policy_target": policy_supervision.target,
        "policy_weight": policy_supervision.weight,
        "sibling_move_target": policy_supervision.sibling_move_target,
        "sibling_move_weight": policy_supervision.sibling_move_weight,
        "same_type_target": policy_supervision.same_type_target,
        "same_type_weight": policy_supervision.same_type_weight,
        "belief_tokens": _grounded_belief_tokens(state),
        "question_tokens": _grounded_question_tokens(state),
    }


def _dream_sequence_windows(
    dataset: list[dict[str, object]],
    horizon: int,
) -> list[list[int]]:
    if horizon <= 1:
        return []
    by_episode: dict[str, list[int]] = {}
    for index, sample in enumerate(dataset):
        state = sample["state"]
        by_episode.setdefault(state.episode_id, []).append(index)
    windows: list[list[int]] = []
    for indices in by_episode.values():
        ordered = sorted(indices, key=lambda idx: int(dataset[idx]["state"].step_index))
        if len(ordered) < horizon:
            continue
        for start in range(0, len(ordered) - horizon + 1):
            window = ordered[start : start + horizon]
            contiguous = True
            for offset in range(len(window) - 1):
                left = dataset[window[offset]]
                right = dataset[window[offset + 1]]
                if left["next_state"].step_index != right["state"].step_index:
                    contiguous = False
                    break
                if left["next_state"].episode_id != right["state"].episode_id:
                    contiguous = False
                    break
            if contiguous:
                windows.append(window)
    return windows


def _dream_window_weight(dataset: list[dict[str, object]], window: list[int]) -> float:
    weight = 1.0
    for index in window:
        sample = dataset[index]
        weight += abs(float(sample["reward"]))
        weight += float(sample["usefulness"])
        weight += 0.5 * float(sample["delta_norm"])
        weight += float(sample.get("teacher_weight", 0.0))
    return float(max(weight, 1e-3))


def _sample_dream_sequences(
    dataset: list[dict[str, object]],
    *,
    horizon: int,
    total_sequences: int,
    seed: int,
) -> list[list[int]]:
    windows = _dream_sequence_windows(dataset, horizon)
    if not windows or total_sequences <= 0:
        return []
    weights = np.asarray([_dream_window_weight(dataset, window) for window in windows], dtype=np.float64)
    if float(weights.sum()) <= 0.0:
        probabilities = None
    else:
        probabilities = weights / weights.sum()
    rng = np.random.default_rng(seed)
    sample_count = min(len(windows), total_sequences)
    chosen = rng.choice(len(windows), size=sample_count, replace=len(windows) < sample_count, p=probabilities)
    return [windows[int(index)] for index in chosen]


def _checkpoint_variant_path(checkpoint_path: Path, variant: str) -> Path:
    return checkpoint_path.with_name(f"{checkpoint_path.stem}.{variant}{checkpoint_path.suffix}")


def _epoch_seed_base(config: SyntheticTrainingConfig, epoch_index: int) -> int:
    seed_sequence = np.random.SeedSequence((int(config.seed), int(epoch_index), int(config.episodes_per_epoch)))
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0])


def _teacher_guidance_state(
    config: SyntheticTrainingConfig,
    history: list[dict[str, object]] | None = None,
) -> dict[str, float]:
    records = history or []
    threshold = float(config.oracle_bootstrap_decay_success_threshold)
    holdout_threshold = float(config.teacher_guidance_holdout_success_threshold)
    stability_epochs = max(1, int(config.oracle_bootstrap_decay_stability_epochs))
    ready_streak = 0
    for record in reversed(records):
        success_rate = float(record.get("collect_success_rate", 0.0))
        if success_rate < threshold:
            break
        holdout_evaluated = bool(record.get("holdout_evaluated", False))
        if holdout_evaluated:
            frontier_success = float(record.get("holdout_frontier_success", 0.0))
            if frontier_success < holdout_threshold:
                break
        ready_streak += 1
    full_epochs = max(0, int(config.oracle_bootstrap_full_epochs))
    decay_epochs = max(0, int(config.oracle_bootstrap_decay_epochs))
    stable_progress = max(0, ready_streak - stability_epochs)
    if stable_progress < full_epochs:
        alpha = 0.0
    elif decay_epochs == 0:
        alpha = 1.0
    else:
        alpha = min(1.0, float(stable_progress - full_epochs + 1) / float(decay_epochs))
    return {
        "ready_streak": float(ready_streak),
        "alpha": float(alpha),
    }


def _bootstrap_schedule(
    config: SyntheticTrainingConfig,
    epoch_index: int,
    history: list[dict[str, object]] | None = None,
) -> tuple[int, int]:
    max_steps = max(0, int(config.oracle_bootstrap_steps))
    min_steps = max(0, min(int(config.oracle_bootstrap_min_steps), max_steps))
    max_stride = max(1, int(config.oracle_bootstrap_stride))
    min_stride = max(1, int(config.oracle_bootstrap_min_stride))
    guidance_state = _teacher_guidance_state(config, history)
    alpha = float(guidance_state["alpha"])
    bootstrap_steps = int(round(max_steps + (min_steps - max_steps) * alpha))
    bootstrap_stride = int(round(max_stride + (min_stride - max_stride) * alpha))
    return max(0, bootstrap_steps), max(1, bootstrap_stride)


def _teacher_episode_fraction(
    config: SyntheticTrainingConfig,
    history: list[dict[str, object]] | None = None,
) -> float:
    initial = float(max(0.0, min(1.0, config.teacher_episode_fraction_initial)))
    floor = float(max(0.0, min(1.0, config.teacher_episode_fraction_floor)))
    high = max(initial, floor)
    low = min(initial, floor)
    guidance_state = _teacher_guidance_state(config, history)
    alpha = float(guidance_state["alpha"])
    return high + ((low - high) * alpha)


def _teacher_takeover_probability(
    config: SyntheticTrainingConfig,
    history: list[dict[str, object]] | None = None,
) -> float:
    initial = float(max(0.0, min(1.0, config.teacher_takeover_prob_initial)))
    floor = float(max(0.0, min(1.0, config.teacher_takeover_prob_floor)))
    high = max(initial, floor)
    low = min(initial, floor)
    guidance_state = _teacher_guidance_state(config, history)
    alpha = float(guidance_state["alpha"])
    return high + ((low - high) * alpha)


def _teacher_policy_supervision(action: str, *, weight: float) -> PolicySupervision:
    if not action or weight <= 0.0:
        return PolicySupervision(target=0.0, weight=0.0)
    family = _shared_action_family(action)
    if family == "move":
        return PolicySupervision(
            target=1.0,
            weight=weight,
            sibling_move_target=0.35,
            sibling_move_weight=0.45 * weight,
        )
    if family in {"click", "select"}:
        return PolicySupervision(
            target=1.0,
            weight=weight,
            same_type_target=0.25,
            same_type_weight=0.35 * weight,
        )
    if family == "interact":
        return PolicySupervision(
            target=1.0,
            weight=weight,
            same_type_target=0.12,
            same_type_weight=0.2 * weight,
        )
    return PolicySupervision(target=0.8, weight=0.6 * weight)


def _curriculum_stages(config: SyntheticTrainingConfig) -> tuple[CurriculumStage, ...]:
    size_options = tuple(sorted(set(int(size) for size in config.size_options)))
    smaller_sizes = tuple(size for size in size_options if size <= 8) or size_options
    larger_sizes = tuple(size for size in size_options if size >= max(smaller_sizes)) or size_options
    heldout_stage0_sizes = tuple(size for size in size_options if size > max(smaller_sizes)) or larger_sizes
    heldout_frontier_sizes = tuple(sorted(set(larger_sizes[-2:]))) or larger_sizes
    return (
        CurriculumStage(
            name="foundation",
            frontier_families=("switch_unlock", "order_collect"),
            train_size_options=smaller_sizes,
            holdout_size_options=heldout_stage0_sizes,
            first_episode_threshold=0.80,
            later_episode_threshold=0.92,
            avg_return_threshold=0.70,
            avg_interactions_cap=10.0,
            regression_first_success_floor=0.0,
            regression_later_success_floor=0.0,
        ),
        CurriculumStage(
            name="hidden_modes",
            frontier_families=("selector_unlock", "delayed_order_unlock"),
            train_size_options=size_options,
            holdout_size_options=heldout_frontier_sizes,
            first_episode_threshold=0.55,
            later_episode_threshold=0.80,
            avg_return_threshold=0.35,
            avg_interactions_cap=18.0,
            regression_first_success_floor=0.65,
            regression_later_success_floor=0.85,
        ),
        CurriculumStage(
            name="compositional",
            frontier_families=("selector_sequence_unlock",),
            train_size_options=size_options,
            holdout_size_options=heldout_frontier_sizes,
            first_episode_threshold=0.35,
            later_episode_threshold=0.65,
            avg_return_threshold=0.15,
            avg_interactions_cap=22.0,
            regression_first_success_floor=0.50,
            regression_later_success_floor=0.75,
        ),
    )


def _fixed_stage_index_for_epoch(config: SyntheticTrainingConfig, epoch_index: int) -> int:
    stages = _curriculum_stages(config)
    if config.epochs <= 1:
        return min(len(stages) - 1, 0)
    first_stage = max(1, config.epochs // 3)
    second_stage = max(first_stage + 1, (2 * config.epochs) // 3)
    if epoch_index < first_stage:
        return 0
    if epoch_index < second_stage:
        return min(1, len(stages) - 1)
    return len(stages) - 1


def _weighted_family_schedule(
    family_weights: dict[str, int],
    episode_count: int,
    *,
    seed: int,
) -> list[str]:
    families = tuple(family_weights.keys())
    if not families:
        return []
    weights = np.asarray([max(1, family_weights[family]) for family in families], dtype=np.float64)
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(seed)
    sampled_indices = rng.choice(len(families), size=episode_count, replace=True, p=probabilities)
    return [families[int(index)] for index in sampled_indices]


def _stable_token_signature(token: str) -> int:
    return sum((index + 1) * ord(character) for index, character in enumerate(token))


def _family_variant_offset(family_mode: str, *, seed: int) -> int:
    variants = family_variants_for_mode(family_mode)
    if len(variants) <= 1:
        return 0
    seed_sequence = np.random.SeedSequence((int(seed), _stable_token_signature(family_mode), len(variants)))
    return int(seed_sequence.generate_state(1, dtype=np.uint32)[0] % len(variants))


def _stage_family_weights(
    config: SyntheticTrainingConfig,
    stages: tuple[CurriculumStage, ...],
    stage_index: int,
) -> dict[str, int]:
    weights: dict[str, int] = {}
    for previous_stage in stages[:stage_index]:
        for family_mode in previous_stage.frontier_families:
            weights[family_mode] = max(weights.get(family_mode, 0), config.previous_stage_replay_weight)
    for family_mode in stages[stage_index].frontier_families:
        weights[family_mode] = max(weights.get(family_mode, 0), config.frontier_replay_weight)
    return weights


def _stage_regression_families(stages: tuple[CurriculumStage, ...], stage_index: int) -> tuple[str, ...]:
    return tuple(dict.fromkeys(family for stage in stages[:stage_index] for family in stage.frontier_families))


def _count_by_value(values: list[str]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        counts[value] = counts.get(value, 0) + 1
    return counts


def _action_family(action: str) -> str:
    return _shared_action_family(action)


def _synthetic_color_name(color: int) -> str | None:
    return {
        3: "red",
        4: "blue",
        5: "green",
        6: "yellow",
        7: "red",
        8: "blue",
    }.get(int(color))


def _nearest_interactable_tokens(state: StructuredState) -> tuple[str, ...]:
    agent = next((obj for obj in state.objects if "agent" in obj.tags), None)
    interactables = [obj for obj in state.objects if "interactable" in obj.tags]
    if agent is None or not interactables:
        return ()
    target = min(
        interactables,
        key=lambda obj: abs(obj.centroid[0] - agent.centroid[0]) + abs(obj.centroid[1] - agent.centroid[1]),
    )
    color_token = _synthetic_color_name(target.color)
    return tuple(token for token in (color_token,) if token is not None)


def _state_has_active_target(state: StructuredState) -> bool:
    return any("target" in obj.tags and "active" in obj.tags for obj in state.objects)


def _has_selector_affordance(state: StructuredState) -> bool:
    context = build_action_schema_context(state.affordances, dict(state.action_roles))
    return any(
        build_action_schema(candidate, context).action_type in {"click", "select"}
        for candidate in state.affordances
    )


def _state_has_target(state: StructuredState) -> bool:
    return any("target" in obj.tags for obj in state.objects)


def _state_symbolic_int(state: StructuredState, key: str, default: int = 0) -> int:
    raw = state.inventory_dict().get(key, str(default))
    try:
        return int(raw)
    except (TypeError, ValueError):
        return default


def _grounded_belief_tokens(
    state: StructuredState,
) -> tuple[str, ...]:
    nearest = _nearest_interactable_tokens(state)
    has_selector = _has_selector_affordance(state)
    target_visible = _state_has_target(state)
    progress_level = _state_symbolic_int(state, "belief_progress_level")
    contradiction_count = _state_symbolic_int(state, "belief_contradiction_count")
    flags = state.flags_dict()
    mode = flags.get("belief_mode", "explore")
    recent_progress = flags.get("belief_recent_progress", "0") == "1"
    last_effect_family = flags.get("belief_last_effect_family", "none")

    if _state_has_active_target(state):
        return ("goal", "active")
    if recent_progress and last_effect_family in {"interact", "click"}:
        return ("confirm", "interact") + nearest[:1]
    if progress_level >= 2 and mode == "commit":
        return ("confirm", "goal")
    if has_selector and contradiction_count > 0:
        return ("uncertain", "rule") + nearest[:1]
    if target_visible:
        return ("goal", "inactive")
    if has_selector and nearest:
        return ("uncertain", "rule") + nearest[:1]
    if nearest:
        return ("goal", "unknown") + nearest[:1]
    return ("goal", "unknown")


def _grounded_question_tokens(
    state: StructuredState,
) -> tuple[str, ...]:
    has_selector = _has_selector_affordance(state)
    has_interactable = any("interactable" in obj.tags for obj in state.objects)
    progress_level = _state_symbolic_int(state, "belief_progress_level")
    contradiction_count = _state_symbolic_int(state, "belief_contradiction_count")
    flags = state.flags_dict()
    mode = flags.get("belief_mode", "explore")
    recent_progress = flags.get("belief_recent_progress", "0") == "1"
    last_effect_family = flags.get("belief_last_effect_family", "none")

    if _state_has_active_target(state) or (progress_level >= 2 and mode == "commit"):
        return ("move", "toward", "target")
    if has_selector and (mode in {"explore", "probe"} or contradiction_count > 0):
        return ("need", "test", "rule")
    if recent_progress and last_effect_family in {"interact", "click"}:
        return ("need", "test", "interact")
    if has_interactable:
        return ("need", "test", "interact")
    return ("need", "explore")


def _summarize_episode_records(records: list[dict[str, object]]) -> dict[str, object]:
    if not records:
        return {
            "episodes": 0,
            "success_rate": 0.0,
            "avg_return": 0.0,
            "avg_steps": 0.0,
            "avg_interactions": 0.0,
        }
    return {
        "episodes": len(records),
        "success_rate": mean(float(record["success"]) for record in records),
        "avg_return": mean(float(record["return"]) for record in records),
        "avg_steps": mean(float(record["steps"]) for record in records),
        "avg_interactions": mean(float(record["interaction_steps"]) for record in records),
    }


def _sorted_metric_breakdown(records_by_key: dict[str, list[dict[str, object]]]) -> dict[str, dict[str, object]]:
    return {
        key: _summarize_episode_records(records)
        for key, records in sorted(records_by_key.items(), key=lambda item: item[0])
    }


def _holdout_failure_reasons(
    stage: CurriculumStage,
    frontier_metrics: dict[str, object],
    regression_metrics: dict[str, object] | None,
) -> list[str]:
    reasons: list[str] = []
    if float(frontier_metrics["first_episode_success"]) < stage.first_episode_threshold:
        reasons.append(
            f"frontier_first_episode_success<{stage.first_episode_threshold:.2f}:"
            f"{float(frontier_metrics['first_episode_success']):.3f}"
        )
    if float(frontier_metrics["later_episode_success"]) < stage.later_episode_threshold:
        reasons.append(
            f"frontier_later_episode_success<{stage.later_episode_threshold:.2f}:"
            f"{float(frontier_metrics['later_episode_success']):.3f}"
        )
    if float(frontier_metrics["avg_return"]) < stage.avg_return_threshold:
        reasons.append(
            f"frontier_avg_return<{stage.avg_return_threshold:.2f}:"
            f"{float(frontier_metrics['avg_return']):.3f}"
        )
    if float(frontier_metrics["avg_interactions"]) > stage.avg_interactions_cap:
        reasons.append(
            f"frontier_avg_interactions>{stage.avg_interactions_cap:.2f}:"
            f"{float(frontier_metrics['avg_interactions']):.3f}"
        )
    if regression_metrics is not None:
        if float(regression_metrics["first_episode_success"]) < stage.regression_first_success_floor:
            reasons.append(
                f"regression_first_episode_success<{stage.regression_first_success_floor:.2f}:"
                f"{float(regression_metrics['first_episode_success']):.3f}"
            )
        if float(regression_metrics["later_episode_success"]) < stage.regression_later_success_floor:
            reasons.append(
                f"regression_later_episode_success<{stage.regression_later_success_floor:.2f}:"
                f"{float(regression_metrics['later_episode_success']):.3f}"
            )
    return reasons


def _top_action_thoughts(thought: object | None, *, limit: int) -> list[dict[str, object]]:
    actions = tuple(getattr(thought, "actions", ())) if thought is not None else ()
    if not actions or limit <= 0:
        return []

    def _rank(entry: object) -> float:
        return (
            float(getattr(entry, "value", 0.0))
            + (0.35 * float(getattr(entry, "policy", 0.0)))
            + (0.25 * float(getattr(entry, "selector_followup", 0.0)))
            - (0.15 * float(getattr(entry, "uncertainty", 0.0)))
        )

    ranked = sorted(actions, key=_rank, reverse=True)[:limit]
    return [
        {
            "action": str(getattr(entry, "action", "")),
            "value": float(getattr(entry, "value", 0.0)),
            "uncertainty": float(getattr(entry, "uncertainty", 0.0)),
            "policy": float(getattr(entry, "policy", 0.0)),
            "selector_followup": float(getattr(entry, "selector_followup", 0.0)),
            "predicted_reward": float(getattr(entry, "predicted_reward", 0.0)),
            "usefulness": float(getattr(entry, "usefulness", 0.0)),
        }
        for entry in ranked
    ]


def _claim_summaries(claims: tuple[object, ...] | list[object], *, limit: int = 3) -> list[dict[str, object]]:
    summaries: list[dict[str, object]] = []
    for claim in tuple(claims)[:limit]:
        summaries.append(
            {
                "claim_type": str(getattr(claim, "claim_type", "")),
                "subject": str(getattr(claim, "subject", "")),
                "relation": str(getattr(claim, "relation", "")),
                "object": str(getattr(claim, "object", "")),
                "confidence": float(getattr(claim, "confidence", 0.0)),
                "evidence": float(getattr(claim, "evidence", 0.0)),
                "salience": float(getattr(claim, "salience", 0.0)),
            }
        )
    return summaries


def _selected_color_name(env: HiddenRuleEnv) -> str | None:
    selected_color = getattr(env, "_selected_color", None)
    if selected_color is None:
        return None
    return str(env._color_name(selected_color))


def _trace_plan_scores(scores: dict[str, float] | None) -> dict[str, float]:
    if not scores:
        return {}
    keys = (
        "total",
        "value",
        "search",
        "entropy",
        "disagreement",
        "empirical_reward",
        "memory",
        "induced",
        "policy_prior",
        "online_bias",
        "global_novelty",
        "family_novelty",
        "parameter_bonus",
        "stuck_bonus",
        "repeat_penalty",
        "action_no_effect_rate",
        "family_no_effect_rate",
        "cycle_penalty",
        "search_budget_remaining",
    )
    return {key: float(scores[key]) for key in keys if key in scores}


def _step_trace_entry(
    *,
    agent: object,
    env: HiddenRuleEnv,
    action: str,
    reward: float,
    info: dict[str, object],
    step_index: int,
    trace_top_actions: int,
) -> dict[str, object]:
    runtime_thought = getattr(agent, "last_runtime_thought", None)
    latest_claims = tuple(getattr(agent, "latest_claims", ()))
    latest_language = tuple(getattr(agent, "latest_language", ()))
    latest_state = getattr(agent, "last_state", None)
    return {
        "step": step_index,
        "action": action,
        "action_family": _action_family(action),
        "reward": float(reward),
        "event": str(info.get("event", "")),
        "belief_tokens": tuple(getattr(runtime_thought, "belief_tokens", ())),
        "question_tokens": tuple(getattr(runtime_thought, "question_tokens", ())),
        "latest_language": latest_language,
        "claims": _claim_summaries(latest_claims),
        "top_actions": _top_action_thoughts(runtime_thought, limit=trace_top_actions),
        "plan_scores": _trace_plan_scores(getattr(agent, "last_plan_scores", {})),
        "goal_active": bool(getattr(env, "_goal_is_active")()),
        "selected_color": _selected_color_name(env),
        "progress_index": int(getattr(env, "_progress_index", 0)),
        "inventory": dict(getattr(env, "_inventory", {})),
        "flags": dict(getattr(env, "_flags", {})),
        "object_count": len(latest_state.objects) if latest_state is not None else 0,
        "affordance_count": len(latest_state.affordances) if latest_state is not None else 0,
        "total_reward": float(getattr(agent, "total_reward", 0.0)),
    }


def _episode_event_counts(step_trace: list[dict[str, object]]) -> dict[str, int]:
    return _count_by_value([str(entry["event"]) for entry in step_trace])


def _max_repeated_action_run(step_trace: list[dict[str, object]]) -> int:
    best = 0
    current = 0
    previous_action = None
    for entry in step_trace:
        action = str(entry["action"])
        if action == previous_action:
            current += 1
        else:
            current = 1
            previous_action = action
        best = max(best, current)
    return best


def _failure_signatures(
    *,
    family_mode: str,
    family_variant: str,
    episode: dict[str, object],
    step_trace: list[dict[str, object]],
) -> list[str]:
    signatures: list[str] = []
    event_counts = _episode_event_counts(step_trace)
    action_family_counts = dict(episode.get("action_family_counts", {}))
    interaction_count = int(action_family_counts.get("interact", 0))
    click_count = int(action_family_counts.get("click", 0))
    move_count = int(action_family_counts.get("move", 0))
    goal_activation_step = next(
        (int(entry["step"]) for entry in step_trace if bool(entry.get("goal_active"))),
        None,
    )
    max_repeat_run = _max_repeated_action_run(step_trace)
    if str(episode.get("termination_reason", "")) == "timeout":
        signatures.append("timeout")
    if goal_activation_step is not None and not bool(episode.get("success", False)):
        signatures.append("goal_activated_but_not_completed")
    if interaction_count == 0 and click_count == 0 and goal_activation_step is None:
        signatures.append("never_tested_mechanic")
    if click_count > 0 and interaction_count == 0 and goal_activation_step is None:
        signatures.append("selector_probe_without_followthrough")
    if (interaction_count + click_count) >= max(4, int(0.35 * float(episode.get("steps", 0.0)))) and goal_activation_step is None:
        signatures.append("interaction_churn_without_progress")
    if event_counts.get("inactive_goal_blocked", 0) >= 3 and interaction_count == 0:
        signatures.append("goal_seen_but_prerequisite_not_tested")
    if event_counts.get("blocked_by_object", 0) >= 3 and interaction_count == 0:
        signatures.append("navigation_loop_around_interactable")
    if max_repeat_run >= 4:
        signatures.append("repeated_action_loop")
    if (
        event_counts.get("wrong_switch", 0)
        + event_counts.get("wrong_selector_or_switch", 0)
        + event_counts.get("wrong_order", 0)
        + event_counts.get("wrong_order_reset", 0)
    ) > 0 and goal_activation_step is None:
        signatures.append("negative_interaction_evidence_not_internalized")
    if family_mode == "switch_unlock" and event_counts.get("wrong_switch", 0) > 0 and event_counts.get("correct_switch", 0) == 0:
        signatures.append("wrong_switch_not_penalized")
    if family_mode == "order_collect" and event_counts.get("wrong_order", 0) > 0:
        signatures.append("sequence_rule_not_internalized")
    if family_mode == "delayed_order_unlock" and event_counts.get("wrong_order_reset", 0) > 0:
        signatures.append("delayed_sequence_not_internalized")
    if family_mode in {"selector_unlock", "selector_sequence_unlock"}:
        final_selected_color = next(
            (str(entry["selected_color"]) for entry in reversed(step_trace) if entry.get("selected_color") is not None),
            None,
        )
        target_color = family_variant.split("__", maxsplit=1)[0]
        if click_count > 0 and final_selected_color != target_color and goal_activation_step is None:
            signatures.append("selector_state_wrong_or_unstable")
        if click_count == 0:
            signatures.append("selector_mechanic_never_tested")
    if move_count == 0:
        signatures.append("no_navigation_attempt")
    return sorted(dict.fromkeys(signatures))


def _build_checkpoint_snapshot(
    config: SyntheticTrainingConfig,
    encoder: StructuredStateEncoder,
    world_model: RecurrentWorldModel,
    language_model: GroundedLanguageModel,
    history: list[dict[str, float | int | bool]],
    holdout_history: list[dict[str, object]],
    training_state: dict[str, float | int | bool],
    last_holdout_result: dict[str, object] | None = None,
) -> dict[str, object]:
    return {
        "config": asdict(config),
        "encoder": encoder.state_dict(),
        "world_model": world_model.state_dict(),
        "language_model": language_model.state_dict(),
        "history": history,
        "holdout_history": holdout_history,
        "training_state": training_state,
        "last_holdout_result": last_holdout_result or {},
    }


def _save_checkpoint(
    checkpoint_path: Path,
    snapshot: dict[str, object],
    *,
    epoch: int | None = None,
    interrupted: bool = False,
) -> dict[str, str]:
    saved_paths = {"latest_checkpoint_path": str(checkpoint_path)}
    torch.save(snapshot, checkpoint_path)
    if epoch is not None:
        epoch_checkpoint_path = _checkpoint_variant_path(checkpoint_path, f"epoch_{epoch:04d}")
        torch.save(snapshot, epoch_checkpoint_path)
        saved_paths["epoch_checkpoint_path"] = str(epoch_checkpoint_path)
    if interrupted:
        interrupt_checkpoint_path = _checkpoint_variant_path(checkpoint_path, "interrupt")
        torch.save(snapshot, interrupt_checkpoint_path)
        saved_paths["interrupt_checkpoint_path"] = str(interrupt_checkpoint_path)
    return saved_paths


def collect_dataset(
    config: SyntheticTrainingConfig,
    *,
    epoch_index: int = 0,
    stage_index: int | None = None,
    history: list[dict[str, object]] | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    explorer = GraphExplorerAgent()
    dataset: list[dict[str, object]] = []
    epoch_seed_base = _epoch_seed_base(config, epoch_index)
    seed_cursor = epoch_seed_base
    stages = _curriculum_stages(config)
    if config.curriculum in {"staged", "gated"}:
        resolved_stage_index = stage_index if stage_index is not None else _fixed_stage_index_for_epoch(config, epoch_index)
        resolved_stage = stages[min(max(resolved_stage_index, 0), len(stages) - 1)]
        family_weights = _stage_family_weights(config, stages, resolved_stage_index)
        family_mode_schedule = _weighted_family_schedule(
            family_weights,
            config.episodes_per_epoch,
            seed=epoch_seed_base,
        )
        size_options = resolved_stage.train_size_options
        stage_name = resolved_stage.name
        frontier_families = resolved_stage.frontier_families
    else:
        family_modes = _family_modes_for_epoch(config, epoch_index)
        family_mode_schedule = [family_modes[episode_index % len(family_modes)] for episode_index in range(config.episodes_per_epoch)]
        size_options = _size_options_for_epoch(config, epoch_index)
        stage_name = "fixed_staged" if config.curriculum == "fixed_staged" else "flat"
        frontier_families = tuple(dict.fromkeys(family_mode_schedule))
    episode_returns: list[float] = []
    episode_steps: list[float] = []
    episode_successes: list[float] = []
    episode_family_modes: list[str] = []
    records_by_family: dict[str, list[dict[str, object]]] = {}
    records_by_variant: dict[str, list[dict[str, object]]] = {}
    records_by_size: dict[str, list[dict[str, object]]] = {}
    family_variant_counts: dict[str, int] = {}
    family_variant_offsets = {
        family_mode: _family_variant_offset(family_mode, seed=epoch_seed_base)
        for family_mode in set(family_mode_schedule)
    }
    size_offset = 0 if not size_options else int(epoch_seed_base % len(size_options))
    guidance_state = _teacher_guidance_state(config, history)
    bootstrap_steps, bootstrap_stride = _bootstrap_schedule(config, epoch_index, history)
    scheduled_teacher_episode_fraction = _teacher_episode_fraction(config, history)
    teacher_takeover_prob = _teacher_takeover_probability(config, history)
    teacher_episode_fraction = (
        1.0
        if config.behavior_policy in {"mixed", "bootstrap"} and epoch_index < max(0, config.oracle_imitation_epochs)
        else scheduled_teacher_episode_fraction
    )
    guidance_rng = np.random.default_rng(epoch_seed_base + 17_941)
    teacher_episode_count = 0
    teacher_controlled_steps = 0
    teacher_labeled_steps = 0
    total_steps = 0
    collect_started = perf_counter()
    for epoch_episode in range(config.episodes_per_epoch):
        family_mode = family_mode_schedule[epoch_episode]
        variants = family_variants_for_mode(family_mode)
        family_occurrence = family_variant_counts.get(family_mode, 0)
        variant = variants[(family_variant_offsets[family_mode] + family_occurrence) % len(variants)]
        family_variant_counts[family_mode] = family_occurrence + 1
        size = size_options[(size_offset + epoch_episode) % len(size_options)]
        env = HiddenRuleEnv(
            size=size,
            family_mode=family_mode,
            family_variant=variant,
            max_steps=config.max_steps,
            seed=seed_cursor,
        )
        seed_cursor += 1
        observation = env.reset(seed=seed_cursor)
        explorer.reset_episode()
        done = False
        episode_return = 0.0
        episode_step_count = 0
        episode_success = False
        episode_interaction_count = 0
        dense_teacher_episode = (
            config.behavior_policy in {"mixed", "bootstrap"}
            and epoch_index < max(0, config.oracle_imitation_epochs)
        )
        sampled_teacher_episode = (
            config.behavior_policy in {"mixed", "bootstrap"}
            and not dense_teacher_episode
            and bool(guidance_rng.random() < teacher_episode_fraction)
        )
        teacher_episode = dense_teacher_episode or sampled_teacher_episode or config.behavior_policy == "oracle"
        if teacher_episode:
            teacher_episode_count += 1
        while not done:
            state: StructuredState
            use_bootstrap_oracle = (
                config.behavior_policy in {"mixed", "bootstrap"}
                and not teacher_episode
                and env._step < bootstrap_steps
                and ((seed_cursor + env._step) % bootstrap_stride == 0)
                and bool(guidance_rng.random() < teacher_takeover_prob)
            )
            teacher_action = ""
            prefetched_state: StructuredState | None = None
            if config.behavior_policy in {"mixed", "bootstrap", "oracle"}:
                prefetched_state = explorer.observe(observation)
                teacher_action = oracle_action(env)
                explorer.last_state = prefetched_state
            if teacher_episode or use_bootstrap_oracle:
                state = prefetched_state if prefetched_state is not None else explorer.observe(observation)
                action = teacher_action
                explorer.last_state = state
                explorer.last_action = action
                teacher_controlled_steps += 1
            else:
                action = explorer.act(observation)
                state = explorer.last_state
                assert state is not None
            teacher_label_action = ""
            teacher_label_weight = 0.0
            if (
                teacher_action
                and action != teacher_action
                and config.behavior_policy in {"mixed", "bootstrap"}
            ):
                teacher_label_action = teacher_action
                teacher_label_weight = float(config.teacher_relabel_weight)
                teacher_labeled_steps += 1
            result = env.step(action)
            next_state = explorer.update_after_step(
                next_observation=result.observation,
                reward=result.reward,
                terminated=result.terminated or result.truncated,
                info=result.info,
            )
            dataset.append(
                _make_sample(
                    state,
                    next_state,
                    action,
                    result.reward,
                    result.info,
                    teacher_action=teacher_label_action,
                    teacher_weight=teacher_label_weight,
                )
            )
            episode_return += result.reward
            episode_step_count += 1
            total_steps += 1
            if _action_family(action) in {"interact", "click"}:
                episode_interaction_count += 1
            if result.info.get("event") == "goal_reached":
                episode_success = True
            observation = result.observation
            done = result.terminated or result.truncated
        episode_returns.append(float(episode_return))
        episode_steps.append(float(episode_step_count))
        episode_successes.append(1.0 if episode_success else 0.0)
        episode_family_modes.append(family_mode)
        episode_record = {
            "success": episode_success,
            "return": float(episode_return),
            "steps": float(episode_step_count),
            "interaction_steps": float(episode_interaction_count),
        }
        records_by_family.setdefault(family_mode, []).append(episode_record)
        records_by_variant.setdefault(f"{family_mode}/{variant}", []).append(episode_record)
        records_by_size.setdefault(str(size), []).append(episode_record)
        should_log_episode = (
            config.log_every_episodes > 0
            and (
                ((epoch_episode + 1) % config.log_every_episodes == 0)
                or (epoch_episode + 1 == config.episodes_per_epoch)
            )
        )
        if should_log_episode:
            interval_start = max(0, len(episode_returns) - config.log_every_episodes)
            interval_returns = episode_returns[interval_start:]
            interval_steps = episode_steps[interval_start:]
            interval_successes = episode_successes[interval_start:]
            interval_family_modes = episode_family_modes[interval_start:]
            print(
                json.dumps(
                    {
                        "type": "episode_progress",
                        "epoch": epoch_index,
                        "episode": epoch_episode + 1,
                        "episodes_per_epoch": config.episodes_per_epoch,
                        "family_mode": family_mode,
                        "family_variant": variant,
                        "size": size,
                        "episode_return": float(episode_return),
                        "episode_steps": float(episode_step_count),
                        "episode_success": episode_success,
                        "running_avg_return": mean(episode_returns),
                        "running_avg_steps": mean(episode_steps),
                        "running_success_rate": mean(episode_successes),
                        "stage_name": stage_name,
                        "frontier_families": frontier_families,
                        "running_family_counts": _count_by_value(episode_family_modes),
                        "interval_avg_return": mean(interval_returns),
                        "interval_avg_steps": mean(interval_steps),
                        "interval_success_rate": mean(interval_successes),
                        "interval_family_counts": _count_by_value(interval_family_modes),
                        "samples_collected": float(len(dataset)),
                        "elapsed_seconds": perf_counter() - collect_started,
                        "bootstrap_steps": bootstrap_steps,
                        "bootstrap_stride": bootstrap_stride,
                        "teacher_guidance_ready_streak": guidance_state["ready_streak"],
                        "teacher_guidance_alpha": guidance_state["alpha"],
                        "teacher_episode_fraction": teacher_episode_fraction,
                        "teacher_takeover_prob": teacher_takeover_prob,
                        "teacher_episode_count": teacher_episode_count,
                        "teacher_step_fraction": (float(teacher_controlled_steps) / float(max(total_steps, 1))),
                        "teacher_relabel_fraction": (float(teacher_labeled_steps) / float(max(total_steps, 1))),
                    }
                ),
                flush=True,
            )
    return dataset, {
        "episodes": float(config.episodes_per_epoch),
        "samples": float(len(dataset)),
        "avg_return": mean(episode_returns) if episode_returns else 0.0,
        "avg_steps": mean(episode_steps) if episode_steps else 0.0,
        "success_rate": mean(episode_successes) if episode_successes else 0.0,
        "stage_name": stage_name,
        "frontier_families": frontier_families,
        "family_counts": _count_by_value(episode_family_modes),
        "family_breakdown": _sorted_metric_breakdown(records_by_family),
        "variant_breakdown": _sorted_metric_breakdown(records_by_variant),
        "size_breakdown": _sorted_metric_breakdown(records_by_size),
        "epoch_seed_base": epoch_seed_base,
        "collect_seconds": perf_counter() - collect_started,
        "bootstrap_steps": float(bootstrap_steps),
        "bootstrap_stride": float(bootstrap_stride),
        "bootstrap_release_epoch": -1.0,
        "bootstrap_ready_streak": float(guidance_state["ready_streak"]),
        "teacher_guidance_ready_streak": float(guidance_state["ready_streak"]),
        "teacher_guidance_alpha": float(guidance_state["alpha"]),
        "teacher_episode_fraction": float(teacher_episode_fraction),
        "teacher_takeover_prob": float(teacher_takeover_prob),
        "teacher_episode_count": float(teacher_episode_count),
        "teacher_step_fraction": float(teacher_controlled_steps) / float(max(total_steps, 1)),
        "teacher_relabel_fraction": float(teacher_labeled_steps) / float(max(total_steps, 1)),
    }


def _run_dream_phase(
    config: SyntheticTrainingConfig,
    *,
    dataset: list[dict[str, object]],
    epoch_index: int,
    encoder: StructuredStateEncoder,
    world_model: RecurrentWorldModel,
    language_model: GroundedLanguageModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> dict[str, float]:
    if config.dream_batches_per_epoch <= 0 or config.dream_batch_size <= 0 or config.dream_horizon <= 1:
        return {
            "dream_sequences": 0.0,
            "dream_steps": 0.0,
            "dream_loss": 0.0,
            "dream_world_loss": 0.0,
            "dream_belief_loss": 0.0,
            "dream_question_loss": 0.0,
            "dream_seconds": 0.0,
        }
    total_sequences = int(config.dream_batches_per_epoch) * int(config.dream_batch_size)
    windows = _sample_dream_sequences(
        dataset,
        horizon=int(config.dream_horizon),
        total_sequences=total_sequences,
        seed=_epoch_seed_base(config, epoch_index) + 911_731,
    )
    if not windows:
        return {
            "dream_sequences": 0.0,
            "dream_steps": 0.0,
            "dream_loss": 0.0,
            "dream_world_loss": 0.0,
            "dream_belief_loss": 0.0,
            "dream_question_loss": 0.0,
            "dream_seconds": 0.0,
        }
    dream_started = perf_counter()
    losses: list[float] = []
    world_losses: list[float] = []
    belief_losses: list[float] = []
    question_losses: list[float] = []
    total_steps = 0
    encoder.train()
    world_model.train()
    language_model.train()
    for window in windows:
        anchor_sample = dataset[window[0]]
        anchor_state = anchor_sample["state"]
        latent = encoder.encode_state(anchor_state, device=device).latent
        hidden: torch.Tensor | None = None
        sequence_terms: list[torch.Tensor] = []
        sequence_world_terms: list[torch.Tensor] = []
        sequence_belief_terms: list[torch.Tensor] = []
        sequence_question_terms: list[torch.Tensor] = []
        for index in window:
            sample = dataset[index]
            state = sample["state"]
            next_state = sample["next_state"]
            action = sample["action"]
            with torch.no_grad():
                next_encoded = encoder.encode_state(next_state, device=device)
            reward_target = torch.tensor([float(sample["reward"])], dtype=torch.float32, device=device)
            delta_target = torch.tensor(sample["delta"], dtype=torch.float32, device=device).unsqueeze(0)
            usefulness_target = torch.tensor([float(sample["usefulness"])], dtype=torch.float32, device=device)
            world_loss, _metrics = world_model.loss(
                latent=latent,
                actions=[action],
                state=state,
                hidden=hidden,
                next_latent_target=next_encoded.latent.detach(),
                reward_target=reward_target,
                delta_target=delta_target,
                usefulness_target=usefulness_target,
            )
            prediction = world_model.step(
                latent,
                actions=[action],
                state=state,
                hidden=hidden,
            )
            belief_loss = language_model.teacher_forcing_loss(
                prediction.next_latent_mean,
                [_grounded_belief_tokens(next_state)],
                mode="belief",
            )
            question_loss = language_model.teacher_forcing_loss(
                prediction.next_latent_mean,
                [_grounded_question_tokens(next_state)],
                mode="question",
            )
            sequence_world_terms.append(world_loss)
            sequence_belief_terms.append(belief_loss)
            sequence_question_terms.append(question_loss)
            sequence_terms.append(
                world_loss
                + (float(config.dream_belief_weight) * belief_loss)
                + (float(config.dream_question_weight) * question_loss)
            )
            hidden = prediction.hidden
            latent = prediction.next_latent_mean
            total_steps += 1
        if not sequence_terms:
            continue
        dream_loss = float(config.dream_loss_weight) * (torch.stack(sequence_terms).mean())
        optimizer.zero_grad()
        dream_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            list(encoder.parameters()) + list(world_model.parameters()) + list(language_model.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        losses.append(float(dream_loss.detach().cpu()))
        world_losses.append(float(torch.stack(sequence_world_terms).mean().detach().cpu()))
        belief_losses.append(float(torch.stack(sequence_belief_terms).mean().detach().cpu()))
        question_losses.append(float(torch.stack(sequence_question_terms).mean().detach().cpu()))
    return {
        "dream_sequences": float(len(losses)),
        "dream_steps": float(total_steps),
        "dream_loss": mean(losses) if losses else 0.0,
        "dream_world_loss": mean(world_losses) if world_losses else 0.0,
        "dream_belief_loss": mean(belief_losses) if belief_losses else 0.0,
        "dream_question_loss": mean(question_losses) if question_losses else 0.0,
        "dream_seconds": perf_counter() - dream_started,
    }


def _build_eval_agent(
    encoder: StructuredStateEncoder,
    world_model: RecurrentWorldModel,
    language_model: GroundedLanguageModel,
    *,
    device: torch.device,
):
    from arcagi.agents.learned_agent import HybridAgent
    from arcagi.memory.episodic import EpisodicMemory

    eval_encoder, eval_world_model, eval_language_model, _ = build_default_modules(device=device)
    eval_encoder.load_state_dict(encoder.state_dict())
    eval_world_model.load_state_dict(world_model.state_dict())
    eval_language_model.load_state_dict(language_model.state_dict())
    planner = HybridPlanner()
    return HybridAgent(
        encoder=eval_encoder,
        world_model=eval_world_model,
        planner=planner,
        language_model=eval_language_model,
        episodic_memory=EpisodicMemory(),
        device=device,
    )


def _run_eval_episode(
    agent,
    env: HiddenRuleEnv,
    *,
    seed: int,
    trace_steps: int,
    trace_top_actions: int,
) -> dict[str, object]:
    observation = env.reset(seed=seed)
    agent.reset_episode()
    rewards: list[float] = []
    interaction_steps = 0
    action_family_counts: dict[str, int] = {}
    step_trace: list[dict[str, object]] = []
    done = False
    steps = 0
    success = False
    final_reason = "unknown"
    while not done:
        action = agent.act(observation)
        family = _action_family(action)
        action_family_counts[family] = action_family_counts.get(family, 0) + 1
        if family in {"interact", "click"}:
            interaction_steps += 1
        result = env.step(action)
        agent.update_after_step(
            next_observation=result.observation,
            reward=result.reward,
            terminated=result.terminated or result.truncated,
            info=result.info,
        )
        rewards.append(result.reward)
        steps += 1
        if result.reward > 0.9:
            success = True
        if steps <= trace_steps:
            step_trace.append(
                _step_trace_entry(
                    agent=agent,
                    env=env,
                    action=action,
                    reward=result.reward,
                    info=result.info,
                    step_index=steps,
                    trace_top_actions=trace_top_actions,
                )
            )
        observation = result.observation
        done = result.terminated or result.truncated
        if done:
            if success:
                final_reason = "goal_reached"
            elif result.truncated:
                final_reason = "timeout"
            else:
                final_reason = str(result.info.get("event", "terminated"))
    return {
        "success": success,
        "return": float(sum(rewards)),
        "steps": float(steps),
        "interaction_steps": float(interaction_steps),
        "action_family_counts": action_family_counts,
        "termination_reason": final_reason,
        "event_counts": _episode_event_counts(step_trace),
        "max_repeated_action_run": _max_repeated_action_run(step_trace),
        "step_trace": step_trace,
    }


def _evaluate_holdout(
    config: SyntheticTrainingConfig,
    encoder: StructuredStateEncoder,
    world_model: RecurrentWorldModel,
    language_model: GroundedLanguageModel,
    *,
    family_modes: tuple[str, ...],
    size_options: tuple[int, ...],
    seed_base: int,
    device: torch.device,
) -> dict[str, object]:
    if not family_modes:
        return {
            "success_rate": 0.0,
            "avg_return": 0.0,
            "avg_steps": 0.0,
            "avg_interactions": 0.0,
            "first_episode_success": 0.0,
            "later_episode_success": 0.0,
            "family_modes": (),
        }
    agent = _build_eval_agent(encoder, world_model, language_model, device=device)
    all_episodes: list[dict[str, object]] = []
    first_episode_success: list[float] = []
    later_episode_success: list[float] = []
    records_by_family: dict[str, list[dict[str, object]]] = {}
    records_by_variant: dict[str, list[dict[str, object]]] = {}
    records_by_size: dict[str, list[dict[str, object]]] = {}
    termination_reasons: list[str] = []
    action_family_totals: dict[str, int] = {}
    failure_examples: list[dict[str, object]] = []
    seed_cursor = seed_base
    size_cycle = size_options or config.size_options
    for family_mode in family_modes:
        for variant_index, variant in enumerate(family_variants_for_mode(family_mode)):
            agent.reset_all()
            variant_episodes: list[dict[str, object]] = []
            for episode_index in range(config.holdout_episodes_per_variant):
                size = size_cycle[(variant_index + episode_index) % len(size_cycle)]
                env = HiddenRuleEnv(
                    size=size,
                    max_steps=config.max_steps,
                    family_mode=family_mode,
                    family_variant=variant,
                    seed=seed_cursor,
                )
                episode = _run_eval_episode(
                    agent,
                    env,
                    seed=seed_cursor,
                    trace_steps=min(config.holdout_trace_steps, config.max_steps),
                    trace_top_actions=config.holdout_trace_top_actions,
                )
                variant_episodes.append(episode)
                records_by_family.setdefault(family_mode, []).append(episode)
                records_by_variant.setdefault(f"{family_mode}/{variant}", []).append(episode)
                records_by_size.setdefault(str(size), []).append(episode)
                termination_reasons.append(str(episode["termination_reason"]))
                for action_family, count in dict(episode["action_family_counts"]).items():
                    action_family_totals[action_family] = action_family_totals.get(action_family, 0) + int(count)
                if (
                    not bool(episode["success"])
                    and len(failure_examples) < config.holdout_failure_examples
                ):
                    failure_examples.append(
                        {
                            "family_mode": family_mode,
                            "family_variant": variant,
                            "size": size,
                            "return": float(episode["return"]),
                            "steps": float(episode["steps"]),
                            "interaction_steps": float(episode["interaction_steps"]),
                            "action_family_counts": dict(episode["action_family_counts"]),
                            "termination_reason": str(episode["termination_reason"]),
                            "event_counts": dict(episode["event_counts"]),
                            "max_repeated_action_run": int(episode["max_repeated_action_run"]),
                            "failure_signatures": _failure_signatures(
                                family_mode=family_mode,
                                family_variant=variant,
                                episode=episode,
                                step_trace=list(episode["step_trace"]),
                            ),
                            "step_trace": list(episode["step_trace"]),
                        }
                    )
                seed_cursor += 1
            all_episodes.extend(variant_episodes)
            first_episode_success.append(float(variant_episodes[0]["success"]))
            later_episode_success.extend(float(item["success"]) for item in variant_episodes[1:])
    action_family_rate = {
        key: value / max(len(all_episodes), 1)
        for key, value in sorted(action_family_totals.items())
    }
    return {
        "success_rate": mean(float(item["success"]) for item in all_episodes) if all_episodes else 0.0,
        "avg_return": mean(float(item["return"]) for item in all_episodes) if all_episodes else 0.0,
        "avg_steps": mean(float(item["steps"]) for item in all_episodes) if all_episodes else 0.0,
        "avg_interactions": mean(float(item["interaction_steps"]) for item in all_episodes) if all_episodes else 0.0,
        "first_episode_success": mean(first_episode_success) if first_episode_success else 0.0,
        "later_episode_success": mean(later_episode_success) if later_episode_success else 0.0,
        "family_modes": family_modes,
        "size_options": size_cycle,
        "family_breakdown": _sorted_metric_breakdown(records_by_family),
        "variant_breakdown": _sorted_metric_breakdown(records_by_variant),
        "size_breakdown": _sorted_metric_breakdown(records_by_size),
        "termination_reasons": _count_by_value(termination_reasons),
        "avg_action_family_counts": action_family_rate,
        "event_counts": _count_by_value(
            [
                event
                for episode in all_episodes
                for event, count in dict(episode["event_counts"]).items()
                for _ in range(int(count))
            ]
        ),
        "failure_examples": failure_examples,
    }


def _holdout_passed(
    stage: CurriculumStage,
    frontier_metrics: dict[str, object],
    regression_metrics: dict[str, object] | None,
) -> bool:
    if float(frontier_metrics["first_episode_success"]) < stage.first_episode_threshold:
        return False
    if float(frontier_metrics["later_episode_success"]) < stage.later_episode_threshold:
        return False
    if float(frontier_metrics["avg_return"]) < stage.avg_return_threshold:
        return False
    if float(frontier_metrics["avg_interactions"]) > stage.avg_interactions_cap:
        return False
    if regression_metrics is None:
        return True
    if float(regression_metrics["first_episode_success"]) < stage.regression_first_success_floor:
        return False
    if float(regression_metrics["later_episode_success"]) < stage.regression_later_success_floor:
        return False
    return True


def train_synthetic(
    config: SyntheticTrainingConfig,
    device: torch.device | None = None,
) -> dict[str, object]:
    seed_everything(config.seed)
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, world_model, language_model, _planner = build_default_modules(device=device)
    stages = _curriculum_stages(config)
    if config.init_checkpoint_path and Path(config.init_checkpoint_path).exists():
        checkpoint = torch.load(config.init_checkpoint_path, map_location=device)
        _load_compatible_state_dict(encoder, checkpoint["encoder"], module_name="encoder")
        _load_compatible_state_dict(world_model, checkpoint["world_model"], module_name="world_model")
        _load_compatible_state_dict(language_model, checkpoint["language_model"], module_name="language_model")
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(world_model.parameters()) + list(language_model.parameters()),
        lr=config.learning_rate,
    )
    history: list[dict[str, float | int | bool]] = []
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    current_epoch = -1
    current_collection_metrics: dict[str, object] = {}
    epoch_losses: list[float] = []
    epoch_uncertainty: list[float] = []
    epoch_world_total_losses: list[float] = []
    epoch_latent_losses: list[float] = []
    epoch_reward_losses: list[float] = []
    epoch_delta_losses: list[float] = []
    epoch_usefulness_losses: list[float] = []
    epoch_belief_losses: list[float] = []
    epoch_question_losses: list[float] = []
    epoch_policy_losses: list[float] = []
    epoch_positive_policy_losses: list[float] = []
    epoch_negative_policy_losses: list[float] = []
    epoch_gate_losses: list[float] = []
    dream_metrics: dict[str, float] = {}
    if config.curriculum in {"staged", "gated"}:
        current_stage_index = 0
    elif config.curriculum == "fixed_staged":
        current_stage_index = _fixed_stage_index_for_epoch(config, 0)
    else:
        current_stage_index = 0
    stage_epoch_count = 0
    consecutive_holdout_passes = 0
    holdout_history: list[dict[str, object]] = []
    last_holdout_result: dict[str, object] | None = None
    try:
        for epoch in range(config.epochs):
            current_epoch = epoch
            epoch_started = perf_counter()
            if config.curriculum == "fixed_staged":
                current_stage_index = _fixed_stage_index_for_epoch(config, epoch)
            dataset, current_collection_metrics = collect_dataset(
                config,
                epoch_index=epoch,
                stage_index=current_stage_index if config.curriculum in {"staged", "gated"} else None,
                history=history,
            )
            epoch_losses = []
            epoch_uncertainty = []
            epoch_world_total_losses = []
            epoch_latent_losses = []
            epoch_reward_losses = []
            epoch_delta_losses = []
            epoch_usefulness_losses = []
            epoch_belief_losses = []
            epoch_question_losses = []
            epoch_policy_losses = []
            epoch_positive_policy_losses = []
            epoch_negative_policy_losses = []
            epoch_gate_losses = []
            dream_metrics = {}
            encoder.train()
            world_model.train()
            language_model.train()
            sequence_episode_id: str | None = None
            sequence_hidden: torch.Tensor | None = None
            train_started = perf_counter()
            for sample in dataset:
                state = sample["state"]
                next_state = sample["next_state"]
                action = sample["action"]
                available_actions = sample["available_actions"]
                reward = sample["reward"]
                delta = sample["delta"]
                usefulness = sample["usefulness"]
                teacher_action = str(sample.get("teacher_action", ""))
                teacher_weight = float(sample.get("teacher_weight", 0.0))
                policy_target = float(sample["policy_target"])
                policy_weight = float(sample["policy_weight"])
                sibling_move_target = float(sample["sibling_move_target"])
                sibling_move_weight = float(sample["sibling_move_weight"])
                same_type_target = float(sample["same_type_target"])
                same_type_weight = float(sample["same_type_weight"])
                belief_tokens = sample["belief_tokens"]
                question_tokens = sample["question_tokens"]
                if sequence_episode_id != state.episode_id or state.step_index == 0:
                    sequence_episode_id = state.episode_id
                    sequence_hidden = None
                encoded = encoder.encode_state(state, device=device)
                with torch.no_grad():
                    next_encoded = encoder.encode_state(next_state, device=device)
                reward_target = torch.tensor([reward], dtype=torch.float32, device=device)
                delta_target = torch.tensor(delta, dtype=torch.float32, device=device).unsqueeze(0)
                usefulness_target = torch.tensor([usefulness], dtype=torch.float32, device=device)
                world_loss, metrics = world_model.loss(
                    latent=encoded.latent,
                    actions=[action],
                    state=state,
                    hidden=sequence_hidden,
                    next_latent_target=next_encoded.latent.detach(),
                    reward_target=reward_target,
                    delta_target=delta_target,
                    usefulness_target=usefulness_target,
                )
                belief_loss = language_model.teacher_forcing_loss(
                    encoded.latent,
                    [belief_tokens or ("goal", "unknown")],
                    mode="belief",
                )
                question_loss = language_model.teacher_forcing_loss(
                    encoded.latent,
                    [question_tokens or ("need", "explore")],
                    mode="question",
                )
                repeated_latent = encoded.latent.repeat(len(available_actions), 1)
                repeated_hidden = None
                if sequence_hidden is not None:
                    repeated_hidden = sequence_hidden.repeat(len(available_actions), 1)
                all_prediction = world_model.step(
                    repeated_latent,
                    actions=available_actions,
                    state=state,
                    hidden=repeated_hidden,
                )
                policy_targets = torch.zeros(len(available_actions), dtype=torch.float32, device=device)
                policy_weights = torch.ones(len(available_actions), dtype=torch.float32, device=device)
                schema_context = build_action_schema_context(available_actions, dict(state.action_roles))
                candidate_schemas = [build_action_schema(candidate, schema_context) for candidate in available_actions]

                def apply_policy_supervision(reference_action: str, supervision: PolicySupervision) -> None:
                    if not reference_action:
                        return
                    if (
                        supervision.weight <= 0.0
                        and supervision.sibling_move_weight <= 0.0
                        and supervision.same_type_weight <= 0.0
                    ):
                        return
                    reference_schema = build_action_schema(reference_action, schema_context)
                    for index, candidate in enumerate(available_actions):
                        candidate_schema = candidate_schemas[index]
                        if candidate == reference_action:
                            policy_targets[index] = max(float(policy_targets[index].item()), supervision.target)
                            policy_weights[index] = max(float(policy_weights[index].item()), supervision.weight)
                            continue
                        if (
                            reference_schema.action_type == "move"
                            and candidate_schema.action_type == "move"
                            and supervision.sibling_move_target > 0.0
                        ):
                            policy_targets[index] = max(
                                float(policy_targets[index].item()),
                                supervision.sibling_move_target,
                            )
                            policy_weights[index] = max(
                                float(policy_weights[index].item()),
                                supervision.sibling_move_weight,
                            )
                        if (
                            candidate_schema.action_type == reference_schema.action_type
                            and supervision.same_type_target > 0.0
                        ):
                            policy_targets[index] = max(
                                float(policy_targets[index].item()),
                                supervision.same_type_target,
                            )
                            policy_weights[index] = max(
                                float(policy_weights[index].item()),
                                supervision.same_type_weight,
                            )

                apply_policy_supervision(
                    action,
                    PolicySupervision(
                        target=policy_target,
                        weight=policy_weight,
                        sibling_move_target=sibling_move_target,
                        sibling_move_weight=sibling_move_weight,
                        same_type_target=same_type_target,
                        same_type_weight=same_type_weight,
                    ),
                )
                apply_policy_supervision(
                    teacher_action,
                    _teacher_policy_supervision(teacher_action, weight=teacher_weight),
                )
                policy_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                    all_prediction.policy,
                    policy_targets,
                    reduction="none",
                )
                policy_loss = (policy_raw * policy_weights).mean()
                positive_mask = policy_targets > 0.5
                negative_mask = ~positive_mask
                if positive_mask.any():
                    positive_policy_loss = (policy_raw[positive_mask] * policy_weights[positive_mask]).mean()
                else:
                    positive_policy_loss = torch.tensor(0.0, device=device)
                if negative_mask.any():
                    negative_policy_loss = (policy_raw[negative_mask] * policy_weights[negative_mask]).mean()
                else:
                    negative_policy_loss = torch.tensor(0.0, device=device)
                gate_target = torch.tensor(config.enhancement_gate_target, dtype=torch.float32, device=device)
                gate_value = torch.tanh(encoder.enhancement_gate)
                gate_loss = config.enhancement_gate_weight * torch.square(gate_value - gate_target)
                loss = world_loss + 0.3 * belief_loss + 0.3 * question_loss + 0.5 * policy_loss + gate_loss
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    list(encoder.parameters()) + list(world_model.parameters()) + list(language_model.parameters()),
                    max_norm=1.0,
                )
                optimizer.step()
                with torch.no_grad():
                    sequence_hidden = world_model.step(
                        encoded.latent.detach(),
                        actions=[action],
                        state=state,
                        hidden=sequence_hidden,
                    ).hidden.detach()
                epoch_losses.append(float(loss.detach().cpu()))
                epoch_uncertainty.append(metrics["uncertainty"])
                epoch_world_total_losses.append(metrics["loss_total"])
                epoch_latent_losses.append(metrics["loss_latent"])
                epoch_reward_losses.append(metrics["loss_reward"])
                epoch_delta_losses.append(metrics["loss_delta"])
                epoch_usefulness_losses.append(metrics["loss_usefulness"])
                epoch_belief_losses.append(float(belief_loss.detach().cpu()))
                epoch_question_losses.append(float(question_loss.detach().cpu()))
                epoch_policy_losses.append(float(policy_loss.detach().cpu()))
                epoch_positive_policy_losses.append(float(positive_policy_loss.detach().cpu()))
                epoch_negative_policy_losses.append(float(negative_policy_loss.detach().cpu()))
                epoch_gate_losses.append(float(gate_loss.detach().cpu()))
            train_seconds = perf_counter() - train_started
            dream_metrics = _run_dream_phase(
                config,
                dataset=dataset,
                epoch_index=epoch,
                encoder=encoder,
                world_model=world_model,
                language_model=language_model,
                optimizer=optimizer,
                device=device,
            )
            stage_epoch_count += 1
            current_stage = stages[min(max(current_stage_index, 0), len(stages) - 1)]
            epoch_stage_index = current_stage_index
            epoch_stage_name = current_stage.name
            epoch_stage_epoch_count = stage_epoch_count
            regression_families = _stage_regression_families(stages, current_stage_index)
            holdout_result: dict[str, object] | None = None
            should_run_holdout = (
                config.curriculum in {"staged", "gated"}
                and config.holdout_eval_every_epochs > 0
                and stage_epoch_count >= 1
                and (
                    stage_epoch_count % config.holdout_eval_every_epochs == 0
                    or epoch == config.epochs - 1
                )
            )
            if should_run_holdout:
                frontier_holdout = _evaluate_holdout(
                    config,
                    encoder,
                    world_model,
                    language_model,
                    family_modes=current_stage.frontier_families,
                    size_options=current_stage.holdout_size_options,
                    seed_base=config.seed + 100_000 + (current_stage_index * 10_000),
                    device=device,
                )
                regression_holdout = (
                    _evaluate_holdout(
                        config,
                        encoder,
                        world_model,
                        language_model,
                        family_modes=regression_families,
                        size_options=current_stage.holdout_size_options,
                        seed_base=config.seed + 200_000 + (current_stage_index * 10_000),
                        device=device,
                    )
                    if regression_families
                    else None
                )
                failure_reasons = _holdout_failure_reasons(current_stage, frontier_holdout, regression_holdout)
                passed_holdout = (
                    stage_epoch_count >= max(1, config.holdout_eval_every_epochs)
                    and _holdout_passed(current_stage, frontier_holdout, regression_holdout)
                )
                consecutive_holdout_passes = consecutive_holdout_passes + 1 if passed_holdout else 0
                holdout_result = {
                    "epoch": epoch,
                    "stage_index": current_stage_index,
                    "stage_name": current_stage.name,
                    "frontier": frontier_holdout,
                    "regression": regression_holdout,
                    "passed": passed_holdout,
                    "failure_reasons": failure_reasons,
                    "consecutive_passes": consecutive_holdout_passes,
                    "required_consecutive_passes": config.promotion_consecutive_evals,
                }
                holdout_history.append(holdout_result)
                last_holdout_result = holdout_result
                print(
                    json.dumps(
                        {
                            "type": "holdout_eval",
                            "epoch": epoch,
                            "stage_index": current_stage_index,
                            "stage_name": current_stage.name,
                            "stage_epoch_count": stage_epoch_count,
                            "frontier": frontier_holdout,
                            "regression": regression_holdout,
                            "passed": passed_holdout,
                            "failure_reasons": failure_reasons,
                            "consecutive_passes": consecutive_holdout_passes,
                            "required_consecutive_passes": config.promotion_consecutive_evals,
                        }
                    ),
                    flush=True,
                )
                if (
                    passed_holdout
                    and consecutive_holdout_passes >= config.promotion_consecutive_evals
                    and current_stage_index < len(stages) - 1
                ):
                    previous_stage_name = current_stage.name
                    current_stage_index += 1
                    stage_epoch_count = 0
                    consecutive_holdout_passes = 0
                    print(
                        json.dumps(
                            {
                                "type": "stage_advance",
                                "epoch": epoch,
                                "from_stage_index": current_stage_index - 1,
                                "from_stage_name": previous_stage_name,
                                "to_stage_index": current_stage_index,
                                "to_stage_name": stages[current_stage_index].name,
                                "reason": "heldout_thresholds_cleared",
                            }
                        ),
                        flush=True,
                    )
            history.append(
                {
                    "epoch": float(epoch),
                    "loss": mean(epoch_losses) if epoch_losses else 0.0,
                    "uncertainty": mean(epoch_uncertainty) if epoch_uncertainty else 0.0,
                    "samples": current_collection_metrics.get("samples", float(len(dataset))),
                    "collect_avg_return": current_collection_metrics.get("avg_return", 0.0),
                    "collect_avg_steps": current_collection_metrics.get("avg_steps", 0.0),
                    "collect_success_rate": current_collection_metrics.get("success_rate", 0.0),
                    "stage_index": float(epoch_stage_index),
                    "stage_name": current_collection_metrics.get("stage_name", epoch_stage_name),
                    "stage_epoch_count": float(epoch_stage_epoch_count),
                    "frontier_families": tuple(current_collection_metrics.get("frontier_families", ())),
                    "family_counts": current_collection_metrics.get("family_counts", {}),
                    "family_breakdown": current_collection_metrics.get("family_breakdown", {}),
                    "variant_breakdown": current_collection_metrics.get("variant_breakdown", {}),
                    "size_breakdown": current_collection_metrics.get("size_breakdown", {}),
                    "epoch_seed_base": current_collection_metrics.get("epoch_seed_base", 0.0),
                    "collect_seconds": current_collection_metrics.get("collect_seconds", 0.0),
                    "train_seconds": train_seconds,
                    "epoch_seconds": perf_counter() - epoch_started,
                    "world_loss": mean(epoch_world_total_losses) if epoch_world_total_losses else 0.0,
                    "latent_loss": mean(epoch_latent_losses) if epoch_latent_losses else 0.0,
                    "reward_loss": mean(epoch_reward_losses) if epoch_reward_losses else 0.0,
                    "delta_loss": mean(epoch_delta_losses) if epoch_delta_losses else 0.0,
                    "usefulness_loss": mean(epoch_usefulness_losses) if epoch_usefulness_losses else 0.0,
                    "belief_loss": mean(epoch_belief_losses) if epoch_belief_losses else 0.0,
                    "question_loss": mean(epoch_question_losses) if epoch_question_losses else 0.0,
                    "policy_loss": mean(epoch_policy_losses) if epoch_policy_losses else 0.0,
                    "positive_policy_loss": mean(epoch_positive_policy_losses) if epoch_positive_policy_losses else 0.0,
                    "negative_policy_loss": mean(epoch_negative_policy_losses) if epoch_negative_policy_losses else 0.0,
                    "gate_loss": mean(epoch_gate_losses) if epoch_gate_losses else 0.0,
                    "encoder_enhancement_gate_raw": float(encoder.enhancement_gate.detach().cpu()),
                    "encoder_enhancement_gate_tanh": float(torch.tanh(encoder.enhancement_gate.detach()).cpu()),
                    "holdout_passed": bool(holdout_result["passed"]) if holdout_result is not None else False,
                    "holdout_evaluated": holdout_result is not None,
                    "holdout_frontier_success": float(holdout_result["frontier"]["success_rate"])
                    if holdout_result is not None
                    else 0.0,
                    "holdout_consecutive_passes": float(holdout_result["consecutive_passes"])
                    if holdout_result is not None
                    else float(consecutive_holdout_passes),
                    "bootstrap_steps": current_collection_metrics.get("bootstrap_steps", 0.0),
                    "bootstrap_stride": current_collection_metrics.get("bootstrap_stride", 1.0),
                    "bootstrap_release_epoch": current_collection_metrics.get("bootstrap_release_epoch", -1.0),
                    "bootstrap_ready_streak": current_collection_metrics.get("bootstrap_ready_streak", 0.0),
                    "teacher_guidance_ready_streak": current_collection_metrics.get("teacher_guidance_ready_streak", 0.0),
                    "teacher_guidance_alpha": current_collection_metrics.get("teacher_guidance_alpha", 0.0),
                    "teacher_episode_fraction": current_collection_metrics.get("teacher_episode_fraction", 0.0),
                    "teacher_takeover_prob": current_collection_metrics.get("teacher_takeover_prob", 0.0),
                    "teacher_episode_count": current_collection_metrics.get("teacher_episode_count", 0.0),
                    "teacher_step_fraction": current_collection_metrics.get("teacher_step_fraction", 0.0),
                    "teacher_relabel_fraction": current_collection_metrics.get("teacher_relabel_fraction", 0.0),
                    "dream_sequences": dream_metrics.get("dream_sequences", 0.0),
                    "dream_steps": dream_metrics.get("dream_steps", 0.0),
                    "dream_loss": dream_metrics.get("dream_loss", 0.0),
                    "dream_world_loss": dream_metrics.get("dream_world_loss", 0.0),
                    "dream_belief_loss": dream_metrics.get("dream_belief_loss", 0.0),
                    "dream_question_loss": dream_metrics.get("dream_question_loss", 0.0),
                    "dream_seconds": dream_metrics.get("dream_seconds", 0.0),
                }
            )
            training_state = {
                "completed_epochs": len(history),
                "active_epoch": epoch,
                "interrupted": False,
                "last_epoch_samples": history[-1]["samples"],
                "stage_index": current_stage_index,
                "stage_name": stages[current_stage_index].name if stages else "",
                "stage_epoch_count": stage_epoch_count,
                "consecutive_holdout_passes": consecutive_holdout_passes,
                "bootstrap_steps": current_collection_metrics.get("bootstrap_steps", 0.0),
                "bootstrap_stride": current_collection_metrics.get("bootstrap_stride", 1.0),
                "bootstrap_release_epoch": current_collection_metrics.get("bootstrap_release_epoch", -1.0),
                "bootstrap_ready_streak": current_collection_metrics.get("bootstrap_ready_streak", 0.0),
                "teacher_guidance_ready_streak": current_collection_metrics.get("teacher_guidance_ready_streak", 0.0),
                "teacher_guidance_alpha": current_collection_metrics.get("teacher_guidance_alpha", 0.0),
                "teacher_episode_fraction": current_collection_metrics.get("teacher_episode_fraction", 0.0),
                "teacher_takeover_prob": current_collection_metrics.get("teacher_takeover_prob", 0.0),
                "teacher_episode_count": current_collection_metrics.get("teacher_episode_count", 0.0),
                "teacher_step_fraction": current_collection_metrics.get("teacher_step_fraction", 0.0),
                "teacher_relabel_fraction": current_collection_metrics.get("teacher_relabel_fraction", 0.0),
                "dream_sequences": dream_metrics.get("dream_sequences", 0.0),
                "dream_steps": dream_metrics.get("dream_steps", 0.0),
                "dream_loss": dream_metrics.get("dream_loss", 0.0),
                "dream_world_loss": dream_metrics.get("dream_world_loss", 0.0),
                "dream_belief_loss": dream_metrics.get("dream_belief_loss", 0.0),
                "dream_question_loss": dream_metrics.get("dream_question_loss", 0.0),
                "dream_seconds": dream_metrics.get("dream_seconds", 0.0),
            }
            snapshot = _build_checkpoint_snapshot(
                config,
                encoder,
                world_model,
                language_model,
                history,
                holdout_history,
                training_state,
                last_holdout_result,
            )
            saved_paths = _save_checkpoint(checkpoint_path, snapshot, epoch=epoch)
            print(
                json.dumps(
                    {
                        "type": "epoch",
                        "epoch": epoch,
                        "loss": history[-1]["loss"],
                        "uncertainty": history[-1]["uncertainty"],
                        "samples": history[-1]["samples"],
                        "stage_index": epoch_stage_index,
                        "stage_name": current_collection_metrics.get("stage_name", epoch_stage_name),
                        "stage_epoch_count": epoch_stage_epoch_count,
                        "frontier_families": current_collection_metrics.get("frontier_families", ()),
                        "collect_avg_return": history[-1]["collect_avg_return"],
                        "collect_avg_steps": history[-1]["collect_avg_steps"],
                        "collect_success_rate": history[-1]["collect_success_rate"],
                        "family_counts": current_collection_metrics.get("family_counts", {}),
                        "family_breakdown": current_collection_metrics.get("family_breakdown", {}),
                        "variant_breakdown": current_collection_metrics.get("variant_breakdown", {}),
                        "size_breakdown": current_collection_metrics.get("size_breakdown", {}),
                        "epoch_seed_base": current_collection_metrics.get("epoch_seed_base", 0.0),
                        "collect_seconds": history[-1]["collect_seconds"],
                        "train_seconds": history[-1]["train_seconds"],
                        "epoch_seconds": history[-1]["epoch_seconds"],
                        "world_loss": history[-1]["world_loss"],
                        "latent_loss": history[-1]["latent_loss"],
                        "reward_loss": history[-1]["reward_loss"],
                        "delta_loss": history[-1]["delta_loss"],
                        "usefulness_loss": history[-1]["usefulness_loss"],
                        "belief_loss": history[-1]["belief_loss"],
                        "question_loss": history[-1]["question_loss"],
                        "policy_loss": history[-1]["policy_loss"],
                        "positive_policy_loss": history[-1]["positive_policy_loss"],
                        "negative_policy_loss": history[-1]["negative_policy_loss"],
                        "gate_loss": history[-1]["gate_loss"],
                        "encoder_enhancement_gate_raw": history[-1]["encoder_enhancement_gate_raw"],
                        "encoder_enhancement_gate_tanh": history[-1]["encoder_enhancement_gate_tanh"],
                        "holdout_passed": history[-1]["holdout_passed"],
                        "holdout_consecutive_passes": history[-1]["holdout_consecutive_passes"],
                        "holdout_failure_reasons": holdout_result["failure_reasons"] if holdout_result is not None else [],
                        "bootstrap_steps": history[-1]["bootstrap_steps"],
                        "bootstrap_stride": history[-1]["bootstrap_stride"],
                        "bootstrap_release_epoch": history[-1]["bootstrap_release_epoch"],
                        "bootstrap_ready_streak": history[-1]["bootstrap_ready_streak"],
                        "dream_sequences": history[-1]["dream_sequences"],
                        "dream_steps": history[-1]["dream_steps"],
                        "dream_loss": history[-1]["dream_loss"],
                        "dream_world_loss": history[-1]["dream_world_loss"],
                        "dream_belief_loss": history[-1]["dream_belief_loss"],
                        "dream_question_loss": history[-1]["dream_question_loss"],
                        "dream_seconds": history[-1]["dream_seconds"],
                        **saved_paths,
                    }
                ),
                flush=True,
            )
    except KeyboardInterrupt:
        training_state = {
            "completed_epochs": len(history),
            "active_epoch": current_epoch,
            "interrupted": True,
            "partial_epoch_loss": mean(epoch_losses) if epoch_losses else 0.0,
            "partial_epoch_uncertainty": mean(epoch_uncertainty) if epoch_uncertainty else 0.0,
            "partial_epoch_samples": float(current_collection_metrics.get("samples", 0.0)),
            "partial_world_loss": mean(epoch_world_total_losses) if epoch_world_total_losses else 0.0,
            "partial_policy_loss": mean(epoch_policy_losses) if epoch_policy_losses else 0.0,
            "stage_index": current_stage_index,
            "stage_name": stages[current_stage_index].name if stages else "",
            "stage_epoch_count": stage_epoch_count,
            "consecutive_holdout_passes": consecutive_holdout_passes,
            "bootstrap_steps": float(current_collection_metrics.get("bootstrap_steps", 0.0)),
            "bootstrap_stride": float(current_collection_metrics.get("bootstrap_stride", 1.0)),
            "bootstrap_release_epoch": float(current_collection_metrics.get("bootstrap_release_epoch", -1.0)),
            "bootstrap_ready_streak": float(current_collection_metrics.get("bootstrap_ready_streak", 0.0)),
            "dream_sequences": float(dream_metrics.get("dream_sequences", 0.0)),
            "dream_steps": float(dream_metrics.get("dream_steps", 0.0)),
            "dream_loss": float(dream_metrics.get("dream_loss", 0.0)),
            "dream_world_loss": float(dream_metrics.get("dream_world_loss", 0.0)),
            "dream_belief_loss": float(dream_metrics.get("dream_belief_loss", 0.0)),
            "dream_question_loss": float(dream_metrics.get("dream_question_loss", 0.0)),
            "dream_seconds": float(dream_metrics.get("dream_seconds", 0.0)),
        }
        snapshot = _build_checkpoint_snapshot(
            config,
            encoder,
            world_model,
            language_model,
            history,
            holdout_history,
            training_state,
            last_holdout_result,
        )
        interrupt_paths = _save_checkpoint(checkpoint_path, snapshot, interrupted=True)
        print(
            json.dumps(
                {
                    "type": "interrupt",
                    "epoch": current_epoch,
                    "completed_epochs": len(history),
                    "partial_epoch_loss": training_state["partial_epoch_loss"],
                    "partial_epoch_uncertainty": training_state["partial_epoch_uncertainty"],
                    "partial_epoch_samples": training_state["partial_epoch_samples"],
                    "partial_dream_loss": training_state["dream_loss"],
                    "partial_dream_steps": training_state["dream_steps"],
                    **interrupt_paths,
                }
            ),
            flush=True,
        )
        return {
            "interrupted": True,
            "epochs_completed": len(history),
            "active_epoch": current_epoch,
            "stage_index": current_stage_index,
            "stage_name": stages[current_stage_index].name if stages else "",
            "samples_last_epoch": history[-1]["samples"] if history else 0.0,
            "loss_last_epoch": history[-1]["loss"] if history else 0.0,
            "uncertainty_last_epoch": history[-1]["uncertainty"] if history else 0.0,
            "last_holdout_result": last_holdout_result or {},
            **interrupt_paths,
        }
    return {
        "interrupted": False,
        "epochs": float(config.epochs),
        "epochs_completed": len(history),
        "stage_index": current_stage_index,
        "stage_name": stages[current_stage_index].name if stages else "",
        "samples_last_epoch": history[-1]["samples"] if history else 0.0,
        "loss_last_epoch": history[-1]["loss"] if history else 0.0,
        "uncertainty_last_epoch": history[-1]["uncertainty"] if history else 0.0,
        "last_holdout_result": last_holdout_result or {},
        "latest_checkpoint_path": str(checkpoint_path),
        "last_epoch_checkpoint_path": str(_checkpoint_variant_path(checkpoint_path, f"epoch_{len(history) - 1:04d}"))
        if history
        else "",
    }


def load_checkpoint(
    checkpoint_path: str,
    device: torch.device | None = None,
) -> tuple[StructuredStateEncoder, RecurrentWorldModel, GroundedLanguageModel]:
    device = device or torch.device("cpu")
    encoder, world_model, language_model, _planner = build_default_modules(device=device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    _load_compatible_state_dict(encoder, checkpoint["encoder"], module_name="encoder")
    _load_compatible_state_dict(world_model, checkpoint["world_model"], module_name="world_model")
    _load_compatible_state_dict(language_model, checkpoint["language_model"], module_name="language_model")
    encoder.eval()
    world_model.eval()
    language_model.eval()
    return encoder, world_model, language_model


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m arcagi.training.synthetic")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--episodes-per-epoch", type=int, default=96)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--checkpoint-path", type=str, default="artifacts/synthetic_hybrid.pt")
    parser.add_argument("--init-checkpoint-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--behavior-policy", type=str, default="mixed")
    parser.add_argument("--curriculum", type=str, default="staged")
    parser.add_argument("--oracle-imitation-epochs", type=int, default=2)
    parser.add_argument("--oracle-bootstrap-steps", type=int, default=16)
    parser.add_argument("--oracle-bootstrap-stride", type=int, default=1)
    parser.add_argument("--oracle-bootstrap-min-steps", type=int, default=8)
    parser.add_argument("--oracle-bootstrap-min-stride", type=int, default=2)
    parser.add_argument("--oracle-bootstrap-full-epochs", type=int, default=4)
    parser.add_argument("--oracle-bootstrap-decay-epochs", type=int, default=8)
    parser.add_argument("--oracle-bootstrap-decay-success-threshold", type=float, default=0.9)
    parser.add_argument("--oracle-bootstrap-decay-stability-epochs", type=int, default=2)
    parser.add_argument("--teacher-guidance-holdout-success-threshold", type=float, default=0.2)
    parser.add_argument("--teacher-episode-fraction-initial", type=float, default=0.2)
    parser.add_argument("--teacher-episode-fraction-floor", type=float, default=0.2)
    parser.add_argument("--teacher-takeover-prob-initial", type=float, default=0.5)
    parser.add_argument("--teacher-takeover-prob-floor", type=float, default=0.2)
    parser.add_argument("--teacher-relabel-weight", type=float, default=0.8)
    parser.add_argument("--dream-batches-per-epoch", type=int, default=8)
    parser.add_argument("--dream-batch-size", type=int, default=8)
    parser.add_argument("--dream-horizon", type=int, default=3)
    parser.add_argument("--dream-loss-weight", type=float, default=0.35)
    parser.add_argument("--dream-belief-weight", type=float, default=0.2)
    parser.add_argument("--dream-question-weight", type=float, default=0.2)
    parser.add_argument("--log-every-episodes", type=int, default=16)
    parser.add_argument("--holdout-eval-every-epochs", type=int, default=4)
    parser.add_argument("--holdout-episodes-per-variant", type=int, default=2)
    parser.add_argument("--promotion-consecutive-evals", type=int, default=2)
    parser.add_argument("--frontier-replay-weight", type=int, default=3)
    parser.add_argument("--previous-stage-replay-weight", type=int, default=1)
    parser.add_argument("--holdout-failure-examples", type=int, default=5)
    parser.add_argument("--holdout-trace-steps", type=int, default=48)
    parser.add_argument("--holdout-trace-top-actions", type=int, default=3)
    parser.add_argument("--enhancement-gate-target", type=float, default=0.3)
    parser.add_argument("--enhancement-gate-weight", type=float, default=0.05)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = SyntheticTrainingConfig(
        epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        learning_rate=args.learning_rate,
        checkpoint_path=args.checkpoint_path,
        init_checkpoint_path=args.init_checkpoint_path,
        seed=args.seed,
        behavior_policy=args.behavior_policy,
        curriculum=args.curriculum,
        oracle_imitation_epochs=args.oracle_imitation_epochs,
        oracle_bootstrap_steps=args.oracle_bootstrap_steps,
        oracle_bootstrap_stride=args.oracle_bootstrap_stride,
        oracle_bootstrap_min_steps=args.oracle_bootstrap_min_steps,
        oracle_bootstrap_min_stride=args.oracle_bootstrap_min_stride,
        oracle_bootstrap_full_epochs=args.oracle_bootstrap_full_epochs,
        oracle_bootstrap_decay_epochs=args.oracle_bootstrap_decay_epochs,
        oracle_bootstrap_decay_success_threshold=args.oracle_bootstrap_decay_success_threshold,
        oracle_bootstrap_decay_stability_epochs=args.oracle_bootstrap_decay_stability_epochs,
        teacher_guidance_holdout_success_threshold=args.teacher_guidance_holdout_success_threshold,
        teacher_episode_fraction_initial=args.teacher_episode_fraction_initial,
        teacher_episode_fraction_floor=args.teacher_episode_fraction_floor,
        teacher_takeover_prob_initial=args.teacher_takeover_prob_initial,
        teacher_takeover_prob_floor=args.teacher_takeover_prob_floor,
        teacher_relabel_weight=args.teacher_relabel_weight,
        dream_batches_per_epoch=args.dream_batches_per_epoch,
        dream_batch_size=args.dream_batch_size,
        dream_horizon=args.dream_horizon,
        dream_loss_weight=args.dream_loss_weight,
        dream_belief_weight=args.dream_belief_weight,
        dream_question_weight=args.dream_question_weight,
        log_every_episodes=args.log_every_episodes,
        holdout_eval_every_epochs=args.holdout_eval_every_epochs,
        holdout_episodes_per_variant=args.holdout_episodes_per_variant,
        promotion_consecutive_evals=args.promotion_consecutive_evals,
        frontier_replay_weight=args.frontier_replay_weight,
        previous_stage_replay_weight=args.previous_stage_replay_weight,
        holdout_failure_examples=args.holdout_failure_examples,
        holdout_trace_steps=args.holdout_trace_steps,
        holdout_trace_top_actions=args.holdout_trace_top_actions,
        enhancement_gate_target=args.enhancement_gate_target,
        enhancement_gate_weight=args.enhancement_gate_weight,
    )
    metrics = train_synthetic(config)
    print(json.dumps(metrics, indent=2))
    return 0


def _family_modes_for_epoch(config: SyntheticTrainingConfig, epoch_index: int) -> tuple[str, ...]:
    if config.curriculum != "staged":
        return config.family_modes
    if config.epochs <= 1:
        return config.family_modes
    first_stage = max(1, config.epochs // 3)
    second_stage = max(first_stage + 1, (2 * config.epochs) // 3)
    if epoch_index < first_stage:
        return ("switch_unlock", "order_collect")
    if epoch_index < second_stage:
        return ("switch_unlock", "order_collect", "selector_unlock", "delayed_order_unlock")
    return config.family_modes


def _size_options_for_epoch(config: SyntheticTrainingConfig, epoch_index: int) -> tuple[int, ...]:
    if config.curriculum != "staged":
        return config.size_options
    if epoch_index < max(1, config.epochs // 3):
        return tuple(size for size in config.size_options if size <= 8) or config.size_options
    return config.size_options


if __name__ == "__main__":
    raise SystemExit(main())
