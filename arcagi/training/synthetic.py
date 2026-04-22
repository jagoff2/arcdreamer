from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass, field
import json
import logging
import multiprocessing as mp
from pathlib import Path
import queue
from statistics import mean
from time import perf_counter

import numpy as np
import torch

from arcagi.agents.graph_agent import GraphExplorerAgent
from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.progress_signals import (
    hindsight_supervision,
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
from arcagi.memory.episodic import EpisodicMemory
from arcagi.perception.object_encoder import extract_structured_state
from arcagi.planning.planner import HybridPlanner, PlannerConfig
from arcagi.training.synthetic_oracle import oracle_action

logger = logging.getLogger(__name__)

_GRAPH_COLLECTION_POLICIES: frozenset[str] = frozenset({"explore", "graph"})
_LEARNED_COLLECTION_POLICIES: frozenset[str] = frozenset({"mixed", "bootstrap", "oracle", "learned", "hybrid"})
_ALLOWED_SYNTHETIC_BEHAVIOR_POLICIES: frozenset[str] = _GRAPH_COLLECTION_POLICIES | _LEARNED_COLLECTION_POLICIES
_ALLOWED_CURRICULA: frozenset[str] = frozenset({"staged", "gated", "fixed_staged", "flat"})


@dataclass(frozen=True)
class SyntheticTrainingConfig:
    family_modes: tuple[str, ...] = DEFAULT_SYNTHETIC_FAMILY_MODES
    episodes_per_epoch: int = 96
    epochs: int = 8
    max_steps: int = 48
    learning_rate: float = 3e-4
    seed: int = 7
    checkpoint_path: str = "artifacts/synthetic_hybrid.pt"
    train_device: str = ""
    async_eval_device: str = ""
    behavior_policy: str = "mixed"
    size_options: tuple[int, ...] = (7, 8, 9)
    init_checkpoint_path: str = ""
    resume_checkpoint_path: str = ""
    allow_weights_only_init_from_training_checkpoint: bool = False
    curriculum: str = "staged"
    log_every_episodes: int = 16
    holdout_eval_every_epochs: int = 4
    holdout_episodes_per_variant: int = 2
    regression_holdout_every_evals: int = 1
    promotion_consecutive_evals: int = 2
    frontier_replay_weight: int = 3
    previous_stage_replay_weight: int = 1
    flat_family_weight_floor: int = 1
    flat_family_weight_ceiling: int = 4
    flat_family_focus_strength: float = 3.0
    flat_family_history_window: int = 3
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
    teacher_episode_fraction_initial: float = 0.1
    teacher_episode_fraction_floor: float = 0.05
    teacher_takeover_prob_initial: float = 0.25
    teacher_takeover_prob_floor: float = 0.05
    teacher_relabel_weight: float = 0.8
    teacher_ownership_window: int = 4
    teacher_agreement_target: float = 0.78
    teacher_success_target: float = 0.8
    trajectory_credit_discount: float = 0.94
    dream_batches_per_epoch: int = 8
    dream_batch_size: int = 8
    dream_horizon: int = 3
    dream_loss_weight: float = 0.35
    dream_belief_weight: float = 0.2
    dream_question_weight: float = 0.2
    dream_plan_weight: float = 0.2
    theory_loss_weight: float = 0.45
    diagnostic_language_loss_weight: float = 0.35
    holdout_failure_examples: int = 5
    holdout_trace_steps: int = 48
    holdout_trace_top_actions: int = 3
    enhancement_gate_target: float = 0.3
    enhancement_gate_weight: float = 0.05


@dataclass
class _AsyncHoldoutRuntime:
    request_queue: object
    result_queue: object
    process: mp.Process
    next_request_index: int = 1
    next_result_index: int = 1
    pending_count: int = 0
    buffered_results: dict[int, dict[str, object]] = field(default_factory=dict)


@dataclass
class _SecondaryTrainingRuntime:
    device: torch.device
    encoder: StructuredStateEncoder
    world_model: RecurrentWorldModel
    language_model: GroundedLanguageModel


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


def _trainable_parameters(
    encoder: StructuredStateEncoder,
    world_model: RecurrentWorldModel,
    language_model: GroundedLanguageModel,
) -> list[torch.nn.Parameter]:
    return list(encoder.parameters()) + list(world_model.parameters()) + list(language_model.parameters())


def _build_secondary_training_runtime(
    encoder: StructuredStateEncoder,
    world_model: RecurrentWorldModel,
    language_model: GroundedLanguageModel,
    *,
    device: torch.device,
) -> _SecondaryTrainingRuntime:
    secondary_encoder, secondary_world_model, secondary_language_model, _ = build_default_modules(device=device)
    secondary_encoder.load_state_dict(encoder.state_dict())
    secondary_world_model.load_state_dict(world_model.state_dict())
    secondary_language_model.load_state_dict(language_model.state_dict())
    secondary_encoder.train(mode=encoder.training)
    secondary_world_model.train(mode=world_model.training)
    secondary_language_model.train(mode=language_model.training)
    return _SecondaryTrainingRuntime(
        device=device,
        encoder=secondary_encoder,
        world_model=secondary_world_model,
        language_model=secondary_language_model,
    )


def _sync_secondary_training_runtime(
    runtime: _SecondaryTrainingRuntime,
    encoder: StructuredStateEncoder,
    world_model: RecurrentWorldModel,
    language_model: GroundedLanguageModel,
) -> None:
    runtime.encoder.load_state_dict(encoder.state_dict())
    runtime.world_model.load_state_dict(world_model.state_dict())
    runtime.language_model.load_state_dict(language_model.state_dict())
    runtime.encoder.train(mode=encoder.training)
    runtime.world_model.train(mode=world_model.training)
    runtime.language_model.train(mode=language_model.training)


def _zero_module_grads(*modules: torch.nn.Module) -> None:
    for module in modules:
        module.zero_grad(set_to_none=True)


def _merge_secondary_gradients(
    encoder: StructuredStateEncoder,
    world_model: RecurrentWorldModel,
    language_model: GroundedLanguageModel,
    secondary_runtime: _SecondaryTrainingRuntime,
    *,
    contributors: int,
) -> None:
    if contributors <= 1:
        return
    primary_parameters = _trainable_parameters(encoder, world_model, language_model)
    secondary_parameters = _trainable_parameters(
        secondary_runtime.encoder,
        secondary_runtime.world_model,
        secondary_runtime.language_model,
    )
    scale = 1.0 / float(contributors)
    for primary_parameter, secondary_parameter in zip(primary_parameters, secondary_parameters, strict=True):
        secondary_grad = secondary_parameter.grad
        if secondary_grad is None:
            if primary_parameter.grad is not None:
                primary_parameter.grad.mul_(scale)
            continue
        moved_secondary = secondary_grad.detach().to(primary_parameter.device)
        if primary_parameter.grad is None:
            primary_parameter.grad = moved_secondary.mul(scale)
            continue
        primary_parameter.grad.mul_(scale)
        primary_parameter.grad.add_(moved_secondary, alpha=scale)


def _validate_synthetic_training_config(config: SyntheticTrainingConfig) -> None:
    if int(config.epochs) <= 0:
        raise ValueError("epochs must be positive")
    if int(config.episodes_per_epoch) <= 0:
        raise ValueError("episodes_per_epoch must be positive")
    if int(config.max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if str(config.behavior_policy) not in _ALLOWED_SYNTHETIC_BEHAVIOR_POLICIES:
        raise ValueError(
            f"unsupported behavior_policy: {config.behavior_policy}. "
            f"expected one of {sorted(_ALLOWED_SYNTHETIC_BEHAVIOR_POLICIES)}"
        )
    if str(config.curriculum) not in _ALLOWED_CURRICULA:
        raise ValueError(f"unsupported curriculum: {config.curriculum}. expected one of {sorted(_ALLOWED_CURRICULA)}")
    if not config.size_options:
        raise ValueError("size_options must not be empty")
    if float(config.teacher_episode_fraction_initial) < 0.0 or float(config.teacher_episode_fraction_initial) > 1.0:
        raise ValueError("teacher_episode_fraction_initial must be within [0, 1]")
    if float(config.teacher_episode_fraction_floor) < 0.0 or float(config.teacher_episode_fraction_floor) > 1.0:
        raise ValueError("teacher_episode_fraction_floor must be within [0, 1]")
    if float(config.teacher_takeover_prob_initial) < 0.0 or float(config.teacher_takeover_prob_initial) > 1.0:
        raise ValueError("teacher_takeover_prob_initial must be within [0, 1]")
    if float(config.teacher_takeover_prob_floor) < 0.0 or float(config.teacher_takeover_prob_floor) > 1.0:
        raise ValueError("teacher_takeover_prob_floor must be within [0, 1]")
    if float(config.teacher_agreement_target) < 0.0 or float(config.teacher_agreement_target) > 1.0:
        raise ValueError("teacher_agreement_target must be within [0, 1]")
    if float(config.teacher_success_target) < 0.0 or float(config.teacher_success_target) > 1.0:
        raise ValueError("teacher_success_target must be within [0, 1]")
    if int(config.teacher_ownership_window) <= 0:
        raise ValueError("teacher_ownership_window must be positive")
    if float(config.trajectory_credit_discount) < 0.0 or float(config.trajectory_credit_discount) > 1.0:
        raise ValueError("trajectory_credit_discount must be within [0, 1]")
    if float(config.theory_loss_weight) < 0.0:
        raise ValueError("theory_loss_weight must be non-negative")
    if float(config.diagnostic_language_loss_weight) < 0.0:
        raise ValueError("diagnostic_language_loss_weight must be non-negative")
    if config.init_checkpoint_path and config.resume_checkpoint_path:
        raise ValueError("init_checkpoint_path and resume_checkpoint_path are mutually exclusive")


def _clone_collection_modules(
    encoder: StructuredStateEncoder,
    world_model: RecurrentWorldModel,
    language_model: GroundedLanguageModel,
    *,
    device: torch.device,
) -> tuple[StructuredStateEncoder, RecurrentWorldModel, GroundedLanguageModel]:
    cloned_encoder, cloned_world_model, cloned_language_model, _ = build_default_modules(device=device)
    cloned_encoder.load_state_dict(encoder.state_dict())
    cloned_world_model.load_state_dict(world_model.state_dict())
    cloned_language_model.load_state_dict(language_model.state_dict())
    cloned_encoder.eval()
    cloned_world_model.eval()
    cloned_language_model.eval()
    return cloned_encoder, cloned_world_model, cloned_language_model


def _build_collection_agent(
    config: SyntheticTrainingConfig,
    *,
    encoder: StructuredStateEncoder | None = None,
    world_model: RecurrentWorldModel | None = None,
    language_model: GroundedLanguageModel | None = None,
    device: torch.device | None = None,
) -> tuple[object, str, bool]:
    policy = str(config.behavior_policy)
    if policy in _GRAPH_COLLECTION_POLICIES or encoder is None or world_model is None or language_model is None:
        return GraphExplorerAgent(), "graph", False
    from arcagi.agents.learned_agent import HybridAgent

    collector_device = device or torch.device("cpu")
    collection_encoder, collection_world_model, collection_language_model = _clone_collection_modules(
        encoder,
        world_model,
        language_model,
        device=collector_device,
    )
    planner = HybridPlanner(
        PlannerConfig(search_depth=2, search_root_width=2, search_branch_width=1, max_world_model_calls=48)
        if collector_device.type == "cpu"
        else PlannerConfig(search_depth=2, search_root_width=2, search_branch_width=1, max_world_model_calls=32)
    )
    return (
        HybridAgent(
            encoder=collection_encoder,
            world_model=collection_world_model,
            planner=planner,
            language_model=collection_language_model,
            episodic_memory=EpisodicMemory(),
            device=collector_device,
        ),
        "hybrid",
        True,
    )


def _latest_regression_snapshot(holdout_history: list[dict[str, object]]) -> dict[str, object] | None:
    for item in reversed(holdout_history):
        regression = item.get("regression")
        if regression is not None:
            return dict(regression)
    return None


def _load_resume_state(
    checkpoint_path: str,
    *,
    encoder: StructuredStateEncoder,
    world_model: RecurrentWorldModel,
    language_model: GroundedLanguageModel,
    device: torch.device,
) -> dict[str, object]:
    checkpoint = torch.load(checkpoint_path, map_location=device)
    _load_compatible_state_dict(encoder, checkpoint["encoder"], module_name="encoder")
    _load_compatible_state_dict(world_model, checkpoint["world_model"], module_name="world_model")
    _load_compatible_state_dict(language_model, checkpoint["language_model"], module_name="language_model")
    history = list(checkpoint.get("history", []))
    holdout_history = list(checkpoint.get("holdout_history", []))
    training_state = dict(checkpoint.get("training_state", {}))
    last_holdout_result = checkpoint.get("last_holdout_result") or None
    completed_epochs = int(training_state.get("completed_epochs", len(history)))
    return {
        "history": history,
        "holdout_history": holdout_history,
        "training_state": training_state,
        "last_holdout_result": last_holdout_result,
        "current_epoch": max(-1, completed_epochs - 1),
        "start_epoch": completed_epochs,
        "stage_index": int(training_state.get("stage_index", 0)),
        "stage_epoch_count": int(training_state.get("stage_epoch_count", 0)),
        "consecutive_holdout_passes": int(training_state.get("consecutive_holdout_passes", 0)),
        "cached_regression_holdout": _latest_regression_snapshot(holdout_history),
    }


def _checkpoint_contains_resume_metadata(checkpoint: dict[str, object]) -> bool:
    history = checkpoint.get("history", [])
    holdout_history = checkpoint.get("holdout_history", [])
    training_state = checkpoint.get("training_state", {})
    if history or holdout_history:
        return True
    if not isinstance(training_state, dict):
        return False
    completed_epochs = int(training_state.get("completed_epochs", 0) or 0)
    stage_epoch_count = int(training_state.get("stage_epoch_count", 0) or 0)
    return completed_epochs > 0 or stage_epoch_count > 0 or bool(training_state.get("stage_name"))


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


_EFFECT_KIND_TO_ID: dict[str, int] = {
    "reward_gain": 0,
    "setback": 1,
    "state_change": 2,
    "latent_shift": 3,
    "no_effect": 4,
}
_SAMPLE_STRONG_POSITIVE_EVENTS: frozenset[str] = frozenset(
    {
        "goal_reached",
        "correct_switch",
        "selector_unlock_complete",
        "correct_order_complete",
        "delayed_sequence_complete",
        "selector_sequence_complete",
        "correct_collect",
        "delayed_correct_collect",
        "selector_sequence_progress",
        "selector_candidate",
    }
)
_SAMPLE_DIAGNOSTIC_EVENTS: frozenset[str] = frozenset({"selector_probe", "local_match_no_unlock"})
_SAMPLE_NEGATIVE_EVENTS: frozenset[str] = frozenset(
    {
        "decoy_reward_reset",
        "false_progress_under_wrong_selector",
        "wrong_switch",
        "wrong_selector_or_switch",
        "wrong_order",
        "wrong_order_reset",
    }
)
_SAMPLE_NO_EFFECT_EVENTS: frozenset[str] = frozenset(
    {
        "empty_interaction",
        "noop",
        "wall",
        "blocked_by_object",
        "invalid_click",
        "unused_click",
        "missing_rule",
        "inactive_goal_blocked",
        "redundant_collect",
        "redundant_post_goal_interaction",
    }
)
_FAMILY_KEYWORD_TOKENS: tuple[str, ...] = ("collect", "unlock", "selector", "switch", "order", "delayed", "sequence")


def _effect_token(effect_kind: str) -> str:
    return {
        "reward_gain": "positive",
        "setback": "negative",
        "state_change": "visible",
        "latent_shift": "hidden",
        "no_effect": "none",
    }.get(effect_kind, "unknown")


def _sample_effect_kind(
    *,
    action: str,
    reward: float,
    event: str,
    delta_norm: float,
) -> str:
    family = _shared_action_family(action)
    event_name = str(event or "")
    if event_name in _SAMPLE_STRONG_POSITIVE_EVENTS or float(reward) > 0.05:
        return "reward_gain"
    if event_name in _SAMPLE_NEGATIVE_EVENTS or float(reward) < -0.04:
        return "setback"
    if event_name in _SAMPLE_DIAGNOSTIC_EVENTS:
        return "latent_shift"
    if event_name in _SAMPLE_NO_EFFECT_EVENTS:
        return "no_effect"
    if family in {"click", "interact", "select"} and float(delta_norm) >= 0.2:
        return "latent_shift"
    if float(delta_norm) >= 0.12:
        return "state_change"
    return "no_effect"


def _family_tokens_for_sample(action: str, event: str) -> tuple[str, ...]:
    context = build_action_schema_context((action,), {action: ""})
    schema = build_action_schema(action, context)
    joined = f"{schema.family} {event}".lower()
    tokens = [token for token in _FAMILY_KEYWORD_TOKENS if token in joined]
    return tuple(tokens[:2])


def _theory_tokens_for_sample(
    sample: dict[str, object],
) -> tuple[str, ...]:
    action = str(sample["action"])
    effect_kind = str(sample.get("effect_kind", "no_effect"))
    state = sample.get("state")
    action_roles = () if state is None else getattr(state, "action_roles", ())
    context = build_action_schema_context(tuple(sample.get("available_actions", (action,))), dict(action_roles))
    schema = build_action_schema(action, context)
    return (
        "rule",
        schema.action_type,
        _effect_token(effect_kind),
        *_family_tokens_for_sample(action, str(sample.get("event", ""))),
    )[:6]


def _causal_value_target_for_sample(sample: dict[str, object]) -> float:
    discounted_return = float(sample.get("discounted_return", sample.get("reward", 0.0)))
    future_progress = float(sample.get("future_progress", 0.0))
    future_setback = float(sample.get("future_setback", 0.0))
    usefulness = float(sample.get("usefulness", 0.0))
    outcome_signal = float(sample.get("outcome_signal", 0.0))
    return float(
        max(
            min(
                (0.45 * discounted_return)
                + (0.35 * future_progress)
                - (0.4 * future_setback)
                + (0.3 * usefulness)
                + (0.45 * outcome_signal),
                1.5,
            ),
            -1.5,
        )
    )


def _diagnostic_target_for_sample(sample: dict[str, object]) -> float:
    action = str(sample["action"])
    effect_kind = str(sample.get("effect_kind", "no_effect"))
    event_name = str(sample.get("event", ""))
    future_progress = max(float(sample.get("future_progress", 0.0)), 0.0)
    future_setback = max(float(sample.get("future_setback", 0.0)), 0.0)
    delta_norm = float(sample.get("delta_norm", 0.0))
    usefulness = abs(float(sample.get("usefulness", 0.0)))
    interactive_bonus = 0.18 if _shared_action_family(action) in {"click", "interact", "select"} else 0.0
    hidden_bonus = 0.22 if effect_kind == "latent_shift" else 0.0
    diagnostic_bonus = 0.2 if event_name in _SAMPLE_DIAGNOSTIC_EVENTS else 0.0
    outcome_bonus = 0.18 * abs(float(sample.get("outcome_signal", 0.0)))
    target = (
        diagnostic_bonus
        + interactive_bonus
        + hidden_bonus
        + (0.14 * min(delta_norm, 1.0))
        + (0.18 * usefulness)
        + (0.22 * future_progress)
        + (0.28 * future_setback)
    )
    return float(max(0.0, min(target + outcome_bonus, 1.25)))


def _diagnostic_tokens_for_sample(sample: dict[str, object]) -> tuple[str, ...]:
    action = str(sample["action"])
    state = sample.get("state")
    action_roles = () if state is None else getattr(state, "action_roles", ())
    context = build_action_schema_context(tuple(sample.get("available_actions", (action,))), dict(action_roles))
    schema = build_action_schema(action, context)
    diagnostic_target = float(sample.get("diagnostic_target", 0.0))
    effect_kind = str(sample.get("effect_kind", "no_effect"))
    if diagnostic_target >= 0.65:
        return ("question", "need", "test", "rule", schema.action_type, _effect_token(effect_kind))
    if diagnostic_target >= 0.3:
        return ("question", "confirm", "rule", schema.action_type, _effect_token(effect_kind))
    return ("plan", "commit", schema.action_type, "because", _effect_token(effect_kind))


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
    event_name = str(info.get("event", ""))
    usefulness = transition_usefulness_target(
        action,
        reward,
        event_name,
        delta_norm,
    )
    policy_supervision = transition_policy_supervision(
        action,
        reward,
        event_name,
        delta_norm,
    )
    plan_action = teacher_action or action
    sample = {
        "state": state,
        "next_state": next_state,
        "action": action,
        "teacher_action": teacher_action,
        "teacher_weight": float(teacher_weight),
        "available_actions": state.affordances,
        "reward": reward,
        "event": event_name,
        "delta": delta.astype(np.float32),
        "delta_norm": delta_norm,
        "usefulness": usefulness,
        "policy_target": policy_supervision.target,
        "policy_weight": policy_supervision.weight,
        "sibling_move_target": policy_supervision.sibling_move_target,
        "sibling_move_weight": policy_supervision.sibling_move_weight,
        "same_type_target": policy_supervision.same_type_target,
        "same_type_weight": policy_supervision.same_type_weight,
        "replay_weight": 1.0,
        "discounted_return": reward,
        "future_progress": 0.0,
        "future_setback": 0.0,
        "outcome_signal": 0.0,
        "belief_tokens": _grounded_belief_tokens(state),
        "question_tokens": _grounded_question_tokens(state),
        "plan_tokens": _grounded_plan_tokens(state, plan_action),
    }
    sample["effect_kind"] = _sample_effect_kind(
        action=action,
        reward=reward,
        event=event_name,
        delta_norm=delta_norm,
    )
    sample["effect_target"] = _EFFECT_KIND_TO_ID[str(sample["effect_kind"])]
    sample["theory_tokens"] = _theory_tokens_for_sample(sample)
    sample["causal_value_target"] = _causal_value_target_for_sample(sample)
    sample["diagnostic_target"] = _diagnostic_target_for_sample(sample)
    sample["diagnostic_tokens"] = _diagnostic_tokens_for_sample(sample)
    return sample


def _apply_episode_hindsight(
    config: SyntheticTrainingConfig,
    episode_samples: list[dict[str, object]],
) -> None:
    if not episode_samples:
        return
    discount = float(config.trajectory_credit_discount)
    trailing_return = 0.0
    trailing_progress = 0.0
    trailing_setback = 0.0
    for sample in reversed(episode_samples):
        reward = float(sample["reward"])
        base_usefulness = float(sample["usefulness"])
        base_policy = PolicySupervision(
            target=float(sample["policy_target"]),
            weight=float(sample["policy_weight"]),
            sibling_move_target=float(sample["sibling_move_target"]),
            sibling_move_weight=float(sample["sibling_move_weight"]),
            same_type_target=float(sample["same_type_target"]),
            same_type_weight=float(sample["same_type_weight"]),
        )
        discounted_return = reward + (discount * trailing_return)
        hindsight = hindsight_supervision(
            base_usefulness=base_usefulness,
            base_policy=base_policy,
            discounted_return=discounted_return,
            future_progress=trailing_progress,
            future_setback=trailing_setback,
            teacher_weight=float(sample.get("teacher_weight", 0.0)),
            teacher_disagrees=bool(sample.get("teacher_action")) and str(sample.get("teacher_action")) != str(sample.get("action")),
        )
        sample["usefulness"] = hindsight.usefulness
        sample["policy_target"] = hindsight.policy_target
        sample["policy_weight"] = hindsight.policy_weight
        sample["sibling_move_weight"] = hindsight.sibling_move_weight
        sample["same_type_weight"] = hindsight.same_type_weight
        sample["teacher_weight"] = hindsight.teacher_weight
        sample["replay_weight"] = hindsight.replay_weight
        sample["discounted_return"] = hindsight.discounted_return
        sample["future_progress"] = hindsight.future_progress
        sample["future_setback"] = hindsight.future_setback
        sample["outcome_signal"] = hindsight.outcome_signal
        sample["causal_value_target"] = _causal_value_target_for_sample(sample)
        sample["diagnostic_target"] = _diagnostic_target_for_sample(sample)
        sample["theory_tokens"] = _theory_tokens_for_sample(sample)
        sample["diagnostic_tokens"] = _diagnostic_tokens_for_sample(sample)
        trailing_return = discounted_return
        trailing_progress = max(0.0, base_usefulness) + (discount * trailing_progress)
        trailing_setback = max(0.0, -base_usefulness) + (discount * trailing_setback)


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
        weight += float(sample.get("replay_weight", 1.0)) - 1.0
        weight += 0.25 * float(sample.get("future_progress", 0.0))
        weight += 0.35 * float(sample.get("future_setback", 0.0))
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


def _dataset_episode_sequences(dataset: list[dict[str, object]]) -> list[list[dict[str, object]]]:
    sequences: list[list[dict[str, object]]] = []
    current_sequence: list[dict[str, object]] = []
    current_episode_id: str | None = None
    for sample in dataset:
        state = sample["state"]
        if current_sequence and state.episode_id != current_episode_id:
            sequences.append(current_sequence)
            current_sequence = []
        current_episode_id = state.episode_id
        current_sequence.append(sample)
    if current_sequence:
        sequences.append(current_sequence)
    return sequences


def _append_replay_metrics(metric_lists: dict[str, list[float]], metrics: dict[str, float]) -> None:
    for key, value in metrics.items():
        metric_lists[key].append(float(value))


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


def _teacher_ownership_signal(
    config: SyntheticTrainingConfig,
    history: list[dict[str, object]] | None = None,
) -> float:
    if not history:
        return 0.0
    window = max(1, int(config.teacher_ownership_window))
    recent = history[-window:]

    def _metric(name: str) -> list[float]:
        values: list[float] = []
        for item in recent:
            if name not in item:
                continue
            values.append(float(item[name]))
        return values

    collect_success_values = _metric("collect_success_rate")
    holdout_success_values = [
        float(item.get("holdout_frontier_success", 0.0))
        for item in recent
        if bool(item.get("holdout_evaluated", False))
    ]
    agreement_values = _metric("learner_teacher_agreement")
    teacher_step_values = _metric("teacher_step_fraction")
    teacher_relabel_values = _metric("teacher_relabel_fraction")
    if not agreement_values and not teacher_step_values and not teacher_relabel_values:
        return 0.0
    success = mean(holdout_success_values) if holdout_success_values else (mean(collect_success_values) if collect_success_values else 0.0)
    agreement = mean(agreement_values) if agreement_values else 0.0
    teacher_step_fraction = mean(teacher_step_values) if teacher_step_values else 1.0
    teacher_relabel_fraction = mean(teacher_relabel_values) if teacher_relabel_values else 1.0
    ownership = max(
        0.0,
        1.0 - max(teacher_step_fraction, 0.6 * teacher_relabel_fraction),
    )
    agreement_signal = max(
        0.0,
        min(1.0, (agreement - 0.5 * float(config.teacher_agreement_target)) / max(0.5 * float(config.teacher_agreement_target), 1e-6)),
    )
    success_signal = max(
        0.0,
        min(1.0, (success - 0.55 * float(config.teacher_success_target)) / max(0.45 * float(config.teacher_success_target), 1e-6)),
    )
    ownership_signal = max(
        0.0,
        min(1.0, (ownership - 0.45) / 0.4),
    )
    return float(
        max(
            0.0,
            min(
                1.0,
                (0.4 * success_signal) + (0.35 * ownership_signal) + (0.25 * agreement_signal),
            ),
        )
    )


def _teacher_episode_fraction(
    config: SyntheticTrainingConfig,
    history: list[dict[str, object]] | None = None,
) -> float:
    initial = float(max(0.0, min(1.0, config.teacher_episode_fraction_initial)))
    floor = float(max(0.0, min(1.0, config.teacher_episode_fraction_floor)))
    high = max(initial, floor)
    low = min(initial, floor)
    guidance_state = _teacher_guidance_state(config, history)
    alpha = max(float(guidance_state["alpha"]), _teacher_ownership_signal(config, history))
    return max(low, high + ((low - high) * alpha))


def _teacher_takeover_probability(
    config: SyntheticTrainingConfig,
    history: list[dict[str, object]] | None = None,
) -> float:
    initial = float(max(0.0, min(1.0, config.teacher_takeover_prob_initial)))
    floor = float(max(0.0, min(1.0, config.teacher_takeover_prob_floor)))
    high = max(initial, floor)
    low = min(initial, floor)
    guidance_state = _teacher_guidance_state(config, history)
    alpha = max(float(guidance_state["alpha"]), _teacher_ownership_signal(config, history))
    return max(low, high + ((low - high) * alpha))


def _dense_teacher_supervision_active(config: SyntheticTrainingConfig, epoch_index: int) -> bool:
    return (
        str(config.behavior_policy) in {"mixed", "bootstrap"}
        and int(epoch_index) < max(0, int(config.oracle_imitation_epochs))
    )


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
    if episode_count <= len(families):
        rng = np.random.default_rng(seed)
        sampled = list(families[:episode_count])
        rng.shuffle(sampled)
        return sampled
    weights = np.asarray([max(1, family_weights[family]) for family in families], dtype=np.float64)
    probabilities = weights / weights.sum()
    rng = np.random.default_rng(seed)
    base_schedule = list(families)
    remaining = episode_count - len(base_schedule)
    sampled_indices = rng.choice(len(families), size=remaining, replace=True, p=probabilities)
    schedule = base_schedule + [families[int(index)] for index in sampled_indices]
    rng.shuffle(schedule)
    return schedule


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


def _flat_family_weights(
    config: SyntheticTrainingConfig,
    history: list[dict[str, object]] | None = None,
) -> dict[str, int]:
    families = tuple(dict.fromkeys(str(family) for family in config.family_modes))
    floor = max(1, int(config.flat_family_weight_floor))
    ceiling = max(floor, int(config.flat_family_weight_ceiling))
    weights = {family: floor for family in families}
    if not history:
        return weights
    window = max(1, int(config.flat_family_history_window))
    recent_entries = [entry for entry in history[-window:] if isinstance(entry.get("family_breakdown"), dict)]
    if not recent_entries:
        return weights

    aggregate: dict[str, dict[str, float]] = {}
    for entry in recent_entries:
        family_breakdown = dict(entry.get("family_breakdown", {}))
        for family in families:
            metrics = family_breakdown.get(family)
            if not isinstance(metrics, dict):
                continue
            episodes = float(metrics.get("episodes", 0.0))
            if episodes <= 0.0:
                continue
            bucket = aggregate.setdefault(
                family,
                {
                    "episodes": 0.0,
                    "success_sum": 0.0,
                    "return_sum": 0.0,
                    "steps_sum": 0.0,
                    "interaction_sum": 0.0,
                },
            )
            bucket["episodes"] += episodes
            bucket["success_sum"] += episodes * float(metrics.get("success_rate", 0.0))
            bucket["return_sum"] += episodes * float(metrics.get("avg_return", 0.0))
            bucket["steps_sum"] += episodes * float(metrics.get("avg_steps", 0.0))
            bucket["interaction_sum"] += episodes * float(metrics.get("avg_interactions", 0.0))
    if not aggregate:
        return weights

    normalized: dict[str, dict[str, float]] = {}
    for family, bucket in aggregate.items():
        episodes = max(bucket["episodes"], 1.0)
        normalized[family] = {
            "success_rate": bucket["success_sum"] / episodes,
            "avg_return": bucket["return_sum"] / episodes,
            "avg_steps": bucket["steps_sum"] / episodes,
            "avg_interactions": bucket["interaction_sum"] / episodes,
        }

    best_success = max(metrics["success_rate"] for metrics in normalized.values())
    best_return = max(metrics["avg_return"] for metrics in normalized.values())
    min_steps = min(metrics["avg_steps"] for metrics in normalized.values())
    min_interactions = min(metrics["avg_interactions"] for metrics in normalized.values())
    focus_strength = max(0.0, float(config.flat_family_focus_strength))
    max_steps = max(float(config.max_steps), 1.0)

    for family, metrics in normalized.items():
        success_gap = max(0.0, best_success - metrics["success_rate"])
        return_gap = max(0.0, best_return - metrics["avg_return"])
        step_gap = max(0.0, metrics["avg_steps"] - min_steps) / max_steps
        interaction_gap = max(0.0, metrics["avg_interactions"] - min_interactions) / max_steps
        difficulty = success_gap + (0.35 * return_gap) + (0.2 * step_gap) + (0.15 * interaction_gap)
        dynamic_bonus = int(round(focus_strength * difficulty))
        weights[family] = max(floor, min(ceiling, floor + dynamic_bonus))
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


def _progress_bucket(value: int) -> str:
    return f"p{max(0, min(int(value), 5))}"


def _language_status_token(
    state: StructuredState,
    *,
    contradiction_count: int,
    recent_progress: bool,
) -> str:
    if _state_has_active_target(state):
        return "active"
    if contradiction_count > 0:
        return "uncertain"
    if recent_progress:
        return "positive"
    if _state_has_target(state):
        return "inactive"
    return "unknown"


def _mode_token(raw_mode: str) -> str:
    normalized = str(raw_mode).strip().lower()
    if normalized in {"explore", "probe", "commit", "confirm"}:
        return normalized
    return "unknown"


def _action_token(action_type: str) -> str:
    if action_type in {"move", "interact", "click", "select", "wait"}:
        return action_type
    return "unknown"


def _focus_token(
    state: StructuredState,
    *,
    action_type: str,
    contradiction_count: int,
) -> str:
    if _state_has_active_target(state):
        return "target"
    if contradiction_count > 0:
        return "contradiction"
    if action_type in {"click", "select"} and _has_selector_affordance(state):
        return "rule"
    if action_type == "interact":
        return "interactable"
    if action_type == "move" and _state_has_target(state):
        return "target"
    if _nearest_interactable_tokens(state):
        return "interactable"
    return "frontier"


def _color_clause(state: StructuredState) -> tuple[str, ...]:
    nearest = _nearest_interactable_tokens(state)
    if nearest:
        return ("color", nearest[0])
    return ()


def _grounded_belief_tokens(
    state: StructuredState,
) -> tuple[str, ...]:
    has_selector = _has_selector_affordance(state)
    progress_level = _state_symbolic_int(state, "belief_progress_level")
    contradiction_count = _state_symbolic_int(state, "belief_contradiction_count")
    flags = state.flags_dict()
    mode = _mode_token(flags.get("belief_mode", "explore"))
    recent_progress = flags.get("belief_recent_progress", "0") == "1"
    action_type = "select" if has_selector else "interact" if _nearest_interactable_tokens(state) else "move"
    status = _language_status_token(
        state,
        contradiction_count=contradiction_count,
        recent_progress=recent_progress,
    )
    focus = _focus_token(
        state,
        action_type=action_type,
        contradiction_count=contradiction_count,
    )
    return (
        "belief",
        "goal",
        status,
        "focus",
        focus,
        "state",
        mode,
        *_color_clause(state),
        _progress_bucket(progress_level),
    )


def _grounded_question_tokens(
    state: StructuredState,
) -> tuple[str, ...]:
    has_selector = _has_selector_affordance(state)
    has_interactable = any("interactable" in obj.tags for obj in state.objects)
    progress_level = _state_symbolic_int(state, "belief_progress_level")
    contradiction_count = _state_symbolic_int(state, "belief_contradiction_count")
    flags = state.flags_dict()
    mode = _mode_token(flags.get("belief_mode", "explore"))
    recent_progress = flags.get("belief_recent_progress", "0") == "1"
    last_effect_family = flags.get("belief_last_effect_family", "none")

    if _state_has_active_target(state) or (progress_level >= 2 and mode == "commit"):
        intent = "move"
        focus = "target"
    elif has_selector and (mode in {"explore", "probe"} or contradiction_count > 0):
        intent = "test"
        focus = "rule"
    elif recent_progress and last_effect_family in {"interact", "click"}:
        intent = "confirm"
        focus = "effect"
    elif has_interactable:
        intent = "test"
        focus = "interactable"
    else:
        intent = "explore"
        focus = "frontier"
    return (
        "question",
        "need",
        intent,
        "focus",
        focus,
        "state",
        mode,
        *_color_clause(state),
        _progress_bucket(progress_level),
    )


def _grounded_plan_tokens(
    state: StructuredState,
    action: str,
) -> tuple[str, ...]:
    context = build_action_schema_context(state.affordances, dict(state.action_roles))
    schema = build_action_schema(action, context)
    progress_level = _state_symbolic_int(state, "belief_progress_level")
    contradiction_count = _state_symbolic_int(state, "belief_contradiction_count")
    flags = state.flags_dict()
    mode = _mode_token(flags.get("belief_mode", "explore"))
    recent_progress = flags.get("belief_recent_progress", "0") == "1"
    focus = _focus_token(
        state,
        action_type=schema.action_type,
        contradiction_count=contradiction_count,
    )
    status = _language_status_token(
        state,
        contradiction_count=contradiction_count,
        recent_progress=recent_progress,
    )
    direction = schema.direction or "none"
    return (
        "plan",
        "action",
        _action_token(schema.action_type),
        "direction",
        direction,
        "focus",
        focus,
        *_color_clause(state),
        "state",
        status if status != "positive" else mode,
        _progress_bucket(progress_level),
    )


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
    completed_epochs = int(training_state.get("completed_epochs", len(history)))
    active_epoch = int(training_state.get("active_epoch", completed_epochs - 1 if completed_epochs > 0 else -1))
    return {
        "config": asdict(config),
        "encoder": encoder.state_dict(),
        "world_model": world_model.state_dict(),
        "language_model": language_model.state_dict(),
        "history": history,
        "holdout_history": holdout_history,
        "training_state": training_state,
        "last_holdout_result": last_holdout_result or {},
        "completed_epochs": completed_epochs,
        "current_epoch": active_epoch,
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
    encoder: StructuredStateEncoder | None = None,
    world_model: RecurrentWorldModel | None = None,
    language_model: GroundedLanguageModel | None = None,
    device: torch.device | None = None,
) -> tuple[list[dict[str, object]], dict[str, object]]:
    _validate_synthetic_training_config(config)
    collector, collector_agent_name, reset_collection_agent_all = _build_collection_agent(
        config,
        encoder=encoder,
        world_model=world_model,
        language_model=language_model,
        device=device,
    )
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
        if config.curriculum == "flat":
            family_weights = _flat_family_weights(config, history)
            family_mode_schedule = _weighted_family_schedule(
                family_weights,
                config.episodes_per_epoch,
                seed=epoch_seed_base,
            )
        else:
            family_weights = {family: 1 for family in family_modes}
            family_mode_schedule = [
                family_modes[episode_index % len(family_modes)]
                for episode_index in range(config.episodes_per_epoch)
            ]
        size_options = _size_options_for_epoch(config, epoch_index)
        stage_name = "fixed_staged" if config.curriculum == "fixed_staged" else "flat"
        frontier_families = tuple(dict.fromkeys(family_modes))
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
    dense_teacher_supervision = _dense_teacher_supervision_active(config, epoch_index)
    teacher_episode_fraction = scheduled_teacher_episode_fraction
    guidance_rng = np.random.default_rng(epoch_seed_base + 17_941)
    teacher_episode_count = 0
    teacher_controlled_steps = 0
    teacher_labeled_steps = 0
    teacher_agreement_steps = 0
    teacher_opportunity_steps = 0
    total_steps = 0
    collect_started = perf_counter()
    for epoch_episode in range(config.episodes_per_epoch):
        family_mode = family_mode_schedule[epoch_episode]
        variants = family_variants_for_mode(family_mode)
        family_occurrence = family_variant_counts.get(family_mode, 0)
        variant = variants[(family_variant_offsets[family_mode] + family_occurrence) % len(variants)]
        family_variant_counts[family_mode] = family_occurrence + 1
        size = size_options[(size_offset + epoch_episode) % len(size_options)]
        episode_seed = seed_cursor
        env = HiddenRuleEnv(
            size=size,
            family_mode=family_mode,
            family_variant=variant,
            max_steps=config.max_steps,
            seed=episode_seed,
        )
        seed_cursor += 1
        observation = env.reset(seed=episode_seed)
        if reset_collection_agent_all:
            collector.reset_all()
        else:
            collector.reset_episode()
        done = False
        episode_return = 0.0
        episode_step_count = 0
        episode_success = False
        episode_interaction_count = 0
        episode_samples: list[dict[str, object]] = []
        sampled_teacher_episode = (
            config.behavior_policy in {"mixed", "bootstrap"}
            and not dense_teacher_supervision
            and bool(guidance_rng.random() < teacher_episode_fraction)
        )
        teacher_episode = sampled_teacher_episode or config.behavior_policy == "oracle"
        if teacher_episode:
            teacher_episode_count += 1
        while not done:
            state: StructuredState
            bootstrap_takeover_candidate = (
                config.behavior_policy in {"mixed", "bootstrap"}
                and not teacher_episode
                and env._step < bootstrap_steps
                and ((seed_cursor + env._step) % bootstrap_stride == 0)
                and bool(guidance_rng.random() < teacher_takeover_prob)
            )
            teacher_action = ""
            prefetched_state: StructuredState | None = None
            if config.behavior_policy in {"mixed", "bootstrap", "oracle"}:
                prefetched_state = collector.observe(observation)
                teacher_action = oracle_action(env)
                collector.last_state = prefetched_state
            if teacher_episode:
                state = prefetched_state if prefetched_state is not None else collector.observe(observation)
                action = teacher_action
                collector.last_state = state
                collector.last_action = action
                teacher_controlled_steps += 1
            else:
                learner_action = collector.act(observation)
                state = collector.last_state
                assert state is not None
                if teacher_action:
                    teacher_opportunity_steps += 1
                    if learner_action == teacher_action:
                        teacher_agreement_steps += 1
                use_bootstrap_oracle = (
                    bootstrap_takeover_candidate
                    and teacher_action
                    and learner_action != teacher_action
                )
                if use_bootstrap_oracle:
                    action = teacher_action
                    collector.last_state = state
                    collector.last_action = action
                    teacher_controlled_steps += 1
                else:
                    action = learner_action
            teacher_label_action = ""
            teacher_label_weight = 0.0
            if teacher_action and config.behavior_policy in {"mixed", "bootstrap"}:
                should_attach_teacher_label = (
                    dense_teacher_supervision
                    and not teacher_episode
                    and action != teacher_action
                ) or (
                    not dense_teacher_supervision
                    and action != teacher_action
                )
                if should_attach_teacher_label:
                    teacher_label_action = teacher_action
                    teacher_label_weight = float(max(config.teacher_relabel_weight, 1.0 if dense_teacher_supervision else 0.0))
                    teacher_labeled_steps += 1
            result = env.step(action)
            next_state = collector.update_after_step(
                next_observation=result.observation,
                reward=result.reward,
                terminated=result.terminated or result.truncated,
                info=result.info,
            )
            episode_samples.append(
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
        _apply_episode_hindsight(config, episode_samples)
        dataset.extend(episode_samples)
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
                        "family_sampling_weights": family_weights,
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
                        "dense_teacher_supervision": dense_teacher_supervision,
                        "teacher_episode_fraction": teacher_episode_fraction,
                        "teacher_takeover_prob": teacher_takeover_prob,
                        "teacher_episode_count": teacher_episode_count,
                        "teacher_step_fraction": (float(teacher_controlled_steps) / float(max(total_steps, 1))),
                        "teacher_relabel_fraction": (float(teacher_labeled_steps) / float(max(total_steps, 1))),
                        "learner_teacher_agreement": (
                            float(teacher_agreement_steps) / float(max(teacher_opportunity_steps, 1))
                        ),
                        "collector_agent": collector_agent_name,
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
        "family_sampling_weights": dict(family_weights),
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
        "dense_teacher_supervision": bool(dense_teacher_supervision),
        "teacher_episode_fraction": float(teacher_episode_fraction),
        "teacher_takeover_prob": float(teacher_takeover_prob),
        "teacher_episode_count": float(teacher_episode_count),
        "teacher_step_fraction": float(teacher_controlled_steps) / float(max(total_steps, 1)),
        "teacher_relabel_fraction": float(teacher_labeled_steps) / float(max(total_steps, 1)),
        "learner_teacher_agreement": float(teacher_agreement_steps) / float(max(teacher_opportunity_steps, 1)),
        "collector_agent": collector_agent_name,
    }


def _prepare_replay_training_step(
    config: SyntheticTrainingConfig,
    sample: dict[str, object],
    *,
    encoder: StructuredStateEncoder,
    world_model: RecurrentWorldModel,
    language_model: GroundedLanguageModel,
    sequence_hidden: torch.Tensor | None,
    device: torch.device,
) -> tuple[torch.Tensor, dict[str, float], torch.Tensor, StructuredState, str]:
    state = sample["state"]
    next_state = sample["next_state"]
    action = sample["action"]
    available_actions = sample["available_actions"]
    reward = float(sample["reward"])
    delta = sample["delta"]
    usefulness = float(sample["usefulness"])
    teacher_action = str(sample.get("teacher_action", ""))
    teacher_weight = float(sample.get("teacher_weight", 0.0))
    replay_weight = float(sample.get("replay_weight", 1.0))
    policy_target = float(sample["policy_target"])
    policy_weight = float(sample["policy_weight"])
    sibling_move_target = float(sample["sibling_move_target"])
    sibling_move_weight = float(sample["sibling_move_weight"])
    same_type_target = float(sample["same_type_target"])
    same_type_weight = float(sample["same_type_weight"])
    belief_tokens = sample["belief_tokens"]
    question_tokens = sample["question_tokens"]
    plan_tokens = sample["plan_tokens"]
    theory_tokens = sample.get("theory_tokens", ())
    diagnostic_tokens = sample.get("diagnostic_tokens", ())
    effect_target_id = int(sample.get("effect_target", _EFFECT_KIND_TO_ID["no_effect"]))
    causal_value_target_scalar = float(sample.get("causal_value_target", sample.get("outcome_signal", 0.0)))
    diagnostic_target_scalar = float(sample.get("diagnostic_target", 0.0))

    encoded = encoder.encode_state(state, device=device)
    with torch.no_grad():
        next_encoded = encoder.encode_state(next_state, device=device)
    reward_target = torch.tensor([reward], dtype=torch.float32, device=device)
    return_target = torch.tensor([float(sample.get("discounted_return", reward))], dtype=torch.float32, device=device)
    delta_target = torch.tensor(delta, dtype=torch.float32, device=device).unsqueeze(0)
    usefulness_target = torch.tensor([usefulness], dtype=torch.float32, device=device)
    effect_target = torch.tensor([effect_target_id], dtype=torch.long, device=device)
    causal_value_target = torch.tensor([causal_value_target_scalar], dtype=torch.float32, device=device)
    diagnostic_target = torch.tensor([diagnostic_target_scalar], dtype=torch.float32, device=device)
    world_loss, world_metrics = world_model.loss(
        latent=encoded.latent,
        actions=[action],
        state=state,
        hidden=sequence_hidden,
        next_latent_target=next_encoded.latent.detach(),
        reward_target=reward_target,
        return_target=return_target,
        delta_target=delta_target,
        usefulness_target=usefulness_target,
        effect_target=effect_target,
        causal_target=causal_value_target,
        diagnostic_target=diagnostic_target,
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
    plan_loss = language_model.teacher_forcing_loss(
        encoded.latent,
        [plan_tokens or ("plan", "action", "unknown")],
        mode="plan",
    )
    theory_loss = language_model.teacher_forcing_loss(
        encoded.latent,
        [theory_tokens or ("rule", "unknown")],
        mode="theory",
    )
    diagnostic_language_loss = language_model.teacher_forcing_loss(
        encoded.latent,
        [diagnostic_tokens or ("question", "need", "test", "rule")],
        mode="diagnostic",
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
    sample_scale = 0.75 + (0.25 * replay_weight)
    weighted_core_loss = sample_scale * (
        world_loss
        + 0.25 * belief_loss
        + 0.25 * question_loss
        + 0.25 * plan_loss
        + (float(config.theory_loss_weight) * theory_loss)
        + (float(config.diagnostic_language_loss_weight) * diagnostic_language_loss)
        + 0.5 * policy_loss
    )
    loss = weighted_core_loss + gate_loss
    replay_metrics = {
        "epoch_losses": float(loss.detach().cpu()),
        "epoch_uncertainty": float(world_metrics["uncertainty"]),
        "epoch_world_total_losses": float(world_metrics["loss_total"]),
        "epoch_latent_losses": float(world_metrics["loss_latent"]),
        "epoch_reward_losses": float(world_metrics["loss_reward"]),
        "epoch_delta_losses": float(world_metrics["loss_delta"]),
        "epoch_usefulness_losses": float(world_metrics["loss_usefulness"]),
        "epoch_causal_losses": float(world_metrics["loss_causal"]),
        "epoch_diagnostic_world_losses": float(world_metrics["loss_diagnostic"]),
        "epoch_effect_losses": float(world_metrics["loss_effect"]),
        "epoch_belief_losses": float(belief_loss.detach().cpu()),
        "epoch_question_losses": float(question_loss.detach().cpu()),
        "epoch_plan_losses": float(plan_loss.detach().cpu()),
        "epoch_theory_losses": float(theory_loss.detach().cpu()),
        "epoch_diagnostic_language_losses": float(diagnostic_language_loss.detach().cpu()),
        "epoch_policy_losses": float(policy_loss.detach().cpu()),
        "epoch_positive_policy_losses": float(positive_policy_loss.detach().cpu()),
        "epoch_negative_policy_losses": float(negative_policy_loss.detach().cpu()),
        "epoch_gate_losses": float(gate_loss.detach().cpu()),
    }
    return loss, replay_metrics, encoded.latent.detach(), state, action


def _advance_sequence_hidden(
    world_model: RecurrentWorldModel,
    *,
    latent: torch.Tensor,
    action: str,
    state: StructuredState,
    sequence_hidden: torch.Tensor | None,
) -> torch.Tensor:
    with torch.no_grad():
        return world_model.step(
            latent,
            actions=[action],
            state=state,
            hidden=sequence_hidden,
        ).hidden.detach()


def _run_multidevice_replay_phase(
    config: SyntheticTrainingConfig,
    *,
    dataset: list[dict[str, object]],
    encoder: StructuredStateEncoder,
    world_model: RecurrentWorldModel,
    language_model: GroundedLanguageModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    secondary_runtime: _SecondaryTrainingRuntime,
    metric_lists: dict[str, list[float]],
) -> dict[str, float]:
    episode_sequences = _dataset_episode_sequences(dataset)
    primary_parameters = _trainable_parameters(encoder, world_model, language_model)
    _sync_secondary_training_runtime(secondary_runtime, encoder, world_model, language_model)
    secondary_replay_samples = 0
    train_started = perf_counter()
    for episode_index in range(0, len(episode_sequences), 2):
        primary_episode = episode_sequences[episode_index]
        secondary_episode = episode_sequences[episode_index + 1] if episode_index + 1 < len(episode_sequences) else []
        primary_hidden: torch.Tensor | None = None
        secondary_hidden: torch.Tensor | None = None
        max_steps = max(len(primary_episode), len(secondary_episode))
        for step_index in range(max_steps):
            optimizer.zero_grad(set_to_none=True)
            _zero_module_grads(
                secondary_runtime.encoder,
                secondary_runtime.world_model,
                secondary_runtime.language_model,
            )
            contributors = 0
            primary_prepared: tuple[torch.Tensor, dict[str, float], torch.Tensor, StructuredState, str] | None = None
            secondary_prepared: tuple[torch.Tensor, dict[str, float], torch.Tensor, StructuredState, str] | None = None
            if step_index < len(primary_episode):
                primary_prepared = _prepare_replay_training_step(
                    config,
                    primary_episode[step_index],
                    encoder=encoder,
                    world_model=world_model,
                    language_model=language_model,
                    sequence_hidden=primary_hidden,
                    device=device,
                )
                primary_prepared[0].backward()
                _append_replay_metrics(metric_lists, primary_prepared[1])
                contributors += 1
            if step_index < len(secondary_episode):
                secondary_prepared = _prepare_replay_training_step(
                    config,
                    secondary_episode[step_index],
                    encoder=secondary_runtime.encoder,
                    world_model=secondary_runtime.world_model,
                    language_model=secondary_runtime.language_model,
                    sequence_hidden=secondary_hidden,
                    device=secondary_runtime.device,
                )
                secondary_prepared[0].backward()
                _append_replay_metrics(metric_lists, secondary_prepared[1])
                contributors += 1
                secondary_replay_samples += 1
            _merge_secondary_gradients(
                encoder,
                world_model,
                language_model,
                secondary_runtime,
                contributors=contributors,
            )
            torch.nn.utils.clip_grad_norm_(primary_parameters, max_norm=1.0)
            optimizer.step()
            _sync_secondary_training_runtime(secondary_runtime, encoder, world_model, language_model)
            if primary_prepared is not None:
                primary_hidden = _advance_sequence_hidden(
                    world_model,
                    latent=primary_prepared[2],
                    action=primary_prepared[4],
                    state=primary_prepared[3],
                    sequence_hidden=primary_hidden,
                )
            if secondary_prepared is not None:
                secondary_hidden = _advance_sequence_hidden(
                    secondary_runtime.world_model,
                    latent=secondary_prepared[2],
                    action=secondary_prepared[4],
                    state=secondary_prepared[3],
                    sequence_hidden=secondary_hidden,
                )
    return {
        "train_seconds": perf_counter() - train_started,
        "secondary_replay_samples": float(secondary_replay_samples),
        "secondary_device": str(secondary_runtime.device),
    }


def _prepare_dream_window(
    config: SyntheticTrainingConfig,
    *,
    dataset: list[dict[str, object]],
    window: list[int],
    encoder: StructuredStateEncoder,
    world_model: RecurrentWorldModel,
    language_model: GroundedLanguageModel,
    device: torch.device,
) -> tuple[torch.Tensor | None, dict[str, float]]:
    anchor_sample = dataset[window[0]]
    anchor_state = anchor_sample["state"]
    latent = encoder.encode_state(anchor_state, device=device).latent
    hidden: torch.Tensor | None = None
    sequence_terms: list[torch.Tensor] = []
    sequence_world_terms: list[torch.Tensor] = []
    sequence_belief_terms: list[torch.Tensor] = []
    sequence_question_terms: list[torch.Tensor] = []
    sequence_plan_terms: list[torch.Tensor] = []
    sequence_theory_terms: list[torch.Tensor] = []
    sequence_diagnostic_language_terms: list[torch.Tensor] = []
    total_steps = 0
    for position, index in enumerate(window):
        sample = dataset[index]
        state = sample["state"]
        next_state = sample["next_state"]
        action = sample["action"]
        with torch.no_grad():
            next_encoded = encoder.encode_state(next_state, device=device)
        reward_target = torch.tensor([float(sample["reward"])], dtype=torch.float32, device=device)
        return_target = torch.tensor([float(sample.get("discounted_return", sample["reward"]))], dtype=torch.float32, device=device)
        delta_target = torch.tensor(sample["delta"], dtype=torch.float32, device=device).unsqueeze(0)
        usefulness_target = torch.tensor([float(sample["usefulness"])], dtype=torch.float32, device=device)
        effect_target = torch.tensor([int(sample.get("effect_target", _EFFECT_KIND_TO_ID["no_effect"]))], dtype=torch.long, device=device)
        causal_target = torch.tensor([float(sample.get("causal_value_target", sample.get("outcome_signal", 0.0)))], dtype=torch.float32, device=device)
        diagnostic_target = torch.tensor([float(sample.get("diagnostic_target", 0.0))], dtype=torch.float32, device=device)
        world_loss, _metrics = world_model.loss(
            latent=latent,
            actions=[action],
            state=state,
            hidden=hidden,
            next_latent_target=next_encoded.latent.detach(),
            reward_target=reward_target,
            return_target=return_target,
            delta_target=delta_target,
            usefulness_target=usefulness_target,
            effect_target=effect_target,
            causal_target=causal_target,
            diagnostic_target=diagnostic_target,
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
        if position + 1 < len(window):
            next_plan_tokens = tuple(dataset[window[position + 1]].get("plan_tokens", ()))
        else:
            next_plan_tokens = _grounded_plan_tokens(
                next_state,
                str(sample.get("teacher_action") or sample["action"]),
            )
        plan_loss = language_model.teacher_forcing_loss(
            prediction.next_latent_mean,
            [next_plan_tokens or ("plan", "action", "unknown")],
            mode="plan",
        )
        theory_loss = language_model.teacher_forcing_loss(
            prediction.next_latent_mean,
            [tuple(sample.get("theory_tokens", ())) or ("rule", "unknown")],
            mode="theory",
        )
        diagnostic_language_loss = language_model.teacher_forcing_loss(
            prediction.next_latent_mean,
            [tuple(sample.get("diagnostic_tokens", ())) or ("question", "need", "test", "rule")],
            mode="diagnostic",
        )
        sequence_world_terms.append(world_loss)
        sequence_belief_terms.append(belief_loss)
        sequence_question_terms.append(question_loss)
        sequence_plan_terms.append(plan_loss)
        sequence_theory_terms.append(theory_loss)
        sequence_diagnostic_language_terms.append(diagnostic_language_loss)
        sequence_terms.append(
            world_loss
            + (float(config.dream_belief_weight) * belief_loss)
            + (float(config.dream_question_weight) * question_loss)
            + (float(config.dream_plan_weight) * plan_loss)
            + (float(config.theory_loss_weight) * theory_loss)
            + (float(config.diagnostic_language_loss_weight) * diagnostic_language_loss)
        )
        hidden = prediction.hidden
        latent = prediction.next_latent_mean
        total_steps += 1
    if not sequence_terms:
        return None, {
            "dream_sequences": 0.0,
            "dream_steps": 0.0,
            "dream_loss": 0.0,
            "dream_world_loss": 0.0,
            "dream_belief_loss": 0.0,
            "dream_question_loss": 0.0,
            "dream_plan_loss": 0.0,
            "dream_theory_loss": 0.0,
            "dream_diagnostic_language_loss": 0.0,
        }
    dream_loss = float(config.dream_loss_weight) * torch.stack(sequence_terms).mean()
    return dream_loss, {
        "dream_sequences": 1.0,
        "dream_steps": float(total_steps),
        "dream_loss": float(dream_loss.detach().cpu()),
        "dream_world_loss": float(torch.stack(sequence_world_terms).mean().detach().cpu()),
        "dream_belief_loss": float(torch.stack(sequence_belief_terms).mean().detach().cpu()),
        "dream_question_loss": float(torch.stack(sequence_question_terms).mean().detach().cpu()),
        "dream_plan_loss": float(torch.stack(sequence_plan_terms).mean().detach().cpu()),
        "dream_theory_loss": float(torch.stack(sequence_theory_terms).mean().detach().cpu()),
        "dream_diagnostic_language_loss": float(torch.stack(sequence_diagnostic_language_terms).mean().detach().cpu()),
    }


def _run_multidevice_dream_phase(
    config: SyntheticTrainingConfig,
    *,
    dataset: list[dict[str, object]],
    epoch_index: int,
    encoder: StructuredStateEncoder,
    world_model: RecurrentWorldModel,
    language_model: GroundedLanguageModel,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    secondary_runtime: _SecondaryTrainingRuntime,
) -> dict[str, float]:
    if config.dream_batches_per_epoch <= 0 or config.dream_batch_size <= 0 or config.dream_horizon <= 1:
        return {
            "dream_sequences": 0.0,
            "dream_steps": 0.0,
            "dream_loss": 0.0,
            "dream_world_loss": 0.0,
            "dream_belief_loss": 0.0,
            "dream_question_loss": 0.0,
            "dream_plan_loss": 0.0,
            "dream_theory_loss": 0.0,
            "dream_diagnostic_language_loss": 0.0,
            "dream_seconds": 0.0,
            "secondary_dream_sequences": 0.0,
            "secondary_device": str(secondary_runtime.device),
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
            "dream_plan_loss": 0.0,
            "dream_theory_loss": 0.0,
            "dream_diagnostic_language_loss": 0.0,
            "dream_seconds": 0.0,
            "secondary_dream_sequences": 0.0,
            "secondary_device": str(secondary_runtime.device),
        }
    encoder.train()
    world_model.train()
    language_model.train()
    secondary_runtime.encoder.train()
    secondary_runtime.world_model.train()
    secondary_runtime.language_model.train()
    _sync_secondary_training_runtime(secondary_runtime, encoder, world_model, language_model)
    primary_parameters = _trainable_parameters(encoder, world_model, language_model)
    dream_started = perf_counter()
    losses: list[float] = []
    world_losses: list[float] = []
    belief_losses: list[float] = []
    question_losses: list[float] = []
    plan_losses: list[float] = []
    theory_losses: list[float] = []
    diagnostic_language_losses: list[float] = []
    total_steps = 0
    secondary_dream_sequences = 0
    for window_index in range(0, len(windows), 2):
        primary_window = windows[window_index]
        secondary_window = windows[window_index + 1] if window_index + 1 < len(windows) else None
        optimizer.zero_grad(set_to_none=True)
        _zero_module_grads(
            secondary_runtime.encoder,
            secondary_runtime.world_model,
            secondary_runtime.language_model,
        )
        contributors = 0
        primary_prepared = _prepare_dream_window(
            config,
            dataset=dataset,
            window=primary_window,
            encoder=encoder,
            world_model=world_model,
            language_model=language_model,
            device=device,
        )
        if primary_prepared[0] is not None:
            primary_prepared[0].backward()
            contributors += 1
            losses.append(float(primary_prepared[1]["dream_loss"]))
            world_losses.append(float(primary_prepared[1]["dream_world_loss"]))
            belief_losses.append(float(primary_prepared[1]["dream_belief_loss"]))
            question_losses.append(float(primary_prepared[1]["dream_question_loss"]))
            plan_losses.append(float(primary_prepared[1]["dream_plan_loss"]))
            theory_losses.append(float(primary_prepared[1]["dream_theory_loss"]))
            diagnostic_language_losses.append(float(primary_prepared[1]["dream_diagnostic_language_loss"]))
            total_steps += int(primary_prepared[1]["dream_steps"])
        secondary_prepared: tuple[torch.Tensor | None, dict[str, float]] | None = None
        if secondary_window is not None:
            secondary_prepared = _prepare_dream_window(
                config,
                dataset=dataset,
                window=secondary_window,
                encoder=secondary_runtime.encoder,
                world_model=secondary_runtime.world_model,
                language_model=secondary_runtime.language_model,
                device=secondary_runtime.device,
            )
            if secondary_prepared[0] is not None:
                secondary_prepared[0].backward()
                contributors += 1
                secondary_dream_sequences += 1
                losses.append(float(secondary_prepared[1]["dream_loss"]))
                world_losses.append(float(secondary_prepared[1]["dream_world_loss"]))
                belief_losses.append(float(secondary_prepared[1]["dream_belief_loss"]))
                question_losses.append(float(secondary_prepared[1]["dream_question_loss"]))
                plan_losses.append(float(secondary_prepared[1]["dream_plan_loss"]))
                theory_losses.append(float(secondary_prepared[1]["dream_theory_loss"]))
                diagnostic_language_losses.append(float(secondary_prepared[1]["dream_diagnostic_language_loss"]))
                total_steps += int(secondary_prepared[1]["dream_steps"])
        _merge_secondary_gradients(
            encoder,
            world_model,
            language_model,
            secondary_runtime,
            contributors=contributors,
        )
        torch.nn.utils.clip_grad_norm_(primary_parameters, max_norm=1.0)
        optimizer.step()
        _sync_secondary_training_runtime(secondary_runtime, encoder, world_model, language_model)
    return {
        "dream_sequences": float(len(losses)),
        "dream_steps": float(total_steps),
        "dream_loss": mean(losses) if losses else 0.0,
        "dream_world_loss": mean(world_losses) if world_losses else 0.0,
        "dream_belief_loss": mean(belief_losses) if belief_losses else 0.0,
        "dream_question_loss": mean(question_losses) if question_losses else 0.0,
        "dream_plan_loss": mean(plan_losses) if plan_losses else 0.0,
        "dream_theory_loss": mean(theory_losses) if theory_losses else 0.0,
        "dream_diagnostic_language_loss": mean(diagnostic_language_losses) if diagnostic_language_losses else 0.0,
        "dream_seconds": perf_counter() - dream_started,
        "secondary_dream_sequences": float(secondary_dream_sequences),
        "secondary_device": str(secondary_runtime.device),
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
            "dream_plan_loss": 0.0,
            "dream_theory_loss": 0.0,
            "dream_diagnostic_language_loss": 0.0,
            "dream_seconds": 0.0,
            "secondary_dream_sequences": 0.0,
            "secondary_device": "",
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
            "dream_plan_loss": 0.0,
            "dream_theory_loss": 0.0,
            "dream_diagnostic_language_loss": 0.0,
            "dream_seconds": 0.0,
            "secondary_dream_sequences": 0.0,
            "secondary_device": "",
        }
    dream_started = perf_counter()
    losses: list[float] = []
    world_losses: list[float] = []
    belief_losses: list[float] = []
    question_losses: list[float] = []
    plan_losses: list[float] = []
    theory_losses: list[float] = []
    diagnostic_language_losses: list[float] = []
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
        sequence_plan_terms: list[torch.Tensor] = []
        sequence_theory_terms: list[torch.Tensor] = []
        sequence_diagnostic_language_terms: list[torch.Tensor] = []
        for position, index in enumerate(window):
            sample = dataset[index]
            state = sample["state"]
            next_state = sample["next_state"]
            action = sample["action"]
            with torch.no_grad():
                next_encoded = encoder.encode_state(next_state, device=device)
            reward_target = torch.tensor([float(sample["reward"])], dtype=torch.float32, device=device)
            return_target = torch.tensor([float(sample.get("discounted_return", sample["reward"]))], dtype=torch.float32, device=device)
            delta_target = torch.tensor(sample["delta"], dtype=torch.float32, device=device).unsqueeze(0)
            usefulness_target = torch.tensor([float(sample["usefulness"])], dtype=torch.float32, device=device)
            effect_target = torch.tensor([int(sample.get("effect_target", _EFFECT_KIND_TO_ID["no_effect"]))], dtype=torch.long, device=device)
            causal_target = torch.tensor([float(sample.get("causal_value_target", sample.get("outcome_signal", 0.0)))], dtype=torch.float32, device=device)
            diagnostic_target = torch.tensor([float(sample.get("diagnostic_target", 0.0))], dtype=torch.float32, device=device)
            world_loss, _metrics = world_model.loss(
                latent=latent,
                actions=[action],
                state=state,
                hidden=hidden,
                next_latent_target=next_encoded.latent.detach(),
                reward_target=reward_target,
                return_target=return_target,
                delta_target=delta_target,
                usefulness_target=usefulness_target,
                effect_target=effect_target,
                causal_target=causal_target,
                diagnostic_target=diagnostic_target,
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
            next_plan_tokens: tuple[str, ...]
            if position + 1 < len(window):
                next_plan_tokens = tuple(dataset[window[position + 1]].get("plan_tokens", ()))
            else:
                next_plan_tokens = _grounded_plan_tokens(
                    next_state,
                    str(sample.get("teacher_action") or sample["action"]),
                )
            plan_loss = language_model.teacher_forcing_loss(
                prediction.next_latent_mean,
                [next_plan_tokens or ("plan", "action", "unknown")],
                mode="plan",
            )
            theory_loss = language_model.teacher_forcing_loss(
                prediction.next_latent_mean,
                [tuple(sample.get("theory_tokens", ())) or ("rule", "unknown")],
                mode="theory",
            )
            diagnostic_language_loss = language_model.teacher_forcing_loss(
                prediction.next_latent_mean,
                [tuple(sample.get("diagnostic_tokens", ())) or ("question", "need", "test", "rule")],
                mode="diagnostic",
            )
            sequence_world_terms.append(world_loss)
            sequence_belief_terms.append(belief_loss)
            sequence_question_terms.append(question_loss)
            sequence_plan_terms.append(plan_loss)
            sequence_theory_terms.append(theory_loss)
            sequence_diagnostic_language_terms.append(diagnostic_language_loss)
            sequence_terms.append(
                world_loss
                + (float(config.dream_belief_weight) * belief_loss)
                + (float(config.dream_question_weight) * question_loss)
                + (float(config.dream_plan_weight) * plan_loss)
                + (float(config.theory_loss_weight) * theory_loss)
                + (float(config.diagnostic_language_loss_weight) * diagnostic_language_loss)
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
        plan_losses.append(float(torch.stack(sequence_plan_terms).mean().detach().cpu()))
        theory_losses.append(float(torch.stack(sequence_theory_terms).mean().detach().cpu()))
        diagnostic_language_losses.append(float(torch.stack(sequence_diagnostic_language_terms).mean().detach().cpu()))
    return {
        "dream_sequences": float(len(losses)),
        "dream_steps": float(total_steps),
        "dream_loss": mean(losses) if losses else 0.0,
        "dream_world_loss": mean(world_losses) if world_losses else 0.0,
        "dream_belief_loss": mean(belief_losses) if belief_losses else 0.0,
        "dream_question_loss": mean(question_losses) if question_losses else 0.0,
        "dream_plan_loss": mean(plan_losses) if plan_losses else 0.0,
        "dream_theory_loss": mean(theory_losses) if theory_losses else 0.0,
        "dream_diagnostic_language_loss": mean(diagnostic_language_losses) if diagnostic_language_losses else 0.0,
        "dream_seconds": perf_counter() - dream_started,
        "secondary_dream_sequences": 0.0,
        "secondary_device": "",
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
    eval_encoder.eval()
    eval_world_model.eval()
    eval_language_model.eval()
    planner = HybridPlanner(
        PlannerConfig(search_depth=2, search_root_width=2, search_branch_width=1, max_world_model_calls=48)
        if device.type == "cpu"
        else PlannerConfig()
    )
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
    agent,
    *,
    family_modes: tuple[str, ...],
    size_options: tuple[int, ...],
    seed_base: int,
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
    use_inference_mode = not bool(getattr(agent, "gradient_world_model_adaptation", False))

    def run_family_sweep() -> None:
        nonlocal seed_cursor
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
    if use_inference_mode:
        with torch.inference_mode():
            run_family_sweep()
    else:
        run_family_sweep()
    action_family_rate = {
        key: value / max(len(all_episodes), 1)
        for key, value in sorted(action_family_totals.items())
    }
    first_success_value = mean(first_episode_success) if first_episode_success else 0.0
    later_success_value = (
        mean(later_episode_success)
        if later_episode_success
        else first_success_value
    )
    return {
        "success_rate": mean(float(item["success"]) for item in all_episodes) if all_episodes else 0.0,
        "avg_return": mean(float(item["return"]) for item in all_episodes) if all_episodes else 0.0,
        "avg_steps": mean(float(item["steps"]) for item in all_episodes) if all_episodes else 0.0,
        "avg_interactions": mean(float(item["interaction_steps"]) for item in all_episodes) if all_episodes else 0.0,
        "first_episode_success": first_success_value,
        "later_episode_success": later_success_value,
        "later_episode_count": float(len(later_episode_success)),
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


def _should_run_regression_holdout(
    config: SyntheticTrainingConfig,
    *,
    holdout_eval_index: int,
    cached_regression_metrics: dict[str, object] | None,
) -> bool:
    cadence = max(1, int(config.regression_holdout_every_evals))
    if cached_regression_metrics is None:
        return True
    return holdout_eval_index % cadence == 0


def _evaluate_loaded_holdout(
    config: SyntheticTrainingConfig,
    encoder: StructuredStateEncoder,
    world_model: RecurrentWorldModel,
    language_model: GroundedLanguageModel,
    *,
    stage_index: int,
    stage_epoch_count: int,
    holdout_eval_index: int,
    device: torch.device,
    cached_regression_holdout: dict[str, object] | None = None,
) -> dict[str, object]:
    stages = _curriculum_stages(config)
    current_stage = stages[min(max(int(stage_index), 0), len(stages) - 1)]
    regression_families = _stage_regression_families(stages, int(stage_index))
    holdout_started = perf_counter()
    eval_agent = _build_eval_agent(
        encoder,
        world_model,
        language_model,
        device=device,
    )
    frontier_started = perf_counter()
    frontier_holdout = _evaluate_holdout(
        config,
        eval_agent,
        family_modes=current_stage.frontier_families,
        size_options=current_stage.holdout_size_options,
        seed_base=config.seed + 100_000 + (int(stage_index) * 10_000),
    )
    frontier_holdout_seconds = perf_counter() - frontier_started
    regression_reference = "none"
    regression_holdout_seconds = 0.0
    regression_holdout_current: dict[str, object] | None = None
    if regression_families and _should_run_regression_holdout(
        config,
        holdout_eval_index=holdout_eval_index,
        cached_regression_metrics=cached_regression_holdout,
    ):
        regression_started = perf_counter()
        regression_holdout_current = _evaluate_holdout(
            config,
            eval_agent,
            family_modes=regression_families,
            size_options=current_stage.holdout_size_options,
            seed_base=config.seed + 200_000 + (int(stage_index) * 10_000),
        )
        regression_holdout_seconds = perf_counter() - regression_started
        regression_reference = "current"
    elif regression_families and cached_regression_holdout is not None:
        regression_reference = "cached"
    regression_holdout = regression_holdout_current or cached_regression_holdout
    holdout_seconds = perf_counter() - holdout_started
    return {
        "epoch": 0,
        "stage_index": int(stage_index),
        "stage_name": current_stage.name,
        "stage_epoch_count": int(stage_epoch_count),
        "holdout_eval_index": int(holdout_eval_index),
        "frontier": frontier_holdout,
        "regression": regression_holdout,
        "threshold_passed": _holdout_passed(current_stage, frontier_holdout, regression_holdout),
        "failure_reasons": _holdout_failure_reasons(current_stage, frontier_holdout, regression_holdout),
        "holdout_seconds": holdout_seconds,
        "frontier_holdout_seconds": frontier_holdout_seconds,
        "regression_holdout_seconds": regression_holdout_seconds,
        "regression_reference": regression_reference,
    }


def _evaluate_checkpoint_holdout(
    checkpoint_path: str,
    config: SyntheticTrainingConfig,
    *,
    epoch: int,
    stage_index: int,
    stage_epoch_count: int,
    holdout_eval_index: int,
    device: torch.device,
    cached_regression_holdout: dict[str, object] | None = None,
) -> dict[str, object]:
    encoder, world_model, language_model = load_checkpoint(checkpoint_path, device=device)
    result = _evaluate_loaded_holdout(
        config,
        encoder,
        world_model,
        language_model,
        stage_index=stage_index,
        stage_epoch_count=stage_epoch_count,
        holdout_eval_index=holdout_eval_index,
        device=device,
        cached_regression_holdout=cached_regression_holdout,
    )
    result["epoch"] = int(epoch)
    result["checkpoint_path"] = str(checkpoint_path)
    return result


def _finalize_holdout_result(
    config: SyntheticTrainingConfig,
    result: dict[str, object],
    *,
    current_stage_index: int,
    consecutive_holdout_passes: int,
) -> tuple[dict[str, object], int]:
    stale_for_stage = int(result["stage_index"]) != int(current_stage_index)
    gated_passed = bool(result["threshold_passed"]) and int(result["stage_epoch_count"]) >= max(
        1,
        int(config.holdout_eval_every_epochs),
    )
    if stale_for_stage:
        next_consecutive = int(consecutive_holdout_passes)
    else:
        next_consecutive = int(consecutive_holdout_passes) + 1 if gated_passed else 0
    holdout_result = {
        "epoch": int(result["epoch"]),
        "stage_index": int(result["stage_index"]),
        "stage_name": str(result["stage_name"]),
        "stage_epoch_count": int(result["stage_epoch_count"]),
        "holdout_eval_index": int(result["holdout_eval_index"]),
        "frontier": result["frontier"],
        "regression": result["regression"],
        "passed": gated_passed,
        "threshold_passed": bool(result["threshold_passed"]),
        "failure_reasons": list(result["failure_reasons"]),
        "consecutive_passes": next_consecutive,
        "required_consecutive_passes": int(config.promotion_consecutive_evals),
        "holdout_seconds": float(result["holdout_seconds"]),
        "frontier_holdout_seconds": float(result["frontier_holdout_seconds"]),
        "regression_holdout_seconds": float(result["regression_holdout_seconds"]),
        "regression_reference": str(result["regression_reference"]),
        "stale_for_stage": stale_for_stage,
        "checkpoint_path": str(result.get("checkpoint_path", "")),
    }
    return holdout_result, next_consecutive


def _update_history_entry_from_holdout(history: list[dict[str, object]], holdout_result: dict[str, object]) -> None:
    epoch = int(holdout_result["epoch"])
    if epoch < 0 or epoch >= len(history):
        return
    entry = history[epoch]
    entry["holdout_passed"] = bool(holdout_result["passed"])
    entry["holdout_evaluated"] = True
    entry["holdout_frontier_success"] = float(holdout_result["frontier"]["success_rate"])
    entry["holdout_consecutive_passes"] = float(holdout_result["consecutive_passes"])
    entry["holdout_seconds"] = float(holdout_result["holdout_seconds"])
    entry["frontier_holdout_seconds"] = float(holdout_result["frontier_holdout_seconds"])
    entry["regression_holdout_seconds"] = float(holdout_result["regression_holdout_seconds"])
    entry["regression_reference"] = str(holdout_result["regression_reference"])


def _async_holdout_worker_loop(request_queue: object, result_queue: object) -> None:
    cached_regression_holdout: dict[str, object] | None = None
    cached_stage_index: int | None = None
    while True:
        request = request_queue.get()
        if request is None:
            break
        try:
            config = SyntheticTrainingConfig(**dict(request["config"]))
            stage_index = int(request["stage_index"])
            if cached_stage_index != stage_index:
                cached_regression_holdout = None
                cached_stage_index = stage_index
            device = torch.device(str(request["device"]))
            if device.type == "cuda" and torch.cuda.is_available():
                torch.cuda.set_device(device)
            result = _evaluate_checkpoint_holdout(
                str(request["checkpoint_path"]),
                config,
                epoch=int(request["epoch"]),
                stage_index=stage_index,
                stage_epoch_count=int(request["stage_epoch_count"]),
                holdout_eval_index=int(request["holdout_eval_index"]),
                device=device,
                cached_regression_holdout=cached_regression_holdout,
            )
            if str(result["regression_reference"]) == "current":
                cached_regression_holdout = result["regression"]
            result_queue.put({"ok": True, **result})
        except Exception as exc:
            result_queue.put(
                {
                    "ok": False,
                    "epoch": int(request.get("epoch", -1)),
                    "stage_index": int(request.get("stage_index", -1)),
                    "holdout_eval_index": int(request.get("holdout_eval_index", -1)),
                    "error": repr(exc),
                }
            )


def _start_async_holdout_runtime(config: SyntheticTrainingConfig) -> _AsyncHoldoutRuntime:
    context = mp.get_context("spawn")
    request_queue = context.Queue()
    result_queue = context.Queue()
    process = context.Process(
        target=_async_holdout_worker_loop,
        args=(request_queue, result_queue),
        daemon=True,
    )
    process.start()
    return _AsyncHoldoutRuntime(
        request_queue=request_queue,
        result_queue=result_queue,
        process=process,
    )


def _submit_async_holdout_request(
    runtime: _AsyncHoldoutRuntime,
    config: SyntheticTrainingConfig,
    *,
    checkpoint_path: str,
    epoch: int,
    stage_index: int,
    stage_epoch_count: int,
) -> int:
    holdout_eval_index = int(runtime.next_request_index)
    runtime.next_request_index += 1
    runtime.pending_count += 1
    runtime.request_queue.put(
        {
            "config": asdict(config),
            "checkpoint_path": str(checkpoint_path),
            "epoch": int(epoch),
            "stage_index": int(stage_index),
            "stage_epoch_count": int(stage_epoch_count),
            "holdout_eval_index": holdout_eval_index,
            "device": str(config.async_eval_device),
        }
    )
    return holdout_eval_index


def _collect_ready_async_holdout_results(
    runtime: _AsyncHoldoutRuntime,
    *,
    block: bool = False,
    timeout_seconds: float = 0.1,
) -> list[dict[str, object]]:
    ready: list[dict[str, object]] = []
    if block and runtime.pending_count > 0:
        try:
            first = runtime.result_queue.get(timeout=timeout_seconds)
        except queue.Empty:
            first = None
        if first is not None:
            runtime.pending_count = max(0, runtime.pending_count - 1)
            runtime.buffered_results[int(first["holdout_eval_index"])] = first
    while True:
        try:
            item = runtime.result_queue.get_nowait()
        except queue.Empty:
            break
        runtime.pending_count = max(0, runtime.pending_count - 1)
        runtime.buffered_results[int(item["holdout_eval_index"])] = item
    while runtime.next_result_index in runtime.buffered_results:
        ready.append(runtime.buffered_results.pop(runtime.next_result_index))
        runtime.next_result_index += 1
    if runtime.pending_count > 0 and not runtime.process.is_alive() and not ready:
        raise RuntimeError("async holdout worker exited before returning all pending results")
    return ready


def _stop_async_holdout_runtime(runtime: _AsyncHoldoutRuntime, *, wait: bool) -> None:
    runtime.request_queue.put(None)
    join_timeout = 300.0 if wait else 0.1
    runtime.process.join(timeout=join_timeout)
    if runtime.process.is_alive() and not wait:
        runtime.process.terminate()
        runtime.process.join(timeout=1.0)


def train_synthetic(
    config: SyntheticTrainingConfig,
    device: torch.device | None = None,
) -> dict[str, object]:
    _validate_synthetic_training_config(config)
    seed_everything(config.seed)
    if device is not None:
        device = torch.device(str(device))
    elif config.train_device:
        device = torch.device(config.train_device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    encoder, world_model, language_model, _planner = build_default_modules(device=device)
    stages = _curriculum_stages(config)
    resume_state: dict[str, object] | None = None
    startup_mode = "scratch"
    startup_checkpoint_path = ""
    if config.resume_checkpoint_path and Path(config.resume_checkpoint_path).exists():
        resume_state = _load_resume_state(
            config.resume_checkpoint_path,
            encoder=encoder,
            world_model=world_model,
            language_model=language_model,
            device=device,
        )
        startup_mode = "resume"
        startup_checkpoint_path = str(config.resume_checkpoint_path)
    elif config.init_checkpoint_path and Path(config.init_checkpoint_path).exists():
        checkpoint = torch.load(config.init_checkpoint_path, map_location=device)
        if _checkpoint_contains_resume_metadata(checkpoint) and not bool(
            config.allow_weights_only_init_from_training_checkpoint
        ):
            raise ValueError(
                "init_checkpoint_path points to a checkpoint with saved training state; "
                "use resume_checkpoint_path to continue curriculum/state or set "
                "allow_weights_only_init_from_training_checkpoint=True for an explicit weights-only load"
            )
        _load_compatible_state_dict(encoder, checkpoint["encoder"], module_name="encoder")
        _load_compatible_state_dict(world_model, checkpoint["world_model"], module_name="world_model")
        _load_compatible_state_dict(language_model, checkpoint["language_model"], module_name="language_model")
        startup_mode = "weights_only_init"
        startup_checkpoint_path = str(config.init_checkpoint_path)
    optimizer = torch.optim.Adam(
        list(encoder.parameters()) + list(world_model.parameters()) + list(language_model.parameters()),
        lr=config.learning_rate,
    )
    history: list[dict[str, float | int | bool]] = list(
        resume_state["history"] if resume_state is not None else []
    )
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    current_epoch = int(resume_state["current_epoch"]) if resume_state is not None else -1
    start_epoch = int(resume_state["start_epoch"]) if resume_state is not None else 0
    current_collection_metrics: dict[str, object] = {}
    epoch_losses: list[float] = []
    epoch_uncertainty: list[float] = []
    epoch_world_total_losses: list[float] = []
    epoch_latent_losses: list[float] = []
    epoch_reward_losses: list[float] = []
    epoch_delta_losses: list[float] = []
    epoch_usefulness_losses: list[float] = []
    epoch_causal_losses: list[float] = []
    epoch_diagnostic_world_losses: list[float] = []
    epoch_effect_losses: list[float] = []
    epoch_belief_losses: list[float] = []
    epoch_question_losses: list[float] = []
    epoch_plan_losses: list[float] = []
    epoch_theory_losses: list[float] = []
    epoch_diagnostic_language_losses: list[float] = []
    epoch_policy_losses: list[float] = []
    epoch_positive_policy_losses: list[float] = []
    epoch_negative_policy_losses: list[float] = []
    epoch_gate_losses: list[float] = []
    dream_metrics: dict[str, float] = {}
    secondary_train_device = ""
    secondary_replay_samples = 0.0
    if resume_state is not None:
        current_stage_index = int(resume_state["stage_index"])
    elif config.curriculum in {"staged", "gated"}:
        current_stage_index = 0
    elif config.curriculum == "fixed_staged":
        current_stage_index = _fixed_stage_index_for_epoch(config, 0)
    else:
        current_stage_index = 0
    stage_epoch_count = int(resume_state["stage_epoch_count"]) if resume_state is not None else 0
    consecutive_holdout_passes = int(resume_state["consecutive_holdout_passes"]) if resume_state is not None else 0
    holdout_history: list[dict[str, object]] = list(
        resume_state["holdout_history"] if resume_state is not None else []
    )
    last_holdout_result: dict[str, object] | None = (
        dict(resume_state["last_holdout_result"]) if resume_state is not None and resume_state["last_holdout_result"] else None
    )
    cached_regression_holdout: dict[str, object] | None = (
        dict(resume_state["cached_regression_holdout"]) if resume_state is not None and resume_state["cached_regression_holdout"] else None
    )
    async_holdout_runtime: _AsyncHoldoutRuntime | None = None
    secondary_training_runtime: _SecondaryTrainingRuntime | None = None
    if config.async_eval_device:
        async_eval_device = torch.device(config.async_eval_device)
        if async_eval_device == device:
            raise ValueError("async_eval_device must differ from the training device")
        if device.type == "cuda" and async_eval_device.type == "cuda" and torch.cuda.is_available():
            secondary_training_runtime = _build_secondary_training_runtime(
                encoder,
                world_model,
                language_model,
                device=async_eval_device,
            )
            print(
                json.dumps(
                    {
                        "type": "secondary_training_enabled",
                        "primary_device": str(device),
                        "secondary_device": str(async_eval_device),
                        "async_holdout_enabled": False,
                    }
                ),
                flush=True,
            )
        else:
            async_holdout_runtime = _start_async_holdout_runtime(config)

    current_stage = stages[min(max(current_stage_index, 0), len(stages) - 1)] if stages else None
    startup_stage_name = (
        "flat"
        if config.curriculum == "flat"
        else "fixed_staged"
        if config.curriculum == "fixed_staged"
        else current_stage.name
        if current_stage is not None
        else ""
    )
    startup_frontier_families = (
        tuple(dict.fromkeys(config.family_modes))
        if config.curriculum == "flat"
        else current_stage.frontier_families
        if current_stage is not None
        else ()
    )
    startup_train_sizes = (
        config.size_options
        if config.curriculum == "flat"
        else current_stage.train_size_options
        if current_stage is not None
        else config.size_options
    )
    startup_holdout_sizes = (
        config.size_options
        if config.curriculum == "flat"
        else current_stage.holdout_size_options
        if current_stage is not None
        else config.size_options
    )
    print(
        json.dumps(
            {
                "type": "training_start",
                "mode": startup_mode,
                "checkpoint_path": startup_checkpoint_path,
                "start_epoch": start_epoch,
                "stage_index": current_stage_index,
                "stage_name": startup_stage_name,
                "stage_epoch_count": stage_epoch_count,
                "frontier_families": startup_frontier_families,
                "train_size_options": startup_train_sizes,
                "holdout_size_options": startup_holdout_sizes,
                "device": str(device),
                "secondary_device": secondary_train_device or str(config.async_eval_device or ""),
            }
        ),
        flush=True,
    )

    def consume_async_holdout_results(*, block: bool, report_epoch: int) -> None:
        nonlocal current_stage_index
        nonlocal stage_epoch_count
        nonlocal consecutive_holdout_passes
        nonlocal last_holdout_result
        if async_holdout_runtime is None:
            return
        timeout_seconds = 3600.0 if block else 0.0
        ready_results = _collect_ready_async_holdout_results(
            async_holdout_runtime,
            block=block,
            timeout_seconds=timeout_seconds,
        )
        for raw_result in ready_results:
            if not bool(raw_result.get("ok", False)):
                raise RuntimeError(
                    "async holdout worker failed for "
                    f"epoch={raw_result.get('epoch')} index={raw_result.get('holdout_eval_index')}: "
                    f"{raw_result.get('error')}"
                )
            holdout_result, consecutive_holdout_passes = _finalize_holdout_result(
                config,
                raw_result,
                current_stage_index=current_stage_index,
                consecutive_holdout_passes=consecutive_holdout_passes,
            )
            holdout_history.append(holdout_result)
            last_holdout_result = holdout_result
            _update_history_entry_from_holdout(history, holdout_result)
            print(
                json.dumps(
                    {
                        "type": "holdout_eval",
                        "async": True,
                        "reported_epoch": report_epoch,
                        "epoch": holdout_result["epoch"],
                        "stage_index": holdout_result["stage_index"],
                        "stage_name": holdout_result["stage_name"],
                        "stage_epoch_count": holdout_result["stage_epoch_count"],
                        "holdout_eval_index": holdout_result["holdout_eval_index"],
                        "frontier": holdout_result["frontier"],
                        "regression": holdout_result["regression"],
                        "passed": holdout_result["passed"],
                        "threshold_passed": holdout_result["threshold_passed"],
                        "failure_reasons": holdout_result["failure_reasons"],
                        "consecutive_passes": holdout_result["consecutive_passes"],
                        "required_consecutive_passes": holdout_result["required_consecutive_passes"],
                        "holdout_seconds": holdout_result["holdout_seconds"],
                        "frontier_holdout_seconds": holdout_result["frontier_holdout_seconds"],
                        "regression_holdout_seconds": holdout_result["regression_holdout_seconds"],
                        "regression_reference": holdout_result["regression_reference"],
                        "stale_for_stage": holdout_result["stale_for_stage"],
                        "checkpoint_path": holdout_result["checkpoint_path"],
                    }
                ),
                flush=True,
            )
            if (
                not bool(holdout_result["stale_for_stage"])
                and bool(holdout_result["passed"])
                and consecutive_holdout_passes >= config.promotion_consecutive_evals
                and current_stage_index < len(stages) - 1
            ):
                previous_stage_name = stages[current_stage_index].name
                current_stage_index += 1
                stage_epoch_count = 0
                consecutive_holdout_passes = 0
                print(
                    json.dumps(
                        {
                            "type": "stage_advance",
                            "async": True,
                            "reported_epoch": report_epoch,
                            "epoch": holdout_result["epoch"],
                            "from_stage_index": current_stage_index - 1,
                            "from_stage_name": previous_stage_name,
                            "to_stage_index": current_stage_index,
                            "to_stage_name": stages[current_stage_index].name,
                            "reason": "async_heldout_thresholds_cleared",
                        }
                    ),
                    flush=True,
                )
    try:
        for epoch in range(start_epoch, start_epoch + config.epochs):
            current_epoch = epoch
            epoch_started = perf_counter()
            consume_async_holdout_results(block=False, report_epoch=epoch)
            if config.curriculum == "fixed_staged":
                current_stage_index = _fixed_stage_index_for_epoch(config, epoch)
            dataset, current_collection_metrics = collect_dataset(
                config,
                epoch_index=epoch,
                stage_index=current_stage_index if config.curriculum in {"staged", "gated"} else None,
                history=history,
                encoder=encoder,
                world_model=world_model,
                language_model=language_model,
                device=device,
            )
            metric_lists: dict[str, list[float]] = {
                "epoch_losses": [],
                "epoch_uncertainty": [],
                "epoch_world_total_losses": [],
                "epoch_latent_losses": [],
                "epoch_reward_losses": [],
                "epoch_delta_losses": [],
                "epoch_usefulness_losses": [],
                "epoch_causal_losses": [],
                "epoch_diagnostic_world_losses": [],
                "epoch_effect_losses": [],
                "epoch_belief_losses": [],
                "epoch_question_losses": [],
                "epoch_plan_losses": [],
                "epoch_theory_losses": [],
                "epoch_diagnostic_language_losses": [],
                "epoch_policy_losses": [],
                "epoch_positive_policy_losses": [],
                "epoch_negative_policy_losses": [],
                "epoch_gate_losses": [],
            }
            epoch_losses = metric_lists["epoch_losses"]
            epoch_uncertainty = metric_lists["epoch_uncertainty"]
            epoch_world_total_losses = metric_lists["epoch_world_total_losses"]
            epoch_latent_losses = metric_lists["epoch_latent_losses"]
            epoch_reward_losses = metric_lists["epoch_reward_losses"]
            epoch_delta_losses = metric_lists["epoch_delta_losses"]
            epoch_usefulness_losses = metric_lists["epoch_usefulness_losses"]
            epoch_causal_losses = metric_lists["epoch_causal_losses"]
            epoch_diagnostic_world_losses = metric_lists["epoch_diagnostic_world_losses"]
            epoch_effect_losses = metric_lists["epoch_effect_losses"]
            epoch_belief_losses = metric_lists["epoch_belief_losses"]
            epoch_question_losses = metric_lists["epoch_question_losses"]
            epoch_plan_losses = metric_lists["epoch_plan_losses"]
            epoch_theory_losses = metric_lists["epoch_theory_losses"]
            epoch_diagnostic_language_losses = metric_lists["epoch_diagnostic_language_losses"]
            epoch_policy_losses = metric_lists["epoch_policy_losses"]
            epoch_positive_policy_losses = metric_lists["epoch_positive_policy_losses"]
            epoch_negative_policy_losses = metric_lists["epoch_negative_policy_losses"]
            epoch_gate_losses = metric_lists["epoch_gate_losses"]
            dream_metrics = {}
            encoder.train()
            world_model.train()
            language_model.train()
            secondary_replay_samples = 0.0
            secondary_train_device = ""
            if secondary_training_runtime is not None:
                replay_metrics = _run_multidevice_replay_phase(
                    config,
                    dataset=dataset,
                    encoder=encoder,
                    world_model=world_model,
                    language_model=language_model,
                    optimizer=optimizer,
                    device=device,
                    secondary_runtime=secondary_training_runtime,
                    metric_lists=metric_lists,
                )
                train_seconds = float(replay_metrics["train_seconds"])
                secondary_replay_samples = float(replay_metrics["secondary_replay_samples"])
                secondary_train_device = str(replay_metrics["secondary_device"])
                dream_metrics = _run_multidevice_dream_phase(
                    config,
                    dataset=dataset,
                    epoch_index=epoch,
                    encoder=encoder,
                    world_model=world_model,
                    language_model=language_model,
                    optimizer=optimizer,
                    device=device,
                    secondary_runtime=secondary_training_runtime,
                )
            else:
                sequence_episode_id: str | None = None
                sequence_hidden: torch.Tensor | None = None
                train_started = perf_counter()
                for sample in dataset:
                    state = sample["state"]
                    if sequence_episode_id != state.episode_id or state.step_index == 0:
                        sequence_episode_id = state.episode_id
                        sequence_hidden = None
                    prepared = _prepare_replay_training_step(
                        config,
                        sample,
                        encoder=encoder,
                        world_model=world_model,
                        language_model=language_model,
                        sequence_hidden=sequence_hidden,
                        device=device,
                    )
                    optimizer.zero_grad(set_to_none=True)
                    prepared[0].backward()
                    torch.nn.utils.clip_grad_norm_(
                        _trainable_parameters(encoder, world_model, language_model),
                        max_norm=1.0,
                    )
                    optimizer.step()
                    sequence_hidden = _advance_sequence_hidden(
                        world_model,
                        latent=prepared[2],
                        action=prepared[4],
                        state=prepared[3],
                        sequence_hidden=sequence_hidden,
                    )
                    _append_replay_metrics(metric_lists, prepared[1])
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
            holdout_seconds = 0.0
            frontier_holdout_seconds = 0.0
            regression_holdout_seconds = 0.0
            enqueue_async_holdout = False
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
                if async_holdout_runtime is None:
                    sync_holdout_result = _evaluate_loaded_holdout(
                        config,
                        encoder,
                        world_model,
                        language_model,
                        stage_index=current_stage_index,
                        stage_epoch_count=stage_epoch_count,
                        holdout_eval_index=len(holdout_history) + 1,
                        device=secondary_training_runtime.device if secondary_training_runtime is not None else device,
                        cached_regression_holdout=cached_regression_holdout,
                    )
                    if str(sync_holdout_result["regression_reference"]) == "current":
                        cached_regression_holdout = sync_holdout_result["regression"]
                    sync_holdout_result["epoch"] = epoch
                    holdout_result, consecutive_holdout_passes = _finalize_holdout_result(
                        config,
                        sync_holdout_result,
                        current_stage_index=current_stage_index,
                        consecutive_holdout_passes=consecutive_holdout_passes,
                    )
                    holdout_history.append(holdout_result)
                    last_holdout_result = holdout_result
                    print(
                        json.dumps(
                            {
                                "type": "holdout_eval",
                                "async": False,
                                "epoch": epoch,
                                "stage_index": holdout_result["stage_index"],
                                "stage_name": holdout_result["stage_name"],
                                "stage_epoch_count": holdout_result["stage_epoch_count"],
                                "holdout_eval_index": holdout_result["holdout_eval_index"],
                                "frontier": holdout_result["frontier"],
                                "regression": holdout_result["regression"],
                                "passed": holdout_result["passed"],
                                "threshold_passed": holdout_result["threshold_passed"],
                                "failure_reasons": holdout_result["failure_reasons"],
                                "consecutive_passes": holdout_result["consecutive_passes"],
                                "required_consecutive_passes": holdout_result["required_consecutive_passes"],
                                "holdout_seconds": holdout_result["holdout_seconds"],
                                "frontier_holdout_seconds": holdout_result["frontier_holdout_seconds"],
                                "regression_holdout_seconds": holdout_result["regression_holdout_seconds"],
                                "regression_reference": holdout_result["regression_reference"],
                                "stale_for_stage": holdout_result["stale_for_stage"],
                            }
                        ),
                        flush=True,
                    )
                    if (
                        not bool(holdout_result["stale_for_stage"])
                        and bool(holdout_result["passed"])
                        and consecutive_holdout_passes >= config.promotion_consecutive_evals
                        and current_stage_index < len(stages) - 1
                    ):
                        previous_stage_name = current_stage.name
                        current_stage_index += 1
                        stage_epoch_count = 0
                        consecutive_holdout_passes = 0
                        cached_regression_holdout = None
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
                else:
                    enqueue_async_holdout = True
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
                    "collector_agent": str(current_collection_metrics.get("collector_agent", "graph")),
                    "family_sampling_weights": current_collection_metrics.get("family_sampling_weights", {}),
                    "family_counts": current_collection_metrics.get("family_counts", {}),
                    "family_breakdown": current_collection_metrics.get("family_breakdown", {}),
                    "variant_breakdown": current_collection_metrics.get("variant_breakdown", {}),
                    "size_breakdown": current_collection_metrics.get("size_breakdown", {}),
                    "epoch_seed_base": current_collection_metrics.get("epoch_seed_base", 0.0),
                    "collect_seconds": current_collection_metrics.get("collect_seconds", 0.0),
                    "train_seconds": train_seconds,
                    "secondary_train_device": secondary_train_device,
                    "secondary_replay_samples": secondary_replay_samples,
                    "epoch_seconds": perf_counter() - epoch_started,
                    "world_loss": mean(epoch_world_total_losses) if epoch_world_total_losses else 0.0,
                    "latent_loss": mean(epoch_latent_losses) if epoch_latent_losses else 0.0,
                    "reward_loss": mean(epoch_reward_losses) if epoch_reward_losses else 0.0,
                    "delta_loss": mean(epoch_delta_losses) if epoch_delta_losses else 0.0,
                    "usefulness_loss": mean(epoch_usefulness_losses) if epoch_usefulness_losses else 0.0,
                    "causal_loss": mean(epoch_causal_losses) if epoch_causal_losses else 0.0,
                    "diagnostic_world_loss": mean(epoch_diagnostic_world_losses) if epoch_diagnostic_world_losses else 0.0,
                    "effect_loss": mean(epoch_effect_losses) if epoch_effect_losses else 0.0,
                    "belief_loss": mean(epoch_belief_losses) if epoch_belief_losses else 0.0,
                    "question_loss": mean(epoch_question_losses) if epoch_question_losses else 0.0,
                    "plan_loss": mean(epoch_plan_losses) if epoch_plan_losses else 0.0,
                    "theory_loss": mean(epoch_theory_losses) if epoch_theory_losses else 0.0,
                    "diagnostic_language_loss": mean(epoch_diagnostic_language_losses) if epoch_diagnostic_language_losses else 0.0,
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
                    "holdout_seconds": float(holdout_result["holdout_seconds"])
                    if holdout_result is not None
                    else 0.0,
                    "frontier_holdout_seconds": float(holdout_result["frontier_holdout_seconds"])
                    if holdout_result is not None
                    else 0.0,
                    "regression_holdout_seconds": float(holdout_result["regression_holdout_seconds"])
                    if holdout_result is not None
                    else 0.0,
                    "regression_reference": str(holdout_result["regression_reference"])
                    if holdout_result is not None
                    else "none",
                    "bootstrap_steps": current_collection_metrics.get("bootstrap_steps", 0.0),
                    "bootstrap_stride": current_collection_metrics.get("bootstrap_stride", 1.0),
                    "bootstrap_release_epoch": current_collection_metrics.get("bootstrap_release_epoch", -1.0),
                    "bootstrap_ready_streak": current_collection_metrics.get("bootstrap_ready_streak", 0.0),
                    "teacher_guidance_ready_streak": current_collection_metrics.get("teacher_guidance_ready_streak", 0.0),
                    "teacher_guidance_alpha": current_collection_metrics.get("teacher_guidance_alpha", 0.0),
                    "dense_teacher_supervision": current_collection_metrics.get("dense_teacher_supervision", False),
                    "teacher_episode_fraction": current_collection_metrics.get("teacher_episode_fraction", 0.0),
                    "teacher_takeover_prob": current_collection_metrics.get("teacher_takeover_prob", 0.0),
                    "teacher_episode_count": current_collection_metrics.get("teacher_episode_count", 0.0),
                    "teacher_step_fraction": current_collection_metrics.get("teacher_step_fraction", 0.0),
                    "teacher_relabel_fraction": current_collection_metrics.get("teacher_relabel_fraction", 0.0),
                    "learner_teacher_agreement": current_collection_metrics.get("learner_teacher_agreement", 0.0),
                    "dream_sequences": dream_metrics.get("dream_sequences", 0.0),
                    "dream_steps": dream_metrics.get("dream_steps", 0.0),
                    "dream_loss": dream_metrics.get("dream_loss", 0.0),
                    "dream_world_loss": dream_metrics.get("dream_world_loss", 0.0),
                    "dream_belief_loss": dream_metrics.get("dream_belief_loss", 0.0),
                    "dream_question_loss": dream_metrics.get("dream_question_loss", 0.0),
                    "dream_plan_loss": dream_metrics.get("dream_plan_loss", 0.0),
                    "dream_theory_loss": dream_metrics.get("dream_theory_loss", 0.0),
                    "dream_diagnostic_language_loss": dream_metrics.get("dream_diagnostic_language_loss", 0.0),
                    "dream_seconds": dream_metrics.get("dream_seconds", 0.0),
                    "secondary_dream_sequences": dream_metrics.get("secondary_dream_sequences", 0.0),
                }
            )
            training_state = {
                "completed_epochs": len(history),
                "active_epoch": epoch,
                "interrupted": False,
                "last_epoch_samples": history[-1]["samples"],
                "stage_index": current_stage_index,
                "stage_name": str(current_collection_metrics.get("stage_name", epoch_stage_name)),
                "stage_epoch_count": stage_epoch_count,
                "consecutive_holdout_passes": consecutive_holdout_passes,
                "holdout_seconds": history[-1]["holdout_seconds"],
                "frontier_holdout_seconds": history[-1]["frontier_holdout_seconds"],
                "regression_holdout_seconds": history[-1]["regression_holdout_seconds"],
                "regression_reference": history[-1]["regression_reference"],
                "bootstrap_steps": current_collection_metrics.get("bootstrap_steps", 0.0),
                "bootstrap_stride": current_collection_metrics.get("bootstrap_stride", 1.0),
                "bootstrap_release_epoch": current_collection_metrics.get("bootstrap_release_epoch", -1.0),
                "bootstrap_ready_streak": current_collection_metrics.get("bootstrap_ready_streak", 0.0),
                "teacher_guidance_ready_streak": current_collection_metrics.get("teacher_guidance_ready_streak", 0.0),
                "teacher_guidance_alpha": current_collection_metrics.get("teacher_guidance_alpha", 0.0),
                "dense_teacher_supervision": current_collection_metrics.get("dense_teacher_supervision", False),
                "teacher_episode_fraction": current_collection_metrics.get("teacher_episode_fraction", 0.0),
                "teacher_takeover_prob": current_collection_metrics.get("teacher_takeover_prob", 0.0),
                "teacher_episode_count": current_collection_metrics.get("teacher_episode_count", 0.0),
                "teacher_step_fraction": current_collection_metrics.get("teacher_step_fraction", 0.0),
                "teacher_relabel_fraction": current_collection_metrics.get("teacher_relabel_fraction", 0.0),
                "collector_agent": current_collection_metrics.get("collector_agent", "graph"),
                "family_sampling_weights": current_collection_metrics.get("family_sampling_weights", {}),
                "dream_sequences": dream_metrics.get("dream_sequences", 0.0),
                "dream_steps": dream_metrics.get("dream_steps", 0.0),
                "dream_loss": dream_metrics.get("dream_loss", 0.0),
                "dream_world_loss": dream_metrics.get("dream_world_loss", 0.0),
                "dream_belief_loss": dream_metrics.get("dream_belief_loss", 0.0),
                "dream_question_loss": dream_metrics.get("dream_question_loss", 0.0),
                "dream_plan_loss": dream_metrics.get("dream_plan_loss", 0.0),
                "dream_seconds": dream_metrics.get("dream_seconds", 0.0),
                "secondary_train_device": secondary_train_device,
                "secondary_replay_samples": secondary_replay_samples,
                "secondary_dream_sequences": dream_metrics.get("secondary_dream_sequences", 0.0),
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
            if enqueue_async_holdout and async_holdout_runtime is not None:
                holdout_eval_index = _submit_async_holdout_request(
                    async_holdout_runtime,
                    config,
                    checkpoint_path=str(saved_paths.get("epoch_checkpoint_path", saved_paths["latest_checkpoint_path"])),
                    epoch=epoch,
                    stage_index=epoch_stage_index,
                    stage_epoch_count=epoch_stage_epoch_count,
                )
                print(
                    json.dumps(
                        {
                            "type": "holdout_enqueued",
                            "async": True,
                            "epoch": epoch,
                            "stage_index": epoch_stage_index,
                            "stage_name": current_collection_metrics.get("stage_name", epoch_stage_name),
                            "stage_epoch_count": epoch_stage_epoch_count,
                            "holdout_eval_index": holdout_eval_index,
                            "checkpoint_path": str(
                                saved_paths.get("epoch_checkpoint_path", saved_paths["latest_checkpoint_path"])
                            ),
                            "device": config.async_eval_device,
                        }
                    ),
                    flush=True,
                )
                consume_async_holdout_results(block=False, report_epoch=epoch)
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
                        "collector_agent": history[-1]["collector_agent"],
                        "family_sampling_weights": history[-1]["family_sampling_weights"],
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
                        "secondary_train_device": history[-1]["secondary_train_device"],
                        "secondary_replay_samples": history[-1]["secondary_replay_samples"],
                        "epoch_seconds": history[-1]["epoch_seconds"],
                        "holdout_seconds": history[-1]["holdout_seconds"],
                        "frontier_holdout_seconds": history[-1]["frontier_holdout_seconds"],
                        "regression_holdout_seconds": history[-1]["regression_holdout_seconds"],
                        "regression_reference": history[-1]["regression_reference"],
                        "world_loss": history[-1]["world_loss"],
                        "latent_loss": history[-1]["latent_loss"],
                        "reward_loss": history[-1]["reward_loss"],
                        "delta_loss": history[-1]["delta_loss"],
                        "usefulness_loss": history[-1]["usefulness_loss"],
                        "causal_loss": history[-1]["causal_loss"],
                        "diagnostic_world_loss": history[-1]["diagnostic_world_loss"],
                        "effect_loss": history[-1]["effect_loss"],
                        "belief_loss": history[-1]["belief_loss"],
                        "question_loss": history[-1]["question_loss"],
                        "plan_loss": history[-1]["plan_loss"],
                        "theory_loss": history[-1]["theory_loss"],
                        "diagnostic_language_loss": history[-1]["diagnostic_language_loss"],
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
                        "dream_plan_loss": history[-1]["dream_plan_loss"],
                        "dream_theory_loss": history[-1]["dream_theory_loss"],
                        "dream_diagnostic_language_loss": history[-1]["dream_diagnostic_language_loss"],
                        "dream_seconds": history[-1]["dream_seconds"],
                        "secondary_dream_sequences": history[-1]["secondary_dream_sequences"],
                        **saved_paths,
                    }
                ),
                flush=True,
            )
    except KeyboardInterrupt:
        if async_holdout_runtime is not None:
            _stop_async_holdout_runtime(async_holdout_runtime, wait=False)
        training_state = {
            "completed_epochs": len(history),
            "active_epoch": current_epoch,
            "interrupted": True,
            "partial_epoch_loss": mean(epoch_losses) if epoch_losses else 0.0,
            "partial_epoch_uncertainty": mean(epoch_uncertainty) if epoch_uncertainty else 0.0,
            "partial_epoch_samples": float(current_collection_metrics.get("samples", 0.0)),
            "partial_world_loss": mean(epoch_world_total_losses) if epoch_world_total_losses else 0.0,
            "partial_causal_loss": mean(epoch_causal_losses) if epoch_causal_losses else 0.0,
            "partial_effect_loss": mean(epoch_effect_losses) if epoch_effect_losses else 0.0,
            "partial_policy_loss": mean(epoch_policy_losses) if epoch_policy_losses else 0.0,
            "stage_index": current_stage_index,
            "stage_name": str(current_collection_metrics.get("stage_name", "flat" if config.curriculum == "flat" else stages[current_stage_index].name if stages else "")),
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
            "dream_plan_loss": float(dream_metrics.get("dream_plan_loss", 0.0)),
            "dream_theory_loss": float(dream_metrics.get("dream_theory_loss", 0.0)),
            "dream_diagnostic_language_loss": float(dream_metrics.get("dream_diagnostic_language_loss", 0.0)),
            "dream_seconds": float(dream_metrics.get("dream_seconds", 0.0)),
            "secondary_train_device": secondary_train_device,
            "secondary_replay_samples": float(secondary_replay_samples),
            "secondary_dream_sequences": float(dream_metrics.get("secondary_dream_sequences", 0.0)),
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
            "stage_name": str(current_collection_metrics.get("stage_name", "flat" if config.curriculum == "flat" else stages[current_stage_index].name if stages else "")),
            "samples_last_epoch": history[-1]["samples"] if history else 0.0,
            "loss_last_epoch": history[-1]["loss"] if history else 0.0,
            "uncertainty_last_epoch": history[-1]["uncertainty"] if history else 0.0,
            "last_holdout_result": last_holdout_result or {},
            "collector_agent": history[-1]["collector_agent"] if history else "",
            **interrupt_paths,
        }
    if async_holdout_runtime is not None:
        while async_holdout_runtime.pending_count > 0:
            consume_async_holdout_results(block=True, report_epoch=current_epoch)
        _stop_async_holdout_runtime(async_holdout_runtime, wait=True)
        final_training_state = {
            "completed_epochs": len(history),
            "active_epoch": current_epoch,
            "interrupted": False,
            "last_epoch_samples": history[-1]["samples"] if history else 0.0,
            "stage_index": current_stage_index,
            "stage_name": str(current_collection_metrics.get("stage_name", "flat" if config.curriculum == "flat" else stages[current_stage_index].name if stages else "")),
            "stage_epoch_count": stage_epoch_count,
            "consecutive_holdout_passes": consecutive_holdout_passes,
            "holdout_seconds": history[-1]["holdout_seconds"] if history else 0.0,
            "frontier_holdout_seconds": history[-1]["frontier_holdout_seconds"] if history else 0.0,
            "regression_holdout_seconds": history[-1]["regression_holdout_seconds"] if history else 0.0,
            "regression_reference": history[-1]["regression_reference"] if history else "none",
            "bootstrap_steps": current_collection_metrics.get("bootstrap_steps", 0.0),
            "bootstrap_stride": current_collection_metrics.get("bootstrap_stride", 1.0),
            "bootstrap_release_epoch": current_collection_metrics.get("bootstrap_release_epoch", -1.0),
            "bootstrap_ready_streak": current_collection_metrics.get("bootstrap_ready_streak", 0.0),
            "teacher_guidance_ready_streak": current_collection_metrics.get("teacher_guidance_ready_streak", 0.0),
            "teacher_guidance_alpha": current_collection_metrics.get("teacher_guidance_alpha", 0.0),
            "dense_teacher_supervision": current_collection_metrics.get("dense_teacher_supervision", False),
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
            "dream_plan_loss": dream_metrics.get("dream_plan_loss", 0.0),
            "dream_seconds": dream_metrics.get("dream_seconds", 0.0),
            "secondary_train_device": secondary_train_device,
            "secondary_replay_samples": secondary_replay_samples,
            "secondary_dream_sequences": dream_metrics.get("secondary_dream_sequences", 0.0),
        }
        final_snapshot = _build_checkpoint_snapshot(
            config,
            encoder,
            world_model,
            language_model,
            history,
            holdout_history,
            final_training_state,
            last_holdout_result,
        )
        _save_checkpoint(
            checkpoint_path,
            final_snapshot,
            epoch=len(history) - 1 if history else None,
        )
    return {
        "interrupted": False,
        "epochs": float(config.epochs),
        "epochs_completed": len(history),
        "stage_index": current_stage_index,
        "stage_name": str(current_collection_metrics.get("stage_name", "flat" if config.curriculum == "flat" else stages[current_stage_index].name if stages else "")),
        "samples_last_epoch": history[-1]["samples"] if history else 0.0,
        "loss_last_epoch": history[-1]["loss"] if history else 0.0,
        "uncertainty_last_epoch": history[-1]["uncertainty"] if history else 0.0,
        "last_holdout_result": last_holdout_result or {},
        "collector_agent": history[-1]["collector_agent"] if history else "",
        "secondary_train_device": history[-1]["secondary_train_device"] if history else "",
        "secondary_replay_samples": history[-1]["secondary_replay_samples"] if history else 0.0,
        "secondary_dream_sequences": history[-1]["secondary_dream_sequences"] if history else 0.0,
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
    defaults = SyntheticTrainingConfig()
    parser = argparse.ArgumentParser(prog="python -m arcagi.training.synthetic")
    parser.add_argument("--epochs", type=int, default=defaults.epochs)
    parser.add_argument("--episodes-per-epoch", type=int, default=defaults.episodes_per_epoch)
    parser.add_argument("--learning-rate", type=float, default=defaults.learning_rate)
    parser.add_argument("--checkpoint-path", type=str, default=defaults.checkpoint_path)
    parser.add_argument("--init-checkpoint-path", type=str, default="")
    parser.add_argument("--resume-checkpoint-path", type=str, default="")
    parser.add_argument(
        "--allow-weights-only-init-from-training-checkpoint",
        action="store_true",
        help="allow init-checkpoint-path to load weights from a checkpoint that also contains saved training state",
    )
    parser.add_argument("--device", type=str, default=defaults.train_device)
    parser.add_argument(
        "--async-eval-device",
        type=str,
        default=defaults.async_eval_device,
        help="secondary device for mirrored replay/dream training when CUDA is available; otherwise async holdout fallback",
    )
    parser.add_argument(
        "--secondary-device",
        type=str,
        default=defaults.async_eval_device,
        help="alias for --async-eval-device",
    )
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument(
        "--behavior-policy",
        type=str,
        default=defaults.behavior_policy,
        choices=sorted(_ALLOWED_SYNTHETIC_BEHAVIOR_POLICIES),
    )
    parser.add_argument("--curriculum", type=str, default=defaults.curriculum, choices=sorted(_ALLOWED_CURRICULA))
    parser.add_argument("--oracle-imitation-epochs", type=int, default=defaults.oracle_imitation_epochs)
    parser.add_argument("--oracle-bootstrap-steps", type=int, default=defaults.oracle_bootstrap_steps)
    parser.add_argument("--oracle-bootstrap-stride", type=int, default=defaults.oracle_bootstrap_stride)
    parser.add_argument("--oracle-bootstrap-min-steps", type=int, default=defaults.oracle_bootstrap_min_steps)
    parser.add_argument("--oracle-bootstrap-min-stride", type=int, default=defaults.oracle_bootstrap_min_stride)
    parser.add_argument("--oracle-bootstrap-full-epochs", type=int, default=defaults.oracle_bootstrap_full_epochs)
    parser.add_argument("--oracle-bootstrap-decay-epochs", type=int, default=defaults.oracle_bootstrap_decay_epochs)
    parser.add_argument(
        "--oracle-bootstrap-decay-success-threshold",
        type=float,
        default=defaults.oracle_bootstrap_decay_success_threshold,
    )
    parser.add_argument(
        "--oracle-bootstrap-decay-stability-epochs",
        type=int,
        default=defaults.oracle_bootstrap_decay_stability_epochs,
    )
    parser.add_argument(
        "--teacher-guidance-holdout-success-threshold",
        type=float,
        default=defaults.teacher_guidance_holdout_success_threshold,
    )
    parser.add_argument(
        "--teacher-episode-fraction-initial",
        type=float,
        default=defaults.teacher_episode_fraction_initial,
    )
    parser.add_argument(
        "--teacher-episode-fraction-floor",
        type=float,
        default=defaults.teacher_episode_fraction_floor,
    )
    parser.add_argument(
        "--teacher-takeover-prob-initial",
        type=float,
        default=defaults.teacher_takeover_prob_initial,
    )
    parser.add_argument(
        "--teacher-takeover-prob-floor",
        type=float,
        default=defaults.teacher_takeover_prob_floor,
    )
    parser.add_argument("--teacher-relabel-weight", type=float, default=defaults.teacher_relabel_weight)
    parser.add_argument("--teacher-ownership-window", type=int, default=defaults.teacher_ownership_window)
    parser.add_argument("--teacher-agreement-target", type=float, default=defaults.teacher_agreement_target)
    parser.add_argument("--teacher-success-target", type=float, default=defaults.teacher_success_target)
    parser.add_argument("--trajectory-credit-discount", type=float, default=defaults.trajectory_credit_discount)
    parser.add_argument("--dream-batches-per-epoch", type=int, default=defaults.dream_batches_per_epoch)
    parser.add_argument("--dream-batch-size", type=int, default=defaults.dream_batch_size)
    parser.add_argument("--dream-horizon", type=int, default=defaults.dream_horizon)
    parser.add_argument("--dream-loss-weight", type=float, default=defaults.dream_loss_weight)
    parser.add_argument("--dream-belief-weight", type=float, default=defaults.dream_belief_weight)
    parser.add_argument("--dream-question-weight", type=float, default=defaults.dream_question_weight)
    parser.add_argument("--dream-plan-weight", type=float, default=defaults.dream_plan_weight)
    parser.add_argument("--theory-loss-weight", type=float, default=defaults.theory_loss_weight)
    parser.add_argument(
        "--diagnostic-language-loss-weight",
        type=float,
        default=defaults.diagnostic_language_loss_weight,
    )
    parser.add_argument("--log-every-episodes", type=int, default=defaults.log_every_episodes)
    parser.add_argument("--holdout-eval-every-epochs", type=int, default=defaults.holdout_eval_every_epochs)
    parser.add_argument(
        "--holdout-episodes-per-variant",
        type=int,
        default=defaults.holdout_episodes_per_variant,
    )
    parser.add_argument(
        "--regression-holdout-every-evals",
        type=int,
        default=defaults.regression_holdout_every_evals,
    )
    parser.add_argument(
        "--promotion-consecutive-evals",
        type=int,
        default=defaults.promotion_consecutive_evals,
    )
    parser.add_argument("--frontier-replay-weight", type=int, default=defaults.frontier_replay_weight)
    parser.add_argument(
        "--previous-stage-replay-weight",
        type=int,
        default=defaults.previous_stage_replay_weight,
    )
    parser.add_argument("--holdout-failure-examples", type=int, default=defaults.holdout_failure_examples)
    parser.add_argument("--holdout-trace-steps", type=int, default=defaults.holdout_trace_steps)
    parser.add_argument("--holdout-trace-top-actions", type=int, default=defaults.holdout_trace_top_actions)
    parser.add_argument("--enhancement-gate-target", type=float, default=defaults.enhancement_gate_target)
    parser.add_argument("--enhancement-gate-weight", type=float, default=defaults.enhancement_gate_weight)
    return parser


def main() -> int:
    mp.freeze_support()
    parser = build_parser()
    args = parser.parse_args()
    secondary_device = args.secondary_device or args.async_eval_device
    config = SyntheticTrainingConfig(
        epochs=args.epochs,
        episodes_per_epoch=args.episodes_per_epoch,
        learning_rate=args.learning_rate,
        checkpoint_path=args.checkpoint_path,
        init_checkpoint_path=args.init_checkpoint_path,
        resume_checkpoint_path=args.resume_checkpoint_path,
        allow_weights_only_init_from_training_checkpoint=args.allow_weights_only_init_from_training_checkpoint,
        train_device=args.device,
        async_eval_device=secondary_device,
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
        teacher_ownership_window=args.teacher_ownership_window,
        teacher_agreement_target=args.teacher_agreement_target,
        teacher_success_target=args.teacher_success_target,
        trajectory_credit_discount=args.trajectory_credit_discount,
        dream_batches_per_epoch=args.dream_batches_per_epoch,
        dream_batch_size=args.dream_batch_size,
        dream_horizon=args.dream_horizon,
        dream_loss_weight=args.dream_loss_weight,
        dream_belief_weight=args.dream_belief_weight,
        dream_question_weight=args.dream_question_weight,
        dream_plan_weight=args.dream_plan_weight,
        theory_loss_weight=args.theory_loss_weight,
        diagnostic_language_loss_weight=args.diagnostic_language_loss_weight,
        log_every_episodes=args.log_every_episodes,
        holdout_eval_every_epochs=args.holdout_eval_every_epochs,
        holdout_episodes_per_variant=args.holdout_episodes_per_variant,
        regression_holdout_every_evals=args.regression_holdout_every_evals,
        promotion_consecutive_evals=args.promotion_consecutive_evals,
        frontier_replay_weight=args.frontier_replay_weight,
        previous_stage_replay_weight=args.previous_stage_replay_weight,
        holdout_failure_examples=args.holdout_failure_examples,
        holdout_trace_steps=args.holdout_trace_steps,
        holdout_trace_top_actions=args.holdout_trace_top_actions,
        enhancement_gate_target=args.enhancement_gate_target,
        enhancement_gate_weight=args.enhancement_gate_weight,
    )
    metrics = train_synthetic(config, device=torch.device(args.device) if args.device else None)
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
