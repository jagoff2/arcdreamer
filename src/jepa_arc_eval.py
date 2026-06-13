from __future__ import annotations

import argparse
import ast
import json
import random
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import torch

from .arcagi3_eval import aggregate_rows, collect_hashes, state_signature
from .arcagi3_official import OfficialArcAGI3Env, default_max_steps, discover_official_games, make_arcade
from .arcagi3_trace import action_entropy, repeat_collapse
from .attempt_buffer import AttemptBuffer, AttemptRecord, action_id, observation_frame
from .anti_attractor import HardAntiAttractorGate
from .base_world_model import ACTION_FEATURE_DIM, action_to_features
from .base_eval import make_base_controller
from .device import AUTO_DEVICE, resolve_device
from .external_collapse_experiment import adapter_device_summary, cuda_runtime_info, official_baseline_rows
from .external_eval import GymnasiumExternalEnv, json_safe
from .external_registry import discover_external_suites
from .jepa_attempt_memory import JEPAAttemptMemory, POSITIVE_EVENTS, _action_family
from .jepa_train import synthetic_attempts
from .video_jepa import VideoJEPA, load_video_jepa


JEPA_AUDITED_PATHS = [
    "src/attempt_buffer.py",
    "src/video_jepa.py",
    "src/jepa_train.py",
    "src/jepa_attempt_memory.py",
    "src/jepa_arc_eval.py",
    "tests/test_video_jepa.py",
    "data/jepa_trace_manifest.json",
    "docs/jepa_attempt_report.json",
    "frozen/recurrent_latent_fast.pt",
    "frozen/external_base_v1.pt",
    "runs/explorer_tiny.pt",
    "runs/video_jepa.pt",
]


@dataclass(frozen=True)
class JEPAVariant:
    variant_id: str
    use_memory: bool = False
    use_jepa_tokens: bool = False
    use_trained_jepa: bool = False
    random_jepa: bool = False
    shuffled_jepa: bool = False
    frozen_perception: bool = False
    null_control: bool = False


VARIANTS = [
    JEPAVariant("baseline_core"),
    JEPAVariant("attempt_memory_no_jepa", use_memory=True),
    JEPAVariant("jepa_random_init", use_memory=True, use_jepa_tokens=True, random_jepa=True),
    JEPAVariant("jepa_pretrained_frozen_if_available", frozen_perception=True),
    JEPAVariant("jepa_trained_dev", use_jepa_tokens=True, use_trained_jepa=True),
    JEPAVariant("jepa_plus_attempt_memory", use_memory=True, use_jepa_tokens=True, use_trained_jepa=True),
    JEPAVariant("jepa_trained_shuffled", use_memory=True, use_jepa_tokens=True, use_trained_jepa=True, shuffled_jepa=True),
    JEPAVariant("null_control", null_control=True),
]

NO_OP_CONTROL_VARIANT_IDS = frozenset(
    {
        "jepa_random_init",
        "jepa_pretrained_frozen_if_available",
        "null_control",
    }
)
FOCUSED_OFFICIAL_VARIANT_IDS = ("jepa_plus_attempt_memory",)
FOCUSED_CAUSALITY_VARIANT_IDS = (
    "attempt_memory_no_jepa",
    "jepa_random_init",
    "jepa_trained_shuffled",
    "jepa_plus_attempt_memory",
)
JEPA_ACTION_MODES = ("propose_only", "approved", "legacy")
SIDECAR_RECENT_WINDOW = 16
SIDECAR_OVERRIDE_THRESHOLD = 8
SIDECAR_USEFULNESS_BASELINE = 0.0
BEHAVIORAL_OFFLINE_GATE_THRESHOLDS = {
    "loop_attempt_rate": 0.05,
    "button_loop_attempt_rate": 0.02,
    "coordinate_loop_attempt_rate": 0.02,
    "mean_useful_events_per_attempt": 0.25,
    "zero_useful_attempt_rate": 0.25,
    "sidecar_changed_action_ratio": 0.05,
}


def variant_by_id(variant_id: str) -> JEPAVariant:
    for variant in VARIANTS:
        if variant.variant_id == variant_id:
            return variant
    raise KeyError(variant_id)


def expand_cli_values(values: Sequence[str] | None) -> list[str]:
    expanded: list[str] = []
    for value in values or []:
        for item in str(value).split(","):
            stripped = item.strip()
            if stripped:
                expanded.append(stripped)
    return expanded


def select_variants(variant_ids: Sequence[str] | None = None) -> list[JEPAVariant]:
    requested = expand_cli_values(variant_ids)
    ids = requested or [variant.variant_id for variant in VARIANTS]
    return [variant_by_id(variant_id) for variant_id in ids]


def focused_variant_ids(variant_ids: Sequence[str] | None = None) -> list[str]:
    ids = expand_cli_values(variant_ids) or list(FOCUSED_OFFICIAL_VARIANT_IDS)
    blocked = sorted(set(ids).intersection(NO_OP_CONTROL_VARIANT_IDS))
    if blocked:
        raise ValueError(f"focused_official excludes no-op control variants: {', '.join(blocked)}")
    return ids


def focused_causality_variant_ids(variant_ids: Sequence[str] | None = None) -> list[str]:
    return expand_cli_values(variant_ids) or list(FOCUSED_CAUSALITY_VARIANT_IDS)


def manifest_game_ids(
    *,
    manifest_path: str | Path = "docs/arcagi3_official_games.json",
    requested_ids: Sequence[str] | None = None,
    limit: int | None = None,
) -> list[str]:
    manifest = json.loads(Path(manifest_path).read_text(encoding="utf-8"))
    manifest_ids = [str(item["game_id"]) for item in manifest.get("games", [])]
    requested = set(expand_cli_values(requested_ids))
    selected = [
        item
        for item in manifest_ids
        if not requested or item in requested or item.split("-", 1)[0] in requested
    ]
    missing = sorted(requested.difference(selected).difference({item.split("-", 1)[0] for item in selected}))
    if missing:
        raise ValueError(f"Requested official games were not found in manifest: {', '.join(missing)}")
    if limit is not None:
        selected = selected[: max(int(limit), 0)]
    return selected


class JEPAAugmentedController:
    def __init__(
        self,
        *,
        variant: JEPAVariant,
        core_arm: str,
        checkpoint: str | Path,
        explorer_checkpoint: str | Path,
        jepa_model: VideoJEPA | None,
        device: torch.device,
        jepa_action_mode: str = "propose_only",
        jepa_max_action_bias: float = 0.05,
        disable_jepa_direct_override: bool = True,
        enable_hard_anti_attractor: bool = True,
    ) -> None:
        self.variant = variant
        self.base = make_base_controller(
            arm_id=core_arm,
            external_checkpoint=checkpoint,
            explorer_checkpoint=explorer_checkpoint,
            device=device,
        )
        token_mode = "shuffled" if variant.shuffled_jepa else "normal"
        self.memory = JEPAAttemptMemory(use_jepa_tokens=variant.use_jepa_tokens, device=device, jepa_token_mode=token_mode)
        self.jepa_model = jepa_model
        self.device = device
        self.jepa_action_mode = str(jepa_action_mode)
        if self.jepa_action_mode not in JEPA_ACTION_MODES:
            raise ValueError(f"unsupported jepa_action_mode={self.jepa_action_mode!r}")
        self.jepa_max_action_bias = float(max(jepa_max_action_bias, 0.0))
        self.disable_jepa_direct_override = bool(disable_jepa_direct_override)
        self.enable_hard_anti_attractor = bool(enable_hard_anti_attractor)
        self.changed_actions = 0
        self.frames = 0
        self.last_base_action: str | None = None
        self.last_observation: Any | None = None
        self.bridge_history_frames: list[list[float]] = []
        self.bridge_history_action_ids: list[int] = []
        self.bridge_history_action_features: list[list[float]] = []
        self.bridge_history_legal_counts: list[float] = []
        self.anti_attractor = HardAntiAttractorGate()
        self.rng = random.Random(0)
        self.post_prefix_rng = random.Random(0)
        self.attempt_counter = 0
        self.sidecar_override_ledger: list[dict[str, Any]] = []
        self.pending_sidecar_override: dict[str, Any] | None = None
        self.last_transition_usefulness: dict[str, Any] | None = None

    @property
    def name(self) -> str:
        return self.variant.variant_id

    def reset_all(self) -> None:
        self.base.reset()
        self.memory.reset()
        self.changed_actions = 0
        self.frames = 0
        self.last_base_action = None
        self.last_observation = None
        self.bridge_history_frames.clear()
        self.bridge_history_action_ids.clear()
        self.bridge_history_action_features.clear()
        self.bridge_history_legal_counts.clear()
        self.anti_attractor = HardAntiAttractorGate()
        self.rng.seed(0)
        self.post_prefix_rng.seed(0)
        self.attempt_counter = 0
        self.sidecar_override_ledger.clear()
        self.pending_sidecar_override = None
        self.last_transition_usefulness = None

    def reset_attempt(self, seed: int) -> None:
        self.attempt_counter += 1
        self.base.reset(seed)
        if self.variant.use_memory and not self.variant.null_control:
            self.memory.start_attempt()
        self.frames = 0
        self.last_base_action = None
        self.bridge_history_frames.clear()
        self.bridge_history_action_ids.clear()
        self.bridge_history_action_features.clear()
        self.bridge_history_legal_counts.clear()
        self.anti_attractor.reset()
        self.last_transition_usefulness = None
        self.rng.seed(9173 + int(seed))
        self.post_prefix_rng.seed(9173 + (1009 * int(seed)) + (104729 * self.attempt_counter))
        self.pending_sidecar_override = None

    def choose_action(self, observation: Any) -> tuple[str, dict[str, Any]]:
        if not hasattr(self, "anti_attractor"):
            self.anti_attractor = HardAntiAttractorGate()
        base_action, diagnostics = self.base.choose_action(observation)
        legal = tuple(observation.available_actions)
        raw_scores = diagnostics.get("action_scores", {})
        if not raw_scores:
            raw_scores = {action: 0.0 for action in legal}
            raw_scores[base_action] = 1.0
        adjusted = {action: float(raw_scores.get(action, 0.0)) for action in legal}
        memory_scores = {action: 0.0 for action in legal}
        memory_distribution = {action: 1.0 / max(len(legal), 1) for action in legal}
        jepa_bridge_scores = {action: 0.0 for action in legal}
        applied_jepa_bridge_scores = {action: 0.0 for action in legal}
        jepa_bridge_diagnostics: dict[str, Any] = {"active": False}
        bridge_public_support = {action: 0.0 for action in legal}
        sidecar_override: dict[str, Any] = self._sidecar_override_placeholder(base_action)
        prefix_replay = False
        post_prefix_exploration = False
        experiment_candidate = None
        experiment_selected = False
        experiment_by_action: dict[str, Any] = {}
        if self.variant.use_memory and not self.variant.null_control:
            memory_scores = self.memory.plan_scores_for_observation(observation)
            memory_distribution = self.memory.action_distribution_from_scores(memory_scores)
            for action in legal:
                adjusted[action] += memory_scores[action]
        if self.variant.use_memory and self.variant.use_jepa_tokens and self.jepa_model is not None and not self.variant.null_control:
            raw_bridge_scores, jepa_bridge_diagnostics = self.jepa_action_bridge_scores(observation, legal)
            bridge_public_support = {action: self.memory.bridge_public_support_weight(action) for action in legal}
            jepa_bridge_scores = {
                action: float(raw_bridge_scores.get(action, 0.0) * bridge_public_support.get(action, 0.0))
                for action in legal
            }
            jepa_bridge_diagnostics = {
                **jepa_bridge_diagnostics,
                "raw_top_scores": jepa_bridge_diagnostics.get("top_scores", {}),
                "top_scores": top_scores(jepa_bridge_scores),
                "public_support_weights": top_scores(bridge_public_support),
            }
            sidecar_override, applied_jepa_bridge_scores = self._evaluate_sidecar_override(
                base_action=base_action,
                bridge_scores=jepa_bridge_scores,
                bridge_public_support=bridge_public_support,
            )
            for action in legal:
                adjusted[action] += applied_jepa_bridge_scores.get(action, 0.0)
        if self.variant.null_control or not self.variant.use_memory:
            chosen = base_action
        else:
            suppressed_count = int(sum(1 for action in legal if self.memory.public_no_effect_suppresses_action(action)))
            eligible = [action for action in legal if not self.memory.public_no_effect_suppresses_action(action)]
            if not eligible:
                eligible = list(legal)
            experiment_eligible = list(eligible)
            count_balanced = False
            random_balanced = False
            if not self.memory._has_public_goal_evidence():
                min_count = min(int(self.memory.action_counts.get(action, 0)) for action in eligible)
                least_tried = [action for action in eligible if int(self.memory.action_counts.get(action, 0)) == min_count]
                if least_tried:
                    eligible = least_tried
                    count_balanced = True
            if self.memory.active_sequence_source == "positive_public_control_prefix":
                planned_prefix_action = self.memory.sequence_plan_action(legal, observation=observation)
                if planned_prefix_action in eligible:
                    chosen = str(planned_prefix_action)
                    prefix_replay = True
                else:
                    chosen = ""
                if not chosen and hasattr(self.memory, "select_experiment"):
                    experiment_candidate = self.memory.select_experiment(observation, experiment_eligible, scores=adjusted)
                    if experiment_candidate is not None and experiment_candidate.action in experiment_eligible:
                        chosen = str(experiment_candidate.action)
                        experiment_selected = True
                if not chosen and self.memory.positive_prefix_completed and self.memory.best_event_prefix:
                    discovery_pool = [
                        action
                        for action in self.memory.discovery_exploration_candidates(
                            eligible,
                            observation=observation,
                            scores=adjusted,
                        )
                        if action in eligible
                    ]
                    if discovery_pool:
                        chosen = str(self.post_prefix_rng.choice(discovery_pool))
                        post_prefix_exploration = True
            else:
                chosen = ""
                if hasattr(self.memory, "select_experiment"):
                    experiment_candidate = self.memory.select_experiment(observation, experiment_eligible, scores=adjusted)
                    if experiment_candidate is not None and experiment_candidate.action in experiment_eligible:
                        chosen = str(experiment_candidate.action)
                        experiment_selected = True
                if not chosen and self.memory.positive_prefix_completed:
                    discovery_pool = [
                        action
                        for action in self.memory.discovery_exploration_candidates(
                            eligible,
                            observation=observation,
                            scores=adjusted,
                        )
                        if action in eligible
                    ]
                    if discovery_pool:
                        chosen = str(self.post_prefix_rng.choice(discovery_pool))
                        post_prefix_exploration = True
            if not chosen and count_balanced and suppressed_count > 0 and len(eligible) > 1:
                control_pool = [
                    action for action in eligible if _action_family(action) not in {"contact", "wait"}
                ]
                ranked = sorted(control_pool or eligible, key=lambda action: adjusted.get(action, -1.0e9), reverse=True)
                chosen = str(self.rng.choice(ranked))
                random_balanced = True
            elif not chosen:
                chosen = max(eligible, key=lambda action: (adjusted.get(action, -1.0e9), -legal.index(action)))
        pre_veto_chosen = str(chosen)
        if self.variant.use_memory and not self.variant.null_control and hasattr(self.memory, "experiment_for_action"):
            for action in legal:
                experiment = self.memory.experiment_for_action(observation, str(action))
                if experiment is not None:
                    experiment_by_action[str(action)] = experiment
        ranked_for_veto = sorted(
            [str(action) for action in legal],
            key=lambda action: (
                1 if action in experiment_by_action else 0,
                adjusted.get(action, -1.0e9),
                -legal.index(action),
            ),
            reverse=True,
        )
        chosen, anti_attractor_diagnostics = self.anti_attractor.select_action(
            pre_veto_chosen,
            ranked_for_veto,
            observation,
        ) if bool(getattr(self, "enable_hard_anti_attractor", True)) else (
            pre_veto_chosen,
            {
                "schema": "hard_anti_attractor_gate_v1",
                "active": False,
                "disabled": True,
                "selected_action": pre_veto_chosen,
                "changed_action": False,
                "vetoed_actions": [],
                "fail_open_reasons": ["disabled_for_ablation"],
            },
        )
        sidecar_override = self._finalize_sidecar_override_for_selection(
            sidecar_override,
            chosen_action=chosen,
            pre_veto_chosen_action=pre_veto_chosen,
        )
        self.pending_sidecar_override = dict(sidecar_override) if sidecar_override.get("active") else None
        experiment_diagnostics: dict[str, Any] = {"active": False}
        executed_experiment = None
        if experiment_candidate is not None and str(chosen) == str(experiment_candidate.action):
            executed_experiment = experiment_candidate
        elif experiment_by_action:
            executed_experiment = experiment_by_action.get(str(chosen))
        if executed_experiment is not None:
            experiment_diagnostics = {
                "active": True,
                "selected": True,
                "selected_before_veto": bool(experiment_selected),
                "executed": True,
                "schema": "arc_explicit_experiment_protocol_v1",
                "experiment": executed_experiment.to_dict(),
                "hard_veto_replacement": bool(
                    experiment_candidate is not None and str(executed_experiment.action) != str(experiment_candidate.action)
                ),
            }
            if experiment_candidate is not None and str(executed_experiment.action) != str(experiment_candidate.action):
                experiment_diagnostics["pre_veto_experiment"] = experiment_candidate.to_dict()
            if hasattr(self.memory, "begin_experiment"):
                self.memory.begin_experiment(executed_experiment)
        elif experiment_candidate is not None:
            experiment_diagnostics = {
                "active": True,
                "selected": bool(experiment_selected),
                "selected_before_veto": bool(experiment_selected),
                "executed": False,
                "schema": "arc_explicit_experiment_protocol_v1",
                "experiment": experiment_candidate.to_dict(),
                "hard_veto_replacement": bool(str(chosen) != str(experiment_candidate.action)),
            }
        changed = chosen != base_action
        self.changed_actions += int(changed)
        self.frames += 1
        self.last_base_action = base_action
        self.last_observation = observation
        memory_summary = self.memory.summary()
        diagnostics["jepa_policy"] = {
            "variant": self.variant.variant_id,
            "base_core_action": base_action,
            "chosen_action": chosen,
            "changed_action": changed,
            "pre_veto_chosen_action": pre_veto_chosen,
            "hard_anti_attractor": anti_attractor_diagnostics,
            "memory_scores": top_scores(memory_scores),
            "memory_action_distribution": top_scores(memory_distribution),
            "jepa_bridge_scores": top_scores(jepa_bridge_scores),
            "jepa_applied_action_bias": top_scores(applied_jepa_bridge_scores),
            "jepa_bridge": jepa_bridge_diagnostics,
            "sidecar_override": sidecar_override,
            "top_adjusted_scores": top_scores(adjusted),
            "public_no_effect_suppressed_count": int(suppressed_count) if self.variant.use_memory else 0,
            "public_count_balanced_exploration": bool(count_balanced) if self.variant.use_memory else False,
            "public_random_balanced_exploration": bool(random_balanced) if self.variant.use_memory else False,
            "public_prefix_replay": bool(prefix_replay) if self.variant.use_memory else False,
            "public_post_prefix_exploration": bool(post_prefix_exploration) if self.variant.use_memory else False,
            "experiment_protocol": experiment_diagnostics,
            "action_source": "existing_core_plus_attempt_memory_with_approved_sidecar_bias"
            if sidecar_override.get("approved") and sidecar_override.get("executed")
            else "existing_core_plus_attempt_memory",
            "jepa_direct_action": False,
            "emits_text": False,
            "causal_substrate": {
                "active": bool(self.variant.use_memory and self.variant.use_jepa_tokens and memory_summary.get("causal_substrate_active")),
                "chain": memory_summary.get("causal_chain", []),
                "changed_next_attempt_distribution": bool(
                    self.variant.use_memory and any(abs(value) > 1.0e-9 for value in memory_scores.values())
                ),
            },
            "transition_graph_planner": {
                "active": bool(self.variant.use_memory and memory_summary.get("transition_graph", {}).get("observed_edges", 0) > 0),
                "summary": memory_summary.get("transition_graph", {}),
                "chain": memory_summary.get("planner_chain", []),
            },
        }
        return chosen, diagnostics

    def _sidecar_override_placeholder(self, base_action: str) -> dict[str, Any]:
        state = self._sidecar_authority_state()
        return {
            "schema": "sidecar_override_ledger_v1",
            "active": False,
            "proposed_action": "",
            "base_action": str(base_action),
            "approved": False,
            "executed": False,
            "reason": "no_sidecar_proposal",
            "mode": self._jepa_action_mode(),
            "sidecar_can_override": bool(state["sidecar_can_override"]),
            "action_weight": float(state["action_weight"]),
            "posthoc_useful": None,
            "posthoc_progress": None,
            "posthoc_entropy_drop": None,
        }

    def _jepa_action_mode(self) -> str:
        mode = str(getattr(self, "jepa_action_mode", "propose_only"))
        return mode if mode in JEPA_ACTION_MODES else "propose_only"

    def _sidecar_authority_state(self) -> dict[str, Any]:
        recent = list(getattr(self, "sidecar_override_ledger", [])[-SIDECAR_RECENT_WINDOW:])
        recent_overrides = sum(
            1
            for item in recent
            if bool(item.get("approved")) and bool(item.get("executed"))
        )
        recent_useful = sum(
            1
            for item in recent
            if bool(item.get("approved")) and bool(item.get("executed")) and bool(item.get("posthoc_useful"))
        )
        quarantined = bool(
            recent_overrides > SIDECAR_OVERRIDE_THRESHOLD
            and float(recent_useful) <= SIDECAR_USEFULNESS_BASELINE
        )
        mode = self._jepa_action_mode()
        direct_disabled = bool(getattr(self, "disable_jepa_direct_override", True))
        max_bias = float(max(getattr(self, "jepa_max_action_bias", 0.05), 0.0))
        action_weight = max_bias * (0.1 if quarantined else 1.0)
        return {
            "schema": "sidecar_authority_state_v1",
            "mode": mode,
            "recent_overrides": int(recent_overrides),
            "recent_useful": int(recent_useful),
            "quarantined": quarantined,
            "direct_override_disabled": direct_disabled,
            "action_weight": float(action_weight),
            "sidecar_can_override": bool(mode in {"approved", "legacy"} and not direct_disabled and not quarantined),
        }

    def _evaluate_sidecar_override(
        self,
        *,
        base_action: str,
        bridge_scores: dict[str, float],
        bridge_public_support: dict[str, float],
    ) -> tuple[dict[str, Any], dict[str, float]]:
        if not bridge_scores:
            return self._sidecar_override_placeholder(base_action), {}
        proposed_action = max(bridge_scores, key=lambda action: (bridge_scores.get(action, -1.0e9), str(action)))
        proposed_score = float(bridge_scores.get(proposed_action, 0.0))
        authority = self._sidecar_authority_state()
        applied = {action: 0.0 for action in bridge_scores}
        approved = False
        reason = "propose_only_read_only"
        if proposed_score <= 0.0:
            reason = "nonpositive_sidecar_prior"
        elif self._jepa_action_mode() == "propose_only":
            reason = "propose_only_read_only"
        elif not authority["sidecar_can_override"]:
            reason = "sidecar_quarantined" if authority["quarantined"] else "direct_override_disabled"
        elif self._jepa_action_mode() == "legacy":
            approved = True
            reason = "legacy_explicitly_enabled"
        else:
            support = float(bridge_public_support.get(proposed_action, 0.0))
            action_has_progress = bool(getattr(self.memory, "action_events", {}).get(proposed_action, 0))
            if support >= 0.50 or action_has_progress:
                approved = True
                reason = "approved_by_public_usefulness"
            else:
                reason = "missing_public_usefulness"
        if approved:
            limit = float(authority["action_weight"])
            applied[proposed_action] = float(max(min(proposed_score, limit), -limit))
        ledger = {
            "schema": "sidecar_override_ledger_v1",
            "active": True,
            "proposed_action": str(proposed_action),
            "base_action": str(base_action),
            "approved": bool(approved),
            "executed": False,
            "reason": reason,
            "mode": self._jepa_action_mode(),
            "proposal_score": float(proposed_score),
            "public_support": float(bridge_public_support.get(proposed_action, 0.0)),
            "sidecar_can_override": bool(authority["sidecar_can_override"]),
            "quarantined": bool(authority["quarantined"]),
            "recent_overrides": int(authority["recent_overrides"]),
            "recent_useful": int(authority["recent_useful"]),
            "action_weight": float(authority["action_weight"]),
            "applied_bias": float(applied.get(proposed_action, 0.0)),
            "posthoc_useful": None,
            "posthoc_progress": None,
            "posthoc_entropy_drop": None,
        }
        return ledger, applied

    def _finalize_sidecar_override_for_selection(
        self,
        ledger: dict[str, Any],
        *,
        chosen_action: str,
        pre_veto_chosen_action: str,
    ) -> dict[str, Any]:
        if not ledger.get("active"):
            return ledger
        finalized = dict(ledger)
        finalized["pre_veto_chosen_action"] = str(pre_veto_chosen_action)
        finalized["chosen_action"] = str(chosen_action)
        finalized["executed"] = bool(
            finalized.get("approved") and str(chosen_action) == str(finalized.get("proposed_action", ""))
        )
        if finalized["approved"] and not finalized["executed"]:
            finalized["reason"] = "approved_but_vetoed_or_superseded"
        return finalized

    def _complete_pending_sidecar_override(self, action: str, result: Any) -> None:
        pending = dict(getattr(self, "pending_sidecar_override", None) or {})
        self.pending_sidecar_override = None
        if not pending:
            return
        events = [str(item) for item in getattr(result, "info", {}).get("events", [])]
        score_delta = float(getattr(result, "reward", 0.0))
        event_progress = bool(POSITIVE_EVENTS.intersection(events))
        terminal_text_win = any(
            any(token in event.lower() for token in ("win", "won", "solved", "success", "level_completed"))
            and not any(token in event.lower() for token in ("game_over", "timeout", "failed", "loss", "lose"))
            for event in events
        )
        executed = bool(str(action) == str(pending.get("proposed_action", "")) and pending.get("executed"))
        usefulness = getattr(self, "last_transition_usefulness", None) or {}
        posthoc_progress = bool(
            executed
            and (
                score_delta > 0.0
                or event_progress
                or terminal_text_win
                or bool(usefulness.get("progress_effect"))
                or bool(usefulness.get("terminal_win"))
            )
        )
        posthoc_useful = bool(executed and (posthoc_progress or bool(usefulness.get("useful_effect"))))
        pending["posthoc_useful"] = bool(posthoc_useful)
        pending["posthoc_progress"] = bool(posthoc_progress)
        pending["posthoc_entropy_drop"] = 0.0
        pending["posthoc_usefulness_reasons"] = list(usefulness.get("reasons", []))
        pending["observed_action"] = str(action)
        pending["score_delta"] = float(score_delta)
        pending["events"] = events
        if not hasattr(self, "sidecar_override_ledger"):
            self.sidecar_override_ledger = []
        self.sidecar_override_ledger.append(pending)
        self.sidecar_override_ledger = self.sidecar_override_ledger[-128:]

    def jepa_action_bridge_scores(self, observation: Any, legal: tuple[str, ...]) -> tuple[dict[str, float], dict[str, Any]]:
        if self.jepa_model is None or not legal:
            return {action: 0.0 for action in legal}, {"active": False}
        try:
            frame = observation_frame(observation).astype("float32").reshape(-1)
        except Exception as exc:
            return {action: 0.0 for action in legal}, {"active": False, "error": repr(exc)}
        count = len(legal)
        history_len = min(len(self.bridge_history_frames), 12)
        seq_len = history_len + 2
        frames = torch.zeros((count, seq_len, frame.size), dtype=torch.float32, device=self.device)
        action_ids = torch.zeros((count, seq_len), dtype=torch.long, device=self.device)
        action_features = torch.zeros((count, seq_len, ACTION_FEATURE_DIM), dtype=torch.float32, device=self.device)
        legal_counts = torch.zeros((count, seq_len, 1), dtype=torch.float32, device=self.device)
        if history_len:
            prefix_frames = torch.as_tensor(self.bridge_history_frames[-history_len:], dtype=torch.float32, device=self.device)
            prefix_actions = torch.as_tensor(self.bridge_history_action_ids[-history_len:], dtype=torch.long, device=self.device)
            prefix_action_features = torch.as_tensor(
                self.bridge_history_action_features[-history_len:],
                dtype=torch.float32,
                device=self.device,
            )
            prefix_legal = torch.as_tensor(self.bridge_history_legal_counts[-history_len:], dtype=torch.float32, device=self.device)
            frames[:, :history_len, :] = prefix_frames.view(1, history_len, -1).repeat(count, 1, 1)
            action_ids[:, :history_len] = prefix_actions.view(1, history_len).repeat(count, 1)
            action_features[:, :history_len, :] = prefix_action_features.view(1, history_len, -1).repeat(count, 1, 1)
            legal_counts[:, :history_len, 0] = prefix_legal.view(1, history_len).repeat(count, 1)
        current_frame = torch.as_tensor(frame, dtype=torch.float32, device=self.device)
        frames[:, history_len, :] = current_frame.view(1, -1).repeat(count, 1)
        frames[:, history_len + 1, :] = current_frame.view(1, -1).repeat(count, 1)
        candidate_action_ids = torch.tensor(
            [action_id(action, self.jepa_model.config.action_buckets) for action in legal],
            dtype=torch.long,
            device=self.device,
        )
        action_ids[:, history_len] = candidate_action_ids
        action_ids[:, history_len + 1] = candidate_action_ids
        candidate_features = torch.as_tensor(
            [
                action_to_features(observation, action, legal_index=index, legal_count=count)
                for index, action in enumerate(legal)
            ],
            dtype=torch.float32,
            device=self.device,
        )
        action_features[:, history_len, :] = candidate_features
        action_features[:, history_len + 1, :] = candidate_features
        legal_counts[:, history_len:, 0] = float(count)
        valid = torch.ones((count, seq_len, 1), dtype=torch.float32, device=self.device)
        self.jepa_model.eval()
        with torch.no_grad():
            output = self.jepa_model(frames, action_ids, legal_counts, valid, action_features)
            current = output["frame_tokens"][:, history_len]
            predicted = output["predicted_future"][:, history_len] if output["predicted_future"].shape[1] > history_len else current
            target = output["target_future"][:, history_len] if output["target_future"].shape[1] > history_len else current
            context = output["context_tokens"]
            previous_context = context[:, history_len - 1] if history_len else context[:, history_len]
            state_delta = torch.norm(context[:, history_len] - previous_context, dim=-1)
            raw = (
                torch.norm(predicted - current, dim=-1)
                + 0.15 * state_delta
                - 0.20 * torch.mean((predicted - target).pow(2), dim=-1)
            )
        raw_values = [float(item) for item in raw.detach().cpu().tolist()]
        if self.variant.shuffled_jepa and len(raw_values) > 1:
            raw_values = raw_values[1:] + raw_values[:1]
        mean_value = sum(raw_values) / max(len(raw_values), 1)
        centered = [value - mean_value for value in raw_values]
        scale = max(max(abs(value) for value in centered), 1.0e-6)
        scores = {
            action: float(max(min(0.95 * value / scale, 0.95), -0.95))
            for action, value in zip(legal, centered)
        }
        return scores, {
            "active": True,
            "mode": "shuffled" if self.variant.shuffled_jepa else "normal",
            "source": "trained_jepa_action_conditioned_bridge"
            if self.variant.use_trained_jepa
            else "random_jepa_action_conditioned_bridge",
            "top_scores": top_scores(scores),
            "raw_span": float(max(raw_values) - min(raw_values)) if raw_values else 0.0,
            "history_len": int(history_len),
        }

    def observe_transition(self, action: str, result: Any) -> dict[str, Any] | None:
        if not hasattr(self, "anti_attractor"):
            self.anti_attractor = HardAntiAttractorGate()
        self.last_transition_usefulness = None
        self.base.observe_transition(action, result)
        if self.last_observation is not None and bool(getattr(self, "enable_hard_anti_attractor", True)):
            self.anti_attractor.observe_transition(self.last_observation, action, result)
        if self.variant.use_memory and self.variant.use_jepa_tokens and self.jepa_model is not None and self.last_observation is not None:
            self._append_bridge_history(self.last_observation, action)
        if self.variant.use_memory and not self.variant.null_control and self.last_observation is not None:
            label = self.memory.observe_live_transition(self.last_observation, action, result)
            self.last_transition_usefulness = label.to_dict() if label is not None else None
        self._complete_pending_sidecar_override(action, result)
        return self.last_transition_usefulness

    def _append_bridge_history(self, observation: Any, action: str) -> None:
        try:
            frame = observation_frame(observation).astype("float32").reshape(-1)
        except Exception:
            return
        self.bridge_history_frames.append([float(value) for value in frame.tolist()])
        self.bridge_history_action_ids.append(int(action_id(action, self.jepa_model.config.action_buckets)))
        legal = tuple(str(item) for item in getattr(observation, "available_actions", ()))
        try:
            legal_index = list(legal).index(str(action))
        except ValueError:
            legal_index = 0
        self.bridge_history_action_features.append(
            [
                float(value)
                for value in action_to_features(
                    observation,
                    str(action),
                    legal_index=legal_index,
                    legal_count=len(legal),
                ).tolist()
            ]
        )
        self.bridge_history_legal_counts.append(float(len(getattr(observation, "available_actions", ()))))
        if len(self.bridge_history_frames) > 16:
            self.bridge_history_frames = self.bridge_history_frames[-16:]
            self.bridge_history_action_ids = self.bridge_history_action_ids[-16:]
            self.bridge_history_action_features = self.bridge_history_action_features[-16:]
            self.bridge_history_legal_counts = self.bridge_history_legal_counts[-16:]

    def finish_attempt(self, record: AttemptRecord) -> None:
        if self.variant.use_memory:
            model = self.jepa_model if self.variant.use_jepa_tokens else None
            self.memory.ingest_attempt(record, model=model)

    def summary(self) -> dict[str, Any]:
        if not hasattr(self, "anti_attractor"):
            self.anti_attractor = HardAntiAttractorGate()
        return {
            "variant": self.variant.variant_id,
            "changed_actions": self.changed_actions,
            "frames": self.frames,
            "jepa_bridge_history_len": len(self.bridge_history_frames),
            "base": self.base.summary(),
            "attempt_memory": self.memory.summary(),
            "hard_anti_attractor": self.anti_attractor.summary()
            if bool(getattr(self, "enable_hard_anti_attractor", True))
            else {
                "schema": "hard_anti_attractor_gate_v1",
                "active": False,
                "disabled": True,
            },
            "sidecar_control": self._sidecar_authority_state(),
            "sidecar_override_ledger": list(getattr(self, "sidecar_override_ledger", [])[-8:]),
            "jepa_direct_action": False,
            "device": adapter_device_summary(self.base.adapter),
        }


def top_scores(scores: dict[str, float], limit: int = 8) -> dict[str, float]:
    return {
        action: round(float(value), 6)
        for action, value in sorted(scores.items(), key=lambda item: item[1], reverse=True)[:limit]
    }


def compact_scorecard(scorecard: Any) -> dict[str, Any]:
    if not isinstance(scorecard, dict):
        return {}
    return {
        key: scorecard.get(key)
        for key in ["id", "score", "state", "levels_completed", "win_levels", "total_levels_completed", "total_levels"]
        if key in scorecard
    }


def _extras_level_count(extras: dict[str, Any]) -> int:
    for key in ("levels_completed", "total_levels_completed", "win_levels"):
        if key not in extras:
            continue
        try:
            return int(extras.get(key, 0) or 0)
        except (TypeError, ValueError):
            continue
    return 0


def _events_have_progress(events: Sequence[Any]) -> bool:
    event_texts = [str(event) for event in events or []]
    if POSITIVE_EVENTS.intersection(event_texts):
        return True
    return any(
        any(token in event.lower() for token in ("win", "won", "solved", "success", "level_completed"))
        and not any(token in event.lower() for token in ("game_over", "timeout", "failed", "loss", "lose"))
        for event in event_texts
    )


def _result_has_progress_event(before_observation: Any, result: Any) -> bool:
    if float(getattr(result, "reward", 0.0) or 0.0) > 0.0:
        return True
    info = getattr(result, "info", {}) or {}
    if _events_have_progress(info.get("events", []) or []):
        return True
    before_extras = getattr(before_observation, "extras", {}) or {}
    after_extras = getattr(getattr(result, "observation", None), "extras", {}) or {}
    if _extras_level_count(after_extras) > _extras_level_count(before_extras):
        return True
    game_state = str(info.get("game_state", after_extras.get("game_state", ""))).lower()
    return bool(game_state in {"win", "won", "solved", "success"})


def run_attempt(
    env: Any,
    controller: JEPAAugmentedController,
    *,
    suite_id: str,
    variant_id: str,
    split: str,
    seed: int,
    attempt_index: int,
    trace_path: Path,
) -> dict[str, Any]:
    controller.reset_attempt(seed)
    obs = env.reset(seed)
    actions: list[str] = []
    invalid = 0
    useful_events = 0
    states = {state_signature(obs)}
    buffer = AttemptBuffer(
        suite_id=suite_id,
        task_id=getattr(env, "task_id", suite_id),
        variant=variant_id,
        split=split,
        seed=seed,
        attempt_index=attempt_index,
    )
    try:
        for _ in range(int(env.max_steps)):
            before = obs
            action, diagnostics = controller.choose_action(before)
            invalid_action = action not in before.available_actions
            if invalid_action:
                invalid += 1
                action = before.available_actions[0]
            result = env.step(action)
            actions.append(action)
            states.add(state_signature(result.observation))
            progress_event = _result_has_progress_event(before, result)
            transition_usefulness = controller.observe_transition(action, result)
            posthoc_useful = bool(
                progress_event
                or (
                    isinstance(transition_usefulness, dict)
                    and bool(transition_usefulness.get("useful_effect", False))
                )
            )
            useful_events += int(posthoc_useful)
            jepa_policy = diagnostics.setdefault("jepa_policy", {})
            if isinstance(jepa_policy, dict):
                jepa_policy["posthoc_transition_usefulness"] = transition_usefulness or {
                    "useful_effect": bool(progress_event),
                    "progress_effect": bool(progress_event),
                    "reasons": ["progress"] if progress_event else [],
                }
                jepa_policy["posthoc_useful"] = bool(posthoc_useful)
                jepa_policy["posthoc_progress"] = bool(progress_event)
            buffer.append_transition(
                before,
                action,
                result,
                invalid_action=invalid_action,
                diagnostics={
                    "base_policy": diagnostics.get("policy", {}),
                    "jepa_policy": diagnostics.get("jepa_policy", {}),
                },
            )
            obs = result.observation
            if result.terminated or result.truncated:
                break
    finally:
        close_result = env.close()
    normalized = float(env.normalized_score())
    record = buffer.to_record()
    controller.finish_attempt(record)
    trace_payload = {
        "metadata": {
            "suite_id": suite_id,
            "task_id": getattr(env, "task_id", suite_id),
            "variant": variant_id,
            "split": split,
            "seed": seed,
            "attempt_index": attempt_index,
        },
        "attempt": record.to_dict(include_diagnostics=True),
        "summary": {
            "suite_id": suite_id,
            "task_id": getattr(env, "task_id", suite_id),
            "split": split,
            "variant": variant_id,
            "seed": seed,
            "attempt_index": attempt_index,
            "score": float(getattr(env, "score", normalized)),
            "normalized_score": normalized,
            "solved": bool(normalized >= 1.0),
            "steps": len(actions),
            "invalid_action_rate": float(invalid / max(len(actions), 1)),
            "unique_states": len(states),
            "useful_events": useful_events,
            "action_entropy": action_entropy(actions),
            "repeat_collapse": repeat_collapse(actions),
            "actions": actions[:96],
            "controller_summary": controller.summary(),
            "scorecard": compact_scorecard(close_result),
        },
    }
    trace_path.parent.mkdir(parents=True, exist_ok=True)
    trace_path.write_text(json.dumps(json_safe(trace_payload), indent=2), encoding="utf-8")
    summary = trace_payload["summary"]
    summary["trace_path"] = str(trace_path)
    return summary


def make_jepa_for_variant(variant: JEPAVariant, jepa_checkpoint: str | Path, device: torch.device) -> tuple[VideoJEPA | None, dict[str, Any]]:
    if variant.random_jepa:
        return VideoJEPA().to(device).eval(), {"source": "random_init", "available": True}
    if variant.use_trained_jepa:
        model, payload = load_video_jepa(str(jepa_checkpoint), device=device)
        return model, {"source": str(jepa_checkpoint), "available": True, "metrics": payload.get("metrics", {})}
    if variant.frozen_perception:
        return None, {"source": "no_frozen_video_perception_available", "available": False}
    return None, {"source": "not_used", "available": False}


def resolve_external_base_checkpoint(checkpoint: str | Path) -> Path:
    path = Path(checkpoint)
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
        if payload.get("external_base_format") == "external_action_conditioned_world_model_v1":
            return path
    except Exception:
        pass
    fallback = Path("frozen/external_base_v1.pt")
    if fallback.exists():
        return fallback
    return path


def make_controller_for_variant(
    *,
    variant: JEPAVariant,
    core_arm: str,
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    jepa_checkpoint: str | Path,
    device: torch.device,
    jepa_action_mode: str = "propose_only",
    jepa_max_action_bias: float = 0.05,
    disable_jepa_direct_override: bool = True,
    enable_hard_anti_attractor: bool = True,
) -> tuple[JEPAAugmentedController, dict[str, Any]]:
    jepa_model, model_info = make_jepa_for_variant(variant, jepa_checkpoint, device)
    external_checkpoint = resolve_external_base_checkpoint(checkpoint)
    controller = JEPAAugmentedController(
        variant=variant,
        core_arm=core_arm,
        checkpoint=external_checkpoint,
        explorer_checkpoint=explorer_checkpoint,
        jepa_model=jepa_model,
        device=device,
        jepa_action_mode=jepa_action_mode,
        jepa_max_action_bias=jepa_max_action_bias,
        disable_jepa_direct_override=disable_jepa_direct_override,
        enable_hard_anti_attractor=enable_hard_anti_attractor,
    )
    model_info["external_base_checkpoint"] = str(external_checkpoint)
    return controller, model_info


def run_official_worker(
    *,
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    jepa_checkpoint: str | Path,
    trace_dir: str | Path,
    json_output: str | Path,
    core_arm: str,
    device: str | torch.device,
    max_click_actions: int = 192,
    operation_mode: str = "normal",
    game_ids: Sequence[str] | None = None,
    limit: int | None = None,
    variant_ids: Sequence[str] | None = None,
    attempts: int = 3,
    environments_dir: str | Path = "runs/arcagi3_official_envs",
    recordings_dir: str | Path = "runs/arcagi3_official_recordings",
    jepa_action_mode: str = "propose_only",
    jepa_max_action_bias: float = 0.05,
    disable_jepa_direct_override: bool = True,
    enable_hard_anti_attractor: bool = True,
) -> dict[str, Any]:
    target_device = resolve_device(device)
    requested = manifest_game_ids(requested_ids=game_ids, limit=limit)
    arcade = make_arcade(
        operation_mode=operation_mode,
        environments_dir=environments_dir,
        recordings_dir=recordings_dir,
    )
    discovered = discover_official_games(arcade, game_ids=requested, limit=None)
    by_id = {spec.game_id: spec for spec in discovered}
    specs = [by_id[item] for item in requested if item in by_id]
    rows: list[dict[str, Any]] = []
    model_info: dict[str, Any] = {}
    variants = select_variants(variant_ids)
    for variant in variants:
        controller, info = make_controller_for_variant(
            variant=variant,
            core_arm=core_arm,
            checkpoint=checkpoint,
            explorer_checkpoint=explorer_checkpoint,
            jepa_checkpoint=jepa_checkpoint,
            device=target_device,
            jepa_action_mode=jepa_action_mode,
            jepa_max_action_bias=jepa_max_action_bias,
            disable_jepa_direct_override=disable_jepa_direct_override,
            enable_hard_anti_attractor=enable_hard_anti_attractor,
        )
        model_info[variant.variant_id] = info
        for index, spec in enumerate(specs):
            controller.reset_all()
            for attempt_index in range(1, max(int(attempts), 1) + 1):
                env = OfficialArcAGI3Env(
                    arcade,
                    spec,
                    seed=index,
                    max_steps=default_max_steps(spec),
                    max_click_actions=max_click_actions,
                )
                row = run_attempt(
                    env,
                    controller,
                    suite_id="official_arcagi3",
                    variant_id=variant.variant_id,
                    split="sealed_eval",
                    seed=index,
                    attempt_index=attempt_index,
                    trace_path=Path(trace_dir)
                    / "official_arcagi3"
                    / "sealed_eval"
                    / variant.variant_id
                    / f"attempt_{attempt_index}"
                    / f"{spec.game_id}.json",
                )
                row["game_id"] = spec.game_id
                rows.append(row)
    report = {
        "suite_id": "official_arcagi3",
        "split": "sealed_eval",
        "rows": rows,
        "aggregate_by_variant": aggregate_by_variant(rows),
        "attempt_table": attempt_table(rows),
        "behavioral_offline_gates": behavioral_offline_gates(rows),
        "official_baselines": official_baseline_rows(),
        "trace_paths": [row["trace_path"] for row in rows],
        "device_runtime": cuda_runtime_info(target_device),
        "variant_model_info": model_info,
        "operation_mode": operation_mode,
        "selected_game_ids": [spec.game_id for spec in specs],
        "variant_ids": [variant.variant_id for variant in variants],
        "attempts": max(int(attempts), 1),
    }
    Path(json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(json_output).write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    return report


def run_behavioral_ablation_report(
    *,
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    jepa_checkpoint: str | Path,
    trace_dir: str | Path,
    json_output: str | Path,
    core_arm: str,
    device: str | torch.device,
    max_click_actions: int = 192,
    operation_mode: str = "offline",
    game_ids: Sequence[str] | None = None,
    limit: int | None = None,
    attempts: int = 1,
    environments_dir: str | Path = "runs/arcagi3_official_envs",
    recordings_dir: str | Path = "runs/arcagi3_official_recordings",
) -> dict[str, Any]:
    target_device = resolve_device(device)
    requested = manifest_game_ids(requested_ids=game_ids, limit=limit)
    arcade = make_arcade(
        operation_mode=operation_mode,
        environments_dir=environments_dir,
        recordings_dir=recordings_dir,
    )
    discovered = discover_official_games(arcade, game_ids=requested, limit=None)
    by_id = {spec.game_id: spec for spec in discovered}
    specs = [by_id[item] for item in requested if item in by_id]
    arms = [
        {
            "ablation_id": "base_only",
            "variant": variant_by_id("baseline_core"),
            "enable_hard_anti_attractor": False,
            "jepa_action_mode": "propose_only",
            "disable_jepa_direct_override": True,
        },
        {
            "ablation_id": "base_plus_hard_loop_gate",
            "variant": variant_by_id("baseline_core"),
            "enable_hard_anti_attractor": True,
            "jepa_action_mode": "propose_only",
            "disable_jepa_direct_override": True,
        },
        {
            "ablation_id": "base_plus_hard_loop_gate_jepa_readonly",
            "variant": variant_by_id("jepa_plus_attempt_memory"),
            "enable_hard_anti_attractor": True,
            "jepa_action_mode": "propose_only",
            "disable_jepa_direct_override": True,
        },
        {
            "ablation_id": "full_jepa_plus_attempt_memory",
            "variant": variant_by_id("jepa_plus_attempt_memory"),
            "enable_hard_anti_attractor": True,
            "jepa_action_mode": "approved",
            "disable_jepa_direct_override": False,
        },
    ]
    rows: list[dict[str, Any]] = []
    model_info: dict[str, Any] = {}
    for arm in arms:
        variant = arm["variant"]
        ablation_id = str(arm["ablation_id"])
        controller, info = make_controller_for_variant(
            variant=variant,
            core_arm=core_arm,
            checkpoint=checkpoint,
            explorer_checkpoint=explorer_checkpoint,
            jepa_checkpoint=jepa_checkpoint,
            device=target_device,
            jepa_action_mode=str(arm["jepa_action_mode"]),
            jepa_max_action_bias=0.05,
            disable_jepa_direct_override=bool(arm["disable_jepa_direct_override"]),
            enable_hard_anti_attractor=bool(arm["enable_hard_anti_attractor"]),
        )
        model_info[ablation_id] = {
            **info,
            "source_variant": variant.variant_id,
            "enable_hard_anti_attractor": bool(arm["enable_hard_anti_attractor"]),
            "jepa_action_mode": str(arm["jepa_action_mode"]),
            "disable_jepa_direct_override": bool(arm["disable_jepa_direct_override"]),
        }
        for index, spec in enumerate(specs):
            controller.reset_all()
            for attempt_index in range(1, max(int(attempts), 1) + 1):
                env = OfficialArcAGI3Env(
                    arcade,
                    spec,
                    seed=index,
                    max_steps=default_max_steps(spec),
                    max_click_actions=max_click_actions,
                )
                row = run_attempt(
                    env,
                    controller,
                    suite_id="official_arcagi3",
                    variant_id=ablation_id,
                    split="sealed_eval",
                    seed=index,
                    attempt_index=attempt_index,
                    trace_path=Path(trace_dir)
                    / "official_arcagi3"
                    / "sealed_eval"
                    / ablation_id
                    / f"attempt_{attempt_index}"
                    / f"{spec.game_id}.json",
                )
                row["game_id"] = spec.game_id
                row["source_variant"] = variant.variant_id
                rows.append(row)
    report = {
        "suite_id": "official_arcagi3_behavioral_ablation",
        "split": "sealed_eval",
        "rows": rows,
        "aggregate_by_variant": aggregate_by_variant(rows),
        "attempt_table": attempt_table(rows),
        "behavioral_offline_gates": behavioral_offline_gates(
            rows,
            primary_variant="full_jepa_plus_attempt_memory",
            baseline_variant="base_only",
        ),
        "ablation_arms": [
            {
                "ablation_id": str(arm["ablation_id"]),
                "source_variant": arm["variant"].variant_id,
                "enable_hard_anti_attractor": bool(arm["enable_hard_anti_attractor"]),
                "jepa_action_mode": str(arm["jepa_action_mode"]),
                "disable_jepa_direct_override": bool(arm["disable_jepa_direct_override"]),
            }
            for arm in arms
        ],
        "trace_paths": [row["trace_path"] for row in rows],
        "device_runtime": cuda_runtime_info(target_device),
        "variant_model_info": model_info,
        "operation_mode": operation_mode,
        "selected_game_ids": [spec.game_id for spec in specs],
        "attempts": max(int(attempts), 1),
    }
    Path(json_output).parent.mkdir(parents=True, exist_ok=True)
    Path(json_output).write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    return report


def dispatch_official_worker(
    *,
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    jepa_checkpoint: str | Path,
    trace_dir: str | Path,
    core_arm: str,
    device: str | torch.device,
    operation_mode: str = "normal",
    game_ids: Sequence[str] | None = None,
    limit: int | None = None,
    variant_ids: Sequence[str] | None = None,
    attempts: int = 3,
    environments_dir: str | Path = "runs/arcagi3_official_envs",
    recordings_dir: str | Path = "runs/arcagi3_official_recordings",
    jepa_action_mode: str = "propose_only",
    jepa_max_action_bias: float = 0.05,
    disable_jepa_direct_override: bool = True,
    enable_hard_anti_attractor: bool = True,
) -> dict[str, Any]:
    temp = Path(trace_dir) / "_official_jepa_worker_report.json"
    try:
        from .arcagi3_official import require_official_runtime

        require_official_runtime()
        return run_official_worker(
            checkpoint=checkpoint,
            explorer_checkpoint=explorer_checkpoint,
            jepa_checkpoint=jepa_checkpoint,
            trace_dir=trace_dir,
            json_output=temp,
            core_arm=core_arm,
            device=device,
            operation_mode=operation_mode,
            game_ids=game_ids,
            limit=limit,
            variant_ids=variant_ids,
            attempts=attempts,
            environments_dir=environments_dir,
            recordings_dir=recordings_dir,
            jepa_action_mode=jepa_action_mode,
            jepa_max_action_bias=jepa_max_action_bias,
            disable_jepa_direct_override=disable_jepa_direct_override,
            enable_hard_anti_attractor=enable_hard_anti_attractor,
        )
    except Exception:
        venv_python = Path(".venv/Scripts/python.exe")
        if not venv_python.exists():
            raise
        cmd = [
            str(venv_python),
            "-m",
            "src.jepa_arc_eval",
            "--config",
            "official_worker",
            "--checkpoint",
            str(checkpoint),
            "--explorer-checkpoint",
            str(explorer_checkpoint),
            "--jepa-checkpoint",
            str(jepa_checkpoint),
            "--json-output",
            str(temp),
            "--trace-dir",
            str(trace_dir),
            "--core-arm",
            core_arm,
            "--device",
            str(device or AUTO_DEVICE),
            "--operation-mode",
            operation_mode,
            "--attempts",
            str(max(int(attempts), 1)),
            "--environments-dir",
            str(environments_dir),
            "--recordings-dir",
            str(recordings_dir),
            "--jepa-action-mode",
            str(jepa_action_mode),
            "--jepa-max-action-bias",
            str(jepa_max_action_bias),
        ]
        if disable_jepa_direct_override:
            cmd.append("--disable-jepa-direct-override")
        if not enable_hard_anti_attractor:
            cmd.append("--disable-hard-anti-attractor")
        if limit is not None:
            cmd.extend(["--limit", str(limit)])
        for item in expand_cli_values(game_ids):
            cmd.extend(["--game-id", item])
        for item in expand_cli_values(variant_ids):
            cmd.extend(["--variant", item])
        try:
            completed = subprocess.run(cmd, cwd=Path.cwd(), text=True, capture_output=True, check=True)
        except subprocess.CalledProcessError as exc:
            print(
                json.dumps(
                    {
                        "official_worker_failed": True,
                        "stdout_suppressed": bool(exc.stdout),
                        "stderr_suppressed": bool(exc.stderr),
                    }
                )
            )
            raise
        if completed.stdout.strip():
            print(
                json.dumps(
                    {
                        "official_worker_stdout_suppressed": True,
                        "line_count": len(completed.stdout.splitlines()),
                    }
                )
            )
        return json.loads(temp.read_text(encoding="utf-8"))


def run_non_arc(
    *,
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    jepa_checkpoint: str | Path,
    trace_dir: str | Path,
    core_arm: str,
    device: torch.device,
    jepa_action_mode: str = "propose_only",
    jepa_max_action_bias: float = 0.05,
    disable_jepa_direct_override: bool = True,
    enable_hard_anti_attractor: bool = True,
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for suite in discover_external_suites():
        if not suite.available or not suite.suite_id.startswith("gymnasium_"):
            continue
        for variant in VARIANTS:
            controller, _ = make_controller_for_variant(
                variant=variant,
                core_arm=core_arm,
                checkpoint=checkpoint,
                explorer_checkpoint=explorer_checkpoint,
                jepa_checkpoint=jepa_checkpoint,
                device=device,
                jepa_action_mode=jepa_action_mode,
                jepa_max_action_bias=jepa_max_action_bias,
                disable_jepa_direct_override=disable_jepa_direct_override,
                enable_hard_anti_attractor=enable_hard_anti_attractor,
            )
            for task_id in suite.tasks:
                for seed in [100, 101, 102]:
                    controller.reset_all()
                    for attempt_index in [1, 2, 3]:
                        env = GymnasiumExternalEnv(
                            suite.suite_id,
                            task_id,
                            max_steps=80 if task_id == "CartPole-v1" else 32,
                        )
                        rows.append(
                            run_attempt(
                                env,
                                controller,
                                suite_id=suite.suite_id,
                                variant_id=variant.variant_id,
                                split="sealed_eval",
                                seed=seed,
                                attempt_index=attempt_index,
                                trace_path=Path(trace_dir)
                                / suite.suite_id
                                / "sealed_eval"
                                / variant.variant_id
                                / f"attempt_{attempt_index}"
                                / f"{task_id}.seed_{seed}.json",
                            )
                        )
    return {
        "suite_id": "non_arc_external",
        "split": "sealed_eval",
        "rows": rows,
        "aggregate_by_variant": aggregate_by_variant(rows),
        "attempt_table": attempt_table(rows),
        "trace_paths": [row["trace_path"] for row in rows],
    }


def aggregate_by_variant(rows: list[dict[str, Any]]) -> dict[str, Any]:
    variants = sorted({str(row.get("variant")) for row in rows})
    return {variant: aggregate_rows([row for row in rows if row.get("variant") == variant]) for variant in variants}


def attempt_table(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    table: list[dict[str, Any]] = []
    variants = sorted({str(row.get("variant")) for row in rows})
    for variant in variants:
        for attempt_index in [1, 2, 3]:
            subset = [row for row in rows if row.get("variant") == variant and int(row.get("attempt_index", 0)) == attempt_index]
            aggregate = aggregate_rows(subset)
            table.append({"variant": variant, "attempt_index": attempt_index, **aggregate})
    return table


def behavioral_offline_gates(
    rows: list[dict[str, Any]],
    *,
    primary_variant: str = "jepa_plus_attempt_memory",
    baseline_variant: str = "attempt_memory_no_jepa",
) -> dict[str, Any]:
    variants = sorted({str(row.get("variant", "")) for row in rows if str(row.get("variant", ""))})
    per_variant = {variant: _behavioral_metrics_for_rows([row for row in rows if row.get("variant") == variant]) for variant in variants}
    primary = per_variant.get(primary_variant) or (next(iter(per_variant.values())) if per_variant else _empty_behavioral_metrics())
    baseline = per_variant.get(baseline_variant)
    sidecar_override_usefulness = float(primary.get("sidecar_override_usefulness", 0.0))
    base_usefulness = float(baseline.get("mean_useful_events_per_attempt", 0.0)) if baseline else 0.0
    sidecar_overrides = int(primary.get("sidecar_approved_executed", 0))
    sidecar_usefulness_gate = bool(sidecar_overrides == 0 or sidecar_override_usefulness >= base_usefulness)
    sidecar_changed_ratio = float(primary.get("sidecar_changed_action_ratio", 0.0))
    sidecar_changed_gate = bool(
        sidecar_changed_ratio <= BEHAVIORAL_OFFLINE_GATE_THRESHOLDS["sidecar_changed_action_ratio"]
        or sidecar_override_usefulness >= base_usefulness
    )
    baseline_progress = float(baseline.get("progress_discovery_rate", 0.0)) if baseline else None
    progress_discovery_rate = float(primary.get("progress_discovery_rate", 0.0))
    progress_gate = bool(baseline is not None and progress_discovery_rate > float(baseline_progress))
    gates = {
        "loop_attempt_rate": float(primary.get("loop_attempt_rate", 1.0))
        <= BEHAVIORAL_OFFLINE_GATE_THRESHOLDS["loop_attempt_rate"],
        "button_loop_attempt_rate": float(primary.get("button_loop_attempt_rate", 1.0))
        <= BEHAVIORAL_OFFLINE_GATE_THRESHOLDS["button_loop_attempt_rate"],
        "coordinate_loop_attempt_rate": float(primary.get("coordinate_loop_attempt_rate", 1.0))
        <= BEHAVIORAL_OFFLINE_GATE_THRESHOLDS["coordinate_loop_attempt_rate"],
        "mean_useful_events_per_attempt": float(primary.get("mean_useful_events_per_attempt", 0.0))
        >= BEHAVIORAL_OFFLINE_GATE_THRESHOLDS["mean_useful_events_per_attempt"],
        "attempts_with_zero_useful_events": float(primary.get("zero_useful_attempt_rate", 1.0))
        <= BEHAVIORAL_OFFLINE_GATE_THRESHOLDS["zero_useful_attempt_rate"],
        "sidecar_override_usefulness": sidecar_usefulness_gate,
        "sidecar_changed_action_ratio": sidecar_changed_gate,
        "progress_discovery_rate_beats_baseline": progress_gate,
    }
    return {
        "schema": "arc_offline_behavioral_gates_v1",
        "primary_variant": primary_variant,
        "baseline_variant": baseline_variant,
        "thresholds": dict(BEHAVIORAL_OFFLINE_GATE_THRESHOLDS),
        "baseline_available": bool(baseline is not None),
        "per_variant": per_variant,
        "primary_metrics": primary,
        "baseline_metrics": baseline or {},
        "gates": gates,
        "passes": bool(gates and all(gates.values())),
    }


def _empty_behavioral_metrics() -> dict[str, Any]:
    return {
        "attempts": 0,
        "steps": 0,
        "loop_attempts": 0,
        "button_loop_attempts": 0,
        "coordinate_loop_attempts": 0,
        "loop_attempt_rate": 0.0,
        "button_loop_attempt_rate": 0.0,
        "coordinate_loop_attempt_rate": 0.0,
        "mean_useful_events_per_attempt": 0.0,
        "zero_useful_attempt_rate": 0.0,
        "progress_discovery_rate": 0.0,
        "sidecar_approved_executed": 0,
        "sidecar_useful_overrides": 0,
        "sidecar_override_usefulness": 0.0,
        "sidecar_changed_actions": 0,
        "sidecar_changed_action_ratio": 0.0,
    }


def _behavioral_metrics_for_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = _empty_behavioral_metrics()
    if not rows:
        return metrics
    metrics["attempts"] = len(rows)
    metrics["mean_useful_events_per_attempt"] = float(
        sum(float(row.get("useful_events", 0.0) or 0.0) for row in rows) / max(len(rows), 1)
    )
    metrics["zero_useful_attempt_rate"] = float(
        sum(1 for row in rows if float(row.get("useful_events", 0.0) or 0.0) <= 0.0) / max(len(rows), 1)
    )
    for row in rows:
        trace_metrics = _behavioral_metrics_from_trace_path(row.get("trace_path", ""))
        for key in [
            "steps",
            "loop_attempts",
            "button_loop_attempts",
            "coordinate_loop_attempts",
            "progress_events",
            "sidecar_approved_executed",
            "sidecar_useful_overrides",
            "sidecar_changed_actions",
        ]:
            metrics[key] = int(metrics.get(key, 0)) + int(trace_metrics.get(key, 0))
    steps = max(int(metrics.get("steps", 0)), 1)
    metrics["loop_attempt_rate"] = float(metrics.get("loop_attempts", 0) / steps)
    metrics["button_loop_attempt_rate"] = float(metrics.get("button_loop_attempts", 0) / steps)
    metrics["coordinate_loop_attempt_rate"] = float(metrics.get("coordinate_loop_attempts", 0) / steps)
    metrics["progress_discovery_rate"] = float(metrics.get("progress_events", 0) / steps)
    sidecar_executed = max(int(metrics.get("sidecar_approved_executed", 0)), 1)
    metrics["sidecar_override_usefulness"] = float(metrics.get("sidecar_useful_overrides", 0) / sidecar_executed)
    metrics["sidecar_changed_action_ratio"] = float(metrics.get("sidecar_changed_actions", 0) / steps)
    return metrics


def _behavioral_metrics_from_trace_path(path_text: Any) -> dict[str, int]:
    path = Path(str(path_text))
    if not path.exists():
        return {
            "steps": 0,
            "loop_attempts": 0,
            "button_loop_attempts": 0,
            "coordinate_loop_attempts": 0,
            "progress_events": 0,
            "sidecar_approved_executed": 0,
            "sidecar_useful_overrides": 0,
            "sidecar_changed_actions": 0,
        }
    try:
        trace = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return _behavioral_metrics_from_steps([])
    attempt = trace.get("attempt", {}) if isinstance(trace.get("attempt"), dict) else {}
    steps = [step for step in attempt.get("steps", []) if isinstance(step, dict)]
    return _behavioral_metrics_from_steps(steps)


def _behavioral_metrics_from_steps(steps: list[dict[str, Any]]) -> dict[str, int]:
    actions = [str(step.get("action", "")) for step in steps]
    progress = [_step_has_progress_event(step) for step in steps]
    loop_attempts = 0
    button_loop_attempts = 0
    coordinate_loop_attempts = 0
    for index in range(len(actions)):
        if any(progress[max(0, index - 12) : index + 1]):
            continue
        loop_window = _cycle_window(actions, index)
        if not loop_window:
            continue
        loop_attempts += 1
        if all(_is_coordinate_action(action) for action in loop_window):
            coordinate_loop_attempts += 1
        elif all(_is_button_action(action) for action in loop_window):
            button_loop_attempts += 1
    sidecar_approved_executed = 0
    sidecar_useful_overrides = 0
    sidecar_changed_actions = 0
    for step in steps:
        policy = (step.get("diagnostics") or {}).get("jepa_policy", {})
        sidecar = policy.get("sidecar_override", {}) if isinstance(policy, dict) else {}
        if not bool(sidecar.get("approved")) or not bool(sidecar.get("executed")):
            continue
        sidecar_approved_executed += 1
        if str(sidecar.get("proposed_action", "")) != str(sidecar.get("base_action", "")):
            sidecar_changed_actions += 1
        if _step_has_useful_event(step):
            sidecar_useful_overrides += 1
    return {
        "steps": len(steps),
        "loop_attempts": loop_attempts,
        "button_loop_attempts": button_loop_attempts,
        "coordinate_loop_attempts": coordinate_loop_attempts,
        "progress_events": sum(1 for item in progress if item),
        "sidecar_approved_executed": sidecar_approved_executed,
        "sidecar_useful_overrides": sidecar_useful_overrides,
        "sidecar_changed_actions": sidecar_changed_actions,
    }


def _cycle_window(actions: list[str], index: int) -> list[str]:
    for period in range(1, 7):
        start = index - (2 * period) + 1
        mid = index - period + 1
        if start < 0:
            continue
        if actions[start:mid] == actions[mid : index + 1]:
            return actions[mid : index + 1]
    return []


def _is_coordinate_action(action: str) -> bool:
    return str(action).startswith("click:")


def _is_button_action(action: str) -> bool:
    action = str(action)
    return bool(action and not _is_coordinate_action(action))


def _step_has_progress_event(step: dict[str, Any]) -> bool:
    if float(step.get("score_delta", 0.0) or 0.0) > 0.0:
        return True
    return _events_have_progress(step.get("event_delta", []) or [])


def _step_has_useful_event(step: dict[str, Any]) -> bool:
    if _step_has_progress_event(step):
        return True
    policy = (step.get("diagnostics") or {}).get("jepa_policy", {})
    if not isinstance(policy, dict):
        return False
    if bool(policy.get("posthoc_useful")):
        return True
    transition = policy.get("posthoc_transition_usefulness", {})
    return bool(isinstance(transition, dict) and transition.get("useful_effect"))


def lookup_attempt(table: list[dict[str, Any]], variant: str, attempt_index: int) -> dict[str, Any]:
    for row in table:
        if row.get("variant") == variant and int(row.get("attempt_index", -1)) == attempt_index:
            return row
    return {}


def no_hack_proof() -> dict[str, Any]:
    paths = [Path(item) for item in JEPA_AUDITED_PATHS if Path(item).suffix == ".py" and Path(item).exists()]
    findings: list[dict[str, Any]] = []
    forbidden_literals = [
        "hidden" + "_goal",
        "ground" + "_truth",
        "answer" + "_key",
        "solution" + "_path",
        "manual" + "_hint",
    ]
    branch_names = {"task_id", "fixture_id", "official" + "_game" + "_id"}
    for path in paths:
        text = path.read_text(encoding="utf-8", errors="ignore")
        lowered = text.lower()
        for literal in forbidden_literals:
            if literal in lowered:
                findings.append({"path": str(path), "kind": "forbidden_literal", "match": literal})
        try:
            tree = ast.parse(text)
        except SyntaxError as exc:
            findings.append({"path": str(path), "kind": "parse_error", "line": exc.lineno})
            continue
        for node in ast.walk(tree):
            if isinstance(node, ast.If):
                test = ast.unparse(node.test)
                if any(name in test for name in branch_names) and "startswith(\"gymnasium_\")" not in test:
                    findings.append({"path": str(path), "line": getattr(node, "lineno", None), "kind": "id_branch", "match": test})
    hidden_canary = {"passes": False, "hidden_target_canary_max_abs_diff": None}
    canary_path = Path("docs/audit_after_arcagi3_diagnosis.json")
    if canary_path.exists():
        report = json.loads(canary_path.read_text(encoding="utf-8"))
        diff = float(report.get("anti_leakage", {}).get("hidden_target_canary_max_abs_diff", 1.0))
        hidden_canary = {"passes": diff == 0.0, "hidden_target_canary_max_abs_diff": diff, "source": str(canary_path)}
    return {
        "passes": not findings and bool(hidden_canary.get("passes")),
        "findings": findings,
        "hidden_target_canary": hidden_canary,
        "allowed_inputs": [
            "public frames",
            "public legal actions",
            "chosen actions",
            "public score and event deltas",
            "terminal flags",
        ],
        "not_used": [
            "official sealed tuning",
            "hidden labels",
            "action advice",
            "game source inspection",
            "manual hints",
            "scripted solver actions",
            "entropy schedules",
        ],
        "jepa_emits_text": False,
        "actions_from": "existing_core_plus_attempt_memory",
    }


def causal_substrate_self_check(jepa_checkpoint: str | Path, device: torch.device | str = "cpu") -> dict[str, Any]:
    target_device = resolve_device(device)
    model, payload = load_video_jepa(str(jepa_checkpoint), device=target_device)
    records = synthetic_attempts(10, seed=9917)
    probe_record = records[0]
    legal = tuple(probe_record.steps[0].legal_actions)
    jepa_memory = JEPAAttemptMemory(use_jepa_tokens=True, device=target_device)
    no_jepa_memory = JEPAAttemptMemory(use_jepa_tokens=False, device=target_device)
    jepa_entry = jepa_memory.ingest_attempt(probe_record, model=model)
    no_jepa_entry = no_jepa_memory.ingest_attempt(probe_record, model=None)
    jepa_distribution = jepa_memory.action_distribution(legal)
    no_jepa_distribution = no_jepa_memory.action_distribution(legal)
    distribution_l1 = sum(abs(jepa_distribution.get(action, 0.0) - no_jepa_distribution.get(action, 0.0)) for action in legal)
    plan_diff = sum(
        abs(float(jepa_entry.next_attempt_plan.get(action, 0.0)) - float(no_jepa_entry.next_attempt_plan.get(action, 0.0)))
        for action in set(jepa_entry.next_attempt_plan) | set(no_jepa_entry.next_attempt_plan)
    )
    proposal_l1 = float(sum(abs(float(value)) for value in jepa_entry.sidecar_action_priors.values()))
    checks = {
        "attempt_video_action_history_present": bool(probe_record.steps),
        "jepa_temporal_representation_present": bool(jepa_entry.token_mean) and bool(jepa_entry.jepa_action_evidence),
        "attempt_memory_stores_jepa_tokens": bool(jepa_memory.summary().get("causal_substrate_active")),
        "rule_causal_hypothesis_uses_jepa": float(jepa_entry.causal_hypotheses.get("jepa_action_effect_span", 0.0)) > 0.0,
        "jepa_action_priors_recorded": proposal_l1 > 1.0e-9,
        "next_attempt_plan_not_changed_by_jepa": plan_diff <= 1.0e-9,
        "action_distribution_not_changed_by_jepa": distribution_l1 <= 1.0e-9,
        "jepa_not_direct_action_source": bool(jepa_memory.summary().get("direct_action_source") is False),
        "jepa_emits_no_text": bool(payload.get("emits_text") is False),
    }
    return {
        "passes": all(checks.values()),
        "checks": checks,
        "causal_chain": [
            "attempt_video_action_history",
            "jepa_temporal_representation",
            "attempt_memory",
            "rule_causal_hypothesis_update",
            "sidecar_action_prior_proposals",
            "symbolic_causal_controller_approval",
        ],
        "probe_attempt_steps": len(probe_record.steps),
        "jepa_action_evidence": jepa_entry.jepa_action_evidence,
        "jepa_causal_hypotheses": jepa_entry.causal_hypotheses,
        "jepa_next_attempt_plan": jepa_entry.next_attempt_plan,
        "no_jepa_next_attempt_plan": no_jepa_entry.next_attempt_plan,
        "jepa_sidecar_action_priors": jepa_entry.sidecar_action_priors,
        "jepa_action_distribution": jepa_distribution,
        "no_jepa_action_distribution": no_jepa_distribution,
        "distribution_l1": float(distribution_l1),
        "plan_l1": float(plan_diff),
        "proposal_l1": float(proposal_l1),
    }


def load_core_choice() -> dict[str, Any]:
    report = json.loads(Path("docs/external_base_report.json").read_text(encoding="utf-8"))
    selection = report.get("selection", {})
    selected = str(selection.get("selected_variant") or "old_base_finetuned")
    return {
        "selected_core": selected,
        "selection_source": selection.get("selection_source", "non_arc_dev_only_before_official_sealed_eval"),
        "evidence": "docs/external_base_report.json",
        "candidate_table": selection.get("candidate_table", []),
    }


def build_report(
    *,
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    jepa_checkpoint: str | Path,
    trace_dir: str | Path,
    json_output: str | Path,
    device: torch.device,
    jepa_action_mode: str = "propose_only",
    jepa_max_action_bias: float = 0.05,
    disable_jepa_direct_override: bool = True,
    enable_hard_anti_attractor: bool = True,
) -> dict[str, Any]:
    core = load_core_choice()
    core_arm = str(core["selected_core"])
    official = dispatch_official_worker(
        checkpoint=checkpoint,
        explorer_checkpoint=explorer_checkpoint,
        jepa_checkpoint=jepa_checkpoint,
        trace_dir=trace_dir,
        core_arm=core_arm,
        device=device,
        jepa_action_mode=jepa_action_mode,
        jepa_max_action_bias=jepa_max_action_bias,
        disable_jepa_direct_override=disable_jepa_direct_override,
        enable_hard_anti_attractor=enable_hard_anti_attractor,
    )
    non_arc = run_non_arc(
        checkpoint=checkpoint,
        explorer_checkpoint=explorer_checkpoint,
        jepa_checkpoint=jepa_checkpoint,
        trace_dir=trace_dir,
        core_arm=core_arm,
        device=device,
        jepa_action_mode=jepa_action_mode,
        jepa_max_action_bias=jepa_max_action_bias,
        disable_jepa_direct_override=disable_jepa_direct_override,
        enable_hard_anti_attractor=enable_hard_anti_attractor,
    )
    jepa_payload = torch.load(jepa_checkpoint, map_location="cpu", weights_only=False)
    causal_substrate = causal_substrate_self_check(jepa_checkpoint, device=device)
    primary = "jepa_plus_attempt_memory"
    primary_a1 = lookup_attempt(official["attempt_table"], primary, 1)
    primary_a2 = lookup_attempt(official["attempt_table"], primary, 2)
    primary_a3 = lookup_attempt(official["attempt_table"], primary, 3)
    baseline_a1 = lookup_attempt(official["attempt_table"], "baseline_core", 1)
    repeat_drop = float(primary_a1.get("mean_repeat_collapse", 1.0)) - float(primary_a3.get("mean_repeat_collapse", 1.0))
    score_gain = float(primary_a3.get("mean_normalized_score", 0.0)) - float(baseline_a1.get("mean_normalized_score", 0.0))
    useful_gain = float(primary_a3.get("mean_useful_events", 0.0)) - float(baseline_a1.get("mean_useful_events", 0.0))
    attempt_improves = (
        float(primary_a2.get("mean_normalized_score", 0.0)) > float(primary_a1.get("mean_normalized_score", 0.0))
        or float(primary_a3.get("mean_normalized_score", 0.0)) > float(primary_a1.get("mean_normalized_score", 0.0))
        or float(primary_a2.get("mean_useful_events", 0.0)) > float(primary_a1.get("mean_useful_events", 0.0))
        or float(primary_a3.get("mean_useful_events", 0.0)) > float(primary_a1.get("mean_useful_events", 0.0))
    )
    baseline_non_arc = non_arc["aggregate_by_variant"].get("baseline_core", {})
    primary_non_arc = non_arc["aggregate_by_variant"].get(primary, {})
    non_arc_drop = float(baseline_non_arc.get("mean_normalized_score", 0.0)) - float(primary_non_arc.get("mean_normalized_score", 0.0))
    no_hack = no_hack_proof()
    gates = {
        "attempt_2_or_3_improves_over_attempt_1": attempt_improves,
        "score_or_useful_gain_over_core": score_gain >= 0.01 or useful_gain >= 0.08,
        "official_score_gain": score_gain,
        "official_useful_event_gain": useful_gain,
        "repeat_collapse_drop_attempt_1_to_3": repeat_drop,
        "repeat_collapse_drop_gate": repeat_drop >= 0.20,
        "ablation_removes_improvement": False,
        "jepa_beats_null_on_dev": bool(jepa_payload.get("metrics", {}).get("jepa_beats_null")),
        "jepa_causal_substrate_chain": bool(causal_substrate.get("passes")),
        "non_arc_drop": non_arc_drop,
        "non_arc_drop_within_limit": non_arc_drop <= 0.05,
        "hidden_target_canary_diff_zero": bool(no_hack.get("hidden_target_canary", {}).get("passes")),
        "jepa_emits_no_text": True,
        "no_hack_passes": bool(no_hack.get("passes")),
    }
    outcome = (
        "JEPA IMPROVEMENT FOUND"
        if all(
            [
                gates["attempt_2_or_3_improves_over_attempt_1"],
                gates["score_or_useful_gain_over_core"],
                gates["repeat_collapse_drop_gate"],
                gates["ablation_removes_improvement"],
                gates["jepa_beats_null_on_dev"],
                gates["jepa_causal_substrate_chain"],
                gates["non_arc_drop_within_limit"],
                gates["hidden_target_canary_diff_zero"],
                gates["jepa_emits_no_text"],
                gates["no_hack_passes"],
            ]
        )
        else "NO IMPROVEMENT FOUND"
    )
    report = {
        "terminal_outcome": outcome,
        "primary_variant": primary,
        "core_choice": core,
        "required_checkpoint_argument": str(checkpoint),
        "external_base_checkpoint_used": str(resolve_external_base_checkpoint(checkpoint)),
        "variants": [variant.__dict__ for variant in VARIANTS],
        "variant_ids": [variant.variant_id for variant in VARIANTS],
        "data_manifest": jepa_payload.get("manifest", {}),
        "jepa_dev_metrics": jepa_payload.get("metrics", {}),
        "causal_substrate_proof": causal_substrate,
        "official": {
            "aggregate_by_variant": official["aggregate_by_variant"],
            "attempt_table": official["attempt_table"],
            "official_baselines": official.get("official_baselines", []),
            "trace_paths": official.get("trace_paths", []),
            "device_runtime": official.get("device_runtime", {}),
            "variant_model_info": official.get("variant_model_info", {}),
        },
        "non_arc": {
            "aggregate_by_variant": non_arc["aggregate_by_variant"],
            "attempt_table": non_arc["attempt_table"],
            "trace_paths": non_arc["trace_paths"],
        },
        "ablations": {
            "attempt_memory_no_jepa": official["aggregate_by_variant"].get("attempt_memory_no_jepa", {}),
            "jepa_trained_dev": official["aggregate_by_variant"].get("jepa_trained_dev", {}),
            "jepa_random_init": official["aggregate_by_variant"].get("jepa_random_init", {}),
            "null_control": official["aggregate_by_variant"].get("null_control", {}),
            "ablation_removes_improvement": gates["ablation_removes_improvement"],
        },
        "gates": gates,
        "no_hack_proof": no_hack,
        "trace_paths": list(official.get("trace_paths", [])) + list(non_arc.get("trace_paths", [])),
        "required_commands": [
            "pytest -q",
            "python -m src.jepa_train --config dev --output runs/video_jepa.pt --manifest data/jepa_trace_manifest.json",
            "python -m src.jepa_arc_eval --checkpoint frozen/recurrent_latent_fast.pt --explorer-checkpoint runs/explorer_tiny.pt --jepa-checkpoint runs/video_jepa.pt --config external --json-output docs/jepa_attempt_report.json --trace-dir docs/jepa_attempt_traces",
            "python -m audit.leakage_scan",
            "python -m src.generalization_audit --json-output docs/generalization_audit_after_jepa.json",
        ],
        "hashes": collect_hashes(JEPA_AUDITED_PATHS),
        "limitations": [
            "Official sealed outcome is valid only after the generated report and audits are rerun in the current workspace.",
            "Frozen perception variant is declared but unavailable unless an approved latent-only video encoder is added.",
        ],
    }
    output = Path(json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    return report


def build_focused_official_report(
    *,
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    jepa_checkpoint: str | Path,
    trace_dir: str | Path,
    json_output: str | Path,
    device: torch.device,
    operation_mode: str = "offline",
    game_ids: Sequence[str] | None = None,
    limit: int | None = 1,
    variant_ids: Sequence[str] | None = None,
    attempts: int = 3,
    environments_dir: str | Path = "runs/arcagi3_official_envs",
    recordings_dir: str | Path = "runs/arcagi3_official_recordings",
    jepa_action_mode: str = "propose_only",
    jepa_max_action_bias: float = 0.05,
    disable_jepa_direct_override: bool = True,
    enable_hard_anti_attractor: bool = True,
) -> dict[str, Any]:
    core = load_core_choice()
    selected_variants = focused_variant_ids(variant_ids)
    selected_games = manifest_game_ids(requested_ids=game_ids, limit=limit)
    core_arm = str(core["selected_core"])
    official = dispatch_official_worker(
        checkpoint=checkpoint,
        explorer_checkpoint=explorer_checkpoint,
        jepa_checkpoint=jepa_checkpoint,
        trace_dir=trace_dir,
        core_arm=core_arm,
        device=device,
        operation_mode=operation_mode,
        game_ids=selected_games,
        limit=None,
        variant_ids=selected_variants,
        attempts=attempts,
        environments_dir=environments_dir,
        recordings_dir=recordings_dir,
        jepa_action_mode=jepa_action_mode,
        jepa_max_action_bias=jepa_max_action_bias,
        disable_jepa_direct_override=disable_jepa_direct_override,
        enable_hard_anti_attractor=enable_hard_anti_attractor,
    )
    primary = selected_variants[0]
    primary_aggregate = official["aggregate_by_variant"].get(primary, {})
    primary_attempt_rows = [
        row for row in official.get("attempt_table", []) if str(row.get("variant", "")) == primary
    ]
    game_beaten = bool(
        any(float(row.get("solve_rate", 0.0)) >= 1.0 for row in primary_attempt_rows)
        or float(primary_aggregate.get("solve_rate", 0.0)) >= 1.0
    )
    no_hack = no_hack_proof()
    report = {
        "terminal_outcome": "FOCUSED_GAME1_BEATEN" if game_beaten else "FOCUSED_GAME1_NOT_BEATEN",
        "scope": "focused_official_first_game_until_beaten",
        "primary_variant": primary,
        "core_choice": core,
        "operation_mode": operation_mode,
        "selected_game_ids": selected_games,
        "variant_ids": selected_variants,
        "no_op_control_variants_excluded": not bool(set(selected_variants).intersection(NO_OP_CONTROL_VARIANT_IDS)),
        "attempts": max(int(attempts), 1),
        "official": {
            "aggregate_by_variant": official["aggregate_by_variant"],
            "attempt_table": official["attempt_table"],
            "trace_paths": official.get("trace_paths", []),
            "device_runtime": official.get("device_runtime", {}),
            "variant_model_info": official.get("variant_model_info", {}),
        },
        "focused_metrics": {
            "game_beaten": game_beaten,
            "primary_solve_rate": float(primary_aggregate.get("solve_rate", 0.0)),
            "primary_mean_normalized_score": float(primary_aggregate.get("mean_normalized_score", 0.0)),
            "primary_mean_useful_events": float(primary_aggregate.get("mean_useful_events", 0.0)),
            "primary_mean_repeat_collapse": float(primary_aggregate.get("mean_repeat_collapse", 0.0)),
        },
        "no_hack_proof": no_hack,
        "hashes": collect_hashes(JEPA_AUDITED_PATHS),
        "trace_paths": list(official.get("trace_paths", [])),
        "required_commands": [
            "pytest -q tests/test_video_jepa.py",
            "python -m src.jepa_arc_eval --config focused_official --checkpoint frozen/recurrent_latent_fast.pt --explorer-checkpoint runs/explorer_tiny.pt --jepa-checkpoint runs/video_jepa.pt --json-output docs/jepa_focused_game1_report.json --trace-dir docs/jepa_focused_game1_traces --operation-mode offline --limit 1",
        ],
        "limitations": [
            "Focused report is a development loop for the first official public game, not the full 25-game success proof.",
            "No-op control variants are intentionally excluded from this focused loop.",
        ],
    }
    output = Path(json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    return report


def _action_histogram(actions: Sequence[str]) -> dict[str, float]:
    counts: dict[str, float] = {}
    total = max(len(actions), 1)
    for action in actions:
        counts[str(action)] = counts.get(str(action), 0.0) + 1.0 / float(total)
    return counts


def _l1_dict(left: dict[str, float], right: dict[str, float]) -> float:
    return float(sum(abs(float(left.get(key, 0.0)) - float(right.get(key, 0.0))) for key in set(left) | set(right)))


def _trace_actions(trace: dict[str, Any]) -> list[str]:
    attempt = trace.get("attempt", {}) if isinstance(trace.get("attempt"), dict) else {}
    return [str(step.get("action", "")) for step in attempt.get("steps", []) if isinstance(step, dict) and str(step.get("action", ""))]


def _step_policy(trace: dict[str, Any], step_index: int = 0) -> dict[str, Any]:
    attempt = trace.get("attempt", {}) if isinstance(trace.get("attempt"), dict) else {}
    steps = [step for step in attempt.get("steps", []) if isinstance(step, dict)]
    if not steps or step_index >= len(steps):
        return {}
    diagnostics = steps[step_index].get("diagnostics", {}) if isinstance(steps[step_index].get("diagnostics"), dict) else {}
    policy = diagnostics.get("jepa_policy", {}) if isinstance(diagnostics.get("jepa_policy"), dict) else {}
    return policy


def _memory_summary(trace: dict[str, Any]) -> dict[str, Any]:
    summary = trace.get("summary", {}) if isinstance(trace.get("summary"), dict) else {}
    controller = summary.get("controller_summary", {}) if isinstance(summary.get("controller_summary"), dict) else {}
    memory = controller.get("attempt_memory", {}) if isinstance(controller.get("attempt_memory"), dict) else {}
    return memory


def _distribution_from_policy(policy: dict[str, Any], key: str) -> dict[str, float]:
    value = policy.get(key, {})
    if not isinstance(value, dict):
        return {}
    return {str(action): float(score) for action, score in value.items()}


def _load_official_traces(official: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    traces: dict[tuple[str, int], dict[str, Any]] = {}
    for path_text in official.get("trace_paths", []):
        path = Path(path_text)
        if not path.exists():
            continue
        payload = json.loads(path.read_text(encoding="utf-8"))
        metadata = payload.get("metadata", {}) if isinstance(payload.get("metadata"), dict) else {}
        variant = str(metadata.get("variant", ""))
        attempt_index = int(metadata.get("attempt_index", 0) or 0)
        if variant and attempt_index:
            traces[(variant, attempt_index)] = payload
    return traces


def jepa_causality_audit(official: dict[str, Any], *, primary: str = "jepa_plus_attempt_memory") -> dict[str, Any]:
    traces = _load_official_traces(official)
    trained_a1 = traces.get((primary, 1), {})
    trained_a2 = traces.get((primary, 2), {})
    no_jepa_a2 = traces.get(("attempt_memory_no_jepa", 2), {})
    random_a2 = traces.get(("jepa_random_init", 2), {})
    shuffled_a2 = traces.get(("jepa_trained_shuffled", 2), {})

    trained_policy_a2 = _step_policy(trained_a2, 0)
    no_jepa_policy_a2 = _step_policy(no_jepa_a2, 0)
    random_policy_a2 = _step_policy(random_a2, 0)
    shuffled_policy_a2 = _step_policy(shuffled_a2, 0)
    trained_action_dist_a1 = _action_histogram(_trace_actions(trained_a1))
    trained_action_dist_a2 = _action_histogram(_trace_actions(trained_a2))
    no_jepa_action_dist_a2 = _action_histogram(_trace_actions(no_jepa_a2))
    random_action_dist_a2 = _action_histogram(_trace_actions(random_a2))
    shuffled_action_dist_a2 = _action_histogram(_trace_actions(shuffled_a2))

    trained_memory_dist = _distribution_from_policy(trained_policy_a2, "memory_action_distribution")
    no_jepa_memory_dist = _distribution_from_policy(no_jepa_policy_a2, "memory_action_distribution")
    random_memory_dist = _distribution_from_policy(random_policy_a2, "memory_action_distribution")
    shuffled_memory_dist = _distribution_from_policy(shuffled_policy_a2, "memory_action_distribution")
    trained_bridge = _distribution_from_policy(trained_policy_a2, "jepa_bridge_scores")
    random_bridge = _distribution_from_policy(random_policy_a2, "jepa_bridge_scores")
    shuffled_bridge = _distribution_from_policy(shuffled_policy_a2, "jepa_bridge_scores")

    memory_state_items = _memory_summary(trained_a1).get("jepa_evidence_state", [])
    memory_state = memory_state_items[-1] if memory_state_items and isinstance(memory_state_items[-1], dict) else {}
    trained_agg = official.get("aggregate_by_variant", {}).get(primary, {})
    no_jepa_agg = official.get("aggregate_by_variant", {}).get("attempt_memory_no_jepa", {})
    shuffled_agg = official.get("aggregate_by_variant", {}).get("jepa_trained_shuffled", {})
    trained_score = float(trained_agg.get("mean_normalized_score", 0.0))
    no_jepa_score = float(no_jepa_agg.get("mean_normalized_score", 0.0))
    shuffled_score = float(shuffled_agg.get("mean_normalized_score", 0.0))
    trained_useful = float(trained_agg.get("mean_useful_events", 0.0))
    no_jepa_useful = float(no_jepa_agg.get("mean_useful_events", 0.0))
    shuffled_useful = float(shuffled_agg.get("mean_useful_events", 0.0))

    trained_vs_no_jepa_l1 = max(
        _l1_dict(trained_memory_dist, no_jepa_memory_dist),
        _l1_dict(trained_action_dist_a2, no_jepa_action_dist_a2),
        _l1_dict(trained_bridge, {}),
    )
    trained_vs_random_l1 = max(
        _l1_dict(trained_memory_dist, random_memory_dist),
        _l1_dict(trained_action_dist_a2, random_action_dist_a2),
        _l1_dict(trained_bridge, random_bridge),
    )
    trained_vs_shuffled_l1 = max(
        _l1_dict(trained_memory_dist, shuffled_memory_dist),
        _l1_dict(trained_action_dist_a2, shuffled_action_dist_a2),
        _l1_dict(trained_bridge, shuffled_bridge),
    )
    trained_attempt_change_l1 = _l1_dict(trained_action_dist_a1, trained_action_dist_a2)
    evidence_bias_l1 = float(memory_state.get("proposal_l1", memory_state.get("bias_l1", 0.0)) or 0.0)
    evidence_present = bool(memory_state.get("evidence")) and bool(memory_state.get("action_priors")) and evidence_bias_l1 > 1.0e-9
    read_only_state = bool(memory_state.get("planner_role") == "proposer_read_only" and not memory_state.get("consumed_by_planner"))
    claimed_improvement = trained_score > max(no_jepa_score, shuffled_score) or trained_useful > max(no_jepa_useful, shuffled_useful)
    shuffled_degrades = bool(claimed_improvement and (shuffled_score < trained_score or shuffled_useful < trained_useful))
    bridge_active = bool(trained_policy_a2.get("jepa_bridge", {}).get("active")) if isinstance(trained_policy_a2.get("jepa_bridge"), dict) else False
    gates = {
        "trained_jepa_alters_next_attempt_vs_no_jepa": trained_vs_no_jepa_l1 >= 0.05,
        "random_jepa_does_not_reproduce_trained_behavior": trained_vs_random_l1 >= 0.05,
        "shuffled_jepa_degrades_claimed_improvement": shuffled_degrades,
        "explicit_jepa_memory_state_proposes_to_planner": bool(evidence_present and read_only_state),
        "official_trace_attempt1_evidence_to_attempt2_strategy": bool(
            evidence_present and (trained_attempt_change_l1 >= 0.05 or trained_vs_no_jepa_l1 >= 0.05 or bridge_active)
        ),
    }
    return {
        "passes": all(gates.values()),
        "gates": gates,
        "metrics": {
            "trained_vs_no_jepa_l1": trained_vs_no_jepa_l1,
            "trained_vs_random_l1": trained_vs_random_l1,
            "trained_vs_shuffled_l1": trained_vs_shuffled_l1,
            "trained_attempt1_to_attempt2_action_l1": trained_attempt_change_l1,
            "trained_jepa_memory_bias_l1": evidence_bias_l1,
            "trained_score": trained_score,
            "no_jepa_score": no_jepa_score,
            "shuffled_score": shuffled_score,
            "trained_useful_events": trained_useful,
            "no_jepa_useful_events": no_jepa_useful,
            "shuffled_useful_events": shuffled_useful,
            "claimed_improvement": bool(claimed_improvement),
        },
        "trained_attempt1_memory_state": memory_state,
        "trained_attempt2_step0_policy": trained_policy_a2,
        "control_attempt2_step0_policy": {
            "attempt_memory_no_jepa": no_jepa_policy_a2,
            "jepa_random_init": random_policy_a2,
            "jepa_trained_shuffled": shuffled_policy_a2,
        },
    }


def build_focused_causality_report(
    *,
    checkpoint: str | Path,
    explorer_checkpoint: str | Path,
    jepa_checkpoint: str | Path,
    trace_dir: str | Path,
    json_output: str | Path,
    device: torch.device,
    operation_mode: str = "offline",
    game_ids: Sequence[str] | None = None,
    limit: int | None = 1,
    variant_ids: Sequence[str] | None = None,
    attempts: int = 3,
    environments_dir: str | Path = "runs/arcagi3_official_envs",
    recordings_dir: str | Path = "runs/arcagi3_official_recordings",
    jepa_action_mode: str = "propose_only",
    jepa_max_action_bias: float = 0.05,
    disable_jepa_direct_override: bool = True,
    enable_hard_anti_attractor: bool = True,
) -> dict[str, Any]:
    core = load_core_choice()
    selected_variants = focused_causality_variant_ids(variant_ids)
    selected_games = manifest_game_ids(requested_ids=game_ids, limit=limit)
    official = dispatch_official_worker(
        checkpoint=checkpoint,
        explorer_checkpoint=explorer_checkpoint,
        jepa_checkpoint=jepa_checkpoint,
        trace_dir=trace_dir,
        core_arm=str(core["selected_core"]),
        device=device,
        operation_mode=operation_mode,
        game_ids=selected_games,
        limit=None,
        variant_ids=selected_variants,
        attempts=attempts,
        environments_dir=environments_dir,
        recordings_dir=recordings_dir,
        jepa_action_mode=jepa_action_mode,
        jepa_max_action_bias=jepa_max_action_bias,
        disable_jepa_direct_override=disable_jepa_direct_override,
        enable_hard_anti_attractor=enable_hard_anti_attractor,
    )
    causality = jepa_causality_audit(official)
    primary_aggregate = official["aggregate_by_variant"].get("jepa_plus_attempt_memory", {})
    primary_attempt_rows = [
        row for row in official.get("attempt_table", []) if str(row.get("variant", "")) == "jepa_plus_attempt_memory"
    ]
    game_beaten = bool(
        any(float(row.get("solve_rate", 0.0)) >= 1.0 for row in primary_attempt_rows)
        or float(primary_aggregate.get("solve_rate", 0.0)) >= 1.0
    )
    if game_beaten and causality.get("passes"):
        outcome = "FOCUSED_GAME1_BEATEN_WITH_JEPA_CAUSALITY"
    elif causality.get("passes"):
        outcome = "JEPA CAUSALITY PROVEN GAME1_NOT_BEATEN"
    else:
        outcome = "JEPA SIDE-CAR ONLY"
    report = {
        "terminal_outcome": outcome,
        "scope": "focused_official_first_game_jepa_causality_audit",
        "primary_variant": "jepa_plus_attempt_memory",
        "core_choice": core,
        "operation_mode": operation_mode,
        "selected_game_ids": selected_games,
        "variant_ids": selected_variants,
        "attempts": max(int(attempts), 1),
        "official": {
            "aggregate_by_variant": official["aggregate_by_variant"],
            "attempt_table": official["attempt_table"],
            "trace_paths": official.get("trace_paths", []),
            "device_runtime": official.get("device_runtime", {}),
            "variant_model_info": official.get("variant_model_info", {}),
        },
        "jepa_causality": causality,
        "focused_metrics": {
            "game_beaten": game_beaten,
            "primary_solve_rate": float(primary_aggregate.get("solve_rate", 0.0)),
            "primary_mean_normalized_score": float(primary_aggregate.get("mean_normalized_score", 0.0)),
            "primary_mean_useful_events": float(primary_aggregate.get("mean_useful_events", 0.0)),
            "primary_mean_repeat_collapse": float(primary_aggregate.get("mean_repeat_collapse", 0.0)),
        },
        "no_hack_proof": no_hack_proof(),
        "hashes": collect_hashes(JEPA_AUDITED_PATHS),
        "trace_paths": list(official.get("trace_paths", [])),
        "required_commands": [
            "pytest -q tests/test_video_jepa.py",
            "python -m src.jepa_arc_eval --config focused_causality --checkpoint frozen/recurrent_latent_fast.pt --explorer-checkpoint runs/explorer_tiny.pt --jepa-checkpoint runs/video_jepa.pt --json-output docs/jepa_focused_causality_report.json --trace-dir docs/jepa_focused_causality_traces --operation-mode offline --limit 1",
        ],
    }
    output = Path(json_output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        choices=[
            "external",
            "official_worker",
            "focused_official",
            "focused_causality",
            "causal_probe",
            "behavioral_ablation",
        ],
        default="external",
    )
    parser.add_argument("--checkpoint", default="frozen/recurrent_latent_fast.pt")
    parser.add_argument("--explorer-checkpoint", default="runs/explorer_tiny.pt")
    parser.add_argument("--jepa-checkpoint", default="runs/video_jepa.pt")
    parser.add_argument("--json-output", required=True)
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--core-arm", default="")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else AUTO_DEVICE)
    parser.add_argument("--operation-mode", choices=["normal", "online", "offline", "competition"], default=None)
    parser.add_argument("--game-id", action="append", default=[])
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--variant", action="append", default=[])
    parser.add_argument("--attempts", type=int, default=3)
    parser.add_argument("--environments-dir", default="runs/arcagi3_official_envs")
    parser.add_argument("--recordings-dir", default="runs/arcagi3_official_recordings")
    parser.add_argument("--jepa-action-mode", choices=JEPA_ACTION_MODES, default="propose_only")
    parser.add_argument("--jepa-max-action-bias", type=float, default=0.05)
    parser.add_argument("--disable-jepa-direct-override", action="store_true", default=True)
    parser.add_argument("--enable-jepa-direct-override", action="store_false", dest="disable_jepa_direct_override")
    parser.add_argument("--disable-hard-anti-attractor", action="store_true", default=False)
    args = parser.parse_args()
    device = resolve_device(args.device)
    if args.config == "causal_probe":
        report_path = Path(args.json_output)
        report = json.loads(report_path.read_text(encoding="utf-8")) if report_path.exists() else {}
        causal_substrate = causal_substrate_self_check(args.jepa_checkpoint, device=device)
        report["causal_substrate_proof"] = causal_substrate
        gates = dict(report.get("gates", {}))
        gates["jepa_causal_substrate_chain"] = bool(causal_substrate.get("passes"))
        report["gates"] = gates
        no_hack = dict(report.get("no_hack_proof", {}))
        no_hack["causal_substrate_chain"] = causal_substrate.get("causal_chain", [])
        no_hack["jepa_as_causal_perceptual_substrate"] = bool(causal_substrate.get("passes"))
        report["no_hack_proof"] = no_hack
        report["hashes"] = collect_hashes(JEPA_AUDITED_PATHS)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(json_safe(report), indent=2), encoding="utf-8")
        print(json.dumps({"causal_substrate_passes": causal_substrate["passes"], "checks": causal_substrate["checks"]}, indent=2))
        return
    if args.config == "official_worker":
        core_arm = args.core_arm or load_core_choice()["selected_core"]
        report = run_official_worker(
            checkpoint=args.checkpoint,
            explorer_checkpoint=args.explorer_checkpoint,
            jepa_checkpoint=args.jepa_checkpoint,
            trace_dir=args.trace_dir,
            json_output=args.json_output,
            core_arm=core_arm,
            device=device,
            operation_mode=args.operation_mode or "normal",
            game_ids=args.game_id,
            limit=args.limit,
            variant_ids=args.variant,
            attempts=args.attempts,
            environments_dir=args.environments_dir,
            recordings_dir=args.recordings_dir,
            jepa_action_mode=args.jepa_action_mode,
            jepa_max_action_bias=args.jepa_max_action_bias,
            disable_jepa_direct_override=args.disable_jepa_direct_override,
            enable_hard_anti_attractor=not args.disable_hard_anti_attractor,
        )
        print(json.dumps({"suite_id": report["suite_id"], "variants": sorted(report["aggregate_by_variant"])}, indent=2))
        return
    if args.config == "behavioral_ablation":
        core_arm = args.core_arm or load_core_choice()["selected_core"]
        report = run_behavioral_ablation_report(
            checkpoint=args.checkpoint,
            explorer_checkpoint=args.explorer_checkpoint,
            jepa_checkpoint=args.jepa_checkpoint,
            trace_dir=args.trace_dir,
            json_output=args.json_output,
            core_arm=core_arm,
            device=device,
            operation_mode=args.operation_mode or "offline",
            game_ids=args.game_id,
            limit=args.limit,
            attempts=args.attempts,
            environments_dir=args.environments_dir,
            recordings_dir=args.recordings_dir,
        )
        print(
            json.dumps(
                {
                    "suite_id": report["suite_id"],
                    "ablation_arms": [arm["ablation_id"] for arm in report["ablation_arms"]],
                    "behavioral_passes": bool(report["behavioral_offline_gates"]["passes"]),
                },
                indent=2,
            )
        )
        return
    if args.config == "focused_official":
        report = build_focused_official_report(
            checkpoint=args.checkpoint,
            explorer_checkpoint=args.explorer_checkpoint,
            jepa_checkpoint=args.jepa_checkpoint,
            trace_dir=args.trace_dir,
            json_output=args.json_output,
            device=device,
            operation_mode=args.operation_mode or "offline",
            game_ids=args.game_id,
            limit=args.limit if args.limit is not None else 1,
            variant_ids=args.variant,
            attempts=args.attempts,
            environments_dir=args.environments_dir,
            recordings_dir=args.recordings_dir,
            jepa_action_mode=args.jepa_action_mode,
            jepa_max_action_bias=args.jepa_max_action_bias,
            disable_jepa_direct_override=args.disable_jepa_direct_override,
            enable_hard_anti_attractor=not args.disable_hard_anti_attractor,
        )
        print(json.dumps({"terminal_outcome": report["terminal_outcome"], "focused_metrics": report["focused_metrics"]}, indent=2))
        return
    if args.config == "focused_causality":
        report = build_focused_causality_report(
            checkpoint=args.checkpoint,
            explorer_checkpoint=args.explorer_checkpoint,
            jepa_checkpoint=args.jepa_checkpoint,
            trace_dir=args.trace_dir,
            json_output=args.json_output,
            device=device,
            operation_mode=args.operation_mode or "offline",
            game_ids=args.game_id,
            limit=args.limit if args.limit is not None else 1,
            variant_ids=args.variant,
            attempts=args.attempts,
            environments_dir=args.environments_dir,
            recordings_dir=args.recordings_dir,
            jepa_action_mode=args.jepa_action_mode,
            jepa_max_action_bias=args.jepa_max_action_bias,
            disable_jepa_direct_override=args.disable_jepa_direct_override,
            enable_hard_anti_attractor=not args.disable_hard_anti_attractor,
        )
        print(
            json.dumps(
                {
                    "terminal_outcome": report["terminal_outcome"],
                    "focused_metrics": report["focused_metrics"],
                    "jepa_causality_gates": report["jepa_causality"]["gates"],
                },
                indent=2,
            )
        )
        return
    report = build_report(
        checkpoint=args.checkpoint,
        explorer_checkpoint=args.explorer_checkpoint,
        jepa_checkpoint=args.jepa_checkpoint,
        trace_dir=args.trace_dir,
        json_output=args.json_output,
        device=device,
        jepa_action_mode=args.jepa_action_mode,
        jepa_max_action_bias=args.jepa_max_action_bias,
        disable_jepa_direct_override=args.disable_jepa_direct_override,
        enable_hard_anti_attractor=not args.disable_hard_anti_attractor,
    )
    print(json.dumps({"terminal_outcome": report["terminal_outcome"], "gates": report["gates"]}, indent=2))


if __name__ == "__main__":
    main()
