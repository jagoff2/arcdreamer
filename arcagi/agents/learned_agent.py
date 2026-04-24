from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np
import torch
from typing import Any

from arcagi.agents.base import BaseAgent
from arcagi.core.action_schema import build_action_schema, build_action_schema_context, no_effect_family_key
from arcagi.core.progress_signals import (
    PolicySupervision,
    action_family as _shared_action_family,
    visible_online_policy_supervision,
    visible_online_usefulness_target,
)
from arcagi.core.types import ActionName, ActionThought, GridObservation, HypothesisProof, PlanOutput, RuntimeThought, StructuredClaim, StructuredState, Transition
from arcagi.memory.episodic import EpisodicMemory
from arcagi.models.encoder import StructuredStateEncoder
from arcagi.models.language import GroundedLanguageModel
from arcagi.models.world_model import RecurrentWorldModel
from arcagi.planning.planner import HybridPlanner
from arcagi.planning.runtime_rule_controller import RuntimeRuleController
from arcagi.planning.rule_induction import EpisodeRuleInducer, ObjectSignature, action_target_signatures
from arcagi.reasoning import EpisodeTheoryManager, HypothesisCompetition, RuleCandidate, TheoryEvent


GENERIC_LANGUAGE_TOKENS: frozenset[str] = frozenset(
    {
        "belief",
        "question",
        "goal",
        "need",
        "test",
        "unknown",
        "uncertain",
        "rule",
        "plan",
        "explore",
        "probe",
        "confirm",
        "commit",
        "move",
        "interact",
        "click",
        "select",
        "wait",
        "toward",
        "target",
        "state",
        "focus",
        "action",
        "direction",
        "color",
        "family",
        "because",
        "frontier",
        "hotspot",
        "anchor",
        "adjacent",
        "active",
        "inactive",
        "present",
        "absent",
        "visible",
        "hidden",
        "positive",
        "negative",
        "none",
        "near",
        "mid",
        "far",
        "p0",
        "p1",
        "p2",
        "p3",
        "p4",
        "p5",
        "n0",
        "n1",
        "n2",
        "n3",
        "n4",
        "n5",
    }
)
CONTENT_LANGUAGE_TOKENS: frozenset[str] = frozenset(
    {
        "proof",
        "support",
        "contradiction",
        "exception",
        "repair",
        "representation",
        "objective",
        "control",
        "mode",
        "split",
        "merge",
        "rebind",
        "collect",
        "unlock",
        "selector",
        "switch",
        "order",
        "delayed",
        "sequence",
        "interactable",
        "blocking",
        "clickable",
        "interface",
        "interface_target",
        "control_binding",
        "reward_model",
        "reward_after_activate",
        "reward_after_interact",
        "reward_after_contact",
        "reward_after_approach",
        "reward_after_avoid",
        "mode_probe_chain",
        "click_then_move",
        "click_then_interact",
        "move_then_interact",
        "bind_then_objective",
        "objective_chain",
        "targets",
        "controls",
        "move_effect",
        "latent_only",
        "selector_candidate",
        "contradiction",
        "effect",
        "red",
        "blue",
        "green",
        "yellow",
        "gray",
        "orange",
        "purple",
        "cyan",
        "c0",
        "c1",
        "c2",
        "c3",
        "c4",
        "c5",
        "c6",
        "c7",
        "c8",
        "c9",
        "c10",
        "c11",
        "up",
        "down",
        "left",
        "right",
    }
)


def _bounded_memory_confidence(value: float) -> float:
    return _clamp(math.tanh(0.7 * max(float(value), 0.0)), lower=0.0, upper=1.0)


def _clamp(value: float, *, lower: float, upper: float) -> float:
    return float(max(min(value, upper), lower))


def _decay_bias_map(mapping: dict[Any, float], *, factor: float, threshold: float = 0.03) -> None:
    for key in list(mapping.keys()):
        decayed = float(mapping[key]) * float(factor)
        if abs(decayed) < threshold:
            del mapping[key]
        else:
            mapping[key] = decayed


def _theory_event_magnitude(salience: float) -> float:
    return 0.18 + (0.42 * _bounded_memory_confidence(float(salience)))


def _action_family(action: ActionName) -> str:
    return _shared_action_family(action)


def _claim_context_tokens(claims: tuple[StructuredClaim, ...], *, limit: int = 6) -> tuple[str, ...]:
    tokens: list[str] = []
    for claim in claims[:limit]:
        tokens.extend(claim.as_tokens())
    return tuple(tokens)


@dataclass(frozen=True)
class LearnedAgentConfig:
    use_language: bool = False
    use_memory: bool = False
    surprise_threshold: float = 0.35
    use_runtime_controller: bool = False
    use_theory_manager: bool = False
    use_online_world_model_adaptation: bool = True
    online_world_model_lr: float = 3e-4
    online_world_model_update_steps: int = 1
    exploration_epsilon: float = 0.0
    online_contrastive_action_sample_limit: int = 0


@dataclass
class LocalModelPatch:
    value_shift: float = 0.0
    policy_shift: float = 0.0
    usefulness_shift: float = 0.0
    uncertainty_shift: float = 0.0
    entries: int = 0

    def observe(
        self,
        *,
        reward_error: float,
        usefulness_error: float,
        policy_error: float,
        delta_error: float,
        predicted_uncertainty: float,
    ) -> None:
        rate = 0.35 / float((self.entries + 1) ** 0.5)
        self.value_shift = _clamp(
            self.value_shift + (rate * ((0.6 * reward_error) + (0.4 * usefulness_error) - (0.08 * delta_error))),
            lower=-2.5,
            upper=2.5,
        )
        self.policy_shift = _clamp(
            self.policy_shift + (rate * policy_error),
            lower=-1.5,
            upper=1.5,
        )
        self.usefulness_shift = _clamp(
            self.usefulness_shift + (rate * usefulness_error),
            lower=-1.5,
            upper=1.5,
        )
        self.uncertainty_shift = _clamp(
            self.uncertainty_shift + (rate * (delta_error - predicted_uncertainty)),
            lower=-1.5,
            upper=1.5,
        )
        self.entries += 1


@dataclass
class _ActionSemanticsStats:
    trials: int = 0
    progress_sum: float = 0.0
    positive_sum: float = 0.0
    harm_sum: float = 0.0
    visible_effects: int = 0
    no_effects: int = 0
    surprise_sum: float = 0.0
    prediction_error_sum: float = 0.0
    predicted_uncertainty_sum: float = 0.0

    def observe(
        self,
        *,
        progress: float,
        visible_effect: bool,
        surprise: float,
        prediction_error: float,
        predicted_uncertainty: float,
    ) -> None:
        self.trials += 1
        self.progress_sum += float(progress)
        self.positive_sum += max(float(progress), 0.0)
        self.harm_sum += max(-float(progress), 0.0)
        self.visible_effects += int(bool(visible_effect))
        self.no_effects += int((not visible_effect) and float(progress) <= 0.02)
        self.surprise_sum += max(float(surprise), 0.0)
        self.prediction_error_sum += max(float(prediction_error), 0.0)
        self.predicted_uncertainty_sum += max(float(predicted_uncertainty), 0.0)

    def mean_progress(self) -> float:
        if self.trials <= 0:
            return 0.0
        return float(self.progress_sum) / float(self.trials)

    def positive_rate(self) -> float:
        if self.trials <= 0:
            return 0.0
        return float(self.positive_sum) / float(self.trials)

    def harm_rate(self) -> float:
        if self.trials <= 0:
            return 0.0
        return float(self.harm_sum) / float(self.trials)

    def visible_rate(self) -> float:
        if self.trials <= 0:
            return 0.0
        return float(self.visible_effects) / float(self.trials)

    def no_effect_rate(self) -> float:
        if self.trials <= 0:
            return 0.0
        return float(self.no_effects) / float(self.trials)

    def prediction_error(self) -> float:
        if self.trials <= 0:
            return 0.0
        return float(self.prediction_error_sum) / float(self.trials)

    def predicted_uncertainty(self) -> float:
        if self.trials <= 0:
            return 0.0
        return float(self.predicted_uncertainty_sum) / float(self.trials)

    def to_dict(self) -> dict[str, float | int]:
        return {
            "trials": int(self.trials),
            "mean_progress": float(self.mean_progress()),
            "positive_rate": float(self.positive_rate()),
            "harm_rate": float(self.harm_rate()),
            "visible_rate": float(self.visible_rate()),
            "no_effect_rate": float(self.no_effect_rate()),
            "prediction_error": float(self.prediction_error()),
            "predicted_uncertainty": float(self.predicted_uncertainty()),
        }


@dataclass(frozen=True)
class ActionSemanticsScore:
    action_bias: float
    diagnostic_value: float
    expected_progress: float
    uncertainty: float
    evidence: float
    prediction_error_gap: float
    no_effect_rate: float


class ActionSemanticsSelfModel:
    """Online self-model of what the agent's own actions currently do.

    The model stores only evidence produced by the current agent-environment
    loop. It does not contain game-specific action scripts or graph search.
    """

    def __init__(self) -> None:
        self.action_stats: dict[ActionName, _ActionSemanticsStats] = {}
        self.family_stats: dict[str, _ActionSemanticsStats] = {}
        self.type_stats: dict[str, _ActionSemanticsStats] = {}
        self.last_update: dict[str, float | str | bool] = {}
        self.last_scores: dict[ActionName, ActionSemanticsScore] = {}

    def reset(self) -> None:
        self.action_stats.clear()
        self.family_stats.clear()
        self.type_stats.clear()
        self.last_update = {}
        self.last_scores = {}

    def observe(
        self,
        transition: Transition,
        *,
        state_change: float,
        progress_signal: float,
        predicted_reward: float,
        predicted_usefulness: float,
        predicted_uncertainty: float,
        delta_error: float,
        surprise: float,
    ) -> None:
        context = build_action_schema_context(transition.state.affordances, dict(transition.state.action_roles))
        schema = build_action_schema(transition.action, context)
        visible_effect = bool(abs(float(transition.reward)) > 1e-9 or float(state_change) > 1e-6)
        prediction_error = (
            abs(float(transition.reward) - float(predicted_reward))
            + abs(float(progress_signal) - float(predicted_usefulness))
            + (0.15 * max(float(delta_error), 0.0))
        )
        for mapping, key in (
            (self.action_stats, transition.action),
            (self.family_stats, schema.family),
            (self.type_stats, schema.action_type),
        ):
            stats = mapping.setdefault(str(key), _ActionSemanticsStats())
            stats.observe(
                progress=float(progress_signal),
                visible_effect=visible_effect,
                surprise=float(surprise),
                prediction_error=prediction_error,
                predicted_uncertainty=float(predicted_uncertainty),
            )
        self.last_update = {
            "action": str(transition.action),
            "family": str(schema.family),
            "action_type": str(schema.action_type),
            "progress_signal": float(progress_signal),
            "state_change": float(state_change),
            "visible_effect": bool(visible_effect),
            "prediction_error": float(prediction_error),
            "predicted_uncertainty": float(predicted_uncertainty),
            "surprise": float(surprise),
        }

    def score_actions(
        self,
        state: StructuredState,
        thought: RuntimeThought | None = None,
    ) -> dict[ActionName, ActionSemanticsScore]:
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        scored: dict[ActionName, ActionSemanticsScore] = {}
        for action in state.affordances:
            schema = build_action_schema(action, context)
            action_stats = self.action_stats.get(action)
            family_stats = self.family_stats.get(schema.family)
            type_stats = self.type_stats.get(schema.action_type)
            action_trials = 0 if action_stats is None else action_stats.trials
            family_trials = 0 if family_stats is None else family_stats.trials
            type_trials = 0 if type_stats is None else type_stats.trials
            evidence = float(action_trials) + (0.35 * float(family_trials)) + (0.15 * float(type_trials))

            def blend(method_name: str, *, include_type: bool = False) -> float:
                numer = 0.0
                denom = 0.0
                sources = [(action_stats, 1.0), (family_stats, 0.35)]
                if include_type:
                    sources.append((type_stats, 0.15))
                for stats, weight in sources:
                    if stats is None or stats.trials <= 0:
                        continue
                    numer += weight * float(getattr(stats, method_name)())
                    denom += weight
                return 0.0 if denom <= 0.0 else numer / denom

            expected_progress = blend("mean_progress")
            positive_rate = blend("positive_rate")
            harm_rate = blend("harm_rate")
            visible_rate = blend("visible_rate")
            no_effect_rate = blend("no_effect_rate")
            prediction_error = blend("prediction_error")
            predicted_uncertainty = blend("predicted_uncertainty")
            diagnostic_error = blend("prediction_error", include_type=True)
            diagnostic_uncertainty = blend("predicted_uncertainty", include_type=True)
            error_gap = max(0.0, prediction_error - predicted_uncertainty)
            diagnostic_error_gap = max(0.0, diagnostic_error - diagnostic_uncertainty)
            thought_action = None if thought is None else thought.for_action(action)
            model_uncertainty = 0.0 if thought_action is None else max(float(thought_action.uncertainty), 0.0)
            model_diagnostic = 0.0 if thought_action is None else max(float(thought_action.diagnostic_value), 0.0)
            uncertainty = _clamp(
                (1.0 / math.sqrt(evidence + 1.0)) + (0.20 * model_uncertainty) + (0.10 * diagnostic_error_gap),
                lower=0.0,
                upper=2.0,
            )
            productive_evidence = max(expected_progress, 0.0) + positive_rate
            action_bias = (
                expected_progress
                + (0.45 * positive_rate)
                + (0.06 * visible_rate * min(productive_evidence, 1.0))
                + (0.05 * error_gap * min(productive_evidence, 1.0))
                - (0.70 * harm_rate)
                - (0.35 * no_effect_rate)
            )
            diagnostic_value = (
                (0.45 * uncertainty)
                + (0.20 * model_diagnostic)
                + (0.20 * diagnostic_error_gap)
                + (0.08 * visible_rate)
                - (0.35 * no_effect_rate)
            )
            if schema.action_type in {"reset", "undo", "wait"}:
                diagnostic_value *= 0.35
                action_bias -= 0.15
            scored[action] = ActionSemanticsScore(
                action_bias=_clamp(action_bias, lower=-1.6, upper=1.8),
                diagnostic_value=_clamp(diagnostic_value, lower=0.0, upper=1.5),
                expected_progress=_clamp(expected_progress, lower=-1.5, upper=1.5),
                uncertainty=float(uncertainty),
                evidence=float(evidence),
                prediction_error_gap=float(error_gap),
                no_effect_rate=float(no_effect_rate),
            )
        self.last_scores = scored
        return scored

    def augment_thought(
        self,
        state: StructuredState,
        thought: RuntimeThought,
        scores: dict[ActionName, ActionSemanticsScore],
    ) -> RuntimeThought:
        if not scores:
            return thought
        ranked_diagnostic = sorted(
            scores.items(),
            key=lambda item: (item[1].diagnostic_value, item[1].uncertainty, -item[1].no_effect_rate),
            reverse=True,
        )
        ranked_productive = sorted(
            scores.items(),
            key=lambda item: (item[1].action_bias, item[1].expected_progress, item[1].evidence),
            reverse=True,
        )
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        question_tokens = list(thought.question_tokens)
        plan_tokens = list(thought.plan_tokens)
        belief_tokens = list(thought.belief_tokens)
        claims = list(thought.claims)
        if ranked_diagnostic:
            action, score = ranked_diagnostic[0]
            schema = build_action_schema(action, context)
            question_tokens.extend(("question", "need", "test", "action", schema.action_type, "unknown"))
            if schema.action_type in {"click", "select", "raw"}:
                question_tokens.extend(("control", "mode", "hidden"))
            plan_tokens.extend(("plan", "test", "action", schema.action_type))
            claims.append(
                StructuredClaim(
                    claim_type="self_model",
                    subject=schema.action_type,
                    relation="uncertain",
                    object="action_effect",
                    confidence=min(float(score.uncertainty), 1.0),
                    evidence=float(score.evidence),
                    salience=float(score.diagnostic_value),
                )
            )
        if ranked_productive:
            action, score = ranked_productive[0]
            if score.evidence > 0.0 and score.action_bias > 0.05:
                schema = build_action_schema(action, context)
                belief_tokens.extend(("belief", "action", schema.action_type, "positive"))
                claims.append(
                    StructuredClaim(
                        claim_type="self_model",
                        subject=schema.action_type,
                        relation="productive",
                        object="episode",
                        confidence=min(max(score.action_bias, 0.0), 1.0),
                        evidence=float(score.evidence),
                        salience=float(score.action_bias),
                    )
                )
        return RuntimeThought(
            belief_tokens=tuple(dict.fromkeys(belief_tokens)),
            question_tokens=tuple(dict.fromkeys(question_tokens)),
            plan_tokens=tuple(dict.fromkeys(plan_tokens)),
            actions=thought.actions,
            claims=tuple(claims),
            world_model_calls=thought.world_model_calls,
        )

    def action_biases(self, scores: dict[ActionName, ActionSemanticsScore]) -> dict[ActionName, float]:
        return {action: score.action_bias for action, score in scores.items() if abs(score.action_bias) > 1e-6}

    def diagnostic_scores(self, scores: dict[ActionName, ActionSemanticsScore]) -> dict[ActionName, float]:
        return {action: score.diagnostic_value for action, score in scores.items() if score.diagnostic_value > 1e-6}

    def diagnostics(self) -> dict[str, object]:
        ranked = sorted(
            self.last_scores.items(),
            key=lambda item: (item[1].action_bias + item[1].diagnostic_value),
            reverse=True,
        )
        return {
            "last_update": dict(self.last_update),
            "action_stats": {action: stats.to_dict() for action, stats in sorted(self.action_stats.items())},
            "family_count": int(len(self.family_stats)),
            "type_count": int(len(self.type_stats)),
            "last_scores": {
                action: {
                    "action_bias": float(score.action_bias),
                    "diagnostic_value": float(score.diagnostic_value),
                    "expected_progress": float(score.expected_progress),
                    "uncertainty": float(score.uncertainty),
                    "evidence": float(score.evidence),
                    "prediction_error_gap": float(score.prediction_error_gap),
                    "no_effect_rate": float(score.no_effect_rate),
                }
                for action, score in ranked[:8]
            },
        }


class AgentSelfBeliefState:
    """Explicit first-person belief state grounded in the agent's own updates."""

    def __init__(self) -> None:
        self.transitions = 0
        self.online_updates = 0
        self.level_boundaries = 0
        self.levels_completed = 0
        self.progress_ema = 0.0
        self.reward_ema = 0.0
        self.prediction_error_ema = 0.0
        self.predicted_uncertainty_ema = 0.0
        self.surprise_ema = 0.0
        self.adaptation_delta_ema = 0.0
        self.last_progress_signal = 0.0
        self.last_prediction_error = 0.0
        self.last_predicted_uncertainty = 0.0
        self.last_online_update = False

    def reset(self) -> None:
        self.__init__()

    def reset_level(self) -> None:
        self.last_progress_signal = 0.0
        self.last_prediction_error = 0.0
        self.last_predicted_uncertainty = 0.0
        self.last_online_update = False

    def observe(
        self,
        transition: Transition,
        *,
        progress_signal: float,
        prediction_error: float,
        predicted_uncertainty: float,
        surprise: float,
        online_update: bool,
    ) -> None:
        alpha = 0.18
        previous_progress = self.progress_ema
        progress = float(progress_signal)
        self.transitions += 1
        self.online_updates += int(bool(online_update))
        self.progress_ema = _ema(self.progress_ema, progress, alpha=alpha, initialized=self.transitions > 1)
        self.reward_ema = _ema(self.reward_ema, float(transition.reward), alpha=alpha, initialized=self.transitions > 1)
        self.prediction_error_ema = _ema(
            self.prediction_error_ema,
            max(float(prediction_error), 0.0),
            alpha=alpha,
            initialized=self.transitions > 1,
        )
        self.predicted_uncertainty_ema = _ema(
            self.predicted_uncertainty_ema,
            max(float(predicted_uncertainty), 0.0),
            alpha=alpha,
            initialized=self.transitions > 1,
        )
        self.surprise_ema = _ema(
            self.surprise_ema,
            max(float(surprise), 0.0),
            alpha=alpha,
            initialized=self.transitions > 1,
        )
        self.adaptation_delta_ema = _ema(
            self.adaptation_delta_ema,
            progress - previous_progress,
            alpha=0.12,
            initialized=self.transitions > 1,
        )
        self.last_progress_signal = progress
        self.last_prediction_error = max(float(prediction_error), 0.0)
        self.last_predicted_uncertainty = max(float(predicted_uncertainty), 0.0)
        self.last_online_update = bool(online_update)
        try:
            level_delta = int(transition.info.get("level_delta", 0) or 0)
        except Exception:
            level_delta = 0
        if bool(transition.info.get("level_boundary", False)):
            self.level_boundaries += 1
        if level_delta > 0:
            self.levels_completed += level_delta

    @property
    def prediction_gap(self) -> float:
        return max(0.0, self.prediction_error_ema - self.predicted_uncertainty_ema)

    def _competence_token(self) -> str:
        if self.progress_ema >= 0.18 or self.reward_ema > 0.08:
            return "high"
        if self.progress_ema >= -0.03:
            return "mid"
        return "low"

    def _reliability_token(self) -> str:
        gap = self.prediction_gap
        if gap <= 0.04 and self.transitions >= 3:
            return "high"
        if gap <= 0.16:
            return "mid"
        return "low"

    def _trend_token(self) -> str:
        if self.adaptation_delta_ema >= 0.03:
            return "positive"
        if self.adaptation_delta_ema <= -0.03:
            return "negative"
        return "uncertain"

    def tokens(self) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
        competence = self._competence_token()
        reliability = self._reliability_token()
        trend = self._trend_token()
        belief = ("belief", "agent", "state", competence, "progress", trend)
        if self.online_updates > 0:
            belief = belief + ("support",)
        if reliability == "low":
            belief = belief + ("uncertain",)
        if reliability == "low" or competence == "low":
            question = ("question", "need", "test", "agent", "uncertain", "action")
            plan = ("plan", "test", "action", "because", "uncertain")
        elif trend == "positive":
            question = ("question", "need", "confirm", "agent", "positive")
            plan = ("plan", "commit", "action", "because", "support")
        else:
            question = ("question", "need", "test", "agent", "action")
            plan = ("plan", "test", "action", "because", "uncertain")
        return belief, question, plan

    def claims(self) -> tuple[StructuredClaim, ...]:
        competence = self._competence_token()
        reliability = self._reliability_token()
        trend = self._trend_token()
        return (
            StructuredClaim(
                claim_type="self_model",
                subject="agent",
                relation="positive" if competence != "low" else "negative",
                object="progress",
                confidence=0.8 if competence == "high" else 0.55 if competence == "mid" else 0.35,
                evidence=abs(self.progress_ema),
                salience=abs(self.progress_ema),
            ),
            StructuredClaim(
                claim_type="self_model",
                subject="agent",
                relation="uncertain" if reliability == "low" else "support",
                object="state",
                confidence=0.8 if reliability == "high" else 0.55 if reliability == "mid" else 0.35,
                evidence=self.prediction_gap,
                salience=self.prediction_gap,
            ),
            StructuredClaim(
                claim_type="self_model",
                subject="agent",
                relation=trend,
                object="progress",
                confidence=min(0.9, 0.35 + abs(self.adaptation_delta_ema)),
                evidence=abs(self.adaptation_delta_ema),
                salience=abs(self.adaptation_delta_ema),
            ),
        )

    def augment_thought(self, thought: RuntimeThought) -> RuntimeThought:
        belief, question, plan = self.tokens()
        return RuntimeThought(
            belief_tokens=tuple(dict.fromkeys(thought.belief_tokens + belief)),
            question_tokens=tuple(dict.fromkeys(thought.question_tokens + question)),
            plan_tokens=tuple(dict.fromkeys(thought.plan_tokens + plan)),
            actions=thought.actions,
            claims=tuple(thought.claims) + self.claims(),
            world_model_calls=thought.world_model_calls,
        )

    def diagnostics(self) -> dict[str, object]:
        belief, question, plan = self.tokens()
        return {
            "transitions": int(self.transitions),
            "online_updates": int(self.online_updates),
            "level_boundaries": int(self.level_boundaries),
            "levels_completed": int(self.levels_completed),
            "progress_ema": float(self.progress_ema),
            "reward_ema": float(self.reward_ema),
            "prediction_error_ema": float(self.prediction_error_ema),
            "predicted_uncertainty_ema": float(self.predicted_uncertainty_ema),
            "prediction_gap": float(self.prediction_gap),
            "surprise_ema": float(self.surprise_ema),
            "adaptation_delta_ema": float(self.adaptation_delta_ema),
            "belief_tokens": belief,
            "question_tokens": question,
            "plan_tokens": plan,
        }


def _ema(previous: float, value: float, *, alpha: float, initialized: bool) -> float:
    if not initialized:
        return float(value)
    return float(((1.0 - alpha) * previous) + (alpha * value))


class LearnedPlanningAgent(BaseAgent):
    def __init__(
        self,
        name: str,
        encoder: StructuredStateEncoder,
        world_model: RecurrentWorldModel,
        planner: HybridPlanner,
        language_model: GroundedLanguageModel | None = None,
        episodic_memory: EpisodicMemory | None = None,
        config: LearnedAgentConfig | None = None,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(name=name)
        self.encoder = encoder
        self.world_model = world_model
        self.planner = planner
        self.language_model = language_model
        self.episodic_memory = episodic_memory
        self.config = config or LearnedAgentConfig()
        self.device = device or torch.device("cpu")
        self.gradient_world_model_adaptation = self.config.use_online_world_model_adaptation and self.device.type != "cpu"
        self.hidden: torch.Tensor | None = None
        self.last_hidden_input: torch.Tensor | None = None
        self.last_latent: torch.Tensor | None = None
        self.last_prediction = None
        self.last_runtime_thought = None
        self.last_plan_scores: dict[str, float] = {}
        self.recent_actions: list[ActionName] = []
        self.latest_rule_candidates: tuple[RuleCandidate, ...] = ()
        self.latest_competitions: tuple[HypothesisCompetition, ...] = ()
        self.online_action_bias: dict[ActionName, float] = {}
        self.online_context_bias: dict[tuple[str, ObjectSignature], float] = {}
        self.local_action_patches: dict[ActionName, LocalModelPatch] = {}
        self.local_context_patches: dict[tuple[str, ObjectSignature], LocalModelPatch] = {}
        self.family_bias: dict[str, float] = defaultdict(float)
        self.language_token_scores: dict[str, float] = defaultdict(float)
        self.pending_belief_tokens: tuple[str, ...] = ()
        self.pending_question_tokens: tuple[str, ...] = ()
        self.pending_plan_tokens: tuple[str, ...] = ()
        self.stable_belief_tokens: tuple[str, ...] = ()
        self.stable_question_tokens: tuple[str, ...] = ()
        self.stable_plan_tokens: tuple[str, ...] = ()
        self.evidence_steps = 0
        self.runtime_rule_controller = RuntimeRuleController() if self.config.use_runtime_controller else None
        self.theory_manager = EpisodeTheoryManager() if self.config.use_theory_manager else None
        self.rule_inducer = EpisodeRuleInducer()
        self.self_model = ActionSemanticsSelfModel()
        self.self_belief = AgentSelfBeliefState()
        self.latest_self_model_scores: dict[ActionName, ActionSemanticsScore] = {}
        self._world_model_base_state = copy.deepcopy(self.world_model.state_dict())
        self.world_model_optimizer = (
            torch.optim.Adam(self.world_model.parameters(), lr=self.config.online_world_model_lr)
            if self.gradient_world_model_adaptation
            else None
        )
        self.global_action_counts: dict[ActionName, int] = defaultdict(int)
        self.global_action_delta_sums: dict[ActionName, float] = defaultdict(float)
        self.global_action_reward_sums: dict[ActionName, float] = defaultdict(float)
        self.global_action_no_effect_counts: dict[ActionName, int] = defaultdict(int)
        self.family_counts: dict[str, int] = defaultdict(int)
        self.family_bins: dict[str, set[tuple[int, int]]] = defaultdict(set)
        self.family_no_effect_counts: dict[str, int] = defaultdict(int)
        self.level_action_counts: dict[ActionName, int] = defaultdict(int)
        self.level_action_progress_sums: dict[ActionName, float] = defaultdict(float)
        self.level_action_no_objective_counts: dict[ActionName, int] = defaultdict(int)
        self.level_family_counts: dict[str, int] = defaultdict(int)
        self.level_family_progress_sums: dict[str, float] = defaultdict(float)
        self.level_family_no_objective_counts: dict[str, int] = defaultdict(int)
        self.stuck_steps = 0
        self.objective_stall_steps = 0
        self.level_stall_steps = 0
        self.max_levels_completed_observed = 0

    def _clear_level_evidence(self) -> None:
        self.level_action_counts.clear()
        self.level_action_progress_sums.clear()
        self.level_action_no_objective_counts.clear()
        self.level_family_counts.clear()
        self.level_family_progress_sums.clear()
        self.level_family_no_objective_counts.clear()

    def reset_episode(self) -> None:
        super().reset_episode()
        if self.gradient_world_model_adaptation and self.world_model_optimizer is not None:
            self.world_model.load_state_dict(self._world_model_base_state)
            self.world_model_optimizer = torch.optim.Adam(
                self.world_model.parameters(),
                lr=self.config.online_world_model_lr,
            )
            self.world_model.eval()
        self.hidden = None
        self.last_hidden_input = None
        self.last_latent = None
        self.last_prediction = None
        self.last_runtime_thought = None
        self.last_plan_scores = {}
        self.recent_actions = []
        self.latest_rule_candidates = ()
        self.latest_competitions = ()
        self.online_action_bias = {}
        self.online_context_bias = {}
        self.local_action_patches = {}
        self.local_context_patches = {}
        self.family_bias.clear()
        self.language_token_scores.clear()
        self.pending_belief_tokens = ()
        self.pending_question_tokens = ()
        self.pending_plan_tokens = ()
        self.stable_belief_tokens = ()
        self.stable_question_tokens = ()
        self.stable_plan_tokens = ()
        self.evidence_steps = 0
        self.global_action_counts.clear()
        self.global_action_delta_sums.clear()
        self.global_action_reward_sums.clear()
        self.global_action_no_effect_counts.clear()
        self.family_counts.clear()
        self.family_bins.clear()
        self.family_no_effect_counts.clear()
        self._clear_level_evidence()
        self.stuck_steps = 0
        self.objective_stall_steps = 0
        self.level_stall_steps = 0
        self.max_levels_completed_observed = 0
        self.self_model.reset()
        self.self_belief.reset()
        self.latest_self_model_scores = {}
        if self.runtime_rule_controller is not None:
            self.runtime_rule_controller.reset_episode()
        if self.theory_manager is not None:
            self.theory_manager.reset_episode()
        self.rule_inducer.clear()

    def reset_level(self) -> None:
        super().reset_level()
        self.hidden = None
        self.last_hidden_input = None
        self.last_latent = None
        self.last_prediction = None
        self.last_runtime_thought = None
        self.last_plan_scores = {}
        self.recent_actions = []
        self.latest_rule_candidates = ()
        self.latest_competitions = ()
        self.pending_belief_tokens = self.stable_belief_tokens
        self.pending_question_tokens = self.stable_question_tokens
        self.pending_plan_tokens = self.stable_plan_tokens
        if self.stable_belief_tokens or self.stable_question_tokens or self.stable_plan_tokens:
            self.latest_language = (
                self.stable_belief_tokens
                + ("|",)
                + self.stable_question_tokens
                + ("|",)
                + self.stable_plan_tokens
            )
        self._clear_level_evidence()
        self.stuck_steps = 0
        self.self_belief.reset_level()
        self.latest_self_model_scores = {}
        reset_level = getattr(self.runtime_rule_controller, "reset_level", None)
        if callable(reset_level):
            reset_level()
        reset_level = getattr(self.theory_manager, "reset_level", None)
        if callable(reset_level):
            reset_level()

    def reset_all(self) -> None:
        super().reset_all()
        if self.gradient_world_model_adaptation and self.world_model_optimizer is not None:
            self.world_model.load_state_dict(self._world_model_base_state)
            self.world_model_optimizer = torch.optim.Adam(
                self.world_model.parameters(),
                lr=self.config.online_world_model_lr,
            )
            self.world_model.eval()
        self.hidden = None
        self.last_hidden_input = None
        self.last_latent = None
        self.last_prediction = None
        self.last_runtime_thought = None
        self.last_plan_scores = {}
        self.recent_actions = []
        self.latest_rule_candidates = ()
        self.latest_competitions = ()
        self.online_action_bias = {}
        self.online_context_bias = {}
        self.local_action_patches = {}
        self.local_context_patches = {}
        self.family_bias.clear()
        self.language_token_scores.clear()
        self.pending_belief_tokens = ()
        self.pending_question_tokens = ()
        self.pending_plan_tokens = ()
        self.stable_belief_tokens = ()
        self.stable_question_tokens = ()
        self.stable_plan_tokens = ()
        self.evidence_steps = 0
        self.global_action_counts.clear()
        self.global_action_delta_sums.clear()
        self.global_action_reward_sums.clear()
        self.global_action_no_effect_counts.clear()
        self.family_counts.clear()
        self.family_bins.clear()
        self.family_no_effect_counts.clear()
        self._clear_level_evidence()
        self.stuck_steps = 0
        self.objective_stall_steps = 0
        self.level_stall_steps = 0
        self.max_levels_completed_observed = 0
        self.self_model.reset()
        self.self_belief.reset()
        self.latest_self_model_scores = {}
        if self.runtime_rule_controller is not None:
            self.runtime_rule_controller.reset_all()
        if self.theory_manager is not None:
            self.theory_manager.reset_episode()
        self.rule_inducer.clear()
        if self.episodic_memory is not None:
            self.episodic_memory.clear()

    def _language_token_threshold(self, token: str) -> float:
        if token in GENERIC_LANGUAGE_TOKENS:
            return -1.0
        if token in CONTENT_LANGUAGE_TOKENS:
            return 0.25 if self.evidence_steps > 0 else 0.55
        return 0.1

    def _filtered_language_tokens(self, tokens: tuple[str, ...]) -> tuple[str, ...]:
        filtered: list[str] = []
        for token in tokens:
            if token in GENERIC_LANGUAGE_TOKENS:
                filtered.append(token)
                continue
            score = float(self.language_token_scores.get(token, 0.0))
            if score >= self._language_token_threshold(token):
                filtered.append(token)
        return tuple(filtered)

    def _stabilize_runtime_thought(self, runtime_thought: RuntimeThought) -> RuntimeThought:
        filtered_belief = self._filtered_language_tokens(runtime_thought.belief_tokens)
        filtered_question = self._filtered_language_tokens(runtime_thought.question_tokens)
        filtered_plan = self._filtered_language_tokens(runtime_thought.plan_tokens)
        if not filtered_belief and self.evidence_steps > 0:
            filtered_belief = tuple(token for token in runtime_thought.belief_tokens if token in GENERIC_LANGUAGE_TOKENS)
        if not filtered_question:
            filtered_question = tuple(token for token in runtime_thought.question_tokens if token in GENERIC_LANGUAGE_TOKENS)
        if not filtered_plan:
            filtered_plan = tuple(token for token in runtime_thought.plan_tokens if token in GENERIC_LANGUAGE_TOKENS)
        return RuntimeThought(
            belief_tokens=filtered_belief,
            question_tokens=filtered_question,
            plan_tokens=filtered_plan,
            actions=runtime_thought.actions,
            claims=runtime_thought.claims,
            world_model_calls=runtime_thought.world_model_calls,
        )

    def _state_claims(self, state) -> tuple[StructuredClaim, ...]:
        claims: list[StructuredClaim] = []
        for key, value in sorted(state.flags):
            if value in {"", "0", "false", "False"}:
                continue
            claim_type = "belief" if key.startswith("belief_") else "flag"
            confidence = 0.75 if claim_type == "belief" else 1.0
            claims.append(
                StructuredClaim(
                    claim_type=claim_type,
                    subject=key,
                    relation="=",
                    object=value,
                    confidence=confidence,
                    evidence=confidence,
                    salience=confidence,
                )
            )
        for key, value in sorted(state.inventory):
            if not value:
                continue
            claim_type = "belief" if key.startswith("belief_") else "inventory"
            confidence = 0.75 if claim_type == "belief" else 1.0
            claims.append(
                StructuredClaim(
                    claim_type=claim_type,
                    subject=key,
                    relation="=",
                    object=value,
                    confidence=confidence,
                    evidence=confidence,
                    salience=confidence,
                )
            )
        return tuple(claims)

    def _episode_family_claims(self, bias_source: dict[str, float] | None = None) -> tuple[StructuredClaim, ...]:
        source = self.family_bias if bias_source is None else bias_source
        claims: list[StructuredClaim] = []
        for family, bias in sorted(source.items(), key=lambda item: abs(item[1]), reverse=True):
            if abs(bias) < 0.5:
                continue
            relation = "productive" if bias > 0.0 else "stalled"
            confidence = min(abs(bias) / 2.0, 1.0)
            claims.append(
                StructuredClaim(
                    claim_type="action_family",
                    subject=family,
                    relation=relation,
                    object="episode",
                    confidence=confidence,
                    evidence=abs(bias),
                    salience=abs(bias),
                )
            )
            if len(claims) >= 3:
                break
        return tuple(claims)

    @staticmethod
    def _merge_bias_maps(*mappings: dict[Any, float] | None) -> dict[Any, float]:
        merged: defaultdict[Any, float] = defaultdict(float)
        for mapping in mappings:
            if not mapping:
                continue
            for key, value in mapping.items():
                merged[key] += float(value)
        return {key: _clamp(value, lower=-3.0, upper=3.0) for key, value in merged.items()}

    def _objective_stall_pressure(self) -> float:
        stall_steps = max(float(self.objective_stall_steps), float(self.level_stall_steps))
        if self.max_levels_completed_observed <= 0:
            return max(0.0, min((stall_steps - 220.0) / 420.0, 1.0))
        return max(0.0, min((stall_steps - 320.0) / 384.0, 1.0))

    def _objective_stall_action_biases(self, state: StructuredState) -> dict[ActionName, float]:
        pressure = self._objective_stall_pressure()
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        game_state = str(state.inventory_dict().get("interface_game_state", "")).upper()
        terminal_reset_ok = "GAME_OVER" in game_state or "SESSION_ENDED" in game_state
        biases: dict[ActionName, float] = {}
        for action in state.affordances:
            schema = build_action_schema(action, context)
            count = int(self.global_action_counts.get(action, 0))
            no_effect_count = int(self.global_action_no_effect_counts.get(action, 0))
            no_effect_rate = float(no_effect_count) / float(max(count, 1))
            if schema.action_type == "reset":
                if not terminal_reset_ok:
                    biases[action] = -2.5 - (1.5 * pressure)
                continue
            if schema.action_type == "undo" and not terminal_reset_ok:
                biases[action] = -1.4 - (1.0 * pressure)
                continue
            if schema.action_type == "wait":
                biases[action] = -1.0 - (0.8 * pressure)
                continue
            if pressure <= 0.0:
                continue
            if schema.action_type == "move":
                level_count = int(self.level_action_counts.get(action, 0))
                level_mean = float(self.level_action_progress_sums.get(action, 0.0)) / float(max(level_count, 1))
                level_no_objective = (
                    float(self.level_action_no_objective_counts.get(action, 0)) / float(max(level_count, 1))
                )
                stale_level = level_count >= 8 and level_mean <= 0.04 and level_no_objective >= 0.85
                count_scale = min(float(max(count, level_count)) / 32.0, 1.0)
                penalty = (3.2 + (1.4 if stale_level else 0.0)) * pressure * (0.35 + (0.65 * count_scale))
                biases[action] = biases.get(action, 0.0) - penalty
                continue
            if schema.action_type in {"click", "select", "interact"} and (count < 4 or no_effect_rate < 0.9):
                novelty = 1.0 / math.sqrt(float(count) + 1.0)
                biases[action] = biases.get(action, 0.0) + (2.4 * pressure * novelty)
        return biases

    def _objective_stall_family_biases(self, state: StructuredState) -> dict[str, float]:
        pressure = self._objective_stall_pressure()
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        game_state = str(state.inventory_dict().get("interface_game_state", "")).upper()
        terminal_reset_ok = "GAME_OVER" in game_state or "SESSION_ENDED" in game_state
        biases: defaultdict[str, float] = defaultdict(float)
        for action in state.affordances:
            schema = build_action_schema(action, context)
            if schema.action_type == "reset":
                if not terminal_reset_ok:
                    biases[schema.family] -= 1.2 + (1.8 * pressure)
                continue
            if schema.action_type == "undo" and not terminal_reset_ok:
                biases[schema.family] -= 0.8 + (1.0 * pressure)
                continue
            if schema.action_type == "wait":
                biases[schema.family] -= 0.6 + (0.8 * pressure)
                continue
            if pressure <= 0.0:
                continue
            if schema.action_type == "move":
                level_count = int(self.level_family_counts.get(schema.family, 0))
                level_mean = float(self.level_family_progress_sums.get(schema.family, 0.0)) / float(max(level_count, 1))
                level_no_objective = (
                    float(self.level_family_no_objective_counts.get(schema.family, 0)) / float(max(level_count, 1))
                )
                stale_level = level_count >= 12 and level_mean <= 0.05 and level_no_objective >= 0.85
                biases[schema.family] -= pressure * (3.0 + (3.0 if stale_level else 0.0))
        return dict(biases)

    def _objective_stall_diagnostic_scores(self, state: StructuredState) -> dict[ActionName, float]:
        pressure = self._objective_stall_pressure()
        if pressure <= 0.0:
            return {}
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        scores: dict[ActionName, float] = {}
        for action in state.affordances:
            schema = build_action_schema(action, context)
            if schema.action_type not in {"click", "select", "interact"}:
                continue
            count = int(self.global_action_counts.get(action, 0))
            level_count = int(self.level_action_counts.get(action, 0))
            no_effect_count = int(self.global_action_no_effect_counts.get(action, 0))
            no_effect_rate = float(no_effect_count) / float(max(count, 1))
            level_no_objective = (
                float(self.level_action_no_objective_counts.get(action, 0)) / float(max(level_count, 1))
            )
            if level_count >= 6 and level_no_objective >= 0.9 and no_effect_rate >= 0.9:
                continue
            novelty = 1.0 / math.sqrt(float(level_count) + 1.0)
            uncertainty = 1.0 if level_count < 2 else max(0.0, 1.0 - level_no_objective)
            scores[action] = 2.4 * pressure * max(novelty, 0.25 * uncertainty)
        return scores

    def _combined_patch(
        self,
        state,
        action: ActionName,
    ) -> LocalModelPatch:
        combined = LocalModelPatch()
        action_patch = self.local_action_patches.get(action)
        if action_patch is not None:
            combined.value_shift += action_patch.value_shift
            combined.policy_shift += action_patch.policy_shift
            combined.usefulness_shift += action_patch.usefulness_shift
            combined.uncertainty_shift += action_patch.uncertainty_shift
        context_keys = self._action_context_keys(state, action)
        if context_keys:
            for key in context_keys:
                patch = self.local_context_patches.get(key)
                if patch is None:
                    continue
                combined.value_shift += 0.5 * patch.value_shift
                combined.policy_shift += 0.5 * patch.policy_shift
                combined.usefulness_shift += 0.5 * patch.usefulness_shift
                combined.uncertainty_shift += 0.5 * patch.uncertainty_shift
        return combined

    def _apply_local_model_patches(self, state, runtime_thought: RuntimeThought) -> RuntimeThought:
        if not runtime_thought.actions:
            return runtime_thought
        patched_actions: list[ActionThought] = []
        patch_claims: list[StructuredClaim] = list(runtime_thought.claims)
        for action_thought in runtime_thought.actions:
            patch = self._combined_patch(state, action_thought.action)
            patched_actions.append(
                ActionThought(
                    action=action_thought.action,
                    value=action_thought.value + patch.value_shift + (0.2 * patch.usefulness_shift),
                    uncertainty=max(action_thought.uncertainty + patch.uncertainty_shift, 0.0),
                    policy=action_thought.policy + patch.policy_shift,
                    policy_weight=action_thought.policy_weight,
                    predicted_reward=action_thought.predicted_reward,
                    predicted_return=action_thought.predicted_return,
                    causal_value=action_thought.causal_value,
                    diagnostic_value=action_thought.diagnostic_value,
                    usefulness=action_thought.usefulness + patch.usefulness_shift,
                    selector_followup=action_thought.selector_followup,
                    next_latent=action_thought.next_latent,
                    next_hidden=action_thought.next_hidden,
                    next_state_proxy=action_thought.next_state_proxy,
                )
            )
            patch_strength = abs(patch.value_shift) + abs(patch.policy_shift) + abs(patch.usefulness_shift)
            if patch_strength >= 0.4:
                patch_claims.append(
                    StructuredClaim(
                        claim_type="local_patch",
                        subject=action_thought.action,
                        relation="edited",
                        object="world_model",
                        confidence=min(patch_strength / 2.0, 1.0),
                        evidence=patch_strength,
                        salience=patch_strength,
                    )
                )
        return RuntimeThought(
            belief_tokens=runtime_thought.belief_tokens,
            question_tokens=runtime_thought.question_tokens,
            plan_tokens=runtime_thought.plan_tokens,
            actions=tuple(patched_actions),
            claims=tuple(patch_claims),
            world_model_calls=runtime_thought.world_model_calls,
        )

    def _write_runtime_proofs(
        self,
        proofs: tuple[HypothesisProof, ...],
    ) -> None:
        if (
            not proofs
            or not self.config.use_memory
            or self.episodic_memory is None
            or self.last_latent is None
        ):
            return
        context_tokens = _claim_context_tokens(self.latest_claims)
        action_history = tuple(self.recent_actions)
        for proof in proofs:
            salience = max(proof.confidence, abs(proof.evidence), 0.25)
            payload = {
                "context_id": self.last_state.task_id if self.last_state is not None else "",
                "recommended_action": proof.action if proof.proof_type == "support" and not proof.exception else None,
                "avoid_action": proof.action if proof.proof_type == "contradiction" or proof.exception else None,
                "action_confidence": _bounded_memory_confidence(salience),
                "proof": {
                    "proof_type": proof.proof_type,
                    "hypothesis_type": proof.hypothesis_type,
                    "subject": proof.subject,
                    "relation": proof.relation,
                    "object": proof.object,
                    "predicted": proof.predicted,
                    "observed": proof.observed,
                    "exception": proof.exception,
                },
            }
            self.episodic_memory.write(
                key=self.last_latent.squeeze(0).detach().cpu().numpy(),
                belief_tokens=tuple(token for token in self.stable_belief_tokens if token not in GENERIC_LANGUAGE_TOKENS),
                question_tokens=tuple(token for token in self.stable_question_tokens if token not in GENERIC_LANGUAGE_TOKENS),
                plan_tokens=proof.as_tokens(),
                context_tokens=context_tokens,
                action_history=action_history,
                reward=0.0,
                salience=salience,
                payload=payload,
            )

    def _write_theory_events(
        self,
        events: tuple[TheoryEvent, ...],
    ) -> None:
        if (
            not events
            or not self.config.use_memory
            or self.episodic_memory is None
            or self.last_latent is None
        ):
            return
        context_tokens = _claim_context_tokens(self.latest_claims)
        action_history = tuple(self.recent_actions)
        for event in events:
            rule_candidate = event.rule_candidate
            payload = {
                "context_id": self.last_state.task_id if self.last_state is not None else "",
                "recommended_action": event.recommended_action,
                "avoid_action": event.avoid_action,
                "action_confidence": _bounded_memory_confidence(max(event.salience, 0.25)),
                "theory_event": {
                    "event_type": event.event_type,
                    "action": event.action,
                    "detail": event.detail,
                    "theory": {
                        "action_type": event.theory.action_type,
                        "action_family": event.theory.action_family,
                        "target_signature": event.theory.target_signature,
                        "effect_kind": event.theory.effect_kind,
                        "confidence": event.theory.confidence,
                        "support": event.theory.support,
                        "contradiction": event.theory.contradiction,
                    },
                },
            }
            if rule_candidate is not None:
                payload["rule_candidate"] = {
                    "rule_id": rule_candidate.rule_id,
                    "scope": rule_candidate.scope,
                    "preconditions": list(rule_candidate.preconditions),
                    "effects": list(rule_candidate.effects),
                    "diagnostic_actions": list(rule_candidate.diagnostic_actions),
                    "confidence": rule_candidate.confidence,
                    "posterior": rule_candidate.posterior,
                    "support": rule_candidate.support,
                    "contradiction": rule_candidate.contradiction,
                    "tests": rule_candidate.tests,
                }
            self.episodic_memory.write(
                key=self.last_latent.squeeze(0).detach().cpu().numpy(),
                belief_tokens=tuple(token for token in self.stable_belief_tokens if token not in GENERIC_LANGUAGE_TOKENS),
                question_tokens=tuple(token for token in self.stable_question_tokens if token not in GENERIC_LANGUAGE_TOKENS),
                plan_tokens=rule_candidate.as_tokens() if rule_candidate is not None else event.theory.as_tokens(),
                context_tokens=context_tokens,
                action_history=action_history,
                reward=0.0,
                salience=max(event.salience, event.theory.salience, 0.25),
                payload=payload,
            )

    def _update_language_support(self, transition: Transition, *, progress_signal: float) -> None:
        action_family = _action_family(transition.action)
        positive_evidence = progress_signal >= 0.25
        negative_probe = progress_signal <= -0.15
        if positive_evidence:
            self.evidence_steps += 1
        raw_tokens = tuple(
            dict.fromkeys(
                self.pending_belief_tokens + self.pending_question_tokens + self.pending_plan_tokens
            )
        )
        for token in raw_tokens:
            delta = 0.0
            if token in GENERIC_LANGUAGE_TOKENS:
                delta = 0.08 if positive_evidence else (-0.03 if negative_probe else 0.0)
            else:
                if positive_evidence:
                    delta = 0.45
                elif negative_probe and action_family in {"interact", "click"}:
                    delta = -0.55
                elif negative_probe:
                    delta = -0.2
                else:
                    delta = -0.05
            self.language_token_scores[token] = _clamp(
                float(self.language_token_scores.get(token, 0.0)) + delta,
                lower=-2.0,
                upper=2.0,
            )

    @torch.no_grad()
    def act(self, observation: GridObservation) -> ActionName:
        state = self.observe(observation)
        encoded = self.encoder.encode_state(state, device=self.device)
        self.last_latent = encoded.latent
        self.last_hidden_input = self.hidden
        runtime_thought = self.planner.build_runtime_thought(
            state=state,
            latent=encoded.latent,
            graph=self.graph,
            world_model=self.world_model,
            hidden=self.hidden,
            language_model=self.language_model if self.config.use_language else None,
        )
        runtime_thought = self._apply_local_model_patches(state, runtime_thought)
        if self.theory_manager is not None:
            runtime_thought = self.theory_manager.augment_runtime_thought(
                state,
                runtime_thought,
                rule_inducer=self.rule_inducer,
            )
            self.latest_rule_candidates = self.theory_manager.rule_candidates
            self.latest_competitions = self.theory_manager.competitions
        if self.config.use_runtime_controller:
            assert self.runtime_rule_controller is not None
            runtime_thought = self.runtime_rule_controller.augment_runtime_thought(state, runtime_thought)
        runtime_thought = self._stabilize_runtime_thought(runtime_thought)
        self_model_scores = self.self_model.score_actions(state, runtime_thought)
        self.latest_self_model_scores = self_model_scores
        runtime_thought = self.self_model.augment_thought(state, runtime_thought, self_model_scores)
        runtime_thought = self.self_belief.augment_thought(runtime_thought)
        state_claims = self._state_claims(state)
        merged_family_bias = self._merge_bias_maps(
            self.family_bias,
            self._objective_stall_family_biases(state),
            None if self.theory_manager is None else self.theory_manager.family_bias,
        )
        merged_action_bias = self._merge_bias_maps(
            self._merge_bias_maps(
                self.online_action_bias,
                self.self_model.action_biases(self_model_scores),
                self._objective_stall_action_biases(state),
            ),
            None if self.theory_manager is None else self.theory_manager.action_bias,
        )
        merged_diagnostic_scores = self._merge_bias_maps(
            self.self_model.diagnostic_scores(self_model_scores),
            self._objective_stall_diagnostic_scores(state),
            None if self.theory_manager is None else self.theory_manager.diagnostic_action_scores,
        )
        runtime_thought = RuntimeThought(
            belief_tokens=runtime_thought.belief_tokens,
            question_tokens=runtime_thought.question_tokens,
            plan_tokens=runtime_thought.plan_tokens,
            actions=runtime_thought.actions,
            claims=tuple(runtime_thought.claims) + state_claims + self._episode_family_claims(merged_family_bias),
            world_model_calls=runtime_thought.world_model_calls,
        )
        planner_plan = self.planner.choose_action(
            state=state,
            latent=encoded.latent,
            graph=self.graph,
            world_model=self.world_model,
            hidden=self.hidden,
            language_model=self.language_model if self.config.use_language else None,
            episodic_memory=self.episodic_memory if self.config.use_memory else None,
            rule_inducer=self.rule_inducer,
            action_bias=merged_action_bias,
            action_counts=self.global_action_counts,
            action_delta_sums=self.global_action_delta_sums,
            action_reward_sums=self.global_action_reward_sums,
            action_no_effect_counts=self.global_action_no_effect_counts,
            family_counts=self.family_counts,
            family_bins=self.family_bins,
            family_no_effect_counts=self.family_no_effect_counts,
            family_bias=merged_family_bias,
            context_bias=self._merge_bias_maps(
                self.online_context_bias,
                None if self.theory_manager is None else self.theory_manager.context_bias,
            ),
            diagnostic_action_scores=merged_diagnostic_scores,
            stuck_steps=self.stuck_steps,
            last_action=self.last_action,
            thought=runtime_thought,
            objective_stall_pressure=self._objective_stall_pressure(),
            level_action_counts=self.level_action_counts,
            level_action_progress_sums=self.level_action_progress_sums,
            level_action_no_objective_counts=self.level_action_no_objective_counts,
        )
        if self.config.use_runtime_controller:
            assert self.runtime_rule_controller is not None
            controller_plan = self.runtime_rule_controller.propose(state, thought=runtime_thought)
            plan = self._arbitrate_plan(
                planner_plan=planner_plan,
                controller_plan=controller_plan,
                runtime_thought=runtime_thought,
            )
        else:
            plan = planner_plan
        action = plan.action
        exploration_epsilon = max(0.0, min(float(self.config.exploration_epsilon), 1.0))
        if exploration_epsilon > 0.0 and len(state.affordances) > 1 and float(np.random.random()) < exploration_epsilon:
            action = state.affordances[int(np.random.randint(len(state.affordances)))]
            plan = PlanOutput(
                action=action,
                scores={
                    **dict(plan.scores),
                    "exploration_override": 1.0,
                    "exploration_epsilon": exploration_epsilon,
                },
                language=plan.language,
                search_path=(action,),
            )
        self.stable_belief_tokens = plan.language.belief_tokens or runtime_thought.belief_tokens
        self.stable_question_tokens = plan.language.question_tokens or runtime_thought.question_tokens
        self.stable_plan_tokens = plan.language.plan_tokens or runtime_thought.plan_tokens
        self.pending_belief_tokens = self.stable_belief_tokens
        self.pending_question_tokens = self.stable_question_tokens
        self.pending_plan_tokens = self.stable_plan_tokens
        self.last_runtime_thought = runtime_thought
        self.last_plan_scores = dict(plan.scores)
        self.latest_claims = tuple(runtime_thought.claims)
        claim_tokens = _claim_context_tokens(self.latest_claims, limit=2)
        self.latest_language = (
            self.stable_belief_tokens
            + claim_tokens
            + ("|",)
            + self.stable_question_tokens
            + ("|",)
            + self.stable_plan_tokens
        )
        self.last_prediction = self.world_model.step(
            encoded.latent,
            actions=[action],
            state=state,
            hidden=self.hidden,
        )
        self.hidden = self.last_prediction.hidden
        self.last_state = state
        self.last_action = action
        self.recent_actions = (self.recent_actions + [action])[-4:]
        return action

    @staticmethod
    def _controller_confidence(plan: Any) -> float:
        scores = plan.scores
        if "momentum" in scores:
            return 4.0 + float(scores["momentum"])
        if "exploit" in scores:
            return 2.0 + float(scores["exploit"]) + float(scores.get("objective_utility", 0.0))
        if "interaction_probe" in scores:
            return 2.5 + float(scores.get("diagnostic", 0.0)) + float(scores.get("target_uncertainty", 0.0))
        if "selector_probe" in scores:
            return float(scores.get("diagnostic", 0.0)) + float(scores.get("selector_followup", 0.0))
        if "untested_move" in scores:
            return float(scores.get("diagnostic", 0.0)) + float(scores.get("latent_value", 0.0))
        return float(scores.get("diagnostic", 0.0))

    def _arbitrate_plan(
        self,
        planner_plan: Any,
        controller_plan: Any,
        runtime_thought,
    ) -> Any:
        if controller_plan is None:
            return planner_plan
        scores = controller_plan.scores
        if "restore" in scores and controller_plan.action == "7":
            return controller_plan
        if "momentum" in scores and float(scores.get("momentum", 0.0)) > 0.0:
            return controller_plan
        if "exploit" in scores and (
            float(scores.get("objective_utility", 0.0)) >= 1.0 or float(scores.get("exploit", 0.0)) >= 2.0
        ):
            return controller_plan
        if "interaction_probe" in scores and (
            controller_plan.action.startswith("interact")
            or float(scores.get("diagnostic", 0.0)) >= 1.5
        ):
            return controller_plan
        if "untested_move" in scores:
            return controller_plan
        planner_action_thought = runtime_thought.for_action(planner_plan.action)
        planner_confidence = float(planner_plan.scores.get("total", 0.0))
        if planner_action_thought is not None:
            planner_confidence += planner_action_thought.value
            planner_confidence -= 0.25 * planner_action_thought.uncertainty
        controller_confidence = self._controller_confidence(controller_plan)
        if controller_plan.action == planner_plan.action:
            if controller_confidence >= planner_confidence:
                return controller_plan
            return planner_plan
        if controller_confidence >= planner_confidence + 0.35:
            return controller_plan
        return planner_plan

    def on_transition(self, transition: Transition) -> None:
        runtime_proofs: tuple[HypothesisProof, ...] = ()
        if self.config.use_runtime_controller:
            assert self.runtime_rule_controller is not None
            self.runtime_rule_controller.observe_transition(transition)
            runtime_proofs = self.runtime_rule_controller.consume_recent_proofs()
        self.rule_inducer.record(transition)
        theory_events: tuple[TheoryEvent, ...] = ()
        if self.theory_manager is not None:
            self.theory_manager.observe_transition(transition)
            theory_events = self.theory_manager.consume_recent_events()
        if self.last_latent is None or self.last_prediction is None:
            return
        with torch.no_grad():
            next_encoded = self.encoder.encode_state(transition.next_state, device=self.device)
        surprise = torch.norm(self.last_prediction.next_latent_mean - next_encoded.latent, dim=-1).item()
        state_change = float(
            np.linalg.norm(transition.next_state.transition_vector() - transition.state.transition_vector())
        )
        predicted_reward = float(self.last_prediction.reward.item())
        predicted_usefulness = float(self.last_prediction.usefulness.item())
        predicted_uncertainty = float(self.last_prediction.uncertainty.item())
        predicted_delta = self.last_prediction.delta.detach().cpu().reshape(-1).numpy()
        actual_delta = transition.next_state.transition_vector() - transition.state.transition_vector()
        delta_error = float(np.linalg.norm(actual_delta - predicted_delta))
        online_prediction_error = (
            abs(float(transition.reward) - predicted_reward)
            + (0.15 * max(delta_error, 0.0))
        )
        progress_signal = visible_online_usefulness_target(
            transition.action,
            float(transition.reward),
            state_change,
            prediction_error=online_prediction_error,
            predicted_uncertainty=predicted_uncertainty,
        )
        try:
            level_delta = int(transition.info.get("level_delta", 0) or 0)
        except Exception:
            level_delta = 0
        try:
            levels_completed_after = int(transition.info.get("levels_completed_after", 0) or 0)
        except Exception:
            levels_completed_after = 0
        if levels_completed_after <= 0 and level_delta > 0:
            levels_completed_after = self.max_levels_completed_observed + level_delta
        effective_level_delta = max(0, levels_completed_after - self.max_levels_completed_observed)
        if levels_completed_after > self.max_levels_completed_observed:
            self.max_levels_completed_observed = levels_completed_after
        game_state_before = str(transition.info.get("game_state_before", "")).upper()
        game_state_after = str(transition.info.get("game_state_after", "")).upper()
        reset_action = bool(transition.info.get("reset_action", False))
        objective_progress = float(transition.reward) >= 0.5 or effective_level_delta > 0 or game_state_after.endswith("WIN")
        if effective_level_delta != level_delta:
            transition_info = dict(transition.info)
            transition_info["level_delta"] = effective_level_delta
            transition_info["level_boundary"] = bool(reset_action or effective_level_delta > 0)
            transition = Transition(
                state=transition.state,
                action=transition.action,
                reward=transition.reward,
                next_state=transition.next_state,
                terminated=transition.terminated,
                info=transition_info,
            )
        terminal_continuation = bool(
            reset_action
            and not objective_progress
            and ("GAME_OVER" in game_state_before or "SESSION_ENDED" in game_state_before)
        )
        terminal_failure = bool(
            (
                bool(transition.terminated)
                or "GAME_OVER" in game_state_after
                or "SESSION_ENDED" in game_state_after
            )
            and not objective_progress
            and not terminal_continuation
        )
        if terminal_continuation:
            progress_signal = max(float(progress_signal), 0.08)
        if terminal_failure:
            progress_signal = min(float(progress_signal), -0.45)
        context = build_action_schema_context(transition.state.affordances, dict(transition.state.action_roles))
        schema = build_action_schema(transition.action, context)
        if schema.action_type == "reset" and not terminal_continuation and not objective_progress:
            progress_signal = min(float(progress_signal), -0.35)
        if schema.action_type == "undo" and not terminal_continuation and not objective_progress:
            progress_signal = min(float(progress_signal), -0.28)
        if schema.action_type == "wait" and not terminal_continuation and not objective_progress:
            progress_signal = min(float(progress_signal), -0.22)
        stall_pressure = self._objective_stall_pressure()
        if (
            stall_pressure > 0.0
            and not objective_progress
            and not terminal_continuation
            and schema.action_type == "move"
        ):
            repeated = min(float(self.level_action_counts.get(transition.action, 0)) / 16.0, 1.0)
            progress_signal = min(
                float(progress_signal),
                -0.12 - (0.18 * stall_pressure) - (0.10 * stall_pressure * repeated),
            )
        if objective_progress:
            self.objective_stall_steps = 0
        else:
            self.objective_stall_steps += 1
        if effective_level_delta > 0 or game_state_after.endswith("WIN"):
            self.level_stall_steps = 0
        else:
            self.level_stall_steps += 1
        self.self_model.observe(
            transition,
            state_change=state_change,
            progress_signal=progress_signal,
            predicted_reward=predicted_reward,
            predicted_usefulness=predicted_usefulness,
            predicted_uncertainty=predicted_uncertainty,
            delta_error=delta_error,
            surprise=surprise,
        )
        self.self_belief.observe(
            transition,
            progress_signal=progress_signal,
            prediction_error=online_prediction_error,
            predicted_uncertainty=predicted_uncertainty,
            surprise=surprise,
            online_update=self.world_model_optimizer is not None,
        )
        self.level_action_counts[transition.action] += 1
        self.level_action_progress_sums[transition.action] += float(progress_signal)
        self.level_family_counts[schema.family] += 1
        self.level_family_progress_sums[schema.family] += float(progress_signal)
        if not objective_progress:
            self.level_action_no_objective_counts[transition.action] += 1
            self.level_family_no_objective_counts[schema.family] += 1
        _decay_bias_map(self.online_action_bias, factor=0.92)
        _decay_bias_map(self.online_context_bias, factor=0.94)
        _decay_bias_map(self.family_bias, factor=0.94)
        self._update_language_support(transition, progress_signal=progress_signal)
        self._write_runtime_proofs(runtime_proofs)
        self._write_theory_events(theory_events)
        self._apply_theory_event_control_updates(transition, theory_events)
        self.global_action_counts[transition.action] += 1
        self.global_action_delta_sums[transition.action] += state_change
        self.global_action_reward_sums[transition.action] += float(transition.reward)
        if progress_signal <= -0.15:
            self.stuck_steps += 1
            self.global_action_no_effect_counts[transition.action] += 1
        else:
            self.stuck_steps = 0
        effect_family = no_effect_family_key(schema)
        family_signal = 0.9 * progress_signal
        self.family_bias[schema.family] = _clamp(
            float(self.family_bias.get(schema.family, 0.0)) + family_signal,
            lower=-3.0,
            upper=3.0,
        )
        self.family_counts[effect_family] += 1
        if schema.coarse_bin is not None:
            self.family_bins[effect_family].add(schema.coarse_bin)
        if progress_signal <= -0.15:
            self.family_no_effect_counts[effect_family] += 1
        if self.config.use_memory and self.episodic_memory is not None and (
            surprise >= self.config.surprise_threshold or transition.reward > 0.0
        ):
            trusted_action = progress_signal >= 0.45 or transition.reward > 0.0
            context_tokens = _claim_context_tokens(self.latest_claims)
            content_language = tuple(
                token
                for token in (self.stable_belief_tokens + self.stable_question_tokens + self.stable_plan_tokens)
                if token not in GENERIC_LANGUAGE_TOKENS
            )
            memory_supported = bool(context_tokens or content_language or trusted_action or progress_signal >= 0.25)
            payload = {
                "context_id": transition.state.task_id,
                "recommended_action": transition.action if trusted_action else None,
                "avoid_action": transition.action if progress_signal <= -0.2 else None,
                "action_confidence": _bounded_memory_confidence(max(progress_signal, 0.0)) if trusted_action else 0.0,
                "claims": [
                    {
                        "claim_type": claim.claim_type,
                        "subject": claim.subject,
                        "relation": claim.relation,
                        "object": claim.object,
                        "confidence": claim.confidence,
                    }
                    for claim in self.latest_claims
                ],
            }
            if memory_supported:
                belief = self.stable_belief_tokens
                question_tokens = self.stable_question_tokens
                plan_tokens = self.stable_plan_tokens
                self.episodic_memory.write(
                    key=self.last_latent.squeeze(0).detach().cpu().numpy(),
                    belief_tokens=belief,
                    question_tokens=question_tokens,
                    plan_tokens=plan_tokens,
                    context_tokens=context_tokens,
                    action_history=tuple(self.recent_actions),
                    reward=transition.reward,
                    salience=max(float(surprise), abs(transition.reward), state_change),
                    payload=payload,
                )
        if self.config.use_memory and self.episodic_memory is not None:
            self._counterfactual_replay(transition, state_change=state_change)
        self._update_online_action_bias(transition, state_change=state_change, progress_signal=progress_signal)
        self._update_local_model_patches(
            transition,
            state_change=state_change,
            progress_signal=progress_signal,
            terminal_failure=terminal_failure,
            terminal_continuation=terminal_continuation,
        )
        self._adapt_world_model(
            transition,
            state_change=state_change,
            progress_signal=progress_signal,
            terminal_failure=terminal_failure,
            terminal_continuation=terminal_continuation,
        )

    def _apply_theory_event_control_updates(
        self,
        transition: Transition,
        theory_events: tuple[TheoryEvent, ...],
    ) -> None:
        if not theory_events:
            return

        def apply_action_delta(action: ActionName, delta: float) -> None:
            self.online_action_bias[action] = _clamp(
                self.online_action_bias.get(action, 0.0) + float(delta),
                lower=-2.5,
                upper=2.5,
            )

        def apply_family_delta(family: str, delta: float) -> None:
            self.family_bias[family] = _clamp(
                self.family_bias.get(family, 0.0) + float(delta),
                lower=-3.0,
                upper=3.0,
            )

        def apply_context_delta(action_type: str, target_signature: ObjectSignature | None, delta: float) -> None:
            if target_signature is None:
                return
            key = (action_type, target_signature)
            self.online_context_bias[key] = _clamp(
                float(self.online_context_bias.get(key, 0.0)) + float(delta),
                lower=-2.5,
                upper=2.5,
            )

        for event in theory_events:
            theory = event.theory
            magnitude = _theory_event_magnitude(event.salience)
            if event.event_type == "counterfactual":
                recommended_action = event.recommended_action or event.action
                apply_action_delta(recommended_action, 0.7 * magnitude)
                apply_family_delta(theory.action_family, 0.2 * magnitude)
                apply_context_delta(theory.action_type, theory.target_signature, 0.25 * magnitude)
                continue

            is_negative_theory = theory.effect_kind in {"setback", "no_effect"}
            if event.event_type == "contradiction" or (event.event_type == "support" and is_negative_theory):
                avoid_action = event.avoid_action or event.action
                apply_action_delta(avoid_action, -1.0 * magnitude)
                apply_family_delta(theory.action_family, -0.18 * magnitude)
                apply_context_delta(theory.action_type, theory.target_signature, -0.35 * magnitude)
                diagnostic_actions = event.rule_candidate.diagnostic_actions if event.rule_candidate is not None else ()
                for diagnostic_action in diagnostic_actions:
                    if diagnostic_action == avoid_action:
                        continue
                    apply_action_delta(diagnostic_action, 0.22 * magnitude)
                continue

            if event.event_type == "support" and theory.effect_kind == "reward_gain":
                recommended_action = event.recommended_action or event.action
                apply_action_delta(recommended_action, 0.45 * magnitude)
                apply_family_delta(theory.action_family, 0.14 * magnitude)
                apply_context_delta(theory.action_type, theory.target_signature, 0.18 * magnitude)

    def _update_online_action_bias(
        self,
        transition: Transition,
        state_change: float,
        progress_signal: float | None = None,
    ) -> None:
        baseline = self.online_action_bias.get(transition.action, 0.0)
        if progress_signal is None:
            progress_signal = visible_online_usefulness_target(
                transition.action,
                float(transition.reward),
                state_change,
            )
        signal = 0.85 * progress_signal
        context_keys = self._action_context_keys(transition.state, transition.action)
        action_scale = 0.35 if context_keys else 1.0
        updated = baseline + (action_scale * signal)
        self.online_action_bias[transition.action] = _clamp(updated, lower=-2.5, upper=2.5)
        if context_keys:
            context_delta = signal / float(len(context_keys))
            for key in context_keys:
                self.online_context_bias[key] = _clamp(
                    float(self.online_context_bias.get(key, 0.0)) + context_delta,
                    lower=-2.5,
                    upper=2.5,
                )
        if len(self.recent_actions) >= 2 and abs(progress_signal) >= 0.18:
            trace_signal = 0.3 * signal
            for distance, previous_action in enumerate(reversed(self.recent_actions[:-1]), start=1):
                previous_baseline = self.online_action_bias.get(previous_action, 0.0)
                delayed_credit = trace_signal * (0.55 ** float(distance - 1))
                self.online_action_bias[previous_action] = _clamp(
                    previous_baseline + delayed_credit,
                    lower=-2.5,
                    upper=2.5,
                )

    def _update_local_model_patches(
        self,
        transition: Transition,
        *,
        state_change: float,
        progress_signal: float,
        terminal_failure: bool = False,
        terminal_continuation: bool = False,
    ) -> None:
        if self.last_prediction is None:
            return
        predicted_reward = float(self.last_prediction.reward.item())
        predicted_usefulness = float(self.last_prediction.usefulness.item())
        predicted_policy = float(torch.sigmoid(self.last_prediction.policy).item())
        predicted_uncertainty = float(self.last_prediction.uncertainty.item())
        predicted_delta = self.last_prediction.delta.detach().cpu().reshape(-1).numpy()
        actual_delta = transition.next_state.transition_vector() - transition.state.transition_vector()
        delta_error = float(np.linalg.norm(actual_delta - predicted_delta))
        online_prediction_error = (
            abs(float(transition.reward) - predicted_reward)
            + abs(float(progress_signal) - predicted_usefulness)
            + (0.15 * max(delta_error, 0.0))
        )
        policy_supervision = visible_online_policy_supervision(
            transition.action,
            float(transition.reward),
            state_change,
            prediction_error=online_prediction_error,
            predicted_uncertainty=predicted_uncertainty,
        )
        if terminal_failure:
            policy_supervision = PolicySupervision(target=0.0, weight=max(float(policy_supervision.weight), 1.8))
        elif terminal_continuation:
            policy_supervision = PolicySupervision(target=0.35, weight=max(float(policy_supervision.weight), 0.75))
        reward_error = float(transition.reward) - predicted_reward
        usefulness_error = progress_signal - predicted_usefulness
        policy_error = float(policy_supervision.target) - predicted_policy

        action_patch = self.local_action_patches.setdefault(transition.action, LocalModelPatch())
        action_patch.observe(
            reward_error=reward_error,
            usefulness_error=usefulness_error,
            policy_error=policy_error,
            delta_error=delta_error,
            predicted_uncertainty=predicted_uncertainty,
        )

        context_keys = self._action_context_keys(transition.state, transition.action)
        for key in context_keys:
            patch = self.local_context_patches.setdefault(key, LocalModelPatch())
            patch.observe(
                reward_error=0.5 * reward_error,
                usefulness_error=0.5 * usefulness_error,
                policy_error=0.5 * policy_error,
                delta_error=delta_error,
                predicted_uncertainty=predicted_uncertainty,
            )

    def _counterfactual_replay(self, transition: Transition, state_change: float) -> None:
        if self.last_latent is None or self.last_prediction is None:
            return
        if transition.reward > -0.01 and state_change > 0.2:
            return
        alternatives = [action for action in transition.state.affordances if action != transition.action]
        if not alternatives:
            return
        hidden = None if self.last_hidden_input is None else self.last_hidden_input.repeat(len(alternatives), 1)
        with torch.no_grad():
            repeated_latent = self.last_latent.repeat(len(alternatives), 1)
            alternative_prediction = self.world_model.step(
                repeated_latent,
                actions=alternatives,
                state=transition.state,
                hidden=hidden,
            )
        actual_return_value = getattr(self.last_prediction, "return_value", None)
        if actual_return_value is None:
            actual_return_value = torch.zeros_like(self.last_prediction.reward)
        alternative_return_value = getattr(alternative_prediction, "return_value", None)
        if alternative_return_value is None:
            alternative_return_value = torch.zeros_like(alternative_prediction.reward)
        actual_score = float(
            (
                self.last_prediction.reward
                + (0.35 * actual_return_value)
                + (0.5 * self.last_prediction.usefulness)
                + (0.25 * self.last_prediction.policy)
                - (0.1 * self.last_prediction.uncertainty)
            ).item()
        )
        rule_scores = [self.rule_inducer.action_score(transition.state, action) for action in alternatives]
        predicted_scores = (
            alternative_prediction.reward
            + (0.35 * alternative_return_value)
            + (0.5 * alternative_prediction.usefulness)
            + (0.25 * alternative_prediction.policy)
            - (0.1 * alternative_prediction.uncertainty)
            + torch.tensor(rule_scores, dtype=torch.float32, device=self.device)
        )
        best_index = int(torch.argmax(predicted_scores).item())
        best_action = alternatives[best_index]
        best_score = float(predicted_scores[best_index].item())
        score_gap = best_score - actual_score
        if score_gap <= 0.1:
            return
        actual_context_keys = self._action_context_keys(transition.state, transition.action)
        best_context_keys = self._action_context_keys(transition.state, best_action)
        self.online_action_bias[transition.action] = _clamp(
            self.online_action_bias.get(transition.action, 0.0)
            - ((0.35 if actual_context_keys else 0.75) * score_gap),
            lower=-2.5,
            upper=2.5,
        )
        self.online_action_bias[best_action] = _clamp(
            self.online_action_bias.get(best_action, 0.0)
            + ((0.2 if best_context_keys else 0.5) * score_gap),
            lower=-2.5,
            upper=2.5,
        )
        if actual_context_keys:
            actual_delta = (-1.0 * score_gap) / float(len(actual_context_keys))
            for key in actual_context_keys:
                self.online_context_bias[key] = _clamp(
                    float(self.online_context_bias.get(key, 0.0)) + actual_delta,
                    lower=-2.5,
                    upper=2.5,
                )
        if best_context_keys:
            best_delta = (0.65 * score_gap) / float(len(best_context_keys))
            for key in best_context_keys:
                self.online_context_bias[key] = _clamp(
                    float(self.online_context_bias.get(key, 0.0)) + best_delta,
                    lower=-2.5,
                    upper=2.5,
                )
        context = build_action_schema_context(transition.state.affordances, dict(transition.state.action_roles))
        actual_family = build_action_schema(transition.action, context).family
        best_family = build_action_schema(best_action, context).family
        self.family_bias[actual_family] = _clamp(self.family_bias.get(actual_family, 0.0) - (0.35 * score_gap), lower=-3.0, upper=3.0)
        self.family_bias[best_family] = _clamp(self.family_bias.get(best_family, 0.0) + (0.2 * score_gap), lower=-3.0, upper=3.0)
        belief = self.stable_belief_tokens
        question_tokens = self.stable_question_tokens
        plan_tokens = self.stable_plan_tokens
        context_tokens = _claim_context_tokens(self.latest_claims)
        self.episodic_memory.write(
            key=self.last_latent.squeeze(0).detach().cpu().numpy(),
            belief_tokens=belief,
            question_tokens=question_tokens,
            plan_tokens=plan_tokens,
            context_tokens=context_tokens,
            action_history=tuple(self.recent_actions),
            reward=max(transition.reward, 0.0),
            salience=max(score_gap, abs(transition.reward), state_change),
            payload={
                "context_id": transition.state.task_id,
                "recommended_action": None,
                "avoid_action": transition.action,
                "action_confidence": _bounded_memory_confidence(min(max(score_gap, 0.2), 2.0)),
                "counterfactual": True,
                "claims": [
                    {
                        "claim_type": claim.claim_type,
                        "subject": claim.subject,
                        "relation": claim.relation,
                        "object": claim.object,
                        "confidence": claim.confidence,
                    }
                    for claim in self.latest_claims
                ],
            },
        )
        if self.theory_manager is not None:
            predicted_delta_norm = float(torch.norm(alternative_prediction.delta[best_index]).item())
            self.theory_manager.observe_counterfactual(
                state=transition.state,
                action=best_action,
                predicted_reward=float(alternative_prediction.reward[best_index].item()),
                predicted_usefulness=float(alternative_prediction.usefulness[best_index].item()),
                predicted_uncertainty=float(alternative_prediction.uncertainty[best_index].item()),
                predicted_delta_norm=predicted_delta_norm,
                score_gap=score_gap,
            )
            self._write_theory_events(self.theory_manager.consume_recent_events())

    def _action_context_keys(self, state, action: ActionName) -> tuple[tuple[str, ObjectSignature], ...]:
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        schema = build_action_schema(action, context)
        return tuple((schema.action_type, signature) for signature in action_target_signatures(state, action))

    def _adapt_world_model(
        self,
        transition: Transition,
        state_change: float,
        progress_signal: float | None = None,
        terminal_failure: bool = False,
        terminal_continuation: bool = False,
    ) -> None:
        if self.world_model_optimizer is None:
            return
        self.world_model.train()
        with torch.no_grad():
            encoded = self.encoder.encode_state(transition.state, device=self.device)
            next_encoded = self.encoder.encode_state(transition.next_state, device=self.device)
        predicted_reward = 0.0 if self.last_prediction is None else float(self.last_prediction.reward.item())
        predicted_uncertainty = 0.0 if self.last_prediction is None else float(self.last_prediction.uncertainty.item())
        if self.last_prediction is not None:
            predicted_delta = self.last_prediction.delta.detach().cpu().reshape(-1).numpy()
            actual_delta = transition.next_state.transition_vector() - transition.state.transition_vector()
            delta_error = float(np.linalg.norm(actual_delta - predicted_delta))
        else:
            delta_error = 0.0
        online_prediction_error = abs(float(transition.reward) - predicted_reward) + (0.15 * max(delta_error, 0.0))
        if progress_signal is None:
            progress_signal = visible_online_usefulness_target(
                transition.action,
                float(transition.reward),
                state_change,
                prediction_error=online_prediction_error,
                predicted_uncertainty=predicted_uncertainty,
            )
        policy_supervision = visible_online_policy_supervision(
            transition.action,
            float(transition.reward),
            state_change,
            prediction_error=online_prediction_error,
            predicted_uncertainty=predicted_uncertainty,
        )
        if terminal_failure:
            policy_supervision = PolicySupervision(target=0.0, weight=max(float(policy_supervision.weight), 1.8))
        elif terminal_continuation:
            policy_supervision = PolicySupervision(target=0.35, weight=max(float(policy_supervision.weight), 0.75))
        reward_target = torch.tensor([transition.reward], dtype=torch.float32, device=self.device)
        return_target = torch.tensor([transition.reward], dtype=torch.float32, device=self.device)
        delta_target = torch.tensor(
            transition.next_state.transition_vector() - transition.state.transition_vector(),
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        usefulness_target = torch.tensor(
            [progress_signal],
            dtype=torch.float32,
            device=self.device,
        )
        policy_target = torch.tensor(
            [policy_supervision.target],
            dtype=torch.float32,
            device=self.device,
        )
        policy_weight = torch.tensor(
            [policy_supervision.weight],
            dtype=torch.float32,
            device=self.device,
        )
        hidden = None if self.last_hidden_input is None else self.last_hidden_input.detach()
        for _ in range(self.config.online_world_model_update_steps):
            self.world_model_optimizer.zero_grad()
            loss, _metrics = self.world_model.loss(
                latent=encoded.latent.detach(),
                actions=[transition.action],
                state=transition.state,
                hidden=hidden,
                next_latent_target=next_encoded.latent.detach(),
                reward_target=reward_target,
                return_target=return_target,
                delta_target=delta_target,
                usefulness_target=usefulness_target,
            )
            prediction = self.world_model.step(
                encoded.latent.detach(),
                actions=[transition.action],
                state=transition.state,
                hidden=hidden,
            )
            policy_raw = torch.nn.functional.binary_cross_entropy_with_logits(
                prediction.policy,
                policy_target,
                reduction="none",
            )
            policy_loss = (policy_raw * policy_weight).mean()
            contrastive_policy_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
            available_actions = tuple(transition.state.affordances)
            if transition.action in available_actions:
                action_index = available_actions.index(transition.action)
                sample_limit = int(self.config.online_contrastive_action_sample_limit)
                if sample_limit > 0 and len(available_actions) > max(sample_limit, 2):
                    sample_limit = max(sample_limit, 2)
                    sampled_indices = {action_index}
                    slots = max(sample_limit - 1, 1)
                    if slots == 1:
                        sampled_indices.add(0)
                    else:
                        span = max(len(available_actions) - 1, 1)
                        for slot in range(slots):
                            sampled_indices.add(int(round((slot * span) / float(slots - 1))))
                    sampled = sorted(sampled_indices)
                    available_actions = tuple(available_actions[index] for index in sampled)
                    action_index = sampled.index(action_index)
                repeated_hidden = None if hidden is None else hidden.repeat(len(available_actions), 1)
                all_prediction = self.world_model.step(
                    encoded.latent.detach().repeat(len(available_actions), 1),
                    actions=available_actions,
                    state=transition.state,
                    hidden=repeated_hidden,
                )
                if float(policy_supervision.target) > 0.05:
                    target_index = torch.tensor([action_index], dtype=torch.long, device=self.device)
                    ce_loss = torch.nn.functional.cross_entropy(all_prediction.policy.view(1, -1), target_index)
                    if len(available_actions) > 1:
                        reference = all_prediction.policy[action_index]
                        alternatives = torch.cat(
                            [
                                all_prediction.policy[:action_index],
                                all_prediction.policy[action_index + 1 :],
                            ]
                        )
                        schema = build_action_schema(
                            transition.action,
                            build_action_schema_context(available_actions, dict(transition.state.action_roles)),
                        )
                        margin = 1.25 if schema.action_type in {"click", "select"} else 0.65
                        margin_loss = torch.relu(margin - (reference - alternatives)).mean()
                    else:
                        margin_loss = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                    contrastive_policy_loss = (
                        float(policy_supervision.weight)
                        * float(policy_supervision.target)
                        * (ce_loss + (0.6 * margin_loss))
                    )
                elif float(policy_supervision.weight) > 0.0:
                    target = torch.zeros_like(all_prediction.policy)
                    weights = torch.zeros_like(all_prediction.policy)
                    weights[action_index] = float(policy_supervision.weight)
                    raw = torch.nn.functional.binary_cross_entropy_with_logits(
                        all_prediction.policy,
                        target,
                        reduction="none",
                    )
                    contrastive_policy_loss = (raw * weights).mean()
            total_loss = loss + (0.25 * policy_loss) + (0.35 * contrastive_policy_loss)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
            self.world_model_optimizer.step()
        self.world_model.eval()

    def diagnostics(self) -> dict[str, object]:
        return {
            "name": self.name,
            "runtime_controller_enabled": bool(self.config.use_runtime_controller),
            "runtime_controller_active": self.runtime_rule_controller is not None,
            "theory_manager_enabled": bool(self.config.use_theory_manager),
            "theory_manager_active": self.theory_manager is not None,
            "use_language": bool(self.config.use_language),
            "use_memory": bool(self.config.use_memory),
            "gradient_world_model_adaptation": bool(self.gradient_world_model_adaptation),
            "online_action_bias": {str(key): float(value) for key, value in sorted(self.online_action_bias.items())},
            "family_bias": {str(key): float(value) for key, value in sorted(self.family_bias.items())},
            "stuck_steps": int(self.stuck_steps),
            "objective_stall_steps": int(self.objective_stall_steps),
            "level_stall_steps": int(self.level_stall_steps),
            "max_levels_completed_observed": int(self.max_levels_completed_observed),
            "objective_stall_pressure": float(self._objective_stall_pressure()),
            "level_action_counts": {str(key): int(value) for key, value in sorted(self.level_action_counts.items())},
            "level_action_no_objective_counts": {
                str(key): int(value) for key, value in sorted(self.level_action_no_objective_counts.items())
            },
            "last_plan_scores": {str(key): float(value) for key, value in self.last_plan_scores.items()},
            "latest_language": tuple(str(token) for token in self.latest_language),
            "self_model": self.self_model.diagnostics(),
            "self_belief": self.self_belief.diagnostics(),
        }


class RecurrentAblationAgent(LearnedPlanningAgent):
    def __init__(
        self,
        encoder: StructuredStateEncoder,
        world_model: RecurrentWorldModel,
        planner: HybridPlanner,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            name="recurrent_no_language",
            encoder=encoder,
            world_model=world_model,
            planner=planner,
            language_model=None,
            episodic_memory=None,
            config=LearnedAgentConfig(
                use_language=False,
                use_memory=False,
                use_runtime_controller=False,
                use_theory_manager=False,
            ),
            device=device,
        )


class LanguageNoMemoryAgent(LearnedPlanningAgent):
    def __init__(
        self,
        encoder: StructuredStateEncoder,
        world_model: RecurrentWorldModel,
        planner: HybridPlanner,
        language_model: GroundedLanguageModel,
        device: torch.device | None = None,
        exploration_epsilon: float = 0.0,
    ) -> None:
        super().__init__(
            name="recurrent_with_language",
            encoder=encoder,
            world_model=world_model,
            planner=planner,
            language_model=language_model,
            episodic_memory=None,
            config=LearnedAgentConfig(
                use_language=True,
                use_memory=False,
                use_runtime_controller=False,
                use_theory_manager=False,
                online_world_model_lr=1e-3,
                online_world_model_update_steps=2,
                exploration_epsilon=exploration_epsilon,
            ),
            device=device,
        )


class OnlineLanguageMemoryAgent(LearnedPlanningAgent):
    def __init__(
        self,
        encoder: StructuredStateEncoder,
        world_model: RecurrentWorldModel,
        planner: HybridPlanner,
        language_model: GroundedLanguageModel,
        episodic_memory: EpisodicMemory | None = None,
        device: torch.device | None = None,
        exploration_epsilon: float = 0.0,
    ) -> None:
        super().__init__(
            name="recurrent_with_language_memory",
            encoder=encoder,
            world_model=world_model,
            planner=planner,
            language_model=language_model,
            episodic_memory=episodic_memory or EpisodicMemory(),
            config=LearnedAgentConfig(
                use_language=True,
                use_memory=True,
                use_runtime_controller=False,
                use_theory_manager=False,
                online_world_model_lr=1e-3,
                online_world_model_update_steps=2,
                exploration_epsilon=exploration_epsilon,
            ),
            device=device,
        )


class HybridAgent(LearnedPlanningAgent):
    def __init__(
        self,
        encoder: StructuredStateEncoder,
        world_model: RecurrentWorldModel,
        planner: HybridPlanner,
        language_model: GroundedLanguageModel,
        episodic_memory: EpisodicMemory,
        device: torch.device | None = None,
    ) -> None:
        super().__init__(
            name="hybrid",
            encoder=encoder,
            world_model=world_model,
            planner=planner,
            language_model=language_model,
            episodic_memory=episodic_memory,
            config=LearnedAgentConfig(
                use_language=True,
                use_memory=True,
                use_runtime_controller=False,
                use_theory_manager=True,
            ),
            device=device,
        )
