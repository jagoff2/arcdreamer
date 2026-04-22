from __future__ import annotations

import copy
from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np
import torch
from typing import Any

from arcagi.agents.base import BaseAgent
from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.progress_signals import (
    action_family as _shared_action_family,
    transition_policy_supervision,
    transition_usefulness_target,
)
from arcagi.core.types import ActionName, ActionThought, GridObservation, HypothesisProof, RuntimeThought, StructuredClaim, Transition
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
        self.stuck_steps = 0

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
        self.stuck_steps = 0
        if self.runtime_rule_controller is not None:
            self.runtime_rule_controller.reset_episode()
        if self.theory_manager is not None:
            self.theory_manager.reset_episode()
        self.rule_inducer.clear()

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
        self.stuck_steps = 0
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

    def _update_language_support(self, transition: Transition, *, state_change: float) -> None:
        action_family = _action_family(transition.action)
        progress_signal = transition_usefulness_target(
            transition.action,
            float(transition.reward),
            None,
            state_change,
        )
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
        state_claims = self._state_claims(state)
        merged_family_bias = self._merge_bias_maps(
            self.family_bias,
            None if self.theory_manager is None else self.theory_manager.family_bias,
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
            action_bias=self._merge_bias_maps(
                self.online_action_bias,
                None if self.theory_manager is None else self.theory_manager.action_bias,
            ),
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
            diagnostic_action_scores=None if self.theory_manager is None else self.theory_manager.diagnostic_action_scores,
            stuck_steps=self.stuck_steps,
            last_action=self.last_action,
            thought=runtime_thought,
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
        progress_signal = transition_usefulness_target(
            transition.action,
            float(transition.reward),
            None,
            state_change,
        )
        _decay_bias_map(self.online_action_bias, factor=0.92)
        _decay_bias_map(self.online_context_bias, factor=0.94)
        _decay_bias_map(self.family_bias, factor=0.94)
        self._update_language_support(transition, state_change=state_change)
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
        context = build_action_schema_context(transition.state.affordances, dict(transition.state.action_roles))
        schema = build_action_schema(transition.action, context)
        family_signal = 0.9 * progress_signal
        self.family_bias[schema.family] = _clamp(
            float(self.family_bias.get(schema.family, 0.0)) + family_signal,
            lower=-3.0,
            upper=3.0,
        )
        self.family_counts[schema.family] += 1
        if schema.coarse_bin is not None:
            self.family_bins[schema.family].add(schema.coarse_bin)
        if progress_signal <= -0.15:
            self.family_no_effect_counts[schema.family] += 1
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
        self._update_online_action_bias(transition, state_change=state_change)
        self._update_local_model_patches(
            transition,
            state_change=state_change,
            progress_signal=progress_signal,
        )
        self._adapt_world_model(transition, state_change=state_change)

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

    def _update_online_action_bias(self, transition: Transition, state_change: float) -> None:
        baseline = self.online_action_bias.get(transition.action, 0.0)
        progress_signal = transition_usefulness_target(
            transition.action,
            float(transition.reward),
            None,
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
        policy_supervision = transition_policy_supervision(
            transition.action,
            float(transition.reward),
            None,
            state_change,
        )
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

    def _adapt_world_model(self, transition: Transition, state_change: float) -> None:
        if self.world_model_optimizer is None:
            return
        self.world_model.train()
        with torch.no_grad():
            encoded = self.encoder.encode_state(transition.state, device=self.device)
            next_encoded = self.encoder.encode_state(transition.next_state, device=self.device)
        progress_signal = transition_usefulness_target(
            transition.action,
            float(transition.reward),
            None,
            state_change,
        )
        policy_supervision = transition_policy_supervision(
            transition.action,
            float(transition.reward),
            None,
            state_change,
        )
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
            total_loss = loss + (0.25 * policy_loss)
            total_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.world_model.parameters(), max_norm=1.0)
            self.world_model_optimizer.step()
        self.world_model.eval()


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
                use_runtime_controller=True,
                use_theory_manager=True,
            ),
            device=device,
        )
