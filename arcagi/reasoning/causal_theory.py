from collections import defaultdict
from dataclasses import dataclass
import math

import numpy as np

from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.types import ActionName, ActionThought, RuntimeThought, StructuredClaim, StructuredState, Transition
from arcagi.planning.rule_induction import EpisodeRuleInducer, ObjectSignature, action_target_signatures

GroupKey = tuple[str, str, ObjectSignature | None]
TheoryKey = tuple[str, str, ObjectSignature | None, str]
_EFFECT_KINDS: tuple[str, ...] = ("reward_gain", "setback", "state_change", "latent_shift", "no_effect")


def _signature_code(signature: ObjectSignature | None) -> str:
    if signature is None:
        return "global"
    color, area, height, width, _tags = signature
    return f"{color}:{area}:{height}x{width}"


def _scope_code(action_type: str, target_signature: ObjectSignature | None) -> str:
    return f"{action_type}:{_signature_code(target_signature)}"


def _confidence_bucket(value: float) -> str:
    if value >= 0.75:
        return "high"
    if value >= 0.5:
        return "mid"
    return "low"


def _effect_token(effect_kind: str) -> str:
    return {
        "reward_gain": "positive",
        "setback": "negative",
        "state_change": "visible",
        "latent_shift": "hidden",
        "no_effect": "none",
    }.get(effect_kind, "unknown")


def _salient_signed_value(*values: float) -> float:
    dominant = 0.0
    for value in values:
        candidate = float(value)
        if abs(candidate) > abs(dominant):
            dominant = candidate
    return dominant


def _predicted_delta_norm(
    state: StructuredState,
    action_thought: ActionThought,
) -> float:
    next_state_proxy = action_thought.next_state_proxy
    if next_state_proxy is None or not hasattr(next_state_proxy, "projected_transition"):
        return 0.0
    current = state.transition_vector()
    projected = np.asarray(next_state_proxy.projected_transition, dtype=np.float32)
    return float(np.linalg.norm(projected - current))


def _predicted_effect_scores(
    action_thought: ActionThought,
    *,
    action_type: str,
    predicted_delta_norm: float,
) -> dict[str, float]:
    positive_reward = max(float(action_thought.predicted_reward), 0.0)
    negative_reward = max(-float(action_thought.predicted_reward), 0.0)
    positive_usefulness = max(float(action_thought.usefulness), 0.0)
    negative_usefulness = max(-float(action_thought.usefulness), 0.0)
    uncertainty = max(float(action_thought.uncertainty), 0.0)
    policy = max(float(action_thought.policy), 0.0)
    selector_followup = max(float(action_thought.selector_followup), 0.0)
    interactive = action_type in {"click", "select", "interact"}

    reward_gain = (
        (0.9 * positive_reward)
        + (0.75 * positive_usefulness)
        + (0.2 * policy)
        + (0.12 * selector_followup)
    )
    setback = (
        (0.9 * negative_reward)
        + (0.75 * negative_usefulness)
        + (0.28 * predicted_delta_norm)
        + (0.16 * uncertainty)
    )
    state_change = (
        (0.95 * predicted_delta_norm)
        + (0.35 * positive_usefulness)
        + (0.1 * selector_followup)
    )
    latent_shift = (
        ((0.8 if interactive else 0.25) * uncertainty)
        + ((0.45 if interactive else 0.15) * predicted_delta_norm)
        + (0.2 * policy)
        + (0.2 * max(positive_usefulness - predicted_delta_norm, 0.0))
    )
    no_effect = (
        max(0.0, 0.55 - predicted_delta_norm)
        + max(0.0, 0.25 - positive_reward)
        + max(0.0, 0.2 - positive_usefulness)
        + (0.05 * negative_reward)
        + (0.05 * negative_usefulness)
        + (0.1 * max(0.25 - policy, 0.0))
        + (0.08 * max(0.2 - selector_followup, 0.0))
    )
    return {
        "reward_gain": max(reward_gain, 1e-3),
        "setback": max(setback, 1e-3),
        "state_change": max(state_change, 1e-3),
        "latent_shift": max(latent_shift, 1e-3),
        "no_effect": max(no_effect, 1e-3),
    }


def _observed_effect_kind(
    action_type: str,
    *,
    reward: float,
    delta_norm: float,
) -> str:
    if reward > 0.05:
        return "reward_gain"
    if reward < -0.04:
        return "setback"
    if action_type in {"click", "select", "interact"} and delta_norm >= 0.25:
        return "latent_shift"
    if delta_norm >= 0.15:
        return "state_change"
    return "no_effect"


def _merge_tokens(base: tuple[str, ...], extra: tuple[str, ...], *, limit: int = 14) -> tuple[str, ...]:
    merged = tuple(dict.fromkeys(base + extra))
    return merged[:limit]


def _append_unique(target: list[TheoryKey], key: TheoryKey) -> None:
    if key not in target:
        target.append(key)


def _group_posteriors(theories: list["CompactTheory"]) -> dict[TheoryKey, float]:
    if not theories:
        return {}
    logits = [theory.posterior_logit for theory in theories]
    max_logit = max(logits)
    weights = [float(np.exp(logit - max_logit)) for logit in logits]
    normalizer = max(sum(weights), 1e-6)
    return {
        theory.key: (weight / normalizer)
        for theory, weight in zip(theories, weights, strict=False)
    }


def _posterior_entropy(posterior: dict[TheoryKey, float]) -> float:
    if len(posterior) <= 1:
        return 0.0
    entropy = 0.0
    for value in posterior.values():
        entropy -= float(value) * math.log(max(float(value), 1e-6))
    return entropy / math.log(float(len(posterior)))


def _normalized_symbol(value: str) -> str:
    return str(value).strip().lower()


def _is_trivial_symbol(value: str) -> bool:
    return _normalized_symbol(value) in {"", "0", "false", "inactive", "none"}


def _state_scope_conditions(state: StructuredState) -> tuple[str, ...]:
    conditions: list[str] = []
    for key, value in sorted(state.flags):
        if not str(key).startswith("belief_") or _is_trivial_symbol(value):
            continue
        conditions.append(f"{key}={value}")
    for key, value in sorted(state.inventory):
        if not str(key).startswith("belief_") or _is_trivial_symbol(value):
            continue
        conditions.append(f"{key}={value}")
    return tuple(conditions[:3])


def _rule_preconditions(
    state: StructuredState,
    *,
    action_type: str,
    action_family: str,
    target_signature: ObjectSignature | None,
) -> tuple[str, ...]:
    preconditions = [
        f"action_type={action_type}",
        f"action_family={action_family}",
        f"scope={_scope_code(action_type, target_signature)}",
    ]
    if target_signature is not None:
        color, area, height, width, tags = target_signature
        preconditions.extend(
            (
                f"target_color={color}",
                f"target_area={area}",
                f"target_shape={height}x{width}",
            )
        )
        for tag in tags[:3]:
            preconditions.append(f"target_tag={tag}")
    preconditions.extend(_state_scope_conditions(state))
    return tuple(preconditions[:8])


def _reward_bucket(value: float) -> str:
    if value > 0.25:
        return "positive"
    if value < -0.05:
        return "negative"
    return "neutral"


def _delta_bucket(value: float) -> str:
    if value >= 0.25:
        return "large"
    if value >= 0.1:
        return "small"
    return "none"


def _rule_effects(theory: "CompactTheory") -> tuple[str, ...]:
    salient_reward = _salient_signed_value(theory.mean_reward, theory.predicted_reward)
    effects = [
        f"effect={theory.effect_kind}",
        f"reward={_reward_bucket(salient_reward)}",
        f"delta={_delta_bucket(max(theory.mean_delta, theory.predicted_delta))}",
    ]
    if theory.predicted_uncertainty >= 0.45 or theory.effect_kind == "latent_shift":
        effects.append("hidden_state=possible")
    if theory.counterfactual_hits > 0:
        effects.append("counterfactual=useful")
    return tuple(effects[:5])


@dataclass(frozen=True)
class RuleCandidate:
    rule_id: str
    scope: str
    action_type: str
    action_family: str
    target_signature: ObjectSignature | None
    preconditions: tuple[str, ...]
    effects: tuple[str, ...]
    diagnostic_actions: tuple[ActionName, ...]
    confidence: float
    posterior: float
    support: float
    contradiction: float
    tests: int
    salience: float

    def as_claim(self) -> StructuredClaim:
        effect = self.effects[0] if self.effects else "effect=unknown"
        return StructuredClaim(
            claim_type="rule",
            subject=self.scope,
            relation="predicts",
            object=effect,
            confidence=self.posterior,
            evidence=self.support - self.contradiction,
            salience=self.salience,
        )

    def as_tokens(self) -> tuple[str, ...]:
        primary_effect = self.effects[0].split("=", 1)[-1] if self.effects else "unknown"
        return (
            "rule",
            self.action_type,
            primary_effect,
            _confidence_bucket(self.posterior),
        )


@dataclass(frozen=True)
class HypothesisCompetition:
    scope: str
    action_type: str
    action_family: str
    target_signature: ObjectSignature | None
    candidates: tuple[RuleCandidate, ...]
    posterior_entropy: float
    competition_margin: float
    diagnostic_actions: tuple[ActionName, ...]


@dataclass
class CompactTheory:
    action_type: str
    action_family: str
    target_signature: ObjectSignature | None
    effect_kind: str
    support: float = 0.0
    contradiction: float = 0.0
    tests: int = 0
    predicted_reward: float = 0.0
    predicted_usefulness: float = 0.0
    predicted_uncertainty: float = 0.0
    predicted_delta: float = 0.0
    predicted_score: float = 0.0
    observed_reward_sum: float = 0.0
    observed_delta_sum: float = 0.0
    counterfactual_hits: int = 0
    last_step_index: int = 0
    last_action: ActionName = ""

    @property
    def key(self) -> TheoryKey:
        return (
            self.action_type,
            self.action_family,
            self.target_signature,
            self.effect_kind,
        )

    @property
    def evidence_balance(self) -> float:
        return self.support - self.contradiction

    @property
    def confidence(self) -> float:
        return float(self.support + 1.0) / float(self.support + self.contradiction + 2.0)

    @property
    def mean_reward(self) -> float:
        if self.tests <= 0:
            return 0.0
        return self.observed_reward_sum / float(self.tests)

    @property
    def mean_delta(self) -> float:
        if self.tests <= 0:
            return 0.0
        return self.observed_delta_sum / float(self.tests)

    @property
    def posterior_logit(self) -> float:
        return (
            (0.85 * self.predicted_score)
            + (1.0 * self.evidence_balance)
            + (0.2 * max(self.mean_reward, 0.0))
            + (0.08 * self.mean_delta)
            + (0.12 * self.counterfactual_hits)
            - (0.18 * self.predicted_uncertainty)
        )

    @property
    def salience(self) -> float:
        return (
            (0.8 * self.confidence)
            + max(self.predicted_score, 0.0)
            + (0.25 * max(self.mean_reward, 0.0))
            + (0.1 * self.mean_delta)
            + (0.06 * self.counterfactual_hits)
        )

    def observe_prediction(
        self,
        *,
        action: ActionName,
        predicted_reward: float,
        predicted_usefulness: float,
        predicted_uncertainty: float,
        predicted_delta: float,
        predicted_score: float,
    ) -> None:
        mix = 0.35 if self.tests > 0 else 1.0
        self.predicted_reward = ((1.0 - mix) * self.predicted_reward) + (mix * predicted_reward)
        self.predicted_usefulness = ((1.0 - mix) * self.predicted_usefulness) + (mix * predicted_usefulness)
        self.predicted_uncertainty = ((1.0 - mix) * self.predicted_uncertainty) + (mix * predicted_uncertainty)
        self.predicted_delta = ((1.0 - mix) * self.predicted_delta) + (mix * predicted_delta)
        self.predicted_score = ((1.0 - mix) * self.predicted_score) + (mix * predicted_score)
        self.last_action = action

    def observe_outcome(
        self,
        *,
        supported: bool,
        reward: float,
        delta_norm: float,
        step_index: int,
        weight: float = 1.0,
    ) -> None:
        self.tests += 1
        if supported:
            self.support += max(weight, 0.25)
        else:
            self.contradiction += max(weight, 0.25)
        self.observed_reward_sum += reward
        self.observed_delta_sum += delta_norm
        self.last_step_index = step_index

    def observe_counterfactual(self) -> None:
        self.counterfactual_hits += 1

    def as_tokens(self) -> tuple[str, ...]:
        return (
            "theory",
            self.action_type,
            self.effect_kind,
            _signature_code(self.target_signature),
            _confidence_bucket(self.confidence),
        )


@dataclass(frozen=True)
class TheoryEvent:
    event_type: str
    action: ActionName
    theory: CompactTheory
    salience: float
    recommended_action: ActionName | None = None
    avoid_action: ActionName | None = None
    detail: str = ""
    rule_candidate: RuleCandidate | None = None


class EpisodeTheoryManager:
    def __init__(self) -> None:
        self.theories: dict[TheoryKey, CompactTheory] = {}
        self.action_bias: dict[ActionName, float] = {}
        self.context_bias: dict[tuple[str, ObjectSignature], float] = {}
        self.family_bias: dict[str, float] = {}
        self.diagnostic_action_scores: dict[ActionName, float] = {}
        self.rule_candidates: tuple[RuleCandidate, ...] = ()
        self.competitions: tuple[HypothesisCompetition, ...] = ()
        self.rule_candidates_by_key: dict[TheoryKey, RuleCandidate] = {}
        self._last_action_theories: dict[ActionName, tuple[TheoryKey, ...]] = {}
        self._recent_events: list[TheoryEvent] = []

    def reset_episode(self) -> None:
        self.theories.clear()
        self.action_bias = {}
        self.context_bias = {}
        self.family_bias = {}
        self.diagnostic_action_scores = {}
        self.rule_candidates = ()
        self.competitions = ()
        self.rule_candidates_by_key = {}
        self._last_action_theories = {}
        self._recent_events.clear()

    def reset_level(self) -> None:
        self.diagnostic_action_scores = {}
        self.rule_candidates = ()
        self.competitions = ()
        self.rule_candidates_by_key = {}
        self._last_action_theories = {}
        self._recent_events.clear()

    def consume_recent_events(self) -> tuple[TheoryEvent, ...]:
        events = tuple(self._recent_events)
        self._recent_events.clear()
        return events

    @staticmethod
    def _fallback_rule_candidate(
        state: StructuredState,
        theory: CompactTheory,
        *,
        posterior: float | None = None,
        diagnostic_actions: tuple[ActionName, ...] = (),
    ) -> RuleCandidate:
        return RuleCandidate(
            rule_id=f"{_scope_code(theory.action_type, theory.target_signature)}->{theory.effect_kind}",
            scope=_scope_code(theory.action_type, theory.target_signature),
            action_type=theory.action_type,
            action_family=theory.action_family,
            target_signature=theory.target_signature,
            preconditions=_rule_preconditions(
                state,
                action_type=theory.action_type,
                action_family=theory.action_family,
                target_signature=theory.target_signature,
            ),
            effects=_rule_effects(theory),
            diagnostic_actions=diagnostic_actions,
            confidence=theory.confidence,
            posterior=theory.confidence if posterior is None else posterior,
            support=theory.support,
            contradiction=theory.contradiction,
            tests=theory.tests,
            salience=theory.salience,
        )

    def augment_runtime_thought(
        self,
        state: StructuredState,
        thought: RuntimeThought,
        *,
        rule_inducer: EpisodeRuleInducer,
    ) -> RuntimeThought:
        if not thought.actions:
            self.action_bias = {}
            self.context_bias = {}
            self.family_bias = {}
            self.diagnostic_action_scores = {}
            self.rule_candidates = ()
            self.competitions = ()
            self.rule_candidates_by_key = {}
            self._last_action_theories = {}
            return thought

        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        group_to_theories: dict[GroupKey, list[TheoryKey]] = defaultdict(list)
        action_to_theories: dict[ActionName, list[TheoryKey]] = defaultdict(list)
        action_meta: dict[ActionName, tuple[str, str, tuple[ObjectSignature | None, ...]]] = {}

        for action_thought in thought.actions:
            schema = build_action_schema(action_thought.action, context)
            target_signatures = action_target_signatures(state, action_thought.action) or (None,)
            action_meta[action_thought.action] = (schema.action_type, schema.family, target_signatures)
            predicted_delta_norm = _predicted_delta_norm(state, action_thought)
            effect_scores = _predicted_effect_scores(
                action_thought,
                action_type=schema.action_type,
                predicted_delta_norm=predicted_delta_norm,
            )
            for target_signature in target_signatures:
                group_key = (schema.action_type, schema.family, target_signature)
                for effect_kind in _EFFECT_KINDS:
                    key = (schema.action_type, schema.family, target_signature, effect_kind)
                    theory = self.theories.setdefault(
                        key,
                        CompactTheory(
                            action_type=schema.action_type,
                            action_family=schema.family,
                            target_signature=target_signature,
                            effect_kind=effect_kind,
                        ),
                    )
                    theory.observe_prediction(
                        action=action_thought.action,
                        predicted_reward=float(action_thought.predicted_reward),
                        predicted_usefulness=float(action_thought.usefulness),
                        predicted_uncertainty=float(action_thought.uncertainty),
                        predicted_delta=predicted_delta_norm,
                        predicted_score=float(effect_scores[effect_kind]),
                    )
                    _append_unique(group_to_theories[group_key], key)
                    _append_unique(action_to_theories[action_thought.action], key)

        action_bias: defaultdict[ActionName, float] = defaultdict(float)
        context_bias: defaultdict[tuple[str, ObjectSignature], float] = defaultdict(float)
        family_bias: defaultdict[str, float] = defaultdict(float)
        diagnostic_scores: defaultdict[ActionName, float] = defaultdict(float)
        claims: list[StructuredClaim] = list(thought.claims)
        conflict_actions: set[ActionName] = set()
        rule_candidates: list[RuleCandidate] = []
        rule_candidates_by_key: dict[TheoryKey, RuleCandidate] = {}
        competitions: list[HypothesisCompetition] = []
        top_belief: RuleCandidate | None = None
        top_belief_score = float("-inf")

        for action, theory_keys in action_to_theories.items():
            ranked = sorted(
                (self.theories[key] for key in theory_keys),
                key=lambda item: item.posterior_logit,
                reverse=True,
            )
            self._last_action_theories[action] = tuple(theory.key for theory in ranked[:4])

        for group_key, theory_keys in group_to_theories.items():
            ranked = sorted(
                (self.theories[key] for key in theory_keys),
                key=lambda item: item.posterior_logit,
                reverse=True,
            )
            if not ranked:
                continue
            posterior = _group_posteriors(ranked)
            entropy = _posterior_entropy(posterior)
            top = ranked[0]
            runner_up = ranked[1] if len(ranked) > 1 else None
            top_posterior = float(posterior.get(top.key, 0.0))
            runner_posterior = 0.0 if runner_up is None else float(posterior.get(runner_up.key, 0.0))
            margin = max(top_posterior - runner_posterior, 0.0)
            diagnostic_value = entropy + max(0.0, 0.22 - margin) + (0.35 * top.predicted_uncertainty)
            target_signature = group_key[2]
            group_actions = tuple(
                action
                for action, (action_type, family, target_signatures) in action_meta.items()
                if action_type == group_key[0] and family == group_key[1] and target_signature in target_signatures
            )
            diagnostic_actions = tuple(
                sorted(
                    group_actions,
                    key=lambda action: (
                        diagnostic_value,
                        thought.uncertainty_for(action),
                        thought.value_for(action),
                    ),
                    reverse=True,
                )
            )

            group_candidates: list[RuleCandidate] = []
            for index, theory in enumerate(ranked[:3]):
                posterior_value = float(posterior.get(theory.key, 0.0))
                if index > 0 and posterior_value < 0.12:
                    continue
                candidate = RuleCandidate(
                    rule_id=f"{_scope_code(theory.action_type, theory.target_signature)}->{theory.effect_kind}",
                    scope=_scope_code(theory.action_type, theory.target_signature),
                    action_type=theory.action_type,
                    action_family=theory.action_family,
                    target_signature=theory.target_signature,
                    preconditions=_rule_preconditions(
                        state,
                        action_type=theory.action_type,
                        action_family=theory.action_family,
                        target_signature=theory.target_signature,
                    ),
                    effects=_rule_effects(theory),
                    diagnostic_actions=diagnostic_actions[:3],
                    confidence=theory.confidence,
                    posterior=posterior_value,
                    support=theory.support,
                    contradiction=theory.contradiction,
                    tests=theory.tests,
                    salience=theory.salience,
                )
                group_candidates.append(candidate)
                rule_candidates.append(candidate)
                rule_candidates_by_key[theory.key] = candidate

            if not group_candidates:
                continue
            competitions.append(
                HypothesisCompetition(
                    scope=_scope_code(group_key[0], target_signature),
                    action_type=group_key[0],
                    action_family=group_key[1],
                    target_signature=target_signature,
                    candidates=tuple(group_candidates),
                    posterior_entropy=entropy,
                    competition_margin=margin,
                    diagnostic_actions=diagnostic_actions[:3],
                )
            )

            claims.append(group_candidates[0].as_claim())
            if len(group_candidates) > 1:
                rival = group_candidates[1]
                claims.append(
                    StructuredClaim(
                        claim_type="competition",
                        subject=group_candidates[0].scope,
                        relation=group_candidates[0].effects[0],
                        object=rival.effects[0],
                        confidence=rival.posterior,
                        evidence=rival.support - rival.contradiction,
                        salience=max(group_candidates[0].salience, rival.salience),
                    )
                )
            if group_candidates[0].posterior > top_belief_score:
                top_belief = group_candidates[0]
                top_belief_score = group_candidates[0].posterior

            for action in diagnostic_actions:
                diagnostic_scores[action] += diagnostic_value

            for action, (action_type, family, target_signatures) in action_meta.items():
                if action_type != group_key[0] or family != group_key[1] or target_signature not in target_signatures:
                    continue
                empirical_effect = self._empirical_effect_kind(
                    rule_inducer,
                    action=action,
                    action_type=action_type,
                    target_signature=target_signature,
                )
                consistency_conflict = empirical_effect is not None and empirical_effect != top.effect_kind
                if target_signature is not None:
                    context_bias[(action_type, target_signature)] += 0.22 * top_posterior
                if consistency_conflict:
                    conflict_score = 0.3 + diagnostic_value + top.predicted_uncertainty
                    action_bias[action] += conflict_score
                    family_bias[family] += 0.16 * conflict_score
                    conflict_actions.add(action)
                    claims.append(
                        StructuredClaim(
                            claim_type="consistency",
                            subject=_scope_code(action_type, target_signature),
                            relation="contradiction",
                            object=str(empirical_effect),
                            confidence=max(top_posterior, 0.25),
                            evidence=top.evidence_balance,
                            salience=conflict_score,
                        )
                    )
                if diagnostic_value > 0.0:
                    action_bias[action] += 0.15 * diagnostic_value
                if top.effect_kind == "no_effect" and top_posterior >= 0.68:
                    action_bias[action] -= 0.25 + (0.18 * top_posterior) + (0.06 * min(top.tests, 4))
                elif top.effect_kind == "setback" and top_posterior >= 0.55:
                    setback_penalty = 0.38 + (0.32 * top_posterior) + (0.08 * min(top.tests, 4))
                    action_bias[action] -= setback_penalty
                    family_bias[family] -= 0.12 * setback_penalty
                elif top.effect_kind == "reward_gain":
                    commit_bonus = 0.3 * top_posterior
                    action_bias[action] += commit_bonus
                    family_bias[family] += 0.1 * commit_bonus
                else:
                    explore_bonus = 0.08 * top_posterior
                    action_bias[action] += explore_bonus
                    family_bias[family] += 0.05 * explore_bonus

        belief_tokens = thought.belief_tokens
        question_tokens = thought.question_tokens
        plan_tokens = thought.plan_tokens
        max_entropy = max((competition.posterior_entropy for competition in competitions), default=0.0)
        if top_belief is not None:
            if top_belief.posterior >= 0.58 and top_belief.support >= top_belief.contradiction + 0.5:
                belief_tokens = _merge_tokens(
                    thought.belief_tokens,
                    ("belief", "rule", "effect", _effect_token(top_belief.effects[0].split("=", 1)[-1]), top_belief.action_type),
                )
            else:
                belief_tokens = _merge_tokens(
                    thought.belief_tokens,
                    ("belief", "uncertain", "rule"),
                )
        if conflict_actions or max_entropy >= 0.55 or (
            top_belief is not None and top_belief.contradiction > top_belief.support
        ):
            question_tokens = _merge_tokens(
                thought.question_tokens,
                ("question", "need", "test", "rule", "because", "contradiction"),
            )
        elif diagnostic_scores:
            question_tokens = _merge_tokens(
                thought.question_tokens,
                ("question", "need", "test", "rule", "probe"),
            )
        if diagnostic_scores:
            best_action = max(diagnostic_scores, key=diagnostic_scores.get)
            schema = build_action_schema(best_action, context)
            plan_extra = ["plan", schema.action_type]
            if schema.direction is not None:
                plan_extra.append(schema.direction)
            plan_extra.extend(("because", "rule"))
            plan_tokens = _merge_tokens(thought.plan_tokens, tuple(plan_extra))

        self.action_bias = dict(action_bias)
        self.context_bias = dict(context_bias)
        self.family_bias = dict(family_bias)
        self.diagnostic_action_scores = dict(diagnostic_scores)
        self.rule_candidates = tuple(sorted(rule_candidates, key=lambda item: item.posterior, reverse=True)[:12])
        self.competitions = tuple(sorted(competitions, key=lambda item: item.posterior_entropy, reverse=True)[:8])
        self.rule_candidates_by_key = rule_candidates_by_key
        claims = sorted(claims, key=lambda item: item.salience, reverse=True)[:12]
        return RuntimeThought(
            belief_tokens=belief_tokens,
            question_tokens=question_tokens,
            plan_tokens=plan_tokens,
            actions=thought.actions,
            claims=tuple(claims),
            world_model_calls=thought.world_model_calls,
        )

    def observe_transition(
        self,
        transition: Transition,
    ) -> None:
        context = build_action_schema_context(transition.state.affordances, dict(transition.state.action_roles))
        schema = build_action_schema(transition.action, context)
        delta_norm = float(np.linalg.norm(transition.next_state.transition_vector() - transition.state.transition_vector()))
        actual_effect = _observed_effect_kind(
            schema.action_type,
            reward=float(transition.reward),
            delta_norm=delta_norm,
        )
        target_signatures = action_target_signatures(transition.state, transition.action) or (None,)
        predicted_theories = set(self._last_action_theories.get(transition.action, ()))
        for target_signature in target_signatures:
            actual_key = (schema.action_type, schema.family, target_signature, actual_effect)
            theory = self.theories.setdefault(
                actual_key,
                CompactTheory(
                    action_type=schema.action_type,
                    action_family=schema.family,
                    target_signature=target_signature,
                    effect_kind=actual_effect,
                ),
            )
            negative_evidence = max(-float(transition.reward), 0.0)
            support_weight = 1.0 + (0.5 if float(transition.reward) > 0.0 else 0.0)
            if actual_effect == "no_effect":
                support_weight += 0.35
            support_weight += 0.25 * negative_evidence
            theory.observe_outcome(
                supported=True,
                reward=float(transition.reward),
                delta_norm=delta_norm,
                step_index=transition.next_state.step_index,
                weight=support_weight,
            )
            self._recent_events.append(
                TheoryEvent(
                    event_type="support",
                    action=transition.action,
                    theory=theory,
                    salience=max(abs(float(transition.reward)), delta_norm, theory.salience),
                    recommended_action=transition.action if actual_effect != "no_effect" else None,
                    detail=actual_effect,
                    rule_candidate=self.rule_candidates_by_key.get(actual_key)
                    or self._fallback_rule_candidate(
                        transition.state,
                        theory,
                        diagnostic_actions=(transition.action,),
                    ),
                )
            )
            for key, rival in list(self.theories.items()):
                if key[:3] != actual_key[:3] or key[3] == actual_effect:
                    continue
                contradiction_weight = 1.0 + (0.45 if actual_effect == "no_effect" else 0.0) + (0.25 * negative_evidence)
                rival.observe_outcome(
                    supported=False,
                    reward=float(transition.reward),
                    delta_norm=delta_norm,
                    step_index=transition.next_state.step_index,
                    weight=contradiction_weight,
                )
                if key in predicted_theories:
                    self._recent_events.append(
                        TheoryEvent(
                            event_type="contradiction",
                            action=transition.action,
                            theory=rival,
                            salience=max(abs(float(transition.reward)), delta_norm, rival.salience),
                            avoid_action=transition.action,
                            detail=actual_effect,
                            rule_candidate=self.rule_candidates_by_key.get(key)
                            or self._fallback_rule_candidate(
                                transition.state,
                                rival,
                                diagnostic_actions=(transition.action,),
                            ),
                        )
                    )

    def observe_counterfactual(
        self,
        *,
        state: StructuredState,
        action: ActionName,
        predicted_reward: float,
        predicted_usefulness: float,
        predicted_uncertainty: float,
        predicted_delta_norm: float,
        score_gap: float,
    ) -> None:
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        schema = build_action_schema(action, context)
        effect_scores = _predicted_effect_scores(
            ActionThought(
                action=action,
                predicted_reward=predicted_reward,
                usefulness=predicted_usefulness,
                uncertainty=predicted_uncertainty,
            ),
            action_type=schema.action_type,
            predicted_delta_norm=predicted_delta_norm,
        )
        effect_kind = max(effect_scores, key=effect_scores.get)
        target_signatures = action_target_signatures(state, action) or (None,)
        for target_signature in target_signatures:
            key = (schema.action_type, schema.family, target_signature, effect_kind)
            theory = self.theories.setdefault(
                key,
                CompactTheory(
                    action_type=schema.action_type,
                    action_family=schema.family,
                    target_signature=target_signature,
                    effect_kind=effect_kind,
                ),
            )
            theory.observe_prediction(
                action=action,
                predicted_reward=predicted_reward,
                predicted_usefulness=predicted_usefulness,
                predicted_uncertainty=predicted_uncertainty,
                predicted_delta=predicted_delta_norm,
                predicted_score=float(effect_scores[effect_kind]),
            )
            theory.observe_counterfactual()
            self._recent_events.append(
                TheoryEvent(
                    event_type="counterfactual",
                    action=action,
                    theory=theory,
                    salience=max(score_gap, 0.25),
                    recommended_action=action,
                    detail=f"gap={score_gap:.2f}",
                    rule_candidate=self.rule_candidates_by_key.get(key)
                    or self._fallback_rule_candidate(
                        state,
                        theory,
                        diagnostic_actions=(action,),
                    ),
                )
            )

    @staticmethod
    def _empirical_effect_kind(
        rule_inducer: EpisodeRuleInducer,
        *,
        action: ActionName,
        action_type: str,
        target_signature: ObjectSignature | None,
    ) -> str | None:
        action_stats = rule_inducer.action_rules.get(action)
        context_stats = None if target_signature is None else rule_inducer.target_context_rules.get((action_type, target_signature))
        visits = 0
        mean_reward = 0.0
        mean_delta = 0.0
        no_effect_rate = 0.0
        if action_stats is not None:
            visits = max(visits, action_stats.visits)
            mean_reward = _salient_signed_value(mean_reward, action_stats.mean_reward)
            mean_delta = max(mean_delta, action_stats.mean_delta)
            no_effect_rate = max(no_effect_rate, action_stats.no_effect_rate)
        if context_stats is not None:
            visits = max(visits, context_stats.visits)
            mean_reward = _salient_signed_value(mean_reward, context_stats.mean_reward)
            mean_delta = max(mean_delta, context_stats.mean_delta)
            no_effect_rate = max(no_effect_rate, context_stats.no_effect_rate)
        if visits <= 0:
            return None
        if mean_reward > 0.05:
            return "reward_gain"
        if mean_reward < -0.04:
            return "setback"
        if action_type in {"click", "select", "interact"} and mean_delta > 0.2:
            return "latent_shift"
        if mean_delta > 0.12:
            return "state_change"
        if no_effect_rate >= 0.6:
            return "no_effect"
        return None
