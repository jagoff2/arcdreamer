from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math

from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.types import (
    ActionName,
    ActionThought,
    HypothesisProof,
    LanguageTrace,
    ObjectState,
    PlanOutput,
    RuntimeThought,
    StructuredClaim,
    StructuredState,
    Transition,
)
from arcagi.planning.rule_induction import ObjectSignature, object_signature

DEFAULT_MODE = "__default__"
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
        "frontier",
        "active",
        "inactive",
        "visible",
        "hidden",
        "positive",
        "negative",
        "none",
    }
)
OBJECTIVE_FAMILIES: tuple[str, ...] = ("approach", "contact", "interact", "activate", "avoid")


@dataclass
class EvidenceCounter:
    support: int = 0
    contradiction: int = 0

    @property
    def confidence(self) -> float:
        return float(self.support + 1) / float(self.support + self.contradiction + 2)

    @property
    def uncertainty(self) -> float:
        return 1.0 - self.confidence

    @property
    def balance(self) -> int:
        return self.support - self.contradiction


@dataclass
class SignatureStats:
    present: int = 0
    moved: int = 0
    stable: int = 0
    area_sum: float = 0.0

    @property
    def stability(self) -> float:
        return float(self.stable + 1) / float(self.present + 2)

    @property
    def motion_rate(self) -> float:
        return float(self.moved) / float(max(self.present, 1))

    @property
    def mean_area(self) -> float:
        return self.area_sum / float(max(self.present, 1))


@dataclass
class MotionRule:
    evidence: EvidenceCounter
    dy_sum: float = 0.0
    dx_sum: float = 0.0
    magnitude_sum: float = 0.0

    def observe(self, dy: float, dx: float) -> None:
        self.evidence.support += 1
        self.dy_sum += dy
        self.dx_sum += dx
        self.magnitude_sum += math.sqrt((dy * dy) + (dx * dx))

    @property
    def mean_delta(self) -> tuple[float, float]:
        denominator = float(max(self.evidence.support, 1))
        return (self.dy_sum / denominator, self.dx_sum / denominator)

    @property
    def mean_magnitude(self) -> float:
        return self.magnitude_sum / float(max(self.evidence.support, 1))


@dataclass
class ControlHypothesis:
    evidence: EvidenceCounter
    moved_steps: int = 0
    stalled_steps: int = 0
    motion_sum: float = 0.0
    reward_sum: float = 0.0

    def observe(self, *, moved: bool, motion: float, reward: float) -> None:
        if moved:
            self.evidence.support += 1
            self.moved_steps += 1
            self.motion_sum += motion
        else:
            self.evidence.contradiction += 1
            self.stalled_steps += 1
        self.reward_sum += reward
        if reward > 0.0:
            self.evidence.support += 2
        elif reward < 0.0:
            self.evidence.contradiction += 1

    @property
    def mean_motion(self) -> float:
        return self.motion_sum / float(max(self.moved_steps, 1))

    @property
    def utility(self) -> float:
        return (
            (1.5 * self.evidence.confidence)
            + (0.7 * self.mean_motion)
            + (0.2 * self.moved_steps)
            - (0.35 * self.stalled_steps)
            + (1.5 * self.reward_sum)
        )


@dataclass
class ObjectiveHypothesis:
    evidence: EvidenceCounter
    progress_sum: float = 0.0
    positive_progress_steps: int = 0
    contact_count: int = 0
    reward_sum: float = 0.0
    reward_hits: int = 0
    stalled_contacts: int = 0
    interaction_tests: int = 0
    successful_interactions: int = 0
    failed_interactions: int = 0
    goal_activations: int = 0
    state_delta_sum: float = 0.0
    supporting_actions: set[ActionName] = field(default_factory=set)

    def observe(
        self,
        *,
        action: ActionName,
        progress: float,
        reward: float,
        contact: bool,
        direct_interaction: bool = False,
        state_delta: float = 0.0,
        goal_activated: bool = False,
    ) -> None:
        if progress > 0.0:
            self.evidence.support += 1
            self.progress_sum += progress
            self.positive_progress_steps += 1
            self.supporting_actions.add(action)
        if contact:
            self.contact_count += 1
        if reward > 0.05:
            self.evidence.support += 2
            self.reward_sum += reward
            self.reward_hits += 1
        elif reward < -0.02:
            self.evidence.contradiction += 1
        if direct_interaction:
            self.interaction_tests += 1
            self.state_delta_sum += state_delta
            self.supporting_actions.add(action)
            if goal_activated:
                self.goal_activations += 1
            if goal_activated or reward > 0.05 or state_delta >= 0.35:
                self.successful_interactions += 1
                self.evidence.support += 2
            else:
                self.failed_interactions += 1
                self.evidence.contradiction += 2 if reward <= 0.0 and state_delta <= 0.05 else 1
        if contact and reward <= 0.0 and progress <= 0.0:
            self.stalled_contacts += 1
            self.evidence.contradiction += 1

    @property
    def utility(self) -> float:
        return (
            (1.5 * self.evidence.confidence)
            + self.progress_sum
            + (0.35 * self.positive_progress_steps)
            + (0.25 * len(self.supporting_actions))
            + (0.5 * self.contact_count)
            + (2.0 * self.reward_sum)
            + (1.5 * self.reward_hits)
            + (1.0 * self.successful_interactions)
            + (1.0 * self.goal_activations)
            - (1.25 * self.failed_interactions)
            - (0.75 * self.stalled_contacts)
        )


@dataclass
class ObjectiveFamilyHypothesis:
    family: str
    evidence: EvidenceCounter
    progress_sum: float = 0.0
    contact_hits: int = 0
    interaction_hits: int = 0
    reward_sum: float = 0.0
    goal_hits: int = 0
    negative_hits: int = 0
    state_delta_sum: float = 0.0
    supporting_actions: set[ActionName] = field(default_factory=set)
    tests: int = 0

    def observe(
        self,
        *,
        action: ActionName,
        progress: float,
        reward: float,
        contact: bool,
        direct_interaction: bool = False,
        state_delta: float = 0.0,
        goal_activated: bool = False,
    ) -> None:
        self.tests += 1
        self.supporting_actions.add(action)
        positive_reward = reward > 0.05
        negative_reward = reward < -0.02
        if progress > 0.0:
            self.progress_sum += progress
        if contact:
            self.contact_hits += 1
        if direct_interaction:
            self.interaction_hits += 1
            self.state_delta_sum += state_delta
        if positive_reward:
            self.reward_sum += reward
        if goal_activated:
            self.goal_hits += 1
        if negative_reward:
            self.negative_hits += 1

        if self.family == "approach":
            if progress > 0.0:
                self.evidence.support += 1
            elif progress < 0.0:
                self.evidence.contradiction += 1
            if positive_reward or goal_activated:
                self.evidence.support += 2
            if contact and reward <= 0.0 and not goal_activated:
                self.evidence.contradiction += 1
        elif self.family == "contact":
            if contact:
                self.evidence.support += 2
            elif progress <= 0.0:
                self.evidence.contradiction += 1
            if positive_reward or goal_activated:
                self.evidence.support += 1
        elif self.family == "interact":
            if not direct_interaction:
                if progress > 0.0:
                    self.evidence.support += 1
                return
            if goal_activated or positive_reward or state_delta >= 0.35:
                self.evidence.support += 2
            else:
                self.evidence.contradiction += 2 if reward <= 0.0 and state_delta <= 0.05 else 1
        elif self.family == "activate":
            if goal_activated:
                self.evidence.support += 3
            elif direct_interaction and (positive_reward or state_delta >= 0.2):
                self.evidence.support += 1
            elif direct_interaction:
                self.evidence.contradiction += 1
        elif self.family == "avoid":
            if negative_reward or (contact and reward <= 0.0 and not goal_activated):
                self.evidence.support += 2
            elif goal_activated or positive_reward:
                self.evidence.contradiction += 2
            elif progress > 0.0:
                self.evidence.contradiction += 1

    @property
    def utility(self) -> float:
        base = (
            (1.3 * self.evidence.confidence)
            + (0.15 * len(self.supporting_actions))
            + (0.25 * self.progress_sum)
            + (1.75 * self.reward_sum)
            + (0.65 * self.goal_hits)
            + (0.25 * self.contact_hits)
            + (0.2 * self.interaction_hits)
            - (0.9 * self.negative_hits)
        )
        if self.family == "contact":
            return base + (0.45 * self.contact_hits)
        if self.family == "interact":
            return base + (0.35 * self.state_delta_sum) + (0.55 * self.interaction_hits)
        if self.family == "activate":
            return base + (1.25 * self.goal_hits) + (0.15 * self.state_delta_sum)
        if self.family == "avoid":
            return (1.2 * self.evidence.confidence) + (0.8 * self.negative_hits) - self.progress_sum
        return base


@dataclass
class ModeHypothesis:
    evidence: EvidenceCounter
    entries: int = 0
    move_effect_steps: int = 0
    stalled_steps: int = 0
    reward_sum: float = 0.0
    mover_support: dict[ObjectSignature, int] = field(default_factory=dict)
    action_support: dict[ActionName, int] = field(default_factory=dict)
    target_support: dict[ObjectSignature, int] = field(default_factory=dict)
    hidden_only_steps: int = 0

    def observe_move(self, movers: set[ObjectSignature], reward: float) -> None:
        if movers:
            self.evidence.support += 1
            self.move_effect_steps += 1
            for signature in movers:
                self.mover_support[signature] = self.mover_support.get(signature, 0) + 1
        else:
            self.evidence.contradiction += 1
            self.stalled_steps += 1
        self.reward_sum += reward
        if reward > 0.0:
            self.evidence.support += 2
        elif reward < 0.0:
            self.evidence.contradiction += 1

    @property
    def utility(self) -> float:
        diversity_bonus = 0.15 * float(len(self.mover_support))
        return (
            (1.25 * self.evidence.confidence)
            + (0.35 * self.move_effect_steps)
            - (0.6 * self.stalled_steps)
            + (1.5 * self.reward_sum)
            + (0.1 * self.hidden_only_steps)
            + diversity_bonus
        )


@dataclass
class RepairHypothesis:
    evidence: EvidenceCounter
    touches: int = 0
    last_reason: str = ""

    def observe(self, *, supported: bool, reason: str) -> None:
        self.touches += 1
        self.last_reason = reason
        if supported:
            self.evidence.support += 1
        else:
            self.evidence.contradiction += 1


@dataclass
class OptionHypothesis:
    option_type: str
    evidence: EvidenceCounter
    action_sequence: tuple[ActionName, ...]
    mode_key: str
    mover_signature: ObjectSignature | None = None
    target_signature: ObjectSignature | None = None
    objective_family: str = ""
    progress_sum: float = 0.0
    reward_sum: float = 0.0
    uses: int = 0
    successes: int = 0
    failures: int = 0

    def observe(self, *, progress: float, reward: float, supported: bool, contradiction: bool = False) -> None:
        self.uses += 1
        self.progress_sum += progress
        self.reward_sum += reward
        if supported:
            self.evidence.support += 1
            self.successes += 1
        if contradiction:
            self.evidence.contradiction += 1
            self.failures += 1

    @property
    def utility(self) -> float:
        return (
            (1.35 * self.evidence.confidence)
            + (0.5 * self.successes)
            - (0.75 * self.failures)
            + self.progress_sum
            + (1.5 * self.reward_sum)
            + (0.1 * len(self.action_sequence))
        )


class RuntimeRuleController:
    def __init__(self) -> None:
        self.signature_stats: dict[ObjectSignature, SignatureStats] = defaultdict(SignatureStats)
        self.motion_rules: dict[tuple[str, ActionName, ObjectSignature], MotionRule] = {}
        self.control_hypotheses: dict[tuple[str, ObjectSignature], ControlHypothesis] = {}
        self.objective_hypotheses: dict[tuple[str, ObjectSignature, ObjectSignature], ObjectiveHypothesis] = {}
        self.objective_family_hypotheses: dict[
            tuple[str, ObjectSignature, ObjectSignature, str], ObjectiveFamilyHypothesis
        ] = {}
        self.mode_hypotheses: dict[str, ModeHypothesis] = {}
        self.repair_hypotheses: dict[tuple[str, ObjectSignature], RepairHypothesis] = {}
        self.option_hypotheses: dict[
            tuple[str, str, ObjectSignature | None, ObjectSignature | None, tuple[ActionName, ...]], OptionHypothesis
        ] = {}
        self.action_visits: dict[tuple[str, ActionName], int] = defaultdict(int)
        self.selector_visits: dict[ActionName, int] = defaultdict(int)
        self.proof_records: list[HypothesisProof] = []
        self._unread_proofs = 0
        self.current_mode_key = DEFAULT_MODE
        self.pending_undo = False
        self.undo_attempts = 0
        self.exploit_started = False
        self.active_goal_anchor: tuple[float, float] | None = None
        self.active_exploit_action: ActionName | None = None
        self.active_pair_key: tuple[str, ObjectSignature, ObjectSignature] | None = None
        self.active_objective_family: str | None = None
        self.active_probe_anchor: tuple[float, float] | None = None
        self.active_probe_pair_key: tuple[str, ObjectSignature, ObjectSignature] | None = None
        self.active_option_key: tuple[str, str, ObjectSignature | None, ObjectSignature | None, tuple[ActionName, ...]] | None = None
        self.active_option_index = 0
        self.reference_state: StructuredState | None = None
        self.reference_fingerprint: str | None = None
        self._pending_action_mode_key = DEFAULT_MODE
        self._recent_selector_action: ActionName | None = None

    def reset_episode(self) -> None:
        self.signature_stats.clear()
        self.motion_rules.clear()
        self.control_hypotheses.clear()
        self.objective_hypotheses.clear()
        self.objective_family_hypotheses.clear()
        self.mode_hypotheses.clear()
        self.repair_hypotheses.clear()
        self.option_hypotheses.clear()
        self.action_visits.clear()
        self.selector_visits.clear()
        self.proof_records.clear()
        self._unread_proofs = 0
        self.current_mode_key = DEFAULT_MODE
        self.pending_undo = False
        self.undo_attempts = 0
        self.exploit_started = False
        self.active_goal_anchor = None
        self.active_exploit_action = None
        self.active_pair_key = None
        self.active_objective_family = None
        self.active_probe_anchor = None
        self.active_probe_pair_key = None
        self.active_option_key = None
        self.active_option_index = 0
        self.reference_state = None
        self.reference_fingerprint = None
        self._pending_action_mode_key = DEFAULT_MODE
        self._recent_selector_action = None

    def reset_all(self) -> None:
        self.reset_episode()

    def consume_recent_proofs(self, limit: int = 8) -> tuple[HypothesisProof, ...]:
        if self._unread_proofs <= 0 or not self.proof_records:
            return ()
        unread = self.proof_records[-self._unread_proofs :]
        self._unread_proofs = 0
        return tuple(unread[-max(int(limit), 1) :])

    def _record_proof(
        self,
        *,
        proof_type: str,
        hypothesis_type: str,
        action: ActionName,
        subject: str,
        relation: str,
        object: str,
        confidence: float,
        evidence: float,
        predicted: str = "",
        observed: str = "",
        step_index: int = 0,
        exception: bool = False,
    ) -> None:
        self.proof_records.append(
            HypothesisProof(
                proof_type=proof_type,
                hypothesis_type=hypothesis_type,
                action=action,
                subject=subject,
                relation=relation,
                object=object,
                confidence=confidence,
                evidence=evidence,
                predicted=predicted,
                observed=observed,
                step_index=step_index,
                exception=exception,
            )
        )
        if len(self.proof_records) > 128:
            overflow = len(self.proof_records) - 128
            self.proof_records = self.proof_records[overflow:]
            self._unread_proofs = max(self._unread_proofs - overflow, 0)
        self._unread_proofs = min(self._unread_proofs + 1, len(self.proof_records))

    def augment_runtime_thought(
        self,
        state: StructuredState,
        thought: RuntimeThought,
    ) -> RuntimeThought:
        if self.reference_state is None:
            self.reference_state = state
            self.reference_fingerprint = state.fingerprint()

        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        move_actions = [action for action in state.affordances if build_action_schema(action, context).action_type == "move"]
        interact_actions = [
            action
            for action in state.affordances
            if build_action_schema(action, context).action_type == "interact"
        ]
        selector_actions = [
            action
            for action in state.affordances
            if build_action_schema(action, context).action_type in {"click", "select"}
        ]

        if self._should_abandon_mode():
            self.current_mode_key = DEFAULT_MODE
            self.active_goal_anchor = None
            self.active_exploit_action = None
            self.active_pair_key = None
            self.active_probe_anchor = None
            self.active_probe_pair_key = None

        control_posteriors = self._control_posteriors(state, move_actions)
        mover_scores = self._mover_scores(state, move_actions)
        action_bonus: dict[ActionName, float] = defaultdict(float)
        claims: list[StructuredClaim] = []
        top_mover_signature: ObjectSignature | None = None

        if mover_scores:
            top_mover_signature, top_mover_score = max(mover_scores.items(), key=lambda item: item[1])
            control = self.control_hypotheses.get((self.current_mode_key, top_mover_signature))
            claims.append(
                StructuredClaim(
                    claim_type="control",
                    subject=_signature_code(top_mover_signature),
                    relation="controllable",
                    object=_mode_code(self.current_mode_key),
                    confidence=control_posteriors.get(top_mover_signature, 0.0),
                    evidence=0.0 if control is None else float(control.evidence.balance),
                    salience=top_mover_score * max(control_posteriors.get(top_mover_signature, 0.0), 1e-6),
                )
            )

        objective_posteriors = self._objective_posteriors(state, mover_scores)
        objective_candidates = sorted(
            self._objective_candidates(state, mover_scores),
            key=lambda item: objective_posteriors.get((item[0], object_signature(item[1])), 0.0),
            reverse=True,
        )
        top_candidates = objective_candidates[:3]
        for mover_signature, target_object, objective in top_candidates:
            posterior = objective_posteriors.get((mover_signature, object_signature(target_object)), 0.0)
            claims.append(
                StructuredClaim(
                    claim_type="objective",
                    subject=_signature_code(mover_signature),
                    relation="toward",
                    object=_signature_code(object_signature(target_object)),
                    confidence=posterior,
                    evidence=float(objective.evidence.balance),
                    salience=max(posterior * max(objective.utility, 1.0), posterior),
                )
            )
            mover_objects = _group_by_signature(state.objects).get(mover_signature, [])
            if not mover_objects:
                continue
            for action in move_actions:
                progress = self._predicted_progress_for_action(state, action, mover_signature, target_object)
                action_bonus[action] += posterior * (
                    progress + (0.15 * objective.utility) + (0.1 * mover_scores.get(mover_signature, 0.0))
                )

        objective_family_posteriors = self._objective_family_posteriors(state, mover_scores)
        objective_family_candidates = sorted(
            self._objective_family_candidates(state, mover_scores),
            key=lambda item: objective_family_posteriors.get((item[0], object_signature(item[1]), item[2]), 0.0),
            reverse=True,
        )
        for mover_signature, target_object, family, _objective, family_hypothesis in objective_family_candidates[:3]:
            posterior = objective_family_posteriors.get((mover_signature, object_signature(target_object), family), 0.0)
            claims.append(
                StructuredClaim(
                    claim_type="objective_family",
                    subject=_signature_code(mover_signature),
                    relation=family,
                    object=_signature_code(object_signature(target_object)),
                    confidence=posterior,
                    evidence=float(family_hypothesis.evidence.balance),
                    salience=max(posterior * max(family_hypothesis.utility, 1.0), posterior),
                )
            )

        if not top_candidates:
            untested_moves = [action for action in move_actions if self.action_visits[(self.current_mode_key, action)] == 0]
            for action in untested_moves:
                action_bonus[action] += 0.6 + (0.35 * self._thought_uncertainty(thought, action))

        for action in interact_actions:
            target_object = self._interaction_target_object(state, action, context)
            if target_object is None:
                action_bonus[action] -= 1.5
                continue
            target_signature = object_signature(target_object)
            for mover_signature in mover_scores:
                objective = self.objective_hypotheses.get((self.current_mode_key, mover_signature, target_signature))
                if objective is None:
                    continue
                if objective.failed_interactions > 0 and objective.successful_interactions == 0:
                    action_bonus[action] -= 1.25

        if selector_actions:
            grouped: dict[str, list[ActionName]] = defaultdict(list)
            for action in selector_actions:
                grouped[_mode_key(action, build_action_schema(action, context))].append(action)
            mode_posteriors = self._mode_posteriors(selector_actions, context, thought)
            ranked_modes: list[tuple[float, str, float]] = []
            for mode_key, actions in grouped.items():
                hypothesis = self.mode_hypotheses.get(mode_key)
                utility = 0.0 if hypothesis is None else hypothesis.utility
                confidence = mode_posteriors.get(mode_key, 0.0)
                best_action = self._best_selector_action_for_mode(mode_key, actions, context)
                followup = max(self._thought_selector_followup(thought, action) for action in actions)
                uncertainty = max(self._thought_uncertainty(thought, action) for action in actions)
                exploration = 1.0 / math.sqrt((0 if hypothesis is None else hypothesis.entries) + 1.0)
                mode_score = utility + max(followup, 0.0) + (0.35 * uncertainty) + exploration
                ranked_modes.append((mode_score, mode_key, followup))
                claims.append(
                    StructuredClaim(
                        claim_type="mode",
                        subject=_mode_code(mode_key),
                        relation="enables",
                        object="move_gain",
                        confidence=confidence,
                        evidence=0.0 if hypothesis is None else float(hypothesis.evidence.balance),
                        salience=max(mode_score * max(confidence, 1e-6), confidence),
                    )
                )
                if best_action is not None:
                    target_object = self._interaction_target_object(state, best_action, context)
                    if target_object is not None:
                        claims.append(
                            StructuredClaim(
                                claim_type="interface",
                                subject=_action_code(best_action, context),
                                relation="targets",
                                object=_signature_code(object_signature(target_object)),
                                confidence=min(0.5 + (0.1 * confidence), 1.0),
                                evidence=0.0 if hypothesis is None else float(hypothesis.target_support.get(object_signature(target_object), 0)),
                                salience=max(0.45 + confidence + (0.1 * followup), 0.1),
                            )
                        )
                    if top_mover_signature is not None:
                        claims.append(
                            StructuredClaim(
                                claim_type="control_binding",
                                subject=_action_code(best_action, context),
                                relation="controls",
                                object=_signature_code(top_mover_signature),
                                confidence=min(max(confidence, 0.05) * max(control_posteriors.get(top_mover_signature, 0.2), 0.2), 1.0),
                                evidence=0.0 if hypothesis is None else float(hypothesis.evidence.balance),
                                salience=max(confidence + control_posteriors.get(top_mover_signature, 0.0) + (0.1 * followup), 0.05),
                            )
                        )
                for action in actions:
                    action_bonus[action] += confidence * max(self._thought_selector_followup(thought, action), 0.0)
                    action_bonus[action] += 0.15 * self._thought_uncertainty(thought, action)
            ranked_modes.sort(reverse=True)
            if ranked_modes and ranked_modes[0][2] <= 0.25 and sum(self.selector_visits.values()) >= len(grouped):
                for action in selector_actions:
                    action_bonus[action] -= 0.75

        repair_candidates = sorted(
            self.repair_hypotheses.items(),
            key=lambda item: (
                item[1].evidence.confidence,
                item[1].evidence.balance,
                item[1].touches,
            ),
            reverse=True,
        )
        for (repair_type, signature), repair in repair_candidates[:2]:
            claims.append(
                StructuredClaim(
                    claim_type="repair",
                    subject=_signature_code(signature),
                    relation=repair_type,
                    object=repair.last_reason or "representation",
                    confidence=repair.evidence.confidence,
                    evidence=float(repair.evidence.balance),
                    salience=float(repair.evidence.confidence + (0.1 * repair.touches)),
                )
            )

        for proof in self.proof_records[-3:]:
            claims.append(
                StructuredClaim(
                    claim_type="proof",
                    subject=proof.hypothesis_type,
                    relation=proof.proof_type,
                    object=proof.relation,
                    confidence=proof.confidence,
                    evidence=proof.evidence,
                    salience=max(proof.confidence, abs(proof.evidence)),
                )
            )

        for option in sorted(self.option_hypotheses.values(), key=lambda item: item.utility, reverse=True)[:2]:
            if option.evidence.support <= 0:
                continue
            claims.append(
                StructuredClaim(
                    claim_type="option",
                    subject=option.option_type,
                    relation=option.objective_family or option.mode_key,
                    object="then".join(option.action_sequence[:3]),
                    confidence=option.evidence.confidence,
                    evidence=float(option.evidence.balance),
                    salience=max(option.utility, option.evidence.confidence),
                )
            )
        for mover_signature, target_object, family, _objective, family_hypothesis in objective_family_candidates[:2]:
            if family_hypothesis.goal_hits <= 0 and family_hypothesis.reward_sum <= 0.0:
                continue
            relation = {
                "activate": "reward_after_activate",
                "interact": "reward_after_interact",
                "contact": "reward_after_contact",
                "approach": "reward_after_approach",
                "avoid": "reward_after_avoid",
            }.get(family, f"reward_after_{family}")
            claims.append(
                StructuredClaim(
                    claim_type="reward_model",
                    subject=_signature_code(object_signature(target_object)),
                    relation=relation,
                    object=_signature_code(mover_signature),
                    confidence=max(family_hypothesis.evidence.confidence, 0.1),
                    evidence=max(family_hypothesis.reward_sum, float(family_hypothesis.goal_hits)),
                    salience=max(family_hypothesis.utility, family_hypothesis.evidence.confidence),
                )
            )

        claims.sort(key=lambda claim: claim.salience, reverse=True)
        updated_actions: list[ActionThought] = []
        for action_thought in thought.actions:
            updated_actions.append(
                ActionThought(
                    action=action_thought.action,
                    value=action_thought.value + action_bonus.get(action_thought.action, 0.0),
                    uncertainty=action_thought.uncertainty,
                    policy=action_thought.policy,
                    policy_weight=action_thought.policy_weight,
                    predicted_reward=action_thought.predicted_reward,
                    usefulness=action_thought.usefulness,
                    selector_followup=action_thought.selector_followup,
                )
            )

        question_tokens = thought.question_tokens
        if selector_actions and not mover_scores:
            question_tokens = ("question", "need", "test", "focus", "control_binding", "state", "probe")
        elif any(repair.evidence.confidence >= 0.55 for repair in self.repair_hypotheses.values()):
            question_tokens = ("question", "need", "repair", "focus", "representation", "state", "probe")
        elif top_candidates:
            question_tokens = ("question", "need", "test", "focus", "objective", "state", "commit")
        elif selector_actions and max((self._thought_selector_followup(thought, action) for action in selector_actions), default=0.0) > 0.5:
            question_tokens = ("question", "need", "test", "focus", "control_binding", "state", "probe")
        elif move_actions:
            question_tokens = ("question", "need", "test", "focus", "move_effect", "state", "probe")

        belief_tokens = thought.belief_tokens
        if selector_actions and not mover_scores:
            belief_tokens = ("belief", "control_binding", "uncertain", "clickable")
        elif claims:
            belief_tokens = ("belief", claims[0].claim_type, claims[0].relation)

        return RuntimeThought(
            belief_tokens=belief_tokens,
            question_tokens=question_tokens,
            plan_tokens=thought.plan_tokens,
            actions=tuple(updated_actions),
            claims=tuple(claims[:6]),
        )

    def propose(self, state: StructuredState, thought: RuntimeThought | None = None) -> PlanOutput | None:
        if self.reference_state is None:
            self.reference_state = state
            self.reference_fingerprint = state.fingerprint()

        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        move_actions = [action for action in state.affordances if build_action_schema(action, context).action_type == "move"]
        interact_actions = [
            action
            for action in state.affordances
            if build_action_schema(action, context).action_type == "interact"
        ]
        selector_actions = [
            action
            for action in state.affordances
            if build_action_schema(action, context).action_type in {"click", "select"}
        ]
        undo_action = next(
            (action for action in state.affordances if build_action_schema(action, context).action_type == "undo"),
            None,
        )

        if self.pending_undo and undo_action is not None and not self.exploit_started:
            if self.reference_fingerprint is not None and state.fingerprint() != self.reference_fingerprint:
                if self.undo_attempts >= 2:
                    self.pending_undo = False
                    self.undo_attempts = 0
                else:
                    self._pending_action_mode_key = self.current_mode_key
                    return PlanOutput(
                        action=undo_action,
                        scores={"diagnostic": 1.0, "restore": 1.0},
                        language=LanguageTrace(
                            belief_tokens=("belief", "non_reference_state"),
                            question_tokens=("question", "undo_probe"),
                            plan_tokens=("plan", undo_action),
                        ),
                        search_path=(undo_action,),
                    )
            self.pending_undo = False

        if self._should_abandon_mode():
            self.current_mode_key = DEFAULT_MODE
            self.active_goal_anchor = None
            self.active_exploit_action = None
            self.active_pair_key = None
            self.active_objective_family = None
            self._clear_active_option()

        mover_scores = self._mover_scores(state, move_actions)

        option_plan = self._option_plan(state, move_actions, interact_actions, selector_actions, mover_scores, thought, context)
        if option_plan is not None:
            self._pending_action_mode_key = self.current_mode_key
            return option_plan

        first_contact_plan = self._first_contact_program_plan(
            state,
            move_actions,
            interact_actions,
            selector_actions,
            mover_scores,
            thought,
            context,
        )
        if first_contact_plan is not None:
            self._pending_action_mode_key = self.current_mode_key
            return first_contact_plan

        momentum_plan = self._momentum_plan(state, move_actions, mover_scores, thought)
        if momentum_plan is not None:
            self._pending_action_mode_key = self.current_mode_key
            return momentum_plan

        probe_momentum_plan = self._probe_momentum_plan(
            state,
            move_actions,
            interact_actions,
            mover_scores,
            thought,
            context,
        )
        if probe_momentum_plan is not None:
            self._pending_action_mode_key = self.current_mode_key
            return probe_momentum_plan

        interaction_probe_plan = self._interaction_probe_plan(
            state,
            move_actions,
            interact_actions,
            mover_scores,
            thought,
            context,
        )
        if interaction_probe_plan is not None:
            self._pending_action_mode_key = self.current_mode_key
            return interaction_probe_plan

        if self._ready_to_exploit(state, move_actions, mover_scores):
            exploit_plan = self._exploit_plan(state, move_actions, mover_scores, thought)
            if exploit_plan is not None:
                self.exploit_started = True
                self.pending_undo = False
                self._pending_action_mode_key = self.current_mode_key
                return exploit_plan

        diagnostic_plan = self._diagnostic_move_plan(state, move_actions, thought, context)
        if diagnostic_plan is not None:
            self._pending_action_mode_key = self.current_mode_key
            return diagnostic_plan

        disambiguation_plan = self._disambiguation_move_plan(state, move_actions, mover_scores, thought)
        if disambiguation_plan is not None:
            self._pending_action_mode_key = self.current_mode_key
            return disambiguation_plan

        selector_plan = self._selector_probe_plan(state, selector_actions, context, mover_scores, thought)
        if selector_plan is not None:
            self._pending_action_mode_key = self.current_mode_key
            return selector_plan

        return None

    def observe_transition(self, transition: Transition) -> None:
        action_mode_key = self._pending_action_mode_key
        context = build_action_schema_context(transition.state.affordances, dict(transition.state.action_roles))
        schema = build_action_schema(transition.action, context)
        self.action_visits[(action_mode_key, transition.action)] += 1
        transition_delta = _state_delta(transition.state, transition.next_state)
        goal_activated = _goal_activated(transition.state, transition.next_state)

        if schema.action_type == "undo" and self.reference_fingerprint is not None:
            self.undo_attempts += 1
            if transition.next_state.fingerprint() == self.reference_fingerprint:
                self.pending_undo = False
                self.undo_attempts = 0
                self.current_mode_key = DEFAULT_MODE
                self.exploit_started = False
                self.active_goal_anchor = None
                self.active_exploit_action = None
                self.active_pair_key = None
                self.active_objective_family = None
                self.active_probe_anchor = None
                self.active_probe_pair_key = None
                self._clear_active_option()
            elif transition.next_state.fingerprint() == transition.state.fingerprint():
                self.pending_undo = False
                self.undo_attempts = 0

        if schema.action_type in {"click", "select"}:
            self.selector_visits[transition.action] += 1
            self.current_mode_key = _mode_key(transition.action, schema)
            self._recent_selector_action = transition.action
            mode = self.mode_hypotheses.setdefault(self.current_mode_key, ModeHypothesis(evidence=EvidenceCounter()))
            mode.entries += 1
            mode.action_support[transition.action] = mode.action_support.get(transition.action, 0) + 1
            clicked_target = self._interaction_target_object(transition.state, transition.action, context)
            if clicked_target is not None:
                clicked_signature = object_signature(clicked_target)
                mode.target_support[clicked_signature] = mode.target_support.get(clicked_signature, 0) + 1
            if transition_delta <= 0.1:
                mode.evidence.support += 1
                mode.hidden_only_steps += 1
                self._record_proof(
                    proof_type="support",
                    hypothesis_type="mode",
                    action=transition.action,
                    subject=_mode_code(self.current_mode_key),
                    relation="changes_hidden_state",
                    object="latent_only",
                    confidence=mode.evidence.confidence,
                    evidence=float(mode.evidence.balance),
                    predicted="selector changes control state",
                    observed="visible change small",
                    step_index=transition.next_state.step_index,
                )

        signature_groups: dict[ObjectSignature, list[tuple[ObjectState, ObjectState]]] = defaultdict(list)
        for signature, before_obj, after_obj in _match_objects(transition.state, transition.next_state):
            signature_groups[signature].append((before_obj, after_obj))
        before_groups = _group_by_signature(transition.state.objects)
        after_groups = _group_by_signature(transition.next_state.objects)
        self._observe_representation_repairs(
            transition=transition,
            before_groups=before_groups,
            after_groups=after_groups,
            signature_groups=signature_groups,
        )

        moving_signatures: set[ObjectSignature] = set()
        motion_summary: dict[ObjectSignature, tuple[float, float, float]] = {}
        for signature, pairs in signature_groups.items():
            stats = self.signature_stats[signature]
            for before_obj, after_obj in pairs:
                stats.present += 1
                stats.area_sum += float(before_obj.area)
            dy_values = [after_obj.centroid[0] - before_obj.centroid[0] for before_obj, after_obj in pairs]
            dx_values = [after_obj.centroid[1] - before_obj.centroid[1] for before_obj, after_obj in pairs]
            avg_dy = sum(dy_values) / float(len(dy_values))
            avg_dx = sum(dx_values) / float(len(dx_values))
            motion = math.sqrt((avg_dy * avg_dy) + (avg_dx * avg_dx))
            motion_summary[signature] = (avg_dy, avg_dx, motion)
            if motion > 1e-6:
                moving_signatures.add(signature)
                stats.moved += len(pairs)
                if schema.action_type == "move":
                    rule_key = (action_mode_key, transition.action, signature)
                    rule = self.motion_rules.setdefault(rule_key, MotionRule(evidence=EvidenceCounter()))
                    mean_dy, mean_dx = rule.mean_delta
                    if rule.evidence.support > 0 and (abs(mean_dy - avg_dy) > 0.35 or abs(mean_dx - avg_dx) > 0.35):
                        rule.evidence.contradiction += 1
                        self._record_proof(
                            proof_type="contradiction",
                            hypothesis_type="control",
                            action=transition.action,
                            subject=_signature_code(signature),
                            relation="moves_as_expected",
                            object=_mode_code(action_mode_key),
                            confidence=rule.evidence.confidence,
                            evidence=float(rule.evidence.balance),
                            predicted=f"{mean_dy:.2f},{mean_dx:.2f}",
                            observed=f"{avg_dy:.2f},{avg_dx:.2f}",
                            step_index=transition.next_state.step_index,
                            exception=True,
                        )
                    rule.observe(avg_dy, avg_dx)
                    self._record_proof(
                        proof_type="support",
                        hypothesis_type="control",
                        action=transition.action,
                        subject=_signature_code(signature),
                        relation="moves_as_expected",
                        object=_mode_code(action_mode_key),
                        confidence=rule.evidence.confidence,
                        evidence=float(rule.evidence.balance),
                        predicted=f"{rule.mean_delta[0]:.2f},{rule.mean_delta[1]:.2f}",
                        observed=f"{avg_dy:.2f},{avg_dx:.2f}",
                        step_index=transition.next_state.step_index,
                    )
            else:
                stats.stable += len(pairs)
                rule = self.motion_rules.get((action_mode_key, transition.action, signature))
                if rule is not None:
                    rule.evidence.contradiction += 1
                    self._record_proof(
                        proof_type="contradiction",
                        hypothesis_type="control",
                        action=transition.action,
                        subject=_signature_code(signature),
                        relation="moves_as_expected",
                        object=_mode_code(action_mode_key),
                        confidence=rule.evidence.confidence,
                        evidence=float(rule.evidence.balance),
                        predicted=f"{rule.mean_delta[0]:.2f},{rule.mean_delta[1]:.2f}",
                        observed="0.00,0.00",
                        step_index=transition.next_state.step_index,
                        exception=True,
                    )

        if schema.action_type == "move":
            mode = self.mode_hypotheses.setdefault(action_mode_key, ModeHypothesis(evidence=EvidenceCounter()))
            mode.observe_move(moving_signatures, float(transition.reward))
            self._record_proof(
                proof_type="support" if moving_signatures else "contradiction",
                hypothesis_type="mode",
                action=transition.action,
                subject=_mode_code(action_mode_key),
                relation="enables_motion",
                object="move",
                confidence=mode.evidence.confidence,
                evidence=float(mode.evidence.balance),
                predicted="move changes world",
                observed="motion" if moving_signatures else "stalled",
                step_index=transition.next_state.step_index,
                exception=not moving_signatures,
            )

            updated_control_signatures: set[ObjectSignature] = set()
            for signature in moving_signatures:
                motion = motion_summary[signature][2]
                control = self.control_hypotheses.setdefault(
                    (action_mode_key, signature),
                    ControlHypothesis(evidence=EvidenceCounter()),
                )
                control.observe(moved=True, motion=motion, reward=float(transition.reward))
                updated_control_signatures.add(signature)
                self._record_proof(
                    proof_type="support",
                    hypothesis_type="control",
                    action=transition.action,
                    subject=_signature_code(signature),
                    relation="controllable",
                    object=_mode_code(action_mode_key),
                    confidence=control.evidence.confidence,
                    evidence=float(control.evidence.balance),
                    predicted="selected object responds",
                    observed=f"motion={motion:.2f}",
                    step_index=transition.next_state.step_index,
                )
            for mode_key, signature in list(self.control_hypotheses.keys()):
                if mode_key != action_mode_key or signature in updated_control_signatures:
                    continue
                control = self.control_hypotheses[(mode_key, signature)]
                if signature in _signatures_in_state(transition.state):
                    control.observe(moved=False, motion=0.0, reward=0.0)
                    self._record_proof(
                        proof_type="contradiction",
                        hypothesis_type="control",
                        action=transition.action,
                        subject=_signature_code(signature),
                        relation="controllable",
                        object=_mode_code(action_mode_key),
                        confidence=control.evidence.confidence,
                        evidence=float(control.evidence.balance),
                        predicted="selected object responds",
                        observed="stalled",
                        step_index=transition.next_state.step_index,
                        exception=True,
                    )

        if schema.action_type == "interact":
            self._observe_interaction_objective_evidence(
                transition=transition,
                action_mode_key=action_mode_key,
                context=context,
                goal_activated=goal_activated,
                state_delta=transition_delta,
                moving_signatures=moving_signatures,
            )
            if goal_activated:
                next_move_actions = [
                    action
                    for action in transition.next_state.affordances
                    if build_action_schema(action, build_action_schema_context(transition.next_state.affordances, dict(transition.next_state.action_roles))).action_type == "move"
                ]
                goal_movers = set(moving_signatures) or set(self._mover_scores(transition.next_state, next_move_actions))
                self._seed_goal_target_objectives(transition.next_state, action_mode_key, goal_movers)
                self._clear_active_exploit()
                self._clear_active_probe()

        self._update_active_option(
            transition,
            action_mode_key=action_mode_key,
            moving_signatures=moving_signatures,
            state_delta=transition_delta,
            goal_activated=goal_activated,
            schema=schema,
        )

        if schema.action_type == "move" and self._recent_selector_action is not None and moving_signatures:
            best_signature = max(
                moving_signatures,
                key=lambda signature: motion_summary.get(signature, (0.0, 0.0, 0.0))[2],
            )
            option_key = self._materialize_option(
                option_type="selector_move",
                action_sequence=(self._recent_selector_action, transition.action),
                mover_signature=best_signature,
                target_signature=None,
                objective_family="mode_followup",
                mode_key=action_mode_key,
            )
            option = self.option_hypotheses[option_key]
            option.observe(
                progress=motion_summary.get(best_signature, (0.0, 0.0, 0.0))[2],
                reward=float(transition.reward),
                supported=True,
            )
            self._record_proof(
                proof_type="support",
                hypothesis_type="option",
                action=transition.action,
                subject="selector_move",
                relation=_mode_code(action_mode_key),
                object="then".join((self._recent_selector_action, transition.action)),
                confidence=option.evidence.confidence,
                evidence=float(option.evidence.balance),
                predicted="selector enables reusable move chain",
                observed=f"motion={motion_summary.get(best_signature, (0.0, 0.0, 0.0))[2]:.2f}",
                step_index=transition.next_state.step_index,
            )
            self._recent_selector_action = None

        if not moving_signatures:
            if schema.action_type not in {"click", "select"}:
                self._recent_selector_action = None
            return

        target_objects = [
            obj
            for obj in transition.next_state.objects
            if self._is_candidate_target_object(obj, moving_signatures, transition.next_state.grid_shape)
        ]
        if not target_objects:
            return

        for mover_signature in moving_signatures:
            mover_before = before_groups.get(mover_signature, [])
            mover_after = after_groups.get(mover_signature, [])
            if not mover_before or not mover_after:
                continue
            for target in target_objects:
                target_signature = object_signature(target)
                if target_signature == mover_signature:
                    continue
                objective = self.objective_hypotheses.setdefault(
                    (action_mode_key, mover_signature, target_signature),
                    ObjectiveHypothesis(evidence=EvidenceCounter()),
                )
                before_best = min(_cell_distance(before_obj, target) for before_obj in mover_before)
                after_best = min(_cell_distance(after_obj, target) for after_obj in mover_after)
                contact = any(_overlap_fraction(after_obj, target) > 0.6 for after_obj in mover_after)
                objective.observe(
                    action=transition.action,
                    progress=before_best - after_best,
                    reward=float(transition.reward),
                    contact=contact,
                    state_delta=transition_delta,
                    goal_activated=goal_activated,
                )
                for family in self._candidate_objective_families(
                    transition.next_state,
                    mover_signature,
                    target,
                    objective,
                ):
                    family_hypothesis = self._ensure_objective_family_hypothesis(
                        mover_signature,
                        target_signature,
                        family,
                        mode_key=action_mode_key,
                    )
                    family_hypothesis.observe(
                        action=transition.action,
                        progress=before_best - after_best,
                        reward=float(transition.reward),
                        contact=contact,
                        state_delta=transition_delta,
                        goal_activated=goal_activated,
                    )
                if before_best - after_best > 0.0 or float(transition.reward) > 0.0:
                    self._record_proof(
                        proof_type="support",
                        hypothesis_type="objective",
                        action=transition.action,
                        subject=_signature_code(mover_signature),
                        relation="toward",
                        object=_signature_code(target_signature),
                        confidence=objective.evidence.confidence,
                        evidence=float(objective.evidence.balance),
                        predicted="distance shrinks",
                        observed=f"{before_best:.2f}->{after_best:.2f}",
                        step_index=transition.next_state.step_index,
                    )
                elif contact and float(transition.reward) <= 0.0:
                    self._record_proof(
                        proof_type="contradiction",
                        hypothesis_type="objective",
                        action=transition.action,
                        subject=_signature_code(mover_signature),
                        relation="toward",
                        object=_signature_code(target_signature),
                        confidence=objective.evidence.confidence,
                        evidence=float(objective.evidence.balance),
                        predicted="contact should help",
                        observed=f"contact/no progress reward={float(transition.reward):.2f}",
                        step_index=transition.next_state.step_index,
                        exception=True,
                    )

        if schema.action_type not in {"click", "select"}:
            self._recent_selector_action = None

    def _should_abandon_mode(self) -> bool:
        if self.current_mode_key == DEFAULT_MODE:
            return False
        mode = self.mode_hypotheses.get(self.current_mode_key)
        if mode is None:
            return False
        if mode.move_effect_steps == 0 and mode.stalled_steps >= 2:
            return True
        if mode.entries >= 2 and mode.utility <= 0.0:
            return True
        return False

    def _repair_pressure(self) -> float:
        if not self.repair_hypotheses:
            return 0.0
        return max(
            (repair.evidence.confidence + (0.25 * max(repair.evidence.balance, 0))) for repair in self.repair_hypotheses.values()
        )

    def _control_posteriors(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
    ) -> dict[ObjectSignature, float]:
        log_weights: dict[ObjectSignature, float] = {}
        signatures = {object_signature(obj) for obj in state.objects}
        for signature in signatures:
            control = self.control_hypotheses.get((self.current_mode_key, signature))
            support = 0 if control is None else control.evidence.support
            contradiction = 0 if control is None else control.evidence.contradiction
            best_magnitude = 0.0
            best_rule_confidence = 0.0
            rule_support = 0
            for obj in state.objects:
                if object_signature(obj) != signature:
                    continue
                for action in move_actions:
                    rule = self._motion_rule_for_object(action, obj)
                    if rule is None:
                        continue
                    support += rule.evidence.support
                    contradiction += rule.evidence.contradiction
                    rule_support += rule.evidence.support
                    best_rule_confidence = max(best_rule_confidence, rule.evidence.confidence)
                    best_magnitude = max(best_magnitude, rule.mean_magnitude)
            if (
                support <= 0
                and rule_support <= 0
                and "agent" not in signature[4]
                and "repaired_mover" not in signature[4]
            ):
                continue
            log_weight = _beta_log_evidence(support, contradiction)
            log_weight += (0.45 * best_magnitude) + (0.25 * best_rule_confidence)
            if control is not None:
                log_weight += 0.1 * max(control.reward_sum, 0.0)
            if "agent" in signature[4] or "repaired_mover" in signature[4]:
                log_weight += 0.3
            log_weights[signature] = log_weight
        return _normalize_log_weights(log_weights)

    def _objective_posteriors(
        self,
        state: StructuredState,
        mover_scores: dict[ObjectSignature, float],
    ) -> dict[tuple[ObjectSignature, ObjectSignature], float]:
        log_weights: dict[tuple[ObjectSignature, ObjectSignature], float] = {}
        mover_signatures = set(mover_scores)
        unresolved_interaction_targets = self._has_unresolved_interaction_targets(state, mover_scores)
        for target_object in state.objects:
            if not self._is_candidate_target_object(target_object, mover_signatures, state.grid_shape):
                continue
            if "target" in target_object.tags and not self._goal_is_active(state) and unresolved_interaction_targets:
                continue
            if self._goal_is_active(state) and "interactable" in target_object.tags and "target" not in target_object.tags:
                continue
            target_signature = object_signature(target_object)
            for mover_signature in mover_scores:
                objective = self.objective_hypotheses.get((self.current_mode_key, mover_signature, target_signature))
                if objective is None:
                    objective = ObjectiveHypothesis(evidence=EvidenceCounter())
                log_weight = _beta_log_evidence(objective.evidence.support, objective.evidence.contradiction)
                log_weight += 0.12 * objective.progress_sum
                log_weight += 0.35 * objective.reward_sum
                log_weight += 0.3 * objective.successful_interactions
                log_weight += 0.2 * objective.goal_activations
                log_weight -= 0.35 * objective.failed_interactions
                log_weight += 0.1 * mover_scores.get(mover_signature, 0.0)
                if self._requires_interaction_confirmation(state, target_object, objective):
                    log_weight -= 0.25
                if "target" in target_object.tags or "repaired_goal" in target_object.tags:
                    log_weight += 0.2
                if "interactable" in target_object.tags or "repaired_control" in target_object.tags:
                    log_weight += 0.1
                log_weights[(mover_signature, target_signature)] = log_weight
        return _normalize_log_weights(log_weights)

    def _ensure_objective_hypothesis(
        self,
        mover_signature: ObjectSignature,
        target_signature: ObjectSignature,
    ) -> ObjectiveHypothesis:
        return self.objective_hypotheses.setdefault(
            (self.current_mode_key, mover_signature, target_signature),
            ObjectiveHypothesis(evidence=EvidenceCounter()),
        )

    def _ensure_objective_family_hypothesis(
        self,
        mover_signature: ObjectSignature,
        target_signature: ObjectSignature,
        family: str,
        *,
        mode_key: str | None = None,
    ) -> ObjectiveFamilyHypothesis:
        active_mode = self.current_mode_key if mode_key is None else mode_key
        return self.objective_family_hypotheses.setdefault(
            (active_mode, mover_signature, target_signature, family),
            ObjectiveFamilyHypothesis(family=family, evidence=EvidenceCounter()),
        )

    def _candidate_objective_families(
        self,
        state: StructuredState,
        mover_signature: ObjectSignature,
        target_object: ObjectState,
        objective: ObjectiveHypothesis,
    ) -> tuple[str, ...]:
        target_signature = object_signature(target_object)
        families: list[str] = ["approach", "contact"]
        if self._is_interaction_candidate_object(target_object, {mover_signature}, state.grid_shape):
            families.append("interact")
        if (
            not self._goal_is_active(state)
            and (
                "target" in target_object.tags
                or "repaired_goal" in target_object.tags
                or "interactable" in target_object.tags
                or "repaired_control" in target_object.tags
            )
        ):
            families.append("activate")
        avoid_hypothesis = self.objective_family_hypotheses.get(
            (self.current_mode_key, mover_signature, target_signature, "avoid")
        )
        if (
            objective.failed_interactions > 0
            or objective.stalled_contacts > 0
            or (avoid_hypothesis is not None and avoid_hypothesis.evidence.support > 0)
        ):
            families.append("avoid")
        return tuple(dict.fromkeys(families))

    def _objective_family_posteriors(
        self,
        state: StructuredState,
        mover_scores: dict[ObjectSignature, float],
    ) -> dict[tuple[ObjectSignature, ObjectSignature, str], float]:
        log_weights: dict[tuple[ObjectSignature, ObjectSignature, str], float] = {}
        mover_signatures = set(mover_scores)
        unresolved_interaction_targets = self._has_unresolved_interaction_targets(state, mover_scores)
        for target_object in state.objects:
            if not self._is_candidate_target_object(target_object, mover_signatures, state.grid_shape):
                continue
            if "target" in target_object.tags and not self._goal_is_active(state) and unresolved_interaction_targets:
                continue
            if self._goal_is_active(state) and "interactable" in target_object.tags and "target" not in target_object.tags:
                continue
            target_signature = object_signature(target_object)
            for mover_signature in mover_scores:
                objective = self._ensure_objective_hypothesis(mover_signature, target_signature)
                for family in self._candidate_objective_families(state, mover_signature, target_object, objective):
                    hypothesis = self._ensure_objective_family_hypothesis(mover_signature, target_signature, family)
                    log_weight = _beta_log_evidence(hypothesis.evidence.support, hypothesis.evidence.contradiction)
                    log_weight += 0.08 * objective.utility
                    log_weight += 0.14 * hypothesis.utility
                    log_weight += 0.1 * mover_scores.get(mover_signature, 0.0)
                    if family == "approach":
                        log_weight += 0.12 * objective.progress_sum
                        if "target" in target_object.tags or "repaired_goal" in target_object.tags:
                            log_weight += 0.2
                    elif family == "contact":
                        log_weight += 0.22 * hypothesis.contact_hits
                        log_weight += 0.15 * objective.contact_count
                    elif family == "interact":
                        if "interactable" in target_object.tags or "repaired_control" in target_object.tags:
                            log_weight += 0.25
                        log_weight += 0.25 * hypothesis.interaction_hits
                    elif family == "activate":
                        if "target" in target_object.tags or "repaired_goal" in target_object.tags:
                            log_weight += 0.2
                        log_weight += 0.35 * hypothesis.goal_hits
                        log_weight += 0.18 * objective.goal_activations
                    elif family == "avoid":
                        log_weight += 0.3 * hypothesis.negative_hits
                        log_weight += 0.2 * objective.failed_interactions
                        log_weight += 0.15 * objective.stalled_contacts
                    log_weights[(mover_signature, target_signature, family)] = log_weight
        return _normalize_log_weights(log_weights)

    def _objective_family_candidates(
        self,
        state: StructuredState,
        mover_scores: dict[ObjectSignature, float],
    ) -> list[tuple[ObjectSignature, ObjectState, str, ObjectiveHypothesis, ObjectiveFamilyHypothesis]]:
        candidates: list[tuple[ObjectSignature, ObjectState, str, ObjectiveHypothesis, ObjectiveFamilyHypothesis]] = []
        family_posteriors = self._objective_family_posteriors(state, mover_scores)
        mover_signatures = set(mover_scores)
        unresolved_interaction_targets = self._has_unresolved_interaction_targets(state, mover_scores)
        for target_object in state.objects:
            if not self._is_candidate_target_object(target_object, mover_signatures, state.grid_shape):
                continue
            if "target" in target_object.tags and not self._goal_is_active(state) and unresolved_interaction_targets:
                continue
            if self._goal_is_active(state) and "interactable" in target_object.tags and "target" not in target_object.tags:
                continue
            target_signature = object_signature(target_object)
            for mover_signature in mover_scores:
                objective = self._ensure_objective_hypothesis(mover_signature, target_signature)
                for family in self._candidate_objective_families(state, mover_signature, target_object, objective):
                    posterior = family_posteriors.get((mover_signature, target_signature, family), 0.0)
                    family_hypothesis = self._ensure_objective_family_hypothesis(mover_signature, target_signature, family)
                    if posterior < 0.04 and family_hypothesis.utility <= 0.5 and objective.utility <= 1.0:
                        continue
                    candidates.append((mover_signature, target_object, family, objective, family_hypothesis))
        return candidates

    def _mode_posteriors(
        self,
        selector_actions: list[ActionName],
        context,
        thought: RuntimeThought | None,
    ) -> dict[str, float]:
        grouped: dict[str, list[ActionName]] = defaultdict(list)
        for action in selector_actions:
            grouped[_mode_key(action, build_action_schema(action, context))].append(action)
        log_weights: dict[str, float] = {}
        for mode_key, actions in grouped.items():
            hypothesis = self.mode_hypotheses.get(mode_key)
            support = 0 if hypothesis is None else hypothesis.evidence.support
            contradiction = 0 if hypothesis is None else hypothesis.evidence.contradiction
            followup = max(self._thought_selector_followup(thought, action) for action in actions)
            uncertainty = max(self._thought_uncertainty(thought, action) for action in actions)
            log_weight = _beta_log_evidence(support, contradiction)
            if hypothesis is not None:
                log_weight += 0.2 * hypothesis.utility
                log_weight += 0.1 * min(hypothesis.entries, 8)
            log_weight += 0.45 * max(followup, 0.0)
            log_weight += 0.15 * uncertainty
            log_weights[mode_key] = log_weight
        return _normalize_log_weights(log_weights)

    def _predicted_progress_for_action(
        self,
        state: StructuredState,
        action: ActionName,
        mover_signature: ObjectSignature,
        target_object: ObjectState,
    ) -> float:
        current_objects = _group_by_signature(state.objects).get(mover_signature, [])
        predicted_objects = self._predict_signature_objects(state, action, mover_signature)
        if not current_objects or not predicted_objects:
            return 0.0
        before_distance = min(_cell_distance(obj, target_object) for obj in current_objects)
        after_distance = min(_cell_distance(obj, target_object) for obj in predicted_objects)
        overlap = max(_overlap_fraction(obj, target_object) for obj in predicted_objects)
        return (before_distance - after_distance) + (3.0 * overlap)

    def _ready_to_exploit(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        mover_scores: dict[ObjectSignature, float],
    ) -> bool:
        if not move_actions or not mover_scores:
            return False
        if self._repair_pressure() >= 1.3:
            return False
        family_candidates = self._objective_family_candidates(state, mover_scores)
        if not family_candidates:
            return False
        family_posteriors = self._objective_family_posteriors(state, mover_scores)
        return any(
            family_posteriors.get((mover_signature, object_signature(target_object), family), 0.0) >= 0.22
            or family_hypothesis.utility >= 1.0
            or objective.utility >= 2.0
            for mover_signature, target_object, family, objective, family_hypothesis in family_candidates
        )

    def _objective_family_sequence(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        interact_actions: list[ActionName],
        mover_signature: ObjectSignature,
        target_object: ObjectState,
        family: str,
        objective: ObjectiveHypothesis,
    ) -> tuple[tuple[ActionName, ...], float] | None:
        target_signature = object_signature(target_object)
        if family == "avoid":
            movers = _group_by_signature(state.objects).get(mover_signature, [])
            if not movers:
                return None
            best_action: ActionName | None = None
            best_retreat = float("-inf")
            for action in move_actions:
                predicted_objects = self._predict_signature_objects(state, action, mover_signature)
                if not predicted_objects:
                    continue
                before_distance = min(_cell_distance(obj, target_object) for obj in movers)
                after_distance = min(_cell_distance(obj, target_object) for obj in predicted_objects)
                retreat = after_distance - before_distance
                if retreat > best_retreat:
                    best_retreat = retreat
                    best_action = action
            if best_action is None or best_retreat <= 0.0:
                return None
            return ((best_action,), best_retreat)

        if family in {"approach", "contact"}:
            path_sequence = self._path_sequence_to_cells(
                state=state,
                move_actions=move_actions,
                mover_signature=mover_signature,
                goal_cells=set(target_object.cells),
            )
            if path_sequence:
                contact_bonus = 0.6 if family == "contact" else 0.0
                return (path_sequence, max(1.5 - (0.1 * float(len(path_sequence))), 0.25) + contact_bonus)
            movers = _group_by_signature(state.objects).get(mover_signature, [])
            if not movers:
                return None
            best_action = None
            best_progress = float("-inf")
            for action in move_actions:
                predicted_objects = self._predict_signature_objects(state, action, mover_signature)
                if not predicted_objects:
                    continue
                before_distance = min(_cell_distance(obj, target_object) for obj in movers)
                after_distance = min(_cell_distance(obj, target_object) for obj in predicted_objects)
                overlap = max(_overlap_fraction(obj, target_object) for obj in predicted_objects)
                progress = (before_distance - after_distance) + ((4.0 if family == "contact" else 3.0) * overlap)
                if progress > best_progress:
                    best_progress = progress
                    best_action = action
            if best_action is None or best_progress <= 0.0:
                return None
            return ((best_action,), best_progress)

        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        if family in {"interact", "activate"}:
            if family == "activate" and ("target" in target_object.tags or "repaired_goal" in target_object.tags) and not interact_actions:
                goal_path = self._path_sequence_to_cells(
                    state=state,
                    move_actions=move_actions,
                    mover_signature=mover_signature,
                    goal_cells=set(target_object.cells),
                )
                if goal_path:
                    return (goal_path, max(1.8 - (0.1 * float(len(goal_path))), 0.35) + 0.9)
            for action in interact_actions:
                interaction_target = self._interaction_target_object(state, action, context)
                if interaction_target is not None and object_signature(interaction_target) == target_signature:
                    activation_bonus = 1.0 if family == "activate" else 0.4
                    return ((action,), 1.5 + activation_bonus + (0.25 * objective.utility))
            path_sequence = self._path_sequence_to_interaction_target(
                state=state,
                move_actions=move_actions,
                interact_actions=interact_actions,
                mover_signature=mover_signature,
                target_object=target_object,
            )
            if path_sequence:
                activation_bonus = 1.0 if family == "activate" else 0.5
                return (path_sequence, max(2.0 - (0.1 * float(len(path_sequence))), 0.35) + activation_bonus)
            path_plan = self._path_action_to_interaction_target(
                state=state,
                move_actions=move_actions,
                mover_signature=mover_signature,
                target_object=target_object,
            )
            if path_plan is not None:
                action, steps = path_plan
                activation_bonus = 1.0 if family == "activate" else 0.5
                return ((action,), max(1.5 - (0.1 * float(steps)), 0.25) + activation_bonus)
        return None

    def _exploit_plan(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        mover_scores: dict[ObjectSignature, float],
        thought: RuntimeThought | None,
    ) -> PlanOutput | None:
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        interact_actions = [
            action for action in state.affordances if build_action_schema(action, context).action_type == "interact"
        ]
        family_posteriors = self._objective_family_posteriors(state, mover_scores)
        family_candidates = self._objective_family_candidates(state, mover_scores)
        if not family_candidates:
            return None

        best_action: ActionName | None = None
        best_search_path: tuple[ActionName, ...] = ()
        best_score = float("-inf")
        best_local_score = float("-inf")
        best_pair_key: tuple[str, ObjectSignature, ObjectSignature] | None = None
        best_target: ObjectState | None = None
        best_family: str | None = None

        for mover_signature, target_object, family, objective, family_hypothesis in family_candidates:
            posterior = family_posteriors.get((mover_signature, object_signature(target_object), family), 0.0)
            if posterior <= 0.0:
                continue
            target_signature = object_signature(target_object)
            sequence_plan = self._objective_family_sequence(
                state=state,
                move_actions=move_actions,
                interact_actions=interact_actions,
                mover_signature=mover_signature,
                target_object=target_object,
                family=family,
                objective=objective,
            )
            if sequence_plan is None:
                continue
            control_score = mover_scores.get(mover_signature, 0.0)
            search_path, local_score = sequence_plan
            if not search_path:
                continue
            thought_bonus = self._thought_value(thought, search_path[0]) + (0.25 * self._thought_policy(thought, search_path[0]))
            score = posterior * (local_score + objective.utility + family_hypothesis.utility + control_score + thought_bonus)
            if score > best_score:
                best_score = score
                best_local_score = local_score
                best_action = search_path[0]
                best_search_path = search_path
                best_pair_key = (self.current_mode_key, mover_signature, target_signature)
                best_target = target_object
                best_family = family

        if best_action is None or best_pair_key is None or best_target is None or best_family is None or best_local_score <= 0.0:
            return None

        self.active_pair_key = best_pair_key
        self.active_objective_family = best_family
        if best_family == "avoid":
            self.active_goal_anchor = None
            self.active_exploit_action = None
        else:
            self.active_goal_anchor = best_target.centroid
            self.active_exploit_action = best_action
        target_signature = best_pair_key[2]
        best_objective = self._ensure_objective_hypothesis(best_pair_key[1], target_signature)
        best_family_hypothesis = self._ensure_objective_family_hypothesis(best_pair_key[1], target_signature, best_family)
        if len(best_search_path) >= 2:
            option_key = self._materialize_option(
                option_type="objective_chain",
                action_sequence=best_search_path,
                mover_signature=best_pair_key[1],
                target_signature=target_signature,
                objective_family=best_family,
            )
            self._activate_option(option_key)
        selector_actions = [
            action for action in state.affordances if build_action_schema(action, context).action_type in {"click", "select"}
        ]
        best_selector_action = self._best_selector_action_for_mode(self.current_mode_key, selector_actions, context)
        if best_selector_action is not None and self.current_mode_key != DEFAULT_MODE:
            self._materialize_option(
                option_type="bind_then_objective",
                action_sequence=(best_selector_action,) + best_search_path,
                mover_signature=best_pair_key[1],
                target_signature=target_signature,
                objective_family=best_family,
                mode_key=self.current_mode_key,
            )
        target_code = f"{target_signature[0]}:{target_signature[1]}:{target_signature[2]}x{target_signature[3]}"
        return PlanOutput(
            action=best_action,
            scores={
                "exploit": best_score,
                "objective_utility": best_objective.utility,
                "objective_posterior": family_posteriors.get((best_pair_key[1], best_pair_key[2], best_family), 0.0),
                "objective_family_utility": best_family_hypothesis.utility,
            },
            language=LanguageTrace(
                belief_tokens=("belief", "objective_hypothesis", best_family, target_code),
                question_tokens=self._question_tokens(thought, ("question", "reduce_objective_distance")),
                plan_tokens=("plan", best_action),
            ),
            search_path=best_search_path or (best_action,),
        )

    def _materialize_option(
        self,
        *,
        option_type: str,
        action_sequence: tuple[ActionName, ...],
        mover_signature: ObjectSignature | None,
        target_signature: ObjectSignature | None,
        objective_family: str = "",
        mode_key: str | None = None,
    ) -> tuple[str, str, ObjectSignature | None, ObjectSignature | None, tuple[ActionName, ...]]:
        active_mode = self.current_mode_key if mode_key is None else mode_key
        key = (option_type, active_mode, mover_signature, target_signature, action_sequence)
        if key not in self.option_hypotheses:
            self.option_hypotheses[key] = OptionHypothesis(
                option_type=option_type,
                evidence=EvidenceCounter(),
                action_sequence=action_sequence,
                mode_key=active_mode,
                mover_signature=mover_signature,
                target_signature=target_signature,
                objective_family=objective_family,
            )
        return key

    def _activate_option(
        self,
        key: tuple[str, str, ObjectSignature | None, ObjectSignature | None, tuple[ActionName, ...]],
    ) -> None:
        self.active_option_key = key
        self.active_option_index = 0

    def _option_plan(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        interact_actions: list[ActionName],
        selector_actions: list[ActionName],
        mover_scores: dict[ObjectSignature, float],
        thought: RuntimeThought | None,
        context,
    ) -> PlanOutput | None:
        if self.active_option_key is not None:
            option = self.option_hypotheses.get(self.active_option_key)
            if option is None:
                self._clear_active_option()
            elif self.active_option_index < len(option.action_sequence):
                remaining = option.action_sequence[self.active_option_index :]
                action = remaining[0]
                if action in state.affordances and self._option_applicable(
                    state,
                    option,
                    remaining=remaining,
                    mover_scores=mover_scores,
                    thought=thought,
                    context=context,
                ):
                    return PlanOutput(
                        action=action,
                        scores={
                            "option": option.utility,
                            "option_confidence": option.evidence.confidence,
                        },
                        language=LanguageTrace(
                            belief_tokens=("belief", "option_active", option.option_type),
                            question_tokens=self._question_tokens(thought, ("question", "continue_option")),
                            plan_tokens=("plan",) + remaining[: min(len(remaining), 3)],
                        ),
                        search_path=remaining,
                    )
                self._clear_active_option()

        family_posteriors = self._objective_family_posteriors(state, mover_scores)
        mode_posteriors = self._mode_posteriors(selector_actions, context, thought) if selector_actions else {}
        best_key = None
        best_option: OptionHypothesis | None = None
        best_score = float("-inf")
        for key, option in self.option_hypotheses.items():
            if len(option.action_sequence) < 2 or option.evidence.confidence < 0.56:
                continue
            if option.action_sequence[0] not in state.affordances:
                continue
            if not self._option_applicable(
                state,
                option,
                remaining=option.action_sequence,
                mover_scores=mover_scores,
                thought=thought,
                context=context,
            ):
                continue
            score = option.utility
            if option.option_type in {"selector_move", "mode_probe_chain"}:
                score += mode_posteriors.get(option.mode_key, 0.0)
                score += self._thought_selector_followup(thought, option.action_sequence[0])
            elif option.option_type == "bind_then_objective":
                mode = self.mode_hypotheses.get(option.mode_key)
                score += mode_posteriors.get(option.mode_key, 0.0)
                score += 0.0 if mode is None else mode.evidence.confidence
                score += self._thought_selector_followup(thought, option.action_sequence[0])
                if option.mover_signature is not None and option.target_signature is not None:
                    score += family_posteriors.get(
                        (option.mover_signature, option.target_signature, option.objective_family or "approach"),
                        0.0,
                    )
            elif option.mover_signature is not None and option.target_signature is not None:
                score += family_posteriors.get(
                    (option.mover_signature, option.target_signature, option.objective_family or "approach"),
                    0.0,
                )
                score += self._thought_value(thought, option.action_sequence[0])
            else:
                score += self._thought_value(thought, option.action_sequence[0])
            if score > best_score:
                best_score = score
                best_key = key
                best_option = option

        if best_key is None or best_option is None or best_score <= 1.1:
            return None
        self._activate_option(best_key)
        return PlanOutput(
            action=best_option.action_sequence[0],
            scores={"option": best_score, "option_confidence": best_option.evidence.confidence},
            language=LanguageTrace(
                belief_tokens=("belief", "option_reuse", best_option.option_type),
                question_tokens=self._question_tokens(thought, ("question", "reuse_option")),
                plan_tokens=("plan",) + best_option.action_sequence[: min(len(best_option.action_sequence), 3)],
            ),
            search_path=best_option.action_sequence,
        )

    def _first_contact_program_plan(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        interact_actions: list[ActionName],
        selector_actions: list[ActionName],
        mover_scores: dict[ObjectSignature, float],
        thought: RuntimeThought | None,
        context,
    ) -> PlanOutput | None:
        if not selector_actions or not move_actions or self._goal_is_active(state):
            return None
        if self.active_pair_key is not None or self.active_probe_pair_key is not None:
            return None
        strongest_control = max(self._control_posteriors(state, move_actions).values(), default=0.0)
        if strongest_control >= 0.45 and mover_scores:
            return None
        mode_posteriors = self._mode_posteriors(selector_actions, context, thought)
        best_selector: ActionName | None = None
        best_mode_key = DEFAULT_MODE
        best_target_signature: ObjectSignature | None = None
        best_score = float("-inf")
        for action in selector_actions:
            schema = build_action_schema(action, context)
            mode_key = _mode_key(action, schema)
            mode = self.mode_hypotheses.get(mode_key)
            target_object = self._interaction_target_object(state, action, context)
            target_bonus = 0.0
            target_signature: ObjectSignature | None = None
            if target_object is not None:
                target_signature = object_signature(target_object)
                target_bonus += 0.35
                if any(tag in target_object.tags for tag in ("clickable", "selector_candidate", "interface_target")):
                    target_bonus += 0.25
                if any(tag in target_object.tags for tag in ("target", "repaired_goal")):
                    target_bonus += 0.2
                if any(tag in target_object.tags for tag in ("interactable", "repaired_control")):
                    target_bonus += 0.2
            mode_entries = 0 if mode is None else mode.entries
            score = (
                (1.0 / math.sqrt(self.selector_visits[action] + 1.0))
                + (1.0 / math.sqrt(mode_entries + 1.0))
                + (0.5 * mode_posteriors.get(mode_key, 0.0))
                + max(self._thought_selector_followup(thought, action), 0.0)
                + (0.4 * self._thought_uncertainty(thought, action))
                + (0.1 * self._thought_value(thought, action))
                + target_bonus
            )
            if score > best_score:
                best_score = score
                best_selector = action
                best_mode_key = mode_key
                best_target_signature = target_signature
        followup_action, followup_kind = self._best_probe_followup_action(
            state,
            move_actions,
            interact_actions,
            thought,
            context,
        )
        if best_selector is None or followup_action is None or best_score <= 0.6:
            return None
        option_key = self._materialize_option(
            option_type="mode_probe_chain",
            action_sequence=(best_selector, followup_action),
            mover_signature=None,
            target_signature=best_target_signature,
            objective_family="control_binding",
            mode_key=best_mode_key,
        )
        self._activate_option(option_key)
        target_code = "unknown" if best_target_signature is None else _signature_code(best_target_signature)
        return PlanOutput(
            action=best_selector,
            scores={
                "diagnostic": best_score,
                "first_contact": 1.0,
                "mode_probe": 1.0,
                "mode_posterior": mode_posteriors.get(best_mode_key, 0.0),
            },
            language=LanguageTrace(
                belief_tokens=("belief", "control_binding", "uncertain", "clickable"),
                question_tokens=self._question_tokens(thought, ("question", "need", "test", "focus", "control_binding", "state", "probe")),
                plan_tokens=(
                    "plan",
                    "click_then_interact" if followup_kind == "interact" else "click_then_move",
                    "focus",
                    "control_binding",
                    "state",
                    "probe",
                ),
            ),
            search_path=(best_selector, followup_action),
        )

    def _best_probe_followup_action(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        interact_actions: list[ActionName],
        thought: RuntimeThought | None,
        context,
    ) -> tuple[ActionName | None, str]:
        best_action: ActionName | None = None
        best_kind = ""
        best_score = float("-inf")
        candidates = [
            action
            for action in move_actions
            if not self._move_action_hits_visible_blocker(state, action, context)
        ]
        if not candidates:
            candidates = list(move_actions)
        for action in candidates:
            score = (
                0.65
                + (0.25 if self.action_visits[(self.current_mode_key, action)] == 0 else 0.0)
                + self._thought_uncertainty(thought, action)
                + (0.35 * self._thought_value(thought, action))
                - (0.08 * self.action_visits[(self.current_mode_key, action)])
            )
            if score > best_score:
                best_action = action
                best_kind = "move"
                best_score = score
        for action in interact_actions:
            target_object = self._interaction_target_object(state, action, context)
            if target_object is None:
                continue
            target_bonus = 0.35
            if any(
                tag in target_object.tags
                for tag in ("interactable", "repaired_control", "clickable", "interface_target", "selector_candidate")
            ):
                target_bonus += 0.25
            if any(tag in target_object.tags for tag in ("target", "repaired_goal")):
                target_bonus += 0.15
            score = (
                0.8
                + target_bonus
                + (0.25 if self.action_visits[(self.current_mode_key, action)] == 0 else 0.0)
                + (0.8 * self._thought_uncertainty(thought, action))
                + (0.4 * self._thought_value(thought, action))
                - (0.08 * self.action_visits[(self.current_mode_key, action)])
            )
            if score > best_score:
                best_action = action
                best_kind = "interact"
                best_score = score
        return best_action, best_kind

    def _best_selector_action_for_mode(
        self,
        mode_key: str,
        selector_actions: list[ActionName],
        context,
    ) -> ActionName | None:
        if not selector_actions:
            return None
        hypothesis = self.mode_hypotheses.get(mode_key)
        if hypothesis is None:
            return max(selector_actions, key=lambda action: -self.selector_visits[action], default=None)
        def _score(action: ActionName) -> tuple[float, float, float]:
            schema = build_action_schema(action, context)
            target_bonus = 0.05 if schema.coarse_bin is not None else 0.0
            return (
                float(hypothesis.action_support.get(action, 0)),
                target_bonus,
                -float(self.selector_visits[action]),
            )
        return max(selector_actions, key=_score, default=None)

    def _option_applicable(
        self,
        state: StructuredState,
        option: OptionHypothesis,
        *,
        remaining: tuple[ActionName, ...],
        mover_scores: dict[ObjectSignature, float],
        thought: RuntimeThought | None,
        context,
    ) -> bool:
        del thought
        if not remaining or remaining[0] not in state.affordances:
            return False
        if option.option_type not in {"selector_move", "mode_probe_chain", "bind_then_objective"} and option.mode_key not in {DEFAULT_MODE, self.current_mode_key}:
            return False
        if option.option_type == "bind_then_objective":
            if remaining == option.action_sequence and self.current_mode_key != DEFAULT_MODE:
                return False
            if remaining != option.action_sequence and option.mode_key != self.current_mode_key:
                return False
        if option.mover_signature is not None and option.mover_signature not in mover_scores:
            return False
        if option.target_signature is not None:
            target_object = next((obj for obj in state.objects if object_signature(obj) == option.target_signature), None)
            if target_object is None:
                return False
            if option.mover_signature is not None and option.objective_family:
                score = self._family_action_separation_score(
                    state,
                    remaining[0],
                    option.mover_signature,
                    target_object,
                    option.objective_family,
                )
                if score < -0.25:
                    return False
        if option.option_type in {"selector_move", "mode_probe_chain"}:
            schema = build_action_schema(remaining[0], context)
            return schema.action_type in {"click", "select", "move", "interact"}
        if option.option_type == "bind_then_objective" and remaining == option.action_sequence:
            schema = build_action_schema(remaining[0], context)
            return schema.action_type in {"click", "select"}
        return True

    def _update_active_option(
        self,
        transition: Transition,
        *,
        action_mode_key: str,
        moving_signatures: set[ObjectSignature],
        state_delta: float,
        goal_activated: bool,
        schema,
    ) -> None:
        if self.active_option_key is None:
            return
        option = self.option_hypotheses.get(self.active_option_key)
        if option is None:
            self._clear_active_option()
            return
        if self.active_option_index >= len(option.action_sequence):
            self._clear_active_option()
            return
        expected_action = option.action_sequence[self.active_option_index]
        if expected_action != transition.action:
            option.observe(progress=0.0, reward=0.0, supported=False, contradiction=True)
            self._clear_active_option()
            return

        progress = 0.0
        if option.mover_signature is not None and option.target_signature is not None:
            before_movers = _group_by_signature(transition.state.objects).get(option.mover_signature, [])
            after_movers = _group_by_signature(transition.next_state.objects).get(option.mover_signature, [])
            target_before = next((obj for obj in transition.state.objects if object_signature(obj) == option.target_signature), None)
            target_after = next((obj for obj in transition.next_state.objects if object_signature(obj) == option.target_signature), None)
            target_object = target_after or target_before
            if before_movers and after_movers and target_object is not None:
                before_distance = min(_cell_distance(obj, target_object) for obj in before_movers)
                after_distance = min(_cell_distance(obj, target_object) for obj in after_movers)
                progress = after_distance - before_distance if option.objective_family == "avoid" else before_distance - after_distance
        supported = (
            goal_activated
            or float(transition.reward) > 0.0
            or progress > 0.0
            or (option.option_type in {"selector_move", "mode_probe_chain", "bind_then_objective"} and schema.action_type in {"click", "select"})
            or (schema.action_type == "move" and bool(moving_signatures))
            or (schema.action_type == "interact" and state_delta >= 0.05)
        )
        contradiction = (
            (option.objective_family == "avoid" and progress < 0.0)
            or (option.objective_family != "avoid" and schema.action_type == "move" and progress < 0.0)
            or (schema.action_type == "interact" and state_delta <= 0.05 and float(transition.reward) <= 0.0 and not goal_activated)
        )
        option.observe(progress=progress, reward=float(transition.reward), supported=supported, contradiction=contradiction)
        self._record_proof(
            proof_type="support" if supported and not contradiction else "contradiction",
            hypothesis_type="option",
            action=transition.action,
            subject=option.option_type,
            relation=option.objective_family or option.mode_key,
            object="then".join(option.action_sequence[:3]),
            confidence=option.evidence.confidence,
            evidence=float(option.evidence.balance),
            predicted="option sequence remains useful",
            observed=f"delta={state_delta:.2f},reward={float(transition.reward):.2f}",
            step_index=transition.next_state.step_index,
            exception=contradiction,
        )
        if contradiction:
            self._clear_active_option()
            return
        if self.active_option_index + 1 < len(option.action_sequence):
            self.active_option_index += 1
            return
        self._clear_active_option()

    def _family_action_separation_score(
        self,
        state: StructuredState,
        action: ActionName,
        mover_signature: ObjectSignature,
        target_object: ObjectState,
        family: str,
    ) -> float:
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        schema = build_action_schema(action, context)
        if family in {"approach", "contact"}:
            score = self._predicted_progress_for_action(state, action, mover_signature, target_object)
            return score + (0.75 if family == "contact" and schema.action_type == "move" else 0.0)
        if family in {"interact", "activate"}:
            if schema.action_type == "interact":
                interaction_target = self._interaction_target_object(state, action, context)
                if interaction_target is not None and object_signature(interaction_target) == object_signature(target_object):
                    return 2.0 if family == "interact" else 2.5
                return -1.0
            movers = _group_by_signature(state.objects).get(mover_signature, [])
            predicted_objects = self._predict_signature_objects(state, action, mover_signature)
            if not movers or not predicted_objects:
                return 0.0
            adjacency_before = min(_adjacency_distance(obj, target_object) for obj in movers)
            adjacency_after = min(_adjacency_distance(obj, target_object) for obj in predicted_objects)
            return (adjacency_before - adjacency_after) + (0.4 if family == "activate" else 0.0)
        if family == "avoid":
            movers = _group_by_signature(state.objects).get(mover_signature, [])
            predicted_objects = self._predict_signature_objects(state, action, mover_signature)
            if not movers or not predicted_objects:
                return 0.0
            before_distance = min(_cell_distance(obj, target_object) for obj in movers)
            after_distance = min(_cell_distance(obj, target_object) for obj in predicted_objects)
            return after_distance - before_distance
        return 0.0

    def _momentum_plan(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        mover_scores: dict[ObjectSignature, float],
        thought: RuntimeThought | None,
    ) -> PlanOutput | None:
        if self.active_goal_anchor is None or self.active_exploit_action is None or self.active_pair_key is None:
            return None
        if self.active_exploit_action not in move_actions:
            self.active_goal_anchor = None
            self.active_exploit_action = None
            self.active_pair_key = None
            self.active_objective_family = None
            return None
        mover_signature = self.active_pair_key[1]
        if mover_signature not in mover_scores:
            self.active_goal_anchor = None
            self.active_exploit_action = None
            self.active_pair_key = None
            self.active_objective_family = None
            return None
        target_object = next((obj for obj in state.objects if object_signature(obj) == self.active_pair_key[2]), None)
        if target_object is not None:
            path_plan = self._path_action_to_object(
                state=state,
                move_actions=move_actions,
                mover_signature=mover_signature,
                target_object=target_object,
                objective=self.objective_hypotheses.get(self.active_pair_key),
            )
            if path_plan is not None:
                action, path_steps = path_plan
                self.active_exploit_action = action
                return PlanOutput(
                    action=action,
                    scores={"momentum": max(1.5 - (0.1 * float(path_steps)), 0.25)},
                    language=LanguageTrace(
                        belief_tokens=("belief", "objective_anchor_active"),
                        question_tokens=self._question_tokens(thought, ("question", "continue_progress")),
                        plan_tokens=("plan", action),
                    ),
                    search_path=(action,),
                )
        movers = _group_by_signature(state.objects).get(mover_signature, [])
        predicted_objects = self._predict_signature_objects(state, self.active_exploit_action, mover_signature)
        if not movers or not predicted_objects:
            self.active_goal_anchor = None
            self.active_exploit_action = None
            self.active_pair_key = None
            self.active_objective_family = None
            return None
        before_distance = min(_anchor_distance(obj, self.active_goal_anchor) for obj in movers)
        after_distance = min(_anchor_distance(obj, self.active_goal_anchor) for obj in predicted_objects)
        if before_distance - after_distance <= 0.0:
            self.active_goal_anchor = None
            self.active_exploit_action = None
            self.active_pair_key = None
            self.active_objective_family = None
            return None
        return PlanOutput(
            action=self.active_exploit_action,
            scores={"momentum": before_distance - after_distance},
            language=LanguageTrace(
                belief_tokens=("belief", "objective_anchor_active"),
                question_tokens=self._question_tokens(thought, ("question", "continue_progress")),
                plan_tokens=("plan", self.active_exploit_action),
            ),
            search_path=(self.active_exploit_action,),
        )

    def _diagnostic_move_plan(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        thought: RuntimeThought | None,
        context,
    ) -> PlanOutput | None:
        if not move_actions:
            return None
        untested_moves = [
            action
            for action in move_actions
            if self.action_visits[(self.current_mode_key, action)] == 0
            and not self._move_action_hits_visible_blocker(state, action, context)
        ]
        if not untested_moves:
            untested_moves = [action for action in move_actions if self.action_visits[(self.current_mode_key, action)] == 0]
        if not untested_moves:
            return None
        action = max(
            untested_moves,
            key=lambda item: (
                self._thought_uncertainty(thought, item),
                self._thought_value(thought, item),
                -self.action_visits[(self.current_mode_key, item)],
            ),
        )
        self.pending_undo = True
        repair_bias = 0.5 if self._repair_pressure() >= 1.3 else 0.0
        return PlanOutput(
            action=action,
            scores={
                "diagnostic": 1.0 + self._thought_uncertainty(thought, action) + repair_bias,
                "untested_move": 1.0,
                "latent_value": self._thought_value(thought, action),
            },
            language=LanguageTrace(
                belief_tokens=("belief", "representation_uncertain") if repair_bias > 0.0 else ("belief", "dynamics_uncertain"),
                question_tokens=self._question_tokens(thought, ("question", "repair_representation" if repair_bias > 0.0 else "test_move_effect")),
                plan_tokens=("plan", action),
            ),
            search_path=(action,),
        )

    def _move_action_hits_visible_blocker(
        self,
        state: StructuredState,
        action: ActionName,
        context,
    ) -> bool:
        schema = build_action_schema(action, context)
        delta = _interaction_delta(schema.direction)
        if schema.action_type != "move" or delta is None:
            return False
        agent_cells = [cell for obj in state.objects if "agent" in obj.tags for cell in obj.cells]
        if not agent_cells:
            return False
        candidate_cells = {(cell[0] + delta[0], cell[1] + delta[1]) for cell in agent_cells}
        for obj in state.objects:
            if "agent" in obj.tags:
                continue
            if not any(cell in candidate_cells for cell in obj.cells):
                continue
            if "target" in obj.tags and "active" in obj.tags:
                return False
            return True
        return False

    def _interaction_probe_plan(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        interact_actions: list[ActionName],
        mover_scores: dict[ObjectSignature, float],
        thought: RuntimeThought | None,
        context,
    ) -> PlanOutput | None:
        if not interact_actions or not mover_scores or self._goal_is_active(state):
            return None
        candidates = self._interaction_probe_candidates(state, mover_scores)
        if not candidates:
            return None
        family_posteriors = self._objective_family_posteriors(state, mover_scores)

        adjacent_options: list[tuple[float, ActionName, ObjectState, ObjectiveHypothesis]] = []
        for action in interact_actions:
            target = self._interaction_target_object(state, action, context)
            if target is None:
                continue
            target_signature = object_signature(target)
            for mover_signature, candidate_object, objective in candidates:
                if object_signature(candidate_object) != target_signature:
                    continue
                posterior = max(
                    family_posteriors.get((mover_signature, target_signature, "interact"), 0.0),
                    family_posteriors.get((mover_signature, target_signature, "activate"), 0.0),
                )
                score = (
                    (2.5 * posterior)
                    + (2.0 / float(objective.interaction_tests + 1))
                    + (0.75 * objective.evidence.uncertainty)
                    + (0.25 * objective.progress_sum)
                    - (1.0 * objective.failed_interactions)
                    + (0.15 * self._thought_uncertainty(thought, action))
                    + (0.1 * self._thought_value(thought, action))
                )
                adjacent_options.append((score, action, target, objective))
        if adjacent_options:
            _, action, target, objective = max(adjacent_options, key=lambda item: item[0])
            target_signature = object_signature(target)
            self._clear_active_probe()
            return PlanOutput(
                action=action,
                scores={
                    "diagnostic": 2.0,
                    "interaction_probe": 1.0,
                    "target_uncertainty": objective.evidence.uncertainty,
                },
                language=LanguageTrace(
                    belief_tokens=("belief", "interaction_target_adjacent", _signature_code(target_signature)),
                    question_tokens=self._question_tokens(thought, ("question", "test_interaction_effect")),
                    plan_tokens=("plan", action),
                ),
                search_path=(action,),
            )

        movers_by_signature = _group_by_signature(state.objects)
        best_action: ActionName | None = None
        best_score = float("-inf")
        best_target_signature: ObjectSignature | None = None
        best_mover_signature: ObjectSignature | None = None
        best_target: ObjectState | None = None
        for mover_signature, target_object, objective in candidates:
            target_signature = object_signature(target_object)
            posterior = max(
                family_posteriors.get((mover_signature, target_signature, "interact"), 0.0),
                family_posteriors.get((mover_signature, target_signature, "activate"), 0.0),
            )
            if posterior <= 0.0:
                continue
            mover_objects = movers_by_signature.get(mover_signature, [])
            if not mover_objects:
                continue
            base_score = (
                (2.5 * posterior)
                + (1.5 / float(objective.interaction_tests + 1))
                + (0.6 * objective.evidence.uncertainty)
                + (0.2 * objective.progress_sum)
                - (1.2 * objective.failed_interactions)
            )
            path_plan = self._path_action_to_interaction_target(
                state=state,
                move_actions=move_actions,
                mover_signature=mover_signature,
                target_object=target_object,
            )
            if path_plan is not None:
                path_action, path_steps = path_plan
                score = (
                    base_score
                    + max(1.5 - (0.1 * float(path_steps)), 0.25)
                    + (0.2 * self._thought_value(thought, path_action))
                    + (0.25 * self._thought_uncertainty(thought, path_action))
                )
                if score > best_score:
                    best_score = score
                    best_action = path_action
                    best_mover_signature = mover_signature
                    best_target_signature = target_signature
                    best_target = target_object
                continue
            for action in move_actions:
                predicted_objects = self._predict_signature_objects(state, action, mover_signature)
                if not predicted_objects:
                    continue
                before_distance = min(_cell_distance(obj, target_object) for obj in mover_objects)
                after_distance = min(_cell_distance(obj, target_object) for obj in predicted_objects)
                progress = before_distance - after_distance
                if progress <= 0.0:
                    continue
                score = (
                    base_score
                    + progress
                    + (0.2 * self._thought_value(thought, action))
                    + (0.25 * self._thought_uncertainty(thought, action))
                )
                if score > best_score:
                    best_score = score
                    best_action = action
                    best_mover_signature = mover_signature
                    best_target_signature = object_signature(target_object)
                    best_target = target_object

        if (
            best_action is None
            or best_target_signature is None
            or best_mover_signature is None
            or best_target is None
            or best_score <= 0.0
        ):
            return None
        self.active_probe_pair_key = (self.current_mode_key, best_mover_signature, best_target_signature)
        self.active_probe_anchor = best_target.centroid
        return PlanOutput(
            action=best_action,
            scores={"diagnostic": best_score, "interaction_probe": 1.0},
            language=LanguageTrace(
                belief_tokens=("belief", "approach_interaction_target", _signature_code(best_target_signature)),
                question_tokens=self._question_tokens(thought, ("question", "move_to_test_interaction")),
                plan_tokens=("plan", best_action),
            ),
            search_path=(best_action,),
        )

    def _probe_momentum_plan(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        interact_actions: list[ActionName],
        mover_scores: dict[ObjectSignature, float],
        thought: RuntimeThought | None,
        context,
    ) -> PlanOutput | None:
        if self.active_probe_pair_key is None or self.active_probe_anchor is None or self._goal_is_active(state):
            return None
        mode_key, mover_signature, target_signature = self.active_probe_pair_key
        if mode_key != self.current_mode_key or mover_signature not in mover_scores:
            self._clear_active_probe()
            return None
        target_object = next((obj for obj in state.objects if object_signature(obj) == target_signature), None)
        if target_object is None:
            self._clear_active_probe()
            return None
        objective = self.objective_hypotheses.get((mode_key, mover_signature, target_signature))
        if objective is not None and (objective.successful_interactions > 0 or objective.failed_interactions > 0):
            self._clear_active_probe()
            return None
        for action in interact_actions:
            interaction_target = self._interaction_target_object(state, action, context)
            if interaction_target is not None and object_signature(interaction_target) == target_signature:
                return PlanOutput(
                    action=action,
                    scores={"diagnostic": 2.0, "interaction_probe": 1.0, "probe_momentum": 1.0},
                    language=LanguageTrace(
                        belief_tokens=("belief", "probe_target_adjacent", _signature_code(target_signature)),
                        question_tokens=self._question_tokens(thought, ("question", "test_interaction_effect")),
                        plan_tokens=("plan", action),
                    ),
                    search_path=(action,),
                )
        movers = _group_by_signature(state.objects).get(mover_signature, [])
        if not movers:
            self._clear_active_probe()
            return None
        path_plan = self._path_action_to_interaction_target(
            state=state,
            move_actions=move_actions,
            mover_signature=mover_signature,
            target_object=target_object,
        )
        if path_plan is not None:
            action, path_steps = path_plan
            return PlanOutput(
                action=action,
                scores={
                    "diagnostic": max(1.5 - (0.1 * float(path_steps)), 0.25),
                    "interaction_probe": 1.0,
                    "probe_momentum": 1.0,
                },
                language=LanguageTrace(
                    belief_tokens=("belief", "probe_target_active", _signature_code(target_signature)),
                    question_tokens=self._question_tokens(thought, ("question", "move_to_test_interaction")),
                    plan_tokens=("plan", action),
                ),
                search_path=(action,),
            )
        best_action: ActionName | None = None
        best_progress = float("-inf")
        for action in move_actions:
            predicted_objects = self._predict_signature_objects(state, action, mover_signature)
            if not predicted_objects:
                continue
            before_distance = min(_cell_distance(obj, target_object) for obj in movers)
            after_distance = min(_cell_distance(obj, target_object) for obj in predicted_objects)
            progress = before_distance - after_distance
            if progress > best_progress:
                best_progress = progress
                best_action = action
        if best_action is None or best_progress <= 0.0:
            self._clear_active_probe()
            return None
        return PlanOutput(
            action=best_action,
            scores={"diagnostic": best_progress, "interaction_probe": 1.0, "probe_momentum": 1.0},
            language=LanguageTrace(
                belief_tokens=("belief", "probe_target_active", _signature_code(target_signature)),
                question_tokens=self._question_tokens(thought, ("question", "move_to_test_interaction")),
                plan_tokens=("plan", best_action),
            ),
            search_path=(best_action,),
        )

    def _disambiguation_move_plan(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        mover_scores: dict[ObjectSignature, float],
        thought: RuntimeThought | None,
    ) -> PlanOutput | None:
        if not move_actions or not mover_scores:
            return None
        family_posteriors = self._objective_family_posteriors(state, mover_scores)
        candidates = sorted(
            self._objective_family_candidates(state, mover_scores),
            key=lambda item: family_posteriors.get((item[0], object_signature(item[1]), item[2]), 0.0),
            reverse=True,
        )
        if len(candidates) < 2:
            return None
        candidates = candidates[:4]

        best_action: ActionName | None = None
        best_score = float("-inf")
        for action in move_actions:
            action_progress: list[tuple[float, float]] = []
            for mover_signature, target_object, family, _objective, _family_hypothesis in candidates:
                posterior = family_posteriors.get((mover_signature, object_signature(target_object), family), 0.0)
                progress = self._family_action_separation_score(state, action, mover_signature, target_object, family)
                action_progress.append((progress, posterior))
            if len(action_progress) < 2:
                continue
            separation = 0.0
            pair_count = 0
            weighted_mean = sum(progress * posterior for progress, posterior in action_progress) / max(
                sum(posterior for _, posterior in action_progress),
                1e-6,
            )
            for index, (progress_a, posterior_a) in enumerate(action_progress):
                for progress_b, posterior_b in action_progress[index + 1 :]:
                    separation += (posterior_a * posterior_b) * abs(progress_a - progress_b)
                    pair_count += 1
            if pair_count <= 0:
                continue
            separation /= float(pair_count)
            variance = sum(
                posterior * ((progress - weighted_mean) ** 2)
                for progress, posterior in action_progress
            )
            exploration = 1.0 / math.sqrt(self.action_visits[(self.current_mode_key, action)] + 1.0)
            score = (
                (1.5 * separation)
                + variance
                + (0.15 * abs(weighted_mean))
                + exploration
                + (0.4 * self._thought_uncertainty(thought, action))
                + (0.2 * self._thought_value(thought, action))
            )
            if score > best_score:
                best_score = score
                best_action = action

        if best_action is None or best_score <= 0.0:
            return None
        return PlanOutput(
            action=best_action,
            scores={"diagnostic": best_score, "hypothesis_variance": best_score},
            language=LanguageTrace(
                belief_tokens=("belief", "objective_competition"),
                question_tokens=self._question_tokens(thought, ("question", "disambiguate_objective")),
                plan_tokens=("plan", best_action),
            ),
            search_path=(best_action,),
        )

    def _selector_probe_plan(
        self,
        state: StructuredState,
        selector_actions: list[ActionName],
        context,
        mover_scores: dict[ObjectSignature, float],
        thought: RuntimeThought | None,
    ) -> PlanOutput | None:
        if not selector_actions:
            return None
        move_actions = [action for action in state.affordances if build_action_schema(action, context).action_type == "move"]
        if any(self.action_visits[(self.current_mode_key, action)] == 0 for action in move_actions):
            return None
        if mover_scores and self._objective_candidates(state, mover_scores):
            return None

        selector_attempts = sum(self.selector_visits.values())
        mode_failures = sum(
            1
            for hypothesis in self.mode_hypotheses.values()
            if hypothesis.entries > 0 and hypothesis.move_effect_steps == 0 and hypothesis.utility <= 0.0
        )
        grouped: dict[str, list[ActionName]] = defaultdict(list)
        for action in selector_actions:
            grouped[_mode_key(action, build_action_schema(action, context))].append(action)
        mode_posteriors = self._mode_posteriors(selector_actions, context, thought)

        best_mode: str | None = None
        best_score = float("-inf")
        best_followup = float("-inf")
        for mode_key, actions in grouped.items():
            hypothesis = self.mode_hypotheses.get(mode_key)
            visits = 0 if hypothesis is None else hypothesis.entries
            utility = 0.0 if hypothesis is None else hypothesis.utility
            uncertainty = 1.0 if hypothesis is None else hypothesis.evidence.uncertainty
            exploration = 1.0 / math.sqrt(visits + 1.0)
            latent_followup = max(self._thought_selector_followup(thought, action) for action in actions)
            latent_uncertainty = max(self._thought_uncertainty(thought, action) for action in actions)
            latent_value = max(self._thought_value(thought, action) for action in actions)
            posterior = mode_posteriors.get(mode_key, 0.0)
            score = (
                exploration
                + posterior
                + uncertainty
                + utility
                + max(latent_followup, 0.0)
                + (0.35 * latent_uncertainty)
                + (0.15 * latent_value)
            )
            if score > best_score:
                best_score = score
                best_mode = mode_key
                best_followup = latent_followup

        best_hypothesis = None if best_mode is None else self.mode_hypotheses.get(best_mode)
        if (
            selector_attempts >= max(2, len(grouped))
            and mode_failures >= 1
            and best_followup < 1.0
        ):
            return None
        if best_hypothesis is not None and best_hypothesis.entries >= 2 and best_followup <= 0.25:
            return None

        if best_mode is None or best_score <= 0.75:
            return None
        action = max(
            grouped[best_mode],
            key=lambda item: (
                self._thought_selector_followup(thought, item),
                self._thought_uncertainty(thought, item),
                -self.selector_visits[item],
            ),
        )
        return PlanOutput(
            action=action,
            scores={
                "diagnostic": best_score,
                "selector_probe": 1.0,
                "selector_followup": self._thought_selector_followup(thought, action),
                "mode_posterior": mode_posteriors.get(best_mode, 0.0),
            },
            language=LanguageTrace(
                belief_tokens=("belief", "mode_uncertain"),
                question_tokens=self._question_tokens(thought, ("question", "test_control_mode")),
                plan_tokens=("plan", action),
            ),
            search_path=(action,),
        )

    def _mover_scores(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
    ) -> dict[ObjectSignature, float]:
        posteriors = self._control_posteriors(state, move_actions)
        scores: dict[ObjectSignature, float] = {}
        for obj in state.objects:
            signature = object_signature(obj)
            control = self.control_hypotheses.get((self.current_mode_key, signature))
            best_confidence = 0.0
            best_magnitude = 0.0
            for action in move_actions:
                rule = self._motion_rule_for_object(action, obj)
                if rule is None:
                    continue
                best_confidence = max(best_confidence, rule.evidence.confidence)
                best_magnitude = max(best_magnitude, rule.mean_magnitude)
            control_utility = 0.0 if control is None else control.utility
            posterior = posteriors.get(signature, 0.0)
            score = posterior * (1.0 + max(control_utility, 0.0) + (0.8 * best_confidence) + (0.5 * best_magnitude))
            if score >= 0.12 or posterior >= 0.18 or (best_confidence >= 0.55 and best_magnitude >= 0.5):
                scores[signature] = score
        return scores

    def _objective_candidates(
        self,
        state: StructuredState,
        mover_scores: dict[ObjectSignature, float],
    ) -> list[tuple[ObjectSignature, ObjectState, ObjectiveHypothesis]]:
        candidates: list[tuple[ObjectSignature, ObjectState, ObjectiveHypothesis]] = []
        objective_posteriors = self._objective_posteriors(state, mover_scores)
        mover_signatures = set(mover_scores)
        unresolved_interaction_targets = self._has_unresolved_interaction_targets(state, mover_scores)
        for target_object in state.objects:
            if not self._is_candidate_target_object(target_object, mover_signatures, state.grid_shape):
                continue
            if "target" in target_object.tags and not self._goal_is_active(state) and unresolved_interaction_targets:
                continue
            if self._goal_is_active(state) and "interactable" in target_object.tags and "target" not in target_object.tags:
                continue
            target_signature = object_signature(target_object)
            for mover_signature in mover_scores:
                objective = self.objective_hypotheses.get((self.current_mode_key, mover_signature, target_signature))
                if objective is None:
                    objective = ObjectiveHypothesis(evidence=EvidenceCounter())
                if objective_posteriors.get((mover_signature, target_signature), 0.0) < 0.05 and objective.utility <= 1.0:
                    continue
                objective = self._ensure_objective_hypothesis(mover_signature, target_signature)
                if self._requires_interaction_confirmation(state, target_object, objective):
                    continue
                candidates.append((mover_signature, target_object, objective))
        return candidates

    def _observe_representation_repairs(
        self,
        *,
        transition: Transition,
        before_groups: dict[ObjectSignature, list[ObjectState]],
        after_groups: dict[ObjectSignature, list[ObjectState]],
        signature_groups: dict[ObjectSignature, list[tuple[ObjectState, ObjectState]]],
    ) -> None:
        for signature in before_groups.keys() | after_groups.keys():
            before_count = len(before_groups.get(signature, ()))
            after_count = len(after_groups.get(signature, ()))
            if before_count != after_count:
                repair_type = "split" if after_count > before_count else "merge"
                repair = self.repair_hypotheses.setdefault((repair_type, signature), RepairHypothesis(evidence=EvidenceCounter()))
                repair.observe(
                    supported=True,
                    reason=f"{before_count}->{after_count}",
                )
                self._record_proof(
                    proof_type="support",
                    hypothesis_type="representation",
                    action=transition.action,
                    subject=_signature_code(signature),
                    relation=repair_type,
                    object="count_shift",
                    confidence=repair.evidence.confidence,
                    evidence=float(repair.evidence.balance),
                    predicted=str(before_count),
                    observed=str(after_count),
                    step_index=transition.next_state.step_index,
                    exception=True,
                )
            before_group = before_groups.get(signature, [])
            after_group = after_groups.get(signature, [])
            if len(before_group) >= 2 and len(after_group) >= 2:
                sorted_cost = _group_alignment_cost(
                    sorted(before_group, key=_sort_key),
                    sorted(after_group, key=_sort_key),
                )
                matched_cost = _group_pair_cost(signature_groups.get(signature, ()))
                if sorted_cost - matched_cost >= 1.5:
                    repair = self.repair_hypotheses.setdefault(("rebind", signature), RepairHypothesis(evidence=EvidenceCounter()))
                    repair.observe(
                        supported=True,
                        reason=f"{sorted_cost:.2f}->{matched_cost:.2f}",
                    )
                    self._record_proof(
                        proof_type="support",
                        hypothesis_type="representation",
                        action=transition.action,
                        subject=_signature_code(signature),
                        relation="rebind",
                        object="nearest_motion",
                        confidence=repair.evidence.confidence,
                        evidence=float(repair.evidence.balance),
                        predicted=f"sorted={sorted_cost:.2f}",
                        observed=f"nearest={matched_cost:.2f}",
                        step_index=transition.next_state.step_index,
                        exception=True,
                    )

    def _interaction_probe_candidates(
        self,
        state: StructuredState,
        mover_scores: dict[ObjectSignature, float],
    ) -> list[tuple[ObjectSignature, ObjectState, ObjectiveHypothesis]]:
        candidates: list[tuple[ObjectSignature, ObjectState, ObjectiveHypothesis]] = []
        mover_signatures = set(mover_scores)
        for target_object in state.objects:
            if not self._is_interaction_candidate_object(target_object, mover_signatures, state.grid_shape):
                continue
            target_signature = object_signature(target_object)
            for mover_signature in mover_scores:
                objective = self.objective_hypotheses.setdefault(
                    (self.current_mode_key, mover_signature, target_signature),
                    ObjectiveHypothesis(evidence=EvidenceCounter()),
                )
                if objective.successful_interactions > 0 and self._goal_is_active(state):
                    continue
                if objective.failed_interactions >= 1 and objective.successful_interactions == 0:
                    continue
                candidates.append((mover_signature, target_object, objective))
        candidates.sort(
            key=lambda item: (
                item[2].successful_interactions,
                -item[2].failed_interactions,
                -item[2].interaction_tests,
                item[2].utility,
            ),
            reverse=True,
        )
        return candidates

    def _has_unresolved_interaction_targets(
        self,
        state: StructuredState,
        mover_scores: dict[ObjectSignature, float],
    ) -> bool:
        if not any("interact" in action for action in state.affordances):
            return False
        mover_signatures = set(mover_scores)
        for target_object in state.objects:
            if not self._is_interaction_candidate_object(target_object, mover_signatures, state.grid_shape):
                continue
            target_signature = object_signature(target_object)
            for mover_signature in mover_scores:
                objective = self.objective_hypotheses.get((self.current_mode_key, mover_signature, target_signature))
                if objective is None or objective.successful_interactions == 0:
                    return True
        return False

    def _requires_interaction_confirmation(
        self,
        state: StructuredState,
        target_object: ObjectState,
        objective: ObjectiveHypothesis,
    ) -> bool:
        if self._goal_is_active(state):
            return False
        if not self._is_interaction_candidate_object(target_object, set(), state.grid_shape):
            return False
        return objective.successful_interactions == 0 and objective.goal_activations == 0 and objective.reward_hits == 0

    def _observe_interaction_objective_evidence(
        self,
        transition: Transition,
        action_mode_key: str,
        context,
        goal_activated: bool,
        state_delta: float,
        moving_signatures: set[ObjectSignature],
    ) -> None:
        target_object = self._interaction_target_object(transition.state, transition.action, context)
        move_actions = [
            action
            for action in transition.state.affordances
            if build_action_schema(action, context).action_type == "move"
        ]
        candidate_movers = set(moving_signatures) or set(self._mover_scores(transition.state, move_actions))
        if target_object is None:
            if self.active_pair_key is not None and self.active_pair_key[0] == action_mode_key:
                active_objective = self.objective_hypotheses.get(self.active_pair_key)
                if active_objective is not None:
                    active_objective.failed_interactions += 1
                    active_objective.evidence.contradiction += 1
                    self._record_proof(
                        proof_type="contradiction",
                        hypothesis_type="objective",
                        action=transition.action,
                        subject=_signature_code(self.active_pair_key[1]),
                        relation="toward",
                        object=_signature_code(self.active_pair_key[2]),
                        confidence=active_objective.evidence.confidence,
                        evidence=float(active_objective.evidence.balance),
                        predicted="interaction targets candidate object",
                        observed="no aligned target object",
                        step_index=transition.next_state.step_index,
                        exception=True,
                    )
                self._clear_active_exploit()
            return
        target_signature = object_signature(target_object)
        for mover_signature in candidate_movers:
            objective = self.objective_hypotheses.setdefault(
                (action_mode_key, mover_signature, target_signature),
                ObjectiveHypothesis(evidence=EvidenceCounter()),
            )
            objective.observe(
                action=transition.action,
                progress=0.0,
                reward=float(transition.reward),
                contact=True,
                direct_interaction=True,
                state_delta=state_delta,
                goal_activated=goal_activated,
            )
            for family in self._candidate_objective_families(
                transition.next_state,
                mover_signature,
                target_object,
                objective,
            ):
                family_hypothesis = self._ensure_objective_family_hypothesis(
                    mover_signature,
                    target_signature,
                    family,
                    mode_key=action_mode_key,
                )
                family_hypothesis.observe(
                    action=transition.action,
                    progress=0.0,
                    reward=float(transition.reward),
                    contact=True,
                    direct_interaction=True,
                    state_delta=state_delta,
                    goal_activated=goal_activated,
                )
            self._record_proof(
                proof_type="support" if objective.successful_interactions > 0 else "contradiction" if objective.failed_interactions > 0 else "support",
                hypothesis_type="objective",
                action=transition.action,
                subject=_signature_code(mover_signature),
                relation="interacts_with",
                object=_signature_code(target_signature),
                confidence=objective.evidence.confidence,
                evidence=float(objective.evidence.balance),
                predicted="interaction changes mechanic",
                observed=f"delta={state_delta:.2f},goal={int(goal_activated)}",
                step_index=transition.next_state.step_index,
                exception=objective.failed_interactions > 0 and objective.successful_interactions == 0,
            )
            if objective.failed_interactions > 0 and objective.successful_interactions == 0:
                self._clear_active_exploit_for_target(action_mode_key, target_signature)
                self._clear_active_probe_for_target(action_mode_key, target_signature)
            if objective.successful_interactions > 0:
                self._clear_active_probe_for_target(action_mode_key, target_signature)

    def _seed_goal_target_objectives(
        self,
        state: StructuredState,
        action_mode_key: str,
        mover_signatures: set[ObjectSignature],
    ) -> None:
        if not mover_signatures:
            return
        goal_targets = [obj for obj in state.objects if "target" in obj.tags]
        for target_object in goal_targets:
            target_signature = object_signature(target_object)
            for mover_signature in mover_signatures:
                objective = self.objective_hypotheses.setdefault(
                    (action_mode_key, mover_signature, target_signature),
                    ObjectiveHypothesis(evidence=EvidenceCounter()),
                )
                objective.goal_activations += 1
                objective.evidence.support += 2
                objective.supporting_actions.add("goal_active")
                activate_hypothesis = self._ensure_objective_family_hypothesis(
                    mover_signature,
                    target_signature,
                    "activate",
                    mode_key=action_mode_key,
                )
                activate_hypothesis.observe(
                    action="goal_active",
                    progress=0.0,
                    reward=1.0,
                    contact=True,
                    direct_interaction=False,
                    state_delta=1.0,
                    goal_activated=True,
                )

    def _interaction_target_object(self, state: StructuredState, action: ActionName, context) -> ObjectState | None:
        schema = build_action_schema(action, context)
        if schema.action_type == "interact":
            delta = _interaction_delta(schema.direction)
            if delta is None:
                return None
            agent_cells = [cell for obj in state.objects if "agent" in obj.tags for cell in obj.cells]
            if not agent_cells:
                return None
            target_cells = {(cell[0] + delta[0], cell[1] + delta[1]) for cell in agent_cells}
            for obj in state.objects:
                if "agent" in obj.tags:
                    continue
                if any(cell in target_cells for cell in obj.cells):
                    return obj
            return None
        if schema.action_type == "click" and schema.click is not None:
            click_x, click_y = schema.click
            for obj in state.objects:
                if (click_y, click_x) in obj.cells:
                    return obj
        return None

    def _path_action_to_object(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        mover_signature: ObjectSignature,
        target_object: ObjectState,
        objective: ObjectiveHypothesis | None,
    ) -> tuple[ActionName, int] | None:
        if "target" in target_object.tags or "repaired_goal" in target_object.tags:
            return self._path_action_to_cells(
                state=state,
                move_actions=move_actions,
                mover_signature=mover_signature,
                goal_cells=set(target_object.cells),
            )
        if objective is not None and objective.successful_interactions > 0 and (
            "interactable" in target_object.tags or "repaired_control" in target_object.tags
        ):
            return self._path_action_to_interaction_target(
                state=state,
                move_actions=move_actions,
                mover_signature=mover_signature,
                target_object=target_object,
            )
        return None

    def _path_action_to_interaction_target(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        mover_signature: ObjectSignature,
        target_object: ObjectState,
    ) -> tuple[ActionName, int] | None:
        movers = _group_by_signature(state.objects).get(mover_signature, [])
        if len(movers) != 1 or movers[0].area != 1:
            return None
        mover = movers[0]
        adjacent_cells: set[tuple[int, int]] = set()
        blocked_cells = _blocked_cells(state, mover)
        for cell in target_object.cells:
            for neighbor in _adjacent_cells(cell):
                if neighbor in blocked_cells and neighbor != mover.cells[0]:
                    continue
                adjacent_cells.add(neighbor)
        if not adjacent_cells:
            return None
        return self._path_action_to_cells(
            state=state,
            move_actions=move_actions,
            mover_signature=mover_signature,
            goal_cells=adjacent_cells,
        )

    def _path_sequence_to_cells(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        mover_signature: ObjectSignature,
        goal_cells: set[tuple[int, int]],
    ) -> tuple[ActionName, ...] | None:
        movers = _group_by_signature(state.objects).get(mover_signature, [])
        if len(movers) != 1 or movers[0].area != 1:
            return None
        mover = movers[0]
        start = mover.cells[0]
        move_edges: dict[ActionName, tuple[int, int]] = {}
        for action in move_actions:
            predicted_objects = self._predict_signature_objects(state, action, mover_signature)
            if len(predicted_objects) != 1 or predicted_objects[0].area != 1:
                continue
            next_cell = predicted_objects[0].cells[0]
            if next_cell == start:
                continue
            move_edges[action] = next_cell
        if not move_edges:
            return None

        blocked_cells = _blocked_cells(state, mover)
        frontier: list[tuple[int, int]] = [start]
        parent: dict[tuple[int, int], tuple[tuple[int, int], ActionName] | None] = {start: None}
        while frontier:
            cell = frontier.pop(0)
            if cell in goal_cells:
                return _reconstruct_action_sequence(cell, parent)
            for action, next_cell in _translated_edges(cell, start, move_edges).items():
                if next_cell in parent:
                    continue
                if not _in_bounds(next_cell, state.grid_shape):
                    continue
                if next_cell in blocked_cells and next_cell not in goal_cells:
                    continue
                parent[next_cell] = (cell, action)
                frontier.append(next_cell)
        return None

    def _path_sequence_to_interaction_target(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        interact_actions: list[ActionName],
        mover_signature: ObjectSignature,
        target_object: ObjectState,
    ) -> tuple[ActionName, ...] | None:
        movers = _group_by_signature(state.objects).get(mover_signature, [])
        if len(movers) != 1 or movers[0].area != 1:
            return None
        mover = movers[0]
        adjacent_cells: set[tuple[int, int]] = set()
        blocked_cells = _blocked_cells(state, mover)
        for cell in target_object.cells:
            for neighbor in _adjacent_cells(cell):
                if neighbor in blocked_cells and neighbor != mover.cells[0]:
                    continue
                adjacent_cells.add(neighbor)
        if not adjacent_cells:
            return None
        path_sequence = self._path_sequence_to_cells(
            state=state,
            move_actions=move_actions,
            mover_signature=mover_signature,
            goal_cells=adjacent_cells,
        )
        if path_sequence is None:
            return None
        endpoint = _advance_cell(mover.cells[0], path_sequence)
        interact_action = _interaction_action_for_endpoint(endpoint, target_object, interact_actions)
        if interact_action is None:
            return None
        return path_sequence + (interact_action,)

    def _path_action_to_cells(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        mover_signature: ObjectSignature,
        goal_cells: set[tuple[int, int]],
    ) -> tuple[ActionName, int] | None:
        path_sequence = self._path_sequence_to_cells(state, move_actions, mover_signature, goal_cells)
        if not path_sequence:
            return None
        return (path_sequence[0], len(path_sequence))

    def _goal_is_active(self, state: StructuredState) -> bool:
        if state.flags_dict().get("goal_active") == "1":
            return True
        return any(("target" in obj.tags or "repaired_goal" in obj.tags) and "active" in obj.tags for obj in state.objects)

    def _is_interaction_candidate_object(
        self,
        obj: ObjectState,
        moving_signatures: set[ObjectSignature],
        grid_shape: tuple[int, int],
    ) -> bool:
        signature = object_signature(obj)
        if signature in moving_signatures:
            return False
        if "agent" in obj.tags or "blocking" in obj.tags:
            return False
        if any(tag in obj.tags for tag in ("interactable", "repaired_control", "clickable", "interface_target", "selector_candidate")):
            return True
        if "target" in obj.tags or "repaired_goal" in obj.tags:
            return False
        return self._is_candidate_target_object(obj, moving_signatures, grid_shape) and obj.area <= 4

    def _clear_active_exploit(self) -> None:
        self.active_goal_anchor = None
        self.active_exploit_action = None
        self.active_pair_key = None
        self.active_objective_family = None
        self.exploit_started = False

    def _clear_active_exploit_for_target(
        self,
        mode_key: str,
        target_signature: ObjectSignature,
    ) -> None:
        if self.active_pair_key is None:
            return
        if self.active_pair_key[0] == mode_key and self.active_pair_key[2] == target_signature:
            self._clear_active_exploit()
        if (
            self.active_option_key is not None
            and self.active_option_key[1] == mode_key
            and self.active_option_key[3] == target_signature
        ):
            self._clear_active_option()

    def _clear_active_probe(self) -> None:
        self.active_probe_anchor = None
        self.active_probe_pair_key = None

    def _clear_active_probe_for_target(
        self,
        mode_key: str,
        target_signature: ObjectSignature,
    ) -> None:
        if self.active_probe_pair_key is None:
            return
        if self.active_probe_pair_key[0] == mode_key and self.active_probe_pair_key[2] == target_signature:
            self._clear_active_probe()

    def _clear_active_option(self) -> None:
        self.active_option_key = None
        self.active_option_index = 0

    def _thought_value(self, thought: RuntimeThought | None, action: ActionName) -> float:
        return 0.0 if thought is None else thought.value_for(action)

    def _thought_uncertainty(self, thought: RuntimeThought | None, action: ActionName) -> float:
        return 0.0 if thought is None else thought.uncertainty_for(action)

    def _thought_policy(self, thought: RuntimeThought | None, action: ActionName) -> float:
        return 0.0 if thought is None else thought.policy_for(action)

    def _thought_selector_followup(self, thought: RuntimeThought | None, action: ActionName) -> float:
        return 0.0 if thought is None else thought.selector_followup_for(action)

    def _question_tokens(
        self,
        thought: RuntimeThought | None,
        fallback: tuple[str, ...],
    ) -> tuple[str, ...]:
        if thought is not None and thought.question_tokens:
            thought_specific = sum(token not in GENERIC_LANGUAGE_TOKENS for token in thought.question_tokens)
            fallback_specific = sum(token not in GENERIC_LANGUAGE_TOKENS for token in fallback)
            if fallback_specific > thought_specific:
                return fallback
            return thought.question_tokens
        return fallback

    def _predict_signature_objects(
        self,
        state: StructuredState,
        action: ActionName,
        mover_signature: ObjectSignature,
    ) -> list[ObjectState]:
        predicted: list[ObjectState] = []
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        schema = build_action_schema(action, context)
        for obj in _group_by_signature(state.objects).get(mover_signature, []):
            rule = self._motion_rule_for_object(action, obj)
            shift_delta: tuple[float, float] | None = None
            if rule is not None and rule.evidence.confidence >= 0.55:
                shift_delta = rule.mean_delta
            elif obj.area == 1 and schema.action_type == "move":
                direction_delta = _interaction_delta(schema.direction)
                if direction_delta is not None:
                    shift_delta = (float(direction_delta[0]), float(direction_delta[1]))
            if shift_delta is None:
                continue
            shifted = _shift_object(obj, shift_delta, state.grid_shape)
            if _blocked_shift(state, obj, shifted):
                predicted.append(obj)
            else:
                predicted.append(shifted)
        return predicted

    def _motion_rule_for_object(self, action: ActionName, obj: ObjectState) -> MotionRule | None:
        exact_rule = self.motion_rules.get((self.current_mode_key, action, object_signature(obj)))
        if exact_rule is not None:
            return exact_rule
        obj_height = (obj.bbox[2] - obj.bbox[0]) + 1
        obj_width = (obj.bbox[3] - obj.bbox[1]) + 1
        best_rule: MotionRule | None = None
        best_score = float("-inf")
        for (mode_key, candidate_action, signature), rule in self.motion_rules.items():
            if mode_key != self.current_mode_key or candidate_action != action:
                continue
            if signature[0] != obj.color:
                continue
            area_delta = abs(signature[1] - obj.area)
            height_delta = abs(signature[2] - obj_height)
            width_delta = abs(signature[3] - obj_width)
            if area_delta > 2 or height_delta > 1 or width_delta > 1:
                continue
            score = rule.evidence.confidence - (0.1 * area_delta) - (0.1 * height_delta) - (0.1 * width_delta)
            if score > best_score:
                best_score = score
                best_rule = rule
        return best_rule

    def _is_candidate_target_object(
        self,
        obj: ObjectState,
        moving_signatures: set[ObjectSignature],
        grid_shape: tuple[int, int],
    ) -> bool:
        signature = object_signature(obj)
        if signature in moving_signatures:
            return False
        grid_area = grid_shape[0] * grid_shape[1]
        if obj.area >= max(int(0.45 * grid_area), 36):
            return False
        if any(tag in obj.tags for tag in ("repaired_goal", "target", "clickable", "interface_target", "selector_candidate")):
            return True
        height = (obj.bbox[2] - obj.bbox[0]) + 1
        width = (obj.bbox[3] - obj.bbox[1]) + 1
        touches_top = obj.bbox[0] == 0
        touches_bottom = obj.bbox[2] == grid_shape[0] - 1
        touches_left = obj.bbox[1] == 0
        touches_right = obj.bbox[3] == grid_shape[1] - 1
        spans_vertical = touches_top and touches_bottom
        spans_horizontal = touches_left and touches_right
        if spans_vertical or spans_horizontal:
            long_side = max(height, width)
            if long_side >= int(0.6 * max(grid_shape)):
                return False
        return True


def _mode_key(action: ActionName, schema) -> str:
    if schema.click is not None:
        return f"{schema.family}:{schema.coarse_bin}"
    return f"{schema.family}:{action}"


def _state_delta(before: StructuredState, after: StructuredState) -> float:
    vector_before = before.summary_vector()
    vector_after = after.summary_vector()
    return float(sum(abs(float(a) - float(b)) for a, b in zip(vector_before, vector_after)))


def _goal_activated(before: StructuredState, after: StructuredState) -> bool:
    before_active = before.flags_dict().get("goal_active") == "1" or any(
        "target" in obj.tags and "active" in obj.tags for obj in before.objects
    )
    after_active = after.flags_dict().get("goal_active") == "1" or any(
        "target" in obj.tags and "active" in obj.tags for obj in after.objects
    )
    return after_active and not before_active


def _interaction_delta(direction: str | None) -> tuple[int, int] | None:
    if direction == "up":
        return (-1, 0)
    if direction == "down":
        return (1, 0)
    if direction == "left":
        return (0, -1)
    if direction == "right":
        return (0, 1)
    return None


def _match_objects(
    before: StructuredState,
    after: StructuredState,
) -> list[tuple[ObjectSignature, ObjectState, ObjectState]]:
    before_groups = _group_by_signature(before.objects)
    after_groups = _group_by_signature(after.objects)
    matches: list[tuple[ObjectSignature, ObjectState, ObjectState]] = []
    for signature in before_groups.keys() & after_groups.keys():
        before_group = sorted(before_groups[signature], key=_sort_key)
        unmatched_after = sorted(after_groups[signature], key=_sort_key)
        for before_obj in before_group:
            if not unmatched_after:
                break
            best_index = min(
                range(len(unmatched_after)),
                key=lambda index: _object_match_cost(before_obj, unmatched_after[index]),
            )
            matches.append((signature, before_obj, unmatched_after.pop(best_index)))
    return matches


def _group_by_signature(objects: tuple[ObjectState, ...]) -> dict[ObjectSignature, list[ObjectState]]:
    groups: dict[ObjectSignature, list[ObjectState]] = defaultdict(list)
    for obj in objects:
        groups[object_signature(obj)].append(obj)
    return groups


def _signatures_in_state(state: StructuredState) -> set[ObjectSignature]:
    return {object_signature(obj) for obj in state.objects}


def _sort_key(obj: ObjectState) -> tuple[float, float, str]:
    return (obj.centroid[0], obj.centroid[1], obj.object_id)


def _object_match_cost(source: ObjectState, target: ObjectState) -> float:
    return (
        abs(source.centroid[0] - target.centroid[0])
        + abs(source.centroid[1] - target.centroid[1])
        + (0.5 * abs(source.area - target.area))
    )


def _group_alignment_cost(
    before_group: list[ObjectState],
    after_group: list[ObjectState],
) -> float:
    pair_count = min(len(before_group), len(after_group))
    if pair_count <= 0:
        return 0.0
    return sum(_object_match_cost(before_group[index], after_group[index]) for index in range(pair_count)) / float(pair_count)


def _group_pair_cost(pairs: list[tuple[ObjectState, ObjectState]] | tuple[tuple[ObjectState, ObjectState], ...]) -> float:
    if not pairs:
        return 0.0
    return sum(_object_match_cost(before_obj, after_obj) for before_obj, after_obj in pairs) / float(len(pairs))


def _shift_object(
    obj: ObjectState,
    delta: tuple[float, float],
    grid_shape: tuple[int, int],
) -> ObjectState:
    dy = int(round(delta[0]))
    dx = int(round(delta[1]))
    height, width = grid_shape
    shifted_cells = []
    for y, x in obj.cells:
        shifted_y = min(max(y + dy, 0), height - 1)
        shifted_x = min(max(x + dx, 0), width - 1)
        shifted_cells.append((shifted_y, shifted_x))
    ys = [cell[0] for cell in shifted_cells]
    xs = [cell[1] for cell in shifted_cells]
    return ObjectState(
        object_id=obj.object_id,
        color=obj.color,
        cells=tuple(shifted_cells),
        bbox=(min(ys), min(xs), max(ys), max(xs)),
        centroid=(float(sum(ys)) / len(ys), float(sum(xs)) / len(xs)),
        area=obj.area,
        tags=obj.tags,
    )


def _blocked_shift(
    state: StructuredState,
    source: ObjectState,
    shifted: ObjectState,
) -> bool:
    shifted_cells = set(shifted.cells)
    source_cells = set(source.cells)
    for other in state.objects:
        if other.object_id == source.object_id:
            continue
        overlap = shifted_cells & set(other.cells)
        if not overlap:
            continue
        if "target" in other.tags and "active" in other.tags:
            continue
        if overlap <= source_cells:
            continue
        return True
    return False


def _blocked_cells(state: StructuredState, mover: ObjectState) -> set[tuple[int, int]]:
    blocked: set[tuple[int, int]] = set()
    for obj in state.objects:
        if obj.object_id == mover.object_id:
            continue
        blocked.update(obj.cells)
    return blocked


def _adjacent_cells(cell: tuple[int, int]) -> tuple[tuple[int, int], ...]:
    return (
        (cell[0] - 1, cell[1]),
        (cell[0] + 1, cell[1]),
        (cell[0], cell[1] - 1),
        (cell[0], cell[1] + 1),
    )


def _translated_edges(
    cell: tuple[int, int],
    start: tuple[int, int],
    move_edges: dict[ActionName, tuple[int, int]],
) -> dict[ActionName, tuple[int, int]]:
    translated: dict[ActionName, tuple[int, int]] = {}
    for action, absolute_next in move_edges.items():
        delta = (absolute_next[0] - start[0], absolute_next[1] - start[1])
        translated[action] = (cell[0] + delta[0], cell[1] + delta[1])
    return translated


def _reconstruct_first_action(
    goal: tuple[int, int],
    parent: dict[tuple[int, int], tuple[tuple[int, int], ActionName] | None],
) -> tuple[ActionName, int] | None:
    path_actions = _reconstruct_action_sequence(goal, parent)
    if not path_actions:
        return None
    return (path_actions[0], len(path_actions))


def _reconstruct_action_sequence(
    goal: tuple[int, int],
    parent: dict[tuple[int, int], tuple[tuple[int, int], ActionName] | None],
) -> tuple[ActionName, ...]:
    path_actions: list[ActionName] = []
    current = goal
    while parent[current] is not None:
        prev, action = parent[current]
        path_actions.append(action)
        current = prev
    path_actions.reverse()
    return tuple(path_actions)


def _in_bounds(cell: tuple[int, int], grid_shape: tuple[int, int]) -> bool:
    return 0 <= cell[0] < grid_shape[0] and 0 <= cell[1] < grid_shape[1]


def _cell_distance(source: ObjectState, target: ObjectState) -> float:
    return min(
        float(abs(source_cell[0] - target_cell[0]) + abs(source_cell[1] - target_cell[1]))
        for source_cell in source.cells
        for target_cell in target.cells
    )


def _adjacency_distance(source: ObjectState, target: ObjectState) -> float:
    adjacent_cells = {neighbor for cell in target.cells for neighbor in _adjacent_cells(cell)}
    if not adjacent_cells:
        return _cell_distance(source, target)
    return min(
        float(abs(source_cell[0] - target_cell[0]) + abs(source_cell[1] - target_cell[1]))
        for source_cell in source.cells
        for target_cell in adjacent_cells
    )


def _overlap_fraction(source: ObjectState, target: ObjectState) -> float:
    source_cells = set(source.cells)
    target_cells = set(target.cells)
    if not target_cells:
        return 0.0
    return float(len(source_cells & target_cells)) / float(len(target_cells))


def _advance_cell(start: tuple[int, int], sequence: tuple[ActionName, ...]) -> tuple[int, int]:
    current = start
    for action in sequence:
        if action == "up":
            current = (current[0] - 1, current[1])
        elif action == "down":
            current = (current[0] + 1, current[1])
        elif action == "left":
            current = (current[0], current[1] - 1)
        elif action == "right":
            current = (current[0], current[1] + 1)
    return current


def _interaction_action_for_endpoint(
    endpoint: tuple[int, int],
    target_object: ObjectState,
    interact_actions: list[ActionName],
) -> ActionName | None:
    for cell in target_object.cells:
        delta = (cell[0] - endpoint[0], cell[1] - endpoint[1])
        if delta == (-1, 0) and "interact_up" in interact_actions:
            return "interact_up"
        if delta == (1, 0) and "interact_down" in interact_actions:
            return "interact_down"
        if delta == (0, -1) and "interact_left" in interact_actions:
            return "interact_left"
        if delta == (0, 1) and "interact_right" in interact_actions:
            return "interact_right"
    return None


def _anchor_distance(source: ObjectState, anchor: tuple[float, float]) -> float:
    return float(abs(source.centroid[0] - anchor[0]) + abs(source.centroid[1] - anchor[1]))


def _signature_code(signature: ObjectSignature) -> str:
    return f"{signature[0]}:{signature[1]}:{signature[2]}x{signature[3]}"


def _action_code(action: ActionName, context) -> str:
    schema = build_action_schema(action, context)
    if schema.coarse_bin is not None:
        return f"click_bin_{schema.coarse_bin[0]}_{schema.coarse_bin[1]}"
    if schema.action_type == "move":
        return f"move_{schema.direction or 'none'}"
    if schema.action_type == "select":
        return "select_cycle"
    if schema.raw_action is not None:
        return f"raw_{schema.raw_action}"
    return schema.family.replace(":", "_")


def _mode_code(mode_key: str) -> str:
    if mode_key == DEFAULT_MODE:
        return "default"
    return mode_key.replace(":", "_")


def _beta_log_evidence(support: int, contradiction: int, alpha: float = 1.0, beta: float = 1.0) -> float:
    return (
        math.lgamma(support + alpha)
        + math.lgamma(contradiction + beta)
        - math.lgamma(support + contradiction + alpha + beta)
        - (math.lgamma(alpha) + math.lgamma(beta) - math.lgamma(alpha + beta))
    )


def _normalize_log_weights(log_weights: dict[object, float]) -> dict[object, float]:
    if not log_weights:
        return {}
    max_log = max(log_weights.values())
    exp_weights = {key: math.exp(value - max_log) for key, value in log_weights.items()}
    total = sum(exp_weights.values())
    if total <= 0.0:
        uniform = 1.0 / float(len(exp_weights))
        return {key: uniform for key in exp_weights}
    return {key: value / total for key, value in exp_weights.items()}
