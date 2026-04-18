from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math

from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.types import (
    ActionName,
    ActionThought,
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
class ModeHypothesis:
    evidence: EvidenceCounter
    entries: int = 0
    move_effect_steps: int = 0
    stalled_steps: int = 0
    reward_sum: float = 0.0
    mover_support: dict[ObjectSignature, int] = field(default_factory=dict)

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
            + diversity_bonus
        )


class RuntimeRuleController:
    def __init__(self) -> None:
        self.signature_stats: dict[ObjectSignature, SignatureStats] = defaultdict(SignatureStats)
        self.motion_rules: dict[tuple[str, ActionName, ObjectSignature], MotionRule] = {}
        self.control_hypotheses: dict[tuple[str, ObjectSignature], ControlHypothesis] = {}
        self.objective_hypotheses: dict[tuple[str, ObjectSignature, ObjectSignature], ObjectiveHypothesis] = {}
        self.mode_hypotheses: dict[str, ModeHypothesis] = {}
        self.action_visits: dict[tuple[str, ActionName], int] = defaultdict(int)
        self.selector_visits: dict[ActionName, int] = defaultdict(int)
        self.current_mode_key = DEFAULT_MODE
        self.pending_undo = False
        self.undo_attempts = 0
        self.exploit_started = False
        self.active_goal_anchor: tuple[float, float] | None = None
        self.active_exploit_action: ActionName | None = None
        self.active_pair_key: tuple[str, ObjectSignature, ObjectSignature] | None = None
        self.active_probe_anchor: tuple[float, float] | None = None
        self.active_probe_pair_key: tuple[str, ObjectSignature, ObjectSignature] | None = None
        self.reference_state: StructuredState | None = None
        self.reference_fingerprint: str | None = None
        self._pending_action_mode_key = DEFAULT_MODE

    def reset_episode(self) -> None:
        self.signature_stats.clear()
        self.motion_rules.clear()
        self.control_hypotheses.clear()
        self.objective_hypotheses.clear()
        self.mode_hypotheses.clear()
        self.action_visits.clear()
        self.selector_visits.clear()
        self.current_mode_key = DEFAULT_MODE
        self.pending_undo = False
        self.undo_attempts = 0
        self.exploit_started = False
        self.active_goal_anchor = None
        self.active_exploit_action = None
        self.active_pair_key = None
        self.active_probe_anchor = None
        self.active_probe_pair_key = None
        self.reference_state = None
        self.reference_fingerprint = None
        self._pending_action_mode_key = DEFAULT_MODE

    def reset_all(self) -> None:
        self.reset_episode()

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

        mover_scores = self._mover_scores(state, move_actions)
        action_bonus: dict[ActionName, float] = defaultdict(float)
        claims: list[StructuredClaim] = []

        if mover_scores:
            top_mover_signature, top_mover_score = max(mover_scores.items(), key=lambda item: item[1])
            control = self.control_hypotheses.get((self.current_mode_key, top_mover_signature))
            claims.append(
                StructuredClaim(
                    claim_type="control",
                    subject=_signature_code(top_mover_signature),
                    relation="controllable",
                    object=_mode_code(self.current_mode_key),
                    confidence=0.0 if control is None else control.evidence.confidence,
                    evidence=0.0 if control is None else float(control.evidence.balance),
                    salience=top_mover_score,
                )
            )

        objective_candidates = sorted(
            self._objective_candidates(state, mover_scores),
            key=lambda item: item[2].utility + mover_scores.get(item[0], 0.0),
            reverse=True,
        )
        top_candidates = objective_candidates[:3]
        for mover_signature, target_object, objective in top_candidates:
            claims.append(
                StructuredClaim(
                    claim_type="objective",
                    subject=_signature_code(mover_signature),
                    relation="toward",
                    object=_signature_code(object_signature(target_object)),
                    confidence=objective.evidence.confidence,
                    evidence=float(objective.evidence.balance),
                    salience=objective.utility,
                )
            )
            mover_objects = _group_by_signature(state.objects).get(mover_signature, [])
            if not mover_objects:
                continue
            for action in move_actions:
                predicted_objects = self._predict_signature_objects(state, action, mover_signature)
                if not predicted_objects:
                    continue
                before_distance = min(_cell_distance(obj, target_object) for obj in mover_objects)
                after_distance = min(_cell_distance(obj, target_object) for obj in predicted_objects)
                overlap = max(_overlap_fraction(obj, target_object) for obj in predicted_objects)
                progress = before_distance - after_distance
                action_bonus[action] = max(
                    action_bonus[action],
                    progress + (2.5 * overlap) + (0.15 * objective.utility) + (0.1 * mover_scores.get(mover_signature, 0.0)),
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
            ranked_modes: list[tuple[float, str, float]] = []
            for mode_key, actions in grouped.items():
                hypothesis = self.mode_hypotheses.get(mode_key)
                utility = 0.0 if hypothesis is None else hypothesis.utility
                confidence = 0.0 if hypothesis is None else hypothesis.evidence.confidence
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
                        salience=mode_score,
                    )
                )
                for action in actions:
                    action_bonus[action] += max(self._thought_selector_followup(thought, action), 0.0)
                    action_bonus[action] += 0.15 * self._thought_uncertainty(thought, action)
            ranked_modes.sort(reverse=True)
            if ranked_modes and ranked_modes[0][2] <= 0.25 and sum(self.selector_visits.values()) >= len(grouped):
                for action in selector_actions:
                    action_bonus[action] -= 0.75

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
        if top_candidates:
            question_tokens = ("question", "reduce_objective_distance")
        elif selector_actions and max((self._thought_selector_followup(thought, action) for action in selector_actions), default=0.0) > 0.5:
            question_tokens = ("question", "test_control_mode")
        elif move_actions:
            question_tokens = ("question", "test_move_effect")

        belief_tokens = thought.belief_tokens
        if claims:
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

        mover_scores = self._mover_scores(state, move_actions)

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
                self.active_probe_anchor = None
                self.active_probe_pair_key = None
            elif transition.next_state.fingerprint() == transition.state.fingerprint():
                self.pending_undo = False
                self.undo_attempts = 0

        if schema.action_type in {"click", "select"}:
            self.selector_visits[transition.action] += 1
            self.current_mode_key = _mode_key(transition.action, schema)
            mode = self.mode_hypotheses.setdefault(self.current_mode_key, ModeHypothesis(evidence=EvidenceCounter()))
            mode.entries += 1
            if transition_delta <= 0.1:
                mode.evidence.support += 1

        signature_groups: dict[ObjectSignature, list[tuple[ObjectState, ObjectState]]] = defaultdict(list)
        for signature, before_obj, after_obj in _match_objects(transition.state, transition.next_state):
            signature_groups[signature].append((before_obj, after_obj))

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
                    rule.observe(avg_dy, avg_dx)
            else:
                stats.stable += len(pairs)
                rule = self.motion_rules.get((action_mode_key, transition.action, signature))
                if rule is not None:
                    rule.evidence.contradiction += 1

        if schema.action_type == "move":
            mode = self.mode_hypotheses.setdefault(action_mode_key, ModeHypothesis(evidence=EvidenceCounter()))
            mode.observe_move(moving_signatures, float(transition.reward))

            updated_control_signatures: set[ObjectSignature] = set()
            for signature in moving_signatures:
                motion = motion_summary[signature][2]
                control = self.control_hypotheses.setdefault(
                    (action_mode_key, signature),
                    ControlHypothesis(evidence=EvidenceCounter()),
                )
                control.observe(moved=True, motion=motion, reward=float(transition.reward))
                updated_control_signatures.add(signature)
            for mode_key, signature in list(self.control_hypotheses.keys()):
                if mode_key != action_mode_key or signature in updated_control_signatures:
                    continue
                control = self.control_hypotheses[(mode_key, signature)]
                if signature in _signatures_in_state(transition.state):
                    control.observe(moved=False, motion=0.0, reward=0.0)

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

        if not moving_signatures:
            return

        target_objects = [
            obj
            for obj in transition.next_state.objects
            if self._is_candidate_target_object(obj, moving_signatures, transition.next_state.grid_shape)
        ]
        if not target_objects:
            return

        before_groups = _group_by_signature(transition.state.objects)
        after_groups = _group_by_signature(transition.next_state.objects)
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

    def _ready_to_exploit(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        mover_scores: dict[ObjectSignature, float],
    ) -> bool:
        if not move_actions or not mover_scores:
            return False
        objective_candidates = self._objective_candidates(state, mover_scores)
        if not objective_candidates:
            return False
        return any(
            objective.utility >= 2.0
            or objective.reward_hits > 0
            or objective.goal_activations > 0
            or objective.successful_interactions > 0
            for _, _, objective in objective_candidates
        )

    def _exploit_plan(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        mover_scores: dict[ObjectSignature, float],
        thought: RuntimeThought | None,
    ) -> PlanOutput | None:
        movers_by_signature = _group_by_signature(state.objects)
        objective_candidates = self._objective_candidates(state, mover_scores)
        if not objective_candidates:
            return None

        best_action: ActionName | None = None
        best_score = float("-inf")
        best_local_progress = float("-inf")
        best_pair_key: tuple[str, ObjectSignature, ObjectSignature] | None = None
        best_target: ObjectState | None = None

        for mover_signature, target_object, objective in objective_candidates:
            mover_objects = movers_by_signature.get(mover_signature, [])
            if not mover_objects:
                continue
            control_score = mover_scores.get(mover_signature, 0.0)
            target_signature = object_signature(target_object)
            path_plan = self._path_action_to_object(
                state=state,
                move_actions=move_actions,
                mover_signature=mover_signature,
                target_object=target_object,
                objective=objective,
            )
            if path_plan is not None:
                path_action, path_steps = path_plan
                thought_bonus = self._thought_value(thought, path_action) + (0.25 * self._thought_policy(thought, path_action))
                local_progress = max(1.5 - (0.1 * float(path_steps)), 0.25)
                score = local_progress + objective.utility + control_score + thought_bonus
                if score > best_score:
                    best_score = score
                    best_local_progress = local_progress
                    best_action = path_action
                    best_pair_key = (self.current_mode_key, mover_signature, target_signature)
                    best_target = target_object
                continue
            for action in move_actions:
                predicted_objects = self._predict_signature_objects(state, action, mover_signature)
                if not predicted_objects:
                    continue
                before_distance = min(_cell_distance(obj, target_object) for obj in mover_objects)
                after_distance = min(_cell_distance(obj, target_object) for obj in predicted_objects)
                progress = before_distance - after_distance
                overlap = max(_overlap_fraction(obj, target_object) for obj in predicted_objects)
                local_progress = progress + (3.0 * overlap)
                thought_bonus = self._thought_value(thought, action) + (0.25 * self._thought_policy(thought, action))
                score = local_progress + objective.utility + control_score + thought_bonus
                if score > best_score:
                    best_score = score
                    best_local_progress = local_progress
                    best_action = action
                    best_pair_key = (self.current_mode_key, mover_signature, target_signature)
                    best_target = target_object

        if best_action is None or best_pair_key is None or best_target is None or best_local_progress <= 0.0:
            return None

        self.active_pair_key = best_pair_key
        self.active_goal_anchor = best_target.centroid
        self.active_exploit_action = best_action
        target_signature = best_pair_key[2]
        target_code = f"{target_signature[0]}:{target_signature[1]}:{target_signature[2]}x{target_signature[3]}"
        return PlanOutput(
            action=best_action,
            scores={"exploit": best_score, "objective_utility": self.objective_hypotheses[best_pair_key].utility},
            language=LanguageTrace(
                belief_tokens=("belief", "objective_hypothesis", target_code),
                question_tokens=self._question_tokens(thought, ("question", "reduce_objective_distance")),
                plan_tokens=("plan", best_action),
            ),
            search_path=(best_action,),
        )

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
            return None
        mover_signature = self.active_pair_key[1]
        if mover_signature not in mover_scores:
            self.active_goal_anchor = None
            self.active_exploit_action = None
            self.active_pair_key = None
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
            return None
        before_distance = min(_anchor_distance(obj, self.active_goal_anchor) for obj in movers)
        after_distance = min(_anchor_distance(obj, self.active_goal_anchor) for obj in predicted_objects)
        if before_distance - after_distance <= 0.0:
            self.active_goal_anchor = None
            self.active_exploit_action = None
            self.active_pair_key = None
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
        return PlanOutput(
            action=action,
            scores={
                "diagnostic": 1.0 + self._thought_uncertainty(thought, action),
                "untested_move": 1.0,
                "latent_value": self._thought_value(thought, action),
            },
            language=LanguageTrace(
                belief_tokens=("belief", "dynamics_uncertain"),
                question_tokens=self._question_tokens(thought, ("question", "test_move_effect")),
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

        adjacent_options: list[tuple[float, ActionName, ObjectState, ObjectiveHypothesis]] = []
        for action in interact_actions:
            target = self._interaction_target_object(state, action, context)
            if target is None:
                continue
            target_signature = object_signature(target)
            for mover_signature, candidate_object, objective in candidates:
                if object_signature(candidate_object) != target_signature:
                    continue
                score = (
                    2.0 / float(objective.interaction_tests + 1)
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
            mover_objects = movers_by_signature.get(mover_signature, [])
            if not mover_objects:
                continue
            base_score = (
                1.5 / float(objective.interaction_tests + 1)
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
                    best_target_signature = object_signature(target_object)
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
        candidates = sorted(
            self._objective_candidates(state, mover_scores),
            key=lambda item: item[2].utility,
            reverse=True,
        )
        if len(candidates) < 2:
            return None
        candidates = candidates[:4]

        best_action: ActionName | None = None
        best_score = float("-inf")
        for action in move_actions:
            action_progress: list[float] = []
            for mover_signature, target_object, objective in candidates:
                predicted_objects = self._predict_signature_objects(state, action, mover_signature)
                current_objects = _group_by_signature(state.objects).get(mover_signature, [])
                if not predicted_objects or not current_objects:
                    continue
                before_distance = min(_cell_distance(obj, target_object) for obj in current_objects)
                after_distance = min(_cell_distance(obj, target_object) for obj in predicted_objects)
                action_progress.append((before_distance - after_distance) + (0.2 * objective.utility))
            if len(action_progress) < 2:
                continue
            mean_progress = sum(action_progress) / float(len(action_progress))
            variance = sum((value - mean_progress) ** 2 for value in action_progress) / float(len(action_progress))
            exploration = 1.0 / math.sqrt(self.action_visits[(self.current_mode_key, action)] + 1.0)
            score = (
                variance
                + (0.15 * abs(mean_progress))
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
            score = exploration + uncertainty + utility + max(latent_followup, 0.0) + (0.35 * latent_uncertainty) + (0.15 * latent_value)
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
            score = control_utility + (1.2 * best_confidence) + (0.5 * best_magnitude)
            if score >= 1.8 or (best_confidence >= 0.55 and best_magnitude >= 0.5):
                scores[signature] = score
        return scores

    def _objective_candidates(
        self,
        state: StructuredState,
        mover_scores: dict[ObjectSignature, float],
    ) -> list[tuple[ObjectSignature, ObjectState, ObjectiveHypothesis]]:
        candidates: list[tuple[ObjectSignature, ObjectState, ObjectiveHypothesis]] = []
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
                if objective is None or objective.utility <= 1.0:
                    continue
                if self._requires_interaction_confirmation(state, target_object, objective):
                    continue
                candidates.append((mover_signature, target_object, objective))
        return candidates

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
        if "target" in target_object.tags:
            return self._path_action_to_cells(
                state=state,
                move_actions=move_actions,
                mover_signature=mover_signature,
                goal_cells=set(target_object.cells),
            )
        if objective is not None and objective.successful_interactions > 0 and "interactable" in target_object.tags:
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

    def _path_action_to_cells(
        self,
        state: StructuredState,
        move_actions: list[ActionName],
        mover_signature: ObjectSignature,
        goal_cells: set[tuple[int, int]],
    ) -> tuple[ActionName, int] | None:
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
                return _reconstruct_first_action(cell, parent)
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

    def _goal_is_active(self, state: StructuredState) -> bool:
        if state.flags_dict().get("goal_active") == "1":
            return True
        return any("target" in obj.tags and "active" in obj.tags for obj in state.objects)

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
        if "interactable" in obj.tags:
            return True
        if "target" in obj.tags:
            return False
        return self._is_candidate_target_object(obj, moving_signatures, grid_shape) and obj.area <= 4

    def _clear_active_exploit(self) -> None:
        self.active_goal_anchor = None
        self.active_exploit_action = None
        self.active_pair_key = None
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
        after_group = sorted(after_groups[signature], key=_sort_key)
        for index in range(min(len(before_group), len(after_group))):
            matches.append((signature, before_group[index], after_group[index]))
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
    path_actions: list[ActionName] = []
    current = goal
    while parent[current] is not None:
        prev, action = parent[current]
        path_actions.append(action)
        current = prev
    if not path_actions:
        return None
    path_actions.reverse()
    return (path_actions[0], len(path_actions))


def _in_bounds(cell: tuple[int, int], grid_shape: tuple[int, int]) -> bool:
    return 0 <= cell[0] < grid_shape[0] and 0 <= cell[1] < grid_shape[1]


def _cell_distance(source: ObjectState, target: ObjectState) -> float:
    return min(
        float(abs(source_cell[0] - target_cell[0]) + abs(source_cell[1] - target_cell[1]))
        for source_cell in source.cells
        for target_cell in target.cells
    )


def _overlap_fraction(source: ObjectState, target: ObjectState) -> float:
    source_cells = set(source.cells)
    target_cells = set(target.cells)
    if not target_cells:
        return 0.0
    return float(len(source_cells & target_cells)) / float(len(target_cells))


def _anchor_distance(source: ObjectState, anchor: tuple[float, float]) -> float:
    return float(abs(source.centroid[0] - anchor[0]) + abs(source.centroid[1] - anchor[1]))


def _signature_code(signature: ObjectSignature) -> str:
    return f"{signature[0]}:{signature[1]}:{signature[2]}x{signature[3]}"


def _mode_code(mode_key: str) -> str:
    if mode_key == DEFAULT_MODE:
        return "default"
    return mode_key.replace(":", "_")
