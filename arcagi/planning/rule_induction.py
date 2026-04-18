from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math

import numpy as np

from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.types import ActionName, ObjectState, StructuredClaim, StructuredState, Transition
from arcagi.core.spatial_workspace import INTERACT_DELTAS

ObjectSignature = tuple[int, int, int, int, tuple[str, ...]]


@dataclass
class SignatureEffectStats:
    present_count: int = 0
    changed_count: int = 0
    disappeared_count: int = 0
    appeared_count: int = 0
    reward_sum: float = 0.0
    delta_sum: float = 0.0
    move_y_sum: float = 0.0
    move_x_sum: float = 0.0
    move_count: int = 0

    @property
    def causal_rate(self) -> float:
        denominator = max(self.present_count + self.appeared_count, 1)
        return (self.changed_count + self.disappeared_count + self.appeared_count) / denominator

    @property
    def mean_reward(self) -> float:
        denominator = max(self.changed_count + self.disappeared_count + self.appeared_count, 1)
        return self.reward_sum / denominator

    @property
    def mean_delta(self) -> float:
        denominator = max(self.changed_count + self.disappeared_count + self.appeared_count, 1)
        return self.delta_sum / denominator

    @property
    def mean_motion(self) -> float:
        if self.move_count == 0:
            return 0.0
        delta_y = self.move_y_sum / self.move_count
        delta_x = self.move_x_sum / self.move_count
        return math.sqrt((delta_y * delta_y) + (delta_x * delta_x))

    @property
    def mean_move_vector(self) -> tuple[float, float]:
        if self.move_count == 0:
            return (0.0, 0.0)
        return (self.move_y_sum / self.move_count, self.move_x_sum / self.move_count)


@dataclass
class ActionRuleStats:
    visits: int = 0
    reward_sum: float = 0.0
    delta_sum: float = 0.0
    no_effect_count: int = 0
    signature_effects: dict[ObjectSignature, SignatureEffectStats] = field(default_factory=dict)

    @property
    def mean_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.reward_sum / self.visits

    @property
    def mean_delta(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.delta_sum / self.visits

    @property
    def no_effect_rate(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.no_effect_count / self.visits


@dataclass
class ActionFamilyStats:
    visits: int = 0
    reward_sum: float = 0.0
    delta_sum: float = 0.0
    no_effect_count: int = 0

    @property
    def mean_reward(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.reward_sum / self.visits

    @property
    def mean_delta(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.delta_sum / self.visits

    @property
    def no_effect_rate(self) -> float:
        if self.visits == 0:
            return 0.0
        return self.no_effect_count / self.visits


@dataclass(frozen=True)
class GroundedPredicate:
    predicate_type: str
    subject: str
    relation: str
    object: str

    def key(self) -> tuple[str, str, str, str]:
        return (self.predicate_type, self.subject, self.relation, self.object)

    def render(self) -> str:
        return f"{self.predicate_type}:{self.subject}{self.relation}{self.object}"

    def as_claim(
        self,
        *,
        subject_prefix: str,
        confidence: float,
        evidence: float,
        salience: float,
    ) -> StructuredClaim:
        return StructuredClaim(
            claim_type="hypothesis",
            subject=subject_prefix,
            relation="predicts",
            object=self.render(),
            confidence=confidence,
            evidence=evidence,
            salience=salience,
        )


@dataclass
class GroundedHypothesis:
    scope: tuple[GroundedPredicate, ...]
    consequence: GroundedPredicate
    support: float = 0.0
    contradiction: float = 0.0
    observations: int = 0

    @property
    def confidence(self) -> float:
        return (self.support + 1.0) / (self.support + self.contradiction + 2.0)

    @property
    def weight(self) -> float:
        return (2.0 * self.confidence) - 1.0

    @property
    def evidence(self) -> float:
        return self.support + self.contradiction

    def scope_key(self) -> tuple[tuple[str, str, str, str], ...]:
        return tuple(predicate.key() for predicate in self.scope)

    def scope_label(self) -> str:
        return ";".join(predicate.render() for predicate in self.scope)


class EpisodeRuleInducer:
    def __init__(self) -> None:
        self.action_rules: dict[ActionName, ActionRuleStats] = {}
        self.family_rules: dict[str, ActionFamilyStats] = {}
        self.target_context_rules: dict[tuple[str, ObjectSignature], ActionFamilyStats] = {}
        self.signature_values: dict[ObjectSignature, float] = defaultdict(float)
        self.total_transitions = 0
        self.hypotheses: dict[
            tuple[tuple[tuple[str, str, str, str], ...], tuple[str, str, str, str]],
            GroundedHypothesis,
        ] = {}
        self.hypotheses_by_scope: dict[tuple[tuple[str, str, str, str], ...], set[
            tuple[tuple[tuple[str, str, str, str], ...], tuple[str, str, str, str]]
        ]] = defaultdict(set)

    def clear(self) -> None:
        self.action_rules.clear()
        self.family_rules.clear()
        self.target_context_rules.clear()
        self.signature_values.clear()
        self.total_transitions = 0
        self.hypotheses.clear()
        self.hypotheses_by_scope.clear()

    def record(self, transition: Transition) -> None:
        self.total_transitions += 1
        action_stats = self.action_rules.setdefault(transition.action, ActionRuleStats())
        action_stats.visits += 1
        action_stats.reward_sum += transition.reward
        transition_delta = self._state_delta(transition.state, transition.next_state)
        action_stats.delta_sum += transition_delta
        if transition_delta < 0.05 and transition.reward <= 0.0:
            action_stats.no_effect_count += 1
        context = build_action_schema_context(transition.state.affordances, dict(transition.state.action_roles))
        family = build_action_schema(transition.action, context).family
        action_type = build_action_schema(transition.action, context).action_type
        family_stats = self.family_rules.setdefault(family, ActionFamilyStats())
        family_stats.visits += 1
        family_stats.reward_sum += transition.reward
        family_stats.delta_sum += transition_delta
        if transition_delta < 0.05 and transition.reward <= 0.0:
            family_stats.no_effect_count += 1
        for signature in action_target_signatures(transition.state, transition.action):
            target_stats = self.target_context_rules.setdefault((action_type, signature), ActionFamilyStats())
            target_stats.visits += 1
            target_stats.reward_sum += transition.reward
            target_stats.delta_sum += transition_delta
            if transition_delta < 0.05 and transition.reward <= 0.0:
                target_stats.no_effect_count += 1

        before_groups = _group_objects(transition.state.objects)
        after_groups = _group_objects(transition.next_state.objects)
        all_signatures = before_groups.keys() | after_groups.keys()
        for signature in all_signatures:
            effect = action_stats.signature_effects.setdefault(signature, SignatureEffectStats())
            before_objects = before_groups.get(signature, [])
            after_objects = after_groups.get(signature, [])
            matched_count = min(len(before_objects), len(after_objects))
            for index in range(matched_count):
                before_obj = before_objects[index]
                after_obj = after_objects[index]
                effect.present_count += 1
                delta = _object_delta(before_obj, after_obj)
                if delta > 0.0:
                    effect.changed_count += 1
                    effect.delta_sum += delta
                    effect.reward_sum += transition.reward
                    self.signature_values[signature] += transition.reward + (0.05 * delta)
                move_y = after_obj.centroid[0] - before_obj.centroid[0]
                move_x = after_obj.centroid[1] - before_obj.centroid[1]
                if abs(move_y) > 1e-6 or abs(move_x) > 1e-6:
                    effect.move_y_sum += move_y
                    effect.move_x_sum += move_x
                    effect.move_count += 1
            for before_obj in before_objects[matched_count:]:
                effect.present_count += 1
                effect.disappeared_count += 1
                effect.delta_sum += float(before_obj.area)
                effect.reward_sum += transition.reward
                self.signature_values[signature] += transition.reward + (0.1 * before_obj.area)
            for after_obj in after_objects[matched_count:]:
                effect.appeared_count += 1
                effect.delta_sum += float(after_obj.area)
                effect.reward_sum += transition.reward
                self.signature_values[signature] += transition.reward + (0.1 * after_obj.area)
        self._record_grounded_hypotheses(transition)

    def action_score(self, state: StructuredState, action: ActionName) -> float:
        action_stats = self.action_rules.get(action)
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        schema = build_action_schema(action, context)
        family = schema.family
        target_signatures = action_target_signatures(state, action)
        family_stats = self.family_rules.get(family)
        if action_stats is None:
            action_score = {
                "move": 0.2,
                "interact": 0.05,
                "click": 0.05,
                "select": 0.05,
                "wait": 0.0,
            }.get(schema.action_type, 0.05)
        else:
            explore_bonus = math.sqrt(math.log(self.total_transitions + 2.0) / (action_stats.visits + 1.0))
            signature_bonus = 0.0
            signatures_to_score = target_signatures or tuple(dict.fromkeys(object_signature(obj) for obj in state.objects))
            for signature in signatures_to_score:
                effect = action_stats.signature_effects.get(signature)
                if effect is None:
                    continue
                signature_bonus += (
                    effect.causal_rate
                    + (0.25 * effect.mean_delta)
                    + (0.2 * effect.mean_motion)
                    + (0.25 * self.signature_values.get(signature, 0.0))
                )
            action_score = (
                (0.9 * action_stats.mean_reward)
                + (0.1 * action_stats.mean_delta)
                + (0.35 * explore_bonus)
                + (0.12 * signature_bonus)
                - (1.25 * action_stats.no_effect_rate)
            )
            if schema.action_type in {"interact", "click", "select"}:
                action_score -= 0.2 * action_stats.no_effect_rate
        if family_stats is None:
            family_score = {
                "move": 0.15,
                "interact": 0.03,
                "click": 0.03,
                "select": 0.03,
                "wait": 0.0,
            }.get(schema.action_type, 0.03)
        else:
            family_explore = math.sqrt(math.log(self.total_transitions + 2.0) / (family_stats.visits + 1.0))
            family_score = (
                (0.35 * family_stats.mean_reward)
                + (0.05 * family_stats.mean_delta)
                + (0.2 * family_explore)
                - (1.0 * family_stats.no_effect_rate)
            )
        progress_bonus = 0.0
        if action_stats is not None:
            progress_bonus = self._progress_bonus(state, action_stats)
        context_bonus = self._target_context_bonus(schema.action_type, target_signatures)
        hypothesis_bonus = self.action_hypothesis_bonus(state, action)
        diagnostic_bonus = self.action_diagnostic_bonus(state, action)
        return action_score + family_score + progress_bonus + context_bonus + hypothesis_bonus + diagnostic_bonus

    def top_claims(
        self,
        *,
        state: StructuredState | None = None,
        action: ActionName | None = None,
        limit: int = 4,
        min_weight: float = 0.1,
    ) -> tuple[StructuredClaim, ...]:
        if state is not None and action is not None:
            candidates = self.applicable_hypotheses(state, action)
        else:
            candidates = tuple(self.hypotheses.values())
        ranked = sorted(
            (
                hypothesis
                for hypothesis in candidates
                if hypothesis.weight >= min_weight
            ),
            key=lambda hypothesis: (hypothesis.weight, hypothesis.support, -hypothesis.contradiction),
            reverse=True,
        )
        claims: list[StructuredClaim] = []
        for hypothesis in ranked[:limit]:
            confidence = max(min(hypothesis.confidence, 1.0), 0.0)
            claims.append(
                hypothesis.consequence.as_claim(
                    subject_prefix=hypothesis.scope_label(),
                    confidence=confidence,
                    evidence=hypothesis.evidence,
                    salience=abs(hypothesis.weight),
                )
            )
        return tuple(claims)

    def applicable_hypotheses(self, state: StructuredState, action: ActionName) -> tuple[GroundedHypothesis, ...]:
        scopes = self._scope_definitions(state, action)
        hypothesis_ids: set[
            tuple[tuple[tuple[str, str, str, str], ...], tuple[str, str, str, str]]
        ] = set()
        for scope_key, _scope_predicates, _target_signature in scopes:
            hypothesis_ids.update(self.hypotheses_by_scope.get(scope_key, set()))
        hypotheses = [self.hypotheses[hypothesis_id] for hypothesis_id in hypothesis_ids]
        hypotheses.sort(key=lambda item: (abs(item.weight), item.support, -item.contradiction), reverse=True)
        return tuple(hypotheses)

    def action_hypothesis_bonus(self, state: StructuredState, action: ActionName) -> float:
        bonus = 0.0
        for hypothesis in self.applicable_hypotheses(state, action)[:8]:
            desirability = _consequence_desirability(hypothesis.consequence)
            if desirability == 0.0:
                continue
            bonus += desirability * hypothesis.weight
        return max(min(bonus, 2.5), -2.5)

    def action_diagnostic_bonus(self, state: StructuredState, action: ActionName) -> float:
        grouped: dict[tuple[str, str], list[GroundedHypothesis]] = defaultdict(list)
        for hypothesis in self.applicable_hypotheses(state, action):
            grouped[(hypothesis.consequence.predicate_type, hypothesis.consequence.subject)].append(hypothesis)
        bonus = 0.0
        for group in grouped.values():
            if len(group) < 2:
                continue
            support = [hypothesis for hypothesis in group if hypothesis.support > 0.0]
            if len(support) < 2:
                continue
            weights = [abs(hypothesis.weight) for hypothesis in support]
            if not weights:
                continue
            uncertainty = 1.0 - max(weights)
            if uncertainty <= 0.0:
                continue
            bonus += 0.2 * uncertainty
        return min(bonus, 0.75)

    @staticmethod
    def _state_delta(before: StructuredState, after: StructuredState) -> float:
        before_summary = before.transition_vector()
        after_summary = after.transition_vector()
        return float(np.linalg.norm(after_summary - before_summary))

    def _progress_bonus(self, state: StructuredState, action_stats: ActionRuleStats) -> float:
        multiplicity: dict[ObjectSignature, int] = defaultdict(int)
        for obj in state.objects:
            multiplicity[object_signature(obj)] += 1
        movers: list[tuple[ObjectState, SignatureEffectStats]] = []
        targets: list[tuple[ObjectState, float]] = []
        for obj in state.objects:
            signature = object_signature(obj)
            effect = action_stats.signature_effects.get(signature)
            if effect is not None and effect.mean_motion >= 0.25:
                movers.append((obj, effect))
                continue
            stability = self._signature_stability(signature)
            rarity = 1.0 / float(multiplicity[signature])
            area_bias = 1.0 / math.sqrt(max(float(obj.area), 1.0))
            priority = stability * rarity * area_bias
            if priority > 0.15:
                targets.append((obj, priority))
        if not movers or not targets:
            return 0.0
        bonus = 0.0
        for mover, effect in movers:
            move_y, move_x = effect.mean_move_vector
            predicted_y = mover.centroid[0] + move_y
            predicted_x = mover.centroid[1] + move_x
            for target, priority in targets:
                current_distance = _cell_distance(mover.centroid, target.centroid)
                predicted_distance = _cell_distance((predicted_y, predicted_x), target.centroid)
                progress = current_distance - predicted_distance
                if progress > 0.0:
                    bonus += priority * progress
        return 0.2 * bonus

    def _signature_stability(self, signature: ObjectSignature) -> float:
        present_count = 0
        move_total = 0.0
        for action_stats in self.action_rules.values():
            effect = action_stats.signature_effects.get(signature)
            if effect is None:
                continue
            present_count += effect.present_count
            move_total += abs(effect.move_y_sum) + abs(effect.move_x_sum)
        if present_count == 0:
            return 0.0
        return 1.0 / (1.0 + (move_total / float(present_count)))

    def _target_context_bonus(self, action_type: str, signatures: tuple[ObjectSignature, ...]) -> float:
        if not signatures:
            return 0.0
        bonus = 0.0
        for signature in signatures:
            stats = self.target_context_rules.get((action_type, signature))
            if stats is None:
                bonus += 0.2 * math.sqrt(math.log(self.total_transitions + 2.0))
                continue
            explore = math.sqrt(math.log(self.total_transitions + 2.0) / (stats.visits + 1.0))
            bonus += (
                (0.9 * stats.mean_reward)
                + (0.1 * stats.mean_delta)
                + (0.35 * explore)
                - (1.5 * stats.no_effect_rate)
            )
        return bonus / float(len(signatures))

    def _record_grounded_hypotheses(self, transition: Transition) -> None:
        scopes = self._scope_definitions(transition.state, transition.action)
        relation_predicates = self._relation_delta_predicates(transition.state, transition.next_state)
        hidden_predicates = self._hidden_state_delta_predicates(transition.state, transition.next_state)
        reward_predicates = self._reward_predicates(transition.reward)
        state_change_predicates = self._state_change_predicates(transition)
        for scope_key, scope_predicates, target_signature in scopes:
            observed = list(reward_predicates)
            observed.extend(state_change_predicates)
            observed.extend(hidden_predicates)
            observed.extend(relation_predicates)
            if target_signature is not None:
                observed.extend(self._target_effect_predicates(transition, target_signature))
            observed_keys = {predicate.key() for predicate in observed}
            existing_ids = set(self.hypotheses_by_scope.get(scope_key, set()))
            for hypothesis_id in existing_ids:
                hypothesis = self.hypotheses[hypothesis_id]
                hypothesis.observations += 1
                if hypothesis.consequence.key() in observed_keys:
                    hypothesis.support += 1.0
                else:
                    hypothesis.contradiction += 1.0
            for predicate in observed:
                hypothesis_id = (scope_key, predicate.key())
                hypothesis = self.hypotheses.get(hypothesis_id)
                if hypothesis is None:
                    hypothesis = GroundedHypothesis(scope=scope_predicates, consequence=predicate, support=1.0, observations=1)
                    self.hypotheses[hypothesis_id] = hypothesis
                    self.hypotheses_by_scope[scope_key].add(hypothesis_id)

    def _scope_definitions(
        self,
        state: StructuredState,
        action: ActionName,
    ) -> tuple[
        tuple[
            tuple[tuple[str, str, str, str], ...],
            tuple[GroundedPredicate, ...],
            ObjectSignature | None,
        ],
        ...
    ]:
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        schema = build_action_schema(action, context)
        base_scope = (
            GroundedPredicate("action", "type", "=", schema.action_type),
            GroundedPredicate("action", "role", "=", schema.role or "unknown"),
        )
        scopes: list[
            tuple[
                tuple[tuple[str, str, str, str], ...],
                tuple[GroundedPredicate, ...],
                ObjectSignature | None,
            ]
        ] = [
            (tuple(predicate.key() for predicate in base_scope), base_scope, None),
        ]
        relation_label = _target_relation_label(schema.action_type)
        for signature in action_target_signatures(state, action):
            signature_token = _signature_token(signature)
            scope_predicates = base_scope + (
                GroundedPredicate("object", "target_signature", "=", signature_token),
                GroundedPredicate("relation", "agent_target", "=", relation_label),
            )
            scopes.append((tuple(predicate.key() for predicate in scope_predicates), scope_predicates, signature))
        return tuple(scopes)

    def _reward_predicates(self, reward: float) -> tuple[GroundedPredicate, ...]:
        if reward > 0.0:
            bucket = "positive"
        elif reward < 0.0:
            bucket = "negative"
        else:
            bucket = "zero"
        return (
            GroundedPredicate("reward", "episode", "=", bucket),
        )

    def _state_change_predicates(self, transition: Transition) -> tuple[GroundedPredicate, ...]:
        delta = self._state_delta(transition.state, transition.next_state)
        bucket = "changed" if delta >= 0.05 else "stable"
        return (
            GroundedPredicate("transition", "state_change", "=", bucket),
        )

    def _hidden_state_delta_predicates(
        self,
        before: StructuredState,
        after: StructuredState,
    ) -> tuple[GroundedPredicate, ...]:
        predicates: list[GroundedPredicate] = []
        before_flags = before.flags_dict()
        after_flags = after.flags_dict()
        for key in sorted(before_flags.keys() | after_flags.keys()):
            if before_flags.get(key, "") == after_flags.get(key, ""):
                continue
            predicates.append(GroundedPredicate("flag", key, "=", after_flags.get(key, "")))
        before_inventory = before.inventory_dict()
        after_inventory = after.inventory_dict()
        for key in sorted(before_inventory.keys() | after_inventory.keys()):
            if before_inventory.get(key, "") == after_inventory.get(key, ""):
                continue
            predicates.append(GroundedPredicate("inventory", key, "=", after_inventory.get(key, "")))
        return tuple(predicates)

    def _relation_delta_predicates(
        self,
        before: StructuredState,
        after: StructuredState,
    ) -> tuple[GroundedPredicate, ...]:
        before_counts = _relation_type_counts(before)
        after_counts = _relation_type_counts(after)
        predicates: list[GroundedPredicate] = []
        for relation_type in sorted(before_counts.keys() | after_counts.keys()):
            delta = int(after_counts.get(relation_type, 0) - before_counts.get(relation_type, 0))
            if delta == 0:
                continue
            predicates.append(
                GroundedPredicate("relation_count", relation_type, "delta", f"{delta:+d}")
            )
        return tuple(predicates)

    def _target_effect_predicates(
        self,
        transition: Transition,
        signature: ObjectSignature,
    ) -> tuple[GroundedPredicate, ...]:
        before_groups = _group_objects(transition.state.objects)
        after_groups = _group_objects(transition.next_state.objects)
        before_objects = before_groups.get(signature, [])
        after_objects = after_groups.get(signature, [])
        predicates: list[GroundedPredicate] = []
        matched_count = min(len(before_objects), len(after_objects))
        changed = False
        moved = False
        for index in range(matched_count):
            before_obj = before_objects[index]
            after_obj = after_objects[index]
            if _object_delta(before_obj, after_obj) > 0.0:
                changed = True
            if abs(after_obj.centroid[0] - before_obj.centroid[0]) > 1e-6 or abs(after_obj.centroid[1] - before_obj.centroid[1]) > 1e-6:
                moved = True
        if changed:
            predicates.append(GroundedPredicate("object", "target_effect", "=", "changed"))
        if moved:
            predicates.append(GroundedPredicate("object", "target_effect", "=", "moved"))
        if len(after_objects) < len(before_objects):
            predicates.append(GroundedPredicate("object", "target_effect", "=", "disappeared"))
        if len(after_objects) > len(before_objects):
            predicates.append(GroundedPredicate("object", "target_effect", "=", "appeared"))
        if not predicates:
            predicates.append(GroundedPredicate("object", "target_effect", "=", "stable"))
        return tuple(predicates)


def object_signature(obj: ObjectState) -> ObjectSignature:
    height = (obj.bbox[2] - obj.bbox[0]) + 1
    width = (obj.bbox[3] - obj.bbox[1]) + 1
    return (obj.color, obj.area, height, width, obj.tags)


def action_target_signatures(state: StructuredState, action: ActionName) -> tuple[ObjectSignature, ...]:
    context = build_action_schema_context(state.affordances, dict(state.action_roles))
    schema = build_action_schema(action, context)
    target_cells: set[tuple[int, int]] = set()
    if schema.action_type == "interact" and action in INTERACT_DELTAS:
        agent_cells = [cell for obj in state.objects if "agent" in obj.tags for cell in obj.cells]
        if agent_cells:
            dy, dx = INTERACT_DELTAS[action]
            target_cells = {(cell[0] + dy, cell[1] + dx) for cell in agent_cells}
    elif schema.action_type == "click" and schema.click is not None:
        click_x, click_y = schema.click
        target_cells = {(click_y, click_x)}
    if not target_cells:
        return ()
    signatures: list[ObjectSignature] = []
    seen: set[ObjectSignature] = set()
    for obj in state.objects:
        if "agent" in obj.tags:
            continue
        if not any(cell in target_cells for cell in obj.cells):
            continue
        signature = object_signature(obj)
        if signature in seen:
            continue
        seen.add(signature)
        signatures.append(signature)
    return tuple(signatures)


def _group_objects(objects: tuple[ObjectState, ...]) -> dict[ObjectSignature, list[ObjectState]]:
    groups: dict[ObjectSignature, list[ObjectState]] = defaultdict(list)
    for obj in objects:
        groups[object_signature(obj)].append(obj)
    for group in groups.values():
        group.sort(key=lambda item: (item.centroid[0], item.centroid[1], item.object_id))
    return groups


def _object_delta(before: ObjectState, after: ObjectState) -> float:
    return (
        abs(after.centroid[0] - before.centroid[0])
        + abs(after.centroid[1] - before.centroid[1])
        + abs(after.area - before.area)
    )


def _cell_distance(source: tuple[float, float], target: tuple[float, float]) -> float:
    return abs(source[0] - target[0]) + abs(source[1] - target[1])


def _relation_type_counts(state: StructuredState) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for relation in state.relations:
        counts[relation.relation_type] += 1
    return counts


def _signature_token(signature: ObjectSignature) -> str:
    color, area, height, width, tags = signature
    tag_part = ",".join(tags)
    return f"c{color}:a{area}:h{height}:w{width}:{tag_part}"


def _target_relation_label(action_type: str) -> str:
    if action_type == "interact":
        return "adjacent"
    if action_type == "click":
        return "clicked"
    if action_type == "select":
        return "selected"
    return "present"


def _consequence_desirability(predicate: GroundedPredicate) -> float:
    if predicate.predicate_type == "reward":
        return {
            "positive": 1.25,
            "zero": -0.1,
            "negative": -1.25,
        }.get(predicate.object, 0.0)
    if predicate.predicate_type == "transition":
        return {
            "changed": 0.55,
            "stable": -0.35,
        }.get(predicate.object, 0.0)
    if predicate.predicate_type == "object" and predicate.subject == "target_effect":
        return {
            "changed": 0.7,
            "moved": 0.45,
            "disappeared": 0.6,
            "appeared": 0.45,
            "stable": -0.35,
        }.get(predicate.object, 0.0)
    if predicate.predicate_type in {"flag", "inventory"}:
        return 0.35
    if predicate.predicate_type == "relation_count":
        if predicate.object.startswith("+"):
            return 0.2
        if predicate.object.startswith("-"):
            return -0.1
    return 0.0
