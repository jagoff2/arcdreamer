from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass, field
import math

import numpy as np

from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.types import ActionName, ObjectState, StructuredState, Transition
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


class EpisodeRuleInducer:
    def __init__(self) -> None:
        self.action_rules: dict[ActionName, ActionRuleStats] = {}
        self.family_rules: dict[str, ActionFamilyStats] = {}
        self.target_context_rules: dict[tuple[str, ObjectSignature], ActionFamilyStats] = {}
        self.signature_values: dict[ObjectSignature, float] = defaultdict(float)
        self.total_transitions = 0

    def clear(self) -> None:
        self.action_rules.clear()
        self.family_rules.clear()
        self.target_context_rules.clear()
        self.signature_values.clear()
        self.total_transitions = 0

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
        return action_score + family_score + progress_bonus + context_bonus

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
