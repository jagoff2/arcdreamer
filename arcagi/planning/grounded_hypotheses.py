from __future__ import annotations
from collections import defaultdict
from dataclasses import dataclass
import numpy as np
from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.spatial_workspace import INTERACT_DELTAS
from arcagi.core.types import ActionName, ObjectState, StructuredClaim, StructuredState, Transition

ObjectSignature = tuple[int, int, int, int, tuple[str, ...]]

@dataclass(frozen=True)
class GroundedPredicate:
    predicate_type: str
    subject: str
    relation: str
    object: str
    family: str
    desirability: float = 0.0
    def identity(self) -> tuple[str, str, str, str]:
        return (self.predicate_type, self.subject, self.relation, self.object)
    def render(self) -> str:
        return f"{self.predicate_type}:{self.subject}{self.relation}{self.object}"
    def as_claim(self, *, subject_prefix: str, confidence: float, evidence: float, salience: float) -> StructuredClaim:
        return StructuredClaim(
            claim_type="hypothesis",
            subject=subject_prefix,
            relation="predicts",
            object=self.render(),
            confidence=confidence,
            evidence=evidence,
            salience=salience,
        )

@dataclass(frozen=True)
class HypothesisScope:
    action: ActionName
    predicates: tuple[GroundedPredicate, ...]
    target_signature: ObjectSignature | None = None
    def key(self) -> tuple[tuple[str, str, str, str], ...]:
        return tuple(p.identity() for p in self.predicates)
    def label(self) -> str:
        return ";".join(p.render() for p in self.predicates)

@dataclass
class GroundedHypothesis:
    scope: HypothesisScope
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
    @property
    def score(self) -> float:
        return abs(self.weight) * min(1.5, 0.35 + (0.15 * self.evidence))

class GroundedHypothesisEngine:
    def __init__(self) -> None:
        self.hypotheses: dict[tuple[tuple[tuple[str, str, str, str], ...], tuple[str, str, str, str]], GroundedHypothesis] = {}
        self.by_scope: dict[tuple[tuple[str, str, str, str], ...], set[tuple[tuple[tuple[str, str, str, str], ...], tuple[str, str, str, str]]]] = defaultdict(set)
        self.by_scope_family: dict[tuple[tuple[str, str, str, str], ...], dict[str, set[tuple[tuple[tuple[str, str, str, str], ...], tuple[str, str, str, str]]]]] = defaultdict(lambda: defaultdict(set))
    def clear(self) -> None:
        self.hypotheses.clear(); self.by_scope.clear(); self.by_scope_family.clear()
    def scope_candidates(self, state: StructuredState, action: ActionName) -> tuple[HypothesisScope, ...]:
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        schema = build_action_schema(action, context)
        base = HypothesisScope(
            action=action,
            predicates=(
                GroundedPredicate("action","type","=",schema.action_type,"scope:action_type"),
                GroundedPredicate("action","role","=",schema.role or "unknown","scope:action_role"),
            ),
        )
        scopes = [base]
        relation_label = _target_relation_label(schema.action_type)
        for signature in action_target_signatures(state, action):
            scopes.append(HypothesisScope(
                action=action,
                predicates=base.predicates + (
                    GroundedPredicate("object","target_signature","=",_signature_token(signature),"scope:target_signature"),
                    GroundedPredicate("relation","agent_target","=",relation_label,"scope:agent_target"),
                ),
                target_signature=signature,
            ))
        return tuple(scopes)
    def record_transition(self, transition: Transition) -> None:
        for scope in self.scope_candidates(transition.state, transition.action):
            observed = self._observed_consequences(transition, scope.target_signature)
            self._update_scope(scope, observed)
    def applicable_hypotheses(self, state: StructuredState, action: ActionName) -> tuple[GroundedHypothesis, ...]:
        ids = set()
        for scope in self.scope_candidates(state, action):
            ids.update(self.by_scope.get(scope.key(), set()))
        items = [self.hypotheses[h] for h in ids]
        items.sort(key=lambda h: (h.score, h.support, -h.contradiction), reverse=True)
        return tuple(items)
    def top_claims(self, *, state: StructuredState | None = None, action: ActionName | None = None, limit: int = 4, min_confidence: float = 0.55) -> tuple[StructuredClaim, ...]:
        items = tuple(self.hypotheses.values()) if state is None or action is None else self.applicable_hypotheses(state, action)
        ranked = [h for h in items if h.confidence >= min_confidence]
        ranked.sort(key=lambda h: (h.score, h.consequence.desirability, h.support), reverse=True)
        return tuple(h.consequence.as_claim(subject_prefix=h.scope.label(), confidence=h.confidence, evidence=h.evidence, salience=h.score) for h in ranked[:limit])
    def predictive_claims_for_state(self, state: StructuredState, *, limit: int = 6, per_action: int = 2, min_confidence: float = 0.55) -> tuple[StructuredClaim, ...]:
        claims: list[tuple[float, StructuredClaim]] = []
        for action in state.affordances:
            for h in self.applicable_hypotheses(state, action)[:max(per_action,1)]:
                if h.confidence < min_confidence:
                    continue
                claims.append((h.score, h.consequence.as_claim(subject_prefix=f"action={action}|{h.scope.label()}", confidence=h.confidence, evidence=h.evidence, salience=h.score)))
        claims.sort(key=lambda x: x[0], reverse=True)
        return tuple(claim for _, claim in claims[:limit])
    def action_hypothesis_bonus(self, state: StructuredState, action: ActionName) -> float:
        bonus = 0.0
        for h in self.applicable_hypotheses(state, action)[:10]:
            if h.consequence.desirability == 0.0:
                continue
            bonus += h.consequence.desirability * h.weight * min(1.25, 0.3 + (0.12 * h.evidence))
        return max(min(bonus, 3.0), -3.0)
    def action_diagnostic_bonus(self, state: StructuredState, action: ActionName) -> float:
        grouped: dict[str, list[GroundedHypothesis]] = defaultdict(list)
        for h in self.applicable_hypotheses(state, action):
            grouped[h.consequence.family].append(h)
        bonus = 0.0
        for group in grouped.values():
            if len(group) < 2:
                continue
            ranked = sorted(group, key=lambda h: h.confidence, reverse=True)
            if ranked[0].evidence < 1.0 or ranked[1].evidence < 1.0:
                continue
            margin = abs(ranked[0].confidence - ranked[1].confidence)
            ambiguity = max(0.0, 0.35 - margin) / 0.35
            bonus += 0.3 * ambiguity
        return min(bonus, 0.9)
    def _update_scope(self, scope: HypothesisScope, observed: dict[str, tuple[GroundedPredicate, ...]]) -> None:
        skey = scope.key()
        for family, predicates in observed.items():
            existing = set(self.by_scope_family.get(skey, {}).get(family, set()))
            observed_ids = {p.identity() for p in predicates}
            for hid in existing:
                h = self.hypotheses[hid]
                h.observations += 1
                if h.consequence.identity() in observed_ids: h.support += 1.0
                else: h.contradiction += 1.0
            for predicate in predicates:
                hid = (skey, predicate.identity())
                if hid in self.hypotheses: continue
                h = GroundedHypothesis(scope=scope, consequence=predicate, support=1.0, observations=1)
                self.hypotheses[hid] = h
                self.by_scope[skey].add(hid)
                self.by_scope_family[skey][family].add(hid)
    def _observed_consequences(self, transition: Transition, target_signature: ObjectSignature | None) -> dict[str, tuple[GroundedPredicate, ...]]:
        obs: dict[str, tuple[GroundedPredicate, ...]] = {}
        for p in self._reward_predicates(transition.reward): obs[p.family] = (p,)
        for p in self._state_change_predicates(transition): obs[p.family] = (p,)
        for p in self._relation_count_predicates(transition.state, transition.next_state): obs[p.family] = (p,)
        for p in self._hidden_state_predicates(transition.state, transition.next_state): obs[p.family] = (p,)
        if target_signature is not None:
            for p in self._target_effect_predicates(transition, target_signature): obs[p.family] = (p,)
        return obs
    def _reward_predicates(self, reward: float) -> tuple[GroundedPredicate, ...]:
        bucket = "positive" if reward > 0.0 else "negative" if reward < 0.0 else "zero"
        desirability = {"positive":1.35,"zero":-0.1,"negative":-1.35}[bucket]
        return (GroundedPredicate("reward","episode","=",bucket,"reward:episode",desirability),)
    def _state_change_predicates(self, transition: Transition) -> tuple[GroundedPredicate, ...]:
        bucket = "changed" if _state_delta(transition.state, transition.next_state) >= 0.05 else "stable"
        desirability = {"changed":0.55,"stable":-0.35}[bucket]
        return (GroundedPredicate("transition","state_change","=",bucket,"transition:state_change",desirability),)
    def _hidden_state_predicates(self, before: StructuredState, after: StructuredState) -> tuple[GroundedPredicate, ...]:
        predicates = []
        bf, af = before.flags_dict(), after.flags_dict()
        for key in sorted(bf.keys() | af.keys()):
            value = af.get(key, "<missing>")
            desirability = 0.35 if value not in {"0","", "<missing>"} else 0.05
            predicates.append(GroundedPredicate("flag", key, "=", value, f"flag:{key}", desirability))
        bi, ai = before.inventory_dict(), after.inventory_dict()
        for key in sorted(bi.keys() | ai.keys()):
            value = ai.get(key, "<missing>")
            desirability = 0.2 if value not in {"","<missing>"} else 0.0
            predicates.append(GroundedPredicate("inventory", key, "=", value, f"inventory:{key}", desirability))
        return tuple(predicates)
    def _relation_count_predicates(self, before: StructuredState, after: StructuredState) -> tuple[GroundedPredicate, ...]:
        bc, ac = _relation_type_counts(before), _relation_type_counts(after)
        preds = []
        for rtype in sorted(bc.keys() | ac.keys()):
            delta = int(ac.get(rtype, 0) - bc.get(rtype, 0))
            desirability = 0.15 if delta > 0 else -0.05 if delta < 0 else 0.0
            preds.append(GroundedPredicate("relation_count", rtype, "delta", f"{delta:+d}", f"relation_count:{rtype}", desirability))
        return tuple(preds)
    def _target_effect_predicates(self, transition: Transition, signature: ObjectSignature) -> tuple[GroundedPredicate, ...]:
        before_groups, after_groups = _group_objects(transition.state.objects), _group_objects(transition.next_state.objects)
        before_objects, after_objects = before_groups.get(signature, []), after_groups.get(signature, [])
        matched = min(len(before_objects), len(after_objects))
        changed = False; moved = False; move_y = 0.0; move_x = 0.0
        for i in range(matched):
            b, a = before_objects[i], after_objects[i]
            if _object_delta(b, a) > 0.0: changed = True
            dy, dx = a.centroid[0] - b.centroid[0], a.centroid[1] - b.centroid[1]
            move_y += dy; move_x += dx
            if abs(dy) > 1e-6 or abs(dx) > 1e-6: moved = True
        presence = "disappeared" if len(after_objects) < len(before_objects) else "appeared" if len(after_objects) > len(before_objects) else "stable"
        return (
            GroundedPredicate("object","target_presence","=",presence,"object:target_presence",{"stable":-0.1,"disappeared":0.7,"appeared":0.35}[presence]),
            GroundedPredicate("object","target_delta","=","changed" if changed or presence != "stable" else "unchanged","object:target_delta",0.55 if changed or presence != "stable" else -0.35),
            GroundedPredicate("object","target_motion","=","moved" if moved else "static","object:target_motion",0.25 if moved else -0.05),
            GroundedPredicate("object","target_motion_direction","=",_direction_bucket(move_y, move_x),"object:target_motion_direction",0.0),
        )

def object_signature(obj: ObjectState) -> ObjectSignature:
    return (obj.color, obj.area, (obj.bbox[2] - obj.bbox[0]) + 1, (obj.bbox[3] - obj.bbox[1]) + 1, obj.tags)

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
    if not target_cells: return ()
    signatures = []; seen: set[ObjectSignature] = set()
    for obj in state.objects:
        if "agent" in obj.tags: continue
        if not any(cell in target_cells for cell in obj.cells): continue
        signature = object_signature(obj)
        if signature in seen: continue
        seen.add(signature); signatures.append(signature)
    return tuple(signatures)

def _group_objects(objects: tuple[ObjectState, ...]) -> dict[ObjectSignature, list[ObjectState]]:
    groups: dict[ObjectSignature, list[ObjectState]] = defaultdict(list)
    for obj in objects: groups[object_signature(obj)].append(obj)
    for group in groups.values(): group.sort(key=lambda item: (item.centroid[0], item.centroid[1], item.object_id))
    return groups

def _object_delta(before: ObjectState, after: ObjectState) -> float:
    return abs(after.centroid[0] - before.centroid[0]) + abs(after.centroid[1] - before.centroid[1]) + abs(after.area - before.area)

def _relation_type_counts(state: StructuredState) -> dict[str, int]:
    counts: dict[str, int] = defaultdict(int)
    for relation in state.relations: counts[relation.relation_type] += 1
    return counts

def _state_delta(before: StructuredState, after: StructuredState) -> float:
    return float(np.linalg.norm(after.transition_vector() - before.transition_vector()))

def _signature_token(signature: ObjectSignature) -> str:
    color, area, height, width, tags = signature
    return f"c{color}:a{area}:h{height}:w{width}:{','.join(tags)}"

def _target_relation_label(action_type: str) -> str:
    return "adjacent" if action_type == "interact" else "clicked" if action_type == "click" else "selected" if action_type == "select" else "present"

def _direction_bucket(delta_y: float, delta_x: float) -> str:
    if abs(delta_y) < 1e-6 and abs(delta_x) < 1e-6: return "none"
    if abs(delta_y) >= abs(delta_x): return "down" if delta_y > 0.0 else "up"
    return "right" if delta_x > 0.0 else "left"
