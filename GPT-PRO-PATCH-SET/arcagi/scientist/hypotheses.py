"""Online causal hypothesis induction, testing, and Bayesian scoring.

The key design is that rules are treated as falsifiable hypotheses, not fixed
controllers.  The engine creates compact rule objects from observed transitions,
updates evidence online, and exposes uncertainty to the planner so that the agent
can choose diagnostic experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Any, Mapping

import numpy as np

from .types import (
    ActionName,
    ObjectToken,
    StructuredState,
    TransitionRecord,
    action_delta,
    action_family,
    action_target_to_grid_cell,
    grid_cell_to_action_coordinates,
    is_interact_action,
    is_move_action,
    is_selector_action,
    make_targeted_action,
    parse_action_target,
)


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


@dataclass
class Evidence:
    support: float = 0.0
    contradiction: float = 0.0
    trials: int = 0

    def update(self, *, support: float = 0.0, contradiction: float = 0.0) -> None:
        self.support += max(0.0, float(support))
        self.contradiction += max(0.0, float(contradiction))
        self.trials += 1

    @property
    def posterior(self) -> float:
        # Beta(1.2, 1.2) prior prevents early certainty.
        return float((self.support + 1.2) / (self.support + self.contradiction + 2.4))

    @property
    def confidence(self) -> float:
        total = self.support + self.contradiction
        return float(1.0 - np.exp(-0.25 * total))

    @property
    def entropy(self) -> float:
        p = _clip(self.posterior, 1e-6, 1.0 - 1e-6)
        return float(-(p * log(p) + (1.0 - p) * log(1.0 - p)))


@dataclass
class Hypothesis:
    hypothesis_id: str
    kind: str
    action_family: str
    params: dict[str, Any]
    description: str
    created_step: int
    evidence: Evidence = field(default_factory=Evidence)
    mdl_penalty: float = 0.0
    last_updated_step: int = 0

    @property
    def posterior(self) -> float:
        return _clip(self.evidence.posterior - self.mdl_penalty, 0.01, 0.99)

    @property
    def uncertainty(self) -> float:
        return self.evidence.entropy * (1.0 - min(self.evidence.confidence, 0.95))

    @property
    def utility_prior(self) -> float:
        if self.kind.startswith("reward") or self.kind.startswith("goal"):
            return 1.5
        if self.kind.startswith("action_moves"):
            return 0.65
        if self.kind.startswith("targeted"):
            return 0.55
        if self.kind.startswith("mode"):
            return 0.40
        return 0.25

    def observe(self, *, supported: bool, strength: float = 1.0, step: int = 0) -> None:
        if supported:
            self.evidence.update(support=strength)
        else:
            self.evidence.update(contradiction=strength)
        self.last_updated_step = step

    def as_tokens(self) -> tuple[str, ...]:
        tokens = [f"hyp:{self.kind}", f"af:{self.action_family}"]
        for key, value in sorted(self.params.items()):
            if isinstance(value, (str, int, float, bool)):
                tokens.append(f"{key}:{value}")
            elif isinstance(value, tuple):
                tokens.append(f"{key}:{','.join(map(str, value[:4]))}")
        tokens.append(f"p:{round(self.posterior, 2)}")
        return tuple(tokens)


@dataclass(frozen=True)
class HypothesisActionScore:
    expected_reward: float
    expected_change: float
    information_gain: float
    risk: float
    posterior_mass: float
    rationale: tuple[str, ...]


class HypothesisEngine:
    """Maintains and updates executable online theories."""

    def __init__(self, *, max_hypotheses: int = 512) -> None:
        self.max_hypotheses = int(max_hypotheses)
        self.hypotheses: dict[str, Hypothesis] = {}
        self.transition_count = 0
        self.positive_examples = 0
        self.recent_actions: list[ActionName] = []

    def reset_episode(self) -> None:
        self.hypotheses.clear()
        self.transition_count = 0
        self.positive_examples = 0
        self.recent_actions.clear()

    def _id(self, kind: str, action_fam: str, params: Mapping[str, Any]) -> str:
        payload = ";".join(f"{k}={params[k]}" for k in sorted(params))
        return f"{kind}|{action_fam}|{payload}"

    def _get_or_create(
        self,
        *,
        kind: str,
        action_fam: str,
        params: Mapping[str, Any],
        description: str,
        step: int,
        mdl_penalty: float = 0.0,
    ) -> Hypothesis:
        hid = self._id(kind, action_fam, params)
        hyp = self.hypotheses.get(hid)
        if hyp is not None:
            return hyp
        hyp = Hypothesis(
            hypothesis_id=hid,
            kind=kind,
            action_family=action_fam,
            params=dict(params),
            description=description,
            created_step=step,
            mdl_penalty=float(mdl_penalty),
        )
        self.hypotheses[hid] = hyp
        self._prune_if_needed()
        return hyp

    def _prune_if_needed(self) -> None:
        if len(self.hypotheses) <= self.max_hypotheses:
            return
        ranked = sorted(
            self.hypotheses.values(),
            key=lambda h: (h.posterior * h.evidence.confidence * h.utility_prior, -h.evidence.contradiction),
            reverse=True,
        )
        keep = {h.hypothesis_id for h in ranked[: self.max_hypotheses]}
        for hid in list(self.hypotheses):
            if hid not in keep:
                del self.hypotheses[hid]

    def observe_transition(self, record: TransitionRecord) -> None:
        self.transition_count += 1
        step = record.before.step_index
        fam = action_family(record.action)
        if record.delta.is_positive:
            self.positive_examples += 1
        self.recent_actions.append(record.action)
        if len(self.recent_actions) > 16:
            self.recent_actions.pop(0)

        # First update existing hypotheses against the new observation.
        for hyp in list(self.hypotheses.values()):
            self._update_hypothesis(hyp, record, fam=fam, step=step)

        # Then induce new explanations from effects.  Induction after updating
        # avoids instantly rewarding a hypothesis for the event that created it.
        self._induce_action_effects(record, fam=fam, step=step)
        self._induce_targeted_effects(record, fam=fam, step=step)
        self._induce_reward_hypotheses(record, fam=fam, step=step)
        self._induce_mode_hypotheses(record, fam=fam, step=step)

    def _update_hypothesis(self, hyp: Hypothesis, record: TransitionRecord, *, fam: str, step: int) -> None:
        if hyp.kind == "action_moves_object":
            if fam != hyp.action_family:
                return
            target_sig = str(hyp.params.get("object_signature"))
            expected_delta = tuple(hyp.params.get("delta", (0, 0)))
            supported = any(
                motion.before_signature == target_sig and tuple(motion.delta) == expected_delta
                for motion in record.delta.moved_objects
            )
            visible_relevant = any(motion.before_signature == target_sig for motion in record.delta.moved_objects)
            if supported:
                hyp.observe(supported=True, strength=1.0 + 0.25 * record.delta.changed_fraction, step=step)
            elif visible_relevant or record.delta.has_visible_effect:
                hyp.observe(supported=False, strength=0.35, step=step)

        elif hyp.kind == "targeted_action_changes_object":
            if fam != hyp.action_family:
                return
            target = action_target_to_grid_cell(record.action, record.before.frame.extras)
            if target is None:
                return
            expected_color = int(hyp.params.get("color", -999))
            changed = record.delta.changed_cells > 0
            touched_color = any(
                (record.before.object_by_id(oid) and record.before.object_by_id(oid).color == expected_color)
                or (record.after.object_by_id(oid) and record.after.object_by_id(oid).color == expected_color)
                for oid in record.delta.touched_objects
            )
            if changed and touched_color:
                hyp.observe(supported=True, strength=1.0, step=step)
            elif touched_color:
                hyp.observe(supported=False, strength=0.5, step=step)

        elif hyp.kind == "reward_when_touch_color":
            color = int(hyp.params.get("color", -999))
            relevant = self._action_touches_color(record, color) or self._state_contains_color(record.after, color)
            if not relevant:
                return
            if record.delta.is_positive:
                hyp.observe(supported=True, strength=1.5 + max(record.reward, 0.0), step=step)
            elif fam == hyp.action_family or is_interact_action(record.action) or is_move_action(record.action):
                hyp.observe(supported=False, strength=0.18, step=step)

        elif hyp.kind == "reward_when_state_has_color":
            color = int(hyp.params.get("color", -999))
            relevant = self._state_contains_color(record.after, color)
            if relevant and record.delta.is_positive:
                hyp.observe(supported=True, strength=1.2, step=step)
            elif relevant and (fam == hyp.action_family or record.delta.has_visible_effect):
                hyp.observe(supported=False, strength=0.12, step=step)

        elif hyp.kind == "mode_action_changes_dynamics":
            if fam == hyp.action_family and record.delta.has_visible_effect:
                hyp.observe(supported=True, strength=0.7, step=step)
            elif fam == hyp.action_family:
                hyp.observe(supported=False, strength=0.25, step=step)

    def _induce_action_effects(self, record: TransitionRecord, *, fam: str, step: int) -> None:
        if not record.delta.moved_objects:
            return
        for motion in record.delta.moved_objects:
            params = {
                "object_signature": motion.before_signature,
                "color": motion.color,
                "delta": tuple(motion.delta),
            }
            desc = f"action family {fam} moves object {motion.before_signature} by {motion.delta}"
            hyp = self._get_or_create(
                kind="action_moves_object",
                action_fam=fam,
                params=params,
                description=desc,
                step=step,
                mdl_penalty=0.02,
            )
            # The creating event is weakly credited because the same transition is
            # not a replication.  Repeated support is required for high posterior.
            hyp.observe(supported=True, strength=0.35, step=step)

    def _induce_targeted_effects(self, record: TransitionRecord, *, fam: str, step: int) -> None:
        if action_target_to_grid_cell(record.action, record.before.frame.extras) is None:
            return
        if not record.delta.has_visible_effect:
            return
        for oid in record.delta.touched_objects:
            obj = record.before.object_by_id(oid) or record.after.object_by_id(oid)
            if obj is None:
                continue
            params = {"color": obj.color, "signature": obj.signature}
            hyp = self._get_or_create(
                kind="targeted_action_changes_object",
                action_fam=fam,
                params=params,
                description=f"targeted {fam} changes object color c{obj.color}",
                step=step,
                mdl_penalty=0.03,
            )
            hyp.observe(supported=True, strength=0.45, step=step)

    def _induce_reward_hypotheses(self, record: TransitionRecord, *, fam: str, step: int) -> None:
        if not record.delta.is_positive:
            return
        touched_objects = [record.before.object_by_id(oid) or record.after.object_by_id(oid) for oid in record.delta.touched_objects]
        moved_after = [record.after.object_by_id(motion.after_id) for motion in record.delta.moved_objects]
        candidate_objects = [obj for obj in [*touched_objects, *moved_after, *record.after.objects[:6]] if obj is not None]
        seen_colors: set[int] = set()
        for obj in candidate_objects:
            if obj.color in seen_colors:
                continue
            seen_colors.add(obj.color)
            params = {"color": obj.color, "signature": obj.signature}
            hyp = self._get_or_create(
                kind="reward_when_touch_color",
                action_fam=fam,
                params=params,
                description=f"reward/progress follows contact or interaction with color c{obj.color}",
                step=step,
                mdl_penalty=0.04,
            )
            hyp.observe(supported=True, strength=1.25 + max(record.reward, record.delta.score_delta, 0.0), step=step)

            state_hyp = self._get_or_create(
                kind="reward_when_state_has_color",
                action_fam=fam,
                params={"color": obj.color},
                description=f"reward/progress occurs when state contains color c{obj.color}",
                step=step,
                mdl_penalty=0.08,
            )
            state_hyp.observe(supported=True, strength=0.8, step=step)

    def _induce_mode_hypotheses(self, record: TransitionRecord, *, fam: str, step: int) -> None:
        if not is_selector_action(record.action):
            return
        before_actions = set(record.before.available_actions)
        after_actions = set(record.after.available_actions)
        action_set_changed = before_actions != after_actions
        if record.delta.has_visible_effect or action_set_changed:
            params = {"changed_action_set": action_set_changed, "visible": record.delta.has_visible_effect}
            hyp = self._get_or_create(
                kind="mode_action_changes_dynamics",
                action_fam=fam,
                params=params,
                description=f"selector-like action {fam} changes latent mode or controls",
                step=step,
                mdl_penalty=0.05,
            )
            hyp.observe(supported=True, strength=0.8, step=step)

    def _state_contains_color(self, state: StructuredState, color: int) -> bool:
        return any(obj.color == color for obj in state.objects)

    def _action_touches_color(self, record: TransitionRecord, color: int) -> bool:
        for oid in record.delta.touched_objects:
            before = record.before.object_by_id(oid)
            after = record.after.object_by_id(oid)
            if (before is not None and before.color == color) or (after is not None and after.color == color):
                return True
        return False

    def score_action(self, state: StructuredState, action: ActionName) -> HypothesisActionScore:
        fam = action_family(action)
        expected_reward = 0.0
        expected_change = 0.0
        info_gain = 0.0
        risk = 0.0
        posterior_mass = 0.0
        rationale: list[str] = []
        target = action_target_to_grid_cell(action, state.frame.extras)

        for hyp in self.hypotheses.values():
            p = hyp.posterior
            relevant = fam == hyp.action_family
            if hyp.kind == "action_moves_object" and relevant:
                expected_change += p * 0.45
                posterior_mass += p
                if hyp.uncertainty > 0.05:
                    info_gain += hyp.uncertainty * 0.60
                    rationale.extend(("test", "movement", fam))

            elif hyp.kind == "targeted_action_changes_object" and relevant:
                color = int(hyp.params.get("color", -999))
                if target is not None:
                    near_color = any(_obj_near_target(obj, target, radius=1) and obj.color == color for obj in state.objects)
                    if near_color:
                        expected_change += p * 0.55
                        info_gain += hyp.uncertainty * 0.75
                        posterior_mass += p
                        rationale.extend(("test", "targeted_change", f"c{color}"))
                else:
                    info_gain += hyp.uncertainty * 0.30

            elif hyp.kind == "reward_when_touch_color":
                color = int(hyp.params.get("color", -999))
                color_present = any(obj.color == color for obj in state.objects)
                if not color_present:
                    continue
                if is_move_action(action) or is_interact_action(action) or fam == hyp.action_family:
                    expected_reward += p * hyp.utility_prior * 0.35
                    posterior_mass += p
                    if hyp.uncertainty > 0.04:
                        info_gain += hyp.uncertainty * 0.45
                    rationale.extend(("pursue", f"c{color}"))

            elif hyp.kind == "reward_when_state_has_color":
                color = int(hyp.params.get("color", -999))
                if any(obj.color == color for obj in state.objects):
                    expected_reward += p * 0.12
                    posterior_mass += p * 0.25

            elif hyp.kind == "mode_action_changes_dynamics" and relevant:
                expected_change += p * 0.20
                info_gain += hyp.uncertainty * 0.65
                posterior_mass += p
                rationale.extend(("test", "mode"))

            if relevant and hyp.posterior < 0.20 and hyp.evidence.contradiction > hyp.evidence.support + 1:
                risk += 0.10

        # Before enough evidence exists, keep raw exploration live.  This avoids
        # prematurely collapsing into a single movement loop.
        if self.transition_count < 12:
            if is_move_action(action) or is_interact_action(action) or is_selector_action(action):
                info_gain += 0.25 / (1.0 + self.transition_count)
            if target is not None:
                info_gain += 0.08

        return HypothesisActionScore(
            expected_reward=float(expected_reward),
            expected_change=float(expected_change),
            information_gain=float(info_gain),
            risk=float(risk),
            posterior_mass=float(posterior_mass),
            rationale=tuple(rationale[:12]),
        )

    def diagnostic_actions(self, state: StructuredState, legal_actions: tuple[ActionName, ...], *, limit: int = 32) -> tuple[ActionName, ...]:
        actions: list[ActionName] = []
        base_by_family = {action_family(a): a for a in legal_actions}
        uncertain = sorted(self.hypotheses.values(), key=lambda h: h.uncertainty * h.utility_prior, reverse=True)
        for hyp in uncertain[:24]:
            if hyp.action_family in base_by_family:
                base = base_by_family[hyp.action_family]
                if hyp.kind == "targeted_action_changes_object":
                    color = int(hyp.params.get("color", -999))
                    for obj in state.objects:
                        if obj.color == color:
                            r, c = obj.center_cell
                            tr, tc = grid_cell_to_action_coordinates(r, c, state.frame.extras)
                            actions.append(make_targeted_action(base, tr, tc))
                else:
                    actions.append(base)
            if len(actions) >= limit:
                break
        # Always include targeted probes of object centers if a click-like action exists.
        click_base = None
        for legal in legal_actions:
            if action_family(legal) in {"click", "action6", "interact_at", "select_at"}:
                click_base = legal
                break
        if click_base is not None:
            for obj in state.objects[:16]:
                r, c = obj.center_cell
                tr, tc = grid_cell_to_action_coordinates(r, c, state.frame.extras)
                actions.append(make_targeted_action(click_base, tr, tc))
                if len(actions) >= limit:
                    break
        return tuple(dict.fromkeys(actions[:limit]))


    def color_progress_priors(self) -> dict[int, float]:
        """Return online evidence that interacting with a color may advance the task."""

        priors: dict[int, float] = {}
        for hyp in self.hypotheses.values():
            if hyp.kind not in {"reward_when_touch_color", "reward_when_state_has_color"}:
                continue
            try:
                color = int(hyp.params.get("color"))
            except Exception:
                continue
            weight = hyp.posterior * hyp.utility_prior * (0.5 + hyp.evidence.confidence)
            priors[color] = max(priors.get(color, 0.0), float(weight))
        return priors

    def controlled_object_colors(self) -> dict[int, float]:
        """Return colors that evidence suggests are moved by primitive actions."""

        colors: dict[int, float] = {}
        for hyp in self.hypotheses.values():
            if hyp.kind != "action_moves_object":
                continue
            try:
                color = int(hyp.params.get("color"))
            except Exception:
                continue
            colors[color] = max(colors.get(color, 0.0), float(hyp.posterior * (0.5 + hyp.evidence.confidence)))
        return colors

    def credible_hypotheses(self, *, min_posterior: float = 0.55, limit: int = 16) -> tuple[Hypothesis, ...]:
        ranked = sorted(
            (h for h in self.hypotheses.values() if h.posterior >= min_posterior),
            key=lambda h: h.posterior * h.evidence.confidence * h.utility_prior,
            reverse=True,
        )
        return tuple(ranked[:limit])

    def uncertain_hypotheses(self, *, limit: int = 12) -> tuple[Hypothesis, ...]:
        ranked = sorted(self.hypotheses.values(), key=lambda h: h.uncertainty * h.utility_prior, reverse=True)
        return tuple(ranked[:limit])


def _obj_near_target(obj: ObjectToken, target: tuple[int, int], *, radius: int) -> bool:
    r, c = target
    orow, ocol = obj.center_cell
    if abs(orow - r) + abs(ocol - c) <= radius:
        return True
    r0, c0, r1, c1 = obj.bbox
    return (r0 - radius) <= r <= (r1 + radius) and (c0 - radius) <= c <= (c1 + radius)
