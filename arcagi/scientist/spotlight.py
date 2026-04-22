"""Action-level spotlight workspace for the scientist agent.

This module adds a bounded serial control layer on top of the existing
hypothesis engine, world model, memory, and planner candidate generation. The
goal is generic online control discipline: selector-like or coordinate-bearing
actions are allowed to create latent control state without being treated as
immediate failures just because the visible grid does not change right away.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from math import exp
from typing import Any, Iterable, Mapping

from .types import (
    ActionDecision,
    ActionName,
    StructuredState,
    TransitionRecord,
    action_family,
    combined_progress_signal,
    is_interact_action,
    is_move_action,
    is_selector_action,
    parse_action_target,
)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def _family(action: ActionName) -> str:
    try:
        return str(action_family(action)).lower()
    except Exception:
        text = str(action).lower()
        if text.startswith("click"):
            return "click"
        if text.startswith("action6"):
            return "action6"
        if text.startswith("action5"):
            return "action5"
        if text in {"up", "down", "left", "right", "action1", "action2", "action3", "action4"}:
            return "move"
        return text.split(":", 1)[0]


def _is_binding_action(action: ActionName) -> bool:
    """Return true for actions whose useful effect may only appear downstream."""

    text = str(action).lower()
    fam = _family(action)
    if fam in {"click", "action6", "select", "select_at", "interact_at", "targeted", "action5"}:
        return True
    if text.startswith(("click", "select", "action5", "action6")):
        return True
    try:
        return bool(is_selector_action(action))
    except Exception:
        return False


def _is_probe_action(action: ActionName) -> bool:
    """Return true for actions suitable for testing a prior latent binder."""

    if _is_binding_action(action):
        return False
    try:
        if is_move_action(action) or is_interact_action(action):
            return True
    except Exception:
        pass
    fam = _family(action)
    return fam in {"move", "up", "down", "left", "right", "interact", "action1", "action2", "action3", "action4"}


def _normalise(values: Mapping[ActionName, float]) -> dict[ActionName, float]:
    if not values:
        return {}
    finite = {action: _safe_float(value) for action, value in values.items()}
    lo = min(finite.values())
    hi = max(finite.values())
    if hi <= lo + 1e-9:
        return {action: 0.0 for action in finite}
    return {action: (value - lo) / (hi - lo) for action, value in finite.items()}


@dataclass(frozen=True)
class SpotlightConfig:
    """Weights and limits for the bounded action workspace."""

    reward_weight: float = 2.20
    information_weight: float = 1.85
    change_weight: float = 0.65
    uncertainty_weight: float = 0.45
    memory_weight: float = 0.65
    coverage_weight: float = 0.95
    binder_weight: float = 1.15
    post_binder_probe_weight: float = 2.20
    contradiction_penalty: float = 1.25
    no_effect_penalty: float = 0.85
    repeat_penalty: float = 0.45
    commitment_bonus: float = 0.40
    max_candidates: int = 96
    trace_capacity: int = 256
    selector_probe_horizon: int = 4
    selector_null_grace: int = 1
    max_commit_steps: int = 5
    min_commit_steps: int = 1
    switch_margin: float = 0.20
    confidence_temperature: float = 0.75


@dataclass
class SpotlightCandidate:
    action: ActionName
    score: float
    components: dict[str, float]
    intent_kind: str
    target: str | None
    hypothesis_id: str | None
    predicted_reward: float
    predicted_change: float
    rationale: tuple[str, ...] = ()


@dataclass
class SpotlightIntention:
    intention_id: int
    action: ActionName
    intent_kind: str
    target: str | None
    hypothesis_id: str | None
    created_step: int
    expires_step: int
    min_until_step: int
    predicted_reward: float
    predicted_change: float
    falsifier: str
    support: float = 0.0
    contradiction: float = 0.0
    probe_count: int = 0

    @property
    def confidence(self) -> float:
        total = self.support + self.contradiction
        if total <= 1e-9:
            return 0.0
        return float(self.support / total)


@dataclass
class PendingBinderProbe:
    binder_action: ActionName
    target: str | None
    created_step: int
    expires_step: int
    before_fingerprint: str
    immediate_visible_effect: bool
    probes_taken: int = 0


@dataclass
class SpotlightBroadcast:
    step: int
    action: ActionName
    focus: str
    target: str | None
    expected_reward: float
    expected_change: float
    score: float
    components: dict[str, float]
    falsifier: str
    workspace_slots: dict[str, str]


class ActionSpotlight:
    """Serial action-level workspace with explicit short-horizon commitments."""

    def __init__(self, config: SpotlightConfig | None = None) -> None:
        self.config = config or SpotlightConfig()
        self._intention_counter = 0
        self.active_intention: SpotlightIntention | None = None
        self.pending_binder_probe: PendingBinderProbe | None = None
        self.state_action_visits: defaultdict[tuple[str, ActionName], int] = defaultdict(int)
        self.abstract_action_visits: defaultdict[tuple[str, str], int] = defaultdict(int)
        self.global_action_visits: defaultdict[ActionName, int] = defaultdict(int)
        self.no_effect_counts: defaultdict[tuple[str, ActionName], int] = defaultdict(int)
        self.no_effect_family_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        self.contradiction_counts: defaultdict[tuple[str, ActionName], float] = defaultdict(float)
        self.binding_success: defaultdict[tuple[str, str], int] = defaultdict(int)
        self.binding_failure: defaultdict[tuple[str, str], int] = defaultdict(int)
        self.last_surprise: float = 0.0
        self.last_broadcast: SpotlightBroadcast | None = None
        self.trace: deque[dict[str, Any]] = deque(maxlen=self.config.trace_capacity)

    def reset_episode(self) -> None:
        self._intention_counter = 0
        self.active_intention = None
        self.pending_binder_probe = None
        self.state_action_visits.clear()
        self.abstract_action_visits.clear()
        self.global_action_visits.clear()
        self.no_effect_counts.clear()
        self.no_effect_family_counts.clear()
        self.contradiction_counts.clear()
        self.binding_success.clear()
        self.binding_failure.clear()
        self.last_surprise = 0.0
        self.last_broadcast = None
        self.trace.clear()

    def choose_action(
        self,
        state: StructuredState,
        *,
        planner: Any,
        engine: Any,
        world_model: Any,
        memory: Any,
        language: Any,
    ) -> ActionDecision:
        candidates = self._candidate_actions(state, planner=planner, engine=engine)
        if not candidates:
            raise RuntimeError("ActionSpotlight received no candidate actions")

        lang_tokens = self._language_tokens(language, state, engine)
        scored = self._score_candidates(
            state,
            candidates,
            engine=engine,
            world_model=world_model,
            memory=memory,
            lang_tokens=lang_tokens,
        )
        if not scored:
            fallback = candidates[0]
            scored = [
                SpotlightCandidate(
                    action=fallback,
                    score=0.0,
                    components={},
                    intent_kind="fallback_probe",
                    target=None,
                    hypothesis_id=None,
                    predicted_reward=0.0,
                    predicted_change=0.0,
                    rationale=("fallback because no subsystem scored candidates",),
                )
            ]

        chosen = self._select_with_commitment(state, scored)
        intention = self._broadcast_intention(state, chosen)
        components = dict(chosen.components)
        components["spotlight_score"] = float(chosen.score)
        components["spotlight_confidence"] = self._confidence_from_margin(scored)
        components["active_intention_confidence"] = float(intention.confidence)

        language_items = self._workspace_language(language, engine, chosen, intention)
        self.state_action_visits[(state.exact_fingerprint, chosen.action)] += 1
        self.abstract_action_visits[(state.abstract_fingerprint, _family(chosen.action))] += 1
        self.global_action_visits[chosen.action] += 1

        return ActionDecision(
            action=chosen.action,
            score=float(chosen.score),
            components=components,
            language=language_items,
            candidate_count=len(candidates),
            chosen_reason="; ".join(chosen.rationale[:4]) or chosen.intent_kind,
        )

    def notify_transition(self, *, record: TransitionRecord, engine: Any | None = None) -> None:
        progress = combined_progress_signal(record.reward, record.delta.score_delta)
        visible_effect = bool(record.delta.has_visible_effect or abs(progress) > 1e-8)
        fam = _family(record.action)
        exact_key = (record.before.exact_fingerprint, record.action)
        abstract_family_key = (record.before.abstract_fingerprint, fam)
        predicted_reward = 0.0
        predicted_change = 0.0
        if self.active_intention is not None and self.active_intention.action == record.action:
            predicted_reward = self.active_intention.predicted_reward
            predicted_change = self.active_intention.predicted_change

        surprise = self._transition_surprise(
            progress=progress,
            visible_effect=visible_effect,
            predicted_reward=predicted_reward,
            predicted_change=predicted_change,
            changed_fraction=_safe_float(getattr(record.delta, "changed_fraction", 0.0)),
        )
        self.last_surprise = surprise

        binding_action = _is_binding_action(record.action)
        if binding_action:
            self.pending_binder_probe = PendingBinderProbe(
                binder_action=record.action,
                target=self._target_text(record.action),
                created_step=record.before.step_index,
                expires_step=record.before.step_index + self.config.selector_probe_horizon,
                before_fingerprint=record.before.exact_fingerprint,
                immediate_visible_effect=visible_effect,
            )

        probe_supported = self._update_pending_binder_probe(record, visible_effect=visible_effect, progress=progress)

        contradiction = False
        if self.active_intention is not None and self.active_intention.action == record.action:
            contradiction = self._contradicted(self.active_intention, visible_effect=visible_effect, progress=progress)
            if contradiction and binding_action:
                contradiction = False
            if contradiction:
                self.active_intention.contradiction += 1.0 + surprise
                self.contradiction_counts[exact_key] += 1.0 + surprise
            elif visible_effect or probe_supported or progress > 0.0:
                self.active_intention.support += 1.0 + max(0.0, progress)
            self.active_intention.probe_count += 1

        if not visible_effect and not binding_action:
            self.no_effect_counts[exact_key] += 1
            self.no_effect_family_counts[abstract_family_key] += 1

        if self.active_intention is not None:
            expired = record.after.step_index >= self.active_intention.expires_step
            if contradiction or expired or progress > 0.0:
                self.active_intention = None

        self.trace.append(
            {
                "step": int(record.before.step_index),
                "action": str(record.action),
                "family": fam,
                "visible_effect": bool(visible_effect),
                "progress": float(progress),
                "surprise": float(surprise),
                "binding_action": bool(binding_action),
                "pending_binder_probe": None if self.pending_binder_probe is None else str(self.pending_binder_probe.binder_action),
                "contradiction": bool(contradiction),
            }
        )

    def diagnostics(self) -> dict[str, Any]:
        return {
            "active_intention": None if self.active_intention is None else self._intention_dict(self.active_intention),
            "pending_binder_probe": None
            if self.pending_binder_probe is None
            else {
                "binder_action": str(self.pending_binder_probe.binder_action),
                "target": self.pending_binder_probe.target,
                "created_step": self.pending_binder_probe.created_step,
                "expires_step": self.pending_binder_probe.expires_step,
                "probes_taken": self.pending_binder_probe.probes_taken,
                "immediate_visible_effect": self.pending_binder_probe.immediate_visible_effect,
            },
            "last_surprise": self.last_surprise,
            "last_broadcast": None
            if self.last_broadcast is None
            else {
                "step": self.last_broadcast.step,
                "action": str(self.last_broadcast.action),
                "focus": self.last_broadcast.focus,
                "target": self.last_broadcast.target,
                "expected_reward": self.last_broadcast.expected_reward,
                "expected_change": self.last_broadcast.expected_change,
                "score": self.last_broadcast.score,
                "falsifier": self.last_broadcast.falsifier,
                "workspace_slots": dict(self.last_broadcast.workspace_slots),
            },
            "no_effect_count_total": int(sum(self.no_effect_counts.values())),
            "binding_success_total": int(sum(self.binding_success.values())),
            "binding_failure_total": int(sum(self.binding_failure.values())),
            "trace_tail": list(self.trace)[-8:],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "no_effect_counts": [(list(key), int(value)) for key, value in self.no_effect_counts.items()],
            "no_effect_family_counts": [(list(key), int(value)) for key, value in self.no_effect_family_counts.items()],
            "contradiction_counts": [(list(key), float(value)) for key, value in self.contradiction_counts.items()],
            "binding_success": [(list(key), int(value)) for key, value in self.binding_success.items()],
            "binding_failure": [(list(key), int(value)) for key, value in self.binding_failure.items()],
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self.no_effect_counts.clear()
        self.no_effect_family_counts.clear()
        self.contradiction_counts.clear()
        self.binding_success.clear()
        self.binding_failure.clear()
        for key, value in state.get("no_effect_counts", []):
            self.no_effect_counts[(str(key[0]), str(key[1]))] = int(value)
        for key, value in state.get("no_effect_family_counts", []):
            self.no_effect_family_counts[(str(key[0]), str(key[1]))] = int(value)
        for key, value in state.get("contradiction_counts", []):
            self.contradiction_counts[(str(key[0]), str(key[1]))] = float(value)
        for key, value in state.get("binding_success", []):
            self.binding_success[(str(key[0]), str(key[1]))] = int(value)
        for key, value in state.get("binding_failure", []):
            self.binding_failure[(str(key[0]), str(key[1]))] = int(value)

    def _candidate_actions(self, state: StructuredState, *, planner: Any, engine: Any) -> tuple[ActionName, ...]:
        candidates: Iterable[ActionName]
        if hasattr(planner, "candidate_actions"):
            candidates = planner.candidate_actions(state, engine=engine)
        else:
            candidates = tuple(getattr(state, "available_actions", ()) or ())
        deduped = tuple(dict.fromkeys(candidates))
        return deduped[: self.config.max_candidates]

    def _score_candidates(
        self,
        state: StructuredState,
        candidates: tuple[ActionName, ...],
        *,
        engine: Any,
        world_model: Any,
        memory: Any,
        lang_tokens: tuple[str, ...],
    ) -> list[SpotlightCandidate]:
        raw: dict[str, dict[ActionName, float]] = {
            "reward": {},
            "info": {},
            "change": {},
            "uncertainty": {},
            "memory": {},
            "coverage": {},
            "binder": {},
            "probe": {},
            "penalty": {},
        }
        rationale_by_action: dict[ActionName, tuple[str, ...]] = {}
        hyp_id_by_action: dict[ActionName, str | None] = {}

        for action in candidates:
            hyp_score = self._hypothesis_score(engine, state, action)
            world = self._world_prediction(world_model, state, action)
            raw["reward"][action] = hyp_score["expected_reward"] + world["reward_mean"]
            raw["info"][action] = hyp_score["information_gain"]
            raw["change"][action] = hyp_score["expected_change"] + world["change_mean"]
            raw["uncertainty"][action] = world["total_uncertainty"]
            raw["memory"][action] = self._memory_bonus(memory, state, action, lang_tokens)
            raw["coverage"][action] = self._coverage_bonus(state, action)
            raw["binder"][action] = self._binder_bonus(action)
            raw["probe"][action] = self._post_binder_probe_bonus(action, state.step_index)
            raw["penalty"][action] = self._penalty(state, action)
            rationale_by_action[action] = tuple(hyp_score.get("rationale", ()))
            hyp_id_by_action[action] = hyp_score.get("hypothesis_id")

        norm = {name: _normalise(values) for name, values in raw.items() if name != "penalty"}
        scored: list[SpotlightCandidate] = []
        for action in candidates:
            reward = norm.get("reward", {}).get(action, 0.0)
            info = norm.get("info", {}).get(action, 0.0)
            change = norm.get("change", {}).get(action, 0.0)
            uncertainty = norm.get("uncertainty", {}).get(action, 0.0)
            memory_bonus = norm.get("memory", {}).get(action, 0.0)
            coverage = norm.get("coverage", {}).get(action, 0.0)
            binder = norm.get("binder", {}).get(action, 0.0)
            probe = norm.get("probe", {}).get(action, 0.0)
            penalty = raw["penalty"].get(action, 0.0)
            commitment = self._commitment_bonus(action, state)
            components = {
                "expected_reward": raw["reward"].get(action, 0.0),
                "information_gain": raw["info"].get(action, 0.0),
                "expected_change": raw["change"].get(action, 0.0),
                "world_uncertainty": raw["uncertainty"].get(action, 0.0),
                "memory_bonus": raw["memory"].get(action, 0.0),
                "coverage": raw["coverage"].get(action, 0.0),
                "binder_bonus": raw["binder"].get(action, 0.0),
                "post_binder_probe_bonus": raw["probe"].get(action, 0.0),
                "commitment_bonus": commitment,
                "penalty": penalty,
            }
            score = (
                self.config.reward_weight * reward
                + self.config.information_weight * info
                + self.config.change_weight * change
                + self.config.uncertainty_weight * uncertainty
                + self.config.memory_weight * memory_bonus
                + self.config.coverage_weight * coverage
                + self.config.binder_weight * binder
                + self.config.post_binder_probe_weight * probe
                + self.config.commitment_bonus * commitment
                - penalty
            )
            scored.append(
                SpotlightCandidate(
                    action=action,
                    score=float(score),
                    components=components,
                    intent_kind=self._intent_kind(action, components),
                    target=self._target_text(action),
                    hypothesis_id=hyp_id_by_action.get(action),
                    predicted_reward=components["expected_reward"],
                    predicted_change=components["expected_change"],
                    rationale=rationale_by_action.get(action, ()) or self._default_rationale(action, components),
                )
            )
        scored.sort(key=lambda candidate: candidate.score, reverse=True)
        return scored

    def _hypothesis_score(self, engine: Any, state: StructuredState, action: ActionName) -> dict[str, Any]:
        try:
            score = engine.score_action(state, action)
        except Exception:
            return {"expected_reward": 0.0, "expected_change": 0.0, "information_gain": 0.0, "rationale": ()}
        return {
            "expected_reward": _safe_float(getattr(score, "expected_reward", 0.0)),
            "expected_change": _safe_float(getattr(score, "expected_change", 0.0)),
            "information_gain": _safe_float(getattr(score, "information_gain", 0.0)),
            "risk": _safe_float(getattr(score, "risk", 0.0)),
            "posterior_mass": _safe_float(getattr(score, "posterior_mass", 0.0)),
            "rationale": tuple(getattr(score, "rationale", ()) or ()),
            "hypothesis_id": None,
        }

    def _world_prediction(self, world_model: Any, state: StructuredState, action: ActionName) -> dict[str, float]:
        try:
            pred = world_model.predict(state, action)
        except Exception:
            return {"reward_mean": 0.0, "change_mean": 0.0, "total_uncertainty": 0.0}
        return {
            "reward_mean": _safe_float(getattr(pred, "reward_mean", 0.0)),
            "change_mean": _safe_float(getattr(pred, "change_mean", 0.0)),
            "total_uncertainty": _safe_float(getattr(pred, "total_uncertainty", 0.0)),
        }

    def _memory_bonus(self, memory: Any, state: StructuredState, action: ActionName, lang_tokens: tuple[str, ...]) -> float:
        try:
            return _safe_float(memory.action_memory_bonus(state, action, lang_tokens))
        except Exception:
            return 0.0

    def _coverage_bonus(self, state: StructuredState, action: ActionName) -> float:
        exact_visits = self.state_action_visits[(state.exact_fingerprint, action)]
        family_visits = self.abstract_action_visits[(state.abstract_fingerprint, _family(action))]
        global_visits = self.global_action_visits[action]
        return float(1.0 / (1.0 + exact_visits) + 0.40 / (1.0 + family_visits) + 0.15 / (1.0 + global_visits))

    def _binder_bonus(self, action: ActionName) -> float:
        if not _is_binding_action(action):
            return 0.0
        fam = _family(action)
        success = sum(value for (binder, _probe), value in self.binding_success.items() if binder == fam)
        failure = sum(value for (binder, _probe), value in self.binding_failure.items() if binder == fam)
        prior = 0.55
        empirical = (success + 1.0) / (success + failure + 2.0)
        return float(prior + 0.75 * empirical)

    def _post_binder_probe_bonus(self, action: ActionName, step_index: int) -> float:
        probe = self.pending_binder_probe
        if probe is None or not _is_probe_action(action):
            return 0.0
        ttl_left = max(0, probe.expires_step - step_index)
        urgency = 1.0 + 0.15 * max(0, self.config.selector_probe_horizon - ttl_left)
        return float(urgency)

    def _penalty(self, state: StructuredState, action: ActionName) -> float:
        exact_no_effect = self.no_effect_counts[(state.exact_fingerprint, action)]
        family_no_effect = self.no_effect_family_counts[(state.abstract_fingerprint, _family(action))]
        contradictions = self.contradiction_counts[(state.exact_fingerprint, action)]
        repeat = self.global_action_visits[action]
        penalty = self.config.no_effect_penalty * min(exact_no_effect, 4) / 4.0
        penalty += 0.35 * self.config.no_effect_penalty * min(family_no_effect, 6) / 6.0
        penalty += self.config.contradiction_penalty * min(contradictions, 4.0) / 4.0
        penalty += self.config.repeat_penalty * min(repeat, 12) / 12.0
        if _is_binding_action(action):
            penalty *= 0.35
        if self.pending_binder_probe is not None and _is_probe_action(action):
            penalty *= 0.25
        return float(penalty)

    def _commitment_bonus(self, action: ActionName, state: StructuredState) -> float:
        intention = self.active_intention
        if intention is None:
            return 0.0
        if state.step_index > intention.expires_step:
            return 0.0
        if action == intention.action and state.step_index <= intention.min_until_step:
            return 1.0
        if self.pending_binder_probe is not None and _is_probe_action(action):
            return 0.75
        return 0.0

    def _select_with_commitment(self, state: StructuredState, scored: list[SpotlightCandidate]) -> SpotlightCandidate:
        best = scored[0]
        intention = self.active_intention
        if intention is None or state.step_index > intention.expires_step:
            return best
        if self.pending_binder_probe is not None:
            for candidate in scored:
                if _is_probe_action(candidate.action):
                    return candidate
        if state.step_index <= intention.min_until_step:
            for candidate in scored:
                if candidate.action == intention.action and best.score <= candidate.score + self.config.switch_margin:
                    return candidate
        return best

    def _broadcast_intention(self, state: StructuredState, chosen: SpotlightCandidate) -> SpotlightIntention:
        current = self.active_intention
        if current is not None and current.action == chosen.action and state.step_index <= current.expires_step:
            intention = current
        else:
            self._intention_counter += 1
            intention = SpotlightIntention(
                intention_id=self._intention_counter,
                action=chosen.action,
                intent_kind=chosen.intent_kind,
                target=chosen.target,
                hypothesis_id=chosen.hypothesis_id,
                created_step=state.step_index,
                expires_step=state.step_index + self.config.max_commit_steps,
                min_until_step=state.step_index + self.config.min_commit_steps,
                predicted_reward=chosen.predicted_reward,
                predicted_change=chosen.predicted_change,
                falsifier=self._falsifier(chosen),
            )
            self.active_intention = intention

        slots = self._workspace_slots(state, chosen, intention)
        self.last_broadcast = SpotlightBroadcast(
            step=state.step_index,
            action=chosen.action,
            focus=chosen.intent_kind,
            target=chosen.target,
            expected_reward=chosen.predicted_reward,
            expected_change=chosen.predicted_change,
            score=chosen.score,
            components=dict(chosen.components),
            falsifier=intention.falsifier,
            workspace_slots=slots,
        )
        return intention

    def _workspace_slots(
        self,
        state: StructuredState,
        chosen: SpotlightCandidate,
        intention: SpotlightIntention,
    ) -> dict[str, str]:
        pending = self.pending_binder_probe
        return {
            "focus": chosen.intent_kind,
            "committed_action": str(chosen.action),
            "target": "none" if chosen.target is None else chosen.target,
            "expected_effect": f"r={chosen.predicted_reward:.3f};d={chosen.predicted_change:.3f}",
            "falsifier": intention.falsifier,
            "pending_binder": "none" if pending is None else str(pending.binder_action),
            "last_surprise": f"{self.last_surprise:.3f}",
            "step": str(state.step_index),
        }

    def _workspace_language(
        self,
        language: Any,
        engine: Any,
        chosen: SpotlightCandidate,
        intention: SpotlightIntention,
    ) -> tuple[str, ...]:
        tokens: list[str] = []
        try:
            tokens.extend(language.belief_sentences(engine, limit=3))
        except Exception:
            pass
        try:
            tokens.extend(language.questions(engine, limit=2))
        except Exception:
            pass
        tokens.extend(
            [
                f"spotlight.focus={chosen.intent_kind}",
                f"spotlight.action={chosen.action}",
                f"spotlight.expected_reward={chosen.predicted_reward:.3f}",
                f"spotlight.expected_change={chosen.predicted_change:.3f}",
                f"spotlight.falsifier={intention.falsifier}",
            ]
        )
        if self.pending_binder_probe is not None:
            tokens.append(f"spotlight.pending_probe_after={self.pending_binder_probe.binder_action}")
        return tuple(tokens)

    def _language_tokens(self, language: Any, state: StructuredState, engine: Any) -> tuple[str, ...]:
        try:
            return tuple(language.memory_tokens(state, engine))
        except Exception:
            return ()

    def _intent_kind(self, action: ActionName, components: Mapping[str, float]) -> str:
        if self.pending_binder_probe is not None and _is_probe_action(action):
            return "probe_after_latent_binding"
        if _is_binding_action(action):
            return "bind_or_test_hidden_control"
        if components.get("expected_reward", 0.0) > 0.10:
            return "exploit_reward_hypothesis"
        if components.get("information_gain", 0.0) > components.get("expected_reward", 0.0):
            return "disambiguate_hypotheses"
        try:
            if is_move_action(action):
                return "navigate_or_contact_probe"
            if is_interact_action(action):
                return "contact_intervention"
        except Exception:
            pass
        return "coverage_probe"

    def _default_rationale(self, action: ActionName, components: Mapping[str, float]) -> tuple[str, ...]:
        reasons = [f"focus:{self._intent_kind(action, components)}"]
        if components.get("post_binder_probe_bonus", 0.0) > 0:
            reasons.append("must test the previous latent binder before judging it")
        if components.get("coverage", 0.0) > 0.5:
            reasons.append("state-action pair is under-tested")
        if components.get("binder_bonus", 0.0) > 0:
            reasons.append("action may bind hidden control state")
        return tuple(reasons)

    def _falsifier(self, candidate: SpotlightCandidate) -> str:
        if _is_binding_action(candidate.action):
            return "no downstream probe changes controllable state before selector_probe_horizon"
        if candidate.predicted_reward > 0.10:
            return "no reward or score progress after committed action"
        if candidate.predicted_change > 0.10:
            return "no visible state delta after committed action"
        return "repeated execution gives neither novelty nor state change"

    def _target_text(self, action: ActionName) -> str | None:
        try:
            target = parse_action_target(action)
        except Exception:
            target = None
        if target is None:
            return None
        return ",".join(str(value) for value in target)

    def _transition_surprise(
        self,
        *,
        progress: float,
        visible_effect: bool,
        predicted_reward: float,
        predicted_change: float,
        changed_fraction: float,
    ) -> float:
        reward_gap = abs(progress - predicted_reward)
        change_target = max(0.0, predicted_change)
        observed_change = changed_fraction if visible_effect else 0.0
        change_gap = abs(observed_change - change_target)
        return float(reward_gap + 0.5 * change_gap)

    def _contradicted(self, intention: SpotlightIntention, *, visible_effect: bool, progress: float) -> bool:
        if intention.predicted_reward > 0.15 and progress <= 1e-8:
            return True
        if intention.predicted_change > 0.20 and not visible_effect:
            return True
        return False

    def _update_pending_binder_probe(self, record: TransitionRecord, *, visible_effect: bool, progress: float) -> bool:
        probe = self.pending_binder_probe
        if probe is None:
            return False
        if _is_binding_action(record.action) and record.action == probe.binder_action:
            return False
        if record.before.step_index > probe.expires_step:
            self.binding_failure[(_family(probe.binder_action), "expired")] += 1
            self.pending_binder_probe = None
            return False
        if not _is_probe_action(record.action):
            return False
        probe.probes_taken += 1
        key = (_family(probe.binder_action), _family(record.action))
        supported = bool(visible_effect or progress > 0.0)
        if supported:
            self.binding_success[key] += 1
            self.pending_binder_probe = None
            return True
        if probe.probes_taken >= self.config.selector_null_grace:
            self.binding_failure[key] += 1
        if probe.probes_taken >= self.config.selector_probe_horizon:
            self.pending_binder_probe = None
        return False

    def _confidence_from_margin(self, scored: list[SpotlightCandidate]) -> float:
        if len(scored) < 2:
            return 1.0
        margin = scored[0].score - scored[1].score
        return float(1.0 / (1.0 + exp(-margin / max(self.config.confidence_temperature, 1e-6))))

    def _intention_dict(self, intention: SpotlightIntention) -> dict[str, Any]:
        return {
            "id": intention.intention_id,
            "action": str(intention.action),
            "intent_kind": intention.intent_kind,
            "target": intention.target,
            "hypothesis_id": intention.hypothesis_id,
            "created_step": intention.created_step,
            "expires_step": intention.expires_step,
            "predicted_reward": intention.predicted_reward,
            "predicted_change": intention.predicted_change,
            "falsifier": intention.falsifier,
            "support": intention.support,
            "contradiction": intention.contradiction,
            "confidence": intention.confidence,
            "probe_count": intention.probe_count,
        }
