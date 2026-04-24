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

import numpy as np

from .adaptation import AdaptationScorer
from .executive import ExecutiveScorer, encode_feature_map
from .habit import HabitPolicy
from .types import (
    ActionDecision,
    ActionName,
    StructuredState,
    TransitionRecord,
    action_family,
    combined_progress_signal,
    is_failure_terminal_game_state,
    is_interact_action,
    is_move_action,
    is_reset_action,
    is_selector_action,
    observation_game_state,
    observation_levels_completed,
    parse_action_target,
)

LEGACY_FEATURE_SCHEMA_VERSION = 1
EXTENDED_FEATURE_SCHEMA_VERSION = 2
CURRENT_FEATURE_SCHEMA_VERSION = 3


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


def _coerce_feature_vector(value: Any, dim: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        array = np.zeros(0, dtype=np.float32)
    if array.shape[0] == dim:
        return array.copy()
    output = np.zeros(dim, dtype=np.float32)
    count = min(dim, int(array.shape[0]))
    if count > 0:
        output[:count] = array[:count]
    return output


@dataclass(frozen=True)
class SpotlightConfig:
    """Weights and limits for the bounded action workspace."""

    adaptation_weight: float = 0.70
    habit_weight: float = 1.0
    executive_weight: float = 1.0
    reward_weight: float = 2.20
    information_weight: float = 1.85
    change_weight: float = 0.65
    uncertainty_weight: float = 0.45
    memory_weight: float = 0.65
    coverage_weight: float = 0.95
    binder_weight: float = 1.15
    post_binder_probe_weight: float = 2.20
    reset_weight: float = 3.20
    risk_weight: float = 1.00
    contradiction_penalty: float = 1.25
    no_effect_penalty: float = 0.85
    repeat_penalty: float = 0.45
    commitment_bonus: float = 0.40
    max_candidates: int = 0
    trace_capacity: int = 256
    selector_probe_horizon: int = 4
    selector_null_grace: int = 1
    max_commit_steps: int = 5
    min_commit_steps: int = 1
    switch_margin: float = 0.20
    override_margin: float = 0.18
    confidence_temperature: float = 0.75
    reset_stall_threshold: int = 56
    reset_cooldown_steps: int = 28
    reset_session_budget: int = 4
    reset_retry_penalty: float = 1.15
    reset_positive_improvement_guard: float = 0.08
    diagnostic_binding_stall_threshold: int = 24


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
    feature_vector: np.ndarray | None = None
    adaptation_value: float = 0.0
    habit_prior: float = 0.0
    habit_probability: float = 0.5
    habit_rank: int = 0
    executive_value: float = 0.0
    executive_uncertainty: float = 0.0
    executive_advantage: float = 0.0
    executive_rank: int = 0


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
class PendingActionUpdate:
    action: ActionName
    feature_vector: np.ndarray
    predicted_advantage: float
    predicted_uncertainty: float
    habit_baseline: float
    habit_rank: int
    executive_rank: int
    habit_best_action: ActionName
    move37_candidate: bool
    candidate_count: int
    step: int


@dataclass
class DeferredTeacherBinding:
    teacher_action: ActionName
    teacher_feature_vector: np.ndarray
    weight: float
    chosen_feature_vector: np.ndarray | None = None
    chosen_zero_weight: float = 0.0
    contrast_feature_vector: np.ndarray | None = None
    contrast_weight: float = 0.0


@dataclass
class AttemptActionRecord:
    action: ActionName
    feature_vector: np.ndarray
    step: int
    progress: float
    visible_effect: bool


@dataclass(frozen=True)
class AttemptOutcome:
    level_key: str
    score: float
    reward: float
    steps: int
    success: bool
    terminal_failure: bool


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
        self.feature_schema_version = CURRENT_FEATURE_SCHEMA_VERSION
        self.executive = ExecutiveScorer()
        self.habit = HabitPolicy(feature_dim=self.executive.feature_dim)
        self.adaptation = AdaptationScorer(feature_dim=self.executive.feature_dim)
        self._intention_counter = 0
        self.active_intention: SpotlightIntention | None = None
        self.pending_binder_probe: PendingBinderProbe | None = None
        self.pending_teacher_binding: DeferredTeacherBinding | None = None
        self.pending_update: PendingActionUpdate | None = None
        self.state_action_visits: defaultdict[tuple[str, ActionName], int] = defaultdict(int)
        self.abstract_action_visits: defaultdict[tuple[str, str], int] = defaultdict(int)
        self.global_action_visits: defaultdict[ActionName, int] = defaultdict(int)
        self.no_effect_counts: defaultdict[tuple[str, ActionName], int] = defaultdict(int)
        self.no_effect_family_counts: defaultdict[tuple[str, str], int] = defaultdict(int)
        self.contradiction_counts: defaultdict[tuple[str, ActionName], float] = defaultdict(float)
        self.binding_success: defaultdict[tuple[str, str], int] = defaultdict(int)
        self.binding_failure: defaultdict[tuple[str, str], int] = defaultdict(int)
        self.probe_baseline_trials: defaultdict[str, int] = defaultdict(int)
        self.probe_baseline_effect_sum: defaultdict[str, float] = defaultdict(float)
        self.prior_binding_success: defaultdict[tuple[str, str], int] = defaultdict(int)
        self.prior_binding_failure: defaultdict[tuple[str, str], int] = defaultdict(int)
        self.steps_since_progress = 0
        self.max_levels_completed = 0
        self.last_surprise: float = 0.0
        self.last_executive_target: float = 0.0
        self.last_executive_loss: float = 0.0
        self.last_habit_loss: float = 0.0
        self.last_adaptation_loss: float = 0.0
        self.last_attempt_improvement: float = 0.0
        self.last_teacher_action: str = ""
        self.move37_candidates: int = 0
        self.move37_validated: int = 0
        self.last_move37_event: dict[str, Any] | None = None
        self._last_scored_candidates: dict[ActionName, SpotlightCandidate] = {}
        self._current_attempt_actions: list[AttemptActionRecord] = []
        self._current_attempt_level_key: str | None = None
        self._current_attempt_reward: float = 0.0
        self._current_attempt_steps: int = 0
        self._previous_attempt_outcome: dict[str, AttemptOutcome] = {}
        self.attempt_improvements: deque[float] = deque(maxlen=64)
        self.last_broadcast: SpotlightBroadcast | None = None
        self.trace: deque[dict[str, Any]] = deque(maxlen=self.config.trace_capacity)
        self.session_reset_count = 0
        self.steps_since_reset = int(self.config.reset_cooldown_steps)

    def reset_episode(self) -> None:
        self._intention_counter = 0
        self.active_intention = None
        self.pending_binder_probe = None
        self.pending_teacher_binding = None
        self.pending_update = None
        self._last_scored_candidates.clear()
        self.state_action_visits.clear()
        self.abstract_action_visits.clear()
        self.global_action_visits.clear()
        self.no_effect_counts.clear()
        self.no_effect_family_counts.clear()
        self.contradiction_counts.clear()
        self.binding_success.clear()
        self.binding_failure.clear()
        self.probe_baseline_trials.clear()
        self.probe_baseline_effect_sum.clear()
        self.steps_since_progress = 0
        self.max_levels_completed = 0
        self.last_surprise = 0.0
        self.last_executive_target = 0.0
        self.last_executive_loss = 0.0
        self.last_habit_loss = 0.0
        self.last_adaptation_loss = 0.0
        self.last_attempt_improvement = 0.0
        self.last_teacher_action = ""
        self.last_move37_event = None
        self._current_attempt_actions.clear()
        self._current_attempt_level_key = None
        self._current_attempt_reward = 0.0
        self._current_attempt_steps = 0
        self._previous_attempt_outcome.clear()
        self.attempt_improvements.clear()
        self.last_broadcast = None
        self.trace.clear()
        self.session_reset_count = 0
        self.steps_since_reset = int(self.config.reset_cooldown_steps)

    def reset_level(self) -> None:
        self._intention_counter = 0
        self.active_intention = None
        self.pending_binder_probe = None
        self.pending_teacher_binding = None
        self.pending_update = None
        self._last_scored_candidates.clear()
        self.steps_since_progress = 0
        self._current_attempt_actions.clear()
        self._current_attempt_level_key = None
        self._current_attempt_reward = 0.0
        self._current_attempt_steps = 0
        self.last_broadcast = None

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
        self._ensure_attempt(state)
        lang_tokens = self._language_tokens(language, state, engine)
        candidates = self._candidate_actions(state, planner=planner, engine=engine, memory=memory, lang_tokens=lang_tokens)
        if not candidates:
            raise RuntimeError("ActionSpotlight received no candidate actions")

        scored = self._score_candidates(
            state,
            candidates,
            planner=planner,
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

        self._last_scored_candidates = {candidate.action: candidate for candidate in scored}
        chosen = self._select_with_commitment(state, scored)
        habit_best = self._habit_best_candidate(scored)
        move37_candidate = self._is_move37_candidate(chosen, habit_best)
        intention = self._broadcast_intention(state, chosen)
        components = dict(chosen.components)
        components["spotlight_score"] = float(chosen.score)
        components["spotlight_confidence"] = self._confidence_from_margin(scored)
        components["active_intention_confidence"] = float(intention.confidence)
        components["adaptation_value"] = float(chosen.adaptation_value)
        components["habit_prior"] = float(chosen.habit_prior)
        components["habit_probability"] = float(chosen.habit_probability)
        components["habit_rank"] = float(chosen.habit_rank)
        components["executive_value"] = float(chosen.executive_value)
        components["executive_uncertainty"] = float(chosen.executive_uncertainty)
        components["executive_advantage"] = float(chosen.executive_advantage)
        components["executive_rank"] = float(chosen.executive_rank)
        components["move37_candidate"] = float(move37_candidate)

        language_items = self._workspace_language(language, engine, chosen, intention)
        self.state_action_visits[(state.exact_fingerprint, chosen.action)] += 1
        self.abstract_action_visits[(state.abstract_fingerprint, _family(chosen.action))] += 1
        self.global_action_visits[chosen.action] += 1
        if chosen.feature_vector is not None:
            self.pending_update = PendingActionUpdate(
                action=chosen.action,
                feature_vector=chosen.feature_vector.copy(),
                predicted_advantage=float(chosen.executive_advantage),
                predicted_uncertainty=float(chosen.executive_uncertainty),
                habit_baseline=float(habit_best.habit_prior),
                habit_rank=int(chosen.habit_rank),
                executive_rank=int(chosen.executive_rank),
                habit_best_action=habit_best.action,
                move37_candidate=bool(move37_candidate),
                candidate_count=len(scored),
                step=int(state.step_index),
            )
            if move37_candidate:
                self.move37_candidates += 1
                self.last_move37_event = {
                    "step": int(state.step_index),
                    "action": str(chosen.action),
                    "habit_best_action": str(habit_best.action),
                    "habit_rank": int(chosen.habit_rank),
                    "executive_rank": int(chosen.executive_rank),
                    "validated": None,
                }

        return ActionDecision(
            action=chosen.action,
            score=float(chosen.score),
            components=components,
            language=language_items,
            candidate_count=len(candidates),
            chosen_reason="; ".join(chosen.rationale[:4]) or chosen.intent_kind,
        )

    def notify_transition(self, *, record: TransitionRecord, engine: Any | None = None) -> dict[str, Any]:
        progress = combined_progress_signal(record.reward, record.delta.score_delta)
        visible_effect = bool(record.delta.has_visible_effect or abs(progress) > 1e-8)
        pending_before = self.pending_binder_probe
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
        before_levels = observation_levels_completed(record.before)
        after_levels = observation_levels_completed(record.after)
        level_progress = max(0, after_levels - before_levels)
        self.max_levels_completed = max(self.max_levels_completed, after_levels)
        self._ensure_attempt(record.before)
        self._current_attempt_reward += float(progress)
        self._current_attempt_steps += 1
        if progress > 1e-8 or level_progress > 0:
            self.steps_since_progress = 0
        else:
            self.steps_since_progress += 1
        if is_reset_action(record.action):
            self.session_reset_count += 1
            self.steps_since_reset = 0
        else:
            self.steps_since_reset = min(self.steps_since_reset + 1, 10_000)

        if self.pending_update is not None and self.pending_update.action == record.action:
            self._current_attempt_actions.append(
                AttemptActionRecord(
                    action=record.action,
                    feature_vector=self.pending_update.feature_vector.copy(),
                    step=int(record.before.step_index),
                    progress=float(progress),
                    visible_effect=bool(visible_effect),
                )
            )

        binding_action = _is_binding_action(record.action)
        if pending_before is None and not binding_action and _is_probe_action(record.action):
            self._update_probe_baseline(record, progress=progress, visible_effect=visible_effect)
        if binding_action:
            self.pending_binder_probe = PendingBinderProbe(
                binder_action=record.action,
                target=self._target_text(record.action),
                created_step=record.before.step_index,
                expires_step=record.before.step_index + self.config.selector_probe_horizon,
                before_fingerprint=record.before.exact_fingerprint,
                immediate_visible_effect=visible_effect,
            )

        probe_supported, probe_failed = self._update_pending_binder_probe(
            record,
            pending_probe=pending_before,
            visible_effect=visible_effect,
            progress=progress,
        )

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

        if self.pending_update is not None and self.pending_update.action == record.action and self.pending_update.move37_candidate:
            validated = bool(progress > 1e-8 or level_progress > 0 or probe_supported)
            if validated:
                self.move37_validated += 1
            self.last_move37_event = {
                "step": int(record.before.step_index),
                "action": str(record.action),
                "habit_best_action": str(self.pending_update.habit_best_action),
                "habit_rank": int(self.pending_update.habit_rank),
                "executive_rank": int(self.pending_update.executive_rank),
                "validated": bool(validated),
                "progress": float(progress),
                "level_delta": int(level_progress),
                "probe_supported": bool(probe_supported),
            }

        if probe_supported:
            self._finalize_deferred_teacher_binding(success=True)
        elif probe_failed:
            self._finalize_deferred_teacher_binding(success=False)

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
        return {
            "binding_action": bool(binding_action),
            "probe_supported": bool(probe_supported),
            "probe_failed": bool(probe_failed),
            "pending_binder_before": bool(pending_before is not None),
            "pending_binder_after": bool(self.pending_binder_probe is not None),
        }

    def observe_teacher_action(self, teacher_action: ActionName, *, weight: float = 1.0) -> None:
        teacher_candidate = self._last_scored_candidates.get(teacher_action)
        if teacher_candidate is None or teacher_candidate.feature_vector is None:
            return
        if _is_binding_action(teacher_action):
            deferred = DeferredTeacherBinding(
                teacher_action=teacher_action,
                teacher_feature_vector=teacher_candidate.feature_vector.copy(),
                weight=float(weight),
            )
            pending = self.pending_update
            if pending is not None and pending.action != teacher_action:
                chosen_candidate = self._last_scored_candidates.get(pending.action)
                if chosen_candidate is not None and chosen_candidate.feature_vector is not None:
                    deferred.chosen_feature_vector = chosen_candidate.feature_vector.copy()
                    deferred.chosen_zero_weight = 0.5 * float(weight)
                    deferred.contrast_feature_vector = chosen_candidate.feature_vector.copy()
                    deferred.contrast_weight = float(weight)
            else:
                habit_best = self._habit_best_candidate(tuple(self._last_scored_candidates.values()))
                if habit_best.action != teacher_action and habit_best.feature_vector is not None:
                    deferred.contrast_feature_vector = habit_best.feature_vector.copy()
                    deferred.contrast_weight = 0.5 * float(weight)
            self.pending_teacher_binding = deferred
            self.last_teacher_action = f"{teacher_action} [deferred]"
            self.last_habit_loss = 0.0
            return
        self.last_teacher_action = str(teacher_action)
        pending = self.pending_update
        chosen_feature_vector: np.ndarray | None = None
        chosen_zero_weight = 0.0
        contrast_feature_vector: np.ndarray | None = None
        contrast_weight = 0.0
        if pending is not None and pending.action != teacher_action:
            chosen_candidate = self._last_scored_candidates.get(pending.action)
            if chosen_candidate is not None and chosen_candidate.feature_vector is not None:
                chosen_feature_vector = chosen_candidate.feature_vector.copy()
                chosen_zero_weight = 0.5 * float(weight)
                contrast_feature_vector = chosen_candidate.feature_vector.copy()
                contrast_weight = float(weight)
        else:
            habit_best = self._habit_best_candidate(tuple(self._last_scored_candidates.values()))
            if habit_best.action != teacher_action and habit_best.feature_vector is not None:
                contrast_feature_vector = habit_best.feature_vector.copy()
                contrast_weight = 0.5 * float(weight)
        self._apply_teacher_feedback(
            teacher_feature_vector=teacher_candidate.feature_vector.copy(),
            weight=float(weight),
            positive=True,
            chosen_feature_vector=chosen_feature_vector,
            chosen_zero_weight=chosen_zero_weight,
            contrast_feature_vector=contrast_feature_vector,
            contrast_weight=contrast_weight,
        )

    def _apply_teacher_feedback(
        self,
        *,
        teacher_feature_vector: np.ndarray,
        weight: float,
        positive: bool,
        chosen_feature_vector: np.ndarray | None = None,
        chosen_zero_weight: float = 0.0,
        contrast_feature_vector: np.ndarray | None = None,
        contrast_weight: float = 0.0,
    ) -> None:
        label = 1.0 if positive else 0.0
        losses: list[float] = [self.habit.update(teacher_feature_vector, label, weight=weight)]
        if positive:
            if chosen_feature_vector is not None and chosen_zero_weight > 0.0:
                losses.append(self.habit.update(chosen_feature_vector, 0.0, weight=chosen_zero_weight))
            if contrast_feature_vector is not None and contrast_weight > 0.0:
                losses.append(
                    self.habit.update_preference(
                        teacher_feature_vector,
                        contrast_feature_vector,
                        weight=contrast_weight,
                    )
                )
        self.last_habit_loss = float(sum(losses) / max(len(losses), 1))

    def _finalize_deferred_teacher_binding(self, *, success: bool) -> None:
        deferred = self.pending_teacher_binding
        if deferred is None:
            return
        self.pending_teacher_binding = None
        self.last_teacher_action = (
            str(deferred.teacher_action)
            if success
            else f"{deferred.teacher_action} [failed]"
        )
        applied_weight = float(deferred.weight if success else (0.35 * deferred.weight))
        self._apply_teacher_feedback(
            teacher_feature_vector=deferred.teacher_feature_vector,
            weight=applied_weight,
            positive=bool(success),
            chosen_feature_vector=deferred.chosen_feature_vector if success else None,
            chosen_zero_weight=deferred.chosen_zero_weight if success else 0.0,
            contrast_feature_vector=deferred.contrast_feature_vector if success else None,
            contrast_weight=deferred.contrast_weight if success else 0.0,
        )

    def learn_from_transition(
        self,
        *,
        record: TransitionRecord,
        planner: Any,
        engine: Any,
        world_model: Any,
        memory: Any,
        language: Any,
    ) -> None:
        pending = self.pending_update
        if pending is None:
            return
        progress = combined_progress_signal(record.reward, record.delta.score_delta)
        level_delta = max(0, observation_levels_completed(record.after) - observation_levels_completed(record.before))
        session_terminal = bool(record.delta.terminated) and not self._can_continue_after_terminal(record.after)
        bootstrap = 0.0 if session_terminal else self._estimate_state_value(
            record.after,
            planner=planner,
            engine=engine,
            world_model=world_model,
            memory=memory,
            language=language,
        )
        reset_penalty = 0.0
        if self._uses_reset_guard_schema() and is_reset_action(record.action):
            terminal_failure = is_failure_terminal_game_state(observation_game_state(record.after))
            if not terminal_failure and level_delta <= 0 and progress <= 1e-8:
                reset_penalty = 0.35
                if self.steps_since_progress <= self.config.reset_stall_threshold:
                    reset_penalty += 0.45
                if self.steps_since_reset < self.config.reset_cooldown_steps:
                    reset_penalty += 0.55
                if self.last_attempt_improvement > self.config.reset_positive_improvement_guard:
                    reset_penalty += 0.35
                reset_penalty += 0.12 * min(float(self._reset_over_budget()), 8.0)
        target = float(progress + level_delta + (self.executive.gamma * bootstrap) - pending.habit_baseline - reset_penalty)
        self.last_executive_loss = float(self.executive.update(pending.feature_vector, target))
        self.last_executive_target = float(target)
        self.pending_update = None

    def diagnostics(self) -> dict[str, Any]:
        return {
            "feature_schema_version": int(self.feature_schema_version),
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
            "prior_binding_success_total": int(sum(self.prior_binding_success.values())),
            "prior_binding_failure_total": int(sum(self.prior_binding_failure.values())),
            "probe_baseline_trials": {family: int(value) for family, value in self.probe_baseline_trials.items()},
            "steps_since_progress": int(self.steps_since_progress),
            "session_reset_count": int(self.session_reset_count),
            "steps_since_reset": int(self.steps_since_reset),
            "max_levels_completed": int(self.max_levels_completed),
            "adaptation_updates": int(self.adaptation.updates),
            "adaptation_last_loss": float(self.last_adaptation_loss),
            "last_attempt_improvement": float(self.last_attempt_improvement),
            "avg_attempt_improvement": float(np.mean(self.attempt_improvements)) if self.attempt_improvements else 0.0,
            "habit_updates": int(self.habit.updates),
            "habit_last_loss": float(self.last_habit_loss),
            "last_teacher_action": self.last_teacher_action,
            "executive_updates": int(self.executive.updates),
            "executive_last_target": float(self.last_executive_target),
            "executive_last_loss": float(self.last_executive_loss),
            "move37_candidates": int(self.move37_candidates),
            "move37_validated": int(self.move37_validated),
            "last_move37_event": None if self.last_move37_event is None else dict(self.last_move37_event),
            "trace_tail": list(self.trace)[-8:],
        }

    def state_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.__dict__,
            "feature_schema_version": int(self.feature_schema_version),
            "executive": self.executive.state_dict(),
            "habit": self.habit.state_dict(),
            "adaptation": self.adaptation.state_dict(),
            "state_action_visits": [(list(key), int(value)) for key, value in self.state_action_visits.items()],
            "abstract_action_visits": [(list(key), int(value)) for key, value in self.abstract_action_visits.items()],
            "global_action_visits": [(str(key), int(value)) for key, value in self.global_action_visits.items()],
            "no_effect_counts": [(list(key), int(value)) for key, value in self.no_effect_counts.items()],
            "no_effect_family_counts": [(list(key), int(value)) for key, value in self.no_effect_family_counts.items()],
            "contradiction_counts": [(list(key), float(value)) for key, value in self.contradiction_counts.items()],
            "binding_success": [(list(key), int(value)) for key, value in self.binding_success.items()],
            "binding_failure": [(list(key), int(value)) for key, value in self.binding_failure.items()],
            "probe_baseline_trials": [(str(key), int(value)) for key, value in self.probe_baseline_trials.items()],
            "probe_baseline_effect_sum": [(str(key), float(value)) for key, value in self.probe_baseline_effect_sum.items()],
            "prior_binding_success": [(list(key), int(value)) for key, value in self.prior_binding_success.items()],
            "prior_binding_failure": [(list(key), int(value)) for key, value in self.prior_binding_failure.items()],
            "steps_since_progress": int(self.steps_since_progress),
            "max_levels_completed": int(self.max_levels_completed),
            "last_surprise": float(self.last_surprise),
            "last_executive_target": float(self.last_executive_target),
            "last_executive_loss": float(self.last_executive_loss),
            "last_habit_loss": float(self.last_habit_loss),
            "last_adaptation_loss": float(self.last_adaptation_loss),
            "last_attempt_improvement": float(self.last_attempt_improvement),
            "last_teacher_action": str(self.last_teacher_action),
            "move37_candidates": int(self.move37_candidates),
            "move37_validated": int(self.move37_validated),
            "last_move37_event": None if self.last_move37_event is None else dict(self.last_move37_event),
            "current_attempt_actions": [
                {
                    "action": record.action,
                    "feature_vector": record.feature_vector.copy(),
                    "step": int(record.step),
                    "progress": float(record.progress),
                    "visible_effect": bool(record.visible_effect),
                }
                for record in self._current_attempt_actions
            ],
            "current_attempt_level_key": self._current_attempt_level_key,
            "current_attempt_reward": float(self._current_attempt_reward),
            "current_attempt_steps": int(self._current_attempt_steps),
            "previous_attempt_outcome": [
                {
                    "level_key": key,
                    "score": float(outcome.score),
                    "reward": float(outcome.reward),
                    "steps": int(outcome.steps),
                    "success": bool(outcome.success),
                    "terminal_failure": bool(outcome.terminal_failure),
                }
                for key, outcome in self._previous_attempt_outcome.items()
            ],
            "attempt_improvements": [float(value) for value in self.attempt_improvements],
            "session_reset_count": int(self.session_reset_count),
            "steps_since_reset": int(self.steps_since_reset),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        self._clear_runtime_state()
        try:
            self.feature_schema_version = int(state.get("feature_schema_version", LEGACY_FEATURE_SCHEMA_VERSION))
        except Exception:
            self.feature_schema_version = LEGACY_FEATURE_SCHEMA_VERSION
        self.no_effect_counts.clear()
        self.no_effect_family_counts.clear()
        self.contradiction_counts.clear()
        self.binding_success.clear()
        self.binding_failure.clear()
        self.prior_binding_success.clear()
        self.prior_binding_failure.clear()
        executive_state = state.get("executive")
        if isinstance(executive_state, Mapping):
            self.executive.load_state_dict(executive_state)
        habit_state = state.get("habit")
        if isinstance(habit_state, Mapping):
            self.habit.load_state_dict(habit_state)
        adaptation_state = state.get("adaptation")
        if isinstance(adaptation_state, Mapping):
            self.adaptation.load_state_dict(adaptation_state)
        self.state_action_visits.clear()
        self.abstract_action_visits.clear()
        self.global_action_visits.clear()
        self.probe_baseline_trials.clear()
        self.probe_baseline_effect_sum.clear()
        for key, value in state.get("state_action_visits", []):
            self.state_action_visits[(str(key[0]), str(key[1]))] = int(value)
        for key, value in state.get("abstract_action_visits", []):
            self.abstract_action_visits[(str(key[0]), str(key[1]))] = int(value)
        for key, value in state.get("global_action_visits", []):
            self.global_action_visits[str(key)] = int(value)
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
        for key, value in state.get("probe_baseline_trials", []):
            self.probe_baseline_trials[str(key)] = int(value)
        for key, value in state.get("probe_baseline_effect_sum", []):
            self.probe_baseline_effect_sum[str(key)] = float(value)
        for key, value in state.get("prior_binding_success", []):
            self.prior_binding_success[(str(key[0]), str(key[1]))] = int(value)
        for key, value in state.get("prior_binding_failure", []):
            self.prior_binding_failure[(str(key[0]), str(key[1]))] = int(value)
        self.steps_since_progress = int(state.get("steps_since_progress", 0))
        self.max_levels_completed = int(state.get("max_levels_completed", 0))
        self.last_surprise = float(state.get("last_surprise", 0.0))
        self.last_executive_target = float(state.get("last_executive_target", 0.0))
        self.last_executive_loss = float(state.get("last_executive_loss", 0.0))
        self.last_habit_loss = float(state.get("last_habit_loss", 0.0))
        self.last_adaptation_loss = float(state.get("last_adaptation_loss", 0.0))
        self.last_attempt_improvement = float(state.get("last_attempt_improvement", 0.0))
        self.last_teacher_action = str(state.get("last_teacher_action", ""))
        self.move37_candidates = int(state.get("move37_candidates", 0))
        self.move37_validated = int(state.get("move37_validated", 0))
        last_move37_event = state.get("last_move37_event")
        self.last_move37_event = dict(last_move37_event) if isinstance(last_move37_event, Mapping) else None
        self._current_attempt_actions.clear()
        for raw in state.get("current_attempt_actions", []):
            if not isinstance(raw, Mapping):
                continue
            self._current_attempt_actions.append(
                AttemptActionRecord(
                    action=str(raw.get("action", "")),
                    feature_vector=_coerce_feature_vector(raw.get("feature_vector"), self.adaptation.feature_dim),
                    step=int(raw.get("step", 0)),
                    progress=float(raw.get("progress", 0.0)),
                    visible_effect=bool(raw.get("visible_effect", False)),
                )
            )
        current_attempt_level_key = state.get("current_attempt_level_key")
        self._current_attempt_level_key = None if current_attempt_level_key is None else str(current_attempt_level_key)
        self._current_attempt_reward = float(state.get("current_attempt_reward", 0.0))
        self._current_attempt_steps = int(state.get("current_attempt_steps", 0))
        self._previous_attempt_outcome.clear()
        for raw in state.get("previous_attempt_outcome", []):
            if not isinstance(raw, Mapping):
                continue
            level_key = str(raw.get("level_key", ""))
            if not level_key:
                continue
            self._previous_attempt_outcome[level_key] = AttemptOutcome(
                level_key=level_key,
                score=float(raw.get("score", 0.0)),
                reward=float(raw.get("reward", 0.0)),
                steps=int(raw.get("steps", 0)),
                success=bool(raw.get("success", False)),
                terminal_failure=bool(raw.get("terminal_failure", False)),
            )
        self.attempt_improvements.clear()
        self.attempt_improvements.extend(float(value) for value in state.get("attempt_improvements", [])[-64:])
        self.session_reset_count = int(state.get("session_reset_count", 0))
        self.steps_since_reset = int(state.get("steps_since_reset", self.config.reset_cooldown_steps))

    def _candidate_actions(
        self,
        state: StructuredState,
        *,
        planner: Any,
        engine: Any,
        memory: Any | None = None,
        lang_tokens: tuple[str, ...] = (),
    ) -> tuple[ActionName, ...]:
        legal = tuple(dict.fromkeys(getattr(state, "available_actions", ()) or ()))
        candidates: Iterable[ActionName]
        if hasattr(planner, "candidate_actions"):
            if self._uses_extended_feature_schema():
                try:
                    candidates = planner.candidate_actions(state, engine=engine, memory=memory, language_tokens=lang_tokens)
                except TypeError:
                    candidates = planner.candidate_actions(state, engine=engine)
            else:
                candidates = planner.candidate_actions(state, engine=engine)
        else:
            candidates = legal
        deduped = tuple(dict.fromkeys((*legal, *tuple(candidates))))
        missing = [action for action in legal if action not in deduped]
        if missing:
            raise RuntimeError(f"candidate surface lost legal actions: {missing[:8]}")
        max_candidates = int(self.config.max_candidates)
        if max_candidates > 0:
            legal_set = set(legal)
            extras = [action for action in deduped if action not in legal_set]
            return (*legal, *extras[: max(0, max_candidates - len(legal))])
        return deduped

    def _score_candidates(
        self,
        state: StructuredState,
        candidates: tuple[ActionName, ...],
        *,
        planner: Any,
        engine: Any,
        world_model: Any,
        memory: Any,
        lang_tokens: tuple[str, ...],
    ) -> list[SpotlightCandidate]:
        raw: dict[str, dict[ActionName, float]] = {
            "adaptation": {},
            "reward": {},
            "info": {},
            "change": {},
            "uncertainty": {},
            "memory": {},
            "coverage": {},
            "binder": {},
            "probe": {},
            "reset": {},
            "risk": {},
            "penalty": {},
        }
        rationale_by_action: dict[ActionName, tuple[str, ...]] = {}
        hyp_id_by_action: dict[ActionName, str | None] = {}
        precomputed: list[dict[str, Any]] = []

        for action in candidates:
            hyp_score = self._hypothesis_score(engine, state, action)
            world = self._world_prediction(world_model, state, action)
            raw["reward"][action] = hyp_score["expected_reward"] + world["reward_mean"]
            raw["info"][action] = hyp_score["information_gain"]
            raw["change"][action] = hyp_score["expected_change"] + world["change_mean"]
            raw["uncertainty"][action] = world["total_uncertainty"]
            raw["memory"][action] = self._memory_bonus(memory, state, action, lang_tokens)
            option_profile = self._option_profile(planner, memory, state, action, lang_tokens) if self._uses_extended_feature_schema() else {}
            raw["coverage"][action] = self._coverage_bonus(state, action)
            raw["binder"][action] = self._binder_bonus(action)
            raw["probe"][action] = self._post_binder_probe_bonus(action, state.step_index)
            raw["reset"][action] = self._reset_bonus(state, action)
            raw["risk"][action] = hyp_score["risk"]
            raw["penalty"][action] = self._penalty(state, action)
            rationale_by_action[action] = tuple(hyp_score.get("rationale", ()))
            hyp_id_by_action[action] = hyp_score.get("hypothesis_id")
            risk = raw["risk"].get(action, 0.0)
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
                "reset_bonus": raw["reset"].get(action, 0.0),
                "risk": risk,
                "commitment_bonus": commitment,
                "penalty": penalty,
            }
            if self._uses_extended_feature_schema():
                components.update(
                    {
                        "option_schema_bonus": _safe_float(option_profile.get("schema_bonus", 0.0)),
                        "option_continuation": _safe_float(option_profile.get("continuation_depth", 0.0)),
                        "option_efficiency": _safe_float(option_profile.get("efficiency", 0.0)),
                        "action_cost": self._planner_action_cost(planner, action, option_profile),
                        "budget_pressure": self._planner_budget_pressure(planner, state),
                        "family_efficiency": self._planner_family_efficiency(planner, action, option_profile),
                    }
                )
            intent_kind = self._intent_kind(action, components)
            precomputed.append(
                {
                    "action": action,
                    "components": components,
                    "intent_kind": intent_kind,
                    "target": self._target_text(action),
                    "hypothesis_id": hyp_id_by_action.get(action),
                    "rationale": rationale_by_action.get(action, ()) or self._default_rationale(action, components),
                }
            )

        norm = {name: _normalise(values) for name, values in raw.items() if name != "penalty"}
        for item in precomputed:
            action = item["action"]
            feature_map = self._workspace_feature_map(
                state,
                action,
                components=item["components"],
                norm_components={name: values.get(action, 0.0) for name, values in norm.items()},
                intent_kind=item["intent_kind"],
            )
            feature_vector = encode_feature_map(feature_map, feature_dim=self.habit.feature_dim)
            _executive_vec, executive_prediction = self.executive.predict(feature_map)
            adaptation_prediction = self.adaptation.predict_encoded(feature_vector)
            habit_prediction = self.habit.predict_encoded(feature_vector)
            item["feature_vector"] = feature_vector
            item["executive_prediction"] = executive_prediction
            item["executive_advantage"] = self.executive.score(executive_prediction)
            item["adaptation_prediction"] = adaptation_prediction
            item["adaptation_value"] = self.adaptation.score(adaptation_prediction)
            item["habit_prediction"] = habit_prediction
            item["habit_prior"] = self.habit.score(habit_prediction)

        habit_sorted = sorted(
            precomputed,
            key=lambda item: (item["habit_prior"], item["habit_prediction"].probability, item["executive_advantage"]),
            reverse=True,
        )
        executive_sorted = sorted(
            precomputed,
            key=lambda item: (item["executive_advantage"], item["habit_prior"]),
            reverse=True,
        )
        habit_rank = {item["action"]: idx + 1 for idx, item in enumerate(habit_sorted)}
        executive_rank = {item["action"]: idx + 1 for idx, item in enumerate(executive_sorted)}

        scored: list[SpotlightCandidate] = []
        for item in precomputed:
            executive_prediction = item["executive_prediction"]
            adaptation_prediction = item["adaptation_prediction"]
            habit_prediction = item["habit_prediction"]
            combined_score = (
                self.config.adaptation_weight * float(item["adaptation_value"])
                + self.config.habit_weight * float(item["habit_prior"])
                + self.config.executive_weight * float(item["executive_advantage"])
            )
            scored.append(
                SpotlightCandidate(
                    action=item["action"],
                    score=float(combined_score),
                    components=dict(item["components"]),
                    intent_kind=item["intent_kind"],
                    target=item["target"],
                    hypothesis_id=item["hypothesis_id"],
                    predicted_reward=item["components"]["expected_reward"],
                    predicted_change=item["components"]["expected_change"],
                    rationale=item["rationale"],
                    feature_vector=item["feature_vector"],
                    adaptation_value=float(item["adaptation_value"]),
                    habit_prior=float(item["habit_prior"]),
                    habit_probability=float(habit_prediction.probability),
                    habit_rank=int(habit_rank[item["action"]]),
                    executive_value=float(executive_prediction.value_mean),
                    executive_uncertainty=float(executive_prediction.value_uncertainty),
                    executive_advantage=float(item["executive_advantage"]),
                    executive_rank=int(executive_rank[item["action"]]),
                )
            )
        scored.sort(key=lambda candidate: candidate.score, reverse=True)
        return scored

    def _workspace_feature_map(
        self,
        state: StructuredState,
        action: ActionName,
        *,
        components: Mapping[str, float],
        norm_components: Mapping[str, float],
        intent_kind: str,
    ) -> dict[str, float]:
        rows, cols = state.grid.shape
        family = _family(action)
        target = parse_action_target(action)
        active_intention = self.active_intention
        pending_probe = self.pending_binder_probe
        exact_key = (state.exact_fingerprint, action)
        abstract_key = (state.abstract_fingerprint, family)
        feature_map: dict[str, float] = {
            "grid_rows": float(rows) / 16.0,
            "grid_cols": float(cols) / 16.0,
            "object_count": float(len(state.objects)) / 24.0,
            "relation_count": float(len(state.relations)) / 96.0,
            "steps_since_progress": float(min(self.steps_since_progress, 96)) / 96.0,
            "max_levels_completed": float(min(self.max_levels_completed, 8)) / 8.0,
            "last_attempt_improvement": float(max(min(self.last_attempt_improvement, 2.0), -2.0)) / 2.0,
            "avg_attempt_improvement": float(np.mean(self.attempt_improvements)) / 2.0 if self.attempt_improvements else 0.0,
            "session_reset_count": float(min(self.session_reset_count, 16)) / 16.0,
            "steps_since_reset": float(min(self.steps_since_reset, max(2 * self.config.reset_cooldown_steps, 1)))
            / max(float(2 * self.config.reset_cooldown_steps), 1.0),
            "has_active_intention": float(active_intention is not None),
            "active_intention_confidence": 0.0 if active_intention is None else float(active_intention.confidence),
            "same_as_active_intention": float(active_intention is not None and active_intention.action == action),
            "has_pending_binder_probe": float(pending_probe is not None),
            "same_as_pending_binder": float(pending_probe is not None and pending_probe.binder_action == action),
            "pending_probe_age": 0.0
            if pending_probe is None
            else float(min(max(state.step_index - pending_probe.created_step, 0), self.config.selector_probe_horizon))
            / max(float(self.config.selector_probe_horizon), 1.0),
            "pending_probe_ttl": 0.0
            if pending_probe is None
            else float(max(pending_probe.expires_step - state.step_index, 0)) / max(float(self.config.selector_probe_horizon), 1.0),
            "exact_state_action_visits": float(min(self.state_action_visits[exact_key], 8)) / 8.0,
            "abstract_family_visits": float(min(self.abstract_action_visits[abstract_key], 8)) / 8.0,
            "global_action_visits": float(min(self.global_action_visits[action], 16)) / 16.0,
            "no_effect_exact": float(min(self.no_effect_counts[exact_key], 6)) / 6.0,
            "no_effect_family": float(min(self.no_effect_family_counts[abstract_key], 8)) / 8.0,
            "contradictions_exact": float(min(self.contradiction_counts[exact_key], 6.0)) / 6.0,
            "is_move": float(is_move_action(action)),
            "is_interact": float(is_interact_action(action)),
            "is_selector": float(is_selector_action(action)),
            "is_binding": float(_is_binding_action(action)),
            "is_probe": float(_is_probe_action(action)),
            "is_reset": float(is_reset_action(action)),
            "has_target": float(target is not None),
            f"family::{family}": 1.0,
            f"intent::{intent_kind}": 1.0,
        }
        if target is not None:
            feature_map["target_row"] = float(target[0]) / max(float(rows - 1), 1.0)
            feature_map["target_col"] = float(target[1]) / max(float(cols - 1), 1.0)
        extras = state.frame.extras if isinstance(state.frame.extras, Mapping) else {}
        flags = extras.get("flags")
        if isinstance(flags, Mapping):
            for key, value in list(flags.items())[:16]:
                text = str(value).strip().lower()
                if text in {"0", "1"}:
                    feature_map[f"flag::{key}"] = float(text == "1")
        inventory = extras.get("inventory")
        if isinstance(inventory, Mapping):
            for key, value in list(inventory.items())[:16]:
                try:
                    numeric = float(value)
                except Exception:
                    continue
                feature_map[f"inventory::{key}"] = _safe_float(numeric)
        for key, value in components.items():
            feature_map[f"raw::{key}"] = _safe_float(value)
        for key, value in norm_components.items():
            feature_map[f"norm::{key}"] = _safe_float(value)
        if self._uses_reset_guard_schema():
            feature_map["reset_over_budget"] = float(min(max(0, self.session_reset_count - self.config.reset_session_budget), 8)) / 8.0
            feature_map["reset_under_cooldown"] = float(self.steps_since_reset < self.config.reset_cooldown_steps)
        return feature_map

    def _uses_extended_feature_schema(self) -> bool:
        return int(self.feature_schema_version) >= EXTENDED_FEATURE_SCHEMA_VERSION

    def _uses_reset_guard_schema(self) -> bool:
        return int(self.feature_schema_version) >= CURRENT_FEATURE_SCHEMA_VERSION

    def _clear_runtime_state(self) -> None:
        self._intention_counter = 0
        self.active_intention = None
        self.pending_binder_probe = None
        self.pending_teacher_binding = None
        self.pending_update = None
        self._last_scored_candidates.clear()
        self.steps_since_progress = 0
        self.max_levels_completed = 0
        self.last_surprise = 0.0
        self.last_executive_target = 0.0
        self.last_executive_loss = 0.0
        self.last_habit_loss = 0.0
        self.last_adaptation_loss = 0.0
        self.last_attempt_improvement = 0.0
        self.last_teacher_action = ""
        self.last_move37_event = None
        self._current_attempt_actions.clear()
        self._current_attempt_level_key = None
        self._current_attempt_reward = 0.0
        self._current_attempt_steps = 0
        self._previous_attempt_outcome.clear()
        self.attempt_improvements.clear()
        self.last_broadcast = None
        self.trace.clear()
        self.session_reset_count = 0
        self.steps_since_reset = int(self.config.reset_cooldown_steps)

    def _estimate_state_value(
        self,
        state: StructuredState,
        *,
        planner: Any,
        engine: Any,
        world_model: Any,
        memory: Any,
        language: Any,
    ) -> float:
        candidates = self._candidate_actions(state, planner=planner, engine=engine)
        if not candidates:
            return 0.0
        scored = self._score_candidates(
            state,
            candidates,
            planner=planner,
            engine=engine,
            world_model=world_model,
            memory=memory,
            lang_tokens=self._language_tokens(language, state, engine),
        )
        if not scored:
            return 0.0
        chosen = self._select_with_commitment(state, scored)
        return float(chosen.score)

    def finalize_attempt(
        self,
        state: StructuredState,
        *,
        success: bool,
        terminal_failure: bool,
        ended_by_reset: bool = False,
    ) -> float:
        level_key = self._current_attempt_level_key or self._level_key(state)
        outcome = AttemptOutcome(
            level_key=level_key,
            score=self._attempt_score(
                reward=self._current_attempt_reward,
                steps=self._current_attempt_steps,
                success=success,
                terminal_failure=terminal_failure,
                ended_by_reset=ended_by_reset,
            ),
            reward=float(self._current_attempt_reward),
            steps=int(self._current_attempt_steps),
            success=bool(success),
            terminal_failure=bool(terminal_failure),
        )
        previous = self._previous_attempt_outcome.get(level_key)
        improvement = 0.0 if previous is None else float(outcome.score - previous.score)
        self.last_attempt_improvement = improvement
        self.attempt_improvements.append(float(improvement))
        if previous is not None:
            losses: list[float] = []
            for record in self._current_attempt_actions:
                losses.append(self.adaptation.update(record.feature_vector, improvement))
            self.last_adaptation_loss = float(sum(losses) / max(len(losses), 1)) if losses else 0.0
        else:
            self.last_adaptation_loss = 0.0
        self._previous_attempt_outcome[level_key] = outcome
        self._current_attempt_actions.clear()
        self._current_attempt_level_key = None
        self._current_attempt_reward = 0.0
        self._current_attempt_steps = 0
        return improvement

    def _ensure_attempt(self, state: StructuredState) -> None:
        level_key = self._level_key(state)
        if self._current_attempt_level_key != level_key:
            self._current_attempt_level_key = level_key
            self._current_attempt_actions.clear()
            self._current_attempt_reward = 0.0
            self._current_attempt_steps = 0

    def _level_key(self, state: StructuredState) -> str:
        extras = state.frame.extras if isinstance(state.frame.extras, Mapping) else {}
        if isinstance(extras, Mapping) and "session_level_index" in extras:
            try:
                level_index = int(extras.get("session_level_index", 0) or 0)
                return f"{state.frame.task_id}/level_{level_index}"
            except Exception:
                pass
        return str(state.frame.task_id)

    def _attempt_score(
        self,
        *,
        reward: float,
        steps: int,
        success: bool,
        terminal_failure: bool,
        ended_by_reset: bool = False,
    ) -> float:
        score = float(reward)
        score -= 0.01 * float(steps)
        if success:
            score += 2.0
        if terminal_failure:
            score -= 0.25
        if ended_by_reset and self._uses_reset_guard_schema():
            score -= 0.15
            if steps < self.config.reset_cooldown_steps:
                score -= 0.25 * (1.0 - (float(steps) / max(float(self.config.reset_cooldown_steps), 1.0)))
            score -= 0.05 * min(float(max(0, self.session_reset_count - self.config.reset_session_budget) + 1), 8.0)
        return score

    def _can_continue_after_terminal(self, state: StructuredState) -> bool:
        if not any(is_reset_action(action) for action in state.available_actions):
            return False
        game_state = observation_game_state(state).strip().upper()
        return game_state.endswith("GAME_OVER") or game_state.endswith("SESSION_ENDED")

    def _hypothesis_score(self, engine: Any, state: StructuredState, action: ActionName) -> dict[str, Any]:
        try:
            try:
                score = engine.score_action(state, action, contextual=True, bounded=True)
            except TypeError:
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

    def _option_profile(
        self,
        planner: Any,
        memory: Any,
        state: StructuredState,
        action: ActionName,
        lang_tokens: tuple[str, ...],
    ) -> dict[str, float]:
        profile_fn = getattr(memory, "action_option_profile", None)
        if callable(profile_fn):
            try:
                profile = profile_fn(state, action, lang_tokens)
                if isinstance(profile, Mapping):
                    return {str(key): _safe_float(value, default=0.0) for key, value in profile.items()}
            except Exception:
                pass
        return {
            "schema_bonus": 0.0,
            "relative_cost": 1.0,
            "efficiency": 0.0,
            "support": 0.0,
            "contradiction": 0.0,
            "continuation_depth": 0.0,
        }

    def _planner_action_cost(self, planner: Any, action: ActionName, option_profile: Mapping[str, float]) -> float:
        fn = getattr(planner, "_action_cost_estimate", None)
        if callable(fn):
            try:
                return _safe_float(fn(action, option_profile=dict(option_profile)))
            except Exception:
                pass
        return float(max(option_profile.get("relative_cost", 1.0), 1.0))

    def _planner_family_efficiency(self, planner: Any, action: ActionName, option_profile: Mapping[str, float]) -> float:
        fn = getattr(planner, "_action_efficiency_prior", None)
        if callable(fn):
            try:
                return _safe_float(fn(action, option_profile=dict(option_profile)))
            except Exception:
                pass
        return _safe_float(option_profile.get("efficiency", 0.0))

    def _planner_budget_pressure(self, planner: Any, state: StructuredState) -> float:
        fn = getattr(planner, "_budget_pressure", None)
        if callable(fn):
            try:
                return _safe_float(fn(state))
            except Exception:
                pass
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
        success += 0.35 * sum(value for (binder, _probe), value in self.prior_binding_success.items() if binder == fam)
        failure = sum(value for (binder, _probe), value in self.binding_failure.items() if binder == fam)
        failure += 0.35 * sum(value for (binder, _probe), value in self.prior_binding_failure.items() if binder == fam)
        prior = 0.55
        empirical = (success + 1.0) / (success + failure + 2.0)
        return float(prior + 0.75 * empirical)

    def _post_binder_probe_bonus(self, action: ActionName, step_index: int) -> float:
        probe = self.pending_binder_probe
        if probe is None or not _is_probe_action(action):
            return 0.0
        binder_family = _family(probe.binder_action)
        probe_family = _family(action)
        pair_success = self.binding_success[(binder_family, probe_family)] + (0.35 * self.prior_binding_success[(binder_family, probe_family)])
        pair_failure = self.binding_failure[(binder_family, probe_family)] + (0.35 * self.prior_binding_failure[(binder_family, probe_family)])
        pair_trials = pair_success + pair_failure
        empirical = (pair_success + 1.0) / (pair_trials + 2.0)
        uncertainty = 1.0 / (1.0 + pair_trials)
        ttl_left = max(0, probe.expires_step - step_index)
        urgency = 1.0 + 0.15 * max(0, self.config.selector_probe_horizon - ttl_left)
        return float(urgency * (0.25 + (0.85 * uncertainty) + (0.35 * empirical)))

    def _reset_over_budget(self) -> int:
        return max(0, int(self.session_reset_count) - int(self.config.reset_session_budget))

    def _allow_nonterminal_reset(self, state: StructuredState) -> bool:
        if self.pending_binder_probe is not None:
            return False
        if self.steps_since_reset < self.config.reset_cooldown_steps:
            return False
        if self.steps_since_progress <= self.config.reset_stall_threshold:
            return False
        if self.last_attempt_improvement > self.config.reset_positive_improvement_guard:
            return False
        if self._reset_over_budget() > 0 and self.last_attempt_improvement >= 0.0:
            return False
        return not is_failure_terminal_game_state(observation_game_state(state))

    def _reset_choice_allowed(self, state: StructuredState, candidate: SpotlightCandidate) -> bool:
        if not is_reset_action(candidate.action):
            return True
        if is_failure_terminal_game_state(observation_game_state(state)):
            return True
        if candidate.components.get("reset_bonus", 0.0) <= 0.0:
            return False
        return self._allow_nonterminal_reset(state)

    def _reset_bonus(self, state: StructuredState, action: ActionName) -> float:
        if not is_reset_action(action):
            return 0.0
        if self._uses_reset_guard_schema():
            game_state = observation_game_state(state)
            if is_failure_terminal_game_state(game_state):
                bonus = 2.25 - (0.20 * min(float(self._reset_over_budget()), 6.0))
                if self.steps_since_reset < self.config.reset_cooldown_steps:
                    cooldown_frac = 1.0 - (float(self.steps_since_reset) / max(float(self.config.reset_cooldown_steps), 1.0))
                    bonus -= 0.75 * max(0.0, cooldown_frac)
                return float(max(0.5, bonus))
            if not self._allow_nonterminal_reset(state):
                return 0.0
            stall_excess = max(0, self.steps_since_progress - self.config.reset_stall_threshold)
            bonus = min(1.25, stall_excess / 24.0)
            bonus /= 1.0 + (0.35 * min(float(self.session_reset_count), 8.0))
            if self.last_attempt_improvement < -0.05:
                bonus += min(0.45, -self.last_attempt_improvement)
            return float(max(0.0, bonus))
        game_state = observation_game_state(state)
        if is_failure_terminal_game_state(game_state):
            return 4.0
        stall_excess = max(0, self.steps_since_progress - self.config.reset_stall_threshold)
        if stall_excess <= 0:
            return 0.0
        return float(min(2.0, stall_excess / 16.0))

    def _penalty(self, state: StructuredState, action: ActionName) -> float:
        if is_reset_action(action):
            if self._uses_reset_guard_schema():
                terminal_failure = is_failure_terminal_game_state(observation_game_state(state))
                penalty = 0.35 if terminal_failure else 3.75
                if self.pending_binder_probe is not None:
                    penalty += 2.0
                if self.steps_since_reset < self.config.reset_cooldown_steps:
                    cooldown_frac = 1.0 - (float(self.steps_since_reset) / max(float(self.config.reset_cooldown_steps), 1.0))
                    penalty += 2.5 * max(0.0, cooldown_frac)
                if not terminal_failure and self.steps_since_progress <= self.config.reset_stall_threshold:
                    penalty += 2.0
                if self.last_attempt_improvement > self.config.reset_positive_improvement_guard:
                    penalty += 1.5 * min(
                        self.last_attempt_improvement / max(self.config.reset_positive_improvement_guard, 1e-6),
                        2.0,
                    )
                penalty += self.config.reset_retry_penalty * min(float(self._reset_over_budget()), 8.0)
                if not terminal_failure and not self._allow_nonterminal_reset(state):
                    penalty += 1.75
                return float(penalty)
            if is_failure_terminal_game_state(observation_game_state(state)):
                return 0.0
            if self.steps_since_progress > self.config.reset_stall_threshold:
                return 0.25
            return 3.5
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
        if self.pending_binder_probe is not None and _is_binding_action(action):
            if action == self.pending_binder_probe.binder_action:
                penalty += 4.0
            else:
                penalty += 1.5
        if self.pending_binder_probe is not None and _is_probe_action(action):
            binder_family = _family(self.pending_binder_probe.binder_action)
            probe_family = _family(action)
            pair_success = self.binding_success[(binder_family, probe_family)] + (0.35 * self.prior_binding_success[(binder_family, probe_family)])
            pair_failure = self.binding_failure[(binder_family, probe_family)] + (0.35 * self.prior_binding_failure[(binder_family, probe_family)])
            if pair_failure > pair_success:
                penalty += 0.35 * min(pair_failure - pair_success, 6.0)
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

    def _habit_best_candidate(self, scored: Iterable[SpotlightCandidate]) -> SpotlightCandidate:
        ordered = list(scored)
        if not ordered:
            raise RuntimeError("habit best candidate requested for empty list")
        return min(
            ordered,
            key=lambda candidate: (
                candidate.habit_rank,
                -candidate.habit_probability,
                -candidate.score,
            ),
        )

    def _default_choice(self, state: StructuredState, scored: list[SpotlightCandidate]) -> SpotlightCandidate:
        best_combined = scored[0]
        habit_best = self._habit_best_candidate(scored)
        if best_combined.action == habit_best.action:
            choice = best_combined
        else:
            override_margin = best_combined.executive_advantage - habit_best.executive_advantage
            if override_margin >= self.config.override_margin and best_combined.score > habit_best.score:
                choice = best_combined
            else:
                choice = habit_best
        if self._uses_reset_guard_schema() and is_reset_action(choice.action) and not self._reset_choice_allowed(state, choice):
            for candidate in scored:
                if not is_reset_action(candidate.action):
                    choice = candidate
                    break
        diagnostic_binding = self._stalled_binding_probe_choice(state, scored)
        if diagnostic_binding is not None and not is_reset_action(choice.action):
            return diagnostic_binding
        return choice

    def _stalled_binding_probe_choice(
        self,
        state: StructuredState,
        scored: list[SpotlightCandidate],
    ) -> SpotlightCandidate | None:
        if self.pending_binder_probe is not None:
            return None
        if is_failure_terminal_game_state(observation_game_state(state)):
            return None
        if self.steps_since_progress < int(self.config.diagnostic_binding_stall_threshold):
            return None
        candidates: list[SpotlightCandidate] = []
        for candidate in scored:
            if is_reset_action(candidate.action) or not _is_binding_action(candidate.action):
                continue
            exact_key = (state.exact_fingerprint, candidate.action)
            if self.state_action_visits[exact_key] > 0:
                continue
            fam = _family(candidate.action)
            known_failures = sum(value for (binder, _probe), value in self.binding_failure.items() if binder == fam)
            known_failures += 0.35 * sum(value for (binder, _probe), value in self.prior_binding_failure.items() if binder == fam)
            known_successes = sum(value for (binder, _probe), value in self.binding_success.items() if binder == fam)
            known_successes += 0.35 * sum(value for (binder, _probe), value in self.prior_binding_success.items() if binder == fam)
            if known_failures > known_successes + 2.0:
                continue
            candidates.append(candidate)
        if not candidates:
            return None
        chosen = max(
            candidates,
            key=lambda candidate: (
                float(candidate.target is not None),
                float(candidate.components.get("coverage", 0.0)),
                float(candidate.executive_uncertainty),
                -float(candidate.components.get("penalty", 0.0)),
                float(candidate.score),
            ),
        )
        chosen.components["diagnostic_binding_probe"] = 1.0
        chosen.components["stalled_steps"] = float(self.steps_since_progress)
        chosen.rationale = (
            "stalled without progress; test an untried latent binding or interaction affordance",
        ) + tuple(chosen.rationale)
        chosen.intent_kind = "diagnostic_binding_probe"
        return chosen

    def _is_move37_candidate(self, chosen: SpotlightCandidate, habit_best: SpotlightCandidate) -> bool:
        if chosen.action == habit_best.action:
            return False
        if chosen.habit_rank <= 1:
            return False
        if chosen.executive_rank != 1:
            return False
        return bool(chosen.executive_advantage >= habit_best.executive_advantage + self.config.override_margin)

    def _select_with_commitment(self, state: StructuredState, scored: list[SpotlightCandidate]) -> SpotlightCandidate:
        if is_failure_terminal_game_state(observation_game_state(state)):
            for candidate in scored:
                if is_reset_action(candidate.action):
                    return candidate
        best = self._default_choice(state, scored)
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
        if is_reset_action(action):
            return "reset_level_retry"
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
        if components.get("reset_bonus", 0.0) > 0:
            reasons.append("reset is the best way to preserve learning and reopen the level")
        if components.get("post_binder_probe_bonus", 0.0) > 0:
            reasons.append("must test the previous latent binder before judging it")
        if components.get("coverage", 0.0) > 0.5:
            reasons.append("state-action pair is under-tested")
        if components.get("binder_bonus", 0.0) > 0:
            reasons.append("action may bind hidden control state")
        return tuple(reasons)

    def _falsifier(self, candidate: SpotlightCandidate) -> str:
        if is_reset_action(candidate.action):
            return "reset does not reopen a controllable nonterminal level state"
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

    def _update_pending_binder_probe(
        self,
        record: TransitionRecord,
        *,
        pending_probe: PendingBinderProbe | None,
        visible_effect: bool,
        progress: float,
    ) -> tuple[bool, bool]:
        probe = pending_probe
        if probe is None:
            return False, False
        if _is_binding_action(record.action) and record.action == probe.binder_action:
            return False, False
        if record.before.step_index > probe.expires_step:
            self.binding_failure[(_family(probe.binder_action), "expired")] += 1
            self.prior_binding_failure[(_family(probe.binder_action), "expired")] += 1
            if self.pending_binder_probe is probe:
                self.pending_binder_probe = None
            return False, True
        if not _is_probe_action(record.action):
            return False, False
        probe.probes_taken += 1
        key = (_family(probe.binder_action), _family(record.action))
        supported = self._probe_supports_binder(
            record,
            probe=probe,
            progress=progress,
            visible_effect=visible_effect,
        )
        if supported:
            self.binding_success[key] += 1
            self.prior_binding_success[key] += 1
            if self.pending_binder_probe is probe:
                self.pending_binder_probe = None
            return True, False
        if probe.probes_taken >= self.config.selector_null_grace:
            self.binding_failure[key] += 1
            self.prior_binding_failure[key] += 1
            if self.pending_binder_probe is probe:
                self.pending_binder_probe = None
            return False, True
        if probe.probes_taken >= self.config.selector_probe_horizon and self.pending_binder_probe is probe:
            self.pending_binder_probe = None
        return False, False

    def _probe_supports_binder(
        self,
        record: TransitionRecord,
        *,
        probe: PendingBinderProbe,
        progress: float,
        visible_effect: bool,
    ) -> bool:
        if progress > 1e-8:
            return True
        action_set_changed = set(record.before.available_actions) != set(record.after.available_actions)
        if action_set_changed or record.delta.appeared or record.delta.disappeared:
            return True
        if not visible_effect:
            return False
        probe_family = _family(record.action)
        baseline_trials = self.probe_baseline_trials[probe_family]
        if baseline_trials < 3:
            return False
        baseline_mean = self._probe_baseline_mean(probe_family)
        effect_score = self._probe_effect_score(record, progress=progress, visible_effect=visible_effect)
        changed_fraction = _safe_float(getattr(record.delta, "changed_fraction", 0.0))
        if changed_fraction < 0.12:
            return False
        return bool(effect_score > (baseline_mean + 0.28))

    def _update_probe_baseline(self, record: TransitionRecord, *, progress: float, visible_effect: bool) -> None:
        family = _family(record.action)
        self.probe_baseline_trials[family] += 1
        self.probe_baseline_effect_sum[family] += self._probe_effect_score(record, progress=progress, visible_effect=visible_effect)

    def _probe_effect_score(self, record: TransitionRecord, *, progress: float, visible_effect: bool) -> float:
        action_set_changed = set(record.before.available_actions) != set(record.after.available_actions)
        effect = max(0.0, float(progress))
        effect += 0.35 * _safe_float(getattr(record.delta, "changed_fraction", 0.0))
        effect += 0.15 * len(getattr(record.delta, "moved_objects", ()))
        effect += 0.08 * len(getattr(record.delta, "touched_objects", ()))
        effect += 0.20 * (len(getattr(record.delta, "appeared", ())) + len(getattr(record.delta, "disappeared", ())))
        if action_set_changed:
            effect += 0.25
        if visible_effect:
            effect += 0.05
        return float(effect)

    def _probe_baseline_mean(self, family: str) -> float:
        trials = self.probe_baseline_trials[family]
        if trials <= 0:
            return 0.0
        return float(self.probe_baseline_effect_sum[family] / max(trials, 1))

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
