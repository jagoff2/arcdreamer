"""Runtime patch that makes ARC useful-credit stricter."""
from __future__ import annotations

from typing import Any

_APPLIED = False


def apply_goal_credit_patch_runtime() -> None:
    global _APPLIED
    if _APPLIED:
        return
    from . import jepa_attempt_memory as m

    def classify(
        *,
        visible_effect: bool,
        events: list[str],
        score_delta: float,
        terminal: bool,
        no_effect_event: bool,
        invalid_action: bool,
        prior_action_count: int = 0,
        prior_family_count: int = 0,
        prior_useful_action_count: int = 0,
        public_controllability_evidence: bool = False,
        public_reachable_state_evidence: bool = False,
        state_class_seen: bool = True,
        goal_entropy_drop: float = 0.0,
        candidate_goal_reachability_improved: float = 0.0,
        attached_goal_predicates: tuple[str, ...] = (),
    ) -> Any:
        progress = bool(m.POSITIVE_EVENTS.intersection(str(e) for e in events)) or float(score_delta) > 0.0
        win = bool(terminal and m._is_terminal_win_event(events))
        no_effect = bool((not visible_effect) or no_effect_event or invalid_action)
        entropy = max(float(goal_entropy_drop), 0.0)
        reach = max(float(candidate_goal_reachability_improved), 0.0)
        attached = tuple(str(x) for x in attached_goal_predicates if str(x))[:8]
        goal_ok = bool(attached and (entropy > m.GOAL_ENTROPY_DROP_THRESHOLD or reach > m.GOAL_REACHABILITY_THRESHOLD))
        useful = bool(progress or win or goal_ok)
        instrumental = bool(visible_effect and not no_effect and not progress and not useful and (public_controllability_evidence or public_reachable_state_evidence or prior_useful_action_count > 0))
        nuisance = bool(visible_effect and not useful)
        reasons: list[str] = []
        if progress:
            reasons.append("progress")
        if win:
            reasons.append("terminal_win")
        if goal_ok:
            reasons.append("attached_goal_distance_improved")
        if instrumental:
            reasons.append("instrumental_not_useful")
        if nuisance:
            reasons.append("visible_without_goal_progress")
        if no_effect:
            reasons.append("no_effect")
        return m.TransitionUsefulness(
            visible_effect=bool(visible_effect),
            useful_effect=useful,
            nuisance_effect=nuisance,
            no_effect=no_effect,
            progress_effect=progress,
            terminal_win=win,
            controllability_effect=False,
            reachable_state_class_effect=False,
            instrumental_evidence=instrumental,
            goal_entropy_drop=entropy,
            candidate_goal_reachability_improved=reach,
            attached_goal_predicates=attached,
            reasons=tuple(reasons),
        )

    def credit(*, useful: Any, no_effect_event: bool, invalid_action: bool, score_delta: float, terminal: bool) -> float:
        if invalid_action:
            return -1.0
        if useful.progress_effect or useful.terminal_win:
            return 1.15 + min(max(float(score_delta), 0.0), 2.0)
        if useful.useful_effect:
            reach = min(max(float(getattr(useful, "candidate_goal_reachability_improved", 0.0)), 0.0), 1.0)
            entropy = min(max(float(getattr(useful, "goal_entropy_drop", 0.0)), 0.0), 1.0)
            return 0.08 + 0.32 * reach + 0.18 * entropy
        if no_effect_event or useful.no_effect:
            return -0.65
        if terminal:
            return -0.90
        if useful.instrumental_evidence:
            return min(float(score_delta), 0.0) - 0.04
        if useful.nuisance_effect:
            return min(float(score_delta), 0.0) - 0.24
        return min(float(score_delta), 0.0) - 0.08

    m._classify_transition_usefulness = classify
    m._transition_credit = credit
    _APPLIED = True
