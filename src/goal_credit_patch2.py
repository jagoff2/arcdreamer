"""Strict usefulness helper for ARC attempt memory."""

from __future__ import annotations

from typing import Any


def strict_useful_label(memory_module: Any, *, visible_effect: bool, events: list[str], score_delta: float, terminal: bool, no_effect_event: bool, invalid_action: bool, goal_entropy_drop: float = 0.0, candidate_goal_reachability_improved: float = 0.0, attached_goal_predicates: tuple[str, ...] = ()) -> Any:
    """Return a stricter TransitionUsefulness label.

    Visible changes are not sufficient.  A transition is useful only when it
    has real progress, a terminal win, or an attached candidate-goal distance
    improvement.  Public controllability remains instrumental evidence.
    """
    progress = bool(memory_module.POSITIVE_EVENTS.intersection(str(e) for e in events)) or float(score_delta) > 0.0
    win = bool(terminal and memory_module._is_terminal_win_event(events))
    no_effect = bool((not visible_effect) or no_effect_event or invalid_action)
    entropy = max(float(goal_entropy_drop), 0.0)
    reach = max(float(candidate_goal_reachability_improved), 0.0)
    attached = tuple(str(x) for x in attached_goal_predicates if str(x))[:8]
    goal_improved = bool(attached and (entropy > memory_module.GOAL_ENTROPY_DROP_THRESHOLD or reach > memory_module.GOAL_REACHABILITY_THRESHOLD))
    useful = bool(progress or win or goal_improved)
    nuisance = bool(visible_effect and not useful)
    reasons: list[str] = []
    if progress:
        reasons.append("progress")
    if win:
        reasons.append("terminal_win")
    if goal_improved:
        reasons.append("attached_goal_distance_improved")
    if nuisance:
        reasons.append("visible_without_goal_progress")
    if no_effect:
        reasons.append("no_effect")
    return memory_module.TransitionUsefulness(
        visible_effect=bool(visible_effect),
        useful_effect=useful,
        nuisance_effect=nuisance,
        no_effect=no_effect,
        progress_effect=progress,
        terminal_win=win,
        controllability_effect=False,
        reachable_state_class_effect=False,
        instrumental_evidence=bool(visible_effect and not no_effect and not progress and not useful),
        goal_entropy_drop=entropy,
        candidate_goal_reachability_improved=reach,
        attached_goal_predicates=attached,
        reasons=tuple(reasons),
    )
