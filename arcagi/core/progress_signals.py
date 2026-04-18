from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class PolicySupervision:
    target: float
    weight: float
    sibling_move_target: float = 0.0
    sibling_move_weight: float = 0.0
    same_type_target: float = 0.0
    same_type_weight: float = 0.0


_STRONG_POSITIVE_EVENTS: frozenset[str] = frozenset(
    {
        "goal_reached",
        "correct_switch",
        "selector_unlock_complete",
        "correct_order_complete",
        "delayed_sequence_complete",
        "selector_sequence_complete",
    }
)
_PROGRESS_EVENTS: frozenset[str] = frozenset(
    {
        "correct_collect",
        "delayed_correct_collect",
        "selector_sequence_progress",
        "selector_candidate",
    }
)
_DIAGNOSTIC_EVENTS: frozenset[str] = frozenset(
    {
        "selector_probe",
        "local_match_no_unlock",
    }
)
_MISLEADING_EVENTS: frozenset[str] = frozenset(
    {
        "decoy_reward_reset",
        "false_progress_under_wrong_selector",
    }
)
_STRUCTURAL_NEGATIVE_EVENTS: frozenset[str] = frozenset(
    {
        "wrong_switch",
        "wrong_selector_or_switch",
        "wrong_order",
        "wrong_order_reset",
    }
)
_NO_EFFECT_EVENTS: frozenset[str] = frozenset(
    {
        "empty_interaction",
        "noop",
        "wall",
        "blocked_by_object",
        "invalid_click",
        "unused_click",
        "missing_rule",
        "inactive_goal_blocked",
        "redundant_collect",
        "redundant_post_goal_interaction",
    }
)
_KNOWN_EVENTS: frozenset[str] = frozenset().union(
    _STRONG_POSITIVE_EVENTS,
    _PROGRESS_EVENTS,
    _DIAGNOSTIC_EVENTS,
    _MISLEADING_EVENTS,
    _STRUCTURAL_NEGATIVE_EVENTS,
    _NO_EFFECT_EVENTS,
    {"move"},
)


def _clamp(value: float, *, lower: float, upper: float) -> float:
    return float(max(lower, min(upper, value)))


def action_family(action: str) -> str:
    if action.startswith("click:"):
        return "click"
    if action.startswith("interact"):
        return "interact"
    if action == "wait":
        return "wait"
    return "move"


def is_known_transition_event(event: str | None) -> bool:
    return str(event or "") in _KNOWN_EVENTS


def transition_usefulness_target(
    action: str,
    reward: float,
    event: str | None,
    delta_norm: float,
) -> float:
    family = action_family(action)
    event_name = str(event or "")
    delta_norm = max(0.0, float(delta_norm))

    if event_name in _STRONG_POSITIVE_EVENTS:
        return 1.1 if event_name == "goal_reached" else 0.85
    if event_name == "correct_collect":
        return 0.55
    if event_name in {"delayed_correct_collect", "selector_sequence_progress"}:
        return 0.3
    if event_name == "selector_candidate":
        return 0.35
    if event_name == "selector_probe":
        return 0.1
    if event_name == "local_match_no_unlock":
        return 0.05
    if event_name == "move":
        return _clamp(0.08 + (0.14 * min(delta_norm, 1.0)), lower=0.05, upper=0.24)
    if event_name in _MISLEADING_EVENTS:
        return -0.2
    if event_name in _STRUCTURAL_NEGATIVE_EVENTS:
        return -0.45
    if event_name in _NO_EFFECT_EVENTS:
        return -0.45 if family in {"interact", "click"} else -0.12

    if reward > 0.0:
        return _clamp(float(reward) + (0.15 * min(delta_norm, 1.0)), lower=0.3, upper=1.1)
    if family in {"click", "select"} and delta_norm >= 0.2:
        return 0.18
    if family == "interact" and delta_norm >= 0.2:
        return 0.12
    if family == "move" and delta_norm >= 0.1:
        return _clamp(0.08 + (0.14 * min(delta_norm, 1.0)), lower=0.05, upper=0.24)
    if delta_norm >= 0.2:
        return 0.08 if family not in {"interact", "click"} else 0.02
    return -0.35 if family in {"interact", "click"} else -0.1


def transition_policy_supervision(
    action: str,
    reward: float,
    event: str | None,
    delta_norm: float,
) -> PolicySupervision:
    family = action_family(action)
    event_name = str(event or "")
    delta_norm = max(0.0, float(delta_norm))

    if event_name in _STRONG_POSITIVE_EVENTS:
        return PolicySupervision(target=1.0, weight=1.6)
    if event_name == "correct_collect":
        return PolicySupervision(target=0.9, weight=1.35)
    if event_name in {"delayed_correct_collect", "selector_sequence_progress"}:
        return PolicySupervision(target=0.65, weight=1.1)
    if event_name == "selector_candidate":
        return PolicySupervision(target=0.65, weight=1.1)
    if event_name == "selector_probe":
        return PolicySupervision(target=0.3, weight=0.85)
    if event_name == "local_match_no_unlock":
        return PolicySupervision(target=0.2, weight=0.8)
    if event_name == "move":
        return PolicySupervision(target=0.4, weight=0.75, sibling_move_target=0.2, sibling_move_weight=0.35)
    if event_name in _MISLEADING_EVENTS:
        return PolicySupervision(target=0.0, weight=1.6 if family in {"interact", "click"} else 1.2)
    if event_name in _STRUCTURAL_NEGATIVE_EVENTS:
        return PolicySupervision(target=0.0, weight=1.8 if family in {"interact", "click"} else 1.3)
    if event_name in _NO_EFFECT_EVENTS:
        return PolicySupervision(target=0.0, weight=2.0 if family in {"interact", "click"} else 1.35)

    if reward > 0.0:
        if family in {"click", "select"}:
            return PolicySupervision(target=0.95, weight=1.35, same_type_target=0.18, same_type_weight=0.35)
        return PolicySupervision(target=1.0, weight=1.5)
    if family in {"click", "select"} and delta_norm >= 0.2:
        return PolicySupervision(target=0.4, weight=0.85, same_type_target=0.15, same_type_weight=0.3)
    if family == "interact" and delta_norm >= 0.2:
        return PolicySupervision(target=0.25, weight=0.85)
    if family == "move" and delta_norm >= 0.1:
        return PolicySupervision(target=0.4, weight=0.75, sibling_move_target=0.25, sibling_move_weight=0.45)
    if delta_norm >= 0.2:
        return PolicySupervision(
            target=0.45 if family == "move" else 0.25,
            weight=0.9,
            sibling_move_target=0.18 if family == "move" else 0.0,
            sibling_move_weight=0.3 if family == "move" else 0.0,
        )
    return PolicySupervision(target=0.0, weight=1.6 if family in {"interact", "click"} else 1.1)
