"""Hindsight target construction for the ARC-AGI-3 goal-scientist trainer.

This module is deliberately independent of torch and the ARC toolkit.  The live
public-ARC trainer imports it after collecting black-box transitions and uses it
to convert sparse rewards and weak transition effects into supervised targets
that teach the learned agent what *experiments* were useful.

The design goal is not to smuggle per-game knowledge.  It only uses generic
transition evidence:

* reward appeared or increased;
* the visible structured state changed;
* an action was a selector / coordinate / interaction action;
* a later event made an earlier no-effect selector look like a necessary prefix;
* repeated state-action pairs stopped producing information.

Those signals are exactly the runtime learning pressure the current repo is
missing when the official public slice is mostly first-contact sparse feedback.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping, MutableMapping, Sequence
import math
import re

try:  # NumPy is available in the repo, but this module stays usable without it.
    import numpy as _np  # type: ignore
except Exception:  # pragma: no cover - exercised only in stripped environments.
    _np = None

_EPS = 1.0e-9
_COORD_RE = re.compile(r"-?\d+")


@dataclass(frozen=True)
class TransitionCredit:
    """Dense training target derived from one black-box transition.

    ``usefulness`` is the scalar target used by the world-model usefulness head.
    ``policy_target`` is a squashed probability-like target for the action head.
    ``diagnostic_weight`` should weight the loss upward when a transition teaches
    a reusable interaction fact, and downward for stale repeated no-effect moves.
    """

    reward: float
    delta_norm: float
    no_effect: bool
    terminal: bool
    action_family: str
    is_interaction: bool
    is_coordinate: bool
    novelty: float
    immediate_usefulness: float
    hindsight_return: float
    sequence_credit: float
    selector_credit: float
    no_effect_penalty: float
    usefulness: float
    policy_target: float
    diagnostic_weight: float
    tags: tuple[str, ...]


def finite_float(value: Any, default: float = 0.0) -> float:
    """Return a finite float for heterogeneous environment values."""

    try:
        out = float(value)
    except Exception:
        return float(default)
    if math.isnan(out) or math.isinf(out):
        return float(default)
    return out


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def action_text(action: Any) -> str:
    if action is None:
        return "none"
    if isinstance(action, str):
        return action
    name = getattr(action, "name", None)
    if isinstance(name, str):
        return name
    value = getattr(action, "value", None)
    if isinstance(value, str):
        return value
    return str(action)


def action_family(action: Any) -> str:
    """Map repo/toolkit action spellings into generic action families."""

    text = action_text(action).strip().lower()
    if not text:
        return "none"
    if text.startswith("click:") or text.startswith("select:"):
        return "coordinate"
    if text in {"action6", "a6", "coord", "coordinate", "click", "select"}:
        return "coordinate"
    if text in {"action5", "a5", "interact", "select", "execute", "use", "toggle", "rotate"}:
        return "interact"
    if text in {"action1", "a1", "up", "north", "move_up"}:
        return "move_up"
    if text in {"action2", "a2", "down", "south", "move_down"}:
        return "move_down"
    if text in {"action3", "a3", "left", "west", "move_left"}:
        return "move_left"
    if text in {"action4", "a4", "right", "east", "move_right"}:
        return "move_right"
    if text in {"reset", "restart"}:
        return "reset"
    if text in {"wait", "noop", "no_op", "none"}:
        return "wait"
    if "click" in text or "coord" in text:
        return "coordinate"
    if "interact" in text or "select" in text or "toggle" in text:
        return "interact"
    if any(token in text for token in ("up", "down", "left", "right")):
        return "move"
    return "other"


def is_coordinate_action(action: Any) -> bool:
    return action_family(action) == "coordinate"


def is_interaction_action(action: Any) -> bool:
    return action_family(action) in {"coordinate", "interact"}


def _iter_vector(value: Any) -> list[float]:
    if value is None:
        return []
    if _np is not None:
        try:
            arr = _np.asarray(value, dtype=float).reshape(-1)
            return [finite_float(x) for x in arr.tolist()]
        except Exception:
            pass
    if isinstance(value, (str, bytes)):
        return []
    try:
        return [finite_float(x) for x in value]
    except Exception:
        return []


def transition_vector(state: Any) -> list[float]:
    """Extract a numeric vector from the repo's StructuredState or a test fake."""

    if state is None:
        return []
    method = getattr(state, "transition_vector", None)
    if callable(method):
        try:
            return _iter_vector(method())
        except Exception:
            return []
    for attr in ("vector", "features", "state_vector"):
        if hasattr(state, attr):
            return _iter_vector(getattr(state, attr))
    return []


def vector_delta_norm(delta: Any) -> float:
    vec = _iter_vector(delta)
    if not vec:
        return 0.0
    return math.sqrt(sum(x * x for x in vec))


def sample_delta_norm(sample: Mapping[str, Any]) -> float:
    if "delta_norm" in sample:
        return max(0.0, finite_float(sample.get("delta_norm")))
    if "delta" in sample:
        return vector_delta_norm(sample.get("delta"))
    state_vec = transition_vector(sample.get("state"))
    next_vec = transition_vector(sample.get("next_state"))
    if not state_vec or not next_vec:
        return 0.0
    width = min(len(state_vec), len(next_vec))
    return math.sqrt(sum((next_vec[i] - state_vec[i]) ** 2 for i in range(width)))


def _hashable(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Mapping):
        return tuple(sorted((str(k), _hashable(v)) for k, v in value.items()))
    if isinstance(value, (list, tuple, set, frozenset)):
        return tuple(_hashable(v) for v in value)
    if hasattr(value, "__dict__"):
        return _hashable(vars(value))
    return repr(value)


def coarse_state_signature(state: Any) -> tuple[Any, ...]:
    """Make a stable, game-agnostic state signature for novelty accounting.

    The signature intentionally ignores raw environment names and any hidden
    metadata.  It uses the public structured state only: grid/object summaries,
    affordances, and generic flags when available.
    """

    if state is None:
        return ("state", "none")

    for attr in ("grid_signature", "signature", "hash_key", "state_hash"):
        if hasattr(state, attr):
            value = getattr(state, attr)
            try:
                value = value() if callable(value) else value
                if value is not None:
                    return ("state", attr, _hashable(value))
            except Exception:
                pass

    parts: list[Any] = ["state"]
    objects = getattr(state, "objects", None)
    if objects is not None:
        object_parts: list[Any] = []
        try:
            iterable = list(objects)
        except Exception:
            iterable = []
        for obj in iterable[:64]:
            object_parts.append(
                _hashable(
                    {
                        "color": getattr(obj, "color", getattr(obj, "value", None)),
                        "kind": getattr(obj, "kind", getattr(obj, "role", None)),
                        "size": getattr(obj, "size", getattr(obj, "area", None)),
                        "centroid": getattr(obj, "centroid", None),
                        "bbox": getattr(obj, "bbox", getattr(obj, "bounds", None)),
                    }
                )
            )
        parts.append(tuple(sorted(object_parts, key=repr)))

    for attr in ("affordances", "flags", "flags_dict", "metadata"):
        if not hasattr(state, attr):
            continue
        value = getattr(state, attr)
        try:
            value = value() if callable(value) else value
        except Exception:
            continue
        if value is not None:
            parts.append((attr, _hashable(value)))

    vec = transition_vector(state)
    if vec:
        # Bucket the vector slightly.  The purpose is novelty, not exact physics.
        parts.append(("vec", tuple(round(x, 3) for x in vec[:64])))
    return tuple(parts)


def state_action_key(sample: Mapping[str, Any]) -> tuple[Any, ...]:
    return coarse_state_signature(sample.get("state")) + ("action", action_family(sample.get("action")))


def target_language_tokens(credit: TransitionCredit) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Build grounded language targets that describe the discovered evidence."""

    if credit.reward > 0.0:
        effect = "reward"
    elif credit.delta_norm > 1.0e-4:
        effect = "state_change"
    elif credit.no_effect:
        effect = "no_effect"
    else:
        effect = "uncertain"

    novelty = "novel" if credit.novelty > 0.5 else "known"
    belief = (
        "belief",
        "effect",
        effect,
        "action",
        credit.action_family,
        "evidence",
        novelty,
        "terminal",
        "yes" if credit.terminal else "no",
    )

    if "sequence_prefix" in credit.tags or "selector_binding" in credit.tags:
        question = ("question", "bind_selector_sequence", "test_prefix", credit.action_family)
        plan = ("plan", "preserve_prefix", "then_probe_objective", "action", credit.action_family)
    elif credit.no_effect and credit.novelty > 0.5:
        question = ("question", "does_action_prime_later_effect", "action", credit.action_family)
        plan = ("plan", "diagnostic_probe", "follow_with_contrast", "frontier")
    elif credit.no_effect:
        question = ("question", "avoid_repeated_no_effect", "action", credit.action_family)
        plan = ("plan", "switch_action_family", "seek_information_gain")
    elif credit.is_interaction:
        question = ("question", "what_object_or_contact_changed", "action", credit.action_family)
        plan = ("plan", "repeat_near_similar_object", "verify_effect")
    else:
        question = ("question", "which_transition_changed_state", "action", credit.action_family)
        plan = ("plan", "extend_path", "test_adjacent_action", credit.action_family)
    return belief, question, plan


def score_transition(
    sample: Mapping[str, Any],
    *,
    future_return: float = 0.0,
    future_event_distance: int | None = None,
    seen_state_actions: set[tuple[Any, ...]] | None = None,
) -> TransitionCredit:
    """Score one transition as a scientific experiment.

    The score is intentionally generic.  It rewards evidence-bearing actions,
    gives delayed credit to selector/contact prefixes, and penalizes repeated
    no-effect transitions that no longer buy information.
    """

    reward = finite_float(sample.get("reward", 0.0))
    positive_reward = max(0.0, reward)
    delta_norm = sample_delta_norm(sample)
    terminal = bool(sample.get("terminated", sample.get("terminal", False)))
    family = action_family(sample.get("action"))
    coordinate = family == "coordinate"
    interaction = family in {"coordinate", "interact"}
    key = state_action_key(sample)
    novelty = 1.0 if seen_state_actions is None or key not in seen_state_actions else 0.0
    no_effect = delta_norm <= 1.0e-5 and positive_reward <= 1.0e-9 and not terminal

    tags: list[str] = []
    if positive_reward > 0.0:
        tags.append("reward")
    if delta_norm > 1.0e-5:
        tags.append("state_change")
    if novelty > 0.5:
        tags.append("novel_probe")
    if interaction:
        tags.append("interaction_probe")
    if coordinate:
        tags.append("coordinate_probe")
    if no_effect:
        tags.append("no_effect")

    bounded_delta = clamp(delta_norm, 0.0, 3.0)
    immediate = 1.50 * positive_reward + 0.35 * bounded_delta + 0.12 * novelty
    if interaction and (positive_reward > 0.0 or delta_norm > 1.0e-5):
        immediate += 0.28
    if terminal and positive_reward > 0.0:
        immediate += 0.45

    future_return = max(0.0, finite_float(future_return))
    distance = future_event_distance if future_event_distance is not None and future_event_distance > 0 else None
    distance_decay = 1.0 / float(distance + 1) if distance is not None else 0.0

    sequence_credit = 0.0
    selector_credit = 0.0
    if future_return > 0.0 and distance is not None:
        sequence_credit = 0.35 * future_return * distance_decay
        if interaction:
            selector_credit = 0.90 * future_return * distance_decay
            tags.append("selector_binding")
        if no_effect:
            tags.append("sequence_prefix")
        if distance <= 3:
            tags.append("near_delayed_effect")
        else:
            tags.append("delayed_effect")

    no_effect_penalty = 0.0
    if no_effect:
        # A first no-effect probe still has diagnostic value; repeated no-effect
        # probes should be hard negatives unless a later event rescues them.
        no_effect_penalty = 0.08 if novelty > 0.5 else 0.34
        if future_return > 0.0:
            no_effect_penalty *= 0.25

    usefulness = clamp(immediate + sequence_credit + selector_credit - no_effect_penalty, 0.0, 3.5)
    policy_target = clamp(1.0 - math.exp(-usefulness), 0.0, 1.0)
    diagnostic_weight = 1.0
    diagnostic_weight += 0.25 * novelty
    diagnostic_weight += 0.40 if "state_change" in tags else 0.0
    diagnostic_weight += 0.65 if "reward" in tags else 0.0
    diagnostic_weight += 0.55 if "selector_binding" in tags else 0.0
    diagnostic_weight -= 0.35 if no_effect and future_return <= 0.0 and novelty <= 0.5 else 0.0
    diagnostic_weight = clamp(diagnostic_weight, 0.25, 3.0)

    return TransitionCredit(
        reward=reward,
        delta_norm=delta_norm,
        no_effect=no_effect,
        terminal=terminal,
        action_family=family,
        is_interaction=interaction,
        is_coordinate=coordinate,
        novelty=novelty,
        immediate_usefulness=immediate,
        hindsight_return=future_return,
        sequence_credit=sequence_credit,
        selector_credit=selector_credit,
        no_effect_penalty=no_effect_penalty,
        usefulness=usefulness,
        policy_target=policy_target,
        diagnostic_weight=diagnostic_weight,
        tags=tuple(dict.fromkeys(tags)),
    )


def _event_value(sample: Mapping[str, Any]) -> float:
    reward_value = max(0.0, finite_float(sample.get("reward", 0.0)))
    delta_value = clamp(sample_delta_norm(sample), 0.0, 3.0) * 0.25
    terminal_bonus = 0.15 if bool(sample.get("terminated", sample.get("terminal", False))) and reward_value > 0.0 else 0.0
    return reward_value + delta_value + terminal_bonus


def _nearest_future_event_distances(samples: Sequence[Mapping[str, Any]], sequence_horizon: int) -> list[int | None]:
    n = len(samples)
    distances: list[int | None] = [None] * n
    event_indices = [i for i, sample in enumerate(samples) if _event_value(sample) > 0.05]
    cursor = 0
    for i in range(n):
        while cursor < len(event_indices) and event_indices[cursor] <= i:
            cursor += 1
        if cursor >= len(event_indices):
            continue
        distance = event_indices[cursor] - i
        if 0 < distance <= sequence_horizon:
            distances[i] = distance
    return distances


def relabel_episode_samples(
    samples: Sequence[Mapping[str, Any]],
    *,
    gamma: float = 0.92,
    sequence_horizon: int = 12,
    seen_state_actions: set[tuple[Any, ...]] | None = None,
) -> list[dict[str, Any]]:
    """Return copies of episode samples with hindsight scientist targets.

    ``samples`` must be one episode in chronological order.  The return keeps all
    original keys and adds:

    * ``usefulness``
    * ``policy_target``
    * ``diagnostic_weight``
    * ``transition_credit``
    * ``belief_tokens`` / ``question_tokens`` / ``plan_tokens``
    * ``hindsight_return``
    * ``future_event_distance``
    """

    if gamma <= 0.0 or gamma > 1.0:
        raise ValueError("gamma must be in (0, 1]")
    if sequence_horizon <= 0:
        raise ValueError("sequence_horizon must be positive")

    local_seen: set[tuple[Any, ...]] = seen_state_actions if seen_state_actions is not None else set()
    raw = [dict(sample) for sample in samples]
    n = len(raw)
    future_distances = _nearest_future_event_distances(raw, sequence_horizon)

    returns_after: list[float] = [0.0] * n
    running = 0.0
    for i in range(n - 1, -1, -1):
        returns_after[i] = gamma * running
        running = _event_value(raw[i]) + gamma * running

    relabeled: list[dict[str, Any]] = []
    for i, sample in enumerate(raw):
        credit = score_transition(
            sample,
            future_return=returns_after[i],
            future_event_distance=future_distances[i],
            seen_state_actions=local_seen,
        )
        key = state_action_key(sample)
        local_seen.add(key)

        belief, question, plan = target_language_tokens(credit)
        out = dict(sample)
        out["delta_norm"] = credit.delta_norm
        out["hindsight_return"] = returns_after[i]
        out["future_event_distance"] = future_distances[i]
        out["usefulness"] = credit.usefulness
        out["policy_target"] = credit.policy_target
        out["diagnostic_weight"] = credit.diagnostic_weight
        out["transition_credit"] = asdict(credit)
        out["credit_tags"] = credit.tags
        out["belief_tokens"] = tuple(out.get("belief_tokens") or belief)
        out["question_tokens"] = tuple(out.get("question_tokens") or question)
        out["plan_tokens"] = tuple(out.get("plan_tokens") or plan)
        relabeled.append(out)
    return relabeled


def relabel_transition_stream(
    samples: Iterable[Mapping[str, Any]],
    *,
    gamma: float = 0.92,
    sequence_horizon: int = 12,
) -> list[dict[str, Any]]:
    """Relabel a mixed stream, grouping by episode when possible."""

    buckets: dict[Any, list[Mapping[str, Any]]] = {}
    order: list[Any] = []
    for index, sample in enumerate(samples):
        episode_id = sample.get("episode_id", sample.get("episode", 0))
        key = episode_id if episode_id is not None else 0
        if key not in buckets:
            buckets[key] = []
            order.append(key)
        buckets[key].append(sample)

    seen: set[tuple[Any, ...]] = set()
    out: list[dict[str, Any]] = []
    for key in order:
        out.extend(relabel_episode_samples(buckets[key], gamma=gamma, sequence_horizon=sequence_horizon, seen_state_actions=seen))
    return out


def summarize_credit(samples: Sequence[Mapping[str, Any]]) -> dict[str, float]:
    """Small metric helper for CLI progress reporting."""

    if not samples:
        return {
            "samples": 0.0,
            "avg_usefulness": 0.0,
            "avg_policy_target": 0.0,
            "avg_diagnostic_weight": 0.0,
            "selector_binding_fraction": 0.0,
            "no_effect_fraction": 0.0,
        }
    count = float(len(samples))
    selector = 0.0
    no_effect = 0.0
    for sample in samples:
        tags = tuple(sample.get("credit_tags", ()))
        selector += 1.0 if "selector_binding" in tags else 0.0
        no_effect += 1.0 if "no_effect" in tags else 0.0
    return {
        "samples": count,
        "avg_usefulness": sum(finite_float(s.get("usefulness")) for s in samples) / count,
        "avg_policy_target": sum(finite_float(s.get("policy_target")) for s in samples) / count,
        "avg_diagnostic_weight": sum(finite_float(s.get("diagnostic_weight"), 1.0) for s in samples) / count,
        "selector_binding_fraction": selector / count,
        "no_effect_fraction": no_effect / count,
    }
