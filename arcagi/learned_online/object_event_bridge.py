from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from arcagi.core.types import ActionName, StructuredState, Transition
from arcagi.learned_online.event_tokens import (
    ActionTokenBatch,
    StateTokenBatch,
    TransitionEventTargets,
    build_transition_targets,
    encode_action_tokens,
    encode_state_tokens,
)


FORBIDDEN_MODEL_INPUT_METADATA_KEYS = frozenset(
    {
        "task_id",
        "episode_id",
        "game_id",
        "arc_game_id",
        "trace_id",
        "trace_cursor",
        "state_hash",
        "state_fingerprint",
        "action_sequence",
        "correct_action_index",
        "latent_rule",
        "teacher_action",
        "teacher_index",
    }
)


@dataclass(frozen=True)
class SelectedActionObservation:
    before: StructuredState
    selected_action: ActionName
    after: StructuredState
    reward: float
    terminated: bool
    legal_actions: tuple[ActionName, ...]
    info: Mapping[str, Any]


@dataclass(frozen=True)
class ObjectEventObservation:
    state_tokens: StateTokenBatch
    action_tokens: ActionTokenBatch
    transition_targets: TransitionEventTargets
    selected_action_index: int
    legal_action_count: int
    metadata: Mapping[str, Any]

    def model_input_metadata(self) -> dict[str, int]:
        return {
            "selected_action_index": int(self.selected_action_index),
            "legal_action_count": int(self.legal_action_count),
        }


def build_object_event_observation(
    observation: SelectedActionObservation,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> ObjectEventObservation:
    legal_actions = tuple(str(action) for action in observation.legal_actions)
    selected_action = str(observation.selected_action)
    if selected_action not in legal_actions:
        raise ValueError("selected action is not present in the legal action surface")
    transition = Transition(
        state=observation.before,
        action=selected_action,
        reward=float(observation.reward),
        next_state=observation.after,
        terminated=bool(observation.terminated),
        info=dict(observation.info),
    )
    targets = build_transition_targets(transition, actions=legal_actions)
    state_tokens = encode_state_tokens(observation.before)
    action_tokens = encode_action_tokens(observation.before, legal_actions)
    diagnostics = dict(metadata or {})
    diagnostics.update(
        {
            "selected_action_index": int(targets.actual_action_index),
            "legal_action_count": int(len(legal_actions)),
        }
    )
    return ObjectEventObservation(
        state_tokens=state_tokens,
        action_tokens=action_tokens,
        transition_targets=targets,
        selected_action_index=int(targets.actual_action_index),
        legal_action_count=int(len(legal_actions)),
        metadata=diagnostics,
    )


def forbidden_metadata_keys(mapping: Mapping[str, Any]) -> tuple[str, ...]:
    return tuple(sorted(FORBIDDEN_MODEL_INPUT_METADATA_KEYS.intersection(str(key) for key in mapping.keys())))


def assert_no_forbidden_metadata_as_model_input(mapping: Mapping[str, Any]) -> None:
    forbidden = forbidden_metadata_keys(mapping)
    if forbidden:
        raise AssertionError(f"forbidden metadata keys in model input: {list(forbidden)}")
