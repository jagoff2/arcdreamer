from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

from arcagi.agents.base import BaseAgent
from arcagi.core.types import ActionName, GridObservation, StructuredState, Transition


class ObjectEventRuntimeExtractor(BaseAgent):
    """Runtime observation extractor with no policy or learned update."""

    def __init__(self) -> None:
        super().__init__(name="object_event_runtime_extractor")

    def act(self, observation: GridObservation) -> ActionName:
        raise RuntimeError("ObjectEventRuntimeExtractor has no action policy")

    def on_transition(self, transition: Transition) -> None:
        return None


@dataclass(frozen=True)
class ExtractedSelectedTransition:
    before_state: StructuredState
    after_state: StructuredState
    transition: Transition


def extract_selected_transition_from_grid(
    extractor: ObjectEventRuntimeExtractor,
    *,
    before_observation: GridObservation,
    selected_action: ActionName,
    after_observation: GridObservation,
    reward: float,
    terminated: bool,
    info: Mapping[str, object] | None = None,
) -> ExtractedSelectedTransition:
    before_state = extractor.observe(before_observation)
    extractor.last_state = before_state
    extractor.last_action = str(selected_action)
    transition_info = dict(info or {})
    after_state = extractor.update_after_step(
        after_observation,
        reward=float(reward),
        terminated=bool(terminated),
        info=transition_info,
    )
    transition = Transition(
        state=before_state,
        action=str(selected_action),
        reward=float(reward),
        next_state=after_state,
        terminated=bool(terminated),
        info=transition_info,
    )
    return ExtractedSelectedTransition(
        before_state=before_state,
        after_state=after_state,
        transition=transition,
    )
