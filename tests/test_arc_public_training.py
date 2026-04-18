from __future__ import annotations

import numpy as np

from arcagi.core.types import GridObservation
from arcagi.perception.object_encoder import extract_structured_state
from arcagi.training.arc_public import _belief_tokens, _plan_tokens, _question_tokens


def _state(actions: tuple[str, ...]) -> object:
    observation = GridObservation(
        task_id="arc/public-test",
        episode_id="episode-0",
        step_index=0,
        grid=np.array(
            [
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 3],
            ],
            dtype=np.int64,
        ),
        available_actions=actions,
        extras={
            "action_roles": {
                "1": "move_up",
                "5": "select_cycle",
                "click:4:4": "click",
                "interact_right": "interact",
            }
        },
    )
    return extract_structured_state(observation)


def test_arc_public_language_targets_probe_rules_when_selector_actions_exist() -> None:
    state = _state(("1", "5", "click:4:4"))

    belief = _belief_tokens(state, "5", reward=0.0, usefulness=0.1)
    question = _question_tokens(state, "5", reward=0.0, usefulness=0.1)

    assert belief == ("belief", "goal", "uncertain", "focus", "rule", "state", "probe")
    assert question == ("question", "need", "test", "focus", "rule", "state", "probe")


def test_arc_public_language_targets_confirm_goal_on_positive_move_reward() -> None:
    state = _state(("1",))

    belief = _belief_tokens(state, "1", reward=1.0, usefulness=0.5)
    question = _question_tokens(state, "1", reward=1.0, usefulness=0.5)

    assert belief == ("belief", "goal", "active", "focus", "target", "state", "explore")
    assert question == ("question", "need", "confirm", "focus", "target", "state", "explore")


def test_arc_public_plan_tokens_encode_action_focus_and_direction() -> None:
    state = _state(("1", "interact_right"))

    plan = _plan_tokens(state, "interact_right", reward=0.0, usefulness=0.2)

    assert plan == ("plan", "action", "interact", "direction", "right", "focus", "interactable", "state", "inactive")
