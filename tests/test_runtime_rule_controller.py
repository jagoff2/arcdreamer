from __future__ import annotations

import numpy as np

from arcagi.core.types import ActionThought, GridObservation, RuntimeThought, Transition
from arcagi.perception.object_encoder import extract_structured_state
from arcagi.planning.rule_induction import object_signature
from arcagi.planning.runtime_rule_controller import RuntimeRuleController


def _observation(
    grid: np.ndarray,
    *,
    step_index: int,
    actions: tuple[str, ...] = ("1", "2", "3", "4", "7"),
    action_roles: dict[str, str] | None = None,
    cell_tags: dict[tuple[int, int], tuple[str, ...]] | None = None,
) -> GridObservation:
    return GridObservation(
        task_id="arc/test",
        episode_id="episode-0",
        step_index=step_index,
        grid=grid,
        available_actions=actions,
        extras={
            "action_roles": action_roles
            or {
                "1": "move_up",
                "2": "move_down",
                "3": "move_left",
                "4": "move_right",
                "5": "select_cycle",
                "6": "click",
                "7": "undo",
            },
            "cell_tags": cell_tags or {},
        },
    )


def _signature_for_tag(state, tag: str):
    for obj in state.objects:
        if tag in obj.tags:
            return object_signature(obj)
    raise AssertionError(f"missing tag {tag}")


def test_runtime_rule_controller_uses_undo_for_move_diagnostics() -> None:
    before = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 2, 0, 3, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 4, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
        )
    )
    after_probe = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 2, 0, 3, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 4, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=1,
        )
    )

    controller = RuntimeRuleController()
    initial_plan = controller.propose(before)

    assert initial_plan is not None
    assert initial_plan.action in {"1", "2", "3", "4"}

    controller.observe_transition(
        Transition(
            state=before,
            action="2",
            reward=0.0,
            next_state=after_probe,
            terminated=False,
        )
    )

    undo_plan = controller.propose(after_probe)

    assert undo_plan is not None
    assert undo_plan.action == "7"


def test_runtime_rule_controller_moves_toward_stable_goal_after_online_probes() -> None:
    start = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 2, 2, 0, 0, 0, 0],
                    [0, 2, 2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 4, 4, 0],
                    [0, 0, 0, 0, 4, 4, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
        )
    )
    moved_down = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 2, 2, 0, 0, 0, 0],
                    [0, 2, 2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 4, 4, 0],
                    [0, 0, 0, 0, 4, 4, 0],
                ],
                dtype=np.int64,
            ),
            step_index=1,
        )
    )
    moved_right = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 2, 2, 0, 0, 0],
                    [0, 0, 2, 2, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 4, 4, 0],
                    [0, 0, 0, 0, 4, 4, 0],
                ],
                dtype=np.int64,
            ),
            step_index=1,
        )
    )
    success = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2, 2, 0],
                    [0, 0, 0, 0, 2, 2, 0],
                    [0, 0, 0, 0, 4, 4, 0],
                    [0, 0, 0, 0, 4, 4, 0],
                ],
                dtype=np.int64,
            ),
            step_index=2,
        )
    )

    controller = RuntimeRuleController()
    controller.propose(start)
    controller.observe_transition(
        Transition(state=start, action="2", reward=0.0, next_state=moved_down, terminated=False)
    )
    controller.pending_undo = False
    controller.observe_transition(
        Transition(state=start, action="4", reward=0.0, next_state=moved_right, terminated=False)
    )
    controller.pending_undo = False
    controller.observe_transition(
        Transition(state=moved_right, action="2", reward=1.0, next_state=success, terminated=False)
    )

    plan = controller.propose(moved_right)

    assert plan is not None
    assert plan.action == "2"


def test_runtime_rule_controller_uses_selector_followup_from_runtime_thought() -> None:
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 3],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=("1", "2", "5", "6"),
        )
    )
    controller = RuntimeRuleController()
    controller.reference_state = state
    controller.reference_fingerprint = state.fingerprint()
    controller.action_visits[(controller.current_mode_key, "1")] = 1
    controller.action_visits[(controller.current_mode_key, "2")] = 1
    thought = RuntimeThought(
        belief_tokens=("uncertain",),
        question_tokens=("need", "test"),
        actions=(
            ActionThought(action="1", value=0.1, uncertainty=0.1),
            ActionThought(action="2", value=0.0, uncertainty=0.1),
            ActionThought(action="5", value=0.1, uncertainty=0.2, selector_followup=2.5),
            ActionThought(action="6", value=0.1, uncertainty=0.2, selector_followup=0.1),
        ),
    )

    plan = controller.propose(state, thought=thought)

    assert plan is not None
    assert plan.action == "5"


def test_runtime_rule_controller_tests_adjacent_interactable_after_motion_learning() -> None:
    actions = ("up", "down", "left", "right", "interact_up", "interact_down", "interact_left", "interact_right")
    start = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 9, 0, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=actions,
            cell_tags={
                (1, 1): ("agent",),
                (1, 3): ("interactable", "switch"),
                (4, 4): ("target",),
            },
        )
    )
    moved_down = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 3, 0, 0],
                    [0, 9, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=1,
            actions=actions,
            cell_tags={
                (2, 1): ("agent",),
                (1, 3): ("interactable", "switch"),
                (4, 4): ("target",),
            },
        )
    )
    moved_right = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 9, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=1,
            actions=actions,
            cell_tags={
                (1, 2): ("agent",),
                (1, 3): ("interactable", "switch"),
                (4, 4): ("target",),
            },
        )
    )

    controller = RuntimeRuleController()
    controller.propose(start)
    controller.observe_transition(Transition(state=start, action="down", reward=0.0, next_state=moved_down, terminated=False))
    controller.pending_undo = False
    controller.observe_transition(Transition(state=start, action="right", reward=0.0, next_state=moved_right, terminated=False))
    controller.pending_undo = False

    plan = controller.propose(moved_right)

    assert plan is not None
    assert plan.action == "interact_right"


def test_runtime_rule_controller_failed_interaction_writes_objective_contradiction() -> None:
    actions = ("up", "down", "left", "right", "interact_up", "interact_down", "interact_left", "interact_right")
    start = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 9, 0, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=actions,
            cell_tags={
                (1, 1): ("agent",),
                (1, 3): ("interactable", "switch"),
                (4, 4): ("target",),
            },
        )
    )
    moved_right = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 9, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=1,
            actions=actions,
            cell_tags={
                (1, 2): ("agent",),
                (1, 3): ("interactable", "switch"),
                (4, 4): ("target",),
            },
        )
    )

    controller = RuntimeRuleController()
    controller.propose(start)
    controller.observe_transition(Transition(state=start, action="right", reward=0.0, next_state=moved_right, terminated=False))
    controller.pending_undo = False
    controller.observe_transition(
        Transition(state=moved_right, action="interact_right", reward=-0.1, next_state=moved_right, terminated=False)
    )

    mover_signature = _signature_for_tag(moved_right, "agent")
    target_signature = _signature_for_tag(moved_right, "interactable")
    objective = controller.objective_hypotheses[(controller.current_mode_key, mover_signature, target_signature)]

    assert objective.interaction_tests == 1
    assert objective.failed_interactions == 1
    assert objective.evidence.contradiction >= 2
