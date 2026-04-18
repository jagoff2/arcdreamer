from __future__ import annotations

import numpy as np

from arcagi.core.types import GridObservation, ObjectState, StructuredState, Transition
from arcagi.perception.object_encoder import extract_structured_state
from arcagi.planning.rule_induction import EpisodeRuleInducer


def _observation(grid: np.ndarray) -> GridObservation:
    return GridObservation(
        task_id="arc/test",
        episode_id="episode-0",
        step_index=0,
        grid=grid,
        available_actions=("up", "right", "wait"),
    )


def _object(
    object_id: str,
    color: int,
    cells: tuple[tuple[int, int], ...],
    *,
    tags: tuple[str, ...],
) -> ObjectState:
    ys = [cell[0] for cell in cells]
    xs = [cell[1] for cell in cells]
    return ObjectState(
        object_id=object_id,
        color=color,
        cells=cells,
        bbox=(min(ys), min(xs), max(ys), max(xs)),
        centroid=(float(sum(ys)) / len(ys), float(sum(xs)) / len(xs)),
        area=len(cells),
        tags=tags,
    )


def _interaction_state(*, step_index: int = 0, right_removed: bool = False) -> StructuredState:
    agent = _object("agent", 2, ((1, 1),), tags=("agent",))
    left = _object("left", 3, ((1, 0),), tags=("interactable",))
    objects = [agent, left]
    if not right_removed:
        objects.append(_object("right", 4, ((1, 2),), tags=("interactable",)))
    grid = np.zeros((3, 3), dtype=np.int64)
    for obj in objects:
        for y, x in obj.cells:
            grid[y, x] = obj.color
    return StructuredState(
        task_id="rule-induction/test",
        episode_id="episode-0",
        step_index=step_index,
        grid_shape=grid.shape,
        grid_signature=tuple(int(value) for value in grid.reshape(-1)),
        objects=tuple(objects),
        relations=(),
        affordances=("interact_left", "interact_right", "wait"),
        action_roles=(("interact_left", "interact"), ("interact_right", "interact"), ("wait", "wait")),
    )


def test_rule_inducer_prefers_action_that_changes_objects_with_reward() -> None:
    before = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int64,
            )
        )
    )
    after = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 2, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int64,
            )
        )
    )
    inducer = EpisodeRuleInducer()
    inducer.record(
        Transition(
            state=before,
            action="right",
            reward=1.0,
            next_state=after,
            terminated=False,
        )
    )

    assert inducer.action_score(before, "right") > inducer.action_score(before, "wait")


def test_rule_inducer_transition_delta_detects_motion_without_tags() -> None:
    before = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 3],
                ],
                dtype=np.int64,
            )
        )
    )
    after = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 2, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 3],
                ],
                dtype=np.int64,
            )
        )
    )

    delta = EpisodeRuleInducer._state_delta(before, after)

    assert delta > 0.1


def test_rule_inducer_can_prefer_progress_toward_stable_object_from_motion_history() -> None:
    before = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 2, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 3],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            )
        )
    )
    moved_right = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 2, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 3],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            )
        )
    )
    moved_left = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 3],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            )
        )
    )

    inducer = EpisodeRuleInducer()
    inducer.record(
        Transition(
            state=before,
            action="right",
            reward=0.0,
            next_state=moved_right,
            terminated=False,
        )
    )
    inducer.record(
        Transition(
            state=before,
            action="up",
            reward=0.0,
            next_state=moved_left,
            terminated=False,
        )
    )

    assert inducer.action_score(before, "right") > inducer.action_score(before, "up")


def test_rule_inducer_targets_local_interaction_signature_instead_of_smearing_family_value() -> None:
    before = _interaction_state(step_index=0, right_removed=False)
    fail_after = _interaction_state(step_index=1, right_removed=False)
    success_after = _interaction_state(step_index=1, right_removed=True)

    inducer = EpisodeRuleInducer()
    inducer.record(
        Transition(
            state=before,
            action="interact_left",
            reward=-0.05,
            next_state=fail_after,
            terminated=False,
        )
    )
    inducer.record(
        Transition(
            state=before,
            action="interact_right",
            reward=0.25,
            next_state=success_after,
            terminated=False,
        )
    )

    assert inducer.action_score(before, "interact_right") > inducer.action_score(before, "interact_left")
