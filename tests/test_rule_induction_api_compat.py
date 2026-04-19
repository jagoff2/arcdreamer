from __future__ import annotations

import numpy as np

from arcagi.core.types import ObjectState, StructuredState, Transition
from arcagi.planning.rule_induction import EpisodeRuleInducer, action_target_signatures, object_signature


def _object(object_id: str, color: int, cells: tuple[tuple[int, int], ...], tags: tuple[str, ...] = ()) -> ObjectState:
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


def _state(step_index: int, objects: tuple[ObjectState, ...], affordances: tuple[str, ...] = ("interact_right", "wait")) -> StructuredState:
    grid = np.zeros((3, 3), dtype=np.int64)
    for obj in objects:
        for y, x in obj.cells:
            grid[y, x] = obj.color
    return StructuredState(
        task_id="task",
        episode_id="episode",
        step_index=step_index,
        grid_shape=grid.shape,
        grid_signature=tuple(int(v) for v in grid.reshape(-1)),
        objects=objects,
        relations=(),
        affordances=affordances,
        flags=(),
        inventory=(),
    )


def test_rule_induction_import_surface_and_core_methods() -> None:
    state = _state(
        0,
        (
            _object("agent", 1, ((1, 1),), tags=("agent",)),
            _object("door", 4, ((1, 2),), tags=("target", "interactable")),
        ),
    )
    signature = object_signature(state.objects[1])
    assert signature[0] == 4
    assert action_target_signatures(state, "interact_right") == (signature,)

    inducer = EpisodeRuleInducer()
    next_state = _state(1, (_object("agent", 1, ((1, 1),), tags=("agent",)),))
    inducer.record(
        Transition(
            state=state,
            action="interact_right",
            reward=1.0,
            next_state=next_state,
            terminated=False,
            info={},
        )
    )

    assert inducer.action_score(state, "interact_right") > inducer.action_score(state, "wait")
    assert inducer.applicable_hypotheses(state, "interact_right")
    assert inducer.top_claims(state=state, action="interact_right")
