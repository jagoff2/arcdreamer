from __future__ import annotations

import numpy as np

from arcagi.core.types import ObjectState, StructuredState, Transition
from arcagi.planning.rule_induction import EpisodeRuleInducer


def _object(object_id: str, color: int, cells: tuple[tuple[int, int], ...], tags: tuple[str, ...] = ()) -> ObjectState:
    ys = [cell[0] for cell in cells]
    xs = [cell[1] for cell in cells]
    bbox = (min(ys), min(xs), max(ys), max(xs))
    centroid = (float(sum(ys)) / len(ys), float(sum(xs)) / len(xs))
    return ObjectState(
        object_id=object_id,
        color=color,
        cells=cells,
        bbox=bbox,
        centroid=centroid,
        area=len(cells),
        tags=tags,
    )


def _state(
    *,
    step_index: int,
    objects: tuple[ObjectState, ...],
    affordances: tuple[str, ...] = ("interact_right", "wait"),
    flags: tuple[tuple[str, str], ...] = (),
    inventory: tuple[tuple[str, str], ...] = (),
) -> StructuredState:
    grid = np.zeros((3, 3), dtype=np.int64)
    for obj in objects:
        for y, x in obj.cells:
            grid[y, x] = obj.color
    return StructuredState(
        task_id="task",
        episode_id="episode",
        step_index=step_index,
        grid_shape=grid.shape,
        grid_signature=tuple(int(value) for value in grid.reshape(-1)),
        objects=objects,
        relations=(),
        affordances=affordances,
        inventory=inventory,
        flags=flags,
    )


def _transition(
    *,
    before: StructuredState,
    after: StructuredState,
    action: str = "interact_right",
    reward: float = 1.0,
) -> Transition:
    return Transition(
        state=before,
        action=action,
        reward=reward,
        next_state=after,
        terminated=False,
        info={},
    )


def test_grounded_hypotheses_capture_target_reward_and_hidden_state() -> None:
    before = _state(
        step_index=0,
        objects=(
            _object("agent", 1, ((1, 1),), tags=("agent",)),
            _object("door", 4, ((1, 2),), tags=("target", "interactable")),
        ),
        flags=(("door_open", "0"),),
    )
    after = _state(
        step_index=1,
        objects=(
            _object("agent", 1, ((1, 1),), tags=("agent",)),
        ),
        flags=(("door_open", "1"),),
    )
    inducer = EpisodeRuleInducer()
    inducer.record(_transition(before=before, after=after, reward=1.0))

    hypotheses = inducer.applicable_hypotheses(before, "interact_right")
    rendered = {hypothesis.consequence.render() for hypothesis in hypotheses}
    assert "reward:episode=positive" in rendered
    assert "object:target_effect=disappeared" in rendered
    assert "flag:door_open=1" in rendered

    claims = inducer.top_claims(state=before, action="interact_right", limit=8)
    claim_objects = {claim.object for claim in claims}
    assert "reward:episode=positive" in claim_objects
    assert "object:target_effect=disappeared" in claim_objects


def test_grounded_hypotheses_keep_competing_reward_models() -> None:
    before = _state(
        step_index=0,
        objects=(
            _object("agent", 1, ((1, 1),), tags=("agent",)),
            _object("door", 4, ((1, 2),), tags=("target", "interactable")),
        ),
    )
    opened = _state(
        step_index=1,
        objects=(
            _object("agent", 1, ((1, 1),), tags=("agent",)),
        ),
        flags=(("door_open", "1"),),
    )
    unchanged = _state(
        step_index=1,
        objects=before.objects,
    )
    inducer = EpisodeRuleInducer()
    inducer.record(_transition(before=before, after=opened, reward=1.0))
    inducer.record(_transition(before=before, after=unchanged, reward=0.0))

    reward_hypotheses = {
        hypothesis.consequence.object: hypothesis
        for hypothesis in inducer.applicable_hypotheses(before, "interact_right")
        if hypothesis.consequence.predicate_type == "reward"
    }
    assert "positive" in reward_hypotheses
    assert "zero" in reward_hypotheses
    assert reward_hypotheses["positive"].contradiction > 0.0
    assert reward_hypotheses["zero"].support > 0.0


def test_grounded_hypothesis_bonus_prefers_supported_action() -> None:
    before = _state(
        step_index=0,
        objects=(
            _object("agent", 1, ((1, 1),), tags=("agent",)),
            _object("door", 4, ((1, 2),), tags=("target", "interactable")),
        ),
    )
    after = _state(
        step_index=1,
        objects=(
            _object("agent", 1, ((1, 1),), tags=("agent",)),
        ),
        flags=(("door_open", "1"),),
    )
    inducer = EpisodeRuleInducer()
    inducer.record(_transition(before=before, after=after, reward=1.0))

    assert inducer.action_score(before, "interact_right") > inducer.action_score(before, "wait")
