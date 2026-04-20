from __future__ import annotations

import numpy as np

from arcagi.core.representation_repair import RepresentationRepairWorkspace
from arcagi.core.types import GridObservation, StructuredState
from arcagi.perception.object_encoder import extract_structured_state


def _state(
    grid: np.ndarray,
    *,
    actions: tuple[str, ...] = ("wait",),
    action_roles: dict[str, str] | None = None,
    cell_tags: dict[tuple[int, int], tuple[str, ...]] | None = None,
) -> StructuredState:
    return extract_structured_state(
        GridObservation(
            task_id="repair/test",
            episode_id="episode-0",
            step_index=0,
            grid=grid,
            available_actions=actions,
            extras={
                "action_roles": action_roles or {"wait": "wait", "right": "move_right"},
                "cell_tags": cell_tags or {},
            },
        )
    )


def test_representation_repair_workspace_executes_split_after_split_evidence() -> None:
    workspace = RepresentationRepairWorkspace()
    merged = _state(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 2, 2, 0],
                [0, 2, 2, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
    )
    split = _state(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 2, 0, 2],
                [0, 2, 0, 2],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
    )

    before = workspace.augment(merged)
    after = workspace.augment(split)
    workspace.observe_transition(before, "wait", 0.0, after, terminated=False)

    repaired = workspace.augment(merged)

    assert len(repaired.objects) == 2


def test_representation_repair_workspace_executes_merge_after_merge_evidence() -> None:
    workspace = RepresentationRepairWorkspace()
    separated = _state(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 2, 0, 2],
                [0, 2, 0, 2],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
    )
    merged = _state(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 2, 2, 0],
                [0, 2, 2, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
    )

    before = workspace.augment(separated)
    after = workspace.augment(merged)
    workspace.observe_transition(before, "wait", 0.0, after, terminated=False)

    repaired = workspace.augment(separated)

    assert len(repaired.objects) == 1
    assert repaired.objects[0].area == 4


def test_representation_repair_workspace_relabels_track_as_mover_after_motion() -> None:
    workspace = RepresentationRepairWorkspace()
    before_raw = _state(
        np.array(
            [
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 0],
            ],
            dtype=np.int64,
        ),
        actions=("right",),
        action_roles={"right": "move_right"},
        cell_tags={(1, 1): ("agent",)},
    )
    after_raw = _state(
        np.array(
            [
                [0, 0, 0],
                [0, 0, 2],
                [0, 0, 0],
            ],
            dtype=np.int64,
        ),
        actions=("right",),
        action_roles={"right": "move_right"},
        cell_tags={(1, 2): ("agent",)},
    )

    before = workspace.augment(before_raw)
    after = workspace.augment(after_raw)
    workspace.observe_transition(before, "right", 0.0, after, terminated=False)

    relabeled = workspace.augment(after_raw)

    assert any("repaired_mover" in obj.tags for obj in relabeled.objects if "agent" in obj.tags)


def test_representation_repair_workspace_learns_split_proposal_across_signatures() -> None:
    workspace = RepresentationRepairWorkspace()
    train_merged = _state(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 2, 2, 0],
                [0, 2, 2, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
    )
    train_split = _state(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 2, 0, 2],
                [0, 2, 0, 2],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
    )
    before = workspace.augment(train_merged)
    after = workspace.augment(train_split)
    workspace.observe_transition(before, "wait", 0.0, after, terminated=False)

    generalization = _state(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 5, 5, 0],
                [0, 5, 5, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
    )
    repaired = workspace.augment(generalization)

    assert len(repaired.objects) == 2


def test_representation_repair_workspace_learns_merge_proposal_across_signatures() -> None:
    workspace = RepresentationRepairWorkspace()
    train_separated = _state(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 2, 0, 2],
                [0, 2, 0, 2],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
    )
    train_merged = _state(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 2, 2, 0],
                [0, 2, 2, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
    )
    before = workspace.augment(train_separated)
    after = workspace.augment(train_merged)
    workspace.observe_transition(before, "wait", 0.0, after, terminated=False)

    generalization = _state(
        np.array(
            [
                [0, 0, 0, 0],
                [0, 5, 0, 5],
                [0, 5, 0, 5],
                [0, 0, 0, 0],
            ],
            dtype=np.int64,
        )
    )
    repaired = workspace.augment(generalization)

    assert len(repaired.objects) == 1
