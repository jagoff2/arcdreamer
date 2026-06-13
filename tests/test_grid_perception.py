from __future__ import annotations

import json

import numpy as np
import pytest

from src.grid_perception import parse_grid, raw_grid_hash


def _component_objects(scene):
    return [obj for obj in scene.objects if obj.kind == "component"]


def _component_by_color(scene, color):
    return [obj for obj in _component_objects(scene) if obj.color == color]


def test_parse_grid_returns_expected_fields_and_deterministic_hash() -> None:
    grid = np.array(
        [
            [0, 0, 0, 0],
            [0, 3, 3, 0],
            [0, 0, 0, 0],
            [0, 2, 2, 2],
        ],
        dtype=np.int64,
    )

    scene_a = parse_grid(grid)
    scene_b = parse_grid(grid)

    assert scene_a.raw_hash == scene_b.raw_hash == raw_grid_hash(grid)
    assert scene_a.shape == (4, 4)
    assert scene_a.background == 0
    assert isinstance(scene_a.grid_embedding, tuple)
    assert scene_a.grid_embedding
    assert scene_a.raw_hash == scene_a.compact()["raw_hash"]
    assert scene_a.compact() == scene_b.compact()
    assert len(scene_a.objects) >= 1
    assert set(scene_a.compact().keys()) == {
        "raw_hash",
        "shape",
        "background",
        "grid_embedding",
        "objects",
        "relation_graph",
        "bindings",
        "salience_map",
        "active_cells",
        "edit_script",
    }


def test_object_slots_are_stable_and_descriptive() -> None:
    grid = np.zeros((7, 7), dtype=np.int64)
    for i in range(1, 6):
        grid[1, i] = 1
        grid[5, i] = 1
        grid[i, 1] = 1
        grid[i, 5] = 1
    grid[2, 2] = 2

    scene = parse_grid(grid)
    components = [obj for obj in scene.objects if obj.kind == "component"]
    component1 = next(obj for obj in components if obj.color == 1)

    assert component1.color == 1
    assert component1.bbox == (1, 1, 5, 5)
    assert component1.area == 16
    assert component1.holes == 1
    assert component1.changed_count == 0
    assert component1.area == len(component1.cells)
    assert component1.centroid == (3.0, 3.0)
    assert component1.canonical_hash != component1.object_id
    assert len(component1.object_id) == 12
    assert len(component1.canonical_hash) == 12
    assert component1.boundary_cells
    assert component1.canonical_cells
    assert component1.symmetry["vertical"] is True
    assert component1.symmetry["horizontal"] is True

    component1_norm = _component_objects(scene)
    ids = [obj.object_id for obj in component1_norm]
    assert len(ids) == len(set(ids))


def test_relation_graph_covers_color_adjacency_repetition_alignment_contains_and_symmetry() -> None:
    grid = np.zeros((7, 7), dtype=np.int64)
    grid[1, 1:3] = 2
    grid[1, 4:6] = 2
    grid[2, 1] = 3

    for i in range(3, 6):
        grid[3, i] = 4
        grid[5, i] = 4
        grid[i, 3] = 4
        grid[i, 5] = 4
    grid[4, 4] = 5

    scene = parse_grid(grid, action="probe", time_index=7)
    comps = _component_objects(scene)
    color_two = sorted([obj for obj in comps if obj.color == 2], key=lambda obj: obj.bbox[1])
    left, right = color_two
    adjacency_target = _component_by_color(scene, 3)[0]
    outer_ring = _component_by_color(scene, 4)[0]
    inner = _component_by_color(scene, 5)[0]

    relation_map = {(r.source, r.relation, r.target, json.dumps(r.detail, sort_keys=True)) for r in scene.relation_graph}

    assert (left.object_id, "same_color", right.object_id, json.dumps({"color": 2}, sort_keys=True)) in relation_map
    assert (left.object_id, "adjacent", adjacency_target.object_id, "{}") in relation_map
    assert (outer_ring.object_id, "contains", inner.object_id, "{}") in relation_map
    assert (left.object_id, "row_aligned", right.object_id, "{}") in relation_map
    assert (left.object_id, "col_aligned", adjacency_target.object_id, "{}") in relation_map
    assert (left.object_id, "repetition", right.object_id, json.dumps({"delta": [0.0, 3.0]}, sort_keys=True)) in relation_map
    assert (outer_ring.object_id, "has_hole", outer_ring.object_id, json.dumps({"count": 1}, sort_keys=True)) in relation_map
    for axis in ("vertical", "horizontal", "diagonal"):
        assert (
            outer_ring.object_id,
            "symmetry",
            outer_ring.object_id,
            json.dumps({"axis": axis}, sort_keys=True),
        ) in relation_map


def test_bindings_are_deterministic_and_structured() -> None:
    grid = np.array(
        [
            [0, 0, 0, 0],
            [0, 2, 2, 0],
            [0, 0, 0, 0],
            [0, 3, 3, 3],
        ],
        dtype=np.int64,
    )

    scene = parse_grid(grid, action="left", time_index=12)
    repeat = parse_grid(grid, action="left", time_index=12)
    later = parse_grid(grid, action="left", time_index=13)

    assert scene.bindings == repeat.bindings
    assert list(scene.bindings) == sorted(scene.bindings, key=lambda item: item.key)

    key_to_vector = {binding.key: binding.vector for binding in scene.bindings}
    assert all(len(vector) == 16 for vector in key_to_vector.values())
    assert all(set(vector).issubset({-1, 1}) for vector in key_to_vector.values())
    assert repeat.bindings != later.bindings
    repeat_vectors = {binding.key: binding.vector for binding in repeat.bindings}
    later_vectors = {binding.key: binding.vector for binding in later.bindings}
    assert repeat_vectors["time:12"] != later_vectors["time:13"]
    assert "action:left" in key_to_vector

    for obj in _component_objects(scene):
        assert f"{obj.object_id}:color:{obj.color}" in key_to_vector
        assert f"{obj.object_id}:bbox:{obj.bbox}" in key_to_vector
        assert f"{obj.object_id}:area:{obj.area}" in key_to_vector
        assert f"{obj.object_id}:centroid:{round(obj.centroid[0], 3)}:{round(obj.centroid[1], 3)}" in key_to_vector

    for relation in scene.relation_graph:
        key = f"relation:{relation.source}:{relation.relation}:{relation.target}:{json.dumps(relation.detail, sort_keys=True)}"
        assert key in key_to_vector


def test_edit_script_tracks_edits_disappearances_appearances_moves_and_active_cells() -> None:
    previous = np.zeros((5, 5), dtype=np.int64)
    previous[1, 1:4] = 7

    current = np.zeros((5, 5), dtype=np.int64)
    current[2, 1:4] = 7

    before = parse_grid(previous)
    after = parse_grid(current, previous_grid=previous, action="slide", time_index=2)

    edit_script = after.edit_script
    assert edit_script.changed_count == 6
    assert edit_script.unchanged_count == current.size - 6
    assert len(edit_script.cell_edits) == 6
    assert len(edit_script.appears) == 3
    assert len(edit_script.disappears) == 3
    assert len(edit_script.moves) == 1

    move = edit_script.moves[0]
    before_component = next(obj for obj in _component_objects(before) if obj.color == 7)
    assert move["object_id"] == before_component.object_id
    assert move["color"] == 7
    assert move["delta"] == [1.0, 0.0]
    assert move["from_centroid"] == [1.0, 2.0]
    assert move["to_centroid"] == [2.0, 2.0]

    changed_cells = set(tuple(edit["cell"]) for edit in edit_script.cell_edits)
    expected_active = set(changed_cells)
    for obj in _component_objects(after):
        expected_active.add((int(round(obj.centroid[0])), int(round(obj.centroid[1]))))
        expected_active.update(obj.boundary_cells)

    assert expected_active <= set(after.active_cells)


def test_parser_supports_64x64_and_rejects_bad_shape() -> None:
    grid = np.zeros((64, 64), dtype=np.int64)
    grid[0, 0] = 2

    valid = parse_grid(grid)
    assert valid.shape == (64, 64)

    with pytest.raises(ValueError, match="2D"):
        parse_grid([1, 2, 3, 4])

    with pytest.raises(ValueError, match="up to 64x64"):
        parse_grid(np.zeros((65, 8), dtype=np.int64))
