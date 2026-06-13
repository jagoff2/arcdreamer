from __future__ import annotations

import torch

from src.action_selection import select_experimental_action
from src.persistent_memory import canonical_observation_hash


class _FakeMemory:
    def __init__(self, transition_graph: dict[str, object] | None = None) -> None:
        self.transition_graph = transition_graph or {}


def test_graph_plan_selection_prefers_shortest_progress_path() -> None:
    observation = {"sensory": torch.tensor([0.1], dtype=torch.float32)}
    intermediate = {"sensory": torch.tensor([0.2], dtype=torch.float32)}
    short_target = {"sensory": torch.tensor([0.3], dtype=torch.float32)}
    long_target = {"sensory": torch.tensor([0.4], dtype=torch.float32)}

    current_state = canonical_observation_hash(observation)
    intermediate_state = canonical_observation_hash(intermediate)
    short_target_state = canonical_observation_hash(short_target)
    long_target_state = canonical_observation_hash(long_target)

    graph = {
        "edges": {
            "state_to_intermediate": {
                "id": "state_to_intermediate",
                "from": current_state,
                "to": intermediate_state,
                "action": "0",
                "count": 1,
                "no_op": False,
                "loop_observed": False,
                "reversible": False,
                "score_delta_sum": 0.0,
                "terminal_count": 0,
            },
            "state_to_target": {
                "id": "state_to_target",
                "from": current_state,
                "to": short_target_state,
                "action": "1",
                "count": 1,
                "no_op": False,
                "loop_observed": False,
                "reversible": False,
                "score_delta_sum": 1.0,
                "terminal_count": 0,
            },
            "intermediate_to_target": {
                "id": "intermediate_to_target",
                "from": intermediate_state,
                "to": long_target_state,
                "action": "2",
                "count": 1,
                "no_op": False,
                "loop_observed": False,
                "reversible": False,
                "score_delta_sum": 1.0,
                "terminal_count": 0,
            },
        },
    }

    selected, diagnostics = select_experimental_action(
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        _FakeMemory(transition_graph=graph),
        observation,
        available_action_mask=[True, True, True],
    )

    selected_row = next(item for item in diagnostics["components"] if item["action"] == selected)
    indirect_row = next(item for item in diagnostics["components"] if item["action"] == 0)

    assert selected == 1
    assert selected_row["graph_plan_value"] > 0.0
    assert selected_row["graph_plan"]["target_state"] == short_target_state
    assert selected_row["graph_plan"]["path_length"] == 1
    assert selected_row["graph_plan"]["action_sequence"] == ["1"]
    assert indirect_row["graph_plan"]["path_length"] == 2
    assert indirect_row["graph_plan"]["action_sequence"] == ["0", "2"]


def test_graph_plan_prefers_shorter_path_when_target_value_comparable() -> None:
    observation = {"sensory": torch.tensor([1.0], dtype=torch.float32)}
    intermediate = {"sensory": torch.tensor([2.0], dtype=torch.float32)}
    short_target = {"sensory": torch.tensor([3.0], dtype=torch.float32)}
    long_target = {"sensory": torch.tensor([4.0], dtype=torch.float32)}

    current_state = canonical_observation_hash(observation)
    intermediate_state = canonical_observation_hash(intermediate)
    short_target_state = canonical_observation_hash(short_target)
    long_target_state = canonical_observation_hash(long_target)

    graph = {
        "edges": {
            "short_edge": {
                "id": "short_edge",
                "from": current_state,
                "to": short_target_state,
                "action": "0",
                "count": 1,
                "no_op": False,
                "loop_observed": False,
                "reversible": False,
                "score_delta_sum": 1.0,
                "terminal_count": 0,
            },
            "long_prefix": {
                "id": "long_prefix",
                "from": current_state,
                "to": intermediate_state,
                "action": "1",
                "count": 1,
                "no_op": False,
                "loop_observed": False,
                "reversible": False,
                "score_delta_sum": 0.0,
                "terminal_count": 0,
            },
            "long_tail": {
                "id": "long_tail",
                "from": intermediate_state,
                "to": long_target_state,
                "action": "2",
                "count": 1,
                "no_op": False,
                "loop_observed": False,
                "reversible": False,
                "score_delta_sum": 1.0,
                "terminal_count": 0,
            },
        },
    }

    _, diagnostics = select_experimental_action(
        torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32),
        _FakeMemory(transition_graph=graph),
        observation,
        available_action_mask=[True, True, True],
    )

    short_row = next(item for item in diagnostics["components"] if item["action"] == 0)
    long_row = next(item for item in diagnostics["components"] if item["action"] == 1)

    assert short_row["graph_plan"]["path_length"] == 1
    assert long_row["graph_plan"]["path_length"] == 2
    assert short_row["graph_plan_value"] > long_row["graph_plan_value"]
    assert diagnostics["selected_action"] == 0


def test_graph_plan_absent_without_positive_score_or_terminal_progress_edges() -> None:
    observation = {"sensory": torch.tensor([5.0], dtype=torch.float32)}
    other = {"sensory": torch.tensor([6.0], dtype=torch.float32)}

    current_state = canonical_observation_hash(observation)
    other_state = canonical_observation_hash(other)

    graph = {
        "edges": {
            "neutral_action": {
                "id": "neutral_action",
                "from": current_state,
                "to": other_state,
                "action": "0",
                "count": 1,
                "no_op": False,
                "loop_observed": False,
                "reversible": False,
                "score_delta_sum": 0.0,
                "terminal_count": 0,
            },
            "neutral_loop": {
                "id": "neutral_loop",
                "from": current_state,
                "to": current_state,
                "action": "1",
                "count": 1,
                "no_op": False,
                "loop_observed": True,
                "reversible": False,
                "score_delta_sum": 0.0,
                "terminal_count": 0,
            },
        },
    }

    _, diagnostics = select_experimental_action(
        torch.tensor([0.0, 0.0], dtype=torch.float32),
        _FakeMemory(transition_graph=graph),
        observation,
        available_action_mask=[True, True],
    )

    for row in diagnostics["components"]:
        assert "graph_plan" not in row
        assert row["graph_plan_value"] == 0.0
