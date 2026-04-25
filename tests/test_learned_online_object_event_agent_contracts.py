from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np

from arcagi.agents.learned_online_object_event_agent import LearnedOnlineObjectEventAgent
from arcagi.core.types import GridObservation, ObjectState, StructuredState, Transition
from arcagi.evaluation.harness import build_agent


def test_object_event_agent_scores_all_legal_actions_and_reports_contract_diagnostics() -> None:
    agent = build_agent("learned_online_object_event")
    observation = _observation(actions=tuple(f"click:{x}:{y}" for y in range(8) for x in range(8)) + ("0", "undo", "up"))

    action = agent.act(observation)
    diagnostics = agent.diagnostics()

    assert action in observation.available_actions
    assert agent.controller_kind == "learned_online_object_event_v1"
    assert diagnostics["controller_kind"] == "learned_online_object_event_v1"
    assert diagnostics["legal_action_count"] == len(observation.available_actions)
    assert diagnostics["scored_action_count"] == len(observation.available_actions)
    assert diagnostics["full_dense_surface_scored"] is True
    assert diagnostics["runtime_trace_cursor"] is False
    assert diagnostics["runtime_action_sequence_replay"] is False
    assert diagnostics["runtime_state_hash_to_action"] is False
    assert diagnostics["runtime_per_game_behavior"] is False


def test_object_event_online_update_count_and_level_belief_boundary_semantics() -> None:
    agent = LearnedOnlineObjectEventAgent(seed=3, device="cpu")
    before = _state(levels_completed=0)
    after = _state(levels_completed=1, remove_blue=True)
    agent.last_state = before
    agent.last_action = before.affordances[0]
    initial_session = float(agent.diagnostics()["session_belief_norm"])

    agent.on_transition(
        Transition(
            state=before,
            action=before.affordances[0],
            reward=1.0,
            next_state=after,
            terminated=False,
            info={"level_boundary": True, "levels_completed_before": 0, "levels_completed_after": 1, "score_delta": 1.0},
        )
    )
    diagnostics = agent.diagnostics()

    assert diagnostics["online_update_count"] == 1
    assert diagnostics["level_epoch"] == 1
    assert diagnostics["level_step"] == 0
    assert diagnostics["session_belief_norm"] > initial_session
    assert diagnostics["level_belief_norm"] == 0.0


def test_object_event_checkpoint_roundtrip_contains_anti_replay_metadata(tmp_path: Path) -> None:
    agent = LearnedOnlineObjectEventAgent(seed=7, device="cpu")
    path = tmp_path / "object_event.pkl"

    agent.save_checkpoint(path)
    restored = LearnedOnlineObjectEventAgent.from_checkpoint(path, device="cpu")
    checkpoint = pickle.loads(path.read_bytes())
    metadata = checkpoint["metadata"]

    assert restored.controller_kind == "learned_online_object_event_v1"
    assert metadata["controller_kind"] == "learned_online_object_event_v1"
    assert metadata["runtime_trace_cursor"] is False
    assert metadata["runtime_action_sequence_replay"] is False
    assert metadata["runtime_state_hash_to_action"] is False
    assert metadata["runtime_per_game_behavior"] is False
    assert metadata["runtime_graph_search_solver"] is False
    assert metadata["runtime_action_pattern_enumerator"] is False
    assert metadata["trace_bootstrap_runtime_replay"] is False
    assert metadata["stores_teacher_action_sequence"] is False
    assert metadata["stores_state_hash_to_action"] is False
    assert metadata["online_update_from_transition_error"] is True
    assert "session_belief" in metadata["online_update_params"]
    assert "level_belief" in metadata["online_update_params"]
    _assert_no_forbidden_payload_keys(checkpoint)


def test_object_event_agent_has_no_forbidden_controller_attributes() -> None:
    agent = build_agent("object_event")

    assert agent.claim_eligible_arc_controller is True
    assert agent.arc_competence_validated is False
    assert agent.uses_trace_replay is False
    assert agent.uses_state_hash_action_lookup is False
    assert agent.uses_per_game_runtime_behavior is False
    assert agent.scores_full_legal_action_surface is True
    assert not hasattr(agent, "runtime_rule_controller")
    assert not hasattr(agent, "theory_manager")
    assert not hasattr(agent, "spotlight")
    assert not hasattr(agent, "planner")
    assert not hasattr(agent, "rule_inducer")


def test_harness_alias_loads_object_event_checkpoint(tmp_path: Path) -> None:
    agent = LearnedOnlineObjectEventAgent(seed=11, device="cpu")
    path = tmp_path / "agent.pkl"
    agent.save_checkpoint(path)

    restored = build_agent("learned_online_object_event_v1", checkpoint_path=str(path), device="cpu")

    assert restored.controller_kind == "learned_online_object_event_v1"
    assert restored.diagnostics()["runtime_trace_cursor"] is False


def _assert_no_forbidden_payload_keys(value: object, *, path: str = "") -> None:
    forbidden = (
        "trace_cursor",
        "teacher_actions",
        "action_sequence",
        "replay_actions",
        "state_hash_to_action",
        "state_action_lookup",
        "per_game",
    )
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if any(fragment in key_text for fragment in forbidden):
                assert child is False, f"forbidden stored payload key {child_path}={child!r}"
            _assert_no_forbidden_payload_keys(child, path=child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_forbidden_payload_keys(child, path=f"{path}[{index}]")


def _observation(*, actions: tuple[str, ...]) -> GridObservation:
    grid = np.zeros((8, 8), dtype=np.int64)
    grid[1, 1] = 2
    grid[6, 6] = 5
    roles = {action: "click" for action in actions if action.startswith("click:")}
    roles.update({"0": "reset_level", "undo": "undo", "up": "move_up"})
    return GridObservation(
        task_id="object_event_contract",
        episode_id="0",
        step_index=0,
        grid=grid,
        available_actions=actions,
        extras={"action_roles": roles},
    )


def _state(*, levels_completed: int, remove_blue: bool = False) -> StructuredState:
    grid = np.zeros((4, 4), dtype=np.int64)
    red = _object("red", 2, (1, 1))
    blue = _object("blue", 5, (2, 2))
    objects = (red,) if remove_blue else (red, blue)
    for obj in objects:
        for row, col in obj.cells:
            grid[row, col] = obj.color
    actions = ("click:1:1", "click:2:2", "0")
    return StructuredState(
        task_id="object_event_contract",
        episode_id="0",
        step_index=levels_completed,
        grid_shape=grid.shape,
        grid_signature=tuple(int(value) for value in grid.reshape(-1)),
        objects=objects,
        relations=(),
        affordances=actions,
        action_roles=(("0", "reset_level"), ("click:1:1", "click"), ("click:2:2", "click")),
        inventory=(("interface_levels_completed", str(levels_completed)),),
        flags=(),
    )


def _object(object_id: str, color: int, cell: tuple[int, int]) -> ObjectState:
    row, col = cell
    return ObjectState(
        object_id=object_id,
        color=color,
        cells=(cell,),
        bbox=(row, col, row, col),
        centroid=(float(row), float(col)),
        area=1,
        tags=(),
    )
