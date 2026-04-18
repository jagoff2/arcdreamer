from __future__ import annotations

from collections import deque

import numpy as np

from arcagi.agents.graph_agent import GraphExplorerAgent
from arcagi.core.inferred_state import InferredStateTracker
from arcagi.core.spatial_workspace import SpatialBeliefWorkspace
from arcagi.core.types import StructuredState
from arcagi.envs.synthetic import COLLECT_RED, HiddenRuleEnv
from arcagi.perception.object_encoder import extract_structured_state


def _symbolic_only_state(
    *,
    inventory: tuple[tuple[str, str], ...] = (),
    flags: tuple[tuple[str, str], ...] = (),
) -> StructuredState:
    return StructuredState(
        task_id="state-features/test",
        episode_id="episode-0",
        step_index=0,
        grid_shape=(5, 5),
        grid_signature=tuple(np.zeros((5, 5), dtype=np.int64).reshape(-1)),
        objects=(),
        relations=(),
        affordances=("wait",),
        inventory=inventory,
        flags=flags,
    )


def _find(grid: np.ndarray, value: int) -> tuple[int, int]:
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if int(grid[y, x]) == value:
                return (y, x)
    raise AssertionError(f"value {value} not found")


def _bfs_path(grid: np.ndarray, start: tuple[int, int], goal: tuple[int, int]) -> list[str]:
    queue = deque([(start, [])])
    visited = {start}
    deltas = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}
    while queue:
        (y, x), path = queue.popleft()
        if (y, x) == goal:
            return path
        for action, (dy, dx) in deltas.items():
            ny, nx = y + dy, x + dx
            if (ny, nx) in visited:
                continue
            cell = int(grid[ny, nx])
            if (ny, nx) != goal and cell != 0:
                continue
            visited.add((ny, nx))
            queue.append(((ny, nx), path + [action]))
    raise AssertionError("no path found")


def _reachable_interaction(grid: np.ndarray, start: tuple[int, int], target: tuple[int, int]) -> tuple[list[str], str]:
    candidates = [
        ((target[0] - 1, target[1]), "interact_down"),
        ((target[0] + 1, target[1]), "interact_up"),
        ((target[0], target[1] - 1), "interact_right"),
        ((target[0], target[1] + 1), "interact_left"),
    ]
    for position, action in candidates:
        if int(grid[position]) != 0:
            continue
        try:
            return _bfs_path(grid, start, position), action
        except AssertionError:
            continue
    raise AssertionError("no reachable interaction side")


def _approach_and_interact(env: HiddenRuleEnv, target: tuple[int, int]):
    path, interact_action = _reachable_interaction(env._grid, env._agent, target)
    for action in path:
        env.step(action)
    return interact_action, env.step(interact_action)


def _follow_actions_with_workspace(
    env: HiddenRuleEnv,
    workspace: SpatialBeliefWorkspace,
    observation,
    actions: list[str],
):
    current_raw = extract_structured_state(observation)
    workspace.observe_state(current_raw)
    current_observation = observation
    current_result = None
    for action in actions:
        current_result = env.step(action)
        next_raw = extract_structured_state(current_result.observation)
        workspace.observe_transition(
            before=current_raw,
            action=action,
            reward=current_result.reward,
            after=next_raw,
            terminated=current_result.terminated or current_result.truncated,
        )
        workspace.observe_state(next_raw)
        current_raw = next_raw
        current_observation = current_result.observation
    return current_observation, current_raw, current_result


def test_summary_and_transition_vectors_keep_nonbinary_symbolic_state() -> None:
    before = _symbolic_only_state(inventory=(("sequence", "r"),))
    after = _symbolic_only_state(inventory=(("sequence", "b"),))

    summary_before = before.summary_vector()
    summary_after = after.summary_vector()
    transition_before = before.transition_vector()
    transition_after = after.transition_vector()

    assert summary_before.shape == (25,)
    assert transition_before.shape == (25,)
    assert not np.allclose(summary_before, summary_after)
    assert not np.allclose(transition_before, transition_after)


def test_synthetic_observation_does_not_leak_hidden_rule_or_progress_state() -> None:
    selector_env = HiddenRuleEnv(family_mode="selector_unlock", family_variant="red", seed=17)
    selector_observation = selector_env.reset(seed=17)
    assert "rule_tokens" not in selector_observation.extras
    assert "question_tokens" not in selector_observation.extras
    assert "inventory" not in selector_observation.extras
    assert "flags" not in selector_observation.extras

    delayed_env = HiddenRuleEnv(family_mode="delayed_order_unlock", family_variant="red_then_blue", seed=19)
    delayed_observation = delayed_env.reset(seed=19)
    assert "inventory" not in delayed_observation.extras
    assert "flags" not in delayed_observation.extras


def test_inferred_state_tracker_recovers_progress_from_visible_transition() -> None:
    env = HiddenRuleEnv(family_mode="order_collect", family_variant="red_then_blue", seed=19)
    observation = env.reset(seed=19)
    tracker = InferredStateTracker()

    current_raw = extract_structured_state(observation)
    augmented_before = tracker.augment(current_raw)
    assert augmented_before.inventory_dict()["belief_progress_level"] == "0"

    path, interact_action = _reachable_interaction(env._grid, env._agent, env._collect_positions[COLLECT_RED])
    for action in path + [interact_action]:
        result = env.step(action)
        next_raw = extract_structured_state(result.observation)
        tracker.observe_transition(
            before=current_raw,
            action=action,
            reward=result.reward,
            after=next_raw,
            terminated=result.terminated or result.truncated,
        )
        current_raw = next_raw
    augmented_after = tracker.augment(current_raw)

    assert int(augmented_after.inventory_dict()["belief_progress_level"]) >= 1
    assert augmented_after.flags_dict()["belief_recent_progress"] == "1"
    assert augmented_after.flags_dict()["belief_last_effect_family"] == "interact"


def test_spatial_workspace_tracks_tested_sites_and_persistent_anchor() -> None:
    env = HiddenRuleEnv(family_mode="order_collect", family_variant="red_then_blue", seed=23)
    observation = env.reset(seed=23)
    workspace = SpatialBeliefWorkspace()
    raw_before = extract_structured_state(observation)
    workspace.observe_state(raw_before)
    start_augmented = workspace.augment(raw_before)

    assert int(start_augmented.inventory_dict()["belief_visited_cells"]) >= 1
    assert start_augmented.flags_dict()["belief_has_spatial_anchor"] == "0"

    path, interact_action = _reachable_interaction(env._grid, env._agent, env._collect_positions[COLLECT_RED])
    observation, raw_after, result = _follow_actions_with_workspace(
        env,
        workspace,
        observation,
        path + [interact_action],
    )
    assert result is not None
    after_augmented = workspace.augment(raw_after)

    assert int(after_augmented.inventory_dict()["belief_tested_sites"]) >= 1
    assert int(after_augmented.inventory_dict()["belief_effect_sites"]) >= 1
    assert after_augmented.flags_dict()["belief_has_spatial_anchor"] == "1"
    assert after_augmented.inventory_dict()["belief_nearest_anchor_distance"] == "near"

    anchor_cell = env._collect_positions[COLLECT_RED]
    empty_cells = [
        (y, x)
        for y in range(env._grid.shape[0])
        for x in range(env._grid.shape[1])
        if int(env._grid[y, x]) == 0
    ]
    far_goal = max(
        empty_cells,
        key=lambda cell: abs(cell[0] - anchor_cell[0]) + abs(cell[1] - anchor_cell[1]),
    )
    move_path = _bfs_path(env._grid, env._agent, far_goal)
    observation, raw_far, result = _follow_actions_with_workspace(env, workspace, observation, move_path)
    far_augmented = workspace.augment(raw_far)

    assert far_augmented.flags_dict()["belief_has_spatial_anchor"] == "1"
    assert far_augmented.inventory_dict()["belief_nearest_anchor_distance"] in {"mid", "far"}


def test_spatial_workspace_does_not_emit_probe_target_heuristics() -> None:
    env = HiddenRuleEnv(family_mode="switch_unlock", family_variant="red", seed=31)
    observation = env.reset(seed=31)
    workspace = SpatialBeliefWorkspace()
    raw_state = extract_structured_state(observation)
    workspace.observe_state(raw_state)
    augmented = workspace.augment(raw_state)

    assert all(not key.startswith("belief_probe_") for key, _ in augmented.inventory)
    assert all(not key.startswith("belief_probe_") for key, _ in augmented.flags)


def test_agent_path_sanitizes_hidden_step_info() -> None:
    env = HiddenRuleEnv(family_mode="switch_unlock", family_variant="red", seed=29)
    observation = env.reset(seed=29)
    agent = GraphExplorerAgent()

    action = agent.act(observation)
    result = env.step(action)

    assert "rule_tokens" in result.info
    assert "question_tokens" in result.info
    assert "family_id" in result.info

    agent.update_after_step(
        result.observation,
        reward=result.reward,
        terminated=result.terminated or result.truncated,
        info=result.info,
    )

    assert agent.last_info == {}
