from __future__ import annotations

from collections import deque

from arcagi.envs.synthetic import (
    AGENT,
    COLLECT_BLUE,
    COLLECT_RED,
    EMPTY,
    GOAL_ACTIVE,
    HiddenRuleEnv,
    SWITCH_YELLOW,
    SWITCH_RED,
    TARGET,
)


def _find(grid, value: int) -> tuple[int, int]:
    for y in range(grid.shape[0]):
        for x in range(grid.shape[1]):
            if int(grid[y, x]) == value:
                return (y, x)
    raise AssertionError(f"value {value} not found")


def _bfs_path(grid, start: tuple[int, int], goal: tuple[int, int]) -> list[str]:
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
            if (ny, nx) != goal and cell not in (EMPTY, TARGET, GOAL_ACTIVE):
                continue
            visited.add((ny, nx))
            queue.append(((ny, nx), path + [action]))
    raise AssertionError("no path found")


def _reachable_interaction(grid, start: tuple[int, int], target: tuple[int, int]) -> tuple[list[str], str]:
    candidates = [
        ((target[0] - 1, target[1]), "interact_down"),
        ((target[0] + 1, target[1]), "interact_up"),
        ((target[0], target[1] - 1), "interact_right"),
        ((target[0], target[1] + 1), "interact_left"),
    ]
    for position, action in candidates:
        if int(grid[position]) != EMPTY:
            continue
        try:
            return _bfs_path(grid, start, position), action
        except AssertionError:
            continue
    raise AssertionError("no reachable interaction side")


def test_switch_unlock_goal_can_be_reached_after_correct_interaction() -> None:
    env = HiddenRuleEnv(family_mode="switch_unlock", family_variant="red", seed=13)
    observation = env.reset(seed=13)
    grid = observation.grid
    agent = _find(grid, AGENT)
    switch = _find(grid, SWITCH_RED)
    target = _find(grid, TARGET)

    candidates = [
        ((switch[0] - 1, switch[1]), "interact_down"),
        ((switch[0] + 1, switch[1]), "interact_up"),
        ((switch[0], switch[1] - 1), "interact_right"),
        ((switch[0], switch[1] + 1), "interact_left"),
    ]
    adjacency, interact_action = next(
        (position, action)
        for position, action in candidates
        if int(grid[position]) == EMPTY
    )
    for action in _bfs_path(grid, agent, adjacency):
        result = env.step(action)
        assert not result.terminated
    result = env.step(interact_action)
    assert result.info["event"] == "correct_switch"

    grid = result.observation.grid
    agent = _find(grid, AGENT)
    goal_path = _bfs_path(grid, agent, target)
    final = None
    for action in goal_path:
        final = env.step(action)
    assert final is not None
    assert final.terminated
    assert final.reward > 0.9


def test_switch_unlock_redundant_interaction_after_goal_activation_is_penalized() -> None:
    env = HiddenRuleEnv(family_mode="switch_unlock", family_variant="red", seed=23)
    observation = env.reset(seed=23)
    grid = observation.grid
    agent = _find(grid, AGENT)
    switch = _find(grid, SWITCH_RED)

    path, interact_action = _reachable_interaction(grid, agent, switch)
    for action in path:
        result = env.step(action)
        assert not result.terminated
    unlocked = env.step(interact_action)
    assert unlocked.info["event"] == "correct_switch"

    repeated = env.step(interact_action)
    assert repeated.info["event"] == "redundant_post_goal_interaction"
    assert repeated.reward < 0.0


def test_selector_unlock_requires_click_before_correct_switch() -> None:
    env = HiddenRuleEnv(family_mode="selector_unlock", family_variant="red", seed=17)
    observation = env.reset(seed=17)
    grid = observation.grid
    agent = _find(grid, AGENT)
    switch = env._switch_positions[SWITCH_RED]
    target = _find(grid, TARGET)

    candidates = [
        ((switch[0] - 1, switch[1]), "interact_down"),
        ((switch[0] + 1, switch[1]), "interact_up"),
        ((switch[0], switch[1] - 1), "interact_right"),
        ((switch[0], switch[1] + 1), "interact_left"),
    ]
    path, interact_action = _reachable_interaction(grid, agent, switch)
    for action in path:
        result = env.step(action)
        assert not result.terminated
    blocked = env.step(interact_action)
    assert blocked.info["event"] != "selector_unlock_complete"
    assert not (blocked.observation.grid == GOAL_ACTIVE).any()

    click_action = next(action for action, color in env._selector_actions.items() if color == SWITCH_RED)
    clicked = env.step(click_action)
    assert clicked.info["event"] == "selector_candidate"

    unlocked = env.step(interact_action)
    assert unlocked.info["event"] == "selector_unlock_complete"

    grid = unlocked.observation.grid
    agent = _find(grid, AGENT)
    goal_path = _bfs_path(grid, agent, target)
    final = None
    for action in goal_path:
        final = env.step(action)
    assert final is not None
    assert final.reward > 0.9


def test_explicit_reset_seed_updates_episode_identity() -> None:
    env = HiddenRuleEnv(family_mode="switch_unlock", family_variant="red", seed=13)

    first = env.reset(seed=41)
    second = env.reset(seed=42)

    assert first.episode_id == "synthetic_hidden_rule/41_0"
    assert second.episode_id == "synthetic_hidden_rule/42_1"


def test_delayed_order_unlock_uses_delayed_completion_and_decoy_reset() -> None:
    env = HiddenRuleEnv(family_mode="delayed_order_unlock", family_variant="red_then_blue", seed=19)
    observation = env.reset(seed=19)
    grid = observation.grid
    agent = _find(grid, AGENT)
    first = env._collect_positions[COLLECT_RED]
    second = env._collect_positions[COLLECT_BLUE]
    decoy = env._collect_positions[SWITCH_YELLOW]

    def approach(target: tuple[int, int], agent_pos: tuple[int, int]):
        path, interact_action = _reachable_interaction(env._grid, agent_pos, target)
        for move in path:
            env.step(move)
        return env._agent, interact_action

    agent, interact = approach(first, agent)
    first_result = env.step(interact)
    assert first_result.info["event"] == "delayed_correct_collect"
    assert first_result.reward == 0.0

    agent, interact = approach(decoy, agent)
    decoy_result = env.step(interact)
    assert decoy_result.info["event"] == "decoy_reward_reset"
    assert decoy_result.reward <= 0.0

    agent, interact = approach(first, agent)
    env.step(interact)
    agent, interact = approach(second, agent)
    second_result = env.step(interact)
    assert second_result.info["event"] == "delayed_sequence_complete"
    assert second_result.reward == 0.0


def test_delayed_order_unlock_decoy_loop_is_not_reward_hackable() -> None:
    env = HiddenRuleEnv(family_mode="delayed_order_unlock", family_variant="red_then_blue", seed=23)
    observation = env.reset(seed=23)
    agent = _find(observation.grid, AGENT)
    decoy = env._collect_positions[SWITCH_YELLOW]

    def approach(target: tuple[int, int], agent_pos: tuple[int, int]):
        path, interact_action = _reachable_interaction(env._grid, agent_pos, target)
        for move in path:
            env.step(move)
        return env._agent, interact_action

    agent, interact = approach(decoy, agent)
    first = env.step(interact)
    agent, interact = approach(decoy, agent)
    second = env.step(interact)

    assert first.info["event"] == "decoy_reward_reset"
    assert second.info["event"] == "decoy_reward_reset"
    assert first.reward <= 0.0
    assert second.reward < 0.0


def test_selector_sequence_observation_surfaces_public_control_and_progress_state() -> None:
    env = HiddenRuleEnv(family_mode="selector_sequence_unlock", family_variant="red__red_then_blue", seed=29)
    observation = env.reset(seed=29)
    selector_click = next(action for action, color in env._selector_actions.items() if color == SWITCH_RED)

    clicked = env.step(selector_click)

    assert clicked.observation.extras["inventory"]["interface_selected_color"] == "red"
    assert clicked.observation.extras["flags"]["interface_selection_active"] == "1"
    selector_position = env._selector_positions[SWITCH_RED]
    assert "active" in clicked.observation.extras["cell_tags"][selector_position]
    assert "selected" in clicked.observation.extras["cell_tags"][selector_position]

    agent = env._agent
    first_target = env._collect_positions[COLLECT_RED]
    path, interact_action = _reachable_interaction(env._grid, agent, first_target)
    for action in path:
        env.step(action)
    progressed = env.step(interact_action)

    assert progressed.info["event"] == "selector_sequence_progress"
    assert progressed.observation.extras["inventory"]["interface_sequence_progress"] == "1"
    assert progressed.observation.extras["inventory"]["interface_sequence_total"] == "2"
    assert progressed.observation.extras["flags"]["interface_sequence_started"] == "1"
