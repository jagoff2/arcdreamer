from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch

import arcagi.training.arc_public as arc_public
import arcagi.training.goal_scientist_public as goal_scientist_public
from arcagi.agents.learned_agent import HybridAgent, LanguageNoMemoryAgent
from arcagi.core.types import GridObservation, StepResult
from arcagi.perception.object_encoder import extract_structured_state
from arcagi.scientist.train_arc import _result_key
from arcagi.training.arc_public import _behavior_agent, _belief_tokens, _plan_tokens, _question_tokens
from arcagi.training.synthetic import build_default_modules


def _state(actions: tuple[str, ...]) -> object:
    observation = GridObservation(
        task_id="arc/public-test",
        episode_id="episode-0",
        step_index=0,
        grid=np.array(
            [
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 3],
            ],
            dtype=np.int64,
        ),
        available_actions=actions,
        extras={
            "action_roles": {
                "1": "move_up",
                "5": "select_cycle",
                "click:4:4": "click",
                "interact_right": "interact",
            }
        },
    )
    return extract_structured_state(observation)


def _session_observation(
    *,
    step_index: int,
    actions: tuple[str, ...],
    game_state: str,
    levels_completed: int,
) -> GridObservation:
    action_roles = {
        "0": "reset",
        "1": "move_up",
        "5": "select_cycle",
        "interact_right": "interact",
    }
    for action in actions:
        if action.startswith("click:"):
            action_roles[action] = "click"
    return GridObservation(
        task_id="arc/public-test",
        episode_id="session-0",
        step_index=step_index,
        grid=np.array(
            [
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 3],
            ],
            dtype=np.int64,
        ),
        available_actions=actions,
        extras={
            "game_state": game_state,
            "levels_completed": levels_completed,
            "action_roles": action_roles,
        },
    )


def _make_arc_public_config(**overrides: Any):
    values: dict[str, Any] = {
        "mode": "offline",
        "game_limit": 1,
        "max_steps": 4,
        "epochs": 1,
        "behavior_policies": ("graph",),
    }
    fields = arc_public.ArcPublicTrainingConfig.__dataclass_fields__
    if "episodes_per_game" in fields:
        values["episodes_per_game"] = 1
    if "sessions_per_game" in fields:
        values["sessions_per_game"] = 1
    values.update(overrides)
    return arc_public.ArcPublicTrainingConfig(**{key: value for key, value in values.items() if key in fields})


def _make_goal_scientist_config(**overrides: Any):
    values: dict[str, Any] = {
        "mode": "offline",
        "game_limit": 1,
        "max_steps": 4,
        "epochs": 1,
        "behavior_policies": ("graph",),
        "save_epoch_snapshots": False,
    }
    fields = goal_scientist_public.GoalScientistPublicTrainingConfig.__dataclass_fields__
    if "episodes_per_game" in fields:
        values["episodes_per_game"] = 1
    if "sessions_per_game" in fields:
        values["sessions_per_game"] = 1
    values.update(overrides)
    return goal_scientist_public.GoalScientistPublicTrainingConfig(**{key: value for key, value in values.items() if key in fields})


def _patch_public_arc_toolkit(monkeypatch: pytest.MonkeyPatch, module: Any, env_cls: type[Any]) -> None:
    monkeypatch.setattr(module, "arc_toolkit_available", lambda: True)
    monkeypatch.setattr(module, "arc_operation_mode", lambda mode: mode)
    monkeypatch.setattr(module, "list_arc_games", lambda operation_mode=None: ["game-0"])
    monkeypatch.setattr(module, "ArcToolkitEnv", env_cls)
    if hasattr(module, "Arcade"):
        monkeypatch.setattr(module, "Arcade", None)


class _ReactiveSessionAgent:
    def __init__(self) -> None:
        self.reset_all_calls = 0
        self.reset_episode_calls = 0
        self.seen_actions: list[str] = []

    def reset_all(self) -> None:
        self.reset_all_calls += 1

    def reset_episode(self) -> None:
        self.reset_episode_calls += 1

    def act(self, observation: GridObservation) -> str:
        game_state = str(observation.extras.get("game_state", "")).upper()
        if game_state.endswith("GAME_OVER") and "0" in observation.available_actions:
            action = "0"
        else:
            action = "1"
        self.seen_actions.append(action)
        return action

    def update_after_step(
        self,
        *,
        next_observation: GridObservation,
        reward: float = 0.0,
        terminated: bool = False,
        info: dict[str, Any] | None = None,
    ) -> None:
        del next_observation, reward, terminated, info


class _TerminalThenResetSessionEnv:
    def __init__(self, game_id: str, operation_mode: str | None = None, arcade: Any | None = None) -> None:
        del game_id, operation_mode, arcade
        self.phase = 0

    def reset(self, seed: int | None = None) -> GridObservation:
        del seed
        self.phase = 0
        return _session_observation(
            step_index=0,
            actions=("1", "0"),
            game_state="GameState.NOT_FINISHED",
            levels_completed=0,
        )

    def step(self, action: str) -> StepResult:
        if self.phase == 0:
            assert action == "1"
            self.phase = 1
            return StepResult(
                observation=_session_observation(
                    step_index=1,
                    actions=("1", "0"),
                    game_state="GameState.GAME_OVER",
                    levels_completed=0,
                ),
                reward=0.0,
                terminated=True,
                truncated=False,
                info={},
            )
        if self.phase == 1:
            assert action == "0"
            self.phase = 2
            return StepResult(
                observation=_session_observation(
                    step_index=2,
                    actions=("1", "0"),
                    game_state="GameState.NOT_FINISHED",
                    levels_completed=0,
                ),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={},
            )
        if self.phase == 2:
            assert action == "1"
            self.phase = 3
            return StepResult(
                observation=_session_observation(
                    step_index=3,
                    actions=("0",),
                    game_state="GameState.WIN",
                    levels_completed=1,
                ),
                reward=1.0,
                terminated=True,
                truncated=False,
                info={},
            )
        raise AssertionError((self.phase, action))

    def close(self) -> None:
        return None


class _LevelBoundarySessionEnv:
    def __init__(self, game_id: str, operation_mode: str | None = None, arcade: Any | None = None) -> None:
        del game_id, operation_mode, arcade
        self.phase = 0

    def reset(self, seed: int | None = None) -> GridObservation:
        del seed
        self.phase = 0
        return _session_observation(
            step_index=0,
            actions=("1", "0"),
            game_state="GameState.NOT_FINISHED",
            levels_completed=0,
        )

    def step(self, action: str) -> StepResult:
        assert action == "1"
        if self.phase == 0:
            self.phase = 1
            return StepResult(
                observation=_session_observation(
                    step_index=1,
                    actions=("1", "0"),
                    game_state="GameState.NOT_FINISHED",
                    levels_completed=1,
                ),
                reward=1.0,
                terminated=False,
                truncated=False,
                info={},
            )
        if self.phase == 1:
            self.phase = 2
            return StepResult(
                observation=_session_observation(
                    step_index=2,
                    actions=("1", "0"),
                    game_state="GameState.NOT_FINISHED",
                    levels_completed=1,
                ),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={},
            )
        if self.phase == 2:
            self.phase = 3
            return StepResult(
                observation=_session_observation(
                    step_index=3,
                    actions=("0",),
                    game_state="GameState.WIN",
                    levels_completed=2,
                ),
                reward=1.0,
                terminated=True,
                truncated=False,
                info={},
            )
        raise AssertionError((self.phase, action))

    def close(self) -> None:
        return None


def test_arc_public_language_targets_probe_rules_when_selector_actions_exist() -> None:
    state = _state(("1", "5", "click:4:4"))

    belief = _belief_tokens(state, "5", reward=0.0, usefulness=0.1)
    question = _question_tokens(state, "5", reward=0.0, usefulness=0.1)

    assert belief == ("belief", "interface", "control_binding", "uncertain", "clickable")
    assert question == ("question", "need", "test", "focus", "control_binding", "state", "probe")


def test_arc_public_language_targets_confirm_goal_on_positive_move_reward() -> None:
    state = _state(("1",))

    belief = _belief_tokens(state, "1", reward=1.0, usefulness=0.5)
    question = _question_tokens(state, "1", reward=1.0, usefulness=0.5)

    assert belief == ("belief", "goal", "active", "focus", "target", "state", "explore")
    assert question == ("question", "need", "confirm", "focus", "target", "state", "explore")


def test_arc_public_plan_tokens_encode_action_focus_and_direction() -> None:
    state = _state(("1", "interact_right"))

    plan = _plan_tokens(state, "interact_right", reward=0.0, usefulness=0.2)

    assert plan == ("plan", "move_then_interact", "direction", "right", "focus", "interactable", "state", "inactive")


def test_extract_structured_state_infers_interface_flags_and_clickable_tags() -> None:
    observation = GridObservation(
        task_id="arc/public-test",
        episode_id="episode-0",
        step_index=0,
        grid=np.array(
            [
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 3],
            ],
            dtype=np.int64,
        ),
        available_actions=("1", "click:1:1", "click:2:2"),
        extras={
            "action_roles": {
                "1": "move_up",
                "click:1:1": "click",
                "click:2:2": "click",
            }
        },
    )

    state = extract_structured_state(observation)

    assert state.flags_dict()["interface_has_click"] == "1"
    assert state.inventory_dict()["interface_click_actions"] == "2"
    assert any("clickable" in obj.tags for obj in state.objects)


def test_arc_public_behavior_agent_distinguishes_learned_and_hybrid() -> None:
    device = torch.device("cpu")
    encoder, world_model, language_model, _ = build_default_modules(device=device)

    learned_agent, learned_reset = _behavior_agent(
        "learned",
        encoder=encoder,
        world_model=world_model,
        language_model=language_model,
        device=device,
    )
    hybrid_agent, hybrid_reset = _behavior_agent(
        "hybrid",
        encoder=encoder,
        world_model=world_model,
        language_model=language_model,
        device=device,
    )

    assert isinstance(learned_agent, LanguageNoMemoryAgent)
    assert isinstance(hybrid_agent, HybridAgent)
    assert learned_reset is True
    assert hybrid_reset is True


def test_arc_public_collector_continues_across_terminal_when_reset_is_available(monkeypatch: pytest.MonkeyPatch) -> None:
    agent = _ReactiveSessionAgent()
    _patch_public_arc_toolkit(monkeypatch, arc_public, _TerminalThenResetSessionEnv)
    monkeypatch.setattr(arc_public, "_behavior_agent", lambda *args, **kwargs: (agent, True))

    dataset = arc_public.collect_arc_public_dataset(_make_arc_public_config())

    assert len(dataset) == 3
    assert [sample["action"] for sample in dataset] == ["1", "0", "1"]
    assert agent.reset_all_calls == 1


def test_goal_scientist_session_collector_continues_across_terminal_when_reset_is_available(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _ReactiveSessionAgent()
    monkeypatch.setattr(goal_scientist_public, "_goal_behavior_agent", lambda *args, **kwargs: (agent, True))

    dataset, summary = goal_scientist_public._collect_goal_scientist_session(
        _TerminalThenResetSessionEnv("game-0"),
        game_id="game-0",
        session_index=0,
        seed=17,
        policy_name="graph",
        config=_make_goal_scientist_config(),
    )

    assert len(dataset) == 3
    assert [sample["action"] for sample in dataset] == ["1", "0", "1"]
    assert agent.reset_all_calls == 1
    assert summary["reset_steps"] == 1
    assert summary["won"] is True


def test_goal_scientist_session_collector_exposes_level_boundary_session_annotations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    agent = _ReactiveSessionAgent()
    monkeypatch.setattr(goal_scientist_public, "_goal_behavior_agent", lambda *args, **kwargs: (agent, True))

    dataset, summary = goal_scientist_public._collect_goal_scientist_session(
        _LevelBoundarySessionEnv("game-0"),
        game_id="game-0",
        session_index=0,
        seed=17,
        policy_name="graph",
        config=_make_goal_scientist_config(),
    )

    assert len(dataset) == 3
    required_keys = {
        "levels_completed_before",
        "levels_completed_after",
        "level_delta",
        "reset_action",
        "level_boundary",
        "failure_terminal",
        "session_terminal",
        "won_session",
    }
    assert required_keys <= set(dataset[0])
    assert dataset[0]["levels_completed_before"] == 0
    assert dataset[0]["levels_completed_after"] == 1
    assert dataset[0]["level_delta"] == 1
    assert dataset[0]["level_boundary"] is True
    assert dataset[-1]["won_session"] is True
    assert dataset[-1]["session_terminal"] is True
    assert summary["level_boundaries"] == 2
    assert summary["levels_completed"] == 2


def test_scientist_train_arc_result_key_prefers_session_outcomes_over_raw_return() -> None:
    high_return_without_progress = {
        "success_rate": 0.0,
        "avg_return": 4.0,
        "avg_steps": 16.0,
        "session_win_rate": 0.0,
        "won_session_rate": 0.0,
        "avg_levels_completed": 0.0,
        "levels_completed": 0.0,
    }
    completed_levels_without_session_win = {
        "success_rate": 0.0,
        "avg_return": 0.25,
        "avg_steps": 64.0,
        "session_win_rate": 0.0,
        "won_session_rate": 0.0,
        "avg_levels_completed": 2.0,
        "levels_completed": 2.0,
    }
    won_session_with_lower_return = {
        "success_rate": 0.0,
        "avg_return": 0.1,
        "avg_steps": 96.0,
        "session_win_rate": 1.0,
        "won_session_rate": 1.0,
        "avg_levels_completed": 3.0,
        "levels_completed": 3.0,
    }

    assert _result_key(completed_levels_without_session_win) > _result_key(high_return_without_progress)
    assert _result_key(won_session_with_lower_return) > _result_key(high_return_without_progress)
