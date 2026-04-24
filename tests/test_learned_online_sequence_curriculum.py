from __future__ import annotations

from arcagi.evaluation.harness import run_episode
from arcagi.learned_online.curriculum import DelayedUnlockTask, MovementRequiredAfterModeTask, VisibleMovementTrapTask


class ScriptedAgent:
    name = "scripted"
    latest_language = ()

    def __init__(self, actions):
        self.actions = tuple(actions)
        self.index = 0

    def reset_episode(self) -> None:
        self.index = 0

    def act(self, _observation):
        action = self.actions[min(self.index, len(self.actions) - 1)]
        self.index += 1
        return action

    def update_after_step(self, **_kwargs) -> None:
        return None


def test_visible_movement_trap_rewards_commit_not_movement() -> None:
    moving = run_episode(ScriptedAgent(["move_right", "move_right", "move_left"]), VisibleMovementTrapTask(), seed=0, max_steps=3)
    commit = run_episode(ScriptedAgent(["commit"]), VisibleMovementTrapTask(), seed=0, max_steps=3)

    assert moving["return"] == 0.0
    assert commit["return"] == 1.0


def test_movement_required_after_mode_requires_sequence() -> None:
    wrong = run_episode(ScriptedAgent(["move_right", "move_right"]), MovementRequiredAfterModeTask(), seed=0, max_steps=3)
    right = run_episode(ScriptedAgent(["mode", "move_right", "move_right"]), MovementRequiredAfterModeTask(), seed=0, max_steps=3)

    assert wrong["return"] == 0.0
    assert right["return"] == 1.0


def test_delayed_unlock_requires_latent_carryover() -> None:
    wrong = run_episode(ScriptedAgent(["goal", "goal"]), DelayedUnlockTask(), seed=0, max_steps=3)
    right = run_episode(ScriptedAgent(["unlock", "goal"]), DelayedUnlockTask(), seed=0, max_steps=3)

    assert wrong["return"] == 0.0
    assert right["return"] == 1.0
