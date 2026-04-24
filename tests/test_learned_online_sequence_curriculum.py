from __future__ import annotations

from arcagi.evaluation.harness import run_episode
from arcagi.learned_online.curriculum import (
    ActionNameRemapHeldoutTask,
    DelayedUnlockTask,
    DenseFamilyMassArbitrationTask,
    ModeThenDenseClickTask,
    MovementRequiredAfterModeTask,
    VisibleMovementTrapTask,
)


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


def test_dense_family_mass_arbitration_rewards_non_click_family() -> None:
    click_only = run_episode(
        ScriptedAgent(["click:0:0", "click:1:0", "click:2:0"]),
        DenseFamilyMassArbitrationTask(size=12),
        seed=0,
        max_steps=3,
    )
    select = run_episode(ScriptedAgent(["5"]), DenseFamilyMassArbitrationTask(size=12), seed=0, max_steps=3)

    assert click_only["return"] == 0.0
    assert select["return"] == 1.0


def test_mode_then_dense_click_requires_nonclick_then_context_click() -> None:
    env = ModeThenDenseClickTask(seed=4, size=6)
    env.reset(seed=4)
    y, x = env.target_cell
    click = f"click:{x}:{y}"

    click_only = run_episode(ScriptedAgent([click, click]), ModeThenDenseClickTask(seed=4, size=6), seed=4, max_steps=3)
    sequenced = run_episode(ScriptedAgent(["5", click]), ModeThenDenseClickTask(seed=4, size=6), seed=4, max_steps=3)

    assert click_only["return"] == 0.0
    assert sequenced["return"] == 1.0


def test_action_name_remap_uses_role_not_literal_action_name() -> None:
    env = ActionNameRemapHeldoutTask(seed=9)
    env.reset(seed=9)
    roles = env._action_roles()
    correct = next(action for action, role in roles.items() if role == env.target_role)
    wrong = next(action for action, role in roles.items() if role != env.target_role)

    wrong_result = run_episode(ScriptedAgent([wrong]), ActionNameRemapHeldoutTask(seed=9), seed=9, max_steps=2)
    right_result = run_episode(ScriptedAgent([correct]), ActionNameRemapHeldoutTask(seed=9), seed=9, max_steps=2)

    assert wrong_result["return"] == 0.0
    assert right_result["return"] == 1.0
