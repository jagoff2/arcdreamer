from __future__ import annotations

from arcagi.evaluation.harness import run_episode
from arcagi.learned_online.curriculum import (
    ActionNameRemapHeldoutTask,
    DelayedUnlockTask,
    DenseFamilyMassArbitrationTask,
    LongDenseDecoyTask,
    LongPostBoundaryCarryoverTask,
    LongSparseMixedFamilyChainTask,
    ModeThenDenseClickTask,
    MovementRequiredAfterModeTask,
    RoleOpaqueRemapTask,
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


def test_long_sparse_chain_requires_full_mixed_family_sequence() -> None:
    env = LongSparseMixedFamilyChainTask(seed=13, chain_length=24, size=6)
    env.reset(seed=13)
    sequence = env.expected_action_sequence()
    wrong = run_episode(ScriptedAgent(["noop"] * len(sequence)), LongSparseMixedFamilyChainTask(seed=13, chain_length=24, size=6), seed=13, max_steps=32)
    right = run_episode(ScriptedAgent(sequence), LongSparseMixedFamilyChainTask(seed=13, chain_length=24, size=6), seed=13, max_steps=32)

    assert wrong["return"] == 0.0
    assert right["return"] == 1.0


def test_long_dense_decoy_rewards_target_click_chain_not_decoy_flash() -> None:
    env = LongDenseDecoyTask(seed=14, chain_length=20, size=6)
    env.reset(seed=14)
    sequence = env.expected_action_sequence()
    decoy_y, decoy_x = env.decoy_cells[0][0]
    decoy = f"click:{decoy_x}:{decoy_y}"

    wrong = run_episode(ScriptedAgent([decoy] * 24), LongDenseDecoyTask(seed=14, chain_length=20, size=6), seed=14, max_steps=28)
    right = run_episode(ScriptedAgent(sequence), LongDenseDecoyTask(seed=14, chain_length=20, size=6), seed=14, max_steps=28)

    assert wrong["return"] == 0.0
    assert right["return"] == 1.0


def test_role_opaque_remap_has_no_action_role_shortcut() -> None:
    env = RoleOpaqueRemapTask(seed=15)
    env.reset(seed=15)
    correct = env.expert_action()
    wrong = next(action for action in env.actions() if action != correct)

    wrong_result = run_episode(ScriptedAgent([wrong]), RoleOpaqueRemapTask(seed=15), seed=15, max_steps=2)
    right_result = run_episode(ScriptedAgent([correct]), RoleOpaqueRemapTask(seed=15), seed=15, max_steps=2)

    assert wrong_result["return"] == 0.0
    assert right_result["return"] == 1.0


def test_post_boundary_carryover_requires_persistent_selector_after_level_transition() -> None:
    env = LongPostBoundaryCarryoverTask(seed=16, level_count=3, size=6)
    env.reset(seed=16)
    sequence = env.expected_action_sequence()
    wrong = run_episode(ScriptedAgent(["noop"] * len(sequence)), LongPostBoundaryCarryoverTask(seed=16, level_count=3, size=6), seed=16, max_steps=12)
    right = run_episode(ScriptedAgent(sequence), LongPostBoundaryCarryoverTask(seed=16, level_count=3, size=6), seed=16, max_steps=12)

    assert wrong["return"] == 0.0
    assert right["return"] == 3.0
    assert right["levels_completed"] == 3
    assert right["won"] is True
