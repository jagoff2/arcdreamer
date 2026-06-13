from __future__ import annotations

import torch

from src.env import (
    ACTION_LEFT,
    ACTION_REST,
    GRID_SIZE,
    NUM_ACTIONS,
    NUM_BODY_SCALARS,
    NUM_COLORS,
    SENSOR_DIM,
    TinyWorldRuntime,
    generate_batch,
    shortest_action,
)


def _color_slice() -> slice:
    start = GRID_SIZE + 2 + NUM_BODY_SCALARS
    return slice(start, start + NUM_COLORS + 1)


def _visible_flag_index() -> int:
    return GRID_SIZE + 2 + NUM_BODY_SCALARS + (NUM_COLORS + 1)


def _object_pos_slice() -> slice:
    start = _visible_flag_index() + 1
    return slice(start, start + GRID_SIZE + 1)


def test_tiny_world_observation_exposes_only_public_tensors() -> None:
    world = TinyWorldRuntime(seed=11)
    observation = world.observation(device="cpu", private_in=7)

    assert set(observation.keys()) == {"sensory", "lang_in", "private_in"}
    assert observation["sensory"].shape == (1, SENSOR_DIM)
    assert observation["lang_in"].shape == (1,)
    assert observation["private_in"].shape == (1,)
    assert observation["sensory"].dtype == torch.float32
    assert observation["lang_in"].dtype == torch.long
    assert observation["private_in"].dtype == torch.long
    assert observation["private_in"].item() == 7
    assert "goal" not in observation
    assert "hazard" not in observation
    assert "raw" not in observation


def test_tiny_world_observation_is_partially_observable_over_time() -> None:
    world = TinyWorldRuntime(seed=17)

    for tick in range(6):
        observation = world.observation(device="cpu")
        sensory = observation["sensory"][0]
        visible_color = int(sensory[_color_slice()].argmax().item())
        visible_pos = int(sensory[_object_pos_slice()].argmax().item())
        visible_flag = float(sensory[_visible_flag_index()].item())

        if tick < 4:
            assert 0 <= visible_color < NUM_COLORS
            assert 0 <= visible_pos < GRID_SIZE
            assert visible_flag == 1.0
        else:
            assert visible_color == NUM_COLORS
            assert visible_pos == GRID_SIZE
            assert visible_flag == 0.0

        world.step(ACTION_LEFT)


def test_tiny_world_actions_update_public_observations() -> None:
    world = TinyWorldRuntime(seed=23)
    initial = world.observation(device="cpu")["sensory"]
    world.step(ACTION_REST)
    after_rest = world.observation(device="cpu")["sensory"]
    world.step(ACTION_REST)
    after_second_rest = world.observation(device="cpu")["sensory"]

    assert not torch.allclose(initial, after_rest)
    assert not torch.allclose(after_rest, after_second_rest)


def test_generate_batch_is_reproducible_and_has_valid_transition_impulses() -> None:
    batch_a = generate_batch(batch_size=4, seq_len=12, base_seed=1234, device="cpu")
    batch_b = generate_batch(batch_size=4, seq_len=12, base_seed=1234, device="cpu")
    batch_c = generate_batch(batch_size=4, seq_len=12, base_seed=1235, device="cpu")

    for key in (
        "sensory",
        "lang_in",
        "private_in",
        "action_target",
        "prev_action",
        "prev_delta",
        "dt",
        "action_mask",
        "delayed_memory_mask",
        "object_mask",
        "grounded_language_mask",
        "self_mask",
    ):
        assert torch.equal(batch_a[key], batch_b[key]), key

    assert not torch.equal(batch_a["sensory"], batch_c["sensory"])

    assert batch_a["prev_action"].shape == (4, 12)
    assert batch_a["prev_delta"].shape == (4, 12, SENSOR_DIM)
    assert batch_a["dt"].shape == (4, 12)
    assert batch_a["action_mask"].shape == (4, 12)
    assert batch_a["delayed_memory_mask"].shape == (4, 12)
    assert batch_a["object_mask"].shape == (4, 12)
    assert batch_a["grounded_language_mask"].shape == (4, 12)
    assert batch_a["self_mask"].shape == (4, 12)

    assert batch_a["prev_action"].dtype == torch.long
    assert batch_a["dt"].dtype == torch.float32
    assert batch_a["action_mask"].dtype == torch.bool
    assert batch_a["delayed_memory_mask"].dtype == torch.bool
    assert batch_a["object_mask"].dtype == torch.bool
    assert batch_a["grounded_language_mask"].dtype == torch.bool
    assert batch_a["self_mask"].dtype == torch.bool

    assert torch.equal(batch_a["prev_action"][:, 0], torch.full((4,), NUM_ACTIONS, dtype=torch.long))
    assert torch.allclose(batch_a["prev_delta"][:, 0], torch.zeros(4, SENSOR_DIM))
    assert torch.allclose(batch_a["dt"], torch.ones(4, 12))
    assert torch.equal(batch_a["prev_delta"][:, 1:], batch_a["sensory"][:, 1:] - batch_a["sensory"][:, :-1])
    assert (batch_a["prev_action"][:, 1:] != NUM_ACTIONS).any()


def test_generated_action_targets_match_closed_loop_expected_action_contract() -> None:
    batch = generate_batch(batch_size=6, seq_len=18, base_seed=777, device="cpu")
    current_pos = batch["sensory"][:, :, :GRID_SIZE].argmax(dim=-1)
    expected = shortest_action(current_pos, batch["world_pos_target"])

    assert torch.equal(batch["action_target"], expected)
