from __future__ import annotations

import torch

from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.types import GridObservation
from arcagi.models.action_encoder import ActionEncoder, build_action_context
from arcagi.models.world_model import RecurrentWorldModel
from arcagi.perception.object_encoder import extract_structured_state


def test_action_encoder_distinguishes_dynamic_click_coordinates() -> None:
    encoder = ActionEncoder(embedding_dim=24)
    context = build_action_context(
        affordances=("1", "click:4:4", "click:10:4", "undo"),
        action_roles={"1": "move", "click:4:4": "click", "click:10:4": "click", "undo": "undo"},
    )

    first = encoder.encode(["click:4:4"], device=torch.device("cpu"))
    second = encoder.encode(["click:10:4"], device=torch.device("cpu"))

    assert first.shape == (1, 24)
    assert second.shape == (1, 24)
    assert not torch.allclose(first, second)


def test_structured_state_carries_action_roles_for_generic_modeling() -> None:
    observation = GridObservation(
        task_id="arc/test",
        episode_id="episode-0",
        step_index=0,
        grid=torch.zeros((3, 3), dtype=torch.int64).numpy(),
        available_actions=("1", "click:4:4", "undo"),
        extras={"action_roles": {"1": "move", "click:4:4": "click", "undo": "undo"}},
    )

    state = extract_structured_state(observation)

    assert dict(state.action_roles)["click:4:4"] == "click"
    assert dict(state.action_roles)["undo"] == "undo"


def test_world_model_accepts_dynamic_arc_actions_without_fixed_ids() -> None:
    observation = GridObservation(
        task_id="arc/test",
        episode_id="episode-1",
        step_index=0,
        grid=torch.zeros((4, 4), dtype=torch.int64).numpy(),
        available_actions=("1", "click:4:4", "click:10:4", "undo"),
        extras={"action_roles": {"1": "move", "click:4:4": "click", "click:10:4": "click", "undo": "undo"}},
    )
    state = extract_structured_state(observation)
    world_model = RecurrentWorldModel(latent_dim=16, hidden_dim=16, action_dim=12, summary_dim=25)
    latent = torch.randn(1, 16)

    prediction = world_model.step(latent, actions=["click:10:4"], state=state)

    assert prediction.hidden.shape == (1, 16)
    assert prediction.next_latent_mean.shape == (1, 16)
    assert prediction.policy.shape == (1,)


def test_action_schema_groups_parametric_clicks_by_family_and_bin() -> None:
    context = build_action_schema_context(
        affordances=("click:4:4", "click:10:4", "undo"),
        action_roles={"click:4:4": "click", "click:10:4": "click", "undo": "undo"},
    )

    first = build_action_schema("click:4:4", context)
    second = build_action_schema("click:10:4", context)

    assert first.family == second.family
    assert first.coarse_bin != second.coarse_bin


def test_action_schema_recognizes_reset_action_type() -> None:
    context = build_action_schema_context(
        affordances=("0", "1", "click:4:4"),
        action_roles={"0": "reset_level", "1": "move_up", "click:4:4": "click"},
    )

    schema = build_action_schema("0", context)

    assert schema.action_type == "reset"
    assert schema.role == "reset_level"
