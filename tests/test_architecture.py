import inspect
from dataclasses import asdict

import torch

from src.env import NUM_ACTIONS, NUM_COLORS, SENSOR_DIM, TinyWorldRuntime, generate_batch
from src.model import RecurrentLatentModel, decode_program_proposals, load_checkpoint
from src import run_unbroken


def test_model_exposes_recurrent_step_and_persistent_latent() -> None:
    model = RecurrentLatentModel(device="cpu")
    batch = generate_batch(batch_size=2, seq_len=3, base_seed=10, device="cpu")
    z0 = model.initial_state(2)
    output, z1 = model.step({"sensory": batch["sensory"][:, 0], "lang_in": batch["lang_in"][:, 0]}, z0)
    _, z2 = model.step({"sensory": batch["sensory"][:, 1], "lang_in": batch["lang_in"][:, 1]}, z1)
    assert z1.shape == z0.shape
    assert z2.shape == z0.shape
    assert not torch.allclose(z0, z1)
    assert not torch.allclose(z1, z2)
    assert output["action_logits"].shape == (2, NUM_ACTIONS)
    assert output["private_logits"].shape[0] == 2
    assert batch["sensory"].shape[-1] == SENSOR_DIM


def test_recurrent_step_accepts_previous_action_and_delta_impulses() -> None:
    model = RecurrentLatentModel(device="cpu")
    batch = generate_batch(batch_size=2, seq_len=3, base_seed=11, device="cpu")
    z0 = model.initial_state(2)
    observation = {
        "sensory": batch["sensory"][:, 1],
        "lang_in": batch["lang_in"][:, 1],
        "private_in": batch["private_in"][:, 1],
        "prev_action": batch["prev_action"][:, 1],
        "prev_delta": batch["prev_delta"][:, 1],
    }

    assert observation["prev_action"].shape == (2,)
    assert observation["prev_delta"].shape == (2, SENSOR_DIM)

    with torch.no_grad():
        model.action_impulse_embedding.weight[1].fill_(0.15)
        model.action_impulse_embedding.weight[2].fill_(-0.15)
        model.delta_impulse_encoder.weight.fill_(0.01)
        model.delta_impulse_encoder.bias.zero_()

    _, with_left = model.step({**observation, "prev_action": torch.ones(2, dtype=torch.long)}, z0)
    _, with_right = model.step(
        {**observation, "prev_action": torch.full((2,), 2, dtype=torch.long)},
        z0,
    )
    _, with_delta = model.step({**observation, "prev_delta": observation["prev_delta"] + 1.0}, z0)

    assert not torch.allclose(with_left, with_right)
    assert not torch.allclose(with_left, with_delta)


def test_generated_batches_include_transition_impulses() -> None:
    batch = generate_batch(batch_size=3, seq_len=5, base_seed=12, device="cpu")

    assert batch["prev_action"].shape == (3, 5)
    assert batch["prev_delta"].shape == (3, 5, SENSOR_DIM)
    assert batch["dt"].shape == (3, 5)
    assert torch.equal(batch["prev_action"][:, 0], torch.full((3,), NUM_ACTIONS))
    assert torch.allclose(batch["prev_delta"][:, 0], torch.zeros(3, SENSOR_DIM))
    assert torch.allclose(batch["prev_delta"][:, 1:], batch["sensory"][:, 1:] - batch["sensory"][:, :-1])
    assert torch.allclose(batch["dt"], torch.ones(3, 5))


def test_recurrent_core_has_liquid_time_constant_dynamics() -> None:
    model = RecurrentLatentModel(device="cpu")
    batch = generate_batch(batch_size=1, seq_len=2, base_seed=13, device="cpu")
    z0 = model.initial_state(1, device="cpu")
    observation = {
        "sensory": batch["sensory"][:, 1],
        "lang_in": batch["lang_in"][:, 1],
        "private_in": batch["private_in"][:, 1],
        "prev_action": batch["prev_action"][:, 1],
        "prev_delta": batch["prev_delta"][:, 1],
    }

    assert hasattr(model, "liquid_time_constant")
    assert hasattr(model, "liquid_drift")

    with torch.no_grad():
        model.liquid_time_constant.weight.zero_()
        model.liquid_time_constant.bias.zero_()
        model.liquid_drift[-1].weight.zero_()
        model.liquid_drift[-1].bias.fill_(0.25)

    _, no_elapsed_time = model.step({**observation, "dt": torch.zeros(1)}, z0)
    _, elapsed_time = model.step({**observation, "dt": torch.full((1,), 2.0)}, z0)

    assert not torch.allclose(no_elapsed_time, elapsed_time)


def test_model_exposes_neural_program_proposal_heads_and_decoder() -> None:
    model = RecurrentLatentModel(device="cpu")
    batch = generate_batch(batch_size=2, seq_len=2, base_seed=14, device="cpu")
    z0 = model.initial_state(2, device="cpu")

    output, _ = model.step(
        {
            "sensory": batch["sensory"][:, 0],
            "lang_in": batch["lang_in"][:, 0],
            "private_in": batch["private_in"][:, 0],
        },
        z0,
    )

    assert output["program_family_logits"].shape == (2, 4)
    assert output["program_field_logits"].shape == (2, 4)
    assert output["program_transform_logits"].shape == (2, 6)
    assert output["program_color_logits"].shape[-1] >= 2

    proposals = decode_program_proposals(
        {
            "program_family_logits": torch.tensor([[0.0, 0.5, 0.25, 4.0], [4.0, 0.0, 0.0, 0.0]]),
            "program_field_logits": torch.tensor([[0.0, 3.0, 0.0, 0.0], [3.0, 0.0, 0.0, 0.0]]),
            "program_transform_logits": torch.tensor([[0.0, 0.0, 0.0, 0.0, 0.0, 5.0], [5.0, 0.0, 0.0, 0.0, 0.0, 0.0]]),
            "program_color_logits": torch.tensor(
                [[0.0, 0.0, 6.0, 0.0][:NUM_COLORS], [6.0, 0.0, 0.0, 0.0][:NUM_COLORS]]
            ),
        },
        action=3,
        top_k=2,
    )

    assert proposals[0]["schema"] == "neural_causal_program_proposal_v1"
    assert proposals[0]["source"] == "neural_proposal"
    assert proposals[0]["family"] == "move_color"
    assert proposals[0]["action"] == "3"
    assert proposals[0]["selector"] == {"color": 2}
    assert proposals[0]["transform"] == {"kind": "translate", "dy": 0.0, "dx": 1.0}
    assert proposals[0]["goal_test"] == {"kind": "component_translation"}


def test_checkpoint_loader_accepts_missing_liquid_dynamics_keys(tmp_path) -> None:
    model = RecurrentLatentModel(device="cpu")
    state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("liquid_time_constant.") and not key.startswith("liquid_drift.")
    }
    checkpoint = tmp_path / "older.pt"
    torch.save(
        {
            "model_config": asdict(model.config),
            "model_state": state,
            "train_config": {},
            "metrics": {},
        },
        checkpoint,
    )

    loaded = load_checkpoint(checkpoint, device="cpu")

    assert isinstance(loaded, RecurrentLatentModel)
    assert hasattr(loaded, "liquid_time_constant")


def test_checkpoint_loader_accepts_missing_program_proposal_keys(tmp_path) -> None:
    model = RecurrentLatentModel(device="cpu")
    state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("program_")
    }
    checkpoint = tmp_path / "older_programless.pt"
    torch.save(
        {
            "model_config": asdict(model.config),
            "model_state": state,
            "train_config": {},
            "metrics": {},
        },
        checkpoint,
    )

    loaded = load_checkpoint(checkpoint, device="cpu")

    assert isinstance(loaded, RecurrentLatentModel)
    assert "program_family_head.weight" in loaded.state_dict()


def test_runtime_does_not_feed_generated_language_back_as_state() -> None:
    source = inspect.getsource(run_unbroken.run_unbroken)
    assert '"prev_action"' in source
    assert '"prev_delta"' in source
    assert "model.step(model_observation, z)" in source
    assert "world.step(action)" in source
    assert "world.step(language" not in source
    assert "prompt" not in source.lower()


def test_generated_language_is_not_state_carrier_behaviorally() -> None:
    model = RecurrentLatentModel()
    world = TinyWorldRuntime(seed=99)
    z = model.initial_state(1)
    generated_language = []
    environment_language = []
    latent_ids = []

    with torch.no_grad():
        for _ in range(12):
            observation = world.observation()
            environment_language.append(int(observation["lang_in"].item()))
            output, z_next = model.step(observation, z)
            generated = int(output["language_logits"].argmax(dim=-1).item())
            generated_language.append(generated)
            latent_ids.append(id(z_next))
            world.step(int(output["action_logits"].argmax(dim=-1).item()))
            z = z_next

    assert environment_language[1:] != generated_language[:-1]
    assert len(set(latent_ids)) > 1
