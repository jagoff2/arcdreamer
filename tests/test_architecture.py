import inspect

import torch

from src.env import NUM_ACTIONS, SENSOR_DIM, TinyWorldRuntime, generate_batch
from src.model import RecurrentLatentModel
from src import run_unbroken


def test_model_exposes_recurrent_step_and_persistent_latent() -> None:
    model = RecurrentLatentModel()
    batch = generate_batch(batch_size=2, seq_len=3, base_seed=10)
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


def test_runtime_does_not_feed_generated_language_back_as_state() -> None:
    source = inspect.getsource(run_unbroken.run_unbroken)
    assert "model.step(observation, z)" in source
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
