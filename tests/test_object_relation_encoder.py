from __future__ import annotations

import numpy as np
import torch

from src.env import NUM_ACTIONS, generate_batch
from src.grid_perception import parse_grid
from src.model import ObjectRelationEncoder, RecurrentLatentModel, load_checkpoint


def _sample_scene() -> object:
    grid = np.array(
        [
            [0, 1, 1, 0, 1, 1],
            [0, 1, 1, 0, 1, 1],
            [0, 0, 0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    return parse_grid(grid)


def test_object_relation_encoder_accepts_scene_and_preserves_hidden_shape() -> None:
    scene = _sample_scene()
    encoder = ObjectRelationEncoder(hidden_dim=32)

    single = encoder(scene)
    pair = encoder([scene, scene])
    tiled = encoder(scene, batch_size=3)

    assert single.shape == (1, 32)
    assert pair.shape == (2, 32)
    assert tiled.shape == (3, 32)


def test_object_relation_encoder_eval_mode_is_deterministic_for_same_scene() -> None:
    scene = _sample_scene()
    encoder = ObjectRelationEncoder(hidden_dim=24)
    encoder.eval()

    with torch.no_grad():
        first = encoder(scene)
        second = encoder(scene)

    torch.testing.assert_close(first, second)


def test_object_relation_encoder_backward_flow_covers_encoder_parameters() -> None:
    scene = _sample_scene()
    encoder = ObjectRelationEncoder(hidden_dim=16)

    embedding = encoder(scene, batch_size=1)
    loss = embedding.mean()
    loss.backward()

    grads = [param.grad for param in encoder.parameters() if param.requires_grad]
    assert any(grad is not None for grad in grads)
    assert any(grad is not None and torch.any(grad != 0.0) for grad in grads)


def test_recurrent_step_uses_optional_object_relation_features_and_preserves_baseline() -> None:
    batch = generate_batch(batch_size=1, seq_len=2, base_seed=17, device="cpu")
    scene = _sample_scene()
    encoder = ObjectRelationEncoder(hidden_dim=64)
    object_features = encoder.object_features_from_scene(scene, device="cpu").unsqueeze(0)
    relation_features = encoder.relation_features_from_scene(scene, device="cpu").unsqueeze(0)

    model = RecurrentLatentModel(device="cpu")
    z0 = model.initial_state(1, device="cpu")
    observation = {
        "sensory": batch["sensory"][:, 0],
        "lang_in": batch["lang_in"][:, 0],
        "private_in": batch["private_in"][:, 0],
    }

    baseline_output, baseline_latent = model.step(observation, z0)
    compatibility_output, compatibility_latent = model.step(observation, z0)
    with_features_output, with_features_latent = model.step(
        {
            **observation,
            "object_features": object_features,
            "relation_features": relation_features,
        },
        z0,
    )

    assert baseline_output["action_logits"].shape == (1, NUM_ACTIONS)
    assert compatibility_output["action_logits"].shape == (1, NUM_ACTIONS)
    assert torch.allclose(baseline_latent, compatibility_latent)
    assert torch.allclose(baseline_output["action_logits"], compatibility_output["action_logits"])
    assert not torch.allclose(baseline_latent, with_features_latent)
    assert not torch.allclose(baseline_output["action_logits"], with_features_output["action_logits"])


def test_checkpoint_loader_accepts_missing_object_relation_encoder_keys(tmp_path) -> None:
    model = RecurrentLatentModel(device="cpu")
    state = {
        key: value
        for key, value in model.state_dict().items()
        if not key.startswith("object_relation_encoder.")
    }
    checkpoint = tmp_path / "older_without_object_relation_encoder.pt"
    torch.save(
        {
            "model_config": {
                key: value
                for key, value in model.config.__dict__.items()
                if key not in {"object_feature_dim", "relation_feature_dim"}
            },
            "model_state": state,
            "train_config": {},
            "metrics": {},
        },
        checkpoint,
    )

    loaded = load_checkpoint(checkpoint, device="cpu")

    assert isinstance(loaded.object_relation_encoder, ObjectRelationEncoder)
