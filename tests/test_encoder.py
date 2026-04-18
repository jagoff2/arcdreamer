from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from arcagi.core.types import GridObservation
from arcagi.models.encoder import GRID_VALUE_BUCKETS, MAX_GRID_SIZE, MAX_OBJECTS, StructuredStateEncoder
from arcagi.models.language import GroundedLanguageModel
from arcagi.models.world_model import RecurrentWorldModel
from arcagi.perception.object_encoder import extract_structured_state
from arcagi.training.synthetic import load_checkpoint


def _many_object_state() -> object:
    grid = np.zeros((18, 18), dtype=np.int64)
    color = 1
    for row in range(0, 18, 2):
        for col in range(0, 18, 2):
            grid[row, col] = color
            color = 1 + (color % 9)
    observation = GridObservation(
        task_id="encoder/test",
        episode_id="episode-0",
        step_index=0,
        grid=grid,
        available_actions=("up", "down", "left", "right"),
    )
    return extract_structured_state(observation)


def test_encoder_represents_more_than_sixteen_objects_and_full_grid() -> None:
    state = _many_object_state()
    assert len(state.objects) > 16

    encoder = StructuredStateEncoder()
    encoded = encoder.encode_state(state, device=torch.device("cpu"))

    assert encoded.latent.shape == (1, encoder.latent_dim)
    assert encoded.object_rows.shape == (1, MAX_OBJECTS, 12)
    assert encoded.grid.shape == (1, MAX_GRID_SIZE, MAX_GRID_SIZE)
    assert encoded.object_rows[0, 16:].abs().sum().item() > 0.0
    assert torch.count_nonzero(encoded.grid).item() >= len(state.objects)


def test_encoder_accepts_grid_values_beyond_original_palette_ceiling() -> None:
    grid = np.full((6, 6), GRID_VALUE_BUCKETS + 37, dtype=np.int64)
    grid[2:4, 2:4] = 0
    observation = GridObservation(
        task_id="encoder/high-colors",
        episode_id="episode-0",
        step_index=0,
        grid=grid,
        available_actions=("wait",),
        extras={"background_color": GRID_VALUE_BUCKETS + 37},
    )
    state = extract_structured_state(observation)
    encoder = StructuredStateEncoder()

    encoded = encoder.encode_state(state, device=torch.device("cpu"))

    assert encoded.latent.shape == (1, encoder.latent_dim)


def test_load_checkpoint_accepts_old_encoder_state_dict(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    encoder = StructuredStateEncoder()
    world_model = RecurrentWorldModel()
    language_model = GroundedLanguageModel()
    legacy_encoder_state = {
        key: value
        for key, value in encoder.state_dict().items()
        if not (
            key.startswith("object_attention.")
            or key == "object_query"
            or key.startswith("grid_embedding.")
            or key.startswith("grid_conv.")
            or key.startswith("enhancement_mlp.")
            or key == "enhancement_gate"
        )
    }
    checkpoint_path = tmp_path / "legacy_encoder.pt"
    torch.save(
        {
            "encoder": legacy_encoder_state,
            "world_model": world_model.state_dict(),
            "language_model": language_model.state_dict(),
        },
        checkpoint_path,
    )

    with caplog.at_level("WARNING"):
        loaded_encoder, loaded_world_model, loaded_language_model = load_checkpoint(str(checkpoint_path))

    assert "Compatibility load for encoder skipped or missed parameters" in caplog.text
    state = _many_object_state()
    encoded = loaded_encoder.encode_state(state, device=torch.device("cpu"))

    assert encoded.latent.shape == (1, loaded_encoder.latent_dim)
    assert loaded_world_model.summary_dim == 25
    assert len(loaded_language_model.vocab.id_to_token) > 0
