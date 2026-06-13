from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch

from src.causal_hypotheses import posterior_predictive_mixture
from src.persistent_memory import PersistentMemoryState
from src.run_unbroken import run_unbroken
from src.train import train_model


def test_fresh_state_initializes_hypothesis_posterior() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    posterior = memory.hypothesis_posterior

    assert posterior["schema"] == "runtime_causal_hypothesis_posterior_v1"
    assert posterior["updates"] == 0
    assert posterior["hypotheses"] == {}
    assert posterior["posterior_mass"] == 0.0
    assert posterior["top_hypotheses"] == []


def test_append_event_generates_change_stable_and_no_op_hypotheses() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = {"sensory": torch.tensor([1.0]), "private": torch.tensor([3])}
    changed = {"sensory": torch.tensor([2.0]), "private": torch.tensor([3])}
    stable = {"sensory": torch.tensor([2.0]), "private": torch.tensor([3])}

    memory.append_event(
        tick=1,
        observation=before,
        action=1,
        next_observation=changed,
    )
    memory.append_event(
        tick=2,
        observation=changed,
        action=1,
        next_observation=stable,
    )

    hypotheses = memory.hypothesis_posterior["hypotheses"]
    assert "field_change|action:1|field:sensory" in hypotheses
    assert "field_stable|action:1|field:private" in hypotheses
    assert "no_op|action:1" in hypotheses


def test_append_event_generates_move_color_hypothesis_for_grid_translation() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = {"grid": torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.long)}
    after = {"grid": torch.tensor([[0, 1, 0], [0, 0, 0], [0, 0, 0]], dtype=torch.long)}

    memory.append_event(
        tick=1,
        observation=before,
        action="move_component",
        next_observation=after,
    )

    hypotheses = memory.hypothesis_posterior["hypotheses"]
    move_keys = [key for key in hypotheses if key.startswith("move_color|action:move_component")]
    assert len(move_keys) == 1

    hypothesis = hypotheses[move_keys[0]]
    assert hypothesis["family"] == "move_color"
    assert hypothesis["selector"]["color"] == 1
    assert hypothesis["transform"]["kind"] == "translate"
    assert hypothesis["transform"]["dy"] == 0.0
    assert hypothesis["transform"]["dx"] == 1.0
    assert hypothesis["goal_test"]["kind"] == "component_translation"


def test_hypothesis_scoring_prefers_shorter_description_given_equal_support() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = {"sensory": torch.tensor([1.0]), "private": torch.tensor([0])}

    # Same stable event twice gives both no-op and field_stable hypotheses with equal support.
    # Shorter descriptions should be preferred in posterior mass.
    memory.append_event(
        tick=1,
        observation=before,
        action=2,
        next_observation=before,
    )
    memory.append_event(
        tick=2,
        observation=before,
        action=2,
        next_observation=before,
    )

    hypotheses = memory.hypothesis_posterior["hypotheses"]
    no_op = hypotheses["no_op|action:2"]
    sensory_stable = hypotheses["field_stable|action:2|field:sensory"]
    private_stable = hypotheses["field_stable|action:2|field:private"]

    assert no_op["support"] == sensory_stable["support"] == private_stable["support"] == 2
    assert math.isclose(memory.hypothesis_posterior["posterior_mass"], 1.0, rel_tol=1.0e-12)
    assert no_op["posterior"] > sensory_stable["posterior"]
    assert no_op["posterior"] > private_stable["posterior"]


def test_repeated_consistent_transitions_increase_support_and_posterior() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    obs = {"x": torch.tensor([0.0])}
    obs1 = {"x": torch.tensor([1.0])}
    obs2 = {"x": torch.tensor([2.0])}
    obs3 = {"x": torch.tensor([3.0])}

    memory.append_event(tick=1, observation=obs, action=0, next_observation=obs1)
    memory.append_event(tick=2, observation=obs1, action=1, next_observation=obs2)
    before = memory.hypothesis_posterior["hypotheses"]["field_change|action:0|field:x"]["posterior"]

    memory.append_event(tick=3, observation=obs2, action=0, next_observation=obs3)
    hypothesis = memory.hypothesis_posterior["hypotheses"]["field_change|action:0|field:x"]

    assert hypothesis["support"] == 2
    assert hypothesis["counterexamples"] == 0
    assert hypothesis["posterior"] > before


def test_counterexample_transition_increases_counterexamples_and_reduces_posterior() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    obs = {"x": torch.tensor([0.0])}
    obs1 = {"x": torch.tensor([1.0])}
    obs2 = {"x": torch.tensor([2.0])}
    obs3 = {"x": torch.tensor([3.0])}

    memory.append_event(tick=1, observation=obs, action=0, next_observation=obs1)
    memory.append_event(tick=2, observation=obs1, action=1, next_observation=obs2)
    memory.append_event(tick=3, observation=obs2, action=0, next_observation=obs3)
    before = memory.hypothesis_posterior["hypotheses"]["field_change|action:0|field:x"]["posterior"]

    memory.append_event(tick=4, observation=obs3, action=0, next_observation=obs3)
    after = memory.hypothesis_posterior["hypotheses"]["field_change|action:0|field:x"]

    assert after["counterexamples"] == 1
    assert after["support"] == 2
    assert after["posterior"] < before


def test_posterior_predictive_mixture_is_weighted_causal_program_distribution() -> None:
    posterior = {
        "hypotheses": {
            "field_change|action:2|field:x": {
                "id": "field_change|action:2|field:x",
                "family": "field_change",
                "action": "2",
                "selector": {"field": "x"},
                "transform": {"kind": "change"},
                "goal_test": {"kind": "field_changed"},
                "posterior": 0.60,
                "support": 3,
                "counterexamples": 0,
                "failed_predictions": 0,
                "description_length": 1.85,
            },
            "no_op|action:2": {
                "id": "no_op|action:2",
                "family": "no_op",
                "action": "2",
                "selector": {"scope": "observation"},
                "transform": {"kind": "identity"},
                "goal_test": {"kind": "no_visible_change"},
                "posterior": 0.30,
                "support": 1,
                "counterexamples": 1,
                "failed_predictions": 0,
                "description_length": 1.0,
            },
            "field_change|action:1|field:x": {
                "id": "field_change|action:1|field:x",
                "family": "field_change",
                "action": "1",
                "selector": {"field": "x"},
                "transform": {"kind": "change"},
                "goal_test": {"kind": "field_changed"},
                "posterior": 0.10,
                "support": 1,
                "counterexamples": 0,
            },
        }
    }

    mixture = posterior_predictive_mixture(posterior, action=2)

    assert mixture["schema"] == "runtime_posterior_predictive_causal_mixture_v1"
    assert mixture["prediction_type"] == "causal_edit_program_mixture"
    assert mixture["pixel_prediction"] is False
    assert mixture["component_count"] == 2
    assert mixture["action_posterior_mass"] == pytest.approx(0.90)
    assert mixture["field_change_probability"]["x"] == pytest.approx(2.0 / 3.0)
    assert mixture["no_op_probability"] == pytest.approx(1.0 / 3.0)
    assert mixture["expected_change_probability"] == pytest.approx(2.0 / 3.0)
    assert mixture["family_mixture"]["field_change"] == pytest.approx(2.0 / 3.0)
    assert mixture["family_mixture"]["no_op"] == pytest.approx(1.0 / 3.0)
    assert sum(component["mixture_weight"] for component in mixture["components"]) == pytest.approx(1.0)


def test_hypotheses_store_typed_program_fields() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = {"sensory": torch.tensor([1.0]), "private": torch.tensor([2])}
    after = {"sensory": torch.tensor([2.0]), "private": torch.tensor([2])}

    memory.append_event(
        tick=1,
        observation=before,
        action=3,
        next_observation=after,
    )
    memory.append_event(
        tick=2,
        observation=after,
        action=3,
        next_observation=after,
    )

    for hypothesis in memory.hypothesis_posterior["hypotheses"].values():
        assert "selector" in hypothesis
        assert "transform" in hypothesis
        assert "goal_test" in hypothesis
        assert isinstance(hypothesis["selector"], dict)
        assert isinstance(hypothesis["transform"], dict)
        assert isinstance(hypothesis["goal_test"], dict)


def test_neural_program_proposal_creates_verifiable_hypothesis() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")

    memory.append_event(
        tick=1,
        observation={"x": torch.tensor([1.0])},
        action=5,
        next_observation={"x": torch.tensor([2.0])},
        prediction_error={
            "neural_program_proposals": [
                {
                    "schema": "neural_causal_program_proposal_v1",
                    "source": "neural_proposal",
                    "family": "field_change",
                    "action": "ignored",
                    "selector": {"field": "x"},
                    "transform": {"kind": "change", "source": "neural_program_head"},
                    "goal_test": {"kind": "field_changed"},
                    "confidence": 0.75,
                }
            ]
        },
    )

    hypothesis = memory.hypothesis_posterior["hypotheses"]["field_change|action:5|field:x"]
    assert hypothesis["support"] == 1
    assert hypothesis["counterexamples"] == 0
    assert hypothesis["neural_proposal_count"] == 1
    assert hypothesis["proposal_confidence"] == 0.75
    assert "neural_proposal" in hypothesis["proposal_sources"]


def test_bad_neural_program_proposal_is_counterexampled_by_event_log() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")

    memory.append_event(
        tick=1,
        observation={"x": torch.tensor([1.0])},
        action=6,
        next_observation={"x": torch.tensor([1.0])},
        prediction_error={
            "neural_program_proposals": [
                {
                    "schema": "neural_causal_program_proposal_v1",
                    "source": "neural_proposal",
                    "family": "field_change",
                    "action": "6",
                    "selector": {"field": "x"},
                    "transform": {"kind": "change", "source": "neural_program_head"},
                    "goal_test": {"kind": "field_changed"},
                    "confidence": 0.9,
                }
            ]
        },
    )

    hypothesis = memory.hypothesis_posterior["hypotheses"]["field_change|action:6|field:x"]
    assert hypothesis["support"] == 0
    assert hypothesis["counterexamples"] == 1
    assert hypothesis["prediction_loss"] >= 1.0
    assert hypothesis["neural_proposal_count"] == 1


def test_hypothesis_posterior_round_trip_saves_and_loads(tmp_path: Path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    memory.append_event(
        tick=1,
        observation={"sensory": torch.tensor([1.0])},
        action=4,
        next_observation={"sensory": torch.tensor([2.0])},
    )
    memory.append_event(
        tick=2,
        observation={"sensory": torch.tensor([2.0])},
        action=4,
        next_observation={"sensory": torch.tensor([3.0])},
    )

    path = tmp_path / "runtime_memory.pt"
    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=4, batch_size=1, device="cpu")

    assert loaded.hypothesis_posterior == memory.hypothesis_posterior


def test_run_unbroken_with_memory_file_round_trips_hypothesis_posterior(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runtime.pt"
    memory_file = tmp_path / "runtime_memory.pt"

    train_model("smoke", output=checkpoint, steps=1, device="cpu")
    stats = run_unbroken(checkpoint, max_ticks=4, log_every=0, device="cpu", memory_file=memory_file)
    payload = torch.load(memory_file, map_location="cpu")
    posterior = payload["hypothesis_posterior"]

    assert stats["event_journal_length"] == 4.0
    assert posterior["schema"] == "runtime_causal_hypothesis_posterior_v1"
    assert posterior["updates"] == 4
    assert posterior["hypotheses"]
