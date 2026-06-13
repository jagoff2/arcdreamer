from __future__ import annotations

import math
from pathlib import Path

import torch

from src.persistent_memory import PersistentMemoryState
from src.run_unbroken import run_unbroken
from src.train import train_model


def _candidates(posterior: dict[str, object], scope: str, kind: str) -> list[dict[str, object]]:
    source = posterior["goals"] if scope == "goal" else posterior["subgoals"]
    return [entry for entry in source.values() if entry["kind"] == kind]


def test_fresh_state_has_goal_posterior_schema_and_empty_fields() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    posterior = memory.goal_posterior

    assert posterior["schema"] == "runtime_goal_posterior_v1"
    assert posterior["updates"] == 0
    assert posterior["goals"] == {}
    assert posterior["subgoals"] == {}
    assert posterior["posterior_mass"] == 0.0
    assert posterior["top_goals"] == []


def test_append_event_generates_score_terminal_and_event_goal_candidates() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    observation = {"sensory": torch.tensor([0.0])}
    next_observation = {"sensory": torch.tensor([1.0])}
    metadata = {
        "score_delta": 1.5,
        "terminal": True,
        "events": ["goal_reached", "opened gate"],
    }

    memory.append_event(
        tick=1,
        observation=observation,
        action=2,
        next_observation=next_observation,
        metadata=metadata,
    )

    goals = memory.goal_posterior["goals"]
    subgoals = memory.goal_posterior["subgoals"]

    assert len(_candidates(memory.goal_posterior, "goal", "positive_reward")) == 1
    assert len(_candidates(memory.goal_posterior, "goal", "terminal_win")) == 1
    assert len(_candidates(memory.goal_posterior, "goal", "event")) == 2
    assert _candidates(memory.goal_posterior, "subgoal", "gate_or_counter")[0]["selector"]["event"] == "opened gate"
    assert len(subgoals) == 1


def test_append_event_generates_disappearance_clearing_and_movement_subgoal_candidates() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = {"grid": torch.tensor([[1, 0], [0, 0]], dtype=torch.long)}
    after = {"grid": torch.tensor([[0, 0], [0, 0]], dtype=torch.long)}

    memory.append_event(
        tick=1,
        observation=before,
        action=0,
        next_observation=after,
        metadata={"score_delta": 1.0, "terminal": False},
    )

    assert len(_candidates(memory.goal_posterior, "goal", "object_disappears")) == 1
    assert len(_candidates(memory.goal_posterior, "goal", "clearing")) == 1


def test_append_event_generates_movement_and_target_changed_subgoals() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = {"grid": torch.tensor([[1, 0], [0, 0]], dtype=torch.long)}
    after = {"grid": torch.tensor([[0, 1], [0, 0]], dtype=torch.long)}

    memory.append_event(
        tick=1,
        observation=before,
        action=0,
        next_observation=after,
        metadata={"score_delta": 1.0, "terminal": False},
    )

    assert len(_candidates(memory.goal_posterior, "subgoal", "movement_subgoal")) == 1
    assert len(_candidates(memory.goal_posterior, "subgoal", "target_changed_cell")) == 1


def test_append_event_generates_alignment_matching_and_filling_from_relation_graph() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = {"grid": torch.tensor([[1, 0, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.long)}
    after = {"grid": torch.tensor([[1, 2, 1], [0, 0, 0], [0, 0, 0]], dtype=torch.long)}

    memory.append_event(
        tick=1,
        observation=before,
        action=1,
        next_observation=after,
        metadata={"score_delta": 0.0, "terminal": False},
    )

    assert len(_candidates(memory.goal_posterior, "goal", "alignment")) == 1
    assert len(_candidates(memory.goal_posterior, "goal", "matching")) == 1
    assert len(_candidates(memory.goal_posterior, "goal", "filling")) == 1
    assert len(_candidates(memory.goal_posterior, "subgoal", "target_changed_cell")) == 1
    goals = memory.goal_posterior["goals"]
    assert math.isclose(sum(goal["posterior"] for goal in goals.values()), 1.0, rel_tol=1.0e-12)


def test_append_event_generates_avoid_no_op_subgoal_when_unchanged() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = {"grid": torch.tensor([[0, 1], [0, 0]], dtype=torch.long)}

    memory.append_event(
        tick=1,
        observation=before,
        action=3,
        next_observation=before,
        metadata={"score_delta": 0.0, "terminal": False},
    )

    assert len(_candidates(memory.goal_posterior, "subgoal", "avoid_no_op")) == 1
    assert memory.goal_posterior["goals"] == {}
    assert memory.goal_posterior["updates"] == 1


def test_repeated_positive_progress_increases_support_and_posterior() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    first = {"sensory": torch.tensor([0.0])}
    second = {"sensory": torch.tensor([1.0])}
    third = {"sensory": torch.tensor([2.0])}

    memory.append_event(
        tick=1,
        observation=first,
        action=0,
        next_observation=second,
        metadata={"score_delta": 1.0, "terminal": True},
    )
    before_posterior = memory.goal_posterior["goals"]["goal|positive_reward"]["posterior"]

    memory.append_event(
        tick=2,
        observation=second,
        action=0,
        next_observation=third,
        metadata={"score_delta": 1.0, "terminal": False},
    )

    positive = memory.goal_posterior["goals"]["goal|positive_reward"]
    terminal = memory.goal_posterior["goals"]["goal|terminal_win"]

    assert positive["support"] == 2
    assert terminal["support"] == 1
    assert positive["posterior"] > before_posterior
    assert math.isclose(
        memory.goal_posterior["posterior_mass"],
        sum(goal["posterior"] for goal in memory.goal_posterior["goals"].values()),
        rel_tol=1.0e-12,
    )


def test_inconsistent_no_progress_increases_inconsistency_and_reduces_posterior() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    base = {"sensory": torch.tensor([0.0])}
    mid = {"sensory": torch.tensor([1.0])}

    memory.append_event(
        tick=1,
        observation=base,
        action=0,
        next_observation=mid,
        metadata={"score_delta": 1.0, "events": ["alpha"]},
    )
    memory.append_event(
        tick=2,
        observation=mid,
        action=0,
        next_observation=mid,
        metadata={"score_delta": 1.0, "events": ["alpha"]},
    )
    before = memory.goal_posterior["goals"]["goal|positive_reward"]["posterior"]

    memory.append_event(
        tick=3,
        observation=mid,
        action=1,
        next_observation=mid,
        metadata={"score_delta": 0.0, "events": ["beta"]},
    )

    positive = memory.goal_posterior["goals"]["goal|positive_reward"]
    alpha_event = memory.goal_posterior["goals"]["goal|event|event:alpha"]
    beta_event = memory.goal_posterior["goals"]["goal|event|event:beta"]

    assert positive["support"] == 2
    assert positive["inconsistency"] == 1
    assert alpha_event["support"] == 2
    assert alpha_event["inconsistency"] == 1
    assert beta_event["support"] == 1
    assert beta_event["inconsistency"] == 0
    assert positive["posterior"] < before
    assert math.isclose(memory.goal_posterior["posterior_mass"], 1.0, rel_tol=1.0e-12)


def test_goal_description_length_penalty_affects_relative_posterior() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    before = {"sensory": torch.tensor([0.0])}
    mid = {"sensory": torch.tensor([1.0])}

    memory.append_event(
        tick=1,
        observation=before,
        action=0,
        next_observation=mid,
        metadata={"score_delta": 1.0, "events": ["progress"]},
    )
    memory.append_event(
        tick=2,
        observation=mid,
        action=0,
        next_observation=mid,
        metadata={"score_delta": 1.0, "events": ["progress"]},
    )

    positive = memory.goal_posterior["goals"]["goal|positive_reward"]
    progress_event = memory.goal_posterior["goals"]["goal|event|event:progress"]

    assert positive["support"] == 2
    assert progress_event["support"] == 2
    assert positive["description_length"] < progress_event["description_length"]
    assert positive["posterior"] > progress_event["posterior"]
    assert math.isclose(sum(goal["posterior"] for goal in memory.goal_posterior["goals"].values()), 1.0, rel_tol=1.0e-12)


def test_goal_posterior_round_trip_via_memory_save_load(tmp_path: Path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    observation = {"sensory": torch.tensor([0.5])}
    next_observation = {"sensory": torch.tensor([1.0])}

    memory.append_event(
        tick=1,
        observation=observation,
        action=1,
        next_observation=next_observation,
        metadata={"score_delta": 1.0, "terminal": False},
    )
    memory.append_event(
        tick=2,
        observation=next_observation,
        action=1,
        next_observation=next_observation,
        metadata={"score_delta": 0.0},
    )

    path = tmp_path / "runtime_memory.pt"
    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=4, batch_size=1, device="cpu")

    assert loaded.goal_posterior == memory.goal_posterior


def test_run_unbroken_roundtrips_goal_posterior_and_updates(tmp_path: Path) -> None:
    checkpoint = tmp_path / "runtime.ckpt"
    memory_file = tmp_path / "runtime_memory.pt"
    train_model("smoke", output=checkpoint, steps=1, device="cpu")

    max_ticks = 4
    stats = run_unbroken(checkpoint, max_ticks=max_ticks, log_every=0, device="cpu", memory_file=memory_file)
    payload = torch.load(memory_file, map_location="cpu")
    posterior = payload["goal_posterior"]

    assert stats["unbroken_ticks"] == float(max_ticks)
    assert stats["event_journal_length"] == float(max_ticks)
    assert posterior["schema"] == "runtime_goal_posterior_v1"
    assert posterior["updates"] == max_ticks
