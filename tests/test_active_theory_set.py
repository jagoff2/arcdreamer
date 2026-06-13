from __future__ import annotations

from pathlib import Path

import torch

from src.persistent_memory import PersistentMemoryState, maintain_active_theory_set


def _fake_hypothesis(
    hypothesis_id: str,
    *,
    family: str,
    action: str,
    posterior: float,
    support: int,
    selector: dict[str, object] | None = None,
    transform: dict[str, object] | None = None,
    goal_test: dict[str, object] | None = None,
    counterexamples: int = 0,
    description_length: float = 1.0,
) -> dict[str, object]:
    if selector is None:
        selector = {}
    if transform is None:
        transform = {}
    if goal_test is None:
        goal_test = {}
    return {
        "id": hypothesis_id,
        "family": family,
        "action": action,
        "selector": selector,
        "transform": transform,
        "goal_test": goal_test,
        "description_length": float(description_length),
        "support": int(support),
        "counterexamples": int(counterexamples),
        "prediction_loss": 0.0,
        "score": 0.0,
        "posterior": float(posterior),
        "context_confidence": 1.0,
    }


def _many_ranked_hypotheses() -> dict[str, dict[str, object]]:
    hypotheses: dict[str, dict[str, object]] = {}
    for family_index in range(10):
        family = f"family-{family_index:02d}"
        for tier in range(3):
            hypothesis_id = f"{family}|action:scan|tier:{tier}"
            hypotheses[hypothesis_id] = _fake_hypothesis(
                hypothesis_id=hypothesis_id,
                family=family,
                action="scan",
                posterior=1.0 - 0.01 * family_index - 0.10 * tier,
                support=10 + (2 - tier),
                counterexamples=tier,
            )
    return hypotheses


def test_fresh_memory_exposes_active_theory_set_schema() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    active = memory.active_theory_set

    assert active["schema"] == "runtime_active_theory_set_v1"
    assert active["updates"] == 0
    assert active["min_candidates"] == 3
    assert active["max_candidates"] == 20
    assert active["theories"] == []


def test_maintain_active_theory_set_with_fewer_than_min_candidates_keeps_all() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    memory.hypothesis_posterior["hypotheses"] = {
        "field_change|action:2|field:x": _fake_hypothesis(
            "field_change|action:2|field:x",
            family="field_change",
            action="2",
            posterior=0.7,
            support=4,
            description_length=1.6,
        ),
        "field_stable|action:2|field:y": _fake_hypothesis(
            "field_stable|action:2|field:y",
            family="field_stable",
            action="2",
            posterior=0.2,
            support=1,
            description_length=1.4,
        ),
    }
    maintain_active_theory_set(memory.active_theory_set, memory.hypothesis_posterior)

    theories = memory.active_theory_set["theories"]
    coverage = memory.active_theory_set["coverage"]

    assert coverage["available_hypotheses"] == 2
    assert coverage["active_count"] == 2
    assert coverage["bounded"] is False
    assert len(theories) == 2
    assert {theory["id"] for theory in theories} == set(memory.hypothesis_posterior["hypotheses"])
    assert set(coverage["families"]) == {"field_change", "field_stable"}
    assert memory.active_theory_set["updates"] == 1


def test_maintain_active_theory_set_prioritizes_high_value_hypotheses_and_preserves_families() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    memory.hypothesis_posterior["hypotheses"] = _many_ranked_hypotheses()
    maintain_active_theory_set(memory.active_theory_set, memory.hypothesis_posterior)

    active = memory.active_theory_set
    theories = active["theories"]
    coverage = active["coverage"]

    assert coverage["available_hypotheses"] == 30
    assert coverage["active_count"] == 20
    assert len(theories) == 20
    assert coverage["bounded"] is True
    assert set(coverage["families"]) == {f"family-{idx:02d}" for idx in range(10)}
    assert coverage["active_count"] >= active["min_candidates"] == 3

    selected_ids = {item["id"] for item in theories}
    for idx in range(10):
        family = f"family-{idx:02d}"
        assert f"{family}|action:scan|tier:0" in selected_ids
        assert f"{family}|action:scan|tier:1" in selected_ids
        assert f"{family}|action:scan|tier:2" not in selected_ids


def test_append_event_updates_active_theory_set_after_posterior_update() -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=4, batch_size=1, device="cpu")
    memory.hypothesis_posterior["hypotheses"] = {
        "field_change|action:3|field:x": _fake_hypothesis(
            "field_change|action:3|field:x",
            family="field_change",
            action="3",
            posterior=0.8,
            support=1,
            selector={"field": "x"},
            description_length=1.6,
        ),
        "field_change|action:3|field:y": _fake_hypothesis(
            "field_change|action:3|field:y",
            family="field_change",
            action="3",
            posterior=0.2,
            support=1,
            selector={"field": "y"},
            description_length=1.6,
        ),
    }
    maintain_active_theory_set(memory.active_theory_set, memory.hypothesis_posterior)

    before_updates = memory.active_theory_set["updates"]
    before_x_support = memory.hypothesis_posterior["hypotheses"]["field_change|action:3|field:x"]["support"]
    before_y_counterexamples = memory.hypothesis_posterior["hypotheses"][
        "field_change|action:3|field:y"
    ]["counterexamples"]

    memory.append_event(
        tick=1,
        observation={"x": torch.tensor([0.0]), "y": torch.tensor([1.0])},
        action=3,
        next_observation={"x": torch.tensor([1.0]), "y": torch.tensor([1.0])},
    )

    hypotheses = memory.hypothesis_posterior["hypotheses"]
    active = {item["id"]: item for item in memory.active_theory_set["theories"]}

    assert hypotheses["field_change|action:3|field:x"]["support"] == before_x_support + 1
    assert hypotheses["field_change|action:3|field:y"]["counterexamples"] == before_y_counterexamples + 1
    assert memory.active_theory_set["updates"] == before_updates + 1
    assert "field_change|action:3|field:x" in active
    assert active["field_change|action:3|field:x"]["support"] == hypotheses["field_change|action:3|field:x"]["support"]
    assert "field_change|action:3|field:y" in active
    assert (
        active["field_change|action:3|field:y"]["counterexamples"]
        == hypotheses["field_change|action:3|field:y"]["counterexamples"]
    )


def test_active_theory_set_round_trips_and_does_not_delete_hypotheses(tmp_path: Path) -> None:
    memory = PersistentMemoryState.fresh(hidden_dim=8, batch_size=1, device="cpu")
    memory.hypothesis_posterior["hypotheses"] = _many_ranked_hypotheses()
    maintain_active_theory_set(memory.active_theory_set, memory.hypothesis_posterior)

    path = tmp_path / "active_theory_set_memory.pt"
    memory.save(path)
    loaded = PersistentMemoryState.load(path, hidden_dim=8, batch_size=1, device="cpu")

    assert loaded.active_theory_set == memory.active_theory_set
    assert loaded.hypothesis_posterior["hypotheses"].keys() == memory.hypothesis_posterior["hypotheses"].keys()
    assert len(loaded.hypothesis_posterior["hypotheses"]) == len(memory.hypothesis_posterior["hypotheses"])
    assert len(memory.hypothesis_posterior["hypotheses"]) > loaded.active_theory_set["coverage"]["active_count"]
