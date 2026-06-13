from __future__ import annotations

import torch
from pathlib import Path
from typing import Any, Iterable, Tuple

from src.run_unbroken import run_unbroken
from src.train import train_model


def _iter_metadata_nodes(metadata: Any, path: str = "") -> Iterable[Tuple[str, Any]]:
    if isinstance(metadata, dict):
        for key, value in metadata.items():
            nested_path = f"{path}.{key}" if path else str(key)
            yield nested_path, value
            yield from _iter_metadata_nodes(value, nested_path)
    elif isinstance(metadata, (list, tuple)):
        for index, value in enumerate(metadata):
            yield f"{path}[{index}]", value
            yield from _iter_metadata_nodes(value, f"{path}[{index}]")


def _is_external_action_like_key(key: str) -> bool:
    lowered = key.lower()
    return (
        "external_action" in lowered
        or ("executed" in lowered and "action" in lowered)
    )


def _assert_single_executed_external_action_field(event: dict[str, Any]) -> None:
    metadata = event["metadata"]
    assert isinstance(metadata, dict)
    external_action_nodes = [
        (path, value)
        for path, value in _iter_metadata_nodes(metadata)
        if _is_external_action_like_key(path.split(".")[-1].split("[")[0])
    ]

    executed_candidates = [
        (path, value)
        for path, value in external_action_nodes
        if not path.endswith("external_action_index")
    ]

    assert len(executed_candidates) <= 1
    for path, value in executed_candidates:
        assert not isinstance(value, (list, tuple, dict)), path


def test_run_unbroken_counts_external_actions_and_journal_length(tmp_path: Path) -> None:
    checkpoint = tmp_path / "smoke_unbroken_checkpoint.pt"
    train_model("smoke", output=checkpoint, steps=1, device="cpu")
    memory_file = tmp_path / "smoke_unbroken_memory.pt"
    tick_count = 7

    stats = run_unbroken(checkpoint, max_ticks=tick_count, memory_file=memory_file, device="cpu", log_every=0)

    assert stats["external_action_count"] == float(tick_count)
    assert stats["event_journal_length"] == float(tick_count)


def test_run_unbroken_persists_external_action_metadata_and_single_action_record(tmp_path: Path) -> None:
    checkpoint = tmp_path / "smoke_unbroken_checkpoint.pt"
    train_model("smoke", output=checkpoint, steps=1, device="cpu")
    memory_file = tmp_path / "smoke_unbroken_memory.pt"
    tick_count = 9

    run_unbroken(checkpoint, max_ticks=tick_count, memory_file=memory_file, device="cpu", log_every=0)
    payload = torch.load(memory_file, map_location="cpu")
    events = payload["event_journal"]

    assert len(events) == tick_count
    for tick, event in enumerate(events, start=1):
        metadata = event["metadata"]
        assert metadata["external_action_index"] == tick
        assert event["action"] == str(metadata["action_selection"]["selected_action"])
        _assert_single_executed_external_action_field(event)
