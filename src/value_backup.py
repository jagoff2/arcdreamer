from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping


def fresh_progress_value_model() -> dict[str, Any]:
    return {
        "schema": "runtime_progress_value_model_v1",
        "updates": 0,
        "backups": 0,
        "discount": 0.85,
        "backup_window": 8,
        "action_values": {},
        "sequence_values": {},
        "recent_progress": [],
        "top_actions": [],
        "top_sequences": [],
    }


def _score_delta(event: Mapping[str, Any]) -> float:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return 0.0
    try:
        return float(metadata.get("score_delta", 0.0) or 0.0)
    except (TypeError, ValueError):
        return 0.0


def _metadata_text(metadata: Mapping[str, Any]) -> str:
    values: list[str] = []
    for key in ("event", "events", "status", "phase", "boundary"):
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values.extend(str(item).lower() for item in value)
        else:
            values.append(str(value).lower())
    return " ".join(values)


def _progress_signal(event: Mapping[str, Any]) -> tuple[str | None, float]:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    score = _score_delta(event)
    if score > 0.0:
        return "score_delta", score
    text = _metadata_text(metadata)
    if "win" in text or "goal" in text or "progress" in text or "level" in text:
        return "event_progress", 1.0
    if bool(metadata.get("terminal", False)) and str(metadata.get("boundary", "")).lower() in {"win", "level", "level_transition"}:
        return "terminal_progress", 1.0
    return None, 0.0


def _sequence_id(actions: list[str]) -> str:
    payload = json.dumps(actions, separators=(",", ":"))
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"value_sequence|{digest}"


def _recent_events(event_journal: list[dict[str, Any]], window: int) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for event in reversed(event_journal):
        selected.append(event)
        metadata = event.get("metadata", {})
        if len(selected) >= int(window):
            break
        if len(selected) > 1 and isinstance(metadata, Mapping) and metadata.get("boundary"):
            break
    selected.reverse()
    return selected


def _update_average(entry: dict[str, Any], value: float) -> None:
    support = int(entry.get("support", 0)) + 1
    total = float(entry.get("value_sum", 0.0)) + float(value)
    entry["support"] = support
    entry["value_sum"] = total
    entry["mean_value"] = float(total / max(support, 1))
    entry["confidence"] = float(min(1.0, support / 3.0) * max(entry["mean_value"], 0.0))


def _normalize(model: dict[str, Any]) -> None:
    actions = sorted(
        model.get("action_values", {}).values(),
        key=lambda item: (float(item.get("confidence", 0.0)), float(item.get("mean_value", 0.0)), int(item.get("support", 0))),
        reverse=True,
    )
    model["top_actions"] = [
        {
            "action": item["action"],
            "support": int(item.get("support", 0)),
            "mean_value": float(item.get("mean_value", 0.0)),
            "confidence": float(item.get("confidence", 0.0)),
        }
        for item in actions[:8]
    ]
    sequences = sorted(
        model.get("sequence_values", {}).values(),
        key=lambda item: (float(item.get("confidence", 0.0)), float(item.get("mean_value", 0.0)), int(item.get("support", 0))),
        reverse=True,
    )
    model["top_sequences"] = [
        {
            "id": item["id"],
            "actions": list(item.get("actions", [])),
            "support": int(item.get("support", 0)),
            "mean_value": float(item.get("mean_value", 0.0)),
            "confidence": float(item.get("confidence", 0.0)),
        }
        for item in sequences[:8]
    ]


def update_progress_value_model(
    model: dict[str, Any],
    *,
    event_journal: list[dict[str, Any]],
    event: Mapping[str, Any],
) -> dict[str, Any]:
    if not model:
        model.update(fresh_progress_value_model())
    model["updates"] = int(model.get("updates", 0)) + 1
    signal, magnitude = _progress_signal(event)
    if signal is None:
        _normalize(model)
        return model

    window = int(model.get("backup_window", 8))
    discount = float(model.get("discount", 0.85))
    recent = _recent_events(event_journal, window)
    actions = [str(item.get("action")) for item in recent if "action" in item]
    if not actions:
        _normalize(model)
        return model

    tick = int(event.get("tick", 0))
    model["backups"] = int(model.get("backups", 0)) + 1
    action_values = model.setdefault("action_values", {})
    for age, action in enumerate(reversed(actions)):
        backed_value = float(magnitude) * (discount ** age)
        entry = action_values.setdefault(
            action,
            {
                "action": action,
                "support": 0,
                "value_sum": 0.0,
                "mean_value": 0.0,
                "confidence": 0.0,
                "evidence_ticks": [],
                "signals": {},
            },
        )
        _update_average(entry, backed_value)
        ticks = entry.setdefault("evidence_ticks", [])
        if tick not in ticks:
            ticks.append(tick)
            ticks.sort()
        signals = entry.setdefault("signals", {})
        signals[signal] = int(signals.get(signal, 0)) + 1

    sequence_id = _sequence_id(actions)
    sequence_entry = model.setdefault("sequence_values", {}).setdefault(
        sequence_id,
        {
            "id": sequence_id,
            "actions": actions,
            "support": 0,
            "value_sum": 0.0,
            "mean_value": 0.0,
            "confidence": 0.0,
            "evidence_ticks": [],
            "signals": {},
        },
    )
    _update_average(sequence_entry, float(magnitude))
    ticks = sequence_entry.setdefault("evidence_ticks", [])
    if tick not in ticks:
        ticks.append(tick)
        ticks.sort()
    signals = sequence_entry.setdefault("signals", {})
    signals[signal] = int(signals.get(signal, 0)) + 1

    progress = model.setdefault("recent_progress", [])
    progress.append(
        {
            "tick": tick,
            "signal": signal,
            "magnitude": float(magnitude),
            "actions": actions,
            "sequence_id": sequence_id,
        }
    )
    if len(progress) > window:
        del progress[: len(progress) - window]
    _normalize(model)
    return model
