from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn.functional as F

from .adaptation import fast_adaptation_contract
from .causal_hypotheses import event_hypothesis_trace, fresh_hypothesis_posterior, update_hypothesis_posterior
from .controllability import fresh_controllability_model, update_controllability_model
from .coordinate_affordance import fresh_coordinate_affordances, update_runtime_coordinate_affordances
from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .env import NUM_ACTIONS, PRIVATE_NONE
from .goal_inference import fresh_goal_posterior, goal_candidates_from_event, update_goal_posterior
from .grid_perception import parse_grid
from .value_backup import fresh_progress_value_model, update_progress_value_model


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, torch.Tensor):
        return _tensor_snapshot(value)
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    return repr(value)


def _tensor_snapshot(value: torch.Tensor) -> dict[str, Any]:
    tensor = value.detach().cpu()
    return {
        "dtype": str(tensor.dtype).replace("torch.", ""),
        "shape": [int(item) for item in tensor.shape],
        "values": tensor.tolist(),
    }


def observation_snapshot(observation: Mapping[str, Any]) -> dict[str, Any]:
    return {str(key): _json_safe(value) for key, value in observation.items()}


def observation_delta(before: Mapping[str, Any], after: Mapping[str, Any]) -> dict[str, Any]:
    delta: dict[str, Any] = {}
    for key in sorted(set(before) | set(after)):
        left = before.get(key)
        right = after.get(key)
        if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor) and tuple(left.shape) == tuple(right.shape):
            left_cpu = left.detach().cpu()
            right_cpu = right.detach().cpu()
            if torch.is_floating_point(left_cpu) or torch.is_floating_point(right_cpu):
                diff = right_cpu.float() - left_cpu.float()
            else:
                diff = right_cpu.long() - left_cpu.long()
            changed = diff != 0
            delta[str(key)] = {
                "dtype": str(diff.dtype).replace("torch.", ""),
                "shape": [int(item) for item in diff.shape],
                "values": diff.tolist(),
                "changed_count": int(changed.sum().item()),
                "l1": float(diff.float().abs().sum().item()),
                "max_abs": float(diff.float().abs().max().item()) if diff.numel() else 0.0,
            }
        else:
            delta[str(key)] = {
                "before": _json_safe(left),
                "after": _json_safe(right),
                "changed": _json_safe(left) != _json_safe(right),
            }
    return delta


def _grid_frame_from_observation(observation: Mapping[str, Any]) -> Any | None:
    explicit_grid = "grid" in observation
    value = observation.get("grid") if explicit_grid else observation.get("sensory")
    if isinstance(value, torch.Tensor):
        tensor = value.detach().cpu()
        if tensor.ndim != 2:
            return None
        if explicit_grid or not torch.is_floating_point(tensor):
            return tensor
        return None
    if isinstance(value, (list, tuple)):
        try:
            tensor = torch.as_tensor(value)
        except (TypeError, ValueError):
            return None
        if tensor.ndim != 2:
            return None
        if explicit_grid or not torch.is_floating_point(tensor):
            return tensor
        return None
    return None


def frame_perception_payload(
    observation: Mapping[str, Any],
    next_observation: Mapping[str, Any],
    *,
    action: int | str,
    tick: int,
) -> dict[str, Any] | None:
    current_grid = _grid_frame_from_observation(observation)
    next_grid = _grid_frame_from_observation(next_observation)
    if current_grid is None and next_grid is None:
        return None

    payload: dict[str, Any] = {"schema": "runtime_frame_perception_v1"}
    if current_grid is not None:
        current_scene = parse_grid(current_grid, action=action, time_index=int(tick))
        payload["current_frame"] = current_scene.compact()
    if next_grid is not None:
        previous = current_grid if current_grid is not None else None
        next_scene = parse_grid(next_grid, previous_grid=previous, action=action, time_index=int(tick))
        payload["next_frame"] = next_scene.compact()
    return payload


def canonical_observation_hash(observation: Mapping[str, Any]) -> str:
    payload = json.dumps(observation_snapshot(observation), sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


def _fresh_transition_graph() -> dict[str, Any]:
    return {
        "schema": "runtime_transition_graph_v1",
        "nodes": {},
        "edges": {},
        "outgoing_edges": {},
        "outgoing_action_edges": {},
        "incoming_edges": {},
        "edge_index_count": 0,
        "visited_sequence": [],
        "last_node_hash": None,
        "last_edge_id": None,
        "observed_actions": {},
        "unexplored_actions": {},
        "progress_targets": {},
        "reversible_pairs": {},
        "undo_actions": {},
        "no_op_edge_count": 0,
        "loop_edge_count": 0,
        "reversible_edge_count": 0,
        "undo_edge_count": 0,
    }


def _fresh_semantic_memory() -> dict[str, Any]:
    return {
        "schema": "runtime_semantic_memory_v1",
        "action_facts": {},
        "invariants": {},
        "counterexamples": {},
        "affordances": {},
        "object_facts": {},
        "relation_facts": {},
        "click_facts": {},
        "obstacle_facts": {},
        "goal_facts": {},
        "procedural_facts": {},
        "facts": {},
    }


def _fresh_self_supervised_adapter(
    input_dim: int = 24,
    delta_dim: int = 8,
    action_buckets: int = NUM_ACTIONS,
    history_capacity: int = 64,
) -> dict[str, Any]:
    return {
        "schema": "runtime_self_supervised_adapter_v1",
        "input_dim": int(input_dim),
        "delta_dim": int(delta_dim),
        "action_buckets": int(action_buckets),
        "lr": 0.05,
        "updates": 0,
        "history_capacity": int(history_capacity),
        "heads": {
            "next_delta": {"weight": [], "bias": []},
            "inverse_action": {"weight": [], "bias": []},
            "no_op_change": {"weight": [], "bias": []},
            "object_persistence": {"weight": [], "bias": []},
        },
        "loss_history": [],
        "latest_losses": {},
        "latest_metrics": {},
    }


def _fresh_plastic_memory(capacity: int = 64, adapter_rank: int = 4) -> dict[str, Any]:
    return {
        "schema": "runtime_plastic_memory_v1",
        "capacity": int(capacity),
        "updates": 0,
        "key_values": [],
        "hebbian_traces": {},
        "low_rank_adapter": {
            "rank": int(adapter_rank),
            "action_factors": {},
            "field_factors": {},
            "norm": 0.0,
        },
        "self_supervised_adapter": _fresh_self_supervised_adapter(history_capacity=capacity),
    }


def fresh_context_state() -> dict[str, Any]:
    return {
        "schema": "runtime_context_state_v1",
        "game_id": None,
        "level_index": 0,
        "episode_index": 0,
        "phase": "running",
        "uncertainty": 1.0,
        "confidence_gates": {
            "boundary_count": 0,
            "fragile_hypothesis_decay": 1.0,
            "fragile_goal_decay": 1.0,
            "verified_min_support": 2,
        },
        "boundaries": [],
        "last_terminal": False,
        "last_boundary": None,
    }


def fresh_macro_policy_library() -> dict[str, Any]:
    return {
        "schema": "runtime_macro_policy_library_v1",
        "macros": {},
        "updates": 0,
        "consolidations": 0,
        "last_consolidated_tick": None,
        "top_macros": [],
    }


def fresh_active_theory_set() -> dict[str, Any]:
    return {
        "schema": "runtime_active_theory_set_v1",
        "updates": 0,
        "min_candidates": 3,
        "max_candidates": 20,
        "theories": [],
        "coverage": {
            "available_hypotheses": 0,
            "active_count": 0,
            "families": [],
            "bounded": False,
        },
    }


def _available_actions(metadata: Mapping[str, Any] | None) -> list[str]:
    if not metadata:
        return []
    if "available_actions" in metadata:
        return [str(item) for item in metadata.get("available_actions", [])]
    mask = metadata.get("available_action_mask")
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().tolist()
    if isinstance(mask, (list, tuple)):
        return [str(index) for index, enabled in enumerate(mask) if bool(enabled)]
    return []


def _is_undo_action(action_text: str) -> bool:
    return action_text.lower() == "undo" or action_text == "7"


def _ensure_transition_graph_fields(graph: dict[str, Any]) -> None:
    graph.setdefault("nodes", {})
    graph.setdefault("edges", {})
    graph.setdefault("outgoing_edges", {})
    graph.setdefault("outgoing_action_edges", {})
    graph.setdefault("incoming_edges", {})
    graph.setdefault("edge_index_count", 0)
    graph.setdefault("visited_sequence", [])
    graph.setdefault("last_node_hash", None)
    graph.setdefault("last_edge_id", None)
    graph.setdefault("observed_actions", {})
    graph.setdefault("unexplored_actions", {})
    graph.setdefault("progress_targets", {})
    graph.setdefault("reversible_pairs", {})
    graph.setdefault("undo_actions", {})
    graph.setdefault("no_op_edge_count", 0)
    graph.setdefault("loop_edge_count", 0)
    graph.setdefault("reversible_edge_count", 0)
    graph.setdefault("undo_edge_count", 0)


def _record_undo_evidence(
    graph: dict[str, Any],
    *,
    undo_edge: Mapping[str, Any],
    reversed_edge: Mapping[str, Any],
    tick: int,
) -> None:
    action_text = str(undo_edge.get("action", ""))
    if not _is_undo_action(action_text):
        return

    undo_edge_id = str(undo_edge.get("id", ""))
    reversed_edge_id = str(reversed_edge.get("id", ""))
    if not undo_edge_id or not reversed_edge_id:
        return

    entry = graph.setdefault("undo_actions", {}).setdefault(
        action_text,
        {
            "action": action_text,
            "support": 0,
            "confidence": 0.0,
            "first_tick": int(tick),
            "last_tick": int(tick),
            "undo_edges": [],
            "reverses_edges": [],
            "state_pairs": [],
            "evidence": {},
        },
    )

    evidence_key = f"{undo_edge_id}<-{reversed_edge_id}"
    evidence = entry.setdefault("evidence", {})
    if evidence_key not in evidence:
        evidence[evidence_key] = {
            "undo_edge": undo_edge_id,
            "reverses_edge": reversed_edge_id,
            "from": str(undo_edge.get("from", "")),
            "to": str(undo_edge.get("to", "")),
            "first_tick": int(tick),
            "last_tick": int(tick),
            "observations": 0,
        }
        graph["undo_edge_count"] = int(graph.get("undo_edge_count", 0)) + 1

    evidence[evidence_key]["last_tick"] = int(tick)
    evidence[evidence_key]["observations"] = max(
        int(evidence[evidence_key].get("observations", 0)),
        int(undo_edge.get("count", 1)),
    )

    entry["last_tick"] = int(tick)
    entry["first_tick"] = min(int(entry.get("first_tick", tick)), int(tick))

    undo_edges = {str(item) for item in entry.get("undo_edges", [])}
    undo_edges.add(undo_edge_id)
    entry["undo_edges"] = sorted(undo_edges)

    reversed_edges = {str(item) for item in entry.get("reverses_edges", [])}
    reversed_edges.add(reversed_edge_id)
    entry["reverses_edges"] = sorted(reversed_edges)

    pair_key = (str(reversed_edge.get("from", "")), str(reversed_edge.get("to", "")))
    state_pairs = {
        (str(item.get("from", "")), str(item.get("to", "")))
        for item in entry.get("state_pairs", [])
        if isinstance(item, Mapping)
    }
    state_pairs.add(pair_key)
    entry["state_pairs"] = [{"from": left, "to": right} for left, right in sorted(state_pairs)]

    edges = graph.get("edges", {})
    support = 0
    for undo_id in entry["undo_edges"]:
        undo_record = edges.get(undo_id, {})
        support += max(int(undo_record.get("count", 1)), 1)
    entry["support"] = int(support)
    entry["confidence"] = float(min(0.99, support / max(support + 1.0, 1.0)))


def _ensure_node(
    graph: dict[str, Any],
    node_hash: str,
    observation: Mapping[str, Any],
    tick: int,
) -> dict[str, Any]:
    nodes = graph.setdefault("nodes", {})
    node = nodes.get(node_hash)
    if node is None:
        node = {
            "hash": node_hash,
            "observation": observation_snapshot(observation),
            "visits": 0,
            "first_tick": int(tick),
            "last_tick": int(tick),
        }
        nodes[node_hash] = node
    return node


def _record_node_visit(graph: dict[str, Any], node_hash: str, tick: int) -> None:
    node = graph["nodes"][node_hash]
    node["visits"] = int(node.get("visits", 0)) + 1
    node["last_tick"] = int(tick)
    graph.setdefault("visited_sequence", []).append(node_hash)
    graph["last_node_hash"] = node_hash


def _sync_unexplored_actions(graph: dict[str, Any], node_hash: str, available_actions: list[str]) -> None:
    observed = set(graph.setdefault("observed_actions", {}).setdefault(node_hash, []))
    if available_actions:
        unexplored = [action for action in available_actions if action not in observed]
        graph.setdefault("unexplored_actions", {})[node_hash] = unexplored


def _index_outgoing_edge(graph: dict[str, Any], from_hash: str, edge_id: str) -> None:
    outgoing = graph.setdefault("outgoing_edges", {}).setdefault(str(from_hash), [])
    if edge_id not in outgoing:
        outgoing.append(edge_id)
        outgoing.sort()


def _index_incoming_edge(graph: dict[str, Any], to_hash: str, edge_id: str) -> None:
    incoming = graph.setdefault("incoming_edges", {}).setdefault(str(to_hash), [])
    if edge_id not in incoming:
        incoming.append(edge_id)
        incoming.sort()


def _index_transition_edge(graph: dict[str, Any], edge: Mapping[str, Any]) -> None:
    edge_id = str(edge.get("id", ""))
    from_hash = str(edge.get("from", ""))
    to_hash = str(edge.get("to", ""))
    action_text = str(edge.get("action", ""))
    if not edge_id or not from_hash or not to_hash:
        return
    _index_outgoing_edge(graph, from_hash, edge_id)
    by_action = graph.setdefault("outgoing_action_edges", {}).setdefault(from_hash, {}).setdefault(action_text, [])
    if edge_id not in by_action:
        by_action.append(edge_id)
        by_action.sort()
    _index_incoming_edge(graph, to_hash, edge_id)
    graph["edge_index_count"] = len(graph.get("edges", {}))


def _rebuild_edge_indexes_if_needed(graph: dict[str, Any]) -> None:
    edges = graph.get("edges", {})
    if not edges:
        graph["edge_index_count"] = 0
        return
    if int(graph.get("edge_index_count", -1)) == len(edges):
        return
    graph["outgoing_edges"] = {}
    graph["outgoing_action_edges"] = {}
    graph["incoming_edges"] = {}
    for edge in edges.values():
        _index_transition_edge(graph, edge)
    graph["edge_index_count"] = len(edges)


def _update_progress_target(graph: dict[str, Any], edge: Mapping[str, Any]) -> None:
    count = max(float(edge.get("count", 1.0)), 1.0)
    score_delta = max(float(edge.get("score_delta_sum", 0.0)) / count, 0.0)
    terminal_rate = float(edge.get("terminal_count", 0.0)) / count
    value = float(score_delta + 1.25 * terminal_rate)
    target = str(edge.get("to", ""))
    edge_id = str(edge.get("id", ""))
    if not target or value <= 0.0:
        return
    targets = graph.setdefault("progress_targets", {})
    existing = targets.get(target, {})
    if not isinstance(existing, Mapping) or value >= float(existing.get("value", 0.0)):
        targets[target] = {
            "state": target,
            "value": value,
            "source_edge": edge_id,
            "score_delta": float(score_delta),
            "terminal_rate": float(terminal_rate),
            "last_tick": int(edge.get("last_tick", 0)),
        }


def update_transition_graph(
    graph: dict[str, Any],
    *,
    tick: int,
    observation: Mapping[str, Any],
    action: int | str,
    next_observation: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    delta: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not graph:
        graph.update(_fresh_transition_graph())
    _ensure_transition_graph_fields(graph)
    _rebuild_edge_indexes_if_needed(graph)
    action_text = str(action)
    from_hash = canonical_observation_hash(observation)
    to_hash = canonical_observation_hash(next_observation)
    _ensure_node(graph, from_hash, observation, tick)
    _ensure_node(graph, to_hash, next_observation, tick)

    if graph.get("last_node_hash") is None:
        _record_node_visit(graph, from_hash, tick)
    elif graph.get("last_node_hash") != from_hash:
        _record_node_visit(graph, from_hash, tick)

    loop_observed = int(graph["nodes"][to_hash].get("visits", 0)) > 0
    _record_node_visit(graph, to_hash, tick)

    available_actions = _available_actions(metadata)
    _sync_unexplored_actions(graph, from_hash, available_actions)
    observed_actions = graph.setdefault("observed_actions", {}).setdefault(from_hash, [])
    if action_text not in observed_actions:
        observed_actions.append(action_text)
        observed_actions.sort()
    _sync_unexplored_actions(graph, from_hash, available_actions)

    edge_id = f"{from_hash}|{action_text}|{to_hash}"
    edges = graph.setdefault("edges", {})
    edge = edges.get(edge_id)
    no_op = from_hash == to_hash
    if edge is None:
        edge = {
            "id": edge_id,
            "from": from_hash,
            "to": to_hash,
            "action": action_text,
            "count": 0,
            "first_tick": int(tick),
            "last_tick": int(tick),
            "no_op": bool(no_op),
            "loop_observed": bool(loop_observed),
            "reversible": False,
            "score_delta_sum": 0.0,
            "terminal_count": 0,
            "delta": _json_safe(delta or {}),
        }
        edges[edge_id] = edge
        _index_transition_edge(graph, edge)
        if no_op:
            graph["no_op_edge_count"] = int(graph.get("no_op_edge_count", 0)) + 1
        if loop_observed:
            graph["loop_edge_count"] = int(graph.get("loop_edge_count", 0)) + 1
    else:
        edge["loop_observed"] = bool(edge.get("loop_observed", False) or loop_observed)
    edge["count"] = int(edge.get("count", 0)) + 1
    edge["last_tick"] = int(tick)
    graph["last_edge_id"] = edge_id
    edge["score_delta_sum"] = float(edge.get("score_delta_sum", 0.0)) + float((metadata or {}).get("score_delta", 0.0))
    edge["terminal_count"] = int(edge.get("terminal_count", 0)) + int(bool((metadata or {}).get("terminal", False)))
    _index_transition_edge(graph, edge)
    _update_progress_target(graph, edge)

    reverse_ids = graph.setdefault("outgoing_edges", {}).get(to_hash, [])
    for reverse_id in list(reverse_ids):
        if reverse_id == edge_id:
            continue
        reverse = edges.get(reverse_id, {})
        if reverse.get("from") == to_hash and reverse.get("to") == from_hash:
            edge["reversible"] = True
            reverse["reversible"] = True
            pair_key = "||".join(sorted([edge_id, reverse_id]))
            pairs = graph.setdefault("reversible_pairs", {})
            if pair_key not in pairs:
                pairs[pair_key] = {"forward": edge_id, "reverse": reverse_id, "first_tick": int(tick)}
                graph["reversible_edge_count"] = int(graph.get("reversible_edge_count", 0)) + 1
            _record_undo_evidence(graph, undo_edge=edge, reversed_edge=reverse, tick=tick)
            _record_undo_evidence(graph, undo_edge=reverse, reversed_edge=edge, tick=tick)
    return graph


def shortest_action_path(graph: Mapping[str, Any], start_hash: str, goal_hash: str) -> list[str] | None:
    if start_hash == goal_hash:
        return []
    edges = graph.get("edges", {})
    frontier: list[tuple[str, list[str]]] = [(str(start_hash), [])]
    seen = {str(start_hash)}
    while frontier:
        node_hash, path = frontier.pop(0)
        outgoing = [
            edge
            for edge in edges.values()
            if str(edge.get("from")) == node_hash
        ]
        outgoing.sort(key=lambda edge: (str(edge.get("action", "")), str(edge.get("to", ""))))
        for edge in outgoing:
            next_hash = str(edge.get("to"))
            if next_hash in seen:
                continue
            next_path = path + [str(edge.get("action"))]
            if next_hash == str(goal_hash):
                return next_path
            seen.add(next_hash)
            frontier.append((next_hash, next_path))
    return None


def _delta_changed(delta_item: Any) -> bool:
    if not isinstance(delta_item, Mapping):
        return bool(delta_item)
    if "changed_count" in delta_item:
        return int(delta_item.get("changed_count", 0)) > 0
    if "changed" in delta_item:
        return bool(delta_item.get("changed"))
    return False


def _delta_l1(delta_item: Any) -> float:
    if isinstance(delta_item, Mapping) and "l1" in delta_item:
        return float(delta_item.get("l1", 0.0))
    return 1.0 if _delta_changed(delta_item) else 0.0


def _fact_confidence(support: int, counterexamples: int) -> float:
    return float(support / max(support + counterexamples, 1))


def _semantic_grid_scene(
    observation: Mapping[str, Any] | None,
    *,
    previous_observation: Mapping[str, Any] | None = None,
    action: int | str | None = None,
    tick: int = 0,
):
    if observation is None:
        return None
    grid = _grid_frame_from_observation(observation)
    if grid is None:
        return None
    previous_grid = _grid_frame_from_observation(previous_observation) if previous_observation is not None else None
    try:
        return parse_grid(grid, previous_grid=previous_grid, action=action, time_index=int(tick))
    except ValueError:
        return None


def _metadata_events(metadata: Mapping[str, Any] | None) -> list[str]:
    if not isinstance(metadata, Mapping):
        return []
    values = metadata.get("events", [])
    if isinstance(values, str):
        return [values]
    if isinstance(values, (list, tuple)):
        return [str(item) for item in values]
    return []


def _metadata_coordinate(metadata: Mapping[str, Any] | None) -> tuple[int, int] | None:
    if not isinstance(metadata, Mapping):
        return None
    value = metadata.get("action_coordinate", metadata.get("coordinate_action"))
    if value is None:
        value = metadata.get("coordinate", metadata.get("click"))
    if isinstance(value, Mapping) and "y" in value and "x" in value:
        return int(value["y"]), int(value["x"])
    if isinstance(value, (list, tuple)) and len(value) >= 2:
        return int(value[0]), int(value[1])
    return None


def _object_at_coordinate(scene: Any, y: int, x: int) -> Mapping[str, Any] | None:
    for obj in scene.objects:
        if obj.kind != "component":
            continue
        if (int(y), int(x)) in obj.cells:
            return obj.compact()
    return None


def _increment_fact(entry: dict[str, Any], *, tick: int, support_key: str = "support") -> None:
    entry[support_key] = int(entry.get(support_key, 0)) + 1
    entry["last_tick"] = int(tick)
    entry.setdefault("first_tick", int(tick))


def _update_semantic_object_and_relation_facts(
    semantic_memory: dict[str, Any],
    *,
    tick: int,
    scene: Any,
    frame_role: str,
) -> None:
    object_facts = semantic_memory.setdefault("object_facts", {})
    relation_facts = semantic_memory.setdefault("relation_facts", {})
    compressed_facts = semantic_memory.setdefault("facts", {})
    seen_objects: set[str] = set()
    object_by_id: dict[str, Any] = {}
    for obj in scene.objects:
        if obj.kind != "component":
            continue
        compact = obj.compact()
        object_by_id[str(obj.object_id)] = obj
        key = f"object:{obj.canonical_hash}:color:{obj.color}:area:{obj.area}"
        if key in seen_objects:
            continue
        seen_objects.add(key)
        fact = object_facts.setdefault(
            key,
            {
                "fact": key,
                "kind": "object_identity",
                "canonical_hash": str(obj.canonical_hash),
                "color": int(obj.color),
                "area": int(obj.area),
                "holes": int(obj.holes),
                "symmetry": dict(obj.symmetry),
                "support": 0,
                "examples": [],
            },
        )
        _increment_fact(fact, tick=tick)
        fact["last_frame_role"] = str(frame_role)
        fact["last_object_id"] = str(obj.object_id)
        fact["last_bbox"] = list(obj.bbox)
        fact["last_centroid"] = [round(float(obj.centroid[0]), 4), round(float(obj.centroid[1]), 4)]
        examples = fact.setdefault("examples", [])
        examples.append(
            {
                "tick": int(tick),
                "frame_role": str(frame_role),
                "object_id": str(obj.object_id),
                "bbox": list(obj.bbox),
                "centroid": [round(float(obj.centroid[0]), 4), round(float(obj.centroid[1]), 4)],
                "salience": float(compact.get("salience", 0.0)),
            }
        )
        if len(examples) > 8:
            del examples[: len(examples) - 8]
        compressed_facts[key] = fact

    for relation in scene.relation_graph:
        source = object_by_id.get(str(relation.source))
        target = object_by_id.get(str(relation.target))
        source_color = int(source.color) if source is not None else None
        target_color = int(target.color) if target is not None else None
        detail = dict(relation.detail)
        detail_key = json.dumps(detail, sort_keys=True, separators=(",", ":"))
        key = f"relation:{relation.relation}:colors:{source_color}:{target_color}:detail:{detail_key}"
        fact = relation_facts.setdefault(
            key,
            {
                "fact": key,
                "kind": "object_relation",
                "relation": str(relation.relation),
                "source_color": source_color,
                "target_color": target_color,
                "detail": detail,
                "support": 0,
            },
        )
        _increment_fact(fact, tick=tick)
        fact["last_frame_role"] = str(frame_role)
        compressed_facts[key] = fact


def _update_semantic_click_facts(
    semantic_memory: dict[str, Any],
    *,
    tick: int,
    action: int | str,
    observation: Mapping[str, Any] | None,
    next_observation: Mapping[str, Any] | None,
    metadata: Mapping[str, Any] | None,
    current_scene: Any,
) -> None:
    coordinate = _metadata_coordinate(metadata)
    if coordinate is None or current_scene is None or observation is None or next_observation is None:
        return
    before_grid = _grid_frame_from_observation(observation)
    after_grid = _grid_frame_from_observation(next_observation)
    if before_grid is None or after_grid is None:
        return
    before_tensor = torch.as_tensor(before_grid)
    after_tensor = torch.as_tensor(after_grid)
    y, x = coordinate
    if not (0 <= y < before_tensor.shape[0] and 0 <= x < before_tensor.shape[1]):
        return
    before_color = int(before_tensor[y, x].item())
    after_color = int(after_tensor[y, x].item())
    changed = bool(torch.any(before_tensor != after_tensor).item())
    color_changed = before_color != after_color
    obj = _object_at_coordinate(current_scene, y, x)
    clicked_empty = before_color == int(current_scene.background)
    if obj is not None:
        cy, cx = obj.get("centroid", [y, x])
        coordinate_role = "object_center" if (int(round(float(cy))), int(round(float(cx)))) == (y, x) else "object_cell"
        object_key = f"object:{obj['canonical_hash']}:color:{obj['color']}:area:{obj['area']}"
        key = f"click:action:{action}:object:{obj['canonical_hash']}:{coordinate_role}"
        kind = "clicked_object_center_changes_color" if coordinate_role == "object_center" and color_changed else "clicked_object"
    else:
        coordinate_role = "empty"
        object_key = None
        key = f"click:action:{action}:empty"
        kind = "click_empty_no_op" if not changed else "click_empty_changed"

    click_facts = semantic_memory.setdefault("click_facts", {})
    compressed_facts = semantic_memory.setdefault("facts", {})
    fact = click_facts.setdefault(
        key,
        {
            "fact": key,
            "kind": kind,
            "action": str(action),
            "coordinate_role": coordinate_role,
            "object_fact": object_key,
            "support": 0,
            "changed_count": 0,
            "no_op_count": 0,
            "color_changed_count": 0,
            "examples": [],
        },
    )
    _increment_fact(fact, tick=tick)
    fact["kind"] = kind
    fact["changed_count"] = int(fact.get("changed_count", 0)) + int(changed)
    fact["no_op_count"] = int(fact.get("no_op_count", 0)) + int(not changed)
    fact["color_changed_count"] = int(fact.get("color_changed_count", 0)) + int(color_changed)
    fact["p_change"] = float(fact["changed_count"] / max(int(fact.get("support", 0)), 1))
    fact["no_op_rate"] = float(fact["no_op_count"] / max(int(fact.get("support", 0)), 1))
    fact["color_change_rate"] = float(fact["color_changed_count"] / max(int(fact.get("support", 0)), 1))
    fact["last_coordinate"] = [int(y), int(x)]
    fact["last_before_color"] = before_color
    fact["last_after_color"] = after_color
    examples = fact.setdefault("examples", [])
    examples.append(
        {
            "tick": int(tick),
            "coordinate": [int(y), int(x)],
            "before_color": before_color,
            "after_color": after_color,
            "changed": changed,
            "color_changed": color_changed,
        }
    )
    if len(examples) > 8:
        del examples[: len(examples) - 8]
    compressed_facts[key] = fact


def _update_semantic_obstacle_facts(
    semantic_memory: dict[str, Any],
    *,
    tick: int,
    action: int | str,
    no_op: bool,
    metadata: Mapping[str, Any] | None,
    current_scene: Any,
) -> None:
    if current_scene is None:
        return
    events = [item.lower() for item in _metadata_events(metadata)]
    event_obstacle = any(
        any(marker in item for marker in ("blocked", "obstacle", "collision", "wall", "hazard"))
        for item in events
    )
    if not no_op and not event_obstacle:
        return
    obstacle_facts = semantic_memory.setdefault("obstacle_facts", {})
    compressed_facts = semantic_memory.setdefault("facts", {})
    colors = sorted({int(obj.color) for obj in current_scene.objects if obj.kind == "component"})
    for color in colors:
        key = f"obstacle:color:{color}"
        fact = obstacle_facts.setdefault(
            key,
            {
                "fact": key,
                "kind": "possible_obstacle_color",
                "color": int(color),
                "support": 0,
                "no_op_support": 0,
                "event_support": 0,
                "actions": {},
            },
        )
        _increment_fact(fact, tick=tick)
        fact["no_op_support"] = int(fact.get("no_op_support", 0)) + int(no_op)
        fact["event_support"] = int(fact.get("event_support", 0)) + int(event_obstacle)
        actions = fact.setdefault("actions", {})
        actions[str(action)] = int(actions.get(str(action), 0)) + 1
        fact["confidence"] = _fact_confidence(
            int(fact.get("no_op_support", 0)) + int(fact.get("event_support", 0)),
            max(int(fact.get("support", 0)) - int(fact.get("no_op_support", 0)) - int(fact.get("event_support", 0)), 0),
        )
        compressed_facts[key] = fact


def _update_semantic_goal_candidate_facts(
    semantic_memory: dict[str, Any],
    *,
    tick: int,
    action: int | str,
    delta: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    observation: Mapping[str, Any],
    next_observation: Mapping[str, Any],
    next_scene: Any,
) -> None:
    goal_facts = semantic_memory.setdefault("goal_facts", {})
    compressed_facts = semantic_memory.setdefault("facts", {})

    candidate_event = {
        "observation": observation_snapshot(observation),
        "action": action,
        "next_observation": observation_snapshot(next_observation),
        "metadata": dict(metadata or {}),
    }
    candidates = goal_candidates_from_event(candidate_event)
    if any(_delta_changed(item) for item in delta.values()):
        candidates.append({"kind": "target_changed_field", "selector": {"fields": sorted(str(key) for key in delta)}})

    for candidate in candidates:
        selector = dict(candidate.get("selector", {}))
        selector_key = json.dumps(selector, sort_keys=True, separators=(",", ":"))
        key = f"goal_candidate:{candidate['kind']}:{selector_key}"
        fact = goal_facts.setdefault(
            key,
            {
                "fact": key,
                "kind": str(candidate["kind"]),
                "scope": str(candidate.get("scope", "goal")),
                "selector": selector,
                "progress_test": dict(candidate.get("progress_test", {})),
                "support": 0,
            },
        )
        _increment_fact(fact, tick=tick)
        fact["confidence"] = _fact_confidence(int(fact.get("support", 0)), 0)
        compressed_facts[key] = fact


def mirror_goal_posterior_to_semantic_memory(
    semantic_memory: dict[str, Any],
    goal_posterior: Mapping[str, Any],
    *,
    tick: int,
) -> dict[str, Any]:
    if not semantic_memory:
        semantic_memory.update(_fresh_semantic_memory())
    goal_facts = semantic_memory.setdefault("goal_facts", {})
    compressed_facts = semantic_memory.setdefault("facts", {})
    for scope in ("goals", "subgoals"):
        for entry in goal_posterior.get(scope, {}).values():
            if not isinstance(entry, Mapping):
                continue
            key = f"posterior_goal_candidate:{entry.get('id', '')}"
            fact = goal_facts.setdefault(
                key,
                {
                    "fact": key,
                    "kind": str(entry.get("kind", "")),
                    "scope": str(entry.get("scope", "goal")),
                    "selector": dict(entry.get("selector", {}) or {}),
                    "support": 0,
                },
            )
            fact["support"] = int(entry.get("support", fact.get("support", 0)))
            fact["inconsistency"] = int(entry.get("inconsistency", 0))
            fact["posterior"] = float(entry.get("posterior", 0.0))
            fact["score"] = float(entry.get("score", 0.0))
            fact["description_length"] = float(entry.get("description_length", 0.0))
            fact["last_tick"] = int(tick)
            fact.setdefault("first_tick", int(tick))
            compressed_facts[key] = fact
    return semantic_memory


def update_semantic_memory(
    semantic_memory: dict[str, Any],
    *,
    tick: int,
    action: int | str,
    delta: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    observation: Mapping[str, Any] | None = None,
    next_observation: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not semantic_memory:
        semantic_memory.update(_fresh_semantic_memory())
    action_text = str(action)
    fields = sorted(str(key) for key in delta.keys())
    changed_fields = [field for field in fields if _delta_changed(delta[field])]
    stable_fields = [field for field in fields if field not in changed_fields]
    no_op = len(changed_fields) == 0

    action_facts = semantic_memory.setdefault("action_facts", {})
    fact = action_facts.setdefault(
        action_text,
        {
            "action": action_text,
            "count": 0,
            "changed_count": 0,
            "no_op_count": 0,
            "terminal_count": 0,
            "score_delta_sum": 0.0,
            "changed_fields": {},
            "stable_fields": {},
            "last_tick": 0,
            "confidence": 0.0,
        },
    )
    fact["count"] = int(fact.get("count", 0)) + 1
    fact["changed_count"] = int(fact.get("changed_count", 0)) + int(not no_op)
    fact["no_op_count"] = int(fact.get("no_op_count", 0)) + int(no_op)
    fact["terminal_count"] = int(fact.get("terminal_count", 0)) + int(bool((metadata or {}).get("terminal", False)))
    fact["score_delta_sum"] = float(fact.get("score_delta_sum", 0.0)) + float((metadata or {}).get("score_delta", 0.0))
    fact["last_tick"] = int(tick)
    for field in changed_fields:
        changed = fact.setdefault("changed_fields", {})
        changed[field] = int(changed.get(field, 0)) + 1
    for field in stable_fields:
        stable = fact.setdefault("stable_fields", {})
        stable[field] = int(stable.get(field, 0)) + 1
    fact["confidence"] = float(fact["changed_count"] / max(fact["count"], 1))

    invariants = semantic_memory.setdefault("invariants", {})
    counterexamples = semantic_memory.setdefault("counterexamples", {})
    compressed_facts = semantic_memory.setdefault("facts", {})
    for field in stable_fields:
        invariant_key = f"action:{action_text}:field:{field}:stable"
        invariant = invariants.setdefault(
            invariant_key,
            {
                "fact": invariant_key,
                "action": action_text,
                "field": field,
                "support": 0,
                "counterexamples": 0,
                "confidence": 0.0,
                "first_tick": int(tick),
                "last_tick": int(tick),
            },
        )
        invariant["support"] = int(invariant.get("support", 0)) + 1
        invariant["last_tick"] = int(tick)
        invariant["confidence"] = _fact_confidence(
            int(invariant.get("support", 0)),
            int(invariant.get("counterexamples", 0)),
        )
        compressed_facts[invariant_key] = invariant
    for field in changed_fields:
        invariant_key = f"action:{action_text}:field:{field}:stable"
        if invariant_key in invariants:
            invariant = invariants[invariant_key]
            invariant["counterexamples"] = int(invariant.get("counterexamples", 0)) + 1
            invariant["last_tick"] = int(tick)
            invariant["confidence"] = _fact_confidence(
                int(invariant.get("support", 0)),
                int(invariant.get("counterexamples", 0)),
            )
            counterexamples.setdefault(invariant_key, []).append(
                {
                    "tick": int(tick),
                    "action": action_text,
                    "field": field,
                    "delta_l1": _delta_l1(delta[field]),
                }
            )
            compressed_facts[invariant_key] = invariant

    affordance = semantic_memory.setdefault("affordances", {}).setdefault(
        action_text,
        {
            "action": action_text,
            "trials": 0,
            "changed_trials": 0,
            "no_op_trials": 0,
            "terminal_trials": 0,
            "mean_score_delta": 0.0,
            "change_rate": 0.0,
            "no_op_rate": 0.0,
        },
    )
    affordance["trials"] = int(affordance.get("trials", 0)) + 1
    affordance["changed_trials"] = int(affordance.get("changed_trials", 0)) + int(not no_op)
    affordance["no_op_trials"] = int(affordance.get("no_op_trials", 0)) + int(no_op)
    affordance["terminal_trials"] = int(affordance.get("terminal_trials", 0)) + int(bool((metadata or {}).get("terminal", False)))
    previous_total = float(affordance.get("mean_score_delta", 0.0)) * float(max(affordance["trials"] - 1, 0))
    score_delta = float((metadata or {}).get("score_delta", 0.0))
    affordance["mean_score_delta"] = float((previous_total + score_delta) / max(affordance["trials"], 1))
    affordance["change_rate"] = float(affordance["changed_trials"] / max(affordance["trials"], 1))
    affordance["no_op_rate"] = float(affordance["no_op_trials"] / max(affordance["trials"], 1))

    action_fact_key = f"action:{action_text}:effect"
    compressed_facts[action_fact_key] = {
        "fact": action_fact_key,
        "action": action_text,
        "support": int(fact["count"]),
        "changed_fields": dict(fact.get("changed_fields", {})),
        "stable_fields": dict(fact.get("stable_fields", {})),
        "change_rate": float(affordance["change_rate"]),
        "no_op_rate": float(affordance["no_op_rate"]),
        "mean_score_delta": float(affordance["mean_score_delta"]),
        "confidence": float(max(affordance["change_rate"], affordance["no_op_rate"])),
    }

    current_scene = _semantic_grid_scene(observation, action=action, tick=tick)
    next_scene = _semantic_grid_scene(
        next_observation,
        previous_observation=observation,
        action=action,
        tick=tick,
    )
    seen_scene_hashes: set[str] = set()
    for frame_role, scene in (("current", current_scene), ("next", next_scene)):
        if scene is None or scene.raw_hash in seen_scene_hashes:
            continue
        seen_scene_hashes.add(scene.raw_hash)
        _update_semantic_object_and_relation_facts(
            semantic_memory,
            tick=tick,
            scene=scene,
            frame_role=frame_role,
        )
    _update_semantic_click_facts(
        semantic_memory,
        tick=tick,
        action=action,
        observation=observation,
        next_observation=next_observation,
        metadata=metadata,
        current_scene=current_scene,
    )
    _update_semantic_obstacle_facts(
        semantic_memory,
        tick=tick,
        action=action,
        no_op=no_op,
        metadata=metadata,
        current_scene=current_scene,
    )
    _update_semantic_goal_candidate_facts(
        semantic_memory,
        tick=tick,
        action=action,
        delta=delta,
        metadata=metadata,
        observation=observation,
        next_observation=next_observation,
        next_scene=next_scene,
    )
    return semantic_memory


def _numeric_values(value: Any) -> list[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, Mapping):
        values: list[float] = []
        for item in value.values():
            values.extend(_numeric_values(item))
        return values
    if isinstance(value, (list, tuple)):
        values: list[float] = []
        for item in value:
            values.extend(_numeric_values(item))
        return values
    return []


def _rank_vector(key: str, rank: int) -> list[float]:
    digest = hashlib.sha256(key.encode("utf-8")).digest()
    values: list[float] = []
    for index in range(rank):
        byte = digest[index % len(digest)]
        values.append((float(byte) / 127.5) - 1.0)
    return values


def _vector_add_scaled(left: list[float], right: list[float], scale: float) -> list[float]:
    return [float(a + scale * b) for a, b in zip(left, right)]


def _adapter_norm(adapter: Mapping[str, Any]) -> float:
    total = 0.0
    for section in ("action_factors", "field_factors"):
        for vector in adapter.get(section, {}).values():
            total += sum(float(item) * float(item) for item in vector)
    return float(total ** 0.5)


def _flatten_numeric_values(value: Any, limit: int = 128) -> list[float]:
    values: list[float] = []

    def visit(item: Any) -> None:
        if len(values) >= limit:
            return
        if item is None or isinstance(item, bool):
            return
        if isinstance(item, (int, float)):
            values.append(float(max(-10.0, min(10.0, float(item)))))
            return
        if isinstance(item, torch.Tensor):
            tensor = item.detach().cpu().float().flatten()
            remaining = max(0, limit - len(values))
            if remaining:
                values.extend(float(max(-10.0, min(10.0, item_value))) for item_value in tensor[:remaining].tolist())
            return
        if isinstance(item, Mapping):
            for key in sorted(item):
                visit(item[key])
                if len(values) >= limit:
                    break
            return
        if isinstance(item, (list, tuple)):
            for child in item:
                visit(child)
                if len(values) >= limit:
                    break

    visit(value)
    return values


def _fixed_vector(values: list[float], dim: int) -> list[float]:
    clipped = [float(max(-10.0, min(10.0, value))) for value in values[:dim]]
    if len(clipped) < dim:
        clipped.extend([0.0] * (dim - len(clipped)))
    return clipped


def _action_bucket(action: int | str, buckets: int) -> int:
    try:
        numeric = int(action)
    except (TypeError, ValueError):
        digest = hashlib.sha256(str(action).encode("utf-8")).digest()
        numeric = int.from_bytes(digest[:4], "little", signed=False)
    return int(numeric % max(int(buckets), 1))


def _transition_input_vector(
    *,
    observation: Mapping[str, Any],
    action: int | str,
    input_dim: int,
    action_buckets: int,
) -> list[float]:
    action_dim = min(max(action_buckets, 1), max(input_dim // 3, 1))
    observation_dim = max(input_dim - action_dim, 1)
    values = _fixed_vector(_flatten_numeric_values(observation), observation_dim)
    one_hot = [0.0] * action_dim
    one_hot[_action_bucket(action, action_dim)] = 1.0
    return _fixed_vector(values + one_hot, input_dim)


def _delta_target_vector(delta: Mapping[str, Any], delta_dim: int) -> list[float]:
    values: list[float] = []
    for field in sorted(delta):
        item = delta[field]
        values.append(float(_delta_changed(item)))
        values.append(float(min(_delta_l1(item), 10.0)))
        if isinstance(item, Mapping):
            values.append(float(min(float(item.get("changed_count", 0.0) or 0.0), 10.0)))
            values.append(float(min(float(item.get("max_abs", 0.0) or 0.0), 10.0)))
            raw_values = item.get("values")
            if raw_values is not None:
                values.extend(_flatten_numeric_values(raw_values, limit=max(delta_dim - len(values), 0)))
    return _fixed_vector(values, delta_dim)


def _grid_signatures(observation: Mapping[str, Any]) -> set[str] | None:
    grid = observation.get("grid")
    if not isinstance(grid, torch.Tensor) or grid.ndim != 2:
        return None
    signatures: set[str] = set()
    for value in torch.unique(grid.detach().cpu().long()).tolist():
        if int(value) == 0:
            continue
        count = int((grid == int(value)).sum().item())
        signatures.add(f"value:{int(value)}|area:{count}")
    return signatures


def _object_persistence_target(
    observation: Mapping[str, Any],
    next_observation: Mapping[str, Any],
    delta: Mapping[str, Any],
) -> float:
    before = _grid_signatures(observation)
    after = _grid_signatures(next_observation)
    if before is not None and after is not None:
        if not before and not after:
            return 1.0
        return float(len(before & after) / max(len(before | after), 1))
    changed_fields = sum(1 for item in delta.values() if _delta_changed(item))
    return 0.0 if changed_fields else 1.0


def _adapter_head_tensor(
    head: dict[str, Any],
    *,
    rows: int,
    cols: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    weight = head.get("weight", [])
    bias = head.get("bias", [])
    if not weight:
        weight_tensor = torch.zeros(rows, cols, dtype=torch.float32)
    else:
        weight_tensor = torch.tensor(weight, dtype=torch.float32).view(rows, cols)
    if not bias:
        bias_tensor = torch.zeros(rows, dtype=torch.float32)
    else:
        bias_tensor = torch.tensor(bias, dtype=torch.float32).view(rows)
    return weight_tensor, bias_tensor


def _store_adapter_head(head: dict[str, Any], weight: torch.Tensor, bias: torch.Tensor) -> None:
    head["weight"] = weight.detach().clamp(-5.0, 5.0).tolist()
    head["bias"] = bias.detach().clamp(-5.0, 5.0).tolist()


def update_self_supervised_adapter(
    plastic_memory: dict[str, Any],
    *,
    tick: int,
    observation: Mapping[str, Any],
    action: int | str,
    next_observation: Mapping[str, Any],
    delta: Mapping[str, Any],
) -> dict[str, Any]:
    adapter = plastic_memory.setdefault("self_supervised_adapter", _fresh_self_supervised_adapter())
    if not adapter:
        adapter.update(_fresh_self_supervised_adapter())
    input_dim = int(adapter.get("input_dim", 24))
    delta_dim = int(adapter.get("delta_dim", 8))
    action_buckets = int(adapter.get("action_buckets", NUM_ACTIONS))
    lr = float(adapter.get("lr", 0.05))
    heads = adapter.setdefault("heads", _fresh_self_supervised_adapter()["heads"])

    x = torch.tensor(
        _transition_input_vector(
            observation=observation,
            action=action,
            input_dim=input_dim,
            action_buckets=action_buckets,
        ),
        dtype=torch.float32,
    )
    delta_target = torch.tensor(_delta_target_vector(delta, delta_dim), dtype=torch.float32)
    action_target = torch.tensor([_action_bucket(action, action_buckets)], dtype=torch.long)
    change_target = torch.tensor([1.0 if any(_delta_changed(item) for item in delta.values()) else 0.0], dtype=torch.float32)
    object_target = torch.tensor([_object_persistence_target(observation, next_observation, delta)], dtype=torch.float32)

    next_w, next_b = _adapter_head_tensor(heads.setdefault("next_delta", {}), rows=delta_dim, cols=input_dim)
    inverse_w, inverse_b = _adapter_head_tensor(
        heads.setdefault("inverse_action", {}),
        rows=action_buckets,
        cols=delta_dim,
    )
    noop_w, noop_b = _adapter_head_tensor(heads.setdefault("no_op_change", {}), rows=1, cols=delta_dim)
    persist_w, persist_b = _adapter_head_tensor(heads.setdefault("object_persistence", {}), rows=1, cols=input_dim)

    pred_delta = next_w.matmul(x) + next_b
    inverse_logits = inverse_w.matmul(delta_target) + inverse_b
    change_logit = noop_w.matmul(delta_target) + noop_b
    persistence_logit = persist_w.matmul(x) + persist_b

    next_delta_loss = F.mse_loss(pred_delta, delta_target)
    inverse_action_loss = F.cross_entropy(inverse_logits.view(1, -1), action_target)
    no_op_change_loss = F.binary_cross_entropy_with_logits(change_logit.view(1), change_target)
    object_persistence_loss = F.binary_cross_entropy_with_logits(persistence_logit.view(1), object_target)
    total_loss = next_delta_loss + inverse_action_loss + no_op_change_loss + object_persistence_loss

    next_grad = (2.0 / max(int(delta_target.numel()), 1)) * (pred_delta - delta_target)
    inverse_grad = torch.softmax(inverse_logits, dim=0)
    inverse_grad[int(action_target.item())] -= 1.0
    change_grad = torch.sigmoid(change_logit.view(1)) - change_target
    persistence_grad = torch.sigmoid(persistence_logit.view(1)) - object_target

    next_w -= lr * next_grad.view(-1, 1).matmul(x.view(1, -1))
    next_b -= lr * next_grad
    inverse_w -= lr * inverse_grad.view(-1, 1).matmul(delta_target.view(1, -1))
    inverse_b -= lr * inverse_grad
    noop_w -= lr * change_grad.view(1, 1).matmul(delta_target.view(1, -1))
    noop_b -= lr * change_grad
    persist_w -= lr * persistence_grad.view(1, 1).matmul(x.view(1, -1))
    persist_b -= lr * persistence_grad
    for parameter in (next_w, next_b, inverse_w, inverse_b, noop_w, noop_b, persist_w, persist_b):
        parameter.clamp_(-5.0, 5.0)

    _store_adapter_head(heads["next_delta"], next_w, next_b)
    _store_adapter_head(heads["inverse_action"], inverse_w, inverse_b)
    _store_adapter_head(heads["no_op_change"], noop_w, noop_b)
    _store_adapter_head(heads["object_persistence"], persist_w, persist_b)

    losses = {
        "next_delta": float(next_delta_loss.detach().item()),
        "inverse_action": float(inverse_action_loss.detach().item()),
        "no_op_change": float(no_op_change_loss.detach().item()),
        "object_persistence": float(object_persistence_loss.detach().item()),
        "total": float(total_loss.detach().item()),
    }
    metrics = {
        "action_bucket": int(action_target.item()),
        "change_target": float(change_target.item()),
        "object_persistence_target": float(object_target.item()),
        "predicted_change_probability": float(torch.sigmoid(change_logit.detach()).item()),
        "predicted_object_persistence": float(torch.sigmoid(persistence_logit.detach()).item()),
        "delta_prediction_l1": float((pred_delta.detach() - delta_target).abs().sum().item()),
    }
    history = adapter.setdefault("loss_history", [])
    history.append({"tick": int(tick), **losses, **metrics})
    capacity = int(adapter.get("history_capacity", plastic_memory.get("capacity", 64)))
    if len(history) > capacity:
        del history[: len(history) - capacity]
    adapter["updates"] = int(adapter.get("updates", 0)) + 1
    adapter["latest_losses"] = losses
    adapter["latest_metrics"] = metrics
    return adapter


def maintain_active_theory_set(
    active_theory_set: dict[str, Any],
    hypothesis_posterior: Mapping[str, Any],
) -> dict[str, Any]:
    if not active_theory_set:
        active_theory_set.update(fresh_active_theory_set())
    active_theory_set["updates"] = int(active_theory_set.get("updates", 0)) + 1
    min_candidates = int(active_theory_set.get("min_candidates", 3))
    max_candidates = int(active_theory_set.get("max_candidates", 20))
    hypotheses = list((hypothesis_posterior or {}).get("hypotheses", {}).values())
    hypotheses.sort(
        key=lambda item: (
            float(item.get("posterior", 0.0)),
            int(item.get("support", 0)),
            -int(item.get("counterexamples", 0)),
            -float(item.get("description_length", 0.0)),
            str(item.get("id", "")),
        ),
        reverse=True,
    )
    selected: list[Mapping[str, Any]] = []
    seen_ids: set[str] = set()
    families = sorted({str(item.get("family", "")) for item in hypotheses if str(item.get("family", ""))})
    for family in families:
        best = next((item for item in hypotheses if str(item.get("family", "")) == family and str(item.get("id", "")) not in seen_ids), None)
        if best is None:
            continue
        selected.append(best)
        seen_ids.add(str(best.get("id", "")))
        if len(selected) >= max_candidates:
            break
    for item in hypotheses:
        if len(selected) >= max_candidates:
            break
        hid = str(item.get("id", ""))
        if hid in seen_ids:
            continue
        selected.append(item)
        seen_ids.add(hid)
    if len(hypotheses) < min_candidates:
        selected = hypotheses
    active_theory_set["theories"] = [
        {
            "id": str(item.get("id", "")),
            "family": str(item.get("family", "")),
            "action": str(item.get("action", "")),
            "posterior": float(item.get("posterior", 0.0)),
            "support": int(item.get("support", 0)),
            "counterexamples": int(item.get("counterexamples", 0)),
            "description_length": float(item.get("description_length", 0.0)),
            "context_confidence": float(item.get("context_confidence", 1.0)),
        }
        for item in selected
    ]
    active_theory_set["coverage"] = {
        "available_hypotheses": len(hypotheses),
        "active_count": len(selected),
        "families": sorted({str(item.get("family", "")) for item in selected if str(item.get("family", ""))}),
        "bounded": bool(len(hypotheses) > max_candidates),
    }
    return active_theory_set


def _neuromodulation(
    *,
    delta: Mapping[str, Any],
    metadata: Mapping[str, Any] | None,
    prediction_error: Mapping[str, Any] | None,
) -> float:
    changed_fields = [field for field, item in delta.items() if _delta_changed(item)]
    delta_energy = sum(_delta_l1(delta[field]) for field in changed_fields)
    prediction_values = _numeric_values(prediction_error or {})
    prediction_signal = sum(max(0.0, item) for item in prediction_values) / max(len(prediction_values), 1)
    score_delta = float((metadata or {}).get("score_delta", 0.0))
    terminal = bool((metadata or {}).get("terminal", False))
    value = (
        0.02
        + min(delta_energy, 5.0) * 0.08
        + min(max(score_delta, 0.0), 2.0) * 0.45
        + min(prediction_signal, 2.0) * 0.20
        + (0.18 if terminal else 0.0)
    )
    return float(max(0.0, value))


def update_plastic_memory(
    plastic_memory: dict[str, Any],
    *,
    tick: int,
    observation: Mapping[str, Any],
    action: int | str,
    next_observation: Mapping[str, Any],
    delta: Mapping[str, Any],
    metadata: Mapping[str, Any] | None = None,
    prediction_error: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not plastic_memory:
        plastic_memory.update(_fresh_plastic_memory())
    action_text = str(action)
    changed_fields = sorted(str(field) for field, item in delta.items() if _delta_changed(item))
    stable_fields = sorted(str(field) for field in delta.keys() if str(field) not in changed_fields)
    modulation = _neuromodulation(delta=delta, metadata=metadata, prediction_error=prediction_error)
    capacity = int(plastic_memory.get("capacity", 64))
    plastic_memory["updates"] = int(plastic_memory.get("updates", 0)) + 1

    trace = {
        "tick": int(tick),
        "action": action_text,
        "observation_hash": canonical_observation_hash(observation),
        "next_observation_hash": canonical_observation_hash(next_observation),
        "changed_fields": changed_fields,
        "stable_fields": stable_fields,
        "neuromodulation": modulation,
        "score_delta": float((metadata or {}).get("score_delta", 0.0)),
        "prediction_error": _json_safe(dict(prediction_error or {})),
    }
    key_values = plastic_memory.setdefault("key_values", [])
    key_values.append(trace)
    if len(key_values) > capacity:
        del key_values[: len(key_values) - capacity]

    hebbian = plastic_memory.setdefault("hebbian_traces", {})
    target_fields = changed_fields or ["no_op"]
    for field in target_fields:
        key = f"action:{action_text}|field:{field}"
        old = float(hebbian.get(key, 0.0))
        field_energy = _delta_l1(delta[field]) if field in delta else 1.0
        hebbian[key] = float(0.95 * old + modulation * max(field_energy, 1.0))

    adapter = plastic_memory.setdefault("low_rank_adapter", _fresh_plastic_memory()["low_rank_adapter"])
    rank = int(adapter.get("rank", 4))
    action_factors = adapter.setdefault("action_factors", {})
    field_factors = adapter.setdefault("field_factors", {})
    action_vector = [float(item) for item in action_factors.get(action_text, [0.0] * rank)]
    action_basis = _rank_vector(f"action:{action_text}", rank)
    action_factors[action_text] = _vector_add_scaled(action_vector, action_basis, modulation * 0.05)
    for field in target_fields:
        field_vector = [float(item) for item in field_factors.get(field, [0.0] * rank)]
        field_basis = _rank_vector(f"field:{field}", rank)
        field_factors[field] = _vector_add_scaled(field_vector, field_basis, modulation * 0.05)
    adapter["norm"] = _adapter_norm(adapter)
    update_self_supervised_adapter(
        plastic_memory,
        tick=tick,
        observation=observation,
        action=action,
        next_observation=next_observation,
        delta=delta,
    )
    return plastic_memory


def _event_score_delta(event: Mapping[str, Any]) -> float:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, Mapping):
        return 0.0
    try:
        return float(metadata.get("score_delta", 0.0))
    except (TypeError, ValueError):
        return 0.0


def _metadata_event_text(metadata: Mapping[str, Any]) -> str:
    values: list[str] = []
    for key in ("event", "events", "status", "phase"):
        value = metadata.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set)):
            values.extend(str(item).lower() for item in value)
        else:
            values.append(str(value).lower())
    return " ".join(values)


def _event_context_key(context_state: Mapping[str, Any]) -> str:
    return "|".join(
        [
            str(context_state.get("game_id")),
            str(int(context_state.get("level_index", 0))),
            str(int(context_state.get("episode_index", 0))),
        ]
    )


def _macro_progress_signal(event: Mapping[str, Any]) -> str | None:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    score_delta = _event_score_delta(event)
    if score_delta > 0.0:
        return "positive_reward"
    boundary = _boundary_kind(metadata)
    if boundary is not None and str(boundary).lower() in {"win", "level", "level_transition"}:
        return "level_progress"
    event_text = _metadata_event_text(metadata)
    if "goal" in event_text or "win" in event_text or "progress" in event_text:
        return "event_progress"
    return None


def _recent_macro_events(event_journal: list[dict[str, Any]], max_len: int = 8) -> list[dict[str, Any]]:
    selected: list[dict[str, Any]] = []
    for event in reversed(event_journal):
        if selected:
            metadata = event.get("metadata", {})
            if isinstance(metadata, Mapping) and _boundary_kind(metadata) is not None:
                break
        selected.append(event)
        if len(selected) >= max_len:
            break
    selected.reverse()
    return selected


def _macro_id(action_sequence: list[str], signal: str) -> str:
    payload = json.dumps(
        {"actions": action_sequence, "signal": signal},
        sort_keys=True,
        separators=(",", ":"),
    )
    digest = hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
    return f"macro|{digest}"


def _normalize_macro_library(library: dict[str, Any]) -> None:
    macros = library.setdefault("macros", {})
    top = sorted(
        macros.values(),
        key=lambda item: (
            float(item.get("confidence", 0.0)),
            int(item.get("support", 0)),
            int(item.get("last_tick", 0)),
        ),
        reverse=True,
    )
    library["top_macros"] = [
        {
            "id": item["id"],
            "objective": item["objective"],
            "action_sequence": list(item.get("action_sequence", [])),
            "support": int(item.get("support", 0)),
            "confidence": float(item.get("confidence", 0.0)),
            "transfer_ready": bool(item.get("transfer_ready", False)),
        }
        for item in top[:8]
    ]


def consolidate_event_journal(state: "PersistentMemoryState", trigger_event: Mapping[str, Any]) -> dict[str, Any] | None:
    library = state.macro_policy_library
    if not library:
        library.update(fresh_macro_policy_library())
    library["updates"] = int(library.get("updates", 0)) + 1

    signal = _macro_progress_signal(trigger_event)
    if signal is None:
        _normalize_macro_library(library)
        return None

    events = _recent_macro_events(state.event_journal)
    action_sequence = [str(event.get("action")) for event in events if "action" in event]
    if not action_sequence:
        _normalize_macro_library(library)
        return None

    ticks = [int(event.get("tick", 0)) for event in events]
    macro_id = _macro_id(action_sequence, signal)
    context = {
        "key": _event_context_key(state.context_state),
        "game_id": state.context_state.get("game_id"),
        "level_index": int(state.context_state.get("level_index", 0)),
        "episode_index": int(state.context_state.get("episode_index", 0)),
        "phase": state.context_state.get("phase", "running"),
    }
    metadata = trigger_event.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    score_delta = _event_score_delta(trigger_event)
    boundary = _boundary_kind(metadata)
    macros = library.setdefault("macros", {})
    macro = macros.setdefault(
        macro_id,
        {
            "schema": "runtime_macro_policy_v1",
            "id": macro_id,
            "objective": signal,
            "action_sequence": action_sequence,
            "length": len(action_sequence),
            "support": 0,
            "success_count": 0,
            "failure_count": 0,
            "terminal_count": 0,
            "boundary_count": 0,
            "score_delta_sum": 0.0,
            "mean_score_delta": 0.0,
            "confidence": 0.0,
            "transfer_ready": False,
            "contexts": [],
            "context_keys": [],
            "source_event_ticks": [],
            "first_tick": int(trigger_event.get("tick", 0)),
            "last_tick": int(trigger_event.get("tick", 0)),
        },
    )
    macro["support"] = int(macro.get("support", 0)) + 1
    macro["success_count"] = int(macro.get("success_count", 0)) + 1
    macro["terminal_count"] = int(macro.get("terminal_count", 0)) + int(bool(metadata.get("terminal", False)))
    macro["boundary_count"] = int(macro.get("boundary_count", 0)) + int(boundary is not None)
    macro["score_delta_sum"] = float(macro.get("score_delta_sum", 0.0)) + score_delta
    macro["mean_score_delta"] = float(macro["score_delta_sum"] / max(int(macro.get("support", 0)), 1))
    macro["last_tick"] = int(trigger_event.get("tick", 0))
    macro["source_event_ticks"] = sorted(set(int(tick) for tick in macro.get("source_event_ticks", []) + ticks))
    context_keys = macro.setdefault("context_keys", [])
    if context["key"] not in context_keys:
        context_keys.append(context["key"])
        macro.setdefault("contexts", []).append(context)
    macro["cross_context_support"] = len(context_keys)
    macro["transfer_ready"] = bool(int(macro.get("support", 0)) >= 2 or len(context_keys) >= 2)
    macro["confidence"] = float(
        int(macro.get("success_count", 0))
        / max(int(macro.get("success_count", 0)) + int(macro.get("failure_count", 0)), 1)
    )

    library["consolidations"] = int(library.get("consolidations", 0)) + 1
    library["last_consolidated_tick"] = int(trigger_event.get("tick", 0))
    _normalize_macro_library(library)

    procedural_facts = state.semantic_memory.setdefault("procedural_facts", {})
    fact_key = f"macro:{macro_id}"
    fact = {
        "fact": fact_key,
        "kind": "macro_policy",
        "macro_id": macro_id,
        "objective": signal,
        "action_sequence": list(action_sequence),
        "support": int(macro.get("support", 0)),
        "confidence": float(macro.get("confidence", 0.0)),
        "transfer_ready": bool(macro.get("transfer_ready", False)),
        "last_tick": int(trigger_event.get("tick", 0)),
    }
    procedural_facts[fact_key] = fact
    state.semantic_memory.setdefault("facts", {})[fact_key] = fact
    return macro


def _boundary_kind(metadata: Mapping[str, Any]) -> str | None:
    explicit = metadata.get("boundary")
    if explicit not in (None, "", False):
        return str(explicit)
    event_text = _metadata_event_text(metadata)
    if "game_over" in event_text or "game over" in event_text:
        return "game_over"
    if "level" in event_text:
        return "level"
    if "win" in event_text or "won" in event_text:
        return "win"
    if "reset" in event_text:
        return "reset"
    if bool(metadata.get("terminal", False)):
        return "terminal"
    return None


def _boundary_phase(kind: str) -> str:
    normalized = str(kind).lower()
    if normalized in {"episode", "terminal"}:
        return "episode_boundary"
    if normalized in {"level", "level_transition"}:
        return "level_transition"
    if normalized == "game_over":
        return "game_over"
    if normalized == "reset":
        return "reset"
    if normalized == "win":
        return "win"
    return "boundary"


def _memory_preservation_counts(state: "PersistentMemoryState") -> dict[str, Any]:
    graph = state.transition_graph or {}
    semantic = state.semantic_memory or {}
    plastic = state.plastic_memory or {}
    hypotheses = state.hypothesis_posterior or {}
    goals = state.goal_posterior or {}
    coordinates = state.coordinate_affordances or {}
    macros = state.macro_policy_library or {}
    controllability = state.controllability_model or {}
    progress_value = state.progress_value_model or {}
    active_theories = state.active_theory_set or {}
    return {
        "event_journal": len(state.event_journal),
        "transition_nodes": len(graph.get("nodes", {})),
        "transition_edges": len(graph.get("edges", {})),
        "semantic_action_facts": len(semantic.get("action_facts", {})),
        "semantic_invariants": len(semantic.get("invariants", {})),
        "semantic_facts": len(semantic.get("facts", {})),
        "plastic_updates": int(plastic.get("updates", 0)),
        "plastic_key_values": len(plastic.get("key_values", [])),
        "hypotheses": len(hypotheses.get("hypotheses", {})),
        "goals": len(goals.get("goals", {})),
        "subgoals": len(goals.get("subgoals", {})),
        "coordinate_updates": int(coordinates.get("updates", 0)),
        "coordinate_candidates": len(coordinates.get("candidates", {})),
        "macro_policies": len(macros.get("macros", {})),
        "macro_consolidations": int(macros.get("consolidations", 0)),
        "controlled_objects": len(controllability.get("controlled_objects", {})),
        "controlled_variables": len(controllability.get("controlled_variables", {})),
        "progress_value_actions": len(progress_value.get("action_values", {})),
        "progress_value_sequences": len(progress_value.get("sequence_values", {})),
        "active_theories": len(active_theories.get("theories", [])),
    }


def _is_fragile_hypothesis(entry: Mapping[str, Any], verified_min_support: int) -> bool:
    support = int(entry.get("support", 0))
    counterexamples = int(entry.get("counterexamples", 0))
    return support < verified_min_support or counterexamples > 0


def _is_fragile_goal(entry: Mapping[str, Any], verified_min_support: int) -> bool:
    support = int(entry.get("support", 0))
    inconsistency = int(entry.get("inconsistency", 0))
    return support < verified_min_support or inconsistency > 0


def _decay_fragile_context_confidence(
    *,
    context_state: dict[str, Any],
    hypothesis_posterior: dict[str, Any],
    goal_posterior: dict[str, Any],
) -> dict[str, list[str]]:
    gates = context_state.setdefault("confidence_gates", {})
    verified_min_support = int(gates.get("verified_min_support", 2))
    hypothesis_decay = 0.75
    goal_decay = 0.75
    gates["fragile_hypothesis_decay"] = float(gates.get("fragile_hypothesis_decay", 1.0)) * hypothesis_decay
    gates["fragile_goal_decay"] = float(gates.get("fragile_goal_decay", 1.0)) * goal_decay

    decayed = {"hypotheses": [], "goals": [], "subgoals": []}
    for hid, hypothesis in hypothesis_posterior.get("hypotheses", {}).items():
        if _is_fragile_hypothesis(hypothesis, verified_min_support):
            hypothesis["context_confidence"] = float(hypothesis.get("context_confidence", 1.0)) * hypothesis_decay
            decayed["hypotheses"].append(str(hid))
        else:
            hypothesis["context_confidence"] = max(float(hypothesis.get("context_confidence", 1.0)), 1.0)

    for section in ("goals", "subgoals"):
        for gid, goal in goal_posterior.get(section, {}).items():
            if _is_fragile_goal(goal, verified_min_support):
                goal["context_confidence"] = float(goal.get("context_confidence", 1.0)) * goal_decay
                decayed[section].append(str(gid))
            else:
                goal["context_confidence"] = max(float(goal.get("context_confidence", 1.0)), 1.0)
    return decayed


def apply_boundary_transform(state: "PersistentMemoryState", event: Mapping[str, Any]) -> dict[str, Any] | None:
    metadata = event.get("metadata", {})
    if not isinstance(metadata, Mapping):
        metadata = {}
    kind = _boundary_kind(metadata)
    context_state = state.context_state
    context_state.setdefault("schema", "runtime_context_state_v1")
    context_state.setdefault("game_id", None)
    context_state.setdefault("level_index", 0)
    context_state.setdefault("episode_index", 0)
    context_state.setdefault("phase", "running")
    context_state.setdefault("uncertainty", 1.0)
    context_state.setdefault("confidence_gates", dict(fresh_context_state()["confidence_gates"]))
    context_state.setdefault("boundaries", [])

    if "game_id" in metadata:
        context_state["game_id"] = metadata.get("game_id")

    if kind is None:
        context_state["last_terminal"] = bool(metadata.get("terminal", False))
        context_state["last_boundary"] = None
        context_state["phase"] = "running"
        context_state["uncertainty"] = float(max(0.05, float(context_state.get("uncertainty", 1.0)) * 0.995))
        return None

    before = {
        "game_id": context_state.get("game_id"),
        "level_index": int(context_state.get("level_index", 0)),
        "episode_index": int(context_state.get("episode_index", 0)),
        "phase": context_state.get("phase", "running"),
        "uncertainty": float(context_state.get("uncertainty", 1.0)),
    }
    normalized = str(kind).lower()
    if "game_id" in metadata:
        context_state["game_id"] = metadata.get("game_id")
    if normalized in {"episode", "reset", "terminal"}:
        context_state["episode_index"] = int(context_state.get("episode_index", 0)) + 1
    if normalized in {"level", "level_transition", "win"}:
        context_state["level_index"] = int(context_state.get("level_index", 0)) + 1
    if normalized == "game_over":
        context_state["episode_index"] = int(context_state.get("episode_index", 0)) + 1

    context_state["phase"] = _boundary_phase(normalized)
    context_state["last_terminal"] = bool(metadata.get("terminal", True))
    context_state["last_boundary"] = normalized
    context_state["uncertainty"] = float(min(1.0, max(0.35, float(context_state.get("uncertainty", 1.0)) * 1.15)))
    gates = context_state.setdefault("confidence_gates", {})
    gates["boundary_count"] = int(gates.get("boundary_count", 0)) + 1
    decayed = _decay_fragile_context_confidence(
        context_state=context_state,
        hypothesis_posterior=state.hypothesis_posterior,
        goal_posterior=state.goal_posterior,
    )

    after = {
        "game_id": context_state.get("game_id"),
        "level_index": int(context_state.get("level_index", 0)),
        "episode_index": int(context_state.get("episode_index", 0)),
        "phase": context_state.get("phase", "running"),
        "uncertainty": float(context_state.get("uncertainty", 1.0)),
    }
    record = {
        "schema": "runtime_boundary_transform_v1",
        "tick": int(event.get("tick", state.tick)),
        "boundary": normalized,
        "terminal": bool(metadata.get("terminal", True)),
        "before": before,
        "after": after,
        "preserved_counts": _memory_preservation_counts(state),
        "confidence_decay": decayed,
    }
    context_state.setdefault("boundaries", []).append(record)
    return record


@dataclass
class PersistentMemoryState:
    latent: torch.Tensor
    private_token: torch.Tensor
    tick: int = 0
    event_journal: list[dict[str, Any]] = field(default_factory=list)
    transition_graph: dict[str, Any] = field(default_factory=_fresh_transition_graph)
    semantic_memory: dict[str, Any] = field(default_factory=_fresh_semantic_memory)
    plastic_memory: dict[str, Any] = field(default_factory=_fresh_plastic_memory)
    hypothesis_posterior: dict[str, Any] = field(default_factory=fresh_hypothesis_posterior)
    active_theory_set: dict[str, Any] = field(default_factory=fresh_active_theory_set)
    goal_posterior: dict[str, Any] = field(default_factory=fresh_goal_posterior)
    macro_policy_library: dict[str, Any] = field(default_factory=fresh_macro_policy_library)
    controllability_model: dict[str, Any] = field(default_factory=fresh_controllability_model)
    progress_value_model: dict[str, Any] = field(default_factory=fresh_progress_value_model)
    coordinate_affordances: dict[str, Any] = field(default_factory=fresh_coordinate_affordances)
    context_state: dict[str, Any] = field(default_factory=fresh_context_state)

    @classmethod
    def fresh(
        cls,
        hidden_dim: int,
        batch_size: int = 1,
        device: DeviceLike = AUTO_DEVICE,
    ) -> "PersistentMemoryState":
        target_device = resolve_device(device)
        return cls(
            latent=torch.zeros(batch_size, hidden_dim, device=target_device),
            private_token=torch.full((batch_size,), PRIVATE_NONE, dtype=torch.long, device=target_device),
            tick=0,
            event_journal=[],
            transition_graph=_fresh_transition_graph(),
            semantic_memory=_fresh_semantic_memory(),
            plastic_memory=_fresh_plastic_memory(),
            hypothesis_posterior=fresh_hypothesis_posterior(),
            active_theory_set=fresh_active_theory_set(),
            goal_posterior=fresh_goal_posterior(),
            macro_policy_library=fresh_macro_policy_library(),
            controllability_model=fresh_controllability_model(),
            progress_value_model=fresh_progress_value_model(),
            coordinate_affordances=fresh_coordinate_affordances(),
            context_state=fresh_context_state(),
        )

    @classmethod
    def load(
        cls,
        path: str | Path,
        hidden_dim: int,
        batch_size: int = 1,
        device: DeviceLike = AUTO_DEVICE,
    ) -> "PersistentMemoryState":
        target_device = resolve_device(device)
        path = Path(path)
        if not path.exists():
            return cls.fresh(hidden_dim, batch_size=batch_size, device=target_device)
        payload = torch.load(path, map_location=target_device)
        latent = payload["latent"].to(target_device).float()
        private_token = payload["private_token"].to(target_device).long()
        if latent.ndim == 1:
            latent = latent.view(1, -1)
        if private_token.ndim == 0:
            private_token = private_token.view(1)
        if latent.shape[-1] != hidden_dim:
            raise ValueError(f"memory latent width {latent.shape[-1]} does not match model hidden_dim {hidden_dim}")
        if latent.shape[0] != batch_size:
            latent = latent[:1].repeat(batch_size, 1)
        if private_token.shape[0] != batch_size:
            private_token = private_token[:1].repeat(batch_size)
        return cls(
            latent=latent,
            private_token=private_token,
            tick=int(payload.get("tick", 0)),
            event_journal=[dict(item) for item in payload.get("event_journal", [])],
            transition_graph=dict(payload.get("transition_graph", _fresh_transition_graph())),
            semantic_memory=dict(payload.get("semantic_memory", _fresh_semantic_memory())),
            plastic_memory=dict(payload.get("plastic_memory", _fresh_plastic_memory())),
            hypothesis_posterior=dict(payload.get("hypothesis_posterior", fresh_hypothesis_posterior())),
            active_theory_set=dict(payload.get("active_theory_set", fresh_active_theory_set())),
            goal_posterior=dict(payload.get("goal_posterior", fresh_goal_posterior())),
            macro_policy_library=dict(payload.get("macro_policy_library", fresh_macro_policy_library())),
            controllability_model=dict(payload.get("controllability_model", fresh_controllability_model())),
            progress_value_model=dict(payload.get("progress_value_model", fresh_progress_value_model())),
            coordinate_affordances=dict(payload.get("coordinate_affordances", fresh_coordinate_affordances())),
            context_state=dict(payload.get("context_state", fresh_context_state())),
        )

    def update(self, latent: torch.Tensor, private_token: torch.Tensor, tick: int) -> None:
        target_device = self.latent.device
        self.latent = latent.detach().to(target_device).clone()
        self.private_token = private_token.detach().to(target_device).clone().long()
        self.tick = int(tick)

    def append_event(
        self,
        *,
        tick: int,
        observation: Mapping[str, Any],
        action: int | str,
        next_observation: Mapping[str, Any],
        metadata: Mapping[str, Any] | None = None,
        prediction_error: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        delta = observation_delta(observation, next_observation)
        perception = frame_perception_payload(
            observation,
            next_observation,
            action=action,
            tick=int(tick),
        )
        record = {
            "schema": "runtime_event_journal_v1",
            "tick": int(tick),
            "observation": observation_snapshot(observation),
            "action": str(action),
            "next_observation": observation_snapshot(next_observation),
            "delta": delta,
            "metadata": _json_safe(dict(metadata or {})),
            "prediction_error": _json_safe(dict(prediction_error or {})),
        }
        if perception is not None:
            record["perception"] = _json_safe(perception)
        self.event_journal.append(record)
        update_transition_graph(
            self.transition_graph,
            tick=int(tick),
            observation=observation,
            action=action,
            next_observation=next_observation,
            metadata=metadata,
            delta=delta,
        )
        update_semantic_memory(
            self.semantic_memory,
            tick=int(tick),
            action=action,
            delta=delta,
            metadata=metadata,
            observation=observation,
            next_observation=next_observation,
        )
        update_plastic_memory(
            self.plastic_memory,
            tick=int(tick),
            observation=observation,
            action=action,
            next_observation=next_observation,
            delta=delta,
            metadata=metadata,
            prediction_error=prediction_error,
        )
        update_hypothesis_posterior(self.hypothesis_posterior, record)
        record["hypothesis_trace"] = _json_safe(event_hypothesis_trace(self.hypothesis_posterior, record))
        maintain_active_theory_set(self.active_theory_set, self.hypothesis_posterior)
        update_goal_posterior(self.goal_posterior, record)
        mirror_goal_posterior_to_semantic_memory(
            self.semantic_memory,
            self.goal_posterior,
            tick=int(tick),
        )
        update_runtime_coordinate_affordances(
            self.coordinate_affordances,
            observation=observation,
            next_observation=next_observation,
            metadata=metadata,
        )
        update_controllability_model(
            self.controllability_model,
            self.semantic_memory,
            tick=int(tick),
            observation=observation,
            action=action,
            next_observation=next_observation,
            delta=delta,
            capacity=int(self.plastic_memory.get("capacity", 64)),
        )
        update_progress_value_model(
            self.progress_value_model,
            event_journal=self.event_journal,
            event=record,
        )
        consolidate_event_journal(self, record)
        apply_boundary_transform(self, record)
        return record

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "latent": self.latent.detach().cpu(),
                "private_token": self.private_token.detach().cpu(),
                "tick": self.tick,
                "event_journal": _json_safe(self.event_journal),
                "transition_graph": _json_safe(self.transition_graph),
                "semantic_memory": _json_safe(self.semantic_memory),
                "plastic_memory": _json_safe(self.plastic_memory),
                "hypothesis_posterior": _json_safe(self.hypothesis_posterior),
                "active_theory_set": _json_safe(self.active_theory_set),
                "goal_posterior": _json_safe(self.goal_posterior),
                "macro_policy_library": _json_safe(self.macro_policy_library),
                "controllability_model": _json_safe(self.controllability_model),
                "progress_value_model": _json_safe(self.progress_value_model),
                "coordinate_affordances": _json_safe(self.coordinate_affordances),
                "context_state": _json_safe(self.context_state),
                "fast_adaptation": _json_safe(fast_adaptation_contract()),
                "format": "persistent_differentiable_tensor_memory_v1",
            },
            path,
        )
