from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from .causal_hypotheses import posterior_predictive_mixture
from .persistent_memory import canonical_observation_hash


def _available_actions(mask: Any, action_count: int) -> list[int]:
    if mask is None:
        return list(range(action_count))
    if isinstance(mask, torch.Tensor):
        mask = mask.detach().cpu().tolist()
    if isinstance(mask, (list, tuple)):
        return [idx for idx, enabled in enumerate(mask[:action_count]) if bool(enabled)]
    return list(range(action_count))


def _softmax_scores(logits: torch.Tensor) -> list[float]:
    vector = logits.detach().float().view(-1)
    probs = torch.softmax(vector, dim=-1)
    return [float(item) for item in probs.tolist()]


def _has_current_edge_indexes(graph: Mapping[str, Any]) -> bool:
    edges = graph.get("edges", {})
    return "edge_index_count" in graph and int(graph.get("edge_index_count", -1)) == len(edges)


def _outgoing_edges(graph: Mapping[str, Any], state_hash: str, action: int) -> list[Mapping[str, Any]]:
    action_text = str(action)
    action_index = graph.get("outgoing_action_edges", {}).get(state_hash, {}).get(action_text)
    if isinstance(action_index, (list, tuple)):
        edges = graph.get("edges", {})
        return [edges[edge_id] for edge_id in action_index if edge_id in edges]
    edge_ids = graph.get("outgoing_edges", {}).get(state_hash)
    if isinstance(edge_ids, (list, tuple)):
        edges = graph.get("edges", {})
        return [
            edges[edge_id]
            for edge_id in edge_ids
            if edge_id in edges and str(edges[edge_id].get("action")) == action_text
        ]
    if _has_current_edge_indexes(graph):
        return []
    return [
        edge
        for edge in graph.get("edges", {}).values()
        if str(edge.get("from")) == state_hash and str(edge.get("action")) == action_text
    ]


def _all_outgoing_edges(graph: Mapping[str, Any], state_hash: str) -> list[Mapping[str, Any]]:
    edge_ids = graph.get("outgoing_edges", {}).get(state_hash)
    if isinstance(edge_ids, (list, tuple)):
        edges = graph.get("edges", {})
        return [edges[edge_id] for edge_id in edge_ids if edge_id in edges]
    if _has_current_edge_indexes(graph):
        return []
    return [
        edge
        for edge in graph.get("edges", {}).values()
        if str(edge.get("from")) == state_hash
    ]


def _action_int(edge: Mapping[str, Any]) -> int | None:
    try:
        return int(str(edge.get("action")))
    except ValueError:
        return None


def _is_undo_action(action: int) -> bool:
    return int(action) == 7


def _undo_entry(graph: Mapping[str, Any], action: int) -> Mapping[str, Any]:
    if not _is_undo_action(action):
        return {}
    return graph.get("undo_actions", {}).get(str(action), {})


def _undo_confidence(graph: Mapping[str, Any], action: int) -> float:
    entry = _undo_entry(graph, action)
    support = int(entry.get("support", 0))
    if support <= 0:
        return 0.0
    return float(max(float(entry.get("confidence", 0.0)), min(0.99, support / max(support + 1.0, 1.0))))


def _last_edge_enters_state(graph: Mapping[str, Any], state_hash: str) -> bool:
    last_edge_id = graph.get("last_edge_id")
    if not last_edge_id:
        return False
    edge = graph.get("edges", {}).get(str(last_edge_id), {})
    if not edge:
        return False
    if bool(edge.get("no_op", False)):
        return False
    return str(edge.get("to")) == state_hash and str(edge.get("from")) != state_hash


def _undo_probe_available(graph: Mapping[str, Any], state_hash: str, action: int) -> bool:
    if not _is_undo_action(action):
        return False
    if _outgoing_edges(graph, state_hash, action):
        return False
    return _last_edge_enters_state(graph, state_hash)


def _undo_value(graph: Mapping[str, Any], state_hash: str, action: int) -> float:
    confidence = _undo_confidence(graph, action)
    if confidence > 0.0:
        support = min(int(_undo_entry(graph, action).get("support", 0)), 4) / 4.0
        return float(confidence * (0.75 + support))
    if _undo_probe_available(graph, state_hash, action):
        return 0.20
    return 0.0


def _semantic_value(semantic_memory: Mapping[str, Any], action: int) -> float:
    action_text = str(action)
    affordance = semantic_memory.get("affordances", {}).get(action_text, {})
    action_fact = semantic_memory.get("action_facts", {}).get(action_text, {})
    mean_score = float(affordance.get("mean_score_delta", action_fact.get("score_delta_sum", 0.0)))
    change_rate = float(affordance.get("change_rate", 0.0))
    terminal_rate = float(affordance.get("terminal_trials", 0.0)) / max(float(affordance.get("trials", 0.0)), 1.0)
    return float(max(mean_score, 0.0) + 0.35 * change_rate + 0.75 * terminal_rate)


def _hypothesis_value(hypothesis_posterior: Mapping[str, Any], action: int) -> float:
    action_text = str(action)
    total = 0.0
    for hypothesis in hypothesis_posterior.get("hypotheses", {}).values():
        if str(hypothesis.get("action")) != action_text:
            continue
        family = str(hypothesis.get("family", ""))
        posterior = float(hypothesis.get("posterior", 0.0))
        support = int(hypothesis.get("support", 0))
        if family != "no_op":
            total += posterior * min(support, 4) / 4.0
    return float(total)


def _macro_plan_values(
    macro_policy_library: Mapping[str, Any],
    *,
    available_actions: list[int],
) -> dict[int, dict[str, Any]]:
    available = {int(action) for action in available_actions}
    plans: dict[int, dict[str, Any]] = {}
    for macro in macro_policy_library.get("macros", {}).values():
        sequence = [str(item) for item in macro.get("action_sequence", [])]
        if not sequence:
            continue
        try:
            first_action = int(sequence[0])
        except ValueError:
            continue
        if first_action not in available:
            continue
        confidence = _clamp01(float(macro.get("confidence", 0.0)))
        support_count = max(int(macro.get("support", 0)), 0)
        support = min(support_count, 6) / 6.0
        cross_context_support = max(int(macro.get("cross_context_support", 0)), 0)
        cross_context = min(cross_context_support, 4) / 4.0
        transfer_ready = bool(macro.get("transfer_ready", False))
        transfer = 0.35 if transfer_ready else 0.0
        mean_score = max(float(macro.get("mean_score_delta", 0.0)), 0.0)
        length = max(int(macro.get("length", len(sequence))), 1)
        length_discount = 0.88 ** max(length - 1, 0)
        value = float((confidence * (0.65 + 0.70 * support + 0.25 * cross_context + transfer) + 0.20 * mean_score) * length_discount)
        plan = {
            "source": "macro_policy_library",
            "macro_id": str(macro.get("id", "")),
            "objective": str(macro.get("objective", "")),
            "action_sequence": list(sequence),
            "next_action": int(first_action),
            "remaining_action_sequence": list(sequence[1:]),
            "path_length": int(length),
            "support": int(support_count),
            "success_count": int(macro.get("success_count", 0)),
            "failure_count": int(macro.get("failure_count", 0)),
            "confidence": float(confidence),
            "transfer_ready": bool(transfer_ready),
            "cross_context_support": int(cross_context_support),
            "mean_score_delta": float(macro.get("mean_score_delta", 0.0)),
            "source_event_ticks": [int(item) for item in macro.get("source_event_ticks", [])],
            "context_keys": [str(item) for item in macro.get("context_keys", [])],
            "value": float(value),
        }
        existing = plans.get(first_action)
        if existing is None or (
            value,
            confidence,
            support_count,
            -length,
            str(macro.get("id", "")),
        ) > (
            float(existing["value"]),
            float(existing["confidence"]),
            int(existing["support"]),
            -int(existing["path_length"]),
            str(existing["macro_id"]),
        ):
            plans[first_action] = plan
    return plans


def _progress_backup_value(progress_value_model: Mapping[str, Any], action: int) -> float:
    entry = progress_value_model.get("action_values", {}).get(str(action), {})
    mean_value = float(entry.get("mean_value", 0.0))
    confidence = float(entry.get("confidence", 0.0))
    support = min(int(entry.get("support", 0)), 4) / 4.0
    return float(max(mean_value, 0.0) * (0.50 + confidence + support))


def _posterior_mixture_value(mixture: Mapping[str, Any]) -> float:
    if not bool(mixture.get("mixture_available", False)):
        return 0.0
    action_mass = _clamp01(float(mixture.get("action_posterior_mass", 0.0)))
    change_probability = _clamp01(float(mixture.get("expected_change_probability", 0.0)))
    certainty = 1.0 - _clamp01(float(mixture.get("uncertainty", 1.0)))
    return float(action_mass * (0.20 + 0.55 * change_probability + 0.25 * certainty))


def _progress_target_values(graph: Mapping[str, Any]) -> dict[str, float]:
    indexed = graph.get("progress_targets", {})
    if isinstance(indexed, Mapping) and indexed:
        targets: dict[str, float] = {}
        for target, entry in indexed.items():
            if isinstance(entry, Mapping):
                value = float(entry.get("value", 0.0))
            else:
                value = float(entry)
            if value > 0.0:
                targets[str(target)] = value
        return targets

    targets: dict[str, float] = {}
    for edge in graph.get("edges", {}).values():
        count = max(float(edge.get("count", 1.0)), 1.0)
        score_delta = max(float(edge.get("score_delta_sum", 0.0)) / count, 0.0)
        terminal_rate = float(edge.get("terminal_count", 0.0)) / count
        if score_delta <= 0.0 and terminal_rate <= 0.0:
            continue
        target = str(edge.get("to", ""))
        if not target:
            continue
        value = score_delta + 1.25 * terminal_rate
        targets[target] = max(float(targets.get(target, 0.0)), float(value))
    return targets


def _graph_plan_values(
    graph: Mapping[str, Any],
    state_hash: str,
    *,
    available_actions: list[int],
    max_depth: int = 8,
) -> dict[int, dict[str, Any]]:
    target_values = _progress_target_values(graph)
    if not target_values:
        return {}

    available = {int(action) for action in available_actions}
    queue: list[tuple[str, list[str], list[str], list[str]]] = [(state_hash, [], [], [])]
    seen = {state_hash}
    plans: dict[int, dict[str, Any]] = {}
    while queue:
        node_hash, action_path, edge_path, state_path = queue.pop(0)
        if action_path and node_hash in target_values:
            first_action = int(action_path[0])
            path_length = len(action_path)
            discount = 0.82 ** max(path_length - 1, 0)
            value = float(target_values[node_hash] * discount)
            plan = {
                "target_state": node_hash,
                "target_value": float(target_values[node_hash]),
                "value": value,
                "path_length": int(path_length),
                "action_sequence": list(action_path),
                "edge_sequence": list(edge_path),
                "state_sequence": list(state_path + [node_hash]),
            }
            existing = plans.get(first_action)
            if existing is None or (value, -path_length) > (float(existing["value"]), -int(existing["path_length"])):
                plans[first_action] = plan
            continue
        if len(action_path) >= max_depth:
            continue
        outgoing = _all_outgoing_edges(graph, node_hash)
        outgoing.sort(
            key=lambda edge: (
                bool(edge.get("no_op", False)),
                bool(edge.get("loop_observed", False)),
                str(edge.get("action", "")),
                str(edge.get("to", "")),
            )
        )
        for edge in outgoing:
            action = _action_int(edge)
            if action is None:
                continue
            if not action_path and action not in available:
                continue
            next_hash = str(edge.get("to", ""))
            if not next_hash or next_hash in seen:
                continue
            if bool(edge.get("no_op", False)) and next_hash == node_hash:
                continue
            seen.add(next_hash)
            queue.append(
                (
                    next_hash,
                    action_path + [str(action)],
                    edge_path + [str(edge.get("id", ""))],
                    state_path + [node_hash],
                )
            )
    return plans


def _clamp01(value: float) -> float:
    return float(max(0.0, min(1.0, value)))


def _edge_count(edge: Mapping[str, Any]) -> float:
    return max(float(edge.get("count", 1.0)), 1.0)


def _observed_graph_futures(edges: list[Mapping[str, Any]], action: int) -> list[dict[str, Any]]:
    if not edges:
        return []
    total_count = sum(_edge_count(edge) for edge in edges)
    max_fraction = max(_edge_count(edge) for edge in edges) / max(total_count, 1.0)
    ensemble_disagreement = _clamp01(1.0 - max_fraction)
    futures: list[dict[str, Any]] = []
    for edge in edges:
        count = _edge_count(edge)
        empirical_mass = count / max(total_count, 1.0)
        confidence = _clamp01(empirical_mass * count / (count + 1.0))
        uncertainty = _clamp01((1.0 - confidence) * 0.55 + ensemble_disagreement * 0.45)
        futures.append(
            {
                "schema": "runtime_imagined_future_v1",
                "source": "observed_graph",
                "risk_order": 0,
                "risk_source": "observed_graph",
                "action": int(action),
                "edge_id": str(edge.get("id", "")),
                "target_state": str(edge.get("to", "")),
                "posterior_mass": float(empirical_mass),
                "confidence": float(confidence),
                "uncertainty": float(uncertainty),
                "sigma": float(uncertainty),
                "ensemble_disagreement": float(ensemble_disagreement),
                "support": int(count),
            }
        )
    return futures


def _symbolic_hypothesis_futures(
    hypothesis_posterior: Mapping[str, Any],
    action: int,
) -> list[dict[str, Any]]:
    action_text = str(action)
    futures: list[dict[str, Any]] = []
    for hypothesis in hypothesis_posterior.get("hypotheses", {}).values():
        if str(hypothesis.get("action")) != action_text:
            continue
        posterior = _clamp01(float(hypothesis.get("posterior", 0.0)))
        support = max(int(hypothesis.get("support", 0)), 0)
        counterexamples = max(int(hypothesis.get("counterexamples", 0)), 0)
        failed = max(int(hypothesis.get("failed_predictions", 0)), 0)
        evidence_confidence = support / max(support + counterexamples + failed + 1.0, 1.0)
        confidence = _clamp01(posterior * evidence_confidence)
        uncertainty = _clamp01(1.0 - confidence)
        verified = support >= 2 and counterexamples == 0 and failed == 0
        source = "verified_symbolic" if verified else "symbolic_hypothesis"
        risk_order = 1 if verified else 2
        futures.append(
            {
                "schema": "runtime_imagined_future_v1",
                "source": source,
                "risk_order": int(risk_order),
                "risk_source": source,
                "action": int(action),
                "hypothesis_id": str(hypothesis.get("id", "")),
                "family": str(hypothesis.get("family", "")),
                "selector": dict(hypothesis.get("selector", {})),
                "transform": dict(hypothesis.get("transform", {})),
                "posterior_mass": float(posterior),
                "confidence": float(confidence),
                "uncertainty": float(uncertainty),
                "sigma": float(uncertainty),
                "support": int(support),
                "counterexamples": int(counterexamples),
                "failed_predictions": int(failed),
            }
        )
    futures.sort(
        key=lambda item: (
            float(item["confidence"]),
            float(item["posterior_mass"]),
            int(item["support"]),
            str(item.get("hypothesis_id", "")),
        ),
        reverse=True,
    )
    return futures[:8]


def _adapter_neural_future(
    plastic_memory: Mapping[str, Any],
    action: int,
    *,
    policy: float,
) -> dict[str, Any] | None:
    adapter = plastic_memory.get("self_supervised_adapter", {})
    if not isinstance(adapter, Mapping):
        return None
    updates = int(adapter.get("updates", 0))
    if updates <= 0:
        return None
    action_predictions = adapter.get("neural_action_predictions", adapter.get("action_predictions", {}))
    action_prediction: Mapping[str, Any] = {}
    if isinstance(action_predictions, Mapping):
        if action_predictions and str(action) not in action_predictions:
            return None
        raw_prediction = action_predictions.get(str(action), {})
        if isinstance(raw_prediction, Mapping):
            action_prediction = raw_prediction
    metrics = dict(adapter.get("latest_metrics", {}) or {})
    metrics.update(dict(action_prediction.get("metrics", {}) or {}))
    losses = dict(adapter.get("latest_losses", {}) or {})
    losses.update(dict(action_prediction.get("losses", {}) or {}))
    if not isinstance(metrics, Mapping) or not isinstance(losses, Mapping):
        return None
    updates = int(action_prediction.get("updates", updates))
    loss_values = [
        float(losses[key])
        for key in ("next_delta", "inverse_action", "no_op_change", "object_persistence")
        if key in losses and math.isfinite(float(losses[key]))
    ]
    if not loss_values:
        return None
    mean_loss = sum(max(value, 0.0) for value in loss_values) / max(len(loss_values), 1)
    prediction_confidence = 1.0 / (1.0 + mean_loss)
    evidence_confidence = min(updates, 16) / 16.0
    change_probability = _clamp01(float(metrics.get("predicted_change_probability", policy)))
    persistence_probability = _clamp01(float(metrics.get("predicted_object_persistence", 0.5)))
    agreement = 1.0 - min(abs(change_probability - 0.5), abs(persistence_probability - 0.5)) * 2.0
    confidence = _clamp01(0.50 * prediction_confidence + 0.30 * evidence_confidence + 0.20 * (1.0 - agreement))
    uncertainty = _clamp01(1.0 - confidence)
    return {
        "schema": "runtime_imagined_future_v1",
        "source": "ensemble_agreed_neural",
        "risk_order": 2,
        "risk_source": "ensemble_agreed_neural",
        "action": int(action),
        "posterior_mass": 0.0,
        "confidence": float(confidence),
        "uncertainty": float(uncertainty),
        "sigma": float(uncertainty),
        "ensemble_disagreement": float(agreement),
        "support": int(updates),
        "adapter_updates": int(updates),
        "predicted_change_probability": float(change_probability),
        "predicted_object_persistence": float(persistence_probability),
        "mean_prediction_loss": float(mean_loss),
    }


def _adapter_prediction(
    plastic_memory: Mapping[str, Any],
    action: int,
    *,
    policy: float,
) -> dict[str, Any] | None:
    adapter = plastic_memory.get("self_supervised_adapter", {})
    if not isinstance(adapter, Mapping):
        return None
    adapter_updates = max(int(adapter.get("updates", 0)), 0)
    if adapter_updates <= 0:
        return None

    action_prediction: Mapping[str, Any] = {}
    action_predictions = adapter.get("neural_action_predictions", adapter.get("action_predictions", {}))
    if isinstance(action_predictions, Mapping):
        if action_predictions and str(action) not in action_predictions:
            return None
        raw_prediction = action_predictions.get(str(action), {})
        if isinstance(raw_prediction, Mapping):
            action_prediction = raw_prediction

    metrics = dict(adapter.get("latest_metrics", {}) or {})
    metrics.update(dict(action_prediction.get("metrics", {}) or {}))
    losses = dict(adapter.get("latest_losses", {}) or {})
    losses.update(dict(action_prediction.get("losses", {}) or {}))
    if not isinstance(metrics, Mapping) or not isinstance(losses, Mapping):
        return None
    loss_values = [
        _finite_float(losses[key])
        for key in ("next_delta", "inverse_action", "no_op_change", "object_persistence", "total")
        if key in losses and math.isfinite(_finite_float(losses[key], float("nan")))
    ]
    if not loss_values:
        return None
    updates = max(int(action_prediction.get("updates", adapter_updates)), 0)
    mean_loss = sum(max(value, 0.0) for value in loss_values) / max(len(loss_values), 1)
    prediction_confidence = 1.0 / (1.0 + mean_loss)
    evidence_confidence = min(updates, 24) / 24.0
    change_probability = _clamp01(_finite_float(metrics.get("predicted_change_probability", policy), policy))
    persistence_probability = _clamp01(_finite_float(metrics.get("predicted_object_persistence", 0.5), 0.5))
    progress_probability = _clamp01(
        _finite_float(
            metrics.get(
                "predicted_progress_probability",
                metrics.get("predicted_goal_progress_probability", max(change_probability - 0.15, 0.0)),
            ),
            0.0,
        )
    )
    score_delta = max(
        _finite_float(metrics.get("predicted_score_delta", metrics.get("predicted_reward", metrics.get("predicted_value", 0.0)))),
        0.0,
    )
    delta_l1 = max(_finite_float(metrics.get("delta_prediction_l1", 0.0)), 0.0)
    consistency = 1.0 / (1.0 + 0.15 * delta_l1)
    confidence = _clamp01(
        0.42 * prediction_confidence
        + 0.28 * evidence_confidence
        + 0.16 * consistency
        + 0.14 * max(progress_probability, change_probability * persistence_probability)
    )
    uncertainty = _clamp01(1.0 - confidence)
    return {
        "action": int(action),
        "adapter_updates": int(updates),
        "adapter_total_updates": int(adapter_updates),
        "mean_prediction_loss": float(mean_loss),
        "prediction_confidence": float(prediction_confidence),
        "evidence_confidence": float(evidence_confidence),
        "predicted_change_probability": float(change_probability),
        "predicted_object_persistence": float(persistence_probability),
        "predicted_progress_probability": float(progress_probability),
        "predicted_score_delta": float(score_delta),
        "delta_prediction_l1": float(delta_l1),
        "confidence": float(confidence),
        "uncertainty": float(uncertainty),
        "source": "self_supervised_adapter",
    }


def _prediction_step_value(prediction: Mapping[str, Any], *, policy: float) -> float:
    change = _clamp01(_finite_float(prediction.get("predicted_change_probability", 0.0)))
    progress = _clamp01(_finite_float(prediction.get("predicted_progress_probability", 0.0)))
    persistence = _clamp01(_finite_float(prediction.get("predicted_object_persistence", 0.5)))
    score_delta = max(_finite_float(prediction.get("predicted_score_delta", 0.0)), 0.0)
    confidence = _clamp01(_finite_float(prediction.get("confidence", 0.0)))
    raw = 0.40 * change + 0.65 * progress + 0.35 * persistence + 0.25 * min(score_delta, 4.0) + 0.15 * policy
    return float(max(raw, 0.0) * confidence)


def _learned_neural_mpc_rollout(
    *,
    action: int,
    available_actions: list[int],
    policy_scores: list[float],
    plastic_memory: Mapping[str, Any],
    horizon: int = 3,
) -> dict[str, Any]:
    horizon = max(int(horizon), 2)
    predictions = {
        int(candidate): _adapter_prediction(
            plastic_memory,
            int(candidate),
            policy=policy_scores[int(candidate)] if int(candidate) < len(policy_scores) else 0.0,
        )
        for candidate in available_actions
    }
    first_prediction = predictions.get(int(action))
    if first_prediction is None:
        return {
            "schema": "runtime_learned_neural_mpc_rollout_v1",
            "source": "self_supervised_adapter",
            "action": int(action),
            "horizon": int(horizon),
            "action_sequence": [str(action)],
            "predicted_steps": [],
            "value": 0.0,
            "raw_value": 0.0,
            "uncertainty": 1.0,
            "confidence": 0.0,
            "plan_status": "hypothesis",
            "evidence_gate": "no_learned_adapter_prediction",
            "high_uncertainty_dependency": True,
            "learned_model_used": False,
        }

    sequence = [int(action)]
    predicted_steps: list[dict[str, Any]] = []
    discount = 0.72
    raw_value = 0.0
    path_confidence = 1.0
    max_depth = min(horizon, max(len(available_actions), 1))
    current_action = int(action)
    for step_index in range(max_depth):
        prediction = predictions.get(current_action)
        if prediction is None:
            break
        policy = policy_scores[current_action] if current_action < len(policy_scores) else 0.0
        step_value = _prediction_step_value(prediction, policy=policy)
        confidence = _clamp01(_finite_float(prediction.get("confidence", 0.0)))
        uncertainty = _clamp01(_finite_float(prediction.get("uncertainty", 1.0)))
        path_confidence *= max(confidence, 1.0e-6)
        raw_value += (discount ** step_index) * step_value
        predicted_steps.append(
            {
                "step": int(step_index + 1),
                "action": int(current_action),
                "source": str(prediction.get("source", "self_supervised_adapter")),
                "confidence": float(confidence),
                "uncertainty": float(uncertainty),
                "value": float(step_value),
                "predicted_change_probability": float(prediction.get("predicted_change_probability", 0.0)),
                "predicted_progress_probability": float(prediction.get("predicted_progress_probability", 0.0)),
                "predicted_object_persistence": float(prediction.get("predicted_object_persistence", 0.0)),
                "predicted_score_delta": float(prediction.get("predicted_score_delta", 0.0)),
                "mean_prediction_loss": float(prediction.get("mean_prediction_loss", 0.0)),
                "adapter_updates": int(prediction.get("adapter_updates", 0)),
            }
        )
        if step_index + 1 >= max_depth:
            break
        next_candidates = [
            candidate
            for candidate in available_actions
            if int(candidate) in predictions and predictions[int(candidate)] is not None
        ]
        if not next_candidates:
            break
        current_action = int(
            max(
                next_candidates,
                key=lambda candidate: (
                    _prediction_step_value(
                        predictions[int(candidate)] or {},
                        policy=policy_scores[int(candidate)] if int(candidate) < len(policy_scores) else 0.0,
                    ),
                    _clamp01(_finite_float((predictions[int(candidate)] or {}).get("confidence", 0.0))),
                    -int(candidate),
                ),
            )
        )
        sequence.append(current_action)

    if not predicted_steps:
        confidence = 0.0
        uncertainty = 1.0
    else:
        confidence = _clamp01(path_confidence ** (1.0 / max(len(predicted_steps), 1)))
        uncertainty = _clamp01(1.0 - confidence)
    min_updates = min((int(step.get("adapter_updates", 0)) for step in predicted_steps), default=0)
    max_loss = max((float(step.get("mean_prediction_loss", 1.0e9)) for step in predicted_steps), default=1.0e9)
    strong_evidence = bool(
        len(predicted_steps) >= 2
        and min_updates >= 8
        and max_loss <= 0.75
        and confidence >= 0.58
        and uncertainty <= 0.42
        and any(float(step.get("predicted_change_probability", 0.0)) > 0.55 for step in predicted_steps)
    )
    value = raw_value if strong_evidence else 0.0
    return {
        "schema": "runtime_learned_neural_mpc_rollout_v1",
        "source": "learned_self_supervised_adapter_latent_dynamics",
        "action": int(action),
        "horizon": int(horizon),
        "action_sequence": [str(item) for item in sequence[: len(predicted_steps)]],
        "predicted_steps": predicted_steps,
        "value": float(max(value, 0.0)),
        "raw_value": float(max(raw_value, 0.0)),
        "uncertainty": float(uncertainty),
        "confidence": float(confidence),
        "plan_status": "plan" if strong_evidence else "hypothesis",
        "evidence_gate": "learned_neural_mpc_plan" if strong_evidence else "learned_neural_mpc_hypothesis",
        "high_uncertainty_dependency": not strong_evidence,
        "learned_model_used": True,
        "min_adapter_updates": int(min_updates),
        "max_prediction_loss": float(max_loss),
        "discount": float(discount),
    }


def _speculative_future(action: int, *, policy: float) -> dict[str, Any]:
    confidence = _clamp01(0.10 + 0.20 * float(policy))
    uncertainty = _clamp01(1.0 - confidence)
    return {
        "schema": "runtime_imagined_future_v1",
        "source": "speculative_prior",
        "risk_order": 3,
        "risk_source": "speculative_prior",
        "action": int(action),
        "posterior_mass": 0.0,
        "confidence": float(confidence),
        "uncertainty": float(uncertainty),
        "sigma": float(uncertainty),
        "ensemble_disagreement": 1.0,
        "support": 0,
    }


def _imagined_futures(
    graph: Mapping[str, Any],
    hypothesis_posterior: Mapping[str, Any],
    plastic_memory: Mapping[str, Any],
    state_hash: str,
    action: int,
    *,
    policy: float,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    observed = _observed_graph_futures(_outgoing_edges(graph, state_hash, action), action)
    symbolic = _symbolic_hypothesis_futures(hypothesis_posterior, action)
    neural = []
    neural_future = _adapter_neural_future(plastic_memory, action, policy=policy)
    if neural_future is not None:
        neural = [neural_future]
    futures = observed + symbolic
    if not futures:
        futures = neural
    if not futures:
        futures = [_speculative_future(action, policy=policy)]
    futures.sort(
        key=lambda item: (
            int(item.get("risk_order", 3)),
            float(item["uncertainty"]),
            -float(item["confidence"]),
            str(item.get("edge_id", item.get("hypothesis_id", ""))),
        )
    )
    futures = futures[:8]
    best_confidence = max(float(item.get("confidence", 0.0)) for item in futures)
    best_uncertainty = min(float(item.get("uncertainty", 1.0)) for item in futures)
    posterior_mass = _clamp01(sum(float(item.get("posterior_mass", 0.0)) for item in futures))
    risk_order = min(int(item.get("risk_order", 3)) for item in futures)
    observed_present = risk_order == 0
    verified_present = risk_order == 1
    status_threshold = 0.40 if observed_present else 0.28
    if verified_present:
        status_threshold = 0.34
    plan_status = "plan" if best_uncertainty <= status_threshold and risk_order <= 1 else "hypothesis"
    high_uncertainty_dependency = bool(best_uncertainty >= 0.50 or risk_order >= 2)
    hypothesis_only = bool(plan_status != "plan")
    if plan_status == "plan":
        evidence_gate = "admissible_plan"
    elif high_uncertainty_dependency:
        evidence_gate = "uncertainty_limited_hypothesis"
    else:
        evidence_gate = "pending_verification_hypothesis"
    aggregate = {
        "imagined_confidence": float(best_confidence),
        "imagined_uncertainty": float(best_uncertainty),
        "imagined_posterior_mass": float(posterior_mass),
        "future_risk_order": int(risk_order),
        "future_source_priority": int(3 - risk_order),
        "plan_status": plan_status,
        "evidence_gate": evidence_gate,
        "high_uncertainty_dependency": high_uncertainty_dependency,
        "hypothesis_only": hypothesis_only,
        "dominant_future_source": str(futures[0].get("source", "")),
    }
    return futures, aggregate


def _goal_value(goal_posterior: Mapping[str, Any], action: int) -> float:
    del action
    posterior_mass = float(goal_posterior.get("posterior_mass", 0.0))
    top = goal_posterior.get("top_goals", [])
    if not top:
        return 0.0
    return float(min(posterior_mass, 1.0) * 0.15)


def _information_gain(
    graph: Mapping[str, Any],
    state_hash: str,
    action: int,
    *,
    available_actions: list[int],
) -> float:
    if _undo_probe_available(graph, state_hash, action):
        return 1.25
    unexplored = graph.get("unexplored_actions", {}).get(state_hash, [])
    unexplored_text = {str(item) for item in unexplored}
    if str(action) in unexplored_text:
        return 1.0
    if not _outgoing_edges(graph, state_hash, action):
        return 0.75 if action in available_actions else 0.0
    return 0.08


def _risk(graph: Mapping[str, Any], state_hash: str, action: int) -> float:
    edges = _outgoing_edges(graph, state_hash, action)
    if not edges:
        if _undo_probe_available(graph, state_hash, action):
            return 0.05
        return 0.25
    risk = 0.0
    for edge in edges:
        count = max(float(edge.get("count", 1.0)), 1.0)
        no_op = bool(edge.get("no_op", False))
        loop = bool(edge.get("loop_observed", False))
        reversible = bool(edge.get("reversible", False))
        terminal = float(edge.get("terminal_count", 0.0)) / count
        risk += (0.65 if no_op else 0.0) + (0.45 if loop else 0.0) + (0.65 * terminal) - (0.40 if reversible else 0.0)
    risk = float(max(risk / max(len(edges), 1), 0.0))
    confidence = _undo_confidence(graph, action)
    if confidence > 0.0:
        risk = max(risk - 0.45 * confidence, 0.0)
    return float(risk)


def _observed_rank(graph: Mapping[str, Any], state_hash: str, action: int) -> str:
    edges = _outgoing_edges(graph, state_hash, action)
    if not edges:
        if _undo_probe_available(graph, state_hash, action):
            return "undo_probe"
        return "speculative"
    if _undo_confidence(graph, action) > 0.0 and any(bool(edge.get("reversible", False)) for edge in edges):
        return "observed_undo"
    if any(bool(edge.get("reversible", False)) for edge in edges):
        return "observed_reversible"
    if any(not bool(edge.get("no_op", False)) for edge in edges):
        return "observed_effect"
    return "observed_no_op"


def _posterior_mixture_evidence_confidence(mixture: Mapping[str, Any]) -> float:
    if not bool(mixture.get("mixture_available", False)):
        return 0.0
    components = mixture.get("components", [])
    if not isinstance(components, list) or not components:
        return 0.0
    weighted_confidence = 0.0
    total_weight = 0.0
    for component in components:
        if not isinstance(component, Mapping):
            continue
        weight = _clamp01(float(component.get("mixture_weight", component.get("posterior_mass", 0.0))))
        support = max(float(component.get("support", 0.0)), 0.0)
        counterexamples = max(float(component.get("counterexamples", 0.0)), 0.0)
        failed = max(float(component.get("failed_predictions", 0.0)), 0.0)
        evidence = support / max(support + counterexamples + failed + 1.0, 1.0)
        weighted_confidence += weight * evidence
        total_weight += weight
    if total_weight <= 0.0:
        return 0.0
    action_mass = _clamp01(float(mixture.get("action_posterior_mass", 0.0)))
    mixture_agreement = 1.0 - _clamp01(float(mixture.get("uncertainty", 1.0)))
    evidence_confidence = weighted_confidence / max(total_weight, 1e-6)
    return float(_clamp01(action_mass * (0.35 + 0.65 * evidence_confidence) * mixture_agreement))


def _posterior_mixture_min_support(mixture: Mapping[str, Any]) -> int:
    if not bool(mixture.get("mixture_available", False)):
        return 0
    components = mixture.get("components", [])
    if not isinstance(components, list) or not components:
        return 0
    supports = [
        max(int(component.get("support", 0)), 0)
        for component in components
        if isinstance(component, Mapping)
    ]
    return min(supports) if supports else 0


def _projection_for_rollout_action(
    action: int,
    *,
    hypothesis_posterior: Mapping[str, Any],
    semantic_memory: Mapping[str, Any],
    goal_posterior: Mapping[str, Any],
    progress_value_model: Mapping[str, Any],
) -> dict[str, Any]:
    mixture = posterior_predictive_mixture(hypothesis_posterior, action=action)
    mixture_value = _posterior_mixture_value(mixture)
    evidence_confidence = _posterior_mixture_evidence_confidence(mixture)
    hypothesis_value = _hypothesis_value(hypothesis_posterior, action)
    semantic_value = _semantic_value(semantic_memory, action)
    goal_value = _goal_value(goal_posterior, action)
    progress_value = _progress_backup_value(progress_value_model, action)
    value = (
        mixture_value * (0.80 + 0.20 * evidence_confidence)
        + 0.45 * hypothesis_value
        + 0.15 * semantic_value
        + 0.15 * goal_value
        + 0.20 * progress_value
    )
    return {
        "action": int(action),
        "value": float(max(value, 0.0)),
        "mixture": mixture,
        "mixture_available": bool(mixture.get("mixture_available", False)),
        "mixture_value": float(mixture_value),
        "evidence_confidence": float(evidence_confidence),
        "hypothesis_value": float(hypothesis_value),
        "semantic_value": float(semantic_value),
        "goal_value": float(goal_value),
        "progress_value": float(progress_value),
    }


def _bounded_internal_rollouts(
    *,
    action: int,
    available_actions: list[int],
    graph_plan: Mapping[str, Any] | None,
    macro_plan: Mapping[str, Any] | None,
    hypothesis_posterior: Mapping[str, Any],
    semantic_memory: Mapping[str, Any],
    goal_posterior: Mapping[str, Any],
    progress_value_model: Mapping[str, Any],
    world_model_mixture: Mapping[str, Any],
    max_depth: int = 3,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    rollouts: list[dict[str, Any]] = []
    bounded_depth = max(2, int(max_depth))
    if graph_plan is not None:
        path_length = max(int(graph_plan.get("path_length", 1)), 1)
        uncertainty = _clamp01(0.08 + 0.04 * max(path_length - 1, 0))
        rollouts.append(
            {
                "schema": "runtime_internal_rollout_v1",
                "source": "observed_graph_rollout",
                "path_source": "transition_graph",
                "first_action": int(action),
                "action_sequence": [str(item) for item in graph_plan.get("action_sequence", [])],
                "state_sequence": [str(item) for item in graph_plan.get("state_sequence", [])],
                "edge_sequence": [str(item) for item in graph_plan.get("edge_sequence", [])],
                "target_state": str(graph_plan.get("target_state", "")),
                "target_value": float(graph_plan.get("target_value", 0.0)),
                "depth": int(path_length),
                "max_depth": int(bounded_depth),
                "value": float(graph_plan.get("value", 0.0)),
                "confidence": float(1.0 - uncertainty),
                "uncertainty": float(uncertainty),
                "posterior_mass": 1.0,
                "plan_status": "plan",
                "evidence_gate": "observed_graph_rollout",
                "high_uncertainty_dependency": False,
                "hypothesis_limited": False,
            }
        )
    if macro_plan is not None:
        confidence = _clamp01(float(macro_plan.get("confidence", 0.0)))
        uncertainty = _clamp01(1.0 - confidence)
        transfer_ready = bool(macro_plan.get("transfer_ready", False)) and confidence >= 0.50
        rollouts.append(
            {
                "schema": "runtime_internal_rollout_v1",
                "source": "macro_policy_rollout",
                "path_source": "macro_policy_library",
                "first_action": int(action),
                "macro_id": str(macro_plan.get("macro_id", "")),
                "objective": str(macro_plan.get("objective", "")),
                "action_sequence": [str(item) for item in macro_plan.get("action_sequence", [])],
                "depth": int(max(int(macro_plan.get("path_length", 1)), 1)),
                "max_depth": int(bounded_depth),
                "value": float(macro_plan.get("value", 0.0)),
                "confidence": float(confidence),
                "uncertainty": float(uncertainty),
                "posterior_mass": 0.0,
                "plan_status": "plan" if transfer_ready else "hypothesis",
                "evidence_gate": "macro_policy_transfer" if transfer_ready else "macro_policy_hypothesis",
                "high_uncertainty_dependency": not transfer_ready,
                "hypothesis_limited": not transfer_ready,
            }
        )

    first_projection = _projection_for_rollout_action(
        action,
        hypothesis_posterior=hypothesis_posterior,
        semantic_memory=semantic_memory,
        goal_posterior=goal_posterior,
        progress_value_model=progress_value_model,
    )
    first_projection["mixture"] = world_model_mixture
    first_projection["mixture_available"] = bool(world_model_mixture.get("mixture_available", False))
    first_projection["mixture_value"] = float(_posterior_mixture_value(world_model_mixture))
    first_projection["evidence_confidence"] = float(_posterior_mixture_evidence_confidence(world_model_mixture))
    if bool(world_model_mixture.get("mixture_available", False)) and bounded_depth >= 2:
        next_projections = [
            _projection_for_rollout_action(
                next_action,
                hypothesis_posterior=hypothesis_posterior,
                semantic_memory=semantic_memory,
                goal_posterior=goal_posterior,
                progress_value_model=progress_value_model,
            )
            for next_action in available_actions
        ]
        next_projections = [
            item
            for item in next_projections
            if bool(item["mixture_available"]) or float(item["hypothesis_value"]) > 0.0
        ]
        next_projections.sort(
            key=lambda item: (
                float(item["value"]),
                float(item["evidence_confidence"]),
                -int(item["action"]),
            ),
            reverse=True,
        )
        first_confidence = float(first_projection["evidence_confidence"])
        first_mass = _clamp01(float(world_model_mixture.get("action_posterior_mass", 0.0)))
        first_change = _clamp01(float(world_model_mixture.get("expected_change_probability", 0.0)))
        first_min_support = _posterior_mixture_min_support(world_model_mixture)
        for next_projection in next_projections[:3]:
            next_action = int(next_projection["action"])
            next_confidence = float(next_projection["evidence_confidence"])
            next_value = float(next_projection["value"])
            next_min_support = _posterior_mixture_min_support(next_projection["mixture"])
            path_confidence = _clamp01(first_confidence * (0.55 + 0.45 * max(next_confidence, 0.25)))
            path_uncertainty = _clamp01(1.0 - path_confidence)
            discount = 0.72
            value = float(discount * path_confidence * next_value)
            support_ready = first_min_support >= 3 and next_min_support >= 3
            plan_ready = support_ready and path_confidence >= 0.58 and path_uncertainty <= 0.42 and first_change > 0.0
            rollouts.append(
                {
                    "schema": "runtime_internal_rollout_v1",
                    "source": "posterior_ensemble_rollout",
                    "path_source": "posterior_predictive_mixture",
                    "first_action": int(action),
                    "action_sequence": [str(action), str(next_action)],
                    "depth": 2,
                    "max_depth": int(bounded_depth),
                    "discount": float(discount),
                    "value": float(value),
                    "first_step_value": float(first_projection["mixture_value"]),
                    "future_value": float(next_value),
                    "confidence": float(path_confidence),
                    "uncertainty": float(path_uncertainty),
                    "posterior_mass": float(_clamp01(first_mass + float(next_projection["mixture"].get("action_posterior_mass", 0.0)))),
                    "first_posterior_mass": float(first_mass),
                    "next_posterior_mass": float(_clamp01(float(next_projection["mixture"].get("action_posterior_mass", 0.0)))),
                    "first_min_support": int(first_min_support),
                    "next_min_support": int(next_min_support),
                    "first_evidence_confidence": float(first_confidence),
                    "next_evidence_confidence": float(next_confidence),
                    "first_expected_change_probability": float(first_change),
                    "next_expected_change_probability": float(
                        _clamp01(float(next_projection["mixture"].get("expected_change_probability", 0.0)))
                    ),
                    "plan_status": "plan" if plan_ready else "hypothesis",
                    "evidence_gate": "posterior_ensemble_rollout" if plan_ready else "posterior_ensemble_hypothesis",
                    "high_uncertainty_dependency": not plan_ready,
                    "hypothesis_limited": not plan_ready,
                }
            )

    if not rollouts:
        rollouts.append(
            {
                "schema": "runtime_internal_rollout_v1",
                "source": "speculative_rollout",
                "path_source": "speculative_prior",
                "first_action": int(action),
                "action_sequence": [str(action)],
                "depth": 1,
                "max_depth": int(bounded_depth),
                "value": 0.0,
                "confidence": 0.05,
                "uncertainty": 0.95,
                "posterior_mass": 0.0,
                "plan_status": "hypothesis",
                "evidence_gate": "uncertainty_limited_hypothesis",
                "high_uncertainty_dependency": True,
                "hypothesis_limited": True,
            }
        )

    rollouts.sort(
        key=lambda item: (
            str(item.get("plan_status")) == "plan",
            float(item.get("value", 0.0)),
            -float(item.get("uncertainty", 1.0)),
            -int(item.get("depth", 1)),
            str(item.get("source", "")),
        ),
        reverse=True,
    )
    best = rollouts[0]
    posterior_candidates = [
        item
        for item in rollouts
        if str(item.get("source")) == "posterior_ensemble_rollout"
        and str(item.get("plan_status")) == "plan"
        and not bool(item.get("high_uncertainty_dependency", True))
    ]
    posterior_rollout_value = max((float(item.get("value", 0.0)) for item in posterior_candidates), default=0.0)
    summary = {
        "internal_rollout_value": float(best.get("value", 0.0)),
        "internal_rollout_depth": int(best.get("depth", 1)),
        "internal_rollout_source": str(best.get("source", "")),
        "internal_rollout_uncertainty": float(best.get("uncertainty", 1.0)),
        "internal_rollout_plan_status": str(best.get("plan_status", "hypothesis")),
        "internal_rollout_evidence_gate": str(best.get("evidence_gate", "")),
        "internal_rollout_high_uncertainty_dependency": bool(best.get("high_uncertainty_dependency", True)),
        "internal_rollout_path_count": int(len(rollouts)),
        "posterior_rollout_value": float(posterior_rollout_value),
    }
    return rollouts[:8], summary


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(number):
        return float(default)
    return float(number)


def _normalise_particle_masses(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    total = sum(max(float(candidate.get("mass", 0.0)), 0.0) for candidate in candidates)
    if total <= 0.0:
        return []
    normalised: list[dict[str, Any]] = []
    for candidate in candidates:
        item = dict(candidate)
        item["mass"] = float(max(float(item.get("mass", 0.0)), 0.0) / total)
        normalised.append(item)
    return normalised


def _hypothesis_belief_candidates(
    hypothesis_posterior: Mapping[str, Any],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    for hypothesis in (hypothesis_posterior or {}).get("hypotheses", {}).values():
        if not isinstance(hypothesis, Mapping):
            continue
        mass = _finite_float(hypothesis.get("posterior", 0.0))
        support = max(int(hypothesis.get("support", 0)), 0)
        counterexamples = max(int(hypothesis.get("counterexamples", 0)), 0)
        failed = max(int(hypothesis.get("failed_predictions", 0)), 0)
        if mass <= 0.0 and support > 0:
            mass = 0.05 * support / max(support + counterexamples + failed + 1.0, 1.0)
        if mass <= 0.0:
            continue
        action_text = str(hypothesis.get("action", ""))
        if action_text == "":
            continue
        evidence_confidence = support / max(support + counterexamples + failed + 1.0, 1.0)
        candidates.append(
            {
                "id": str(hypothesis.get("id", "")),
                "action": action_text,
                "family": str(hypothesis.get("family", "")),
                "selector": dict(hypothesis.get("selector", {}) or {}),
                "transform": dict(hypothesis.get("transform", {}) or {}),
                "goal_test": dict(hypothesis.get("goal_test", {}) or {}),
                "mass": float(mass),
                "posterior": float(mass),
                "support": int(support),
                "counterexamples": int(counterexamples),
                "failed_predictions": int(failed),
                "evidence_confidence": float(_clamp01(evidence_confidence)),
                "description_length": _finite_float(hypothesis.get("description_length", 1.0), 1.0),
                "is_null": False,
            }
        )
    candidates.sort(
        key=lambda item: (
            float(item["mass"]),
            int(item["support"]),
            -int(item["counterexamples"]),
            str(item["id"]),
        ),
        reverse=True,
    )
    return _normalise_particle_masses(candidates[: max(int(limit), 1)])


def _goal_belief_candidates(
    goal_posterior: Mapping[str, Any],
    *,
    limit: int = 8,
) -> list[dict[str, Any]]:
    candidates: list[dict[str, Any]] = []
    seen: set[str] = set()
    for scope in ("goals", "subgoals"):
        store = (goal_posterior or {}).get(scope, {})
        if not isinstance(store, Mapping):
            continue
        for goal in store.values():
            if not isinstance(goal, Mapping):
                continue
            goal_id = str(goal.get("id", ""))
            if not goal_id or goal_id in seen:
                continue
            seen.add(goal_id)
            support = max(int(goal.get("support", 0)), 0)
            inconsistency = max(int(goal.get("inconsistency", 0)), 0)
            mass = _finite_float(goal.get("posterior", 0.0))
            if mass <= 0.0 and support > 0:
                mass = 0.05 * support / max(support + inconsistency + 1.0, 1.0)
            if mass <= 0.0:
                continue
            evidence_confidence = support / max(support + inconsistency + 1.0, 1.0)
            candidates.append(
                {
                    "id": goal_id,
                    "scope": str(goal.get("scope", scope[:-1])),
                    "kind": str(goal.get("kind", "")),
                    "selector": dict(goal.get("selector", {}) or {}),
                    "progress_test": dict(goal.get("progress_test", {}) or {}),
                    "mass": float(mass),
                    "posterior": float(mass),
                    "support": int(support),
                    "inconsistency": int(inconsistency),
                    "evidence_confidence": float(_clamp01(evidence_confidence)),
                    "description_length": _finite_float(goal.get("description_length", 1.0), 1.0),
                    "is_null": False,
                }
            )
    candidates.sort(
        key=lambda item: (
            float(item["mass"]),
            int(item["support"]),
            -int(item["inconsistency"]),
            str(item["id"]),
        ),
        reverse=True,
    )
    return _normalise_particle_masses(candidates[: max(int(limit), 1)])


def _null_hypothesis_particle() -> dict[str, Any]:
    return {
        "id": "",
        "action": "",
        "family": "unknown",
        "selector": {},
        "transform": {},
        "goal_test": {},
        "mass": 1.0,
        "posterior": 0.0,
        "support": 0,
        "counterexamples": 0,
        "failed_predictions": 0,
        "evidence_confidence": 0.0,
        "description_length": 0.0,
        "is_null": True,
    }


def _null_goal_particle() -> dict[str, Any]:
    return {
        "id": "",
        "scope": "unknown",
        "kind": "unknown_goal",
        "selector": {},
        "progress_test": {},
        "mass": 1.0,
        "posterior": 0.0,
        "support": 0,
        "inconsistency": 0,
        "evidence_confidence": 0.0,
        "description_length": 0.0,
        "is_null": True,
    }


def _goal_direct_action_match(goal: Mapping[str, Any], action: int) -> float:
    if bool(goal.get("is_null", False)):
        return 0.0
    action_text = str(action)
    selector = goal.get("selector", {})
    progress_test = goal.get("progress_test", {})
    for source in (selector, progress_test):
        if not isinstance(source, Mapping):
            continue
        if str(source.get("action", "")) == action_text:
            return 1.0
        actions = source.get("actions")
        if isinstance(actions, (list, tuple, set)) and action_text in {str(item) for item in actions}:
            return 1.0
    return 0.0


def _hypothesis_goal_alignment(hypothesis: Mapping[str, Any], goal: Mapping[str, Any]) -> float:
    if bool(goal.get("is_null", False)):
        return 0.45 if not bool(hypothesis.get("is_null", False)) else 0.0
    if bool(hypothesis.get("is_null", False)):
        return 0.25
    family = str(hypothesis.get("family", ""))
    selector = hypothesis.get("selector", {})
    goal_test = hypothesis.get("goal_test", {})
    progress_test = goal.get("progress_test", {})
    goal_selector = goal.get("selector", {})
    if not isinstance(selector, Mapping):
        selector = {}
    if not isinstance(goal_test, Mapping):
        goal_test = {}
    if not isinstance(progress_test, Mapping):
        progress_test = {}
    if not isinstance(goal_selector, Mapping):
        goal_selector = {}

    alignment = 0.30
    if str(goal_test.get("kind")) == "field_changed" or "changed_count_gt" in progress_test:
        alignment = max(alignment, 0.85 if family in {"field_change", "move_color"} else 0.25)
    if "move_color" in progress_test or "moved_object" in progress_test or str(goal_test.get("kind")) == "component_translation":
        alignment = max(alignment, 0.90 if family == "move_color" else 0.35)
    if "score_delta_gt" in progress_test or progress_test.get("terminal") is True:
        alignment = max(alignment, 0.65)
    if str(goal_test.get("kind")) == str(goal.get("kind", "")) and str(goal.get("kind", "")):
        alignment = max(alignment, 0.55)
    hyp_field = str(selector.get("field", ""))
    goal_field = str(goal_selector.get("field", progress_test.get("field", "")))
    if hyp_field and goal_field and hyp_field == goal_field:
        alignment = max(alignment, 0.95)
    return float(_clamp01(alignment))


def _posterior_tree_particle_scores(
    *,
    action: int,
    hypothesis: Mapping[str, Any],
    goal: Mapping[str, Any],
    projection: Mapping[str, Any],
    graph_plan: Mapping[str, Any] | None,
    macro_plan: Mapping[str, Any] | None,
) -> tuple[float, float]:
    hypothesis_matches = not bool(hypothesis.get("is_null", False)) and str(hypothesis.get("action", "")) == str(action)
    goal_matches = _goal_direct_action_match(goal, action) > 0.0
    alignment = _hypothesis_goal_alignment(hypothesis, goal)
    hypothesis_mass = float(hypothesis.get("mass", 0.0)) if hypothesis_matches else 0.0
    goal_mass = 0.0 if bool(goal.get("is_null", False)) else float(goal.get("mass", 0.0))
    hypothesis_confidence = float(hypothesis.get("evidence_confidence", 0.0)) if hypothesis_matches else 0.0
    goal_confidence = float(goal.get("evidence_confidence", 0.0)) if not bool(goal.get("is_null", False)) else 0.0
    change_bonus = 0.25 if str(hypothesis.get("family", "")) != "no_op" else -0.10

    projection_value = max(float(projection.get("value", 0.0)), 0.0)
    solution = 0.45 * projection_value
    if hypothesis_matches:
        solution += hypothesis_mass * (0.40 + 0.35 * hypothesis_confidence + change_bonus) * max(alignment, 0.20)
    if goal_matches:
        solution += goal_mass * (0.35 + 0.40 * goal_confidence)
    elif hypothesis_matches and goal_mass > 0.0:
        solution += 0.45 * goal_mass * alignment * (0.25 + 0.45 * goal_confidence)
    if graph_plan is not None:
        solution += 0.50 * max(float(graph_plan.get("value", 0.0)), 0.0)
    if macro_plan is not None:
        solution += 0.45 * max(float(macro_plan.get("value", 0.0)), 0.0)

    hypothesis_uncertainty = 1.0 - hypothesis_confidence
    goal_uncertainty = 1.0 - goal_confidence if goal_mass > 0.0 else 0.0
    experiment = 0.0
    if hypothesis_matches:
        experiment += hypothesis_mass * hypothesis_uncertainty * (0.55 + 0.35 * alignment)
    if goal_mass > 0.0:
        experiment += goal_mass * goal_uncertainty * (0.25 + (0.45 * alignment if hypothesis_matches else 0.10))
    if hypothesis_matches and goal_mass > 0.0:
        experiment += 0.20 * hypothesis_mass * goal_mass * abs(hypothesis_confidence - goal_confidence)
    return float(max(solution, 0.0)), float(max(experiment, 0.0))


def _posterior_tree_empty(
    *,
    available_actions: list[int],
    horizon: int,
    simulations: int,
) -> dict[str, Any]:
    action_values = {
        str(action): {
            "action": int(action),
            "visits": 0,
            "prior_value": 0.0,
            "q_value": 0.0,
            "solution_value": 0.0,
            "experiment_value": 0.0,
            "action_value": 0.0,
        }
        for action in available_actions
    }
    return {
        "schema": "runtime_posterior_conditioned_pomcp_search_v1",
        "belief_tree": "root_action_ucb",
        "sampled_belief_particles": [],
        "simulations": 0,
        "requested_simulations": int(simulations),
        "horizon": int(horizon),
        "root_visit_count": 0,
        "action_values": action_values,
        "selected_action": int(available_actions[0]) if available_actions else -1,
        "selected_hypothesis_ids": [],
        "selected_goal_ids": [],
        "solution_search_value": 0.0,
        "experiment_search_value": 0.0,
        "selected_value": 0.0,
        "joint_posterior_conditioned": False,
        "plan_status": "hypothesis",
        "evidence_gate": "no_posterior_belief_particles",
        "high_uncertainty_dependency": True,
    }


def _posterior_conditioned_tree_search(
    *,
    available_actions: list[int],
    hypothesis_posterior: Mapping[str, Any],
    semantic_memory: Mapping[str, Any],
    goal_posterior: Mapping[str, Any],
    progress_value_model: Mapping[str, Any],
    graph_plans: Mapping[int, Mapping[str, Any]],
    macro_plans: Mapping[int, Mapping[str, Any]],
    horizon: int = 3,
    simulations: int = 48,
) -> tuple[dict[str, Any], dict[int, float]]:
    horizon = max(int(horizon), 1)
    simulations = max(int(simulations), 1)
    if not available_actions:
        return _posterior_tree_empty(available_actions=[], horizon=horizon, simulations=simulations), {}

    real_hypotheses = _hypothesis_belief_candidates(hypothesis_posterior)
    real_goals = _goal_belief_candidates(goal_posterior)
    if not real_hypotheses and not real_goals:
        search = _posterior_tree_empty(
            available_actions=available_actions,
            horizon=horizon,
            simulations=simulations,
        )
        return search, {int(action): 0.0 for action in available_actions}

    hypotheses = real_hypotheses if real_hypotheses else [_null_hypothesis_particle()]
    goals = real_goals if real_goals else [_null_goal_particle()]
    projections = {
        int(action): _projection_for_rollout_action(
            int(action),
            hypothesis_posterior=hypothesis_posterior,
            semantic_memory=semantic_memory,
            goal_posterior=goal_posterior,
            progress_value_model=progress_value_model,
        )
        for action in available_actions
    }
    particles: list[dict[str, Any]] = []
    for hypothesis in hypotheses:
        for goal in goals:
            weight = float(hypothesis.get("mass", 0.0)) * float(goal.get("mass", 0.0))
            if weight <= 0.0:
                continue
            particles.append(
                {
                    "hypothesis": hypothesis,
                    "goal": goal,
                    "mass": float(weight),
                    "weight": float(weight),
                }
            )
    particles.sort(
        key=lambda item: (
            float(item["weight"]),
            int(item["hypothesis"].get("support", 0)),
            int(item["goal"].get("support", 0)),
            str(item["hypothesis"].get("id", "")),
            str(item["goal"].get("id", "")),
        ),
        reverse=True,
    )
    particles = particles[:16]
    particles = _normalise_particle_masses(particles)
    if not particles:
        search = _posterior_tree_empty(
            available_actions=available_actions,
            horizon=horizon,
            simulations=simulations,
        )
        return search, {int(action): 0.0 for action in available_actions}

    prior_values: dict[int, float] = {}
    for action in available_actions:
        projection = projections[int(action)]
        graph_value = max(float(graph_plans.get(int(action), {}).get("value", 0.0)), 0.0)
        macro_value = max(float(macro_plans.get(int(action), {}).get("value", 0.0)), 0.0)
        prior_values[int(action)] = float(max(float(projection.get("value", 0.0)), 0.0) + 0.30 * graph_value + 0.25 * macro_value)

    stats: dict[int, dict[str, float]] = {
        int(action): {
            "visits": 0.0,
            "value_sum": 0.0,
            "solution_sum": 0.0,
            "experiment_sum": 0.0,
        }
        for action in available_actions
    }
    discount = 0.70
    exploration = 0.55
    future_prior = max(prior_values.values(), default=0.0)
    for simulation_index in range(simulations):
        particle = particles[simulation_index % len(particles)]
        total_visits = max(1.0, float(simulation_index + 1))
        if simulation_index < len(available_actions):
            chosen_action = int(available_actions[simulation_index])
        else:
            chosen_action = max(
                available_actions,
                key=lambda candidate: (
                    (
                        (stats[int(candidate)]["value_sum"] / stats[int(candidate)]["visits"])
                        if stats[int(candidate)]["visits"] > 0.0
                        else prior_values[int(candidate)]
                    )
                    + exploration
                    * math.sqrt(math.log(total_visits + 1.0) / max(stats[int(candidate)]["visits"], 1.0))
                    + 0.15 * prior_values[int(candidate)],
                    prior_values[int(candidate)],
                    -int(candidate),
                ),
            )
        solution, experiment = _posterior_tree_particle_scores(
            action=int(chosen_action),
            hypothesis=particle["hypothesis"],
            goal=particle["goal"],
            projection=projections[int(chosen_action)],
            graph_plan=graph_plans.get(int(chosen_action)),
            macro_plan=macro_plans.get(int(chosen_action)),
        )
        rollout_tail = 0.0
        for depth in range(1, horizon):
            rollout_tail += (discount ** depth) * 0.35 * future_prior / max(depth, 1)
        particle_weight = float(particle.get("mass", particle.get("weight", 1.0)))
        search_return = (solution + 0.60 * experiment + rollout_tail) * max(particle_weight, 1.0e-6)
        entry = stats[int(chosen_action)]
        entry["visits"] += 1.0
        entry["value_sum"] += float(search_return)
        entry["solution_sum"] += float(solution * max(particle_weight, 1.0e-6))
        entry["experiment_sum"] += float(experiment * max(particle_weight, 1.0e-6))

    action_values: dict[str, dict[str, Any]] = {}
    values_by_action: dict[int, float] = {}
    for action in available_actions:
        entry = stats[int(action)]
        visits = max(float(entry["visits"]), 0.0)
        q_value = float(entry["value_sum"] / max(visits, 1.0)) if visits > 0.0 else 0.0
        solution_value = float(entry["solution_sum"] / max(visits, 1.0)) if visits > 0.0 else 0.0
        experiment_value = float(entry["experiment_sum"] / max(visits, 1.0)) if visits > 0.0 else 0.0
        action_value = float(q_value + 0.15 * prior_values[int(action)])
        values_by_action[int(action)] = float(max(action_value, 0.0))
        action_values[str(action)] = {
            "action": int(action),
            "visits": int(visits),
            "prior_value": float(prior_values[int(action)]),
            "q_value": float(q_value),
            "solution_value": float(solution_value),
            "experiment_value": float(experiment_value),
            "action_value": float(max(action_value, 0.0)),
        }

    selected_action = max(
        available_actions,
        key=lambda action: (
            values_by_action[int(action)],
            action_values[str(action)]["solution_value"],
            action_values[str(action)]["experiment_value"],
            -int(action),
        ),
    )
    selected_hypotheses = [
        item
        for item in real_hypotheses
        if str(item.get("action", "")) == str(selected_action)
    ]
    selected_goals = sorted(
        real_goals,
        key=lambda item: (
            _hypothesis_goal_alignment(selected_hypotheses[0], item) if selected_hypotheses else 0.0,
            float(item.get("mass", 0.0)),
            int(item.get("support", 0)),
            str(item.get("id", "")),
        ),
        reverse=True,
    )
    selected_hypothesis_ids = [str(item["id"]) for item in selected_hypotheses[:4] if str(item.get("id", ""))]
    selected_goal_ids = [str(item["id"]) for item in selected_goals[:4] if str(item.get("id", ""))]
    best_action_values = action_values[str(selected_action)]
    solution_value = float(best_action_values["solution_value"])
    experiment_value = float(best_action_values["experiment_value"])
    min_hyp_support = min((int(item.get("support", 0)) for item in selected_hypotheses[:2]), default=0)
    min_goal_support = min((int(item.get("support", 0)) for item in selected_goals[:2]), default=0)
    strong_joint_evidence = bool(
        selected_hypothesis_ids
        and selected_goal_ids
        and min_hyp_support >= 3
        and min_goal_support >= 2
        and solution_value >= experiment_value
    )
    sampled_particles = [
        {
            "hypothesis_id": str(item["hypothesis"].get("id", "")),
            "goal_id": str(item["goal"].get("id", "")),
            "weight": float(item.get("mass", item.get("weight", 0.0))),
            "hypothesis_posterior": float(item["hypothesis"].get("posterior", 0.0)),
            "goal_posterior": float(item["goal"].get("posterior", 0.0)),
            "hypothesis_action": str(item["hypothesis"].get("action", "")),
            "goal_kind": str(item["goal"].get("kind", "")),
        }
        for item in particles[:12]
    ]
    search = {
        "schema": "runtime_posterior_conditioned_pomcp_search_v1",
        "belief_tree": "root_action_ucb",
        "sampled_belief_particles": sampled_particles,
        "simulations": int(sum(int(item["visits"]) for item in action_values.values())),
        "requested_simulations": int(simulations),
        "horizon": int(horizon),
        "discount": float(discount),
        "exploration_constant": float(exploration),
        "root_visit_count": int(sum(int(item["visits"]) for item in action_values.values())),
        "action_values": action_values,
        "selected_action": int(selected_action),
        "selected_hypothesis_ids": selected_hypothesis_ids,
        "selected_goal_ids": selected_goal_ids,
        "solution_search_value": float(solution_value),
        "experiment_search_value": float(experiment_value),
        "selected_value": float(values_by_action[int(selected_action)]),
        "joint_posterior_conditioned": bool(real_hypotheses and real_goals),
        "plan_status": "plan" if strong_joint_evidence else "hypothesis",
        "evidence_gate": "posterior_conditioned_tree_plan" if strong_joint_evidence else "posterior_conditioned_tree_hypothesis",
        "high_uncertainty_dependency": not strong_joint_evidence,
    }
    return search, values_by_action


def _joint_world_goal_value(
    *,
    action: int,
    hypothesis_posterior: Mapping[str, Any],
    semantic_memory: Mapping[str, Any],
    goal_posterior: Mapping[str, Any],
    progress_value_model: Mapping[str, Any],
    graph_plan: Mapping[str, Any] | None,
    macro_plan: Mapping[str, Any] | None,
) -> dict[str, Any]:
    hypotheses = _hypothesis_belief_candidates(hypothesis_posterior, limit=12)
    goals = _goal_belief_candidates(goal_posterior, limit=12)
    action_text = str(action)
    hypothesis_marginal = sum(
        float(hypothesis.get("mass", 0.0))
        for hypothesis in hypotheses
        if str(hypothesis.get("action", "")) == action_text
    )
    if not hypotheses or not goals or hypothesis_marginal <= 0.0:
        return {
            "schema": "runtime_joint_world_goal_value_v1",
            "action": int(action),
            "hypothesis_goal_terms": [],
            "hypothesis_marginal": float(max(hypothesis_marginal, 0.0)),
            "goal_marginal": 0.0,
            "joint_mass": 0.0,
            "expected_solution_value": 0.0,
            "expected_information_value": 0.0,
            "value": 0.0,
            "joint_uncertainty_propagated": False,
            "selected_hypothesis_ids": [],
            "selected_goal_ids": [],
            "term_count": 0,
        }

    projection = _projection_for_rollout_action(
        action,
        hypothesis_posterior=hypothesis_posterior,
        semantic_memory=semantic_memory,
        goal_posterior=goal_posterior,
        progress_value_model=progress_value_model,
    )
    terms: list[dict[str, Any]] = []
    expected_solution = 0.0
    expected_information = 0.0
    joint_mass = 0.0
    selected_hypotheses: set[str] = set()
    selected_goals: set[str] = set()
    for hypothesis in hypotheses:
        if str(hypothesis.get("action", "")) != action_text:
            continue
        for goal in goals:
            term_mass = float(hypothesis.get("mass", 0.0)) * float(goal.get("mass", 0.0))
            if term_mass <= 0.0:
                continue
            solution, information = _posterior_tree_particle_scores(
                action=action,
                hypothesis=hypothesis,
                goal=goal,
                projection=projection,
                graph_plan=graph_plan,
                macro_plan=macro_plan,
            )
            weighted_solution = term_mass * solution
            weighted_information = term_mass * information
            alignment = _hypothesis_goal_alignment(hypothesis, goal)
            term_value = weighted_solution + 0.60 * weighted_information
            terms.append(
                {
                    "hypothesis_id": str(hypothesis.get("id", "")),
                    "goal_id": str(goal.get("id", "")),
                    "hypothesis_action": str(hypothesis.get("action", "")),
                    "hypothesis_family": str(hypothesis.get("family", "")),
                    "goal_kind": str(goal.get("kind", "")),
                    "joint_weight": float(term_mass),
                    "alignment": float(alignment),
                    "solution_value": float(solution),
                    "information_value": float(information),
                    "weighted_solution_value": float(weighted_solution),
                    "weighted_information_value": float(weighted_information),
                    "term_value": float(term_value),
                    "hypothesis_support": int(hypothesis.get("support", 0)),
                    "goal_support": int(goal.get("support", 0)),
                }
            )
            expected_solution += weighted_solution
            expected_information += weighted_information
            joint_mass += term_mass
            selected_hypotheses.add(str(hypothesis.get("id", "")))
            selected_goals.add(str(goal.get("id", "")))
    terms.sort(
        key=lambda item: (
            float(item["term_value"]),
            float(item["joint_weight"]),
            float(item["alignment"]),
            str(item["hypothesis_id"]),
            str(item["goal_id"]),
        ),
        reverse=True,
    )
    value = expected_solution + 0.60 * expected_information
    goal_marginal = sum(float(goal.get("mass", 0.0)) for goal in goals) if terms else 0.0
    return {
        "schema": "runtime_joint_world_goal_value_v1",
        "action": int(action),
        "hypothesis_goal_terms": terms[:12],
        "hypothesis_marginal": float(_clamp01(hypothesis_marginal)),
        "goal_marginal": float(_clamp01(goal_marginal)),
        "joint_mass": float(_clamp01(joint_mass)),
        "expected_solution_value": float(max(expected_solution, 0.0)),
        "expected_information_value": float(max(expected_information, 0.0)),
        "value": float(max(value, 0.0)),
        "joint_uncertainty_propagated": bool(terms),
        "selected_hypothesis_ids": sorted(item for item in selected_hypotheses if item)[:8],
        "selected_goal_ids": sorted(item for item in selected_goals if item)[:8],
        "term_count": int(len(terms)),
    }


def select_experimental_action(
    action_logits: torch.Tensor,
    memory: Any,
    observation: Mapping[str, Any],
    *,
    available_action_mask: Any = None,
) -> tuple[int, dict[str, Any]]:
    policy_scores = _softmax_scores(action_logits)
    action_count = len(policy_scores)
    available_actions = _available_actions(available_action_mask, action_count)
    if not available_actions:
        available_actions = list(range(action_count))
    graph = getattr(memory, "transition_graph", {}) or {}
    semantic_memory = getattr(memory, "semantic_memory", {}) or {}
    hypothesis_posterior = getattr(memory, "hypothesis_posterior", {}) or {}
    goal_posterior = getattr(memory, "goal_posterior", {}) or {}
    macro_policy_library = getattr(memory, "macro_policy_library", {}) or {}
    progress_value_model = getattr(memory, "progress_value_model", {}) or {}
    plastic_memory = getattr(memory, "plastic_memory", {}) or {}
    state_hash = canonical_observation_hash(observation)
    graph_plans = _graph_plan_values(graph, state_hash, available_actions=available_actions)
    macro_plans = _macro_plan_values(macro_policy_library, available_actions=available_actions)
    posterior_tree_search, posterior_tree_values = _posterior_conditioned_tree_search(
        available_actions=available_actions,
        hypothesis_posterior=hypothesis_posterior,
        semantic_memory=semantic_memory,
        goal_posterior=goal_posterior,
        progress_value_model=progress_value_model,
        graph_plans=graph_plans,
        macro_plans=macro_plans,
    )
    rows: list[dict[str, Any]] = []
    for action in available_actions:
        world_model_mixture = posterior_predictive_mixture(hypothesis_posterior, action=action)
        posterior_mixture_value = _posterior_mixture_value(world_model_mixture)
        posterior_tree_value = float(posterior_tree_values.get(int(action), 0.0))
        graph_plan = graph_plans.get(action)
        graph_plan_value = float(graph_plan.get("value", 0.0)) if graph_plan is not None else 0.0
        macro_plan = macro_plans.get(action)
        macro_plan_value = float(macro_plan.get("value", 0.0)) if macro_plan is not None else 0.0
        joint_world_goal_plan = _joint_world_goal_value(
            action=action,
            hypothesis_posterior=hypothesis_posterior,
            semantic_memory=semantic_memory,
            goal_posterior=goal_posterior,
            progress_value_model=progress_value_model,
            graph_plan=graph_plan,
            macro_plan=macro_plan,
        )
        joint_world_goal_value = float(joint_world_goal_plan.get("value", 0.0))
        internal_rollouts, rollout_summary = _bounded_internal_rollouts(
            action=action,
            available_actions=available_actions,
            graph_plan=graph_plan,
            macro_plan=macro_plan,
            hypothesis_posterior=hypothesis_posterior,
            semantic_memory=semantic_memory,
            goal_posterior=goal_posterior,
            progress_value_model=progress_value_model,
            world_model_mixture=world_model_mixture,
        )
        posterior_rollout_value = float(rollout_summary["posterior_rollout_value"])
        neural_mpc_rollout = _learned_neural_mpc_rollout(
            action=action,
            available_actions=available_actions,
            policy_scores=policy_scores,
            plastic_memory=plastic_memory,
        )
        neural_mpc_value = float(neural_mpc_rollout.get("value", 0.0))
        expected_value = (
            _semantic_value(semantic_memory, action)
            + _hypothesis_value(hypothesis_posterior, action)
            + posterior_mixture_value
            + posterior_rollout_value
            + posterior_tree_value
            + joint_world_goal_value
            + neural_mpc_value
            + _goal_value(goal_posterior, action)
            + macro_plan_value
            + _progress_backup_value(progress_value_model, action)
            + _undo_value(graph, state_hash, action)
            + graph_plan_value
        )
        information_gain = _information_gain(graph, state_hash, action, available_actions=available_actions)
        risk = _risk(graph, state_hash, action)
        action_cost = 0.01 + 0.002 * float(action)
        policy = policy_scores[action]
        imagined_futures, imagined_summary = _imagined_futures(
            graph,
            hypothesis_posterior,
            plastic_memory,
            state_hash,
            action,
            policy=policy,
        )
        macro_plan_copy: dict[str, Any] | None = None
        if macro_plan is not None:
            macro_plan_copy = dict(macro_plan)
            macro_uncertainty = float(max(0.0, 1.0 - float(macro_plan_copy.get("confidence", 0.0))))
            macro_transfer_plan = (
                bool(macro_plan_copy.get("transfer_ready", False))
                and float(macro_plan_copy.get("confidence", 0.0)) >= 0.50
            )
            macro_plan_copy["plan_status"] = "plan" if macro_transfer_plan else imagined_summary["plan_status"]
            macro_plan_copy["evidence_gate"] = "macro_policy_transfer" if macro_transfer_plan else imagined_summary["evidence_gate"]
            macro_plan_copy["uncertainty"] = macro_uncertainty
            macro_plan_copy["high_uncertainty_dependency"] = not macro_transfer_plan
            if macro_transfer_plan:
                imagined_summary = dict(imagined_summary)
                imagined_summary["plan_status"] = "plan"
                imagined_summary["evidence_gate"] = "macro_policy_transfer"
                imagined_summary["high_uncertainty_dependency"] = False
                imagined_summary["hypothesis_only"] = False
                imagined_summary["imagined_confidence"] = max(
                    float(imagined_summary["imagined_confidence"]),
                    float(macro_plan_copy.get("confidence", 0.0)),
                )
                imagined_summary["imagined_uncertainty"] = min(
                    float(imagined_summary["imagined_uncertainty"]),
                    macro_uncertainty,
                )
                imagined_summary["future_risk_order"] = min(int(imagined_summary["future_risk_order"]), 1)
                imagined_summary["future_source_priority"] = max(int(imagined_summary["future_source_priority"]), 2)
                imagined_summary["dominant_future_source"] = "macro_policy"
        if (
            str(rollout_summary["internal_rollout_plan_status"]) == "plan"
            and not bool(rollout_summary["internal_rollout_high_uncertainty_dependency"])
            and str(imagined_summary["plan_status"]) != "plan"
        ):
            imagined_summary = dict(imagined_summary)
            imagined_summary["plan_status"] = "plan"
            imagined_summary["evidence_gate"] = str(rollout_summary["internal_rollout_evidence_gate"])
            imagined_summary["high_uncertainty_dependency"] = False
            imagined_summary["hypothesis_only"] = False
            imagined_summary["imagined_confidence"] = max(
                float(imagined_summary["imagined_confidence"]),
                float(1.0 - rollout_summary["internal_rollout_uncertainty"]),
            )
            imagined_summary["imagined_uncertainty"] = min(
                float(imagined_summary["imagined_uncertainty"]),
                float(rollout_summary["internal_rollout_uncertainty"]),
            )
            imagined_summary["future_risk_order"] = min(int(imagined_summary["future_risk_order"]), 1)
            imagined_summary["future_source_priority"] = max(int(imagined_summary["future_source_priority"]), 2)
            imagined_summary["dominant_future_source"] = str(rollout_summary["internal_rollout_source"])
        future_risk_order = int(imagined_summary["future_risk_order"])
        score = (
            0.70 * policy
            + 1.05 * expected_value
            + 0.85 * information_gain
            - 0.70 * risk
            - action_cost
            - 0.12 * float(future_risk_order)
        )
        rows.append(
            {
                "action": int(action),
                "score": float(score),
                "policy": float(policy),
                "expected_value": float(expected_value),
                "information_gain": float(information_gain),
                "risk": float(risk),
                "action_cost": float(action_cost),
                "world_model_mixture": world_model_mixture,
                "posterior_mixture_value": float(posterior_mixture_value),
                "posterior_tree_search": posterior_tree_search,
                "posterior_tree_value": float(posterior_tree_value),
                "joint_world_goal_plan": joint_world_goal_plan,
                "joint_world_goal_value": float(joint_world_goal_value),
                "neural_mpc_rollout": neural_mpc_rollout,
                "neural_mpc_value": float(neural_mpc_value),
                "internal_rollouts": internal_rollouts,
                **rollout_summary,
                "imagined_futures": imagined_futures,
                **imagined_summary,
                "observed_rank": _observed_rank(graph, state_hash, action),
                "undo_candidate": bool(_is_undo_action(action)),
                "undo_confidence": float(_undo_confidence(graph, action)),
                "undo_probe": bool(_undo_probe_available(graph, state_hash, action)),
                "graph_plan_value": graph_plan_value,
                "macro_plan_value": macro_plan_value,
            }
        )
        if graph_plan is not None:
            plan_copy = dict(graph_plan)
            plan_copy["plan_status"] = rows[-1]["plan_status"]
            plan_copy["evidence_gate"] = rows[-1]["evidence_gate"]
            plan_copy["uncertainty"] = rows[-1]["imagined_uncertainty"]
            plan_copy["high_uncertainty_dependency"] = rows[-1]["high_uncertainty_dependency"]
            rows[-1]["graph_plan"] = plan_copy
        if macro_plan_copy is not None:
            rows[-1]["macro_plan"] = macro_plan_copy
    rows.sort(
        key=lambda item: (
            float(item["score"]),
            float(item["expected_value"]),
            float(item["information_gain"]),
            -float(item["risk"]),
            -int(item["action"]),
        ),
        reverse=True,
    )
    selected = rows[0]
    selected_is_admissible_solution = (
        str(selected.get("plan_status")) == "plan"
        and not bool(selected.get("high_uncertainty_dependency", True))
        and float(selected["expected_value"]) >= float(selected["information_gain"])
    )
    diagnostics = {
        "schema": "runtime_experimental_action_selection_v1",
        "selected_action": int(selected["action"]),
        "state_hash": state_hash,
        "mode": "solution" if selected_is_admissible_solution else "experiment",
        "selected_plan_status": str(selected.get("plan_status")),
        "selected_evidence_gate": str(selected.get("evidence_gate")),
        "selected_high_uncertainty_dependency": bool(selected.get("high_uncertainty_dependency", True)),
        "selected_macro_id": str(selected.get("macro_plan", {}).get("macro_id", "")) if "macro_plan" in selected else "",
        "selected_posterior_tree_search": dict(selected.get("posterior_tree_search", {})),
        "selected_joint_world_goal_plan": dict(selected.get("joint_world_goal_plan", {})),
        "selected_neural_mpc_rollout": dict(selected.get("neural_mpc_rollout", {})),
        "candidate_count": len(rows),
        "available_actions": [int(item) for item in available_actions],
        "components": rows,
        "weights": {
            "policy": 0.70,
            "expected_value": 1.05,
            "information_gain": 0.85,
            "risk": -0.70,
            "action_cost": -1.0,
            "future_risk_order": -0.12,
        },
    }
    if "macro_plan" in selected:
        diagnostics["selected_macro_plan"] = dict(selected["macro_plan"])
    return int(selected["action"]), diagnostics
