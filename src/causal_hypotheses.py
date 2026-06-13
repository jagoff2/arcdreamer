from __future__ import annotations

import math
from typing import Any, Mapping

import torch

from .grid_perception import parse_grid


NEURAL_PROPOSAL_SCHEMA = "neural_causal_program_proposal_v1"
NEURAL_PROPOSAL_FAMILIES = {"no_op", "field_change", "field_stable", "move_color"}


def fresh_hypothesis_posterior() -> dict[str, Any]:
    return {
        "schema": "runtime_causal_hypothesis_posterior_v1",
        "updates": 0,
        "hypotheses": {},
        "posterior_mass": 0.0,
        "top_hypotheses": [],
    }


def _tensor_from_snapshot(snapshot: Any) -> torch.Tensor | None:
    if not isinstance(snapshot, Mapping):
        return None
    if "values" not in snapshot:
        return None
    try:
        return torch.as_tensor(snapshot["values"])
    except (TypeError, ValueError):
        return None


def _changed(delta_item: Any) -> bool:
    if isinstance(delta_item, Mapping) and "changed_count" in delta_item:
        return int(delta_item.get("changed_count", 0)) > 0
    if isinstance(delta_item, Mapping) and "changed" in delta_item:
        return bool(delta_item.get("changed"))
    return bool(delta_item)


def _delta_l1(delta_item: Any) -> float:
    if isinstance(delta_item, Mapping) and "l1" in delta_item:
        return float(delta_item.get("l1", 0.0))
    return 1.0 if _changed(delta_item) else 0.0


def _grid_from_observation_snapshot(snapshot: Mapping[str, Any]) -> torch.Tensor | None:
    if "grid" in snapshot:
        return _tensor_from_snapshot(snapshot["grid"])
    if "sensory" in snapshot:
        tensor = _tensor_from_snapshot(snapshot["sensory"])
        if tensor is not None and tensor.ndim == 2 and not torch.is_floating_point(tensor):
            return tensor
    return None


def _hypothesis_id(program: Mapping[str, Any]) -> str:
    family = str(program["family"])
    action = str(program["action"])
    selector = program.get("selector", {})
    transform = program.get("transform", {})
    if family in {"field_change", "field_stable"}:
        return f"{family}|action:{action}|field:{selector.get('field')}"
    if family == "no_op":
        return f"{family}|action:{action}"
    if family == "move_color":
        return (
            f"{family}|action:{action}|color:{selector.get('color')}|"
            f"dy:{transform.get('dy')}|dx:{transform.get('dx')}"
        )
    return f"{family}|action:{action}|{selector}|{transform}"


def _description_length(program: Mapping[str, Any]) -> float:
    family_cost = {
        "no_op": 1.0,
        "field_stable": 1.4,
        "field_change": 1.6,
        "move_color": 2.4,
    }.get(str(program.get("family")), 2.0)
    selector_cost = 0.25 * len(program.get("selector", {}))
    transform_cost = 0.25 * len(program.get("transform", {}))
    return float(family_cost + selector_cost + transform_cost)


def _candidate_programs(
    *,
    action: str,
    event: Mapping[str, Any],
) -> list[dict[str, Any]]:
    delta = event.get("delta", {})
    programs: list[dict[str, Any]] = []
    changed_fields = [str(field) for field, item in delta.items() if _changed(item)]
    stable_fields = [str(field) for field in delta if str(field) not in changed_fields]
    if not changed_fields:
        programs.append(
            {
                "family": "no_op",
                "action": action,
                "selector": {"scope": "observation"},
                "transform": {"kind": "identity"},
                "goal_test": {"kind": "no_visible_change"},
            }
        )
    for field in changed_fields:
        programs.append(
            {
                "family": "field_change",
                "action": action,
                "selector": {"field": field},
                "transform": {"kind": "change", "l1": round(_delta_l1(delta[field]), 6)},
                "goal_test": {"kind": "field_changed"},
            }
        )
    for field in stable_fields:
        programs.append(
            {
                "family": "field_stable",
                "action": action,
                "selector": {"field": field},
                "transform": {"kind": "identity"},
                "goal_test": {"kind": "field_unchanged"},
            }
        )
    programs.extend(_grid_move_programs(action=action, event=event))
    return programs


def _bounded_confidence(value: Any) -> float:
    try:
        return float(min(max(float(value), 0.0), 1.0))
    except (TypeError, ValueError):
        return 0.0


def _neural_candidate_programs(
    *,
    action: str,
    event: Mapping[str, Any],
    limit: int = 8,
) -> list[dict[str, Any]]:
    prediction_error = event.get("prediction_error", {})
    if not isinstance(prediction_error, Mapping):
        return []
    proposals = prediction_error.get("neural_program_proposals", [])
    if not isinstance(proposals, (list, tuple)):
        return []

    programs: list[dict[str, Any]] = []
    for proposal in proposals:
        if len(programs) >= limit:
            break
        if not isinstance(proposal, Mapping):
            continue
        family = str(proposal.get("family", ""))
        if family not in NEURAL_PROPOSAL_FAMILIES:
            continue
        if str(proposal.get("schema", "")) != NEURAL_PROPOSAL_SCHEMA:
            continue
        selector = proposal.get("selector", {})
        transform = proposal.get("transform", {})
        goal_test = proposal.get("goal_test", {})
        if not isinstance(selector, Mapping) or not isinstance(transform, Mapping) or not isinstance(goal_test, Mapping):
            continue
        programs.append(
            {
                "family": family,
                "action": action,
                "selector": dict(selector),
                "transform": dict(transform),
                "goal_test": dict(goal_test),
                "source": "neural_proposal",
                "confidence": _bounded_confidence(proposal.get("confidence", 0.0)),
                "proposal": {
                    key: proposal[key]
                    for key in (
                        "family_probability",
                        "field_probability",
                        "transform_probability",
                        "color_probability",
                        "family_index",
                        "field_index",
                        "transform_index",
                        "color_index",
                    )
                    if key in proposal
                },
            }
        )
    return programs


def _grid_move_programs(*, action: str, event: Mapping[str, Any]) -> list[dict[str, Any]]:
    before = event.get("observation", {})
    after = event.get("next_observation", {})
    if not isinstance(before, Mapping) or not isinstance(after, Mapping):
        return []
    before_grid = _grid_from_observation_snapshot(before)
    after_grid = _grid_from_observation_snapshot(after)
    if before_grid is None or after_grid is None or before_grid.ndim != 2 or after_grid.ndim != 2:
        return []
    try:
        scene = parse_grid(after_grid.numpy(), previous_grid=before_grid.numpy(), action=action)
    except ValueError:
        return []
    programs = []
    for move in scene.edit_script.moves:
        dy, dx = move.get("delta", [0.0, 0.0])
        programs.append(
            {
                "family": "move_color",
                "action": action,
                "selector": {"color": int(move["color"])},
                "transform": {"kind": "translate", "dy": float(dy), "dx": float(dx)},
                "goal_test": {"kind": "component_translation"},
            }
        )
    return programs


def _predicts(program: Mapping[str, Any], event: Mapping[str, Any]) -> bool:
    delta = event.get("delta", {})
    family = str(program.get("family"))
    if family == "no_op":
        return not any(_changed(item) for item in delta.values())
    if family == "field_change":
        field = str(program.get("selector", {}).get("field"))
        return field in delta and _changed(delta[field])
    if family == "field_stable":
        field = str(program.get("selector", {}).get("field"))
        return field in delta and not _changed(delta[field])
    if family == "move_color":
        for candidate in _grid_move_programs(action=str(program.get("action")), event=event):
            if (
                candidate.get("selector") == program.get("selector")
                and candidate.get("transform") == program.get("transform")
            ):
                return True
        return False
    return False


def _score_hypothesis(hypothesis: Mapping[str, Any]) -> float:
    return float(
        1.20 * int(hypothesis.get("support", 0))
        - 1.80 * int(hypothesis.get("counterexamples", 0))
        - 0.85 * int(hypothesis.get("failed_predictions", 0))
        - 0.20 * float(hypothesis.get("prediction_loss", 0.0))
        - 0.35 * float(hypothesis.get("description_length", 1.0))
    )


def _normalize(posterior: dict[str, Any]) -> None:
    hypotheses = posterior.setdefault("hypotheses", {})
    if not hypotheses:
        posterior["posterior_mass"] = 0.0
        posterior["top_hypotheses"] = []
        return
    scores = [float(item.get("score", 0.0)) for item in hypotheses.values()]
    max_score = max(scores)
    weights = {
        key: math.exp(float(item.get("score", 0.0)) - max_score)
        for key, item in hypotheses.items()
    }
    total = sum(weights.values())
    for key, item in hypotheses.items():
        item["posterior"] = float(weights[key] / max(total, 1.0e-12))
    posterior["posterior_mass"] = float(sum(item["posterior"] for item in hypotheses.values()))
    top = sorted(
        hypotheses.values(),
        key=lambda item: (float(item.get("posterior", 0.0)), int(item.get("support", 0))),
        reverse=True,
    )
    posterior["top_hypotheses"] = [
        {
            "id": item["id"],
            "family": item["family"],
            "posterior": item["posterior"],
            "support": item["support"],
            "counterexamples": item["counterexamples"],
            "failed_predictions": int(item.get("failed_predictions", 0)),
        }
        for item in top[:8]
    ]


def _posterior_mass(hypothesis: Mapping[str, Any]) -> float:
    try:
        value = float(hypothesis.get("posterior", 0.0))
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(value):
        return 0.0
    return max(value, 0.0)


def _mixture_uncertainty(weights: list[float]) -> float:
    if not weights:
        return 1.0
    concentration = max(weights)
    entropy = -sum(weight * math.log(max(weight, 1.0e-12)) for weight in weights)
    max_entropy = math.log(max(len(weights), 2))
    entropy_fraction = entropy / max(max_entropy, 1.0e-12)
    return float(min(max(0.55 * (1.0 - concentration) + 0.45 * entropy_fraction, 0.0), 1.0))


def posterior_predictive_mixture(
    posterior: Mapping[str, Any],
    *,
    action: int | str,
    top_k: int = 8,
) -> dict[str, Any]:
    """Return a posterior-weighted distribution over typed causal edit programs."""

    action_text = str(action)
    raw_components: list[tuple[float, Mapping[str, Any]]] = []
    for hypothesis in (posterior or {}).get("hypotheses", {}).values():
        if not isinstance(hypothesis, Mapping):
            continue
        if str(hypothesis.get("action")) != action_text:
            continue
        if str(hypothesis.get("family")) not in {"field_change", "field_stable", "no_op", "move_color"}:
            continue
        mass = _posterior_mass(hypothesis)
        if mass <= 0.0:
            continue
        raw_components.append((mass, hypothesis))

    raw_components.sort(
        key=lambda item: (
            float(item[0]),
            int(item[1].get("support", 0)),
            str(item[1].get("id", "")),
        ),
        reverse=True,
    )
    raw_components = raw_components[: max(int(top_k), 1)]
    total_mass = sum(mass for mass, _ in raw_components)
    weights = [mass / max(total_mass, 1.0e-12) for mass, _ in raw_components]

    components: list[dict[str, Any]] = []
    family_mixture: dict[str, float] = {}
    field_change_probability: dict[str, float] = {}
    field_stable_probability: dict[str, float] = {}
    move_color_probability: dict[str, float] = {}
    no_op_probability = 0.0
    expected_change_probability = 0.0

    for weight, (mass, hypothesis) in zip(weights, raw_components):
        family = str(hypothesis.get("family", ""))
        selector = dict(hypothesis.get("selector", {}) or {})
        transform = dict(hypothesis.get("transform", {}) or {})
        family_mixture[family] = float(family_mixture.get(family, 0.0) + weight)
        if family == "no_op":
            no_op_probability += weight
        elif family == "field_change":
            field = str(selector.get("field", ""))
            if field:
                field_change_probability[field] = float(field_change_probability.get(field, 0.0) + weight)
            expected_change_probability += weight
        elif family == "field_stable":
            field = str(selector.get("field", ""))
            if field:
                field_stable_probability[field] = float(field_stable_probability.get(field, 0.0) + weight)
        elif family == "move_color":
            key = (
                f"color:{selector.get('color')}|"
                f"dy:{transform.get('dy', 0.0)}|dx:{transform.get('dx', 0.0)}"
            )
            move_color_probability[key] = float(move_color_probability.get(key, 0.0) + weight)
            expected_change_probability += weight
        components.append(
            {
                "id": str(hypothesis.get("id", "")),
                "family": family,
                "selector": selector,
                "transform": transform,
                "goal_test": dict(hypothesis.get("goal_test", {}) or {}),
                "posterior_mass": float(mass),
                "mixture_weight": float(weight),
                "support": int(hypothesis.get("support", 0)),
                "counterexamples": int(hypothesis.get("counterexamples", 0)),
                "failed_predictions": int(hypothesis.get("failed_predictions", 0)),
                "description_length": float(hypothesis.get("description_length", 0.0)),
                "source": str(hypothesis.get("source", "event_evidence")),
            }
        )

    mixture_available = bool(components)
    return {
        "schema": "runtime_posterior_predictive_causal_mixture_v1",
        "prediction_type": "causal_edit_program_mixture",
        "pixel_prediction": False,
        "action": action_text,
        "component_count": len(components),
        "action_posterior_mass": float(min(total_mass, 1.0)),
        "normalized_component_mass": float(sum(weights)),
        "expected_change_probability": float(min(max(expected_change_probability, 0.0), 1.0)),
        "no_op_probability": float(min(max(no_op_probability, 0.0), 1.0)),
        "field_change_probability": field_change_probability,
        "field_stable_probability": field_stable_probability,
        "move_color_probability": move_color_probability,
        "family_mixture": family_mixture,
        "uncertainty": _mixture_uncertainty(weights),
        "mixture_available": mixture_available,
        "components": components,
    }


def event_hypothesis_trace(
    posterior: Mapping[str, Any],
    event: Mapping[str, Any],
) -> dict[str, Any]:
    delta = event.get("delta", {})
    if not isinstance(delta, Mapping):
        delta = {}
    changed_fields = sorted(str(field) for field, item in delta.items() if _changed(item))
    invariant_fields = sorted(str(field) for field in delta if str(field) not in changed_fields)
    action = str(event.get("action"))
    predicted: list[dict[str, Any]] = []
    contradicted: list[dict[str, Any]] = []
    for hypothesis in (posterior or {}).get("hypotheses", {}).values():
        if str(hypothesis.get("action")) != action:
            continue
        row = {
            "id": str(hypothesis.get("id", "")),
            "family": str(hypothesis.get("family", "")),
            "action": str(hypothesis.get("action", "")),
            "selector": dict(hypothesis.get("selector", {})),
            "transform": dict(hypothesis.get("transform", {})),
            "posterior": float(hypothesis.get("posterior", 0.0)),
            "support": int(hypothesis.get("support", 0)),
            "counterexamples": int(hypothesis.get("counterexamples", 0)),
            "failed_predictions": int(hypothesis.get("failed_predictions", 0)),
        }
        if _predicts(hypothesis, event):
            predicted.append(row)
        elif str(hypothesis.get("family")) in {"field_change", "field_stable", "no_op", "move_color"}:
            contradicted.append(row)

    predicted.sort(key=lambda item: (float(item["posterior"]), int(item["support"]), item["id"]), reverse=True)
    contradicted.sort(
        key=lambda item: (float(item["posterior"]), int(item["counterexamples"]), item["id"]),
        reverse=True,
    )
    return {
        "schema": "runtime_hypothesis_trace_v1",
        "changed_fields": changed_fields,
        "invariant_fields": invariant_fields,
        "predicted_hypotheses": predicted[:16],
        "contradicted_hypotheses": contradicted[:16],
    }


def _prediction_failure_context(prediction_error: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: prediction_error[key]
        for key in ("expected", "observed", "loss", "prediction_loss", "family", "action", "field", "color", "selector")
        if key in prediction_error
    }


def _append_failure_record(records: list[dict[str, Any]], raw: Any, context: Mapping[str, Any]) -> None:
    if raw is None:
        return
    if isinstance(raw, Mapping):
        record = dict(context)
        record.update(dict(raw))
    else:
        record = dict(context)
        record["id"] = str(raw)
    if "hypothesis_id" in record and "id" not in record:
        record["id"] = record["hypothesis_id"]
    records.append(record)


def _prediction_failure_records(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    prediction_error = event.get("prediction_error", {})
    if not isinstance(prediction_error, Mapping):
        return []

    context = _prediction_failure_context(prediction_error)
    records: list[dict[str, Any]] = []
    for key in ("failed_predictions", "hypothesis_failures", "contradictions", "failures"):
        value = prediction_error.get(key)
        if isinstance(value, (list, tuple)):
            for item in value:
                _append_failure_record(records, item, context)
        elif value is not None:
            _append_failure_record(records, value, context)

    for key in ("failed_hypotheses", "failed_hypothesis_ids", "contradicted_hypothesis_ids"):
        value = prediction_error.get(key)
        if isinstance(value, (list, tuple, set)):
            for item in value:
                _append_failure_record(records, {"id": str(item)}, context)
        elif value is not None:
            _append_failure_record(records, {"id": str(value)}, context)

    if bool(prediction_error.get("prediction_failed", False)) or bool(prediction_error.get("failed", False)):
        _append_failure_record(records, context, {})
    return records


def _failure_loss(failure: Mapping[str, Any]) -> float:
    for key in ("loss", "prediction_loss"):
        if key not in failure:
            continue
        try:
            return float(max(float(failure[key]), 0.0))
        except (TypeError, ValueError):
            continue
    return 1.0


def _failure_matches_hypothesis(
    hypothesis: Mapping[str, Any],
    failure: Mapping[str, Any],
    *,
    event_action: str,
) -> bool:
    failure_id = failure.get("id", failure.get("hypothesis_id"))
    if failure_id is not None:
        return str(failure_id) == str(hypothesis.get("id"))

    failure_action = failure.get("action")
    if failure_action is not None and str(failure_action) != str(hypothesis.get("action", event_action)):
        return False
    if failure_action is None and str(hypothesis.get("action")) != event_action:
        return False

    failure_family = failure.get("family")
    if failure_family is not None and str(failure_family) != str(hypothesis.get("family")):
        return False

    selector = failure.get("selector", {})
    if not isinstance(selector, Mapping):
        selector = {}
    field = failure.get("field", selector.get("field"))
    color = failure.get("color", selector.get("color"))
    hypothesis_selector = hypothesis.get("selector", {})
    if not isinstance(hypothesis_selector, Mapping):
        hypothesis_selector = {}
    if field is not None and str(hypothesis_selector.get("field")) != str(field):
        return False
    if color is not None and str(hypothesis_selector.get("color")) != str(color):
        return False

    return failure_family is not None or field is not None or color is not None


def _record_prediction_failure(
    hypothesis: dict[str, Any],
    failure: Mapping[str, Any],
    *,
    event: Mapping[str, Any],
    loss: float,
) -> None:
    hypothesis["failed_predictions"] = int(hypothesis.get("failed_predictions", 0)) + 1
    hypothesis["prediction_loss"] = float(hypothesis.get("prediction_loss", 0.0)) + float(loss)
    hypothesis["last_failure_tick"] = int(event.get("tick", 0))
    patches = hypothesis.setdefault("patches", [])
    patch = {
        "kind": "prediction_failure",
        "tick": int(event.get("tick", 0)),
        "action": str(event.get("action")),
        "loss": float(loss),
        "failure": dict(failure),
        "delta_fields": sorted(str(field) for field in event.get("delta", {}).keys()),
    }
    patches.append(patch)
    if len(patches) > 16:
        del patches[: len(patches) - 16]


def _record_neural_proposal(
    hypothesis: dict[str, Any],
    program: Mapping[str, Any],
    *,
    event: Mapping[str, Any],
) -> None:
    sources = hypothesis.setdefault("proposal_sources", [])
    if "neural_proposal" not in sources:
        sources.append("neural_proposal")
    hypothesis["neural_proposal_count"] = int(hypothesis.get("neural_proposal_count", 0)) + 1
    hypothesis["last_neural_proposal_tick"] = int(event.get("tick", 0))
    confidence = _bounded_confidence(program.get("confidence", 0.0))
    hypothesis["proposal_confidence"] = max(float(hypothesis.get("proposal_confidence", 0.0)), confidence)
    records = hypothesis.setdefault("neural_proposals", [])
    records.append(
        {
            "tick": int(event.get("tick", 0)),
            "confidence": confidence,
            "proposal": dict(program.get("proposal", {})),
        }
    )
    if len(records) > 16:
        del records[: len(records) - 16]


def update_hypothesis_posterior(
    posterior: dict[str, Any],
    event: Mapping[str, Any],
) -> dict[str, Any]:
    if not posterior:
        posterior.update(fresh_hypothesis_posterior())
    action = str(event.get("action"))
    hypotheses = posterior.setdefault("hypotheses", {})
    failures = _prediction_failure_records(event)
    candidate_ids: set[str] = set()
    candidate_programs = _candidate_programs(action=action, event=event)
    candidate_programs.extend(_neural_candidate_programs(action=action, event=event))
    for program in candidate_programs:
        hid = _hypothesis_id(program)
        candidate_ids.add(hid)
        if hid not in hypotheses:
            source = str(program.get("source", "event_evidence"))
            hypotheses[hid] = {
                "id": hid,
                "family": program["family"],
                "action": action,
                "selector": dict(program.get("selector", {})),
                "transform": dict(program.get("transform", {})),
                "goal_test": dict(program.get("goal_test", {})),
                "description_length": _description_length(program),
                "source": source,
                "proposal_sources": [source],
                "neural_proposal_count": 0,
                "proposal_confidence": 0.0,
                "support": 0,
                "counterexamples": 0,
                "failed_predictions": 0,
                "prediction_loss": 0.0,
                "patches": [],
                "score": 0.0,
                "posterior": 0.0,
            }
        elif str(program.get("source", "")) == "neural_proposal":
            hypotheses[hid].setdefault("proposal_sources", ["event_evidence"])
        if str(program.get("source", "")) == "neural_proposal":
            _record_neural_proposal(hypotheses[hid], program, event=event)
    for hid, hypothesis in hypotheses.items():
        if str(hypothesis.get("action")) != action:
            continue
        hypothesis.setdefault("failed_predictions", 0)
        hypothesis.setdefault("patches", [])
        predicts = _predicts(hypothesis, event)
        if predicts:
            hypothesis["support"] = int(hypothesis.get("support", 0)) + 1
            hypothesis["prediction_loss"] = float(hypothesis.get("prediction_loss", 0.0))
        elif hid in candidate_ids or hypothesis.get("family") in {"field_change", "field_stable", "no_op", "move_color"}:
            hypothesis["counterexamples"] = int(hypothesis.get("counterexamples", 0)) + 1
            hypothesis["prediction_loss"] = float(hypothesis.get("prediction_loss", 0.0)) + 1.0
        matched_failures = [
            failure
            for failure in failures
            if _failure_matches_hypothesis(hypothesis, failure, event_action=action)
        ]
        if matched_failures:
            if predicts:
                hypothesis["counterexamples"] = int(hypothesis.get("counterexamples", 0)) + len(matched_failures)
            for failure in matched_failures:
                _record_prediction_failure(
                    hypothesis,
                    failure,
                    event=event,
                    loss=_failure_loss(failure),
                )
        hypothesis["score"] = _score_hypothesis(hypothesis)
    posterior["updates"] = int(posterior.get("updates", 0)) + 1
    _normalize(posterior)
    return posterior
