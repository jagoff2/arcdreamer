from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .arcagi3_adapter import ArcAGI3Observation, ArcAGI3StepResult
from .arc_affordance_search import AffordanceSearchState, SearchWeights


@dataclass(frozen=True)
class AffordanceVariantSpec:
    variant_id: str
    component: bool = False
    change: bool = False
    graph: bool = False
    event: bool = False
    object_persistence: bool = False
    comparison_only: bool = False
    ablation_of: str | None = None

    def enabled_features(self) -> set[str]:
        features: set[str] = set()
        if self.component:
            features.add("component")
        if self.change:
            features.add("change")
        if self.graph:
            features.add("graph")
        if self.event:
            features.add("event")
        if self.object_persistence:
            features.add("object_persistence")
        return features


AFFORDANCE_VARIANTS = [
    AffordanceVariantSpec("component_click_search", component=True),
    AffordanceVariantSpec("change_memory_search", change=True, event=True),
    AffordanceVariantSpec("state_graph_affordance", graph=True),
    AffordanceVariantSpec("event_linked_ranking", event=True),
    AffordanceVariantSpec("object_persistence_search", component=True, object_persistence=True),
    AffordanceVariantSpec(
        "combined_affordance_search",
        component=True,
        change=True,
        graph=True,
        event=True,
        object_persistence=True,
    ),
]

AFFORDANCE_ABLATIONS = [
    AffordanceVariantSpec(
        "ablation_no_component_memory",
        change=True,
        graph=True,
        event=True,
        object_persistence=True,
        ablation_of="combined_affordance_search",
    ),
    AffordanceVariantSpec(
        "ablation_no_change_memory",
        component=True,
        graph=True,
        event=True,
        object_persistence=True,
        ablation_of="combined_affordance_search",
    ),
    AffordanceVariantSpec(
        "ablation_no_event_memory",
        component=True,
        change=True,
        graph=True,
        object_persistence=True,
        ablation_of="combined_affordance_search",
    ),
]


def variant_by_id(variant_id: str) -> AffordanceVariantSpec:
    for variant in [*AFFORDANCE_VARIANTS, *AFFORDANCE_ABLATIONS]:
        if variant.variant_id == variant_id:
            return variant
    raise KeyError(variant_id)


class ArcAffordancePolicy:
    def __init__(self, variant: AffordanceVariantSpec | str, *, weights: SearchWeights | None = None) -> None:
        self.variant = variant_by_id(variant) if isinstance(variant, str) else variant
        self.search = AffordanceSearchState()
        self.weights = weights or SearchWeights()
        self.name = self.variant.variant_id
        self.choice_count = 0
        self.affordance_choices = 0
        self.reason_counts: dict[str, int] = {name: 0 for name in ["component", "change", "graph", "event", "object_persistence"]}
        self._last_observation: ArcAGI3Observation | None = None
        self._last_choice_was_affordance = False

    def reset(self, seed: int | None = None) -> None:
        del seed
        self.search.reset()
        self.choice_count = 0
        self.affordance_choices = 0
        self.reason_counts = {name: 0 for name in ["component", "change", "graph", "event", "object_persistence"]}
        self._last_observation = None
        self._last_choice_was_affordance = False

    def choose(self, observation: ArcAGI3Observation) -> str:
        action, _ = self.choose_action(observation)
        return action

    def choose_action(self, observation: ArcAGI3Observation) -> tuple[str, dict[str, Any]]:
        self._last_observation = observation
        self.search.objects.observe_frame(observation)
        legal = tuple(observation.available_actions)
        if not legal:
            return "wait", {"policy": {"variant": self.name, "choice_reason": {"fallback": "no_legal_actions"}}}
        scores: dict[str, float] = {}
        details: dict[str, Any] = {}
        for action in legal:
            score, detail = self.search.score_action(
                observation,
                action,
                enable_component=self.variant.component,
                enable_change=self.variant.change,
                enable_graph=self.variant.graph,
                enable_event=self.variant.event,
                enable_object=self.variant.object_persistence,
                weights=self.weights,
            )
            scores[action] = score
            details[action] = detail
        chosen = max(legal, key=lambda action: (scores[action], -legal.index(action)))
        top = sorted(scores.items(), key=lambda item: item[1], reverse=True)[:8]
        enabled = self.variant.enabled_features()
        choice_was_affordance = bool(enabled) and scores[chosen] > 0.05
        self._last_choice_was_affordance = choice_was_affordance
        self.search.effect.record_choice(observation, chosen)
        self.choice_count += 1
        self.affordance_choices += int(choice_was_affordance)
        for name in enabled:
            if name in details[chosen]:
                self.reason_counts[name] = self.reason_counts.get(name, 0) + 1
        reason = {
            "variant": self.name,
            "enabled_features": sorted(enabled),
            "chosen_score": round(float(scores[chosen]), 6),
            "choice_was_affordance": choice_was_affordance,
            "chosen_detail": details[chosen],
            "top_scores": {action: round(float(score), 6) for action, score in top},
        }
        self.search.last_choice_reason = reason
        diagnostics = {
            "policy": {
                "variant": self.name,
                "chosen_action": chosen,
                "choice_reason": reason,
                "forced_cycle": False,
                "component_summary": self.search.diagnostics(observation)["component_summary"],
                "action_effect_memory": self.search.effect.snapshot(),
                "repeat_cycle_stats": self.search.graph.cycle_stats(observation),
            }
        }
        return chosen, diagnostics

    def observe(self, before: ArcAGI3Observation, action: str, result: ArcAGI3StepResult) -> None:
        self.observe_transition(action, result, before=before)

    def observe_transition(
        self,
        action: str,
        result: ArcAGI3StepResult,
        *,
        before: ArcAGI3Observation | None = None,
    ) -> None:
        prior = before or self._last_observation
        if prior is None:
            return
        self.search.observe(prior, action, result, choice_was_affordance=self._last_choice_was_affordance)

    def summary(self) -> dict[str, Any]:
        return {
            "variant": self.name,
            "choice_count": self.choice_count,
            "affordance_choices": self.affordance_choices,
            "affordance_choice_rate": float(self.affordance_choices / max(self.choice_count, 1)),
            "reason_counts": dict(self.reason_counts),
            "no_forced_cycle": True,
            "action_effect_hit_rate": self.search.effect.action_effect_hit_rate(),
            "no_op_avoidance_rate": self.search.effect.no_op_avoidance_rate(),
        }


def build_affordance_variants(*, include_ablations: bool = True) -> list[ArcAffordancePolicy]:
    variants = [ArcAffordancePolicy(variant) for variant in AFFORDANCE_VARIANTS]
    if include_ablations:
        variants.extend(ArcAffordancePolicy(variant) for variant in AFFORDANCE_ABLATIONS)
    return variants
