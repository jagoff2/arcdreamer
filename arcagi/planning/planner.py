from __future__ import annotations

from dataclasses import dataclass
import math

import numpy as np
import torch

from arcagi.core.action_schema import ActionSchemaContext, build_action_schema, build_action_schema_context, click_to_grid_cell, no_effect_family_key, parse_click_action
from arcagi.core.types import ActionName, ActionThought, LanguageTrace, PlanOutput, RuntimeThought, StructuredState
from arcagi.planning.rule_induction import EpisodeRuleInducer, ObjectSignature, action_target_signatures
from arcagi.memory.episodic import EpisodicMemory
from arcagi.memory.graph import StateGraph
from arcagi.models.language import GroundedLanguageModel
from arcagi.models.world_model import RecurrentWorldModel

GENERIC_MEMORY_TOKENS: frozenset[str] = frozenset(
    {
        "belief",
        "question",
        "goal",
        "need",
        "test",
        "unknown",
        "uncertain",
        "rule",
        "plan",
        "explore",
        "probe",
        "confirm",
        "move",
        "interact",
        "click",
        "select",
        "toward",
        "target",
        "action",
        "focus",
        "state",
        "direction",
        "color",
        "family",
        "because",
        "active",
        "inactive",
        "present",
        "absent",
        "visible",
        "hidden",
        "positive",
        "negative",
        "none",
        "near",
        "mid",
        "far",
        "high",
        "mid",
        "low",
    }
)


@dataclass(frozen=True)
class PlannerConfig:
    novelty_weight: float = 1.0
    value_weight: float = 1.0
    info_gain_weight: float = 0.8
    memory_weight: float = 0.6
    policy_prior_weight: float = 1.0
    graph_signal_weight: float = 1.0
    search_depth: int = 3
    search_root_width: int = 3
    search_branch_width: int = 2
    max_world_model_calls: int = 96
    discount: float = 0.7


@dataclass(frozen=True)
class _ImaginationStateProxy:
    affordances: tuple[ActionName, ...]
    action_roles: tuple[tuple[ActionName, str], ...]
    step_index: int
    projected_transition: tuple[float, ...]


INTERACT_DELTAS: dict[str, tuple[int, int]] = {
    "interact_up": (-1, 0),
    "interact_down": (1, 0),
    "interact_left": (0, -1),
    "interact_right": (0, 1),
}


def _interaction_grounding_score(state: StructuredState | _ImaginationStateProxy, action: ActionName) -> float:
    if not isinstance(state, StructuredState):
        return 0.0
    click = parse_click_action(action)
    if click is not None:
        grid_cell = click_to_grid_cell(
            click,
            grid_shape=state.grid_shape,
            inventory=state.inventory_dict(),
        )
        if grid_cell is None:
            return -0.7
        hit = next((obj for obj in state.objects if grid_cell in obj.cells), None)
        if hit is None:
            return -0.45
        if "agent" in hit.tags:
            return -0.1
        if "interactable" in hit.tags or "target" in hit.tags or "selector" in hit.tags:
            return 0.55
        return 0.18
    if action not in INTERACT_DELTAS:
        return 0.0
    agent = next((obj for obj in state.objects if "agent" in obj.tags), None)
    if agent is None or not agent.cells:
        return 0.0
    ay, ax = sorted(agent.cells)[0]
    dy, dx = INTERACT_DELTAS[action]
    target = (ay + dy, ax + dx)
    hit = next(
        (
            obj
            for obj in state.objects
            if obj is not agent and target in obj.cells
        ),
        None,
    )
    if hit is None:
        return -1.25
    if "interactable" in hit.tags or "target" in hit.tags or "selector" in hit.tags:
        return 0.45
    return 0.15


def _parametric_novelty_scale(schema: object, interaction_grounding: float) -> float:
    if not hasattr(schema, "coarse_bin") or getattr(schema, "coarse_bin") is None:
        return 1.0
    if interaction_grounding < -0.2:
        return 0.15
    if interaction_grounding < 0.25:
        return 0.45
    return 0.75


def _policy_bonus(
    predicted_reward: float,
    usefulness: float,
    policy_prior: float,
    policy_weight: float,
) -> float:
    base_progress = predicted_reward + (0.5 * usefulness)
    effective_weight = policy_weight if base_progress > 0.0 else (0.1 * policy_weight)
    return 0.35 * effective_weight * policy_prior


def _predicted_outcome_value(
    predicted_reward: float,
    usefulness: float,
    predicted_return: float,
    causal_value: float,
    policy_prior: float,
    policy_weight: float,
) -> float:
    return (
        predicted_reward
        + (0.5 * usefulness)
        + (0.35 * predicted_return)
        + (0.45 * causal_value)
        + _policy_bonus(
            predicted_reward,
            usefulness,
            policy_prior,
            policy_weight,
        )
    )


def _memory_entry_signal(
    *,
    score: float,
    salience: float,
    action_confidence: float,
) -> float:
    bounded_score = math.tanh(1.5 * float(score))
    bounded_salience = math.tanh(0.2 * max(float(salience), 0.0))
    bounded_confidence = math.tanh(0.7 * max(float(action_confidence), 0.0))
    return bounded_score * (0.5 + (0.5 * bounded_salience)) * (0.25 + (0.75 * bounded_confidence))


def _belief_inventory_value(state: StructuredState, key: str, default: str = "") -> str:
    return state.inventory_dict().get(key, default)


def _belief_flag_value(state: StructuredState, key: str, default: str = "0") -> str:
    return state.flags_dict().get(key, default)



def _move_direction_bonus(action: ActionName, direction: str, *, bonus: float) -> float:
    if not direction or direction == "none":
        return 0.0
    return bonus if action == direction else 0.0


def _spatial_action_bonus(state: StructuredState | _ImaginationStateProxy, action: ActionName, *, question_tokens: tuple[str, ...]) -> float:
    if not isinstance(state, StructuredState):
        return 0.0
    frontier_direction = _belief_inventory_value(state, "belief_frontier_direction", "none")
    anchor_direction = _belief_inventory_value(state, "belief_anchor_direction", "none")
    contradiction_direction = _belief_inventory_value(state, "belief_contradiction_direction", "none")
    frontier_distance = _belief_inventory_value(state, "belief_frontier_distance", "none")
    has_anchor = _belief_flag_value(state, "belief_has_spatial_anchor", "0") == "1"
    near_anchor = _belief_flag_value(state, "belief_near_spatial_anchor", "0") == "1"
    has_contradiction = _belief_flag_value(state, "belief_has_contradiction_hotspot", "0") == "1"
    wants_explore = any(token in {"need", "explore", "test", "uncertain"} for token in question_tokens)
    wants_confirm = any(token in {"confirm", "target"} for token in question_tokens)

    bonus = 0.0
    if action in {"up", "down", "left", "right"}:
        if frontier_distance != "none":
            bonus += _move_direction_bonus(action, frontier_direction, bonus=0.18 if wants_explore else 0.1)
        if has_anchor and not near_anchor:
            bonus += _move_direction_bonus(action, anchor_direction, bonus=0.28 if wants_confirm else 0.18)
        if has_contradiction:
            bonus += _move_direction_bonus(action, contradiction_direction, bonus=0.08 if wants_explore else 0.04)
    elif action == "wait" and frontier_distance != "none":
        bonus -= 0.15
    return bonus


def _context_bias_bonus(
    state: StructuredState,
    action: ActionName,
    *,
    schema,
    context_bias: dict[tuple[str, ObjectSignature], float] | None,
) -> float:
    if not context_bias:
        return 0.0
    signatures = action_target_signatures(state, action)
    if not signatures:
        return 0.0
    return sum(float(context_bias.get((schema.action_type, signature), 0.0)) for signature in signatures) / float(
        len(signatures)
    )


def _signature_color_tokens(signature: ObjectSignature) -> tuple[str, ...]:
    color = int(signature[0])
    named = {
        3: "red",
        4: "blue",
        5: "green",
        6: "yellow",
        7: "red",
        8: "blue",
    }.get(color)
    bucket = f"c{max(0, min(color, 11))}"
    if named is None:
        return (bucket,)
    return (named, bucket)


def _plan_alignment_bonus(
    state: StructuredState | _ImaginationStateProxy,
    action: ActionName,
    *,
    schema,
    plan_tokens: tuple[str, ...],
) -> float:
    if not plan_tokens:
        return 0.0
    token_set = set(plan_tokens)
    score = 0.0
    if "action" in token_set and schema.action_type in token_set:
        score += 0.28
    elif schema.action_type in token_set:
        score += 0.16
    if schema.direction is not None and schema.direction in token_set:
        score += 0.22
    if schema.action_type == "wait" and "wait" in token_set:
        score += 0.2
    if "focus" in token_set:
        if "target" in token_set and schema.action_type == "move":
            score += 0.12
        if "interactable" in token_set and schema.action_type == "interact":
            score += 0.16
        if "rule" in token_set and schema.action_type in {"click", "select"}:
            score += 0.16
        if "frontier" in token_set and schema.action_type == "move":
            score += 0.08
    if isinstance(state, StructuredState):
        target_signatures = action_target_signatures(state, action)
        if target_signatures:
            signature_tokens = {
                token
                for signature in target_signatures
                for token in _signature_color_tokens(signature)
            }
            if token_set.intersection(signature_tokens):
                score += 0.18
    return score


class HybridPlanner:
    def __init__(self, config: PlannerConfig | None = None) -> None:
        self.config = config or PlannerConfig()

    @torch.no_grad()
    def build_runtime_thought(
        self,
        state: StructuredState,
        latent: torch.Tensor | None,
        graph: StateGraph,
        world_model: RecurrentWorldModel | None = None,
        hidden: torch.Tensor | None = None,
        language_model: GroundedLanguageModel | None = None,
    ) -> RuntimeThought:
        belief_tokens = ()
        question_tokens = ()
        plan_tokens = ()
        if language_model is not None and latent is not None:
            belief_tokens = language_model.decode(latent, mode="belief")
            question_tokens = language_model.decode(latent, mode="question")
            plan_tokens = language_model.decode(latent, mode="plan")
        if world_model is None or latent is None:
            return RuntimeThought(
                belief_tokens=belief_tokens,
                question_tokens=question_tokens,
                plan_tokens=plan_tokens,
            )
        action_thoughts, world_model_calls = self._build_action_thoughts(
            state,
            latent,
            world_model,
            hidden,
            graph=graph,
        )
        return RuntimeThought(
            belief_tokens=belief_tokens,
            question_tokens=question_tokens,
            plan_tokens=plan_tokens,
            actions=action_thoughts,
            world_model_calls=world_model_calls,
        )

    @torch.no_grad()
    def choose_action(
        self,
        state: StructuredState,
        latent: torch.Tensor | None,
        graph: StateGraph,
        world_model: RecurrentWorldModel | None = None,
        hidden: torch.Tensor | None = None,
        language_model: GroundedLanguageModel | None = None,
        episodic_memory: EpisodicMemory | None = None,
        rule_inducer: EpisodeRuleInducer | None = None,
        action_bias: dict[ActionName, float] | None = None,
        action_counts: dict[ActionName, int] | None = None,
        action_delta_sums: dict[ActionName, float] | None = None,
        action_reward_sums: dict[ActionName, float] | None = None,
        action_no_effect_counts: dict[ActionName, int] | None = None,
        family_counts: dict[str, int] | None = None,
        family_bins: dict[str, set[tuple[int, int]]] | None = None,
        family_no_effect_counts: dict[str, int] | None = None,
        family_bias: dict[str, float] | None = None,
        context_bias: dict[tuple[str, ObjectSignature], float] | None = None,
        diagnostic_action_scores: dict[ActionName, float] | None = None,
        stuck_steps: int = 0,
        last_action: ActionName | None = None,
        thought: RuntimeThought | None = None,
        objective_stall_pressure: float = 0.0,
        level_action_counts: dict[ActionName, int] | None = None,
        level_action_progress_sums: dict[ActionName, float] | None = None,
        level_action_no_objective_counts: dict[ActionName, int] | None = None,
    ) -> PlanOutput:
        if thought is None:
            thought = self.build_runtime_thought(
                state=state,
                latent=latent,
                graph=graph,
                world_model=world_model,
                hidden=hidden,
                language_model=language_model,
            )
        belief_tokens = thought.belief_tokens
        question_tokens = thought.question_tokens
        plan_tokens = thought.plan_tokens
        search_candidates = self._ordered_search_actions(
            state=state,
            thought=thought,
            graph=graph,
            action_bias=action_bias,
            action_counts=action_counts,
            family_counts=family_counts,
            family_bins=family_bins,
            action_no_effect_counts=action_no_effect_counts,
            family_no_effect_counts=family_no_effect_counts,
            family_bias=family_bias,
            context_bias=context_bias,
            diagnostic_action_scores=diagnostic_action_scores,
            stuck_steps=stuck_steps,
            last_action=last_action,
            objective_stall_pressure=objective_stall_pressure,
            level_action_counts=level_action_counts,
            level_action_progress_sums=level_action_progress_sums,
            level_action_no_objective_counts=level_action_no_objective_counts,
        )
        search_roots = set(search_candidates[: max(self.config.search_root_width, 0)])
        search_budget = max(self.config.max_world_model_calls - thought.world_model_calls, 0)
        context = build_action_schema_context(state.affordances, dict(state.action_roles))

        best_action = state.affordances[0]
        best_total = float("-inf")
        best_scores: dict[str, float] = {}
        latent_np = latent.squeeze(0).detach().cpu().numpy() if latent is not None else state.summary_vector()
        memory_query_tokens = tuple(
            token
            for token in dict.fromkeys(belief_tokens + question_tokens + plan_tokens + thought.claim_tokens(limit=6))
            if token not in GENERIC_MEMORY_TOKENS
        )
        for action in state.affordances:
            schema = build_action_schema(action, context)
            graph_weight = max(float(self.config.graph_signal_weight), 0.0)
            if graph_weight > 0.0:
                novelty = graph_weight * graph.action_novelty(state, action)
                entropy = graph_weight * graph.action_outcome_entropy(state, action)
                cycle_penalty = graph_weight * graph.action_cycle_penalty(state, action)
                graph_stats = graph.get_action_stats(state, action)
                empirical_reward = graph_weight * graph_stats.mean_reward
            else:
                novelty = 0.0
                entropy = 0.0
                cycle_penalty = 0.0
                graph_stats = None
                empirical_reward = 0.0
            effect_family = no_effect_family_key(schema)
            global_count = 0 if action_counts is None else int(action_counts.get(action, 0))
            level_count = 0 if level_action_counts is None else int(level_action_counts.get(action, 0))
            level_mean_progress = (
                0.0
                if level_action_progress_sums is None or level_count == 0
                else float(level_action_progress_sums.get(action, 0.0)) / float(level_count)
            )
            level_no_objective_rate = (
                0.0
                if level_action_no_objective_counts is None or level_count == 0
                else float(level_action_no_objective_counts.get(action, 0)) / float(level_count)
            )
            global_novelty = 1.0 / math.sqrt(global_count + 1.0)
            mean_delta = 0.0 if action_delta_sums is None or global_count == 0 else float(action_delta_sums.get(action, 0.0)) / global_count
            mean_reward = 0.0 if action_reward_sums is None or global_count == 0 else float(action_reward_sums.get(action, 0.0)) / global_count
            action_no_effect_rate = (
                0.0
                if action_no_effect_counts is None or global_count == 0
                else float(action_no_effect_counts.get(action, 0)) / float(global_count)
            )
            family_count = 0 if family_counts is None else int(family_counts.get(effect_family, 0))
            family_novelty = 1.0 / math.sqrt(family_count + 1.0)
            family_no_effect_rate = (
                0.0
                if family_no_effect_counts is None or family_count == 0
                else float(family_no_effect_counts.get(effect_family, 0)) / float(family_count)
            )
            family_online_bias = 0.0 if family_bias is None else float(family_bias.get(schema.family, 0.0))
            interaction_grounding = _interaction_grounding_score(state, action)
            parametric_novelty_scale = _parametric_novelty_scale(schema, interaction_grounding)
            if schema.coarse_bin is not None:
                global_novelty = min(global_novelty, parametric_novelty_scale * family_novelty)
            spatial_bonus = _spatial_action_bonus(state, action, question_tokens=question_tokens)
            context_online_bias = _context_bias_bonus(
                state,
                action,
                schema=schema,
                context_bias=context_bias,
            )
            diagnostic_bonus = 0.0 if diagnostic_action_scores is None else float(diagnostic_action_scores.get(action, 0.0))
            plan_alignment = _plan_alignment_bonus(
                state,
                action,
                schema=schema,
                plan_tokens=plan_tokens,
            )
            question_alignment = _plan_alignment_bonus(
                state,
                action,
                schema=schema,
                plan_tokens=question_tokens,
            )
            plan_alignment += 0.75 * question_alignment
            wait_penalty = 0.35 if action == "wait" else 0.0
            parameter_bonus = 0.0
            if schema.coarse_bin is not None and family_bins is not None and schema.coarse_bin not in family_bins.get(effect_family, set()):
                parameter_bonus = 0.35 * max(0.0, 1.0 - family_no_effect_rate)
            repeat_penalty = 0.35 if last_action == action else 0.0
            stuck_bonus = (0.1 * min(stuck_steps, 8)) * global_novelty
            value = 0.0
            search = 0.0
            disagreement = 0.0
            policy_prior = 0.0
            policy_weight = 0.0
            diagnostic_model_bonus = 0.0
            action_thought = thought.for_action(action)
            if action_thought is not None:
                value = action_thought.value
                disagreement = action_thought.uncertainty
                policy_prior = action_thought.policy
                policy_weight = action_thought.policy_weight
                diagnostic_model_bonus = action_thought.diagnostic_value
            elif world_model is not None and latent is not None:
                prediction = world_model.step(latent, actions=[action], state=state, hidden=hidden)
                policy_prior = float(prediction.policy.item())
                graph_visits = 0 if graph_stats is None else int(graph_stats.visits)
                policy_weight = 0.35 / float(graph_visits + 1)
                predicted_reward = float(prediction.reward.item())
                predicted_return = float(prediction.return_value.item())
                causal_value = float(prediction.causal_value.item())
                usefulness = float(prediction.usefulness.item())
                value = _predicted_outcome_value(
                    predicted_reward,
                    usefulness,
                    predicted_return,
                    causal_value,
                    policy_prior,
                    policy_weight,
                )
                disagreement = float(prediction.uncertainty.item())
                diagnostic_model_bonus = float(prediction.diagnostic_value.item())
            if (
                world_model is not None
                and latent is not None
                and self.config.search_depth > 1
                and action in search_roots
            ):
                prediction = None
                if action_thought is not None and action_thought.next_latent is not None and action_thought.next_hidden is not None:
                    prediction = action_thought
                elif search_budget > 0:
                    prediction = world_model.step(latent, actions=[action], state=state, hidden=hidden)
                    search_budget -= 1
                if prediction is not None:
                    next_latent = prediction.next_latent if action_thought is not None and prediction is action_thought else prediction.next_latent_mean
                    next_hidden = prediction.next_hidden if action_thought is not None and prediction is action_thought else prediction.hidden
                    next_state = (
                        prediction.next_state_proxy
                        if action_thought is not None and prediction is action_thought
                        else self._next_state_proxy(state, prediction.delta)
                    )
                    budget_ref = [search_budget]
                    search = self._lookahead(
                        next_latent,
                        next_hidden,
                        world_model,
                        depth=self.config.search_depth - 1,
                        state=next_state,
                        budget=budget_ref,
                        branch_width=self.config.search_branch_width,
                    )
                    search_budget = budget_ref[0]
            memory_bonus = 0.0
            if episodic_memory is not None and memory_query_tokens:
                memory_support = 0.0
                memory_avoid = 0.0
                for score, entry in episodic_memory.query(latent_np, query_tokens=memory_query_tokens, top_k=3):
                    signal = _memory_entry_signal(
                        score=score,
                        salience=entry.salience,
                        action_confidence=float(entry.payload.get("action_confidence", 0.0)),
                    )
                    if (
                        entry.payload.get("recommended_action") == action
                        and not bool(entry.payload.get("counterfactual", False))
                    ):
                        memory_support += signal
                    if entry.payload.get("avoid_action") == action:
                        memory_avoid += abs(signal)
                memory_bonus = 2.0 * math.tanh((memory_support - memory_avoid) / 2.0)
            induced_bonus = rule_inducer.action_score(state, action) if rule_inducer is not None else 0.0
            online_bias = 0.0 if action_bias is None else float(action_bias.get(action, 0.0))
            prior_reliability = max(
                0.2,
                min(
                    1.0,
                    1.0
                    - (0.7 * max(action_no_effect_rate, family_no_effect_rate))
                    - (0.45 * cycle_penalty)
                    - (0.25 * float(last_action == action)),
                ),
            )
            stale_move_prior = bool(
                float(objective_stall_pressure) >= 0.25
                and schema.action_type == "move"
                and global_count >= 32
                and level_count >= 8
                and level_mean_progress <= 0.04
                and level_no_objective_rate >= 0.85
            )
            if stale_move_prior:
                prior_reliability = min(prior_reliability, 0.08)
            effective_online_bias = online_bias * prior_reliability if online_bias > 0.0 else online_bias
            effective_family_online_bias = (
                family_online_bias * prior_reliability if family_online_bias > 0.0 else family_online_bias
            )
            effective_induced_bonus = (1.35 * math.tanh(induced_bonus / 1.35)) * (0.25 + (0.75 * prior_reliability))
            effective_memory_bonus = memory_bonus * (0.35 + (0.65 * prior_reliability))
            effective_diagnostic_bonus = (
                diagnostic_bonus * 0.15
                if stale_move_prior
                else diagnostic_bonus * (1.0 + (0.5 * (1.0 - prior_reliability)))
            )
            if schema.coarse_bin is not None and interaction_grounding < -0.2:
                effective_diagnostic_bonus *= 0.35
            effective_policy_prior = policy_prior * max(policy_weight, 0.15) * prior_reliability
            effective_plan_alignment = plan_alignment
            if stale_move_prior and effective_plan_alignment > 0.0:
                effective_plan_alignment *= 0.20
            stale_move_penalty = 1.5 * float(objective_stall_pressure) if stale_move_prior else 0.0
            total = (
                self.config.novelty_weight * novelty
                + self.config.value_weight * (value + search)
                + self.config.info_gain_weight * (entropy + disagreement)
                + self.config.memory_weight * effective_memory_bonus
                + self.config.policy_prior_weight * effective_policy_prior
                + 0.8 * empirical_reward
                + 0.5 * effective_induced_bonus
                + effective_online_bias
                + (0.8 * global_novelty)
                + (0.6 * family_novelty * max(0.0, 1.0 - family_no_effect_rate))
                + (0.003 * mean_delta)
                + (0.5 * mean_reward)
                + parameter_bonus
                + effective_family_online_bias
                + interaction_grounding
                + spatial_bonus
                + context_online_bias
                + (1.8 * effective_diagnostic_bonus)
                + (0.75 * diagnostic_model_bonus)
                + (1.6 * effective_plan_alignment)
                + stuck_bonus
                - 0.9 * cycle_penalty
                - (1.5 * action_no_effect_rate)
                - (1.25 * family_no_effect_rate)
                - repeat_penalty
                - wait_penalty
                - stale_move_penalty
            )
            if total > best_total:
                best_total = total
                best_action = action
                best_scores = {
                    "total": total,
                    "novelty": novelty,
                    "value": value,
                    "search": search,
                    "entropy": entropy,
                    "disagreement": disagreement,
                    "empirical_reward": empirical_reward,
                    "memory": effective_memory_bonus,
                    "induced": effective_induced_bonus,
                    "policy_prior": policy_prior,
                    "policy_weight": policy_weight,
                    "effective_policy_prior": effective_policy_prior,
                    "online_bias": effective_online_bias,
                    "global_novelty": global_novelty,
                    "family_novelty": family_novelty,
                    "mean_delta": mean_delta,
                    "mean_reward": mean_reward,
                    "parameter_bonus": parameter_bonus,
                    "family_online_bias": effective_family_online_bias,
                    "interaction_grounding": interaction_grounding,
                    "spatial_bonus": spatial_bonus,
                    "context_online_bias": context_online_bias,
                    "diagnostic_bonus": effective_diagnostic_bonus,
                    "diagnostic_model": diagnostic_model_bonus,
                    "plan_alignment": plan_alignment,
                    "question_alignment": question_alignment,
                    "effective_plan_alignment": effective_plan_alignment,
                    "level_action_count": float(level_count),
                    "level_mean_progress": level_mean_progress,
                    "level_no_objective_rate": level_no_objective_rate,
                    "stale_move_prior": float(stale_move_prior),
                    "stale_move_penalty": stale_move_penalty,
                    "wait_penalty": wait_penalty,
                    "stuck_bonus": stuck_bonus,
                    "repeat_penalty": repeat_penalty,
                    "action_no_effect_rate": action_no_effect_rate,
                    "family_no_effect_rate": family_no_effect_rate,
                    "cycle_penalty": cycle_penalty,
                    "prior_reliability": prior_reliability,
                    "search_budget_remaining": float(search_budget),
                }
        return PlanOutput(
            action=best_action,
            scores=best_scores,
            language=LanguageTrace(
                belief_tokens=belief_tokens,
                question_tokens=question_tokens,
                plan_tokens=plan_tokens,
            ),
            search_path=(best_action,),
        )

    def _lookahead(
        self,
        latent: torch.Tensor,
        hidden: torch.Tensor,
        world_model: RecurrentWorldModel,
        depth: int,
        state: StructuredState | _ImaginationStateProxy,
        budget: list[int],
        branch_width: int,
    ) -> float:
        if depth <= 0 or budget[0] <= 0:
            return 0.0
        thought, _calls = self._build_action_thoughts(
            state,
            latent,
            world_model,
            hidden,
            budget=budget,
        )
        if not thought:
            return 0.0
        ranked_actions = self._ordered_imagined_actions(state, thought)
        scores: list[float] = []
        width = max(1, min(branch_width, len(ranked_actions)))
        for action in ranked_actions[:width]:
            action_thought = next((candidate for candidate in thought if candidate.action == action), None)
            if action_thought is None or action_thought.next_latent is None or action_thought.next_hidden is None:
                continue
            immediate = (
                _predicted_outcome_value(
                    action_thought.predicted_reward,
                    action_thought.usefulness,
                    action_thought.predicted_return,
                    action_thought.causal_value,
                    action_thought.policy,
                    action_thought.policy_weight,
                )
                + (0.25 * action_thought.diagnostic_value)
                - (0.1 * action_thought.uncertainty)
            )
            future = self._lookahead(
                action_thought.next_latent,
                action_thought.next_hidden,
                world_model,
                depth=depth - 1,
                state=action_thought.next_state_proxy or state,
                budget=budget,
                branch_width=branch_width,
            )
            scores.append(immediate + self.config.discount * future)
        return max(scores) if scores else 0.0

    def _ordered_search_actions(
        self,
        state: StructuredState,
        thought: RuntimeThought,
        graph: StateGraph,
        action_bias: dict[ActionName, float] | None,
        action_counts: dict[ActionName, int] | None,
        family_counts: dict[str, int] | None,
        family_bins: dict[str, set[tuple[int, int]]] | None,
        action_no_effect_counts: dict[ActionName, int] | None,
        family_no_effect_counts: dict[str, int] | None,
        family_bias: dict[str, float] | None,
        context_bias: dict[tuple[str, ObjectSignature], float] | None,
        diagnostic_action_scores: dict[ActionName, float] | None,
        stuck_steps: int,
        last_action: ActionName | None,
        objective_stall_pressure: float = 0.0,
        level_action_counts: dict[ActionName, int] | None = None,
        level_action_progress_sums: dict[ActionName, float] | None = None,
        level_action_no_objective_counts: dict[ActionName, int] | None = None,
    ) -> tuple[ActionName, ...]:
        context = build_action_schema_context(state.affordances, dict(state.action_roles))
        def rank_score(action: ActionName) -> float:
            schema = build_action_schema(action, context)
            effect_family = no_effect_family_key(schema)
            global_count = 0 if action_counts is None else int(action_counts.get(action, 0))
            family_count = 0 if family_counts is None else int(family_counts.get(effect_family, 0))
            action_no_effect_rate = (
                0.0
                if action_no_effect_counts is None or global_count == 0
                else float(action_no_effect_counts.get(action, 0)) / float(global_count)
            )
            family_no_effect_rate = (
                0.0
                if family_no_effect_counts is None or family_count == 0
                else float(family_no_effect_counts.get(effect_family, 0)) / float(family_count)
            )
            level_count = 0 if level_action_counts is None else int(level_action_counts.get(action, 0))
            level_mean_progress = (
                0.0
                if level_action_progress_sums is None or level_count == 0
                else float(level_action_progress_sums.get(action, 0.0)) / float(level_count)
            )
            level_no_objective_rate = (
                0.0
                if level_action_no_objective_counts is None or level_count == 0
                else float(level_action_no_objective_counts.get(action, 0)) / float(level_count)
            )
            stale_move_prior = (
                float(objective_stall_pressure) >= 0.25
                and schema.action_type == "move"
                and global_count >= 32
                and level_count >= 8
                and level_mean_progress <= 0.04
                and level_no_objective_rate >= 0.85
            )
            parameter_bonus = 0.0
            if (
                schema.coarse_bin is not None
                and family_bins is not None
                and schema.coarse_bin not in family_bins.get(effect_family, set())
                and family_no_effect_rate < 0.75
            ):
                parameter_bonus = 0.35
            interaction_grounding = _interaction_grounding_score(state, action)
            parametric_novelty_scale = _parametric_novelty_scale(schema, interaction_grounding)
            action_novelty = (0.8 / math.sqrt(global_count + 1.0)) * parametric_novelty_scale
            family_novelty = 0.6 / math.sqrt(family_count + 1.0)
            diagnostic_bonus = 0.0 if diagnostic_action_scores is None else float(diagnostic_action_scores.get(action, 0.0))
            if schema.coarse_bin is not None and interaction_grounding < -0.2:
                diagnostic_bonus *= 0.35
            return (
                action_novelty
                + family_novelty
                + (0.0 if family_bias is None else float(family_bias.get(schema.family, 0.0)))
                + interaction_grounding
                + _context_bias_bonus(state, action, schema=schema, context_bias=context_bias)
                + (1.55 * diagnostic_bonus)
                + thought.value_for(action)
                + (0.85 * thought.diagnostic_value_for(action))
                + (0.8 * thought.selector_followup_for(action))
                + (0.4 * thought.uncertainty_for(action))
                + (float(self.config.graph_signal_weight) * 0.35 * graph.action_novelty(state, action))
                + (float(self.config.graph_signal_weight) * 0.25 * graph.action_outcome_entropy(state, action))
                + (0.0 if action_bias is None else float(action_bias.get(action, 0.0)))
                + parameter_bonus
                + ((0.1 * min(stuck_steps, 8)) * (1.0 / math.sqrt(global_count + 1.0)) * parametric_novelty_scale)
                - (float(self.config.graph_signal_weight) * 0.9 * graph.action_cycle_penalty(state, action))
                - (0.35 if action == "wait" else 0.0)
                - (1.5 * action_no_effect_rate)
                - (1.25 * family_no_effect_rate)
                - (0.35 if last_action == action else 0.0)
                - ((1.5 * float(objective_stall_pressure)) if stale_move_prior else 0.0)
            )

        ranked = sorted(state.affordances, key=rank_score, reverse=True)
        return tuple(ranked)

    def _build_action_thoughts(
        self,
        state: StructuredState | _ImaginationStateProxy,
        latent: torch.Tensor,
        world_model: RecurrentWorldModel,
        hidden: torch.Tensor | None,
        *,
        graph: StateGraph | None = None,
        budget: list[int] | None = None,
    ) -> tuple[tuple[ActionThought, ...], int]:
        affordances = tuple(getattr(state, "affordances", ()))
        action_roles = tuple(getattr(state, "action_roles", ()))
        context = build_action_schema_context(affordances, dict(action_roles))
        move_actions = [
            action
            for action in affordances
            if build_action_schema(action, context).action_type == "move"
        ]
        predictions: dict[
            ActionName,
            tuple[object, float, float, float, float, float, float, float, float, float, _ImaginationStateProxy],
        ] = {}
        world_model_calls = 0
        baseline_move_value = float("-inf")
        if budget is not None and budget[0] <= 0:
            return (), 0
        if budget is None:
            predicted_actions = affordances
        else:
            predicted_actions = affordances[: max(int(budget[0]), 0)]
            budget[0] -= len(predicted_actions)
        if not predicted_actions:
            return (), 0
        followup_budget = budget
        if followup_budget is None:
            followup_budget = [max(int(self.config.max_world_model_calls) - len(predicted_actions), 0)]
        repeated_latent = latent.repeat(len(predicted_actions), 1)
        repeated_hidden = None if hidden is None else hidden.repeat(len(predicted_actions), 1)
        batch_prediction = world_model.step(
            repeated_latent,
            actions=predicted_actions,
            state=state if isinstance(state, StructuredState) else None,
            hidden=repeated_hidden,
        )
        if int(batch_prediction.policy.reshape(-1).shape[0]) == len(predicted_actions):
            world_model_calls += len(predicted_actions)
            for index, action in enumerate(predicted_actions):
                graph_visits = 0
                if graph is not None and isinstance(state, StructuredState) and float(self.config.graph_signal_weight) > 0.0:
                    graph_visits = graph.get_action_stats(state, action).visits
                policy_weight = 0.35 / float(graph_visits + 1)
                policy_prior = math.tanh(float(batch_prediction.policy[index].item()) / 2.0)
                predicted_reward = float(batch_prediction.reward[index].item())
                predicted_return = float(batch_prediction.return_value[index].item())
                causal_value = float(batch_prediction.causal_value[index].item())
                diagnostic_value = float(batch_prediction.diagnostic_value[index].item())
                usefulness = float(batch_prediction.usefulness[index].item())
                uncertainty = float(batch_prediction.uncertainty[index].item())
                value = _predicted_outcome_value(
                    predicted_reward,
                    usefulness,
                    predicted_return,
                    causal_value,
                    policy_prior,
                    policy_weight,
                )
                next_latent = batch_prediction.next_latent_mean[index : index + 1]
                next_hidden = batch_prediction.hidden[index : index + 1]
                next_state_proxy = self._next_state_proxy(state, batch_prediction.delta[index])
                predictions[action] = (
                    (next_latent, next_hidden),
                    value,
                    uncertainty,
                    policy_prior,
                    policy_weight,
                    predicted_reward,
                    predicted_return,
                    causal_value,
                    diagnostic_value,
                    usefulness,
                    next_state_proxy,
                )
                if build_action_schema(action, context).action_type == "move":
                    baseline_move_value = max(baseline_move_value, value - (0.1 * uncertainty))
        else:
            for action in predicted_actions:
                prediction = world_model.step(latent, actions=[action], state=state, hidden=hidden)
                world_model_calls += 1
                graph_visits = 0
                if graph is not None and isinstance(state, StructuredState) and float(self.config.graph_signal_weight) > 0.0:
                    graph_visits = graph.get_action_stats(state, action).visits
                policy_weight = 0.35 / float(graph_visits + 1)
                policy_prior = math.tanh(float(prediction.policy.item()) / 2.0)
                predicted_reward = float(prediction.reward.item())
                predicted_return = float(prediction.return_value.item())
                causal_value = float(prediction.causal_value.item())
                diagnostic_value = float(prediction.diagnostic_value.item())
                usefulness = float(prediction.usefulness.item())
                uncertainty = float(prediction.uncertainty.item())
                value = _predicted_outcome_value(
                    predicted_reward,
                    usefulness,
                    predicted_return,
                    causal_value,
                    policy_prior,
                    policy_weight,
                )
                next_state_proxy = self._next_state_proxy(state, prediction.delta)
                predictions[action] = (
                    (prediction.next_latent_mean, prediction.hidden),
                    value,
                    uncertainty,
                    policy_prior,
                    policy_weight,
                    predicted_reward,
                    predicted_return,
                    causal_value,
                    diagnostic_value,
                    usefulness,
                    next_state_proxy,
                )
                if build_action_schema(action, context).action_type == "move":
                    baseline_move_value = max(baseline_move_value, value - (0.1 * uncertainty))
        if baseline_move_value == float("-inf"):
            baseline_move_value = 0.0

        action_thoughts: list[ActionThought] = []
        for action, (
            prediction,
            value,
            uncertainty,
            policy_prior,
            policy_weight,
            predicted_reward,
            predicted_return,
            causal_value,
            diagnostic_value,
            usefulness,
            next_state_proxy,
        ) in predictions.items():
            selector_followup = 0.0
            schema = build_action_schema(action, context)
            if schema.action_type in {"click", "select"} and move_actions:
                move_batch = move_actions
                if followup_budget is not None:
                    if followup_budget[0] <= 0:
                        move_batch = []
                    else:
                        move_batch = move_actions[: max(int(followup_budget[0]), 0)]
                        followup_budget[0] -= len(move_batch)
                if move_batch:
                    next_latent, next_hidden = prediction
                    followup = world_model.step(
                        next_latent.repeat(len(move_batch), 1),
                        actions=move_batch,
                        state=None,
                        hidden=next_hidden.repeat(len(move_batch), 1),
                    )
                    world_model_calls += len(move_batch)
                    followup_scores = (
                        followup.reward
                        + (0.5 * followup.usefulness)
                        + (0.35 * followup.return_value)
                        + (0.45 * followup.causal_value)
                        + (0.35 * 0.25 * torch.tanh(followup.policy / 2.0))
                        + (0.25 * followup.diagnostic_value)
                        - (0.1 * followup.uncertainty)
                    )
                    selector_followup = float(torch.max(followup_scores).item()) - baseline_move_value
            action_thoughts.append(
                ActionThought(
                    action=action,
                    value=value,
                    uncertainty=uncertainty,
                    policy=policy_prior,
                    policy_weight=policy_weight,
                    predicted_reward=predicted_reward,
                    predicted_return=predicted_return,
                    causal_value=causal_value,
                    diagnostic_value=diagnostic_value,
                    usefulness=usefulness,
                    selector_followup=selector_followup,
                    next_latent=prediction[0],
                    next_hidden=prediction[1],
                    next_state_proxy=next_state_proxy,
                )
            )
        return tuple(action_thoughts), world_model_calls

    def _next_state_proxy(
        self,
        state: StructuredState | _ImaginationStateProxy,
        predicted_delta: torch.Tensor,
    ) -> _ImaginationStateProxy:
        base_transition = self._transition_projection(state)
        next_transition = tuple(
            (base_transition + predicted_delta.detach().cpu().reshape(-1).numpy()).astype("float32").tolist()
        )
        return _ImaginationStateProxy(
            affordances=tuple(getattr(state, "affordances", ())),
            action_roles=tuple(getattr(state, "action_roles", ())),
            step_index=int(getattr(state, "step_index", 0)) + 1,
            projected_transition=next_transition,
        )

    def _ordered_imagined_actions(
        self,
        state: StructuredState | _ImaginationStateProxy,
        thought: tuple[ActionThought, ...],
    ) -> tuple[ActionName, ...]:
        context = build_action_schema_context(tuple(getattr(state, "affordances", ())), dict(getattr(state, "action_roles", ())))
        thought_by_action = {candidate.action: candidate for candidate in thought}
        ranked = sorted(
            thought_by_action.keys(),
            key=lambda action: (
                thought_by_action[action].value
                + (0.45 * thought_by_action[action].diagnostic_value)
                + (0.8 * thought_by_action[action].selector_followup)
                - (0.15 * thought_by_action[action].uncertainty)
                + (0.1 if build_action_schema(action, context).action_type in {"move", "interact"} else 0.0)
            ),
            reverse=True,
        )
        return tuple(ranked)

    @staticmethod
    def _transition_projection(state: StructuredState | _ImaginationStateProxy) -> np.ndarray:
        if isinstance(state, _ImaginationStateProxy):
            return np.asarray(state.projected_transition, dtype=np.float32)
        return state.transition_vector()
