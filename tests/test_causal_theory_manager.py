from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import torch

from arcagi.core.types import (
    ActionThought,
    GridObservation,
    LanguageTrace,
    ObjectState,
    PlanOutput,
    RuntimeThought,
    StructuredState,
    Transition,
)
from arcagi.evaluation.harness import build_agent
from arcagi.planning.rule_induction import EpisodeRuleInducer
from arcagi.reasoning import EpisodeTheoryManager


def _object(
    object_id: str,
    color: int,
    cells: tuple[tuple[int, int], ...],
    *,
    tags: tuple[str, ...],
) -> ObjectState:
    ys = [cell[0] for cell in cells]
    xs = [cell[1] for cell in cells]
    return ObjectState(
        object_id=object_id,
        color=color,
        cells=cells,
        bbox=(min(ys), min(xs), max(ys), max(xs)),
        centroid=(float(sum(ys)) / len(ys), float(sum(xs)) / len(xs)),
        area=len(cells),
        tags=tags,
    )


def _interaction_state(*, step_index: int = 0, right_removed: bool = False) -> StructuredState:
    agent = _object("agent", 2, ((1, 1),), tags=("agent",))
    left = _object("left", 3, ((1, 0),), tags=("interactable",))
    objects = [agent, left]
    if not right_removed:
        objects.append(_object("right", 4, ((1, 2),), tags=("interactable",)))
    grid = np.zeros((3, 3), dtype=np.int64)
    for obj in objects:
        for y, x in obj.cells:
            grid[y, x] = obj.color
    return StructuredState(
        task_id="theory/test",
        episode_id="episode-0",
        step_index=step_index,
        grid_shape=grid.shape,
        grid_signature=tuple(int(value) for value in grid.reshape(-1)),
        objects=tuple(objects),
        relations=(),
        affordances=("interact_right", "wait"),
        action_roles=(("interact_right", "interact"), ("wait", "wait")),
    )


def _interaction_thought(state: StructuredState) -> RuntimeThought:
    current = state.transition_vector()
    projected = current.copy()
    projected[1] = min(projected[1] + 0.2, 1.0)
    return RuntimeThought(
        actions=(
            ActionThought(
                action="interact_right",
                value=0.55,
                uncertainty=0.6,
                policy=0.2,
                predicted_reward=0.45,
                usefulness=0.3,
                next_state_proxy=SimpleNamespace(projected_transition=tuple(projected.tolist())),
            ),
        )
    )


def _move_observation(*, step_index: int, agent_x: int = 1) -> GridObservation:
    grid = np.zeros((3, 3), dtype=np.int64)
    grid[1, agent_x] = 2
    return GridObservation(
        task_id="hybrid-theory/test",
        episode_id="episode-0",
        step_index=step_index,
        grid=grid,
        available_actions=("left", "right"),
        extras={
            "action_roles": {
                "left": "move_left",
                "right": "move_right",
            },
            "cell_tags": {
                (1, agent_x): ("agent",),
            },
        },
    )


def test_theory_manager_emits_rival_claims_and_diagnostic_question_on_consistency_conflict() -> None:
    before = _interaction_state(step_index=0, right_removed=False)
    no_effect_after = _interaction_state(step_index=1, right_removed=False)
    inducer = EpisodeRuleInducer()
    inducer.record(
        Transition(
            state=before,
            action="interact_right",
            reward=0.0,
            next_state=no_effect_after,
            terminated=False,
        )
    )
    manager = EpisodeTheoryManager()

    augmented = manager.augment_runtime_thought(
        before,
        _interaction_thought(before),
        rule_inducer=inducer,
    )

    assert any(claim.claim_type == "rule" for claim in augmented.claims)
    assert manager.rule_candidates
    assert manager.competitions
    assert manager.rule_candidates[0].preconditions
    assert manager.rule_candidates[0].effects
    assert manager.rule_candidates[0].diagnostic_actions
    assert any(claim.claim_type == "consistency" for claim in augmented.claims)
    assert {"need", "test", "rule"}.issubset(set(augmented.question_tokens))
    assert float(manager.action_bias.get("interact_right", 0.0)) > 0.0
    assert float(manager.diagnostic_action_scores.get("interact_right", 0.0)) > 0.0


def test_theory_manager_records_contradiction_support_and_counterfactual_events() -> None:
    before = _interaction_state(step_index=0, right_removed=False)
    no_effect_after = _interaction_state(step_index=1, right_removed=False)
    manager = EpisodeTheoryManager()
    manager.augment_runtime_thought(
        before,
        _interaction_thought(before),
        rule_inducer=EpisodeRuleInducer(),
    )

    manager.observe_transition(
        Transition(
            state=before,
            action="interact_right",
            reward=0.0,
            next_state=no_effect_after,
            terminated=False,
        )
    )
    transition_events = manager.consume_recent_events()

    assert any(event.event_type == "support" and event.theory.effect_kind == "no_effect" for event in transition_events)
    assert any(event.event_type == "contradiction" for event in transition_events)
    assert any(event.rule_candidate is not None for event in transition_events)

    manager.observe_counterfactual(
        state=before,
        action="wait",
        predicted_reward=0.35,
        predicted_usefulness=0.2,
        predicted_uncertainty=0.1,
        predicted_delta_norm=0.0,
        score_gap=0.6,
    )
    counterfactual_events = manager.consume_recent_events()

    assert any(event.event_type == "counterfactual" for event in counterfactual_events)


def test_theory_manager_tracks_negative_reward_as_setback_rule() -> None:
    before = _interaction_state(step_index=0, right_removed=False)
    after = _interaction_state(step_index=1, right_removed=False)
    manager = EpisodeTheoryManager()
    manager.augment_runtime_thought(
        before,
        _interaction_thought(before),
        rule_inducer=EpisodeRuleInducer(),
    )

    manager.observe_transition(
        Transition(
            state=before,
            action="interact_right",
            reward=-0.08,
            next_state=after,
            terminated=False,
        )
    )
    transition_events = manager.consume_recent_events()

    assert any(event.event_type == "support" and event.theory.effect_kind == "setback" for event in transition_events)
    assert any(
        event.rule_candidate is not None and any(effect == "effect=setback" for effect in event.rule_candidate.effects)
        for event in transition_events
    )


def test_theory_manager_builds_fallback_rule_candidate_for_unranked_event() -> None:
    before = _interaction_state(step_index=0, right_removed=False)
    no_effect_after = _interaction_state(step_index=1, right_removed=False)
    manager = EpisodeTheoryManager()
    manager.augment_runtime_thought(
        before,
        _interaction_thought(before),
        rule_inducer=EpisodeRuleInducer(),
    )
    manager.rule_candidates_by_key = {}

    manager.observe_transition(
        Transition(
            state=before,
            action="interact_right",
            reward=0.0,
            next_state=no_effect_after,
            terminated=False,
        )
    )
    transition_events = manager.consume_recent_events()

    assert transition_events
    assert all(event.rule_candidate is not None for event in transition_events)


def test_hybrid_agent_routes_theory_claims_and_writes_theory_events_to_memory() -> None:
    agent = build_agent("hybrid", device=torch.device("cpu"))
    agent.config = replace(agent.config, use_runtime_controller=False)
    agent.runtime_rule_controller = None
    captured: dict[str, object] = {}

    def fake_build_runtime_thought(**kwargs):
        state = kwargs["state"]
        current = state.transition_vector()
        projected = current.copy()
        projected[2] = min(projected[2] + 0.25, 1.0)
        return RuntimeThought(
            actions=(
                ActionThought(
                    action="left",
                    value=0.0,
                    uncertainty=0.2,
                    policy=0.0,
                    predicted_reward=0.0,
                    usefulness=0.0,
                    next_state_proxy=SimpleNamespace(projected_transition=tuple(current.tolist())),
                ),
                ActionThought(
                    action="right",
                    value=0.55,
                    uncertainty=0.15,
                    policy=0.2,
                    predicted_reward=0.45,
                    usefulness=0.3,
                    next_state_proxy=SimpleNamespace(projected_transition=tuple(projected.tolist())),
                ),
            )
        )

    def fake_choose_action(**kwargs):
        captured["thought"] = kwargs["thought"]
        captured["action_bias"] = dict(kwargs["action_bias"] or {})
        captured["diagnostic_action_scores"] = dict(kwargs["diagnostic_action_scores"] or {})
        return PlanOutput(action="right", scores={"total": 1.0}, language=LanguageTrace())

    def fake_world_step(latent, actions, state=None, hidden=None):
        batch = len(actions)
        width = len(state.transition_vector()) if state is not None else 25
        zeros = torch.zeros(batch, dtype=latent.dtype, device=latent.device)
        return SimpleNamespace(
            next_latent_mean=torch.zeros_like(latent),
            hidden=None,
            reward=zeros,
            usefulness=zeros,
            policy=zeros,
            uncertainty=zeros,
            delta=torch.zeros((batch, width), dtype=latent.dtype, device=latent.device),
        )

    agent.planner.build_runtime_thought = fake_build_runtime_thought
    agent.planner.choose_action = fake_choose_action
    agent.world_model.step = fake_world_step

    action = agent.act(_move_observation(step_index=0, agent_x=1))
    agent.update_after_step(
        next_observation=_move_observation(step_index=1, agent_x=1),
        reward=0.0,
        terminated=False,
        info={},
    )

    assert action == "right"
    thought = captured["thought"]
    assert isinstance(thought, RuntimeThought)
    assert any(claim.claim_type == "rule" for claim in thought.claims)
    assert float(captured["action_bias"]["right"]) > 0.0
    assert float(captured["diagnostic_action_scores"]["right"]) >= 0.0
    assert agent.latest_rule_candidates
    assert agent.latest_competitions
    assert any(
        entry.payload.get("theory_event", {}).get("event_type") == "contradiction"
        for entry in agent.episodic_memory.entries
    )
    assert any("rule_candidate" in entry.payload for entry in agent.episodic_memory.entries)
