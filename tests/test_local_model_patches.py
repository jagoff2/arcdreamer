from __future__ import annotations

import numpy as np
import torch
from types import SimpleNamespace

from arcagi.agents.learned_agent import ActionSemanticsSelfModel, AgentSelfBeliefState, LocalModelPatch
from arcagi.core.types import ActionThought, GridObservation, RuntimeThought, Transition
from arcagi.evaluation.harness import build_agent
from arcagi.memory.graph import StateGraph
from arcagi.perception.object_encoder import extract_structured_state


def _observation() -> GridObservation:
    return GridObservation(
        task_id="patch/test",
        episode_id="episode-0",
        step_index=0,
        grid=np.array(
            [
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 3],
            ],
            dtype=np.int64,
        ),
        available_actions=("left", "right"),
        extras={
            "action_roles": {
                "left": "move_left",
                "right": "move_right",
            },
            "cell_tags": {
                (1, 1): ("agent",),
            },
        },
    )


def _stalled_observation() -> GridObservation:
    return GridObservation(
        task_id="patch/stall",
        episode_id="episode-0",
        step_index=0,
        grid=np.array(
            [
                [0, 0, 0],
                [0, 2, 4],
                [0, 0, 0],
            ],
            dtype=np.int64,
        ),
        available_actions=("0", "left", "right", "click:1:1"),
        extras={
            "action_roles": {
                "0": "reset_level",
                "left": "move_left",
                "right": "move_right",
                "click:1:1": "click",
            },
            "inventory": {"interface_game_state": "GameState.NOT_FINISHED"},
            "cell_tags": {
                (1, 1): ("agent",),
                (1, 2): ("clickable",),
            },
        },
    )


def _click_observation() -> GridObservation:
    return GridObservation(
        task_id="patch/clicks",
        episode_id="episode-0",
        step_index=0,
        grid=np.array(
            [
                [2, 0, 0, 0, 3],
                [0, 0, 0, 0, 0],
                [0, 0, 4, 0, 0],
                [0, 0, 0, 0, 0],
                [5, 0, 0, 0, 6],
            ],
            dtype=np.int64,
        ),
        available_actions=("click:0:0", "click:4:4"),
        extras={
            "action_roles": {
                "click:0:0": "click",
                "click:4:4": "click",
            },
        },
    )


def test_hybrid_agent_applies_local_model_patches_to_runtime_thought() -> None:
    agent = build_agent("hybrid", device=torch.device("cpu"))
    state = extract_structured_state(_observation())
    agent.local_action_patches["right"] = LocalModelPatch(
        value_shift=1.0,
        policy_shift=0.5,
        usefulness_shift=0.3,
        uncertainty_shift=-0.2,
    )
    thought = RuntimeThought(
        actions=(
            ActionThought(action="left", value=0.2, uncertainty=0.4, policy=0.0, usefulness=0.1),
            ActionThought(action="right", value=0.2, uncertainty=0.4, policy=0.0, usefulness=0.1),
        )
    )

    patched = agent._apply_local_model_patches(state, thought)
    left = patched.for_action("left")
    right = patched.for_action("right")

    assert left is not None
    assert right is not None
    assert right.value > left.value
    assert right.policy > left.policy
    assert right.usefulness > left.usefulness
    assert right.uncertainty < left.uncertainty
    assert any(claim.claim_type == "local_patch" and claim.subject == "right" for claim in patched.claims)


def test_hybrid_agent_propagates_online_bias_across_recent_action_trace() -> None:
    agent = build_agent("hybrid", device=torch.device("cpu"))
    state = extract_structured_state(_observation())
    agent.recent_actions = ["left", "right", "interact"]

    agent._update_online_action_bias(
        Transition(
            state=state,
            action="interact",
            reward=0.0,
            next_state=state,
            terminated=False,
            info={"event": "empty_interaction"},
        ),
        state_change=0.0,
    )

    assert agent.online_action_bias["interact"] < 0.0
    assert agent.online_action_bias["right"] < 0.0
    assert agent.online_action_bias["left"] < 0.0


def test_action_semantics_self_model_updates_from_single_transition() -> None:
    state = extract_structured_state(_observation())
    model = ActionSemanticsSelfModel()

    model.observe(
        Transition(
            state=state,
            action="right",
            reward=1.0,
            next_state=state,
            terminated=False,
            info={},
        ),
        state_change=0.0,
        progress_signal=1.0,
        predicted_reward=0.0,
        predicted_usefulness=0.0,
        predicted_uncertainty=0.1,
        delta_error=0.0,
        surprise=0.9,
    )
    scores = model.score_actions(state, RuntimeThought())

    assert scores["right"].action_bias > scores["left"].action_bias
    assert scores["right"].expected_progress > 0.0
    assert model.diagnostics()["last_update"]["prediction_error"] > 0.0


def test_action_semantics_self_model_generates_grounded_question_tokens() -> None:
    state = extract_structured_state(_observation())
    model = ActionSemanticsSelfModel()

    scores = model.score_actions(
        state,
        RuntimeThought(
            actions=(
                ActionThought(action="left", uncertainty=0.1),
                ActionThought(action="right", uncertainty=0.9),
            )
        ),
    )
    thought = model.augment_thought(state, RuntimeThought(), scores)

    assert ("question", "need", "test") == thought.question_tokens[:3]
    assert "action" in thought.question_tokens
    assert any(claim.claim_type == "self_model" and claim.relation == "uncertain" for claim in thought.claims)


def test_action_semantics_self_model_keeps_negative_visible_moves_out_of_exploit_bias() -> None:
    state = extract_structured_state(_observation())
    model = ActionSemanticsSelfModel()

    model.observe(
        Transition(
            state=state,
            action="right",
            reward=0.0,
            next_state=state,
            terminated=False,
            info={},
        ),
        state_change=0.05,
        progress_signal=-0.08,
        predicted_reward=0.0,
        predicted_usefulness=0.0,
        predicted_uncertainty=0.02,
        delta_error=0.4,
        surprise=1.0,
    )
    scores = model.score_actions(state, RuntimeThought())

    assert scores["right"].action_bias <= 0.0
    assert scores["right"].diagnostic_value > 0.0


def test_agent_self_belief_emits_first_person_uncertainty_and_survives_level_reset() -> None:
    state = extract_structured_state(_observation())
    belief = AgentSelfBeliefState()
    belief.observe(
        Transition(
            state=state,
            action="right",
            reward=0.0,
            next_state=state,
            terminated=False,
            info={"level_boundary": True, "level_delta": 1},
        ),
        progress_signal=-0.08,
        prediction_error=0.45,
        predicted_uncertainty=0.02,
        surprise=1.0,
        online_update=True,
    )

    belief_tokens, question_tokens, plan_tokens = belief.tokens()
    diagnostics = belief.diagnostics()

    assert belief_tokens[:3] == ("belief", "agent", "state")
    assert "uncertain" in belief_tokens
    assert ("question", "need", "test") == question_tokens[:3]
    assert plan_tokens == ("plan", "test", "action", "because", "uncertain")
    assert diagnostics["online_updates"] == 1
    assert diagnostics["level_boundaries"] == 1
    assert diagnostics["levels_completed"] == 1

    belief.reset_level()

    assert belief.diagnostics()["online_updates"] == 1
    assert belief.diagnostics()["level_boundaries"] == 1


def test_clean_language_agent_can_route_self_model_scores_into_planner() -> None:
    agent = build_agent("language", device=torch.device("cpu"))
    state = extract_structured_state(_observation())
    graph = StateGraph()
    graph.visit(state)
    agent.self_model.observe(
        Transition(
            state=state,
            action="right",
            reward=1.0,
            next_state=state,
            terminated=False,
            info={},
        ),
        state_change=0.0,
        progress_signal=1.0,
        predicted_reward=0.0,
        predicted_usefulness=0.0,
        predicted_uncertainty=0.1,
        delta_error=0.0,
        surprise=0.9,
    )
    agent.self_belief.observe(
        Transition(
            state=state,
            action="right",
            reward=1.0,
            next_state=state,
            terminated=False,
            info={"level_boundary": True, "level_delta": 1},
        ),
        progress_signal=1.0,
        prediction_error=0.1,
        predicted_uncertainty=0.05,
        surprise=0.9,
        online_update=True,
    )
    scores = agent.self_model.score_actions(state, RuntimeThought())
    thought = agent.self_model.augment_thought(state, RuntimeThought(), scores)

    plan = agent.planner.choose_action(
        state=state,
        latent=None,
        graph=graph,
        world_model=None,
        hidden=None,
        language_model=None,
        action_bias=agent.self_model.action_biases(scores),
        diagnostic_action_scores=agent.self_model.diagnostic_scores(scores),
        thought=thought,
    )

    assert plan.action == "right"
    assert plan.scores["online_bias"] > 0.0
    assert agent.diagnostics()["runtime_controller_active"] is False


def test_language_agent_reset_level_preserves_online_learning_carryover() -> None:
    agent = build_agent("language", device=torch.device("cpu"))
    state = extract_structured_state(_observation())
    target_signature = (2, 1, 1, 1, ("agent",))
    agent.online_action_bias["right"] = 1.25
    agent.online_context_bias[("move", target_signature)] = 0.75
    agent.local_action_patches["right"] = LocalModelPatch(value_shift=0.8, entries=1)
    agent.local_context_patches[("move", target_signature)] = LocalModelPatch(policy_shift=0.4, entries=1)
    agent.family_bias["move"] = 0.6
    agent.language_token_scores["control"] = 0.9
    agent.evidence_steps = 3
    agent.global_action_counts["right"] = 2
    agent.family_counts["move"] = 2
    agent.level_action_counts["right"] = 2
    agent.level_action_progress_sums["right"] = 0.5
    agent.level_action_no_objective_counts["right"] = 1
    agent.level_family_counts["move"] = 2
    agent.level_family_progress_sums["move"] = 0.5
    agent.level_family_no_objective_counts["move"] = 1
    agent.self_model.observe(
        Transition(
            state=state,
            action="right",
            reward=1.0,
            next_state=state,
            terminated=False,
            info={},
        ),
        state_change=0.0,
        progress_signal=1.0,
        predicted_reward=0.0,
        predicted_usefulness=0.0,
        predicted_uncertainty=0.2,
        delta_error=0.1,
        surprise=0.9,
    )
    agent.self_belief.observe(
        Transition(
            state=state,
            action="right",
            reward=1.0,
            next_state=state,
            terminated=False,
            info={"level_boundary": True, "level_delta": 1},
        ),
        progress_signal=1.0,
        prediction_error=0.1,
        predicted_uncertainty=0.05,
        surprise=0.9,
        online_update=True,
    )
    agent.last_state = state
    agent.last_action = "right"
    agent.hidden = torch.zeros(1, agent.world_model.hidden_dim)
    agent.objective_stall_steps = 99
    agent.level_stall_steps = 123
    agent.max_levels_completed_observed = 1

    agent.reset_level()

    assert agent.last_state is None
    assert agent.last_action is None
    assert agent.hidden is None
    assert agent.recent_actions == []
    assert agent.online_action_bias["right"] == 1.25
    assert agent.online_context_bias[("move", target_signature)] == 0.75
    assert agent.local_action_patches["right"].value_shift == 0.8
    assert agent.local_context_patches[("move", target_signature)].policy_shift == 0.4
    assert agent.family_bias["move"] == 0.6
    assert agent.language_token_scores["control"] == 0.9
    assert agent.evidence_steps == 3
    assert agent.global_action_counts["right"] == 2
    assert agent.family_counts["move"] == 2
    assert not agent.level_action_counts
    assert not agent.level_family_counts
    assert agent.self_model.diagnostics()["action_stats"]["right"]["trials"] == 1
    assert agent.self_belief.diagnostics()["online_updates"] == 1
    assert agent.self_belief.diagnostics()["levels_completed"] == 1
    assert agent.objective_stall_steps == 99
    assert agent.level_stall_steps == 123
    assert agent.max_levels_completed_observed == 1


def test_language_agent_reset_level_preserves_grounded_language_carryover() -> None:
    agent = build_agent("language", device=torch.device("cpu"))
    agent.stable_belief_tokens = ("belief", "control", "support")
    agent.stable_question_tokens = ("question", "need", "test")
    agent.stable_plan_tokens = ("plan", "commit", "action")

    agent.reset_level()

    assert agent.stable_belief_tokens == ("belief", "control", "support")
    assert agent.pending_belief_tokens == agent.stable_belief_tokens
    assert "control" in agent.latest_language


def test_objective_stall_bias_shifts_from_motion_to_diagnostics_without_resetting() -> None:
    agent = build_agent("language", device=torch.device("cpu"))
    state = extract_structured_state(_stalled_observation())
    agent.objective_stall_steps = 800
    agent.max_levels_completed_observed = 1
    agent.global_action_counts["left"] = 80
    agent.global_action_counts["right"] = 80

    biases = agent._objective_stall_action_biases(state)
    family_biases = agent._objective_stall_family_biases(state)
    diagnostic_scores = agent._objective_stall_diagnostic_scores(state)

    assert biases["0"] < -1.0
    assert biases["left"] < -1.0
    assert biases["right"] < -1.0
    assert min(family_biases.values()) < -1.0
    assert biases["click:1:1"] > 0.0
    assert diagnostic_scores["click:1:1"] > 0.0


def test_first_level_gets_longer_window_before_objective_stall_pressure() -> None:
    agent = build_agent("language", device=torch.device("cpu"))
    state = extract_structured_state(_stalled_observation())
    agent.objective_stall_steps = 180
    agent.level_stall_steps = 180
    agent.max_levels_completed_observed = 0

    assert agent._objective_stall_pressure() == 0.0
    assert agent._objective_stall_action_biases(state).get("left", 0.0) == 0.0
    family_biases = agent._objective_stall_family_biases(state)
    assert any(key.startswith("reset:") and value < 0.0 for key, value in family_biases.items())

    agent.objective_stall_steps = 300
    agent.level_stall_steps = 300

    assert agent._objective_stall_pressure() > 0.0
    assert agent._objective_stall_action_biases(state).get("left", 0.0) < 0.0


def test_online_contrastive_update_defaults_to_full_action_set() -> None:
    agent = build_agent("language", device=torch.device("cpu"))

    assert agent.config.online_contrastive_action_sample_limit == 0


def test_terminal_failure_forces_negative_local_policy_patch() -> None:
    agent = build_agent("language", device=torch.device("cpu"))
    state = extract_structured_state(_observation())
    agent.last_prediction = SimpleNamespace(
        reward=torch.tensor([0.0]),
        usefulness=torch.tensor([0.2]),
        policy=torch.tensor([0.0]),
        uncertainty=torch.tensor([0.05]),
        delta=torch.zeros(1, state.transition_vector().shape[0]),
    )

    agent._update_local_model_patches(
        Transition(
            state=state,
            action="right",
            reward=0.0,
            next_state=state,
            terminated=True,
            info={"game_state_after": "GameState.GAME_OVER"},
        ),
        state_change=0.0,
        progress_signal=-0.45,
        terminal_failure=True,
    )

    patch = agent.local_action_patches["right"]
    assert patch.policy_shift < 0.0
    assert patch.usefulness_shift < 0.0


def test_terminal_reset_continuation_gets_positive_local_policy_patch() -> None:
    agent = build_agent("language", device=torch.device("cpu"))
    state = extract_structured_state(_stalled_observation())
    agent.last_prediction = SimpleNamespace(
        reward=torch.tensor([0.0]),
        usefulness=torch.tensor([-0.1]),
        policy=torch.tensor([-0.8]),
        uncertainty=torch.tensor([0.05]),
        delta=torch.zeros(1, state.transition_vector().shape[0]),
    )

    agent._update_local_model_patches(
        Transition(
            state=state,
            action="0",
            reward=0.0,
            next_state=state,
            terminated=False,
            info={
                "reset_action": True,
                "game_state_before": "GameState.GAME_OVER",
                "game_state_after": "GameState.NOT_FINISHED",
            },
        ),
        state_change=0.0,
        progress_signal=0.08,
        terminal_continuation=True,
    )

    patch = agent.local_action_patches["0"]
    assert patch.policy_shift > 0.0
    assert patch.usefulness_shift > 0.0


def test_click_no_effect_counts_do_not_smear_across_all_parametric_clicks() -> None:
    agent = build_agent("language", device=torch.device("cpu"))
    state = extract_structured_state(_click_observation())

    agent.global_action_counts["click:0:0"] = 1
    agent.global_action_no_effect_counts["click:0:0"] = 1
    context = {action: role for action, role in state.action_roles}
    from arcagi.core.action_schema import build_action_schema, build_action_schema_context, no_effect_family_key

    schema_a = build_action_schema("click:0:0", build_action_schema_context(state.affordances, context))
    schema_b = build_action_schema("click:4:4", build_action_schema_context(state.affordances, context))
    agent.family_counts[no_effect_family_key(schema_a)] = 1
    agent.family_no_effect_counts[no_effect_family_key(schema_a)] = 1

    graph = StateGraph()
    graph.visit(state)
    plan = agent.planner.choose_action(
        state=state,
        latent=None,
        graph=graph,
        world_model=None,
        action_counts=agent.global_action_counts,
        action_no_effect_counts=agent.global_action_no_effect_counts,
        family_counts=agent.family_counts,
        family_no_effect_counts=agent.family_no_effect_counts,
        thought=RuntimeThought(),
    )

    assert no_effect_family_key(schema_a) != no_effect_family_key(schema_b)
    assert plan.action == "click:4:4"
