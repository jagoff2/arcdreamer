from __future__ import annotations

import numpy as np

from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.types import ActionThought, GridObservation, RuntimeThought, Transition
from arcagi.perception.object_encoder import extract_structured_state
from arcagi.planning.rule_induction import object_signature
from arcagi.planning.runtime_rule_controller import (
    ControlHypothesis,
    EvidenceCounter,
    ModeHypothesis,
    ObjectiveFamilyHypothesis,
    ObjectiveHypothesis,
    RuntimeRuleController,
)


def _observation(
    grid: np.ndarray,
    *,
    step_index: int,
    actions: tuple[str, ...] = ("1", "2", "3", "4", "7"),
    action_roles: dict[str, str] | None = None,
    cell_tags: dict[tuple[int, int], tuple[str, ...]] | None = None,
) -> GridObservation:
    return GridObservation(
        task_id="arc/test",
        episode_id="episode-0",
        step_index=step_index,
        grid=grid,
        available_actions=actions,
        extras={
            "action_roles": action_roles
            or {
                "1": "move_up",
                "2": "move_down",
                "3": "move_left",
                "4": "move_right",
                "5": "select_cycle",
                "6": "click",
                "7": "undo",
            },
            "cell_tags": cell_tags or {},
        },
    )


def _signature_for_tag(state, tag: str):
    for obj in state.objects:
        if tag in obj.tags:
            return object_signature(obj)
    raise AssertionError(f"missing tag {tag}")


def test_runtime_rule_controller_uses_undo_for_move_diagnostics() -> None:
    before = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 2, 0, 3, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 4, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
        )
    )
    after_probe = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 2, 0, 3, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 4, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=1,
        )
    )

    controller = RuntimeRuleController()
    initial_plan = controller.propose(before)

    assert initial_plan is not None
    assert initial_plan.action in {"1", "2", "3", "4"}

    controller.observe_transition(
        Transition(
            state=before,
            action="2",
            reward=0.0,
            next_state=after_probe,
            terminated=False,
        )
    )

    undo_plan = controller.propose(after_probe)

    assert undo_plan is not None
    assert undo_plan.action == "7"


def test_runtime_rule_controller_moves_toward_stable_goal_after_online_probes() -> None:
    start = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 2, 2, 0, 0, 0, 0],
                    [0, 2, 2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 4, 4, 0],
                    [0, 0, 0, 0, 4, 4, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
        )
    )
    moved_down = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 2, 2, 0, 0, 0, 0],
                    [0, 2, 2, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 4, 4, 0],
                    [0, 0, 0, 0, 4, 4, 0],
                ],
                dtype=np.int64,
            ),
            step_index=1,
        )
    )
    moved_right = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 2, 2, 0, 0, 0],
                    [0, 0, 2, 2, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 4, 4, 0],
                    [0, 0, 0, 0, 4, 4, 0],
                ],
                dtype=np.int64,
            ),
            step_index=1,
        )
    )
    success = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2, 2, 0],
                    [0, 0, 0, 0, 2, 2, 0],
                    [0, 0, 0, 0, 4, 4, 0],
                    [0, 0, 0, 0, 4, 4, 0],
                ],
                dtype=np.int64,
            ),
            step_index=2,
        )
    )

    controller = RuntimeRuleController()
    controller.propose(start)
    controller.observe_transition(
        Transition(state=start, action="2", reward=0.0, next_state=moved_down, terminated=False)
    )
    controller.pending_undo = False
    controller.observe_transition(
        Transition(state=start, action="4", reward=0.0, next_state=moved_right, terminated=False)
    )
    controller.pending_undo = False
    controller.observe_transition(
        Transition(state=moved_right, action="2", reward=1.0, next_state=success, terminated=False)
    )

    plan = controller.propose(moved_right)

    assert plan is not None
    assert plan.action == "2"


def test_runtime_rule_controller_uses_selector_followup_from_runtime_thought() -> None:
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 3],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=("1", "2", "5", "6"),
        )
    )
    controller = RuntimeRuleController()
    controller.reference_state = state
    controller.reference_fingerprint = state.fingerprint()
    controller.action_visits[(controller.current_mode_key, "1")] = 1
    controller.action_visits[(controller.current_mode_key, "2")] = 1
    thought = RuntimeThought(
        belief_tokens=("uncertain",),
        question_tokens=("need", "test"),
        actions=(
            ActionThought(action="1", value=0.1, uncertainty=0.1),
            ActionThought(action="2", value=0.0, uncertainty=0.1),
            ActionThought(action="5", value=0.1, uncertainty=0.2, selector_followup=2.5),
            ActionThought(action="6", value=0.1, uncertainty=0.2, selector_followup=0.1),
        ),
    )

    plan = controller.propose(state, thought=thought)

    assert plan is not None
    assert plan.action == "5"


def test_runtime_rule_controller_tests_adjacent_interactable_after_motion_learning() -> None:
    actions = ("up", "down", "left", "right", "interact_up", "interact_down", "interact_left", "interact_right")
    start = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 9, 0, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=actions,
            cell_tags={
                (1, 1): ("agent",),
                (1, 3): ("interactable", "switch"),
                (4, 4): ("target",),
            },
        )
    )
    moved_down = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 3, 0, 0],
                    [0, 9, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=1,
            actions=actions,
            cell_tags={
                (2, 1): ("agent",),
                (1, 3): ("interactable", "switch"),
                (4, 4): ("target",),
            },
        )
    )
    moved_right = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 9, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=1,
            actions=actions,
            cell_tags={
                (1, 2): ("agent",),
                (1, 3): ("interactable", "switch"),
                (4, 4): ("target",),
            },
        )
    )

    controller = RuntimeRuleController()
    controller.propose(start)
    controller.observe_transition(Transition(state=start, action="down", reward=0.0, next_state=moved_down, terminated=False))
    controller.pending_undo = False
    controller.observe_transition(Transition(state=start, action="right", reward=0.0, next_state=moved_right, terminated=False))
    controller.pending_undo = False

    plan = controller.propose(moved_right)

    assert plan is not None
    assert plan.action == "interact_right"


def test_runtime_rule_controller_failed_interaction_writes_objective_contradiction() -> None:
    actions = ("up", "down", "left", "right", "interact_up", "interact_down", "interact_left", "interact_right")
    start = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 9, 0, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=actions,
            cell_tags={
                (1, 1): ("agent",),
                (1, 3): ("interactable", "switch"),
                (4, 4): ("target",),
            },
        )
    )
    moved_right = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 9, 3, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=1,
            actions=actions,
            cell_tags={
                (1, 2): ("agent",),
                (1, 3): ("interactable", "switch"),
                (4, 4): ("target",),
            },
        )
    )

    controller = RuntimeRuleController()
    controller.propose(start)
    controller.observe_transition(Transition(state=start, action="right", reward=0.0, next_state=moved_right, terminated=False))
    controller.pending_undo = False
    controller.observe_transition(
        Transition(state=moved_right, action="interact_right", reward=-0.1, next_state=moved_right, terminated=False)
    )

    mover_signature = _signature_for_tag(moved_right, "agent")
    target_signature = _signature_for_tag(moved_right, "interactable")
    objective = controller.objective_hypotheses[(controller.current_mode_key, mover_signature, target_signature)]
    proofs = controller.consume_recent_proofs(limit=8)

    assert objective.interaction_tests == 1
    assert objective.failed_interactions == 1
    assert objective.evidence.contradiction >= 2
    assert any(
        proof.hypothesis_type == "objective"
        and proof.proof_type == "contradiction"
        and proof.exception
        for proof in proofs
    )


def test_runtime_rule_controller_records_representation_repair_on_count_change() -> None:
    actions = ("wait",)
    before = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=actions,
        )
    )
    after = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 2, 0, 0],
                    [0, 0, 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=1,
            actions=actions,
        )
    )

    controller = RuntimeRuleController()
    controller.observe_transition(
        Transition(state=before, action="wait", reward=0.0, next_state=after, terminated=False)
    )

    proofs = controller.consume_recent_proofs(limit=8)

    assert any(
        proof.hypothesis_type == "representation" and proof.relation == "split"
        for proof in proofs
    )


def test_runtime_rule_controller_objective_posteriors_normalize_and_favor_supported_hypothesis() -> None:
    actions = ("up", "down", "left", "right")
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 9, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 3, 0],
                    [0, 0, 4, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=actions,
            cell_tags={
                (1, 1): ("agent",),
                (3, 3): ("target",),
                (4, 2): ("target",),
            },
        )
    )
    controller = RuntimeRuleController()
    mover_signature = _signature_for_tag(state, "agent")
    target_signatures = [object_signature(obj) for obj in state.objects if "target" in obj.tags]
    controller.control_hypotheses[(controller.current_mode_key, mover_signature)] = ControlHypothesis(
        evidence=EvidenceCounter(support=5, contradiction=1)
    )
    controller.objective_hypotheses[(controller.current_mode_key, mover_signature, target_signatures[0])] = ObjectiveHypothesis(
        evidence=EvidenceCounter(support=6, contradiction=1),
        progress_sum=2.0,
        successful_interactions=1,
    )
    controller.objective_hypotheses[(controller.current_mode_key, mover_signature, target_signatures[1])] = ObjectiveHypothesis(
        evidence=EvidenceCounter(support=1, contradiction=5),
        progress_sum=0.1,
        failed_interactions=2,
    )

    posteriors = controller._objective_posteriors(state, {mover_signature: 1.0})

    assert abs(sum(posteriors.values()) - 1.0) < 1e-6
    assert posteriors[(mover_signature, target_signatures[0])] > posteriors[(mover_signature, target_signatures[1])]


def test_runtime_rule_controller_materializes_posterior_selected_objective_for_exploit() -> None:
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0],
                    [0, 9, 0, 3, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=("up", "down", "left", "right"),
            cell_tags={
                (1, 1): ("agent",),
                (1, 3): ("target",),
            },
        )
    )
    controller = RuntimeRuleController()
    mover_signature = _signature_for_tag(state, "agent")
    target_signature = _signature_for_tag(state, "target")

    plan = controller._exploit_plan(
        state=state,
        move_actions=["up", "down", "left", "right"],
        mover_scores={mover_signature: 1.0},
        thought=None,
    )

    assert plan is not None
    assert plan.action == "right"
    assert (controller.current_mode_key, mover_signature, target_signature) in controller.objective_hypotheses


def test_runtime_rule_controller_mode_posteriors_normalize_and_favor_supported_mode() -> None:
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 2, 0, 0],
                    [0, 0, 0, 0],
                    [0, 0, 0, 3],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=("1", "2", "click:0:0", "click:3:0"),
            action_roles={"1": "move_up", "2": "move_down", "click:0:0": "click", "click:3:0": "click"},
        )
    )
    controller = RuntimeRuleController()
    context = build_action_schema_context(state.affordances, dict(state.action_roles))
    schema_a = build_action_schema("click:0:0", context)
    schema_b = build_action_schema("click:3:0", context)
    mode_a = f"{schema_a.family}:{schema_a.coarse_bin}"
    mode_b = f"{schema_b.family}:{schema_b.coarse_bin}"
    controller.mode_hypotheses[mode_a] = ModeHypothesis(
        evidence=EvidenceCounter(support=5, contradiction=1),
        entries=3,
        move_effect_steps=2,
    )
    controller.mode_hypotheses[mode_b] = ModeHypothesis(
        evidence=EvidenceCounter(support=1, contradiction=5),
        entries=3,
        stalled_steps=2,
    )
    posteriors = controller._mode_posteriors(
        selector_actions=["click:0:0", "click:3:0"],
        context=context,
        thought=RuntimeThought(
            actions=(
                ActionThought(action="click:0:0", selector_followup=1.5, uncertainty=0.2),
                ActionThought(action="click:3:0", selector_followup=0.1, uncertainty=0.2),
            )
        ),
    )

    assert abs(sum(posteriors.values()) - 1.0) < 1e-6
    assert posteriors[mode_a] > posteriors[mode_b]


def test_runtime_rule_controller_objective_family_posteriors_normalize_and_prefer_interact() -> None:
    actions = ("up", "down", "left", "right", "interact_right")
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 9, 3, 0],
                    [0, 0, 0, 0],
                    [0, 0, 2, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=actions,
            cell_tags={
                (1, 1): ("agent",),
                (1, 2): ("interactable", "switch"),
                (3, 2): ("target",),
            },
        )
    )
    controller = RuntimeRuleController()
    mover_signature = _signature_for_tag(state, "agent")
    target_signature = _signature_for_tag(state, "interactable")
    controller.control_hypotheses[(controller.current_mode_key, mover_signature)] = ControlHypothesis(
        evidence=EvidenceCounter(support=5, contradiction=1),
        moved_steps=3,
        motion_sum=3.0,
    )
    controller.objective_hypotheses[(controller.current_mode_key, mover_signature, target_signature)] = ObjectiveHypothesis(
        evidence=EvidenceCounter(support=4, contradiction=1),
        progress_sum=1.0,
        successful_interactions=1,
    )
    controller.objective_family_hypotheses[(controller.current_mode_key, mover_signature, target_signature, "approach")] = (
        ObjectiveFamilyHypothesis(
            family="approach",
            evidence=EvidenceCounter(support=1, contradiction=4),
            progress_sum=0.1,
        )
    )
    controller.objective_family_hypotheses[(controller.current_mode_key, mover_signature, target_signature, "interact")] = (
        ObjectiveFamilyHypothesis(
            family="interact",
            evidence=EvidenceCounter(support=6, contradiction=1),
            interaction_hits=2,
            state_delta_sum=1.0,
        )
    )

    posteriors = controller._objective_family_posteriors(state, {mover_signature: 1.0})

    assert abs(sum(posteriors.values()) - 1.0) < 1e-6
    assert posteriors[(mover_signature, target_signature, "interact")] > posteriors[
        (mover_signature, target_signature, "approach")
    ]


def test_runtime_rule_controller_exploit_plan_induces_and_reuses_objective_option() -> None:
    actions = ("up", "down", "left", "right")
    start = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 9, 0, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=actions,
            cell_tags={
                (1, 1): ("agent",),
                (1, 4): ("target",),
            },
        )
    )
    after_one = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 9, 0, 2, 0],
                    [0, 0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=1,
            actions=actions,
            cell_tags={
                (1, 2): ("agent",),
                (1, 4): ("target",),
            },
        )
    )

    controller = RuntimeRuleController()
    mover_signature = _signature_for_tag(start, "agent")
    target_signature = _signature_for_tag(start, "target")
    controller.control_hypotheses[(controller.current_mode_key, mover_signature)] = ControlHypothesis(
        evidence=EvidenceCounter(support=5, contradiction=1),
        moved_steps=3,
        motion_sum=3.0,
    )
    controller.objective_hypotheses[(controller.current_mode_key, mover_signature, target_signature)] = ObjectiveHypothesis(
        evidence=EvidenceCounter(support=6, contradiction=1),
        progress_sum=2.0,
    )
    controller.objective_family_hypotheses[(controller.current_mode_key, mover_signature, target_signature, "approach")] = (
        ObjectiveFamilyHypothesis(
            family="approach",
            evidence=EvidenceCounter(support=6, contradiction=1),
            progress_sum=2.0,
        )
    )

    plan = controller.propose(start)

    assert plan is not None
    assert plan.action == "right"
    assert len(plan.search_path) >= 2
    assert any(option.option_type == "objective_chain" for option in controller.option_hypotheses.values())

    controller.observe_transition(Transition(state=start, action="right", reward=0.0, next_state=after_one, terminated=False))
    continued = controller.propose(after_one)

    assert continued is not None
    assert continued.action == "right"
    assert "option" in continued.scores


def test_runtime_rule_controller_induces_selector_move_option_from_followup_motion() -> None:
    actions = ("down", "click:0:0")
    start = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0],
                    [0, 9, 0],
                    [0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=actions,
            action_roles={"down": "move_down", "click:0:0": "click"},
            cell_tags={(1, 1): ("agent",)},
        )
    )
    after_selector = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0],
                    [0, 9, 0],
                    [0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=1,
            actions=actions,
            action_roles={"down": "move_down", "click:0:0": "click"},
            cell_tags={(1, 1): ("agent",)},
        )
    )
    after_move = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 9, 0],
                ],
                dtype=np.int64,
            ),
            step_index=2,
            actions=actions,
            action_roles={"down": "move_down", "click:0:0": "click"},
            cell_tags={(2, 1): ("agent",)},
        )
    )

    controller = RuntimeRuleController()
    controller.observe_transition(Transition(state=start, action="click:0:0", reward=0.0, next_state=after_selector, terminated=False))
    controller.observe_transition(Transition(state=after_selector, action="down", reward=0.0, next_state=after_move, terminated=False))

    assert any(option.option_type == "selector_move" for option in controller.option_hypotheses.values())


def test_runtime_rule_controller_first_contact_plan_prefers_specific_question_and_probe_chain() -> None:
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=("up", "right", "click:1:1"),
            action_roles={"up": "move_up", "right": "move_right", "click:1:1": "click"},
        )
    )

    controller = RuntimeRuleController()
    plan = controller.propose(
        state,
        thought=RuntimeThought(
            question_tokens=("question", "need", "focus", "state", "probe"),
            actions=(
                ActionThought(action="up", value=0.1, uncertainty=0.4),
                ActionThought(action="right", value=0.0, uncertainty=0.2),
                ActionThought(action="click:1:1", value=0.2, uncertainty=0.5, selector_followup=1.1),
            ),
        ),
    )

    assert plan is not None
    assert plan.action == "click:1:1"
    assert len(plan.search_path) == 2
    assert plan.search_path[0] == "click:1:1"
    assert plan.search_path[1] in {"up", "right"}
    assert plan.language.question_tokens == ("question", "need", "test", "focus", "control_binding", "state", "probe")
    assert plan.language.plan_tokens == ("plan", "click_then_move", "focus", "control_binding", "state", "probe")
    assert any(option.option_type == "mode_probe_chain" for option in controller.option_hypotheses.values())


def test_runtime_rule_controller_augment_runtime_thought_emits_interface_and_control_binding_claims() -> None:
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 9, 0, 3],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=("up", "down", "click:3:1"),
            action_roles={"up": "move_up", "down": "move_down", "click:3:1": "click"},
            cell_tags={(1, 1): ("agent",)},
        )
    )

    controller = RuntimeRuleController()
    mover_signature = _signature_for_tag(state, "agent")
    click_target = next(obj for obj in state.objects if (1, 3) in obj.cells)
    target_signature = object_signature(click_target)
    context = build_action_schema_context(state.affordances, dict(state.action_roles))
    mode_key = f"{build_action_schema('click:3:1', context).family}:{build_action_schema('click:3:1', context).coarse_bin}"
    controller.mode_hypotheses[mode_key] = ModeHypothesis(
        evidence=EvidenceCounter(support=5, contradiction=1),
        entries=3,
        action_support={"click:3:1": 3},
        target_support={target_signature: 2},
        hidden_only_steps=1,
    )
    controller.control_hypotheses[(controller.current_mode_key, mover_signature)] = ControlHypothesis(
        evidence=EvidenceCounter(support=5, contradiction=1),
        moved_steps=3,
        motion_sum=3.0,
    )

    thought = RuntimeThought(
        belief_tokens=("belief", "goal", "unknown"),
        question_tokens=("question", "need", "explore"),
        actions=(
            ActionThought(action="up", value=0.1, uncertainty=0.2),
            ActionThought(action="down", value=0.0, uncertainty=0.2),
            ActionThought(action="click:3:1", value=0.2, uncertainty=0.3, selector_followup=1.0),
        ),
    )

    augmented = controller.augment_runtime_thought(state, thought)

    assert any(claim.claim_type == "interface" and claim.relation == "targets" for claim in augmented.claims)
    assert any(claim.claim_type == "control_binding" and claim.relation == "controls" for claim in augmented.claims)
    assert augmented.question_tokens == ("question", "need", "test", "focus", "control_binding", "state", "probe")


def test_runtime_rule_controller_exploit_plan_materializes_bind_then_objective_option() -> None:
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [5, 0, 0, 0],
                    [0, 9, 0, 2],
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=("right", "click:0:0"),
            action_roles={"right": "move_right", "click:0:0": "click"},
            cell_tags={
                (1, 1): ("agent",),
                (1, 3): ("target",),
            },
        )
    )

    controller = RuntimeRuleController()
    mover_signature = _signature_for_tag(state, "agent")
    target_signature = _signature_for_tag(state, "target")
    context = build_action_schema_context(state.affordances, dict(state.action_roles))
    click_schema = build_action_schema("click:0:0", context)
    controller.current_mode_key = f"{click_schema.family}:{click_schema.coarse_bin}"
    controller.mode_hypotheses[controller.current_mode_key] = ModeHypothesis(
        evidence=EvidenceCounter(support=4, contradiction=1),
        entries=2,
        action_support={"click:0:0": 2},
    )
    controller.control_hypotheses[(controller.current_mode_key, mover_signature)] = ControlHypothesis(
        evidence=EvidenceCounter(support=5, contradiction=1),
        moved_steps=3,
        motion_sum=3.0,
    )
    controller.objective_hypotheses[(controller.current_mode_key, mover_signature, target_signature)] = ObjectiveHypothesis(
        evidence=EvidenceCounter(support=6, contradiction=1),
        progress_sum=2.0,
    )
    controller.objective_family_hypotheses[(controller.current_mode_key, mover_signature, target_signature, "approach")] = (
        ObjectiveFamilyHypothesis(
            family="approach",
            evidence=EvidenceCounter(support=6, contradiction=1),
            progress_sum=2.0,
        )
    )

    plan = controller._exploit_plan(
        state=state,
        move_actions=["right"],
        mover_scores={mover_signature: 1.0},
        thought=RuntimeThought(actions=(ActionThought(action="right", value=0.2, uncertainty=0.1),)),
    )

    assert plan is not None
    assert any(option.option_type == "bind_then_objective" for option in controller.option_hypotheses.values())


def test_runtime_rule_controller_keeps_interface_only_targets_out_of_reward_objectives_without_evidence() -> None:
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [5, 0, 0, 0, 0],
                    [0, 9, 0, 2, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                    [0, 0, 0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=("right", "interact_right", "click:0:0"),
            action_roles={"right": "move_right", "interact_right": "interact", "click:0:0": "click"},
            cell_tags={
                (1, 1): ("agent",),
                (1, 3): ("target",),
            },
        )
    )
    controller = RuntimeRuleController()
    mover_signature = _signature_for_tag(state, "agent")
    target_signature = _signature_for_tag(state, "target")
    interface_signature = next(
        object_signature(obj)
        for obj in state.objects
        if (0, 0) in obj.cells
    )
    controller.control_hypotheses[(controller.current_mode_key, mover_signature)] = ControlHypothesis(
        evidence=EvidenceCounter(support=5, contradiction=1),
        moved_steps=3,
        motion_sum=3.0,
    )

    candidates = controller._objective_candidates(state, {mover_signature: 1.0})

    candidate_signatures = {object_signature(target_object) for _, target_object, _ in candidates}
    assert target_signature in candidate_signatures
    assert interface_signature not in candidate_signatures
    assert controller._has_unresolved_interaction_targets(state, {mover_signature: 1.0}) is False


def test_runtime_rule_controller_allows_interface_target_into_reward_objectives_after_reward_evidence() -> None:
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [5, 0, 0],
                    [0, 9, 0],
                    [0, 0, 0],
                ],
                dtype=np.int64,
            ),
            step_index=0,
            actions=("right", "interact_right", "click:0:0"),
            action_roles={"right": "move_right", "interact_right": "interact", "click:0:0": "click"},
            cell_tags={(1, 1): ("agent",)},
        )
    )
    controller = RuntimeRuleController()
    mover_signature = _signature_for_tag(state, "agent")
    interface_signature = next(
        object_signature(obj)
        for obj in state.objects
        if (0, 0) in obj.cells
    )
    controller.control_hypotheses[(controller.current_mode_key, mover_signature)] = ControlHypothesis(
        evidence=EvidenceCounter(support=5, contradiction=1),
        moved_steps=3,
        motion_sum=3.0,
    )
    controller.objective_hypotheses[(controller.current_mode_key, mover_signature, interface_signature)] = ObjectiveHypothesis(
        evidence=EvidenceCounter(support=3, contradiction=0),
        successful_interactions=1,
        reward_hits=1,
        reward_sum=0.5,
    )

    candidates = controller._objective_candidates(state, {mover_signature: 1.0})

    candidate_signatures = {object_signature(target_object) for _, target_object, _ in candidates}
    assert interface_signature in candidate_signatures
