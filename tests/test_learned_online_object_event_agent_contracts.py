from __future__ import annotations

import pickle
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

from arcagi.agents.learned_online_object_event_agent import LearnedOnlineObjectEventAgent
from arcagi.core.types import GridObservation, ObjectState, StructuredState, Transition
from arcagi.evaluation.harness import build_agent, run_episode
from arcagi.learned_online.event_tokens import OUT_NO_EFFECT_NONPROGRESS, OUT_VISIBLE_CHANGE, OUT_VISIBLE_ONLY_NONPROGRESS
from arcagi.learned_online.object_event_curriculum import (
    ActiveOnlineObjectEventConfig,
    OnlineObjectEventCurriculumConfig,
    apply_synthetic_object_event_action,
    build_active_online_object_event_curriculum,
    build_online_object_event_curriculum,
    level_to_grid_observation,
)
from arcagi.learned_online.object_event_model import ObjectEventModelOutput


def test_object_event_agent_scores_all_legal_actions_and_reports_contract_diagnostics() -> None:
    agent = build_agent("learned_online_object_event")
    observation = _observation(actions=tuple(f"click:{x}:{y}" for y in range(8) for x in range(8)) + ("0", "undo", "up"))

    action = agent.act(observation)
    diagnostics = agent.diagnostics()

    assert action in observation.available_actions
    assert agent.controller_kind == "learned_online_object_event_v1"
    assert diagnostics["controller_kind"] == "learned_online_object_event_v1"
    assert diagnostics["legal_action_count"] == len(observation.available_actions)
    assert diagnostics["scored_action_count"] == len(observation.available_actions)
    assert diagnostics["full_dense_surface_scored"] is True
    assert diagnostics["runtime_trace_cursor"] is False
    assert diagnostics["runtime_action_sequence_replay"] is False
    assert diagnostics["runtime_state_hash_to_action"] is False
    assert diagnostics["runtime_per_game_behavior"] is False
    assert diagnostics["runtime_graph_search_solver"] is False
    assert diagnostics["runtime_action_pattern_enumerator"] is False
    assert diagnostics["runtime_external_api_or_knowledge"] is False
    assert diagnostics["runtime_rank_logits_used"] is True
    assert diagnostics["runtime_greedy_rank_selection"] is True


def test_object_event_online_update_count_and_level_belief_boundary_semantics() -> None:
    agent = LearnedOnlineObjectEventAgent(seed=3, device="cpu")
    before = _state(levels_completed=0)
    after = _state(levels_completed=1, remove_blue=True)
    agent.last_state = before
    agent.last_action = before.affordances[0]
    initial_session = float(agent.diagnostics()["session_belief_norm"])

    agent.on_transition(
        Transition(
            state=before,
            action=before.affordances[0],
            reward=1.0,
            next_state=after,
            terminated=False,
            info={"level_boundary": True, "levels_completed_before": 0, "levels_completed_after": 1, "score_delta": 1.0},
        )
    )
    diagnostics = agent.diagnostics()

    assert diagnostics["online_update_count"] == 1
    assert diagnostics["level_epoch"] == 1
    assert diagnostics["level_step"] == 0
    assert diagnostics["session_belief_norm"] > initial_session
    assert diagnostics["level_belief_norm"] == 0.0


def test_object_event_agent_relation_memory_carries_across_level_transition() -> None:
    split = build_online_object_event_curriculum(
        OnlineObjectEventCurriculumConfig(
            seed=31,
            train_sessions=1,
            heldout_sessions=0,
            levels_per_session=3,
            max_objects=3,
            include_distractors=False,
        )
    )
    session = split.train[0]
    support = session.levels[0].example
    query = session.levels[1].example
    agent = LearnedOnlineObjectEventAgent(seed=4, device="cpu", temperature=0.1, epsilon_floor=0.0)

    before_scores = agent.score_actions_for_state(query.state, query.legal_actions)
    before_diag = agent.diagnostics()
    wrong_index = (
        int(support.metadata["blue_action_index"])
        if int(support.correct_action_index) == int(support.metadata["red_action_index"])
        else int(support.metadata["red_action_index"])
    )
    agent.on_transition(
        Transition(
            state=support.state,
            action=support.legal_actions[wrong_index],
            reward=0.0,
            next_state=support.state,
            terminated=False,
            info={"score_delta": 0.0},
        )
    )
    mid_diag = agent.diagnostics()
    after_scores = agent.score_actions_for_state(query.state, query.legal_actions)
    correct = query.legal_actions[query.correct_action_index]

    assert before_diag["legal_action_count"] == 68
    assert before_diag["scored_action_count"] == 68
    assert mid_diag["online_update_count"] == 1
    assert mid_diag["session_belief_norm"] > 0.0
    assert mid_diag["level_belief_norm"] > 0.0
    assert after_scores[correct].score != before_scores[correct].score

    session_norm = float(agent.diagnostics()["session_belief_norm"])
    agent.reset_level()
    final = agent.diagnostics()

    assert final["level_belief_norm"] == 0.0
    assert final["session_belief_norm"] == session_norm
    assert final["runtime_trace_cursor"] is False
    assert final["runtime_action_sequence_replay"] is False
    assert final["runtime_state_hash_to_action"] is False
    assert final["runtime_per_game_behavior"] is False


def test_object_event_agent_active_update_splits_session_and_level_belief() -> None:
    split = build_online_object_event_curriculum(
        OnlineObjectEventCurriculumConfig(
            seed=32,
            train_sessions=1,
            heldout_sessions=0,
            levels_per_session=2,
            max_objects=3,
            include_distractors=False,
        )
    )
    support = split.train[0].levels[0].example
    wrong_index = (
        int(support.metadata["blue_action_index"])
        if int(support.correct_action_index) == int(support.metadata["red_action_index"])
        else int(support.metadata["red_action_index"])
    )
    failed_action = support.legal_actions[wrong_index]
    agent = LearnedOnlineObjectEventAgent(seed=5, device="cpu", temperature=0.1, epsilon_floor=0.0)

    before_scores = agent.score_actions_for_state(support.state, support.legal_actions)
    agent.on_transition(
        Transition(
            state=support.state,
            action=failed_action,
            reward=0.0,
            next_state=support.state,
            terminated=False,
            info={"score_delta": 0.0},
        )
    )
    after_scores = agent.score_actions_for_state(support.state, support.legal_actions)
    after_diag = agent.diagnostics()
    session_norm = float(after_diag["session_belief_norm"])

    agent.reset_level()
    reset_scores = agent.score_actions_for_state(support.state, support.legal_actions)
    reset_diag = agent.diagnostics()

    assert after_diag["online_update_count"] == 1
    assert session_norm > 0.0
    assert float(after_diag["level_belief_norm"]) > 0.0
    assert after_scores[failed_action].score < before_scores[failed_action].score
    assert reset_diag["level_belief_norm"] == 0.0
    assert reset_diag["session_belief_norm"] == session_norm
    assert reset_scores[failed_action].score > after_scores[failed_action].score


def test_object_event_online_targets_ignore_internal_belief_inventory_changes() -> None:
    agent = LearnedOnlineObjectEventAgent(seed=12, device="cpu")
    before = _state(levels_completed=0)
    after = StructuredState(
        task_id=before.task_id,
        episode_id=before.episode_id,
        step_index=before.step_index + 1,
        grid_shape=before.grid_shape,
        grid_signature=before.grid_signature,
        objects=before.objects,
        relations=before.relations,
        affordances=before.affordances,
        action_roles=before.action_roles,
        inventory=tuple(before.inventory) + (("belief_tested_sites", "1"),),
        flags=tuple(before.flags) + (("belief_near_tested_site", "1"),),
    )

    agent.on_transition(
        Transition(
            state=before,
            action=before.affordances[0],
            reward=0.0,
            next_state=after,
            terminated=False,
            info={},
        )
    )
    observed = agent.diagnostics()["last_observed_outcome"]

    assert observed[OUT_VISIBLE_CHANGE] == 0.0
    assert observed[OUT_VISIBLE_ONLY_NONPROGRESS] == 0.0
    assert observed[OUT_NO_EFFECT_NONPROGRESS] == 1.0


def test_object_event_agent_runtime_uses_model_rank_logits_when_outcomes_are_flat() -> None:
    agent = LearnedOnlineObjectEventAgent(seed=6, device="cpu", temperature=0.1, epsilon_floor=0.0)
    state = _state(levels_completed=0)
    actions = state.affordances
    action_count = len(actions)

    def fake_forward(_state_arg, _actions_arg):
        output = ObjectEventModelOutput(
            outcome_logits=torch.zeros((1, action_count, 10), dtype=torch.float32),
            delta_pred=torch.zeros((1, action_count, 25), dtype=torch.float32),
            value_logits=torch.zeros((1, action_count), dtype=torch.float32),
            action_repr=torch.zeros((1, action_count, agent.config.d_model), dtype=torch.float32),
            encoded_state=torch.zeros((1, 1, agent.config.d_model), dtype=torch.float32),
            rank_logits=torch.as_tensor([[0.0, 9.0, -4.0]], dtype=torch.float32),
        )
        return output, SimpleNamespace(mask=np.ones((action_count,), dtype=bool))

    agent._forward_state_actions = fake_forward  # type: ignore[method-assign]
    scores = agent.score_actions_for_state(state, actions)
    diagnostics = agent.diagnostics()

    assert max(scores.values(), key=lambda decision: decision.score).action == actions[1]
    assert diagnostics["runtime_rank_logits_used"] is True
    assert diagnostics["runtime_rank_score_std"] > 0.0


def test_object_event_checkpoint_roundtrip_preserves_runtime_rank_policy(tmp_path: Path) -> None:
    agent = LearnedOnlineObjectEventAgent(seed=8, device="cpu")
    path = tmp_path / "rank_policy.pkl"

    agent.save_checkpoint(path)
    restored = LearnedOnlineObjectEventAgent.from_checkpoint(path, device="cpu")
    observation = _observation(actions=tuple(f"click:{x}:{y}" for y in range(8) for x in range(8)) + ("0", "undo", "up"))
    restored.act(observation)
    diagnostics = restored.diagnostics()

    assert restored.metadata["runtime_uses_policy_rank_logits"] is True
    assert restored.metadata["runtime_greedy_rank_selection"] is True
    assert diagnostics["runtime_rank_logits_used"] is True
    assert diagnostics["full_dense_surface_scored"] is True
    assert diagnostics["runtime_trace_cursor"] is False
    assert diagnostics["runtime_action_sequence_replay"] is False
    assert diagnostics["runtime_state_hash_to_action"] is False
    assert diagnostics["runtime_per_game_behavior"] is False
    assert diagnostics["runtime_graph_search_solver"] is False
    assert diagnostics["runtime_action_pattern_enumerator"] is False
    assert diagnostics["runtime_external_api_or_knowledge"] is False


def test_object_event_checkpoint_roundtrip_contains_anti_replay_metadata(tmp_path: Path) -> None:
    agent = LearnedOnlineObjectEventAgent(seed=7, device="cpu")
    path = tmp_path / "object_event.pkl"

    agent.save_checkpoint(path)
    restored = LearnedOnlineObjectEventAgent.from_checkpoint(path, device="cpu")
    checkpoint = pickle.loads(path.read_bytes())
    metadata = checkpoint["metadata"]

    assert restored.controller_kind == "learned_online_object_event_v1"
    assert metadata["controller_kind"] == "learned_online_object_event_v1"
    assert metadata["runtime_trace_cursor"] is False
    assert metadata["runtime_action_sequence_replay"] is False
    assert metadata["runtime_state_hash_to_action"] is False
    assert metadata["runtime_per_game_behavior"] is False
    assert metadata["runtime_graph_search_solver"] is False
    assert metadata["runtime_action_pattern_enumerator"] is False
    assert metadata["trace_bootstrap_runtime_replay"] is False
    assert metadata["stores_teacher_action_sequence"] is False
    assert metadata["stores_state_hash_to_action"] is False
    assert metadata["online_update_from_transition_error"] is True
    assert "session_belief" in metadata["online_update_params"]
    assert "level_belief" in metadata["online_update_params"]
    assert "level_belief_action_basis_slots" in metadata["online_update_params"]
    assert "level_belief_action_family_slots" in metadata["online_update_params"]
    _assert_no_forbidden_payload_keys(checkpoint)


def test_object_event_agent_has_no_forbidden_controller_attributes() -> None:
    agent = build_agent("object_event")

    assert agent.claim_eligible_arc_controller is True
    assert agent.arc_competence_validated is False
    assert agent.uses_trace_replay is False
    assert agent.uses_state_hash_action_lookup is False
    assert agent.uses_per_game_runtime_behavior is False
    assert agent.scores_full_legal_action_surface is True
    assert not hasattr(agent, "runtime_rule_controller")
    assert not hasattr(agent, "theory_manager")
    assert not hasattr(agent, "spotlight")
    assert not hasattr(agent, "planner")
    assert not hasattr(agent, "rule_inducer")


def test_object_event_agent_scores_arc_scale_parametric_surface_without_cap() -> None:
    level = _parametric_level()
    agent = LearnedOnlineObjectEventAgent(seed=21, device="cpu", temperature=0.1, epsilon_floor=0.0)

    action = agent.act(level_to_grid_observation(level))
    diagnostics = agent.diagnostics()

    assert action in level.example.legal_actions
    assert diagnostics["legal_action_count"] == 447
    assert diagnostics["scored_action_count"] == 447
    assert diagnostics["full_dense_surface_scored"] is True
    assert diagnostics["object_event_action_surface_capped"] is False
    assert diagnostics["runtime_graph_search_solver"] is False
    assert diagnostics["runtime_action_pattern_enumerator"] is False
    assert diagnostics["runtime_external_api_or_knowledge"] is False


def test_object_event_agent_parametric_no_effect_update_keeps_failed_action_scored() -> None:
    level = _parametric_level()
    example = level.example
    failed = next(
        index
        for index, value in enumerate(example.candidate_targets.value)
        if float(value) == 0.0 and example.legal_actions[index].startswith("click:")
    )
    failed_action = example.legal_actions[failed]
    agent = LearnedOnlineObjectEventAgent(seed=22, device="cpu", temperature=0.1, epsilon_floor=0.0)
    with torch.no_grad():
        agent.model.failed_action_memory_rank.rank_mlp[-1].weight.fill_(-0.25)
        agent.model.failed_action_memory_rank.rank_mlp[-1].bias.zero_()

    before = agent.score_actions_for_state(example.state, example.legal_actions)
    result = apply_synthetic_object_event_action(level, failed)
    agent.on_transition(
        Transition(
            state=example.state,
            action=failed_action,
            reward=0.0,
            next_state=example.state,
            terminated=False,
            info={"score_delta": 0.0},
        )
    )
    after = agent.score_actions_for_state(example.state, example.legal_actions)
    diagnostics = agent.diagnostics()

    assert diagnostics["online_update_count"] == 1
    assert result.no_effect is True
    assert failed_action in after
    assert np.isfinite(after[failed_action].score)
    assert diagnostics["legal_action_count"] == 447
    assert diagnostics["scored_action_count"] == 447
    assert float(diagnostics["coordinate_noeffect_memory_norm"]) > 0.0
    assert float(diagnostics["coordinate_noeffect_count"]) > 0.0
    assert float(diagnostics["axis_noeffect_memory_norm"]) > 0.0
    assert float(diagnostics["axis_noeffect_count"]) > 0.0
    assert float(diagnostics["action_family_belief_norm"]) > 0.0
    assert float(diagnostics["action_family_evidence_count"]) > 0.0
    assert float(diagnostics["action_family_noeffect_count"]) > 0.0
    assert float(diagnostics["action_basis_belief_norm"]) > 0.0
    assert float(diagnostics["action_basis_evidence_count"]) > 0.0
    assert float(diagnostics["action_basis_noeffect_count"]) > 0.0
    assert "rank_component_axis_noeffect_std" in diagnostics
    assert "rank_component_relation_std" in diagnostics
    assert "rank_component_axis_noeffect_raw_std" in diagnostics
    assert "rank_component_relation_raw_std" in diagnostics
    assert "rank_component_gate_relation" in diagnostics
    assert "rank_component_gate_axis_noeffect" in diagnostics
    assert "relation_object_prior_scale" in diagnostics
    assert "relation_positive_prior_scale" in diagnostics
    assert "relation_repeat_penalty_scale" in diagnostics
    assert "relation_contradiction_gate_mean" in diagnostics
    assert "top_score_same_x_fraction" in diagnostics
    assert diagnostics["runtime_learned_diagnostic_utility_used"] is True
    assert "runtime_learned_diagnostic_utility_std" in diagnostics
    assert "runtime_entropy_tiebreak_std" in diagnostics
    assert "runtime_diagnostic_mix" in diagnostics
    assert "runtime_diagnostic_mix_effective" in diagnostics
    assert "runtime_diagnostic_mix_evidence_gate" in diagnostics
    assert "runtime_rank_weight_effective" in diagnostics
    assert "learned_diagnostic_utility_std" in diagnostics
    assert "diagnostic_mix_model_value" in diagnostics
    assert "action_family_belief_uncertainty_mean" in diagnostics
    assert "action_family_belief_noeffect_mean" in diagnostics
    assert "action_basis_belief_uncertainty_mean" in diagnostics
    assert "action_basis_belief_noeffect_mean" in diagnostics
    assert "family_assignment_effective_count" in diagnostics
    assert "family_assignment_usage_max" in diagnostics
    assert "basis_assignment_effective_count" in diagnostics
    assert "basis_assignment_usage_max" in diagnostics
    assert np.isfinite(float(diagnostics["rank_component_gate_relation"]))
    assert np.isfinite(float(diagnostics["relation_object_prior_scale"]))
    assert np.isfinite(float(diagnostics["runtime_learned_diagnostic_utility_std"]))
    assert np.isfinite(float(diagnostics["action_family_belief_uncertainty_mean"]))
    assert np.isfinite(float(diagnostics["family_assignment_effective_count"]))
    assert np.isfinite(float(diagnostics["action_basis_belief_uncertainty_mean"]))
    assert np.isfinite(float(diagnostics["basis_assignment_effective_count"]))
    assert 0.0 <= float(diagnostics["runtime_diagnostic_mix"]) <= 1.0
    assert 0.0 <= float(diagnostics["runtime_diagnostic_mix_effective"]) <= float(diagnostics["runtime_diagnostic_mix_max"])
    assert float(diagnostics["runtime_rank_weight_effective"]) >= 1.0 - float(diagnostics["runtime_diagnostic_mix_max"])
    assert after[failed_action].score != before[failed_action].score
    for forbidden in (
        "tried_actions",
        "blocked_actions",
        "blacklist",
        "action_sequence",
        "trace_cursor",
        "state_hash_to_action",
        "frontier",
        "coverage_queue",
        "sweep_index",
        "visited_bins",
        "least_visited",
        "untried",
        "tried_families",
        "family_blacklist",
        "basis_blacklist",
        "visited_basis",
        "probe_counter",
    ):
        assert not hasattr(agent, forbidden)


def test_object_event_agent_diagnostic_mix_is_bounded_by_evidence_gate() -> None:
    level = _parametric_level()
    example = level.example
    failed = next(
        index
        for index, value in enumerate(example.candidate_targets.value)
        if float(value) == 0.0 and example.legal_actions[index].startswith("click:")
    )
    failed_action = example.legal_actions[failed]
    agent = LearnedOnlineObjectEventAgent(seed=33, device="cpu", temperature=0.1, epsilon_floor=0.0)

    agent.score_actions_for_state(example.state, example.legal_actions)
    before = agent.diagnostics()
    agent.on_transition(
        Transition(
            state=example.state,
            action=failed_action,
            reward=0.0,
            next_state=example.state,
            terminated=False,
            info={"score_delta": 0.0},
        )
    )
    agent.score_actions_for_state(example.state, example.legal_actions)
    after = agent.diagnostics()

    assert float(before["runtime_diagnostic_mix_effective"]) == 0.0
    assert float(before["runtime_rank_weight_effective"]) == 1.0
    assert 0.0 <= float(after["runtime_diagnostic_mix_effective"]) <= float(after["runtime_diagnostic_mix_max"])
    assert float(after["runtime_rank_weight_effective"]) >= 1.0 - float(after["runtime_diagnostic_mix_max"])
    for forbidden in ("least_visited", "untried", "tried_basis", "basis_blacklist", "coverage_queue", "frontier", "sweep_index"):
        assert not hasattr(agent, forbidden)


def test_agent_exposes_hygiene_diversity_diagnostics_without_controller_state() -> None:
    level = _parametric_level()
    agent = LearnedOnlineObjectEventAgent(seed=34, device="cpu", temperature=0.1, epsilon_floor=0.0)

    agent.score_actions_for_state(level.example.state, level.example.legal_actions)
    diagnostics = agent.diagnostics()

    assert diagnostics["runtime_hygiene_diversity_diagnostics_only"] is True
    assert diagnostics["runtime_diversity_controller_active"] is False
    assert "top_score_same_mapped_col_fraction" in diagnostics
    assert "runtime_diagnostic_mix_effective" in diagnostics
    assert "runtime_rank_weight_effective" in diagnostics
    for forbidden in (
        "least_visited",
        "untried",
        "tried_actions",
        "tried_basis",
        "basis_blacklist",
        "coverage_queue",
        "frontier",
        "sweep_index",
        "probe_counter",
        "action_pattern_enumerator",
    ):
        assert not hasattr(agent, forbidden)


def test_harness_reports_hygiene_diversity_metrics_without_controller_state() -> None:
    class ScriptedAgent:
        def __init__(self) -> None:
            self.actions = iter(("click:1:1", "click:1:2", "click:4:2"))

        def reset_episode(self) -> None:
            return None

        def act(self, observation: GridObservation) -> str:
            return next(self.actions)

        def diagnostics(self) -> dict[str, object]:
            return {
                "runtime_diagnostic_mix_effective": 0.1,
                "runtime_rank_weight_effective": 0.9,
                "top_score_same_mapped_col_fraction": 0.8,
            }

    class Env:
        def reset(self, seed: int = 0) -> GridObservation:
            return _observation(actions=("click:1:1", "click:1:2", "click:4:2"))

        def step(self, action: str) -> SimpleNamespace:
            return SimpleNamespace(
                observation=_observation(actions=("click:1:1", "click:1:2", "click:4:2")),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={},
            )

    result = run_episode(ScriptedAgent(), Env(), seed=0, max_steps=3)

    assert result["unique_action_count"] == 3
    assert result["unique_click_x_count"] == 2
    assert result["unique_mapped_col_count"] == 2
    assert result["max_same_click_x_streak"] == 2
    assert result["max_same_mapped_col_streak"] == 2
    assert result["post_noeffect_next_action_same_x_rate"] == 0.5
    assert result["post_noeffect_next_action_same_mapped_col_rate"] == 0.5
    assert result["top_score_same_mapped_col_fraction"] == 0.8
    assert result["runtime_diagnostic_mix_effective"] == 0.1
    assert result["runtime_rank_weight_effective"] == 0.9
    assert not hasattr(ScriptedAgent(), "coverage_queue")


def test_harness_alias_loads_object_event_checkpoint(tmp_path: Path) -> None:
    agent = LearnedOnlineObjectEventAgent(seed=11, device="cpu")
    path = tmp_path / "agent.pkl"
    agent.save_checkpoint(path)

    restored = build_agent("learned_online_object_event_v1", checkpoint_path=str(path), device="cpu")

    assert restored.controller_kind == "learned_online_object_event_v1"
    assert restored.diagnostics()["runtime_trace_cursor"] is False


def _assert_no_forbidden_payload_keys(value: object, *, path: str = "") -> None:
    forbidden = (
        "trace_cursor",
        "teacher_actions",
        "action_sequence",
        "replay_actions",
        "state_hash_to_action",
        "state_action_lookup",
        "per_game",
    )
    if isinstance(value, dict):
        for key, child in value.items():
            key_text = str(key)
            child_path = f"{path}.{key_text}" if path else key_text
            if any(fragment in key_text for fragment in forbidden):
                assert child is False, f"forbidden stored payload key {child_path}={child!r}"
            _assert_no_forbidden_payload_keys(child, path=child_path)
    elif isinstance(value, (list, tuple)):
        for index, child in enumerate(value):
            _assert_no_forbidden_payload_keys(child, path=f"{path}[{index}]")


def _observation(*, actions: tuple[str, ...]) -> GridObservation:
    grid = np.zeros((8, 8), dtype=np.int64)
    grid[1, 1] = 2
    grid[6, 6] = 5
    roles = {action: "click" for action in actions if action.startswith("click:")}
    roles.update({"0": "reset_level", "undo": "undo", "up": "move_up"})
    return GridObservation(
        task_id="object_event_contract",
        episode_id="0",
        step_index=0,
        grid=grid,
        available_actions=actions,
        extras={"action_roles": roles},
    )


def _state(*, levels_completed: int, remove_blue: bool = False) -> StructuredState:
    grid = np.zeros((4, 4), dtype=np.int64)
    red = _object("red", 2, (1, 1))
    blue = _object("blue", 5, (2, 2))
    objects = (red,) if remove_blue else (red, blue)
    for obj in objects:
        for row, col in obj.cells:
            grid[row, col] = obj.color
    actions = ("click:1:1", "click:2:2", "0")
    return StructuredState(
        task_id="object_event_contract",
        episode_id="0",
        step_index=levels_completed,
        grid_shape=grid.shape,
        grid_signature=tuple(int(value) for value in grid.reshape(-1)),
        objects=objects,
        relations=(),
        affordances=actions,
        action_roles=(("0", "reset_level"), ("click:1:1", "click"), ("click:2:2", "click")),
        inventory=(("interface_levels_completed", str(levels_completed)),),
        flags=(),
    )


def _object(object_id: str, color: int, cell: tuple[int, int]) -> ObjectState:
    row, col = cell
    return ObjectState(
        object_id=object_id,
        color=color,
        cells=(cell,),
        bbox=(row, col, row, col),
        centroid=(float(row), float(col)),
        area=1,
        tags=(),
    )


def _parametric_level():
    split = build_active_online_object_event_curriculum(
        ActiveOnlineObjectEventConfig(
            seed=333,
            train_sessions=1,
            heldout_sessions=0,
            levels_per_session=3,
            max_distractors=2,
            curriculum="latent_rule_variable_palette",
            palette_size=8,
            require_role_balanced_colors=True,
            action_surface="arc_scale_parametric",
            action_surface_size=447,
            coordinate_grid_size=64,
            empty_click_fraction=0.80,
            positive_region_radius=1,
        )
    )
    return split.train[0].levels[0]
