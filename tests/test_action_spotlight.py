from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arcagi.evaluation.harness import build_agent
from arcagi.agents.spotlight_scientist_agent import SpotlightScientistAgent, SpotlightScientistConfig
from arcagi.scientist.planner import PlannerConfig
from arcagi.scientist.perception import compare_states, extract_state
from arcagi.scientist.spotlight import (
    ActionSpotlight,
    ActionEvidence,
    AttemptActionRecord,
    AttemptOutcome,
    CURRENT_FEATURE_SCHEMA_VERSION,
    LEGACY_FEATURE_SCHEMA_VERSION,
    SpotlightConfig,
)
from arcagi.scientist.types import GridFrame, action_delta, action_family, is_interact_action, is_move_action, is_reset_action, is_selector_action


def test_spotlight_scientist_agent_exposes_spotlight_diagnostics() -> None:
    agent = SpotlightScientistAgent()
    diagnostics = agent.diagnostics()
    assert "spotlight" in diagnostics
    assert diagnostics["spotlight"]["feature_schema_version"] == CURRENT_FEATURE_SCHEMA_VERSION


def test_harness_supports_scientist_and_spotlight_aliases() -> None:
    for name in ("scientist", "spotlight"):
        agent = build_agent(name)
        diagnostics = agent.diagnostics()
        assert "spotlight" in diagnostics
        assert diagnostics["spotlight"]["feature_schema_version"] == CURRENT_FEATURE_SCHEMA_VERSION


def test_spotlight_loads_missing_feature_schema_version_as_legacy() -> None:
    spotlight = ActionSpotlight()
    state = spotlight.state_dict()
    assert state["feature_schema_version"] == CURRENT_FEATURE_SCHEMA_VERSION
    del state["feature_schema_version"]

    restored = ActionSpotlight()
    restored.load_state_dict(state)

    diagnostics = restored.diagnostics()
    assert diagnostics["feature_schema_version"] == LEGACY_FEATURE_SCHEMA_VERSION


def test_legacy_spotlight_schema_omits_extended_cost_schema_components() -> None:
    spotlight = ActionSpotlight()
    state = spotlight.state_dict()
    del state["feature_schema_version"]
    spotlight.load_state_dict(state)
    grid_state = extract_state(GridFrame("task", "episode", 0, np.array([[0, 1], [0, 0]], dtype=np.int64), ("3", "4")))

    class Planner:
        @staticmethod
        def candidate_actions(_state, *, engine):
            return ("3", "4")

    class Engine:
        @staticmethod
        def score_action(_state, _action, **_kwargs):
            return SimpleNamespace(
                expected_reward=0.0,
                expected_change=0.0,
                information_gain=0.0,
                risk=0.0,
                rationale=(),
            )

    class WorldModel:
        @staticmethod
        def predict(_state, _action):
            return SimpleNamespace(reward_mean=0.0, change_mean=0.0, total_uncertainty=0.0)

    class Memory:
        @staticmethod
        def action_memory_bonus(_state, _action, _lang_tokens):
            return 0.0

        @staticmethod
        def action_option_profile(_state, _action, _lang_tokens):
            return {
                "schema_bonus": 9.0,
                "relative_cost": 5.0,
                "efficiency": 4.0,
                "support": 3.0,
                "contradiction": 0.0,
                "continuation_depth": 1.0,
            }

    class Language:
        @staticmethod
        def memory_tokens(_state, _engine):
            return ()

        @staticmethod
        def belief_sentences(_engine, *, limit=3):
            return ()

        @staticmethod
        def questions(_engine, *, limit=2):
            return ()

    decision = spotlight.choose_action(
        grid_state,
        planner=Planner(),
        engine=Engine(),
        world_model=WorldModel(),
        memory=Memory(),
        language=Language(),
    )

    assert "option_schema_bonus" not in decision.components
    assert "action_cost" not in decision.components


def test_stalled_spotlight_promotes_untried_binding_probe() -> None:
    spotlight = ActionSpotlight(SpotlightConfig(diagnostic_binding_stall_threshold=2))
    spotlight.steps_since_progress = 3
    grid_state = extract_state(
        GridFrame("task", "episode", 3, np.array([[0, 1], [0, 0]], dtype=np.int64), ("1", "click:0:1"))
    )

    class Planner:
        @staticmethod
        def candidate_actions(_state, *, engine):
            return ("1", "click:0:1")

    class Engine:
        @staticmethod
        def score_action(_state, _action, **_kwargs):
            return SimpleNamespace(
                expected_reward=0.0,
                expected_change=0.0,
                information_gain=0.0,
                risk=0.0,
                rationale=(),
            )

    class WorldModel:
        @staticmethod
        def predict(_state, _action):
            return SimpleNamespace(reward_mean=0.0, change_mean=0.0, total_uncertainty=0.0)

    class Memory:
        @staticmethod
        def action_memory_bonus(_state, _action, _lang_tokens):
            return 0.0

        @staticmethod
        def action_option_profile(_state, _action, _lang_tokens):
            return {}

    class Language:
        @staticmethod
        def memory_tokens(_state, _engine):
            return ()

        @staticmethod
        def belief_sentences(_engine, *, limit=3):
            return ()

        @staticmethod
        def questions(_engine, *, limit=2):
            return ()

    decision = spotlight.choose_action(
        grid_state,
        planner=Planner(),
        engine=Engine(),
        world_model=WorldModel(),
        memory=Memory(),
        language=Language(),
    )

    assert decision.action == "click:0:1"
    assert decision.components["diagnostic_binding_probe"] == 1.0
    assert "stalled without progress" in decision.chosen_reason


def test_spotlight_agent_loads_missing_spotlight_state_as_legacy_schema() -> None:
    agent = SpotlightScientistAgent()
    state = {
        "config": {},
        "world_model": agent.world_model.state_dict(),
    }

    agent.load_state_dict(state)

    diagnostics = agent.diagnostics()
    assert diagnostics["spotlight"]["feature_schema_version"] == LEGACY_FEATURE_SCHEMA_VERSION


def test_spotlight_agent_from_checkpoint_restores_saved_config(tmp_path) -> None:
    config = SpotlightScientistConfig(
        planner=PlannerConfig(max_candidates=17, reward_weight=3.1),
        spotlight=SpotlightConfig(max_candidates=13, override_margin=0.77),
    )
    agent = SpotlightScientistAgent(config=config)
    agent.engine.transition_count = 5
    agent.memory.recent_actions = ["5", "4"]
    agent.planner.family_effect_count["right"] = 3
    agent.spotlight.global_action_visits["4"] = 7
    agent.spotlight.state_action_visits[("exact-a", "4")] = 6
    agent.spotlight.abstract_action_visits[("abstract-a", "right")] = 5
    agent.spotlight.binding_success[("5", "4")] = 2
    agent.spotlight.probe_baseline_trials["right"] = 9
    agent.spotlight.probe_baseline_effect_sum["right"] = 1.25
    agent.spotlight.action_evidence[("exact-a", "4")] = ActionEvidence(
        attempts=7.0,
        successes=1.0,
        failures=4.0,
        no_effects=3.0,
        contradictions=1.0,
        recent_failures=2.5,
        last_success_step=3,
        last_attempt_step=9,
    )
    agent.spotlight.family_evidence[("abstract-a", "right")] = ActionEvidence(attempts=9.0, failures=6.0, no_effects=5.0)
    agent.spotlight.steps_since_progress = 8
    agent.spotlight.session_reset_count = 2
    agent.spotlight.steps_since_reset = 3
    agent.spotlight.last_attempt_improvement = 0.25
    agent.spotlight.attempt_improvements.append(0.25)
    agent.spotlight._previous_attempt_outcome["task/level_0"] = AttemptOutcome(
        level_key="task/level_0",
        score=0.4,
        reward=0.5,
        steps=10,
        success=False,
        terminal_failure=False,
    )
    agent.spotlight._current_attempt_actions.append(
        AttemptActionRecord(
            action="4",
            feature_vector=np.ones(agent.spotlight.adaptation.feature_dim, dtype=np.float32),
            step=4,
            progress=0.1,
            visible_effect=True,
        )
    )
    agent.spotlight._current_attempt_level_key = "task/level_0"
    agent.spotlight._current_attempt_reward = 0.1
    agent.spotlight._current_attempt_steps = 1
    agent.spotlight.micro_attempt_updates = 4
    agent.spotlight.last_micro_attempt_target = -0.5
    path = tmp_path / "spotlight_roundtrip.pkl"
    agent.save_checkpoint(path)

    restored = SpotlightScientistAgent.from_checkpoint(path)

    assert restored.config.planner.max_candidates == 17
    assert restored.config.planner.reward_weight == 3.1
    assert restored.config.spotlight.max_candidates == 13
    assert restored.config.spotlight.override_margin == 0.77
    assert restored.diagnostics()["spotlight"]["feature_schema_version"] == CURRENT_FEATURE_SCHEMA_VERSION
    assert restored.engine.transition_count == 5
    assert restored.memory.recent_actions == ["5", "4"]
    assert restored.planner.family_effect_count["right"] == 3
    assert restored.spotlight.global_action_visits["4"] == 7
    assert restored.spotlight.state_action_visits[("exact-a", "4")] == 6
    assert restored.spotlight.abstract_action_visits[("abstract-a", "right")] == 5
    assert restored.spotlight.binding_success[("5", "4")] == 2
    assert restored.spotlight.probe_baseline_trials["right"] == 9
    assert restored.spotlight.probe_baseline_effect_sum["right"] == 1.25
    restored_exact_evidence = restored.spotlight.action_evidence[("exact-a", "4")]
    assert restored_exact_evidence.attempts == 7.0
    assert restored_exact_evidence.successes == 1.0
    assert restored_exact_evidence.failures == 4.0
    assert restored_exact_evidence.no_effects == 3.0
    assert restored_exact_evidence.contradictions == 1.0
    assert restored_exact_evidence.last_success_step == 3
    assert restored.spotlight.family_evidence[("abstract-a", "right")].failures == 6.0
    assert restored.spotlight.steps_since_progress == 8
    assert restored.spotlight.session_reset_count == 2
    assert restored.spotlight.steps_since_reset == 3
    assert restored.spotlight.last_attempt_improvement == 0.25
    assert list(restored.spotlight.attempt_improvements) == [0.25]
    assert restored.spotlight._previous_attempt_outcome["task/level_0"].score == 0.4
    assert restored.spotlight._current_attempt_level_key == "task/level_0"
    assert restored.spotlight._current_attempt_steps == 1
    assert len(restored.spotlight._current_attempt_actions) == 1
    assert restored.spotlight.micro_attempt_updates == 4
    assert restored.spotlight.last_micro_attempt_target == -0.5


def test_spotlight_agent_normalizes_mapping_configs_before_runtime_use() -> None:
    config = {
        "memory_capacity": 2048,
        "max_hypotheses": 512,
        "planner": {"max_candidates": 11, "reward_weight": 2.9},
        "spotlight": {"max_candidates": 19, "override_margin": 0.33},
        "world_learning_rate": 0.08,
        "seed": 0,
        "keep_world_weights_between_episodes": True,
    }

    agent = SpotlightScientistAgent(config=config)  # type: ignore[arg-type]

    assert agent.config.planner.max_candidates == 11
    assert agent.planner.config.max_candidates == 11
    assert agent.config.spotlight.max_candidates == 19
    assert agent.spotlight.config.max_candidates == 19


def test_binding_action_creates_pending_probe_and_probe_clears_on_support() -> None:
    spotlight = ActionSpotlight()

    before_bind = extract_state(
        GridFrame("task", "episode", 0, np.array([[0, 1], [0, 0]], dtype=np.int64), ("action5", "right"))
    )
    after_bind = extract_state(
        GridFrame("task", "episode", 1, np.array([[0, 1], [0, 0]], dtype=np.int64), ("action5", "right"))
    )
    bind_record = compare_states(before_bind, after_bind, action="action5")
    spotlight.notify_transition(record=bind_record)

    diagnostics = spotlight.diagnostics()
    assert diagnostics["pending_binder_probe"] is not None
    assert diagnostics["pending_binder_probe"]["binder_action"] == "action5"

    after_probe = extract_state(
        GridFrame("task", "episode", 2, np.array([[0, 0], [0, 1]], dtype=np.int64), ("action5", "right"))
    )
    probe_record = compare_states(after_bind, after_probe, action="right", reward=1.0)
    spotlight.notify_transition(record=probe_record)

    diagnostics = spotlight.diagnostics()
    assert diagnostics["pending_binder_probe"] is None
    assert diagnostics["binding_success_total"] >= 1


def test_arc_numeric_actions_map_to_generic_semantics() -> None:
    assert action_family("0") == "reset"
    assert action_family("1") == "up"
    assert is_move_action("1")
    assert action_delta("4") == (0, 1)
    assert is_selector_action("5")
    assert is_interact_action("6")
    assert is_reset_action("0")


def test_spotlight_persists_family_binding_priors_across_episode_reset() -> None:
    spotlight = ActionSpotlight()
    before_bind = extract_state(
        GridFrame("task", "episode", 0, np.array([[0, 1], [0, 0]], dtype=np.int64), ("5", "1"))
    )
    after_bind = extract_state(
        GridFrame("task", "episode", 1, np.array([[0, 1], [0, 0]], dtype=np.int64), ("5", "1"))
    )
    bind_record = compare_states(before_bind, after_bind, action="5")
    spotlight.notify_transition(record=bind_record)

    after_probe = extract_state(
        GridFrame("task", "episode", 2, np.array([[1, 0], [0, 0]], dtype=np.int64), ("5", "1"))
    )
    probe_record = compare_states(after_bind, after_probe, action="1", reward=1.0)
    spotlight.notify_transition(record=probe_record)

    diagnostics = spotlight.diagnostics()
    assert diagnostics["binding_success_total"] >= 1
    assert diagnostics["prior_binding_success_total"] >= 1

    spotlight.reset_episode()
    diagnostics = spotlight.diagnostics()
    assert diagnostics["binding_success_total"] == 0
    assert diagnostics["prior_binding_success_total"] >= 1


def test_spotlight_reset_level_preserves_online_session_evidence() -> None:
    spotlight = ActionSpotlight()
    spotlight.state_action_visits[("state-a", "click:1:1")] = 2
    spotlight.abstract_action_visits[("abstract-a", "click")] = 3
    spotlight.global_action_visits["click:1:1"] = 4
    spotlight.no_effect_counts[("state-a", "click:1:1")] = 5
    spotlight.no_effect_family_counts[("state-a", "click")] = 6
    spotlight.contradiction_counts[("state-a", "click:1:1")] = 0.75
    spotlight.binding_success[("click", "right")] = 7
    spotlight.binding_failure[("click", "left")] = 8
    spotlight.probe_baseline_trials["right"] = 9
    spotlight.probe_baseline_effect_sum["right"] = 1.5
    spotlight.prior_binding_success[("click", "right")] = 10
    spotlight.prior_binding_failure[("click", "left")] = 11
    spotlight.pending_update = SimpleNamespace(action="click:1:1")
    spotlight.steps_since_progress = 12
    spotlight._current_attempt_actions.append(SimpleNamespace(action="click:1:1"))

    spotlight.reset_level()

    assert spotlight.state_action_visits[("state-a", "click:1:1")] == 2
    assert spotlight.abstract_action_visits[("abstract-a", "click")] == 3
    assert spotlight.global_action_visits["click:1:1"] == 4
    assert spotlight.no_effect_counts[("state-a", "click:1:1")] == 5
    assert spotlight.no_effect_family_counts[("state-a", "click")] == 6
    assert spotlight.contradiction_counts[("state-a", "click:1:1")] == 0.75
    assert spotlight.binding_success[("click", "right")] == 7
    assert spotlight.binding_failure[("click", "left")] == 8
    assert spotlight.probe_baseline_trials["right"] == 9
    assert spotlight.probe_baseline_effect_sum["right"] == 1.5
    assert spotlight.prior_binding_success[("click", "right")] == 10
    assert spotlight.prior_binding_failure[("click", "left")] == 11
    assert spotlight.pending_update is None
    assert spotlight.steps_since_progress == 0
    assert spotlight._current_attempt_actions == []


def test_spotlight_candidate_surface_includes_legal_actions_when_planner_drops_them() -> None:
    spotlight = ActionSpotlight()
    legal = ("1", "2", "click:0:0", "click:1:0", "click:0:1", "click:1:1")
    state = extract_state(GridFrame("task", "episode", 0, np.zeros((2, 2), dtype=np.int64), legal))

    class Planner:
        @staticmethod
        def candidate_actions(_state, *, engine, memory=None, language_tokens=()):
            return ("1",)

    candidates = spotlight._candidate_actions(
        state,
        planner=Planner(),
        engine=SimpleNamespace(),
        memory=None,
        lang_tokens=(),
    )

    assert set(legal).issubset(set(candidates))
    assert candidates[: len(legal)] == legal


def test_spotlight_max_candidates_never_truncates_legal_surface() -> None:
    spotlight = ActionSpotlight(SpotlightConfig(max_candidates=3))
    legal = tuple(f"click:{index}:0" for index in range(12))
    state = extract_state(GridFrame("task", "episode", 0, np.zeros((1, 12), dtype=np.int64), legal))

    class Planner:
        @staticmethod
        def candidate_actions(_state, *, engine, memory=None, language_tokens=()):
            return ("extra",)

    candidates = spotlight._candidate_actions(
        state,
        planner=Planner(),
        engine=SimpleNamespace(),
        memory=None,
        lang_tokens=(),
    )

    assert len(candidates) == len(legal)
    assert candidates == legal


def test_spotlight_evidence_score_suppresses_repeated_falsified_action() -> None:
    spotlight = ActionSpotlight()
    spotlight.executive.weights[:] = 0.0
    spotlight.executive.exploration_bonus = 0.0
    state = extract_state(
        GridFrame(
            "task",
            "episode",
            0,
            np.array([[0, 1], [0, 0]], dtype=np.int64),
            ("3", "4"),
        )
    )
    spotlight.no_effect_counts[(state.exact_fingerprint, "3")] = 8
    spotlight.no_effect_family_counts[(state.abstract_fingerprint, "left")] = 12
    spotlight.contradiction_counts[(state.exact_fingerprint, "3")] = 8.0
    spotlight.global_action_visits["3"] = 20

    class Planner:
        @staticmethod
        def candidate_actions(_state, *, engine, memory=None, language_tokens=()):
            return ("3", "4")

    class Engine:
        @staticmethod
        def score_action(_state, action, **_kwargs):
            if action == "3":
                return SimpleNamespace(
                    expected_reward=0.0,
                    expected_change=1.0,
                    information_gain=0.0,
                    risk=0.0,
                    rationale=("stale",),
                )
            return SimpleNamespace(
                expected_reward=0.0,
                expected_change=0.0,
                information_gain=0.15,
                risk=0.0,
                rationale=("alternative",),
            )

    class WorldModel:
        @staticmethod
        def predict(_state, _action):
            return SimpleNamespace(reward_mean=0.0, change_mean=0.0, total_uncertainty=0.0)

    class Memory:
        @staticmethod
        def action_memory_bonus(_state, _action, _lang_tokens):
            return 0.0

        @staticmethod
        def action_option_profile(_state, _action, _lang_tokens):
            return {}

    class Language:
        @staticmethod
        def memory_tokens(_state, _engine):
            return ()

        @staticmethod
        def belief_sentences(_engine, *, limit=3):
            return ()

        @staticmethod
        def questions(_engine, *, limit=2):
            return ()

    decision = spotlight.choose_action(
        state,
        planner=Planner(),
        engine=Engine(),
        world_model=WorldModel(),
        memory=Memory(),
        language=Language(),
    )

    assert decision.action == "4"
    assert decision.components["evidence_score"] > 0.0


def test_spotlight_reliability_recovers_after_success() -> None:
    spotlight = ActionSpotlight()
    state = extract_state(GridFrame("task", "episode", 0, np.array([[0, 1], [0, 0]], dtype=np.int64), ("3",)))
    for step in range(6):
        spotlight._observe_action_evidence(
            state,
            "3",
            step=step,
            no_effect=1.0,
            contradiction=1.0,
        )

    failed_reliability = spotlight._reliability(state, "3")
    failed_penalty = spotlight._falsification_penalty(state, "3")

    spotlight._observe_action_evidence(state, "3", step=7, success=3.0)

    assert spotlight._reliability(state, "3") > failed_reliability
    assert spotlight._falsification_penalty(state, "3") <= failed_penalty


def test_failed_action_remains_candidate_after_reliability_demotes_score() -> None:
    spotlight = ActionSpotlight()
    state = extract_state(GridFrame("task", "episode", 0, np.zeros((2, 2), dtype=np.int64), ("3", "4")))
    for step in range(10):
        spotlight._observe_action_evidence(state, "3", step=step, no_effect=1.0, contradiction=1.0)

    class Planner:
        @staticmethod
        def candidate_actions(_state, *, engine, memory=None, language_tokens=()):
            return ("3", "4")

    candidates = spotlight._candidate_actions(
        state,
        planner=Planner(),
        engine=SimpleNamespace(),
        memory=None,
        lang_tokens=(),
    )

    assert "3" in candidates
    assert "4" in candidates
    assert spotlight._falsification_penalty(state, "3") > spotlight._falsification_penalty(state, "4")


def test_micro_attempt_updates_inside_long_nonterminal_episode() -> None:
    spotlight = ActionSpotlight()
    spotlight.executive.weights[:] = 0.0
    spotlight.executive.exploration_bonus = 0.0

    class Planner:
        @staticmethod
        def candidate_actions(_state, *, engine, memory=None, language_tokens=()):
            return ("3",)

    class Engine:
        @staticmethod
        def score_action(_state, _action, **_kwargs):
            return SimpleNamespace(
                expected_reward=0.0,
                expected_change=0.0,
                information_gain=0.0,
                risk=0.0,
                rationale=(),
            )

    class WorldModel:
        @staticmethod
        def predict(_state, _action):
            return SimpleNamespace(reward_mean=0.0, change_mean=0.0, total_uncertainty=0.0)

    class Memory:
        @staticmethod
        def action_memory_bonus(_state, _action, _lang_tokens):
            return 0.0

        @staticmethod
        def action_option_profile(_state, _action, _lang_tokens):
            return {}

    class Language:
        @staticmethod
        def memory_tokens(_state, _engine):
            return ()

        @staticmethod
        def belief_sentences(_engine, *, limit=3):
            return ()

        @staticmethod
        def questions(_engine, *, limit=2):
            return ()

    for step in range(6):
        before = extract_state(
            GridFrame(
                "task",
                "episode",
                step,
                np.array([[0, 1], [0, 0]], dtype=np.int64),
                ("3",),
                extras={"session_level_index": 0},
            )
        )
        after = extract_state(
            GridFrame(
                "task",
                "episode",
                step + 1,
                np.array([[0, 1], [0, 0]], dtype=np.int64),
                ("3",),
                extras={"session_level_index": 0},
            )
        )
        decision = spotlight.choose_action(
            before,
            planner=Planner(),
            engine=Engine(),
            world_model=WorldModel(),
            memory=Memory(),
            language=Language(),
        )
        spotlight.notify_transition(record=compare_states(before, after, action=decision.action))

    diagnostics = spotlight.diagnostics()
    assert diagnostics["micro_attempt_updates"] >= 1
    assert diagnostics["adaptation_updates"] >= 1
    assert diagnostics["micro_attempt_steps"] == 0
    assert spotlight._current_attempt_steps == 6


def test_baseline_movement_after_binder_does_not_count_as_binding_success() -> None:
    spotlight = ActionSpotlight()

    baseline_before = extract_state(
        GridFrame("task", "episode", 0, np.array([[1, 0], [0, 0]], dtype=np.int64), ("4", "5"))
    )
    baseline_after = extract_state(
        GridFrame("task", "episode", 1, np.array([[0, 1], [0, 0]], dtype=np.int64), ("4", "5"))
    )
    baseline_move = compare_states(baseline_before, baseline_after, action="4")
    spotlight.notify_transition(record=baseline_move)

    bind_before = extract_state(
        GridFrame("task", "episode", 2, np.array([[1, 0], [0, 0]], dtype=np.int64), ("4", "5"))
    )
    bind_after = extract_state(
        GridFrame("task", "episode", 3, np.array([[1, 0], [0, 0]], dtype=np.int64), ("4", "5"))
    )
    bind_record = compare_states(bind_before, bind_after, action="5")
    spotlight.notify_transition(record=bind_record)

    probe_after = extract_state(
        GridFrame("task", "episode", 4, np.array([[0, 1], [0, 0]], dtype=np.int64), ("4", "5"))
    )
    probe_record = compare_states(bind_after, probe_after, action="4")
    spotlight.notify_transition(record=probe_record)

    diagnostics = spotlight.diagnostics()
    assert diagnostics["binding_success_total"] == 0
    assert diagnostics["binding_failure_total"] >= 1


def test_reset_guard_schema_makes_nonterminal_reset_more_conservative() -> None:
    spotlight = ActionSpotlight()
    spotlight.feature_schema_version = CURRENT_FEATURE_SCHEMA_VERSION
    spotlight.steps_since_progress = spotlight.config.reset_stall_threshold + 24
    spotlight.steps_since_reset = 0
    spotlight.session_reset_count = spotlight.config.reset_session_budget + 1
    spotlight.last_attempt_improvement = 0.2
    state = extract_state(
        GridFrame("task", "episode", 0, np.array([[0, 1], [0, 0]], dtype=np.int64), ("0", "1"))
    )

    assert spotlight._reset_bonus(state, "0") == 0.0
    assert spotlight._penalty(state, "0") > 6.0


def test_reset_ended_attempt_scores_worse_under_reset_guard_schema() -> None:
    spotlight = ActionSpotlight()
    spotlight.feature_schema_version = CURRENT_FEATURE_SCHEMA_VERSION
    spotlight.session_reset_count = spotlight.config.reset_session_budget + 2

    ordinary = spotlight._attempt_score(reward=0.0, steps=8, success=False, terminal_failure=False, ended_by_reset=False)
    reset_ended = spotlight._attempt_score(reward=0.0, steps=8, success=False, terminal_failure=False, ended_by_reset=True)

    assert reset_ended < ordinary


def test_spotlight_habit_policy_prefers_teacher_labeled_action_after_update() -> None:
    spotlight = ActionSpotlight()
    spotlight.executive.weights[:] = 0.0
    spotlight.executive.exploration_bonus = 0.0
    state = extract_state(GridFrame("task", "episode", 0, np.array([[0, 1], [0, 0]], dtype=np.int64), ("3", "4")))

    class Planner:
        @staticmethod
        def candidate_actions(_state, *, engine):
            return ("3", "4")

    class Engine:
        @staticmethod
        def score_action(_state, _action, **_kwargs):
            return SimpleNamespace(
                expected_reward=0.0,
                expected_change=0.0,
                information_gain=0.0,
                risk=0.0,
                rationale=(),
            )

    class WorldModel:
        @staticmethod
        def predict(_state, _action):
            return SimpleNamespace(reward_mean=0.0, change_mean=0.0, total_uncertainty=0.0)

    class Memory:
        @staticmethod
        def action_memory_bonus(_state, _action, _lang_tokens):
            return 0.0

    class Language:
        @staticmethod
        def memory_tokens(_state, _engine):
            return ()

        @staticmethod
        def belief_sentences(_engine, *, limit=3):
            return ()

        @staticmethod
        def questions(_engine, *, limit=2):
            return ()

    first = spotlight.choose_action(
        state,
        planner=Planner(),
        engine=Engine(),
        world_model=WorldModel(),
        memory=Memory(),
        language=Language(),
    )
    assert first.action == "3"

    spotlight.observe_teacher_action("4")
    assert spotlight.diagnostics()["last_teacher_action"] == "4"
    spotlight.reset_level()

    second = spotlight.choose_action(
        state,
        planner=Planner(),
        engine=Engine(),
        world_model=WorldModel(),
        memory=Memory(),
        language=Language(),
    )

    assert second.action == "4"
    diagnostics = spotlight.diagnostics()
    assert diagnostics["habit_updates"] > 0


def test_binding_teacher_label_is_deferred_until_probe_resolution() -> None:
    spotlight = ActionSpotlight()
    spotlight.executive.weights[:] = 0.0
    spotlight.executive.exploration_bonus = 0.0
    state = extract_state(GridFrame("task", "episode", 0, np.array([[0, 1], [0, 0]], dtype=np.int64), ("3", "5")))

    class Planner:
        @staticmethod
        def candidate_actions(_state, *, engine):
            return ("3", "5")

    class Engine:
        @staticmethod
        def score_action(_state, _action, **_kwargs):
            return SimpleNamespace(
                expected_reward=0.0,
                expected_change=0.0,
                information_gain=0.0,
                risk=0.0,
                rationale=(),
            )

    class WorldModel:
        @staticmethod
        def predict(_state, _action):
            return SimpleNamespace(reward_mean=0.0, change_mean=0.0, total_uncertainty=0.0)

    class Memory:
        @staticmethod
        def action_memory_bonus(_state, _action, _lang_tokens):
            return 0.0

    class Language:
        @staticmethod
        def memory_tokens(_state, _engine):
            return ()

        @staticmethod
        def belief_sentences(_engine, *, limit=3):
            return ()

        @staticmethod
        def questions(_engine, *, limit=2):
            return ()

    spotlight.choose_action(
        state,
        planner=Planner(),
        engine=Engine(),
        world_model=WorldModel(),
        memory=Memory(),
        language=Language(),
    )
    spotlight.observe_teacher_action("5")
    diagnostics = spotlight.diagnostics()
    assert diagnostics["habit_updates"] == 0
    assert diagnostics["last_teacher_action"] == "5 [deferred]"

    bind_before = extract_state(GridFrame("task", "episode", 0, np.array([[1, 0], [0, 0]], dtype=np.int64), ("3", "5")))
    bind_after = extract_state(GridFrame("task", "episode", 1, np.array([[1, 0], [0, 0]], dtype=np.int64), ("3", "5")))
    spotlight.notify_transition(record=compare_states(bind_before, bind_after, action="5"))

    baseline_before = extract_state(GridFrame("task", "episode", 2, np.array([[1, 0], [0, 0]], dtype=np.int64), ("3", "5")))
    baseline_after = extract_state(GridFrame("task", "episode", 3, np.array([[0, 1], [0, 0]], dtype=np.int64), ("3", "5")))
    spotlight.notify_transition(record=compare_states(baseline_before, baseline_after, action="3"))
    bind_before = extract_state(GridFrame("task", "episode", 4, np.array([[1, 0], [0, 0]], dtype=np.int64), ("3", "5")))
    bind_after = extract_state(GridFrame("task", "episode", 5, np.array([[1, 0], [0, 0]], dtype=np.int64), ("3", "5")))
    spotlight.notify_transition(record=compare_states(bind_before, bind_after, action="5"))
    probe_after = extract_state(GridFrame("task", "episode", 6, np.array([[0, 1], [0, 0]], dtype=np.int64), ("3", "5")))
    spotlight.notify_transition(record=compare_states(bind_after, probe_after, action="3"))

    diagnostics = spotlight.diagnostics()
    assert diagnostics["habit_updates"] > 0
    assert diagnostics["last_teacher_action"] == "5 [failed]"
    assert diagnostics["binding_failure_total"] >= 1


def test_spotlight_logs_validated_move37_style_override() -> None:
    spotlight = ActionSpotlight()
    spotlight.executive.weights[:] = 0.0
    spotlight.executive.exploration_bonus = 0.0
    state = extract_state(GridFrame("task", "episode", 0, np.array([[1, 0], [0, 0]], dtype=np.int64), ("3", "4")))

    class Planner:
        @staticmethod
        def candidate_actions(_state, *, engine):
            return ("3", "4")

    class Engine:
        @staticmethod
        def score_action(_state, _action, **_kwargs):
            return SimpleNamespace(
                expected_reward=0.0,
                expected_change=0.0,
                information_gain=0.0,
                risk=0.0,
                rationale=(),
            )

    class WorldModel:
        @staticmethod
        def predict(_state, _action):
            return SimpleNamespace(reward_mean=0.0, change_mean=0.0, total_uncertainty=0.0)

    class Memory:
        @staticmethod
        def action_memory_bonus(_state, _action, _lang_tokens):
            return 0.0

    class Language:
        @staticmethod
        def memory_tokens(_state, _engine):
            return ()

        @staticmethod
        def belief_sentences(_engine, *, limit=3):
            return ()

        @staticmethod
        def questions(_engine, *, limit=2):
            return ()

    for _ in range(3):
        spotlight.choose_action(
            state,
            planner=Planner(),
            engine=Engine(),
            world_model=WorldModel(),
            memory=Memory(),
            language=Language(),
        )
        spotlight.observe_teacher_action("3")

    real_encode = spotlight.executive.encode

    class Executive:
        gamma = 0.92
        updates = 0

        @staticmethod
        def predict(feature_map):
            feature_vector = real_encode(feature_map)
            score = 1.0 if feature_map.get("family::right", 0.0) > 0.0 else -0.25
            return feature_vector, SimpleNamespace(value_mean=score, value_uncertainty=0.0)

        @staticmethod
        def score(prediction):
            return float(prediction.value_mean)

        @staticmethod
        def update(_feature_vector, _target):
            return 0.0

    spotlight.executive = Executive()

    decision = spotlight.choose_action(
        state,
        planner=Planner(),
        engine=Engine(),
        world_model=WorldModel(),
        memory=Memory(),
        language=Language(),
    )

    assert decision.action == "4"
    assert decision.components["move37_candidate"] == 1.0

    after = extract_state(GridFrame("task", "episode", 1, np.array([[0, 1], [0, 0]], dtype=np.int64), ("3", "4")))
    record = compare_states(state, after, action="4", reward=1.0)
    spotlight.notify_transition(record=record)

    diagnostics = spotlight.diagnostics()
    assert diagnostics["move37_candidates"] >= 1
    assert diagnostics["move37_validated"] >= 1
    assert diagnostics["last_move37_event"]["validated"] is True


def test_spotlight_learns_from_attempt_improvement() -> None:
    spotlight = ActionSpotlight()
    spotlight.executive.weights[:] = 0.0
    spotlight.executive.exploration_bonus = 0.0

    class Planner:
        @staticmethod
        def candidate_actions(_state, *, engine):
            return ("3", "4")

    class Engine:
        @staticmethod
        def score_action(_state, _action, **_kwargs):
            return SimpleNamespace(
                expected_reward=0.0,
                expected_change=0.0,
                information_gain=0.0,
                risk=0.0,
                rationale=(),
            )

    class WorldModel:
        @staticmethod
        def predict(_state, _action):
            return SimpleNamespace(reward_mean=0.0, change_mean=0.0, total_uncertainty=0.0)

    class Memory:
        @staticmethod
        def action_memory_bonus(_state, _action, _lang_tokens):
            return 0.0

    class Language:
        @staticmethod
        def memory_tokens(_state, _engine):
            return ()

        @staticmethod
        def belief_sentences(_engine, *, limit=3):
            return ()

        @staticmethod
        def questions(_engine, *, limit=2):
            return ()

    first_before = extract_state(
        GridFrame("task", "episode", 0, np.array([[0, 1], [0, 0]], dtype=np.int64), ("3", "4"), extras={"session_level_index": 0})
    )
    first_after = extract_state(
        GridFrame("task", "episode", 1, np.array([[0, 1], [0, 0]], dtype=np.int64), ("3", "4"), extras={"session_level_index": 0})
    )
    first_decision = spotlight.choose_action(
        first_before,
        planner=Planner(),
        engine=Engine(),
        world_model=WorldModel(),
        memory=Memory(),
        language=Language(),
    )
    spotlight.notify_transition(record=compare_states(first_before, first_after, action=first_decision.action, reward=0.0, terminated=True))
    spotlight.finalize_attempt(first_after, success=False, terminal_failure=True)

    second_before = extract_state(
        GridFrame("task", "episode", 0, np.array([[0, 1], [0, 0]], dtype=np.int64), ("3", "4"), extras={"session_level_index": 0})
    )
    second_after = extract_state(
        GridFrame("task", "episode", 1, np.array([[1, 0], [0, 0]], dtype=np.int64), ("3", "4"), extras={"session_level_index": 0})
    )
    second_decision = spotlight.choose_action(
        second_before,
        planner=Planner(),
        engine=Engine(),
        world_model=WorldModel(),
        memory=Memory(),
        language=Language(),
    )
    spotlight.notify_transition(record=compare_states(second_before, second_after, action=second_decision.action, reward=1.0, terminated=True))
    spotlight.finalize_attempt(second_after, success=True, terminal_failure=False)

    diagnostics = spotlight.diagnostics()
    assert diagnostics["adaptation_updates"] > 0
    assert diagnostics["last_attempt_improvement"] > 0.0
    assert diagnostics["avg_attempt_improvement"] > 0.0


def test_spotlight_subtracts_hypothesis_risk_when_scoring_candidates() -> None:
    spotlight = ActionSpotlight()
    state = extract_state(GridFrame("task", "episode", 0, np.array([[0, 1], [0, 0]], dtype=np.int64), ("3", "4")))

    class Planner:
        @staticmethod
        def candidate_actions(_state, *, engine):
            return ("3", "4")

    class Engine:
        @staticmethod
        def score_action(_state, action):
            if action == "4":
                return SimpleNamespace(
                    expected_reward=0.8,
                    expected_change=0.9,
                    information_gain=0.7,
                    risk=5.0,
                    rationale=("risky",),
                )
            return SimpleNamespace(
                expected_reward=0.2,
                expected_change=0.2,
                information_gain=0.2,
                risk=0.0,
                rationale=("safe",),
            )

    class WorldModel:
        @staticmethod
        def predict(_state, _action):
            return SimpleNamespace(reward_mean=0.0, change_mean=0.0, total_uncertainty=0.0)

    class Memory:
        @staticmethod
        def action_memory_bonus(_state, _action, _lang_tokens):
            return 0.0

    class Language:
        @staticmethod
        def memory_tokens(_state, _engine):
            return ()

        @staticmethod
        def belief_sentences(_engine, *, limit=3):
            return ()

        @staticmethod
        def questions(_engine, *, limit=2):
            return ()

    decision = spotlight.choose_action(
        state,
        planner=Planner(),
        engine=Engine(),
        world_model=WorldModel(),
        memory=Memory(),
        language=Language(),
    )

    assert decision.action == "3"
    assert decision.components["risk"] == 0.0


def test_spotlight_uses_learned_executive_score_over_workspace_features() -> None:
    spotlight = ActionSpotlight()
    state = extract_state(GridFrame("task", "episode", 0, np.array([[0, 1], [0, 0]], dtype=np.int64), ("3", "4")))

    class Planner:
        @staticmethod
        def candidate_actions(_state, *, engine):
            return ("3", "4")

    class Engine:
        @staticmethod
        def score_action(_state, _action, **_kwargs):
            return SimpleNamespace(
                expected_reward=0.0,
                expected_change=0.0,
                information_gain=0.0,
                risk=0.0,
                rationale=(),
            )

    class WorldModel:
        @staticmethod
        def predict(_state, action):
            if action == "4":
                return SimpleNamespace(reward_mean=0.9, change_mean=0.2, total_uncertainty=0.0)
            return SimpleNamespace(reward_mean=0.1, change_mean=0.2, total_uncertainty=0.0)

    class Memory:
        @staticmethod
        def action_memory_bonus(_state, _action, _lang_tokens):
            return 0.0

    class Language:
        @staticmethod
        def memory_tokens(_state, _engine):
            return ()

        @staticmethod
        def belief_sentences(_engine, *, limit=3):
            return ()

        @staticmethod
        def questions(_engine, *, limit=2):
            return ()

    class Executive:
        updates = 0

        @staticmethod
        def predict(feature_map):
            score = float(feature_map.get("raw::expected_reward", 0.0))
            return np.array([score], dtype=np.float32), SimpleNamespace(value_mean=score, value_uncertainty=0.0)

        @staticmethod
        def score(prediction):
            return float(prediction.value_mean)

    spotlight.executive = Executive()

    decision = spotlight.choose_action(
        state,
        planner=Planner(),
        engine=Engine(),
        world_model=WorldModel(),
        memory=Memory(),
        language=Language(),
    )

    assert decision.action == "4"
    assert decision.components["expected_reward"] > 0.5
    assert decision.components["executive_value"] > 0.5
