from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arcagi.evaluation.harness import build_agent, run_episode as run_harness_episode
from arcagi.scientist import HiddenRuleGridEnv, ScientistAgent, SyntheticConfig, load_scientist_checkpoint, run_episode, save_scientist_checkpoint
from arcagi.scientist.hypotheses import HypothesisEngine
from arcagi.scientist.memory import EpisodicMemory
from arcagi.scientist.planner import ScientistPlanner
from arcagi.scientist.perception import compare_states, extract_state
from arcagi.scientist.train_arc import _result_key
from arcagi.scientist.types import GridFrame, combined_progress_signal
from arcagi.scientist.world_model import OnlineWorldModel


def test_perception_segments_non_background_objects() -> None:
    grid = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    state = extract_state(GridFrame("t", "e", 0, grid, ("up",)))
    colors = sorted(obj.color for obj in state.objects)
    assert colors == [1, 2]
    assert state.abstract_fingerprint
    assert len(state.relations) >= 1


def test_hypothesis_engine_induces_movement_rule() -> None:
    before = GridFrame("t", "e", 0, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), ("right",))
    after = GridFrame("t", "e", 1, np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]), ("right",))
    record = compare_states(extract_state(before), extract_state(after), action="right")
    engine = HypothesisEngine()
    engine.observe_transition(record)
    kinds = {hyp.kind for hyp in engine.hypotheses.values()}
    assert "action_moves_object" in kinds


def test_hypothesis_engine_contradicts_movement_rule_when_expected_object_stalls() -> None:
    before = extract_state(GridFrame("t", "e", 0, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), ("right",)))
    moved = extract_state(GridFrame("t", "e", 1, np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]), ("right",)))
    stalled_after = extract_state(GridFrame("t", "e", 2, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), ("right",)))

    engine = HypothesisEngine()
    engine.observe_transition(compare_states(before, moved, action="right"))

    movement_hypothesis = next(h for h in engine.hypotheses.values() if h.kind == "action_moves_object")
    contradiction_before = movement_hypothesis.evidence.contradiction

    engine.observe_transition(compare_states(before, stalled_after, action="right"))

    updated = engine.hypotheses[movement_hypothesis.hypothesis_id]
    assert updated.evidence.contradiction > contradiction_before


def test_hypothesis_engine_movement_scores_are_contextual_and_bounded() -> None:
    state = extract_state(GridFrame("t", "e", 0, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), ("right",)))
    engine = HypothesisEngine()
    engine.transition_count = 20

    for index in range(12):
        hyp = engine._get_or_create(
            kind="action_moves_object",
            action_fam="right",
            params={"object_signature": f"missing:{index}", "color": 9, "delta": (0, index + 1)},
            description="missing object movement",
            step=0,
            mdl_penalty=0.0,
        )
        hyp.observe(supported=True, strength=1.0, step=0)

    missing_score = engine.score_action(state, "right", contextual=True, bounded=True)
    assert missing_score.expected_change == 0.0

    present_signature = state.objects[0].signature
    for index in range(12):
        hyp = engine._get_or_create(
            kind="action_moves_object",
            action_fam="right",
            params={"object_signature": present_signature, "color": 1, "delta": (0, index + 1)},
            description="present object movement",
            step=0,
            mdl_penalty=0.0,
        )
        hyp.observe(supported=True, strength=1.0, step=0)

    present_score = engine.score_action(state, "right", contextual=True, bounded=True)
    assert 0.0 < present_score.expected_change <= 1.0
    assert present_score.posterior_mass <= 2.0


def test_hypothesis_engine_does_not_induce_world_dynamics_from_reset_action() -> None:
    before = extract_state(GridFrame("t", "e", 0, np.array([[0, 0], [1, 0]]), ("0", "right")))
    after = extract_state(GridFrame("t", "e", 1, np.array([[1, 0], [0, 0]]), ("0", "right")))
    record = compare_states(before, after, action="0")
    engine = HypothesisEngine()

    engine.observe_transition(record)

    assert engine.hypotheses == {}
    score = engine.score_action(before, "0")
    assert score.expected_change == 0.0
    assert score.information_gain == 0.0


def test_hypothesis_engine_induces_generic_effect_beliefs_for_disappearance_and_large_motion() -> None:
    engine = HypothesisEngine()

    before_disappear = extract_state(
        GridFrame("t", "e", 0, np.array([[0, 2, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int64), ("up",))
    )
    after_disappear = extract_state(
        GridFrame("t", "e", 1, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int64), ("up",))
    )
    engine.observe_transition(compare_states(before_disappear, after_disappear, action="up"))

    before_motion = extract_state(
        GridFrame("t", "e", 2, np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64), ("right",))
    )
    after_motion = extract_state(
        GridFrame("t", "e", 3, np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.int64), ("right",))
    )
    engine.observe_transition(compare_states(before_motion, after_motion, action="right"))

    kinds = {hyp.kind for hyp in engine.hypotheses.values()}
    assert "effect_contact_disappears_object" in kinds
    assert "effect_contact_large_displacement" in kinds

    effect_priors = engine.mechanic_color_priors()
    assert 1 in effect_priors or 2 in effect_priors


def test_episodic_memory_records_option_for_salient_effect_without_reward() -> None:
    memory = EpisodicMemory()
    before = extract_state(
        GridFrame("t", "e", 0, np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64), ("right",))
    )
    after = extract_state(
        GridFrame("t", "e", 1, np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.int64), ("right",))
    )
    record = compare_states(before, after, action="right", reward=0.0)

    memory.write_transition(record, surprise=0.5, language_tokens=("effect", "motion"))

    assert len(memory.options) == 1
    assert memory.options[0].effect_value > 0.0
    assert memory.schemas


def test_episodic_memory_can_skip_and_contradict_option_writes() -> None:
    memory = EpisodicMemory()
    before = extract_state(
        GridFrame("t", "e", 0, np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64), ("right",))
    )
    after = extract_state(
        GridFrame("t", "e", 1, np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.int64), ("right",))
    )
    record = compare_states(before, after, action="right", reward=0.0)

    memory.recent_actions = ["click:1:1"]
    memory.recent_salient_flags = [False]
    memory.write_transition(record, surprise=0.7, language_tokens=("effect:large_motion",), option_mode="skip")

    assert len(memory.options) == 0
    assert memory.schemas == {}

    memory.recent_actions = ["click:1:1"]
    memory.recent_salient_flags = [False]
    memory.write_transition(record, surprise=0.7, language_tokens=("effect:large_motion",), option_mode="write")

    assert len(memory.options) == 1
    option = memory.options[0]
    assert option.action_sequence == ("click:1:1", "right")
    assert option.contradiction == 0.0

    memory.recent_actions = ["click:1:1"]
    memory.recent_salient_flags = [False]
    memory.write_transition(record, surprise=0.7, language_tokens=("effect:large_motion",), option_mode="contradict")

    assert len(memory.options) == 1
    assert memory.options[0].contradiction > 0.0
    schema = next(iter(memory.schemas.values()))
    assert schema.contradiction > 0.0


def test_episodic_memory_option_profile_tracks_effect_tags_and_relative_cost() -> None:
    memory = EpisodicMemory()
    before = extract_state(
        GridFrame("t", "e", 0, np.array([[0, 2, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int64), ("up",))
    )
    after = extract_state(
        GridFrame("t", "e", 1, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]], dtype=np.int64), ("up",))
    )
    record = compare_states(before, after, action="up", reward=0.0)

    memory.write_transition(record, surprise=0.6, language_tokens=("effect:disappear", "effect:contact"))

    profile = memory.action_option_profile(before, "up", ("effect:disappear", "effect:contact"))
    assert profile["schema_bonus"] > 0.0
    assert profile["relative_cost"] >= 1.0
    assert memory.options[0].effect_tags
    assert "effect:disappear" in memory.options[0].effect_tags
    assert memory.retrieve_schemas(before, ("effect:disappear", "effect:contact"))


def test_episodic_memory_option_profile_supports_sequence_continuation() -> None:
    memory = EpisodicMemory()
    before = extract_state(
        GridFrame("t", "e", 0, np.array([[0, 0, 0, 0], [1, 0, 0, 0], [0, 0, 0, 0]], dtype=np.int64), ("right",))
    )
    after = extract_state(
        GridFrame("t", "e", 1, np.array([[0, 0, 0, 0], [0, 0, 0, 1], [0, 0, 0, 0]], dtype=np.int64), ("right",))
    )
    memory.recent_actions = ["up"]
    memory.recent_salient_flags = [False]
    record = compare_states(before, after, action="right", reward=0.0)

    memory.write_transition(record, surprise=0.7, language_tokens=("effect:large_motion",))
    memory.recent_actions = ["up"]
    memory.recent_salient_flags = [False]

    profile = memory.action_option_profile(before, "right", ("effect:large_motion",))
    assert profile["schema_bonus"] > 0.0
    assert profile["continuation_depth"] > 0.0
    assert profile["relative_cost"] < memory.options[0].relative_cost


def test_planner_candidate_actions_only_adds_option_entries_compatible_with_current_legal_actions() -> None:
    planner = ScientistPlanner()
    engine = HypothesisEngine()
    memory = EpisodicMemory()
    state = extract_state(GridFrame("t", "e", 0, np.array([[0, 1], [0, 0]], dtype=np.int64), ("3", "4")))

    compatible = memory.options.__class__.__args__[0] if False else None  # type: ignore[attr-defined]
    del compatible

    from arcagi.scientist.memory import OptionItem

    memory.options.append(
        OptionItem(
            action_sequence=("click:9:9", "3"),
            first_action="click:9:9",
            family_sequence=("click", "left"),
            state_vector=np.zeros(memory.feature_dim, dtype=np.float32),
            language_tokens=frozenset({"effect:visible_change"}),
            effect_tags=frozenset({"effect:visible_change"}),
            precondition_tokens=frozenset({"shape:2x2"}),
            relative_cost=2.0,
            effect_value=0.4,
            reward=0.2,
            uses=1,
            successes=1,
            support=1.0,
        )
    )
    memory.options.append(
        OptionItem(
            action_sequence=("4",),
            first_action="4",
            family_sequence=("right",),
            state_vector=np.zeros(memory.feature_dim, dtype=np.float32),
            language_tokens=frozenset({"effect:visible_change"}),
            effect_tags=frozenset({"effect:visible_change"}),
            precondition_tokens=frozenset({"shape:2x2"}),
            relative_cost=1.0,
            effect_value=0.2,
            reward=0.1,
            uses=1,
            successes=1,
            support=1.0,
        )
    )

    candidates = planner.candidate_actions(state, engine=engine, memory=memory, language_tokens=("effect:visible_change",))

    assert "4" in candidates
    assert "click:9:9" not in candidates


def test_planner_budget_pressure_penalizes_high_cost_families() -> None:
    planner = ScientistPlanner()
    high_pressure_state = extract_state(
        GridFrame(
            "t",
            "e",
            80,
            np.array([[0, 1], [0, 0]], dtype=np.int64),
            ("3", "4"),
            extras={"session_retry_index": 3},
        )
    )
    low_pressure_state = extract_state(
        GridFrame(
            "t",
            "e",
            2,
            np.array([[0, 1], [0, 0]], dtype=np.int64),
            ("3", "4"),
            extras={"session_retry_index": 0},
        )
    )

    planner.family_cost_sum["left"] = 4.0
    planner.family_cost_count["left"] = 2
    planner.family_effect_sum["left"] = 0.2
    planner.family_effect_count["left"] = 2
    planner.family_cost_sum["right"] = 2.0
    planner.family_cost_count["right"] = 2
    planner.family_effect_sum["right"] = 1.2
    planner.family_effect_count["right"] = 2

    high_left = planner._action_cost_estimate("3", option_profile={"relative_cost": 1.0})
    high_right = planner._action_cost_estimate("4", option_profile={"relative_cost": 1.0})
    assert high_left > high_right
    assert planner._budget_pressure(high_pressure_state) > planner._budget_pressure(low_pressure_state)
    assert planner._action_efficiency_prior("4", option_profile={"efficiency": 0.0}) > planner._action_efficiency_prior(
        "3", option_profile={"efficiency": 0.0}
    )


def test_planner_budget_pressure_uses_generic_numeric_channels() -> None:
    planner = ScientistPlanner()
    engine = HypothesisEngine()
    before = extract_state(
        GridFrame(
            "t",
            "e",
            4,
            np.array([[0, 1], [0, 0]], dtype=np.int64),
            ("3", "4"),
            extras={"inventory": {"energy": "5"}},
        )
    )
    after = extract_state(
        GridFrame(
            "t",
            "e",
            5,
            np.array([[0, 1], [0, 0]], dtype=np.int64),
            ("3", "4"),
            extras={"inventory": {"energy": "4"}},
        )
    )
    low = extract_state(
        GridFrame(
            "t",
            "e",
            6,
            np.array([[0, 1], [0, 0]], dtype=np.int64),
            ("3", "4"),
            extras={"inventory": {"energy": "1"}},
        )
    )

    planner.notify_transition(changed=False, record=compare_states(before, after, action="3"), engine=engine)

    assert planner._budget_pressure(low) > planner._budget_pressure(before)


def test_combined_progress_signal_does_not_double_count_score_delta() -> None:
    assert combined_progress_signal(0.1, 0.1) == 0.1
    assert combined_progress_signal(0.1, 0.0) == 0.1
    assert combined_progress_signal(0.1, 0.2) == 0.30000000000000004


def test_scientist_agent_runs_updates_and_solves_hidden_rule_smoke_test() -> None:
    env = HiddenRuleGridEnv(SyntheticConfig(size=7, requires_key=True, seed=3, max_steps=80))
    agent = ScientistAgent()
    result = run_episode(env, agent, max_steps=80, seed=3)
    assert result.steps > 0
    assert result.diagnostics["transitions_observed"] == result.steps
    assert result.diagnostics["hypothesis_count"] > 0
    assert result.diagnostics["memory_items"] > 0
    assert agent.trace
    assert result.total_reward >= 1.0
    assert result.terminated


def test_scientist_checkpoint_round_trip_and_harness_load(tmp_path) -> None:
    env = HiddenRuleGridEnv(SyntheticConfig(size=7, requires_key=True, seed=5, max_steps=80))
    agent = ScientistAgent()
    run_episode(env, agent, max_steps=80, seed=5)

    path = tmp_path / "scientist.pkl"
    save_scientist_checkpoint(agent, path)

    restored = load_scientist_checkpoint(path)
    np.testing.assert_allclose(agent.world_model.reward_w, restored.world_model.reward_w)
    np.testing.assert_allclose(agent.world_model.change_w, restored.world_model.change_w)
    np.testing.assert_allclose(agent.world_model.recurrent_input_w, restored.world_model.recurrent_input_w)
    np.testing.assert_allclose(agent.world_model.recurrent_w, restored.world_model.recurrent_w)
    np.testing.assert_allclose(agent.world_model.reward_recurrent_w, restored.world_model.reward_recurrent_w)
    np.testing.assert_allclose(agent.world_model.change_recurrent_w, restored.world_model.change_recurrent_w)
    np.testing.assert_allclose(agent.world_model.hidden, restored.world_model.hidden)
    assert agent.world_model.updates == restored.world_model.updates

    harness_agent = build_agent("scientist", checkpoint_path=str(path))
    np.testing.assert_allclose(agent.world_model.reward_w, harness_agent.world_model.reward_w)
    assert harness_agent.diagnostics()["world_model"]["model"] == "online_recurrent_bootstrap"


def test_scientist_world_model_uses_recurrent_context() -> None:
    before = extract_state(
        GridFrame("t", "e", 0, np.array([[0, 1], [0, 0]], dtype=np.int64), ("4",))
    )
    after = extract_state(
        GridFrame("t", "e", 1, np.array([[0, 0], [0, 1]], dtype=np.int64), ("4",))
    )
    record = compare_states(before, after, action="4", reward=0.25)
    agent = ScientistAgent()

    first = agent.world_model.predict(before, "4")
    loss = agent.world_model.update(record)
    second = agent.world_model.predict(before, "4")

    assert loss >= 0.0
    assert agent.world_model.diagnostics()["model_schema_version"] == 2
    assert agent.world_model.diagnostics()["hidden_norm"] > 0.0
    assert second.reward_mean != first.reward_mean or second.change_mean != first.change_mean


def test_legacy_scientist_world_model_load_preserves_linear_predictions() -> None:
    model = OnlineWorldModel(seed=11)
    legacy_state = {
        key: value
        for key, value in model.state_dict().items()
        if key
        in {
            "feature_dim",
            "ensemble_size",
            "learning_rate",
            "weight_decay",
            "updates",
            "reward_w",
            "change_w",
        }
    }

    restored = OnlineWorldModel(seed=99)
    restored.load_state_dict(legacy_state)

    assert np.allclose(restored.reward_recurrent_w, 0.0)
    assert np.allclose(restored.change_recurrent_w, 0.0)


def test_scientist_arc_result_key_prefers_success_then_return_then_steps() -> None:
    fail_fast = {"success_rate": 0.0, "avg_return": 0.0, "avg_steps": 50.0}
    fail_slow = {"success_rate": 0.0, "avg_return": 0.0, "avg_steps": 80.0}
    better_return = {"success_rate": 0.0, "avg_return": 0.2, "avg_steps": 80.0}
    success = {"success_rate": 1.0, "avg_return": 1.0, "avg_steps": 120.0}

    assert _result_key(fail_fast) > _result_key(fail_slow)
    assert _result_key(better_return) > _result_key(fail_fast)
    assert _result_key(success) > _result_key(better_return)


def test_scientist_runtime_run_episode_continues_after_terminal_when_reset_is_available() -> None:
    class ScriptedAgent:
        def __init__(self) -> None:
            self.actions = iter(("1", "0", "1"))
            self.reset_calls = 0
            self.latest_language: tuple[str, ...] = ()
            self.transitions: list[tuple[str, float, bool]] = []

        def reset_episode(self) -> None:
            self.reset_calls += 1

        def act(self, observation):
            return next(self.actions)

        def observe_result(self, *, action, before_observation, after_observation, reward=0.0, terminated=False, info=None):
            self.transitions.append((str(action), float(reward), bool(terminated)))

        def diagnostics(self):
            return {"transitions": len(self.transitions)}

    class ScriptedEnv:
        def __init__(self) -> None:
            self.phase = 0

        def reset(self, seed=None):
            self.phase = 0
            return self._obs("GameState.NOT_FINISHED", 0, ("1", "0"))

        def step(self, action):
            if self.phase == 0 and action == "1":
                self.phase = 1
                return SimpleNamespace(
                    observation=self._obs("GameState.GAME_OVER", 0, ("1", "0")),
                    reward=0.0,
                    terminated=True,
                    truncated=False,
                    info={},
                )
            if self.phase == 1 and action == "0":
                self.phase = 2
                return SimpleNamespace(
                    observation=self._obs("GameState.NOT_FINISHED", 0, ("1", "0")),
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    info={},
                )
            if self.phase == 2 and action == "1":
                self.phase = 3
                return SimpleNamespace(
                    observation=self._obs("GameState.WIN", 1, ("0",)),
                    reward=1.0,
                    terminated=True,
                    truncated=False,
                    info={},
                )
            raise AssertionError((self.phase, action))

        @staticmethod
        def _obs(game_state: str, levels_completed: int, actions: tuple[str, ...]):
            return SimpleNamespace(
                available_actions=actions,
                extras={
                    "game_state": game_state,
                    "levels_completed": levels_completed,
                    "action_roles": {"0": "reset_level", "1": "select_cycle"},
                },
            )

    result = run_episode(ScriptedEnv(), ScriptedAgent(), max_steps=4, seed=0)

    assert result.steps == 3
    assert result.won is True
    assert result.levels_completed == 1
    assert result.reset_steps == 1
    assert result.total_reward == 1.0


def test_harness_run_episode_continues_after_terminal_until_reset_and_win() -> None:
    class ScriptedAgent:
        def __init__(self) -> None:
            self.actions = iter(("1", "0", "1"))
            self.reset_calls = 0
            self.level_reset_calls = 0
            self.latest_language: tuple[str, ...] = ()

        def reset_episode(self) -> None:
            self.reset_calls += 1

        def reset_level(self) -> None:
            self.level_reset_calls += 1

        def act(self, observation):
            return next(self.actions)

        def update_after_step(self, *, next_observation, reward=0.0, terminated=False, info=None):
            return None

        def diagnostics(self):
            return {"ok": True}

    class ScriptedEnv:
        family_id = "arc/test"

        def __init__(self) -> None:
            self.phase = 0

        def reset(self, seed=None):
            self.phase = 0
            return self._obs("GameState.NOT_FINISHED", 0, ("1", "0"))

        def step(self, action):
            if self.phase == 0 and action == "1":
                self.phase = 1
                return SimpleNamespace(
                    observation=self._obs("GameState.GAME_OVER", 0, ("1", "0")),
                    reward=0.0,
                    terminated=True,
                    truncated=False,
                    info={},
                )
            if self.phase == 1 and action == "0":
                self.phase = 2
                return SimpleNamespace(
                    observation=self._obs("GameState.NOT_FINISHED", 0, ("1", "0")),
                    reward=0.0,
                    terminated=False,
                    truncated=False,
                    info={},
                )
            if self.phase == 2 and action == "1":
                self.phase = 3
                return SimpleNamespace(
                    observation=self._obs("GameState.WIN", 1, ("0",)),
                    reward=1.0,
                    terminated=True,
                    truncated=False,
                    info={},
                )
            raise AssertionError((self.phase, action))

        @staticmethod
        def _obs(game_state: str, levels_completed: int, actions: tuple[str, ...]):
            return SimpleNamespace(
                available_actions=actions,
                extras={
                    "game_state": game_state,
                    "levels_completed": levels_completed,
                    "action_roles": {"0": "reset_level", "1": "select_cycle"},
                },
            )

    agent = ScriptedAgent()
    result = run_harness_episode(agent, ScriptedEnv(), seed=0, max_steps=4)

    assert result["steps"] == 3
    assert result["won"] is True
    assert result["levels_completed"] == 1
    assert result["reset_steps"] == 1
    assert result["interaction_steps"] == 2
    assert result["success"] is True
    assert agent.reset_calls == 1
    assert agent.level_reset_calls == 1
