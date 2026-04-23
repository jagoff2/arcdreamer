from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from arcagi.evaluation.harness import build_agent, run_episode as run_harness_episode
from arcagi.scientist import HiddenRuleGridEnv, ScientistAgent, SyntheticConfig, load_scientist_checkpoint, run_episode, save_scientist_checkpoint
from arcagi.scientist.hypotheses import HypothesisEngine
from arcagi.scientist.perception import compare_states, extract_state
from arcagi.scientist.train_arc import _result_key
from arcagi.scientist.types import GridFrame, combined_progress_signal


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
    assert agent.world_model.updates == restored.world_model.updates

    harness_agent = build_agent("scientist", checkpoint_path=str(path))
    np.testing.assert_allclose(agent.world_model.reward_w, harness_agent.world_model.reward_w)


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
                extras={"game_state": game_state, "levels_completed": levels_completed},
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
            self.latest_language: tuple[str, ...] = ()

        def reset_episode(self) -> None:
            self.reset_calls += 1

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
                extras={"game_state": game_state, "levels_completed": levels_completed},
            )

    agent = ScriptedAgent()
    result = run_harness_episode(agent, ScriptedEnv(), seed=0, max_steps=4)

    assert result["steps"] == 3
    assert result["won"] is True
    assert result["levels_completed"] == 1
    assert result["reset_steps"] == 1
    assert result["success"] is True
    assert agent.reset_calls == 1
