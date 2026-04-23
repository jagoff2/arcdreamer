from __future__ import annotations

from pathlib import Path

from arcagi.core.types import GridObservation, StepResult
from arcagi.envs.session import PersistentLevelSessionEnv
from arcagi.agents.scientist_agent import load_spotlight_scientist_checkpoint
from arcagi.scientist.train import ScientistTrainingConfig, train_scientist


class _FailThenWinLevel:
    def __init__(self) -> None:
        self.reset_count = 0

    def legal_actions(self):
        return ("1",)

    def reset(self, seed=None):
        self.reset_count += 1
        return GridObservation(
            task_id="scripted/level",
            episode_id="scripted/level/0",
            step_index=0,
            grid=[[0]],
            available_actions=("1",),
            extras={},
        )

    def step(self, action):
        assert action == "1"
        if self.reset_count == 1:
            return StepResult(
                observation=GridObservation(
                    task_id="scripted/level",
                    episode_id="scripted/level/0",
                    step_index=1,
                    grid=[[0]],
                    available_actions=("1",),
                    extras={},
                ),
                reward=0.0,
                terminated=True,
                truncated=False,
                info={"event": "failed"},
            )
        return StepResult(
            observation=GridObservation(
                task_id="scripted/level",
                episode_id="scripted/level/0",
                step_index=1,
                grid=[[1]],
                available_actions=("1",),
                extras={},
            ),
            reward=1.0,
            terminated=True,
            truncated=False,
            info={"event": "goal_reached"},
        )


class _ImmediateWinLevel:
    def legal_actions(self):
        return ("1",)

    def reset(self, seed=None):
        return GridObservation(
            task_id="scripted/level",
            episode_id="scripted/level/1",
            step_index=0,
            grid=[[0]],
            available_actions=("1",),
            extras={},
        )

    def step(self, action):
        assert action == "1"
        return StepResult(
            observation=GridObservation(
                task_id="scripted/level",
                episode_id="scripted/level/1",
                step_index=1,
                grid=[[2]],
                available_actions=("1",),
                extras={},
            ),
            reward=1.0,
            terminated=True,
            truncated=False,
            info={"event": "goal_reached"},
        )


def test_persistent_level_session_env_supports_retry_reset_and_level_advance() -> None:
    first_level = _FailThenWinLevel()
    second_level = _ImmediateWinLevel()
    env = PersistentLevelSessionEnv(
        level_builders=(
            lambda seed: first_level,
            lambda seed: second_level,
        ),
        task_id="synthetic/test_session",
        family_id="synthetic/test_family",
    )

    observation = env.reset(seed=0)
    assert observation.extras["levels_completed"] == 0
    assert "0" in observation.available_actions

    failed = env.step("1")
    assert failed.terminated is True
    assert failed.observation.extras["game_state"] == "GAME_OVER"
    assert "0" in failed.observation.available_actions

    retried = env.step("0")
    assert retried.terminated is False
    assert retried.observation.extras["session_retry_index"] == 1
    assert retried.observation.extras["levels_completed"] == 0

    advanced = env.step("1")
    assert advanced.terminated is False
    assert advanced.observation.extras["levels_completed"] == 1
    assert advanced.info["event"] == "level_advanced"

    won = env.step("1")
    assert won.terminated is True
    assert won.observation.extras["game_state"] == "WIN"
    assert won.observation.extras["levels_completed"] == 2


def test_train_scientist_runs_tiny_session_smoke(tmp_path: Path) -> None:
    summary = train_scientist(
        ScientistTrainingConfig(
            seed=3,
            stage1_sessions=1,
            stage2_sessions=1,
            eval_every=2,
            simple_max_steps=16,
            rich_level_max_steps=12,
            rich_session_max_steps=32,
            rich_levels_per_session=2,
            holdout_simple_sessions=1,
            holdout_rich_sessions=1,
            checkpoint_path=str(tmp_path / "best.pkl"),
            latest_checkpoint_path=str(tmp_path / "latest.pkl"),
        )
    )

    assert summary["sessions"] == 2
    assert "train_session_win_rate" in summary
    assert "train_avg_levels_completed" in summary
    assert "train_avg_attempt_improvement" in summary
    assert "final_holdout" in summary
    assert "simple_avg_attempt_improvement" in summary["final_holdout"]
    assert "rich_avg_attempt_improvement" in summary["final_holdout"]
    assert (tmp_path / "best.pkl").exists()
    assert (tmp_path / "latest.pkl").exists()


def test_train_scientist_uses_teacher_habit_path(tmp_path: Path, monkeypatch) -> None:
    teacher_calls = {"count": 0}

    from arcagi.scientist import train as scientist_train_module
    from arcagi.training.synthetic_oracle import teacher_action as real_teacher_action

    def counted_teacher_action(env, *, observation=None):
        teacher_calls["count"] += 1
        return real_teacher_action(env, observation=observation)

    monkeypatch.setattr(scientist_train_module, "synthetic_teacher_action", counted_teacher_action)

    train_scientist(
        ScientistTrainingConfig(
            seed=17,
            stage1_sessions=1,
            stage2_sessions=1,
            eval_every=2,
            simple_max_steps=12,
            rich_level_max_steps=10,
            rich_session_max_steps=24,
            rich_levels_per_session=2,
            holdout_simple_sessions=1,
            holdout_rich_sessions=1,
            checkpoint_path=str(tmp_path / "best.pkl"),
            latest_checkpoint_path=str(tmp_path / "latest.pkl"),
        )
    )

    restored = load_spotlight_scientist_checkpoint(tmp_path / "latest.pkl")
    diagnostics = restored.diagnostics()["spotlight"]

    assert teacher_calls["count"] > 0
    assert diagnostics["habit_updates"] > 0
    assert "adaptation_updates" in diagnostics
