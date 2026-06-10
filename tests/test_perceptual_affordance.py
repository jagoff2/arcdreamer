from __future__ import annotations

import numpy as np

from src.arcagi3_adapter import ArcAGI3Observation, ArcAGI3StepResult
from src.external_perception_experiment import PerceptionController, no_hack_proof, variant_by_id
from src.perceptual_affordance import OnlinePerceptualAffordance, extract_regions


def obs(grid: list[list[int]], actions: tuple[str, ...] = ("click:1:1", "click:0:0")) -> ArcAGI3Observation:
    return ArcAGI3Observation(
        task_id="external/test",
        episode_id="episode",
        step_index=0,
        grid=np.asarray(grid, dtype=np.int64),
        available_actions=actions,
        extras={},
    )


def result(next_obs: ArcAGI3Observation, reward: float = 0.0, events: list[str] | None = None) -> ArcAGI3StepResult:
    return ArcAGI3StepResult(next_obs, reward, False, False, {"events": events or [], "score": reward, "normalized_score": reward})


class DummyAdapter:
    def __init__(self) -> None:
        self.device = "cpu"
        self.observed = []

    def reset(self) -> None:
        self.observed.clear()

    def choose_action(self, observation: ArcAGI3Observation):
        first, second = observation.available_actions[:2]
        return (
            first,
            {
                "action_scores": {first: 1.0, second: 0.98},
                "policy": {},
                "memory_recall": {},
                "drive": {},
                "hypothesis_state": {},
            },
        )

    def observe_transition(self, action: str, transition: ArcAGI3StepResult) -> None:
        self.observed.append((action, transition.reward))


def test_extract_regions_components_clicks_and_changes() -> None:
    before = np.asarray([[0, 0, 0], [0, 7, 0], [0, 0, 0]], dtype=np.int64)
    observation = obs([[0, 0, 0], [0, 7, 0], [0, 0, 5]], actions=("click:1:1", "click:2:2"))
    frame = extract_regions(observation, before)
    assert frame.background == 0
    assert len(frame.regions) == 2
    assert frame.changed_count == 1
    assert len(frame.click_regions) == 2
    assert frame.region_at(1, 1) is not None


def test_temporal_slots_track_region_for_three_frames() -> None:
    memory = OnlinePerceptualAffordance()
    observation = obs([[0, 0, 0], [0, 4, 0], [0, 0, 0]])
    for _ in range(3):
        frame = memory.perceive(observation)
    assert frame.regions[0].stable_frames >= 3
    assert memory.diagnostics()["stable_tracks"] >= 1


def test_action_effect_memory_records_public_change_reward_and_noop() -> None:
    memory = OnlinePerceptualAffordance()
    before = obs([[0, 0, 0], [0, 4, 0], [0, 0, 0]])
    after = obs([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    memory.perceive(before)
    row = memory.observe_transition(before, "click:1:1", result(after, reward=1.0, events=["positive_reward"]))
    assert row["changed"] is True
    assert row["stats"]["mean_reward"] > 0.0
    assert memory.diagnostics()["effect_entries"] >= 1


def test_prediction_beats_no_change_null_after_repeated_changes() -> None:
    memory = OnlinePerceptualAffordance()
    before = obs([[0, 0, 0], [0, 4, 0], [0, 0, 0]])
    after = obs([[0, 0, 0], [0, 0, 0], [0, 0, 0]])
    for _ in range(4):
        memory.perceive(before)
        memory.observe_transition(before, "click:1:1", result(after, reward=0.0, events=[]))
    summary = memory.prediction_summary()
    assert summary["model_accuracy"] > summary["null_accuracy"]


def test_null_patch_preserves_model_action_while_perception_runs() -> None:
    controller = PerceptionController(DummyAdapter(), variant_by_id("null_patch_control"))
    observation = obs([[0, 0, 0], [0, 4, 0], [0, 0, 0]], actions=("click:0:0", "click:1:1"))
    action, diag = controller.choose_action(observation)
    assert action == "click:0:0"
    assert diag["policy"]["changed_action"] is False
    assert controller.summary()["perception"]["component_count"] > 0


def test_full_perception_can_change_action_after_learned_effect() -> None:
    controller = PerceptionController(DummyAdapter(), variant_by_id("full_perceptual_affordance"))
    before = obs([[0, 0, 0], [0, 4, 0], [0, 0, 0]], actions=("click:0:0", "click:1:1"))
    after = obs([[0, 0, 0], [0, 0, 0], [0, 0, 0]], actions=("click:0:0", "click:1:1"))
    controller._last_observation = before
    controller.memory.perceive(before)
    controller.memory.observe_transition(before, "click:1:1", result(after, reward=1.0, events=["positive_reward"]))
    action, diag = controller.choose_action(before)
    assert action == "click:1:1"
    assert diag["policy"]["changed_action"] is True


def test_perception_no_hack_source_scan_passes() -> None:
    assert no_hack_proof()["passes"] is True
