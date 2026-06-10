from __future__ import annotations

import numpy as np
import torch

from src.arcagi3_adapter import ArcAGI3Observation, ArcAGI3StepResult
from src.external_affordance import affordance_bias, summarize_affordances
from src.external_collapse_experiment import CUDA_PREFERRED_DEVICE, CollapseController, cuda_runtime_info, no_hack_proof, variant_by_id
from src.external_valence import ConsequenceValence, public_valence


def obs(grid: list[list[int]], actions: tuple[str, ...] = ("1", "2")) -> ArcAGI3Observation:
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
        self.observed = []

    def reset(self) -> None:
        self.observed.clear()

    def choose_action(self, observation: ArcAGI3Observation):
        return (
            observation.available_actions[0],
            {
                "action_scores": {observation.available_actions[0]: 1.0, observation.available_actions[1]: 0.9},
                "policy": {},
                "memory_recall": {},
                "drive": {},
                "hypothesis_state": {},
            },
        )

    def observe_transition(self, action: str, transition: ArcAGI3StepResult) -> None:
        self.observed.append((action, transition.reward))


def test_valence_records_public_no_effect_only_after_observation() -> None:
    before = obs([[0, 0], [0, 2]])
    transition = result(before, reward=0.0, events=[])
    valence = ConsequenceValence()
    assert valence.loop_penalty(before, "1") == 0.0
    item = valence.observe(before, "1", transition)
    assert item["no_effect"] is True
    assert valence.loop_penalty(before, "1") > 0.0
    assert valence.score(before, "1") < 0.0


def test_public_valence_uses_reward_event_and_visible_change() -> None:
    before = obs([[0, 0], [0, 2]])
    after = obs([[0, 3], [0, 2]])
    item = public_valence(before, "1", result(after, reward=1.0, events=["positive_reward"]))
    assert item["positive_event"] is True
    assert item["changed"] is True
    assert item["value"] > 1.0


def test_affordance_summary_and_bias_are_visible_grid_only() -> None:
    observation = obs([[0, 0, 0], [0, 9, 0], [0, 0, 0]], actions=("click:1:1", "click:0:0"))
    summary = summarize_affordances(observation)
    assert summary["non_background_count"] == 1
    assert summary["click_region_count"] == 2
    assert affordance_bias(observation, "click:1:1", summary) > affordance_bias(observation, "click:0:0", summary)


def test_loop_aversion_changes_action_only_after_same_state_no_effect() -> None:
    adapter = DummyAdapter()
    controller = CollapseController(adapter, variant_by_id("loop_aversion_only"))
    observation = obs([[0, 0], [0, 2]])
    first, first_diag = controller.choose_action(observation)
    assert first == "1"
    assert first_diag["policy"]["changed_action"] is False
    controller.observe_transition(first, result(observation, reward=0.0, events=[]))
    second, second_diag = controller.choose_action(observation)
    assert second == "2"
    assert second_diag["policy"]["changed_action"] is True
    assert second_diag["policy"]["forced_cycle"] is False


def test_null_patch_control_preserves_model_action() -> None:
    adapter = DummyAdapter()
    controller = CollapseController(adapter, variant_by_id("null_patch_control"))
    action, diag = controller.choose_action(obs([[0, 0], [0, 2]]))
    assert action == "1"
    assert diag["policy"]["base_action"] == "1"
    assert diag["policy"]["changed_action"] is False


def test_cuda_preferred_device_and_runtime_info_match_torch() -> None:
    expected_default = "cuda" if torch.cuda.is_available() else "auto"
    assert CUDA_PREFERRED_DEVICE == expected_default
    info = cuda_runtime_info(expected_default)
    assert info["torch_cuda_available"] is torch.cuda.is_available()
    assert info["resolved_device"] == ("cuda" if torch.cuda.is_available() else "cpu")


def test_controller_summary_reports_device_without_dummy_model() -> None:
    adapter = DummyAdapter()
    controller = CollapseController(adapter, variant_by_id("baseline_unchanged"))
    assert controller.summary()["device"]["explorer_parameter_device"] == "unknown"


def test_collapse_no_hack_source_scan_passes() -> None:
    assert no_hack_proof()["passes"] is True
