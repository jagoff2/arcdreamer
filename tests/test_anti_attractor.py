from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from src.anti_attractor import HardAntiAttractorGate
from src import jepa_arc_eval


@dataclass
class _Obs:
    grid: np.ndarray
    available_actions: tuple[str, ...] = ("1", "2", "3", "4", "5", "7")
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass
class _Result:
    observation: _Obs
    reward: float = 0.0
    terminated: bool = False
    truncated: bool = False
    info: dict[str, Any] = field(default_factory=lambda: {"events": []})


def _obs(value: int = 0, actions: tuple[str, ...] = ("1", "2", "3", "4", "5", "7")) -> _Obs:
    return _Obs(grid=np.full((64, 64), value, dtype=np.int64), available_actions=actions)


def _result(next_obs: _Obs, reward: float = 0.0, events: list[str] | None = None) -> _Result:
    return _Result(observation=next_obs, reward=reward, info={"events": list(events or [])})


def test_no_5757_undo_cycle_sb26() -> None:
    gate = HardAntiAttractorGate(window=8, threshold=0.80)
    obs = _obs()
    for action in ["5", "7", "5", "7"]:
        gate.observe_transition(obs, action, _result(obs))

    selected, diagnostics = gate.select_action("5", ["5", "1", "2"], obs)

    assert selected == "1"
    assert diagnostics["changed_action"] is True
    assert diagnostics["vetoed_actions"][0]["action"] == "5"
    assert any("period_2" in reason for reason in diagnostics["vetoed_actions"][0]["reasons"])


def test_no_repeated_7_without_harm_or_deadend_su15() -> None:
    gate = HardAntiAttractorGate()
    obs = _obs()

    selected, diagnostics = gate.select_action("7", ["7", "2"], obs)

    assert selected == "2"
    assert diagnostics["vetoed_actions"][0]["action"] == "7"
    assert "undo_without_harm_deadend_verified_macro_or_first_probe" in diagnostics["vetoed_actions"][0]["reasons"]


def test_step_cost_does_not_make_repeated_undo_allowed() -> None:
    gate = HardAntiAttractorGate()
    obs = _obs()
    gate.observe_transition(obs, "7", _result(obs, reward=-0.001))

    selected, diagnostics = gate.select_action("7", ["7", "2"], obs)

    assert selected == "2"
    assert diagnostics["vetoed_actions"][0]["action"] == "7"
    assert "undo_without_harm_deadend_verified_macro_or_first_probe" in diagnostics["vetoed_actions"][0]["reasons"]


def test_undo_probe_allowed_once_after_nontrivial_transition_then_quarantined() -> None:
    gate = HardAntiAttractorGate()
    before = _obs(0)
    after = _obs(1)
    gate.observe_transition(before, "2", _result(after))

    selected, diagnostics = gate.select_action("7", ["7", "3"], after)
    assert selected == "7"
    assert diagnostics["vetoed_actions"] == []

    gate.observe_transition(after, "7", _result(before))
    selected, diagnostics = gate.select_action("7", ["7", "3"], before)

    assert selected == "3"
    assert diagnostics["vetoed_actions"][0]["action"] == "7"


def test_no_repeated_5_without_useful_effect_g50t() -> None:
    gate = HardAntiAttractorGate(window=8, threshold=0.80)
    obs = _obs()
    for _ in range(4):
        gate.observe_transition(obs, "5", _result(obs))

    selected, diagnostics = gate.select_action("5", ["5", "4"], obs)

    assert selected == "4"
    assert diagnostics["vetoed_actions"][0]["action"] == "5"
    assert any("period_1" in reason for reason in diagnostics["vetoed_actions"][0]["reasons"])


def test_immediate_simple_repeat_without_progress_is_vetoed() -> None:
    gate = HardAntiAttractorGate(window=8, threshold=0.80)
    obs = _obs()
    gate.observe_transition(obs, "4", _result(obs))

    selected, diagnostics = gate.select_action("4", ["4", "2"], obs)

    assert selected == "2"
    assert diagnostics["vetoed_actions"][0]["action"] == "4"
    assert "immediate_repeat:simple:no_progress_window" in diagnostics["vetoed_actions"][0]["reasons"]


def test_no_coordinate_two_click_oscillation_lf52() -> None:
    actions = ("click:42:19", "click:42:44", "click:10:10")
    gate = HardAntiAttractorGate(window=8, threshold=0.80)
    obs = _obs(actions=actions)
    for action in ["click:42:19", "click:42:44", "click:42:19", "click:42:44"]:
        gate.observe_transition(obs, action, _result(obs))

    selected, diagnostics = gate.select_action("click:42:19", actions, obs)

    assert selected == "click:10:10"
    assert diagnostics["vetoed_actions"][0]["action"] == "click:42:19"
    assert any("period_2" in reason for reason in diagnostics["vetoed_actions"][0]["reasons"])


def test_coordinate_repeat_stays_quarantined_after_single_alternate_click() -> None:
    actions = ("click:34:16", "click:28:16", "5")
    gate = HardAntiAttractorGate(window=12, threshold=0.80)
    obs = _obs(actions=actions)
    for _ in range(8):
        gate.observe_transition(obs, "click:34:16", _result(obs))
    gate.observe_transition(obs, "click:28:16", _result(obs))

    selected, diagnostics = gate.select_action("click:34:16", actions, obs)

    assert selected != "click:34:16"
    assert diagnostics["vetoed_actions"][0]["action"] == "click:34:16"
    assert any("action_repeat:coordinate" in reason for reason in diagnostics["vetoed_actions"][0]["reasons"])


def test_progress_event_prevents_loop_veto_inside_recent_window() -> None:
    gate = HardAntiAttractorGate(window=8, threshold=0.80)
    obs = _obs()
    gate.observe_transition(obs, "5", _result(obs, reward=0.1, events=["level_completed"]))
    for _ in range(3):
        gate.observe_transition(obs, "5", _result(obs))

    selected, diagnostics = gate.select_action("5", ["5", "4"], obs)

    assert selected == "5"
    assert diagnostics["vetoed_actions"] == []


def test_jepa_augmented_controller_applies_hard_veto_after_scoring() -> None:
    class _Base:
        def choose_action(self, observation: _Obs) -> tuple[str, dict[str, Any]]:
            del observation
            return "5", {"action_scores": {"5": 9.0, "1": 1.0}, "policy": {}}

        def observe_transition(self, action: str, result: _Result) -> None:
            del action, result

        def summary(self) -> dict[str, Any]:
            return {}

    class _Memory:
        action_counts: dict[str, int] = {"5": 4, "1": 4}
        active_sequence_source = ""
        positive_prefix_completed = False
        best_event_prefix: list[str] = []

        def start_attempt(self) -> None:
            pass

        def plan_scores_for_observation(self, observation: _Obs) -> dict[str, float]:
            return {action: 0.0 for action in observation.available_actions}

        def action_distribution_from_scores(self, scores: dict[str, float]) -> dict[str, float]:
            return {action: 1.0 / max(len(scores), 1) for action in scores}

        def public_no_effect_suppresses_action(self, action: str) -> bool:
            del action
            return False

        def _has_public_goal_evidence(self) -> bool:
            return False

        def summary(self) -> dict[str, Any]:
            return {}

    controller = object.__new__(jepa_arc_eval.JEPAAugmentedController)
    controller.variant = jepa_arc_eval.JEPAVariant("unit", use_memory=True)
    controller.base = _Base()
    controller.memory = _Memory()
    controller.jepa_model = None
    controller.device = "cpu"
    controller.changed_actions = 0
    controller.frames = 0
    controller.last_base_action = None
    controller.last_observation = None
    controller.bridge_history_frames = []
    controller.bridge_history_action_ids = []
    controller.bridge_history_action_features = []
    controller.bridge_history_legal_counts = []
    controller.rng = jepa_arc_eval.random.Random(0)
    controller.post_prefix_rng = jepa_arc_eval.random.Random(0)
    controller.anti_attractor = HardAntiAttractorGate(window=8, threshold=0.80)
    obs = _obs(actions=("5", "1"))
    for _ in range(4):
        controller.anti_attractor.observe_transition(obs, "5", _result(obs))

    action, diagnostics = controller.choose_action(obs)

    assert action == "1"
    policy = diagnostics["jepa_policy"]
    assert policy["pre_veto_chosen_action"] == "5"
    assert policy["hard_anti_attractor"]["changed_action"] is True
    assert policy["hard_anti_attractor"]["vetoed_actions"][0]["action"] == "5"


def test_jepa_augmented_controller_can_disable_hard_veto_for_ablation() -> None:
    class _Base:
        def choose_action(self, observation: _Obs) -> tuple[str, dict[str, Any]]:
            del observation
            return "5", {"action_scores": {"5": 9.0, "1": 1.0}, "policy": {}}

        def observe_transition(self, action: str, result: _Result) -> None:
            del action, result

        def summary(self) -> dict[str, Any]:
            return {}

    class _Memory:
        action_counts: dict[str, int] = {"5": 4, "1": 4}
        active_sequence_source = ""
        positive_prefix_completed = False
        best_event_prefix: list[str] = []

        def start_attempt(self) -> None:
            pass

        def plan_scores_for_observation(self, observation: _Obs) -> dict[str, float]:
            return {action: 0.0 for action in observation.available_actions}

        def action_distribution_from_scores(self, scores: dict[str, float]) -> dict[str, float]:
            return {action: 1.0 / max(len(scores), 1) for action in scores}

        def public_no_effect_suppresses_action(self, action: str) -> bool:
            del action
            return False

        def _has_public_goal_evidence(self) -> bool:
            return False

        def summary(self) -> dict[str, Any]:
            return {}

    controller = object.__new__(jepa_arc_eval.JEPAAugmentedController)
    controller.variant = jepa_arc_eval.JEPAVariant("unit", use_memory=True)
    controller.base = _Base()
    controller.memory = _Memory()
    controller.jepa_model = None
    controller.device = "cpu"
    controller.changed_actions = 0
    controller.frames = 0
    controller.last_base_action = None
    controller.last_observation = None
    controller.bridge_history_frames = []
    controller.bridge_history_action_ids = []
    controller.bridge_history_action_features = []
    controller.bridge_history_legal_counts = []
    controller.rng = jepa_arc_eval.random.Random(0)
    controller.post_prefix_rng = jepa_arc_eval.random.Random(0)
    controller.anti_attractor = HardAntiAttractorGate(window=8, threshold=0.80)
    controller.enable_hard_anti_attractor = False
    obs = _obs(actions=("5", "1"))
    for _ in range(4):
        controller.anti_attractor.observe_transition(obs, "5", _result(obs))

    action, diagnostics = controller.choose_action(obs)

    assert action == "5"
    policy = diagnostics["jepa_policy"]
    assert policy["pre_veto_chosen_action"] == "5"
    assert policy["hard_anti_attractor"]["disabled"] is True
    assert policy["hard_anti_attractor"]["changed_action"] is False
    assert policy["hard_anti_attractor"]["fail_open_reasons"] == ["disabled_for_ablation"]
