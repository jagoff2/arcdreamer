from __future__ import annotations

from dataclasses import dataclass
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from arcagi.training.goal_scientist_targets import (  # noqa: E402
    action_family,
    relabel_episode_samples,
    score_transition,
    summarize_credit,
)


@dataclass
class FakeState:
    episode_id: str
    step_index: int
    vector: tuple[float, ...]
    affordances: tuple[str, ...] = ("1", "2", "3", "4", "5", "click:1:1")

    def transition_vector(self):
        return self.vector

    def flags_dict(self):
        return {"interface_has_mode_actions": "1"}


def make_sample(index: int, action: str, reward: float, delta: tuple[float, ...], *, terminated: bool = False):
    state = FakeState("episode", index, (float(index), 0.0, 0.0))
    next_state = FakeState("episode", index + 1, tuple(state.vector[i] + delta[i] for i in range(len(delta))))
    return {
        "state": state,
        "next_state": next_state,
        "action": action,
        "reward": reward,
        "delta": delta,
        "terminated": terminated,
        "episode_id": "episode",
    }


def test_action_family_accepts_arc_and_adapter_spellings():
    assert action_family("ACTION1") == "move_up"
    assert action_family("2") == "other"  # raw numeric actions remain schema-dependent for the repo encoder
    assert action_family("ACTION5") == "interact"
    assert action_family("ACTION6") == "coordinate"
    assert action_family("click:3:4") == "coordinate"


def test_hindsight_relabels_no_effect_selector_as_sequence_prefix():
    samples = [
        make_sample(0, "click:1:1", 0.0, (0.0, 0.0, 0.0)),
        make_sample(1, "ACTION1", 0.0, (0.0, 0.0, 0.0)),
        make_sample(2, "ACTION5", 1.0, (0.5, 0.0, 0.0)),
    ]
    relabeled = relabel_episode_samples(samples, gamma=0.92, sequence_horizon=5)
    assert "sequence_prefix" in relabeled[0]["credit_tags"]
    assert "selector_binding" in relabeled[0]["credit_tags"]
    assert relabeled[0]["usefulness"] > 0.2
    assert relabeled[0]["policy_target"] > relabeled[1]["policy_target"]
    assert relabeled[2]["usefulness"] > relabeled[0]["usefulness"]


def test_repeated_no_effect_probe_becomes_hard_negative():
    state = FakeState("episode", 0, (0.0, 0.0, 0.0))
    sample = {
        "state": state,
        "next_state": state,
        "action": "ACTION1",
        "reward": 0.0,
        "delta": (0.0, 0.0, 0.0),
        "terminated": False,
    }
    seen = set()
    first = score_transition(sample, seen_state_actions=seen)
    seen.add(("state", "grid_signature", None, "action", "move_up"))  # deliberately wrong shape should not matter
    # Add the actual key by scoring through relabeling twice in one episode.
    relabeled = relabel_episode_samples([sample, sample], seen_state_actions=set())
    assert first.novelty == 1.0
    assert relabeled[1]["transition_credit"]["novelty"] == 0.0
    assert relabeled[1]["usefulness"] < relabeled[0]["usefulness"]
    assert relabeled[1]["diagnostic_weight"] < relabeled[0]["diagnostic_weight"]


def test_summary_reports_selector_and_no_effect_fractions():
    samples = [
        make_sample(0, "click:0:0", 0.0, (0.0, 0.0, 0.0)),
        make_sample(1, "ACTION5", 1.0, (1.0, 0.0, 0.0)),
    ]
    relabeled = relabel_episode_samples(samples)
    summary = summarize_credit(relabeled)
    assert summary["samples"] == 2.0
    assert summary["avg_policy_target"] > 0.0
    assert summary["no_effect_fraction"] >= 0.5


def test_terminal_failure_becomes_negative_outcome_signal():
    relabeled = relabel_episode_samples([make_sample(0, "ACTION1", 0.0, (0.6, 0.0, 0.0), terminated=True)])

    assert "terminal_failure" in relabeled[0]["credit_tags"]
    assert relabeled[0]["outcome_signal"] < 0.0
    assert relabeled[0]["usefulness"] < 0.0
    assert relabeled[0]["policy_target"] == 0.0


def test_move_only_delta_is_not_treated_like_structural_progress():
    move = score_transition(make_sample(0, "ACTION1", 0.0, (1.0, 0.0, 0.0)))
    interact = score_transition(make_sample(0, "ACTION5", 0.0, (1.0, 0.0, 0.0)))

    assert move.state_signal < interact.state_signal
    assert move.usefulness < interact.usefulness
