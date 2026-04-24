from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np

from arcagi.agents.learned_online_recurrent_agent import LearnedOnlineRecurrentAgent
from arcagi.core.types import GridObservation
from arcagi.evaluation.harness import build_agent, run_episode
from arcagi.learned_online.action_features import ActionFeatureConfig, encode_action_candidates
from arcagi.learned_online.curriculum import NullDenseClickTask, RowMajorSweepPolicy
from arcagi.learned_online.memory import OnlineEpisodicMemory, OnlineMemoryEntry
from arcagi.learned_online.minimal_model import MinimalOnlineModel
from arcagi.learned_online.questions import QuestionToken
from arcagi.learned_online.recurrent_policy import RECURRENT_CANDIDATE_INPUT_DIM
from arcagi.learned_online.recurrent_model import RecurrentOnlineModel
from arcagi.learned_online.signals import TransitionLabels


def test_recurrent_agent_scores_full_dense_surface_and_updates_hidden() -> None:
    agent = build_agent("learned_online_recurrent")
    actions = tuple(f"click:{x}:{y}" for y in range(6) for x in range(6))
    observation = GridObservation("recurrent_dense", "0", 0, np.zeros((6, 6), dtype=np.int64), actions)

    chosen = agent.act(observation)
    before_hidden = float(agent.diagnostics()["hidden_norm"])
    agent.update_after_step(observation, reward=0.0, terminated=False, info={})
    after_hidden = float(agent.diagnostics()["hidden_norm"])

    assert chosen in actions
    assert agent.diagnostics()["last_scored_action_count"] == len(actions)
    assert agent.diagnostics()["last_legal_action_count"] == len(actions)
    assert after_hidden != before_hidden


def test_recurrent_agent_does_not_construct_forbidden_controllers() -> None:
    agent = build_agent("learned_online_recurrent")

    assert agent.controller_kind == "learned_online_recurrent_v1"
    assert agent.claim_eligible_arc_controller is True
    assert agent.arc_competence_validated is False
    assert not hasattr(agent, "runtime_rule_controller")
    assert not hasattr(agent, "theory_manager")
    assert not hasattr(agent, "spotlight")
    assert not hasattr(agent, "planner")
    assert not hasattr(agent, "rule_inducer")


def test_recurrent_import_boundary_is_clean() -> None:
    root = Path(__file__).resolve().parents[1]
    checked = [
        root / "arcagi" / "agents" / "learned_online_recurrent_agent.py",
        root / "arcagi" / "learned_online" / "recurrent_model.py",
        root / "arcagi" / "learned_online" / "recurrent_policy.py",
        root / "arcagi" / "learned_online" / "sequence.py",
    ]
    forbidden = (
        "RuntimeRuleController",
        "HybridPlanner",
        "EpisodeRuleInducer",
        "EpisodeTheoryManager",
        "ActionSpotlight",
        "SpotlightScientistAgent",
        "GraphExplorerAgent",
    )
    for path in checked:
        source = path.read_text(encoding="utf-8")
        for token in forbidden:
            assert token not in source, f"{token} leaked into {path}"


def test_recurrent_source_has_no_diagnostic_binding_probe_dependency() -> None:
    import arcagi.agents.learned_online_recurrent_agent as mod

    source = inspect.getsource(mod)
    assert "diagnostic_binding_probe" not in source


def test_recurrent_update_uses_pre_outcome_belief_features(monkeypatch) -> None:
    agent = LearnedOnlineRecurrentAgent(seed=11)
    before = _obs(levels_completed=0, grid=np.zeros((2, 2), dtype=np.int64), actions=("a", "b"))
    state = agent.observe(before)
    agent.last_state = state
    agent.last_action = "a"
    agent.last_question = QuestionToken.TEST_ACTION_MEANING
    seen_update_counts: list[int] = []
    original = agent.policy.feature_for_action

    def wrapped_feature_for_action(state_arg, action_arg, *, question):
        seen_update_counts.append(int(agent.belief.online_update_count))
        return original(state_arg, action_arg, question=question)

    monkeypatch.setattr(agent.policy, "feature_for_action", wrapped_feature_for_action)
    agent.update_after_step(before, reward=0.0, terminated=False, info={})

    assert seen_update_counts
    assert seen_update_counts[0] == 0
    assert agent.diagnostics()["online_updates"] == 1


def test_checkpoint_load_resets_online_adaptation_counter_not_pretrain_count() -> None:
    labels = TransitionLabels(no_effect_nonprogress=1.0)
    model = MinimalOnlineModel(input_dim=3, learning_rate=0.5)
    feature = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    for _ in range(5):
        model.online_update(feature, labels)
    state = model.state_dict()

    restored = MinimalOnlineModel(input_dim=3, learning_rate=0.5)
    restored.load_state_dict(state)

    assert restored.updates == 5
    assert restored.pretrain_updates == 5
    assert restored.online_adapt_updates == 0
    restored.online_update(feature, labels)
    assert restored.online_adapt_updates == 1
    assert restored.updates == 6


def test_online_update_counter_increments_once_per_transition_update() -> None:
    model = MinimalOnlineModel(input_dim=3, learning_rate=0.5)
    feature = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    labels = TransitionLabels(no_effect_nonprogress=1.0)

    loss = model.online_update(feature, labels, realized_info_gain=0.7, return_credit=0.9)

    assert loss > 0.0
    assert model.online_adapt_updates == 1
    assert model.updates == 1


def test_return_credit_trains_value_not_immediate_usefulness() -> None:
    model = MinimalOnlineModel(input_dim=3, learning_rate=0.8)
    feature = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    labels = TransitionLabels(no_effect_nonprogress=1.0)
    before = model.predict(feature[None, :])

    model.online_update(feature, labels, return_credit=0.9)
    after = model.predict(feature[None, :])

    assert float(after.value[0]) > float(before.value[0])
    assert float(after.useful[0]) < float(before.useful[0])


def test_blended_success_credit_keeps_long_prefix_nontrivial() -> None:
    memory = OnlineEpisodicMemory(capacity=400)
    labels = TransitionLabels(no_effect_nonprogress=1.0)
    for index in range(301):
        memory.entries.append(
            OnlineMemoryEntry(
                state_key=np.zeros((2,), dtype=np.float32),
                context_key="context",
                action=f"a{index}",
                question=QuestionToken.TEST_ACTION_MEANING,
                labels=labels,
                realized_info_gain=0.0,
                feature=np.asarray([1.0, 0.0], dtype=np.float32),
                level_epoch=0,
            )
        )

    credited = memory.credit_recent_success(level_epoch=0)

    assert len(credited) == 301
    assert credited[0].return_credit > 0.30
    assert credited[-1].return_credit == 1.0


def test_value_ranking_raises_success_prefix_above_observed_decoy() -> None:
    model = MinimalOnlineModel(input_dim=4, learning_rate=0.8)
    positive = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    negative = np.asarray([0.0, 1.0, 0.0, 0.0], dtype=np.float32)
    before_gap = model.value_logit(positive) - model.value_logit(negative)

    violation = model.ranking_update_value(positive, negative, margin=0.2, weight=1.0)
    after_gap = model.value_logit(positive) - model.value_logit(negative)

    assert violation > 0.0
    assert after_gap > before_gap
    assert model.online_adapt_updates == 1


def test_imitation_update_ranks_demonstrated_feature_without_changing_value_logit() -> None:
    model = MinimalOnlineModel(input_dim=4, learning_rate=0.8)
    positive = np.asarray([1.0, 0.0, 0.0, 0.0], dtype=np.float32)
    negatives = np.asarray(
        [
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
        ],
        dtype=np.float32,
    )
    before_value = model.value_logit(positive)
    before_imitation_gap = model.imitation_logit(positive) - model.imitation_logit(negatives[0])

    loss = model.imitation_update(positive, negatives, margin=0.2, weight=1.0)
    after_imitation_gap = model.imitation_logit(positive) - model.imitation_logit(negatives[0])

    assert loss > 0.0
    assert after_imitation_gap > before_imitation_gap
    assert model.value_logit(positive) == before_value


def test_recurrent_value_ranking_uses_supplied_hidden_snapshots() -> None:
    model = RecurrentOnlineModel(candidate_input_dim=2, event_input_dim=2, hidden_dim=2, learning_rate=0.8, seed=2)
    positive = np.asarray([1.0, 0.0], dtype=np.float32)
    negative = np.asarray([1.0, 0.0], dtype=np.float32)
    positive_hidden = np.asarray([1.0, 0.0], dtype=np.float32)
    negative_hidden = np.asarray([0.0, 1.0], dtype=np.float32)
    before_gap = model.fast_heads.value_logit(model.augment_features(positive, hidden=positive_hidden)) - model.fast_heads.value_logit(
        model.augment_features(negative, hidden=negative_hidden)
    )

    violation = model.ranking_update_value(
        positive,
        negative,
        positive_hidden=positive_hidden,
        negative_hidden=negative_hidden,
        margin=0.2,
        weight=1.0,
    )
    after_gap = model.fast_heads.value_logit(model.augment_features(positive, hidden=positive_hidden)) - model.fast_heads.value_logit(
        model.augment_features(negative, hidden=negative_hidden)
    )

    assert violation > 0.0
    assert after_gap > before_gap


def test_success_credit_ranks_observed_sequence_action_above_no_effect_decoy() -> None:
    agent = LearnedOnlineRecurrentAgent(seed=21)
    level_epoch = 0
    negative_feature = np.zeros((RECURRENT_CANDIDATE_INPUT_DIM,), dtype=np.float32)
    positive_feature = np.zeros((RECURRENT_CANDIDATE_INPUT_DIM,), dtype=np.float32)
    reward_feature = np.zeros((RECURRENT_CANDIDATE_INPUT_DIM,), dtype=np.float32)
    negative_feature[0] = 1.0
    positive_feature[1] = 1.0
    reward_feature[2] = 1.0
    hidden = np.zeros((agent.hidden_dim,), dtype=np.float32)
    entries = [
        OnlineMemoryEntry(
            state_key=np.zeros((2,), dtype=np.float32),
            context_key="context",
            action="decoy",
            question=QuestionToken.TEST_ACTION_MEANING,
            labels=TransitionLabels(no_effect_nonprogress=1.0),
            realized_info_gain=0.0,
            feature=negative_feature,
            hidden=hidden,
            level_epoch=level_epoch,
        ),
        OnlineMemoryEntry(
            state_key=np.zeros((2,), dtype=np.float32),
            context_key="context",
            action="setup",
            question=QuestionToken.TEST_ACTION_MEANING,
            labels=TransitionLabels(action_availability_changed=1.0, mechanic_change=1.0, visible_change=1.0),
            realized_info_gain=0.0,
            feature=positive_feature,
            hidden=hidden,
            level_epoch=level_epoch,
        ),
        OnlineMemoryEntry(
            state_key=np.zeros((2,), dtype=np.float32),
            context_key="context",
            action="goal",
            question=QuestionToken.TEST_ACTION_MEANING,
            labels=TransitionLabels(reward_progress=1.0, objective_progress=1.0, visible_change=1.0),
            realized_info_gain=0.0,
            feature=reward_feature,
            hidden=hidden,
            level_epoch=level_epoch,
        ),
    ]
    agent.memory.entries.extend(entries)

    agent._credit_current_level_success(level_epoch)
    positive_logit = agent.model.fast_heads.value_logit(agent.model.augment_features(positive_feature, hidden=hidden))
    negative_logit = agent.model.fast_heads.value_logit(agent.model.augment_features(negative_feature, hidden=hidden))

    assert entries[0].return_credit > 0.0
    assert entries[1].return_credit > 0.0
    assert positive_logit > negative_logit


def test_probe_loss_uses_memory_hidden_snapshot_not_current_hidden() -> None:
    model = RecurrentOnlineModel(candidate_input_dim=3, event_input_dim=2, hidden_dim=2, seed=3)
    labels = TransitionLabels(no_effect_nonprogress=1.0)
    feature = np.asarray([1.0, 0.0, 0.0], dtype=np.float32)
    hidden_snapshot = np.asarray([0.0, 0.0], dtype=np.float32)
    model.fast_heads.weights["cost"][-1] = 1.0
    model.hidden = np.asarray([0.0, 5.0], dtype=np.float32)
    entry = OnlineMemoryEntry(
        state_key=np.zeros((2,), dtype=np.float32),
        context_key="context",
        action="a",
        question=QuestionToken.TEST_ACTION_MEANING,
        labels=labels,
        realized_info_gain=0.0,
        feature=feature,
        hidden=hidden_snapshot,
    )

    batch_loss = model.batch_prediction_loss([entry])
    snapshot_loss = model.prediction_loss(feature, labels, realized_info_gain=0.0, hidden=hidden_snapshot)
    current_loss = model.prediction_loss(feature, labels, realized_info_gain=0.0, hidden=model.hidden)

    assert abs(batch_loss - snapshot_loss) <= 1e-8
    assert abs(batch_loss - current_loss) > 1e-4


def test_level_boundary_preserves_session_belief_but_resets_level_belief_and_credits_memory() -> None:
    agent = LearnedOnlineRecurrentAgent(seed=12)
    before = _obs(levels_completed=0, grid=np.zeros((2, 2), dtype=np.int64), actions=("a", "b"))
    action = agent.act(before)
    after = _obs(levels_completed=1, grid=np.ones((2, 2), dtype=np.int64), actions=("a", "b"))
    agent.update_after_step(after, reward=1.0, terminated=False, info={})
    diagnostics = agent.diagnostics()
    belief = diagnostics["belief"]

    assert action in ("a", "b")
    assert diagnostics["level_epoch"] == 1
    assert diagnostics["level_step"] == 0
    assert belief["family_count"] >= 1
    assert belief["level_family_count"] == 0
    assert agent.memory.entries
    assert agent.memory.entries[-1].return_credit > 0.0


def test_trace_compact_diagnostics_include_recurrent_scores(tmp_path) -> None:
    class OneStepEnv:
        task_id = "one_step"
        family_id = "one_step"

        def reset(self, seed=None):
            return _obs(levels_completed=0, grid=np.zeros((2, 2), dtype=np.int64), actions=("a", "b"))

        def step(self, action):
            from arcagi.core.types import StepResult

            return StepResult(
                observation=_obs(levels_completed=0, grid=np.zeros((2, 2), dtype=np.int64), actions=("a", "b")),
                reward=0.0,
                terminated=False,
                truncated=False,
                info={},
            )

    trace_path = tmp_path / "trace.jsonl"
    run_episode(LearnedOnlineRecurrentAgent(seed=13), OneStepEnv(), seed=0, max_steps=1, trace_path=trace_path)
    rows = [json.loads(line) for line in trace_path.read_text().splitlines()]
    step = next(row for row in rows if row["event"] == "step")

    assert "last_top_scores" in step["diagnostics"]
    assert "score_entropy" in step["diagnostics"]
    assert "online_adapt_updates" in step["diagnostics"]


def test_clean_recurrent_action_features_disable_exact_action_identity() -> None:
    agent = LearnedOnlineRecurrentAgent(seed=14)
    observation = GridObservation("features", "0", 0, np.zeros((2, 2), dtype=np.int64), ("foo_a", "foo_b"))
    state = agent.observe(observation)
    clean = encode_action_candidates(state, ("foo_a", "foo_b"))
    exact = encode_action_candidates(
        state,
        ("foo_a", "foo_b"),
        config=ActionFeatureConfig(include_exact_action_projection=True),
    )

    assert np.allclose(clean.features[0], clean.features[1])
    assert not np.allclose(exact.features[0], exact.features[1])
    assert agent.policy.action_feature_config.include_exact_action_projection is False


def test_factorized_softmax_scores_full_dense_surface_with_full_support() -> None:
    agent = LearnedOnlineRecurrentAgent(seed=15)
    actions = tuple(f"click:{x}:{y}" for y in range(10) for x in range(10)) + ("1", "5", "7")
    observation = _obs(levels_completed=0, grid=np.zeros((10, 10), dtype=np.int64), actions=actions)
    state = agent.observe(observation)
    decisions = agent.policy.score_actions(state, actions, question=QuestionToken.TEST_PARAMETER_GROUNDING)
    probabilities, family_probabilities = agent.policy.factorized_action_probabilities(
        state,
        actions,
        {action: decision.score for action, decision in decisions.items()},
    )

    assert len(decisions) == len(actions)
    assert set(probabilities) == set(actions)
    assert all(value > 0.0 for value in probabilities.values())
    assert abs(sum(probabilities.values()) - 1.0) < 1e-8
    assert abs(sum(family_probabilities.values()) - 1.0) < 1e-8


def test_factorized_softmax_action_order_equivariant() -> None:
    agent = LearnedOnlineRecurrentAgent(seed=16)
    actions = tuple(f"click:{x}:{y}" for y in range(5) for x in range(5)) + ("1", "5", "7")
    observation = _obs(levels_completed=0, grid=np.zeros((5, 5), dtype=np.int64), actions=actions)
    state = agent.observe(observation)
    decisions = agent.policy.score_actions(state, actions, question=QuestionToken.TEST_PARAMETER_GROUNDING)
    probs_a, _families_a = agent.policy.factorized_action_probabilities(
        state,
        actions,
        {action: decision.score for action, decision in decisions.items()},
    )
    shuffled = list(actions)
    np.random.default_rng(16).shuffle(shuffled)
    decisions_b = agent.policy.score_actions(state, shuffled, question=QuestionToken.TEST_PARAMETER_GROUNDING)
    probs_b, _families_b = agent.policy.factorized_action_probabilities(
        state,
        shuffled,
        {action: decision.score for action, decision in decisions_b.items()},
    )

    assert set(probs_a) == set(probs_b)
    for action in actions:
        assert abs(probs_a[action] - probs_b[action]) < 1e-8


def test_flat_scores_are_family_count_normalized() -> None:
    agent = LearnedOnlineRecurrentAgent(seed=17)
    clicks = tuple(f"click:{x}:{y}" for y in range(20) for x in range(20))
    actions = clicks + ("0", "1", "5", "7")
    observation = _obs(levels_completed=0, grid=np.zeros((20, 20), dtype=np.int64), actions=actions)
    state = agent.observe(observation)
    flat_scores = {action: 0.0 for action in actions}
    probabilities, family_probabilities = agent.policy.factorized_action_probabilities(state, actions, flat_scores)

    assert 0.18 <= family_probabilities["click:none"] <= 0.22
    assert 0.18 <= family_probabilities["move:none"] <= 0.22
    assert 0.18 <= family_probabilities["select:none"] <= 0.22
    assert 0.18 <= family_probabilities["undo:none"] <= 0.22
    assert 0.18 <= family_probabilities["reset:none"] <= 0.22
    assert abs(sum(probabilities[action] for action in clicks) - family_probabilities["click:none"]) < 1e-8
    assert max(probabilities[action] for action in clicks) < 0.001


def test_factorized_softmax_not_row_major_or_untried_enumerator_on_null_dense_task() -> None:
    agent_coords = _rollout_click_coords(
        LearnedOnlineRecurrentAgent(seed=18),
        NullDenseClickTask(width=10, height=10),
        steps=30,
    )
    sweep_coords = _rollout_click_coords(RowMajorSweepPolicy(), NullDenseClickTask(width=10, height=10), steps=30)

    assert _monotone_sweep_score(agent_coords) < _monotone_sweep_score(sweep_coords)
    assert agent_coords != sweep_coords
    assert len(set(agent_coords)) < len(agent_coords)


def _obs(*, levels_completed: int, grid: np.ndarray, actions: tuple[str, ...]) -> GridObservation:
    roles = {action: "raw" for action in actions}
    roles.update({"0": "reset_level", "1": "move_up", "5": "select_cycle", "7": "undo"})
    return GridObservation(
        task_id="recurrent_test",
        episode_id="0",
        step_index=levels_completed,
        grid=grid,
        available_actions=actions,
        extras={
            "inventory": {
                "interface_levels_completed": str(levels_completed),
                "interface_game_state": "GameState.NOT_FINISHED",
            },
            "action_roles": roles,
        },
    )


def _rollout_click_coords(agent, env: NullDenseClickTask, *, steps: int) -> list[tuple[int, int]]:
    observation = env.reset(seed=0)
    reset = getattr(agent, "reset_episode", None)
    if callable(reset):
        reset()
    coords: list[tuple[int, int]] = []
    for _ in range(steps):
        action = str(agent.act(observation))
        parts = action.split(":")
        if len(parts) == 3 and parts[0] == "click":
            coords.append((int(parts[1]), int(parts[2])))
        result = env.step(action)
        update = getattr(agent, "update_after_step", None)
        if callable(update):
            update(
                next_observation=result.observation,
                reward=result.reward,
                terminated=result.terminated or result.truncated,
                info=result.info,
            )
        observation = result.observation
    return coords


def _monotone_sweep_score(coords: list[tuple[int, int]]) -> float:
    if len(coords) < 2:
        return 0.0
    ranks = [y * 1000 + x for x, y in coords]
    adjacent_increments = sum(1 for left, right in zip(ranks, ranks[1:]) if right == left + 1)
    monotone = sum(1 for left, right in zip(ranks, ranks[1:]) if right > left)
    return max(adjacent_increments, monotone) / float(len(ranks) - 1)
