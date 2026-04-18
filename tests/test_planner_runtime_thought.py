from __future__ import annotations

from dataclasses import replace

import numpy as np
import torch

from arcagi.core.types import GridObservation, ObjectState, RuntimeThought, StructuredState
from arcagi.envs.synthetic import HiddenRuleEnv
from arcagi.memory.graph import StateGraph
from arcagi.models.world_model import WorldModelPrediction
from arcagi.perception.object_encoder import extract_structured_state
from arcagi.planning.planner import HybridPlanner, PlannerConfig
from arcagi.planning.rule_induction import object_signature


def _observation(
    grid: np.ndarray,
    *,
    actions: tuple[str, ...] = ("1", "2", "5"),
) -> GridObservation:
    return GridObservation(
        task_id="arc/test",
        episode_id="episode-0",
        step_index=0,
        grid=grid,
        available_actions=actions,
        extras={
            "action_roles": {
                "1": "move_up",
                "2": "move_down",
                "5": "select_cycle",
            }
        },
    )


class _FakeLanguageModel:
    def decode(self, _latent: torch.Tensor, mode: str = "belief", max_length: int = 8) -> tuple[str, ...]:
        if mode == "belief":
            return ("belief", "goal", "uncertain", "focus", "rule", "state", "probe")
        if mode == "question":
            return ("question", "need", "test", "focus", "rule", "state", "probe")
        return ("plan", "action", "select", "direction", "none", "focus", "rule", "state", "uncertain")


class _RawLanguageModel:
    def decode(self, _latent: torch.Tensor, mode: str = "belief", max_length: int = 8) -> tuple[str, ...]:
        if mode == "belief":
            return ("belief", "goal", "active", "focus", "target", "state", "commit")
        if mode == "question":
            return ("question", "need", "move", "focus", "target", "state", "commit")
        return ("plan", "action", "move", "direction", "up", "focus", "target", "state", "active")


def _object(
    object_id: str,
    color: int,
    cells: tuple[tuple[int, int], ...],
    *,
    tags: tuple[str, ...],
) -> ObjectState:
    ys = [cell[0] for cell in cells]
    xs = [cell[1] for cell in cells]
    return ObjectState(
        object_id=object_id,
        color=color,
        cells=cells,
        bbox=(min(ys), min(xs), max(ys), max(xs)),
        centroid=(float(sum(ys)) / len(ys), float(sum(xs)) / len(xs)),
        area=len(cells),
        tags=tags,
    )


def _interaction_state() -> tuple[StructuredState, ObjectState, ObjectState]:
    agent = _object("agent", 2, ((1, 1),), tags=("agent",))
    left = _object("left", 3, ((1, 0),), tags=("interactable",))
    right = _object("right", 4, ((1, 2),), tags=("interactable",))
    grid = np.zeros((3, 3), dtype=np.int64)
    for obj in (agent, left, right):
        for y, x in obj.cells:
            grid[y, x] = obj.color
    state = StructuredState(
        task_id="planner/test",
        episode_id="episode-0",
        step_index=0,
        grid_shape=grid.shape,
        grid_signature=tuple(int(value) for value in grid.reshape(-1)),
        objects=(agent, left, right),
        relations=(),
        affordances=("interact_left", "interact_right", "wait"),
        action_roles=(("interact_left", "interact"), ("interact_right", "interact"), ("wait", "wait")),
    )
    return state, left, right


class _FakeWorldModel:
    def __init__(self) -> None:
        self.calls: list[str] = []

    def step(
        self,
        latent: torch.Tensor,
        actions,
        state=None,
        hidden: torch.Tensor | None = None,
        action_embeddings=None,
    ) -> WorldModelPrediction:
        action = actions[0]
        self.calls.append(action)
        hidden_state = torch.zeros((1, 1), dtype=torch.float32, device=latent.device) if hidden is None else hidden.clone()
        reward = 0.0
        usefulness = 0.0
        policy = 0.0
        if action == "1":
            reward = 1.5 if float(hidden_state[0, 0]) > 0.5 else 0.1
            usefulness = 0.3 if float(hidden_state[0, 0]) > 0.5 else 0.05
            policy = 0.4 if float(hidden_state[0, 0]) > 0.5 else 0.1
        elif action == "2":
            reward = 0.0
            usefulness = 0.0
            policy = 0.05
        elif action == "5":
            hidden_state = torch.ones((1, 1), dtype=torch.float32, device=latent.device)
            policy = 0.1
        return WorldModelPrediction(
            hidden=hidden_state,
            next_latent_mean=latent.clone(),
            next_latent_std=torch.zeros_like(latent),
            reward=torch.tensor([reward], dtype=torch.float32, device=latent.device),
            usefulness=torch.tensor([usefulness], dtype=torch.float32, device=latent.device),
            policy=torch.tensor([policy], dtype=torch.float32, device=latent.device),
            delta=torch.zeros((1, 25), dtype=torch.float32, device=latent.device),
            uncertainty=torch.tensor([0.05], dtype=torch.float32, device=latent.device),
        )


class _ProxySensitiveWorldModel:
    def __init__(self, original_state) -> None:
        self.original_state = original_state

    def step(
        self,
        latent: torch.Tensor,
        actions,
        state=None,
        hidden: torch.Tensor | None = None,
        action_embeddings=None,
    ) -> WorldModelPrediction:
        action = actions[0]
        reward = 0.0
        usefulness = 0.0
        policy = 0.0
        next_hidden = torch.zeros((1, 1), dtype=torch.float32, device=latent.device)
        if action == "5":
            next_hidden = torch.ones((1, 1), dtype=torch.float32, device=latent.device)
        elif action == "1" and state is not self.original_state:
            reward = 1.25
            usefulness = 0.5
            policy = 0.4
        elif action == "1":
            reward = 0.0
            usefulness = 0.0
            policy = 0.05
        return WorldModelPrediction(
            hidden=next_hidden if hidden is None else hidden.clone(),
            next_latent_mean=latent.clone(),
            next_latent_std=torch.zeros_like(latent),
            reward=torch.tensor([reward], dtype=torch.float32, device=latent.device),
            usefulness=torch.tensor([usefulness], dtype=torch.float32, device=latent.device),
            policy=torch.tensor([policy], dtype=torch.float32, device=latent.device),
            delta=torch.ones((1, 25), dtype=torch.float32, device=latent.device) * 0.1,
            uncertainty=torch.tensor([0.05], dtype=torch.float32, device=latent.device),
        )


class _UncertaintyTrapWorldModel:
    def step(
        self,
        latent: torch.Tensor,
        actions,
        state=None,
        hidden: torch.Tensor | None = None,
        action_embeddings=None,
    ) -> WorldModelPrediction:
        action = actions[0]
        hidden_state = torch.zeros((1, 1), dtype=torch.float32, device=latent.device) if hidden is None else hidden.clone()
        reward = 0.0
        usefulness = 0.0
        policy = 0.0
        uncertainty = 0.9
        next_hidden = hidden_state
        if action == "1" and float(hidden_state[0, 0]) <= 0.5:
            reward = 0.4
            usefulness = 0.2
            policy = 0.2
            uncertainty = 0.1
        elif action == "2":
            next_hidden = torch.ones((1, 1), dtype=torch.float32, device=latent.device)
            reward = 0.0
            usefulness = 0.0
            policy = 0.05
            uncertainty = 1.5
        return WorldModelPrediction(
            hidden=next_hidden,
            next_latent_mean=latent.clone(),
            next_latent_std=torch.zeros_like(latent),
            reward=torch.tensor([reward], dtype=torch.float32, device=latent.device),
            usefulness=torch.tensor([usefulness], dtype=torch.float32, device=latent.device),
            policy=torch.tensor([policy], dtype=torch.float32, device=latent.device),
            delta=torch.zeros((1, 25), dtype=torch.float32, device=latent.device),
            uncertainty=torch.tensor([uncertainty], dtype=torch.float32, device=latent.device),
        )


class _PolicyTrapWorldModel:
    def step(
        self,
        latent: torch.Tensor,
        actions,
        state=None,
        hidden: torch.Tensor | None = None,
        action_embeddings=None,
    ) -> WorldModelPrediction:
        action = actions[0]
        if action == "1":
            reward = 0.2
            usefulness = 0.2
            policy = 0.05
        else:
            reward = -0.25
            usefulness = -0.35
            policy = 3.0
        return WorldModelPrediction(
            hidden=torch.zeros((1, 1), dtype=torch.float32, device=latent.device),
            next_latent_mean=latent.clone(),
            next_latent_std=torch.zeros_like(latent),
            reward=torch.tensor([reward], dtype=torch.float32, device=latent.device),
            usefulness=torch.tensor([usefulness], dtype=torch.float32, device=latent.device),
            policy=torch.tensor([policy], dtype=torch.float32, device=latent.device),
            delta=torch.zeros((1, 25), dtype=torch.float32, device=latent.device),
            uncertainty=torch.tensor([0.05], dtype=torch.float32, device=latent.device),
        )


def test_runtime_thought_estimates_selector_followup_gain() -> None:
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 0, 3],
                ],
                dtype=np.int64,
            )
        )
    )
    planner = HybridPlanner()
    graph = StateGraph()
    graph.visit(state)
    latent = torch.zeros((1, 4), dtype=torch.float32)
    thought = planner.build_runtime_thought(
        state=state,
        latent=latent,
        graph=graph,
        world_model=_FakeWorldModel(),
        hidden=None,
        language_model=_FakeLanguageModel(),
    )

    selector_thought = thought.for_action("5")

    assert thought.question_tokens[:4] == ("question", "need", "test", "focus")
    assert thought.plan_tokens[:4] == ("plan", "action", "select", "direction")
    assert selector_thought is not None
    assert selector_thought.selector_followup > 1.0
    assert selector_thought.next_latent is not None
    assert selector_thought.next_hidden is not None
    assert thought.world_model_calls == 5


def test_runtime_thought_keeps_decoded_language_tokens_without_synthesizing_grounded_replacements() -> None:
    env = HiddenRuleEnv(family_mode="switch_unlock", family_variant="red", size=9, seed=13)
    state = extract_structured_state(env.reset(seed=13))
    planner = HybridPlanner()
    graph = StateGraph()
    graph.visit(state)
    latent = torch.zeros((1, 4), dtype=torch.float32)

    thought = planner.build_runtime_thought(
        state=state,
        latent=latent,
        graph=graph,
        world_model=_FakeWorldModel(),
        hidden=None,
        language_model=_RawLanguageModel(),
    )

    assert thought.belief_tokens == ("belief", "goal", "active", "focus", "target", "state", "commit")
    assert thought.question_tokens == ("question", "need", "move", "focus", "target", "state", "commit")
    assert thought.plan_tokens == ("plan", "action", "move", "direction", "up", "focus", "target", "state", "active")


def test_runtime_thought_does_not_synthesize_question_tokens_without_language() -> None:
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 0, 3],
                ],
                dtype=np.int64,
            )
        )
    )
    planner = HybridPlanner()
    graph = StateGraph()
    graph.visit(state)
    latent = torch.zeros((1, 4), dtype=torch.float32)
    thought = planner.build_runtime_thought(
        state=state,
        latent=latent,
        graph=graph,
        world_model=_FakeWorldModel(),
        hidden=None,
        language_model=None,
    )

    assert thought.question_tokens == ()
    assert thought.plan_tokens == ()
    assert thought.for_action("5") is not None


def test_runtime_thought_uses_imagined_state_proxy_for_selector_followup() -> None:
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 0, 3],
                ],
                dtype=np.int64,
            )
        )
    )
    planner = HybridPlanner()
    graph = StateGraph()
    graph.visit(state)
    latent = torch.zeros((1, 4), dtype=torch.float32)

    thought = planner.build_runtime_thought(
        state=state,
        latent=latent,
        graph=graph,
        world_model=_ProxySensitiveWorldModel(state),
        hidden=None,
        language_model=None,
    )

    selector_thought = thought.for_action("5")

    assert selector_thought is not None
    assert selector_thought.selector_followup > 1.0
    assert selector_thought.next_state_proxy is not None
    assert selector_thought.next_state_proxy is not state


def test_planner_explicitly_uses_spatial_workspace_directional_signal() -> None:
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 0, 0],
                ],
                dtype=np.int64,
            ),
            actions=("up", "down", "wait"),
        )
    )
    state = replace(
        state,
        inventory=tuple(
            sorted(
                {
                    ("belief_frontier_direction", "up"),
                    ("belief_frontier_distance", "near"),
                    ("belief_visited_cells", "1"),
                }
            )
        ),
        flags=tuple(sorted({("belief_has_spatial_anchor", "0")})),
    )
    planner = HybridPlanner(PlannerConfig(search_depth=1))
    graph = StateGraph()
    graph.visit(state)

    plan = planner.choose_action(
        state=state,
        latent=None,
        graph=graph,
        world_model=None,
        hidden=None,
        language_model=None,
        thought=RuntimeThought(question_tokens=("need", "explore")),
    )

    assert plan.action == "up"


def test_planner_uses_context_bias_for_targeted_interaction() -> None:
    state, left, right = _interaction_state()
    planner = HybridPlanner(PlannerConfig(search_depth=1))
    graph = StateGraph()
    graph.visit(state)

    plan = planner.choose_action(
        state=state,
        latent=None,
        graph=graph,
        world_model=None,
        hidden=None,
        language_model=None,
        context_bias={
            ("interact", object_signature(left)): -1.0,
            ("interact", object_signature(right)): 1.0,
        },
        thought=RuntimeThought(question_tokens=("need", "test")),
    )

    assert plan.action == "interact_right"
    assert plan.scores["context_online_bias"] > 0.0


def test_budgeted_search_reuses_root_predictions_and_prunes_rollouts() -> None:
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 0, 3],
                ],
                dtype=np.int64,
            ),
            actions=("1", "2", "5"),
        )
    )
    planner = HybridPlanner(
        PlannerConfig(
            search_depth=4,
            search_root_width=2,
            search_branch_width=1,
            max_world_model_calls=11,
        )
    )
    graph = StateGraph()
    graph.visit(state)
    latent = torch.zeros((1, 4), dtype=torch.float32)
    world_model = _FakeWorldModel()
    thought = planner.build_runtime_thought(
        state=state,
        latent=latent,
        graph=graph,
        world_model=world_model,
        hidden=None,
        language_model=_FakeLanguageModel(),
    )

    plan = planner.choose_action(
        state=state,
        latent=latent,
        graph=graph,
        world_model=world_model,
        hidden=None,
        language_model=_FakeLanguageModel(),
        thought=thought,
    )

    assert plan.action in {"1", "5"}
    assert thought.world_model_calls == 5
    assert len(world_model.calls) <= 11
    assert plan.scores["search_budget_remaining"] >= 0.0


def test_lookahead_does_not_reward_branch_only_for_uncertainty() -> None:
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 0, 3],
                ],
                dtype=np.int64,
            ),
            actions=("1", "2"),
        )
    )
    planner = HybridPlanner(
        PlannerConfig(
            search_depth=3,
            search_root_width=2,
            search_branch_width=2,
            max_world_model_calls=32,
        )
    )
    graph = StateGraph()
    graph.visit(state)
    latent = torch.zeros((1, 4), dtype=torch.float32)
    world_model = _UncertaintyTrapWorldModel()
    thoughts, _ = planner._build_action_thoughts(
        state,
        latent,
        world_model,
        None,
        graph=graph,
    )
    thought_by_action = {candidate.action: candidate for candidate in thoughts}

    search_one = planner._lookahead(
        thought_by_action["1"].next_latent,
        thought_by_action["1"].next_hidden,
        world_model,
        depth=2,
        state=thought_by_action["1"].next_state_proxy,
        budget=[32],
        branch_width=2,
    )
    search_two = planner._lookahead(
        thought_by_action["2"].next_latent,
        thought_by_action["2"].next_hidden,
        world_model,
        depth=2,
        state=thought_by_action["2"].next_state_proxy,
        budget=[32],
        branch_width=2,
    )

    assert search_one > search_two


def test_planner_does_not_let_policy_prior_rescue_negative_value_action() -> None:
    state = extract_structured_state(
        _observation(
            np.array(
                [
                    [0, 0, 0],
                    [0, 2, 0],
                    [0, 0, 3],
                ],
                dtype=np.int64,
            ),
            actions=("1", "2"),
        )
    )
    planner = HybridPlanner(
        PlannerConfig(
            search_depth=2,
            search_root_width=2,
            search_branch_width=2,
            max_world_model_calls=16,
        )
    )
    graph = StateGraph()
    graph.visit(state)
    latent = torch.zeros((1, 4), dtype=torch.float32)
    world_model = _PolicyTrapWorldModel()

    thought = planner.build_runtime_thought(
        state=state,
        latent=latent,
        graph=graph,
        world_model=world_model,
        hidden=None,
        language_model=None,
    )
    plan = planner.choose_action(
        state=state,
        latent=latent,
        graph=graph,
        world_model=world_model,
        hidden=None,
        language_model=None,
        thought=thought,
    )

    assert plan.action == "1"
