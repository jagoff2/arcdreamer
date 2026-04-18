from __future__ import annotations

import numpy as np

from arcagi.core.types import ObjectState, Relation, StructuredState, Transition
from arcagi.envs.synthetic import HiddenRuleEnv
from arcagi.memory.graph import StateGraph
from arcagi.perception.object_encoder import extract_structured_state


def test_graph_novelty_decreases_after_update() -> None:
    env = HiddenRuleEnv(family_mode="switch_unlock", family_variant="blue", seed=5)
    observation = env.reset(seed=5)
    state = extract_structured_state(observation)
    graph = StateGraph()
    before = graph.action_novelty(state, "wait")
    result = env.step("wait")
    next_state = extract_structured_state(result.observation)
    graph.update(
        Transition(
            state=state,
            action="wait",
            reward=result.reward,
            next_state=next_state,
            terminated=False,
            info=result.info,
        )
    )
    after = graph.action_novelty(state, "wait")
    assert before > after


def test_graph_cycle_penalty_rises_for_recent_deterministic_loop() -> None:
    env = HiddenRuleEnv(family_mode="switch_unlock", family_variant="blue", seed=5)
    observation = env.reset(seed=5)
    state = extract_structured_state(observation)
    graph = StateGraph()
    graph.visit(state)

    result = env.step("wait")
    next_state = extract_structured_state(result.observation)
    graph.update(
        Transition(
            state=state,
            action="wait",
            reward=result.reward,
            next_state=next_state,
            terminated=False,
            info=result.info,
        )
    )
    graph.update(
        Transition(
            state=state,
            action="wait",
            reward=result.reward,
            next_state=next_state,
            terminated=False,
            info=result.info,
        )
    )

    assert graph.action_cycle_penalty(state, "wait") > 0.0


def test_graph_cycle_penalty_uses_monotonic_recency_not_episode_local_step_reset() -> None:
    env = HiddenRuleEnv(family_mode="switch_unlock", family_variant="blue", seed=5)
    graph = StateGraph()

    first_observation = env.reset(seed=5)
    first_state = extract_structured_state(first_observation)
    graph.visit(first_state)
    first_result = env.step("wait")
    first_next = extract_structured_state(first_result.observation)
    graph.update(
        Transition(
            state=first_state,
            action="wait",
            reward=first_result.reward,
            next_state=first_next,
            terminated=False,
            info=first_result.info,
        )
    )

    for advance_seed in range(6, 14):
        observation = env.reset(seed=advance_seed)
        graph.visit(extract_structured_state(observation))

    second_observation = env.reset(seed=15)
    second_state = extract_structured_state(second_observation)
    graph.visit(second_state)

    assert graph.action_cycle_penalty(second_state, "wait") == 0.0


def test_structured_state_fingerprint_is_invariant_to_object_id_permutation() -> None:
    object_a = ObjectState(
        object_id="obj_a",
        color=2,
        cells=((1, 1),),
        bbox=(1, 1, 1, 1),
        centroid=(1.0, 1.0),
        area=1,
        tags=("agent",),
    )
    object_b = ObjectState(
        object_id="obj_b",
        color=3,
        cells=((3, 3),),
        bbox=(3, 3, 3, 3),
        centroid=(3.0, 3.0),
        area=1,
        tags=("target",),
    )
    state_a = StructuredState(
        task_id="graph/test",
        episode_id="episode-0",
        step_index=0,
        grid_shape=(5, 5),
        grid_signature=tuple(np.zeros((5, 5), dtype=np.int64).reshape(-1)),
        objects=(object_a, object_b),
        relations=(Relation("near", "obj_a", "obj_b", 4.0),),
        affordances=("wait",),
    )
    swapped_object_a = ObjectState(
        object_id="obj_x",
        color=2,
        cells=((1, 1),),
        bbox=(1, 1, 1, 1),
        centroid=(1.0, 1.0),
        area=1,
        tags=("agent",),
    )
    swapped_object_b = ObjectState(
        object_id="obj_y",
        color=3,
        cells=((3, 3),),
        bbox=(3, 3, 3, 3),
        centroid=(3.0, 3.0),
        area=1,
        tags=("target",),
    )
    state_b = StructuredState(
        task_id="graph/test",
        episode_id="episode-0",
        step_index=0,
        grid_shape=(5, 5),
        grid_signature=tuple(np.zeros((5, 5), dtype=np.int64).reshape(-1)),
        objects=(swapped_object_b, swapped_object_a),
        relations=(Relation("near", "obj_x", "obj_y", 4.0),),
        affordances=("wait",),
    )

    assert state_a.fingerprint() == state_b.fingerprint()
