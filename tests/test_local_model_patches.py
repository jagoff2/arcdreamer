from __future__ import annotations

import numpy as np
import torch

from arcagi.agents.learned_agent import LocalModelPatch
from arcagi.core.types import ActionThought, GridObservation, RuntimeThought
from arcagi.evaluation.harness import build_agent
from arcagi.perception.object_encoder import extract_structured_state


def _observation() -> GridObservation:
    return GridObservation(
        task_id="patch/test",
        episode_id="episode-0",
        step_index=0,
        grid=np.array(
            [
                [0, 0, 0],
                [0, 2, 0],
                [0, 0, 3],
            ],
            dtype=np.int64,
        ),
        available_actions=("left", "right"),
        extras={
            "action_roles": {
                "left": "move_left",
                "right": "move_right",
            },
            "cell_tags": {
                (1, 1): ("agent",),
            },
        },
    )


def test_hybrid_agent_applies_local_model_patches_to_runtime_thought() -> None:
    agent = build_agent("hybrid", device=torch.device("cpu"))
    state = extract_structured_state(_observation())
    agent.local_action_patches["right"] = LocalModelPatch(
        value_shift=1.0,
        policy_shift=0.5,
        usefulness_shift=0.3,
        uncertainty_shift=-0.2,
    )
    thought = RuntimeThought(
        actions=(
            ActionThought(action="left", value=0.2, uncertainty=0.4, policy=0.0, usefulness=0.1),
            ActionThought(action="right", value=0.2, uncertainty=0.4, policy=0.0, usefulness=0.1),
        )
    )

    patched = agent._apply_local_model_patches(state, thought)
    left = patched.for_action("left")
    right = patched.for_action("right")

    assert left is not None
    assert right is not None
    assert right.value > left.value
    assert right.policy > left.policy
    assert right.usefulness > left.usefulness
    assert right.uncertainty < left.uncertainty
    assert any(claim.claim_type == "local_patch" and claim.subject == "right" for claim in patched.claims)
