from __future__ import annotations

import numpy as np

from arcagi.core.types import GridObservation
from arcagi.perception.object_encoder import extract_structured_state


def test_generic_perception_does_not_infer_synthetic_roles_from_color_ids() -> None:
    observation = GridObservation(
        task_id="generic/test",
        episode_id="generic/test/episode_0",
        step_index=0,
        grid=np.asarray(
            [
                [0, 3, 0],
                [7, 9, 2],
                [0, 0, 0],
            ],
            dtype=np.int64,
        ),
        extras={},
    )
    state = extract_structured_state(observation)
    assert all(not obj.tags for obj in state.objects)
