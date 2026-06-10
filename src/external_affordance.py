from __future__ import annotations

from collections import Counter
from typing import Any

import numpy as np

from .arcagi3_adapter import ArcAGI3Observation, MOVE_DELTAS, find_agent, in_bounds, parse_click


def summarize_affordances(observation: ArcAGI3Observation, previous_grid: np.ndarray | None = None) -> dict[str, Any]:
    grid = np.asarray(observation.grid, dtype=np.int64)
    counts = Counter(int(item) for item in grid.reshape(-1))
    background = counts.most_common(1)[0][0] if counts else 0
    non_background = np.argwhere(grid != background)
    changed = np.zeros(grid.shape, dtype=bool)
    if previous_grid is not None and previous_grid.shape == grid.shape:
        changed = previous_grid != grid
    rare_values = {
        value
        for value, count in counts.items()
        if value != background and count <= max(1, int(grid.size * 0.05))
    }
    click_regions = []
    for action in observation.available_actions:
        click = parse_click(action)
        if click is None:
            continue
        y, x = click
        if in_bounds(grid, y, x):
            click_regions.append(
                {
                    "action": action,
                    "cell": int(grid[y, x]),
                    "non_background": bool(grid[y, x] != background),
                    "rare": bool(int(grid[y, x]) in rare_values),
                    "changed": bool(changed[y, x]),
                }
            )
    return {
        "shape": list(grid.shape),
        "background": int(background),
        "non_background_count": int(len(non_background)),
        "changed_count": int(np.count_nonzero(changed)),
        "rare_value_count": int(len(rare_values)),
        "click_region_count": int(len(click_regions)),
        "click_regions_sample": click_regions[:8],
    }


def affordance_bias(
    observation: ArcAGI3Observation,
    action: str,
    summary: dict[str, Any],
    previous_grid: np.ndarray | None = None,
) -> float:
    grid = np.asarray(observation.grid, dtype=np.int64)
    background = int(summary.get("background", 0))
    changed = np.zeros(grid.shape, dtype=bool)
    if previous_grid is not None and previous_grid.shape == grid.shape:
        changed = previous_grid != grid
    click = parse_click(action)
    if click is not None:
        y, x = click
        if not in_bounds(grid, y, x):
            return -0.05
        value = int(grid[y, x])
        bias = 0.0
        bias += 0.16 if value != background else -0.02
        bias += 0.08 if changed[y, x] else 0.0
        return float(bias)
    agent = find_agent(grid)
    if action in MOVE_DELTAS and agent is not None:
        dy, dx = MOVE_DELTAS[action]
        ny, nx = agent[0] + dy, agent[1] + dx
        if not in_bounds(grid, ny, nx):
            return -0.08
        value = int(grid[ny, nx])
        bias = 0.04 if value != background else 0.0
        bias += 0.04 if changed[ny, nx] else 0.0
        return float(bias)
    return 0.0
