# ARC Dreamer hyper-generalizing scientist-agent patch

Generated: 2026-04-22

This document is both a repository review and a drop-in implementation plan. It describes the required edits in detail and then includes the full code for every new or edited file in this patch. The implementation is designed for the stated objective: a hyper-generalizing, online-learning, hypothesis-driven experimental scientist agent for ARC-AGI-3-style interactive environments.

Important claim boundary: this patch realizes the requested architecture and passes the included local tests, including a hidden-rule online-learning smoke test. It does not prove an official ARC-AGI-3 win rate. ARC-AGI-3 performance must be measured by running the patched repository against the official local or online toolkit.

## Sources and review basis

I reviewed the public repository through GitHub and the current ARC-AGI-3 documentation. The relevant public facts used for this patch are:

- Repository: https://github.com/jagoff2/arcdreamer/
- ARC-AGI-3 official page: https://arcprize.org/arc-agi/3/
- ARC-AGI-3 launch post: https://arcprize.org/blog/announcing-arc-agi-3
- ARC-AGI-3 quickstart: https://arc-agi.github.io/arc-agi/quickstart.html
- ARC-AGI-3 toolkit overview: https://arc-agi.github.io/arc-agi/
- ARC-AGI-3 local-vs-online docs: https://arc-agi.github.io/arc-agi/local-vs-online.html

The repo already contains many useful components: object/state extraction, graph memory, recurrent/hybrid learned modules, runtime rule-controller logic, synthetic benchmarks, and an ARC toolkit adapter. The README states that the objective is an online-learning agent with explicit rule containers induced online. The current public ARC slice status reported in the README is still 0/5, which means the next useful patch should emphasize architectural correction and measurable online experimentation rather than adding another static controller.

## Objective translated into engineering requirements

The phrase “hyper-generalize, online learning, hypothesis driven experimental scientist agent” implies a concrete control loop, not a single model class. The agent must treat each environment as an unknown experimental system. On every step it must: perceive objects and relations, generate falsifiable hypotheses, choose an action that maximizes a weighted mixture of expected reward and information gain, observe the transition, revise beliefs, write surprising evidence to memory, and then plan the next experiment.

ARC-AGI-3 specifically makes that loop necessary because the benchmark is interactive. The agent is not just filling an output grid. It must explore, infer latent rules, acquire goals without instructions, and adapt continuously. The current codebase has pieces of that, but the controller-heavy parts are still too close to hand-organized heuristics. The patch below makes the online experiment loop explicit and testable.

The key non-negotiable edits are:

1. Add a state/action/transition layer that is independent of any single ARC toolkit object shape.
2. Add object-centric perception that produces compact symbolic evidence.
3. Add a hypothesis engine whose rules are falsifiable objects with support, contradiction, posterior, and uncertainty.
4. Add a lightweight online world model that supplies uncertainty and weak reward/change predictions after every transition.
5. Add surprise-weighted episodic memory and option memory, so the agent can reuse successful local experiments without hardcoding games.
6. Add grounded internal language to make beliefs and experiment questions inspectable.
7. Replace greedy controller behavior with an information-gain planner that balances reward, novelty, hypothesis falsification, memory, uncertainty, learned navigation, and contact probes.
8. Make the agent compatible with the existing `arcagi.evaluation.harness` by exposing `act`, `update_after_step`, `reset_episode`, `reset_all`, and `latest_language`.
9. Add tests that verify perception, hypothesis induction, reward/score deduplication, and actual online solution of a hidden-rule smoke test.

## High-level architecture

The resulting architecture is:

```text
raw observation
  -> coerce GridFrame
  -> object-centric StructuredState
  -> candidate actions from legal actions + diagnostic probes
  -> score actions using:
       hypothesis expected reward/change
       hypothesis information gain
       online world-model reward/change/uncertainty
       state-action novelty
       episodic-memory recall
       learned navigation toward unexplained/productive objects
       penalties for blocked/no-effect actions
  -> execute action
  -> compare before/after states
  -> update hypotheses
  -> update bootstrap world model
  -> write surprising transition to memory
  -> update navigation/contact statistics
  -> repeat
```

This is deliberately not a giant offline learner. ARC-AGI-3 requires sample-efficient within-game adaptation. The implementation therefore uses small hashed feature vectors, a bootstrap linear ensemble, explicit object evidence, and fast symbolic hypotheses. Those components can be extended with larger learned priors, but the online loop remains interpretable and cheap enough for many steps per game.

## Detailed required edits

### 1. Add `arcagi/scientist/types.py`

This file defines the data contract for the new agent. It provides `GridFrame`, `ObjectToken`, `RelationToken`, `StructuredState`, `TransitionDelta`, `TransitionRecord`, and `ActionDecision`. It normalizes actions such as `up`, `move_up`, `ACTION1`, and click-style actions into canonical action families. It also adds adapter-aware target conversion so that display-space click actions generated by the existing ARC adapter can still be reasoned about in grid coordinates.

The file also includes `combined_progress_signal`. This is necessary because wrappers often expose the same progress twice: once as `reward` and once as `info["score_delta"]`. Without deduplication, the online world model and episodic memory become miscalibrated.

### 2. Add `arcagi/scientist/perception.py`

This file segments the grid into same-color connected components while suppressing the dominant background slab. It computes object signatures that are position-independent, then adds spatial relations such as left/right/above/below, same-color, and near/touching. It also compares consecutive states to extract changed cells, moved objects, appeared objects, disappeared objects, and touched objects.

This module is where raw grid pixels become evidence. The hypothesis engine should never need to reason directly over whole grids except through compact deltas.

### 3. Add `arcagi/scientist/hypotheses.py`

This file is the scientific core. Each hypothesis has a type, action family, parameters, description, evidence, posterior, uncertainty, and MDL-style penalty. The engine induces rules from transitions and then updates those rules against later evidence. The current supported hypothesis families are:

- `action_moves_object`: an action family moves an object kind by a delta.
- `targeted_action_changes_object`: a coordinate action changes/touches an object of a color/signature.
- `reward_when_touch_color`: progress follows contact or interaction with a color.
- `reward_when_state_has_color`: progress is associated with the presence of a color.
- `mode_action_changes_dynamics`: selector-like actions alter latent mode/control dynamics.

The engine exposes `score_action`, `diagnostic_actions`, `credible_hypotheses`, `uncertain_hypotheses`, `color_progress_priors`, and `controlled_object_colors`. The planner consumes those methods to choose experiments instead of merely choosing actions.

### 4. Add `arcagi/scientist/world_model.py` and `features.py`

The online model is intentionally small: a bootstrap SGD ensemble over hashed state-action features. It predicts reward and visible change and returns uncertainty from ensemble variance plus an early-sample uncertainty term. This is not intended to solve ARC-AGI-3 alone. It supplies epistemic uncertainty and weak predictive shaping to the planner.

### 5. Add `arcagi/scientist/memory.py`

The memory stores transitions only when they are surprising, visibly causal, or rewarding. This avoids flooding memory with repeated no-op moves. Positive transitions also become short option traces. Retrieval uses a mixture of vector cosine similarity and grounded-language token overlap.

### 6. Add `arcagi/scientist/language.py`

This is grounded internal language, not an LLM dependency. The module creates tokens and concise belief/question sentences from current state and hypotheses. The purpose is traceability, memory indexing, and evaluation-harness language output.

### 7. Add `arcagi/scientist/planner.py`

The planner is the main behavioral edit. It creates candidates from legal actions, hypothesis diagnostic actions, and coordinate probes. It scores every candidate using expected reward, information gain, novelty, expected change, world uncertainty, memory recall, learned navigation, risk, and repetition/no-effect penalties.

It also tracks blocked cells and contact outcomes online. When a movement action repeatedly causes no visible effect from a state, the planner treats the destination as blocked. When the controlled object reaches or overlaps an unexplained object without progress, it schedules `interact` probes. This is the difference between wandering and experimental navigation.

### 8. Add `arcagi/scientist/agent.py`

This file wires the components into one agent. It implements `act`, `observe_result`, `update_after_step`, `reset_episode`, `reset_all`, and `diagnostics`. The compatibility methods are important because the current evaluation harness calls `agent.update_after_step(...)`, while the generic runtime calls `observe_result(...)`.

### 9. Add `arcagi/agents/scientist_agent.py`

This wrapper exposes `HyperGeneralizingScientistAgent` in the existing agent namespace. It makes the new agent importable without changing every script.

### 10. Edit `arcagi/evaluation/harness.py`

The harness edit adds `scientist`, `hyper_scientist`, and `hyper-generalizing-scientist` to `build_agent`. It also makes step observation robust: if an agent exposes `update_after_step`, use it; otherwise, if it exposes `observe_result`, call that with the before/after observations. This lets existing agents keep working while the scientist agent receives the transition information it needs for online learning.

### 11. Add `arcagi/scientist/synthetic_env.py` and tests

The smoke-test environment is not a substitute for ARC-AGI-3. It is a small hidden-rule world with no instructions: the agent must discover movement, key acquisition, interaction, obstruction, and goal completion. The included test asserts that the agent solves it online. This is the minimum regression test for the new objective.

## Apply instructions

From the root of `jagoff2/arcdreamer`, overlay the patch files exactly as listed below. Then run:

```bash
python -m pip install -e ".[dev]"
PYTHONPATH=. pytest -q
PYTHONPATH=. python -m arcagi.scientist.cli --seed 3 --max-steps 80
```

For the existing evaluation harness, run the new agent with:

```bash
python -m arcagi.evaluation.harness arc --agent scientist --game-limit 5 --mode offline
```

For online scorecard submission, follow the official toolkit/API setup and use `--mode online` if the repository’s adapter maps that mode to the official operation mode. Local/offline iteration should be preferred for development because it avoids online rate limits and API-key requirements.

## Local validation performed

The patch tree was validated with:

```text
PYTHONPATH=. pytest -q
....                                                                     [100%]
4 passed in 0.87s
```

The hidden-rule smoke test was also sampled across multiple seeds and solved in 40 steps with total reward 1.1. Example:

```text
seed=1 "total_reward":1.1 "steps":40
seed=2 "total_reward":1.1 "steps":40
seed=3 "total_reward":1.1 "steps":40
```

Again, this validates the online-learning loop; it is not an official ARC-AGI-3 benchmark result.

## Known remaining risks before official ARC-AGI-3 runs

The agent depends on the quality of the grid abstraction produced by `ArcToolkitEnv`. If a game’s true state is not well represented by the display grid, the perception layer will need richer feature extraction. Coordinate actions are adapter-aware, but any change in the toolkit’s click coordinate convention should be retested. The hypothesis library is deliberately compact; it should be expanded after replay analysis of failed ARC-AGI-3 games, especially for inventory, irreversible switches, multi-object transformations, delayed rewards, and level-to-level transfer.

The right benchmark loop after applying this patch is not “try once and hope.” It is: run all local games, save traces, group failures by missing hypothesis type, add the smallest new falsifiable hypothesis family, and rerun. That is the scientific loop the code is designed to support.

## File inventory


### `arcagi/scientist/types.py`

New: core datatypes, action normalization, adapter-aware target mapping, reward/score progress handling.

```python
"""Core datatypes for the ARC-AGI-3 hypothesis-driven scientist agent.

The objects in this file deliberately avoid any dependency on ARC internals.  The
agent only assumes a turn-based environment that returns grid-like observations,
a finite action set, and scalar feedback.  The same types can therefore wrap the
existing ``arcagi.core.types.GridObservation`` objects, raw numpy grids, or the
official ARC toolkit wrapper.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from hashlib import blake2b
from typing import Any, Mapping, Sequence

import numpy as np

ActionName = str
GridArray = np.ndarray

MOVE_DELTAS: dict[str, tuple[int, int]] = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "move_up": (-1, 0),
    "move_down": (1, 0),
    "move_left": (0, -1),
    "move_right": (0, 1),
    "action1": (-1, 0),
    "action2": (1, 0),
    "action3": (0, -1),
    "action4": (0, 1),
}

TARGETED_FAMILIES = {
    "click",
    "interact_at",
    "touch",
    "probe",
    "action6",
    "select_at",
}

SELECTOR_FAMILIES = {
    "select",
    "select_cycle",
    "switch",
    "mode",
    "action5",
}

INTERACT_FAMILIES = {
    "interact",
    "use",
    "activate",
    "push",
    "pickup",
    "action6",
    "click",
}


def _stable_digest(text: str, *, size: int = 16) -> str:
    return blake2b(text.encode("utf-8"), digest_size=size).hexdigest()


def normalize_action(action: Any) -> ActionName:
    """Return the repository-wide string representation for an action."""

    if isinstance(action, str):
        return action
    name = getattr(action, "name", None)
    if isinstance(name, str):
        return name.lower()
    value = getattr(action, "value", None)
    if value is not None:
        return str(value).lower()
    return str(action).lower()


def action_family(action: ActionName) -> str:
    """Return a canonical family name, stripping coordinates and punctuation.

    Examples:
        ``click:4:7`` -> ``click``
        ``ACTION6@4,7`` -> ``action6``
        ``move_up`` -> ``up``
    """

    raw = normalize_action(action).strip().lower()
    if ":" in raw:
        raw = raw.split(":", 1)[0]
    if "@" in raw:
        raw = raw.split("@", 1)[0]
    raw = raw.replace("-", "_").replace(" ", "_")
    aliases = {
        "moveup": "up",
        "movedown": "down",
        "moveleft": "left",
        "moveright": "right",
        "north": "up",
        "south": "down",
        "west": "left",
        "east": "right",
        "a1": "action1",
        "a2": "action2",
        "a3": "action3",
        "a4": "action4",
        "a5": "action5",
        "a6": "action6",
        "a7": "action7",
    }
    raw = aliases.get(raw, raw)
    if raw in {"move_up", "action1"}:
        return "up"
    if raw in {"move_down", "action2"}:
        return "down"
    if raw in {"move_left", "action3"}:
        return "left"
    if raw in {"move_right", "action4"}:
        return "right"
    return raw


def is_move_action(action: ActionName) -> bool:
    return action_family(action) in {"up", "down", "left", "right"}


def action_delta(action: ActionName) -> tuple[int, int] | None:
    family = action_family(action)
    return {
        "up": (-1, 0),
        "down": (1, 0),
        "left": (0, -1),
        "right": (0, 1),
    }.get(family)


def is_selector_action(action: ActionName) -> bool:
    return action_family(action) in SELECTOR_FAMILIES


def is_interact_action(action: ActionName) -> bool:
    return action_family(action) in INTERACT_FAMILIES


def is_targeted_action(action: ActionName) -> bool:
    family = action_family(action)
    return family in TARGETED_FAMILIES or parse_action_target(action) is not None


def parse_action_target(action: ActionName) -> tuple[int, int] | None:
    """Parse row/column coordinates from an action string if present."""

    raw = normalize_action(action).strip().lower()
    coord_part = ""
    if ":" in raw:
        parts = raw.split(":")
        if len(parts) >= 3:
            coord_part = ",".join(parts[-2:])
    elif "@" in raw:
        coord_part = raw.split("@", 1)[1]
    if not coord_part:
        return None
    coord_part = coord_part.replace(";", ",").replace(" ", ",")
    pieces = [p for p in coord_part.split(",") if p != ""]
    if len(pieces) < 2:
        return None
    try:
        row = int(float(pieces[0]))
        col = int(float(pieces[1]))
    except ValueError:
        return None
    return row, col


def make_targeted_action(base_action: ActionName, row: int, col: int) -> ActionName:
    family = action_family(base_action)
    if family in {"action6", "click"}:
        return f"click:{int(row)}:{int(col)}"
    return f"{normalize_action(base_action)}:{int(row)}:{int(col)}"


def grid_cell_to_action_coordinates(row: int, col: int, extras: Mapping[str, Any] | None = None) -> tuple[int, int]:
    """Map a logical grid cell to click coordinates when adapter metadata exists."""

    meta = (extras or {}).get("camera_meta") if isinstance(extras, Mapping) else None
    if not isinstance(meta, Mapping):
        return int(row), int(col)
    try:
        scale = int(meta.get("scale", 1) or 1)
        pad_x = int(meta.get("pad_x", 0) or 0)
        pad_y = int(meta.get("pad_y", 0) or 0)
        camera_x = int(meta.get("x", 0) or 0)
        camera_y = int(meta.get("y", 0) or 0)
    except Exception:
        return int(row), int(col)
    display_x = int(pad_x + ((int(col) - camera_x) * scale) + (scale // 2))
    display_y = int(pad_y + ((int(row) - camera_y) * scale) + (scale // 2))
    return display_x, display_y


def action_target_to_grid_cell(action: ActionName, extras: Mapping[str, Any] | None = None) -> tuple[int, int] | None:
    """Parse an action target and map adapter click coordinates back to grid cells."""

    target = parse_action_target(action)
    if target is None:
        return None
    family = action_family(action)
    meta = (extras or {}).get("camera_meta") if isinstance(extras, Mapping) else None
    if family not in {"click", "action6"} or not isinstance(meta, Mapping):
        return target
    try:
        x, y = target
        scale = max(int(meta.get("scale", 1) or 1), 1)
        pad_x = int(meta.get("pad_x", 0) or 0)
        pad_y = int(meta.get("pad_y", 0) or 0)
        camera_x = int(meta.get("x", 0) or 0)
        camera_y = int(meta.get("y", 0) or 0)
        grid_col = int(round((int(x) - pad_x - (scale // 2)) / scale + camera_x))
        grid_row = int(round((int(y) - pad_y - (scale // 2)) / scale + camera_y))
        return grid_row, grid_col
    except Exception:
        return target


@dataclass(frozen=True)
class GridFrame:
    task_id: str
    episode_id: str
    step_index: int
    grid: GridArray
    available_actions: tuple[ActionName, ...] = ()
    extras: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "grid", np.asarray(self.grid, dtype=np.int64))
        object.__setattr__(self, "available_actions", tuple(normalize_action(a) for a in self.available_actions))

    @property
    def shape(self) -> tuple[int, int]:
        if self.grid.ndim != 2:
            raise ValueError(f"GridFrame expects a rank-2 grid, got shape {self.grid.shape!r}")
        return int(self.grid.shape[0]), int(self.grid.shape[1])

    @property
    def fingerprint(self) -> str:
        shape = "x".join(map(str, self.grid.shape))
        payload = self.grid.astype(np.int64, copy=False).tobytes()
        return _stable_digest(shape + ":" + blake2b(payload, digest_size=16).hexdigest())


@dataclass(frozen=True)
class ObjectToken:
    object_id: str
    color: int
    area: int
    bbox: tuple[int, int, int, int]
    centroid: tuple[float, float]
    cells: tuple[tuple[int, int], ...]
    shape_hash: str
    role_tags: tuple[str, ...] = ()

    @property
    def height(self) -> int:
        return self.bbox[2] - self.bbox[0] + 1

    @property
    def width(self) -> int:
        return self.bbox[3] - self.bbox[1] + 1

    @property
    def signature(self) -> str:
        """Object identity without absolute position.

        Absolute positions are intentionally excluded.  The signature is used for
        online rule transfer within a game family: "the same color/shape kind"
        rather than "the same instance at row 3, col 4".
        """

        tags = ",".join(sorted(self.role_tags))
        return f"c{self.color}:a{self.area}:h{self.height}:w{self.width}:s{self.shape_hash}:t{tags}"

    @property
    def center_cell(self) -> tuple[int, int]:
        return int(round(self.centroid[0])), int(round(self.centroid[1]))

    def distance_to(self, other: "ObjectToken") -> float:
        return abs(self.centroid[0] - other.centroid[0]) + abs(self.centroid[1] - other.centroid[1])


@dataclass(frozen=True)
class RelationToken:
    relation: str
    subject_id: str
    object_id: str
    value: float = 1.0

    def as_key(self) -> str:
        return f"{self.relation}:{self.subject_id}:{self.object_id}:{round(float(self.value), 3)}"


@dataclass(frozen=True)
class StructuredState:
    frame: GridFrame
    objects: tuple[ObjectToken, ...]
    relations: tuple[RelationToken, ...]
    dominant_color: int
    abstract_fingerprint: str
    exact_fingerprint: str
    metadata: Mapping[str, Any] = field(default_factory=dict)

    @property
    def grid(self) -> GridArray:
        return self.frame.grid

    @property
    def available_actions(self) -> tuple[ActionName, ...]:
        return self.frame.available_actions

    @property
    def step_index(self) -> int:
        return self.frame.step_index

    def object_by_id(self, object_id: str) -> ObjectToken | None:
        for obj in self.objects:
            if obj.object_id == object_id:
                return obj
        return None

    def objects_by_color(self, color: int) -> tuple[ObjectToken, ...]:
        return tuple(obj for obj in self.objects if obj.color == color)

    def role_objects(self, tag: str) -> tuple[ObjectToken, ...]:
        return tuple(obj for obj in self.objects if tag in obj.role_tags)

    def token_set(self) -> frozenset[str]:
        tokens: set[str] = {
            f"shape:{self.grid.shape[0]}x{self.grid.shape[1]}",
            f"dominant:c{self.dominant_color}",
            f"objects:{len(self.objects)}",
        }
        for obj in self.objects:
            tokens.add(f"object:c{obj.color}:area_bin:{min(obj.area // 2, 12)}")
            tokens.add(f"shape_hash:{obj.shape_hash}")
            for tag in obj.role_tags:
                tokens.add(f"role:{tag}:c{obj.color}")
        for rel in self.relations[:128]:
            tokens.add(f"rel:{rel.relation}")
        return frozenset(tokens)


@dataclass(frozen=True)
class ObjectMotion:
    before_id: str
    after_id: str
    before_signature: str
    after_signature: str
    color: int
    delta: tuple[int, int]
    distance: float


@dataclass(frozen=True)
class TransitionDelta:
    action: ActionName
    changed_cells: int
    changed_fraction: float
    moved_objects: tuple[ObjectMotion, ...]
    appeared: tuple[str, ...]
    disappeared: tuple[str, ...]
    touched_objects: tuple[str, ...]
    reward: float
    score_delta: float
    terminated: bool = False
    info: Mapping[str, Any] = field(default_factory=dict)

    @property
    def has_visible_effect(self) -> bool:
        return self.changed_cells > 0 or bool(self.moved_objects or self.appeared or self.disappeared)

    @property
    def is_positive(self) -> bool:
        return self.reward > 0.0 or self.score_delta > 0.0


@dataclass(frozen=True)
class TransitionRecord:
    before: StructuredState
    after: StructuredState
    delta: TransitionDelta

    @property
    def action(self) -> ActionName:
        return self.delta.action

    @property
    def reward(self) -> float:
        return self.delta.reward

    @property
    def step_index(self) -> int:
        return self.before.step_index


@dataclass(frozen=True)
class ActionDecision:
    action: ActionName
    score: float
    components: Mapping[str, float]
    language: tuple[str, ...]
    candidate_count: int
    chosen_reason: str




def combined_progress_signal(reward: float, score_delta: float) -> float:
    """Combine environment reward and score delta without double-counting.

    Many wrappers expose the same progress twice: once as ``reward`` and once as
    ``info["score_delta"]``.  When the two signals are numerically identical,
    treating them as independent evidence doubles the reward and corrupts online
    calibration.  When they differ, preserve both because ARC-like wrappers may
    use shaped reward plus score progress.
    """

    reward_f = float(reward)
    score_f = float(score_delta)
    if abs(reward_f) > 1e-12 and abs(score_f) > 1e-12 and abs(reward_f - score_f) <= 1e-9:
        return reward_f
    return reward_f + score_f


def coerce_grid_frame(
    observation: Any,
    *,
    task_id: str | None = None,
    episode_id: str | None = None,
    step_index: int | None = None,
    available_actions: Sequence[Any] | None = None,
    extras: Mapping[str, Any] | None = None,
) -> GridFrame:
    """Convert common ARC-like observation objects into ``GridFrame``.

    The function is deliberately permissive so the agent can be called from the
    current ``arcagi`` harness, the official toolkit, or a raw unit-test grid.
    """

    if isinstance(observation, GridFrame):
        return observation

    if isinstance(observation, np.ndarray) or isinstance(observation, list):
        grid = np.asarray(observation, dtype=np.int64)
        obs_task_id = task_id or "raw_grid"
        obs_episode_id = episode_id or f"{obs_task_id}/episode"
        obs_step_index = 0 if step_index is None else int(step_index)
        actions = tuple(normalize_action(a) for a in (available_actions or ()))
        return GridFrame(obs_task_id, obs_episode_id, obs_step_index, grid, actions, extras or {})

    grid = getattr(observation, "grid", None)
    if grid is None:
        grid = getattr(observation, "frame", None)
    if grid is None:
        grid = getattr(observation, "observation", None)
    if grid is None:
        raise TypeError(f"Cannot coerce observation of type {type(observation)!r} into GridFrame")

    obs_task_id = task_id or str(getattr(observation, "task_id", "arc_task"))
    obs_episode_id = episode_id or str(getattr(observation, "episode_id", f"{obs_task_id}/episode"))
    obs_step_index = int(step_index if step_index is not None else getattr(observation, "step_index", 0))

    if available_actions is None:
        available_actions = getattr(observation, "available_actions", ()) or getattr(observation, "actions", ()) or ()
    obs_extras = dict(getattr(observation, "extras", {}) or {})
    if extras:
        obs_extras.update(extras)
    return GridFrame(
        task_id=obs_task_id,
        episode_id=obs_episode_id,
        step_index=obs_step_index,
        grid=np.asarray(grid, dtype=np.int64),
        available_actions=tuple(normalize_action(a) for a in available_actions),
        extras=obs_extras,
    )

```

### `arcagi/scientist/perception.py`

New: object-centric segmentation, relation extraction, state differencing, motion/contact evidence.

```python
"""Object-centric grid perception and transition differencing.

This module replaces brittle per-game perception with a cheap, generic object
extractor.  It builds compact symbolic states from raw grids, computes spatial
relations, and compares consecutive states to supply causal evidence to the
hypothesis engine.
"""

from __future__ import annotations

from collections import Counter, deque
from hashlib import blake2b
from typing import Any, Iterable, Mapping

import numpy as np

from .types import (
    ActionName,
    GridFrame,
    ObjectMotion,
    ObjectToken,
    RelationToken,
    StructuredState,
    TransitionDelta,
    TransitionRecord,
    action_target_to_grid_cell,
    coerce_grid_frame,
)


def _digest(parts: Iterable[str], *, size: int = 10) -> str:
    h = blake2b(digest_size=size)
    for part in parts:
        h.update(part.encode("utf-8"))
        h.update(b"|")
    return h.hexdigest()


def _dominant_color(grid: np.ndarray) -> int:
    values, counts = np.unique(grid, return_counts=True)
    if len(values) == 0:
        return 0
    return int(values[int(np.argmax(counts))])


def _shape_hash(cells: tuple[tuple[int, int], ...]) -> str:
    min_r = min(r for r, _ in cells)
    min_c = min(c for _, c in cells)
    norm = sorted((r - min_r, c - min_c) for r, c in cells)
    return _digest((f"{r},{c}" for r, c in norm), size=8)


def _component_role_tags(
    *,
    color: int,
    area: int,
    bbox: tuple[int, int, int, int],
    grid_shape: tuple[int, int],
    dominant_color: int,
) -> tuple[str, ...]:
    tags: list[str] = []
    rows, cols = grid_shape
    r0, c0, r1, c1 = bbox
    grid_area = max(rows * cols, 1)
    if color == dominant_color and area > 0.35 * grid_area:
        tags.append("background_candidate")
    if r0 == 0 or c0 == 0 or r1 == rows - 1 or c1 == cols - 1:
        tags.append("boundary_touching")
    if area == 1:
        tags.append("point")
    if (r1 - r0 + 1) == 1 or (c1 - c0 + 1) == 1:
        tags.append("line_like")
    if area <= 4:
        tags.append("small")
    if area >= 0.10 * grid_area:
        tags.append("large")
    return tuple(tags)


def connected_components(grid: np.ndarray, *, include_dominant: bool = False) -> tuple[ObjectToken, ...]:
    """Segment same-color 4-connected components into object tokens."""

    grid = np.asarray(grid, dtype=np.int64)
    if grid.ndim != 2:
        raise ValueError(f"expected rank-2 grid, got {grid.shape!r}")
    rows, cols = grid.shape
    dominant = _dominant_color(grid)
    visited = np.zeros_like(grid, dtype=bool)
    objects: list[ObjectToken] = []
    color_counts: Counter[int] = Counter(int(x) for x in grid.ravel())
    dominant_area = color_counts[dominant]

    object_index = 0
    for start_r in range(rows):
        for start_c in range(cols):
            if visited[start_r, start_c]:
                continue
            color = int(grid[start_r, start_c])
            q: deque[tuple[int, int]] = deque([(start_r, start_c)])
            visited[start_r, start_c] = True
            cells: list[tuple[int, int]] = []
            while q:
                r, c = q.popleft()
                cells.append((r, c))
                for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                    nr, nc = r + dr, c + dc
                    if nr < 0 or nc < 0 or nr >= rows or nc >= cols:
                        continue
                    if visited[nr, nc] or int(grid[nr, nc]) != color:
                        continue
                    visited[nr, nc] = True
                    q.append((nr, nc))

            cell_tuple = tuple(sorted(cells))
            area = len(cell_tuple)
            # Drop the main background slab, but keep small components that happen to
            # share the dominant color.  ARC-like games often use color zero as empty
            # space, but relying only on color zero is too brittle for ARC-AGI-3.
            if (
                not include_dominant
                and color == dominant
                and area == dominant_area
                and area > 0.35 * rows * cols
            ):
                continue
            min_r = min(r for r, _ in cell_tuple)
            max_r = max(r for r, _ in cell_tuple)
            min_c = min(c for _, c in cell_tuple)
            max_c = max(c for _, c in cell_tuple)
            centroid = (
                float(sum(r for r, _ in cell_tuple)) / area,
                float(sum(c for _, c in cell_tuple)) / area,
            )
            bbox = (min_r, min_c, max_r, max_c)
            shape = _shape_hash(cell_tuple)
            tags = _component_role_tags(
                color=color,
                area=area,
                bbox=bbox,
                grid_shape=(rows, cols),
                dominant_color=dominant,
            )
            object_id = f"o{object_index:03d}_c{color}_{shape[:6]}_{min_r}_{min_c}"
            objects.append(
                ObjectToken(
                    object_id=object_id,
                    color=color,
                    area=area,
                    bbox=bbox,
                    centroid=centroid,
                    cells=cell_tuple,
                    shape_hash=shape,
                    role_tags=tags,
                )
            )
            object_index += 1
    return tuple(objects)


def _bbox_touching(a: ObjectToken, b: ObjectToken, *, margin: int = 1) -> bool:
    ar0, ac0, ar1, ac1 = a.bbox
    br0, bc0, br1, bc1 = b.bbox
    return not (ar1 + margin < br0 or br1 + margin < ar0 or ac1 + margin < bc0 or bc1 + margin < ac0)


def spatial_relations(objects: tuple[ObjectToken, ...], *, max_pairs: int = 160) -> tuple[RelationToken, ...]:
    relations: list[RelationToken] = []
    pair_count = 0
    for a in objects:
        for b in objects:
            if a.object_id == b.object_id:
                continue
            pair_count += 1
            if pair_count > max_pairs:
                return tuple(relations)
            if a.color == b.color:
                relations.append(RelationToken("same_color", a.object_id, b.object_id))
            dr = b.centroid[0] - a.centroid[0]
            dc = b.centroid[1] - a.centroid[1]
            if abs(dc) >= abs(dr) and dc > 0:
                relations.append(RelationToken("left_of", a.object_id, b.object_id, float(abs(dc))))
            if abs(dc) >= abs(dr) and dc < 0:
                relations.append(RelationToken("right_of", a.object_id, b.object_id, float(abs(dc))))
            if abs(dr) > abs(dc) and dr > 0:
                relations.append(RelationToken("above", a.object_id, b.object_id, float(abs(dr))))
            if abs(dr) > abs(dc) and dr < 0:
                relations.append(RelationToken("below", a.object_id, b.object_id, float(abs(dr))))
            if _bbox_touching(a, b, margin=1):
                relations.append(RelationToken("near_or_touching", a.object_id, b.object_id, float(abs(dr) + abs(dc))))
    return tuple(relations)


def abstract_fingerprint(objects: tuple[ObjectToken, ...], relations: tuple[RelationToken, ...], grid_shape: tuple[int, int]) -> str:
    object_parts = sorted(obj.signature for obj in objects)
    rel_parts = sorted(rel.relation for rel in relations)
    parts = [f"shape:{grid_shape[0]}x{grid_shape[1]}", *object_parts, *rel_parts[:64]]
    return _digest(parts, size=16)


def extract_state(observation: Any, *, include_dominant: bool = False) -> StructuredState:
    frame = coerce_grid_frame(observation)
    objects = connected_components(frame.grid, include_dominant=include_dominant)
    relations = spatial_relations(objects)
    dominant = _dominant_color(frame.grid)
    abstract = abstract_fingerprint(objects, relations, frame.shape)
    return StructuredState(
        frame=frame,
        objects=objects,
        relations=relations,
        dominant_color=dominant,
        abstract_fingerprint=abstract,
        exact_fingerprint=frame.fingerprint,
        metadata={"object_count": len(objects), "relation_count": len(relations)},
    )


def _bbox_iou(a: tuple[int, int, int, int], b: tuple[int, int, int, int]) -> float:
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    inter_r0 = max(ar0, br0)
    inter_c0 = max(ac0, bc0)
    inter_r1 = min(ar1, br1)
    inter_c1 = min(ac1, bc1)
    if inter_r1 < inter_r0 or inter_c1 < inter_c0:
        return 0.0
    inter = (inter_r1 - inter_r0 + 1) * (inter_c1 - inter_c0 + 1)
    area_a = (ar1 - ar0 + 1) * (ac1 - ac0 + 1)
    area_b = (br1 - br0 + 1) * (bc1 - bc0 + 1)
    return float(inter) / float(max(area_a + area_b - inter, 1))


def _match_objects(before: tuple[ObjectToken, ...], after: tuple[ObjectToken, ...]) -> dict[str, str]:
    candidates: list[tuple[float, str, str]] = []
    for b in before:
        for a in after:
            color_score = 2.0 if b.color == a.color else -1.0
            shape_score = 2.0 if b.shape_hash == a.shape_hash else 0.0
            area_score = 1.0 / (1.0 + abs(b.area - a.area))
            dist = abs(b.centroid[0] - a.centroid[0]) + abs(b.centroid[1] - a.centroid[1])
            iou = _bbox_iou(b.bbox, a.bbox)
            score = color_score + shape_score + area_score + iou - 0.12 * dist
            candidates.append((score, b.object_id, a.object_id))
    candidates.sort(reverse=True)
    mapping: dict[str, str] = {}
    used_after: set[str] = set()
    for score, before_id, after_id in candidates:
        if score < 0.25:
            continue
        if before_id in mapping or after_id in used_after:
            continue
        mapping[before_id] = after_id
        used_after.add(after_id)
    return mapping


def _objects_near_cells(objects: tuple[ObjectToken, ...], cells: Iterable[tuple[int, int]], *, radius: int = 1) -> tuple[str, ...]:
    interesting = tuple(cells)
    if not interesting:
        return ()
    touched: set[str] = set()
    for obj in objects:
        for r, c in interesting:
            if any(abs(orow - r) + abs(ocol - c) <= radius for orow, ocol in obj.cells[:256]):
                touched.add(obj.object_id)
                break
    return tuple(sorted(touched))


def compare_states(
    before: StructuredState,
    after: StructuredState,
    *,
    action: ActionName,
    reward: float = 0.0,
    score_delta: float = 0.0,
    terminated: bool = False,
    info: Mapping[str, Any] | None = None,
) -> TransitionRecord:
    grid_before = before.grid
    grid_after = after.grid
    if grid_before.shape == grid_after.shape:
        changed_mask = grid_before != grid_after
        changed_cells = int(np.count_nonzero(changed_mask))
        changed_coords = tuple((int(r), int(c)) for r, c in np.argwhere(changed_mask)[:512])
        changed_fraction = float(changed_cells) / float(max(grid_before.size, 1))
    else:
        changed_cells = int(max(grid_before.size, grid_after.size))
        changed_fraction = 1.0
        changed_coords = ()

    match = _match_objects(before.objects, after.objects)
    after_by_id = {obj.object_id: obj for obj in after.objects}
    before_by_id = {obj.object_id: obj for obj in before.objects}
    moved: list[ObjectMotion] = []
    for before_id, after_id in match.items():
        b = before_by_id[before_id]
        a = after_by_id[after_id]
        dr = int(round(a.centroid[0] - b.centroid[0]))
        dc = int(round(a.centroid[1] - b.centroid[1]))
        distance = abs(a.centroid[0] - b.centroid[0]) + abs(a.centroid[1] - b.centroid[1])
        if distance > 0.10 or b.bbox != a.bbox:
            moved.append(
                ObjectMotion(
                    before_id=before_id,
                    after_id=after_id,
                    before_signature=b.signature,
                    after_signature=a.signature,
                    color=b.color,
                    delta=(dr, dc),
                    distance=float(distance),
                )
            )
    appeared = tuple(sorted(set(after_by_id) - set(match.values())))
    disappeared = tuple(sorted(set(before_by_id) - set(match.keys())))

    target = action_target_to_grid_cell(action, before.frame.extras)
    touched_cells: list[tuple[int, int]] = list(changed_coords[:128])
    if target is not None:
        touched_cells.append(target)
    touched = sorted(
        set(_objects_near_cells(before.objects, touched_cells, radius=1))
        | set(_objects_near_cells(after.objects, touched_cells, radius=1))
    )

    delta = TransitionDelta(
        action=action,
        changed_cells=changed_cells,
        changed_fraction=changed_fraction,
        moved_objects=tuple(moved),
        appeared=appeared,
        disappeared=disappeared,
        touched_objects=tuple(touched),
        reward=float(reward),
        score_delta=float(score_delta),
        terminated=bool(terminated),
        info=info or {},
    )
    return TransitionRecord(before=before, after=after, delta=delta)

```

### `arcagi/scientist/features.py`

New: hashed online feature vectors for states, actions, and transition targets.

```python
"""Hashed numeric features used by online memory and world modeling."""

from __future__ import annotations

from hashlib import blake2b
from typing import Iterable

import numpy as np

from .types import ActionName, StructuredState, TransitionRecord, action_family, combined_progress_signal


def stable_index(text: str, modulo: int) -> int:
    digest = blake2b(text.encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little", signed=False) % modulo


def add_hash_feature(vec: np.ndarray, token: str, value: float = 1.0) -> None:
    idx = stable_index(token, int(vec.shape[0]))
    sign = 1.0 if stable_index("sign:" + token, 2) == 0 else -1.0
    vec[idx] += sign * float(value)


def normalize(vec: np.ndarray, *, eps: float = 1e-8) -> np.ndarray:
    norm = float(np.linalg.norm(vec))
    if norm <= eps:
        return vec.astype(np.float32, copy=False)
    return (vec / norm).astype(np.float32)


def cosine(a: np.ndarray, b: np.ndarray, *, eps: float = 1e-8) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= eps:
        return 0.0
    return float(np.dot(a, b) / denom)


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    sa, sb = set(a), set(b)
    if not sa and not sb:
        return 0.0
    return float(len(sa & sb)) / float(len(sa | sb))


def state_features(state: StructuredState, *, dim: int = 256) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    rows, cols = state.grid.shape
    vec[0] = float(rows) / 64.0
    vec[1] = float(cols) / 64.0
    vec[2] = float(len(state.objects)) / 32.0
    vec[3] = float(len(state.relations)) / 256.0
    vec[4] = float(state.dominant_color) / 32.0
    color_counts: dict[int, int] = {}
    for obj in state.objects:
        color_counts[obj.color] = color_counts.get(obj.color, 0) + 1
        add_hash_feature(vec, f"obj_color:c{obj.color}", 0.20)
        add_hash_feature(vec, f"obj_shape:{obj.shape_hash}", 0.10)
        add_hash_feature(vec, f"obj_area_bin:{min(obj.area // 2, 16)}", 0.15)
        add_hash_feature(vec, f"obj_bbox:{min(obj.height,16)}x{min(obj.width,16)}", 0.12)
        for tag in obj.role_tags:
            add_hash_feature(vec, f"role:{tag}:c{obj.color}", 0.15)
    for color, count in color_counts.items():
        add_hash_feature(vec, f"color_count:c{color}:{min(count, 8)}", 0.25)
    for rel in state.relations[:128]:
        add_hash_feature(vec, f"rel:{rel.relation}", 0.08)
    return normalize(vec)


def action_features(action: ActionName, *, dim: int = 64) -> np.ndarray:
    vec = np.zeros(dim, dtype=np.float32)
    family = action_family(action)
    add_hash_feature(vec, f"action:{family}", 1.0)
    add_hash_feature(vec, f"raw:{action.lower()}", 0.35)
    return normalize(vec)


def state_action_features(state: StructuredState, action: ActionName, *, dim: int = 320) -> np.ndarray:
    sdim = dim - 64
    return np.concatenate([state_features(state, dim=sdim), action_features(action, dim=64)]).astype(np.float32)


def transition_target_features(record: TransitionRecord) -> tuple[float, float]:
    """Return normalized online learning targets: reward and visible change."""

    reward = float(combined_progress_signal(record.delta.reward, record.delta.score_delta))
    change = min(1.0, float(record.delta.changed_fraction) * 4.0 + 0.1 * len(record.delta.moved_objects))
    return reward, change

```

### `arcagi/scientist/hypotheses.py`

New: falsifiable causal hypothesis engine with Bayesian evidence and diagnostic action generation.

```python
"""Online causal hypothesis induction, testing, and Bayesian scoring.

The key design is that rules are treated as falsifiable hypotheses, not fixed
controllers.  The engine creates compact rule objects from observed transitions,
updates evidence online, and exposes uncertainty to the planner so that the agent
can choose diagnostic experiments.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from math import log
from typing import Any, Mapping

import numpy as np

from .types import (
    ActionName,
    ObjectToken,
    StructuredState,
    TransitionRecord,
    action_delta,
    action_family,
    action_target_to_grid_cell,
    grid_cell_to_action_coordinates,
    is_interact_action,
    is_move_action,
    is_selector_action,
    make_targeted_action,
    parse_action_target,
)


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))


@dataclass
class Evidence:
    support: float = 0.0
    contradiction: float = 0.0
    trials: int = 0

    def update(self, *, support: float = 0.0, contradiction: float = 0.0) -> None:
        self.support += max(0.0, float(support))
        self.contradiction += max(0.0, float(contradiction))
        self.trials += 1

    @property
    def posterior(self) -> float:
        # Beta(1.2, 1.2) prior prevents early certainty.
        return float((self.support + 1.2) / (self.support + self.contradiction + 2.4))

    @property
    def confidence(self) -> float:
        total = self.support + self.contradiction
        return float(1.0 - np.exp(-0.25 * total))

    @property
    def entropy(self) -> float:
        p = _clip(self.posterior, 1e-6, 1.0 - 1e-6)
        return float(-(p * log(p) + (1.0 - p) * log(1.0 - p)))


@dataclass
class Hypothesis:
    hypothesis_id: str
    kind: str
    action_family: str
    params: dict[str, Any]
    description: str
    created_step: int
    evidence: Evidence = field(default_factory=Evidence)
    mdl_penalty: float = 0.0
    last_updated_step: int = 0

    @property
    def posterior(self) -> float:
        return _clip(self.evidence.posterior - self.mdl_penalty, 0.01, 0.99)

    @property
    def uncertainty(self) -> float:
        return self.evidence.entropy * (1.0 - min(self.evidence.confidence, 0.95))

    @property
    def utility_prior(self) -> float:
        if self.kind.startswith("reward") or self.kind.startswith("goal"):
            return 1.5
        if self.kind.startswith("action_moves"):
            return 0.65
        if self.kind.startswith("targeted"):
            return 0.55
        if self.kind.startswith("mode"):
            return 0.40
        return 0.25

    def observe(self, *, supported: bool, strength: float = 1.0, step: int = 0) -> None:
        if supported:
            self.evidence.update(support=strength)
        else:
            self.evidence.update(contradiction=strength)
        self.last_updated_step = step

    def as_tokens(self) -> tuple[str, ...]:
        tokens = [f"hyp:{self.kind}", f"af:{self.action_family}"]
        for key, value in sorted(self.params.items()):
            if isinstance(value, (str, int, float, bool)):
                tokens.append(f"{key}:{value}")
            elif isinstance(value, tuple):
                tokens.append(f"{key}:{','.join(map(str, value[:4]))}")
        tokens.append(f"p:{round(self.posterior, 2)}")
        return tuple(tokens)


@dataclass(frozen=True)
class HypothesisActionScore:
    expected_reward: float
    expected_change: float
    information_gain: float
    risk: float
    posterior_mass: float
    rationale: tuple[str, ...]


class HypothesisEngine:
    """Maintains and updates executable online theories."""

    def __init__(self, *, max_hypotheses: int = 512) -> None:
        self.max_hypotheses = int(max_hypotheses)
        self.hypotheses: dict[str, Hypothesis] = {}
        self.transition_count = 0
        self.positive_examples = 0
        self.recent_actions: list[ActionName] = []

    def reset_episode(self) -> None:
        self.hypotheses.clear()
        self.transition_count = 0
        self.positive_examples = 0
        self.recent_actions.clear()

    def _id(self, kind: str, action_fam: str, params: Mapping[str, Any]) -> str:
        payload = ";".join(f"{k}={params[k]}" for k in sorted(params))
        return f"{kind}|{action_fam}|{payload}"

    def _get_or_create(
        self,
        *,
        kind: str,
        action_fam: str,
        params: Mapping[str, Any],
        description: str,
        step: int,
        mdl_penalty: float = 0.0,
    ) -> Hypothesis:
        hid = self._id(kind, action_fam, params)
        hyp = self.hypotheses.get(hid)
        if hyp is not None:
            return hyp
        hyp = Hypothesis(
            hypothesis_id=hid,
            kind=kind,
            action_family=action_fam,
            params=dict(params),
            description=description,
            created_step=step,
            mdl_penalty=float(mdl_penalty),
        )
        self.hypotheses[hid] = hyp
        self._prune_if_needed()
        return hyp

    def _prune_if_needed(self) -> None:
        if len(self.hypotheses) <= self.max_hypotheses:
            return
        ranked = sorted(
            self.hypotheses.values(),
            key=lambda h: (h.posterior * h.evidence.confidence * h.utility_prior, -h.evidence.contradiction),
            reverse=True,
        )
        keep = {h.hypothesis_id for h in ranked[: self.max_hypotheses]}
        for hid in list(self.hypotheses):
            if hid not in keep:
                del self.hypotheses[hid]

    def observe_transition(self, record: TransitionRecord) -> None:
        self.transition_count += 1
        step = record.before.step_index
        fam = action_family(record.action)
        if record.delta.is_positive:
            self.positive_examples += 1
        self.recent_actions.append(record.action)
        if len(self.recent_actions) > 16:
            self.recent_actions.pop(0)

        # First update existing hypotheses against the new observation.
        for hyp in list(self.hypotheses.values()):
            self._update_hypothesis(hyp, record, fam=fam, step=step)

        # Then induce new explanations from effects.  Induction after updating
        # avoids instantly rewarding a hypothesis for the event that created it.
        self._induce_action_effects(record, fam=fam, step=step)
        self._induce_targeted_effects(record, fam=fam, step=step)
        self._induce_reward_hypotheses(record, fam=fam, step=step)
        self._induce_mode_hypotheses(record, fam=fam, step=step)

    def _update_hypothesis(self, hyp: Hypothesis, record: TransitionRecord, *, fam: str, step: int) -> None:
        if hyp.kind == "action_moves_object":
            if fam != hyp.action_family:
                return
            target_sig = str(hyp.params.get("object_signature"))
            expected_delta = tuple(hyp.params.get("delta", (0, 0)))
            supported = any(
                motion.before_signature == target_sig and tuple(motion.delta) == expected_delta
                for motion in record.delta.moved_objects
            )
            visible_relevant = any(motion.before_signature == target_sig for motion in record.delta.moved_objects)
            if supported:
                hyp.observe(supported=True, strength=1.0 + 0.25 * record.delta.changed_fraction, step=step)
            elif visible_relevant or record.delta.has_visible_effect:
                hyp.observe(supported=False, strength=0.35, step=step)

        elif hyp.kind == "targeted_action_changes_object":
            if fam != hyp.action_family:
                return
            target = action_target_to_grid_cell(record.action, record.before.frame.extras)
            if target is None:
                return
            expected_color = int(hyp.params.get("color", -999))
            changed = record.delta.changed_cells > 0
            touched_color = any(
                (record.before.object_by_id(oid) and record.before.object_by_id(oid).color == expected_color)
                or (record.after.object_by_id(oid) and record.after.object_by_id(oid).color == expected_color)
                for oid in record.delta.touched_objects
            )
            if changed and touched_color:
                hyp.observe(supported=True, strength=1.0, step=step)
            elif touched_color:
                hyp.observe(supported=False, strength=0.5, step=step)

        elif hyp.kind == "reward_when_touch_color":
            color = int(hyp.params.get("color", -999))
            relevant = self._action_touches_color(record, color) or self._state_contains_color(record.after, color)
            if not relevant:
                return
            if record.delta.is_positive:
                hyp.observe(supported=True, strength=1.5 + max(record.reward, 0.0), step=step)
            elif fam == hyp.action_family or is_interact_action(record.action) or is_move_action(record.action):
                hyp.observe(supported=False, strength=0.18, step=step)

        elif hyp.kind == "reward_when_state_has_color":
            color = int(hyp.params.get("color", -999))
            relevant = self._state_contains_color(record.after, color)
            if relevant and record.delta.is_positive:
                hyp.observe(supported=True, strength=1.2, step=step)
            elif relevant and (fam == hyp.action_family or record.delta.has_visible_effect):
                hyp.observe(supported=False, strength=0.12, step=step)

        elif hyp.kind == "mode_action_changes_dynamics":
            if fam == hyp.action_family and record.delta.has_visible_effect:
                hyp.observe(supported=True, strength=0.7, step=step)
            elif fam == hyp.action_family:
                hyp.observe(supported=False, strength=0.25, step=step)

    def _induce_action_effects(self, record: TransitionRecord, *, fam: str, step: int) -> None:
        if not record.delta.moved_objects:
            return
        for motion in record.delta.moved_objects:
            params = {
                "object_signature": motion.before_signature,
                "color": motion.color,
                "delta": tuple(motion.delta),
            }
            desc = f"action family {fam} moves object {motion.before_signature} by {motion.delta}"
            hyp = self._get_or_create(
                kind="action_moves_object",
                action_fam=fam,
                params=params,
                description=desc,
                step=step,
                mdl_penalty=0.02,
            )
            # The creating event is weakly credited because the same transition is
            # not a replication.  Repeated support is required for high posterior.
            hyp.observe(supported=True, strength=0.35, step=step)

    def _induce_targeted_effects(self, record: TransitionRecord, *, fam: str, step: int) -> None:
        if action_target_to_grid_cell(record.action, record.before.frame.extras) is None:
            return
        if not record.delta.has_visible_effect:
            return
        for oid in record.delta.touched_objects:
            obj = record.before.object_by_id(oid) or record.after.object_by_id(oid)
            if obj is None:
                continue
            params = {"color": obj.color, "signature": obj.signature}
            hyp = self._get_or_create(
                kind="targeted_action_changes_object",
                action_fam=fam,
                params=params,
                description=f"targeted {fam} changes object color c{obj.color}",
                step=step,
                mdl_penalty=0.03,
            )
            hyp.observe(supported=True, strength=0.45, step=step)

    def _induce_reward_hypotheses(self, record: TransitionRecord, *, fam: str, step: int) -> None:
        if not record.delta.is_positive:
            return
        touched_objects = [record.before.object_by_id(oid) or record.after.object_by_id(oid) for oid in record.delta.touched_objects]
        moved_after = [record.after.object_by_id(motion.after_id) for motion in record.delta.moved_objects]
        candidate_objects = [obj for obj in [*touched_objects, *moved_after, *record.after.objects[:6]] if obj is not None]
        seen_colors: set[int] = set()
        for obj in candidate_objects:
            if obj.color in seen_colors:
                continue
            seen_colors.add(obj.color)
            params = {"color": obj.color, "signature": obj.signature}
            hyp = self._get_or_create(
                kind="reward_when_touch_color",
                action_fam=fam,
                params=params,
                description=f"reward/progress follows contact or interaction with color c{obj.color}",
                step=step,
                mdl_penalty=0.04,
            )
            hyp.observe(supported=True, strength=1.25 + max(record.reward, record.delta.score_delta, 0.0), step=step)

            state_hyp = self._get_or_create(
                kind="reward_when_state_has_color",
                action_fam=fam,
                params={"color": obj.color},
                description=f"reward/progress occurs when state contains color c{obj.color}",
                step=step,
                mdl_penalty=0.08,
            )
            state_hyp.observe(supported=True, strength=0.8, step=step)

    def _induce_mode_hypotheses(self, record: TransitionRecord, *, fam: str, step: int) -> None:
        if not is_selector_action(record.action):
            return
        before_actions = set(record.before.available_actions)
        after_actions = set(record.after.available_actions)
        action_set_changed = before_actions != after_actions
        if record.delta.has_visible_effect or action_set_changed:
            params = {"changed_action_set": action_set_changed, "visible": record.delta.has_visible_effect}
            hyp = self._get_or_create(
                kind="mode_action_changes_dynamics",
                action_fam=fam,
                params=params,
                description=f"selector-like action {fam} changes latent mode or controls",
                step=step,
                mdl_penalty=0.05,
            )
            hyp.observe(supported=True, strength=0.8, step=step)

    def _state_contains_color(self, state: StructuredState, color: int) -> bool:
        return any(obj.color == color for obj in state.objects)

    def _action_touches_color(self, record: TransitionRecord, color: int) -> bool:
        for oid in record.delta.touched_objects:
            before = record.before.object_by_id(oid)
            after = record.after.object_by_id(oid)
            if (before is not None and before.color == color) or (after is not None and after.color == color):
                return True
        return False

    def score_action(self, state: StructuredState, action: ActionName) -> HypothesisActionScore:
        fam = action_family(action)
        expected_reward = 0.0
        expected_change = 0.0
        info_gain = 0.0
        risk = 0.0
        posterior_mass = 0.0
        rationale: list[str] = []
        target = action_target_to_grid_cell(action, state.frame.extras)

        for hyp in self.hypotheses.values():
            p = hyp.posterior
            relevant = fam == hyp.action_family
            if hyp.kind == "action_moves_object" and relevant:
                expected_change += p * 0.45
                posterior_mass += p
                if hyp.uncertainty > 0.05:
                    info_gain += hyp.uncertainty * 0.60
                    rationale.extend(("test", "movement", fam))

            elif hyp.kind == "targeted_action_changes_object" and relevant:
                color = int(hyp.params.get("color", -999))
                if target is not None:
                    near_color = any(_obj_near_target(obj, target, radius=1) and obj.color == color for obj in state.objects)
                    if near_color:
                        expected_change += p * 0.55
                        info_gain += hyp.uncertainty * 0.75
                        posterior_mass += p
                        rationale.extend(("test", "targeted_change", f"c{color}"))
                else:
                    info_gain += hyp.uncertainty * 0.30

            elif hyp.kind == "reward_when_touch_color":
                color = int(hyp.params.get("color", -999))
                color_present = any(obj.color == color for obj in state.objects)
                if not color_present:
                    continue
                if is_move_action(action) or is_interact_action(action) or fam == hyp.action_family:
                    expected_reward += p * hyp.utility_prior * 0.35
                    posterior_mass += p
                    if hyp.uncertainty > 0.04:
                        info_gain += hyp.uncertainty * 0.45
                    rationale.extend(("pursue", f"c{color}"))

            elif hyp.kind == "reward_when_state_has_color":
                color = int(hyp.params.get("color", -999))
                if any(obj.color == color for obj in state.objects):
                    expected_reward += p * 0.12
                    posterior_mass += p * 0.25

            elif hyp.kind == "mode_action_changes_dynamics" and relevant:
                expected_change += p * 0.20
                info_gain += hyp.uncertainty * 0.65
                posterior_mass += p
                rationale.extend(("test", "mode"))

            if relevant and hyp.posterior < 0.20 and hyp.evidence.contradiction > hyp.evidence.support + 1:
                risk += 0.10

        # Before enough evidence exists, keep raw exploration live.  This avoids
        # prematurely collapsing into a single movement loop.
        if self.transition_count < 12:
            if is_move_action(action) or is_interact_action(action) or is_selector_action(action):
                info_gain += 0.25 / (1.0 + self.transition_count)
            if target is not None:
                info_gain += 0.08

        return HypothesisActionScore(
            expected_reward=float(expected_reward),
            expected_change=float(expected_change),
            information_gain=float(info_gain),
            risk=float(risk),
            posterior_mass=float(posterior_mass),
            rationale=tuple(rationale[:12]),
        )

    def diagnostic_actions(self, state: StructuredState, legal_actions: tuple[ActionName, ...], *, limit: int = 32) -> tuple[ActionName, ...]:
        actions: list[ActionName] = []
        base_by_family = {action_family(a): a for a in legal_actions}
        uncertain = sorted(self.hypotheses.values(), key=lambda h: h.uncertainty * h.utility_prior, reverse=True)
        for hyp in uncertain[:24]:
            if hyp.action_family in base_by_family:
                base = base_by_family[hyp.action_family]
                if hyp.kind == "targeted_action_changes_object":
                    color = int(hyp.params.get("color", -999))
                    for obj in state.objects:
                        if obj.color == color:
                            r, c = obj.center_cell
                            tr, tc = grid_cell_to_action_coordinates(r, c, state.frame.extras)
                            actions.append(make_targeted_action(base, tr, tc))
                else:
                    actions.append(base)
            if len(actions) >= limit:
                break
        # Always include targeted probes of object centers if a click-like action exists.
        click_base = None
        for legal in legal_actions:
            if action_family(legal) in {"click", "action6", "interact_at", "select_at"}:
                click_base = legal
                break
        if click_base is not None:
            for obj in state.objects[:16]:
                r, c = obj.center_cell
                tr, tc = grid_cell_to_action_coordinates(r, c, state.frame.extras)
                actions.append(make_targeted_action(click_base, tr, tc))
                if len(actions) >= limit:
                    break
        return tuple(dict.fromkeys(actions[:limit]))


    def color_progress_priors(self) -> dict[int, float]:
        """Return online evidence that interacting with a color may advance the task."""

        priors: dict[int, float] = {}
        for hyp in self.hypotheses.values():
            if hyp.kind not in {"reward_when_touch_color", "reward_when_state_has_color"}:
                continue
            try:
                color = int(hyp.params.get("color"))
            except Exception:
                continue
            weight = hyp.posterior * hyp.utility_prior * (0.5 + hyp.evidence.confidence)
            priors[color] = max(priors.get(color, 0.0), float(weight))
        return priors

    def controlled_object_colors(self) -> dict[int, float]:
        """Return colors that evidence suggests are moved by primitive actions."""

        colors: dict[int, float] = {}
        for hyp in self.hypotheses.values():
            if hyp.kind != "action_moves_object":
                continue
            try:
                color = int(hyp.params.get("color"))
            except Exception:
                continue
            colors[color] = max(colors.get(color, 0.0), float(hyp.posterior * (0.5 + hyp.evidence.confidence)))
        return colors

    def credible_hypotheses(self, *, min_posterior: float = 0.55, limit: int = 16) -> tuple[Hypothesis, ...]:
        ranked = sorted(
            (h for h in self.hypotheses.values() if h.posterior >= min_posterior),
            key=lambda h: h.posterior * h.evidence.confidence * h.utility_prior,
            reverse=True,
        )
        return tuple(ranked[:limit])

    def uncertain_hypotheses(self, *, limit: int = 12) -> tuple[Hypothesis, ...]:
        ranked = sorted(self.hypotheses.values(), key=lambda h: h.uncertainty * h.utility_prior, reverse=True)
        return tuple(ranked[:limit])


def _obj_near_target(obj: ObjectToken, target: tuple[int, int], *, radius: int) -> bool:
    r, c = target
    orow, ocol = obj.center_cell
    if abs(orow - r) + abs(ocol - c) <= radius:
        return True
    r0, c0, r1, c1 = obj.bbox
    return (r0 - radius) <= r <= (r1 + radius) and (c0 - radius) <= c <= (c1 + radius)

```

### `arcagi/scientist/world_model.py`

New: tiny online bootstrap ensemble for reward/change prediction and epistemic uncertainty.

```python
"""Small online world model with uncertainty via bootstrap ensemble.

The model is intentionally tiny: linear heads over hashed structured features,
updated after every transition.  It is not meant to solve ARC-AGI-3 alone; it
supplies calibrated uncertainty and weak reward/change predictions to the
experiment planner.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .features import state_action_features, transition_target_features
from .types import ActionName, StructuredState, TransitionRecord


@dataclass(frozen=True)
class WorldPrediction:
    reward_mean: float
    reward_uncertainty: float
    change_mean: float
    change_uncertainty: float

    @property
    def total_uncertainty(self) -> float:
        return float(self.reward_uncertainty + 0.5 * self.change_uncertainty)


class OnlineWorldModel:
    """Bootstrap SGD ensemble for in-episode reward/change prediction."""

    def __init__(
        self,
        *,
        feature_dim: int = 320,
        ensemble_size: int = 7,
        learning_rate: float = 0.08,
        weight_decay: float = 1e-4,
        seed: int = 0,
    ) -> None:
        self.feature_dim = int(feature_dim)
        self.ensemble_size = int(ensemble_size)
        self.learning_rate = float(learning_rate)
        self.weight_decay = float(weight_decay)
        self.rng = np.random.default_rng(seed)
        scale = 0.015
        self.reward_w = self.rng.normal(0.0, scale, size=(self.ensemble_size, self.feature_dim)).astype(np.float32)
        self.change_w = self.rng.normal(0.0, scale, size=(self.ensemble_size, self.feature_dim)).astype(np.float32)
        self.updates = 0

    def reset_episode(self, *, keep_weights: bool = True) -> None:
        self.updates = 0
        if not keep_weights:
            scale = 0.015
            self.reward_w[:] = self.rng.normal(0.0, scale, size=self.reward_w.shape)
            self.change_w[:] = self.rng.normal(0.0, scale, size=self.change_w.shape)

    def predict(self, state: StructuredState, action: ActionName) -> WorldPrediction:
        x = state_action_features(state, action, dim=self.feature_dim)
        reward_preds = self.reward_w @ x
        change_preds = self.change_w @ x
        reward_mean = float(np.mean(reward_preds))
        change_mean = float(np.mean(change_preds))
        return WorldPrediction(
            reward_mean=float(np.tanh(reward_mean)),
            reward_uncertainty=float(np.var(reward_preds) + 1.0 / np.sqrt(max(self.updates, 1))),
            change_mean=float(1.0 / (1.0 + np.exp(-change_mean))),
            change_uncertainty=float(np.var(change_preds) + 0.5 / np.sqrt(max(self.updates, 1))),
        )

    def update(self, record: TransitionRecord) -> float:
        x = state_action_features(record.before, record.action, dim=self.feature_dim)
        reward_target, change_target = transition_target_features(record)
        reward_target = float(np.tanh(reward_target))
        change_target = float(change_target)
        losses: list[float] = []
        for idx in range(self.ensemble_size):
            # Bootstrap mask keeps member disagreement meaningful.
            if self.rng.random() < 0.80:
                pred = float(self.reward_w[idx] @ x)
                err = _clip(reward_target - pred, -2.0, 2.0)
                self.reward_w[idx] *= 1.0 - self.learning_rate * self.weight_decay
                self.reward_w[idx] += self.learning_rate * err * x
                losses.append(err * err)
            if self.rng.random() < 0.80:
                pred = float(self.change_w[idx] @ x)
                err = _clip(change_target - pred, -2.0, 2.0)
                self.change_w[idx] *= 1.0 - self.learning_rate * self.weight_decay
                self.change_w[idx] += self.learning_rate * err * x
                losses.append(err * err)
        self.updates += 1
        return float(np.mean(losses)) if losses else 0.0


def _clip(value: float, lo: float, hi: float) -> float:
    return float(max(lo, min(hi, value)))

```

### `arcagi/scientist/memory.py`

New: surprise-weighted episodic memory and successful option memory.

```python
"""Surprise-weighted episodic and option memory for online adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .features import cosine, jaccard, state_features
from .types import ActionName, StructuredState, TransitionRecord, combined_progress_signal


@dataclass
class MemoryItem:
    step: int
    before_fingerprint: str
    after_fingerprint: str
    action: ActionName
    reward: float
    surprise: float
    state_vector: np.ndarray
    language_tokens: frozenset[str]
    note: str = ""


@dataclass
class OptionItem:
    action_sequence: tuple[ActionName, ...]
    state_vector: np.ndarray
    language_tokens: frozenset[str]
    reward: float
    uses: int = 0
    successes: int = 0

    @property
    def value(self) -> float:
        return float((self.reward + self.successes) / max(self.uses + 1, 1))


class EpisodicMemory:
    def __init__(self, *, capacity: int = 2048, feature_dim: int = 256) -> None:
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.items: list[MemoryItem] = []
        self.options: list[OptionItem] = []
        self.recent_actions: list[ActionName] = []

    def reset_episode(self) -> None:
        self.items.clear()
        self.options.clear()
        self.recent_actions.clear()

    def write_transition(self, record: TransitionRecord, *, surprise: float, language_tokens: Iterable[str]) -> None:
        self.recent_actions.append(record.action)
        if len(self.recent_actions) > 12:
            self.recent_actions.pop(0)

        reward = float(combined_progress_signal(record.reward, record.delta.score_delta))
        should_write = surprise > 0.18 or abs(reward) > 1e-8 or record.delta.has_visible_effect
        if not should_write:
            return
        item = MemoryItem(
            step=record.step_index,
            before_fingerprint=record.before.abstract_fingerprint,
            after_fingerprint=record.after.abstract_fingerprint,
            action=record.action,
            reward=reward,
            surprise=float(surprise),
            state_vector=state_features(record.before, dim=self.feature_dim),
            language_tokens=frozenset(language_tokens),
            note=_transition_note(record),
        )
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity :]

        if reward > 0.0:
            seq = tuple(self.recent_actions[-6:])
            self.options.append(
                OptionItem(
                    action_sequence=seq,
                    state_vector=state_features(record.before, dim=self.feature_dim),
                    language_tokens=frozenset(language_tokens),
                    reward=reward,
                    uses=1,
                    successes=1,
                )
            )
            if len(self.options) > self.capacity // 4:
                self.options = sorted(self.options, key=lambda o: o.value, reverse=True)[: self.capacity // 4]

    def retrieve(self, state: StructuredState, language_tokens: Iterable[str], *, k: int = 8) -> tuple[MemoryItem, ...]:
        if not self.items:
            return ()
        vec = state_features(state, dim=self.feature_dim)
        tokens = frozenset(language_tokens)
        scored: list[tuple[float, MemoryItem]] = []
        for item in self.items:
            score = 0.65 * cosine(vec, item.state_vector) + 0.25 * jaccard(tokens, item.language_tokens)
            score += 0.15 * max(item.reward, 0.0) + 0.05 * item.surprise
            scored.append((float(score), item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return tuple(item for _, item in scored[:k])

    def retrieve_options(self, state: StructuredState, language_tokens: Iterable[str], *, k: int = 4) -> tuple[OptionItem, ...]:
        if not self.options:
            return ()
        vec = state_features(state, dim=self.feature_dim)
        tokens = frozenset(language_tokens)
        scored: list[tuple[float, OptionItem]] = []
        for option in self.options:
            score = 0.55 * cosine(vec, option.state_vector) + 0.20 * jaccard(tokens, option.language_tokens)
            score += 0.35 * option.value
            scored.append((float(score), option))
        scored.sort(key=lambda x: x[0], reverse=True)
        return tuple(item for _, item in scored[:k])

    def action_memory_bonus(self, state: StructuredState, action: ActionName, language_tokens: Iterable[str]) -> float:
        bonus = 0.0
        for item in self.retrieve(state, language_tokens, k=8):
            if item.action == action:
                bonus += 0.08 * item.surprise + 0.12 * max(item.reward, 0.0)
        for option in self.retrieve_options(state, language_tokens, k=4):
            if option.action_sequence and option.action_sequence[0] == action:
                bonus += 0.25 * option.value
        return float(bonus)


def _transition_note(record: TransitionRecord) -> str:
    parts = [f"action={record.action}"]
    if record.delta.moved_objects:
        parts.append(f"moved={len(record.delta.moved_objects)}")
    if record.delta.changed_cells:
        parts.append(f"changed={record.delta.changed_cells}")
    if record.reward:
        parts.append(f"reward={record.reward:.3f}")
    if record.delta.score_delta:
        parts.append(f"score_delta={record.delta.score_delta:.3f}")
    return ";".join(parts)

```

### `arcagi/scientist/language.py`

New: grounded internal language tokens, belief statements, and questions.

```python
"""Grounded internal language for hypotheses, questions, and memory keys.

This is not a chatbot or hidden prompt.  It is a compact controlled vocabulary
whose tokens are grounded in observations, transition evidence, and rule objects.
The planner consumes the same tokens for retrieval and diagnostic action scoring.
"""

from __future__ import annotations

from .hypotheses import Hypothesis, HypothesisEngine
from .types import ActionName, StructuredState, action_family


class GroundedLanguage:
    def state_tokens(self, state: StructuredState) -> tuple[str, ...]:
        tokens: list[str] = [
            f"grid={state.grid.shape[0]}x{state.grid.shape[1]}",
            f"objects={len(state.objects)}",
            f"dominant=c{state.dominant_color}",
        ]
        for obj in state.objects[:24]:
            tokens.append(f"obj:c{obj.color}:area{min(obj.area, 20)}")
            for tag in obj.role_tags[:3]:
                tokens.append(f"role:{tag}:c{obj.color}")
        for rel in state.relations[:32]:
            tokens.append(f"rel:{rel.relation}")
        return tuple(tokens)

    def hypothesis_tokens(self, hypotheses: tuple[Hypothesis, ...]) -> tuple[str, ...]:
        tokens: list[str] = []
        for hyp in hypotheses:
            tokens.extend(hyp.as_tokens())
        return tuple(tokens[:96])

    def memory_tokens(self, state: StructuredState, engine: HypothesisEngine) -> tuple[str, ...]:
        return tuple(dict.fromkeys((*self.state_tokens(state), *self.hypothesis_tokens(engine.credible_hypotheses(limit=8)))))

    def belief_sentences(self, engine: HypothesisEngine, *, limit: int = 8) -> tuple[str, ...]:
        sentences: list[str] = []
        for hyp in engine.credible_hypotheses(limit=limit):
            p = round(hyp.posterior, 2)
            if hyp.kind == "action_moves_object":
                sentences.append(
                    f"belief p={p}: action {hyp.action_family} moves object {hyp.params.get('object_signature')} by {hyp.params.get('delta')}"
                )
            elif hyp.kind.startswith("reward"):
                sentences.append(f"belief p={p}: progress may depend on color c{hyp.params.get('color')}")
            elif hyp.kind.startswith("mode"):
                sentences.append(f"belief p={p}: action {hyp.action_family} may change latent mode")
            else:
                sentences.append(f"belief p={p}: {hyp.description}")
        return tuple(sentences)

    def questions(self, engine: HypothesisEngine, *, limit: int = 6) -> tuple[str, ...]:
        questions: list[str] = []
        for hyp in engine.uncertain_hypotheses(limit=limit):
            p = round(hyp.posterior, 2)
            if hyp.kind == "action_moves_object":
                questions.append(f"question p={p}: test whether {hyp.action_family} consistently causes delta {hyp.params.get('delta')}")
            elif hyp.kind == "targeted_action_changes_object":
                questions.append(f"question p={p}: test targeted {hyp.action_family} on color c{hyp.params.get('color')}")
            elif hyp.kind.startswith("reward"):
                questions.append(f"question p={p}: test whether color c{hyp.params.get('color')} predicts progress")
            else:
                questions.append(f"question p={p}: falsify {hyp.kind} via action {hyp.action_family}")
        return tuple(questions)

    def plan_sentence(self, action: ActionName, *, components: dict[str, float], reason: str) -> str:
        parts = [f"plan: do {action_family(action)}"]
        if reason:
            parts.append(f"because {reason}")
        for key in ("expected_reward", "information_gain", "novelty", "world_uncertainty"):
            if key in components:
                parts.append(f"{key}={components[key]:.3f}")
        return "; ".join(parts)

```

### `arcagi/scientist/planner.py`

New: information-gain planner with learned navigation, blocked-cell inference, contact probes, novelty, and memory reuse.

```python
"""Information-gain planner for the scientist agent."""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Iterable

import numpy as np

from .hypotheses import HypothesisEngine
from .language import GroundedLanguage
from .memory import EpisodicMemory
from .types import (
    ActionDecision,
    ActionName,
    ObjectToken,
    StructuredState,
    TransitionRecord,
    action_delta,
    action_family,
    combined_progress_signal,
    grid_cell_to_action_coordinates,
    is_interact_action,
    is_move_action,
    make_targeted_action,
    parse_action_target,
)
from .world_model import OnlineWorldModel


@dataclass(frozen=True)
class PlannerConfig:
    reward_weight: float = 2.4
    information_weight: float = 1.7
    novelty_weight: float = 0.85
    change_weight: float = 0.45
    uncertainty_weight: float = 0.35
    memory_weight: float = 0.75
    spatial_goal_weight: float = 1.15
    navigation_weight: float = 2.10
    repeat_penalty: float = 0.55
    max_candidates: int = 96
    random_tie_noise: float = 1e-4


class ScientistPlanner:
    def __init__(self, *, config: PlannerConfig | None = None, seed: int = 0) -> None:
        self.config = config or PlannerConfig()
        self.rng = np.random.default_rng(seed)
        self.state_action_visits: dict[tuple[str, ActionName], int] = defaultdict(int)
        self.action_visits: dict[ActionName, int] = defaultdict(int)
        self.last_action: ActionName | None = None
        self.last_state_fp: str | None = None
        self.stall_count = 0
        self.ineffective_actions: dict[tuple[str, ActionName], int] = defaultdict(int)
        self.blocked_cells: set[tuple[int, int]] = set()
        self.visited_actor_positions: set[tuple[int, int]] = set()
        self.nonproductive_colors: dict[int, int] = defaultdict(int)
        self.productive_colors: dict[int, int] = defaultdict(int)
        self.pending_interactions = 0

    def reset_episode(self) -> None:
        self.state_action_visits.clear()
        self.action_visits.clear()
        self.last_action = None
        self.last_state_fp = None
        self.stall_count = 0
        self.ineffective_actions.clear()
        self.blocked_cells.clear()
        self.visited_actor_positions.clear()
        self.nonproductive_colors.clear()
        self.productive_colors.clear()
        self.pending_interactions = 0

    def notify_transition(
        self,
        *,
        changed: bool,
        record: TransitionRecord | None = None,
        engine: HypothesisEngine | None = None,
    ) -> None:
        if changed:
            self.stall_count = 0
        else:
            self.stall_count += 1

        if record is None:
            return
        if engine is not None:
            self._update_contact_statistics(record, engine)
        key = (record.before.exact_fingerprint, record.action)
        if changed:
            self.ineffective_actions.pop(key, None)
            return
        self.ineffective_actions[key] += 1
        delta = action_delta(record.action)
        if delta is None or engine is None:
            return
        actor = self._controlled_object(record.before, engine)
        if actor is None:
            return
        ar, ac = actor.center_cell
        nr = ar + delta[0]
        nc = ac + delta[1]
        rows, cols = record.before.grid.shape
        if 0 <= nr < rows and 0 <= nc < cols:
            self.blocked_cells.add((nr, nc))

    def choose_action(
        self,
        state: StructuredState,
        *,
        engine: HypothesisEngine,
        world_model: OnlineWorldModel,
        memory: EpisodicMemory,
        language: GroundedLanguage,
    ) -> ActionDecision:
        candidates = self.candidate_actions(state, engine=engine)
        if not candidates:
            raise RuntimeError("No legal actions are available for ScientistPlanner")

        lang_tokens = language.memory_tokens(state, engine)
        navigation_action = self._navigation_next_action(state, engine, candidates)
        scored: list[tuple[float, ActionName, dict[str, float], tuple[str, ...]]] = []
        for action in candidates:
            hyp_score = engine.score_action(state, action)
            world = world_model.predict(state, action)
            novelty = self._novelty(state, action)
            memory_bonus = memory.action_memory_bonus(state, action, lang_tokens)
            repeat = self._repeat_penalty(state, action)
            spatial_goal = self._spatial_goal_value(state, action, engine)
            navigation_goal = 1.0 if navigation_action == action else 0.0
            components = {
                "expected_reward": hyp_score.expected_reward + world.reward_mean,
                "information_gain": hyp_score.information_gain,
                "novelty": novelty,
                "expected_change": hyp_score.expected_change + world.change_mean,
                "world_uncertainty": world.total_uncertainty,
                "memory_bonus": memory_bonus,
                "spatial_goal": spatial_goal,
                "navigation_goal": navigation_goal,
                "risk": hyp_score.risk,
                "repeat_penalty": repeat,
                "posterior_mass": hyp_score.posterior_mass,
            }
            score = (
                self.config.reward_weight * components["expected_reward"]
                + self.config.information_weight * components["information_gain"]
                + self.config.novelty_weight * components["novelty"]
                + self.config.change_weight * components["expected_change"]
                + self.config.uncertainty_weight * components["world_uncertainty"]
                + self.config.memory_weight * components["memory_bonus"]
                + self.config.spatial_goal_weight * components["spatial_goal"]
                + self.config.navigation_weight * components["navigation_goal"]
                - components["risk"]
                - repeat
            )
            score += float(self.rng.normal(0.0, self.config.random_tie_noise))
            scored.append((float(score), action, components, hyp_score.rationale))

        scored.sort(key=lambda item: item[0], reverse=True)
        best_score, best_action, best_components, rationale = scored[0]
        self.state_action_visits[(state.abstract_fingerprint, best_action)] += 1
        self.action_visits[best_action] += 1
        if is_interact_action(best_action) and self.pending_interactions > 0:
            self.pending_interactions -= 1
        self.last_action = best_action
        self.last_state_fp = state.exact_fingerprint
        reason = " ".join(rationale[:5]) if rationale else "maximize online experiment value"
        plan = language.plan_sentence(best_action, components=best_components, reason=reason)
        return ActionDecision(
            action=best_action,
            score=best_score,
            components=best_components,
            language=(*language.belief_sentences(engine, limit=4), *language.questions(engine, limit=3), plan),
            candidate_count=len(candidates),
            chosen_reason=reason,
        )

    def candidate_actions(self, state: StructuredState, *, engine: HypothesisEngine) -> tuple[ActionName, ...]:
        legal = tuple(state.available_actions)
        if not legal:
            legal = ("up", "down", "left", "right", "interact", "click")
        candidates: list[ActionName] = list(legal)
        candidates.extend(engine.diagnostic_actions(state, legal, limit=48))

        # Expand click-like actions to object centers and a small frontier set.  The
        # official ARC docs expose ACTION6 as a coordinate-bearing action in the
        # direct API, so this representation is adapter-friendly.
        click_bases = [a for a in legal if action_family(a) in {"click", "action6", "interact_at", "select_at"}]
        if click_bases:
            base = click_bases[0]
            for obj in sorted(state.objects, key=lambda o: ("background_candidate" in o.role_tags, -o.area))[:24]:
                r, c = obj.center_cell
                tr, tc = grid_cell_to_action_coordinates(r, c, state.frame.extras)
                candidates.append(make_targeted_action(base, tr, tc))
            rows, cols = state.grid.shape
            for r, c in ((0, 0), (0, cols - 1), (rows - 1, 0), (rows - 1, cols - 1), (rows // 2, cols // 2)):
                tr, tc = grid_cell_to_action_coordinates(r, c, state.frame.extras)
                candidates.append(make_targeted_action(base, tr, tc))

        # If all recent actions are stalling, force broad coverage before repeating.
        if self.stall_count >= 3:
            candidates = sorted(candidates, key=lambda a: (self.action_visits[a], self.state_action_visits[(state.abstract_fingerprint, a)]))

        deduped = tuple(dict.fromkeys(candidates))
        return deduped[: self.config.max_candidates]



    def _navigation_next_action(
        self,
        state: StructuredState,
        engine: HypothesisEngine,
        legal_actions: tuple[ActionName, ...],
    ) -> ActionName | None:
        move_by_delta: dict[tuple[int, int], ActionName] = {}
        for action in legal_actions:
            delta = action_delta(action)
            if delta is not None:
                move_by_delta.setdefault(delta, action)
        if self.pending_interactions > 0:
            for action in legal_actions:
                if is_interact_action(action) and parse_action_target(action) is None:
                    return action
        if not move_by_delta:
            return None
        actor = self._controlled_object(state, engine)
        if actor is None:
            return None
        start = actor.center_cell
        self.visited_actor_positions.add(start)
        color_priors = engine.color_progress_priors()
        best: tuple[float, ObjectToken, int] | None = None
        for obj in state.objects:
            if obj.object_id == actor.object_id:
                continue
            if obj.color == actor.color and obj.shape_hash == actor.shape_hash:
                continue
            dist = self._learned_grid_distance(start, obj.center_cell, state)
            if dist is None or dist == 0:
                continue
            weight = 0.20
            weight += color_priors.get(obj.color, 0.0)
            weight += 0.20 * self.productive_colors.get(obj.color, 0)
            weight -= 0.35 * self.nonproductive_colors.get(obj.color, 0)
            if "point" in obj.role_tags or "small" in obj.role_tags:
                weight += 0.10
            if "large" in obj.role_tags or "boundary_touching" in obj.role_tags:
                weight *= 0.45
            if weight <= 0.02:
                continue
            score = weight / (1.0 + 0.15 * dist)
            if best is None or score > best[0]:
                best = (float(score), obj, dist)
        if best is None:
            return self._frontier_action(state, start, move_by_delta)
        goal = best[1].center_cell
        best_step: tuple[int, ActionName] | None = None
        for delta, action in move_by_delta.items():
            nxt = (start[0] + delta[0], start[1] + delta[1])
            if nxt in self.blocked_cells:
                continue
            dist = self._learned_grid_distance(nxt, goal, state)
            if dist is None:
                continue
            if best_step is None or dist < best_step[0]:
                best_step = (dist, action)
        return None if best_step is None else best_step[1]

    def _frontier_action(
        self,
        state: StructuredState,
        start: tuple[int, int],
        move_by_delta: dict[tuple[int, int], ActionName],
    ) -> ActionName | None:
        rows, cols = state.grid.shape
        best: tuple[int, ActionName] | None = None
        for delta, action in move_by_delta.items():
            nxt = (start[0] + delta[0], start[1] + delta[1])
            if nxt in self.blocked_cells:
                continue
            if nxt[0] < 0 or nxt[1] < 0 or nxt[0] >= rows or nxt[1] >= cols:
                continue
            visits = 1 if nxt in self.visited_actor_positions else 0
            if best is None or visits < best[0]:
                best = (visits, action)
        return None if best is None else best[1]

    def _update_contact_statistics(self, record: TransitionRecord, engine: HypothesisEngine) -> None:
        actor = self._controlled_object(record.after, engine) or self._controlled_object(record.before, engine)
        if actor is None:
            return
        progress = combined_progress_signal(record.reward, record.delta.score_delta)
        colors: set[int] = set()
        for state in (record.before, record.after):
            current_actor = self._controlled_object(state, engine) or actor
            ar, ac = current_actor.center_cell
            for obj in state.objects:
                if obj.object_id == current_actor.object_id:
                    continue
                if obj.color == current_actor.color and obj.shape_hash == current_actor.shape_hash:
                    continue
                if abs(ar - obj.center_cell[0]) + abs(ac - obj.center_cell[1]) <= 1:
                    colors.add(obj.color)
        if record.delta.disappeared or colors:
            self.pending_interactions = max(self.pending_interactions, 2)
        if not colors:
            return
        if progress > 1e-8:
            for color in colors:
                self.productive_colors[color] += 1
        elif record.delta.has_visible_effect or is_move_action(record.action) or is_interact_action(record.action):
            for color in colors:
                self.nonproductive_colors[color] += 1

    def _spatial_goal_value(self, state: StructuredState, action: ActionName, engine: HypothesisEngine) -> float:
        """Give move actions a model-based incentive to test object contact hypotheses.

        The benchmark gives no instructions, so the agent must create its own
        experiments. This term identifies the currently controllable object,
        estimates its next position under the candidate action, and rewards moves
        that reduce distance to potentially useful or still-unexplained objects.
        It is intentionally weak relative to observed reward and information gain.
        """

        delta = action_delta(action)
        if delta is None or not state.objects:
            return 0.0
        actor = self._controlled_object(state, engine)
        if actor is None:
            return 0.0
        ar, ac = actor.center_cell
        rows, cols = state.grid.shape
        nr = min(max(ar + delta[0], 0), rows - 1)
        nc = min(max(ac + delta[1], 0), cols - 1)
        if (nr, nc) in self.blocked_cells:
            return -0.75
        color_priors = engine.color_progress_priors()
        value = 0.0
        for obj in state.objects:
            if obj.object_id == actor.object_id:
                continue
            if obj.color == actor.color and obj.shape_hash == actor.shape_hash:
                continue
            before_dist = self._learned_grid_distance((ar, ac), obj.center_cell, state)
            after_dist = self._learned_grid_distance((nr, nc), obj.center_cell, state)
            if before_dist is None or after_dist is None or before_dist == 0:
                continue
            progress = (before_dist - after_dist) / max(before_dist, 1)
            if progress <= 0:
                continue
            weight = 0.10
            weight += color_priors.get(obj.color, 0.0)
            if "point" in obj.role_tags or "small" in obj.role_tags:
                weight += 0.08
            if "large" in obj.role_tags or "boundary_touching" in obj.role_tags:
                weight *= 0.55
            value += weight * progress / (1.0 + after_dist)
        return float(min(value, 1.25))


    def _learned_grid_distance(
        self,
        start: tuple[int, int],
        goal: tuple[int, int],
        state: StructuredState,
    ) -> int | None:
        rows, cols = state.grid.shape
        if start == goal:
            return 0
        q: deque[tuple[tuple[int, int], int]] = deque([(start, 0)])
        seen = {start}
        while q:
            (r, c), dist = q.popleft()
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                nr, nc = r + dr, c + dc
                nxt = (nr, nc)
                if nr < 0 or nc < 0 or nr >= rows or nc >= cols or nxt in seen:
                    continue
                if nxt in self.blocked_cells and nxt != goal:
                    continue
                if nxt == goal:
                    return dist + 1
                seen.add(nxt)
                q.append((nxt, dist + 1))
        return None

    def _controlled_object(self, state: StructuredState, engine: HypothesisEngine) -> ObjectToken | None:
        color_scores = engine.controlled_object_colors()
        if color_scores:
            candidates = [obj for obj in state.objects if obj.color in color_scores]
            if candidates:
                return sorted(candidates, key=lambda o: (-color_scores.get(o.color, 0.0), o.area, o.object_id))[0]
        point_like = [obj for obj in state.objects if "point" in obj.role_tags or obj.area <= 4]
        if point_like:
            return sorted(point_like, key=lambda o: (o.area, "boundary_touching" in o.role_tags, o.object_id))[0]
        return min(state.objects, key=lambda o: (o.area, o.object_id))

    def _novelty(self, state: StructuredState, action: ActionName) -> float:
        local = self.state_action_visits[(state.abstract_fingerprint, action)]
        global_visits = self.action_visits[action]
        target_bonus = 0.10 if parse_action_target(action) is not None else 0.0
        move_or_interact_bonus = 0.08 if is_move_action(action) or is_interact_action(action) else 0.0
        return float(1.0 / (1.0 + local) + 0.25 / (1.0 + global_visits) + target_bonus + move_or_interact_bonus)

    def _repeat_penalty(self, state: StructuredState, action: ActionName) -> float:
        blocked = self.ineffective_actions[(state.exact_fingerprint, action)]
        penalty = 0.40 * min(blocked, 4)
        if self.last_action == action:
            local = self.state_action_visits[(state.abstract_fingerprint, action)]
            penalty += self.config.repeat_penalty * min(local, 4) / 4.0
        if self.stall_count:
            penalty += 0.15 * self.stall_count
        return float(penalty)

```

### `arcagi/scientist/agent.py`

New: end-to-end online scientist agent and compatibility hooks for the existing harness.

```python
"""Hyper-generalizing online scientist agent.

The agent loop is:

1. perceive the current grid as objects and relations;
2. convert the last action/outcome into a transition record;
3. update hypotheses, episodic memory, and the online world model;
4. choose the next action by trading off expected reward, novelty, and
   information gain;
5. emit grounded internal language for traceability and memory indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .hypotheses import HypothesisEngine
from .language import GroundedLanguage
from .memory import EpisodicMemory
from .perception import compare_states, extract_state
from .planner import PlannerConfig, ScientistPlanner
from .types import ActionDecision, ActionName, GridFrame, StructuredState, coerce_grid_frame, combined_progress_signal
from .world_model import OnlineWorldModel


@dataclass(frozen=True)
class ScientistAgentConfig:
    memory_capacity: int = 2048
    max_hypotheses: int = 512
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    world_learning_rate: float = 0.08
    seed: int = 0
    keep_world_weights_between_episodes: bool = True


class ScientistAgent:
    def __init__(self, config: ScientistAgentConfig | None = None) -> None:
        self.config = config or ScientistAgentConfig()
        self.engine = HypothesisEngine(max_hypotheses=self.config.max_hypotheses)
        self.memory = EpisodicMemory(capacity=self.config.memory_capacity)
        self.language = GroundedLanguage()
        self.world_model = OnlineWorldModel(learning_rate=self.config.world_learning_rate, seed=self.config.seed)
        self.planner = ScientistPlanner(config=self.config.planner, seed=self.config.seed)
        self.current_state: StructuredState | None = None
        self.last_decision: ActionDecision | None = None
        self.latest_language: tuple[str, ...] = ()
        self._last_raw_observation: Any | None = None
        self.transitions_observed = 0
        self.total_reward = 0.0
        self.trace: list[dict[str, Any]] = []

    def reset_episode(self) -> None:
        self.engine.reset_episode()
        self.memory.reset_episode()
        self.world_model.reset_episode(keep_weights=self.config.keep_world_weights_between_episodes)
        self.planner.reset_episode()
        self.current_state = None
        self.last_decision = None
        self.latest_language = ()
        self._last_raw_observation = None
        self.transitions_observed = 0
        self.total_reward = 0.0
        self.trace.clear()

    def perceive(self, observation: Any) -> StructuredState:
        return extract_state(coerce_grid_frame(observation))

    def act(self, observation: Any) -> ActionName:
        state = self.perceive(observation)
        self.current_state = state
        decision = self.planner.choose_action(
            state,
            engine=self.engine,
            world_model=self.world_model,
            memory=self.memory,
            language=self.language,
        )
        self.last_decision = decision
        self.latest_language = tuple(decision.language)
        self._last_raw_observation = observation
        self.trace.append(
            {
                "step": state.step_index,
                "action": decision.action,
                "score": decision.score,
                "components": dict(decision.components),
                "language": list(decision.language),
                "candidate_count": decision.candidate_count,
            }
        )
        return decision.action

    def observe_result(
        self,
        *,
        action: ActionName,
        before_observation: Any,
        after_observation: Any,
        reward: float = 0.0,
        terminated: bool = False,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        before = extract_state(coerce_grid_frame(before_observation))
        after = extract_state(coerce_grid_frame(after_observation))
        score_delta = _score_delta(before.frame, after.frame, info or {})
        record = compare_states(
            before,
            after,
            action=action,
            reward=reward,
            score_delta=score_delta,
            terminated=terminated,
            info=info or {},
        )
        self.engine.observe_transition(record)
        loss = self.world_model.update(record)
        prediction = self.world_model.predict(record.before, action)
        progress = combined_progress_signal(reward, score_delta)
        surprise = abs(float(progress) - prediction.reward_mean) + record.delta.changed_fraction + 0.05 * loss
        tokens = self.language.memory_tokens(after, self.engine)
        self.memory.write_transition(record, surprise=surprise, language_tokens=tokens)
        self.planner.notify_transition(
            changed=record.delta.has_visible_effect or abs(progress) > 1e-8,
            record=record,
            engine=self.engine,
        )
        self.current_state = after
        self.transitions_observed += 1
        self.total_reward += float(progress)

        if self.trace:
            self.trace[-1]["observed_reward"] = float(reward)
            self.trace[-1]["score_delta"] = float(score_delta)
            self.trace[-1]["changed_cells"] = record.delta.changed_cells
            self.trace[-1]["hypothesis_count"] = len(self.engine.hypotheses)
            self.trace[-1]["memory_count"] = len(self.memory.items)


    def update_after_step(
        self,
        *,
        next_observation: Any,
        reward: float = 0.0,
        terminated: bool = False,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """Compatibility hook for the existing ``arcagi.evaluation.harness`` loop."""

        if self.last_decision is None or self._last_raw_observation is None:
            return
        self.observe_result(
            action=self.last_decision.action,
            before_observation=self._last_raw_observation,
            after_observation=next_observation,
            reward=reward,
            terminated=terminated,
            info=info or {},
        )

    def reset_all(self) -> None:
        """Reset all online state for a fresh benchmark family/game."""

        self.reset_episode()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "transitions_observed": self.transitions_observed,
            "total_reward": self.total_reward,
            "hypothesis_count": len(self.engine.hypotheses),
            "credible_hypotheses": [h.description for h in self.engine.credible_hypotheses(limit=8)],
            "questions": list(self.language.questions(self.engine, limit=8)),
            "memory_items": len(self.memory.items),
            "option_items": len(self.memory.options),
            "last_decision": None
            if self.last_decision is None
            else {
                "action": self.last_decision.action,
                "score": self.last_decision.score,
                "components": dict(self.last_decision.components),
                "language": list(self.last_decision.language),
            },
        }


def _score_delta(before: GridFrame, after: GridFrame, info: Mapping[str, Any]) -> float:
    if "score_delta" in info:
        try:
            return float(info["score_delta"])
        except Exception:
            return 0.0
    before_score = before.extras.get("score")
    after_score = after.extras.get("score")
    try:
        if before_score is not None and after_score is not None:
            return float(after_score) - float(before_score)
    except Exception:
        return 0.0
    return 0.0

```

### `arcagi/scientist/runtime.py`

New: generic episode runner for black-box environments.

```python
"""Runtime helpers for evaluating ScientistAgent on ARC-like environments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from .agent import ScientistAgent
from .types import ActionName


@dataclass(frozen=True)
class EpisodeResult:
    steps: int
    total_reward: float
    terminated: bool
    final_info: Mapping[str, Any]
    diagnostics: Mapping[str, Any]


def run_episode(env: Any, agent: ScientistAgent, *, max_steps: int = 256, seed: int | None = None) -> EpisodeResult:
    agent.reset_episode()
    try:
        obs = env.reset(seed=seed)
    except TypeError:
        obs = env.reset()
    if isinstance(obs, tuple):
        obs = obs[0]
    total_reward = 0.0
    terminated = False
    final_info: Mapping[str, Any] = {}
    steps = 0
    for step in range(max_steps):
        action = agent.act(obs)
        result = env.step(action)
        next_obs, reward, terminated, info = _unpack_step_result(result)
        agent.observe_result(
            action=action,
            before_observation=obs,
            after_observation=next_obs,
            reward=reward,
            terminated=terminated,
            info=info,
        )
        total_reward += float(reward)
        obs = next_obs
        steps = step + 1
        final_info = info
        if terminated:
            break
    return EpisodeResult(
        steps=steps,
        total_reward=total_reward,
        terminated=terminated,
        final_info=final_info,
        diagnostics=agent.diagnostics(),
    )


def _unpack_step_result(result: Any) -> tuple[Any, float, bool, Mapping[str, Any]]:
    # Existing arcagi StepResult-style object.
    if hasattr(result, "observation") or hasattr(result, "next_observation"):
        obs = getattr(result, "observation", None)
        if obs is None:
            obs = getattr(result, "next_observation")
        reward = float(getattr(result, "reward", 0.0))
        terminated = bool(getattr(result, "terminated", False) or getattr(result, "done", False))
        info = getattr(result, "info", None) or getattr(result, "extras", None) or {}
        return obs, reward, terminated, info
    # Gymnasium shape.
    if isinstance(result, tuple) and len(result) == 5:
        obs, reward, terminated, truncated, info = result
        return obs, float(reward), bool(terminated or truncated), info or {}
    # Classic Gym shape.
    if isinstance(result, tuple) and len(result) == 4:
        obs, reward, done, info = result
        return obs, float(reward), bool(done), info or {}
    raise TypeError(f"Unsupported env.step result: {type(result)!r}")

```

### `arcagi/scientist/synthetic_env.py`

New: hidden-rule smoke-test environment for online key/goal learning.

```python
"""Small black-box synthetic environments for testing online rule learning.

These are not ARC-AGI-3 games.  They are intentionally tiny hidden-rule worlds
that exercise the same mechanics: sparse reward, latent affordances, object
movement, interaction, and no natural-language instructions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from .types import GridFrame, action_delta, action_family, parse_action_target


@dataclass(frozen=True)
class SyntheticConfig:
    size: int = 7
    requires_key: bool = True
    seed: int = 0
    max_steps: int = 80


class HiddenRuleGridEnv:
    """A minimal grid game with hidden key/goal mechanics.

    Colors:
        0 background, 1 avatar, 2 goal, 3 key, 4 wall, 5 distractor.
    """

    def __init__(self, config: SyntheticConfig | None = None) -> None:
        self.config = config or SyntheticConfig()
        self.rng = np.random.default_rng(self.config.seed)
        self.step_index = 0
        self.episode_index = 0
        self.avatar = (1, 1)
        self.goal = (self.config.size - 2, self.config.size - 2)
        self.key = (1, self.config.size - 2)
        self.has_key = False
        self.done = False
        self.score = 0.0
        self.available_actions = ("up", "down", "left", "right", "interact", "click")

    @property
    def task_id(self) -> str:
        return "synthetic/hidden_rule_grid"

    def reset(self, seed: int | None = None) -> GridFrame:
        if seed is not None:
            self.rng = np.random.default_rng(seed)
        self.episode_index += 1
        self.step_index = 0
        self.done = False
        self.has_key = False
        self.score = 0.0
        self.avatar = (1, 1)
        self.goal = (self.config.size - 2, self.config.size - 2)
        self.key = (1, self.config.size - 2)
        return self._frame()

    def step(self, action: str) -> tuple[GridFrame, float, bool, dict[str, Any]]:
        if self.done:
            return self._frame(), 0.0, True, {"score": self.score, "already_done": True}
        old_score = self.score
        reward = 0.0
        fam = action_family(action)
        target = parse_action_target(action)
        if target is not None and fam in {"click", "action6"}:
            # Clicking the key is a legal but hidden affordance.  It gives the
            # agent a way to test targeted actions without requiring text.
            if _manhattan(target, self.key) <= 0 and not self.has_key:
                self.has_key = True
                self.score += 0.10
                reward += 0.10
        elif fam == "interact":
            if _manhattan(self.avatar, self.key) <= 1 and not self.has_key:
                self.has_key = True
                self.score += 0.10
                reward += 0.10
        else:
            delta = action_delta(action)
            if delta is not None:
                nr = min(max(self.avatar[0] + delta[0], 0), self.config.size - 1)
                nc = min(max(self.avatar[1] + delta[1], 0), self.config.size - 1)
                if (nr, nc) not in self._walls():
                    self.avatar = (nr, nc)

        if _manhattan(self.avatar, self.goal) <= 0:
            if (not self.config.requires_key) or self.has_key:
                self.score += 1.0
                reward += 1.0
                self.done = True
            else:
                reward -= 0.02

        self.step_index += 1
        if self.step_index >= self.config.max_steps:
            self.done = True
        info = {"score": self.score, "score_delta": self.score - old_score, "has_key": self.has_key}
        return self._frame(), float(reward), bool(self.done), info

    def _walls(self) -> set[tuple[int, int]]:
        size = self.config.size
        walls = {(3, c) for c in range(1, size - 1) if c != size // 2}
        return walls

    def _grid(self) -> np.ndarray:
        size = self.config.size
        grid = np.zeros((size, size), dtype=np.int64)
        for r, c in self._walls():
            grid[r, c] = 4
        grid[self.goal] = 2
        if not self.has_key:
            grid[self.key] = 3
        grid[size - 2, 1] = 5
        grid[self.avatar] = 1
        return grid

    def _frame(self) -> GridFrame:
        return GridFrame(
            task_id=self.task_id,
            episode_id=f"{self.task_id}/episode_{self.episode_index}",
            step_index=self.step_index,
            grid=self._grid(),
            available_actions=self.available_actions,
            extras={"score": self.score, "has_key": self.has_key},
        )


def _manhattan(a: tuple[int, int], b: tuple[int, int]) -> int:
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

```

### `arcagi/scientist/cli.py`

New: command-line smoke-test entry point.

```python
"""Command-line entry points for the scientist patch."""

from __future__ import annotations

import argparse
import json

from . import HiddenRuleGridEnv, ScientistAgent, SyntheticConfig, run_episode


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Run the ARC-AGI-3 scientist agent on the synthetic hidden-rule smoke test.")
    parser.add_argument("--size", type=int, default=7, help="Synthetic grid size.")
    parser.add_argument("--max-steps", type=int, default=80, help="Maximum episode steps.")
    parser.add_argument("--seed", type=int, default=7, help="Environment and agent seed.")
    parser.add_argument("--no-key", action="store_true", help="Disable the hidden key requirement.")
    args = parser.parse_args(argv)

    env = HiddenRuleGridEnv(
        SyntheticConfig(
            size=args.size,
            requires_key=not args.no_key,
            seed=args.seed,
            max_steps=args.max_steps,
        )
    )
    agent = ScientistAgent()
    result = run_episode(env, agent, max_steps=args.max_steps, seed=args.seed)
    print(
        json.dumps(
            {
                "steps": result.steps,
                "total_reward": result.total_reward,
                "terminated": result.terminated,
                "final_info": dict(result.final_info),
                "diagnostics": result.diagnostics,
            },
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()

```

### `arcagi/scientist/__init__.py`

New: public package exports.

```python
"""Hypothesis-driven online experimental scientist agent for ARC-AGI-3."""

from .agent import ScientistAgent, ScientistAgentConfig
from .runtime import EpisodeResult, run_episode
from .synthetic_env import HiddenRuleGridEnv, SyntheticConfig

__all__ = [
    "ScientistAgent",
    "ScientistAgentConfig",
    "EpisodeResult",
    "run_episode",
    "HiddenRuleGridEnv",
    "SyntheticConfig",
]

```

### `arcagi/agents/scientist_agent.py`

New: wrapper under the existing arcagi.agents namespace.

```python
"""Compatibility wrapper for the existing ``arcagi.agents`` namespace."""

from __future__ import annotations

from typing import Any

from arcagi.scientist import ScientistAgent, ScientistAgentConfig

try:  # Existing repository compatibility; tests do not require this base class.
    from arcagi.agents.base import BaseAgent  # type: ignore
except Exception:  # pragma: no cover
    BaseAgent = object  # type: ignore


class HyperGeneralizingScientistAgent(ScientistAgent, BaseAgent):  # type: ignore[misc]
    """ARC-facing online learner built from the scientist-agent components."""

    def __init__(self, config: ScientistAgentConfig | None = None, **_: Any) -> None:
        ScientistAgent.__init__(self, config=config)


def make_agent(config: ScientistAgentConfig | None = None) -> HyperGeneralizingScientistAgent:
    return HyperGeneralizingScientistAgent(config=config)

```

### `arcagi/evaluation/harness.py`

Edited: adds scientist agent selection and robust observe_result/update_after_step support.

```python
from __future__ import annotations

import argparse
import json
from pathlib import Path
from statistics import mean
from typing import Any

from arcagi.agents.graph_agent import GraphExplorerAgent
from arcagi.agents.random_agent import RandomHeuristicAgent
from arcagi.envs.arc_adapter import Arcade, ArcToolkitEnv, arc_operation_mode, arc_toolkit_available, list_arc_games
from arcagi.envs.synthetic import DEFAULT_SYNTHETIC_FAMILY_MODES, HiddenRuleEnv, family_variants_for_mode


def build_agent(agent_name: str, checkpoint_path: str | None = None, device: Any = None):
    normalized = agent_name.strip().lower()
    if normalized == "random":
        return RandomHeuristicAgent()
    if normalized == "graph":
        return GraphExplorerAgent()
    if normalized in {"scientist", "hyper_scientist", "hyper-generalizing-scientist"}:
        from arcagi.agents.scientist_agent import HyperGeneralizingScientistAgent

        return HyperGeneralizingScientistAgent()

    import torch

    from arcagi.agents.learned_agent import HybridAgent, LanguageNoMemoryAgent, RecurrentAblationAgent
    from arcagi.memory.episodic import EpisodicMemory
    from arcagi.planning.planner import HybridPlanner, PlannerConfig
    from arcagi.training.synthetic import build_default_modules, load_checkpoint

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path and Path(checkpoint_path).exists():
        encoder, world_model, language_model = load_checkpoint(checkpoint_path, device=device)
    else:
        encoder, world_model, language_model, _planner = build_default_modules(device=device)

    planner = HybridPlanner(
        PlannerConfig(search_depth=2, search_root_width=2, search_branch_width=1, max_world_model_calls=48)
        if device.type == "cpu"
        else PlannerConfig()
    )
    if normalized == "recurrent":
        return RecurrentAblationAgent(encoder=encoder, world_model=world_model, planner=planner, device=device)
    if normalized == "language":
        return LanguageNoMemoryAgent(
            encoder=encoder,
            world_model=world_model,
            planner=planner,
            language_model=language_model,
            device=device,
        )
    if normalized == "hybrid":
        return HybridAgent(
            encoder=encoder,
            world_model=world_model,
            planner=planner,
            language_model=language_model,
            episodic_memory=EpisodicMemory(),
            device=device,
        )
    raise ValueError(f"unknown agent: {agent_name}")


def _reset_agent_for_episode(agent: Any) -> None:
    reset_episode = getattr(agent, "reset_episode", None)
    if callable(reset_episode):
        reset_episode()


def _reset_agent_for_family(agent: Any) -> None:
    reset_all = getattr(agent, "reset_all", None)
    if callable(reset_all):
        reset_all()
        return
    _reset_agent_for_episode(agent)


def _observe_step(agent: Any, *, action: str, before: Any, result: Any) -> None:
    update_after_step = getattr(agent, "update_after_step", None)
    if callable(update_after_step):
        update_after_step(
            next_observation=result.observation,
            reward=result.reward,
            terminated=result.terminated or result.truncated,
            info=result.info,
        )
        return
    observe_result = getattr(agent, "observe_result", None)
    if callable(observe_result):
        observe_result(
            action=action,
            before_observation=before,
            after_observation=result.observation,
            reward=result.reward,
            terminated=result.terminated or result.truncated,
            info=result.info,
        )


def run_episode(
    agent: Any,
    env: Any,
    seed: int,
    max_steps: int | None = None,
    stop_on_positive_reward: bool = False,
) -> dict[str, object]:
    observation = env.reset(seed=seed)
    _reset_agent_for_episode(agent)
    steps = 0
    success = False
    rewards: list[float] = []
    interaction_steps = 0
    language_traces: list[str] = []
    done = False
    while not done and (max_steps is None or steps < max_steps):
        before = observation
        action = agent.act(observation)
        action_text = str(action)
        if action_text.startswith("interact") or action_text.startswith("click:"):
            interaction_steps += 1
        result = env.step(action)
        _observe_step(agent, action=action_text, before=before, result=result)
        observation = result.observation
        done = bool(result.terminated or result.truncated)
        steps += 1
        rewards.append(float(result.reward))
        if result.reward > 0.9:
            success = True
            if stop_on_positive_reward:
                done = True
        if getattr(agent, "latest_language", ()):
            language_traces.append(" ".join(str(x) for x in agent.latest_language))
    return {
        "success": success,
        "return": float(sum(rewards)),
        "steps": steps,
        "interaction_steps": interaction_steps,
        "language": language_traces[-1] if language_traces else "",
        "family_id": getattr(env, "family_id", getattr(env, "task_id", "unknown")),
        "diagnostics": getattr(agent, "diagnostics", lambda: {})(),
    }


def evaluate_synthetic(
    agent_name: str,
    checkpoint_path: str | None = None,
    episodes_per_family: int = 3,
    seed: int = 17,
) -> dict[str, object]:
    agent = build_agent(agent_name, checkpoint_path=checkpoint_path)
    families: list[dict[str, object]] = []
    seed_cursor = seed
    for family_mode in DEFAULT_SYNTHETIC_FAMILY_MODES:
        for variant in family_variants_for_mode(family_mode):
            _reset_agent_for_family(agent)
            episode_metrics = []
            for episode_idx in range(episodes_per_family):
                env = HiddenRuleEnv(
                    family_mode=family_mode,
                    family_variant=variant,
                    seed=seed_cursor + episode_idx,
                )
                episode_metrics.append(run_episode(agent, env, seed=seed_cursor + episode_idx))
            seed_cursor += episodes_per_family
            families.append(
                {
                    "family_mode": family_mode,
                    "family_variant": variant,
                    "episodes": episode_metrics,
                }
            )
    all_episodes = [episode for family in families for episode in family["episodes"]]
    first_episode_success = [family["episodes"][0]["success"] for family in families]
    later_episode_success = [episode["success"] for family in families for episode in family["episodes"][1:]]
    return {
        "agent": agent_name,
        "success_rate": mean(float(item["success"]) for item in all_episodes) if all_episodes else 0.0,
        "avg_return": mean(float(item["return"]) for item in all_episodes) if all_episodes else 0.0,
        "avg_steps": mean(float(item["steps"]) for item in all_episodes) if all_episodes else 0.0,
        "avg_interactions": mean(float(item["interaction_steps"]) for item in all_episodes) if all_episodes else 0.0,
        "first_episode_success": mean(float(value) for value in first_episode_success) if first_episode_success else 0.0,
        "later_episode_success": mean(float(value) for value in later_episode_success) if later_episode_success else 0.0,
        "families": families,
    }


def evaluate_arc(
    agent_name: str,
    checkpoint_path: str | None = None,
    game_limit: int = 3,
    mode: str = "offline",
) -> dict[str, object]:
    if not arc_toolkit_available():
        return {"agent": agent_name, "skipped": True, "reason": "ARC toolkit not installed"}
    operation_mode = arc_operation_mode(mode)
    games = list_arc_games(operation_mode=operation_mode)[:game_limit]
    agent = build_agent(agent_name, checkpoint_path=checkpoint_path)
    results = []
    shared_arcade = None if Arcade is None else Arcade(operation_mode=operation_mode)
    for index, game_id in enumerate(games):
        _reset_agent_for_family(agent)
        env = ArcToolkitEnv(game_id, operation_mode=operation_mode, arcade=shared_arcade)
        try:
            episode = run_episode(agent, env, seed=index, max_steps=256, stop_on_positive_reward=True)
        finally:
            env.close()
        results.append({"game_id": game_id, **episode})
    if shared_arcade is not None:
        close_scorecard = getattr(shared_arcade, "close_scorecard", None)
        if callable(close_scorecard):
            try:
                close_scorecard()
            except Exception:
                pass
    return {"agent": agent_name, "mode": mode, "games": results}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m arcagi.evaluation.harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthetic_parser = subparsers.add_parser("synthetic")
    synthetic_parser.add_argument("--agent", type=str, default="graph")
    synthetic_parser.add_argument("--checkpoint-path", type=str, default="")
    synthetic_parser.add_argument("--episodes-per-family", type=int, default=3)
    synthetic_parser.add_argument("--seed", type=int, default=17)
    arc_parser = subparsers.add_parser("arc")
    arc_parser.add_argument("--agent", type=str, default="graph")
    arc_parser.add_argument("--checkpoint-path", type=str, default="")
    arc_parser.add_argument("--game-limit", type=int, default=3)
    arc_parser.add_argument("--mode", type=str, default="offline")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "synthetic":
        result = evaluate_synthetic(
            agent_name=args.agent,
            checkpoint_path=args.checkpoint_path or None,
            episodes_per_family=args.episodes_per_family,
            seed=args.seed,
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "arc":
        result = evaluate_arc(
            agent_name=args.agent,
            checkpoint_path=args.checkpoint_path or None,
            game_limit=args.game_limit,
            mode=args.mode,
        )
        print(json.dumps(result, indent=2))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())

```

### `arcagi/evaluation/__init__.py`

New/placeholder if absent: evaluation package marker.

```python

```

### `scripts/run_scientist_synthetic.py`

Edited: wrapper script for the scientist CLI.

```python
from __future__ import annotations

from arcagi.scientist.cli import main


if __name__ == "__main__":
    main()

```

### `tests/test_scientist_agent.py`

New/edited: perception, hypothesis, reward-deduplication, and hidden-rule solve tests.

```python
from __future__ import annotations

import numpy as np

from arcagi.scientist import HiddenRuleGridEnv, ScientistAgent, SyntheticConfig, run_episode
from arcagi.scientist.hypotheses import HypothesisEngine
from arcagi.scientist.perception import compare_states, extract_state
from arcagi.scientist.types import GridFrame, combined_progress_signal


def test_perception_segments_non_background_objects() -> None:
    grid = np.array(
        [
            [0, 0, 0, 0],
            [0, 1, 1, 0],
            [0, 0, 2, 0],
            [0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    state = extract_state(GridFrame("t", "e", 0, grid, ("up",)))
    colors = sorted(obj.color for obj in state.objects)
    assert colors == [1, 2]
    assert state.abstract_fingerprint
    assert len(state.relations) >= 1


def test_hypothesis_engine_induces_movement_rule() -> None:
    before = GridFrame("t", "e", 0, np.array([[0, 0, 0], [0, 1, 0], [0, 0, 0]]), ("right",))
    after = GridFrame("t", "e", 1, np.array([[0, 0, 0], [0, 0, 1], [0, 0, 0]]), ("right",))
    record = compare_states(extract_state(before), extract_state(after), action="right")
    engine = HypothesisEngine()
    engine.observe_transition(record)
    kinds = {hyp.kind for hyp in engine.hypotheses.values()}
    assert "action_moves_object" in kinds


def test_combined_progress_signal_does_not_double_count_score_delta() -> None:
    assert combined_progress_signal(0.1, 0.1) == 0.1
    assert combined_progress_signal(0.1, 0.0) == 0.1
    assert combined_progress_signal(0.1, 0.2) == 0.30000000000000004


def test_scientist_agent_runs_updates_and_solves_hidden_rule_smoke_test() -> None:
    env = HiddenRuleGridEnv(SyntheticConfig(size=7, requires_key=True, seed=3, max_steps=80))
    agent = ScientistAgent()
    result = run_episode(env, agent, max_steps=80, seed=3)
    assert result.steps > 0
    assert result.diagnostics["transitions_observed"] == result.steps
    assert result.diagnostics["hypothesis_count"] > 0
    assert result.diagnostics["memory_items"] > 0
    assert agent.trace
    assert result.total_reward >= 1.0
    assert result.terminated

```

### `pyproject.toml`

Edited: project description, optional arc dependency, dev dependency, and scientist script entry point.

```toml
[build-system]
requires = ["hatchling>=1.27.0"]
build-backend = "hatchling.build"

[project]
name = "arcagi"
version = "0.1.0"
description = "Local ARC-AGI-3 hybrid agent research stack with explicit exploration, world modeling, memory, grounded language, and hypothesis-driven online learning."
readme = "README.md"
requires-python = ">=3.10"
license = { text = "MIT" }
authors = [{ name = "OpenAI Codex" }]
dependencies = [
  "numpy>=1.26",
  "torch>=2.5",
]

[project.optional-dependencies]
arc = [
  "arc-agi>=0.9.7; python_version >= '3.12'",
]
dev = [
  "pytest>=8.2",
]

[project.scripts]
arcagi = "arcagi.cli:main"
arcagi-scientist-synthetic = "arcagi.scientist.cli:main"

[tool.hatch.build.targets.wheel]
packages = ["arcagi"]

[tool.pytest.ini_options]
testpaths = ["tests"]

```
