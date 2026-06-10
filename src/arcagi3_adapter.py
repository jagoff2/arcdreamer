from __future__ import annotations

import copy
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
import torch.nn.functional as F

from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .explore_env import (
    ACTION_AVOID,
    ACTION_FORAGE,
    ACTION_INSPECT,
    ACTION_OPEN,
    ACTION_REFUSE,
    ACTION_RETURN,
    ACTION_TEST,
    ACTION_WAIT,
    HYPOTHESIS_DIM,
    INTRINSIC_DIM,
    NUM_AGENTS,
    NUM_DOORS,
    NUM_EXPLORER_ACTIONS,
    NUM_HAZARDS,
    NUM_LOCATIONS,
    NUM_OBJECTS,
    NUM_PHASES,
    NUM_PROJECTS,
    NUM_RESOURCES,
    NUM_RULES,
    NUM_SKILLS,
    NUM_SOURCES,
    NUM_TOOLS,
    OBS_DIM,
    PROJECT_STATE_DIM,
    SKILL_MEMORY_DIM,
    SOCIAL_STATE_DIM,
    Z_DIM,
    _category_latent,
)
from .head_collapse import HeadCollapsedExplorer


FIXTURE_PATH = Path("docs/arcagi3_public_fixtures.json")

CELL_EMPTY = 0
CELL_WALL = 1
CELL_AGENT = 2
CELL_KEY = 3
CELL_DOOR = 4
CELL_GOAL = 5
CELL_HAZARD = 6
CELL_RESOURCE = 7
CELL_UNKNOWN = 8

CHAR_TO_CELL = {
    ".": CELL_EMPTY,
    "#": CELL_WALL,
    "A": CELL_AGENT,
    "K": CELL_KEY,
    "D": CELL_DOOR,
    "G": CELL_GOAL,
    "H": CELL_HAZARD,
    "R": CELL_RESOURCE,
    "?": CELL_UNKNOWN,
}
CELL_TO_CHAR = {value: key for key, value in CHAR_TO_CELL.items()}

MOVE_DELTAS = {
    "up": (-1, 0),
    "down": (1, 0),
    "left": (0, -1),
    "right": (0, 1),
    "1": (-1, 0),
    "2": (1, 0),
    "3": (0, -1),
    "4": (0, 1),
}


@dataclass(frozen=True)
class ArcAGI3Observation:
    task_id: str
    episode_id: str
    step_index: int
    grid: np.ndarray
    available_actions: tuple[str, ...]
    extras: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ArcAGI3StepResult:
    observation: ArcAGI3Observation
    reward: float
    terminated: bool
    truncated: bool
    info: dict[str, Any] = field(default_factory=dict)


@dataclass
class ArcAGI3Fixture:
    fixture_id: str
    title: str
    rows: list[str]
    max_steps: int
    actions: tuple[str, ...]
    visibility_radius: int = 99
    reveal_steps: int = 0
    require_key: bool = False
    click_goal: bool = False
    max_score: float = 1.0
    public_rules: dict[str, Any] = field(default_factory=dict)


def load_public_fixtures(path: str | Path = FIXTURE_PATH) -> list[ArcAGI3Fixture]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    fixtures: list[ArcAGI3Fixture] = []
    for item in payload["games"]:
        fixtures.append(
            ArcAGI3Fixture(
                fixture_id=str(item["id"]),
                title=str(item["title"]),
                rows=list(item["grid"]),
                max_steps=int(item["max_steps"]),
                actions=tuple(str(action) for action in item["actions"]),
                visibility_radius=int(item.get("visibility_radius", 99)),
                reveal_steps=int(item.get("reveal_steps", 0)),
                require_key=bool(item.get("require_key", False)),
                click_goal=bool(item.get("click_goal", False)),
                max_score=float(item.get("max_score", 1.0)),
                public_rules=dict(item.get("public_rules", {})),
            )
        )
    return fixtures


def grid_to_rows(grid: np.ndarray) -> list[str]:
    rows: list[str] = []
    for row in np.asarray(grid, dtype=np.int64):
        rows.append("".join(CELL_TO_CHAR.get(int(value), "?") for value in row))
    return rows


def rows_to_grid(rows: Iterable[str]) -> tuple[np.ndarray, tuple[int, int]]:
    parsed: list[list[int]] = []
    agent = (-1, -1)
    for y, row in enumerate(rows):
        parsed_row: list[int] = []
        for x, char in enumerate(row):
            value = CHAR_TO_CELL[char]
            if value == CELL_AGENT:
                agent = (y, x)
                value = CELL_EMPTY
            parsed_row.append(value)
        parsed.append(parsed_row)
    if agent == (-1, -1):
        raise ValueError("fixture grid must contain an A agent marker")
    return np.asarray(parsed, dtype=np.int64), agent


def action_family(action: str) -> str:
    if action.startswith("click:"):
        return "click"
    if action in MOVE_DELTAS:
        return "move"
    if action == "0":
        return "reset"
    if action in {"wait", "noop"}:
        return "wait"
    return "other"


def parse_click(action: str) -> tuple[int, int] | None:
    if not action.startswith("click:"):
        return None
    parts = action.split(":")
    if len(parts) != 3 or not parts[1].lstrip("-").isdigit() or not parts[2].lstrip("-").isdigit():
        return None
    return int(parts[1]), int(parts[2])


class ArcAGI3FixtureEnv:
    def __init__(self, fixture: ArcAGI3Fixture, seed: int = 0) -> None:
        self.fixture = fixture
        self.seed = int(seed)
        self._base_grid, self._start = rows_to_grid(fixture.rows)
        self._rng = np.random.default_rng(self.seed)
        self.episode_id = f"arcagi3/{fixture.fixture_id}/episode_{self.seed}"
        self.reset(seed=seed)

    @property
    def task_id(self) -> str:
        return f"arcagi3/{self.fixture.fixture_id}"

    def clone(self) -> "ArcAGI3FixtureEnv":
        return copy.deepcopy(self)

    def legal_actions(self) -> tuple[str, ...]:
        return self.fixture.actions

    def reset(self, seed: int | None = None) -> ArcAGI3Observation:
        if seed is not None:
            self.seed = int(seed)
            self._rng = np.random.default_rng(self.seed)
            self.episode_id = f"arcagi3/{self.fixture.fixture_id}/episode_{self.seed}"
        self.grid = self._base_grid.copy()
        self.agent = tuple(self._start)
        self.step_index = 0
        self.have_key = False
        self.door_open = False
        self.score = 0.0
        self.positive_score = 0.0
        self.terminated = False
        self.truncated = False
        self.events: list[str] = []
        return self._observation()

    def step(self, action: str) -> ArcAGI3StepResult:
        events: list[str] = []
        reward = 0.0
        if self.terminated or self.truncated:
            obs = self._observation()
            return ArcAGI3StepResult(obs, 0.0, self.terminated, self.truncated, {"events": ["already_terminal"]})
        if action not in self.legal_actions():
            reward -= 0.05
            events.append("invalid_action")
        else:
            family = action_family(action)
            if family == "move":
                reward += self._move(action, events)
            elif family == "click":
                reward += self._click(action, events)
            elif family == "wait":
                reward -= 0.01
                events.append("wait")
            elif family == "reset":
                self.reset(seed=self.seed)
                events.append("reset")
        self.step_index += 1
        if self.step_index >= self.fixture.max_steps and not self.terminated:
            self.truncated = True
            events.append("step_limit")
        self.score += float(reward)
        if reward > 0.0:
            self.positive_score += float(reward)
        self.events.extend(events)
        obs = self._observation()
        info = {
            "events": events,
            "score": self.score,
            "positive_score": self.positive_score,
            "normalized_score": self.normalized_score(),
            "have_key": self.have_key,
            "door_open": self.door_open,
            "game_state": "WIN" if self.terminated else "RUNNING",
        }
        return ArcAGI3StepResult(obs, float(reward), self.terminated, self.truncated, info)

    def normalized_score(self) -> float:
        return float(max(0.0, min(1.0, self.score / max(self.fixture.max_score, 1.0e-6))))

    def _move(self, action: str, events: list[str]) -> float:
        dy, dx = MOVE_DELTAS[action]
        y, x = self.agent
        ny, nx = y + dy, x + dx
        if not self._in_bounds(ny, nx):
            events.append("blocked_edge")
            return -0.03
        cell = int(self.grid[ny, nx])
        if cell == CELL_WALL:
            events.append("blocked_wall")
            return -0.03
        if cell == CELL_DOOR and self.fixture.require_key and not self.have_key:
            events.append("blocked_door")
            return -0.02
        self.agent = (ny, nx)
        events.append("move")
        reward = -0.005
        if cell == CELL_KEY:
            self.have_key = True
            self.grid[ny, nx] = CELL_EMPTY
            reward += 0.25
            events.append("key_collected")
        if cell == CELL_RESOURCE:
            self.grid[ny, nx] = CELL_EMPTY
            reward += 0.12
            events.append("resource_collected")
        if cell == CELL_DOOR:
            self.door_open = True
            self.grid[ny, nx] = CELL_EMPTY
            reward += 0.15
            events.append("door_opened")
        if cell == CELL_HAZARD:
            reward -= 0.25
            events.append("hazard_hit")
        if cell == CELL_GOAL and (not self.fixture.require_key or self.have_key):
            reward += 1.0
            self.terminated = True
            events.append("goal_reached")
        return reward

    def _click(self, action: str, events: list[str]) -> float:
        click = parse_click(action)
        if click is None:
            events.append("bad_click_format")
            return -0.03
        x, y = click
        if not self._in_bounds(y, x):
            events.append("bad_click_bounds")
            return -0.03
        cell = int(self.grid[y, x])
        if self.fixture.click_goal and cell == CELL_GOAL:
            self.terminated = True
            events.append("goal_clicked")
            return 1.0
        if cell in {CELL_KEY, CELL_RESOURCE}:
            events.append("useful_click")
            return 0.08
        events.append("no_effect_click")
        return -0.02

    def _observation(self) -> ArcAGI3Observation:
        observed = self.grid.copy()
        ay, ax = self.agent
        radius = self.fixture.visibility_radius
        if self.step_index >= self.fixture.reveal_steps and radius < 99:
            masked = np.full_like(observed, CELL_UNKNOWN)
            for y in range(observed.shape[0]):
                for x in range(observed.shape[1]):
                    if abs(y - ay) + abs(x - ax) <= radius:
                        masked[y, x] = observed[y, x]
            observed = masked
        observed[ay, ax] = CELL_AGENT
        return ArcAGI3Observation(
            task_id=self.task_id,
            episode_id=self.episode_id,
            step_index=self.step_index,
            grid=observed,
            available_actions=self.legal_actions(),
            extras={
                "adapter": "committed_public_fixture",
                "fixture_id": self.fixture.fixture_id,
                "title": self.fixture.title,
                "score": self.score,
                "normalized_score": self.normalized_score(),
                "have_key": self.have_key,
                "door_open": self.door_open,
                "public_rules": self.fixture.public_rules,
                "max_score": self.fixture.max_score,
                "game_state": "WIN" if self.terminated else "RUNNING",
            },
        )

    def _in_bounds(self, y: int, x: int) -> bool:
        return y >= 0 and x >= 0 and y < self.grid.shape[0] and x < self.grid.shape[1]


@dataclass
class ArcAdapterMemory:
    visited: set[tuple[int, int]] = field(default_factory=set)
    remembered: dict[str, set[tuple[int, int]]] = field(
        default_factory=lambda: {"goal": set(), "key": set(), "door": set(), "hazard": set(), "resource": set()}
    )
    action_values: dict[str, list[float]] = field(default_factory=dict)
    last_grid: np.ndarray | None = None
    have_key: bool = False
    door_open: bool = False
    previous_semantic_action: int = ACTION_INSPECT

    def update_observation(self, observation: ArcAGI3Observation) -> None:
        grid = np.asarray(observation.grid, dtype=np.int64)
        agent = find_agent(grid)
        if agent is not None:
            self.visited.add(agent)
        for name, code in [
            ("goal", CELL_GOAL),
            ("key", CELL_KEY),
            ("door", CELL_DOOR),
            ("hazard", CELL_HAZARD),
            ("resource", CELL_RESOURCE),
        ]:
            for y, x in positions_of(grid, code):
                self.remembered[name].add((y, x))
        self.have_key = bool(observation.extras.get("have_key", self.have_key))
        self.door_open = bool(observation.extras.get("door_open", self.door_open))

    def update_transition(self, action: str, result: ArcAGI3StepResult) -> None:
        self.action_values.setdefault(action_family(action), []).append(float(result.reward))
        self.last_grid = np.asarray(result.observation.grid, dtype=np.int64).copy()
        self.update_observation(result.observation)

    def recall(self) -> dict[str, Any]:
        return {
            "visited_cells": len(self.visited),
            "goal": sorted(self.remembered["goal"])[:4],
            "key": sorted(self.remembered["key"])[:4],
            "door": sorted(self.remembered["door"])[:4],
            "hazard_count": len(self.remembered["hazard"]),
            "resource": sorted(self.remembered["resource"])[:4],
            "have_key": self.have_key,
            "door_open": self.door_open,
        }

    def corrupted(self) -> "ArcAdapterMemory":
        bad = copy.deepcopy(self)
        bad.remembered["goal"], bad.remembered["hazard"] = bad.remembered["hazard"], bad.remembered["goal"]
        bad.remembered["key"] = set()
        bad.have_key = False
        bad.previous_semantic_action = ACTION_WAIT
        return bad


def find_agent(grid: np.ndarray) -> tuple[int, int] | None:
    cells = np.argwhere(np.asarray(grid) == CELL_AGENT)
    if cells.size == 0:
        return None
    y, x = cells[0]
    return int(y), int(x)


def positions_of(grid: np.ndarray, code: int) -> list[tuple[int, int]]:
    return [(int(y), int(x)) for y, x in np.argwhere(np.asarray(grid) == int(code))]


def nearest_position(origin: tuple[int, int] | None, positions: Iterable[tuple[int, int]]) -> tuple[int, int] | None:
    items = list(positions)
    if origin is None or not items:
        return items[0] if items else None
    oy, ox = origin
    return min(items, key=lambda item: abs(item[0] - oy) + abs(item[1] - ox))


def one_hot(index: int, classes: int, device: torch.device) -> torch.Tensor:
    idx = torch.tensor([max(0, min(classes - 1, int(index)))], dtype=torch.long, device=device)
    return F.one_hot(idx, classes).float()


class ARCAGI3Adapter:
    def __init__(
        self,
        explorer: HeadCollapsedExplorer,
        *,
        device: DeviceLike = AUTO_DEVICE,
        mode: str = "normal",
    ) -> None:
        self.explorer = explorer
        self.device = resolve_device(device)
        self.mode = mode
        self.memory = ArcAdapterMemory()

    def reset(self) -> None:
        self.memory = ArcAdapterMemory()

    def choose_action(self, observation: ArcAGI3Observation) -> tuple[str, dict[str, Any]]:
        self.memory.update_observation(observation)
        tensors, feature_state = self.tensorize(observation)
        ablations = {
            "z_enabled": self.mode != "zero_z",
            "hypothesis_enabled": self.mode not in {"no_hypothesis_memory", "no_planner_imagination"},
            "skill_enabled": self.mode != "corrupt_memory",
            "project_enabled": self.mode not in {"corrupt_memory", "no_hypothesis_memory"},
            "social_enabled": self.mode != "corrupt_memory",
            "intrinsic_enabled": self.mode not in {"corrupt_drive", "no_novelty_drive"},
        }
        if self.mode == "shuffled_z":
            tensors["z"] = -tensors["z"].roll(7, dims=-1)
        if self.mode == "corrupt_memory":
            tensors = dict(tensors)
            tensors["hypothesis"] = torch.zeros_like(tensors["hypothesis"])
            tensors["skill_memory"] = torch.zeros_like(tensors["skill_memory"])
            tensors["project_state"] = torch.zeros_like(tensors["project_state"])
            tensors["social_state"] = torch.zeros_like(tensors["social_state"])
        with torch.no_grad():
            output = self.explorer.forward(tensors, **ablations)
        behavior = output["behavior"]
        semantic_action = int(behavior["action"].view(-1)[0].item())
        scoring_memory = self.memory.corrupted() if self.mode == "corrupt_memory" else self.memory
        scoring_features = dict(feature_state)
        if self.mode in {"no_hypothesis_memory", "no_planner_imagination"}:
            scoring_memory = ArcAdapterMemory()
            scoring_memory.update_observation(observation)
            scoring_features["target"] = None
            scoring_features["hypothesis_state"] = {
                "semantic_action": ACTION_INSPECT,
                "target": None,
                "need_key": False,
                "have_key": False,
                "confidence": 0.0,
            }
            semantic_action = ACTION_INSPECT
        if self.mode in {"corrupt_drive", "no_novelty_drive"}:
            semantic_action = ACTION_WAIT
        action_scores = score_legal_actions(
            observation,
            scoring_memory,
            semantic_action,
            scoring_features,
        )
        legal = tuple(observation.available_actions)
        chosen = max(legal, key=lambda action: (action_scores.get(action, -1.0e9), -legal.index(action)))
        if self.mode == "zero_z" and "wait" in legal:
            chosen = "wait"
        self.memory.previous_semantic_action = semantic_action
        diagnostics = {
            "semantic_action": semantic_action,
            "action_scores": action_scores,
            "behavior": {key: int(value.view(-1)[0].item()) for key, value in behavior.items()},
            "memory_recall": scoring_memory.recall(),
            "drive": scoring_features["drive"],
            "hypothesis_state": scoring_features["hypothesis_state"],
            "causal_trace": output["trace"].__dict__,
        }
        return chosen, diagnostics

    def observe_transition(self, action: str, result: ArcAGI3StepResult) -> None:
        self.memory.update_transition(action, result)

    def tensorize(self, observation: ArcAGI3Observation) -> tuple[dict[str, torch.Tensor], dict[str, Any]]:
        feature_state = public_feature_state(observation, self.memory)
        semantic = int(feature_state["semantic_action"])
        rule_id = int(feature_state["rule_id"])
        project_id = int(feature_state["project_id"])
        skill_id = int(feature_state["skill_id"])
        partner_id = int(feature_state["partner_id"])
        device = self.device
        obs_scalars = torch.tensor(
            [[
                float(feature_state["visible_target"]),
                float(feature_state["blocked"]),
                float(feature_state["novelty"]),
                float(feature_state["conflict"]),
                float(feature_state["uncertainty"]),
                float(feature_state["restart"]),
                float(feature_state["body_energy"]),
                float(feature_state["confidence"]),
                float(feature_state["hazard_level"]) / 3.0,
                float(feature_state["resource_level"]) / 3.0,
            ]],
            dtype=torch.float32,
            device=device,
        )
        obs = torch.cat(
            [
                one_hot(feature_state["object_id"], NUM_OBJECTS, device),
                one_hot(feature_state["tool_id"], NUM_TOOLS, device),
                one_hot(feature_state["door_state"], NUM_DOORS, device),
                one_hot(feature_state["hazard_level"], NUM_HAZARDS, device),
                one_hot(feature_state["resource_level"], NUM_RESOURCES, device),
                one_hot(feature_state["agent_id"], NUM_AGENTS, device),
                one_hot(rule_id, NUM_RULES, device),
                one_hot(feature_state["source_id"], NUM_SOURCES, device),
                one_hot(feature_state["location_id"], NUM_LOCATIONS, device),
                one_hot(feature_state["phase_id"], NUM_PHASES, device),
                one_hot(semantic, NUM_EXPLORER_ACTIONS, device),
                obs_scalars,
            ],
            dim=-1,
        )
        hypothesis = torch.cat(
            [
                one_hot(rule_id, NUM_RULES, device),
                one_hot(feature_state["source_id"], NUM_SOURCES, device),
                torch.tensor(
                    [[
                        feature_state["confidence"],
                        feature_state["conflict"],
                        feature_state["visible_target"],
                        feature_state["project_step"] / 3.0,
                        feature_state["novelty"],
                        feature_state["blocked"],
                    ]],
                    dtype=torch.float32,
                    device=device,
                ),
            ],
            dim=-1,
        )
        skill_memory = torch.cat(
            [
                one_hot(skill_id, NUM_SKILLS, device),
                torch.tensor(
                    [[
                        float(feature_state["transfer"]),
                        feature_state["tool_id"] / max(NUM_TOOLS - 1, 1),
                        feature_state["hazard_level"] / 3.0,
                        feature_state["confidence"],
                    ]],
                    dtype=torch.float32,
                    device=device,
                ),
            ],
            dim=-1,
        )
        project_state = torch.cat(
            [
                one_hot(project_id, NUM_PROJECTS, device),
                torch.tensor(
                    [[
                        feature_state["project_step"] / 3.0,
                        feature_state["restart"],
                        float(not feature_state["have_key"]),
                        float(feature_state["hazard_level"] >= 2),
                        min(observation.step_index / max(1, observation.extras.get("max_steps", 32)), 1.0),
                    ]],
                    dtype=torch.float32,
                    device=device,
                ),
            ],
            dim=-1,
        )
        social_state = torch.cat(
            [
                one_hot(partner_id, NUM_AGENTS + 3, device)[:, :NUM_AGENTS + 3],
                torch.tensor(
                    [[
                        feature_state["uncertainty"],
                        feature_state["conflict"],
                        feature_state["confidence"],
                        feature_state["source_id"] / max(NUM_SOURCES - 1, 1),
                        feature_state["restart"],
                        feature_state["visible_target"],
                    ]],
                    dtype=torch.float32,
                    device=device,
                ),
            ],
            dim=-1,
        )
        if social_state.shape[-1] != SOCIAL_STATE_DIM:
            social_state = F.pad(social_state, (0, max(0, SOCIAL_STATE_DIM - social_state.shape[-1])))[:, :SOCIAL_STATE_DIM]
        z = _category_latent(
            torch.tensor([rule_id], dtype=torch.long, device=device),
            torch.tensor([project_id], dtype=torch.long, device=device),
            torch.tensor([skill_id], dtype=torch.long, device=device),
            torch.tensor([partner_id], dtype=torch.long, device=device),
            0,
            device,
        )
        intrinsic = torch.tensor(
            [[
                feature_state["confidence"],
                feature_state["conflict"],
                feature_state["novelty"],
                feature_state["blocked"],
                feature_state["hazard_level"] / 3.0,
                feature_state["resource_level"] / 3.0,
                feature_state["body_energy"],
                feature_state["project_step"] / 3.0,
                feature_state["restart"],
                feature_state["uncertainty"],
            ]],
            dtype=torch.float32,
            device=device,
        )
        if intrinsic.shape[-1] != INTRINSIC_DIM:
            intrinsic = F.pad(intrinsic, (0, max(0, INTRINSIC_DIM - intrinsic.shape[-1])))[:, :INTRINSIC_DIM]
        tensors = {
            "obs": obs[:, :OBS_DIM],
            "z": z[:, :Z_DIM],
            "hypothesis": hypothesis[:, :HYPOTHESIS_DIM],
            "skill_memory": skill_memory[:, :SKILL_MEMORY_DIM],
            "project_state": project_state[:, :PROJECT_STATE_DIM],
            "social_state": social_state[:, :SOCIAL_STATE_DIM],
            "intrinsic": intrinsic[:, :INTRINSIC_DIM],
        }
        return tensors, feature_state


def public_feature_state(observation: ArcAGI3Observation, memory: ArcAdapterMemory) -> dict[str, Any]:
    grid = np.asarray(observation.grid, dtype=np.int64)
    agent = find_agent(grid)
    goals = set(positions_of(grid, CELL_GOAL)) | memory.remembered["goal"]
    keys = set(positions_of(grid, CELL_KEY)) | memory.remembered["key"]
    doors = set(positions_of(grid, CELL_DOOR)) | memory.remembered["door"]
    hazards = set(positions_of(grid, CELL_HAZARD)) | memory.remembered["hazard"]
    resources = set(positions_of(grid, CELL_RESOURCE)) | memory.remembered["resource"]
    unknowns = set(positions_of(grid, CELL_UNKNOWN))
    have_key = bool(observation.extras.get("have_key", memory.have_key))
    need_key = bool(observation.extras.get("public_rules", {}).get("requires_key", False))
    adjacent_hazard = agent is not None and any(manhattan(agent, hazard) <= 1 for hazard in hazards)
    visible_target = bool(positions_of(grid, CELL_GOAL) or positions_of(grid, CELL_KEY) or positions_of(grid, CELL_RESOURCE))
    if adjacent_hazard:
        semantic = ACTION_AVOID
    elif need_key and not have_key and keys:
        semantic = ACTION_FORAGE
    elif resources and not goals:
        semantic = ACTION_FORAGE
    elif doors and have_key and positions_of(grid, CELL_DOOR):
        semantic = ACTION_OPEN
    elif goals:
        semantic = ACTION_RETURN
    elif unknowns:
        semantic = ACTION_INSPECT
    else:
        semantic = ACTION_TEST
    hazard_level = min(3, len([hazard for hazard in hazards if agent is None or manhattan(agent, hazard) <= 3]))
    resource_level = min(3, len(resources) + int(bool(keys and not have_key)))
    target = nearest_position(agent, keys if need_key and not have_key else goals or resources or unknowns)
    project_step = 0 if not have_key and keys else 1 if doors else 2 if goals else 3
    confidence = 0.95 if target is not None else 0.45
    novelty = float(any(item not in memory.visited for item in unknowns) or observation.step_index < 2)
    blocked = float(agent is None)
    rule_id = semantic % NUM_RULES
    return {
        "semantic_action": semantic,
        "object_id": int(CELL_GOAL in grid) % NUM_OBJECTS,
        "tool_id": min(NUM_TOOLS - 1, int(bool(keys)) + int(have_key) * 2),
        "door_state": min(NUM_DOORS - 1, int(bool(doors)) + int(have_key) + int(bool(observation.extras.get("door_open", False)))),
        "hazard_level": hazard_level,
        "resource_level": resource_level,
        "agent_id": 0,
        "rule_id": rule_id,
        "source_id": 0,
        "location_id": 0 if target is None or agent is None else min(NUM_LOCATIONS - 1, manhattan(agent, target)),
        "phase_id": int(observation.step_index) % NUM_PHASES,
        "visible_target": float(visible_target),
        "blocked": blocked,
        "novelty": float(novelty),
        "conflict": 0.0,
        "uncertainty": float(target is None),
        "restart": 0.0,
        "body_energy": max(0.1, 1.0 - 0.04 * observation.step_index - 0.10 * hazard_level),
        "confidence": confidence,
        "project_id": semantic % NUM_PROJECTS,
        "project_step": project_step,
        "skill_id": semantic % NUM_SKILLS,
        "partner_id": 0,
        "transfer": 1.0,
        "have_key": have_key,
        "target": target,
        "drive": {
            "novelty": float(novelty),
            "hazard_level": hazard_level,
            "resource_level": resource_level,
            "body_energy": max(0.1, 1.0 - 0.04 * observation.step_index - 0.10 * hazard_level),
        },
        "hypothesis_state": {
            "semantic_action": semantic,
            "target": target,
            "need_key": need_key,
            "have_key": have_key,
            "confidence": confidence,
        },
    }


def score_legal_actions(
    observation: ArcAGI3Observation,
    memory: ArcAdapterMemory,
    semantic_action: int,
    feature_state: dict[str, Any],
) -> dict[str, float]:
    grid = np.asarray(observation.grid, dtype=np.int64)
    agent = find_agent(grid)
    scores: dict[str, float] = {}
    legal = tuple(observation.available_actions)
    goals = set(positions_of(grid, CELL_GOAL)) | memory.remembered["goal"]
    keys = set(positions_of(grid, CELL_KEY)) | memory.remembered["key"]
    doors = set(positions_of(grid, CELL_DOOR)) | memory.remembered["door"]
    hazards = set(positions_of(grid, CELL_HAZARD)) | memory.remembered["hazard"]
    resources = set(positions_of(grid, CELL_RESOURCE)) | memory.remembered["resource"]
    unknowns = set(positions_of(grid, CELL_UNKNOWN))
    need_key = bool(observation.extras.get("public_rules", {}).get("requires_key", False))
    have_key = bool(observation.extras.get("have_key", memory.have_key))
    if semantic_action == ACTION_FORAGE:
        targets = keys if need_key and not have_key else resources or keys
    elif semantic_action in {ACTION_OPEN, ACTION_RETURN}:
        targets = doors if semantic_action == ACTION_OPEN and have_key and doors else goals
    elif semantic_action == ACTION_AVOID:
        targets = hazards
    elif semantic_action in {ACTION_INSPECT, ACTION_TEST}:
        targets = unknowns or (goals | keys | resources)
    elif semantic_action in {ACTION_WAIT, ACTION_REFUSE}:
        targets = set()
    else:
        targets = goals or keys or resources or unknowns
    target = feature_state.get("target") or nearest_position(agent, targets)
    for action in legal:
        family = action_family(action)
        score = -0.05
        if family == "wait":
            score = 0.1 if semantic_action in {ACTION_WAIT, ACTION_REFUSE} else -0.2
        elif family == "move" and agent is not None:
            dy, dx = MOVE_DELTAS[action]
            ny, nx = agent[0] + dy, agent[1] + dx
            if not in_bounds(grid, ny, nx):
                score = -0.4
            else:
                cell = int(grid[ny, nx])
                if cell in {CELL_WALL, CELL_DOOR} and not (cell == CELL_DOOR and have_key):
                    score = -0.35
                else:
                    next_pos = (ny, nx)
                    if semantic_action == ACTION_AVOID:
                        current_hazard = nearest_distance(agent, hazards)
                        next_hazard = nearest_distance(next_pos, hazards)
                        score = float(next_hazard - current_hazard) + (0.5 if cell != CELL_HAZARD else -1.0)
                        if target is not None:
                            score += 0.35 * float(manhattan(agent, target) - manhattan(next_pos, target))
                        if next_pos not in memory.visited:
                            score += 0.35
                    elif target is not None:
                        score = float(manhattan(agent, target) - manhattan(next_pos, target))
                    if next_pos not in memory.visited and semantic_action in {ACTION_INSPECT, ACTION_TEST}:
                        score += 0.6
                    if cell == CELL_KEY and semantic_action == ACTION_FORAGE:
                        score += 2.0
                    if cell == CELL_DOOR and semantic_action == ACTION_OPEN:
                        score += 1.6
                    if cell == CELL_GOAL and semantic_action in {ACTION_RETURN, ACTION_OPEN}:
                        score += 3.0
                    if cell == CELL_HAZARD:
                        score -= 2.0
        elif family == "click":
            click = parse_click(action)
            if click is not None:
                x, y = click
                if in_bounds(grid, y, x):
                    cell = int(grid[y, x])
                    click_pos = (y, x)
                    score = -0.1
                    if semantic_action in {ACTION_RETURN, ACTION_OPEN, ACTION_TEST} and cell == CELL_GOAL:
                        score = 3.0
                    elif semantic_action in {ACTION_FORAGE, ACTION_TEST} and cell in {CELL_KEY, CELL_RESOURCE}:
                        score = 1.2
                    elif target is not None:
                        score = -0.01 * manhattan(click_pos, target)
        scores[action] = float(score)
    return scores


def nearest_distance(origin: tuple[int, int], targets: Iterable[tuple[int, int]]) -> int:
    items = list(targets)
    if not items:
        return 0
    return min(manhattan(origin, item) for item in items)


def manhattan(left: tuple[int, int], right: tuple[int, int]) -> int:
    return abs(int(left[0]) - int(right[0])) + abs(int(left[1]) - int(right[1]))


def in_bounds(grid: np.ndarray, y: int, x: int) -> bool:
    return y >= 0 and x >= 0 and y < grid.shape[0] and x < grid.shape[1]
