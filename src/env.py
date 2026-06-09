from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F


GRID_SIZE = 5
NUM_COLORS = 4
NUM_ACTIONS = 3
ACTION_STAY = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2

TOK_NONE = 0
TOK_OBSERVE_OBJECT = 1
TOK_TOLD_GOAL = 2
TOK_IMAGINE = 3
TOK_INFER_OBJECT = 4
TOK_ASK_COLOR = 5
TOK_ASK_OBJECT_POS = 6
TOK_ASK_START_POS = 7
TOK_ASK_CURRENT_POS = 8
TOK_ASK_ENERGY = 9
TOK_ASK_ACTION = 10
TOK_ASK_GOAL = 11
NUM_INPUT_TOKENS = 12

PROV_OBSERVED = 0
PROV_REMEMBERED = 1
PROV_IMAGINED = 2
PROV_TOLD = 3
NUM_PROVENANCE = 4

ANS_COLOR = 0
ANS_OBJECT_POS = ANS_COLOR + NUM_COLORS
ANS_START_POS = ANS_OBJECT_POS + GRID_SIZE
ANS_CURRENT_POS = ANS_START_POS + GRID_SIZE
ANS_ENERGY = ANS_CURRENT_POS + GRID_SIZE
ANS_ACTION = ANS_ENERGY + 3
ANS_GOAL = ANS_ACTION + NUM_ACTIONS
ANS_IMAGINED = ANS_GOAL + NUM_COLORS
NUM_LANGUAGE_TOKENS = ANS_IMAGINED + NUM_COLORS

SENSOR_DIM = GRID_SIZE + 2 + 2 + (NUM_COLORS + 1) + 1 + (GRID_SIZE + 1)

QUERY_CYCLE: List[int] = [
    TOK_ASK_COLOR,
    TOK_ASK_OBJECT_POS,
    TOK_ASK_START_POS,
    TOK_ASK_CURRENT_POS,
    TOK_ASK_ENERGY,
    TOK_ASK_ACTION,
    TOK_ASK_GOAL,
]


@dataclass(frozen=True)
class WorldConfig:
    seq_len: int = 80
    grid_size: int = GRID_SIZE
    num_colors: int = NUM_COLORS


def token_for_tick(tick: int) -> int:
    if tick == 0:
        return TOK_OBSERVE_OBJECT
    if tick == 8:
        return TOK_TOLD_GOAL
    if tick % 29 == 11:
        return TOK_IMAGINE
    if tick % 31 == 17:
        return TOK_INFER_OBJECT
    return QUERY_CYCLE[tick % len(QUERY_CYCLE)]


def shortest_action(current_pos: torch.Tensor, target_pos: torch.Tensor) -> torch.Tensor:
    delta = (target_pos - current_pos) % GRID_SIZE
    left_delta = (current_pos - target_pos) % GRID_SIZE
    action = torch.full_like(current_pos, ACTION_STAY)
    action = torch.where(delta == 0, action, torch.full_like(action, ACTION_RIGHT))
    action = torch.where((left_delta < delta) & (left_delta > 0), torch.full_like(action, ACTION_LEFT), action)
    return action


def energy_bin(energy: torch.Tensor) -> torch.Tensor:
    return torch.clamp((energy * 3.0).long(), 0, 2)


def build_sensory(
    current_pos: torch.Tensor,
    orientation: torch.Tensor,
    energy: torch.Tensor,
    fatigue: torch.Tensor,
    visible_color: torch.Tensor,
    visible_object_pos: torch.Tensor,
) -> torch.Tensor:
    batch_size = current_pos.shape[0]
    pos_oh = F.one_hot(current_pos, GRID_SIZE).float()
    orient_oh = F.one_hot(orientation, 2).float()
    body = torch.stack([energy, fatigue], dim=-1).float()
    color_oh = F.one_hot(visible_color, NUM_COLORS + 1).float()
    obj_pos_oh = F.one_hot(visible_object_pos, GRID_SIZE + 1).float()
    visible_flag = (visible_color != NUM_COLORS).float().view(batch_size, 1)
    return torch.cat([pos_oh, orient_oh, body, color_oh, visible_flag, obj_pos_oh], dim=-1)


def language_target(
    token: int,
    target_color: torch.Tensor,
    target_pos: torch.Tensor,
    start_pos: torch.Tensor,
    current_pos: torch.Tensor,
    energy: torch.Tensor,
    action: torch.Tensor,
) -> torch.Tensor:
    if token in (TOK_OBSERVE_OBJECT, TOK_ASK_COLOR):
        return ANS_COLOR + target_color
    if token == TOK_ASK_OBJECT_POS or token == TOK_INFER_OBJECT:
        return ANS_OBJECT_POS + target_pos
    if token == TOK_ASK_START_POS:
        return ANS_START_POS + start_pos
    if token == TOK_ASK_CURRENT_POS:
        return ANS_CURRENT_POS + current_pos
    if token == TOK_ASK_ENERGY:
        return ANS_ENERGY + energy_bin(energy)
    if token == TOK_ASK_ACTION:
        return ANS_ACTION + action
    if token == TOK_TOLD_GOAL or token == TOK_ASK_GOAL:
        return ANS_GOAL + target_color
    if token == TOK_IMAGINE:
        return ANS_IMAGINED + ((target_color + 1) % NUM_COLORS)
    return ANS_COLOR + target_color


def provenance_target(token: int, tick: int) -> int:
    if token == TOK_IMAGINE:
        return PROV_IMAGINED
    if token in (TOK_TOLD_GOAL, TOK_INFER_OBJECT):
        return PROV_TOLD
    if tick >= 64 and token in (
        TOK_ASK_COLOR,
        TOK_ASK_OBJECT_POS,
        TOK_ASK_START_POS,
        TOK_ASK_ACTION,
        TOK_ASK_GOAL,
    ):
        return PROV_REMEMBERED
    return PROV_OBSERVED


def generate_batch(
    batch_size: int,
    seq_len: int = 80,
    base_seed: int = 0,
    device: torch.device | str = "cpu",
) -> Dict[str, torch.Tensor]:
    generator = torch.Generator(device="cpu")
    generator.manual_seed(base_seed)

    start_pos = torch.randint(0, GRID_SIZE, (batch_size,), generator=generator)
    target_pos = torch.randint(0, GRID_SIZE, (batch_size,), generator=generator)
    target_color = torch.randint(0, NUM_COLORS, (batch_size,), generator=generator)
    start_energy = 0.68 + 0.30 * torch.rand(batch_size, generator=generator)
    walk = torch.randint(0, NUM_ACTIONS, (batch_size, seq_len), generator=generator)

    sensory = torch.zeros(batch_size, seq_len, SENSOR_DIM)
    lang_in = torch.zeros(batch_size, seq_len, dtype=torch.long)
    action_target = torch.zeros(batch_size, seq_len, dtype=torch.long)
    language = torch.zeros(batch_size, seq_len, dtype=torch.long)
    provenance = torch.zeros(batch_size, seq_len, dtype=torch.long)
    world_color = torch.zeros(batch_size, seq_len, dtype=torch.long)
    world_pos = torch.zeros(batch_size, seq_len, dtype=torch.long)
    memory_color = torch.zeros(batch_size, seq_len, dtype=torch.long)
    self_start = torch.zeros(batch_size, seq_len, dtype=torch.long)

    action_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    delayed_memory_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    object_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    grounded_language_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)
    self_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool)

    current_pos = start_pos.clone()
    orientation = torch.randint(0, 2, (batch_size,), generator=generator)
    for tick in range(seq_len):
        token = token_for_tick(tick)
        energy = torch.clamp(start_energy - 0.0045 * tick, 0.05, 1.0)
        fatigue = 1.0 - energy
        visible_color = torch.where(
            torch.full_like(target_color, tick < 4, dtype=torch.bool),
            target_color,
            torch.full_like(target_color, NUM_COLORS),
        )
        visible_object_pos = torch.where(
            torch.full_like(target_pos, tick < 4, dtype=torch.bool),
            target_pos,
            torch.full_like(target_pos, GRID_SIZE),
        )
        action = shortest_action(current_pos, target_pos)

        sensory[:, tick, :] = build_sensory(
            current_pos, orientation, energy, fatigue, visible_color, visible_object_pos
        )
        lang_in[:, tick] = token
        action_target[:, tick] = action
        language[:, tick] = language_target(
            token, target_color, target_pos, start_pos, current_pos, energy, action
        )
        provenance[:, tick] = provenance_target(token, tick)
        world_color[:, tick] = target_color
        world_pos[:, tick] = target_pos
        memory_color[:, tick] = target_color
        self_start[:, tick] = start_pos

        action_mask[:, tick] = True
        if tick >= 64 and token in (TOK_ASK_COLOR, TOK_ASK_OBJECT_POS, TOK_ASK_GOAL):
            delayed_memory_mask[:, tick] = True
        if tick >= 4:
            object_mask[:, tick] = True
        if tick >= 64 and token in (
            TOK_ASK_COLOR,
            TOK_ASK_OBJECT_POS,
            TOK_ASK_START_POS,
            TOK_ASK_ACTION,
            TOK_ASK_GOAL,
        ):
            grounded_language_mask[:, tick] = True
        if tick >= 64 and token == TOK_ASK_START_POS:
            self_mask[:, tick] = True

        move = walk[:, tick]
        current_pos = torch.where(move == ACTION_LEFT, (current_pos - 1) % GRID_SIZE, current_pos)
        current_pos = torch.where(move == ACTION_RIGHT, (current_pos + 1) % GRID_SIZE, current_pos)
        orientation = torch.where(move == ACTION_LEFT, torch.zeros_like(orientation), orientation)
        orientation = torch.where(move == ACTION_RIGHT, torch.ones_like(orientation), orientation)

    batch = {
        "sensory": sensory,
        "lang_in": lang_in,
        "action_target": action_target,
        "language_target": language,
        "provenance_target": provenance,
        "world_color_target": world_color,
        "world_pos_target": world_pos,
        "memory_color_target": memory_color,
        "self_start_target": self_start,
        "action_mask": action_mask,
        "delayed_memory_mask": delayed_memory_mask,
        "object_mask": object_mask,
        "grounded_language_mask": grounded_language_mask,
        "self_mask": self_mask,
    }
    return {key: value.to(device) for key, value in batch.items()}


class TinyWorldRuntime:
    def __init__(self, seed: int = 0, episode_len: int = 80) -> None:
        self.seed = seed
        self.episode_len = episode_len
        self.generator = torch.Generator(device="cpu")
        self.generator.manual_seed(seed)
        self.global_tick = 0
        self.local_tick = 0
        self.start_pos = 0
        self.target_pos = 0
        self.target_color = 0
        self.current_pos = 0
        self.orientation = 0
        self.energy = 1.0
        self._new_episode()

    def _rand_int(self, high: int) -> int:
        return int(torch.randint(0, high, (1,), generator=self.generator).item())

    def _new_episode(self) -> None:
        self.local_tick = 0
        self.start_pos = self._rand_int(GRID_SIZE)
        self.target_pos = self._rand_int(GRID_SIZE)
        self.target_color = self._rand_int(NUM_COLORS)
        self.current_pos = self.start_pos
        self.orientation = self._rand_int(2)
        self.energy = float(0.68 + 0.30 * torch.rand(1, generator=self.generator).item())

    def observation(self, device: torch.device | str = "cpu") -> Dict[str, torch.Tensor]:
        token = token_for_tick(self.local_tick)
        visible = self.target_color if self.local_tick < 4 else NUM_COLORS
        visible_pos = self.target_pos if self.local_tick < 4 else GRID_SIZE
        current_pos = torch.tensor([self.current_pos], dtype=torch.long)
        orientation = torch.tensor([self.orientation], dtype=torch.long)
        energy = torch.tensor([self.energy], dtype=torch.float32)
        fatigue = 1.0 - energy
        visible_color = torch.tensor([visible], dtype=torch.long)
        visible_object_pos = torch.tensor([visible_pos], dtype=torch.long)
        sensory = build_sensory(
            current_pos, orientation, energy, fatigue, visible_color, visible_object_pos
        )
        return {
            "sensory": sensory.to(device),
            "lang_in": torch.tensor([token], dtype=torch.long, device=device),
        }

    def expected_action(self) -> int:
        current = torch.tensor([self.current_pos], dtype=torch.long)
        target = torch.tensor([self.target_pos], dtype=torch.long)
        return int(shortest_action(current, target).item())

    def step(self, action: int) -> None:
        if action == ACTION_LEFT:
            self.current_pos = (self.current_pos - 1) % GRID_SIZE
            self.orientation = 0
            self.energy = max(0.05, self.energy - 0.006)
        elif action == ACTION_RIGHT:
            self.current_pos = (self.current_pos + 1) % GRID_SIZE
            self.orientation = 1
            self.energy = max(0.05, self.energy - 0.006)
        else:
            self.energy = max(0.05, self.energy - 0.002)
        self.global_tick += 1
        self.local_tick += 1
        if self.local_tick >= self.episode_len:
            self._new_episode()
