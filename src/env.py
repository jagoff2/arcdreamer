from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List

import torch
import torch.nn.functional as F

from .device import AUTO_DEVICE, DeviceLike, make_generator, resolve_device


GRID_SIZE = 5
NUM_COLORS = 4

ACTION_STAY = 0
ACTION_LEFT = 1
ACTION_RIGHT = 2
ACTION_FORAGE = 3
ACTION_REST = 4
NUM_ACTIONS = 5

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
TOK_CURRICULUM_ALIAS = 12
NUM_INPUT_TOKENS = 13

PRIVATE_NONE = 0
PRIVATE_GOAL_COLOR = 1
PRIVATE_GOAL_POS = PRIVATE_GOAL_COLOR + NUM_COLORS
PRIVATE_BODY_LOW = PRIVATE_GOAL_POS + GRID_SIZE
PRIVATE_BODY_STABLE = PRIVATE_BODY_LOW + 1
PRIVATE_IDLE_BASE = PRIVATE_BODY_STABLE + 1
NUM_IDLE_PRIVATE_TOKENS = 6
NUM_PRIVATE_TOKENS = PRIVATE_IDLE_BASE + NUM_IDLE_PRIVATE_TOKENS

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
ANS_IDLE = ANS_IMAGINED + NUM_COLORS
NUM_IDLE_LANGUAGE_TOKENS = 8
ANS_CURRICULUM = ANS_IDLE + NUM_IDLE_LANGUAGE_TOKENS
NUM_CURRICULUM_CONCEPTS = 3
NUM_LANGUAGE_TOKENS = ANS_CURRICULUM + NUM_CURRICULUM_CONCEPTS

NUM_BODY_SCALARS = 4
BODY_ENERGY = 0
BODY_FATIGUE = 1
BODY_DAMAGE = 2
BODY_RESOURCE = 3
SENSOR_DIM = GRID_SIZE + 2 + NUM_BODY_SCALARS + (NUM_COLORS + 1) + 1 + (GRID_SIZE + 1)

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


def damage_bin(damage: torch.Tensor) -> torch.Tensor:
    return torch.clamp((damage * 3.0).long(), 0, 2)


def living_policy(
    current_pos: torch.Tensor,
    target_pos: torch.Tensor,
    energy: torch.Tensor,
    damage: torch.Tensor,
    resource: torch.Tensor,
) -> torch.Tensor:
    base = shortest_action(current_pos, target_pos)
    needs_rest = (energy < 0.16) | (damage > 0.70)
    can_forage = (energy < 0.30) & (resource < 0.25) & (~needs_rest)
    action = torch.where(can_forage, torch.full_like(base, ACTION_FORAGE), base)
    action = torch.where(needs_rest, torch.full_like(base, ACTION_REST), action)
    return action


def build_sensory(
    current_pos: torch.Tensor,
    orientation: torch.Tensor,
    energy: torch.Tensor,
    fatigue: torch.Tensor,
    damage: torch.Tensor,
    resource: torch.Tensor,
    visible_color: torch.Tensor,
    visible_object_pos: torch.Tensor,
) -> torch.Tensor:
    batch_size = current_pos.shape[0]
    pos_oh = F.one_hot(current_pos, GRID_SIZE).float()
    orient_oh = F.one_hot(orientation, 2).float()
    body = torch.stack([energy, fatigue, damage, resource], dim=-1).float()
    color_oh = F.one_hot(visible_color, NUM_COLORS + 1).float()
    obj_pos_oh = F.one_hot(visible_object_pos, GRID_SIZE + 1).float()
    visible_flag = (visible_color != NUM_COLORS).float().view(batch_size, 1)
    return torch.cat([pos_oh, orient_oh, body, color_oh, visible_flag, obj_pos_oh], dim=-1)


def idle_language_target(
    tick: int,
    target_color: torch.Tensor,
    target_pos: torch.Tensor,
    current_pos: torch.Tensor,
    energy: torch.Tensor,
    damage: torch.Tensor,
) -> torch.Tensor:
    phase = torch.full_like(target_color, tick % NUM_IDLE_LANGUAGE_TOKENS)
    return ANS_IDLE + phase


def private_target(
    tick: int,
    token: int,
    target_color: torch.Tensor,
    target_pos: torch.Tensor,
    energy: torch.Tensor,
    damage: torch.Tensor,
) -> torch.Tensor:
    idle = PRIVATE_IDLE_BASE + ((tick + target_color + target_pos) % NUM_IDLE_PRIVATE_TOKENS)
    color_token = PRIVATE_GOAL_COLOR + target_color
    pos_token = PRIVATE_GOAL_POS + target_pos
    body_token = torch.where(
        (energy < 0.25) | (damage > 0.55),
        torch.full_like(target_color, PRIVATE_BODY_LOW),
        torch.full_like(target_color, PRIVATE_BODY_STABLE),
    )
    if token in (TOK_OBSERVE_OBJECT, TOK_ASK_COLOR, TOK_ASK_GOAL, TOK_TOLD_GOAL):
        return color_token
    if token in (TOK_ASK_OBJECT_POS, TOK_ASK_ACTION, TOK_INFER_OBJECT):
        return pos_token
    if token in (TOK_ASK_ENERGY, TOK_ASK_CURRENT_POS):
        return body_token
    if token == TOK_NONE:
        return idle
    return color_token if tick % 2 == 0 else idle


def language_target(
    token: int,
    target_color: torch.Tensor,
    target_pos: torch.Tensor,
    start_pos: torch.Tensor,
    current_pos: torch.Tensor,
    energy: torch.Tensor,
    action: torch.Tensor,
    damage: torch.Tensor | None = None,
) -> torch.Tensor:
    if damage is None:
        damage = torch.zeros_like(energy)
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
    if token == TOK_CURRICULUM_ALIAS:
        return ANS_CURRICULUM + (target_color % NUM_CURRICULUM_CONCEPTS)
    return idle_language_target(tick=0, target_color=target_color, target_pos=target_pos, current_pos=current_pos, energy=energy, damage=damage)


def provenance_target(token: int, tick: int) -> int:
    if token == TOK_IMAGINE:
        return PROV_IMAGINED
    if token in (TOK_TOLD_GOAL, TOK_INFER_OBJECT, TOK_CURRICULUM_ALIAS):
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
    device: DeviceLike = AUTO_DEVICE,
) -> Dict[str, torch.Tensor]:
    target_device = resolve_device(device)
    generator = make_generator(base_seed, target_device)

    start_pos = torch.randint(0, GRID_SIZE, (batch_size,), generator=generator, device=target_device)
    target_pos = torch.randint(0, GRID_SIZE, (batch_size,), generator=generator, device=target_device)
    target_color = torch.randint(0, NUM_COLORS, (batch_size,), generator=generator, device=target_device)
    hazard_pos = torch.randint(0, GRID_SIZE, (batch_size,), generator=generator, device=target_device)
    start_energy = 0.58 + 0.38 * torch.rand(batch_size, generator=generator, device=target_device)
    start_damage = 0.10 * torch.rand(batch_size, generator=generator, device=target_device)
    start_resource = 0.20 + 0.55 * torch.rand(batch_size, generator=generator, device=target_device)
    walk = torch.randint(0, NUM_ACTIONS, (batch_size, seq_len), generator=generator, device=target_device)

    sensory = torch.zeros(batch_size, seq_len, SENSOR_DIM, device=target_device)
    lang_in = torch.zeros(batch_size, seq_len, dtype=torch.long, device=target_device)
    private_in = torch.zeros(batch_size, seq_len, dtype=torch.long, device=target_device)
    action_target = torch.zeros(batch_size, seq_len, dtype=torch.long, device=target_device)
    language = torch.zeros(batch_size, seq_len, dtype=torch.long, device=target_device)
    private = torch.zeros(batch_size, seq_len, dtype=torch.long, device=target_device)
    provenance = torch.zeros(batch_size, seq_len, dtype=torch.long, device=target_device)
    world_color = torch.zeros(batch_size, seq_len, dtype=torch.long, device=target_device)
    world_pos = torch.zeros(batch_size, seq_len, dtype=torch.long, device=target_device)
    memory_color = torch.zeros(batch_size, seq_len, dtype=torch.long, device=target_device)
    self_start = torch.zeros(batch_size, seq_len, dtype=torch.long, device=target_device)
    prev_action = torch.full((batch_size, seq_len), NUM_ACTIONS, dtype=torch.long, device=target_device)
    prev_delta = torch.zeros(batch_size, seq_len, SENSOR_DIM, device=target_device)
    dt = torch.ones(batch_size, seq_len, dtype=torch.float32, device=target_device)

    action_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=target_device)
    delayed_memory_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=target_device)
    object_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=target_device)
    grounded_language_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=target_device)
    self_mask = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=target_device)

    current_pos = start_pos.clone()
    orientation = torch.randint(0, 2, (batch_size,), generator=generator, device=target_device)
    energy = start_energy.clone()
    damage = start_damage.clone()
    resource = start_resource.clone()
    for tick in range(seq_len):
        token = token_for_tick(tick)
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
            current_pos,
            orientation,
            energy,
            fatigue,
            damage,
            resource,
            visible_color,
            visible_object_pos,
        )
        lang_in[:, tick] = token
        action_target[:, tick] = action
        language[:, tick] = language_target(
            token, target_color, target_pos, start_pos, current_pos, energy, action, damage
        )
        private[:, tick] = private_target(tick, token, target_color, target_pos, energy, damage)
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
        failed = ((move == ACTION_LEFT) | (move == ACTION_RIGHT)) & ((energy < 0.08) | (damage > 0.86))
        can_move = ~failed
        current_pos = torch.where((move == ACTION_LEFT) & can_move, (current_pos - 1) % GRID_SIZE, current_pos)
        current_pos = torch.where((move == ACTION_RIGHT) & can_move, (current_pos + 1) % GRID_SIZE, current_pos)
        orientation = torch.where((move == ACTION_LEFT) & can_move, torch.zeros_like(orientation), orientation)
        orientation = torch.where((move == ACTION_RIGHT) & can_move, torch.ones_like(orientation), orientation)
        at_hazard = current_pos == hazard_pos
        move_cost = torch.where((move == ACTION_LEFT) | (move == ACTION_RIGHT), 0.030, 0.006)
        energy = torch.clamp(energy - move_cost - at_hazard.float() * 0.020, 0.02, 1.0)
        damage = torch.clamp(damage + failed.float() * 0.030 + at_hazard.float() * 0.045, 0.0, 1.0)
        forage = move == ACTION_FORAGE
        rest = move == ACTION_REST
        energy = torch.clamp(energy + forage.float() * 0.070 + rest.float() * 0.045, 0.02, 1.0)
        damage = torch.clamp(damage - rest.float() * 0.030 + forage.float() * 0.004, 0.0, 1.0)
        resource = torch.clamp(resource + forage.float() * 0.090 - rest.float() * 0.006 - 0.004, 0.0, 1.0)

    private_in[:, 1:] = private[:, :-1]

    batch = {
        "sensory": sensory,
        "lang_in": lang_in,
        "private_in": private_in,
        "action_target": action_target,
        "language_target": language,
        "private_target": private,
        "provenance_target": provenance,
        "world_color_target": world_color,
        "world_pos_target": world_pos,
        "memory_color_target": memory_color,
        "self_start_target": self_start,
        "prev_action": prev_action,
        "prev_delta": prev_delta,
        "dt": dt,
        "action_mask": action_mask,
        "delayed_memory_mask": delayed_memory_mask,
        "object_mask": object_mask,
        "grounded_language_mask": grounded_language_mask,
        "self_mask": self_mask,
    }
    if seq_len > 1:
        batch["prev_action"][:, 1:] = walk[:, :-1]
        batch["prev_delta"][:, 1:] = sensory[:, 1:] - sensory[:, :-1]
    return batch


class TinyWorldRuntime:
    def __init__(self, seed: int = 0, episode_len: int = 80) -> None:
        self.seed = seed
        self.episode_len = episode_len
        self.rng_device = resolve_device()
        self.generator = make_generator(seed, self.rng_device)
        self.global_tick = 0
        self.local_tick = 0
        self.start_pos = 0
        self.target_pos = 0
        self.target_color = 0
        self.current_pos = 0
        self.orientation = 0
        self.energy = 1.0
        self.damage = 0.0
        self.resource = 0.5
        self.hazard_pos = 0
        self.failed_actions = 0
        self.damage_events = 0
        self._new_episode()

    def _rand_int(self, high: int) -> int:
        return int(torch.randint(0, high, (1,), generator=self.generator, device=self.rng_device).item())

    def _new_episode(self) -> None:
        self.local_tick = 0
        self.start_pos = self._rand_int(GRID_SIZE)
        self.target_pos = self._rand_int(GRID_SIZE)
        self.target_color = self._rand_int(NUM_COLORS)
        self.current_pos = self.start_pos
        self.orientation = self._rand_int(2)
        self.hazard_pos = self._rand_int(GRID_SIZE)
        self.energy = float(0.58 + 0.38 * torch.rand(1, generator=self.generator, device=self.rng_device).item())
        self.damage = float(0.10 * torch.rand(1, generator=self.generator, device=self.rng_device).item())
        self.resource = float(0.20 + 0.55 * torch.rand(1, generator=self.generator, device=self.rng_device).item())

    def observation(
        self,
        device: DeviceLike = AUTO_DEVICE,
        private_in: int = PRIVATE_NONE,
    ) -> Dict[str, torch.Tensor]:
        target_device = resolve_device(device)
        token = token_for_tick(self.local_tick)
        visible = self.target_color if self.local_tick < 4 else NUM_COLORS
        visible_pos = self.target_pos if self.local_tick < 4 else GRID_SIZE
        current_pos = torch.tensor([self.current_pos], dtype=torch.long, device=target_device)
        orientation = torch.tensor([self.orientation], dtype=torch.long, device=target_device)
        energy = torch.tensor([self.energy], dtype=torch.float32, device=target_device)
        fatigue = 1.0 - energy
        damage = torch.tensor([self.damage], dtype=torch.float32, device=target_device)
        resource = torch.tensor([self.resource], dtype=torch.float32, device=target_device)
        visible_color = torch.tensor([visible], dtype=torch.long, device=target_device)
        visible_object_pos = torch.tensor([visible_pos], dtype=torch.long, device=target_device)
        sensory = build_sensory(
            current_pos,
            orientation,
            energy,
            fatigue,
            damage,
            resource,
            visible_color,
            visible_object_pos,
        )
        return {
            "sensory": sensory,
            "lang_in": torch.tensor([token], dtype=torch.long, device=target_device),
            "private_in": torch.tensor([private_in], dtype=torch.long, device=target_device),
        }

    def expected_action(self) -> int:
        current = torch.tensor([self.current_pos], dtype=torch.long)
        target = torch.tensor([self.target_pos], dtype=torch.long)
        return int(shortest_action(current, target).item())

    def expected_body_action(self) -> int:
        current = torch.tensor([self.current_pos], dtype=torch.long)
        target = torch.tensor([self.target_pos], dtype=torch.long)
        energy = torch.tensor([self.energy], dtype=torch.float32)
        damage = torch.tensor([self.damage], dtype=torch.float32)
        resource = torch.tensor([self.resource], dtype=torch.float32)
        return int(living_policy(current, target, energy, damage, resource).item())

    def step(self, action: int) -> None:
        failed = action in (ACTION_LEFT, ACTION_RIGHT) and (self.energy < 0.08 or self.damage > 0.86)
        if failed:
            self.failed_actions += 1
            self.damage = min(1.0, self.damage + 0.03)
            self.energy = max(0.02, self.energy - 0.01)
        elif action == ACTION_LEFT:
            self.current_pos = (self.current_pos - 1) % GRID_SIZE
            self.orientation = 0
            self.energy = max(0.02, self.energy - 0.030)
            self.resource = max(0.0, self.resource - 0.012)
        elif action == ACTION_RIGHT:
            self.current_pos = (self.current_pos + 1) % GRID_SIZE
            self.orientation = 1
            self.energy = max(0.02, self.energy - 0.030)
            self.resource = max(0.0, self.resource - 0.012)
        elif action == ACTION_FORAGE:
            self.energy = min(1.0, self.energy + 0.070)
            self.resource = min(1.0, self.resource + 0.090)
            self.damage = min(1.0, self.damage + 0.004)
        elif action == ACTION_REST:
            self.energy = min(1.0, self.energy + 0.045)
            self.resource = max(0.0, self.resource - 0.006)
            self.damage = max(0.0, self.damage - 0.030)
        else:
            self.energy = max(0.02, self.energy - 0.006)
            self.resource = max(0.0, self.resource - 0.004)

        if self.current_pos == self.hazard_pos and action != ACTION_REST:
            self.damage_events += 1
            self.damage = min(1.0, self.damage + 0.045)
            self.energy = max(0.02, self.energy - 0.020)
        if self.energy < 0.05:
            self.damage = min(1.0, self.damage + 0.010)

        self.global_tick += 1
        self.local_tick += 1
        if self.local_tick >= self.episode_len:
            self._new_episode()
