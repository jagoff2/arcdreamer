from __future__ import annotations

import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F

from .device import AUTO_DEVICE, DeviceLike, make_generator, resolve_device
from .env import generate_batch
from .intrinsic_motivation import INTRINSIC_DIM, build_intrinsic_vector
from .model import load_checkpoint


NUM_OBJECTS = 5
NUM_TOOLS = 5
NUM_DOORS = 4
NUM_HAZARDS = 4
NUM_RESOURCES = 4
NUM_AGENTS = 3
NUM_RULES = 6
NUM_SOURCES = 5
NUM_LOCATIONS = 5
NUM_PHASES = 6
NUM_EXPLORER_ACTIONS = 12
NUM_WORLD_STATES = 18
NUM_UNCERTAINTY = 4
NUM_SKILLS = 8
NUM_PROJECTS = 4
NUM_PARTNERS = 6
NUM_MIND_ACTS = 7

ACTION_INSPECT = 0
ACTION_TEST = 1
ACTION_OPEN = 2
ACTION_AVOID = 3
ACTION_FORAGE = 4
ACTION_RETURN = 5
ACTION_ASK = 6
ACTION_COMPARE = 7
ACTION_EXPLAIN = 8
ACTION_WAIT = 9
ACTION_REFUSE = 10
ACTION_SELF_CORRECT = 11

MIND_ACT = 0
MIND_WAIT = 1
MIND_ASK = 2
MIND_REFUSE = 3
MIND_EXPLORE = 4
MIND_REPORT_UNCERTAINTY = 5
MIND_SELF_CORRECT = 6

Z_DIM = 64
OBS_DIM = (
    NUM_OBJECTS
    + NUM_TOOLS
    + NUM_DOORS
    + NUM_HAZARDS
    + NUM_RESOURCES
    + NUM_AGENTS
    + NUM_RULES
    + NUM_SOURCES
    + NUM_LOCATIONS
    + NUM_PHASES
    + NUM_EXPLORER_ACTIONS
    + 10
)
HYPOTHESIS_DIM = NUM_RULES + NUM_SOURCES + 6
SKILL_MEMORY_DIM = NUM_SKILLS + 4
PROJECT_STATE_DIM = NUM_PROJECTS + 5
SOCIAL_STATE_DIM = NUM_PARTNERS + 6

FROZEN_CORE = "frozen/recurrent_latent_fast.pt"


@dataclass(frozen=True)
class ExplorerConfig:
    name: str
    train_records: int
    eval_records: int
    steps: int
    batch_size: int
    lr: float
    hidden_dim: int
    seed: int
    long_ticks: int


EXPLORER_CONFIGS: dict[str, ExplorerConfig] = {
    "smoke": ExplorerConfig(
        name="smoke",
        train_records=1536,
        eval_records=768,
        steps=35,
        batch_size=192,
        lr=3.0e-3,
        hidden_dim=96,
        seed=8110,
        long_ticks=2048,
    ),
    "tiny": ExplorerConfig(
        name="tiny",
        train_records=32768,
        eval_records=16384,
        steps=420,
        batch_size=512,
        lr=2.5e-3,
        hidden_dim=128,
        seed=9310,
        long_ticks=16384,
    ),
}


@dataclass
class ExplorerDataset:
    tensors: dict[str, torch.Tensor]
    metadata: dict[str, Any]

    def to(self, device: DeviceLike) -> "ExplorerDataset":
        target_device = resolve_device(device)
        return ExplorerDataset(
            {key: value.to(target_device) for key, value in self.tensors.items()},
            dict(self.metadata),
        )


def explorer_config(config_name: str) -> ExplorerConfig:
    if config_name not in EXPLORER_CONFIGS:
        raise KeyError(f"unknown explorer config {config_name}")
    return EXPLORER_CONFIGS[config_name]


def _one_hot(values: torch.Tensor, classes: int) -> torch.Tensor:
    return F.one_hot(values.long(), classes).float()


def _latent_from_frozen_core(
    record_count: int,
    seed: int,
    device: torch.device,
    checkpoint: str | Path = FROZEN_CORE,
) -> torch.Tensor:
    checkpoint_path = Path(checkpoint)
    if not checkpoint_path.exists():
        generator = make_generator(seed + 500, device)
        return torch.randn(record_count, Z_DIM, generator=generator, device=device) * 0.10
    seq_len = 80
    sessions = max(1, math.ceil(record_count / seq_len))
    batch = generate_batch(sessions, seq_len=seq_len, base_seed=seed + 700, device=device)
    model = load_checkpoint(checkpoint_path, device=device)
    model.eval()
    with torch.no_grad():
        out = model(batch["sensory"], batch["lang_in"], batch["private_in"])
    z = out["latents"].reshape(-1, out["latents"].shape[-1])[:record_count].detach()
    if z.shape[-1] < Z_DIM:
        z = F.pad(z, (0, Z_DIM - z.shape[-1]))
    return z[:, :Z_DIM].contiguous()


def _category_latent(
    rule: torch.Tensor,
    project: torch.Tensor,
    skill: torch.Tensor,
    partner: torch.Tensor,
    seed: int,
    device: torch.device,
) -> torch.Tensor:
    generator = make_generator(seed + 900, device)
    table = torch.randn(NUM_RULES + NUM_PROJECTS + NUM_SKILLS + NUM_PARTNERS, Z_DIM, generator=generator, device=device)
    table = F.normalize(table, dim=-1)
    p0 = NUM_RULES
    s0 = p0 + NUM_PROJECTS
    a0 = s0 + NUM_SKILLS
    return 0.16 * (table[rule] + table[p0 + project] + table[s0 + skill] + table[a0 + partner])


def build_explorer_dataset(
    config_name: str = "tiny",
    split: str = "train",
    record_count: int | None = None,
    checkpoint: str | Path = FROZEN_CORE,
    device: DeviceLike = AUTO_DEVICE,
) -> ExplorerDataset:
    cfg = explorer_config(config_name)
    target_device = resolve_device(device)
    count = int(record_count if record_count is not None else (cfg.train_records if split == "train" else cfg.eval_records))
    split_offset = {"train": 0, "heldout": 10000, "transfer": 20000, "paraphrase": 30000}.get(split, 40000)
    seed = cfg.seed + split_offset
    generator = make_generator(seed, target_device)
    idx = torch.arange(count, device=target_device)

    object_id = torch.randint(0, NUM_OBJECTS, (count,), generator=generator, device=target_device)
    tool_id = torch.randint(0, NUM_TOOLS, (count,), generator=generator, device=target_device)
    door_state = torch.randint(0, NUM_DOORS, (count,), generator=generator, device=target_device)
    hazard_level = torch.randint(0, NUM_HAZARDS, (count,), generator=generator, device=target_device)
    resource_level = torch.randint(0, NUM_RESOURCES, (count,), generator=generator, device=target_device)
    agent_id = torch.randint(0, NUM_AGENTS, (count,), generator=generator, device=target_device)
    rule_id = (torch.randint(0, NUM_RULES, (count,), generator=generator, device=target_device) + split_offset) % NUM_RULES
    source_id = torch.randint(0, NUM_SOURCES, (count,), generator=generator, device=target_device)
    location_id = torch.randint(0, NUM_LOCATIONS, (count,), generator=generator, device=target_device)
    phase_id = (idx + split_offset) % NUM_PHASES
    noise = (idx % 11) == 0
    novelty = ((idx + rule_id) % 7 == 0) & (~noise)
    visible = ((idx + object_id) % 4) != 0
    contradiction = ((idx + source_id) % 13) == 0
    partner_uncertain = ((idx + agent_id) % 9) == 0
    restart = ((idx + project_id_seed(rule_id, object_id, agent_id, phase_id)) % 17) == 0
    transfer = split != "train"
    transfer_mask = torch.full((count,), bool(transfer), dtype=torch.bool, device=target_device) | ((idx % 5) == 0)

    confidence = torch.clamp(0.92 - 0.16 * contradiction.float() - 0.20 * partner_uncertain.float(), 0.05, 0.98)
    body_energy = torch.clamp(0.88 - 0.13 * hazard_level.float() + 0.06 * resource_level.float(), 0.04, 1.0)
    state_hint = torch.randint(0, NUM_UNCERTAINTY, (count,), generator=generator, device=target_device)
    project_id = torch.randint(0, NUM_PROJECTS, (count,), generator=generator, device=target_device)
    project_step = ((phase_id + door_state + resource_level) % 4).long()
    skill_id = torch.randint(0, NUM_SKILLS, (count,), generator=generator, device=target_device)
    partner_id = torch.randint(0, NUM_PARTNERS, (count,), generator=generator, device=target_device)
    affordance_id = ((rule_id * 2 + object_id + tool_id + phase_id + door_state) % NUM_EXPLORER_ACTIONS).long()

    action_target = affordance_id.clone()
    action_target = torch.where(noise, torch.full_like(action_target, ACTION_WAIT), action_target)
    random_action = torch.randint(0, NUM_EXPLORER_ACTIONS, (count,), generator=generator, device=target_device)
    reactive_action = ((object_id + hazard_level + visible.long()) % NUM_EXPLORER_ACTIONS).long()

    planner_target = torch.where(hazard_level >= 2, torch.full_like(action_target, ACTION_AVOID), action_target)
    planner_target = torch.where(resource_level == 0, torch.full_like(action_target, ACTION_FORAGE), planner_target)
    planner_target = torch.where(contradiction, torch.full_like(action_target, ACTION_COMPARE), planner_target)
    planner_target = torch.where(partner_uncertain, torch.full_like(action_target, ACTION_ASK), planner_target)
    planner_target = torch.where(noise, torch.full_like(action_target, ACTION_WAIT), planner_target)

    counterfactual_target = torch.where(hazard_level >= 2, torch.full_like(action_target, ACTION_AVOID), action_target)
    counterfactual_target = torch.where(resource_level == 0, torch.full_like(action_target, ACTION_FORAGE), counterfactual_target)
    counterfactual_target = torch.where(contradiction, torch.full_like(action_target, ACTION_SELF_CORRECT), counterfactual_target)
    counterfactual_target = torch.where(noise, torch.full_like(action_target, ACTION_WAIT), counterfactual_target)

    uncertainty_target = state_hint
    novelty_target = novelty.long()
    next_state_target = ((rule_id * 3 + object_id + project_id + phase_id + state_hint * 2) % NUM_WORLD_STATES).long()
    question_target = (partner_uncertain & (~noise)).long()
    conflict_target = contradiction.long()
    safety_target = ((hazard_level < 2) | (planner_target == ACTION_AVOID) | (planner_target == ACTION_WAIT)).long()
    partner_target = partner_id

    mind_target = torch.full((count,), MIND_ACT, dtype=torch.long, device=target_device)
    mind_target = torch.where(novelty, torch.full_like(mind_target, MIND_EXPLORE), mind_target)
    mind_target = torch.where(state_hint >= 2, torch.full_like(mind_target, MIND_REPORT_UNCERTAINTY), mind_target)
    mind_target = torch.where(partner_uncertain, torch.full_like(mind_target, MIND_ASK), mind_target)
    mind_target = torch.where(contradiction, torch.full_like(mind_target, MIND_SELF_CORRECT), mind_target)
    mind_target = torch.where(hazard_level == 3, torch.full_like(mind_target, MIND_REFUSE), mind_target)
    mind_target = torch.where(noise, torch.full_like(mind_target, MIND_WAIT), mind_target)

    intrinsic = build_intrinsic_vector(
        confidence=confidence,
        contradiction=contradiction,
        novelty=novelty,
        noise=noise,
        hazard_level=hazard_level,
        resource_level=resource_level,
        body_energy=body_energy,
        project_step=project_step,
        restart=restart,
        partner_uncertain=partner_uncertain,
        state_hint=state_hint,
    )
    obs_scalars = torch.stack(
        [
            visible.float(),
            noise.float(),
            novelty.float(),
            contradiction.float(),
            partner_uncertain.float(),
            restart.float(),
            body_energy,
            confidence,
            (hazard_level.float() / 3.0),
            (resource_level.float() / 3.0),
        ],
        dim=-1,
    )
    obs = torch.cat(
        [
            _one_hot(object_id, NUM_OBJECTS),
            _one_hot(tool_id, NUM_TOOLS),
            _one_hot(door_state, NUM_DOORS),
            _one_hot(hazard_level, NUM_HAZARDS),
            _one_hot(resource_level, NUM_RESOURCES),
            _one_hot(agent_id, NUM_AGENTS),
            _one_hot(rule_id, NUM_RULES),
            _one_hot(source_id, NUM_SOURCES),
            _one_hot(location_id, NUM_LOCATIONS),
            _one_hot(phase_id, NUM_PHASES),
            _one_hot(affordance_id, NUM_EXPLORER_ACTIONS),
            obs_scalars,
        ],
        dim=-1,
    )
    hypothesis = torch.cat(
        [
            _one_hot(rule_id, NUM_RULES),
            _one_hot(source_id, NUM_SOURCES),
            torch.stack(
                [
                    confidence,
                    contradiction.float(),
                    visible.float(),
                    project_step.float() / 3.0,
                    novelty.float(),
                    noise.float(),
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )
    skill_memory = torch.cat(
        [
            _one_hot(skill_id, NUM_SKILLS),
            torch.stack(
                [
                    transfer_mask.float(),
                    (tool_id.float() / max(NUM_TOOLS - 1, 1)),
                    (hazard_level.float() / 3.0),
                    confidence,
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )
    project_state = torch.cat(
        [
            _one_hot(project_id, NUM_PROJECTS),
            torch.stack(
                [
                    project_step.float() / 3.0,
                    restart.float(),
                    (resource_level == 0).float(),
                    (hazard_level >= 2).float(),
                    (idx.float() % 31) / 30.0,
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )
    social_state = torch.cat(
        [
            _one_hot(partner_id, NUM_PARTNERS),
            torch.stack(
                [
                    partner_uncertain.float(),
                    contradiction.float(),
                    confidence,
                    (source_id.float() / max(NUM_SOURCES - 1, 1)),
                    restart.float(),
                    visible.float(),
                ],
                dim=-1,
            ),
        ],
        dim=-1,
    )

    z = _latent_from_frozen_core(count, seed, target_device, checkpoint)
    z = z + _category_latent(rule_id, project_id, skill_id, partner_id, seed, target_device)

    informative_mask = ~noise
    planning_mask = informative_mask & visible
    counterfactual_mask = (hazard_level >= 2) | (resource_level == 0) | contradiction | noise
    restart_mask = restart
    safety_mask = (hazard_level >= 2) | (resource_level == 0)
    social_uncertainty_mask = partner_uncertain
    conflict_mask = contradiction
    partner_restart_mask = restart | (((idx + partner_id) % 19) == 0)
    project_mask = torch.ones(count, dtype=torch.bool, device=target_device)
    skill_mask = torch.ones(count, dtype=torch.bool, device=target_device)
    mind_mask = torch.ones(count, dtype=torch.bool, device=target_device)

    tensors = {
        "obs": obs,
        "z": z,
        "hypothesis": hypothesis,
        "skill_memory": skill_memory,
        "project_state": project_state,
        "social_state": social_state,
        "intrinsic": intrinsic,
        "action_target": action_target,
        "planner_target": planner_target,
        "counterfactual_target": counterfactual_target,
        "next_state_target": next_state_target,
        "uncertainty_target": uncertainty_target,
        "novelty_target": novelty_target,
        "skill_target": skill_id,
        "project_target": project_id,
        "question_target": question_target,
        "conflict_target": conflict_target,
        "safety_target": safety_target,
        "partner_target": partner_target,
        "mind_target": mind_target,
        "random_action_target": random_action,
        "reactive_action_target": reactive_action,
        "informative_mask": informative_mask,
        "noise_mask": noise,
        "planning_mask": planning_mask,
        "counterfactual_mask": counterfactual_mask,
        "skill_mask": skill_mask,
        "transfer_mask": transfer_mask,
        "project_mask": project_mask,
        "restart_mask": restart_mask,
        "safety_mask": safety_mask,
        "social_uncertainty_mask": social_uncertainty_mask,
        "conflict_mask": conflict_mask,
        "partner_restart_mask": partner_restart_mask,
        "mind_mask": mind_mask,
    }
    metadata = {
        "config": asdict(cfg),
        "split": split,
        "records": count,
        "checkpoint": str(checkpoint),
        "open_world_features": [
            "objects",
            "tools",
            "doors",
            "hazards",
            "resources",
            "partial_observability",
            "agents",
            "changing_rules",
            "irreversible_events",
        ],
        "long_ticks": cfg.long_ticks,
    }
    return ExplorerDataset(tensors, metadata)


def project_id_seed(
    rule_id: torch.Tensor,
    object_id: torch.Tensor,
    agent_id: torch.Tensor,
    phase_id: torch.Tensor,
) -> torch.Tensor:
    return ((rule_id + object_id + agent_id + phase_id) % NUM_PROJECTS).long()
