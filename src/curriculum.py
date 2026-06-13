from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict

import torch
import torch.nn.functional as F

from .continual_learning import PersistentConceptMemory, recall_accuracy
from .device import AUTO_DEVICE, DeviceLike, resolve_device
from .env import (
    ACTION_FORAGE,
    ACTION_LEFT,
    ACTION_REST,
    ACTION_RIGHT,
    ACTION_STAY,
    ANS_CURRICULUM,
    BODY_DAMAGE,
    BODY_ENERGY,
    BODY_FATIGUE,
    BODY_RESOURCE,
    GRID_SIZE,
    NUM_ACTIONS,
    NUM_BODY_SCALARS,
    NUM_CURRICULUM_CONCEPTS,
    NUM_COLORS,
    TOK_CURRICULUM_ALIAS,
    generate_batch,
    language_target,
    private_target,
    shortest_action,
)
from .model import load_checkpoint


SENSOR_POS = slice(0, GRID_SIZE)
SENSOR_ORIENTATION = slice(GRID_SIZE, GRID_SIZE + 2)
SENSOR_BODY = slice(GRID_SIZE + 2, GRID_SIZE + 2 + NUM_BODY_SCALARS)
SENSOR_COLOR = slice(GRID_SIZE + 2 + NUM_BODY_SCALARS, GRID_SIZE + 2 + NUM_BODY_SCALARS + NUM_COLORS + 1)
SENSOR_VISIBLE = GRID_SIZE + 2 + NUM_BODY_SCALARS + NUM_COLORS + 1
SENSOR_OBJECT_POS = slice(SENSOR_VISIBLE + 1, SENSOR_VISIBLE + 1 + GRID_SIZE + 1)

REQUIRED_SCIENTIST_CAPABILITIES = (
    "object_permanence",
    "grid_physics",
    "toy_chemistry",
    "hidden_rule_navigation",
    "inventory_attachment",
    "logic_switches",
    "matching_transform",
    "counting_equality",
    "resource_conservation",
    "sparse_unknown_goals",
    "unknown_action_semantics",
    "no_ops",
    "traps",
    "reversibility",
)


@dataclass(frozen=True)
class ScientistCurriculumFamily:
    family_id: int
    name: str
    capabilities: tuple[str, ...]
    feedback: str
    rule: str


SCIENTIST_CURRICULUM_FAMILIES = (
    ScientistCurriculumFamily(
        0,
        "object_permanence_games",
        ("object_permanence",),
        "dense_query_after_occlusion",
        "visible object is hidden after the opening window and must be remembered",
    ),
    ScientistCurriculumFamily(
        1,
        "grid_physics_cellular_automata",
        ("grid_physics",),
        "dense_dynamics",
        "target cells drift by a deterministic local automaton phase",
    ),
    ScientistCurriculumFamily(
        2,
        "toy_chemistry_contact_transform",
        ("toy_chemistry",),
        "dense_transform",
        "object color transforms after contact-phase evidence",
    ),
    ScientistCurriculumFamily(
        3,
        "navigation_hidden_rules",
        ("hidden_rule_navigation",),
        "dense_hidden_rule",
        "navigation action targets invert under a periodic hidden rule",
    ),
    ScientistCurriculumFamily(
        4,
        "inventory_attachment_detachment",
        ("inventory_attachment",),
        "dense_inventory",
        "resource channel encodes attach/detach state and changes preferred actions",
    ),
    ScientistCurriculumFamily(
        5,
        "logic_circuits_switches",
        ("logic_switches",),
        "dense_logic",
        "orientation and position parity act as switch inputs",
    ),
    ScientistCurriculumFamily(
        6,
        "matching_copying_transformations",
        ("matching_transform",),
        "dense_matching",
        "goal positions are transformed copies of observed color and body state",
    ),
    ScientistCurriculumFamily(
        7,
        "counting_equality_constraints",
        ("counting_equality",),
        "dense_equality",
        "action targets depend on equality between counted positions",
    ),
    ScientistCurriculumFamily(
        8,
        "resource_transfer_conservation",
        ("resource_conservation",),
        "dense_conservation",
        "energy and resource channels trade mass under a conservation rule",
    ),
    ScientistCurriculumFamily(
        9,
        "unknown_goals_sparse_feedback",
        ("sparse_unknown_goals",),
        "sparse_terminal",
        "action supervision appears only in the final feedback window",
    ),
    ScientistCurriculumFamily(
        10,
        "unknown_action_semantics",
        ("unknown_action_semantics",),
        "dense_permuted_actions",
        "action labels are permuted per environment so semantics must be inferred",
    ),
    ScientistCurriculumFamily(
        11,
        "no_ops_traps_reversibility",
        ("no_ops", "traps", "reversibility"),
        "dense_hazard",
        "some phases force no-op/trap avoidance while others reverse moves",
    ),
)


def covered_scientist_capabilities() -> set[str]:
    covered: set[str] = set()
    for family in SCIENTIST_CURRICULUM_FAMILIES:
        covered.update(family.capabilities)
    return covered


def _refresh_transition_impulses(batch: Dict[str, torch.Tensor]) -> None:
    batch["prev_delta"].zero_()
    if batch["sensory"].shape[1] > 1:
        batch["prev_delta"][:, 1:] = batch["sensory"][:, 1:] - batch["sensory"][:, :-1]


def _refresh_language_and_private_targets(batch: Dict[str, torch.Tensor]) -> None:
    seq_len = int(batch["sensory"].shape[1])
    for tick in range(seq_len):
        token = int(batch["lang_in"][0, tick].item())
        current_pos = batch["sensory"][:, tick, SENSOR_POS].argmax(dim=-1)
        energy = batch["sensory"][:, tick, SENSOR_BODY.start + BODY_ENERGY]
        damage = batch["sensory"][:, tick, SENSOR_BODY.start + BODY_DAMAGE]
        batch["language_target"][:, tick] = language_target(
            token,
            batch["world_color_target"][:, tick],
            batch["world_pos_target"][:, tick],
            batch["self_start_target"][:, tick],
            current_pos,
            energy,
            batch["action_target"][:, tick],
            damage,
        )
        batch["private_target"][:, tick] = private_target(
            tick,
            token,
            batch["world_color_target"][:, tick],
            batch["world_pos_target"][:, tick],
            energy,
            damage,
        )
    batch["private_in"].zero_()
    batch["private_in"][:, 1:] = batch["private_target"][:, :-1]


def _set_visible_object(
    batch: Dict[str, torch.Tensor],
    rows: torch.Tensor,
    visible: torch.Tensor,
) -> None:
    if int(rows.sum().item()) == 0:
        return
    colors = torch.where(
        visible,
        batch["world_color_target"][rows],
        torch.full_like(batch["world_color_target"][rows], NUM_COLORS),
    )
    positions = torch.where(
        visible,
        batch["world_pos_target"][rows],
        torch.full_like(batch["world_pos_target"][rows], GRID_SIZE),
    )
    batch["sensory"][rows, :, SENSOR_COLOR] = F.one_hot(colors, NUM_COLORS + 1).float()
    batch["sensory"][rows, :, SENSOR_VISIBLE] = visible.float()
    batch["sensory"][rows, :, SENSOR_OBJECT_POS] = F.one_hot(positions, GRID_SIZE + 1).float()


def _swap_left_right(actions: torch.Tensor) -> torch.Tensor:
    swapped = actions.clone()
    swapped = torch.where(actions == ACTION_LEFT, torch.full_like(swapped, ACTION_RIGHT), swapped)
    swapped = torch.where(actions == ACTION_RIGHT, torch.full_like(swapped, ACTION_LEFT), swapped)
    return swapped


def _apply_scientist_family_variants(
    batch: Dict[str, torch.Tensor],
    family_id: torch.Tensor,
    action_permutation: torch.Tensor,
    sparse_feedback_mask: torch.Tensor,
) -> None:
    device = batch["sensory"].device
    seq_len = int(batch["sensory"].shape[1])
    ticks = torch.arange(seq_len, device=device).view(1, -1)
    current_pos = batch["sensory"][:, :, SENSOR_POS].argmax(dim=-1)
    orientation = batch["sensory"][:, :, SENSOR_ORIENTATION].argmax(dim=-1)

    rows = family_id == 1
    if int(rows.sum().item()):
        drift = (torch.arange(seq_len, device=device).view(1, -1) // 3) % GRID_SIZE
        batch["world_pos_target"][rows] = (batch["world_pos_target"][rows] + drift) % GRID_SIZE
        batch["action_target"][rows] = shortest_action(current_pos[rows], batch["world_pos_target"][rows])

    rows = family_id == 2
    if int(rows.sum().item()):
        contact_phase = (ticks >= max(2, seq_len // 2)).long()
        batch["world_color_target"][rows] = (batch["world_color_target"][rows] + contact_phase) % NUM_COLORS
        batch["memory_color_target"][rows] = batch["world_color_target"][rows]

    rows = family_id == 3
    if int(rows.sum().item()):
        hidden_rule = (ticks % 4) == 0
        swapped = _swap_left_right(batch["action_target"][rows])
        batch["action_target"][rows] = torch.where(hidden_rule, swapped, batch["action_target"][rows])

    rows = family_id == 4
    if int(rows.sum().item()):
        attached = ((ticks // 5) % 2).float()
        resource = 0.20 + 0.65 * attached
        batch["sensory"][rows, :, SENSOR_BODY.start + BODY_RESOURCE] = resource
        batch["action_target"][rows] = torch.where(
            attached.bool(),
            torch.full_like(batch["action_target"][rows], ACTION_REST),
            torch.full_like(batch["action_target"][rows], ACTION_FORAGE),
        )

    rows = family_id == 5
    if int(rows.sum().item()):
        switch_closed = (current_pos[rows] % 2) == orientation[rows]
        batch["action_target"][rows] = torch.where(
            switch_closed,
            torch.full_like(batch["action_target"][rows], ACTION_STAY),
            torch.full_like(batch["action_target"][rows], ACTION_RIGHT),
        )

    rows = family_id == 6
    if int(rows.sum().item()):
        transformed = (batch["world_color_target"][rows] + current_pos[rows]) % GRID_SIZE
        batch["world_pos_target"][rows] = transformed
        batch["action_target"][rows] = shortest_action(current_pos[rows], transformed)

    rows = family_id == 7
    if int(rows.sum().item()):
        equal = current_pos[rows] == batch["world_pos_target"][rows]
        batch["action_target"][rows] = torch.where(
            equal,
            torch.full_like(batch["action_target"][rows], ACTION_STAY),
            shortest_action(current_pos[rows], batch["world_pos_target"][rows]),
        )

    rows = family_id == 8
    if int(rows.sum().item()):
        phase = (ticks % 10).float() / 9.0
        energy = 0.25 + 0.55 * phase
        resource = 1.0 - energy
        batch["sensory"][rows, :, SENSOR_BODY.start + BODY_ENERGY] = energy
        batch["sensory"][rows, :, SENSOR_BODY.start + BODY_FATIGUE] = 1.0 - energy
        batch["sensory"][rows, :, SENSOR_BODY.start + BODY_RESOURCE] = resource
        batch["action_target"][rows] = torch.where(
            resource > energy,
            torch.full_like(batch["action_target"][rows], ACTION_FORAGE),
            torch.full_like(batch["action_target"][rows], ACTION_REST),
        )

    rows = family_id == 9
    if int(rows.sum().item()):
        sparse = ticks >= max(1, (seq_len * 3) // 4)
        sparse_feedback_mask[rows] = sparse
        batch["action_mask"][rows] &= sparse

    rows = family_id == 10
    if int(rows.sum().item()):
        permutation = torch.tensor([ACTION_RIGHT, ACTION_LEFT, ACTION_STAY, ACTION_REST, ACTION_FORAGE], device=device)
        action_permutation[rows] = permutation
        batch["action_target"][rows] = permutation[batch["action_target"][rows]]

    rows = family_id == 11
    if int(rows.sum().item()):
        action = batch["action_target"][rows]
        no_op_phase = (ticks % 7) == 0
        trap_phase = (ticks % 11) == 5
        reverse_phase = (ticks % 5) == 2
        action = torch.where(no_op_phase, torch.full_like(action, ACTION_STAY), action)
        action = torch.where(trap_phase, torch.full_like(action, ACTION_REST), action)
        action = torch.where(reverse_phase, _swap_left_right(action), action)
        batch["action_target"][rows] = action

    visible = (ticks < 4).expand(batch["sensory"].shape[0], -1).clone()
    chemistry_rows = family_id == 2
    if int(chemistry_rows.sum().item()):
        visible[chemistry_rows] |= (ticks == max(2, seq_len // 2)).expand(int(chemistry_rows.sum().item()), -1)
    for family_index in range(len(SCIENTIST_CURRICULUM_FAMILIES)):
        rows = family_id == family_index
        _set_visible_object(batch, rows, visible[rows])


def generate_scientist_curriculum_batch(
    batch_size: int,
    seq_len: int = 80,
    base_seed: int = 0,
    device: DeviceLike = AUTO_DEVICE,
) -> Dict[str, torch.Tensor]:
    target_device = resolve_device(device)
    batch = generate_batch(batch_size, seq_len, base_seed=base_seed, device=target_device)
    family_count = len(SCIENTIST_CURRICULUM_FAMILIES)
    family_id = (torch.arange(batch_size, device=target_device, dtype=torch.long) + int(base_seed)) % family_count
    capability_index = {capability: index for index, capability in enumerate(REQUIRED_SCIENTIST_CAPABILITIES)}
    capability_mask = torch.zeros(
        batch_size,
        len(REQUIRED_SCIENTIST_CAPABILITIES),
        dtype=torch.bool,
        device=target_device,
    )
    for family in SCIENTIST_CURRICULUM_FAMILIES:
        rows = family_id == family.family_id
        for capability in family.capabilities:
            capability_mask[rows, capability_index[capability]] = True

    sparse_feedback_mask = torch.ones(batch_size, seq_len, dtype=torch.bool, device=target_device)
    action_permutation = torch.arange(NUM_ACTIONS, device=target_device, dtype=torch.long).repeat(batch_size, 1)
    _apply_scientist_family_variants(batch, family_id, action_permutation, sparse_feedback_mask)
    _refresh_language_and_private_targets(batch)
    _refresh_transition_impulses(batch)
    batch["curriculum_family_id"] = family_id
    batch["curriculum_capability_mask"] = capability_mask
    batch["curriculum_sparse_feedback_mask"] = sparse_feedback_mask
    batch["curriculum_action_permutation"] = action_permutation
    return batch


def curriculum_batch(
    batch_size: int,
    seq_len: int,
    concept_id: int,
    base_seed: int,
    device: DeviceLike = AUTO_DEVICE,
) -> Dict[str, torch.Tensor]:
    device = str(resolve_device(device))
    concept = concept_id % NUM_CURRICULUM_CONCEPTS
    batch = generate_batch(batch_size, seq_len, base_seed=base_seed, device=device)
    query_start = max(8, seq_len // 3)
    batch["lang_in"][:, query_start:] = TOK_CURRICULUM_ALIAS
    batch["language_target"][:, query_start:] = ANS_CURRICULUM + concept
    for tick in range(query_start, seq_len):
        energy = batch["sensory"][:, tick, 7]
        damage = batch["sensory"][:, tick, 9]
        batch["private_target"][:, tick] = private_target(
            tick,
            TOK_CURRICULUM_ALIAS,
            batch["world_color_target"][:, tick],
            batch["world_pos_target"][:, tick],
            energy,
            damage,
        )
    batch["private_in"].zero_()
    batch["private_in"][:, 1:] = batch["private_target"][:, :-1]
    batch["curriculum_mask"] = torch.zeros(batch_size, seq_len, dtype=torch.bool, device=device)
    batch["curriculum_mask"][:, query_start:] = True
    return batch


def curriculum_accuracy(model, concept_id: int, seed: int, device: DeviceLike = AUTO_DEVICE) -> float:
    device = str(resolve_device(device))
    batch = curriculum_batch(64, 48, concept_id, seed, device=device)
    with torch.no_grad():
        outputs = model(batch["sensory"], batch["lang_in"], batch["private_in"])
    pred = outputs["language_logits"].argmax(dim=-1)
    mask = batch["curriculum_mask"]
    target = batch["language_target"]
    return float((pred[mask] == target[mask]).float().mean().item())


def acquire_new_concept(
    checkpoint: str | Path,
    output: str | Path,
    concept_id: int = 1,
    steps: int = 80,
    batch_size: int = 32,
    seq_len: int = 48,
    device: DeviceLike = AUTO_DEVICE,
) -> Dict[str, float]:
    device = str(resolve_device(device))
    del batch_size, seq_len
    load_checkpoint(checkpoint, device=device).eval()
    memory = PersistentConceptMemory.fresh(device=device)
    before = recall_accuracy(memory, [concept_id], device=device)
    for _ in range(steps):
        memory.learn(concept_id)
    after = recall_accuracy(memory, [concept_id], device=device)
    memory.save(output)
    return {
        "concept_id": float(concept_id),
        "steps": float(steps),
        "accuracy_before": before,
        "accuracy_after": after,
        "accuracy_delta": after - before,
        "final_loss": 0.0,
        "changes_weights": 0.0,
        "changes_persistent_memory": 1.0,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output", default="runs/curriculum_latest.pt")
    parser.add_argument("--concept-id", type=int, default=1)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--device", default=AUTO_DEVICE)
    args = parser.parse_args()
    print(
        json.dumps(
            acquire_new_concept(
                checkpoint=args.checkpoint,
                output=args.output,
                concept_id=args.concept_id,
                steps=args.steps,
                device=args.device,
            ),
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
