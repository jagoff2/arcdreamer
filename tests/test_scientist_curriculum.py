from __future__ import annotations

import torch

from src import curriculum


def test_scientist_curriculum_families_cover_required_capabilities() -> None:
    families = curriculum.SCIENTIST_CURRICULUM_FAMILIES
    required = set(curriculum.REQUIRED_SCIENTIST_CAPABILITIES)
    covered = curriculum.covered_scientist_capabilities()

    assert families
    assert required.issubset(covered)
    assert all(len(family.capabilities) > 0 for family in families)


def test_generate_scientist_curriculum_batch_is_deterministic_and_tensorized() -> None:
    family_count = len(curriculum.SCIENTIST_CURRICULUM_FAMILIES)
    kwargs = {
        "batch_size": family_count,
        "seq_len": 24,
        "base_seed": 17,
        "device": "cpu",
    }
    batch = curriculum.generate_scientist_curriculum_batch(**kwargs)
    duplicate_batch = curriculum.generate_scientist_curriculum_batch(**kwargs)

    assert batch.keys() == duplicate_batch.keys()
    for key in batch:
        assert torch.is_tensor(batch[key])
        assert torch.equal(batch[key], duplicate_batch[key])

    family_ids = batch["curriculum_family_id"]
    assert family_ids.dtype == torch.long
    assert family_ids.numel() == family_count
    assert set(int(i) for i in torch.unique(family_ids).tolist()) == set(range(family_count))

    sparse_feedback_mask = batch["curriculum_sparse_feedback_mask"]
    assert not sparse_feedback_mask.all()

    action_permutation = batch["curriculum_action_permutation"]
    identity = torch.arange(action_permutation.shape[1], device=action_permutation.device)
    non_identity_rows = (action_permutation != identity).any(dim=1)
    assert non_identity_rows.any()
