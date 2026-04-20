from __future__ import annotations

import torch

from arcagi.evaluation.harness import build_agent


def test_only_full_hybrid_constructs_runtime_controller() -> None:
    device = torch.device("cpu")

    recurrent = build_agent("recurrent", device=device)
    language = build_agent("language", device=device)
    hybrid = build_agent("hybrid", device=device)

    assert recurrent.config.use_runtime_controller is False
    assert recurrent.runtime_rule_controller is None
    assert language.config.use_runtime_controller is False
    assert language.runtime_rule_controller is None
    assert hybrid.config.use_runtime_controller is True
    assert hybrid.runtime_rule_controller is not None
