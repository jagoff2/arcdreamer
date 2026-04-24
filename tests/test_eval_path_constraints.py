from __future__ import annotations

import torch

from arcagi.evaluation.harness import build_agent


def test_clean_eval_agents_do_not_construct_runtime_controller() -> None:
    device = torch.device("cpu")

    recurrent = build_agent("recurrent", device=device)
    language = build_agent("language", device=device)
    language_memory = build_agent("online_language_memory", device=device)
    hybrid = build_agent("hybrid", device=device)

    assert recurrent.config.use_runtime_controller is False
    assert recurrent.runtime_rule_controller is None
    assert recurrent.config.use_theory_manager is False
    assert recurrent.theory_manager is None
    assert language.config.use_runtime_controller is False
    assert language.runtime_rule_controller is None
    assert language.config.use_theory_manager is False
    assert language.theory_manager is None
    assert language_memory.config.use_memory is True
    assert language_memory.episodic_memory is not None
    assert language_memory.config.use_runtime_controller is False
    assert language_memory.runtime_rule_controller is None
    assert language_memory.config.use_theory_manager is False
    assert language_memory.theory_manager is None
    assert hybrid.config.use_runtime_controller is False
    assert hybrid.runtime_rule_controller is None
    assert hybrid.config.use_theory_manager is True
    assert hybrid.theory_manager is not None
