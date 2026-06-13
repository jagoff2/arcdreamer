from __future__ import annotations

import ast
import inspect

import torch

from src import evaluate, run_unbroken
from src.env import ACTION_LEFT, ACTION_RIGHT, NUM_ACTIONS, TinyWorldRuntime


HIDDEN_RUNTIME_ATTRS = {
    "start_pos",
    "target_pos",
    "target_color",
    "current_pos",
    "orientation",
    "hazard_pos",
    "energy",
    "damage",
    "resource",
    "failed_actions",
    "damage_events",
    "generator",
    "rng_device",
}


def _attribute_names(function) -> set[str]:
    tree = ast.parse(inspect.getsource(function))
    return {node.attr for node in ast.walk(tree) if isinstance(node, ast.Attribute)}


def test_core_runtime_loops_do_not_read_hidden_world_fields() -> None:
    for function in (evaluate.closed_loop_action_success, run_unbroken.run_unbroken):
        attrs = _attribute_names(function)
        leaked = sorted(attrs & HIDDEN_RUNTIME_ATTRS)
        assert leaked == [], f"{function.__name__} reads hidden simulator fields: {leaked}"


def test_run_unbroken_does_not_call_expected_action_helpers() -> None:
    attrs = _attribute_names(run_unbroken.run_unbroken)

    assert "expected_action" not in attrs
    assert "expected_body_action" not in attrs


def test_closed_loop_scoring_cannot_override_model_chosen_actions(monkeypatch) -> None:
    created_worlds: list[SpyWorld] = []

    class SpyWorld(TinyWorldRuntime):
        def __init__(self, seed: int = 0, episode_len: int = 80) -> None:
            super().__init__(seed=seed, episode_len=episode_len)
            self.actions_seen: list[int] = []
            self.expected_action_calls = 0
            created_worlds.append(self)

        def expected_action(self) -> int:
            self.expected_action_calls += 1
            return ACTION_LEFT

        def step(self, action: int) -> None:
            self.actions_seen.append(int(action))
            super().step(action)

    class FixedActionModel:
        def initial_state(self, batch_size: int, device: str = "cpu") -> torch.Tensor:
            return torch.zeros(batch_size, 4, device=device)

        def step(self, observation, z):
            del observation
            action_logits = torch.full((1, NUM_ACTIONS), -20.0, device=z.device)
            action_logits[:, ACTION_RIGHT] = 20.0
            return {
                "action_logits": action_logits,
                "private_logits": torch.zeros(1, 2, device=z.device),
            }, z + 1.0

    monkeypatch.setattr(evaluate, "TinyWorldRuntime", SpyWorld)

    score = evaluate.closed_loop_action_success(
        model=FixedActionModel(),
        episodes=1,
        seq_len=70,
        seed=123,
        device="cpu",
    )

    assert score == 0.0
    assert len(created_worlds) == 1
    assert created_worlds[0].expected_action_calls == 1
    assert created_worlds[0].actions_seen
    assert set(created_worlds[0].actions_seen) == {ACTION_RIGHT}
