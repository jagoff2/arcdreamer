from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping

import torch

from src import run_unbroken
from src.causal_hypotheses import fresh_hypothesis_posterior, update_hypothesis_posterior
from src.env import GRID_SIZE, NUM_ACTIONS, NUM_COLORS, generate_batch
from src.model import PROGRAM_FIELDS, PROGRAM_TRANSFORMS, PROGRAM_FAMILIES, RecurrentLatentModel, decode_program_proposals


def _find_neural_program_proposals(event: Mapping[str, Any]) -> list[dict[str, Any]]:
    for section in (event.get("prediction_error"), event.get("metadata")):
        if not isinstance(section, Mapping):
            continue
        for key in (
            "neural_program_proposals",
            "program_proposals",
            "neural_program_head_proposals",
        ):
            value = section.get(key)
            if isinstance(value, list) and value:
                return [item for item in value if isinstance(item, Mapping)]
    return []


def test_step_exposes_neural_program_logits_and_diagnostics() -> None:
    torch.manual_seed(2026)
    model = RecurrentLatentModel(device="cpu")
    batch = generate_batch(batch_size=1, seq_len=2, base_seed=2026, device="cpu")
    observation = {
        "sensory": batch["sensory"][:, 0],
        "lang_in": batch["lang_in"][:, 0],
        "private_in": batch["private_in"][:, 0],
    }

    output, _ = model.step(observation, model.initial_state(1))

    assert output["program_family_logits"].shape == (1, len(PROGRAM_FAMILIES))
    assert output["program_field_logits"].shape == (1, len(PROGRAM_FIELDS))
    assert output["program_transform_logits"].shape == (1, len(PROGRAM_TRANSFORMS))

    proposals = output.get("neural_program_proposals") or output.get("program_proposals")
    if not proposals:
        proposals = decode_program_proposals(output, action=0, top_k=2)

    assert proposals is not None
    assert len(proposals) > 0
    first = proposals[0]
    assert {"family", "selector", "transform", "source", "action", "goal_test"} <= set(first)


def test_update_hypothesis_posterior_adds_neural_program_candidates() -> None:
    posterior = fresh_hypothesis_posterior()
    event = {
        "action": 4,
        "delta": {},
        "prediction_error": {
            "neural_program_proposals": [
                {
                    "schema": "neural_causal_program_proposal_v1",
                    "source": "neural_proposal",
                    "family": "move_color",
                    "action": "4",
                    "selector": {"color": 2},
                    "transform": {"kind": "translate", "dy": 0.0, "dx": 1.0},
                    "goal_test": {"kind": "component_translation"},
                    "confidence": 0.94,
                }
            ]
        },
    }

    update_hypothesis_posterior(posterior, event)

    assert posterior["updates"] == 1
    hypotheses = posterior["hypotheses"]
    neural_candidates = [
        hypothesis
        for hypothesis in hypotheses.values()
        if hypothesis.get("source") == "neural_proposal"
        and hypothesis.get("family") == "move_color"
        and hypothesis.get("selector", {}).get("color") == 2
    ]

    assert len(neural_candidates) == 1
    candidate = neural_candidates[0]
    assert candidate["transform"] == {"kind": "translate", "dy": 0.0, "dx": 1.0}
    assert candidate["source"] == "neural_proposal"
    assert "support" in candidate and isinstance(candidate["support"], int)
    assert "family" in candidate and "selector" in candidate and "transform" in candidate
    assert "no_op|action:4" in hypotheses


def test_run_unbroken_persists_neural_program_proposals(tmp_path: Path) -> None:
    class FakeModel:
        def __init__(self) -> None:
            self.config = type("Config", (), {"hidden_dim": 4})()

        def eval(self) -> None:
            return None

        def step(self, observation: Mapping[str, torch.Tensor], z: torch.Tensor) -> tuple[dict[str, torch.Tensor], torch.Tensor]:
            del observation
            z_next = z + 0.1
            return {
                "action_logits": torch.tensor([[4.0, -1.0, -1.0, -1.0, -1.0]], dtype=torch.float32, device=z.device),
                "language_logits": torch.zeros(1, 13, dtype=torch.float32, device=z.device),
                "private_logits": torch.tensor([[0.2, 0.8]], dtype=torch.float32, device=z.device),
                "provenance_logits": torch.zeros(1, 4, dtype=torch.float32, device=z.device),
                "world_color_logits": torch.zeros(1, NUM_COLORS, dtype=torch.float32, device=z.device),
                "world_pos_logits": torch.zeros(1, GRID_SIZE, dtype=torch.float32, device=z.device),
                "memory_color_logits": torch.zeros(1, NUM_COLORS, dtype=torch.float32, device=z.device),
                "self_start_logits": torch.zeros(1, GRID_SIZE, dtype=torch.float32, device=z.device),
                "program_family_logits": torch.tensor([[3.0, 0.0, 0.0, 0.0]], dtype=torch.float32, device=z.device),
                "program_field_logits": torch.tensor([[0.0, 2.0, 0.0, -1.0]], dtype=torch.float32, device=z.device),
                "program_transform_logits": torch.tensor([[0.0, 0.0, 1.5, 0.0, 0.0, 0.0]], dtype=torch.float32, device=z.device),
                "program_color_logits": torch.tensor([[0.0, 1.0, 0.0, -1.0]], dtype=torch.float32, device=z.device),
            }, z_next

    class FakeWorld:
        def __init__(self, seed: int, episode_len: int = 80) -> None:
            self.global_tick = 0
            self.local_tick = 0
            self._sensory = torch.tensor([0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], dtype=torch.float32)
            self._lang = torch.tensor([3], dtype=torch.long)
            self.private_in = 0

        def observation(self, device: str = "cpu", private_in: int = 0) -> dict[str, torch.Tensor]:
            return {
                "sensory": self._sensory.to(device),
                "lang_in": self._lang.to(device),
                "private_in": torch.tensor([private_in], dtype=torch.long, device=device),
            }

        def step(self, action: int) -> None:
            self.global_tick += 1
            self.local_tick += 1
            self._sensory = self._sensory.roll(shifts=1)

    def forced_action(
        logits: torch.Tensor,
        memory: Any,
        observation: Mapping[str, Any],
        available_action_mask: list[bool] | None = None,
    ) -> tuple[int, dict[str, Any]]:
        del logits, memory, observation, available_action_mask
        return 0, {
            "schema": "runtime_experimental_action_selection_v1",
            "selected_action": 0,
            "candidate_count": 1,
            "components": [{"action": 0}],
            "weights": {},
            "available_actions": list(range(NUM_ACTIONS)),
        }

    checkpoint = tmp_path / "runtime.ckpt"
    torch.save({"model_config": RecurrentLatentModel(device="cpu").config.__dict__, "model_state": {}, "train_config": {}, "metrics": {}}, checkpoint)
    memory_file = tmp_path / "runtime_memory.pt"

    run_unbroken_module = run_unbroken
    original_world = run_unbroken_module.TinyWorldRuntime
    original_selector = run_unbroken_module.select_experimental_action
    original_loader = run_unbroken_module.load_checkpoint
    run_unbroken_module.TinyWorldRuntime = FakeWorld  # type: ignore[attr-defined]
    run_unbroken_module.select_experimental_action = forced_action  # type: ignore[assignment]
    run_unbroken_module.load_checkpoint = lambda _checkpoint, device="cpu": FakeModel()  # type: ignore[assignment]

    try:
        run_unbroken_module.run_unbroken(
            checkpoint,
            max_ticks=3,
            log_every=0,
            seed=900000,
            device="cpu",
            memory_file=memory_file,
        )
    finally:
        run_unbroken_module.TinyWorldRuntime = original_world  # type: ignore[attr-defined]
        run_unbroken_module.select_experimental_action = original_selector  # type: ignore[assignment]
        run_unbroken_module.load_checkpoint = original_loader  # type: ignore[assignment]

    payload = torch.load(memory_file, map_location="cpu")
    events = payload["event_journal"]
    assert len(events) == 3

    for event in events:
        proposals = _find_neural_program_proposals(event)
        assert proposals
        proposal = proposals[0]
        assert proposal["family"] in PROGRAM_FAMILIES
        assert proposal["selector"] != {}
        assert proposal["transform"] != {}
