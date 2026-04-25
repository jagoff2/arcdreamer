from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from arcagi.core.types import ActionName, ObjectState, StructuredState, Transition
from arcagi.learned_online.event_tokens import (
    OBJECT,
    OUT_NO_EFFECT_NONPROGRESS,
    OUT_OBJECTIVE_PROGRESS,
    OUT_REWARD_PROGRESS,
    OUT_VISIBLE_CHANGE,
    build_transition_targets,
    encode_action_tokens,
    encode_state_tokens,
    stack_action_tokens,
    stack_state_tokens,
)
from arcagi.learned_online.object_event_curriculum import (
    ActiveOnlineObjectEventConfig,
    ObjectEventCurriculumConfig,
    ObjectEventExample,
    build_active_online_object_event_curriculum,
    build_paired_color_click_curriculum,
    collate_object_event_examples,
)
from arcagi.learned_online.object_event_model import (
    ObjectEventModel,
    ObjectEventModelConfig,
    policy_rank_logits_from_predictions,
)
from scripts.train_learned_online_object_event import (
    _batch_to_tensors,
    _forward_tensors,
    _with_observed_candidate_targets,
)


@dataclass(frozen=True)
class CueClickCase:
    state: StructuredState
    next_state: StructuredState
    action: ActionName
    legal_actions: tuple[ActionName, ...]
    correct_action_index: int
    red_action_index: int
    blue_action_index: int

    def transition(self) -> Transition:
        return Transition(
            state=self.state,
            action=self.action,
            reward=1.0,
            next_state=self.next_state,
            terminated=False,
            info={"score_delta": 1.0},
        )


def test_paired_cue_tokens_and_targets_are_valid_and_nonleaking() -> None:
    pair_a, pair_b = _paired_cue_cases(seed=123)

    assert pair_a.legal_actions == pair_b.legal_actions
    assert pair_a.red_action_index == pair_b.red_action_index
    assert pair_a.blue_action_index == pair_b.blue_action_index
    assert pair_a.correct_action_index == pair_a.red_action_index
    assert pair_b.correct_action_index == pair_b.blue_action_index

    state_a = encode_state_tokens(pair_a.state)
    state_b = encode_state_tokens(pair_b.state)
    action_a = encode_action_tokens(pair_a.state, pair_a.legal_actions)
    action_b = encode_action_tokens(pair_b.state, pair_b.legal_actions)

    assert float(np.abs(state_a.numeric - state_b.numeric).sum()) > 0.0
    object_mask_a = state_a.type_ids == OBJECT
    object_mask_b = state_b.type_ids == OBJECT
    assert int(np.sum(object_mask_a)) >= 3
    assert int(np.sum(object_mask_b)) >= 3
    object_diff = float(np.abs(state_a.numeric[object_mask_a] - state_b.numeric[object_mask_b]).sum())
    assert object_diff > 1.0e-3, {
        "object_diff": object_diff,
        "state_a_object_tokens": state_a.numeric[object_mask_a].tolist(),
        "state_b_object_tokens": state_b.numeric[object_mask_b].tolist(),
    }
    assert np.allclose(action_a.numeric[pair_a.red_action_index], action_b.numeric[pair_b.red_action_index])
    assert np.allclose(action_a.numeric[pair_a.blue_action_index], action_b.numeric[pair_b.blue_action_index])
    red_row = action_a.numeric[pair_a.red_action_index]
    blue_row = action_a.numeric[pair_a.blue_action_index]
    assert not np.allclose(red_row, blue_row, atol=1.0e-6, rtol=1.0e-6), {
        "red_action": pair_a.legal_actions[pair_a.red_action_index],
        "blue_action": pair_a.legal_actions[pair_a.blue_action_index],
        "red_row": red_row.tolist(),
        "blue_row": blue_row.tolist(),
    }
    assert float(np.abs(red_row - blue_row).sum()) > 1.0e-3

    tensors = _batch_tensors((pair_a, pair_b))
    values = tensors["candidate_value_targets"]
    assert float(values[0, pair_a.red_action_index]) == 1.0
    assert float(values[0, pair_a.blue_action_index]) == 0.0
    assert float(values[1, pair_b.red_action_index]) == 0.0
    assert float(values[1, pair_b.blue_action_index]) == 1.0
    assert int(tensors["actual_action_index"][0]) == pair_a.red_action_index
    assert int(tensors["actual_action_index"][1]) == pair_b.blue_action_index
    assert torch.allclose(tensors["target_outcome"][0], tensors["target_outcome"][1])
    assert torch.allclose(tensors["target_delta"][0], tensors["target_delta"][1])


def test_token_attention_probe_can_overfit_paired_cue_task() -> None:
    batch = _batch_tensors(_make_paired_cue_cases(num_pairs=32, seed=100))
    logits = _paired_cue_rule_logits(batch)
    top1 = float((logits.argmax(dim=-1) == batch["actual_action_index"]).float().mean())
    zero_batch = {
        **batch,
        "inputs": {
            **batch["inputs"],
            "state_numeric": torch.zeros_like(batch["inputs"]["state_numeric"]),
        },
    }
    zero_logits = _paired_cue_rule_logits(zero_batch)
    zero_top1 = float((zero_logits.argmax(dim=-1) == batch["actual_action_index"]).float().mean())

    assert top1 >= 0.95, {
        "attention_probe_train_top1": top1,
        "zero_state_top1": zero_top1,
    }
    assert zero_top1 <= 0.65, {
        "rule_probe_train_top1": top1,
        "zero_state_top1": zero_top1,
    }


def test_paired_cue_token_fields_support_rule_oracle() -> None:
    cases = _make_paired_cue_cases(num_pairs=32, seed=333)
    batch = _batch_tensors(cases)
    object_color_field = 0
    object_centroid_y_field = 2
    object_centroid_x_field = 3
    action_contains_object_field = 11
    action_target_color_field = 13
    logits: list[torch.Tensor] = []
    for batch_index in range(batch["inputs"]["state_numeric"].shape[0]):
        object_mask = batch["inputs"]["state_type_ids"][batch_index] == OBJECT
        objects = batch["inputs"]["state_numeric"][batch_index, object_mask]
        assert objects.shape[0] >= 3
        cue_index = torch.argmin(objects[:, object_centroid_y_field] + objects[:, object_centroid_x_field])
        cue_color = float(objects[cue_index, object_color_field])
        wants_red = cue_color < (1.5 / 11.0)
        action_contains = batch["inputs"]["action_numeric"][batch_index, :, action_contains_object_field] > 0.5
        action_colors = batch["inputs"]["action_numeric"][batch_index, :, action_target_color_field]
        target_color = (2.0 / 11.0) if wants_red else (5.0 / 11.0)
        scores = -(action_colors - target_color).abs()
        scores = scores.masked_fill(~action_contains, -1.0e9)
        scores = scores.masked_fill(~batch["action_mask"][batch_index].bool(), -1.0e9)
        logits.append(scores)
    stacked = torch.stack(logits, dim=0)
    top1 = float((stacked.argmax(dim=-1) == batch["actual_action_index"]).float().mean())

    assert top1 >= 0.95, {
        "oracle_top1": top1,
        "message": "Token fields do not expose cue color plus clicked-object color.",
    }


def test_object_event_model_can_overfit_one_paired_cue_geometry() -> None:
    torch.manual_seed(2)
    pair_a, pair_b = _paired_cue_cases(seed=7)
    model = ObjectEventModel(
        ObjectEventModelConfig(
            d_model=64,
            n_heads=4,
            state_layers=1,
            action_cross_layers=1,
            dropout=0.0,
            online_rank=4,
        )
    )
    state_diag = _state_path_diagnostics(model, (pair_a, pair_b))
    assert state_diag["state_grad"] > 0.0, state_diag
    assert state_diag["action_grad"] > 0.0, state_diag
    assert state_diag["cross_grad"] > 0.0 or state_diag["film_grad"] > 0.0, state_diag

    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    for _step in range(150):
        tensors = _batch_tensors((pair_a, pair_b))
        output = model(**tensors["inputs"])
        losses = model.loss(
            output,
            target_outcome=tensors["target_outcome"],
            target_delta=tensors["target_delta"],
            actual_action_index=tensors["actual_action_index"],
            action_mask=tensors["action_mask"],
            candidate_outcome_targets=tensors["candidate_outcome_targets"],
            candidate_value_targets=tensors["candidate_value_targets"],
            loss_weights={"outcome": 0.0, "rank": 10.0, "inverse": 0.0, "value": 0.0, "delta": 0.0},
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    final = _eval_metrics(model, (pair_a, pair_b))
    assert final["top1"] == 1.0, {"final": final, "state_path": state_diag}
    assert final["score_margin"] > 1.0, {"final": final, "state_path": state_diag}
    assert final["rank_loss"] < 0.10, {"final": final, "state_path": state_diag}


def test_object_event_model_learns_state_conditioned_inverse_and_event_mapping() -> None:
    torch.manual_seed(0)
    train_cases = _paired_cue_cases(seed=19)
    model = ObjectEventModel(
        ObjectEventModelConfig(
            d_model=48,
            n_heads=4,
            state_layers=1,
            action_cross_layers=1,
            dropout=0.0,
            online_rank=4,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3)
    initial = _eval_metrics(model, train_cases)

    for _step in range(300):
        batch = train_cases
        tensors = _batch_tensors(batch)
        output = model(**tensors["inputs"])
        losses = model.loss(
            output,
            target_outcome=tensors["target_outcome"],
            target_delta=tensors["target_delta"],
            actual_action_index=tensors["actual_action_index"],
            action_mask=tensors["action_mask"],
            candidate_outcome_targets=tensors["candidate_outcome_targets"],
            candidate_value_targets=tensors["candidate_value_targets"],
            loss_weights={"outcome": 1.0, "rank": 5.0, "inverse": 0.1, "value": 0.5, "delta": 0.0},
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    final = _eval_metrics(model, train_cases)
    zero_state = _eval_metrics(model, train_cases, zero_state=True)
    shuffled_state = _eval_metrics(model, train_cases, shuffle_state=True)
    action_only = _eval_metrics(model, train_cases, action_only=True)

    assert final["top1"] == 1.0
    assert final["top3"] >= 0.95
    assert final["outcome_loss"] <= initial["outcome_loss"] * 0.65
    assert final["score_margin"] >= 1.0
    assert zero_state["top1"] <= 0.60
    assert shuffled_state["top1"] <= 0.60
    assert action_only["top1"] <= 0.60


def test_object_event_model_dense_curriculum_rank_overfits_and_uses_state() -> None:
    torch.manual_seed(9)
    split = build_paired_color_click_curriculum(
        ObjectEventCurriculumConfig(seed=4, train_geometries=8, heldout_geometries=0, grid_size=8)
    )
    cases = split.train
    assert cases
    assert {len(case.legal_actions) for case in cases} == {68}

    first, second = cases[0], cases[1]
    assert first.metadata["geometry_seed"] == second.metadata["geometry_seed"]
    assert first.legal_actions == second.legal_actions
    first_red = int(first.metadata["red_action_index"])
    first_blue = int(first.metadata["blue_action_index"])
    assert np.allclose(first.action_tokens.numeric[first_red], second.action_tokens.numeric[first_red])
    assert np.allclose(first.action_tokens.numeric[first_blue], second.action_tokens.numeric[first_blue])

    tensors = _curriculum_batch_tensors(cases)
    model = ObjectEventModel(
        ObjectEventModelConfig(
            d_model=64,
            n_heads=4,
            state_layers=1,
            action_cross_layers=1,
            dropout=0.0,
            online_rank=4,
        )
    )
    optimizer = torch.optim.AdamW(model.parameters(), lr=3.0e-3, weight_decay=1.0e-4)
    for _step in range(200):
        output = model(**tensors["inputs"])
        rank_logits = policy_rank_logits_from_predictions(output, tensors["action_mask"])
        loss = F.cross_entropy(rank_logits, tensors["actual_action_index"])
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

    normal = _curriculum_rank_metrics(model, cases)
    zero_state = _curriculum_rank_metrics(model, cases, zero_state=True)
    shuffled_state = _curriculum_rank_metrics(model, cases, shuffle_state=True)
    action_only = _curriculum_rank_metrics(model, cases, action_only=True)
    assert normal["top1"] >= 0.875, {
        "normal": normal,
        "zero_state": zero_state,
        "shuffled_state": shuffled_state,
        "action_only": action_only,
    }
    assert zero_state["top1"] <= normal["top1"] - 0.25
    assert shuffled_state["top1"] <= normal["top1"] - 0.25
    assert action_only["top1"] <= normal["top1"] - 0.25
    assert normal["combined_logits"][0, first_red] > normal["combined_logits"][0, first_blue]
    assert normal["combined_logits"][1, first_blue] > normal["combined_logits"][1, first_red]
    assert abs(float(normal["rank_logits"][0, first_red] - normal["rank_logits"][1, first_red])) > 1.0e-2


def test_online_update_from_observed_transition_changes_future_action_scores() -> None:
    torch.manual_seed(1)
    case = _cue_click_case(np.random.default_rng(5), 5)
    model = ObjectEventModel(
        ObjectEventModelConfig(
            d_model=32,
            n_heads=4,
            state_layers=1,
            action_cross_layers=1,
            dropout=0.0,
            online_rank=4,
        )
    )
    tensors = _batch_tensors((case,))
    with torch.no_grad():
        before = model(**tensors["inputs"]).value_logits[0].detach().clone()
    optimizer = torch.optim.AdamW(model.online_parameters(), lr=1.0e-2)
    output = model(**tensors["inputs"])
    losses = model.loss(
        output,
        target_outcome=tensors["target_outcome"],
        target_delta=tensors["target_delta"],
        actual_action_index=tensors["actual_action_index"],
        action_mask=tensors["action_mask"],
        candidate_outcome_targets=tensors["candidate_outcome_targets"],
        candidate_value_targets=tensors["candidate_value_targets"],
        loss_weights={"outcome": 1.0, "rank": 3.0, "inverse": 0.0, "value": 0.5, "delta": 0.0},
    )
    optimizer.zero_grad(set_to_none=True)
    losses["loss"].backward()
    optimizer.step()

    with torch.no_grad():
        after = model(**tensors["inputs"]).value_logits[0].detach()

    assert not torch.allclose(before, after)


def test_object_event_level_negative_memory_discourages_repeated_failed_candidate() -> None:
    model = ObjectEventModel(
        ObjectEventModelConfig(
            d_model=64,
            n_heads=4,
            state_layers=1,
            action_cross_layers=1,
            dropout=0.0,
            online_rank=4,
        )
    )
    level = build_active_online_object_event_curriculum(
        ActiveOnlineObjectEventConfig(seed=61, train_sessions=1, heldout_sessions=0, levels_per_session=2, max_distractors=1)
    ).train[0].levels[0]
    example = level.example
    failed_index = (
        int(example.metadata["blue_action_index"])
        if int(example.correct_action_index) == int(example.metadata["red_action_index"])
        else int(example.metadata["red_action_index"])
    )
    tensors = _batch_to_tensors(collate_object_event_examples((example,)), device=torch.device("cpu"))
    observed = _with_observed_candidate_targets(tensors, torch.as_tensor([failed_index], dtype=torch.long))

    with torch.no_grad():
        before = policy_rank_logits_from_predictions(_forward_tensors(model, tensors), tensors["action_mask"])
        deltas = model.observed_event_belief_deltas(
            _forward_tensors(model, observed),
            target_outcome=observed["target_outcome"],
            target_delta=observed["target_delta"],
            actual_action_index=observed["actual_action_index"],
            state_numeric=observed["inputs"]["state_numeric"],
            state_type_ids=observed["inputs"]["state_type_ids"],
            state_mask=observed["inputs"]["state_mask"],
            action_numeric=observed["inputs"]["action_numeric"],
        )
        session_only = policy_rank_logits_from_predictions(
            _forward_tensors(model, tensors, session_belief=deltas.session_delta),
            tensors["action_mask"],
        )
        level_only = policy_rank_logits_from_predictions(
            _forward_tensors(model, tensors, level_belief=deltas.level_delta),
            tensors["action_mask"],
        )

    assert float(level_only[0, failed_index] - before[0, failed_index]) < -8.0
    assert float(session_only[0, failed_index] - level_only[0, failed_index]) > 8.0


def _eval_metrics(
    model: ObjectEventModel,
    cases: tuple[CueClickCase, ...],
    *,
    zero_state: bool = False,
    shuffle_state: bool = False,
    action_only: bool = False,
) -> dict[str, float]:
    model.eval()
    tensors = _batch_tensors(cases)
    state_numeric = tensors["inputs"]["state_numeric"].clone()
    state_type_ids = tensors["inputs"]["state_type_ids"].clone()
    state_mask = tensors["inputs"]["state_mask"].clone()
    if zero_state or action_only:
        state_numeric.zero_()
    if action_only:
        state_type_ids.zero_()
        state_mask.fill_(True)
    if shuffle_state:
        perm = torch.arange(state_numeric.shape[0] - 1, -1, -1)
        state_numeric = state_numeric[perm]
        state_type_ids = state_type_ids[perm]
        state_mask = state_mask[perm]
    with torch.no_grad():
        output = model(
            state_numeric=state_numeric,
            state_type_ids=state_type_ids,
            state_mask=state_mask,
            action_numeric=tensors["inputs"]["action_numeric"],
            action_type_ids=tensors["inputs"]["action_type_ids"],
            direction_ids=tensors["inputs"]["direction_ids"],
            action_mask=tensors["inputs"]["action_mask"],
        )
        rank_logits = policy_rank_logits_from_predictions(output, tensors["action_mask"])
        ranks = torch.argsort(rank_logits, dim=1, descending=True)
        inverse_logits = model.inverse_logits(
            output.action_repr,
            target_outcome=tensors["target_outcome"],
            target_delta=tensors["target_delta"],
            action_mask=tensors["action_mask"],
        )
        inverse_ranks = torch.argsort(inverse_logits, dim=1, descending=True)
        actual = tensors["actual_action_index"]
        top1 = torch.mean((ranks[:, 0] == actual).float()).item()
        top3 = torch.mean((ranks[:, :3] == actual[:, None]).any(dim=1).float()).item()
        inverse_top1 = torch.mean((inverse_ranks[:, 0] == actual).float()).item()
        outcome_loss = F.binary_cross_entropy_with_logits(
            output.outcome_logits[tensors["action_mask"]],
            tensors["candidate_outcome_targets"][tensors["action_mask"]],
        ).item()
        batch = torch.arange(output.outcome_logits.shape[0])
        chosen = rank_logits[batch, actual]
        selected = torch.zeros_like(rank_logits, dtype=torch.bool)
        selected[batch, actual] = True
        negatives = rank_logits.masked_fill(selected, -1.0e9)
        margin = torch.mean(chosen - torch.max(negatives, dim=1).values).item()
        rank_loss = F.cross_entropy(rank_logits, actual).item()
    model.train()
    return {
        "top1": float(top1),
        "top3": float(top3),
        "inverse_top1": float(inverse_top1),
        "outcome_loss": float(outcome_loss),
        "score_margin": float(margin),
        "rank_loss": float(rank_loss),
    }


def _curriculum_rank_metrics(
    model: ObjectEventModel,
    cases: tuple[ObjectEventExample, ...],
    *,
    zero_state: bool = False,
    shuffle_state: bool = False,
    action_only: bool = False,
) -> dict[str, float | torch.Tensor]:
    model.eval()
    tensors = _curriculum_batch_tensors(cases)
    state_numeric = tensors["inputs"]["state_numeric"].clone()
    state_type_ids = tensors["inputs"]["state_type_ids"].clone()
    state_mask = tensors["inputs"]["state_mask"].clone()
    if zero_state or action_only:
        state_numeric.zero_()
    if action_only:
        state_type_ids.zero_()
        state_mask.fill_(True)
    if shuffle_state:
        perm = torch.arange(state_numeric.shape[0] - 1, -1, -1)
        state_numeric = state_numeric[perm]
        state_type_ids = state_type_ids[perm]
        state_mask = state_mask[perm]
    with torch.no_grad():
        output = model(
            state_numeric=state_numeric,
            state_type_ids=state_type_ids,
            state_mask=state_mask,
            action_numeric=tensors["inputs"]["action_numeric"],
            action_type_ids=tensors["inputs"]["action_type_ids"],
            direction_ids=tensors["inputs"]["direction_ids"],
            action_mask=tensors["inputs"]["action_mask"],
        )
        combined_logits = policy_rank_logits_from_predictions(output, tensors["action_mask"])
        actual = tensors["actual_action_index"]
        top1 = torch.mean((combined_logits.argmax(dim=-1) == actual).float()).item()
        rank_loss = F.cross_entropy(combined_logits, actual).item()
        rank_logits = output.rank_logits if output.rank_logits is not None else combined_logits
    model.train()
    return {
        "top1": float(top1),
        "rank_loss": float(rank_loss),
        "combined_logits": combined_logits.detach().cpu(),
        "rank_logits": rank_logits.detach().cpu(),
    }


def _paired_cue_rule_logits(batch: dict[str, torch.Tensor | dict[str, torch.Tensor]]) -> torch.Tensor:
    object_color_field = 0
    object_centroid_y_field = 2
    object_centroid_x_field = 3
    action_contains_object_field = 11
    action_target_color_field = 13
    logits: list[torch.Tensor] = []
    inputs = batch["inputs"]
    assert isinstance(inputs, dict)
    for batch_index in range(inputs["state_numeric"].shape[0]):
        object_mask = inputs["state_type_ids"][batch_index] == OBJECT
        objects = inputs["state_numeric"][batch_index, object_mask]
        cue_index = torch.argmin(objects[:, object_centroid_y_field] + objects[:, object_centroid_x_field])
        cue_color = float(objects[cue_index, object_color_field])
        wants_red = cue_color < (1.5 / 11.0)
        action_contains = inputs["action_numeric"][batch_index, :, action_contains_object_field] > 0.5
        action_colors = inputs["action_numeric"][batch_index, :, action_target_color_field]
        target_color = (2.0 / 11.0) if wants_red else (5.0 / 11.0)
        scores = -(action_colors - target_color).abs()
        scores = scores.masked_fill(~action_contains, -1.0e9)
        scores = scores.masked_fill(~batch["action_mask"][batch_index].bool(), -1.0e9)
        logits.append(scores)
    return torch.stack(logits, dim=0)


class _TokenAttentionPairProbe(torch.nn.Module):
    def __init__(
        self,
        *,
        state_dim: int,
        action_dim: int,
        num_state_types: int = 16,
        num_action_types: int = 16,
        num_directions: int = 8,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        self.state_proj = torch.nn.Linear(state_dim, hidden)
        self.state_type_emb = torch.nn.Embedding(num_state_types, hidden)
        self.action_proj = torch.nn.Linear(action_dim, hidden)
        self.action_type_emb = torch.nn.Embedding(num_action_types, hidden)
        self.direction_emb = torch.nn.Embedding(num_directions, hidden)
        self.q = torch.nn.Linear(hidden, hidden, bias=False)
        self.k = torch.nn.Linear(hidden, hidden, bias=False)
        self.v = torch.nn.Linear(hidden, hidden, bias=False)
        self.score = torch.nn.Sequential(
            torch.nn.LayerNorm(3 * hidden),
            torch.nn.Linear(3 * hidden, hidden),
            torch.nn.GELU(),
            torch.nn.Linear(hidden, 1),
        )

    def forward(
        self,
        state_numeric: torch.Tensor,
        state_type_ids: torch.Tensor,
        state_mask: torch.Tensor,
        action_numeric: torch.Tensor,
        action_type_ids: torch.Tensor,
        direction_ids: torch.Tensor,
        action_mask: torch.Tensor,
    ) -> torch.Tensor:
        state_type_ids = state_type_ids.clamp(min=0, max=self.state_type_emb.num_embeddings - 1)
        action_type_ids = action_type_ids.clamp(min=0, max=self.action_type_emb.num_embeddings - 1)
        direction_ids = direction_ids.clamp(min=0, max=self.direction_emb.num_embeddings - 1)
        state_features = self.state_proj(state_numeric) + self.state_type_emb(state_type_ids)
        action_features = (
            self.action_proj(action_numeric)
            + self.action_type_emb(action_type_ids)
            + self.direction_emb(direction_ids)
        )
        query = self.q(action_features)
        key = self.k(state_features)
        value = self.v(state_features)
        scale = float(query.shape[-1]) ** -0.5
        attention_logits = torch.einsum("bah,bsh->bas", query, key) * scale
        attention_logits = attention_logits.masked_fill(~state_mask.bool().unsqueeze(1), -1.0e9)
        attention = torch.softmax(attention_logits, dim=-1)
        context = torch.einsum("bas,bsh->bah", attention, value)
        logits = self.score(torch.cat([action_features, context, action_features * context], dim=-1)).squeeze(-1)
        return logits.masked_fill(~action_mask.bool(), -1.0e9)


def _state_path_diagnostics(model: ObjectEventModel, cases: tuple[CueClickCase, ...]) -> dict[str, float]:
    model.train()
    tensors = _batch_tensors(cases)
    output = model(**tensors["inputs"])
    rank_logits = policy_rank_logits_from_predictions(output, tensors["action_mask"])
    pre_rank_diff = float((rank_logits[0] - rank_logits[1]).abs().mean().detach()) if rank_logits.shape[0] >= 2 else 0.0
    loss = F.cross_entropy(rank_logits, tensors["actual_action_index"])
    model.zero_grad(set_to_none=True)
    loss.backward()
    state_grad = 0.0
    action_grad = 0.0
    cross_grad = 0.0
    film_grad = 0.0
    for name, parameter in model.named_parameters():
        if parameter.grad is None:
            continue
        grad_sum = float(parameter.grad.detach().abs().sum())
        if "state" in name:
            state_grad += grad_sum
        if "action" in name:
            action_grad += grad_sum
        if "cross" in name or "attn" in name:
            cross_grad += grad_sum
        if "film" in name:
            film_grad += grad_sum
    model.zero_grad(set_to_none=True)
    return {
        "pre_rank_diff": pre_rank_diff,
        "state_grad": state_grad,
        "action_grad": action_grad,
        "cross_grad": cross_grad,
        "film_grad": film_grad,
        "rank_loss": float(loss.detach()),
    }


def _batch_tensors(cases: tuple[CueClickCase, ...]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    state_tokens = [encode_state_tokens(case.state) for case in cases]
    action_tokens = [encode_action_tokens(case.state, case.legal_actions) for case in cases]
    targets = [build_transition_targets(case.transition(), actions=case.legal_actions) for case in cases]
    state_numeric, state_type_ids, state_mask = stack_state_tokens(state_tokens)
    action_numeric, action_type_ids, direction_ids, action_mask = stack_action_tokens(action_tokens)
    return {
        "inputs": {
            "state_numeric": torch.as_tensor(state_numeric, dtype=torch.float32),
            "state_type_ids": torch.as_tensor(state_type_ids, dtype=torch.long),
            "state_mask": torch.as_tensor(state_mask, dtype=torch.bool),
            "action_numeric": torch.as_tensor(action_numeric, dtype=torch.float32),
            "action_type_ids": torch.as_tensor(action_type_ids, dtype=torch.long),
            "direction_ids": torch.as_tensor(direction_ids, dtype=torch.long),
            "action_mask": torch.as_tensor(action_mask, dtype=torch.bool),
        },
        "target_outcome": torch.as_tensor(np.stack([target.outcome for target in targets]), dtype=torch.float32),
        "target_delta": torch.as_tensor(np.stack([target.delta for target in targets]), dtype=torch.float32),
        "actual_action_index": torch.as_tensor([target.actual_action_index for target in targets], dtype=torch.long),
        "action_mask": torch.as_tensor(action_mask, dtype=torch.bool),
        "candidate_value_targets": _value_targets(targets, action_mask.shape[1]),
        "candidate_outcome_targets": _candidate_outcome_targets(targets, action_mask.shape[1]),
    }


def _curriculum_batch_tensors(cases: tuple[ObjectEventExample, ...]) -> dict[str, torch.Tensor | dict[str, torch.Tensor]]:
    batch = collate_object_event_examples(cases)
    return {
        "inputs": {
            "state_numeric": torch.as_tensor(batch.state_numeric, dtype=torch.float32),
            "state_type_ids": torch.as_tensor(batch.state_type_ids, dtype=torch.long),
            "state_mask": torch.as_tensor(batch.state_mask, dtype=torch.bool),
            "action_numeric": torch.as_tensor(batch.action_numeric, dtype=torch.float32),
            "action_type_ids": torch.as_tensor(batch.action_type_ids, dtype=torch.long),
            "direction_ids": torch.as_tensor(batch.direction_ids, dtype=torch.long),
            "action_mask": torch.as_tensor(batch.action_mask, dtype=torch.bool),
        },
        "target_outcome": torch.as_tensor(batch.target_outcome, dtype=torch.float32),
        "target_delta": torch.as_tensor(batch.target_delta, dtype=torch.float32),
        "actual_action_index": torch.as_tensor(batch.actual_action_index, dtype=torch.long),
        "action_mask": torch.as_tensor(batch.action_mask, dtype=torch.bool),
        "candidate_value_targets": torch.as_tensor(batch.candidate_value_targets, dtype=torch.float32),
        "candidate_outcome_targets": torch.as_tensor(batch.candidate_outcome_targets, dtype=torch.float32),
    }


def _value_targets(targets, max_actions: int) -> torch.Tensor:
    values = np.zeros((len(targets), max_actions), dtype=np.float32)
    for row, target in enumerate(targets):
        values[row, int(target.actual_action_index)] = 1.0
    return torch.as_tensor(values, dtype=torch.float32)


def _candidate_outcome_targets(targets, max_actions: int) -> torch.Tensor:
    outcomes = np.zeros((len(targets), max_actions, 10), dtype=np.float32)
    outcomes[:, :, OUT_NO_EFFECT_NONPROGRESS] = 1.0
    for row, target in enumerate(targets):
        index = int(target.actual_action_index)
        outcomes[row, index, :] = 0.0
        outcomes[row, index, OUT_VISIBLE_CHANGE] = 1.0
        outcomes[row, index, OUT_OBJECTIVE_PROGRESS] = 1.0
        outcomes[row, index, OUT_REWARD_PROGRESS] = 1.0
    return torch.as_tensor(outcomes, dtype=torch.float32)


def _paired_cue_cases(seed: int) -> tuple[CueClickCase, CueClickCase]:
    rng = np.random.default_rng(0)
    pair_index = int(seed) * 2
    return _cue_click_case(rng, pair_index), _cue_click_case(rng, pair_index + 1)


def _make_paired_cue_cases(*, num_pairs: int, seed: int) -> tuple[CueClickCase, ...]:
    rng = np.random.default_rng(0)
    cases: list[CueClickCase] = []
    for pair_offset in range(int(num_pairs)):
        pair_index = (int(seed) + pair_offset) * 2
        cases.append(_cue_click_case(rng, pair_index))
        cases.append(_cue_click_case(rng, pair_index + 1))
    return tuple(cases)


def _cue_click_case(rng: np.random.Generator, index: int) -> CueClickCase:
    del rng
    pair_index = int(index // 2)
    cue_mode = int(index % 2)
    pair_rng = np.random.default_rng(10000 + pair_index)
    grid = np.zeros((8, 8), dtype=np.int64)
    cue_color = 1 if cue_mode == 0 else 2
    red_pos = _random_free_cell(pair_rng, {(0, 0)})
    blue_pos = _random_free_cell(pair_rng, {(0, 0), red_pos})
    cue = _object("cue", cue_color, (0, 0), tags=("agent", "clickable"))
    red = _object("red", 2, red_pos)
    blue = _object("blue", 5, blue_pos)
    for obj in (cue, red, blue):
        for row, col in obj.cells:
            grid[row, col] = obj.color
    actions = [f"click:{red_pos[1]}:{red_pos[0]}", f"click:{blue_pos[1]}:{blue_pos[0]}"]
    occupied = {(0, 0), red_pos, blue_pos}
    while len(actions) < 12:
        row, col = _random_free_cell(pair_rng, occupied)
        occupied.add((row, col))
        actions.append(f"click:{col}:{row}")
    pair_rng.shuffle(actions)
    correct = f"click:{red_pos[1]}:{red_pos[0]}" if cue_mode == 0 else f"click:{blue_pos[1]}:{blue_pos[0]}"
    red_action = f"click:{red_pos[1]}:{red_pos[0]}"
    blue_action = f"click:{blue_pos[1]}:{blue_pos[0]}"
    roles = {action: "click" for action in actions}
    state = StructuredState(
        task_id="cue_click",
        episode_id="0",
        step_index=0,
        grid_shape=grid.shape,
        grid_signature=tuple(int(value) for value in grid.reshape(-1)),
        objects=(cue, red, blue),
        relations=(),
        affordances=tuple(actions),
        action_roles=tuple(sorted(roles.items())),
    )
    next_state = StructuredState(
        task_id=state.task_id,
        episode_id=state.episode_id,
        step_index=1,
        grid_shape=state.grid_shape,
        grid_signature=state.grid_signature,
        objects=state.objects,
        relations=(),
        affordances=state.affordances,
        action_roles=state.action_roles,
        flags=(("synthetic_success_latch", "1"),),
    )
    return CueClickCase(
        state=state,
        next_state=next_state,
        action=correct,
        legal_actions=state.affordances,
        correct_action_index=state.affordances.index(correct),
        red_action_index=state.affordances.index(red_action),
        blue_action_index=state.affordances.index(blue_action),
    )


def _object(
    object_id: str,
    color: int,
    cell: tuple[int, int],
    *,
    tags: tuple[str, ...] = (),
) -> ObjectState:
    row, col = cell
    return ObjectState(
        object_id=object_id,
        color=color,
        cells=(cell,),
        bbox=(row, col, row, col),
        centroid=(float(row), float(col)),
        area=1,
        tags=tags,
    )


def _random_free_cell(rng: np.random.Generator, occupied: set[tuple[int, int]]) -> tuple[int, int]:
    while True:
        cell = (int(rng.integers(1, 8)), int(rng.integers(0, 8)))
        if cell not in occupied:
            return cell
