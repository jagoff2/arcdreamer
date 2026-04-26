from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import pickle
from typing import Sequence

import numpy as np
import torch

from arcagi.agents.base import BaseAgent
from arcagi.core.action_schema import build_action_schema, build_action_schema_context, parse_click_action
from arcagi.core.types import ActionName, GridObservation, StructuredState, Transition
from arcagi.learned_online.event_tokens import (
    ACTION_FEATURE_GRID_COL,
    ACTION_FEATURE_HAS_GRID_CELL,
    OUT_ACTION_AVAIL_CHANGED,
    OUT_APPEARED_OR_DISAPPEARED,
    OUT_HARM,
    OUT_MECHANIC_CHANGE,
    OUT_NO_EFFECT_NONPROGRESS,
    OUT_OBJECTIVE_PROGRESS,
    OUT_REWARD_PROGRESS,
    OUT_TERMINAL_PROGRESS,
    OUT_VISIBLE_CHANGE,
    OUT_VISIBLE_ONLY_NONPROGRESS,
    build_transition_targets,
    encode_action_tokens,
    encode_state_tokens,
)
from arcagi.learned_online.object_event_model import (
    ObjectEventModel,
    ObjectEventModelConfig,
    policy_rank_logits_from_predictions,
)


@dataclass(frozen=True)
class ObjectEventDecision:
    action: ActionName
    score: float
    probability: float
    outcome: tuple[float, ...]
    value: float


class LearnedOnlineObjectEventAgent(BaseAgent):
    controller_kind = "learned_online_object_event_v1"
    claim_eligible_arc_controller = True
    arc_competence_validated = False
    learned_online_controller = True
    handles_level_boundaries = True
    uses_trace_replay = False
    uses_state_hash_action_lookup = False
    uses_per_game_runtime_behavior = False
    scores_full_legal_action_surface = True

    def __init__(
        self,
        *,
        seed: int = 0,
        config: ObjectEventModelConfig | None = None,
        device: str | torch.device | None = None,
        online_lr: float = 3.0e-3,
        temperature: float = 0.35,
        epsilon_floor: float = 0.01,
        diagnostic_utility_weight: float = 0.75,
        entropy_tiebreak_weight: float = 0.05,
        diagnostic_mix_max: float = 0.35,
        diagnostic_mix_min_evidence: float = 0.25,
        diagnostic_mix_slope: float = 1.0,
    ) -> None:
        super().__init__(name="learned_online_object_event")
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.config = config or ObjectEventModelConfig()
        self.model = ObjectEventModel(self.config).to(self.device)
        self.online_lr = float(online_lr)
        self.temperature = float(temperature)
        self.epsilon_floor = float(epsilon_floor)
        self.diagnostic_utility_weight = float(diagnostic_utility_weight)
        self.entropy_tiebreak_weight = float(entropy_tiebreak_weight)
        self.diagnostic_mix_max = min(max(float(diagnostic_mix_max), 0.0), 1.0)
        self.diagnostic_mix_min_evidence = min(max(float(diagnostic_mix_min_evidence), 0.0), 1.0)
        self.diagnostic_mix_slope = max(float(diagnostic_mix_slope), 1.0e-6)
        self.session_belief = torch.zeros((self.config.d_model,), dtype=torch.float32, device=self.device)
        self.level_belief = torch.zeros((self.config.d_model,), dtype=torch.float32, device=self.device)
        self.level_epoch = 0
        self.level_step = 0
        self.online_update_count = 0
        self.last_online_loss = 0.0
        self.last_prediction_error = 0.0
        self.last_inverse_loss = 0.0
        self.last_outcome_loss = 0.0
        self.last_delta_loss = 0.0
        self.last_decision: ObjectEventDecision | None = None
        self.last_scores: dict[ActionName, float] = {}
        self.last_probabilities: dict[ActionName, float] = {}
        self.last_outcomes: dict[ActionName, tuple[float, ...]] = {}
        self.last_values: dict[ActionName, float] = {}
        self.last_legal_action_count = 0
        self.last_scored_action_count = 0
        self.last_score_entropy = 0.0
        self.last_effective_action_support = 0.0
        self.last_selected_probability = 0.0
        self.last_predicted_outcome: tuple[float, ...] = tuple([0.0] * 10)
        self.last_observed_outcome: tuple[float, ...] = tuple([0.0] * 10)
        self.last_runtime_rank_logits_used = False
        self.last_runtime_rank_score_mean = 0.0
        self.last_runtime_rank_score_std = 0.0
        self.last_runtime_diagnostic_utility_mean = 0.0
        self.last_runtime_learned_diagnostic_utility_used = False
        self.last_runtime_learned_diagnostic_utility_mean = 0.0
        self.last_runtime_learned_diagnostic_utility_std = 0.0
        self.last_runtime_entropy_tiebreak_std = 0.0
        self.last_runtime_diagnostic_mix = 0.0
        self.last_runtime_diagnostic_mix_model_value = 0.0
        self.last_runtime_diagnostic_mix_evidence_gate = 0.0
        self.last_runtime_rank_weight_effective = 1.0
        self.last_rank_component_stats: dict[str, object] = {}
        self.metadata = self.checkpoint_metadata()

    def reset_episode(self) -> None:
        super().reset_episode()
        self.session_belief = torch.zeros_like(self.session_belief)
        self.level_belief = torch.zeros_like(self.level_belief)
        self.level_epoch = 0
        self.level_step = 0
        self.online_update_count = 0
        self.last_online_loss = 0.0
        self.last_prediction_error = 0.0
        self.last_inverse_loss = 0.0
        self.last_outcome_loss = 0.0
        self.last_delta_loss = 0.0
        self.last_decision = None
        self.last_scores.clear()
        self.last_probabilities.clear()
        self.last_outcomes.clear()
        self.last_values.clear()
        self.last_legal_action_count = 0
        self.last_scored_action_count = 0
        self.last_score_entropy = 0.0
        self.last_effective_action_support = 0.0
        self.last_selected_probability = 0.0
        self.last_predicted_outcome = tuple([0.0] * 10)
        self.last_observed_outcome = tuple([0.0] * 10)
        self.last_runtime_rank_logits_used = False
        self.last_runtime_rank_score_mean = 0.0
        self.last_runtime_rank_score_std = 0.0
        self.last_runtime_diagnostic_utility_mean = 0.0
        self.last_runtime_learned_diagnostic_utility_used = False
        self.last_runtime_learned_diagnostic_utility_mean = 0.0
        self.last_runtime_learned_diagnostic_utility_std = 0.0
        self.last_runtime_entropy_tiebreak_std = 0.0
        self.last_runtime_diagnostic_mix = 0.0
        self.last_runtime_diagnostic_mix_model_value = 0.0
        self.last_runtime_diagnostic_mix_evidence_gate = 0.0
        self.last_runtime_rank_weight_effective = 1.0
        self.last_rank_component_stats = {}

    def reset_level(self) -> None:
        BaseAgent.reset_level(self)
        self._start_new_level()

    def act(self, observation: GridObservation) -> ActionName:
        state = self.observe(observation)
        decisions = self.score_actions_for_state(state, state.affordances)
        if not decisions:
            raise ValueError("learned_online_object_event received an observation with no legal actions")
        actions = tuple(decisions)
        probabilities = np.asarray([decisions[action].probability for action in actions], dtype=np.float64)
        if self.epsilon_floor > 0.0 and float(self.rng.random()) < self.epsilon_floor:
            selected_index = int(self.rng.integers(len(actions)))
        else:
            selected_index = int(np.argmax(probabilities))
        selected = decisions[actions[selected_index]]
        self.last_state = state
        self.last_action = selected.action
        self.last_decision = selected
        self.last_selected_probability = float(selected.probability)
        self.latest_language = (
            "question:test_action_event",
            f"legal_actions:{len(actions)}",
            f"entropy:{self.last_score_entropy:.3f}",
        )
        return selected.action

    def score_actions_for_state(
        self,
        state: StructuredState,
        actions: Sequence[ActionName],
    ) -> dict[ActionName, ObjectEventDecision]:
        action_tuple = tuple(actions)
        if not action_tuple:
            return {}
        with torch.no_grad():
            output, action_batch = self._forward_state_actions(state, action_tuple)
            probabilities, scores, outcomes, values = self._calibrated_distribution(output, action_batch.mask)
            component_stats = self._rank_component_diagnostics(
                output,
                action_tuple,
                action_batch.mask,
                getattr(action_batch, "numeric", None),
            )
        decisions: dict[ActionName, ObjectEventDecision] = {}
        for index, action in enumerate(action_tuple):
            outcome_tuple = tuple(float(value) for value in outcomes[index].tolist())
            decisions[action] = ObjectEventDecision(
                action=action,
                score=float(scores[index]),
                probability=float(probabilities[index]),
                outcome=outcome_tuple,
                value=float(values[index]),
            )
        self.last_scores = {action: decision.score for action, decision in decisions.items()}
        self.last_probabilities = {action: decision.probability for action, decision in decisions.items()}
        self.last_outcomes = {action: decision.outcome for action, decision in decisions.items()}
        self.last_values = {action: decision.value for action, decision in decisions.items()}
        self.last_legal_action_count = len(action_tuple)
        self.last_scored_action_count = len(decisions)
        self.last_score_entropy = _entropy(tuple(decision.probability for decision in decisions.values()))
        self.last_effective_action_support = float(np.exp(self.last_score_entropy))
        self.last_rank_component_stats = component_stats
        return decisions

    def on_transition(self, transition: Transition) -> None:
        actions = tuple(transition.state.affordances)
        if transition.action not in actions:
            return
        target_transition = _external_transition_for_targets(transition)
        targets = build_transition_targets(target_transition, actions=actions)
        output, action_batch = self._forward_state_actions(transition.state, actions)
        device = self.device
        target_outcome = torch.as_tensor(targets.outcome[None, :], dtype=torch.float32, device=device)
        target_delta = torch.as_tensor(targets.delta[None, :], dtype=torch.float32, device=device)
        action_index = torch.as_tensor([targets.actual_action_index], dtype=torch.long, device=device)
        action_mask = torch.as_tensor(action_batch.mask[None, :], dtype=torch.bool, device=device)
        with torch.no_grad():
            predicted = torch.sigmoid(output.outcome_logits[0, targets.actual_action_index]).detach().cpu().numpy()
        self.last_predicted_outcome = tuple(float(value) for value in predicted.tolist())
        self.last_observed_outcome = tuple(float(value) for value in targets.outcome.tolist())

        session_param = torch.nn.Parameter(self.session_belief.detach().clone())
        level_param = torch.nn.Parameter(self.level_belief.detach().clone())
        optimizer = torch.optim.AdamW([*self.model.online_parameters(), session_param, level_param], lr=self.online_lr)
        output = self._forward_state_actions_with_beliefs(transition.state, actions, session_param, level_param)[0]
        state_batch = encode_state_tokens(transition.state)
        action_tokens = encode_action_tokens(transition.state, actions)
        state_numeric = torch.as_tensor(state_batch.numeric[None, :, :], dtype=torch.float32, device=device)
        state_type_ids = torch.as_tensor(state_batch.type_ids[None, :], dtype=torch.long, device=device)
        state_mask = torch.as_tensor(state_batch.mask[None, :], dtype=torch.bool, device=device)
        action_numeric = torch.as_tensor(action_tokens.numeric[None, :, :], dtype=torch.float32, device=device)
        losses = self.model.loss(
            output,
            target_outcome=target_outcome,
            target_delta=target_delta,
            actual_action_index=action_index,
            action_mask=action_mask,
        )
        optimizer.zero_grad(set_to_none=True)
        losses["loss"].backward()
        torch.nn.utils.clip_grad_norm_([*self.model.online_parameters(), session_param, level_param], max_norm=1.0)
        optimizer.step()
        with torch.no_grad():
            event = torch.cat([target_outcome, target_delta], dim=-1)
            residual = torch.tanh(self.model.event_encoder(event)).squeeze(0)
            belief_delta = self.model.observed_event_belief_deltas(
                output,
                target_outcome=target_outcome,
                target_delta=target_delta,
                actual_action_index=action_index,
                state_numeric=state_numeric,
                state_type_ids=state_type_ids,
                state_mask=state_mask,
                action_numeric=action_numeric,
            )
            session_delta = belief_delta.session_delta.squeeze(0)
            level_delta = belief_delta.level_delta.squeeze(0)
            self.session_belief = (session_param.detach() + 0.03 * residual + session_delta).detach()
            self.level_belief = (level_param.detach() + 0.05 * residual + level_delta).detach()
        self.online_update_count += 1
        self.last_online_loss = float(losses["loss"].detach().cpu())
        self.last_prediction_error = float(
            np.mean((np.asarray(self.last_predicted_outcome, dtype=np.float32) - targets.outcome) ** 2)
        )
        self.last_inverse_loss = float(losses["inverse_loss"].detach().cpu())
        self.last_outcome_loss = float(losses["outcome_loss"].detach().cpu())
        self.last_delta_loss = float(losses["delta_loss"].detach().cpu())
        if bool(transition.info.get("level_boundary", False)) or targets.outcome[OUT_OBJECTIVE_PROGRESS] > 0.0 or targets.outcome[OUT_TERMINAL_PROGRESS] > 0.0:
            self._start_new_level()
        else:
            self.level_step += 1

    def diagnostics(self) -> dict[str, object]:
        top_action = ""
        top_score = 0.0
        if self.last_scores:
            top_action, top_score = max(self.last_scores.items(), key=lambda item: item[1])
        selected = self.last_decision.action if self.last_decision is not None else ""
        selected_score = self.last_decision.score if self.last_decision is not None else 0.0
        coord_start = int(getattr(self.model.coordinate_noeffect_memory_rank, "start", 0))
        coord_stop = int(getattr(self.model.coordinate_noeffect_memory_rank, "stop", coord_start))
        coord_memory = self.level_belief[coord_start:coord_stop] if coord_stop > coord_start else self.level_belief[:0]
        coord_count = float(coord_memory[0].detach().cpu()) if coord_memory.numel() > 0 else 0.0
        coord_norm = float(torch.linalg.vector_norm(coord_memory).detach().cpu()) if coord_memory.numel() > 0 else 0.0
        axis_start = int(getattr(self.model.axis_noeffect_memory_rank, "start", 0))
        axis_stop = int(getattr(self.model.axis_noeffect_memory_rank, "stop", axis_start))
        axis_memory = self.level_belief[axis_start:axis_stop] if axis_stop > axis_start else self.level_belief[:0]
        axis_count = float(axis_memory[0].detach().cpu()) if axis_memory.numel() > 0 else 0.0
        axis_norm = float(torch.linalg.vector_norm(axis_memory).detach().cpu()) if axis_memory.numel() > 0 else 0.0
        family_start = int(getattr(self.model.action_family_belief, "start", 0))
        family_stop = int(getattr(self.model.action_family_belief, "stop", family_start))
        family_count = int(getattr(self.model.action_family_belief, "num_families", 0))
        family_memory = self.level_belief[family_start:family_stop] if family_stop > family_start else self.level_belief[:0]
        family_norm = float(torch.linalg.vector_norm(family_memory).detach().cpu()) if family_memory.numel() > 0 else 0.0
        family_evidence = float(torch.sum(family_memory[:family_count].clamp_min(0.0)).detach().cpu()) if family_count > 0 else 0.0
        family_noeffect = (
            float(torch.sum(family_memory[2 * family_count : 3 * family_count].clamp_min(0.0)).detach().cpu())
            if family_count > 0
            else 0.0
        )
        basis_start = int(getattr(self.model.action_basis_belief, "start", 0))
        basis_stop = int(getattr(self.model.action_basis_belief, "stop", basis_start))
        basis_count = int(getattr(self.model.action_basis_belief, "num_bases", 0))
        basis_memory = self.level_belief[basis_start:basis_stop] if basis_stop > basis_start else self.level_belief[:0]
        basis_norm = float(torch.linalg.vector_norm(basis_memory).detach().cpu()) if basis_memory.numel() > 0 else 0.0
        basis_evidence = float(torch.sum(basis_memory[:basis_count].clamp_min(0.0)).detach().cpu()) if basis_count > 0 else 0.0
        basis_noeffect = (
            float(torch.sum(basis_memory[2 * basis_count : 3 * basis_count].clamp_min(0.0)).detach().cpu())
            if basis_count > 0
            else 0.0
        )
        return {
            "controller_kind": self.controller_kind,
            "claim_eligible_arc_controller": bool(self.claim_eligible_arc_controller),
            "arc_competence_validated": bool(self.arc_competence_validated),
            "learned_online_controller": True,
            "legal_action_count": int(self.last_legal_action_count),
            "scored_action_count": int(self.last_scored_action_count),
            "full_dense_surface_scored": bool(self.last_legal_action_count == self.last_scored_action_count),
            "object_event_bridge_enabled": True,
            "object_event_legal_action_count": int(self.last_legal_action_count),
            "object_event_scored_action_count": int(self.last_scored_action_count),
            "object_event_online_update_count": int(self.online_update_count),
            "object_event_session_belief_norm": float(torch.linalg.vector_norm(self.session_belief).detach().cpu()),
            "object_event_level_belief_norm": float(torch.linalg.vector_norm(self.level_belief).detach().cpu()),
            "object_event_coordinate_noeffect_memory_norm": coord_norm,
            "object_event_coordinate_noeffect_count": coord_count,
            "object_event_axis_noeffect_memory_norm": axis_norm,
            "object_event_axis_noeffect_count": axis_count,
            "object_event_action_family_belief_norm": family_norm,
            "object_event_action_family_evidence_count": family_evidence,
            "object_event_action_family_noeffect_count": family_noeffect,
            "object_event_action_basis_belief_norm": basis_norm,
            "object_event_action_basis_evidence_count": basis_evidence,
            "object_event_action_basis_noeffect_count": basis_noeffect,
            "object_event_action_surface_capped": False,
            "object_event_oracle_support_used": False,
            "object_event_trace_replay_used": False,
            "object_event_graph_controller_used": False,
            "selected_action": selected,
            "selected_action_score": float(selected_score),
            "selected_action_probability": float(self.last_selected_probability),
            "score_entropy": float(self.last_score_entropy),
            "effective_action_support": float(self.last_effective_action_support),
            "top_action": top_action,
            "top_action_score": float(top_score),
            "top_action_family": self._action_family(top_action),
            "online_update_count": int(self.online_update_count),
            "last_online_loss": float(self.last_online_loss),
            "last_prediction_error": float(self.last_prediction_error),
            "last_inverse_loss": float(self.last_inverse_loss),
            "last_outcome_loss": float(self.last_outcome_loss),
            "last_delta_loss": float(self.last_delta_loss),
            "session_belief_norm": float(torch.linalg.vector_norm(self.session_belief).detach().cpu()),
            "level_belief_norm": float(torch.linalg.vector_norm(self.level_belief).detach().cpu()),
            "coordinate_noeffect_memory_norm": coord_norm,
            "coordinate_noeffect_count": coord_count,
            "axis_noeffect_memory_norm": axis_norm,
            "axis_noeffect_count": axis_count,
            "action_family_belief_norm": family_norm,
            "action_family_evidence_count": family_evidence,
            "action_family_noeffect_count": family_noeffect,
            "action_basis_belief_norm": basis_norm,
            "action_basis_evidence_count": basis_evidence,
            "action_basis_noeffect_count": basis_noeffect,
            "level_epoch": int(self.level_epoch),
            "level_step": int(self.level_step),
            "last_predicted_outcome": tuple(self.last_predicted_outcome),
            "last_observed_outcome": tuple(self.last_observed_outcome),
            "runtime_rank_logits_used": bool(self.last_runtime_rank_logits_used),
            "runtime_rank_score_mean": float(self.last_runtime_rank_score_mean),
            "runtime_rank_score_std": float(self.last_runtime_rank_score_std),
            "runtime_diagnostic_utility_mean": float(self.last_runtime_diagnostic_utility_mean),
            "runtime_learned_diagnostic_utility_used": bool(self.last_runtime_learned_diagnostic_utility_used),
            "runtime_learned_diagnostic_utility_mean": float(self.last_runtime_learned_diagnostic_utility_mean),
            "runtime_learned_diagnostic_utility_std": float(self.last_runtime_learned_diagnostic_utility_std),
            "runtime_entropy_tiebreak_std": float(self.last_runtime_entropy_tiebreak_std),
            "runtime_diagnostic_mix": float(self.last_runtime_diagnostic_mix),
            "runtime_diagnostic_mix_effective": float(self.last_runtime_diagnostic_mix),
            "runtime_diagnostic_mix_model_value": float(self.last_runtime_diagnostic_mix_model_value),
            "runtime_diagnostic_mix_evidence_gate": float(self.last_runtime_diagnostic_mix_evidence_gate),
            "runtime_diagnostic_mix_max": float(self.diagnostic_mix_max),
            "runtime_rank_weight_effective": float(self.last_runtime_rank_weight_effective),
            **self.last_rank_component_stats,
            "runtime_greedy_rank_selection": True,
            "runtime_trace_cursor": False,
            "runtime_action_sequence_replay": False,
            "runtime_state_hash_to_action": False,
            "runtime_per_game_behavior": False,
            "runtime_graph_search_solver": False,
            "runtime_action_pattern_enumerator": False,
            "runtime_external_api_or_knowledge": False,
            "last_top_scores": self._last_top_scores(limit=12),
        }

    def state_dict(self) -> dict[str, object]:
        return {
            "seed": int(self.seed),
            "config": self.config.to_dict(),
            "model_state": self.model.state_dict(),
            "metadata": self.checkpoint_metadata(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.seed = int(state.get("seed", self.seed) or self.seed)
        config_data = state.get("config", {})
        if isinstance(config_data, dict):
            self.config = ObjectEventModelConfig.from_dict(config_data)
            self.model = ObjectEventModel(self.config).to(self.device)
            self.session_belief = torch.zeros((self.config.d_model,), dtype=torch.float32, device=self.device)
            self.level_belief = torch.zeros((self.config.d_model,), dtype=torch.float32, device=self.device)
        model_state = state.get("model_state", {})
        if isinstance(model_state, dict):
            current = self.model.state_dict()
            filtered = {
                key: value
                for key, value in model_state.items()
                if key in current and hasattr(value, "shape") and tuple(value.shape) == tuple(current[key].shape)
            }
            self.model.load_state_dict(filtered, strict=False)
        metadata = state.get("metadata", {})
        self.metadata = metadata if isinstance(metadata, dict) else self.checkpoint_metadata()

    def save_checkpoint(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self.state_dict(), handle)

    @classmethod
    def from_checkpoint(
        cls,
        path: str | Path,
        *,
        seed: int = 0,
        device: str | torch.device | None = None,
    ) -> "LearnedOnlineObjectEventAgent":
        agent = cls(seed=seed, device=device)
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        if not isinstance(state, dict):
            raise ValueError(f"learned online object-event checkpoint at {path!s} is not a mapping")
        agent.load_state_dict(state)
        return agent

    @staticmethod
    def checkpoint_metadata(
        *,
        trace_bootstrap_used: bool = False,
        trace_bootstrap_transition_count: int = 0,
    ) -> dict[str, object]:
        return {
            "controller_kind": "learned_online_object_event_v1",
            "architecture": "object_event_state_space_v1",
            "arc_competence_validated": False,
            "claim_eligible_arc_controller": True,
            "learned_online_controller": True,
            "handles_level_boundaries": True,
            "scores_full_legal_action_surface": True,
            "runtime_trace_cursor": False,
            "runtime_action_sequence_replay": False,
            "runtime_state_hash_to_action": False,
            "runtime_per_game_behavior": False,
            "runtime_graph_search_solver": False,
            "runtime_action_pattern_enumerator": False,
            "runtime_external_api_or_knowledge": False,
            "runtime_uses_policy_rank_logits": True,
            "runtime_uses_learned_diagnostic_utility": True,
            "runtime_uncertainty_diagnostic_utility": True,
            "runtime_bounded_diagnostic_mix": True,
            "runtime_diagnostic_mix_max": 0.35,
            "runtime_greedy_rank_selection": True,
            "trace_bootstrap_used": bool(trace_bootstrap_used),
            "trace_bootstrap_transition_count": int(trace_bootstrap_transition_count),
            "trace_bootstrap_runtime_replay": False,
            "stores_teacher_action_sequence": False,
            "stores_state_hash_to_action": False,
            "online_update_enabled": True,
            "online_update_from_transition_error": True,
            "online_update_params": [
                "online_adapter",
                "fast_outcome_head",
                "fast_delta_head",
                "fast_value_head",
                "observed_event_belief_encoder",
                "level_belief_coordinate_noeffect_slots",
                "level_belief_axis_noeffect_slots",
                "level_belief_action_basis_slots",
                "level_belief_action_family_slots",
                "session_belief",
                "level_belief",
            ],
        }

    def _forward_state_actions(self, state: StructuredState, actions: Sequence[ActionName]):
        return self._forward_state_actions_with_beliefs(state, actions, self.session_belief, self.level_belief)

    def _forward_state_actions_with_beliefs(
        self,
        state: StructuredState,
        actions: Sequence[ActionName],
        session_belief: torch.Tensor,
        level_belief: torch.Tensor,
    ):
        state_batch = encode_state_tokens(state)
        action_batch = encode_action_tokens(state, actions)
        output = self.model(
            state_numeric=torch.as_tensor(state_batch.numeric[None, :, :], dtype=torch.float32, device=self.device),
            state_type_ids=torch.as_tensor(state_batch.type_ids[None, :], dtype=torch.long, device=self.device),
            state_mask=torch.as_tensor(state_batch.mask[None, :], dtype=torch.bool, device=self.device),
            action_numeric=torch.as_tensor(action_batch.numeric[None, :, :], dtype=torch.float32, device=self.device),
            action_type_ids=torch.as_tensor(action_batch.action_type_ids[None, :], dtype=torch.long, device=self.device),
            direction_ids=torch.as_tensor(action_batch.direction_ids[None, :], dtype=torch.long, device=self.device),
            action_mask=torch.as_tensor(action_batch.mask[None, :], dtype=torch.bool, device=self.device),
            session_belief=session_belief.reshape(1, -1),
            level_belief=level_belief.reshape(1, -1),
        )
        return output, action_batch

    def _calibrated_distribution(self, output, action_mask: np.ndarray):
        outcome_probs = torch.sigmoid(output.outcome_logits[0]).detach().cpu().numpy()
        value = torch.sigmoid(output.value_logits[0]).detach().cpu().numpy()
        valid = np.asarray(action_mask, dtype=bool)
        mask_tensor = torch.as_tensor(action_mask[None, :], dtype=torch.bool, device=output.outcome_logits.device)
        with torch.no_grad():
            rank_logits = policy_rank_logits_from_predictions(output, mask_tensor)[0].detach().cpu().numpy()
        rank_scores = _standardize_valid(rank_logits, valid)
        entropy = _binary_entropy(outcome_probs)
        diagnostic_logits = getattr(output, "diagnostic_utility_logits", None)
        if diagnostic_logits is None:
            learned_diagnostic = np.zeros_like(rank_scores, dtype=np.float64)
            learned_diagnostic_raw = learned_diagnostic
            learned_used = False
        else:
            learned_diagnostic_raw = diagnostic_logits[0].detach().cpu().numpy()
            learned_diagnostic = _standardize_valid(learned_diagnostic_raw, valid)
            learned_used = True
        entropy_tiebreak = _standardize_valid(entropy, valid)
        mix_logit = getattr(output, "diagnostic_mix_logit", None)
        if mix_logit is None:
            model_mix = 0.0
        else:
            model_mix = float(torch.sigmoid(mix_logit[0]).detach().cpu())
        evidence_strength = self._diagnostic_evidence_strength()
        denominator = max(1.0 - self.diagnostic_mix_min_evidence, 1.0e-6)
        evidence_gate = np.clip((evidence_strength - self.diagnostic_mix_min_evidence) / denominator, 0.0, 1.0)
        evidence_gate = float(evidence_gate ** self.diagnostic_mix_slope)
        diagnostic_mix = self.diagnostic_mix_max * evidence_gate * min(max(model_mix, 0.0), 1.0)
        diagnostic_utility = diagnostic_mix * learned_diagnostic + self.entropy_tiebreak_weight * entropy_tiebreak
        rank_weight = 1.0 - diagnostic_mix
        scores = rank_weight * rank_scores + diagnostic_utility
        scores = np.asarray(scores, dtype=np.float64)
        logits = np.full_like(scores, -1.0e9, dtype=np.float64)
        logits[valid] = scores[valid] / max(self.temperature, 1.0e-6)
        logits -= np.max(logits[valid])
        probs = np.exp(logits)
        probs[~valid] = 0.0
        probs /= max(float(np.sum(probs)), 1.0e-12)
        eps = min(max(self.epsilon_floor, 0.0), 1.0)
        if valid.any():
            floor = np.zeros_like(probs)
            floor[valid] = 1.0 / float(np.count_nonzero(valid))
            probs = (1.0 - eps) * probs + eps * floor
        self.last_runtime_rank_logits_used = True
        self.last_runtime_rank_score_mean = float(np.mean(rank_scores[valid])) if valid.any() else 0.0
        self.last_runtime_rank_score_std = float(np.std(rank_scores[valid])) if valid.any() else 0.0
        self.last_runtime_diagnostic_utility_mean = float(np.mean(diagnostic_utility[valid])) if valid.any() else 0.0
        self.last_runtime_learned_diagnostic_utility_used = bool(learned_used)
        self.last_runtime_learned_diagnostic_utility_mean = (
            float(np.mean(learned_diagnostic_raw[valid])) if valid.any() else 0.0
        )
        self.last_runtime_learned_diagnostic_utility_std = (
            float(np.std(learned_diagnostic[valid])) if valid.any() else 0.0
        )
        self.last_runtime_entropy_tiebreak_std = float(np.std(entropy_tiebreak[valid])) if valid.any() else 0.0
        self.last_runtime_diagnostic_mix = float(diagnostic_mix)
        self.last_runtime_diagnostic_mix_model_value = float(model_mix)
        self.last_runtime_diagnostic_mix_evidence_gate = float(evidence_gate)
        self.last_runtime_rank_weight_effective = float(rank_weight)
        return probs, scores, outcome_probs, value

    def _diagnostic_evidence_strength(self) -> float:
        coord_start = int(getattr(self.model.coordinate_noeffect_memory_rank, "start", 0))
        coord_stop = int(getattr(self.model.coordinate_noeffect_memory_rank, "stop", coord_start))
        axis_start = int(getattr(self.model.axis_noeffect_memory_rank, "start", 0))
        axis_stop = int(getattr(self.model.axis_noeffect_memory_rank, "stop", axis_start))
        coord_count = float(self.level_belief[coord_start].detach().cpu()) if coord_stop > coord_start else 0.0
        axis_count = float(self.level_belief[axis_start].detach().cpu()) if axis_stop > axis_start else 0.0
        family_start = int(getattr(self.model.action_family_belief, "start", 0))
        family_count = int(getattr(self.model.action_family_belief, "num_families", 0))
        family_noeffect = 0.0
        if family_count > 0:
            start = family_start + 2 * family_count
            stop = family_start + 3 * family_count
            family_noeffect = float(torch.sum(self.level_belief[start:stop].clamp_min(0.0)).detach().cpu())
        basis_start = int(getattr(self.model.action_basis_belief, "start", 0))
        basis_count = int(getattr(self.model.action_basis_belief, "num_bases", 0))
        basis_noeffect = 0.0
        if basis_count > 0:
            start = basis_start + 2 * basis_count
            stop = basis_start + 3 * basis_count
            basis_noeffect = float(torch.sum(self.level_belief[start:stop].clamp_min(0.0)).detach().cpu())
        evidence = max(coord_count, 0.0) + max(axis_count, 0.0) + max(family_noeffect, 0.0) + max(basis_noeffect, 0.0)
        return float(np.tanh(evidence))

    def _rank_component_diagnostics(
        self,
        output,
        actions: Sequence[ActionName],
        action_mask: np.ndarray,
        action_numeric: np.ndarray | None = None,
    ) -> dict[str, object]:
        components = getattr(output, "rank_components", None)
        valid = np.asarray(action_mask, dtype=bool)
        if components is None or not np.any(valid):
            return {}
        action_tuple = tuple(actions)

        def values(name: str) -> np.ndarray:
            tensor = getattr(components, name, None)
            if tensor is None:
                return np.zeros_like(values("total"))
            return tensor[0].detach().cpu().numpy().astype(np.float64)

        stats: dict[str, object] = {}
        for name in ("base", "relation", "failed_action", "coordinate_noeffect", "axis_noeffect", "total"):
            raw = values(name)
            active = raw[valid]
            stats[f"rank_component_{name}_std"] = float(np.std(active)) if active.size else 0.0
            stats[f"rank_component_{name}_mean"] = float(np.mean(active)) if active.size else 0.0
            if active.size:
                valid_indices = np.flatnonzero(valid)
                top_index = int(valid_indices[int(np.argmax(active))])
                stats[f"rank_component_{name}_top_action"] = action_tuple[top_index]
        for name in ("base", "relation", "failed_action", "coordinate_noeffect", "axis_noeffect"):
            raw = values(f"{name}_raw")
            active = raw[valid]
            stats[f"rank_component_{name}_raw_std"] = float(np.std(active)) if active.size else 0.0
            stats[f"rank_component_{name}_raw_mean"] = float(np.mean(active)) if active.size else 0.0
        gate_values = getattr(components, "component_gates", None)
        if gate_values is not None:
            gates = gate_values.detach().cpu().numpy().astype(np.float64)
            for index, name in enumerate(("base", "relation", "failed_action", "coordinate_noeffect", "axis_noeffect")):
                if index < gates.shape[0]:
                    stats[f"rank_component_gate_{name}"] = float(gates[index])
        boost = getattr(components, "noeffect_boost", None)
        if boost is not None:
            stats["rank_component_noeffect_boost_mean"] = float(np.mean(boost.detach().cpu().numpy().astype(np.float64)))
        relation = getattr(self.model, "event_relation_memory_rank", None)
        if relation is not None:
            stats["relation_object_prior_scale"] = float(2.0 * torch.sigmoid(relation.object_prior_scale_raw).detach().cpu())
            stats["relation_positive_prior_scale"] = float(4.0 * torch.sigmoid(relation.positive_prior_scale_raw).detach().cpu())
            stats["relation_negative_prior_scale"] = float(4.0 * torch.sigmoid(relation.negative_prior_scale_raw).detach().cpu())
            stats["relation_repeat_penalty_scale"] = float(6.0 * torch.sigmoid(relation.repeat_penalty_scale_raw).detach().cpu())
        diagnostic_logits = getattr(output, "diagnostic_utility_logits", None)
        if diagnostic_logits is not None:
            diagnostic_raw = diagnostic_logits[0].detach().cpu().numpy().astype(np.float64)
            diagnostic = _standardize_valid(diagnostic_raw, valid)
            active = diagnostic[valid]
            stats["learned_diagnostic_utility_raw_std"] = float(np.std(diagnostic_raw[valid])) if active.size else 0.0
            stats["learned_diagnostic_utility_std"] = float(np.std(active)) if active.size else 0.0
            stats["learned_diagnostic_utility_mean"] = float(np.mean(active)) if active.size else 0.0
            if active.size:
                valid_indices = np.flatnonzero(valid)
                top_index = int(valid_indices[int(np.argmax(active))])
                stats["learned_diagnostic_utility_top_action"] = action_tuple[top_index]
                stats["runtime_rank_diag_top_action_agreement"] = float(
                    action_tuple[top_index] == stats.get("rank_component_total_top_action", "")
                )
        else:
            stats["learned_diagnostic_utility_raw_std"] = 0.0
            stats["learned_diagnostic_utility_std"] = 0.0
            stats["learned_diagnostic_utility_mean"] = 0.0
            stats["runtime_rank_diag_top_action_agreement"] = 0.0
        mix_logit = getattr(output, "diagnostic_mix_logit", None)
        if mix_logit is not None:
            stats["diagnostic_mix_model_value"] = float(torch.sigmoid(mix_logit[0]).detach().cpu())
        else:
            stats["diagnostic_mix_model_value"] = 0.0
        family_features = getattr(output, "action_family_posterior_features", None)
        if family_features is not None:
            posterior = family_features[0].detach().cpu().numpy().astype(np.float64)
            active = posterior[valid]
            if active.size:
                stats["action_family_belief_mean_effect_mean"] = float(np.mean(active[:, 0]))
                stats["action_family_belief_uncertainty_mean"] = float(np.mean(active[:, 1]))
                stats["action_family_belief_count_mean"] = float(np.mean(active[:, 2]))
                stats["action_family_belief_noeffect_mean"] = float(np.mean(active[:, 3]))
                stats["action_family_belief_uncertainty_top_value"] = float(np.max(active[:, 1]))
                stats["action_family_belief_noeffect_top_value"] = float(np.max(active[:, 3]))
            else:
                stats["action_family_belief_mean_effect_mean"] = 0.0
                stats["action_family_belief_uncertainty_mean"] = 0.0
                stats["action_family_belief_count_mean"] = 0.0
                stats["action_family_belief_noeffect_mean"] = 0.0
                stats["action_family_belief_uncertainty_top_value"] = 0.0
                stats["action_family_belief_noeffect_top_value"] = 0.0
        else:
            stats["action_family_belief_mean_effect_mean"] = 0.0
            stats["action_family_belief_uncertainty_mean"] = 0.0
            stats["action_family_belief_count_mean"] = 0.0
            stats["action_family_belief_noeffect_mean"] = 0.0
            stats["action_family_belief_uncertainty_top_value"] = 0.0
            stats["action_family_belief_noeffect_top_value"] = 0.0
        basis_features = getattr(output, "action_basis_posterior_features", None)
        if basis_features is not None:
            posterior = basis_features[0].detach().cpu().numpy().astype(np.float64)
            active = posterior[valid]
            if active.size:
                stats["action_basis_belief_mean_effect_mean"] = float(np.mean(active[:, 0]))
                stats["action_basis_belief_uncertainty_mean"] = float(np.mean(active[:, 1]))
                stats["action_basis_belief_count_mean"] = float(np.mean(active[:, 2]))
                stats["action_basis_belief_noeffect_mean"] = float(np.mean(active[:, 3]))
                stats["action_basis_belief_uncertainty_top_value"] = float(np.max(active[:, 1]))
                stats["action_basis_belief_noeffect_top_value"] = float(np.max(active[:, 3]))
            else:
                stats["action_basis_belief_mean_effect_mean"] = 0.0
                stats["action_basis_belief_uncertainty_mean"] = 0.0
                stats["action_basis_belief_count_mean"] = 0.0
                stats["action_basis_belief_noeffect_mean"] = 0.0
                stats["action_basis_belief_uncertainty_top_value"] = 0.0
                stats["action_basis_belief_noeffect_top_value"] = 0.0
        else:
            stats["action_basis_belief_mean_effect_mean"] = 0.0
            stats["action_basis_belief_uncertainty_mean"] = 0.0
            stats["action_basis_belief_count_mean"] = 0.0
            stats["action_basis_belief_noeffect_mean"] = 0.0
            stats["action_basis_belief_uncertainty_top_value"] = 0.0
            stats["action_basis_belief_noeffect_top_value"] = 0.0
        family_probs = getattr(output, "action_family_probs", None)
        if family_probs is not None and family_probs.shape[-1] > 0:
            probs = family_probs[0].detach().cpu().numpy().astype(np.float64)
            active = probs[valid]
            if active.size:
                usage = np.mean(active, axis=0)
                usage = usage / max(float(np.sum(usage)), 1.0e-12)
                surface_entropy = -float(np.sum(usage * np.log(np.clip(usage, 1.0e-12, 1.0))))
                per_action_entropy = -np.sum(active * np.log(np.clip(active, 1.0e-12, 1.0)), axis=-1)
                stats["family_assignment_surface_entropy"] = surface_entropy
                stats["family_assignment_per_action_entropy"] = float(np.mean(per_action_entropy))
                stats["family_assignment_usage_min"] = float(np.min(usage))
                stats["family_assignment_usage_max"] = float(np.max(usage))
                stats["family_assignment_effective_count"] = float(np.exp(surface_entropy))
            else:
                stats["family_assignment_surface_entropy"] = 0.0
                stats["family_assignment_per_action_entropy"] = 0.0
                stats["family_assignment_usage_min"] = 0.0
                stats["family_assignment_usage_max"] = 0.0
                stats["family_assignment_effective_count"] = 0.0
        else:
            stats["family_assignment_surface_entropy"] = 0.0
            stats["family_assignment_per_action_entropy"] = 0.0
            stats["family_assignment_usage_min"] = 0.0
            stats["family_assignment_usage_max"] = 0.0
            stats["family_assignment_effective_count"] = 0.0
        if action_numeric is not None and int(getattr(self.model.action_basis_belief, "num_bases", 0)) > 0:
            numeric = torch.as_tensor(
                np.asarray(action_numeric, dtype=np.float32)[None, :, :],
                dtype=torch.float32,
                device=output.outcome_logits.device,
            )
            with torch.no_grad():
                basis = self.model.action_basis_belief.basis(numeric)[0].detach().cpu().numpy().astype(np.float64)
            active = basis[valid]
            if active.size:
                usage = np.mean(active, axis=0)
                usage = usage / max(float(np.sum(usage)), 1.0e-12)
                surface_entropy = -float(np.sum(usage * np.log(np.clip(usage, 1.0e-12, 1.0))))
                per_action_entropy = -np.sum(active * np.log(np.clip(active, 1.0e-12, 1.0)), axis=-1)
                stats["basis_assignment_surface_entropy"] = surface_entropy
                stats["basis_assignment_per_action_entropy"] = float(np.mean(per_action_entropy))
                stats["basis_assignment_usage_min"] = float(np.min(usage))
                stats["basis_assignment_usage_max"] = float(np.max(usage))
                stats["basis_assignment_effective_count"] = float(np.exp(surface_entropy))
            else:
                stats["basis_assignment_surface_entropy"] = 0.0
                stats["basis_assignment_per_action_entropy"] = 0.0
                stats["basis_assignment_usage_min"] = 0.0
                stats["basis_assignment_usage_max"] = 0.0
                stats["basis_assignment_effective_count"] = 0.0
        else:
            stats["basis_assignment_surface_entropy"] = 0.0
            stats["basis_assignment_per_action_entropy"] = 0.0
            stats["basis_assignment_usage_min"] = 0.0
            stats["basis_assignment_usage_max"] = 0.0
            stats["basis_assignment_effective_count"] = 0.0
        for name in (
            "relation_object_prior",
            "relation_positive_prior",
            "relation_negative_prior",
            "relation_repeat_penalty",
            "relation_contradiction_gate",
        ):
            raw = values(name)
            active = raw[valid]
            stats[f"{name}_mean"] = float(np.mean(active)) if active.size else 0.0
            stats[f"{name}_std"] = float(np.std(active)) if active.size else 0.0
        stats["top_score_same_x_fraction"] = 0.0
        stats["top_score_same_mapped_col_fraction"] = 0.0
        stats["rank_component_axis_failed_column_delta_mean"] = 0.0
        stats["rank_component_relation_minus_axis_failed_column_mean"] = 0.0
        stats["rank_component_relation_minus_noeffect_failed_column_mean"] = 0.0
        stats["relation_contradiction_gate_failed_column_mean"] = 0.0
        stats["relation_object_prior_failed_column_mean"] = 0.0
        stats["relation_positive_prior_failed_column_mean"] = 0.0
        stats["relation_prior_after_contradiction_failed_column_mean"] = 0.0
        selected = self.last_decision.action if self.last_decision is not None else ""
        reference = selected or str(stats.get("rank_component_total_top_action", ""))
        click = parse_click_action(reference)
        if click is not None:
            same_x = []
            for index, action in enumerate(action_tuple):
                parsed = parse_click_action(action)
                same_x.append(bool(valid[index] and parsed is not None and int(parsed[0]) == int(click[0])))
            same_x_mask = np.asarray(same_x, dtype=bool)
            same_mapped_col_mask = same_x_mask
            if action_numeric is not None:
                numeric = np.asarray(action_numeric, dtype=np.float64)
                has_grid = numeric[:, ACTION_FEATURE_HAS_GRID_CELL] > 0.5
                grid_col = numeric[:, ACTION_FEATURE_GRID_COL]
                top_reference_index = int(np.argmax(values("total")[valid]))
                valid_indices_for_ref = np.flatnonzero(valid)
                reference_index = int(valid_indices_for_ref[top_reference_index])
                if bool(has_grid[reference_index]):
                    same_mapped_col_mask = valid & has_grid & (np.abs(grid_col - grid_col[reference_index]) <= 0.15)
            total = values("total")
            axis = values("axis_noeffect")
            coordinate = values("coordinate_noeffect")
            failed_action = values("failed_action")
            relation = values("relation")
            relation_object_prior = values("relation_object_prior")
            relation_positive_prior = values("relation_positive_prior")
            relation_negative_prior = values("relation_negative_prior")
            relation_repeat_penalty = values("relation_repeat_penalty")
            relation_prior_after = relation_object_prior + relation_positive_prior + relation_negative_prior - relation_repeat_penalty
            relation_gate = values("relation_contradiction_gate")
            top_count = min(12, int(np.count_nonzero(valid)))
            if top_count > 0:
                top_indices = np.argsort(total[valid])[-top_count:]
                valid_indices = np.flatnonzero(valid)
                actual_top = valid_indices[top_indices]
                stats["top_score_same_x_fraction"] = float(np.mean(same_x_mask[actual_top]))
                stats["top_score_same_mapped_col_fraction"] = float(np.mean(same_mapped_col_mask[actual_top]))
            if np.any(same_x_mask):
                stats["rank_component_axis_failed_column_delta_mean"] = float(np.mean(axis[same_x_mask]))
                stats["rank_component_relation_minus_axis_failed_column_mean"] = float(
                    np.mean(relation[same_x_mask] - axis[same_x_mask])
                )
                stats["rank_component_relation_minus_noeffect_failed_column_mean"] = float(
                    np.mean(relation[same_x_mask] - (axis[same_x_mask] + coordinate[same_x_mask] + failed_action[same_x_mask]))
                )
                stats["relation_contradiction_gate_failed_column_mean"] = float(np.mean(relation_gate[same_x_mask]))
                stats["relation_object_prior_failed_column_mean"] = float(np.mean(relation_object_prior[same_x_mask]))
                stats["relation_positive_prior_failed_column_mean"] = float(np.mean(relation_positive_prior[same_x_mask]))
                stats["relation_prior_after_contradiction_failed_column_mean"] = float(np.mean(relation_prior_after[same_x_mask]))
        return stats

    def _start_new_level(self) -> None:
        self.level_belief = torch.zeros_like(self.level_belief)
        self.level_epoch += 1
        self.level_step = 0

    def _last_top_scores(self, *, limit: int) -> list[dict[str, object]]:
        ranked = sorted(self.last_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [
            {
                "action": action,
                "score": float(score),
                "probability": float(self.last_probabilities.get(action, 0.0)),
                "value": float(self.last_values.get(action, 0.0)),
            }
            for action, score in ranked
        ]

    def _action_family(self, action: ActionName) -> str:
        if not action:
            return ""
        if self.last_state is None:
            return ""
        schema = build_action_schema(action, build_action_schema_context(self.last_state.affordances, dict(self.last_state.action_roles)))
        return schema.family


def _binary_entropy(probs: np.ndarray) -> np.ndarray:
    clipped = np.clip(probs, 1.0e-6, 1.0 - 1.0e-6)
    return -np.mean(clipped * np.log(clipped) + (1.0 - clipped) * np.log(1.0 - clipped), axis=-1)


def _standardize_valid(values: np.ndarray, valid: np.ndarray) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64).copy()
    active_mask = np.asarray(valid, dtype=bool)
    if not np.any(active_mask):
        return out
    active = out[active_mask]
    out[active_mask] = (active - float(np.mean(active))) / max(float(np.std(active)), 1.0e-6)
    out[~active_mask] = -1.0e9
    return out


def _external_transition_for_targets(transition: Transition) -> Transition:
    return Transition(
        state=_strip_internal_belief_state(transition.state),
        action=transition.action,
        reward=float(transition.reward),
        next_state=_strip_internal_belief_state(transition.next_state),
        terminated=bool(transition.terminated),
        info=dict(transition.info),
    )


def _strip_internal_belief_state(state: StructuredState) -> StructuredState:
    return StructuredState(
        task_id=state.task_id,
        episode_id=state.episode_id,
        step_index=state.step_index,
        grid_shape=state.grid_shape,
        grid_signature=state.grid_signature,
        objects=state.objects,
        relations=state.relations,
        affordances=state.affordances,
        action_roles=state.action_roles,
        inventory=tuple((key, value) for key, value in state.inventory if not str(key).startswith("belief_")),
        flags=tuple((key, value) for key, value in state.flags if not str(key).startswith("belief_")),
    )


def _entropy(probs: Sequence[float]) -> float:
    values = np.asarray(tuple(probs), dtype=np.float64)
    values = values[values > 0.0]
    if values.size == 0:
        return 0.0
    return float(-np.sum(values * np.log(values)))
