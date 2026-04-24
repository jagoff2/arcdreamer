from __future__ import annotations

from pathlib import Path
import pickle

import numpy as np

from arcagi.agents.base import BaseAgent
from arcagi.core.types import ActionName, GridObservation, StructuredState, Transition
from arcagi.learned_online.fast_belief import OnlineBeliefState
from arcagi.learned_online.memory import OnlineEpisodicMemory
from arcagi.learned_online.action_features import ActionFeatureConfig
from arcagi.learned_online.questions import QuestionToken, question_tokens, select_question
from arcagi.learned_online.recurrent_model import RecurrentOnlineModel
from arcagi.learned_online.recurrent_policy import RECURRENT_CANDIDATE_INPUT_DIM, RecurrentOnlinePolicy, RecurrentPolicyDecision
from arcagi.learned_online.sequence import EVENT_FEATURE_DIM, event_features
from arcagi.learned_online.signals import labels_from_transition


class LearnedOnlineRecurrentAgent(BaseAgent):
    controller_kind = "learned_online_recurrent_v1"
    claim_eligible_arc_controller = True
    arc_competence_validated = False
    learned_online_controller = True
    handles_level_boundaries = True
    role = "recurrent_learned_online_candidate"

    def __init__(self, *, seed: int = 0, chunk_size: int = 4096, hidden_dim: int = 64) -> None:
        super().__init__(name="learned_online_recurrent")
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.hidden_dim = int(hidden_dim)
        self.belief = OnlineBeliefState()
        self.memory = OnlineEpisodicMemory()
        self.model = RecurrentOnlineModel(
            candidate_input_dim=RECURRENT_CANDIDATE_INPUT_DIM,
            event_input_dim=EVENT_FEATURE_DIM,
            hidden_dim=self.hidden_dim,
            seed=self.seed,
        )
        self.policy = RecurrentOnlinePolicy(model=self.model, belief=self.belief, memory=self.memory, seed=self.seed)
        self.current_question = QuestionToken.TEST_ACTION_MEANING
        self.last_decision: RecurrentPolicyDecision | None = None
        self.last_question: QuestionToken = self.current_question
        self.last_loss: float = 0.0
        self.last_realized_info_gain: float = 0.0
        self.last_prediction_error: float = 0.0
        self.credit_gamma: float = 0.985
        self.credit_horizon: int | None = 512

    def reset_episode(self) -> None:
        super().reset_episode()
        self.belief.reset()
        self.memory.reset()
        self.model.reset_episode(keep_weights=True)
        self.policy.last_scored_action_count = 0
        self.policy.last_legal_action_count = 0
        self.policy.last_scores.clear()
        self.policy.last_components.clear()
        self.policy.last_policy_diagnostics.clear()
        self.current_question = QuestionToken.TEST_ACTION_MEANING
        self.last_decision = None
        self.last_question = self.current_question
        self.last_loss = 0.0
        self.last_realized_info_gain = 0.0
        self.last_prediction_error = 0.0

    def reset_level(self) -> None:
        BaseAgent.reset_level(self)

    def act(self, observation: GridObservation) -> ActionName:
        state = self.observe(observation)
        self.current_question = select_question(self.belief)
        decision = self.policy.choose_action(
            state,
            state.affordances,
            question=self.current_question,
            chunk_size=self.chunk_size,
        )
        self.last_state = state
        self.last_action = decision.action
        self.last_question = self.current_question
        self.last_decision = decision
        self.latest_language = question_tokens(self.current_question)
        return decision.action

    def on_transition(self, transition: Transition) -> None:
        labels = labels_from_transition(transition)
        pre_feature = self.policy.feature_for_action(transition.state, transition.action, question=self.last_question)
        hidden_before = self.model.hidden.copy()
        probe_batch = self.memory.sample_probe_batch(k=8, exclude_action=transition.action)
        pre_probe_loss = self.model.batch_prediction_loss(probe_batch) if probe_batch else None
        pre_loss = self.model.prediction_loss(
            pre_feature,
            labels,
            realized_info_gain=0.0,
            hidden=hidden_before,
        )
        update_loss = self.model.online_update(
            pre_feature,
            labels,
            realized_info_gain=0.0,
            hidden=hidden_before,
        )
        post_probe_loss = self.model.batch_prediction_loss(probe_batch) if probe_batch else None
        if pre_probe_loss is None or post_probe_loss is None:
            realized_info_gain = 0.0
        else:
            realized_info_gain = float(np.clip(pre_probe_loss - post_probe_loss, -1.0, 1.0))
        if realized_info_gain > 0.0:
            self.model.online_update(
                pre_feature,
                labels,
                realized_info_gain=realized_info_gain,
                hidden=hidden_before,
            )
        prediction_error = float(pre_loss)
        self.belief.recent_prediction_error = prediction_error
        level_epoch = int(self.belief.level_epoch)
        self.belief.observe(
            transition.state,
            transition.action,
            labels,
            realized_info_gain=realized_info_gain,
            prediction_error=prediction_error,
        )
        self.memory.write(
            state=transition.state,
            action=transition.action,
            question=self.last_question,
            labels=labels,
            realized_info_gain=realized_info_gain,
            feature=pre_feature,
            hidden=hidden_before,
            level_epoch=level_epoch,
            level_step=int(self.belief.level_step),
        )
        if labels.objective_progress > 0.0 or labels.terminal_progress > 0.0:
            self._credit_current_level_success(level_epoch)
        self.model.observe_event(
            event_features(
                transition.state,
                transition.action,
                labels,
                question=self.last_question,
                prediction_error=prediction_error,
                realized_info_gain=realized_info_gain,
            )
        )
        self.last_loss = float(update_loss)
        self.last_realized_info_gain = realized_info_gain
        self.last_prediction_error = prediction_error
        if labels.objective_progress > 0.0 or labels.terminal_progress > 0.0:
            self.belief.start_new_level()
            self.memory.start_new_level(self.belief.level_epoch)

    def score_actions_for_state(
        self,
        state: StructuredState,
        actions: tuple[ActionName, ...] | list[ActionName],
        *,
        question: QuestionToken | None = None,
    ) -> dict[ActionName, RecurrentPolicyDecision]:
        return self.policy.score_actions(
            state,
            tuple(actions),
            question=question or select_question(self.belief),
            chunk_size=self.chunk_size,
        )

    def diagnostics(self) -> dict[str, object]:
        return {
            "controller_kind": self.controller_kind,
            "claim_eligible": bool(self.claim_eligible_arc_controller),
            "arc_competence_validated": bool(self.arc_competence_validated),
            "learned_online_controller": True,
            "role": self.role,
            "last_legal_action_count": int(self.policy.last_legal_action_count),
            "last_scored_action_count": int(self.policy.last_scored_action_count),
            "legal_action_count": int(self.policy.last_legal_action_count),
            "scored_action_count": int(self.policy.last_scored_action_count),
            "online_updates": int(self.belief.online_update_count),
            "model_updates": int(self.model.updates),
            "pretrain_updates": int(self.model.pretrain_updates),
            "online_adapt_updates": int(self.model.online_adapt_updates),
            "last_loss": float(self.last_loss),
            "last_realized_info_gain": float(self.last_realized_info_gain),
            "last_prediction_error": float(self.last_prediction_error),
            "current_question": self.current_question.name,
            "memory_items": len(self.memory.entries),
            "hidden_norm": float(np.linalg.norm(self.model.hidden)),
            "level_epoch": int(self.belief.level_epoch),
            "level_step": int(self.belief.level_step),
            "belief": self.belief.summary(),
            **self.policy.last_policy_diagnostics,
            "family_probabilities": dict(self.policy.last_family_probabilities),
            "action_feature_config": self.policy.action_feature_config.to_dict(),
            "last_top_scores": self._last_top_scores(limit=12),
        }

    def _credit_current_level_success(self, level_epoch: int) -> None:
        credited = self.memory.credit_recent_success(
            level_epoch=level_epoch,
            gamma=self.credit_gamma,
            max_entries=self.credit_horizon,
        )
        for entry in credited:
            if entry.feature is None:
                continue
            self.model.online_update(
                entry.feature,
                entry.labels,
                realized_info_gain=float(entry.realized_info_gain),
                hidden=entry.hidden,
                return_credit=float(entry.return_credit),
            )

    def _last_top_scores(self, *, limit: int) -> list[dict[str, object]]:
        ranked = sorted(self.policy.last_scores.items(), key=lambda item: item[1], reverse=True)[:limit]
        return [
            {
                "action": action,
                "score": float(score),
                "components": dict(self.policy.last_components.get(action, {})),
            }
            for action, score in ranked
        ]

    def state_dict(self) -> dict[str, object]:
        return {
            "seed": int(self.seed),
            "chunk_size": int(self.chunk_size),
            "hidden_dim": int(self.hidden_dim),
            "model": self.model.state_dict(),
            "selection_mode": self.policy.selection_mode,
            "family_temperature": float(self.policy.family_temperature),
            "action_temperature": float(self.policy.action_temperature),
            "action_feature_config": self.policy.action_feature_config.to_dict(),
        }

    def load_state_dict(self, state: dict[str, object]) -> None:
        self.seed = int(state.get("seed", self.seed) or self.seed)
        self.chunk_size = int(state.get("chunk_size", self.chunk_size) or self.chunk_size)
        model_state = state.get("model", {})
        if isinstance(model_state, dict):
            self.model.load_state_dict(model_state)
        self.policy.selection_mode = str(state.get("selection_mode", self.policy.selection_mode) or self.policy.selection_mode)
        self.policy.family_temperature = float(state.get("family_temperature", self.policy.family_temperature) or self.policy.family_temperature)
        self.policy.action_temperature = float(state.get("action_temperature", self.policy.action_temperature) or self.policy.action_temperature)
        feature_config = state.get("action_feature_config", {})
        if isinstance(feature_config, dict):
            self.policy.action_feature_config = ActionFeatureConfig(
                include_exact_action_projection=bool(feature_config.get("include_exact_action_projection", False)),
                include_role_projection=bool(feature_config.get("include_role_projection", True)),
                include_family_projection=bool(feature_config.get("include_family_projection", True)),
            )

    def save_checkpoint(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with target.open("wb") as handle:
            pickle.dump(self.state_dict(), handle)

    @classmethod
    def from_checkpoint(cls, path: str | Path, *, seed: int = 0) -> "LearnedOnlineRecurrentAgent":
        agent = cls(seed=seed)
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        if not isinstance(state, dict):
            raise ValueError(f"learned online recurrent checkpoint at {path!s} is not a mapping")
        agent.load_state_dict(state)
        return agent
