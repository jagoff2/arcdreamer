from __future__ import annotations

from typing import Any

import numpy as np

from arcagi.agents.base import BaseAgent
from arcagi.core.types import ActionName, GridObservation, StructuredState, Transition
from arcagi.learned_online.fast_belief import OnlineBeliefState
from arcagi.learned_online.memory import OnlineEpisodicMemory
from arcagi.learned_online.minimal_model import MinimalOnlineModel
from arcagi.learned_online.policy import LEARNED_ONLINE_INPUT_DIM, LearnedOnlinePolicy, PolicyDecision
from arcagi.learned_online.questions import QuestionToken, question_tokens, select_question
from arcagi.learned_online.signals import labels_from_transition


class LearnedOnlineMinimalAgent(BaseAgent):
    controller_kind = "learned_online_minimal"
    claim_eligible_arc_controller = True
    learned_online_controller = True
    handles_level_boundaries = True

    def __init__(self, *, seed: int = 0, chunk_size: int = 4096) -> None:
        super().__init__(name="learned_online_minimal")
        self.seed = int(seed)
        self.chunk_size = int(chunk_size)
        self.belief = OnlineBeliefState()
        self.memory = OnlineEpisodicMemory()
        self.model = MinimalOnlineModel(input_dim=LEARNED_ONLINE_INPUT_DIM)
        self.policy = LearnedOnlinePolicy(model=self.model, belief=self.belief, memory=self.memory, seed=self.seed)
        self.current_question = QuestionToken.TEST_ACTION_MEANING
        self.last_decision: PolicyDecision | None = None
        self.last_question: QuestionToken = self.current_question
        self.last_loss: float = 0.0
        self.last_realized_info_gain: float = 0.0
        self.last_prediction_error: float = 0.0

    def reset_episode(self) -> None:
        super().reset_episode()
        self.belief.reset()
        self.memory.reset()
        self.model = MinimalOnlineModel(input_dim=LEARNED_ONLINE_INPUT_DIM)
        self.policy = LearnedOnlinePolicy(model=self.model, belief=self.belief, memory=self.memory, seed=self.seed)
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
        self.belief.observe(
            transition.state,
            transition.action,
            labels,
            realized_info_gain=0.0,
            prediction_error=0.0,
        )
        feature = self.policy.feature_for_action(transition.state, transition.action, question=self.last_question)
        pre_loss = self.model.prediction_loss(feature, labels, realized_info_gain=0.0)
        update_loss = self.model.online_update(feature, labels, realized_info_gain=0.0)
        post_loss = self.model.prediction_loss(feature, labels, realized_info_gain=0.0)
        realized_info_gain = float(np.clip(pre_loss - post_loss, -1.0, 1.0))
        if realized_info_gain > 0.0:
            self.model.online_update(feature, labels, realized_info_gain=realized_info_gain)
        prediction_error = float(pre_loss)
        self.belief.recent_prediction_error = prediction_error
        self.memory.write(
            state=transition.state,
            action=transition.action,
            question=self.last_question,
            labels=labels,
            realized_info_gain=realized_info_gain,
        )
        self.last_loss = float(update_loss)
        self.last_realized_info_gain = realized_info_gain
        self.last_prediction_error = prediction_error

    def score_actions_for_state(
        self,
        state: StructuredState,
        actions: tuple[ActionName, ...] | list[ActionName],
        *,
        question: QuestionToken | None = None,
    ) -> dict[ActionName, PolicyDecision]:
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
            "learned_online_controller": True,
            "last_legal_action_count": int(self.policy.last_legal_action_count),
            "last_scored_action_count": int(self.policy.last_scored_action_count),
            "legal_action_count": int(self.policy.last_legal_action_count),
            "scored_action_count": int(self.policy.last_scored_action_count),
            "online_updates": int(self.belief.online_update_count),
            "model_updates": int(self.model.updates),
            "last_loss": float(self.last_loss),
            "last_realized_info_gain": float(self.last_realized_info_gain),
            "last_prediction_error": float(self.last_prediction_error),
            "current_question": self.current_question.name,
            "memory_items": len(self.memory.entries),
            "belief": self.belief.summary(),
            "last_scores": dict(self.policy.last_scores),
            "last_components": dict(self.policy.last_components),
        }
