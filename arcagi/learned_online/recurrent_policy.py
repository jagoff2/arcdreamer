from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from arcagi.core.types import ActionName, StructuredState
from arcagi.learned_online.action_features import ACTION_FEATURE_DIM, encode_action_candidates
from arcagi.learned_online.fast_belief import BELIEF_FEATURE_DIM, OnlineBeliefState
from arcagi.learned_online.memory import MEMORY_FEATURE_DIM, OnlineEpisodicMemory
from arcagi.learned_online.questions import QUESTION_FEATURE_DIM, QuestionToken, question_features
from arcagi.learned_online.recurrent_model import RecurrentOnlineModel

RECURRENT_CANDIDATE_INPUT_DIM = ACTION_FEATURE_DIM + BELIEF_FEATURE_DIM + MEMORY_FEATURE_DIM + QUESTION_FEATURE_DIM


@dataclass(frozen=True)
class RecurrentPolicyDecision:
    action: ActionName
    score: float
    question: QuestionToken
    legal_action_count: int
    scored_action_count: int
    components: dict[str, float]


class RecurrentOnlinePolicy:
    def __init__(
        self,
        *,
        model: RecurrentOnlineModel,
        belief: OnlineBeliefState,
        memory: OnlineEpisodicMemory,
        beta_info: float = 0.35,
        seed: int = 0,
    ) -> None:
        self.model = model
        self.belief = belief
        self.memory = memory
        self.beta_info = float(beta_info)
        self.rng = np.random.default_rng(seed)
        self.last_scored_action_count = 0
        self.last_legal_action_count = 0
        self.last_scores: dict[ActionName, float] = {}
        self.last_components: dict[ActionName, dict[str, float]] = {}

    def build_feature_matrix(
        self,
        state: StructuredState,
        actions: Sequence[ActionName],
        *,
        question: QuestionToken,
    ) -> np.ndarray:
        batch = encode_action_candidates(state, actions)
        q_features = question_features(question)
        rows: list[np.ndarray] = []
        for action, action_features in zip(batch.actions, batch.features):
            rows.append(
                np.concatenate(
                    [
                        action_features,
                        self.belief.features_for(state, action),
                        self.memory.features_for(state, action),
                        q_features,
                    ]
                ).astype(np.float32)
            )
        return np.asarray(rows, dtype=np.float32)

    def score_actions(
        self,
        state: StructuredState,
        legal_actions: Sequence[ActionName],
        *,
        question: QuestionToken,
        chunk_size: int = 4096,
    ) -> dict[ActionName, RecurrentPolicyDecision]:
        actions = tuple(str(action) for action in legal_actions)
        if not actions:
            raise RuntimeError("RecurrentOnlinePolicy received no legal actions")
        results: dict[ActionName, RecurrentPolicyDecision] = {}
        total_scored = 0
        for start in range(0, len(actions), max(int(chunk_size), 1)):
            chunk = actions[start : start + max(int(chunk_size), 1)]
            features = self.build_feature_matrix(state, chunk, question=question)
            pred = self.model.predict(features)
            total_scored += len(chunk)
            for index, action in enumerate(chunk):
                q_progress = float(pred.reward[index]) + float(pred.useful[index])
                q_info = float(pred.info_gain[index])
                learned_cost = float(pred.cost[index])
                score = q_progress + (self.beta_info * q_info) - learned_cost
                components = {
                    "q_progress": float(q_progress),
                    "q_info": float(q_info),
                    "pred_reward": float(pred.reward[index]),
                    "pred_useful": float(pred.useful[index]),
                    "pred_visible": float(pred.visible[index]),
                    "pred_info_gain": float(pred.info_gain[index]),
                    "pred_uncertainty": float(pred.uncertainty[index]),
                    "learned_cost": float(learned_cost),
                }
                results[action] = RecurrentPolicyDecision(
                    action=action,
                    score=float(score),
                    question=question,
                    legal_action_count=len(actions),
                    scored_action_count=total_scored,
                    components=components,
                )
        if total_scored != len(actions):
            raise RuntimeError(f"scored {total_scored} actions but legal surface has {len(actions)}")
        self.last_scored_action_count = total_scored
        self.last_legal_action_count = len(actions)
        self.last_scores = {action: decision.score for action, decision in results.items()}
        self.last_components = {action: dict(decision.components) for action, decision in results.items()}
        return results

    def choose_action(
        self,
        state: StructuredState,
        legal_actions: Sequence[ActionName],
        *,
        question: QuestionToken,
        chunk_size: int = 4096,
    ) -> RecurrentPolicyDecision:
        decisions = self.score_actions(state, legal_actions, question=question, chunk_size=chunk_size)
        best_score = max(decision.score for decision in decisions.values())
        tied = [decision for decision in decisions.values() if abs(decision.score - best_score) <= 1e-8]
        if len(tied) == 1:
            return tied[0]
        return tied[int(self.rng.integers(len(tied)))]

    def feature_for_action(
        self,
        state: StructuredState,
        action: ActionName,
        *,
        question: QuestionToken,
    ) -> np.ndarray:
        return self.build_feature_matrix(state, (str(action),), question=question)[0]
