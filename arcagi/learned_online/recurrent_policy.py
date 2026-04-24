from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.types import ActionName, StructuredState
from arcagi.learned_online.action_features import (
    ACTION_FEATURE_DIM,
    DEFAULT_ACTION_FEATURE_CONFIG,
    ActionFeatureConfig,
    encode_action_candidates,
)
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
        selection_mode: str = "factorized_softmax",
        family_temperature: float = 0.10,
        action_temperature: float = 0.07,
        action_feature_config: ActionFeatureConfig = DEFAULT_ACTION_FEATURE_CONFIG,
    ) -> None:
        self.model = model
        self.belief = belief
        self.memory = memory
        self.beta_info = float(beta_info)
        self.rng = np.random.default_rng(seed)
        self.selection_mode = str(selection_mode)
        self.family_temperature = float(family_temperature)
        self.action_temperature = float(action_temperature)
        self.action_feature_config = action_feature_config
        self.last_scored_action_count = 0
        self.last_legal_action_count = 0
        self.last_scores: dict[ActionName, float] = {}
        self.last_components: dict[ActionName, dict[str, float]] = {}
        self.last_policy_diagnostics: dict[str, float | bool] = {}
        self.last_action_probabilities: dict[ActionName, float] = {}
        self.last_family_probabilities: dict[str, float] = {}

    def build_feature_matrix(
        self,
        state: StructuredState,
        actions: Sequence[ActionName],
        *,
        question: QuestionToken,
    ) -> np.ndarray:
        batch = encode_action_candidates(state, actions, config=self.action_feature_config)
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
                q_progress = float(pred.value[index]) + float(pred.reward[index]) + (0.5 * float(pred.useful[index]))
                q_info = float(pred.info_gain[index])
                learned_cost = float(pred.cost[index])
                score = q_progress + (self.beta_info * q_info) - learned_cost
                components = {
                    "q_progress": float(q_progress),
                    "q_info": float(q_info),
                    "pred_reward": float(pred.reward[index]),
                    "pred_useful": float(pred.useful[index]),
                    "pred_value": float(pred.value[index]),
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
        self.last_policy_diagnostics = _policy_diagnostics(self.last_scores, self.last_components)
        return results

    def factorized_action_probabilities(
        self,
        state: StructuredState,
        actions: Sequence[ActionName],
        scores_by_action: dict[ActionName, float],
    ) -> tuple[dict[ActionName, float], dict[str, float]]:
        action_tuple = tuple(str(action) for action in actions)
        scores = np.asarray([float(scores_by_action[action]) for action in action_tuple], dtype=np.float64)
        groups: dict[str, list[int]] = {}
        for index, action in enumerate(action_tuple):
            family = _policy_family_key(state, action)
            groups.setdefault(family, []).append(index)
        families = tuple(groups.keys())
        family_logits = np.asarray(
            [
                _logmeanexp(scores[groups[family]], temperature=self.action_temperature)
                for family in families
            ],
            dtype=np.float64,
        )
        p_family = _softmax(family_logits, temperature=self.family_temperature)
        probabilities = np.zeros((len(action_tuple),), dtype=np.float64)
        for family_index, family in enumerate(families):
            indices = groups[family]
            local_scores = scores[indices]
            p_local = _softmax(local_scores, temperature=self.action_temperature)
            for local_offset, action_index in enumerate(indices):
                probabilities[action_index] = float(p_family[family_index]) * float(p_local[local_offset])
        total = max(float(np.sum(probabilities)), 1e-12)
        probabilities /= total
        return (
            {action_tuple[index]: float(probabilities[index]) for index in range(len(action_tuple))},
            {families[index]: float(p_family[index]) for index in range(len(families))},
        )

    def choose_action(
        self,
        state: StructuredState,
        legal_actions: Sequence[ActionName],
        *,
        question: QuestionToken,
        chunk_size: int = 4096,
    ) -> RecurrentPolicyDecision:
        actions = tuple(str(action) for action in legal_actions)
        decisions = self.score_actions(state, actions, question=question, chunk_size=chunk_size)
        if self.selection_mode == "factorized_softmax":
            action_probs, family_probs = self.factorized_action_probabilities(
                state,
                actions,
                {action: decision.score for action, decision in decisions.items()},
            )
            probs = np.asarray([action_probs[action] for action in actions], dtype=np.float64)
            probs = probs / max(float(np.sum(probs)), 1e-12)
            choice_index = int(self.rng.choice(len(actions), p=probs))
            action = actions[choice_index]
            self.last_action_probabilities = action_probs
            self.last_family_probabilities = family_probs
            family = _policy_family_key(state, action)
            self.last_policy_diagnostics.update(
                {
                    "selection_mode": self.selection_mode,
                    "selected_action_probability": float(action_probs.get(action, 0.0)),
                    "selected_family_probability": float(family_probs.get(family, 0.0)),
                    "effective_action_support": _effective_support(probs),
                    "effective_family_support": _effective_support(np.asarray(list(family_probs.values()), dtype=np.float64)),
                    "family_temperature": float(self.family_temperature),
                    "action_temperature": float(self.action_temperature),
                }
            )
            return decisions[action]
        if self.selection_mode != "greedy":
            raise ValueError(f"unknown recurrent selection_mode={self.selection_mode!r}")
        best_score = max(decision.score for decision in decisions.values())
        tied = [decision for decision in decisions.values() if abs(decision.score - best_score) <= 1e-8]
        self.last_action_probabilities = {}
        self.last_family_probabilities = {}
        self.last_policy_diagnostics.update({"selection_mode": self.selection_mode})
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


def _policy_diagnostics(
    scores_by_action: dict[ActionName, float],
    components_by_action: dict[ActionName, dict[str, float]],
) -> dict[str, float | bool]:
    if not scores_by_action:
        return {}
    scores = np.asarray(list(scores_by_action.values()), dtype=np.float64)
    centered = scores - float(np.max(scores))
    exp_scores = np.exp(np.clip(centered, -60.0, 0.0))
    probs = exp_scores / max(float(np.sum(exp_scores)), 1e-12)
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    ranked = np.sort(scores)[::-1]
    margin = float(ranked[0] - ranked[1]) if len(ranked) > 1 else float("inf")
    components = list(components_by_action.values())
    return {
        "score_entropy": entropy,
        "score_margin_top2": margin,
        "all_negative_scores": bool(np.all(scores < 0.0)),
        "top_score": float(ranked[0]),
        "mean_score": float(np.mean(scores)),
        "mean_pred_cost": _component_mean(components, "learned_cost"),
        "mean_q_progress": _component_mean(components, "q_progress"),
        "mean_q_info": _component_mean(components, "q_info"),
    }


def _component_mean(components: list[dict[str, float]], key: str) -> float:
    if not components:
        return 0.0
    return float(np.mean([float(item.get(key, 0.0)) for item in components]))


def _policy_family_key(state: StructuredState, action: ActionName) -> str:
    context = build_action_schema_context(state.affordances, dict(state.action_roles))
    schema = build_action_schema(str(action), context)
    return f"{schema.action_type}:{schema.direction or 'none'}"


def _softmax(logits: np.ndarray, *, temperature: float) -> np.ndarray:
    temp = max(float(temperature), 1e-6)
    values = np.asarray(logits, dtype=np.float64) / temp
    values = values - float(np.max(values))
    exp_values = np.exp(np.clip(values, -60.0, 60.0))
    return exp_values / max(float(np.sum(exp_values)), 1e-12)


def _logmeanexp(values: np.ndarray, *, temperature: float) -> float:
    temp = max(float(temperature), 1e-6)
    scaled = np.asarray(values, dtype=np.float64) / temp
    maximum = float(np.max(scaled))
    return float(temp * (maximum + np.log(np.mean(np.exp(np.clip(scaled - maximum, -60.0, 60.0))))))


def _effective_support(probabilities: np.ndarray) -> float:
    probs = np.asarray(probabilities, dtype=np.float64)
    probs = probs / max(float(np.sum(probs)), 1e-12)
    entropy = float(-np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0))))
    return float(np.exp(entropy))
