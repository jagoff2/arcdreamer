"""Hyper-generalizing online scientist agent.

The agent loop is:

1. perceive the current grid as objects and relations;
2. convert the last action/outcome into a transition record;
3. update hypotheses, episodic memory, and the online world model;
4. choose the next action by trading off expected reward, novelty, and
   information gain;
5. emit grounded internal language for traceability and memory indexing.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from .hypotheses import HypothesisEngine
from .language import GroundedLanguage
from .memory import EpisodicMemory
from .perception import compare_states, extract_state
from .planner import PlannerConfig, ScientistPlanner
from .types import ActionDecision, ActionName, GridFrame, StructuredState, coerce_grid_frame, combined_progress_signal
from .world_model import OnlineWorldModel


@dataclass(frozen=True)
class ScientistAgentConfig:
    memory_capacity: int = 2048
    max_hypotheses: int = 512
    planner: PlannerConfig = field(default_factory=PlannerConfig)
    world_learning_rate: float = 0.08
    seed: int = 0
    keep_world_weights_between_episodes: bool = True


class ScientistAgent:
    def __init__(self, config: ScientistAgentConfig | None = None) -> None:
        self.config = config or ScientistAgentConfig()
        self.engine = HypothesisEngine(max_hypotheses=self.config.max_hypotheses)
        self.memory = EpisodicMemory(capacity=self.config.memory_capacity)
        self.language = GroundedLanguage()
        self.world_model = OnlineWorldModel(learning_rate=self.config.world_learning_rate, seed=self.config.seed)
        self.planner = ScientistPlanner(config=self.config.planner, seed=self.config.seed)
        self.current_state: StructuredState | None = None
        self.last_decision: ActionDecision | None = None
        self.latest_language: tuple[str, ...] = ()
        self._last_raw_observation: Any | None = None
        self.transitions_observed = 0
        self.total_reward = 0.0
        self.trace: list[dict[str, Any]] = []

    def reset_episode(self) -> None:
        self.engine.reset_episode()
        self.memory.reset_episode()
        self.world_model.reset_episode(keep_weights=self.config.keep_world_weights_between_episodes)
        self.planner.reset_episode()
        self.current_state = None
        self.last_decision = None
        self.latest_language = ()
        self._last_raw_observation = None
        self.transitions_observed = 0
        self.total_reward = 0.0
        self.trace.clear()

    def perceive(self, observation: Any) -> StructuredState:
        return extract_state(coerce_grid_frame(observation))

    def act(self, observation: Any) -> ActionName:
        state = self.perceive(observation)
        self.current_state = state
        decision = self.planner.choose_action(
            state,
            engine=self.engine,
            world_model=self.world_model,
            memory=self.memory,
            language=self.language,
        )
        self.last_decision = decision
        self.latest_language = tuple(decision.language)
        self._last_raw_observation = observation
        self.trace.append(
            {
                "step": state.step_index,
                "action": decision.action,
                "score": decision.score,
                "components": dict(decision.components),
                "language": list(decision.language),
                "candidate_count": decision.candidate_count,
            }
        )
        return decision.action

    def observe_result(
        self,
        *,
        action: ActionName,
        before_observation: Any,
        after_observation: Any,
        reward: float = 0.0,
        terminated: bool = False,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        before = extract_state(coerce_grid_frame(before_observation))
        after = extract_state(coerce_grid_frame(after_observation))
        score_delta = _score_delta(before.frame, after.frame, info or {})
        record = compare_states(
            before,
            after,
            action=action,
            reward=reward,
            score_delta=score_delta,
            terminated=terminated,
            info=info or {},
        )
        self.engine.observe_transition(record)
        loss = self.world_model.update(record)
        prediction = self.world_model.predict(record.before, action)
        progress = combined_progress_signal(reward, score_delta)
        surprise = abs(float(progress) - prediction.reward_mean) + record.delta.changed_fraction + 0.05 * loss
        tokens = self.language.memory_tokens(after, self.engine)
        self.memory.write_transition(record, surprise=surprise, language_tokens=tokens)
        self.planner.notify_transition(
            changed=record.delta.has_visible_effect or abs(progress) > 1e-8,
            record=record,
            engine=self.engine,
        )
        self.current_state = after
        self.transitions_observed += 1
        self.total_reward += float(progress)

        if self.trace:
            self.trace[-1]["observed_reward"] = float(reward)
            self.trace[-1]["score_delta"] = float(score_delta)
            self.trace[-1]["changed_cells"] = record.delta.changed_cells
            self.trace[-1]["hypothesis_count"] = len(self.engine.hypotheses)
            self.trace[-1]["memory_count"] = len(self.memory.items)


    def update_after_step(
        self,
        *,
        next_observation: Any,
        reward: float = 0.0,
        terminated: bool = False,
        info: Mapping[str, Any] | None = None,
    ) -> None:
        """Compatibility hook for the existing ``arcagi.evaluation.harness`` loop."""

        if self.last_decision is None or self._last_raw_observation is None:
            return
        self.observe_result(
            action=self.last_decision.action,
            before_observation=self._last_raw_observation,
            after_observation=next_observation,
            reward=reward,
            terminated=terminated,
            info=info or {},
        )

    def reset_all(self) -> None:
        """Reset all online state for a fresh benchmark family/game."""

        self.reset_episode()

    def diagnostics(self) -> dict[str, Any]:
        return {
            "transitions_observed": self.transitions_observed,
            "total_reward": self.total_reward,
            "hypothesis_count": len(self.engine.hypotheses),
            "credible_hypotheses": [h.description for h in self.engine.credible_hypotheses(limit=8)],
            "questions": list(self.language.questions(self.engine, limit=8)),
            "memory_items": len(self.memory.items),
            "option_items": len(self.memory.options),
            "last_decision": None
            if self.last_decision is None
            else {
                "action": self.last_decision.action,
                "score": self.last_decision.score,
                "components": dict(self.last_decision.components),
                "language": list(self.last_decision.language),
            },
        }


def _score_delta(before: GridFrame, after: GridFrame, info: Mapping[str, Any]) -> float:
    if "score_delta" in info:
        try:
            return float(info["score_delta"])
        except Exception:
            return 0.0
    before_score = before.extras.get("score")
    after_score = after.extras.get("score")
    try:
        if before_score is not None and after_score is not None:
            return float(after_score) - float(before_score)
    except Exception:
        return 0.0
    return 0.0
