"""Scientist agent variant with an action-level spotlight workspace.

This file is intentionally separate from the base scientist agent so the patch
can be dropped into the current repository without destabilising the existing
implementation.  The public class aliases at the bottom let the existing
`--agent scientist` path use this agent after `arcagi/agents/scientist_agent.py`
is replaced by the compatibility wrapper included in this patch.
"""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from arcagi.scientist.agent import ScientistAgent, ScientistAgentConfig, _score_delta
from arcagi.scientist.perception import compare_states, extract_state
from arcagi.scientist.spotlight import ActionSpotlight, SpotlightConfig
from arcagi.scientist.types import ActionDecision, ActionName, combined_progress_signal, coerce_grid_frame


@dataclass(frozen=True)
class SpotlightScientistConfig(ScientistAgentConfig):
    """Configuration for the action-spotlight scientist.

    All base scientist settings are inherited.  The added spotlight block controls
    the serial workspace, latent binder grace period, and probe commitment.
    """

    spotlight: SpotlightConfig = field(default_factory=SpotlightConfig)


class SpotlightScientistAgent(ScientistAgent):
    """ARC-facing online scientist with a serial action-level workspace."""

    def __init__(self, config: SpotlightScientistConfig | None = None, **_: Any) -> None:
        super().__init__(config=config or SpotlightScientistConfig())
        # `self.config` is set by ScientistAgent.__init__.
        spotlight_cfg = getattr(self.config, "spotlight", SpotlightConfig())
        self.spotlight = ActionSpotlight(config=spotlight_cfg)

    def reset_episode(self) -> None:
        super().reset_episode()
        self.spotlight.reset_episode()

    def act(self, observation: Any) -> ActionName:
        state = self.perceive(observation)
        self.current_state = state
        decision = self.spotlight.choose_action(
            state,
            planner=self.planner,
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
                "spotlight": self.spotlight.diagnostics().get("last_broadcast"),
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

        # Slow and explicit learners update first so the spotlight can interpret
        # the same transition against the latest hypothesis set.
        self.engine.observe_transition(record)
        loss = self.world_model.update(record)
        prediction = self.world_model.predict(record.before, action)
        progress = combined_progress_signal(reward, score_delta)
        surprise = abs(float(progress) - prediction.reward_mean) + record.delta.changed_fraction + 0.05 * loss
        tokens = self.language.memory_tokens(after, self.engine)
        self.memory.write_transition(record, surprise=surprise, language_tokens=tokens)

        self.spotlight.notify_transition(record=record, engine=self.engine)
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
            self.trace[-1]["spotlight_after"] = self.spotlight.diagnostics()

    def diagnostics(self) -> dict[str, Any]:
        base = super().diagnostics()
        base["spotlight"] = self.spotlight.diagnostics()
        return base

    def state_dict(self) -> dict[str, Any]:
        state = {
            "config": asdict(self.config),
            "world_model": self.world_model.state_dict(),
            "spotlight": self.spotlight.state_dict(),
        }
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        spotlight_state = state.get("spotlight")
        if isinstance(spotlight_state, Mapping):
            self.spotlight.load_state_dict(spotlight_state)

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
        config: SpotlightScientistConfig | None = None,
    ) -> "SpotlightScientistAgent":
        agent = cls(config=config)
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        if not isinstance(state, Mapping):
            raise ValueError(f"spotlight scientist checkpoint at {path!s} is not a mapping")
        agent.load_state_dict(state)
        return agent


def save_spotlight_scientist_checkpoint(agent: SpotlightScientistAgent, path: str | Path) -> None:
    agent.save_checkpoint(path)


def load_spotlight_scientist_checkpoint(
    path: str | Path,
    *,
    config: SpotlightScientistConfig | None = None,
) -> SpotlightScientistAgent:
    return SpotlightScientistAgent.from_checkpoint(path, config=config)


def make_agent(config: SpotlightScientistConfig | None = None) -> SpotlightScientistAgent:
    return SpotlightScientistAgent(config=config)
