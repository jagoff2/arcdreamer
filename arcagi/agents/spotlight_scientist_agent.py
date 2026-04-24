"""Scientist agent variant with an action-level spotlight workspace."""

from __future__ import annotations

import pickle
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

from arcagi.scientist.agent import (
    ScientistAgent,
    ScientistAgentConfig,
    _coerce_planner_config,
    _score_delta,
    normalize_scientist_agent_config,
)
from arcagi.scientist.planner import PlannerConfig
from arcagi.scientist.perception import compare_states, extract_state
from arcagi.scientist.spotlight import ActionSpotlight, LEGACY_FEATURE_SCHEMA_VERSION, SpotlightConfig
from arcagi.scientist.types import (
    ActionName,
    coerce_grid_frame,
    combined_progress_signal,
    is_failure_terminal_game_state,
    is_reset_action,
    observation_game_state,
    observation_levels_completed,
)


@dataclass(frozen=True)
class SpotlightScientistConfig(ScientistAgentConfig):
    """Configuration for the action-spotlight scientist."""

    spotlight: SpotlightConfig = field(default_factory=SpotlightConfig)


def normalize_spotlight_scientist_config(
    config: SpotlightScientistConfig | ScientistAgentConfig | Mapping[str, Any] | None,
) -> SpotlightScientistConfig:
    if (
        isinstance(config, SpotlightScientistConfig)
        and isinstance(config.planner, PlannerConfig)
        and isinstance(config.spotlight, SpotlightConfig)
    ):
        return config
    if isinstance(config, Mapping):
        base = normalize_scientist_agent_config(config)
        spotlight_cfg = config.get("spotlight")
        if isinstance(spotlight_cfg, Mapping):
            try:
                spotlight = SpotlightConfig(**dict(spotlight_cfg))
            except TypeError:
                spotlight = SpotlightConfig()
        elif isinstance(spotlight_cfg, SpotlightConfig):
            spotlight = spotlight_cfg
        else:
            spotlight = SpotlightConfig()
        return SpotlightScientistConfig(
            memory_capacity=base.memory_capacity,
            max_hypotheses=base.max_hypotheses,
            planner=base.planner,
            world_learning_rate=base.world_learning_rate,
            seed=base.seed,
            keep_world_weights_between_episodes=base.keep_world_weights_between_episodes,
            spotlight=spotlight,
        )
    if config is None:
        return SpotlightScientistConfig()
    base = normalize_scientist_agent_config(config)
    spotlight_cfg = getattr(config, "spotlight", SpotlightConfig())
    if isinstance(spotlight_cfg, Mapping):
        try:
            spotlight = SpotlightConfig(**dict(spotlight_cfg))
        except TypeError:
            spotlight = SpotlightConfig()
    elif isinstance(spotlight_cfg, SpotlightConfig):
        spotlight = spotlight_cfg
    else:
        spotlight = SpotlightConfig()
    return SpotlightScientistConfig(
        memory_capacity=base.memory_capacity,
        max_hypotheses=base.max_hypotheses,
        planner=_coerce_planner_config(base.planner),
        world_learning_rate=base.world_learning_rate,
        seed=base.seed,
        keep_world_weights_between_episodes=base.keep_world_weights_between_episodes,
        spotlight=spotlight,
    )


class SpotlightScientistAgent(ScientistAgent):
    """Online scientist agent with serial action commitments and delayed binder credit."""

    def __init__(self, config: SpotlightScientistConfig | None = None, **_: Any) -> None:
        normalized = normalize_spotlight_scientist_config(config)
        super().__init__(config=normalized)
        spotlight_cfg = getattr(self.config, "spotlight", SpotlightConfig())
        self.spotlight = ActionSpotlight(config=spotlight_cfg)

    def reset_episode(self) -> None:
        super().reset_episode()
        self.spotlight.reset_episode()

    def reset_level(self) -> None:
        super().reset_level()
        self.spotlight.reset_level()

    def observe_teacher_action(self, teacher_action: ActionName, *, weight: float = 1.0) -> None:
        self.spotlight.observe_teacher_action(teacher_action, weight=weight)

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

        self.engine.observe_transition(record)
        loss = self.world_model.update(record)
        prediction = self.world_model.predict(record.before, action)
        progress = combined_progress_signal(reward, score_delta)
        surprise = abs(float(progress) - prediction.reward_mean) + record.delta.changed_fraction + 0.05 * loss
        tokens = self.language.memory_tokens(after, self.engine)
        spotlight_feedback = self.spotlight.notify_transition(record=record, engine=self.engine)
        option_mode = "write"
        if isinstance(spotlight_feedback, Mapping):
            if bool(spotlight_feedback.get("binding_action")):
                option_mode = "skip"
            elif bool(spotlight_feedback.get("probe_supported")):
                option_mode = "write"
            elif bool(spotlight_feedback.get("probe_failed")):
                option_mode = "contradict"
            elif bool(spotlight_feedback.get("pending_binder_before")):
                option_mode = "skip"
        self.memory.write_transition(record, surprise=surprise, language_tokens=tokens, option_mode=option_mode)

        self.planner.notify_transition(
            changed=record.delta.has_visible_effect or abs(progress) > 1e-8,
            record=record,
            engine=self.engine,
        )
        level_success = observation_levels_completed(after) > observation_levels_completed(before)
        terminal_failure = bool(terminated and is_failure_terminal_game_state(observation_game_state(after)))
        level_boundary = is_reset_action(action) or level_success
        self.spotlight.learn_from_transition(
            record=record,
            planner=self.planner,
            engine=self.engine,
            world_model=self.world_model,
            memory=self.memory,
            language=self.language,
        )
        if level_boundary or terminal_failure:
            self.spotlight.finalize_attempt(
                after,
                success=bool(level_success),
                terminal_failure=bool(terminal_failure),
                ended_by_reset=bool(is_reset_action(action)),
            )
        if level_boundary:
            self.reset_level()

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
        return {
            "config": asdict(self.config),
            "world_model": self.world_model.state_dict(),
            "spotlight": self.spotlight.state_dict(),
            "checkpoint_metadata": dict(self.checkpoint_metadata),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        spotlight_state = state.get("spotlight")
        if isinstance(spotlight_state, Mapping):
            self.spotlight.load_state_dict(spotlight_state)
        else:
            spotlight_cfg = getattr(self.config, "spotlight", SpotlightConfig())
            self.spotlight = ActionSpotlight(config=spotlight_cfg)
            self.spotlight.feature_schema_version = LEGACY_FEATURE_SCHEMA_VERSION

    def save_checkpoint(self, path: str | Path, *, metadata: Mapping[str, Any] | None = None) -> None:
        if metadata is not None:
            self.checkpoint_metadata = dict(metadata)
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
        with Path(path).open("rb") as handle:
            state = pickle.load(handle)
        if not isinstance(state, Mapping):
            raise ValueError(f"spotlight scientist checkpoint at {path!s} is not a mapping")
        if config is None:
            config_state = state.get("config")
            if isinstance(config_state, Mapping):
                config = normalize_spotlight_scientist_config(config_state)
        agent = cls(config=config)
        agent.load_state_dict(state)
        return agent


def save_spotlight_scientist_checkpoint(
    agent: SpotlightScientistAgent,
    path: str | Path,
    *,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    agent.save_checkpoint(path, metadata=metadata)


def load_spotlight_scientist_checkpoint(
    path: str | Path,
    *,
    config: SpotlightScientistConfig | None = None,
) -> SpotlightScientistAgent:
    return SpotlightScientistAgent.from_checkpoint(path, config=config)


def make_agent(config: SpotlightScientistConfig | None = None) -> SpotlightScientistAgent:
    return SpotlightScientistAgent(config=config)
