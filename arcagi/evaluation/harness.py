from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from statistics import mean
from typing import Any

from arcagi.agents.graph_agent import GraphExplorerAgent
from arcagi.agents.random_agent import RandomHeuristicAgent
from arcagi.core.progress_signals import action_family
from arcagi.learned_online.object_event_bridge import (
    SelectedActionObservation,
    assert_no_forbidden_metadata_as_model_input,
    build_object_event_observation,
)
from arcagi.perception.object_encoder import extract_structured_state
from arcagi.envs.arc_adapter import (
    Arcade,
    ArcToolkitEnv,
    arc_operation_mode,
    arc_toolkit_available,
    list_arc_games,
    require_dense_arc_action_surface,
)
from arcagi.envs.synthetic import DEFAULT_SYNTHETIC_FAMILY_MODES, HiddenRuleEnv, family_variants_for_mode


def build_agent(agent_name: str, checkpoint_path: str | None = None, device: Any = None):
    normalized = agent_name.strip().lower()
    if normalized == "random":
        return RandomHeuristicAgent()
    if normalized == "graph":
        return GraphExplorerAgent()
    if normalized in {"learned_online_minimal", "learned_online", "online_minimal"}:
        from arcagi.agents.learned_online_minimal_agent import LearnedOnlineMinimalAgent

        if checkpoint_path and Path(checkpoint_path).exists():
            return LearnedOnlineMinimalAgent.from_checkpoint(checkpoint_path)
        return LearnedOnlineMinimalAgent()
    if normalized in {"learned_online_recurrent", "learned_online_recurrent_v1", "online_recurrent"}:
        from arcagi.agents.learned_online_recurrent_agent import LearnedOnlineRecurrentAgent

        if checkpoint_path and Path(checkpoint_path).exists():
            return LearnedOnlineRecurrentAgent.from_checkpoint(checkpoint_path)
        return LearnedOnlineRecurrentAgent()
    if normalized in {"learned_online_object_event", "learned_online_object_event_v1", "online_object_event", "object_event"}:
        from arcagi.agents.learned_online_object_event_agent import LearnedOnlineObjectEventAgent

        if checkpoint_path and Path(checkpoint_path).exists():
            return LearnedOnlineObjectEventAgent.from_checkpoint(checkpoint_path)
        return LearnedOnlineObjectEventAgent(device=device)
    if normalized in {"scientist", "hyper_scientist", "hyper-generalizing-scientist", "spotlight", "spotlight_scientist"}:
        from arcagi.agents.scientist_agent import (
            HyperGeneralizingScientistAgent,
            load_spotlight_scientist_checkpoint,
        )

        if checkpoint_path and Path(checkpoint_path).exists():
            return load_spotlight_scientist_checkpoint(checkpoint_path)
        return HyperGeneralizingScientistAgent()

    import torch

    from arcagi.agents.learned_agent import HybridAgent, LanguageNoMemoryAgent, OnlineLanguageMemoryAgent, RecurrentAblationAgent
    from arcagi.memory.episodic import EpisodicMemory
    from arcagi.planning.planner import HybridPlanner, PlannerConfig
    from arcagi.training.synthetic import build_default_modules, load_checkpoint

    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if checkpoint_path and Path(checkpoint_path).exists():
        encoder, world_model, language_model = load_checkpoint(checkpoint_path, device=device)
    else:
        encoder, world_model, language_model, _planner = build_default_modules(device=device)

    planner = HybridPlanner(
        PlannerConfig(
            search_depth=2,
            search_root_width=2,
            search_branch_width=1,
            max_world_model_calls=48,
            graph_signal_weight=0.0,
            policy_prior_weight=3.0,
        )
        if device.type == "cpu"
        else PlannerConfig(graph_signal_weight=0.0, policy_prior_weight=3.0)
    )
    if normalized == "recurrent":
        return RecurrentAblationAgent(encoder=encoder, world_model=world_model, planner=planner, device=device)
    if normalized == "language":
        return LanguageNoMemoryAgent(
            encoder=encoder,
            world_model=world_model,
            planner=planner,
            language_model=language_model,
            device=device,
        )
    if normalized in {"online_language_memory", "language_memory", "memory_language"}:
        return OnlineLanguageMemoryAgent(
            encoder=encoder,
            world_model=world_model,
            planner=planner,
            language_model=language_model,
            episodic_memory=EpisodicMemory(),
            device=device,
        )
    if normalized == "hybrid":
        return HybridAgent(
            encoder=encoder,
            world_model=world_model,
            planner=planner,
            language_model=language_model,
            episodic_memory=EpisodicMemory(),
            device=device,
        )
    raise ValueError(f"unknown agent: {agent_name}")


def _reset_agent_for_episode(agent: Any) -> None:
    reset_episode = getattr(agent, "reset_episode", None)
    if callable(reset_episode):
        reset_episode()


def _reset_agent_for_level(agent: Any) -> None:
    reset_level = getattr(agent, "reset_level", None)
    if callable(reset_level):
        reset_level()
        return
    _reset_agent_for_episode(agent)


def _agent_handles_level_boundaries(agent: Any) -> bool:
    return bool(getattr(agent, "handles_level_boundaries", False))


def _reset_agent_for_family(agent: Any) -> None:
    reset_all = getattr(agent, "reset_all", None)
    if callable(reset_all):
        reset_all()
        return
    _reset_agent_for_episode(agent)


def _observe_step(agent: Any, *, action: str, before: Any, result: Any) -> None:
    update_after_step = getattr(agent, "update_after_step", None)
    if callable(update_after_step):
        update_after_step(
            next_observation=result.observation,
            reward=result.reward,
            terminated=result.terminated or result.truncated,
            info=result.info,
        )
        return
    observe_result = getattr(agent, "observe_result", None)
    if callable(observe_result):
        observe_result(
            action=action,
            before_observation=before,
            after_observation=result.observation,
            reward=result.reward,
            terminated=result.terminated or result.truncated,
            info=result.info,
        )


def _spotlight_feature_schema_version(agent: Any) -> int:
    diagnostics_fn = getattr(agent, "diagnostics", None)
    if not callable(diagnostics_fn):
        return 0
    try:
        diagnostics = diagnostics_fn()
    except Exception:
        return 0
    if not isinstance(diagnostics, dict):
        return 0
    spotlight = diagnostics.get("spotlight")
    if not isinstance(spotlight, dict):
        return 0
    try:
        return int(spotlight.get("feature_schema_version", 0) or 0)
    except Exception:
        return 0


def _agent_claim_metadata(agent: Any) -> dict[str, Any]:
    diagnostics_fn = getattr(agent, "diagnostics", None)
    diagnostics: dict[str, Any] = {}
    if callable(diagnostics_fn):
        try:
            raw = diagnostics_fn()
            if isinstance(raw, dict):
                diagnostics = raw
        except Exception:
            diagnostics = {}
    controller_kind = getattr(agent, "controller_kind", diagnostics.get("controller_kind", type(agent).__name__))
    claim_eligible = getattr(agent, "claim_eligible_arc_controller", diagnostics.get("claim_eligible", False))
    learned_online = getattr(agent, "learned_online_controller", diagnostics.get("learned_online_controller", False))
    arc_competence_validated = getattr(
        agent,
        "arc_competence_validated",
        diagnostics.get("arc_competence_validated", False),
    )
    role = getattr(agent, "role", diagnostics.get("role", ""))
    scored_action_count = diagnostics.get("scored_action_count", diagnostics.get("last_scored_action_count", 0))
    legal_action_count = diagnostics.get("legal_action_count", diagnostics.get("last_legal_action_count", 0))
    return {
        "controller_kind": str(controller_kind),
        "claim_eligible": bool(claim_eligible),
        "learned_online_controller": bool(learned_online),
        "arc_competence_validated": bool(arc_competence_validated),
        "controller_role": str(role),
        "scored_action_count": int(scored_action_count or 0),
        "legal_action_count": int(legal_action_count or 0),
    }


def run_episode(
    agent: Any,
    env: Any,
    seed: int,
    max_steps: int | None = None,
    stop_on_positive_reward: bool = False,
    progress_every: int = 0,
    trace_path: str | Path | None = None,
    object_event_bridge_diagnostics: bool = False,
) -> dict[str, object]:
    observation = env.reset(seed=seed)
    _reset_agent_for_episode(agent)
    steps = 0
    success = False
    won = False
    rewards: list[float] = []
    interaction_steps = 0
    reset_steps = 0
    levels_completed = _levels_completed(observation)
    language_traces: list[str] = []
    done = False
    spotlight_feature_schema_version = _spotlight_feature_schema_version(agent)
    claim_metadata = _agent_claim_metadata(agent)
    action_counts: Counter[str] = Counter()
    family_counts: Counter[str] = Counter()
    max_same_action_streak = 0
    same_action_streak = 0
    previous_action = ""
    last_bridge_diagnostics: dict[str, Any] = {}
    trace_handle = None
    try:
        if trace_path is not None:
            trace_file = Path(trace_path)
            trace_file.parent.mkdir(parents=True, exist_ok=True)
            trace_handle = trace_file.open("w", encoding="utf-8")
            _write_trace_row(
                trace_handle,
                {
                    "event": "episode_start",
                    "family_id": getattr(env, "family_id", getattr(env, "task_id", "unknown")),
                    "seed": int(seed),
                    "max_steps": max_steps,
                    "agent": getattr(agent, "name", type(agent).__name__),
                    "initial_game_state": _game_state(observation),
                    "initial_levels_completed": levels_completed,
                    "initial_available_action_count": len(tuple(getattr(observation, "available_actions", ()) or ())),
                    "spotlight_feature_schema_version": spotlight_feature_schema_version,
                    **claim_metadata,
                },
            )
        while not done and (max_steps is None or steps < max_steps):
            before = observation
            before_levels = _levels_completed(before)
            before_game_state = _game_state(before)
            before_actions = tuple(str(item) for item in getattr(before, "available_actions", ()) or ())
            action = agent.act(observation)
            action_text = str(action)
            action_kind = action_family(action_text)
            action_counts[action_text] += 1
            family_counts[action_kind] += 1
            if action_text == previous_action:
                same_action_streak += 1
            else:
                same_action_streak = 1
                previous_action = action_text
            max_same_action_streak = max(max_same_action_streak, same_action_streak)
            if _is_interaction_action(action_text, before):
                interaction_steps += 1
            reset_action = _is_reset_action(action_text, before)
            if reset_action:
                reset_steps += 1
            result = env.step(action)
            bridge_diagnostics = (
                _object_event_bridge_diagnostics(
                    before=before,
                    action=action_text,
                    result=result,
                )
                if object_event_bridge_diagnostics
                else {}
            )
            if bridge_diagnostics:
                last_bridge_diagnostics = bridge_diagnostics
            _observe_step(agent, action=action_text, before=before, result=result)
            observation = result.observation
            after_levels = _levels_completed(observation)
            after_game_state = _game_state(observation)
            after_actions = tuple(str(item) for item in getattr(observation, "available_actions", ()) or ())
            steps += 1
            rewards.append(float(result.reward))
            levels_completed = max(levels_completed, after_levels)
            won = won or _is_win_state(observation)
            level_boundary = bool(reset_action or after_levels > before_levels)
            if level_boundary and not won and not _agent_handles_level_boundaries(agent):
                _reset_agent_for_level(agent)
            if result.reward > 0.9:
                success = True
                if stop_on_positive_reward:
                    done = True
            if won:
                success = True
                done = True
            if not done:
                done = bool(result.terminated or result.truncated) and not _should_continue_after_terminal(observation)
            latest_language = tuple(str(x) for x in getattr(agent, "latest_language", ()) or ())
            if latest_language:
                language_traces.append(" ".join(latest_language))
            if trace_handle is not None:
                _write_trace_row(
                    trace_handle,
                    {
                        "event": "step",
                        "step": steps,
                        "family_id": getattr(env, "family_id", getattr(env, "task_id", "unknown")),
                        "action": action_text,
                        "action_family": action_kind,
                        "action_repeat_count": action_counts[action_text],
                        "same_action_streak": same_action_streak,
                        "family_count": family_counts[action_kind],
                        "before_available_action_count": len(before_actions),
                        "after_available_action_count": len(after_actions),
                        "before_levels_completed": before_levels,
                        "after_levels_completed": after_levels,
                        "level_delta": max(after_levels - before_levels, 0),
                        "level_boundary": level_boundary,
                        "reset_action": reset_action,
                        "interaction_action": _is_interaction_action(action_text, before),
                        "reward": float(result.reward),
                        "return": float(sum(rewards)),
                        "terminated": bool(result.terminated),
                        "truncated": bool(result.truncated),
                        "continued_after_terminal": bool(
                            (result.terminated or result.truncated) and _should_continue_after_terminal(observation)
                        ),
                        "done": bool(done),
                        "won": bool(won),
                        "before_game_state": before_game_state,
                        "after_game_state": after_game_state,
                        "latest_language": latest_language,
                        "last_plan_scores": _json_safe(getattr(agent, "last_plan_scores", {})),
                        "object_event_bridge": bridge_diagnostics,
                        "diagnostics": _compact_diagnostics(getattr(agent, "diagnostics", lambda: {})()),
                    },
                )
            if progress_every > 0 and steps % progress_every == 0:
                print(
                    json.dumps(
                        {
                            "event": "arc_runtime_progress",
                            "family_id": getattr(env, "family_id", getattr(env, "task_id", "unknown")),
                            "steps": steps,
                            "return": float(sum(rewards)),
                            "interaction_steps": interaction_steps,
                            "reset_steps": reset_steps,
                            "levels_completed": levels_completed,
                            "game_state": _game_state(observation),
                            "last_action": action_text,
                            "same_action_streak": same_action_streak,
                            "action_repeat_count": action_counts[action_text],
                            "family_counts": dict(family_counts),
                            "spotlight_feature_schema_version": spotlight_feature_schema_version,
                            "object_event_bridge": bridge_diagnostics,
                            **_agent_claim_metadata(agent),
                        },
                        sort_keys=True,
                    )
                )
    finally:
        if trace_handle is not None:
            _write_trace_row(
                trace_handle,
                {
                    "event": "episode_end",
                    "steps": steps,
                    "success": success,
                    "won": won,
                    "return": float(sum(rewards)),
                    "interaction_steps": interaction_steps,
                    "reset_steps": reset_steps,
                    "levels_completed": levels_completed,
                    "game_state": _game_state(observation),
                    "action_histogram": dict(action_counts),
                    "family_histogram": dict(family_counts),
                    "max_same_action_streak": max_same_action_streak,
                    "object_event_bridge": last_bridge_diagnostics,
                    "diagnostics": _compact_diagnostics(getattr(agent, "diagnostics", lambda: {})()),
                    **_agent_claim_metadata(agent),
                },
            )
            trace_handle.close()
    return {
        "success": success,
        "won": won,
        "return": float(sum(rewards)),
        "steps": steps,
        "interaction_steps": interaction_steps,
        "reset_steps": reset_steps,
        "levels_completed": levels_completed,
        "game_state": _game_state(observation),
        "language": language_traces[-1] if language_traces else "",
        "family_id": getattr(env, "family_id", getattr(env, "task_id", "unknown")),
        "spotlight_feature_schema_version": spotlight_feature_schema_version,
        **_agent_claim_metadata(agent),
        "action_histogram": dict(action_counts),
        "family_histogram": dict(family_counts),
        "max_same_action_streak": max_same_action_streak,
        "trace_path": str(trace_path) if trace_path is not None else "",
        "object_event_bridge": last_bridge_diagnostics,
        "diagnostics": getattr(agent, "diagnostics", lambda: {})(),
    }


def evaluate_synthetic(
    agent_name: str,
    checkpoint_path: str | None = None,
    episodes_per_family: int = 3,
    seed: int = 17,
    max_steps: int | None = None,
) -> dict[str, object]:
    agent = build_agent(agent_name, checkpoint_path=checkpoint_path)
    families: list[dict[str, object]] = []
    seed_cursor = seed
    for family_mode in DEFAULT_SYNTHETIC_FAMILY_MODES:
        for variant in family_variants_for_mode(family_mode):
            _reset_agent_for_family(agent)
            episode_metrics = []
            for episode_idx in range(episodes_per_family):
                env = HiddenRuleEnv(
                    family_mode=family_mode,
                    family_variant=variant,
                    seed=seed_cursor + episode_idx,
                    max_steps=max_steps,
                )
                episode_metrics.append(run_episode(agent, env, seed=seed_cursor + episode_idx, max_steps=max_steps))
            seed_cursor += episodes_per_family
            families.append(
                {
                    "family_mode": family_mode,
                    "family_variant": variant,
                    "episodes": episode_metrics,
                }
            )
    all_episodes = [episode for family in families for episode in family["episodes"]]
    first_episode_success = [family["episodes"][0]["success"] for family in families]
    later_episode_success = [episode["success"] for family in families for episode in family["episodes"][1:]]
    return {
        "agent": agent_name,
        "success_rate": mean(float(item["success"]) for item in all_episodes) if all_episodes else 0.0,
        "avg_return": mean(float(item["return"]) for item in all_episodes) if all_episodes else 0.0,
        "avg_steps": mean(float(item["steps"]) for item in all_episodes) if all_episodes else 0.0,
        "avg_interactions": mean(float(item["interaction_steps"]) for item in all_episodes) if all_episodes else 0.0,
        "first_episode_success": mean(float(value) for value in first_episode_success) if first_episode_success else 0.0,
        "later_episode_success": mean(float(value) for value in later_episode_success) if later_episode_success else 0.0,
        "families": families,
    }


def evaluate_arc(
    agent_name: str,
    checkpoint_path: str | None = None,
    game_limit: int = 3,
    mode: str = "offline",
    game_id: str | None = None,
    max_steps: int | None = 256,
    progress_every: int = 0,
    trace_path: str | None = None,
    allow_sparse_click_smoke: bool = False,
    object_event_bridge_diagnostics: bool = False,
) -> dict[str, object]:
    if not arc_toolkit_available():
        return {"agent": agent_name, "skipped": True, "reason": "ARC toolkit not installed"}
    action_surface = require_dense_arc_action_surface(
        context="ARC evaluation",
        allow_sparse_click_smoke=allow_sparse_click_smoke,
    )
    operation_mode = arc_operation_mode(mode)
    if game_id:
        games = [game_id]
    else:
        games = list_arc_games(operation_mode=operation_mode)[:game_limit]
    agent = build_agent(agent_name, checkpoint_path=checkpoint_path)
    results = []
    shared_arcade = None if Arcade is None else Arcade(operation_mode=operation_mode)
    for index, game_id in enumerate(games):
        _reset_agent_for_family(agent)
        env = ArcToolkitEnv(game_id, operation_mode=operation_mode, arcade=shared_arcade)
        try:
            episode = run_episode(
                agent,
                env,
                seed=index,
                max_steps=max_steps,
                stop_on_positive_reward=False,
                progress_every=progress_every,
                trace_path=_trace_path_for_game(trace_path, game_id, index, len(games)) if trace_path else None,
                object_event_bridge_diagnostics=bool(object_event_bridge_diagnostics),
            )
        finally:
            env.close()
        results.append({"game_id": game_id, **episode})
    if shared_arcade is not None:
        close_scorecard = getattr(shared_arcade, "close_scorecard", None)
        if callable(close_scorecard):
            try:
                close_scorecard()
            except Exception:
                pass
    return {"agent": agent_name, "mode": mode, **action_surface, "games": results}


def _trace_path_for_game(base_path: str, game_id: str, index: int, game_count: int) -> Path:
    base = Path(base_path)
    if game_count == 1:
        return base
    safe_game = "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(game_id))
    if base.suffix:
        return base.with_name(f"{base.stem}.{index:02d}.{safe_game}{base.suffix}")
    return base / f"{index:02d}_{safe_game}.jsonl"


def _object_event_bridge_diagnostics(*, before: Any, action: str, result: Any) -> dict[str, Any]:
    try:
        before_state = extract_structured_state(before)
        after_state = extract_structured_state(result.observation)
        legal_actions = tuple(str(item) for item in getattr(before, "available_actions", ()) or ())
        event = build_object_event_observation(
            SelectedActionObservation(
                before=before_state,
                selected_action=str(action),
                after=after_state,
                reward=float(result.reward),
                terminated=bool(result.terminated or result.truncated),
                legal_actions=legal_actions,
                info=dict(getattr(result, "info", {}) or {}),
            ),
            metadata={
                "object_event_bridge_enabled": True,
                "diagnostic_only": True,
            },
        )
        metadata_forbidden_count = 0
        try:
            assert_no_forbidden_metadata_as_model_input(event.model_input_metadata())
        except AssertionError:
            metadata_forbidden_count = 1
        return {
            "object_event_bridge_enabled": True,
            "object_event_legal_action_count": int(event.legal_action_count),
            "object_event_scored_action_count": int(event.legal_action_count),
            "object_event_selected_action_index": int(event.selected_action_index),
            "object_event_metadata_input_forbidden_count": int(metadata_forbidden_count),
            "object_event_action_surface_capped": False,
            "object_event_oracle_support_used": False,
            "object_event_trace_replay_used": False,
            "object_event_graph_controller_used": False,
        }
    except Exception as exc:
        return {
            "object_event_bridge_enabled": False,
            "object_event_bridge_error": repr(exc),
        }


def _write_trace_row(handle: Any, row: dict[str, Any]) -> None:
    handle.write(json.dumps(_json_safe(row), sort_keys=True) + "\n")
    handle.flush()


def _compact_diagnostics(diagnostics: Any) -> dict[str, Any]:
    if not isinstance(diagnostics, dict):
        return {}
    keep_keys = {
        "controller_kind",
        "claim_eligible",
        "learned_online_controller",
        "arc_competence_validated",
        "role",
        "scored_action_count",
        "legal_action_count",
        "last_scored_action_count",
        "last_legal_action_count",
        "full_dense_surface_scored",
        "object_event_bridge_enabled",
        "object_event_legal_action_count",
        "object_event_scored_action_count",
        "object_event_selected_action_index",
        "object_event_online_update_count",
        "object_event_session_belief_norm",
        "object_event_level_belief_norm",
        "object_event_metadata_input_forbidden_count",
        "object_event_action_surface_capped",
        "object_event_oracle_support_used",
        "object_event_trace_replay_used",
        "object_event_graph_controller_used",
        "runtime_trace_cursor",
        "runtime_action_sequence_replay",
        "runtime_state_hash_to_action",
        "runtime_per_game_behavior",
        "runtime_graph_search_solver",
        "runtime_action_pattern_enumerator",
        "runtime_external_api_or_knowledge",
        "runtime_rank_logits_used",
        "runtime_rank_score_mean",
        "runtime_rank_score_std",
        "runtime_diagnostic_utility_mean",
        "runtime_greedy_rank_selection",
        "runtime_controller_active",
        "objective_stall_steps",
        "level_stall_steps",
        "max_levels_completed_observed",
        "last_plan_scores",
        "latest_language",
        "last_top_scores",
        "latest_self_model_scores",
        "self_belief",
        "self_model",
        "spotlight",
        "world_model",
        "hypothesis_count",
        "memory_items",
        "transitions_observed",
        "score_entropy",
        "score_margin_top2",
        "all_negative_scores",
        "top_score",
        "mean_score",
        "mean_pred_cost",
        "mean_q_progress",
        "mean_q_info",
        "mean_q_imitation",
        "level_epoch",
        "level_step",
        "online_adapt_updates",
        "pretrain_updates",
        "selection_mode",
        "selected_action_probability",
        "selected_family_probability",
        "family_probabilities",
        "effective_action_support",
        "effective_family_support",
        "family_temperature",
        "action_temperature",
        "action_feature_config",
    }
    compact: dict[str, Any] = {}
    for key in keep_keys:
        if key in diagnostics:
            compact[key] = diagnostics[key]
    return _json_safe(compact)


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set, frozenset)):
        return [_json_safe(item) for item in value]
    try:
        import numpy as np

        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, np.ndarray):
            return value.tolist()
    except Exception:
        pass
    try:
        import torch

        if isinstance(value, torch.Tensor):
            return value.detach().cpu().tolist()
    except Exception:
        pass
    return repr(value)


def _observation_extras(observation: Any) -> dict[str, Any]:
    extras = getattr(observation, "extras", None)
    return dict(extras) if isinstance(extras, dict) else {}


def _levels_completed(observation: Any) -> int:
    extras = _observation_extras(observation)
    try:
        return int(extras.get("levels_completed", getattr(observation, "levels_completed", 0)) or 0)
    except Exception:
        return 0


def _game_state(observation: Any) -> str:
    extras = _observation_extras(observation)
    return str(extras.get("game_state", getattr(observation, "state", "")) or "")


def _is_win_state(observation: Any) -> bool:
    return _game_state(observation).strip().upper().endswith("WIN")


def _is_reset_action(action: str, observation: Any) -> bool:
    extras = _observation_extras(observation)
    roles = extras.get("action_roles", {})
    role = ""
    if isinstance(roles, dict):
        role = str(roles.get(action, ""))
    return str(action) == "0" or "reset" in role


def _is_interaction_action(action: str, observation: Any) -> bool:
    extras = _observation_extras(observation)
    roles = extras.get("action_roles", {})
    role = ""
    if isinstance(roles, dict):
        role = str(roles.get(action, ""))
    action_text = str(action)
    if action_text.startswith("interact") or action_text.startswith("click:"):
        return True
    role_lower = role.lower()
    return any(token in role_lower for token in ("interact", "click", "select", "use", "confirm"))


def _should_continue_after_terminal(observation: Any) -> bool:
    available_actions = tuple(str(item) for item in getattr(observation, "available_actions", ()) or ())
    game_state = _game_state(observation).strip().upper()
    if "0" not in available_actions:
        return False
    return game_state.endswith("GAME_OVER") or game_state.endswith("SESSION_ENDED")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m arcagi.evaluation.harness")
    subparsers = parser.add_subparsers(dest="command", required=True)
    synthetic_parser = subparsers.add_parser("synthetic")
    synthetic_parser.add_argument("--agent", type=str, default="language")
    synthetic_parser.add_argument("--checkpoint-path", type=str, default="")
    synthetic_parser.add_argument("--episodes-per-family", type=int, default=3)
    synthetic_parser.add_argument("--seed", type=int, default=17)
    synthetic_parser.add_argument("--max-steps", type=int, default=0)
    arc_parser = subparsers.add_parser("arc")
    arc_parser.add_argument("--agent", type=str, default="language")
    arc_parser.add_argument("--checkpoint-path", type=str, default="")
    arc_parser.add_argument("--game-limit", type=int, default=3)
    arc_parser.add_argument("--game-id", type=str, default="")
    arc_parser.add_argument("--max-steps", type=int, default=256)
    arc_parser.add_argument("--progress-every", type=int, default=0)
    arc_parser.add_argument("--trace-path", type=str, default="")
    arc_parser.add_argument("--mode", type=str, default="offline")
    arc_parser.add_argument(
        "--allow-sparse-click-smoke",
        action="store_true",
        help="allow ARCAGI_SPARSE_CLICKS_BASELINE=1 for explicitly invalid smoke/debug runs",
    )
    arc_parser.add_argument(
        "--object-event-bridge-diagnostics",
        action="store_true",
        help="emit passive selected-action object-event bridge diagnostics without changing control",
    )
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    if args.command == "synthetic":
        result = evaluate_synthetic(
            agent_name=args.agent,
            checkpoint_path=args.checkpoint_path or None,
            episodes_per_family=args.episodes_per_family,
            seed=args.seed,
            max_steps=None if int(args.max_steps) <= 0 else int(args.max_steps),
        )
        print(json.dumps(result, indent=2))
        return 0
    if args.command == "arc":
        result = evaluate_arc(
            agent_name=args.agent,
            checkpoint_path=args.checkpoint_path or None,
            game_limit=args.game_limit,
            game_id=args.game_id or None,
            max_steps=None if int(args.max_steps) <= 0 else int(args.max_steps),
            progress_every=max(0, int(args.progress_every)),
            mode=args.mode,
            trace_path=args.trace_path or None,
            allow_sparse_click_smoke=bool(args.allow_sparse_click_smoke),
            object_event_bridge_diagnostics=bool(args.object_event_bridge_diagnostics),
        )
        print(json.dumps(result, indent=2))
        return 0
    parser.error(f"unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
