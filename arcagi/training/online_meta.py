from __future__ import annotations

import argparse
import copy
import json
from dataclasses import asdict, dataclass, replace
from pathlib import Path
from statistics import mean
from typing import Any

import torch

from arcagi.agents.learned_agent import OnlineLanguageMemoryAgent
from arcagi.envs.synthetic import DEFAULT_SYNTHETIC_FAMILY_MODES, HiddenRuleEnv, family_variants_for_mode
from arcagi.planning.planner import HybridPlanner, PlannerConfig
from arcagi.training.synthetic import build_default_modules, load_checkpoint, seed_everything


@dataclass(frozen=True)
class OnlineMetaConfig:
    checkpoint_path: str = "artifacts/online_meta_language.pt"
    init_checkpoint_path: str = ""
    iterations: int = 64
    support_episodes: int = 2
    query_episodes: int = 1
    max_steps: int = 650
    meta_step_size: float = 0.12
    acceptance_margin: float = 0.0
    support_exploration_epsilon: float = 0.12
    query_exploration_epsilon: float = 0.0
    support_success_update_scale: float = 0.35
    online_update_steps: int = 1
    seed: int = 17
    device: str = ""
    eval_every: int = 8


def _build_language_agent(
    encoder: torch.nn.Module,
    world_model: torch.nn.Module,
    language_model: torch.nn.Module,
    *,
    device: torch.device,
    online_update_steps: int = 1,
) -> OnlineLanguageMemoryAgent:
    planner = HybridPlanner(
        PlannerConfig(
            graph_signal_weight=0.0,
            policy_prior_weight=3.0,
            search_depth=2,
            search_root_width=2,
            search_branch_width=1,
            max_world_model_calls=48,
        )
    )
    agent = OnlineLanguageMemoryAgent(
        encoder=encoder,  # type: ignore[arg-type]
        world_model=world_model,  # type: ignore[arg-type]
        planner=planner,
        language_model=language_model,  # type: ignore[arg-type]
        device=device,
    )
    agent.config = replace(agent.config, online_world_model_update_steps=max(int(online_update_steps), 0))
    return agent


def _run_agent_episode(
    agent: OnlineLanguageMemoryAgent,
    *,
    family_mode: str,
    family_variant: str,
    seed: int,
    max_steps: int,
    preserve_online: bool,
    exploration_epsilon: float = 0.0,
) -> dict[str, Any]:
    env = HiddenRuleEnv(family_mode=family_mode, family_variant=family_variant, seed=seed, max_steps=max_steps)
    observation = env.reset(seed=seed)
    if preserve_online:
        agent.reset_level()
    else:
        agent.reset_all()
    original_config = getattr(agent, "config", None)
    if original_config is not None:
        agent.config = replace(original_config, exploration_epsilon=max(0.0, min(float(exploration_epsilon), 1.0)))
    steps = 0
    total_reward = 0.0
    interactions = 0
    success = False
    try:
        while steps < max_steps:
            action = str(agent.act(observation))
            if action.startswith("click:") or action.startswith("interact"):
                interactions += 1
            result = env.step(action)
            agent.update_after_step(
                next_observation=result.observation,
                reward=float(result.reward),
                terminated=bool(result.terminated or result.truncated),
                info=dict(result.info or {}),
            )
            observation = result.observation
            total_reward += float(result.reward)
            steps += 1
            if result.reward > 0.9:
                success = True
            if result.terminated or result.truncated:
                break
    finally:
        if original_config is not None:
            agent.config = original_config
    return {
        "success": bool(success),
        "return": float(total_reward),
        "steps": int(steps),
        "interactions": int(interactions),
    }


def _move_modules_toward(
    base_modules: tuple[torch.nn.Module, ...],
    adapted_modules: tuple[torch.nn.Module, ...],
    *,
    step_size: float,
) -> None:
    step = float(step_size)
    with torch.no_grad():
        for base, adapted in zip(base_modules, adapted_modules, strict=True):
            for base_param, adapted_param in zip(base.parameters(), adapted.parameters(), strict=True):
                base_param.add_(step * (adapted_param.detach() - base_param))


def _task_for_iteration(iteration: int) -> tuple[str, str]:
    families = tuple(DEFAULT_SYNTHETIC_FAMILY_MODES)
    family_mode = families[iteration % len(families)]
    variants = tuple(family_variants_for_mode(family_mode))
    family_variant = variants[(iteration // len(families)) % len(variants)]
    return str(family_mode), str(family_variant)


def train_online_meta(config: OnlineMetaConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = torch.device(config.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    if device.type == "cpu":
        raise RuntimeError("online meta-training requires CUDA because runtime gradient adaptation is disabled on CPU")
    if config.init_checkpoint_path and Path(config.init_checkpoint_path).exists():
        encoder, world_model, language_model = load_checkpoint(config.init_checkpoint_path, device=device)
    else:
        encoder, world_model, language_model, _planner = build_default_modules(device=device)
    encoder.eval()
    world_model.eval()
    language_model.eval()

    history: list[dict[str, Any]] = []
    for iteration in range(int(config.iterations)):
        family_mode, family_variant = _task_for_iteration(iteration)
        clone_encoder = copy.deepcopy(encoder).to(device).eval()
        clone_world_model = copy.deepcopy(world_model).to(device).eval()
        clone_language_model = copy.deepcopy(language_model).to(device).eval()
        agent = _build_language_agent(
            clone_encoder,
            clone_world_model,
            clone_language_model,
            device=device,
            online_update_steps=config.online_update_steps,
        )

        seed_base = int(config.seed + (iteration * 997))
        cold_query = _run_agent_episode(
            _build_language_agent(
                copy.deepcopy(encoder).to(device).eval(),
                copy.deepcopy(world_model).to(device).eval(),
                copy.deepcopy(language_model).to(device).eval(),
                device=device,
                online_update_steps=config.online_update_steps,
            ),
            family_mode=family_mode,
            family_variant=family_variant,
            seed=seed_base + 10_000,
            max_steps=config.max_steps,
            preserve_online=False,
            exploration_epsilon=0.0,
        )
        support_metrics = []
        for support_index in range(int(config.support_episodes)):
            support_metrics.append(
                _run_agent_episode(
                    agent,
                    family_mode=family_mode,
                    family_variant=family_variant,
                    seed=seed_base + support_index,
                    max_steps=config.max_steps,
                    preserve_online=support_index > 0,
                    exploration_epsilon=config.support_exploration_epsilon,
                )
            )
        query_metrics = []
        for query_index in range(int(config.query_episodes)):
            query_metrics.append(
                _run_agent_episode(
                    agent,
                    family_mode=family_mode,
                    family_variant=family_variant,
                    seed=seed_base + 20_000 + query_index,
                    max_steps=config.max_steps,
                    preserve_online=True,
                    exploration_epsilon=config.query_exploration_epsilon,
                )
            )

        query_success_rate = mean(float(item["success"]) for item in query_metrics) if query_metrics else 0.0
        query_return = mean(float(item["return"]) for item in query_metrics) if query_metrics else 0.0
        query_minus_cold_return = query_return - float(cold_query["return"]) if query_metrics else 0.0
        support_success_rate = mean(float(item["success"]) for item in support_metrics) if support_metrics else 0.0
        support_return = mean(float(item["return"]) for item in support_metrics) if support_metrics else 0.0
        support_minus_cold_return = support_return - float(cold_query["return"]) if support_metrics else 0.0
        query_update = bool(
            query_success_rate > float(cold_query["success"])
            or query_minus_cold_return >= float(config.acceptance_margin)
        )
        support_update = bool(
            support_success_rate > float(cold_query["success"])
            or support_minus_cold_return >= float(config.acceptance_margin)
        )
        meta_update_applied = bool(query_update or support_update)
        update_source = "none"
        applied_step_size = 0.0
        if meta_update_applied:
            step_scale = min(1.0 + max(query_minus_cold_return, 0.0), 2.0)
            if query_update:
                update_source = "query"
                applied_step_size = float(config.meta_step_size) * step_scale
            else:
                update_source = "support"
                support_scale = min(1.0 + max(support_minus_cold_return, 0.0) + support_success_rate, 2.0)
                applied_step_size = (
                    float(config.meta_step_size)
                    * max(0.0, float(config.support_success_update_scale))
                    * support_scale
                )
            _move_modules_toward((world_model,), (clone_world_model,), step_size=applied_step_size)
        row = {
            "iteration": int(iteration),
            "family_mode": family_mode,
            "family_variant": family_variant,
            "cold_success": float(cold_query["success"]),
            "cold_return": float(cold_query["return"]),
            "support_success_rate": support_success_rate,
            "support_return": support_return,
            "support_minus_cold_return": support_minus_cold_return,
            "query_success_rate": query_success_rate,
            "query_return": query_return,
            "query_minus_cold_return": query_minus_cold_return,
            "meta_update_applied": meta_update_applied,
            "update_source": update_source,
            "applied_step_size": applied_step_size,
        }
        history.append(row)
        if config.eval_every > 0 and (iteration == 0 or (iteration + 1) % int(config.eval_every) == 0):
            print(json.dumps({"event": "online_meta_progress", **row}, sort_keys=True), flush=True)

    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "encoder": encoder.state_dict(),
            "world_model": world_model.state_dict(),
            "language_model": language_model.state_dict(),
            "history": history,
            "training_state": {
                "mode": "visible_online_meta_reptile",
                "runtime_hidden_events": "stripped",
                "oracle_labels": False,
                "graph_signal_weight": 0.0,
                "updated_modules": ("world_model",),
                "acceptance_margin": float(config.acceptance_margin),
                "support_exploration_epsilon": float(config.support_exploration_epsilon),
                "query_exploration_epsilon": float(config.query_exploration_epsilon),
                "support_success_update_scale": float(config.support_success_update_scale),
                "online_update_steps": int(config.online_update_steps),
            },
        },
        checkpoint_path,
    )
    return {
        "checkpoint_path": str(checkpoint_path),
        "iterations": int(config.iterations),
        "query_success_rate_last": float(history[-1]["query_success_rate"]) if history else 0.0,
        "query_return_last": float(history[-1]["query_return"]) if history else 0.0,
        "query_minus_cold_return_last": float(history[-1]["query_minus_cold_return"]) if history else 0.0,
    }


def build_parser() -> argparse.ArgumentParser:
    defaults = OnlineMetaConfig()
    parser = argparse.ArgumentParser(prog="python -m arcagi.training.online_meta")
    parser.add_argument("--checkpoint-path", type=str, default=defaults.checkpoint_path)
    parser.add_argument("--init-checkpoint-path", type=str, default=defaults.init_checkpoint_path)
    parser.add_argument("--iterations", type=int, default=defaults.iterations)
    parser.add_argument("--support-episodes", type=int, default=defaults.support_episodes)
    parser.add_argument("--query-episodes", type=int, default=defaults.query_episodes)
    parser.add_argument("--max-steps", type=int, default=defaults.max_steps)
    parser.add_argument("--meta-step-size", type=float, default=defaults.meta_step_size)
    parser.add_argument("--acceptance-margin", type=float, default=defaults.acceptance_margin)
    parser.add_argument("--support-exploration-epsilon", type=float, default=defaults.support_exploration_epsilon)
    parser.add_argument("--query-exploration-epsilon", type=float, default=defaults.query_exploration_epsilon)
    parser.add_argument("--support-success-update-scale", type=float, default=defaults.support_success_update_scale)
    parser.add_argument("--online-update-steps", type=int, default=defaults.online_update_steps)
    parser.add_argument("--seed", type=int, default=defaults.seed)
    parser.add_argument("--device", type=str, default=defaults.device)
    parser.add_argument("--eval-every", type=int, default=defaults.eval_every)
    return parser


def main() -> int:
    args = build_parser().parse_args()
    result = train_online_meta(
        OnlineMetaConfig(
            checkpoint_path=args.checkpoint_path,
            init_checkpoint_path=args.init_checkpoint_path,
            iterations=args.iterations,
            support_episodes=args.support_episodes,
            query_episodes=args.query_episodes,
            max_steps=args.max_steps,
            meta_step_size=args.meta_step_size,
            acceptance_margin=args.acceptance_margin,
            support_exploration_epsilon=args.support_exploration_epsilon,
            query_exploration_epsilon=args.query_exploration_epsilon,
            support_success_update_scale=args.support_success_update_scale,
            online_update_steps=args.online_update_steps,
            seed=args.seed,
            device=args.device,
            eval_every=args.eval_every,
        )
    )
    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
