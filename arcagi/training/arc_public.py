from __future__ import annotations

import argparse
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean

import numpy as np
import torch

from arcagi.agents.graph_agent import GraphExplorerAgent
from arcagi.agents.random_agent import RandomHeuristicAgent
from arcagi.core.action_schema import build_action_schema, build_action_schema_context
from arcagi.core.types import StructuredState
from arcagi.core.utils import seed_everything
from arcagi.envs.arc_adapter import ArcToolkitEnv, arc_operation_mode, arc_toolkit_available, list_arc_games
from arcagi.models.encoder import StructuredStateEncoder
from arcagi.models.language import GroundedLanguageModel
from arcagi.models.world_model import RecurrentWorldModel
from arcagi.memory.episodic import EpisodicMemory
from arcagi.perception.object_encoder import extract_structured_state
from arcagi.planning.planner import HybridPlanner, PlannerConfig
from arcagi.training.arc_session import annotate_session_returns, collect_arc_session
from arcagi.training.synthetic import build_default_modules, load_checkpoint


@dataclass(frozen=True)
class ArcPublicTrainingConfig:
    mode: str = "online"
    game_limit: int = 25
    sessions_per_game: int = 2
    max_steps: int = 96
    epochs: int = 2
    learning_rate: float = 1e-4
    seed: int = 17
    device: str = ""
    checkpoint_path: str = "artifacts/arc_public_hybrid.pt"
    init_checkpoint_path: str = "artifacts/mixed_policy_hybrid.pt"
    behavior_policies: tuple[str, ...] = ("graph", "learned", "random")
    policy_positive_threshold: float = 0.15
    freeze_encoder: bool = True
    language_loss_weight: float = 0.2


_ALLOWED_ARC_PUBLIC_BEHAVIOR_POLICIES: frozenset[str] = frozenset({"graph", "random", "learned", "hybrid"})


def _validate_arc_public_training_config(config: ArcPublicTrainingConfig) -> None:
    if int(config.game_limit) <= 0:
        raise ValueError("game_limit must be positive")
    if int(config.sessions_per_game) <= 0:
        raise ValueError("sessions_per_game must be positive")
    if int(config.max_steps) <= 0:
        raise ValueError("max_steps must be positive")
    if int(config.epochs) <= 0:
        raise ValueError("epochs must be positive")
    if float(config.learning_rate) <= 0.0:
        raise ValueError("learning_rate must be positive")
    if not config.behavior_policies:
        raise ValueError("behavior_policies must not be empty")
    invalid = [policy for policy in config.behavior_policies if policy not in _ALLOWED_ARC_PUBLIC_BEHAVIOR_POLICIES]
    if invalid:
        raise ValueError(
            f"unsupported ARC public behavior policies: {invalid}. "
            f"expected a subset of {sorted(_ALLOWED_ARC_PUBLIC_BEHAVIOR_POLICIES)}"
        )


def _clone_modules(
    encoder: StructuredStateEncoder,
    world_model: RecurrentWorldModel,
    language_model: GroundedLanguageModel,
    *,
    device: torch.device,
) -> tuple[StructuredStateEncoder, RecurrentWorldModel, GroundedLanguageModel]:
    cloned_encoder, cloned_world_model, cloned_language_model, _ = build_default_modules(device=device)
    cloned_encoder.load_state_dict(encoder.state_dict())
    cloned_world_model.load_state_dict(world_model.state_dict())
    cloned_language_model.load_state_dict(language_model.state_dict())
    cloned_encoder.eval()
    cloned_world_model.eval()
    cloned_language_model.eval()
    return cloned_encoder, cloned_world_model, cloned_language_model


def _behavior_agent(
    name: str,
    *,
    encoder: StructuredStateEncoder | None = None,
    world_model: RecurrentWorldModel | None = None,
    language_model: GroundedLanguageModel | None = None,
    device: torch.device | None = None,
):
    if name == "graph":
        return GraphExplorerAgent(), True
    if name == "random":
        return RandomHeuristicAgent(), True
    if name in {"learned", "hybrid"}:
        from arcagi.agents.learned_agent import HybridAgent, LanguageNoMemoryAgent

        collector_device = device or torch.device("cpu")
        if encoder is None or world_model is None or language_model is None:
            collection_encoder, collection_world_model, collection_language_model, _ = build_default_modules(device=collector_device)
        else:
            collection_encoder, collection_world_model, collection_language_model = _clone_modules(
                encoder,
                world_model,
                language_model,
                device=collector_device,
            )
        planner = HybridPlanner(
            PlannerConfig(search_depth=2, search_root_width=2, search_branch_width=1, max_world_model_calls=48)
            if collector_device.type == "cpu"
            else PlannerConfig(search_depth=2, search_root_width=2, search_branch_width=1, max_world_model_calls=32)
        )
        if name == "learned":
            return (
                LanguageNoMemoryAgent(
                    encoder=collection_encoder,
                    world_model=collection_world_model,
                    planner=planner,
                    language_model=collection_language_model,
                    device=collector_device,
                ),
                True,
            )
        return (
            HybridAgent(
                encoder=collection_encoder,
                world_model=collection_world_model,
                planner=planner,
                language_model=collection_language_model,
                episodic_memory=EpisodicMemory(),
                device=collector_device,
            ),
            True,
        )
    raise ValueError(f"unsupported behavior policy: {name}")


def _make_sample(
    state: StructuredState,
    next_state: StructuredState,
    action: str,
    reward: float,
) -> dict[str, object]:
    delta = next_state.transition_vector() - state.transition_vector()
    delta_norm = float(np.linalg.norm(delta))
    usefulness = max(float(reward), 0.0) + (0.5 * delta_norm)
    return {
        "state": state,
        "next_state": next_state,
        "action": action,
        "reward": float(reward),
        "delta": delta.astype(np.float32),
        "usefulness": usefulness,
        "belief_tokens": _belief_tokens(state, action, reward, usefulness),
        "question_tokens": _question_tokens(state, action, reward, usefulness),
        "plan_tokens": _plan_tokens(state, action, reward, usefulness),
    }


def _belief_tokens(
    state: StructuredState,
    action: str,
    reward: float,
    usefulness: float,
) -> tuple[str, ...]:
    context = build_action_schema_context(state.affordances, dict(state.action_roles))
    schema = build_action_schema(action, context)
    flags = state.flags_dict()
    has_selector = flags.get("interface_has_mode_actions") == "1" or any(
        build_action_schema(candidate, context).action_type in {"click", "select"} for candidate in state.affordances
    )
    if reward > 0.0 and has_selector:
        return ("belief", "reward_model", "reward_after_activate", "control_binding")
    if schema.action_type in {"click", "select"} or has_selector:
        return ("belief", "interface", "control_binding", "uncertain", "clickable")
    status = "active" if reward > 0.0 else "inactive"
    focus = "target" if reward > 0.0 else "frontier"
    state_token = "explore"
    return ("belief", "goal", status, "focus", focus, "state", state_token)


def _question_tokens(
    state: StructuredState,
    action: str,
    reward: float,
    usefulness: float,
) -> tuple[str, ...]:
    context = build_action_schema_context(state.affordances, dict(state.action_roles))
    schema = build_action_schema(action, context)
    has_selector = state.flags_dict().get("interface_has_mode_actions") == "1" or any(
        build_action_schema(candidate, context).action_type in {"click", "select"} for candidate in state.affordances
    )
    if reward > 0.0 and has_selector:
        intent = "confirm"
        focus = "reward_model"
    elif reward > 0.0:
        intent = "confirm"
        focus = "target"
    elif schema.action_type in {"click", "select"}:
        intent = "test"
        focus = "control_binding"
    elif schema.action_type == "interact":
        intent = "test"
        focus = "reward_model"
    elif usefulness >= 0.25:
        intent = "test" if has_selector else "explore"
        focus = "control_binding" if has_selector else "frontier"
    else:
        intent = "test" if has_selector else "explore"
        focus = "control_binding" if has_selector else "frontier"
    return ("question", "need", intent, "focus", focus, "state", "probe" if has_selector else "explore")


def _plan_tokens(
    state: StructuredState,
    action: str,
    reward: float,
    usefulness: float,
) -> tuple[str, ...]:
    context = build_action_schema_context(state.affordances, dict(state.action_roles))
    schema = build_action_schema(action, context)
    has_selector = state.flags_dict().get("interface_has_mode_actions") == "1" or any(
        build_action_schema(candidate, context).action_type in {"click", "select"} for candidate in state.affordances
    )
    status = "active" if reward > 0.0 else "uncertain" if usefulness >= 0.25 else "inactive"
    direction = schema.direction or "none"
    if schema.action_type in {"click", "select"}:
        return ("plan", "click_then_move", "focus", "control_binding", "state", status)
    if schema.action_type == "interact":
        return ("plan", "move_then_interact", "direction", direction, "focus", "interactable", "state", status)
    if reward > 0.0 and has_selector:
        return ("plan", "bind_then_objective", "direction", direction, "focus", "target", "state", status)
    action_token = schema.action_type if schema.action_type in {"move", "wait"} else "unknown"
    focus = "target" if reward > 0.0 else "frontier"
    return ("plan", "action", action_token, "direction", direction, "focus", focus, "state", status)


def collect_arc_public_sessions(
    config: ArcPublicTrainingConfig,
    *,
    encoder: StructuredStateEncoder | None = None,
    world_model: RecurrentWorldModel | None = None,
    language_model: GroundedLanguageModel | None = None,
    device: torch.device | None = None,
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    _validate_arc_public_training_config(config)
    if not arc_toolkit_available():
        raise RuntimeError("ARC toolkit is not installed in this environment.")
    operation_mode = arc_operation_mode(config.mode)
    games = list_arc_games(operation_mode=operation_mode)[: config.game_limit]
    dataset: list[dict[str, object]] = []
    session_summaries: list[dict[str, object]] = []
    seed_cursor = config.seed
    agents: dict[str, object] = {}
    reset_all_flags: dict[str, bool] = {}
    for game_index, game_id in enumerate(games):
        env = ArcToolkitEnv(game_id, operation_mode=operation_mode)
        try:
            for session_index in range(config.sessions_per_game):
                policy_name = config.behavior_policies[(game_index + session_index) % len(config.behavior_policies)]
                if policy_name not in agents:
                    agent, reset_all = _behavior_agent(
                        policy_name,
                        encoder=encoder,
                        world_model=world_model,
                        language_model=language_model,
                        device=device,
                    )
                    agents[policy_name] = agent
                    reset_all_flags[policy_name] = reset_all
                agent = agents[policy_name]
                if reset_all_flags[policy_name]:
                    agent.reset_all()
                else:
                    agent.reset_episode()
                session_samples, session_summary = collect_arc_session(
                    env,
                    agent=agent,
                    game_id=str(game_id),
                    session_index=session_index,
                    seed=seed_cursor,
                    max_steps=config.max_steps,
                    policy_name=policy_name,
                    sample_builder=_make_sample,
                )
                seed_cursor += 1
                annotate_session_returns(session_samples)
                dataset.extend(session_samples)
                session_summaries.append(asdict(session_summary))
        finally:
            env.close()
    return dataset, session_summaries


def collect_arc_public_dataset(
    config: ArcPublicTrainingConfig,
    *,
    encoder: StructuredStateEncoder | None = None,
    world_model: RecurrentWorldModel | None = None,
    language_model: GroundedLanguageModel | None = None,
    device: torch.device | None = None,
) -> list[dict[str, object]]:
    dataset, _session_summaries = collect_arc_public_sessions(
        config,
        encoder=encoder,
        world_model=world_model,
        language_model=language_model,
        device=device,
    )
    return dataset


def train_arc_public(
    config: ArcPublicTrainingConfig,
    device: torch.device | None = None,
) -> dict[str, float]:
    _validate_arc_public_training_config(config)
    seed_everything(config.seed)
    if not arc_toolkit_available():
        raise RuntimeError("ARC toolkit is not installed in this environment.")
    if device is not None:
        device = torch.device(str(device))
    elif config.device:
        device = torch.device(config.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if config.init_checkpoint_path and Path(config.init_checkpoint_path).exists():
        encoder, world_model, language_model = load_checkpoint(config.init_checkpoint_path, device=device)
    else:
        encoder, world_model, language_model, _planner = build_default_modules(device=device)
    for parameter in encoder.parameters():
        parameter.requires_grad = not config.freeze_encoder
    optimizer = torch.optim.Adam(
        [
            parameter
            for parameter in list(encoder.parameters()) + list(world_model.parameters()) + list(language_model.parameters())
            if parameter.requires_grad
        ],
        lr=config.learning_rate,
    )
    history: list[dict[str, float]] = []
    for epoch in range(config.epochs):
        dataset, session_summaries = collect_arc_public_sessions(
            config,
            encoder=encoder,
            world_model=world_model,
            language_model=language_model,
            device=device,
        )
        encoder.train()
        world_model.train()
        language_model.train()
        epoch_losses: list[float] = []
        epoch_policy_losses: list[float] = []
        epoch_plan_losses: list[float] = []
        epoch_uncertainty: list[float] = []
        sequence_id_cursor: str | None = None
        sequence_hidden: torch.Tensor | None = None
        for sample in dataset:
            state = sample["state"]
            next_state = sample["next_state"]
            action = sample["action"]
            reward = float(sample["reward"])
            delta = sample["delta"]
            usefulness = float(sample["usefulness"])
            belief_tokens = sample["belief_tokens"]
            question_tokens = sample["question_tokens"]
            plan_tokens = sample["plan_tokens"]
            sample_sequence_id = str(sample.get("sequence_id", state.episode_id))
            if sequence_id_cursor != sample_sequence_id:
                sequence_id_cursor = sample_sequence_id
                sequence_hidden = None
            if config.freeze_encoder:
                with torch.no_grad():
                    encoded = encoder.encode_state(state, device=device)
            else:
                encoded = encoder.encode_state(state, device=device)
            with torch.no_grad():
                next_encoded = encoder.encode_state(next_state, device=device)
            reward_target = torch.tensor([reward], dtype=torch.float32, device=device)
            return_target = torch.tensor([float(sample.get("discounted_return", reward))], dtype=torch.float32, device=device)
            delta_target = torch.tensor(delta, dtype=torch.float32, device=device).unsqueeze(0)
            usefulness_target = torch.tensor([usefulness], dtype=torch.float32, device=device)
            world_loss, metrics = world_model.loss(
                latent=encoded.latent,
                actions=[action],
                state=state,
                hidden=sequence_hidden,
                next_latent_target=next_encoded.latent.detach(),
                reward_target=reward_target,
                return_target=return_target,
                delta_target=delta_target,
                usefulness_target=usefulness_target,
            )
            prediction = world_model.step(
                encoded.latent,
                actions=[action],
                state=state,
                hidden=sequence_hidden,
            )
            policy_target = torch.tensor(
                [1.0 if usefulness >= config.policy_positive_threshold else 0.0],
                dtype=torch.float32,
                device=device,
            )
            policy_loss = torch.nn.functional.binary_cross_entropy_with_logits(
                prediction.policy,
                policy_target,
            )
            belief_loss = language_model.teacher_forcing_loss(
                encoded.latent,
                [belief_tokens or ("goal", "unknown")],
                mode="belief",
            )
            question_loss = language_model.teacher_forcing_loss(
                encoded.latent,
                [question_tokens or ("need", "explore")],
                mode="question",
            )
            plan_loss = language_model.teacher_forcing_loss(
                encoded.latent,
                [plan_tokens or ("plan", "action", "unknown")],
                mode="plan",
            )
            loss = world_loss + (0.25 * policy_loss) + (config.language_loss_weight * (belief_loss + question_loss + plan_loss))
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                [
                    parameter
                    for parameter in list(encoder.parameters()) + list(world_model.parameters()) + list(language_model.parameters())
                    if parameter.requires_grad
                ],
                max_norm=1.0,
            )
            optimizer.step()
            with torch.no_grad():
                sequence_hidden = world_model.step(
                    encoded.latent.detach(),
                    actions=[action],
                    state=state,
                    hidden=sequence_hidden,
                ).hidden.detach()
            epoch_losses.append(float(loss.detach().cpu()))
            epoch_policy_losses.append(float(policy_loss.detach().cpu()))
            epoch_plan_losses.append(float(plan_loss.detach().cpu()))
            epoch_uncertainty.append(metrics["uncertainty"])
        history.append(
            {
                "epoch": float(epoch),
                "loss": mean(epoch_losses) if epoch_losses else 0.0,
                "policy_loss": mean(epoch_policy_losses) if epoch_policy_losses else 0.0,
                "plan_loss": mean(epoch_plan_losses) if epoch_plan_losses else 0.0,
                "uncertainty": mean(epoch_uncertainty) if epoch_uncertainty else 0.0,
                "samples": float(len(dataset)),
                "sessions": float(len(session_summaries)),
                "session_win_rate": mean(float(item["won"]) for item in session_summaries) if session_summaries else 0.0,
                "avg_levels_completed": mean(float(item["levels_completed"]) for item in session_summaries)
                if session_summaries
                else 0.0,
                "avg_session_steps": mean(float(item["steps"]) for item in session_summaries) if session_summaries else 0.0,
                "avg_reset_steps": mean(float(item["reset_steps"]) for item in session_summaries) if session_summaries else 0.0,
            }
        )
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "config": asdict(config),
            "encoder": encoder.state_dict(),
            "world_model": world_model.state_dict(),
            "language_model": language_model.state_dict(),
            "history": history,
        },
        checkpoint_path,
    )
    return {
        "epochs": float(config.epochs),
        "samples_last_epoch": history[-1]["samples"] if history else 0.0,
        "loss_last_epoch": history[-1]["loss"] if history else 0.0,
        "policy_loss_last_epoch": history[-1]["policy_loss"] if history else 0.0,
        "plan_loss_last_epoch": history[-1]["plan_loss"] if history else 0.0,
        "uncertainty_last_epoch": history[-1]["uncertainty"] if history else 0.0,
        "session_win_rate_last_epoch": history[-1]["session_win_rate"] if history else 0.0,
        "avg_levels_completed_last_epoch": history[-1]["avg_levels_completed"] if history else 0.0,
        }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m arcagi.training.arc_public")
    parser.add_argument("--mode", type=str, default="online")
    parser.add_argument("--game-limit", type=int, default=25)
    parser.add_argument("--sessions-per-game", "--episodes-per-game", dest="sessions_per_game", type=int, default=2)
    parser.add_argument("--max-steps", type=int, default=96)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--checkpoint-path", type=str, default="artifacts/arc_public_hybrid.pt")
    parser.add_argument("--init-checkpoint-path", type=str, default="artifacts/mixed_policy_hybrid.pt")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--behavior-policies",
        type=str,
        default="graph,learned,random",
        help="comma-separated collector policies from {graph,random,learned}",
    )
    parser.add_argument("--unfreeze-encoder", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = ArcPublicTrainingConfig(
        mode=args.mode,
        game_limit=args.game_limit,
        sessions_per_game=args.sessions_per_game,
        max_steps=args.max_steps,
        epochs=args.epochs,
        learning_rate=args.learning_rate,
        device=args.device,
        checkpoint_path=args.checkpoint_path,
        init_checkpoint_path=args.init_checkpoint_path,
        seed=args.seed,
        behavior_policies=tuple(policy.strip() for policy in args.behavior_policies.split(",") if policy.strip()),
        freeze_encoder=not args.unfreeze_encoder,
    )
    metrics = train_arc_public(config)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
