"""Public ARC goal-scientist training loop.

Drop this file into ``arcagi/training/goal_scientist_public.py`` and run it with
``python -m arcagi.training.goal_scientist_public`` from the repository root.

It is a strict extension of the current ``arc_public`` trainer:

* it keeps black-box ARC interaction only;
* it saves checkpoints with the same ``encoder`` / ``world_model`` /
  ``language_model`` keys consumed by the existing harness;
* it adds a diagnostic collector that performs generic first-contact probes;
* it performs hindsight credit assignment for sparse rewards and delayed
  selector/control-binding effects;
* it trains the causal/effect/diagnostic heads already present in the current
  world model instead of leaving them weakly supervised on public ARC samples.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import asdict, dataclass
import json
from pathlib import Path
from statistics import mean
import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from arcagi.core.utils import seed_everything
from arcagi.envs.arc_adapter import Arcade, ArcToolkitEnv, arc_operation_mode, arc_toolkit_available, list_arc_games
from arcagi.models.world_model import EFFECT_KINDS
from arcagi.perception.object_encoder import extract_structured_state
from arcagi.training.arc_session import annotate_session_returns, collect_arc_session
from arcagi.training.arc_public import _behavior_agent, _make_sample
from arcagi.training.goal_scientist_targets import (
    action_family,
    finite_float,
    relabel_transition_stream,
    sample_delta_norm,
    summarize_credit,
)
from arcagi.training.synthetic import build_default_modules, load_checkpoint


@dataclass(frozen=True)
class GoalScientistPublicTrainingConfig:
    mode: str = "offline"
    game_limit: int = 25
    sessions_per_game: int = 8
    max_steps: int = 192
    epochs: int = 12
    learning_rate: float = 1.0e-4
    seed: int = 17
    device: str = ""
    checkpoint_path: str = "artifacts/goal_scientist_public.pt"
    init_checkpoint_path: str = ""
    behavior_policies: tuple[str, ...] = ("learned", "random")
    freeze_encoder: bool = False
    language_loss_weight: float = 0.25
    policy_loss_weight: float = 0.35
    hindsight_gamma: float = 0.92
    sequence_horizon: int = 14
    max_coordinate_probes: int = 0
    save_epoch_snapshots: bool = True
    sessions_per_update: int = 1
    replay_buffer_sessions: int = 4
    repeat_epoch_seeds: bool = False


_ALLOWED_POLICIES = frozenset({"graph", "random", "learned", "hybrid", "diagnostic"})
_EFFECT_TO_INDEX = {name: index for index, name in enumerate(EFFECT_KINDS)}


def _validate_config(config: GoalScientistPublicTrainingConfig) -> None:
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
    invalid = [policy for policy in config.behavior_policies if policy not in _ALLOWED_POLICIES]
    if invalid:
        raise ValueError(f"unsupported behavior_policies={invalid}; allowed={sorted(_ALLOWED_POLICIES)}")
    if config.hindsight_gamma <= 0.0 or config.hindsight_gamma > 1.0:
        raise ValueError("hindsight_gamma must be in (0, 1]")
    if config.sequence_horizon <= 0:
        raise ValueError("sequence_horizon must be positive")
    if int(config.sessions_per_update) <= 0:
        raise ValueError("sessions_per_update must be positive")
    if int(config.replay_buffer_sessions) <= 0:
        raise ValueError("replay_buffer_sessions must be positive")
    if int(config.replay_buffer_sessions) < int(config.sessions_per_update):
        raise ValueError("replay_buffer_sessions must be >= sessions_per_update")


def _mean_or_zero(values: Sequence[float]) -> float:
    return float(mean(values)) if values else 0.0


def _close_arcade(arcade: Any | None) -> None:
    if arcade is None:
        return
    close_scorecard = getattr(arcade, "close_scorecard", None)
    if callable(close_scorecard):
        try:
            close_scorecard()
        except Exception:
            pass


def _collect_goal_scientist_session(
    env: ArcToolkitEnv,
    *,
    game_id: str,
    session_index: int,
    seed: int,
    policy_name: str,
    config: GoalScientistPublicTrainingConfig,
    encoder: Any | None = None,
    world_model: Any | None = None,
    language_model: Any | None = None,
    device: torch.device | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    agent, reset_all = _goal_behavior_agent(
        policy_name,
        encoder=encoder,
        world_model=world_model,
        language_model=language_model,
        device=device,
        config=config,
    )
    if reset_all:
        agent.reset_all()
    else:
        agent.reset_episode()

    session_start = time.perf_counter()
    raw_session, session_summary = collect_arc_session(
        env,
        agent=agent,
        game_id=str(game_id),
        session_index=session_index,
        seed=seed,
        max_steps=config.max_steps,
        policy_name=policy_name,
        sample_builder=_make_sample,
    )
    for sample in raw_session:
        sample["behavior_policy"] = policy_name
        sample["episode_id"] = str(sample.get("sequence_id", sample.get("session_id", f"{game_id}/{session_index}")))
    annotate_session_returns(raw_session)
    relabeled = relabel_transition_stream(
        raw_session,
        gamma=config.hindsight_gamma,
        sequence_horizon=config.sequence_horizon,
    )
    return relabeled, {
        **session_summary.__dict__,
        "samples_added": int(len(relabeled)),
        "collection_seconds": float(time.perf_counter() - session_start),
    }


def _train_goal_scientist_batch(
    dataset: Sequence[Mapping[str, Any]],
    *,
    encoder: Any,
    world_model: Any,
    language_model: Any,
    optimizer: torch.optim.Optimizer,
    trainable_parameters: Sequence[torch.nn.Parameter],
    config: GoalScientistPublicTrainingConfig,
    device: torch.device,
) -> dict[str, float]:
    if not dataset:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "plan_loss": 0.0,
            "uncertainty": 0.0,
            "effect_target_mean": 0.0,
            "samples": 0.0,
        }

    encoder.train()
    world_model.train()
    language_model.train()

    losses: list[float] = []
    policy_losses: list[float] = []
    plan_losses: list[float] = []
    uncertainty_values: list[float] = []
    effect_targets: list[float] = []

    sequence_id_cursor: str | None = None
    sequence_hidden: torch.Tensor | None = None

    for sample in dataset:
        state = sample["state"]
        next_state = sample["next_state"]
        action = str(sample["action"])
        reward = float(sample["reward"])
        delta = sample["delta"]
        usefulness = float(sample["usefulness"])
        outcome_signal = float(sample.get("outcome_signal", usefulness))
        belief_tokens = sample.get("belief_tokens")
        question_tokens = sample.get("question_tokens")
        plan_tokens = sample.get("plan_tokens")

        sample_sequence_id = str(sample.get("sequence_id", sample.get("episode_id", getattr(state, "episode_id", ""))))
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
        return_target = torch.tensor(
            [float(sample.get("discounted_return", sample.get("hindsight_return", reward)))],
            dtype=torch.float32,
            device=device,
        )
        delta_target = torch.tensor(delta, dtype=torch.float32, device=device).unsqueeze(0)
        usefulness_target = torch.tensor([usefulness], dtype=torch.float32, device=device)
        effect_target = torch.tensor([_effect_target(sample)], dtype=torch.long, device=device)
        causal_target = torch.tensor([_causal_target(sample)], dtype=torch.float32, device=device)
        diagnostic_target = torch.tensor([_diagnostic_target(sample)], dtype=torch.float32, device=device)

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
            effect_target=effect_target,
            causal_target=causal_target,
            diagnostic_target=diagnostic_target,
        )

        prediction = world_model.step(encoded.latent, actions=[action], state=state, hidden=sequence_hidden)
        policy_target = torch.tensor(
            [float(sample.get("policy_target", 1.0 if usefulness >= 0.2 else 0.0))],
            dtype=torch.float32,
            device=device,
        )
        policy_loss = torch.nn.functional.binary_cross_entropy_with_logits(prediction.policy, policy_target)
        belief_loss = language_model.teacher_forcing_loss(
            encoded.latent,
            [belief_tokens or ("belief", "effect", "unknown")],
            mode="belief",
        )
        question_loss = language_model.teacher_forcing_loss(
            encoded.latent,
            [question_tokens or ("question", "test", "frontier")],
            mode="question",
        )
        plan_loss = language_model.teacher_forcing_loss(
            encoded.latent,
            [plan_tokens or ("plan", "seek_information_gain")],
            mode="plan",
        )

        sample_weight = torch.tensor(
            [float(sample.get("diagnostic_weight", 1.0)) + (0.25 * abs(outcome_signal))],
            dtype=torch.float32,
            device=device,
        ).mean()
        loss = (
            sample_weight * world_loss
            + (config.policy_loss_weight * sample_weight * policy_loss)
            + (config.language_loss_weight * (belief_loss + question_loss + plan_loss))
        )

        optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable_parameters, max_norm=1.0)
        optimizer.step()

        with torch.no_grad():
            sequence_hidden = world_model.step(
                encoded.latent.detach(),
                actions=[action],
                state=state,
                hidden=sequence_hidden,
            ).hidden.detach()

        losses.append(float(loss.detach().cpu()))
        policy_losses.append(float(policy_loss.detach().cpu()))
        plan_losses.append(float(plan_loss.detach().cpu()))
        uncertainty_values.append(float(metrics.get("uncertainty", 0.0)))
        effect_targets.append(float(effect_target.detach().cpu().item()))

    return {
        "loss": _mean_or_zero(losses),
        "policy_loss": _mean_or_zero(policy_losses),
        "plan_loss": _mean_or_zero(plan_losses),
        "uncertainty": _mean_or_zero(uncertainty_values),
        "effect_target_mean": _mean_or_zero(effect_targets),
        "samples": float(len(dataset)),
    }


def _grid_from_observation(observation: Any) -> np.ndarray | None:
    grid = getattr(observation, "grid", None)
    if grid is None:
        return None
    try:
        arr = np.asarray(grid)
    except Exception:
        return None
    if arr.ndim != 2 or arr.size <= 0:
        return None
    return arr


def _available_actions(observation: Any) -> tuple[str, ...]:
    actions = getattr(observation, "available_actions", ()) or ()
    return tuple(str(action) for action in actions)


def _canonical_probe_order(actions: Sequence[str]) -> list[str]:
    priority = {
        "coordinate": 0,
        "interact": 1,
        "move_up": 2,
        "move_down": 3,
        "move_left": 4,
        "move_right": 5,
        "move": 6,
        "wait": 7,
        "other": 8,
        "reset": 99,
    }
    return sorted(dict.fromkeys(str(a) for a in actions), key=lambda a: (priority.get(action_family(a), 50), str(a)))


def _color_centroid_clicks(grid: np.ndarray, *, limit: int) -> list[str]:
    """Return salience clicks as ``click:x:y`` using only visible grid data."""

    arr = np.asarray(grid)
    if arr.ndim != 2:
        return []
    height, width = arr.shape
    clicks: list[tuple[int, int, int]] = []
    values, counts = np.unique(arr, return_counts=True)
    pairs = [(int(value), int(count)) for value, count in zip(values.tolist(), counts.tolist())]
    # Prefer small, non-background components, but include the largest nonzero
    # colors too.  This covers buttons, agents, doors, keys, and goals without
    # naming any of them.
    for value, _count in sorted(pairs, key=lambda item: (item[0] == 0, item[1], item[0])):
        if value == 0 and len(pairs) > 1:
            continue
        ys, xs = np.where(arr == value)
        if len(xs) == 0:
            continue
        cx = int(round(float(xs.mean())))
        cy = int(round(float(ys.mean())))
        clicks.append((abs(len(xs) - (height * width // 2)), cx, cy))
    corner_and_center = [
        (0, width // 2, height // 2),
        (0, 0, 0),
        (0, max(width - 1, 0), 0),
        (0, 0, max(height - 1, 0)),
        (0, max(width - 1, 0), max(height - 1, 0)),
    ]
    clicks.extend(corner_and_center)

    out: list[str] = []
    for _score, x, y in clicks:
        x = max(0, min(width - 1, int(x)))
        y = max(0, min(height - 1, int(y)))
        text = f"click:{x}:{y}"
        if text not in out:
            out.append(text)
        if limit > 0 and len(out) >= limit:
            break
    return out


class PublicDiagnosticAgendaAgent:
    """Generic first-contact collector for public ARC environments.

    This agent is not used as the prize-facing runtime controller.  It is a data
    collector that ensures the model sees enough no-effect, effect, selector, and
    coordinate contrasts to train its online hypothesis machinery.  It never
    reads environment files or game IDs and never uses semantic role names.
    """

    def __init__(self, *, max_coordinate_probes: int = 0) -> None:
        self.max_coordinate_probes = int(max_coordinate_probes)
        self._agenda: list[str] = []
        self._cursor = 0
        self._last_signature: tuple[Any, ...] | None = None
        self._tested: set[tuple[tuple[Any, ...], str]] = set()

    def reset_all(self) -> None:
        self._tested.clear()
        self.reset_episode()

    def reset_episode(self) -> None:
        self._agenda = []
        self._cursor = 0
        self._last_signature = None

    def reset_level(self) -> None:
        self._agenda = []
        self._cursor = 0
        self._last_signature = None

    def update_after_step(self, *, next_observation: Any, reward: float, terminated: bool, info: Mapping[str, Any] | None = None) -> None:
        del next_observation, reward, terminated, info

    def act(self, observation: Any) -> str:
        state = extract_structured_state(observation)
        signature = self._signature(state)
        if signature != self._last_signature or not self._agenda:
            self._agenda = self._build_agenda(observation, state)
            self._cursor = 0
            self._last_signature = signature

        actions = _available_actions(observation)
        fallback = _canonical_probe_order(actions)[0] if actions else "1"
        while self._cursor < len(self._agenda):
            candidate = self._agenda[self._cursor]
            self._cursor += 1
            family = action_family(candidate)
            key = (signature, family if family in {"coordinate", "interact"} else candidate)
            if key in self._tested:
                continue
            if self._is_available(candidate, actions):
                self._tested.add(key)
                return candidate
        return fallback

    def _build_agenda(self, observation: Any, state: Any) -> list[str]:
        actions = _canonical_probe_order(_available_actions(observation))
        grid = _grid_from_observation(observation)
        agenda: list[str] = []

        existing_clicks = [action for action in actions if action.startswith("click:")]
        if existing_clicks:
            agenda.extend(existing_clicks if self.max_coordinate_probes <= 0 else existing_clicks[: self.max_coordinate_probes])
        elif self._has_parametric_click(actions) and grid is not None:
            agenda.extend(_color_centroid_clicks(grid, limit=self.max_coordinate_probes))

        # Alternate interaction and movement probes after selectors.  This is
        # the generic sequence-binding curriculum: selector/contact first, then
        # motion and interaction contrasts.
        agenda.extend(action for action in actions if action_family(action) == "interact")
        agenda.extend(action for action in actions if action_family(action).startswith("move"))
        agenda.extend(action for action in actions if action not in agenda and action_family(action) not in {"reset"})
        return list(dict.fromkeys(agenda))

    @staticmethod
    def _is_available(candidate: str, actions: Sequence[str]) -> bool:
        if candidate in actions:
            return True
        if candidate.startswith("click:"):
            return PublicDiagnosticAgendaAgent._has_parametric_click(actions)
        return False

    @staticmethod
    def _has_parametric_click(actions: Sequence[str]) -> bool:
        lowered = {str(action).lower() for action in actions}
        return bool({"6", "action6", "a6", "click", "coordinate"} & lowered) or any(str(a).startswith("click:") for a in actions)

    @staticmethod
    def _signature(state: Any) -> tuple[Any, ...]:
        vector = getattr(state, "transition_vector", None)
        try:
            if callable(vector):
                values = tuple(round(float(x), 3) for x in np.asarray(vector()).reshape(-1)[:64].tolist())
                return (getattr(state, "episode_id", "episode"), values)
        except Exception:
            pass
        return (repr(state)[:512],)


def _goal_behavior_agent(
    name: str,
    *,
    encoder: Any | None,
    world_model: Any | None,
    language_model: Any | None,
    device: torch.device | None,
    config: GoalScientistPublicTrainingConfig,
):
    if name == "diagnostic":
        return PublicDiagnosticAgendaAgent(max_coordinate_probes=config.max_coordinate_probes), True
    return _behavior_agent(name, encoder=encoder, world_model=world_model, language_model=language_model, device=device)


def _effect_target(sample: Mapping[str, Any]) -> int:
    reward = finite_float(sample.get("reward", 0.0))
    tags = tuple(sample.get("credit_tags", ()))
    outcome_signal = finite_float(sample.get("outcome_signal", 0.0))
    future_setback = finite_float(sample.get("future_setback", 0.0))
    transition_credit = sample.get("transition_credit", {})
    state_signal = finite_float(
        transition_credit.get("state_signal", sample_delta_norm(sample)) if isinstance(transition_credit, Mapping) else sample_delta_norm(sample)
    )
    if reward > 0.0:
        return _EFFECT_TO_INDEX.get("reward_gain", 0)
    if reward < 0.0 or "terminal_failure" in tags or outcome_signal < -0.2 or future_setback > 0.2:
        return _EFFECT_TO_INDEX.get("setback", 1)
    if state_signal > 0.05:
        return _EFFECT_TO_INDEX.get("state_change", 2)
    if "sequence_prefix" in tags or "selector_binding" in tags:
        return _EFFECT_TO_INDEX.get("latent_shift", 3)
    return _EFFECT_TO_INDEX.get("no_effect", len(EFFECT_KINDS) - 1)


def _causal_target(sample: Mapping[str, Any]) -> float:
    tags = tuple(sample.get("credit_tags", ()))
    outcome_signal = finite_float(sample.get("outcome_signal", 0.0))
    transition_credit = sample.get("transition_credit", {})
    state_signal = finite_float(
        transition_credit.get("state_signal", sample_delta_norm(sample)) if isinstance(transition_credit, Mapping) else sample_delta_norm(sample)
    )
    value = 0.0
    if finite_float(sample.get("reward", 0.0)) > 0.0:
        value += 1.0
    if state_signal > 1.0e-5:
        value += 0.35 * min(state_signal, 1.0)
    if "selector_binding" in tags:
        value += 0.65
    if "sequence_prefix" in tags:
        value += 0.35
    if "terminal_failure" in tags or outcome_signal < -0.1:
        value -= 0.55 + (0.35 * min(abs(outcome_signal), 1.0))
    return max(-1.5, min(1.5, value))


def _diagnostic_target(sample: Mapping[str, Any]) -> float:
    tags = tuple(sample.get("credit_tags", ()))
    outcome_signal = finite_float(sample.get("outcome_signal", 0.0))
    value = finite_float(sample.get("policy_target", 0.0))
    if "novel_probe" in tags:
        value += 0.25
    if "no_effect" in tags and "sequence_prefix" not in tags:
        value = min(value, 0.25)
    if "selector_binding" in tags:
        value += 0.45
    if "terminal_failure" in tags:
        value += 0.15 if "novel_probe" in tags else -0.20
    if outcome_signal < 0.0:
        value -= 0.20 * min(abs(outcome_signal), 1.5)
    return max(-1.0, min(1.5, value))


def collect_goal_scientist_public_dataset(
    config: GoalScientistPublicTrainingConfig,
    *,
    encoder: Any | None = None,
    world_model: Any | None = None,
    language_model: Any | None = None,
    device: torch.device | None = None,
    epoch: int | None = None,
    games: Sequence[str] | None = None,
    shared_arcade: Any | None = None,
) -> list[dict[str, Any]]:
    _validate_config(config)
    if not arc_toolkit_available():
        raise RuntimeError("ARC toolkit is not installed in this environment.")

    operation_mode = arc_operation_mode(config.mode)
    game_ids = list(games) if games is not None else list_arc_games(operation_mode=operation_mode)[: config.game_limit]
    dataset: list[dict[str, Any]] = []
    seed_cursor = int(config.seed)
    total_sessions = len(game_ids) * int(config.sessions_per_game)
    collected_sessions = 0
    owns_arcade = False

    if shared_arcade is None and Arcade is not None:
        shared_arcade = Arcade(operation_mode=operation_mode)
        owns_arcade = True

    try:
        for game_index, game_id in enumerate(game_ids):
            env = ArcToolkitEnv(game_id, operation_mode=operation_mode, arcade=shared_arcade)
            try:
                for session_index in range(config.sessions_per_game):
                    policy_name = config.behavior_policies[(game_index + session_index) % len(config.behavior_policies)]
                    relabeled, session_record = _collect_goal_scientist_session(
                        env,
                        game_id=str(game_id),
                        session_index=session_index,
                        seed=seed_cursor,
                        policy_name=policy_name,
                        config=config,
                        encoder=encoder,
                        world_model=world_model,
                        language_model=language_model,
                        device=device,
                    )
                    seed_cursor += 1
                    dataset.extend(relabeled)
                    collected_sessions += 1
                    print(
                        json.dumps(
                            {
                                "goal_scientist_collect": {
                                    "epoch": -1 if epoch is None else int(epoch),
                                    "sessions_complete": int(collected_sessions),
                                    "sessions_total": int(total_sessions),
                                    "episodes_complete": int(collected_sessions),
                                    "episodes_total": int(total_sessions),
                                    "dataset_samples": int(len(dataset)),
                                    **session_record,
                                }
                            },
                            sort_keys=True,
                        ),
                        flush=True,
                    )
            finally:
                env.close()
    finally:
        if owns_arcade:
            _close_arcade(shared_arcade)
    return dataset


def train_goal_scientist_public(
    config: GoalScientistPublicTrainingConfig,
    device: torch.device | None = None,
) -> dict[str, float]:
    _validate_config(config)
    seed_everything(config.seed)
    if not arc_toolkit_available():
        raise RuntimeError("ARC toolkit is not installed in this environment.")
    operation_mode = arc_operation_mode(config.mode)
    games = list_arc_games(operation_mode=operation_mode)[: config.game_limit]
    if not games:
        raise RuntimeError(f"No ARC games available for mode={config.mode!r}.")

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

    trainable_parameters = [
        parameter
        for parameter in list(encoder.parameters()) + list(world_model.parameters()) + list(language_model.parameters())
        if parameter.requires_grad
    ]
    if not trainable_parameters:
        raise RuntimeError("No trainable parameters remain; unset freeze_encoder or check checkpoint loading.")
    optimizer = torch.optim.Adam(trainable_parameters, lr=config.learning_rate)

    history: list[dict[str, float]] = []
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    shared_arcade = None if Arcade is None else Arcade(operation_mode=operation_mode)
    total_sessions_per_epoch = len(games) * int(config.sessions_per_game)
    seed_cursor = int(config.seed)
    print(
        json.dumps(
            {
                "goal_scientist_train_start": {
                    "mode": config.mode,
                    "games": list(games),
                    "epochs": int(config.epochs),
                    "sessions_per_game": int(config.sessions_per_game),
                    "max_steps": int(config.max_steps),
                    "device": str(device),
                    "behavior_policies": list(config.behavior_policies),
                    "sessions_per_update": int(config.sessions_per_update),
                    "replay_buffer_sessions": int(config.replay_buffer_sessions),
                    "repeat_epoch_seeds": bool(config.repeat_epoch_seeds),
                }
            },
            sort_keys=True,
        ),
        flush=True,
    )

    try:
        for epoch in range(config.epochs):
            if config.repeat_epoch_seeds:
                seed_cursor = int(config.seed)

            collection_seconds = 0.0
            train_seconds = 0.0
            epoch_dataset: list[dict[str, Any]] = []
            session_rewards: list[float] = []
            session_steps: list[float] = []
            session_levels_completed: list[float] = []
            session_reset_steps: list[float] = []
            positive_reward_sessions = 0
            won_sessions = 0
            weighted_loss_sum = 0.0
            weighted_policy_loss_sum = 0.0
            weighted_plan_loss_sum = 0.0
            weighted_uncertainty_sum = 0.0
            weighted_effect_sum = 0.0
            trained_samples = 0.0
            update_index = 0
            sessions_since_update = 0
            sessions_complete = 0
            session_buffer: deque[list[dict[str, Any]]] = deque(maxlen=int(config.replay_buffer_sessions))

            for game_index, game_id in enumerate(games):
                env = ArcToolkitEnv(game_id, operation_mode=operation_mode, arcade=shared_arcade)
                try:
                    for session_index in range(config.sessions_per_game):
                        policy_name = config.behavior_policies[(game_index + session_index) % len(config.behavior_policies)]
                        relabeled, session_record = _collect_goal_scientist_session(
                            env,
                            game_id=str(game_id),
                            session_index=session_index,
                            seed=seed_cursor,
                            policy_name=policy_name,
                            config=config,
                            encoder=encoder,
                            world_model=world_model,
                            language_model=language_model,
                            device=device,
                        )
                        seed_cursor += 1
                        epoch_dataset.extend(relabeled)
                        session_buffer.append(relabeled)
                        sessions_complete += 1
                        sessions_since_update += 1
                        collection_seconds += float(session_record["collection_seconds"])
                        session_rewards.append(float(session_record["reward"]))
                        session_steps.append(float(session_record["steps"]))
                        session_levels_completed.append(float(session_record["levels_completed"]))
                        session_reset_steps.append(float(session_record["reset_steps"]))
                        if float(session_record["reward"]) > 0.0:
                            positive_reward_sessions += 1
                        if bool(session_record["won"]):
                            won_sessions += 1
                        print(
                            json.dumps(
                                {
                                    "goal_scientist_collect": {
                                        "epoch": int(epoch),
                                        "sessions_complete": int(sessions_complete),
                                        "sessions_total": int(total_sessions_per_epoch),
                                        "episodes_complete": int(sessions_complete),
                                        "episodes_total": int(total_sessions_per_epoch),
                                        "dataset_samples": int(len(epoch_dataset)),
                                        **session_record,
                                    }
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )

                        should_update = (
                            sessions_since_update >= int(config.sessions_per_update)
                            or sessions_complete >= total_sessions_per_epoch
                        )
                        if not should_update:
                            continue

                        update_dataset = [sample for session_samples in session_buffer for sample in session_samples]
                        update_start = time.perf_counter()
                        update_metrics = _train_goal_scientist_batch(
                            update_dataset,
                            encoder=encoder,
                            world_model=world_model,
                            language_model=language_model,
                            optimizer=optimizer,
                            trainable_parameters=trainable_parameters,
                            config=config,
                            device=device,
                        )
                        train_seconds += time.perf_counter() - update_start
                        batch_samples = float(update_metrics.get("samples", 0.0))
                        trained_samples += batch_samples
                        weighted_loss_sum += float(update_metrics.get("loss", 0.0)) * batch_samples
                        weighted_policy_loss_sum += float(update_metrics.get("policy_loss", 0.0)) * batch_samples
                        weighted_plan_loss_sum += float(update_metrics.get("plan_loss", 0.0)) * batch_samples
                        weighted_uncertainty_sum += float(update_metrics.get("uncertainty", 0.0)) * batch_samples
                        weighted_effect_sum += float(update_metrics.get("effect_target_mean", 0.0)) * batch_samples
                        print(
                            json.dumps(
                                {
                                    "goal_scientist_update": {
                                        "epoch": int(epoch),
                                        "update_index": int(update_index),
                                        "sessions_complete": int(sessions_complete),
                                        "sessions_total": int(total_sessions_per_epoch),
                                        "episodes_complete": int(sessions_complete),
                                        "episodes_total": int(total_sessions_per_epoch),
                                        "batch_sessions": int(len(session_buffer)),
                                        "batch_episodes": int(len(session_buffer)),
                                        "batch_samples": int(batch_samples),
                                        **{key: float(value) for key, value in update_metrics.items()},
                                    }
                                },
                                sort_keys=True,
                            ),
                            flush=True,
                        )
                        update_index += 1
                        sessions_since_update = 0
                finally:
                    env.close()

            credit_summary = summarize_credit(epoch_dataset)
            train_denominator = trained_samples if trained_samples > 0.0 else 1.0
            record = {
                "epoch": float(epoch),
                "loss": float(weighted_loss_sum / train_denominator),
                "policy_loss": float(weighted_policy_loss_sum / train_denominator),
                "plan_loss": float(weighted_plan_loss_sum / train_denominator),
                "uncertainty": float(weighted_uncertainty_sum / train_denominator),
                "effect_target_mean": float(weighted_effect_sum / train_denominator),
                "samples": float(len(epoch_dataset)),
                "trained_samples": float(trained_samples),
                "games": float(len(games)),
                "sessions": float(total_sessions_per_epoch),
                "episodes": float(total_sessions_per_epoch),
                "updates": float(update_index),
                "avg_session_reward": _mean_or_zero(session_rewards),
                "avg_session_steps": _mean_or_zero(session_steps),
                "avg_levels_completed": _mean_or_zero(session_levels_completed),
                "avg_reset_steps": _mean_or_zero(session_reset_steps),
                "positive_reward_sessions": float(positive_reward_sessions),
                "won_sessions": float(won_sessions),
                "session_win_rate": float(won_sessions) / float(max(total_sessions_per_epoch, 1)),
                "avg_episode_reward": _mean_or_zero(session_rewards),
                "avg_episode_steps": _mean_or_zero(session_steps),
                "positive_reward_episodes": float(positive_reward_sessions),
                "solved_episodes": float(won_sessions),
                "collection_seconds": float(collection_seconds),
                "train_seconds": float(train_seconds),
                "sessions_per_update": float(config.sessions_per_update),
                "replay_buffer_sessions": float(config.replay_buffer_sessions),
                "episodes_per_update": float(config.sessions_per_update),
                "replay_buffer_episodes": float(config.replay_buffer_sessions),
                **{f"credit_{key}": float(value) for key, value in credit_summary.items()},
            }
            history.append(record)
            print(json.dumps({"goal_scientist_epoch": record}, sort_keys=True), flush=True)

            payload = {
                "config": asdict(config),
                "encoder": encoder.state_dict(),
                "world_model": world_model.state_dict(),
                "language_model": language_model.state_dict(),
                "history": history,
                "trainer": "goal_scientist_public",
            }
            torch.save(payload, checkpoint_path)
            if config.save_epoch_snapshots:
                torch.save(payload, checkpoint_path.with_suffix(f".epoch_{epoch:04d}.pt"))
    finally:
        _close_arcade(shared_arcade)

    last = history[-1] if history else {}
    return {
        "epochs": float(config.epochs),
        "samples_last_epoch": float(last.get("samples", 0.0)),
        "loss_last_epoch": float(last.get("loss", 0.0)),
        "policy_loss_last_epoch": float(last.get("policy_loss", 0.0)),
        "plan_loss_last_epoch": float(last.get("plan_loss", 0.0)),
        "uncertainty_last_epoch": float(last.get("uncertainty", 0.0)),
        "positive_reward_sessions_last_epoch": float(last.get("positive_reward_sessions", 0.0)),
        "avg_session_steps_last_epoch": float(last.get("avg_session_steps", 0.0)),
        "session_win_rate_last_epoch": float(last.get("session_win_rate", 0.0)),
        "avg_levels_completed_last_epoch": float(last.get("avg_levels_completed", 0.0)),
        "positive_reward_episodes_last_epoch": float(last.get("positive_reward_sessions", 0.0)),
        "avg_episode_steps_last_epoch": float(last.get("avg_session_steps", 0.0)),
        "checkpoint_path": str(checkpoint_path),
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="python -m arcagi.training.goal_scientist_public")
    parser.add_argument("--mode", type=str, default="offline")
    parser.add_argument("--game-limit", type=int, default=25)
    parser.add_argument("--sessions-per-game", "--episodes-per-game", dest="sessions_per_game", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=192)
    parser.add_argument("--epochs", type=int, default=12)
    parser.add_argument("--learning-rate", type=float, default=1.0e-4)
    parser.add_argument("--device", type=str, default="")
    parser.add_argument("--checkpoint-path", type=str, default="artifacts/goal_scientist_public.pt")
    parser.add_argument("--init-checkpoint-path", type=str, default="")
    parser.add_argument("--seed", type=int, default=17)
    parser.add_argument(
        "--behavior-policies",
        type=str,
        default="learned,random",
        help="comma-separated collector policies from {learned,random}; graph/hybrid/diagnostic are baseline-only and not valid for clean success claims",
    )
    parser.add_argument("--freeze-encoder", action="store_true")
    parser.add_argument("--language-loss-weight", type=float, default=0.25)
    parser.add_argument("--policy-loss-weight", type=float, default=0.35)
    parser.add_argument("--hindsight-gamma", type=float, default=0.92)
    parser.add_argument("--sequence-horizon", type=int, default=14)
    parser.add_argument("--max-coordinate-probes", type=int, default=0)
    parser.add_argument("--no-epoch-snapshots", action="store_true")
    parser.add_argument("--sessions-per-update", "--episodes-per-update", dest="sessions_per_update", type=int, default=1)
    parser.add_argument(
        "--replay-buffer-sessions",
        "--replay-buffer-episodes",
        dest="replay_buffer_sessions",
        type=int,
        default=4,
    )
    parser.add_argument("--repeat-epoch-seeds", action="store_true")
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    config = GoalScientistPublicTrainingConfig(
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
        freeze_encoder=bool(args.freeze_encoder),
        language_loss_weight=args.language_loss_weight,
        policy_loss_weight=args.policy_loss_weight,
        hindsight_gamma=args.hindsight_gamma,
        sequence_horizon=args.sequence_horizon,
        max_coordinate_probes=args.max_coordinate_probes,
        save_epoch_snapshots=not bool(args.no_epoch_snapshots),
        sessions_per_update=args.sessions_per_update,
        replay_buffer_sessions=args.replay_buffer_sessions,
        repeat_epoch_seeds=bool(args.repeat_epoch_seeds),
    )
    metrics = train_goal_scientist_public(config)
    print(json.dumps(metrics, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
