"""Surprise-weighted episodic and option memory for online adaptation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping

import numpy as np

from .effects import numeric_delta_tags, state_schema_tokens, transition_numeric_deltas
from .features import cosine, jaccard, state_features
from .types import ActionName, StructuredState, TransitionRecord, action_family, combined_progress_signal


@dataclass
class MemoryItem:
    step: int
    before_fingerprint: str
    after_fingerprint: str
    action: ActionName
    reward: float
    surprise: float
    state_vector: np.ndarray
    language_tokens: frozenset[str]
    note: str = ""


@dataclass
class OptionItem:
    action_sequence: tuple[ActionName, ...]
    first_action: ActionName
    family_sequence: tuple[str, ...]
    state_vector: np.ndarray
    language_tokens: frozenset[str]
    effect_tags: frozenset[str]
    precondition_tokens: frozenset[str]
    relative_cost: float
    effect_value: float
    reward: float
    uses: int = 0
    successes: int = 0
    support: float = 0.0
    contradiction: float = 0.0

    def update(
        self,
        *,
        precondition_tokens: frozenset[str],
        relative_cost: float,
        effect_value: float,
        reward: float,
        success: bool,
    ) -> None:
        self.uses += 1
        if success:
            self.successes += 1
        self.support += 1.0 if (reward > 0.0 or effect_value >= 0.12) else 0.25
        decay = 1.0 / max(float(self.uses), 1.0)
        self.relative_cost = float(((1.0 - decay) * self.relative_cost) + (decay * relative_cost))
        self.effect_value = float(((1.0 - decay) * self.effect_value) + (decay * effect_value))
        self.reward = float(((1.0 - decay) * self.reward) + (decay * reward))
        merged = set(self.precondition_tokens)
        merged.update(precondition_tokens)
        self.precondition_tokens = frozenset(sorted(merged)[:24])

    @property
    def value(self) -> float:
        signal = self.effect_value + (0.85 * self.reward) + (0.35 * self.successes) + (0.12 * self.support)
        signal -= 0.15 * self.contradiction
        return float(signal / max(self.relative_cost, 1.0))

    @property
    def efficiency(self) -> float:
        return float((self.effect_value + max(self.reward, 0.0)) / max(self.relative_cost, 1.0))


@dataclass
class EffectSchema:
    schema_key: str
    family_sequence: tuple[str, ...]
    effect_tags: frozenset[str]
    avg_relative_cost: float
    avg_effect_value: float
    avg_reward: float
    uses: int = 0
    successes: int = 0
    support: float = 0.0
    contradiction: float = 0.0
    precondition_weights: dict[str, float] = field(default_factory=dict)

    @property
    def precondition_tokens(self) -> frozenset[str]:
        ranked = sorted(self.precondition_weights.items(), key=lambda item: (item[1], item[0]), reverse=True)
        return frozenset(token for token, _ in ranked[:16])

    @property
    def value(self) -> float:
        signal = self.avg_effect_value + (0.85 * self.avg_reward)
        signal += 0.25 * (float(self.successes) / max(float(self.uses), 1.0))
        signal += 0.10 * self.support
        signal -= 0.12 * self.contradiction
        return float(signal / max(self.avg_relative_cost, 1.0))

    @property
    def efficiency(self) -> float:
        return float((self.avg_effect_value + max(self.avg_reward, 0.0)) / max(self.avg_relative_cost, 1.0))

    @property
    def confidence(self) -> float:
        posterior = (self.support + 1.0) / (self.support + self.contradiction + 2.0)
        saturation = min(float(self.uses), 8.0) / 8.0
        return float(posterior * (0.35 + 0.65 * saturation))

    def update(
        self,
        *,
        precondition_tokens: frozenset[str],
        relative_cost: float,
        effect_value: float,
        reward: float,
        success: bool,
    ) -> None:
        self.uses += 1
        if success:
            self.successes += 1
        self.support += 1.0 if (reward > 0.0 or effect_value >= 0.12) else 0.25
        decay = 1.0 / max(float(self.uses), 1.0)
        self.avg_relative_cost = float(((1.0 - decay) * self.avg_relative_cost) + (decay * relative_cost))
        self.avg_effect_value = float(((1.0 - decay) * self.avg_effect_value) + (decay * effect_value))
        self.avg_reward = float(((1.0 - decay) * self.avg_reward) + (decay * reward))
        for token in precondition_tokens:
            self.precondition_weights[token] = self.precondition_weights.get(token, 0.0) + 1.0


class EpisodicMemory:
    def __init__(self, *, capacity: int = 2048, feature_dim: int = 256) -> None:
        self.capacity = int(capacity)
        self.feature_dim = int(feature_dim)
        self.items: list[MemoryItem] = []
        self.options: list[OptionItem] = []
        self.schemas: dict[str, EffectSchema] = {}
        self.recent_actions: list[ActionName] = []
        self.recent_salient_flags: list[bool] = []

    def reset_episode(self) -> None:
        self.items.clear()
        self.options.clear()
        self.schemas.clear()
        self.recent_actions.clear()
        self.recent_salient_flags.clear()

    def reset_level(self) -> None:
        self.recent_actions.clear()
        self.recent_salient_flags.clear()

    def state_dict(self) -> dict[str, object]:
        return {
            "capacity": int(self.capacity),
            "feature_dim": int(self.feature_dim),
            "recent_actions": list(self.recent_actions),
            "recent_salient_flags": [bool(item) for item in self.recent_salient_flags],
            "items": [
                {
                    "step": int(item.step),
                    "before_fingerprint": item.before_fingerprint,
                    "after_fingerprint": item.after_fingerprint,
                    "action": item.action,
                    "reward": float(item.reward),
                    "surprise": float(item.surprise),
                    "state_vector": item.state_vector.copy(),
                    "language_tokens": tuple(sorted(item.language_tokens)),
                    "note": item.note,
                }
                for item in self.items
            ],
            "options": [
                {
                    "action_sequence": tuple(option.action_sequence),
                    "first_action": option.first_action,
                    "family_sequence": tuple(option.family_sequence),
                    "state_vector": option.state_vector.copy(),
                    "language_tokens": tuple(sorted(option.language_tokens)),
                    "effect_tags": tuple(sorted(option.effect_tags)),
                    "precondition_tokens": tuple(sorted(option.precondition_tokens)),
                    "relative_cost": float(option.relative_cost),
                    "effect_value": float(option.effect_value),
                    "reward": float(option.reward),
                    "uses": int(option.uses),
                    "successes": int(option.successes),
                    "support": float(option.support),
                    "contradiction": float(option.contradiction),
                }
                for option in self.options
            ],
            "schemas": [
                {
                    "schema_key": schema.schema_key,
                    "family_sequence": tuple(schema.family_sequence),
                    "effect_tags": tuple(sorted(schema.effect_tags)),
                    "avg_relative_cost": float(schema.avg_relative_cost),
                    "avg_effect_value": float(schema.avg_effect_value),
                    "avg_reward": float(schema.avg_reward),
                    "uses": int(schema.uses),
                    "successes": int(schema.successes),
                    "support": float(schema.support),
                    "contradiction": float(schema.contradiction),
                    "precondition_weights": dict(schema.precondition_weights),
                }
                for schema in self.schemas.values()
            ],
        }

    def load_state_dict(self, state: Mapping[str, object]) -> None:
        self.capacity = int(state.get("capacity", self.capacity))
        self.feature_dim = int(state.get("feature_dim", self.feature_dim))
        self.items.clear()
        self.options.clear()
        self.schemas.clear()
        self.recent_actions = [str(action) for action in state.get("recent_actions", [])][-12:]
        self.recent_salient_flags = [bool(item) for item in state.get("recent_salient_flags", [])][-12:]
        if not self.recent_actions:
            self.recent_salient_flags.clear()
        elif len(self.recent_salient_flags) > len(self.recent_actions):
            self.recent_salient_flags = self.recent_salient_flags[-len(self.recent_actions) :]
        while len(self.recent_salient_flags) < len(self.recent_actions):
            self.recent_salient_flags.insert(0, False)

        for raw in state.get("items", []):
            if not isinstance(raw, Mapping):
                continue
            self.items.append(
                MemoryItem(
                    step=int(raw.get("step", 0)),
                    before_fingerprint=str(raw.get("before_fingerprint", "")),
                    after_fingerprint=str(raw.get("after_fingerprint", "")),
                    action=str(raw.get("action", "")),
                    reward=float(raw.get("reward", 0.0)),
                    surprise=float(raw.get("surprise", 0.0)),
                    state_vector=_coerce_memory_vector(raw.get("state_vector"), self.feature_dim),
                    language_tokens=frozenset(str(token) for token in raw.get("language_tokens", ())),
                    note=str(raw.get("note", "")),
                )
            )
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity :]

        for raw in state.get("options", []):
            if not isinstance(raw, Mapping):
                continue
            self.options.append(
                OptionItem(
                    action_sequence=tuple(str(action) for action in raw.get("action_sequence", ())),
                    first_action=str(raw.get("first_action", "")),
                    family_sequence=tuple(str(family) for family in raw.get("family_sequence", ())),
                    state_vector=_coerce_memory_vector(raw.get("state_vector"), self.feature_dim),
                    language_tokens=frozenset(str(token) for token in raw.get("language_tokens", ())),
                    effect_tags=frozenset(str(token) for token in raw.get("effect_tags", ())),
                    precondition_tokens=frozenset(str(token) for token in raw.get("precondition_tokens", ())),
                    relative_cost=float(raw.get("relative_cost", 1.0)),
                    effect_value=float(raw.get("effect_value", 0.0)),
                    reward=float(raw.get("reward", 0.0)),
                    uses=int(raw.get("uses", 0)),
                    successes=int(raw.get("successes", 0)),
                    support=float(raw.get("support", 0.0)),
                    contradiction=float(raw.get("contradiction", 0.0)),
                )
            )
        if len(self.options) > self.capacity // 4:
            self.options = sorted(self.options, key=lambda o: o.value, reverse=True)[: self.capacity // 4]

        for raw in state.get("schemas", []):
            if not isinstance(raw, Mapping):
                continue
            weights = raw.get("precondition_weights", {})
            self.schemas[str(raw.get("schema_key", ""))] = EffectSchema(
                schema_key=str(raw.get("schema_key", "")),
                family_sequence=tuple(str(family) for family in raw.get("family_sequence", ())),
                effect_tags=frozenset(str(token) for token in raw.get("effect_tags", ())),
                avg_relative_cost=float(raw.get("avg_relative_cost", 1.0)),
                avg_effect_value=float(raw.get("avg_effect_value", 0.0)),
                avg_reward=float(raw.get("avg_reward", 0.0)),
                uses=int(raw.get("uses", 0)),
                successes=int(raw.get("successes", 0)),
                support=float(raw.get("support", 0.0)),
                contradiction=float(raw.get("contradiction", 0.0)),
                precondition_weights={
                    str(key): float(value)
                    for key, value in (weights.items() if isinstance(weights, Mapping) else ())
                },
            )
        self.schemas = {key: schema for key, schema in self.schemas.items() if key}

    def write_transition(
        self,
        record: TransitionRecord,
        *,
        surprise: float,
        language_tokens: Iterable[str],
        option_mode: str = "write",
    ) -> None:
        reward = float(combined_progress_signal(record.reward, record.delta.score_delta))
        salient = abs(reward) > 1e-8 or record.delta.has_visible_effect or record.delta.terminated
        self.recent_actions.append(record.action)
        self.recent_salient_flags.append(bool(salient))
        if len(self.recent_actions) > 12:
            self.recent_actions.pop(0)
        if len(self.recent_salient_flags) > 12:
            self.recent_salient_flags.pop(0)
        effect_tags = _effect_signature_tags(record)
        mechanic_value = _mechanic_option_value(record, effect_tags=effect_tags)
        precondition_tokens = state_schema_tokens(record.before)
        should_write = surprise > 0.18 or abs(reward) > 1e-8 or record.delta.has_visible_effect
        if not should_write:
            return
        item = MemoryItem(
            step=record.step_index,
            before_fingerprint=record.before.abstract_fingerprint,
            after_fingerprint=record.after.abstract_fingerprint,
            action=record.action,
            reward=reward,
            surprise=float(surprise),
            state_vector=state_features(record.before, dim=self.feature_dim),
            language_tokens=frozenset(language_tokens),
            note=_transition_note(record),
        )
        self.items.append(item)
        if len(self.items) > self.capacity:
            self.items = self.items[-self.capacity :]

        option_mode = str(option_mode).strip().lower()
        seq = self._recent_option_sequence()
        family_sequence = tuple(action_family(action) for action in seq) if seq else (action_family(record.action),)
        relative_cost = _relative_option_cost(seq, record)
        if option_mode == "contradict":
            self._apply_option_contradiction(action_sequence=seq, family_sequence=family_sequence, effect_tags=effect_tags)
            return
        if option_mode != "write":
            return

        option_value = max(reward, mechanic_value)
        if option_value > 0.0:
            first_action = seq[0] if seq else record.action
            option = self._find_option(seq, effect_tags)
            if option is None:
                self.options.append(
                    OptionItem(
                        action_sequence=seq,
                        first_action=first_action,
                        family_sequence=family_sequence,
                        state_vector=state_features(record.before, dim=self.feature_dim),
                        language_tokens=frozenset(language_tokens),
                        effect_tags=effect_tags,
                        precondition_tokens=precondition_tokens,
                        relative_cost=relative_cost,
                        effect_value=mechanic_value,
                        reward=reward,
                        uses=1,
                        successes=1 if reward > 0.0 else 0,
                        support=1.0 if (reward > 0.0 or mechanic_value >= 0.12) else 0.25,
                        contradiction=0.0,
                    )
                )
            else:
                option.state_vector = state_features(record.before, dim=self.feature_dim)
                option.language_tokens = frozenset(set(option.language_tokens) | set(language_tokens))
                option.update(
                    precondition_tokens=precondition_tokens,
                    relative_cost=relative_cost,
                    effect_value=mechanic_value,
                    reward=reward,
                    success=reward > 0.0,
                )
            if len(self.options) > self.capacity // 4:
                self.options = sorted(self.options, key=lambda o: o.value, reverse=True)[: self.capacity // 4]
            self._update_schema(
                family_sequence=family_sequence,
                precondition_tokens=precondition_tokens,
                effect_tags=effect_tags,
                relative_cost=relative_cost,
                effect_value=mechanic_value,
                reward=reward,
                success=reward > 0.0,
            )

    def retrieve(self, state: StructuredState, language_tokens: Iterable[str], *, k: int = 8) -> tuple[MemoryItem, ...]:
        if not self.items:
            return ()
        vec = state_features(state, dim=self.feature_dim)
        tokens = frozenset(language_tokens)
        scored: list[tuple[float, MemoryItem]] = []
        for item in self.items:
            score = 0.65 * cosine(vec, item.state_vector) + 0.25 * jaccard(tokens, item.language_tokens)
            score += 0.15 * max(item.reward, 0.0) + 0.05 * item.surprise
            scored.append((float(score), item))
        scored.sort(key=lambda x: x[0], reverse=True)
        return tuple(item for _, item in scored[:k])

    def retrieve_options(self, state: StructuredState, language_tokens: Iterable[str], *, k: int = 4) -> tuple[OptionItem, ...]:
        if not self.options:
            return ()
        vec = state_features(state, dim=self.feature_dim)
        tokens = frozenset(language_tokens)
        scored: list[tuple[float, OptionItem]] = []
        for option in self.options:
            score = 0.55 * cosine(vec, option.state_vector) + 0.20 * jaccard(tokens, option.language_tokens)
            score += 0.20 * jaccard(tokens, option.effect_tags)
            score += 0.15 * jaccard(state_schema_tokens(state), option.precondition_tokens)
            score += 0.35 * option.value
            scored.append((float(score), option))
        scored.sort(key=lambda x: x[0], reverse=True)
        return tuple(item for _, item in scored[:k])

    def retrieve_schemas(self, state: StructuredState, language_tokens: Iterable[str], *, k: int = 6) -> tuple[EffectSchema, ...]:
        if not self.schemas:
            return ()
        tokens = frozenset(language_tokens)
        state_tokens = state_schema_tokens(state)
        scored: list[tuple[float, EffectSchema]] = []
        for schema in self.schemas.values():
            score = 0.45 * jaccard(state_tokens, schema.precondition_tokens)
            score += 0.20 * jaccard(tokens, schema.effect_tags)
            score += 0.30 * schema.value
            score += 0.10 * schema.confidence
            scored.append((float(score), schema))
        scored.sort(key=lambda item: item[0], reverse=True)
        return tuple(schema for _, schema in scored[:k])

    def action_memory_bonus(self, state: StructuredState, action: ActionName, language_tokens: Iterable[str]) -> float:
        bonus = 0.0
        recent_actions = tuple(self.recent_actions[-6:])
        for item in self.retrieve(state, language_tokens, k=8):
            if item.action == action:
                bonus += 0.08 * item.surprise + 0.12 * max(item.reward, 0.0)
        for option in self.retrieve_options(state, language_tokens, k=4):
            alignment = _option_alignment(recent_actions, action, option)
            if alignment is not None:
                bonus += alignment["weight"] * (
                    (0.15 * option.value) + (0.10 * option.efficiency) + (0.05 * option.support)
                )
        recent_families = tuple(action_family(item) for item in recent_actions)
        for schema in self.retrieve_schemas(state, language_tokens, k=6):
            alignment = _schema_alignment(recent_families, action, schema)
            if alignment is not None:
                bonus += alignment["weight"] * ((0.12 * schema.value) + (0.08 * schema.confidence))
        return float(bonus)

    def action_option_profile(self, state: StructuredState, action: ActionName, language_tokens: Iterable[str]) -> dict[str, float]:
        recent_actions = tuple(self.recent_actions[-6:])
        recent_families = tuple(action_family(item) for item in recent_actions)
        weighted_entries: list[dict[str, float]] = []

        for option in self.retrieve_options(state, language_tokens, k=12):
            alignment = _option_alignment(recent_actions, action, option)
            if alignment is None:
                continue
            weighted_entries.append(
                {
                    "weight": alignment["weight"],
                    "schema_bonus": max(option.value, 0.0),
                    "relative_cost": max(option.relative_cost * alignment["remaining_fraction"], 1.0),
                    "efficiency": option.efficiency,
                    "support": option.support,
                    "contradiction": option.contradiction,
                    "continuation_depth": alignment["depth"],
                }
            )

        state_tokens = state_schema_tokens(state)
        for schema in self.retrieve_schemas(state, language_tokens, k=12):
            alignment = _schema_alignment(recent_families, action, schema)
            if alignment is None:
                continue
            relevance = 0.60 + (0.40 * jaccard(state_tokens, schema.precondition_tokens))
            weighted_entries.append(
                {
                    "weight": alignment["weight"] * relevance,
                    "schema_bonus": max(schema.value, 0.0),
                    "relative_cost": max(schema.avg_relative_cost * alignment["remaining_fraction"], 1.0),
                    "efficiency": schema.efficiency,
                    "support": schema.support,
                    "contradiction": schema.contradiction,
                    "continuation_depth": alignment["depth"],
                }
            )

        if not weighted_entries:
            return {
                "schema_bonus": 0.0,
                "relative_cost": 1.0,
                "efficiency": 0.0,
                "support": 0.0,
                "contradiction": 0.0,
                "continuation_depth": 0.0,
            }
        total_weight = sum(max(entry["weight"], 1e-6) for entry in weighted_entries)
        best = max(weighted_entries, key=lambda entry: (entry["weight"] * entry["schema_bonus"], entry["continuation_depth"]))
        weighted = lambda key, default=0.0: float(
            sum(entry["weight"] * entry[key] for entry in weighted_entries) / max(total_weight, 1e-6)
        )
        base_bonus = float(max(best["schema_bonus"], 0.0))
        continuation_depth = weighted("continuation_depth")
        continuation_bonus = 0.35 * continuation_depth * min(base_bonus, 1.0)
        return {
            "schema_bonus": float(base_bonus + continuation_bonus),
            "relative_cost": weighted("relative_cost", default=1.0),
            "efficiency": weighted("efficiency"),
            "support": weighted("support"),
            "contradiction": weighted("contradiction"),
            "continuation_depth": continuation_depth,
        }

    def _update_schema(
        self,
        *,
        family_sequence: tuple[str, ...],
        precondition_tokens: frozenset[str],
        effect_tags: frozenset[str],
        relative_cost: float,
        effect_value: float,
        reward: float,
        success: bool,
    ) -> None:
        schema_key = _schema_key(family_sequence, effect_tags)
        schema = self.schemas.get(schema_key)
        if schema is None:
            schema = EffectSchema(
                schema_key=schema_key,
                family_sequence=family_sequence,
                effect_tags=effect_tags,
                avg_relative_cost=float(max(relative_cost, 1.0)),
                avg_effect_value=float(effect_value),
                avg_reward=float(reward),
                uses=0,
                successes=0,
                support=0.0,
                contradiction=0.0,
            )
            self.schemas[schema_key] = schema
        schema.update(
            precondition_tokens=precondition_tokens,
            relative_cost=relative_cost,
            effect_value=effect_value,
            reward=reward,
            success=success,
        )
        if len(self.schemas) > self.capacity // 3:
            keep = sorted(
                self.schemas.values(),
                key=lambda item: (item.value, item.confidence, item.support - item.contradiction),
                reverse=True,
            )[: self.capacity // 3]
            self.schemas = {item.schema_key: item for item in keep}

    def _recent_option_sequence(self) -> tuple[ActionName, ...]:
        start = max(0, len(self.recent_actions) - 6)
        if len(self.recent_actions) <= 1:
            return tuple(self.recent_actions[-6:])
        for index in range(len(self.recent_actions) - 2, start - 1, -1):
            if self.recent_salient_flags[index]:
                start = index + 1
                break
        return tuple(self.recent_actions[start:])

    def _find_option(self, action_sequence: tuple[ActionName, ...], effect_tags: frozenset[str]) -> OptionItem | None:
        for option in reversed(self.options):
            if option.action_sequence == action_sequence and option.effect_tags == effect_tags:
                return option
        return None

    def _apply_option_contradiction(
        self,
        *,
        action_sequence: tuple[ActionName, ...],
        family_sequence: tuple[str, ...],
        effect_tags: frozenset[str],
    ) -> None:
        option = self._find_option(action_sequence, effect_tags)
        if option is not None:
            option.contradiction += 1.0
        schema = self.schemas.get(_schema_key(family_sequence, effect_tags))
        if schema is not None:
            schema.contradiction += 1.0


def _transition_note(record: TransitionRecord) -> str:
    parts = [f"action={record.action}"]
    if record.delta.moved_objects:
        parts.append(f"moved={len(record.delta.moved_objects)}")
    if record.delta.changed_cells:
        parts.append(f"changed={record.delta.changed_cells}")
    if record.reward:
        parts.append(f"reward={record.reward:.3f}")
    if record.delta.score_delta:
        parts.append(f"score_delta={record.delta.score_delta:.3f}")
    return ";".join(parts)


def _effect_signature_tags(record: TransitionRecord) -> frozenset[str]:
    tags: set[str] = set()
    if record.delta.disappeared:
        tags.add("effect:disappear")
    if any(motion.distance > 1.25 for motion in record.delta.moved_objects):
        tags.add("effect:large_motion")
    if tuple(record.before.available_actions) != tuple(record.after.available_actions):
        tags.add("effect:action_regime_change")
    if record.delta.changed_fraction >= 0.10:
        tags.add("effect:visible_change")
    if record.delta.is_positive:
        tags.add("effect:progress")
    if record.delta.touched_objects:
        tags.add("effect:contact")
    tags.update(numeric_delta_tags(record))
    return frozenset(tags)


def _mechanic_option_value(record: TransitionRecord, *, effect_tags: frozenset[str]) -> float:
    action_regime_changed = tuple(record.before.available_actions) != tuple(record.after.available_actions)
    structural = bool(
        record.delta.is_positive
        or record.delta.disappeared
        or action_regime_changed
        or "effect:numeric_increase" in effect_tags
        or "effect:numeric_rewrite" in effect_tags
    )
    pure_visible_nonprogress = bool(record.delta.has_visible_effect and not structural and not record.delta.is_positive)
    if pure_visible_nonprogress:
        return 0.0

    value = 0.0
    if "effect:disappear" in effect_tags:
        value += 0.18
    if "effect:large_motion" in effect_tags:
        value += 0.24
    if "effect:action_regime_change" in effect_tags:
        value += 0.16
    if "effect:visible_change" in effect_tags and not record.delta.is_positive:
        value += 0.08
    if "effect:progress" in effect_tags:
        value += 0.12
    if "effect:numeric_increase" in effect_tags:
        value += 0.10
    if "effect:numeric_rewrite" in effect_tags:
        value += 0.08
    return float(min(value, 0.45))


def _relative_option_cost(action_sequence: tuple[ActionName, ...], record: TransitionRecord) -> float:
    cost = float(max(len(action_sequence), 1))
    action_regime_changed = tuple(record.before.available_actions) != tuple(record.after.available_actions)
    structural = bool(record.delta.is_positive or record.delta.disappeared or action_regime_changed)
    if record.delta.has_visible_effect and not structural:
        cost += 0.55
    if not record.delta.has_visible_effect and not record.delta.is_positive:
        cost += 0.35
    if record.delta.terminated and not record.delta.is_positive:
        cost += 0.45
    numeric_spend = sum(max(-delta, 0.0) for delta in transition_numeric_deltas(record).values())
    cost += 0.15 * min(numeric_spend, 2.0)
    return float(min(cost, 8.0))


def _coerce_memory_vector(value: Any, dim: int) -> np.ndarray:
    try:
        array = np.asarray(value, dtype=np.float32).reshape(-1)
    except Exception:
        array = np.zeros(0, dtype=np.float32)
    if array.shape[0] == dim:
        return array.copy()
    output = np.zeros(dim, dtype=np.float32)
    count = min(dim, int(array.shape[0]))
    if count > 0:
        output[:count] = array[:count]
    return output


def _schema_key(family_sequence: tuple[str, ...], effect_tags: frozenset[str]) -> str:
    sequence = "/".join(family_sequence) if family_sequence else "none"
    effects = "/".join(sorted(effect_tags)) if effect_tags else "none"
    return f"{sequence}|{effects}"


def _option_alignment(
    recent_actions: tuple[ActionName, ...],
    action: ActionName,
    option: OptionItem,
) -> Mapping[str, float] | None:
    recent = recent_actions[-len(option.action_sequence) :]
    recent_families = tuple(action_family(item) for item in recent)
    candidate_family = action_family(action)
    best: dict[str, float] | None = None
    for index in range(len(option.family_sequence)):
        prefix_len = index
        exact_prefix = prefix_len == 0 or tuple(recent[-prefix_len:]) == option.action_sequence[:prefix_len]
        family_prefix = prefix_len == 0 or recent_families[-prefix_len:] == option.family_sequence[:prefix_len]
        if not exact_prefix and not family_prefix:
            continue
        action_exact = index < len(option.action_sequence) and action == option.action_sequence[index]
        action_family_match = candidate_family == option.family_sequence[index]
        if not action_exact and not action_family_match:
            continue
        prefix_weight = 1.0 if exact_prefix else 0.72
        action_weight = 1.0 if action_exact else 0.68
        depth = float(index + 1) / max(float(len(option.family_sequence)), 1.0)
        weight = prefix_weight * action_weight * (0.70 + (0.30 * depth))
        remaining_fraction = float(len(option.family_sequence) - index) / max(float(len(option.family_sequence)), 1.0)
        candidate = {"weight": float(weight), "depth": float(depth), "remaining_fraction": float(remaining_fraction)}
        if best is None or candidate["weight"] > best["weight"]:
            best = candidate
    return best


def _schema_alignment(
    recent_families: tuple[str, ...],
    action: ActionName,
    schema: EffectSchema,
) -> Mapping[str, float] | None:
    candidate_family = action_family(action)
    best: dict[str, float] | None = None
    for index in range(len(schema.family_sequence)):
        prefix_len = index
        if prefix_len > len(recent_families):
            continue
        if prefix_len and recent_families[-prefix_len:] != schema.family_sequence[:prefix_len]:
            continue
        if candidate_family != schema.family_sequence[index]:
            continue
        depth = float(index + 1) / max(float(len(schema.family_sequence)), 1.0)
        weight = schema.confidence * (0.68 + (0.32 * depth))
        remaining_fraction = float(len(schema.family_sequence) - index) / max(float(len(schema.family_sequence)), 1.0)
        candidate = {"weight": float(weight), "depth": float(depth), "remaining_fraction": float(remaining_fraction)}
        if best is None or candidate["weight"] > best["weight"]:
            best = candidate
    return best
