from __future__ import annotations

import importlib.util
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExternalSuite:
    suite_id: str
    name: str
    source: str
    tasks: tuple[str, ...]
    available: bool
    generated_by_repo: bool
    split: str
    interpreter: str
    reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        row = asdict(self)
        row["tasks"] = list(self.tasks)
        return row


@dataclass(frozen=True)
class ClaimSpec:
    claim_id: str
    mechanism: str
    external_behavioral_prediction: str
    external_metric: str
    baseline_to_beat: str
    ablation_expected_to_hurt: str

    def to_dict(self) -> dict[str, str]:
        return asdict(self)


CLAIMS: tuple[ClaimSpec, ...] = (
    ClaimSpec(
        "memory",
        "persistent latent/private-token memory",
        "Delayed or sparse external reward cues should improve held-out external return and corrupt-memory ablations should hurt.",
        "sealed_eval.mean_normalized_score",
        "best external baseline",
        "corrupt_memory",
    ),
    ClaimSpec(
        "exploration",
        "unified policy over public observation, z, memory, and affordance channels",
        "Explorer should beat simple legal-action baselines on external interactive tasks.",
        "sealed_eval.mean_normalized_score",
        "best external baseline",
        "zero_z",
    ),
    ClaimSpec(
        "dialogue",
        "grounded dialogue organ and language projections",
        "Removing dialogue/private-language paths should reduce external task performance if language is externally useful.",
        "sealed_eval.mean_normalized_score",
        "best external baseline",
        "no_dialogue",
    ),
    ClaimSpec(
        "head_collapse",
        "behavior routed through unified affordance policy rather than old probe heads",
        "Unified head-collapsed policy should preserve or improve external task performance over simple baselines.",
        "sealed_eval.mean_normalized_score",
        "best external baseline",
        "disable_planner_imagination",
    ),
    ClaimSpec(
        "planner",
        "hypothesis/project tensors and action-sequence selection",
        "Disabling planner/imagination should lower external score or event progress.",
        "sealed_eval.mean_normalized_score",
        "best external baseline",
        "disable_planner_imagination",
    ),
    ClaimSpec(
        "curiosity",
        "intrinsic drive and novelty channels",
        "Corrupting drive should reduce external exploration progress.",
        "sealed_eval.mean_unique_states",
        "best external baseline",
        "corrupt_drive",
    ),
    ClaimSpec(
        "social_state",
        "partner/social-state tensor channels",
        "Social-state perturbation should affect external multi-agent or social tasks.",
        "sealed_eval.mean_normalized_score",
        "best external baseline",
        "no_social_state",
    ),
)


def module_available(name: str) -> bool:
    return importlib.util.find_spec(name) is not None


def discover_external_suites() -> list[ExternalSuite]:
    venv_python = Path(".venv/Scripts/python.exe")
    arc_available = bool(venv_python.exists()) and Path("docs/arcagi3_official_games.json").exists()
    gymnasium_available = module_available("gymnasium")
    return [
        ExternalSuite(
            suite_id="official_arcagi3",
            name="Official ARC-AGI-3 public games",
            source="arc-agi runtime and cached official reports",
            tasks=tuple(game_ids_from_manifest()),
            available=arc_available,
            generated_by_repo=False,
            split="sealed_eval",
            interpreter=str(venv_python) if venv_python.exists() else "unavailable",
            reason="" if arc_available else "requires .venv ARC runtime and docs/arcagi3_official_games.json",
        ),
        ExternalSuite(
            suite_id="gymnasium_classic_control",
            name="Gymnasium Classic Control",
            source="gymnasium package",
            tasks=("CartPole-v1",),
            available=gymnasium_available,
            generated_by_repo=False,
            split="dev_and_sealed_eval",
            interpreter="default_python",
            reason="" if gymnasium_available else "gymnasium is not installed",
        ),
        ExternalSuite(
            suite_id="gymnasium_toy_text",
            name="Gymnasium ToyText",
            source="gymnasium package",
            tasks=("FrozenLake-v1",),
            available=gymnasium_available,
            generated_by_repo=False,
            split="dev_and_sealed_eval",
            interpreter="default_python",
            reason="" if gymnasium_available else "gymnasium is not installed",
        ),
    ]


def game_ids_from_manifest(path: str | Path = "docs/arcagi3_official_games.json") -> list[str]:
    item = Path(path)
    if not item.exists():
        return []
    import json

    payload = json.loads(item.read_text(encoding="utf-8"))
    return [str(game["game_id"]) for game in payload.get("games", [])]


def claim_registry_template() -> list[dict[str, Any]]:
    return [
        {
            **claim.to_dict(),
            "result": "not evaluated",
            "status": "not tested",
        }
        for claim in CLAIMS
    ]

