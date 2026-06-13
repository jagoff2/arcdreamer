from __future__ import annotations

import importlib
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT / "src"
RUNS_DIR = ROOT / "runs"

REQUIRED_RUNTIME_MODULES = ("env", "model", "train", "evaluate", "metrics", "run_unbroken")


def _snapshot_runs_tree() -> list[str]:
    if not RUNS_DIR.exists():
        return []
    return sorted(str(path.relative_to(RUNS_DIR)) for path in RUNS_DIR.rglob("*"))


def _is_under_src(module_file: str) -> bool:
    candidate = Path(module_file).resolve()
    return candidate.is_relative_to(SRC_DIR.resolve())


def test_src_exposes_required_runtime_modules() -> None:
    import src

    assert hasattr(src, "REQUIRED_RUNTIME_MODULES"), "src.REQUIRED_RUNTIME_MODULES is missing"
    runtime_modules = src.REQUIRED_RUNTIME_MODULES
    assert isinstance(runtime_modules, (tuple, list, set, frozenset))

    for name in REQUIRED_RUNTIME_MODULES:
        assert name in runtime_modules


def test_required_runtime_modules_importable_and_located_under_src() -> None:
    for module_name in REQUIRED_RUNTIME_MODULES:
        module_full_name = f"src.{module_name}"
        runs_before = _snapshot_runs_tree()

        # Force module execution so import-time side effects are observed here.
        sys.modules.pop(module_full_name, None)
        module = importlib.import_module(module_full_name)

        assert module.__file__ is not None
        assert _is_under_src(module.__file__), module.__file__
        assert module.__file__.startswith(str(SRC_DIR))

        runs_after = _snapshot_runs_tree()
        assert runs_after == runs_before


def test_required_runtime_modules_export_expected_symbols() -> None:
    module_symbol_pairs = [
        ("env", "TinyWorldRuntime"),
        ("model", "RecurrentLatentModel"),
        ("train", "train_model"),
        ("evaluate", "evaluate_checkpoint"),
        ("metrics", "pass_fail"),
        ("run_unbroken", "run_unbroken"),
    ]

    for module_name, symbol_name in module_symbol_pairs:
        module_full_name = f"src.{module_name}"
        sys.modules.pop(module_full_name, None)
        module = importlib.import_module(module_full_name)
        assert hasattr(module, symbol_name), f"{module_full_name} missing {symbol_name}"
