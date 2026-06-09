from __future__ import annotations

import hashlib
import json
from pathlib import Path

from src.model import load_checkpoint


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest().upper()


def test_model_py_is_frozen_for_heldout_evaluation() -> None:
    manifest = json.loads(Path("frozen/manifest.json").read_text(encoding="utf-8"))
    assert sha256(Path("src/model.py")) == manifest["model_sha256"]


def test_frozen_checkpoint_hash_and_loadability() -> None:
    manifest = json.loads(Path("frozen/manifest.json").read_text(encoding="utf-8"))
    checkpoint = Path(manifest["checkpoint"])
    assert checkpoint.exists()
    assert checkpoint.stat().st_size == manifest["checkpoint_size_bytes"]
    assert sha256(checkpoint) == manifest["checkpoint_sha256"]
    model = load_checkpoint(checkpoint)
    assert sum(param.numel() for param in model.parameters()) == 39531
