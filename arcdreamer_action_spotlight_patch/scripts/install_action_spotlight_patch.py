#!/usr/bin/env python3
"""Install the action-level spotlight patch into an arcdreamer checkout.

Run this from the repository root after unzipping the patch:

    python scripts/install_action_spotlight_patch.py

The file copy step is normally unnecessary if the ZIP was extracted at the repo
root, but the harness patch is useful because the existing harness returns the
base scientist when a checkpoint path exists.  This script updates that branch to
use the spotlight checkpoint loader as well.
"""

from __future__ import annotations

from pathlib import Path


OLD = (
    'if normalized in {"scientist", "hyper_scientist", "hyper-generalizing-scientist"}: '
    'from arcagi.scientist import load_scientist_checkpoint '
    'from arcagi.agents.scientist_agent import HyperGeneralizingScientistAgent '
    'if checkpoint_path and Path(checkpoint_path).exists(): '
    'return load_scientist_checkpoint(checkpoint_path) '
    'return HyperGeneralizingScientistAgent()'
)

NEW = (
    'if normalized in {"scientist", "hyper_scientist", "hyper-generalizing-scientist", "spotlight", "spotlight_scientist"}: '
    'from arcagi.agents.scientist_agent import HyperGeneralizingScientistAgent, load_spotlight_scientist_checkpoint '
    'if checkpoint_path and Path(checkpoint_path).exists(): '
    'return load_spotlight_scientist_checkpoint(checkpoint_path) '
    'return HyperGeneralizingScientistAgent()'
)


def main() -> int:
    root = Path.cwd()
    harness = root / "arcagi" / "evaluation" / "harness.py"
    if not harness.exists():
        print(f"ERROR: {harness} not found. Run from the arcdreamer repository root.")
        return 2
    text = harness.read_text(encoding="utf-8")
    if NEW in text:
        print("harness already patched for spotlight scientist")
        return 0
    if OLD not in text:
        print("WARNING: exact harness branch was not found; no automatic edit made.")
        print("Add 'spotlight' to build_agent and use load_spotlight_scientist_checkpoint manually.")
        return 1
    harness.write_text(text.replace(OLD, NEW), encoding="utf-8")
    print("patched arcagi/evaluation/harness.py for spotlight scientist checkpoint loading")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
