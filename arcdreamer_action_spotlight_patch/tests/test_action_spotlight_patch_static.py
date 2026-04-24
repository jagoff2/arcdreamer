"""Static checks for the action spotlight patch.

These tests are deliberately light because they should run even before the full
ARC toolkit is installed.  The repository's normal tests should still be run
after the patch is applied.
"""

from pathlib import Path


def test_patch_files_exist() -> None:
    root = Path(__file__).resolve().parents[1]
    assert (root / "arcagi" / "scientist" / "spotlight.py").exists()
    assert (root / "arcagi" / "agents" / "spotlight_scientist_agent.py").exists()
    assert (root / "arcagi" / "agents" / "scientist_agent.py").exists()


def test_spotlight_source_mentions_binder_probe() -> None:
    root = Path(__file__).resolve().parents[1]
    source = (root / "arcagi" / "scientist" / "spotlight.py").read_text(encoding="utf-8")
    assert "PendingBinderProbe" in source
    assert "probe_after_latent_binding" in source
    assert "ACTION6" in source
