from __future__ import annotations

from pathlib import Path

import torch

from src.explore_env import build_explorer_dataset
from src.explorer_train import train_explorer
from src.head_collapse import HeadCollapsedExplorer
from src.head_collapse_eval import anti_crutch_scan, evaluate_head_collapse
from src.heldout_causal import FROZEN_CHECKPOINT
from src.world_model import ExplorerCore, load_explorer_checkpoint


def test_unified_route_ignores_structured_probe_outputs() -> None:
    data = build_explorer_dataset("smoke", "heldout", record_count=96, checkpoint=FROZEN_CHECKPOINT)
    model = ExplorerCore()
    enabled = HeadCollapsedExplorer(model, disable_structured_heads=False, remove_probe_modules=False)
    disabled = HeadCollapsedExplorer(model, disable_structured_heads=True, remove_probe_modules=False)
    removed = HeadCollapsedExplorer(model, disable_structured_heads=True, remove_probe_modules=True)
    with torch.no_grad():
        a = enabled.forward(data.tensors)
        b = disabled.forward(data.tensors)
        c = removed.forward(data.tensors)
    assert a["trace"].old_head_logits_used_for_behavior is False
    assert b["trace"].old_head_logits_used_for_behavior is False
    assert c["trace"].old_head_logits_used_for_behavior is False
    for key in a["behavior"]:
        assert torch.equal(a["behavior"][key], b["behavior"][key])
        assert torch.equal(b["behavior"][key], c["behavior"][key])
    assert b["probes"] == {}
    assert c["probes"] == {}


def test_head_collapse_source_scan_blocks_probe_crutches() -> None:
    scan = anti_crutch_scan()
    assert scan["passes"] is True
    assert scan["findings"] == []


def test_head_collapse_smoke_train_and_eval_runs(tmp_path: Path) -> None:
    output = tmp_path / "explorer_smoke.pt"
    train_explorer("smoke", output=output, checkpoint=FROZEN_CHECKPOINT)
    assert output.exists()
    model = load_explorer_checkpoint(output)
    assert hasattr(model, "unified_policy")

    report = evaluate_head_collapse(
        FROZEN_CHECKPOINT,
        output,
        config_name="smoke",
        json_output=tmp_path / "head_collapse_report.json",
        include_prior=False,
    )
    assert report["anti_crutch_audit"]["passes"] is True
    assert report["anti_crutch_audit"]["runtime_trace"]["old_head_logits_used_for_behavior"] is False
    assert report["probe_only_proof"]["disable_delta"] == 0.0
    assert report["probe_only_proof"]["remove_delta"] == 0.0
    assert Path(tmp_path / "head_collapse_report.json").exists()
