# ARC Results

Last updated: 2026-06-12.

## Current Best Retained Official Public Score

| Evidence | Variant | Solved | Mean normalized score | Useful events | Invalid action rate | Repeat collapse | Outcome |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| `docs/arc_affordance_report.json` | `state_graph_affordance` | `0/25` | `0.005` | `0.04` | `0.0` | `0.22784841859240573` | `NO SIGNAL FOUND` |

This is the best retained official mean normalized score, but it is not a success claim. It solved no games and failed its improvement gate.

## Latest Official JEPA Run

Command:

```bash
python -m src.jepa_arc_eval --config external --checkpoint frozen/recurrent_latent_fast.pt --explorer-checkpoint runs/explorer_tiny.pt --jepa-checkpoint runs/video_jepa.pt --json-output docs/jepa_attempt_report.json --trace-dir docs/jepa_attempt_traces --device cuda
```

Result: `NO IMPROVEMENT FOUND`.

| Variant | Solved | Mean normalized score | Useful events | Invalid action rate | Repeat collapse |
| --- | ---: | ---: | ---: | ---: | ---: |
| `baseline_core` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `attempt_memory_no_jepa` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.853671079685245` |
| `jepa_random_init` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8612995528875755` |
| `jepa_plus_attempt_memory` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8753405853487634` |
| `jepa_pretrained_frozen_if_available` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `jepa_trained_dev` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `null_control` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |

JEPA gates:

- `official_score_gain`: `0.0`
- `official_useful_event_gain`: `0.0`
- `attempt_2_or_3_improves_over_attempt_1`: `false`
- `repeat_collapse_drop_attempt_1_to_3`: `-0.044466188032200926`
- `jepa_causal_substrate_chain`: `true`
- `no_hack_passes`: `true`

## Verification Commands

```bash
pytest -q tests\test_video_jepa.py
pytest -q
python -m src.jepa_arc_eval --config external --checkpoint frozen/recurrent_latent_fast.pt --explorer-checkpoint runs/explorer_tiny.pt --jepa-checkpoint runs/video_jepa.pt --json-output docs/jepa_attempt_report.json --trace-dir docs/jepa_attempt_traces --device cuda
python -m src.generalization_audit --json-output docs/generalization_audit_after_jepa.json
python -m audit.leakage_scan
pytest -q
```

Verification results:

- Focused JEPA tests: `8 passed in 2.65s`.
- Full tests before official rerun: `89 passed in 52.45s`.
- Generalization audit: `EXTERNAL GENERALIZATION AUDIT PROVEN`.
- Leakage scan: `passes=true`.
- Full tests after compaction/audits: `89 passed in 52.27s`.

## Artifact Hashes

- `src/jepa_attempt_memory.py`: `FFFFE092A5FEAF56303D17E75D5CCBE390CDB76DDF52EC4B052D2409B85AF1C4`
- `tests/test_video_jepa.py`: `978EF4F22C4A9C9010EFCF74BA1E747DBCF523925D9BCD4DD2459BDCD34C8385`
- `docs/jepa_attempt_report.json`: `3928452C28B72F1D9CFE59F94FB8B40F75C34BBA6097F827B0F2BA6202923654`
- `docs/generalization_audit_after_jepa.json`: `C0A8B4645603F9EF49CA1460A851663546A0A5CBF0D721EE0B1AD2B57EEED4A4`
- `docs/jepa_attempt_traces/_official_jepa_worker_report.json`: `89AE13B5CF85379FFC4E73E6BBC036160742CE4C1EA4627C6E4B3AD9371637AC`

## Trace State

The latest JEPA official/external run regenerated 652 trace/report JSON files under `docs/jepa_attempt_traces`. The 651 trace files were compacted after the run from `2205285267` bytes to `127131041` bytes while preserving the audit-required attempt timeline fields and summaries.
