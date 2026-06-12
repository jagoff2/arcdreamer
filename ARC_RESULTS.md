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
| `attempt_memory_no_jepa` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8448250960415711` |
| `jepa_random_init` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8516683631497388` |
| `jepa_plus_attempt_memory` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8607239187052945` |
| `jepa_pretrained_frozen_if_available` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `jepa_trained_dev` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `null_control` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |

JEPA gates:

- `official_score_gain`: `0.0`
- `official_useful_event_gain`: `0.0`
- `attempt_2_or_3_improves_over_attempt_1`: `false`
- `repeat_collapse_drop_attempt_1_to_3`: `-0.051449664832514785`
- `repeat_collapse_drop_gate`: `false`
- `jepa_causal_substrate_chain`: `true`
- `no_hack_passes`: `true`

Primary attempt table:

| Attempt | Mean normalized score | Useful events | Repeat collapse | Action entropy |
| ---: | ---: | ---: | ---: | ---: |
| `1` | `0.004` | `0.04` | `0.8579215341806236` | `0.7487791837206761` |
| `2` | `0.0` | `0.0` | `0.8148790229221212` | `0.8774161563528656` |
| `3` | `0.0` | `0.0` | `0.9093711990131383` | `0.43308076426099906` |

Attempt 2 reduced repeat collapse versus attempt 1, but no later attempt improved score or useful events.

## Verification Commands

```bash
python -m py_compile src\attempt_buffer.py src\jepa_attempt_memory.py src\jepa_arc_eval.py
pytest -q tests\test_video_jepa.py
pytest -q
python -m src.jepa_arc_eval --config external --checkpoint frozen/recurrent_latent_fast.pt --explorer-checkpoint runs/explorer_tiny.pt --jepa-checkpoint runs/video_jepa.pt --json-output docs/jepa_attempt_report.json --trace-dir docs/jepa_attempt_traces --device cuda
python -m src.generalization_audit --json-output docs/generalization_audit_after_jepa.json
python -m audit.leakage_scan
pytest -q
```

Verification results:

- Syntax check: passed.
- Focused JEPA tests: `13 passed in 2.57s`.
- Full tests before official rerun: `94 passed in 54.08s`.
- Runtime credential persistence scan: generic phrase scan found no matches; one-off exact-token scan returned `hits=0` without printing the token value.
- Generalization audit: `EXTERNAL GENERALIZATION AUDIT PROVEN`.
- Leakage scan: `passes=true`, no findings.
- Full tests after compaction/audits: `94 passed in 55.26s`.

## Artifact Hashes

- `src/attempt_buffer.py`: `33E69313AC81DDFEF6826EEB6E4708BFD3DA95D1AB0D54AFE75FB12ADF564157`
- `src/jepa_attempt_memory.py`: `D9E98898FA50E314D4C7AED82476189DE278C7B68449E7C7281D8AC2C4392CE5`
- `src/jepa_arc_eval.py`: `7A9E80E6E427E9A64DA77936D4949FBBAAB03B4516BD466F7B333E1D2B233705`
- `tests/test_video_jepa.py`: `B32C15E9084317F059A0DB5199CE10B0854EFD1C509389B93D1F1B851BB51AE2`
- `docs/jepa_attempt_report.json`: `E37177906C3A84C10C547B407D0D966D9F62D094BED22FED299BBF297E160F64`
- `docs/generalization_audit_after_jepa.json`: `52023C63F540E0F69EE3F7D97266F08D39398860676F47CB3DE3AAC7F2F65949`
- `docs/jepa_attempt_traces/_official_jepa_worker_report.json`: `5B3C8A13373FB3B03A7CEAD44DC9A3B7E72D86381059B4BB8821B5E66C758EFF`

## Trace State

The latest JEPA official/external run regenerated 652 trace/report JSON files under `docs/jepa_attempt_traces`. The 651 trace files were compacted after the run from `2319825117` bytes to `246863146` bytes while preserving the audit-required attempt timeline fields, `next_frame`, summaries, compact transition-graph policy diagnostics, object causal hypotheses, object memory summary, and sequence plan summaries. The reduction ratio was `0.893585449959`.
