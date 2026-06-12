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
| `attempt_memory_no_jepa` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8285113405201712` |
| `jepa_random_init` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8272583052000829` |
| `jepa_plus_attempt_memory` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8283416385334161` |
| `jepa_pretrained_frozen_if_available` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `jepa_trained_dev` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `null_control` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |

JEPA gates:

- `official_score_gain`: `0.0`
- `official_useful_event_gain`: `0.0`
- `attempt_2_or_3_improves_over_attempt_1`: `false`
- `score_or_useful_gain_over_core`: `false`
- `repeat_collapse_drop_attempt_1_to_3`: `0.0013723367224568461`
- `repeat_collapse_drop_gate`: `false`
- `ablation_removes_improvement`: `false`
- `jepa_beats_null_on_dev`: `true`
- `jepa_causal_substrate_chain`: `true`
- `non_arc_drop_within_limit`: `true`
- `hidden_target_canary_diff_zero`: `true`
- `jepa_emits_no_text`: `true`
- `no_hack_passes`: `true`

Primary attempt table:

| Attempt | Mean normalized score | Useful events | Repeat collapse | Action entropy |
| ---: | ---: | ---: | ---: | ---: |
| `1` | `0.004` | `0.04` | `0.8579215341806236` | `0.7487791837206761` |
| `2` | `0.0` | `0.0` | `0.7705541839614582` | `1.146861043117688` |
| `3` | `0.0` | `0.0` | `0.8565491974581667` | `0.6238160988414669` |

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
- Focused JEPA tests: `18 passed in 2.57s`.
- Full tests before official rerun: `99 passed in 52.53s`.
- Runtime credential persistence scan: generic phrase scan found no matches.
- Generalization audit: `EXTERNAL GENERALIZATION AUDIT PROVEN`.
- Leakage scan: `passes=true`, no findings.
- Full tests after compaction/audits: `99 passed in 54.98s`.
- Official evaluation completed successfully in about 33 minutes. Component extraction is cached during scoring, but the full external evaluation matrix remains slow.

## Artifact Hashes

- `src/attempt_buffer.py`: `33E69313AC81DDFEF6826EEB6E4708BFD3DA95D1AB0D54AFE75FB12ADF564157`
- `src/jepa_attempt_memory.py`: `9FFE00781F0FA592394D510F6F722082CD34AE6686A256A479B22C7CA11D5F6A`
- `src/jepa_arc_eval.py`: `0BBA29B725955CAEE6BFAA1EAC3E9D3573791DAB7713B5606AD52CD2A577430B`
- `tests/test_video_jepa.py`: `A9788C91EA1ACEA3690EA98DF6F7E937DC93BC4473C5AD5D18D962D66D112193`
- `docs/jepa_attempt_report.json`: `D129BACCE1322729E51675300E32860FEF87125F3B0EEEFA14DFF4D12BF99E87`
- `docs/generalization_audit_after_jepa.json`: `7F1F124BD5F6DED5596CED4C0C54C90EEEA1440EAB6E01930E00DB5FC3F5C916`
- `docs/jepa_attempt_traces/_official_jepa_worker_report.json`: `51C8C7637893710D28A16E91F7FD7C8DDF7DEE128A70C79DCADEF655924D0339`

## Trace State

The latest JEPA official/external run regenerated 652 trace/report JSON files under `docs/jepa_attempt_traces`. The 651 trace files were compacted after the run from `2372838066` bytes to `247849378` bytes while preserving the audit-required attempt timeline fields, `next_frame`, summaries, compact transition-graph policy diagnostics, object causal hypotheses, component causal hypotheses, component-transition prediction summaries, object/component memory summaries, and sequence plan summaries. The reduction ratio was `0.895547285105`.
