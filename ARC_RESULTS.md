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
| `attempt_memory_no_jepa` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8327157421919169` |
| `jepa_random_init` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.834521182315609` |
| `jepa_plus_attempt_memory` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8345795011033633` |
| `jepa_pretrained_frozen_if_available` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `jepa_trained_dev` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `null_control` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |

JEPA gates:

- `official_score_gain`: `0.0`
- `official_useful_event_gain`: `0.0`
- `attempt_2_or_3_improves_over_attempt_1`: `false`
- `score_or_useful_gain_over_core`: `false`
- `repeat_collapse_drop_attempt_1_to_3`: `0.007177159865328631`
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
| `2` | `0.0` | `0.0` | `0.7950725948141715` | `1.0749442526300725` |
| `3` | `0.0` | `0.0` | `0.8507443743152949` | `0.6758009712882054` |

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
- Focused JEPA tests: `20 passed in 2.55s`.
- Full tests before official rerun: `101 passed in 54.51s`.
- Runtime credential persistence scan: generic phrase scan found no matches.
- Compact trace schema check: preserved required step fields, chain planner marker, chain summary keys, and component state signatures.
- Generalization audit: `EXTERNAL GENERALIZATION AUDIT PROVEN`.
- Leakage scan: `passes=true`, no findings.
- Full tests after compaction/audits: `101 passed in 54.93s`.
- Official evaluation completed successfully in about 33 minutes. Component extraction is cached during scoring, but the full external evaluation matrix remains slow.

## Artifact Hashes

- `src/attempt_buffer.py`: `33E69313AC81DDFEF6826EEB6E4708BFD3DA95D1AB0D54AFE75FB12ADF564157`
- `src/jepa_attempt_memory.py`: `EE5ABD36E57738C797FE9F77A01C177A87357B4104DDF8C28001DD4F164F92FF`
- `src/jepa_arc_eval.py`: `0BBA29B725955CAEE6BFAA1EAC3E9D3573791DAB7713B5606AD52CD2A577430B`
- `tests/test_video_jepa.py`: `B9CE28B6C36B3FB74716E95D5DF333B7BAA88F2A4E96C23A5526D4F5D0F8FB9B`
- `docs/jepa_attempt_report.json`: `22B79FF6224999E36E48BA40A0947FB53A42D38723777F7F1933A6439E3A8D3F`
- `docs/generalization_audit_after_jepa.json`: `070C675AF0B980340031E5A34614FD4815068862AAFB0D064CDD5E295DE410A2`
- `docs/jepa_attempt_traces/_official_jepa_worker_report.json`: `1342BF494C6E48805112F1CB21F16393C67C4435B3D236CB0F9C55EB02E2CEAB`

## Trace State

The latest JEPA official/external run regenerated 652 trace/report JSON files under `docs/jepa_attempt_traces`. The 651 trace files were compacted after the run from `2357075018` bytes to `257782792` bytes while preserving the audit-required attempt timeline fields, `next_frame`, summaries, compact transition-graph policy diagnostics, object causal hypotheses, component causal hypotheses, component-transition prediction summaries, component transition-goal chain summaries, object/component memory summaries, and sequence plan summaries. The reduction ratio was `0.8906344558270652`. Total trace tree size after compaction, including the worker report, is `276711501` bytes.
