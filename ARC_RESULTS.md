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
| `attempt_memory_no_jepa` | `0/25` | `0.002666666666666667` | `0.02666666666666667` | `0.0` | `0.8454468253707098` |
| `jepa_random_init` | `0/25` | `0.004000000000000001` | `0.04` | `0.0` | `0.8463359286668403` |
| `jepa_plus_attempt_memory` | `0/25` | `0.004000000000000001` | `0.04` | `0.0` | `0.8474902052213587` |
| `jepa_pretrained_frozen_if_available` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `jepa_trained_dev` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `null_control` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |

JEPA gates:

- `official_score_gain`: `0.004`
- `official_useful_event_gain`: `0.04`
- `attempt_2_or_3_improves_over_attempt_1`: `false`
- `score_or_useful_gain_over_core`: `false`
- `repeat_collapse_drop_attempt_1_to_3`: `0.002394162555590662`
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
| `2` | `0.004` | `0.04` | `0.8290217098584192` | `0.9094204107398244` |
| `3` | `0.004` | `0.04` | `0.8555273716250329` | `0.7028048978667634` |

Attempt 2 reduced repeat collapse versus attempt 1, but no later attempt improved score or useful events over attempt 1. Attempt 3 stayed slightly below attempt 1 on repeat collapse, but the drop did not pass the gate.

## Verification Commands

```bash
python -m py_compile src\jepa_attempt_memory.py tests\test_video_jepa.py
pytest -q tests\test_video_jepa.py
pytest -q
python -m src.jepa_arc_eval --config external --checkpoint frozen/recurrent_latent_fast.pt --explorer-checkpoint runs/explorer_tiny.pt --jepa-checkpoint runs/video_jepa.pt --json-output docs/jepa_attempt_report.json --trace-dir docs/jepa_attempt_traces --device cuda
python -m src.generalization_audit --json-output docs/generalization_audit_after_jepa.json
python -m audit.leakage_scan
pytest -q
```

Verification results:

- Syntax check: passed.
- Focused JEPA tests: `26 passed in 2.63s`.
- Full tests before official rerun: `107 passed in 52.10s`.
- Runtime credential persistence scan: generic phrase scan found no matches.
- Compact trace schema check: preserved required step fields, transition-graph diagnostics, relation-chain planner marker, relation-delta planner marker, `relation_delta_sequence_planner`, relation-delta sequence candidate count, object memory summary keys, component relation signatures/action templates, and component relation-delta tokens.
- Generalization audit: `EXTERNAL GENERALIZATION AUDIT PROVEN`.
- Leakage scan: `passes=true`, no findings.
- Full tests after compaction/audits: `107 passed in 55.02s`.
- Official evaluation completed successfully. Component extraction is cached during scoring, but the full external evaluation matrix remains slow.

## Artifact Hashes

- `src/jepa_attempt_memory.py`: `B0EB12207B87015AFFE0A39C78F61BF7C4B29F01E442ABCF032507AE286872B5`
- `src/jepa_arc_eval.py`: `0BBA29B725955CAEE6BFAA1EAC3E9D3573791DAB7713B5606AD52CD2A577430B`
- `tests/test_video_jepa.py`: `43591E8FEFDB0AB9CF3010A63496D0EDFF78E1C9C66B71871F31F2078F4C0132`
- `docs/jepa_attempt_report.json`: `B396646147E4F6A3D73DFC597346382ABB27AC68A312CF7D1334F973380DD589`
- `docs/generalization_audit_after_jepa.json`: `4DA1A1E833C2DE3A180B9D3FE1340E2B8386FFCA6C52AB2D5AA870C5BB38AE13`
- `docs/jepa_attempt_traces/_official_jepa_worker_report.json`: `ED94BF91BB910A9D5D25D6E59AA9D6694019305FB7027CE7B5A3677ABD3C9872`

## Trace State

The latest JEPA official/external run regenerated 652 trace/report JSON files under `docs/jepa_attempt_traces`. The 651 non-worker trace files were compacted after the run from `2416415531` bytes to `235464399` bytes while preserving the audit-required attempt timeline fields, `next_frame`, summaries, compact transition-graph policy diagnostics, object causal hypotheses, component causal hypotheses, component-transition prediction summaries, exact and relation-level component transition-goal chain summaries, relation-delta mechanism summaries/tokens, relation-delta sequence planner markers, object/component memory summaries, and sequence plan summaries. The reduction ratio was `0.9025563294146862`. Total trace tree size after compaction, including the worker report, is `259118140` bytes.
