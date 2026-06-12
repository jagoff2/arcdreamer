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
| `attempt_memory_no_jepa` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.840588517496806` |
| `jepa_random_init` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8419128489822757` |
| `jepa_plus_attempt_memory` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8425362339951956` |
| `jepa_pretrained_frozen_if_available` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `jepa_trained_dev` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `null_control` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |

JEPA gates:

- `official_score_gain`: `0.0`
- `official_useful_event_gain`: `0.0`
- `attempt_2_or_3_improves_over_attempt_1`: `false`
- `score_or_useful_gain_over_core`: `false`
- `repeat_collapse_drop_attempt_1_to_3`: `-0.019193038810168206`
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
| `2` | `0.0` | `0.0` | `0.7925725948141715` | `1.0879785904096957` |
| `3` | `0.0` | `0.0` | `0.8771145729907918` | `0.5793500162176334` |

Attempt 2 reduced repeat collapse versus attempt 1, but no later attempt improved score or useful events. Attempt 3 regressed on repeat collapse.

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
- Focused JEPA tests: `22 passed in 2.58s`.
- Full tests before official rerun: `103 passed in 53.78s`.
- Runtime credential persistence scan: generic phrase scan found no matches.
- Compact trace schema check: preserved required step fields, transition-graph diagnostics, relation-chain planner marker, relation-chain summary keys, and component relation signatures/action templates.
- Generalization audit: `EXTERNAL GENERALIZATION AUDIT PROVEN`.
- Leakage scan: `passes=true`, no findings.
- Full tests after compaction/audits/docs: `103 passed in 53.52s`.
- Official evaluation completed successfully in about 33 minutes. Component extraction is cached during scoring, but the full external evaluation matrix remains slow.

## Artifact Hashes

- `src/attempt_buffer.py`: `33E69313AC81DDFEF6826EEB6E4708BFD3DA95D1AB0D54AFE75FB12ADF564157`
- `src/jepa_attempt_memory.py`: `59EF54E5907D53F7566C5278C1B1DA1C1FD7AD98E9EA56392683BA9056A95ECB`
- `src/jepa_arc_eval.py`: `0BBA29B725955CAEE6BFAA1EAC3E9D3573791DAB7713B5606AD52CD2A577430B`
- `tests/test_video_jepa.py`: `7A704B15C1A136920E75649D7615A20EAE532AE98AE0BA2639F209B871464EF3`
- `docs/jepa_attempt_report.json`: `80C167F43E083CC7AE8CD43699672E55310808384D7AA9E23C878A5C99C0AED8`
- `docs/generalization_audit_after_jepa.json`: `D4C8E8EF49FE88A5910D4D026ECA5CD8A08E9C7C6BB68EE935461304FD1F2E2B`
- `docs/jepa_attempt_traces/_official_jepa_worker_report.json`: `5CA75E7A13612C31D82A098D542452D115563C19FAA672C7C9140549C6286FCF`

## Trace State

The latest JEPA official/external run regenerated 652 trace/report JSON files under `docs/jepa_attempt_traces`. The 651 non-worker trace files were compacted after the run from `2362266983` bytes to `304635583` bytes while preserving the audit-required attempt timeline fields, `next_frame`, summaries, compact transition-graph policy diagnostics, object causal hypotheses, component causal hypotheses, component-transition prediction summaries, exact and relation-level component transition-goal chain summaries, object/component memory summaries, and sequence plan summaries. The reduction ratio was `0.8710410020576408`. Total trace tree size after compaction, including the worker report, is `325105710` bytes.
