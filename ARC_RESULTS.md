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
| `attempt_memory_no_jepa` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8867226507234964` |
| `jepa_random_init` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.888397678769673` |
| `jepa_plus_attempt_memory` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8752519080124866` |
| `jepa_pretrained_frozen_if_available` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `jepa_trained_dev` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `null_control` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |

JEPA gates:

- `official_score_gain`: `0.0`
- `official_useful_event_gain`: `0.0`
- `attempt_2_or_3_improves_over_attempt_1`: `false`
- `repeat_collapse_drop_attempt_1_to_3`: `-0.04420015602337091`
- `jepa_causal_substrate_chain`: `true`
- `no_hack_passes`: `true`

Primary attempt table:

| Attempt | Mean normalized score | Useful events | Repeat collapse |
| ---: | ---: | ---: | ---: |
| `1` | `0.004` | `0.04` | `0.8579215341806236` |
| `2` | `0.0` | `0.0` | `0.8657124996528418` |
| `3` | `0.0` | `0.0` | `0.9021216902039945` |

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

- Focused JEPA tests: `10 passed in 2.56s`.
- Full tests before official rerun: `91 passed in 53.92s`.
- Generalization audit: `EXTERNAL GENERALIZATION AUDIT PROVEN`.
- Leakage scan: `passes=true`.
- Full tests after compaction/audits: `91 passed in 51.92s`.

## Artifact Hashes

- `src/jepa_attempt_memory.py`: `EC48E8DE673EE50D106B917E11EA62818E48EC8A6BA5C423FF879CF01252C2B5`
- `src/jepa_arc_eval.py`: `4F1D63A355253DE7E6296CC4733B96B16808381F32F96C18CE46B9F62523A32B`
- `tests/test_video_jepa.py`: `FB674C11BA4461647BA4923B399EB67866B2968C3546E1C35A36B04A221A2328`
- `docs/jepa_attempt_report.json`: `B05A9483B157D2904C57DC070657EA4BD1ACBA91BDE4904A0251AB5B38525AC2`
- `docs/generalization_audit_after_jepa.json`: `A0B00B94D5E32649FD2CAAEB7252E14B83BE49DA02418FA76DB663467B66DA30`
- `docs/jepa_attempt_traces/_official_jepa_worker_report.json`: `9628FB2DA526552935210341BD59EAC0860BCF017A625B288E6E134BA4401D93`

## Trace State

The latest JEPA official/external run regenerated 652 trace/report JSON files under `docs/jepa_attempt_traces`. The 651 trace files were compacted after the run from `2242492282` bytes to `188625753` bytes while preserving the audit-required attempt timeline fields, summaries, and compact transition-graph policy diagnostics.
