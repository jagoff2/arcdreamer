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
| `attempt_memory_no_jepa` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8280664986589359` |
| `jepa_random_init` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8466478084455438` |
| `jepa_plus_attempt_memory` | `0/25` | `0.0013333333333333335` | `0.013333333333333334` | `0.0` | `0.8470217966721809` |
| `jepa_pretrained_frozen_if_available` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `jepa_trained_dev` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |
| `null_control` | `0/25` | `0.0` | `0.0` | `0.0` | `0.833294177107705` |

JEPA gates:

- `official_score_gain`: `0.0`
- `official_useful_event_gain`: `0.0`
- `attempt_2_or_3_improves_over_attempt_1`: `false`
- `score_or_useful_gain_over_core`: `false`
- `repeat_collapse_drop_attempt_1_to_3`: `-0.02607127977033996`
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
| `2` | `0.0` | `0.0` | `0.7991510418849557` | `1.0152907719333566` |
| `3` | `0.0` | `0.0` | `0.8839928139509635` | `0.53837363482775` |

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
- Focused JEPA tests: `16 passed in 2.50s`.
- Full tests before official rerun: `97 passed in 54.66s`.
- Runtime credential persistence scan: generic phrase scan found no matches; one-off exact-token scan returned `hits=0` without printing the token value.
- Generalization audit: `EXTERNAL GENERALIZATION AUDIT PROVEN`.
- Leakage scan: `passes=true`, no findings.
- Full tests after compaction/audits: `97 passed in 54.84s`.
- Official evaluation completed successfully but was slow because component extraction currently runs inside legal-action scoring.

## Artifact Hashes

- `src/attempt_buffer.py`: `33E69313AC81DDFEF6826EEB6E4708BFD3DA95D1AB0D54AFE75FB12ADF564157`
- `src/jepa_attempt_memory.py`: `3321CE66FD323CBA16A53E225B3A02F8E185BE634220699EB752FE4F12046F80`
- `src/jepa_arc_eval.py`: `0BBA29B725955CAEE6BFAA1EAC3E9D3573791DAB7713B5606AD52CD2A577430B`
- `tests/test_video_jepa.py`: `8BB30E2A9AE2D98C74AB7850A2BD2192C10D613BF353E1B3D40A98B64F41544C`
- `docs/jepa_attempt_report.json`: `2D4A9DC42443F900581C8E0CE2B9C07F53BB11F830327EFFC9CC07BEBEA5ED57`
- `docs/generalization_audit_after_jepa.json`: `46ED10EE8D9F369FB12A27F61B1A2CD7B5557C4EBEC3259E8A295DC6ED58C928`
- `docs/jepa_attempt_traces/_official_jepa_worker_report.json`: `9F7EFADDEA9C59DE0FD2EE657759BE879684166BB94F563095E574636E45F222`

## Trace State

The latest JEPA official/external run regenerated 652 trace/report JSON files under `docs/jepa_attempt_traces`. The 651 trace files were compacted after the run from `2368977159` bytes to `264835587` bytes while preserving the audit-required attempt timeline fields, `next_frame`, summaries, compact transition-graph policy diagnostics, object causal hypotheses, component causal hypotheses, object/component memory summaries, and sequence plan summaries. The reduction ratio was `0.888206779034`.
