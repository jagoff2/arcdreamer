# Methods Attempted

Last updated: 2026-06-12.

## Strict Attempt Outcome Classification

- Rationale: JEPA attempt memory counted any non-positive transition as failed, including movement or other actions that visibly changed public state but had only the official step cost.
- Code touched: `src/jepa_attempt_memory.py`, `tests/test_video_jepa.py`.
- Mechanism: separate strict no-effect or blocked actions from visible-effect actions; add visible-effect rate and repeated-action fraction to causal hypotheses; add small controllability bias for visible-effect actions; penalize repeated no-progress actions from attempt evidence.
- Result: focused tests passed, full tests passed, official runtime still reported `NO IMPROVEMENT FOUND`.
- Status: retained as a correctness fix, not a performance success.

## JEPA Causal Attempt Substrate

- Rationale: make full attempt video/action history causally affect next-attempt behavior instead of being a sidecar diagnostic.
- Code touched: `src/attempt_buffer.py`, `src/video_jepa.py`, `src/jepa_train.py`, `src/jepa_attempt_memory.py`, `src/jepa_arc_eval.py`, `tests/test_video_jepa.py`.
- Result: causal substrate checks pass; JEPA changes next-attempt action distributions; official public performance did not improve.
- Status: retained as architecture progress, insufficient for score.

## External Base Retraining

- Rationale: test whether the old toy-trained base was the bottleneck by training action-conditioned world-model/affordance arms on unlabeled external interaction traces.
- Evidence: `docs/external_base_report.json`.
- Result: terminal outcome `NO IMPROVEMENT FOUND`; selected `old_base_finetuned`; official score gain over old base `-0.004`, useful-event gain `-0.04`.
- Status: negative.

## ARC Affordance Baseline

- Rationale: test non-neural public-observation affordance discovery, component click search, state graph ranking, and no-op avoidance.
- Evidence: `docs/arc_affordance_report.json`.
- Result: terminal outcome `NO SIGNAL FOUND`; selected `state_graph_affordance`; official mean normalized score `0.005`, useful events `0.04`, invalid action rate `0.0`, repeat collapse `0.22784841859240573`; gate failed.
- Status: current best retained official score, but far below target.

## Perceptual Affordance Experiment

- Rationale: add component extraction, temporal slots, action-effect memory, and predictive object model.
- Evidence: `docs/perceptual_affordance_report.json`.
- Result: terminal outcome `NO IMPROVEMENT FOUND`; selected `full_perceptual_affordance`; official score and useful events regressed versus best existing baseline.
- Status: negative.

## External Collapse / Valence Patch

- Rationale: reduce repeat collapse through valence and loop aversion without forced schedules.
- Evidence: `docs/external_collapse_report.json`.
- Result: terminal outcome `NO MINIMAL IMPROVEMENT FOUND`; selected `valence_only`; repeat collapse dropped by about `0.2899`, but official score gain `0.005` was below threshold.
- Status: partial behavioral signal, insufficient.
