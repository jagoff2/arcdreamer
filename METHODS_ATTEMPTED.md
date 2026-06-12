# Methods Attempted

Last updated: 2026-06-12.

## Region/Object Causal Hypotheses and Sequence Candidate Planner

- Rationale: transition-graph memory was public and causal, but too local. It scored individual state-action edges without identifying changed objects/regions or preserving a compact multi-step experiment across attempts.
- Code touched: `src/attempt_buffer.py`, `src/jepa_attempt_memory.py`, `src/jepa_arc_eval.py`, `tests/test_video_jepa.py`.
- Mechanism: store `next_frame` in attempt traces; infer public-frame movement, spawn, removal, toggle/transform, visual transform, and blocked/no-effect hypotheses; record changed regions/colors and delayed public-event region links; score legal actions by object-region contact and action-family evidence; save short prior-event action windows as sequence candidates and replay the best legal candidate at the next attempt start.
- Result: focused tests passed, full tests passed, generalization/leakage audits passed, official runtime still reported `NO IMPROVEMENT FOUND`. Primary `jepa_plus_attempt_memory` stayed at `0/25` solved, mean normalized score `0.0013333333333333335`, useful events `0.013333333333333334`, invalid action rate `0.0`, repeat collapse `0.8607239187052945`.
- Status: retained as a legal public-evidence substrate, not a performance success. Attempt 2 reduced repeat collapse versus attempt 1, but no later-attempt score/useful-event improvement occurred.

## Transition-Graph Next-Attempt Planner

- Rationale: current attempt memory changed action distributions but did not convert prior attempts into targeted next-attempt experiments over state transitions and delayed effects.
- Code touched: `src/jepa_attempt_memory.py`, `src/jepa_arc_eval.py`, `tests/test_video_jepa.py`.
- Mechanism: build a public `(observation_hash, action)` transition graph from attempt traces; record visible-effect, no-effect, positive-event, and delayed-event-proximity credits; score next actions from the current public observation without game IDs, hidden labels, source inspection, fixed action schedules, text advice, or external solvers.
- Result: focused tests passed, full tests passed, generalization/leakage audits passed, official runtime still reported `NO IMPROVEMENT FOUND`. Primary `jepa_plus_attempt_memory` stayed at `0/25` solved, mean normalized score `0.0013333333333333335`, useful events `0.013333333333333334`, invalid action rate `0.0`, repeat collapse `0.8752519080124866`.
- Status: retained as a legal public-evidence substrate, not a performance success. The next bottleneck is object/region causal inference plus sequence-level plan state.

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
