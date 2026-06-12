# Methods Attempted

Last updated: 2026-06-12.

## Relation-Level Component Goal Chains

- Rationale: exact public component-state chains were legal but too sparse. The same component relation can recur at different coordinates, so exact cells should not be the only bridge between attempts.
- Code touched: `src/jepa_attempt_memory.py`, `tests/test_video_jepa.py`.
- Mechanism: create public relation-state signatures from frame shape, background, component value/area counts, alignment, adjacency, and containment without exact cells; abstract contact actions into relation templates such as component value and area bucket; store relation-chain edges keyed by `(before_relation_state, action_template, after_relation_state)` with values, goal values, failures, contradictions, expectations, and delayed public-event credit; resolve relation templates to currently legal actions at planning time; abort stale relation-chain plans when live public relation postconditions contradict expectations.
- Result: syntax check passed, focused JEPA tests passed, full tests passed before official evaluation, compact trace schema remained valid after compaction, generalization/leakage audits passed, and official runtime still reported `NO IMPROVEMENT FOUND`. Primary `jepa_plus_attempt_memory` stayed at `0/25` solved, mean normalized score `0.0013333333333333335`, useful events `0.013333333333333334`, invalid action rate `0.0`, repeat collapse `0.8425362339951956`.
- Status: retained as a legal public-evidence substrate and relation-generalization improvement in focused tests, not a performance success. Attempt 2 repeat collapse improved to `0.7925725948141715`, but attempts 2 and 3 still had zero score and zero useful events.

## Component Transition-Goal Chain Search

- Rationale: predicted component transitions scored one action at a time. They did not search over public state transitions toward previously goal-linked component states or preserve expected postconditions across a multi-step experiment.
- Code touched: `src/jepa_attempt_memory.py`, `tests/test_video_jepa.py`.
- Mechanism: record exact public component-state signatures before and after each action; store component chain edges keyed by `(before_state, action, after_state)` with counts, values, failures, contradictions, expectations, delayed public-event credit, and goal-state values; search short action chains from the current public component state; activate the best chain as the live sequence plan; reject stale chains when observed public postconditions contradict the predicted after-state.
- Result: syntax check passed, focused JEPA tests passed, full tests passed before and after official evaluation, generalization/leakage audits passed, trace schema remained valid after compaction, official runtime still reported `NO IMPROVEMENT FOUND`. Primary `jepa_plus_attempt_memory` stayed at `0/25` solved, mean normalized score `0.0013333333333333335`, useful events `0.013333333333333334`, invalid action rate `0.0`, repeat collapse `0.8345795011033633`.
- Status: retained as a legal public-evidence substrate and sequence-grounding improvement, not a performance success. Attempt 2 repeat collapse improved to `0.7950725948141715`, but attempts 2 and 3 still had zero score and zero useful events.

## Predicted Component-Transition Planning and Scoring Cache

- Rationale: the component graph stored public component relations and sequence expectations, but legal action scoring still mostly rewarded historical relation/contact values. It did not explicitly score expected next component mechanisms, and component extraction was recomputed during per-action scoring.
- Code touched: `src/jepa_attempt_memory.py`, `tests/test_video_jepa.py`.
- Mechanism: record component relation-to-mechanism predictions for movement, appearance, disappearance, color/shape transform, split/merge, visual transform, stable, and blocked/no-effect outcomes; score legal actions by expected productive public component transitions, goal-linked transition evidence, family-backed transition evidence, and contradiction penalties; penalize stale predicted transitions after live sequence contradictions; cache connected components once per observation and reuse component-target relations across legal actions.
- Result: syntax check passed, focused JEPA tests passed, full tests passed before and after official evaluation, generalization/leakage audits passed, trace schema remained valid after compaction, official runtime still reported `NO IMPROVEMENT FOUND`. Primary `jepa_plus_attempt_memory` stayed at `0/25` solved, mean normalized score `0.0013333333333333335`, useful events `0.013333333333333334`, invalid action rate `0.0`, repeat collapse `0.8283416385334161`.
- Status: retained as a legal public-evidence substrate and runtime-efficiency fix, not a performance success. Attempt 2 repeat collapse improved to `0.7705541839614582`, but no later-attempt score/useful-event improvement occurred.

## Component-Level Public Causal Graph and Grounded Sequence Check

- Rationale: coarse region/object memory could detect changed cells and replay prior public-event windows, but it could not distinguish component movement, component contact, stale sequence hypotheses, or specific public component transitions.
- Code touched: `src/jepa_attempt_memory.py`, `src/jepa_arc_eval.py`, `tests/test_video_jepa.py`.
- Mechanism: extract connected components from public pre-action and post-action frames; infer movement, appearance, disappearance, color/shape transforms, and blocked/no-effect transitions; link action target cells and action families to component relations; record component-goal links and delayed public-event credit; attach expected component transitions to sequence candidates; abort active sequence replay when live public frame diffs contradict the expected component change.
- Result: syntax check passed, focused JEPA tests passed, full tests passed before and after audit/compaction, generalization/leakage audits passed, official runtime still reported `NO IMPROVEMENT FOUND`. Primary `jepa_plus_attempt_memory` stayed at `0/25` solved, mean normalized score `0.0013333333333333335`, useful events `0.013333333333333334`, invalid action rate `0.0`, repeat collapse `0.8470217966721809`.
- Status: retained as a legal public-evidence substrate, not a performance success. Attempt 2 reduced repeat collapse to `0.7991510418849557`, but no later-attempt score/useful-event improvement occurred.

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
