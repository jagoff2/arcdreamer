# Research State

Last updated: 2026-06-12.

## Active Objective

Build an ARC-legal recurrent ARC-AGI-3 agent that completes at least 80% of official public games on the official runtime. Current state is far below that target.

## Source State

`GOAL.md` still contains the older recurrent-latent toy objective. The active thread objective is the official ARC-AGI-3 public-runtime objective. This document tracks the active ARC objective and should be read with `CONTEXT.md`, `METHODS_ATTEMPTED.md`, `ARC_RESULTS.md`, `FAILURE_ANALYSIS.md`, and `TODO_NEXT.md`.

## Current Best Official Public Result

Best retained official public result: `0/25` games solved, mean normalized score `0.005`.

Evidence: `docs/arc_affordance_report.json`, selected variant `state_graph_affordance`, mean normalized score `0.005`, useful events `0.04`, invalid action rate `0.0`, repeat collapse `0.22784841859240573`. This did not pass its improvement gate.

Latest recurrent/JEPA run: `docs/jepa_attempt_report.json`, primary variant `jepa_plus_attempt_memory`, `0/25` solved, mean normalized score `0.0013333333333333335`, useful events `0.013333333333333334`, invalid action rate `0.0`, repeat collapse `0.8283416385334161`. Outcome remains `NO IMPROVEMENT FOUND`.

## Current Architecture

- Official runtime adapter using only public frames, legal actions, public score/event deltas, terminal flags, and internal memory.
- Frozen recurrent latent base and external base arms.
- Attempt buffer storing frame/action/legal-action/score/event/terminal traces, including next public frame snapshots after actions.
- Video-JEPA temporal encoder used as a causal perceptual substrate for attempt memory.
- Attempt memory updates causal hypotheses and next-attempt action distributions without emitting text or direct action advice.
- Transition-graph attempt memory keyed by public observation hash and action, with visible-effect, no-effect, and delayed-public-event credits.
- Region/object causal memory built from public frame diffs, coarse click-to-region contact, changed colors, delayed region links, and compact prior-event sequence candidates.
- Component-level public causal graph over connected components, component action contact relations, public component transforms, component-goal links, live component-grounded sequence contradiction checks, and predicted component-transition scoring.

## Latest Change

Implemented predicted component-transition planning and per-observation component caching in `src/jepa_attempt_memory.py`.

The mechanism records relation-to-mechanism predictions such as component movement, appearance, disappearance, color transform, visual transform, split/merge, stable, and blocked/no-effect. Legal action scoring now uses expected next public component transitions, goal-linked transition evidence, and contradiction penalties. It caches public connected components once per observation for legal-action scoring. It does not use game IDs, hidden labels, source inspection, fixed action schedules, text action advice, or external solvers.

## Active Hypothesis

Predicted component transitions are a legal causal substrate and reduced repeat collapse versus the prior component run, but they still do not identify the objective or produce later-attempt score/useful-event gains. The current planner predicts local public visual effects better than it infers multi-step mechanics or goals.

## Current Bottleneck

Official traces still show no solved games and negligible useful events. Attempt 2 reduced repeat collapse to `0.7705541839614582` and raised entropy to `1.146861043117688`, but attempts 2 and 3 still had zero score and zero useful events. The strongest repeated pattern remains exploration stuck/cycle plus absent or wrong goal/mechanic inference.

## Completion Status

Not complete. The 80% official-public completion target is unproven and contradicted by current evidence.
