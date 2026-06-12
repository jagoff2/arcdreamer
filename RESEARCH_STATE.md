# Research State

Last updated: 2026-06-12.

## Active Objective

Build an ARC-legal recurrent ARC-AGI-3 agent that completes at least 80% of official public games on the official runtime. Current state is far below that target.

## Source State

`GOAL.md` still contains the older recurrent-latent toy objective. The active thread objective is the official ARC-AGI-3 public-runtime objective. This document tracks the active ARC objective and should be read with `CONTEXT.md`, `METHODS_ATTEMPTED.md`, `ARC_RESULTS.md`, `FAILURE_ANALYSIS.md`, and `TODO_NEXT.md`.

## Current Best Official Public Result

Best retained official public result: `0/25` games solved, mean normalized score `0.005`.

Evidence: `docs/arc_affordance_report.json`, selected variant `state_graph_affordance`, mean normalized score `0.005`, useful events `0.04`, invalid action rate `0.0`, repeat collapse `0.22784841859240573`. This did not pass its improvement gate.

Latest recurrent/JEPA run: `docs/jepa_attempt_report.json`, primary variant `jepa_plus_attempt_memory`, `0/25` solved, mean normalized score `0.004000000000000001`, useful events `0.04`, invalid action rate `0.0`, repeat collapse `0.8474902052213587`. Outcome remains `NO IMPROVEMENT FOUND`.

## Current Architecture

- Official runtime adapter using only public frames, legal actions, public score/event deltas, terminal flags, and internal memory.
- Frozen recurrent latent base and external base arms.
- Attempt buffer storing frame/action/legal-action/score/event/terminal traces, including next public frame snapshots after actions.
- Video-JEPA temporal encoder used as a causal perceptual substrate for attempt memory.
- Attempt memory updates causal hypotheses and next-attempt action distributions without emitting text or direct action advice.
- Transition-graph attempt memory keyed by public observation hash and action, with visible-effect, no-effect, and delayed-public-event credits.
- Region/object causal memory built from public frame diffs, coarse click-to-region contact, changed colors, delayed region links, and compact prior-event sequence candidates.
- Component-level public causal graph over connected components, component action contact relations, public component transforms, component-goal links, live component-grounded sequence contradiction checks, predicted component-transition scoring, exact public component-state transition-goal chain search, relation-level component goal-chain abstraction, generic relation-delta mechanism mining from public score/event-linked component changes, and relation-delta-grounded sequence candidate resolution.

## Latest Change

Implemented relation-delta-grounded sequence candidate planning in `src/jepa_attempt_memory.py`.

The mechanism promotes positive public-event windows into relation-delta sequence candidates when their component expectations contain generic public relation-delta tokens. Sequence steps now carry generalized action templates, relation keys, fallback relations, target values, and expected delta tokens. Planning resolves a learned sequence step to currently legal shifted actions when the current public frame matches the expected relation-delta scope, and live transition observation advances or aborts the sequence using public postcondition checks. It does not use game IDs, hidden labels, source inspection, fixed action schedules, text action advice, or external solvers.

## Active Hypothesis

Relation-delta-grounded sequence planning is a legal causal substrate and can resolve a learned two-step public relation-delta plan across shifted component positions in focused tests. Official traces still show no solved games and no attempt 2 or 3 improvement over attempt 1. The mechanism improves the representation of action sequences, but sparse positive events, weak objective inference, and weak activation diagnostics remain dominant.

## Current Bottleneck

Official traces still show no solved games. The latest primary variant reached mean normalized score `0.004000000000000001` and useful events `0.04`, but this only matches the sparse attempt-1 signal and remains below the retained best `0.005`. Attempts 1, 2, and 3 all had mean normalized score `0.004` and useful events `0.04`, so later attempts did not improve. Attempt 2 reduced repeat collapse to `0.8290217098584192`, but attempt 3 remained high at `0.8555273716250329`. The strongest repeated pattern remains exploration stuck/cycle plus absent or wrong goal/mechanic inference.

## Completion Status

Not complete. The 80% official-public completion target is unproven and contradicted by current evidence.
