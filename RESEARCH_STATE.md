# Research State

Last updated: 2026-06-12.

## Active Objective

Build an ARC-legal recurrent ARC-AGI-3 agent that completes at least 80% of official public games on the official runtime. Current state is far below that target.

## Source State

`GOAL.md` still contains the older recurrent-latent toy objective. The active thread objective is the official ARC-AGI-3 public-runtime objective. This document tracks the active ARC objective and should be read with `CONTEXT.md`, `METHODS_ATTEMPTED.md`, `ARC_RESULTS.md`, `FAILURE_ANALYSIS.md`, and `TODO_NEXT.md`.

## Current Best Official Public Result

Best retained official public result: `0/25` games solved, mean normalized score `0.005`.

Evidence: `docs/arc_affordance_report.json`, selected variant `state_graph_affordance`, mean normalized score `0.005`, useful events `0.04`, invalid action rate `0.0`, repeat collapse `0.22784841859240573`. This did not pass its improvement gate.

Latest recurrent/JEPA run: `docs/jepa_attempt_report.json`, primary variant `jepa_plus_attempt_memory`, `0/25` solved, mean normalized score `0.0013333333333333335`, useful events `0.013333333333333334`, invalid action rate `0.0`, repeat collapse `0.8753405853487634`. Outcome remains `NO IMPROVEMENT FOUND`.

## Current Architecture

- Official runtime adapter using only public frames, legal actions, public score/event deltas, terminal flags, and internal memory.
- Frozen recurrent latent base and external base arms.
- Attempt buffer storing frame/action/legal-action/score/event/terminal traces.
- Video-JEPA temporal encoder used as a causal perceptual substrate for attempt memory.
- Attempt memory updates causal hypotheses and next-attempt action distributions without emitting text or direct action advice.

## Latest Change

Patched `src/jepa_attempt_memory.py` so neutral visible-effect transitions are not treated as failed actions merely because the official runtime gives a small step cost. The memory now separates:

- positive event actions;
- strict no-effect or blocked actions;
- visible-effect actions;
- repeated actions after no-progress attempts.

Regression tests were added in `tests/test_video_jepa.py`.

## Active Hypothesis

The previous attempt-memory failure was partly caused by over-penalizing actions that changed public state without immediate reward. The patch improved the causal accounting but did not improve official outcomes. The remaining bottleneck is not just event classification; the agent lacks a strong next-attempt experiment planner that can form and test multi-step state-transition hypotheses from full attempts.

## Current Bottleneck

Official traces still show no solved games, negligible useful events, and high repeat collapse in recurrent/JEPA variants. The strongest repeated pattern is exploration stuck/cycle plus absent or wrong goal inference.

## Completion Status

Not complete. The 80% official-public completion target is unproven and contradicted by current evidence.
