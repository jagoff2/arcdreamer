# Research State

Last updated: 2026-06-12.

## Active Objective

Build an ARC-legal recurrent ARC-AGI-3 agent that completes at least 80% of official public games on the official runtime. Current state is far below that target.

## Source State

`GOAL.md` still contains the older recurrent-latent toy objective. The active thread objective is the official ARC-AGI-3 public-runtime objective. This document tracks the active ARC objective and should be read with `CONTEXT.md`, `METHODS_ATTEMPTED.md`, `ARC_RESULTS.md`, `FAILURE_ANALYSIS.md`, and `TODO_NEXT.md`.

## Current Best Official Public Result

Best retained official public result: `0/25` games solved, mean normalized score `0.005`.

Evidence: `docs/arc_affordance_report.json`, selected variant `state_graph_affordance`, mean normalized score `0.005`, useful events `0.04`, invalid action rate `0.0`, repeat collapse `0.22784841859240573`. This did not pass its improvement gate.

Latest recurrent/JEPA run: `docs/jepa_attempt_report.json`, primary variant `jepa_plus_attempt_memory`, `0/25` solved, mean normalized score `0.0013333333333333335`, useful events `0.013333333333333334`, invalid action rate `0.0`, repeat collapse `0.8607239187052945`. Outcome remains `NO IMPROVEMENT FOUND`.

## Current Architecture

- Official runtime adapter using only public frames, legal actions, public score/event deltas, terminal flags, and internal memory.
- Frozen recurrent latent base and external base arms.
- Attempt buffer storing frame/action/legal-action/score/event/terminal traces, including next public frame snapshots after actions.
- Video-JEPA temporal encoder used as a causal perceptual substrate for attempt memory.
- Attempt memory updates causal hypotheses and next-attempt action distributions without emitting text or direct action advice.
- Transition-graph attempt memory keyed by public observation hash and action, with visible-effect, no-effect, and delayed-public-event credits.
- Region/object causal memory built from public frame diffs, coarse click-to-region contact, changed colors, delayed region links, and compact prior-event sequence candidates.

## Latest Change

Implemented region/object causal hypotheses and a compact sequence candidate planner in `src/jepa_attempt_memory.py`, with `next_frame` capture in `src/attempt_buffer.py` and active sequence replay wiring in `src/jepa_arc_eval.py`.

The mechanism infers public-frame movement, spawn, removal, toggle/transform, visual transform, and blocked/no-effect hypotheses. It records changed regions/colors, family-region/color values, failed regions, delayed public-event region links, and short action windows preceding positive public events. It then biases future legal actions through public object-region memory and can replay the best prior public-event action window across attempts. It does not use game IDs, hidden labels, source inspection, fixed action schedules, text action advice, or external solvers.

## Active Hypothesis

The region/object layer is a legal causal substrate, but its current evidence is too coarse. The 8x8 public-frame abstraction can identify changed cells and rough contacts, but it does not yet infer the underlying game mechanic, goal condition, or robust multi-step plan grounding needed to improve official attempts.

## Current Bottleneck

Official traces still show no solved games, negligible useful events, and high repeat collapse in recurrent/JEPA variants. Attempt 2 reduced repeat collapse versus attempt 1 in the primary variant, but attempts 2 and 3 still had zero score and zero useful events. The strongest repeated pattern remains exploration stuck/cycle plus absent or wrong goal/mechanic inference.

## Completion Status

Not complete. The 80% official-public completion target is unproven and contradicted by current evidence.
