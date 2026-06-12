# Research State

Last updated: 2026-06-12.

## Active Objective

Build an ARC-legal recurrent ARC-AGI-3 agent that completes at least 80% of official public games on the official runtime. Current state is far below that target.

## Source State

`GOAL.md` still contains the older recurrent-latent toy objective. The active thread objective is the official ARC-AGI-3 public-runtime objective. This document tracks the active ARC objective and should be read with `CONTEXT.md`, `METHODS_ATTEMPTED.md`, `ARC_RESULTS.md`, `FAILURE_ANALYSIS.md`, and `TODO_NEXT.md`.

## Current Best Official Public Result

Best retained official public result: `0/25` games solved, mean normalized score `0.005`.

Evidence: `docs/arc_affordance_report.json`, selected variant `state_graph_affordance`, mean normalized score `0.005`, useful events `0.04`, invalid action rate `0.0`, repeat collapse `0.22784841859240573`. This did not pass its improvement gate.

Latest recurrent/JEPA run: `docs/jepa_attempt_report.json`, primary variant `jepa_plus_attempt_memory`, `0/25` solved, mean normalized score `0.0013333333333333335`, useful events `0.013333333333333334`, invalid action rate `0.0`, repeat collapse `0.8752519080124866`. Outcome remains `NO IMPROVEMENT FOUND`.

## Current Architecture

- Official runtime adapter using only public frames, legal actions, public score/event deltas, terminal flags, and internal memory.
- Frozen recurrent latent base and external base arms.
- Attempt buffer storing frame/action/legal-action/score/event/terminal traces.
- Video-JEPA temporal encoder used as a causal perceptual substrate for attempt memory.
- Attempt memory updates causal hypotheses and next-attempt action distributions without emitting text or direct action advice.
- Transition-graph attempt memory keyed by public observation hash and action, with visible-effect, no-effect, and delayed-public-event credits.

## Latest Change

Implemented a transition-graph next-attempt planner in `src/jepa_attempt_memory.py` and wired it through `src/jepa_arc_eval.py`.

The planner records public `(observation_hash, action)` edges from prior attempts, tracks visible state changes, blocked/no-effect edges, positive public events, delayed credit for actions shortly before public events, and observation-specific next-action scores. It does not use game IDs, hidden labels, source inspection, fixed action schedules, or text action advice.

Regression tests were added in `tests/test_video_jepa.py` for public no-effect edge penalties and delayed public-event predecessor credit.

## Active Hypothesis

The transition graph is a correct public-evidence substrate, but the current use is still too local and scalar. It can bias a current action by known state-action edge values, but it does not yet form robust object/region hypotheses or maintain a multi-step experiment plan across states.

## Current Bottleneck

Official traces still show no solved games, negligible useful events, and high repeat collapse in recurrent/JEPA variants. Attempt 1 of `jepa_plus_attempt_memory` found the only useful events in the latest run; attempts 2 and 3 regressed to zero score/useful events. The strongest repeated pattern remains exploration stuck/cycle plus absent or wrong goal inference.

## Completion Status

Not complete. The 80% official-public completion target is unproven and contradicted by current evidence.
