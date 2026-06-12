# Research State

Last updated: 2026-06-12.

## Active Objective

Build an ARC-legal recurrent ARC-AGI-3 agent that completes at least 80% of official public games on the official runtime. Current state is far below that target.

## Source State

`GOAL.md` still contains the older recurrent-latent toy objective. The active thread objective is the official ARC-AGI-3 public-runtime objective. This document tracks the active ARC objective and should be read with `CONTEXT.md`, `METHODS_ATTEMPTED.md`, `ARC_RESULTS.md`, `FAILURE_ANALYSIS.md`, and `TODO_NEXT.md`.

## Current Best Official Public Result

Best retained official public result: `0/25` games solved, mean normalized score `0.005`.

Evidence: `docs/arc_affordance_report.json`, selected variant `state_graph_affordance`, mean normalized score `0.005`, useful events `0.04`, invalid action rate `0.0`, repeat collapse `0.22784841859240573`. This did not pass its improvement gate.

Latest recurrent/JEPA run: `docs/jepa_attempt_report.json`, primary variant `jepa_plus_attempt_memory`, `0/25` solved, mean normalized score `0.0013333333333333335`, useful events `0.013333333333333334`, invalid action rate `0.0`, repeat collapse `0.8470217966721809`. Outcome remains `NO IMPROVEMENT FOUND`.

## Current Architecture

- Official runtime adapter using only public frames, legal actions, public score/event deltas, terminal flags, and internal memory.
- Frozen recurrent latent base and external base arms.
- Attempt buffer storing frame/action/legal-action/score/event/terminal traces, including next public frame snapshots after actions.
- Video-JEPA temporal encoder used as a causal perceptual substrate for attempt memory.
- Attempt memory updates causal hypotheses and next-attempt action distributions without emitting text or direct action advice.
- Transition-graph attempt memory keyed by public observation hash and action, with visible-effect, no-effect, and delayed-public-event credits.
- Region/object causal memory built from public frame diffs, coarse click-to-region contact, changed colors, delayed region links, and compact prior-event sequence candidates.
- Component-level public causal graph over connected components, component action contact relations, public component transforms, component-goal links, and live component-grounded sequence contradiction checks.

## Latest Change

Implemented component-level public causal graph memory and live sequence contradiction checks in `src/jepa_attempt_memory.py`, with observation-to-transition wiring in `src/jepa_arc_eval.py`.

The mechanism extracts public connected components from pre-action and post-action frames, records component movement, appearance, disappearance, color/shape transforms, target-contact relations, delayed public-event links, and component-goal relations. Sequence candidates now carry expected component transitions and abort live replay when the next public frame contradicts the expected component change. It does not use game IDs, hidden labels, source inspection, fixed action schedules, text action advice, or external solvers.

## Active Hypothesis

The component graph is a legal causal substrate, but its current evidence is still not state-predictive enough. Public component contact and transform relations can detect some action effects, but they do not yet infer the underlying mechanic, goal condition, or robust multi-step plan needed to improve official attempts.

## Current Bottleneck

Official traces still show no solved games, negligible useful events, and high repeat collapse in recurrent/JEPA variants. Attempt 2 reduced repeat collapse versus attempt 1 in the primary variant and had higher entropy, but attempts 2 and 3 still had zero score and zero useful events. The strongest repeated pattern remains exploration stuck/cycle plus absent or wrong goal/mechanic inference.

## Completion Status

Not complete. The 80% official-public completion target is unproven and contradicted by current evidence.
