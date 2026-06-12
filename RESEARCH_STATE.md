# Research State

Last updated: 2026-06-12.

## Active Objective

Build an ARC-legal recurrent ARC-AGI-3 agent that completes at least 80% of official public games on the official runtime. Current state is far below that target.

## Source State

`GOAL.md` still contains the older recurrent-latent toy objective. The active thread objective is the official ARC-AGI-3 public-runtime objective. This document tracks the active ARC objective and should be read with `CONTEXT.md`, `METHODS_ATTEMPTED.md`, `ARC_RESULTS.md`, `FAILURE_ANALYSIS.md`, and `TODO_NEXT.md`.

## Current Best Official Public Result

Best retained official public result: `0/25` games solved, mean normalized score `0.005`.

Evidence: `docs/arc_affordance_report.json`, selected variant `state_graph_affordance`, mean normalized score `0.005`, useful events `0.04`, invalid action rate `0.0`, repeat collapse `0.22784841859240573`. This did not pass its improvement gate.

Latest recurrent/JEPA run: `docs/jepa_attempt_report.json`, primary variant `jepa_plus_attempt_memory`, `0/25` solved, mean normalized score `0.0013333333333333335`, useful events `0.013333333333333334`, invalid action rate `0.0`, repeat collapse `0.8493619977971841`. Outcome remains `NO IMPROVEMENT FOUND`.

## Current Architecture

- Official runtime adapter using only public frames, legal actions, public score/event deltas, terminal flags, and internal memory.
- Frozen recurrent latent base and external base arms.
- Attempt buffer storing frame/action/legal-action/score/event/terminal traces, including next public frame snapshots after actions.
- Video-JEPA temporal encoder used as a causal perceptual substrate for attempt memory.
- Attempt memory updates causal hypotheses and next-attempt action distributions without emitting text or direct action advice.
- Transition-graph attempt memory keyed by public observation hash and action, with visible-effect, no-effect, and delayed-public-event credits.
- Region/object causal memory built from public frame diffs, coarse click-to-region contact, changed colors, delayed region links, and compact prior-event sequence candidates.
- Component-level public causal graph over connected components, component action contact relations, public component transforms, component-goal links, live component-grounded sequence contradiction checks, predicted component-transition scoring, exact public component-state transition-goal chain search, relation-level component goal-chain abstraction, and generic relation-delta mechanism mining from public score/event-linked component changes.

## Latest Change

Implemented generic relation-delta event mechanism mining in `src/jepa_attempt_memory.py`.

The mechanism converts public component before/after hypotheses into generic scoped relation-delta tokens such as movement direction, value transform, appearance, disappearance, visual transform, and blocked/no-effect. Tokens are scoped by public action template, generalized relation key, action family, and component value, receive direct and delayed public-event credit, score current legal actions through public relation/template matches, and accumulate contradiction penalties from live public postcondition failures. It does not use game IDs, hidden labels, source inspection, fixed action schedules, text action advice, or external solvers.

## Active Hypothesis

Relation-delta event mining is a legal causal substrate and can generalize a learned public component movement across shifted positions in focused tests, but official traces show it still does not infer the objective or produce later-attempt score/useful-event gains. The abstraction bridges exact coordinate and whole-state changes, but sparse positive events and weak goal inference remain dominant.

## Current Bottleneck

Official traces still show no solved games and negligible useful events. Attempt 2 reduced repeat collapse to `0.8247975948141715` and raised entropy to `0.9206747911567998`, but attempts 2 and 3 still had zero score and zero useful events. Attempt 3 repeat collapse worsened to `0.8653668643967575`. The strongest repeated pattern remains exploration stuck/cycle plus absent or wrong goal/mechanic inference.

## Completion Status

Not complete. The 80% official-public completion target is unproven and contradicted by current evidence.
