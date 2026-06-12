# Failure Analysis

Last updated: 2026-06-12.

## Aggregate Failure

Current official-public completion is `0/25` games solved. No retained experiment has solved an official public game.

Best retained mean normalized score is `0.005`, far below the `>=80%` completion target.

## Latest JEPA Failure

The relation-delta-grounded sequence planner added shifted action resolution for positive-event sequence candidates with expected public relation-delta tokens, but the official runtime result stayed negative:

- `jepa_plus_attempt_memory` mean normalized score: `0.004000000000000001`
- useful events: `0.04`
- repeat collapse: `0.8474902052213587`
- official score gain: `0.004`
- useful-event gain: `0.04`
- attempts 2 or 3 did not improve over attempt 1

Primary attempts 1, 2, and 3 each had mean normalized score `0.004` and useful events `0.04`. Attempt 2 reduced repeat collapse to `0.8290217098584192` and raised action entropy to `0.9094204107398244`, but that behavioral change did not improve score or useful events over attempt 1. Attempt 3 repeat collapse was `0.8555273716250329`.

The latest repeat-collapse drop attempt 1 to 3 is `0.002394162555590662`, failing the required repeat-collapse improvement threshold.

## Repeated Patterns

From `docs/arcagi3_failure_report.json`:

- primary `C exploration stuck/cycle`: `21`
- primary `F wrong/absent goal inference`: `4`
- secondary `F wrong/absent goal inference`: `19`
- secondary `G planner cannot form sequence`: `21`
- secondary `H unsupported mechanic`: `19`

Observed across latest JEPA traces:

- attempt memory changes action distributions but does not make later attempts better than the first attempt;
- public object/region, component transition, relation-chain, relation-delta, and relation-delta sequence hypotheses are recorded, but their contact/transform abstractions do not identify the actual objective;
- positive events remain sparse and mostly isolated;
- short sequence candidates, exact/relation-level component transition-goal chains, relation-delta mechanisms, and relation-delta sequence candidates carry public expectations and can abort or penalize contradictions, but they are not yet predictive enough to infer useful mechanics or goals;
- official variants often repeat actions or fail to transform public observations into multi-step plans;
- no invalid-action issue is present, so the bottleneck is not action legality;
- lower repeat collapse alone has not been sufficient to create score.

## Strongest Bottleneck

The agent still lacks robust mechanic and goal inference from public observation dynamics. Current memory can record changed regions, changed colors, component movement/transforms, action contacts, predicted transition mechanisms, exact public component-state edges, relation-level component-state edges, generic relation-delta mechanisms, and shifted relation-delta sequence candidates, but it cannot infer which public component relation matters, which latent rule produced the score event, or how to form a durable multi-step experiment after the state changes.

The next useful pressure should be higher-resolution public component dynamics and plan grounding, not another scalar action bonus. Candidate directions:

- explicit public relation hypotheses such as color match, object removal, object transfer, repeated transform, and movement-relative-to-target;
- activation diagnostics for relation-chain, relation-delta, and relation-delta sequence planners so weak evidence can be separated from stale templates;
- mechanism mining from recurring public partial-score patterns expressed only as generic relation dynamics;
- contradiction-weighted pruning that reduces stale chain hypotheses without suppressing exploration of unseen legal relations;
- trace analysis of recurring partial-score event patterns only as generic public mechanism mining, not game-specific branching.

## Current Non-Blocker

Leakage/no-hack audits pass. The current blocker is capability, not legality.
