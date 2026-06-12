# Failure Analysis

Last updated: 2026-06-12.

## Aggregate Failure

Current official-public completion is `0/25` games solved. No retained experiment has solved an official public game.

Best retained mean normalized score is `0.005`, far below the `>=80%` completion target.

## Latest JEPA Failure

The region/object causal hypothesis layer added public frame-diff hypotheses, object-region action scoring, delayed region links, and prior-event sequence candidates, but the official runtime result stayed negative:

- `jepa_plus_attempt_memory` mean normalized score: `0.0013333333333333335`
- useful events: `0.013333333333333334`
- repeat collapse: `0.8607239187052945`
- official score gain: `0.0`
- useful-event gain: `0.0`
- attempts 2 or 3 did not improve over attempt 1

Primary attempt 1 had mean normalized score `0.004` and useful events `0.04`; attempts 2 and 3 both had score `0.0` and useful events `0.0`. Attempt 2 reduced repeat collapse to `0.8148790229221212`, but that behavioral change did not translate into score. Attempt 3 repeat collapse rose to `0.9093711990131383`.

The latest repeat-collapse regression gate is `-0.051449664832514785`, still failing the required repeat-collapse improvement threshold.

## Repeated Patterns

From `docs/arcagi3_failure_report.json`:

- primary `C exploration stuck/cycle`: `21`
- primary `F wrong/absent goal inference`: `4`
- secondary `F wrong/absent goal inference`: `19`
- secondary `G planner cannot form sequence`: `21`
- secondary `H unsupported mechanic`: `19`

Observed across latest JEPA traces:

- attempt memory changes action distributions but does not make later attempts better;
- public object/region hypotheses are recorded, but their coarse contact and region abstractions do not identify the actual objective;
- positive events remain sparse and mostly isolated;
- short sequence candidates replay prior event windows but are not grounded strongly enough to generalize across changed public states;
- official variants often repeat actions or fail to transform public observations into multi-step plans;
- no invalid-action issue is present, so the bottleneck is not action legality;
- lower repeat collapse alone has not been sufficient to create score.

## Strongest Bottleneck

The agent still lacks robust mechanic and goal inference from public observation dynamics. Current memory can record changed regions, changed colors, coarse action contacts, and short positive-event windows, but it cannot infer which public object relation matters, which latent rule produced the score event, or how to form a durable multi-step experiment after the state changes.

The next useful pressure should be higher-resolution public component dynamics and plan grounding, not another scalar action bonus. Candidate directions:

- component-level causal graph over public connected components instead of only 8x8 regions;
- explicit public relation hypotheses such as contact, containment, alignment, color match, object removal, and object transfer;
- sequence state tied to predicted public state transitions, with contradictions lowering stale hypotheses;
- trace analysis of recurring partial-score event patterns only as generic public mechanism mining, not game-specific branching.

## Current Non-Blocker

Leakage/no-hack audits pass. The current blocker is capability, not legality.
