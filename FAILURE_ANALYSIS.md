# Failure Analysis

Last updated: 2026-06-12.

## Aggregate Failure

Current official-public completion is `0/25` games solved. No retained experiment has solved an official public game.

Best retained mean normalized score is `0.005`, far below the `>=80%` completion target.

## Latest JEPA Failure

The strict outcome-classifier patch fixed a real causal-accounting bug, but the official runtime result stayed negative:

- `jepa_plus_attempt_memory` mean normalized score: `0.0013333333333333335`
- useful events: `0.013333333333333334`
- repeat collapse: `0.8753405853487634`
- official score gain: `0.0`
- useful-event gain: `0.0`
- attempts 2 or 3 did not improve over attempt 1

The patch slightly reduced the severity of repeat-collapse regression in the primary JEPA gate versus the previous report, from `-0.05311750710721985` to `-0.044466188032200926`, but this is not enough to count as an improvement.

## Repeated Patterns

From `docs/arcagi3_failure_report.json`:

- primary `C exploration stuck/cycle`: `21`
- primary `F wrong/absent goal inference`: `4`
- secondary `F wrong/absent goal inference`: `19`
- secondary `G planner cannot form sequence`: `21`
- secondary `H unsupported mechanic`: `19`

Observed across latest JEPA traces:

- attempt memory changes action distributions but does not make later attempts better;
- positive events remain sparse and mostly isolated;
- official variants often repeat actions or fail to transform public observations into multi-step plans;
- no invalid-action issue is present, so the bottleneck is not action legality;
- lower repeat collapse alone has not been sufficient to create score.

## Strongest Bottleneck

The agent lacks a planner that converts full-attempt evidence into targeted next-attempt experiments over state transitions, delayed effects, object/contact changes, and goal hypotheses. Current memory can score actions, but it does not build a robust causal graph of action sequences and outcomes.

## Current Non-Blocker

Leakage/no-hack audits pass. The current blocker is capability, not legality.
