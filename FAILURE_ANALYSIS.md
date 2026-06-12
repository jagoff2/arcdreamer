# Failure Analysis

Last updated: 2026-06-12.

## Aggregate Failure

Current official-public completion is `0/25` games solved. No retained experiment has solved an official public game.

Best retained mean normalized score is `0.005`, far below the `>=80%` completion target.

## Latest JEPA Failure

The transition-graph planner added public state-action edge memory and delayed-event credit, but the official runtime result stayed negative:

- `jepa_plus_attempt_memory` mean normalized score: `0.0013333333333333335`
- useful events: `0.013333333333333334`
- repeat collapse: `0.8752519080124866`
- official score gain: `0.0`
- useful-event gain: `0.0`
- attempts 2 or 3 did not improve over attempt 1

Primary attempt 1 had mean normalized score `0.004` and useful events `0.04`; attempts 2 and 3 both had score `0.0` and useful events `0.0`. The transition graph therefore did not produce the intended next-attempt improvement.

The latest repeat-collapse regression gate is `-0.04420015602337091`, still failing the required repeat-collapse improvement threshold.

## Repeated Patterns

From `docs/arcagi3_failure_report.json`:

- primary `C exploration stuck/cycle`: `21`
- primary `F wrong/absent goal inference`: `4`
- secondary `F wrong/absent goal inference`: `19`
- secondary `G planner cannot form sequence`: `21`
- secondary `H unsupported mechanic`: `19`

Observed across latest JEPA traces:

- attempt memory changes action distributions but does not make later attempts better;
- the transition graph records public edge evidence but does not convert it into durable multi-step experiments;
- positive events remain sparse and mostly isolated;
- official variants often repeat actions or fail to transform public observations into multi-step plans;
- no invalid-action issue is present, so the bottleneck is not action legality;
- lower repeat collapse alone has not been sufficient to create score.

## Strongest Bottleneck

The agent lacks object/region causal hypotheses and a compact sequence-level plan state. Current memory can record and score public state-action edges, but it still cannot identify which changed regions, contacts, pushes, toggles, spawns, removals, or delayed public effects should define the next multi-step experiment.

## Current Non-Blocker

Leakage/no-hack audits pass. The current blocker is capability, not legality.
