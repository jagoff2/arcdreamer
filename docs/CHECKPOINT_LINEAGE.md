# Checkpoint Lineage

This file exists to stop silent comparison mistakes between scientist checkpoints produced under different runtime feature schemas.

## Current Rule

- `spotlight_feature_schema_version = 1`
  - legacy spotlight checkpoint surface
  - does **not** include the later generic cost/schema feature keys in the live spotlight hashed feature map
- `spotlight_feature_schema_version = 2`
  - current spotlight checkpoint surface
  - includes extended generic cost/budget/schema signals in the live spotlight feature map
- `spotlight_feature_schema_version = 3`
  - current reset-guard spotlight checkpoint surface
  - preserves the `v2` generic cost/schema inputs and adds reset cooldown / reset-budget features plus reset-penalized update targets

Do not compare `v1`, `v2`, and `v3` runs as if they were the same runtime.

## Baselines

- legacy baseline, best:
  - [spotlight_exec_curriculum_best_legacy_v1.pkl](/G:/arcagi/artifacts/spotlight_exec_curriculum_best_legacy_v1.pkl)
- legacy baseline, latest:
  - [spotlight_exec_curriculum_latest_legacy_v1.pkl](/G:/arcagi/artifacts/spotlight_exec_curriculum_latest_legacy_v1.pkl)

These are copies of the old pre-compatibility-fix scientist artifacts and should be treated as `feature_schema_version = 1` baselines.

## Logging Boundary

Current harness and trainer outputs now surface:

- `spotlight_feature_schema_version`

Use that field before comparing:

- training summaries
- training eval events
- uncapped ARC harness traces
- checkpoint-derived ARC runs

## Compatibility Behavior

- old checkpoints with no stored `feature_schema_version` now load as legacy `v1`
- existing extended-runtime checkpoints still save/load as `feature_schema_version = 2`
- fresh reset-guard checkpoints now save `feature_schema_version = 3`

This preserves old baselines while allowing fresh retraining on the extended runtime.
