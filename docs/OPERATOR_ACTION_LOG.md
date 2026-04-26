# Operator Action Log

This file tracks Codex operator actions, not agent actions.

Rule:

- Append a dated entry for each execution slice before moving to the next slice.
- Include commands run, files read or changed, tests, training/eval runs, subprocesses, and known lost outputs.
- Keep entries concise, factual, and reconstructable.
- Mark reconstructed entries explicitly when the exact command output was not captured.

## 2026-04-24 08:49 EDT, Reconstructed From Transcript

Status:

- This entry is incomplete before this timestamp because I did not maintain a command-by-command Markdown ledger earlier in the session.
- The missing ledger is a real process failure. The chat transcript and tool outputs are the source for this reconstruction.
- Clean ARC success has not been achieved in the current clean path.

Operator actions reconstructed:

1. Resumed after context compaction with prior state summary.
2. Tried to poll old cleaned `hybrid` eval session `88340`; tool returned `Unknown process id 88340`, so its buffered final result was not recoverable from the PTY.
3. Ran `git status --short`, `rg --files`, and an artifact listing under `artifacts/`.
4. Spawned sidecar explorer `Carver` for minimum ingredients of a minimally conscious online learning agent.
5. Spawned sidecar explorer `Huygens` for runtime/training reset, action-surface, progress-target, and controller audit.
6. Read these files or relevant sections:
   - `arcagi/agents/base.py`
   - `arcagi/training/arc_public.py`
   - `arcagi/models/action_encoder.py`
   - `arcagi/training/arc_session.py`
   - `arcagi/core/progress_signals.py`
   - `arcagi/agents/learned_agent.py`
   - `tests/test_base_agent.py`
   - `tests/test_arc_public_training.py`
   - `tests/test_local_model_patches.py`
   - `tests/test_action_encoder.py`
   - `arcagi/core/inferred_state.py`
   - `arcagi/core/spatial_workspace.py`
   - `arcagi/core/representation_repair.py`
   - `arcagi/reasoning/causal_theory.py`
   - `arcagi/training/goal_scientist_public.py`
   - `arcagi/planning/planner.py`
   - `docs/GOAL_SCIENTIST_TRAINING.md`
7. Patched `arcagi/core/inferred_state.py` to add `reset_level()` that preserves cumulative progress/setback/contradiction counters while clearing transient mode/effect flags.
8. Patched `arcagi/core/spatial_workspace.py` to add `reset_level()` that clears only per-level visited positions and preserves tested/effect/contradiction sites and anchor memory.
9. Patched `arcagi/agents/base.py` so level reset calls component-level `reset_level()` where available instead of hard-resetting all inferred and spatial belief state.
10. Patched `arcagi/models/action_encoder.py` with an `ActionTargetLookup` cache so dense click/action encoding does not rescan all objects for every action.
11. Patched `arcagi/training/arc_public.py` so public ARC samples use visible online usefulness/policy targets and an objective label pass instead of `reward + 0.5 * visible_delta`.
12. Patched `arcagi/training/goal_scientist_public.py` to default `max_coordinate_probes` to `0` and stop slicing existing click actions unless an explicit positive cap is provided.
13. Patched `arcagi/core/progress_signals.py` so zero-reward movement visible delta is not positive objective usefulness by default.
14. Patched `arcagi/training/arc_session.py` to remove positive session-progress credit for reset after game over.
15. Patched `arcagi/envs/arc_adapter.py` so legal ARC click action `6` without camera metadata raises instead of hiding dense click parameters behind raw `"6"`.
16. Patched tests:
   - `tests/test_progress_signals.py`
   - `tests/test_arc_adapter_helpers.py`
17. Patched `docs/GOAL_SCIENTIST_TRAINING.md` so the documented coordinate probe cap is `0`, not `18`.
18. Ran `.venv313\Scripts\python.exe -m py_compile` on patched runtime/training files. It passed.
19. Tried `.venv313\Scripts\python.exe -m pytest ...`; it failed because `pytest` is not installed in `.venv313`.
20. Checked interpreter/test availability with `where.exe python`, `where.exe pytest`, `.venv313\Scripts\python.exe -m pip show pytest`, and `python -m pytest --version`.
21. Ran system Python targeted tests:
   - `python -m pytest tests/test_arc_adapter_helpers.py tests/test_action_encoder.py tests/test_base_agent.py tests/test_arc_public_training.py tests/test_progress_signals.py tests/test_local_model_patches.py tests/test_eval_path_constraints.py -q`
   - Result: `57 passed in 2.89s`.
22. Checked `arc_public` and `goal_scientist_public` CLI help under `.venv313`.
23. Checked ARC toolkit availability and game listings under `.venv313`.
   - Offline listing: `['ar25-0c556536']`.
   - Online listing fetched 25 environment ids. This was an operator/tooling check, not a runtime-agent dependency, but it should not be used as submission/runtime evidence.
24. Misinterpreted the user's complaint as agent action telemetry and patched `arcagi/evaluation/harness.py` to add optional `--trace-path` JSONL action tracing. This may still be useful for eval diagnosis, but it did not answer the user's actual complaint about my operator-action ledger.
25. Ran `.venv313\Scripts\python.exe -m py_compile arcagi\evaluation\harness.py`; it passed.
26. User clarified they meant my operator actions, not agent actions.
27. Created this file as the durable operator-action ledger.

Current unverified changes after this entry:

- `arcagi/evaluation/harness.py` action trace patch has not yet been covered by tests.
- The operator-action log itself has not yet been linked from `CONTEXT.MD`.
- No new clean ARC eval or training run has been executed after the objective-label/carryover patches.

## 2026-04-24 08:51 EDT

Operator actions:

1. Checked whether `docs/OPERATOR_ACTION_LOG.md` existed with `Test-Path`; it did not.
2. Added `docs/OPERATOR_ACTION_LOG.md`.
3. Updated `CONTEXT.MD` to state that this file is the durable Codex operator-action ledger and that agent runtime traces do not satisfy the operator-ledger requirement.

State after actions:

- Operator-action logging is now explicit in Markdown.
- The prior part of the session remains reconstructed, not complete.
- Next slice is harness-trace verification and short ARC probe.

## 2026-04-24 08:52 EDT

Operator actions:

1. Read the existing harness continuation test section in `tests/test_scientist_agent.py` to check coverage around `run_episode`.
2. Ran:
   - `python -m pytest tests/test_scientist_agent.py tests/test_arc_adapter_helpers.py tests/test_progress_signals.py tests/test_eval_path_constraints.py -q`
3. Test result:
   - `45 passed in 6.32s`

State after actions:

- Harness imports and the changed reset/progress/action-expansion tests still pass under the system Python test environment.
- The new `--trace-path` JSONL behavior itself is not directly asserted by a test yet.

## 2026-04-24 08:53 EDT

Operator actions:

1. Ran a short clean ARC probe with trace output:
   - `.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent online_language_memory --checkpoint-path artifacts/online_meta_language_memory_v6_350.pt --game-id ar25-0c556536 --mode offline --max-steps 5 --progress-every 1 --trace-path artifacts\traces\operator_20260424_0853_ar25_v6_5.jsonl`
2. Read the generated trace file:
   - `Get-Content -Path artifacts\traces\operator_20260424_0853_ar25_v6_5.jsonl`
3. Checked trace file size:
   - `(Get-Item artifacts\traces\operator_20260424_0853_ar25_v6_5.jsonl).Length`

Observed behavior:

- Initial legal action surface was `447` actions, confirming dense click expansion on `AR25`.
- Result was not success:
  - `success=false`
  - `won=false`
  - `levels_completed=0`
  - `return=0.0`
  - `steps=5`
- Action histogram:
  - `4`: 2
  - `5`: 1
  - `click:31:1`: 1
  - `2`: 1
- Family histogram:
  - `move`: 3
  - `select`: 1
  - `click`: 1
- `max_same_action_streak=1`.
- Runtime controller was inactive.
- The trace file exists and is about `41174` bytes.

Reasoning impact:

- The failure is now durably inspectable instead of living only in chat.
- The short run shows no immediate repeated-action loop in the first 5 steps, but it also shows no objective progress and confirms the policy still churns among movement/control/click without learned objective discovery.
- The next useful run must be training or longer eval with trace, not another unlogged blind eval.

## 2026-04-24 08:54-08:57 EDT

Operator actions:

1. Started ARC public training with patched labels and full dense action surface:
   - `.venv313\Scripts\python.exe -m arcagi.training.arc_public --mode offline --game-limit 1 --sessions-per-game 3 --max-steps 160 --epochs 2 --learning-rate 0.0002 --device cuda --init-checkpoint-path artifacts/online_meta_language_memory_v6_350.pt --checkpoint-path artifacts/arc_public_objective_v9.pt --behavior-policies learned,random`
2. Polled the background process several times. It printed toolkit startup only and no epoch metrics.
3. Tried to interrupt via stdin with Ctrl-C. The tool reported stdin was closed for that session.
4. Found matching processes with:
   - `Get-CimInstance Win32_Process | Where-Object { $_.CommandLine -like '*arcagi.training.arc_public*arc_public_objective_v9.pt*' } | Select-Object ProcessId,CommandLine`
5. Stopped matching Python processes:
   - `Stop-Process -Id 35752,29360 -Force`
6. Polled the original session again and it exited with code `1`.

Observed behavior:

- The attempted training run was too slow for fast iteration and did not emit useful metrics before interruption.
- It should not be treated as a valid completed artifact.

Reasoning impact:

- Next training run must be smaller but must keep the action surface uncapped.
- The iteration limit should be session count / step budget, not candidate/action pruning.

## 2026-04-24 08:58-08:59 EDT

Operator actions:

1. Checked whether interrupted artifact `artifacts\arc_public_objective_v9.pt` existed.
   - Result: `False`.
2. Checked for leftover `arc_public` or ARC eval processes.
   - Result: no matching Python process remained.
3. Started a smaller uncapped ARC public training slice:
   - `.venv313\Scripts\python.exe -m arcagi.training.arc_public --mode offline --game-limit 1 --sessions-per-game 1 --max-steps 32 --epochs 1 --learning-rate 0.0002 --device cuda --init-checkpoint-path artifacts/online_meta_language_memory_v6_350.pt --checkpoint-path artifacts/arc_public_objective_v9_smoke.pt --behavior-policies learned`
4. Polled the process once after startup.

Training result:

- Completed normally.
- Output checkpoint path: `artifacts\arc_public_objective_v9_smoke.pt`.
- Metrics:
  - `epochs=1.0`
  - `samples_last_epoch=32.0`
  - `loss_last_epoch=97591.6513671875`
  - `policy_loss_last_epoch=4.59873940050602`
  - `plan_loss_last_epoch=1881.5000228881836`
  - `uncertainty_last_epoch=0.12381348269991577`
  - `session_win_rate_last_epoch=0.0`
  - `avg_levels_completed_last_epoch=0.0`

Reasoning impact:

- The run proves the patched trainer can complete and write an artifact, but it did not discover ARC reward or level progress.
- The new checkpoint must be evaluated, but it should not be expected to be good from these metrics.

## 2026-04-24 09:00-09:01 EDT

Operator actions:

1. Started a clean 50-step ARC eval for the smoke-trained checkpoint:
   - `.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent online_language_memory --checkpoint-path artifacts/arc_public_objective_v9_smoke.pt --game-id ar25-0c556536 --mode offline --max-steps 50 --progress-every 10 --trace-path artifacts\traces\operator_20260424_0900_ar25_v9_smoke_50.jsonl`
2. User correctly objected that killing the earlier 2.5-minute neural training run because it was silent was a bad operator decision.
3. Acknowledged that mistake and committed not to kill future training runs for stdout silence alone.
4. Polled the active eval and let it finish.

Eval result:

- `success=false`
- `won=false`
- `return=0.0`
- `levels_completed=0`
- `steps=50`
- `interaction_steps=10`
- `reset_steps=0`
- `max_same_action_streak=1`
- Action histogram:
  - moves: `1` x9, `2` x10, `3` x10, `4` x9
  - clicks: 9 distinct click actions
  - undo `7` x2
  - select `5` x1
- Family histogram:
  - `move=38`
  - `click=9`
  - `undo=2`
  - `select=1`
- Trace path:
  - `artifacts\traces\operator_20260424_0900_ar25_v9_smoke_50.jsonl`

Reasoning impact:

- The smoke-trained v9 checkpoint did not improve ARC success.
- The failure is not a tight same-action infinite loop in this 50-step run. It is broader objective-free action churn with no reward discovery.
- Future neural training runs should be allowed to run long enough to matter, with progress logging added rather than premature interruption.

## 2026-04-24 09:02 EDT, Stop State

Operator action:

- User instructed: write what I am doing to file and end the turn.
- I am stopping active work now.

Current state:

- No clean ARC success has been achieved.
- The last completed eval was `artifacts\arc_public_objective_v9_smoke.pt` on `ar25-0c556536` for 50 steps.
- That eval failed:
  - `success=false`
  - `won=false`
  - `levels_completed=0`
  - `return=0.0`
  - `steps=50`
- Trace for that eval:
  - `artifacts\traces\operator_20260424_0900_ar25_v9_smoke_50.jsonl`
- The earlier longer ARC public training run toward `artifacts\arc_public_objective_v9.pt` was killed by me prematurely after about 2.5 minutes of stdout silence. That was an operator mistake and the artifact was not produced.
- The completed small training artifact is:
  - `artifacts\arc_public_objective_v9_smoke.pt`
- Targeted tests last run:
  - `python -m pytest tests/test_scientist_agent.py tests/test_arc_adapter_helpers.py tests/test_progress_signals.py tests/test_eval_path_constraints.py -q`
  - Result: `45 passed in 6.32s`

Known code/documentation changes in this slice:

- Added `docs/OPERATOR_ACTION_LOG.md`.
- Linked the operator-action log from `CONTEXT.MD`.
- Added optional `--trace-path` agent action JSONL tracing to `arcagi/evaluation/harness.py`.
- Changed ARC public training labels to reduce visible-delta pseudo-progress and add objective-oriented labels.
- Changed level reset behavior to preserve more learned belief state.
- Changed dense click handling to fail if legal click action `6` cannot be expanded.
- Changed default diagnostic coordinate cap to `0`.
- Changed zero-reward movement visible delta so it is not positive objective usefulness.

Next action should not begin until the user resumes:

- If resumed, continue by logging every operator action here before moving to the next execution slice.

## 2026-04-24 09:35 EDT

Operator actions:

1. Resumed after user instructed that GPT-Pro's response should be used for immediate implementation, not repeated back as prose.
2. Applied GPT-Pro's highest-priority repo-review fixes:
   - Added `ScientistAgent.handles_level_boundaries = True`.
   - Changed `arcagi.evaluation.harness.run_episode` so the harness does not call a second level reset when an agent declares it handles level boundaries internally.
   - Changed `ActionSpotlight.reset_level()` to preserve online session evidence across ARC level transitions while still clearing transient current-attempt state.
   - Changed `ScientistPlanner.candidate_actions()` and `ActionSpotlight._candidate_actions()` so `max_candidates` cannot truncate the legal ARC action surface.
3. Added regression tests:
   - `test_spotlight_reset_level_preserves_online_session_evidence`
   - `test_spotlight_candidate_surface_includes_legal_actions_when_planner_drops_them`
   - `test_spotlight_max_candidates_never_truncates_legal_surface`
   - `test_harness_does_not_double_reset_agent_that_handles_level_boundaries`

Validation:

- Compile check:
  - `.venv313\Scripts\python.exe -m py_compile arcagi\scientist\agent.py arcagi\evaluation\harness.py arcagi\scientist\spotlight.py arcagi\scientist\planner.py tests\test_action_spotlight.py tests\test_scientist_agent.py`
  - Result: passed.
- Targeted tests:
  - `python -m pytest tests/test_action_spotlight.py tests/test_scientist_agent.py tests/test_arc_adapter_helpers.py tests/test_eval_path_constraints.py -q`
  - Result: `59 passed in 6.02s`.

Current conclusion:

- The known double-reset state wipe at ARC level boundaries is fixed and guarded.
- The known legal-action candidate truncation path is fixed and guarded.
- This is not an ARC success claim. It removes two blockers that could invalidate online learning and dense action training.

Follow-up preservation action:

- Committed the slice:
  - `cf9c069 Preserve online state across ARC level boundaries`
- Pushed to `origin/main`.

## 2026-04-24 09:48 EDT

Operator actions:

1. Spawned a read-only explorer subagent to inspect checkpoint/state persistence and dense-action-surface blockers.
2. Implemented full online-state checkpoint persistence for the scientist path:
   - `HypothesisEngine.state_dict/load_state_dict`
   - `EpisodicMemory.state_dict/load_state_dict`
   - `ScientistPlanner.state_dict/load_state_dict`
   - `ScientistAgent.state_dict/load_state_dict` now includes engine, memory, planner, world model, and runtime counters.
   - `SpotlightScientistAgent.state_dict` now builds on the base full-state payload.
3. Implemented spotlight online-credit persistence:
   - exact/abstract/global action visit maps
   - no-effect, contradiction, binding, and probe-baseline maps
   - reset/stall/attempt counters
   - previous attempt outcomes and current attempt action records
4. Added checkpoint round-trip assertions for these fields.

Validation:

- Compile check:
  - `.venv313\Scripts\python.exe -m py_compile arcagi\scientist\spotlight.py tests\test_action_spotlight.py arcagi\scientist\memory.py arcagi\scientist\planner.py arcagi\scientist\hypotheses.py arcagi\scientist\agent.py arcagi\agents\spotlight_scientist_agent.py`
  - Result: passed.
- Focused checkpoint tests:
  - `python -m pytest tests/test_action_spotlight.py::test_spotlight_agent_from_checkpoint_restores_saved_config tests/test_scientist_agent.py::test_scientist_checkpoint_round_trip_and_harness_load -q`
  - Result: `2 passed in 2.57s`.
- Broader targeted tests:
  - `python -m pytest tests/test_action_spotlight.py tests/test_scientist_agent.py tests/test_scientist_training.py tests/test_arc_adapter_helpers.py tests/test_eval_path_constraints.py -q`
  - Result: `65 passed in 126.28s`.

Current conclusion:

- Checkpoints and train/eval holdout snapshots no longer drop the main online-learned scientist state.
- This is still not an ARC success claim. It removes another blocker that made training/eval state discontinuous.

Follow-up preservation action:

- Committed the slice:
  - `401f9fa Persist scientist online learning state`
- Pushed to `origin/main`.

## 2026-04-24 09:59 EDT

Operator actions:

1. Implemented a dense-action-surface guard for ARC train/eval paths:
   - Added `require_dense_arc_action_surface()` in `arcagi.envs.arc_adapter`.
   - Wired it into `arcagi.evaluation.harness.evaluate_arc`.
   - Wired it into `arcagi.scientist.train_arc`.
   - Wired it into `arcagi.training.arc_public`.
   - Added explicit `--allow-sparse-click-smoke` flags for debug-only reduced-click runs.
2. Added tests proving clean ARC eval/training rejects `ARCAGI_SPARSE_CLICKS_BASELINE=1` unless explicitly marked as smoke/debug.

Validation:

- Compile check:
  - `.venv313\Scripts\python.exe -m py_compile arcagi\envs\arc_adapter.py arcagi\evaluation\harness.py arcagi\scientist\train_arc.py arcagi\training\arc_public.py tests\test_eval_path_constraints.py tests\test_arc_public_training.py`
  - Result: passed.
- Sparse-click guard tests:
  - `python -m pytest tests/test_eval_path_constraints.py tests/test_arc_adapter_helpers.py tests/test_arc_public_training.py -q`
  - Result: `27 passed in 1.42s`.
- Combined targeted tests:
  - `python -m pytest tests/test_action_spotlight.py tests/test_scientist_agent.py tests/test_scientist_training.py tests/test_arc_adapter_helpers.py tests/test_eval_path_constraints.py tests/test_arc_public_training.py -q`
  - Result: `78 passed in 126.22s`.

Current conclusion:

- Clean ARC train/eval paths now fail fast if the sparse-click debug baseline would hide dense legal click parameters.
- Explicit smoke/debug reduced-action runs remain possible but are labeled by metadata and cannot be confused with clean success.

Follow-up preservation action:

- Committed the slice:
  - `d73ca8b Guard ARC eval against sparse click surface`
- Pushed to `origin/main`.

## 2026-04-24 10:15 EDT

Operator actions:

1. Ran clean repaired baseline eval:
   - `.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent spotlight --game-id ar25-0c556536 --mode offline --max-steps 80 --progress-every 20 --trace-path artifacts\traces\post_repair_spotlight_clean_80.jsonl`
   - Result: `success=false`, `won=false`, `levels_completed=0`, `return=0.0`, `steps=80`.
   - Failure mode: only 2 clicks, 62 moves, max same-action streak 17.
2. Implemented direct evidence-score action selection in `ActionSpotlight`:
   - Combines expected reward, information gain, expected change, uncertainty, memory, coverage, binder/probe/reset utility, action cost, and online no-effect/contradiction penalties.
   - Lets a sufficiently better combined online evidence score override habit tie-breaking.
   - Added regression proving repeated falsified actions lose to alternative diagnostic actions.
3. Ran targeted tests:
   - `python -m pytest tests/test_action_spotlight.py -q`
   - Result: `24 passed in 0.18s`.
   - `python -m pytest tests/test_action_spotlight.py tests/test_scientist_agent.py tests/test_scientist_training.py tests/test_arc_adapter_helpers.py tests/test_eval_path_constraints.py tests/test_arc_public_training.py -q`
   - Result: `79 passed in 113.56s`.
4. Ran clean evidence-score eval:
   - `.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent spotlight --game-id ar25-0c556536 --mode offline --max-steps 80 --progress-every 20 --trace-path artifacts\traces\post_evidence_score_spotlight_clean_80.jsonl`
   - Result: `success=false`, `won=false`, `levels_completed=0`, `return=0.0`, `steps=80`.
   - Behavior changed: clicks increased from 2 to 14, move count dropped from 62 to 57, option memory wrote 2 options, max same-action streak dropped from 17 to 14.

Current conclusion:

- Evidence-score selection improves dense action exposure but does not solve ARC.
- Next action is longer state-carrying ARC training with the repaired checkpoints and evidence-score controller.

Follow-up preservation action:

- Committed the slice:
  - `17f7672 Use online evidence score in spotlight selection`
- Pushed to `origin/main`.

## 2026-04-24 11:49 EDT

Operator actions:

1. While the long ARC training/holdout process was active, user requested a GPT-Pro second opinion on current progress.
2. Asked GPT-Pro for concrete next code changes using the current commit state, eval results, and live 320-step training diagnostics.
3. Saved the actionable advice summary:
   - `docs/GPT_PRO_SECOND_OPINION_2026-04-24.md`
4. Implemented GPT-Pro's first two recommendations:
   - Added online reliability/falsification evidence for exact actions and action families.
   - Reliability now gates exploitation-like score terms and adds explicit falsification penalty without pruning legal candidates.
   - Added within-session micro-attempt updates that finalize learning segments during long nonterminal failures without resetting the environment or clearing level state.
   - Added checkpoint persistence for the new evidence/micro-attempt counters.
5. Validation:
   - `.venv313\Scripts\python.exe -m py_compile arcagi\scientist\spotlight.py tests\test_action_spotlight.py`
   - Result: passed.
   - `python -m pytest tests/test_action_spotlight.py -q`
   - Result: `27 passed in 0.21s`.
   - `python -m pytest tests/test_action_spotlight.py tests/test_scientist_agent.py tests/test_scientist_training.py tests/test_arc_adapter_helpers.py tests/test_eval_path_constraints.py tests/test_arc_public_training.py -q`
   - Result: `82 passed in 123.61s`.

Live ARC training status:

- The 2-session, 320-step ARC training command is still running in the original Python process started before this patch.
- Session 1 holdout failed: `success_rate=0.0`, `avg_levels_completed=0.0`, `avg_return=0.0`, `avg_steps=320`.
- Session 2 holdout failed: `success_rate=0.0`, `avg_levels_completed=0.0`, `avg_return=0.0`, `avg_steps=320`.
- Final holdout failed: `success_rate=0.0`, `avg_levels_completed=0.0`, `avg_return=0.0`, `avg_steps=320`, `dense_action_surface=true`.

Current conclusion:

- The current long run has not achieved ARC success.
- The new reliability/micro-attempt code has not yet been exercised by an ARC run because the active process predates the patch.

Follow-up preservation action:

- Committed the slice:
  - `e5c4a9c Add reliability gates and micro-attempt learning`
- Pushed to `origin/main`.

## 2026-04-24 12:03 EDT

Operator actions:

1. Resumed after context compaction and found the saved terminal session handle was gone, so reconstructed state from the process table and trace files instead of killing any Python process.
2. Read the clean reliability/micro eval trace:
   - `.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent spotlight --game-id ar25-0c556536 --mode offline --max-steps 80 --progress-every 20 --trace-path artifacts\traces\post_reliability_micro_spotlight_clean_80.jsonl`
   - Result: `success=false`, `won=false`, `levels_completed=0`, `return=0.0`, `steps=80`, `dense_action_surface=true`.
   - Action histogram: `click=2`, `move=48`, `select=2`, `undo=28`.
   - Diagnostics: `micro_attempt_updates=6`, `adaptation_updates=70`, `executive_updates=80`, `binding_failure_total=4`, `no_effect_count_total=1`, `max_same_action_streak=13`.

Current conclusion:

- Reliability and micro-attempt learning are active, but clean ARC success is still zero.
- The failure mode shifted from sparse interaction and binder failure to overvaluing visible reversible movement and undo as information/progress.
- Next implementation target is score normalization plus anti-perseveration and objective-progress separation, without pruning legal actions or adding scripted/search control.

## 2026-04-24 12:45 EDT

Operator actions:

1. Implemented score normalization, score-only anti-perseveration, weaker visible-only evidence, negative micro-attempt targets for nonprogress motion, and a guard preventing zero-update habit priors from overriding evidence.
2. Implemented follow-up reset/option-memory fixes:
   - Disabled the hand-authored nonterminal reset escape hatch in the ARC-facing spotlight controller.
   - Kept rewardless salient option memory, but capped reversible visible-motion option value and gated continuation bonus by option value.
3. Validation:
   - `python -m pytest tests/test_action_spotlight.py -q`
   - Result after habit patch: `30 passed in 0.23s`.
   - `python -m pytest tests/test_action_spotlight.py tests/test_scientist_agent.py -q`
   - Result after reset/option patch: `53 passed in 4.95s`.
   - `python -m pytest tests/test_action_spotlight.py tests/test_scientist_agent.py tests/test_scientist_training.py tests/test_arc_adapter_helpers.py tests/test_eval_path_constraints.py tests/test_arc_public_training.py -q`
   - Result after reset/option patch: `86 passed in 279.99s`.
4. Ran clean normalized/anti-perseveration eval:
   - `.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent spotlight --game-id ar25-0c556536 --mode offline --max-steps 80 --progress-every 20 --trace-path artifacts\traces\post_norm_antipersev_spotlight_clean_80.jsonl`
   - Result: `success=false`, `won=false`, `levels_completed=0`, `return=0.0`, `steps=80`.
   - Action histogram: `click=5`, `move=55`, `select=3`, `undo=17`, `max_same_action_streak=5`, `micro_attempt_updates=2`, `last_micro_attempt_target=-1.0`.
5. Ran clean habit-gated eval:
   - `.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent spotlight --game-id ar25-0c556536 --mode offline --max-steps 80 --progress-every 20 --trace-path artifacts\traces\post_habit_gate_spotlight_clean_80.jsonl`
   - Result: `success=false`, `won=false`, `levels_completed=0`, `return=0.0`, `steps=80`.
   - Action histogram: `click=3`, `move=63`, `select=2`, `undo=11`, `reset=1`, `max_same_action_streak=4`.
6. Ran clean reset/option-gated eval:
   - `.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent spotlight --game-id ar25-0c556536 --mode offline --max-steps 80 --progress-every 20 --trace-path artifacts\traces\post_option_reset_gate_spotlight_clean_80.jsonl`
   - Result: `success=false`, `won=false`, `levels_completed=0`, `return=0.0`, `steps=80`.
   - Action histogram: `click=4`, `move=57`, `select=3`, `undo=16`, `reset=0`, `max_same_action_streak=4`.
   - Last-decision diagnostics: `memory_bonus=0.0023`, `option_schema_bonus=0.0`, `last_micro_attempt_target=-0.94`.

Current conclusion:

- Clean ARC success is still zero.
- The fixed regressions are real: repeated-action streaks are down, nonterminal reset is gone, and rewardless option memory no longer dominates scoring.
- Remaining failure: executive/world-model scoring still assigns positive value to reversible motion and undo hypotheses despite 80 steps with no reward or level progress.

Follow-up preservation action:

- Patched executive TD target to discount bootstrap value and add a learned penalty for no-objective-progress transitions, especially visible-only motion and undo.
- Validation:
  - `.venv313\Scripts\python.exe -m py_compile arcagi\scientist\spotlight.py arcagi\scientist\memory.py tests\test_action_spotlight.py tests\test_scientist_agent.py`
  - Result: passed.
  - `python -m pytest tests\test_action_spotlight.py tests\test_scientist_agent.py -q`
  - Result: `53 passed in 5.12s`.

## 2026-04-24 13:40 EDT

Operator actions:

1. Asked GPT-Pro to continue the same review thread and work through epistemics, process, architecture, training, and code realization instead of another local Spotlight patch.
2. Ran two follow-up consensus rounds in the same GPT-Pro thread:
   - First round established that the last 24 hours improved falsification telemetry but did not prove ARC learning.
   - Second round stress-tested the learned-online plan and clarified the boundary between acceptable inductive bias and forbidden scripted/search control.
   - Third round locked the immediate implementation order and stop conditions.
3. Saved the consensus document:
   - `docs/GPT_PRO_CONSENSUS_2026-04-24.md`

Current conclusion:

- Spotlight/TheoryManager is now treated as `instrumented_spotlight_baseline`.
- The dirty visible-only/objective-failure Spotlight patch can be preserved only as non-claimable telemetry and trace-generation infrastructure.
- The next valid code path is a minimal learned online decision owner with full dense action scoring and synthetic online-adaptation gates before ARC claiming eval.

Follow-up implementation:

1. Added `arcagi.scientist.boundary` and harness claim metadata:
   - `controller_kind`
   - `claim_eligible`
   - `learned_online_controller`
   - `legal_action_count`
   - `scored_action_count`
2. Added `arcagi.learned_online` with a minimal learned decision owner:
   - generic transition labels;
   - full dense action feature encoding;
   - explicit online belief/action-semantics state;
   - fast online prediction heads;
   - episodic memory features;
   - grounded question tokens;
   - chunked all-action scoring.
3. Registered `--agent learned_online_minimal`.
4. Added learned-online gate tests for boundary enforcement, dense action scoring, same-state score change after evidence, no-sweep behavior, synthetic online-vs-frozen sanity, and controlled memory/question ablations.

Validation:

- Compile check:
  - `.venv313\Scripts\python.exe -m py_compile arcagi\evaluation\harness.py arcagi\agents\learned_online_minimal_agent.py arcagi\learned_online\signals.py arcagi\learned_online\action_features.py arcagi\learned_online\fast_belief.py arcagi\learned_online\questions.py arcagi\learned_online\memory.py arcagi\learned_online\minimal_model.py arcagi\learned_online\policy.py arcagi\learned_online\curriculum.py`
  - Result: passed.
- Learned-online focused tests:
  - `python -m pytest tests\test_learned_online_claim_boundary.py tests\test_learned_online_dense_surface.py tests\test_learned_online_update_loop.py tests\test_learned_online_no_sweep.py tests\test_learned_online_synthetic_gates.py tests\test_learned_online_ablation_gates.py tests\test_eval_path_constraints.py -q`
  - Result: `19 passed in 3.75s`.
- Combined targeted regression:
  - `python -m pytest tests\test_learned_online_claim_boundary.py tests\test_learned_online_dense_surface.py tests\test_learned_online_update_loop.py tests\test_learned_online_no_sweep.py tests\test_learned_online_synthetic_gates.py tests\test_learned_online_ablation_gates.py tests\test_action_spotlight.py tests\test_scientist_agent.py tests\test_scientist_training.py tests\test_arc_adapter_helpers.py tests\test_eval_path_constraints.py tests\test_arc_public_training.py -q`
  - Result: `104 passed in 123.95s`.

## 2026-04-24 14:18 EDT

Operator actions:

1. Ran clean `learned_online_minimal` ARC smoke through `.venv313` because system Python lacks the ARC toolkit:
   - `.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent learned_online_minimal --game-id ar25-0c556536 --mode offline --max-steps 80 --progress-every 20 --trace-path artifacts\traces\learned_online_minimal_clean_80.jsonl`
   - Result: `success=false`, `return=0.0`, `levels_completed=0`, `select=64`, `max_same_action_streak=63`.
2. Added a generic no-effect nonprogress cost label and reran:
   - Trace: `artifacts\traces\learned_online_minimal_noeffect_cost_80.jsonl`
   - Result: `success=false`, `return=0.0`, `levels_completed=0`, `click=39`, `select=35`, `max_same_action_streak=34`.
3. Strengthened generic learned cost priors and trained a synthetic outcome checkpoint:
   - `python -m scripts.train_learned_online_minimal --output artifacts\learned_online_minimal_synth_2000.pkl --episodes 2000 --max-steps 12 --seed 41`
   - Result: synthetic `success_rate=0.76`, `model_updates=22546`.
4. ARC with that checkpoint still failed:
   - Trace: `artifacts\traces\learned_online_minimal_synth2000_80.jsonl`
   - Result: `success=false`, `return=0.0`, `levels_completed=0`, `click=74`, `max_same_action_streak=1`.
5. Added parametric-click family/context evidence inheritance and null dense coverage tests to avoid every dense click coordinate getting a clean exact-action slate.
6. ARC with the synthetic checkpoint after parametric inheritance still failed:
   - Trace: `artifacts\traces\learned_online_minimal_param_belief_80.jsonl`
   - Result: `success=false`, `return=0.0`, `levels_completed=0`, `move=73`, `click=5`, `max_same_action_streak=21`.
7. Asked GPT-Pro for a focused stop-condition review. GPT-Pro agreed:
   - `learned_online_minimal` is valid infrastructure and falsification scaffold.
   - It is not validated ARC competence.
   - Stop ARC tuning on the shallow scorer.
   - Next path is a clean recurrent latent-state learned online agent with stronger synthetic sequence gates.
8. Implemented the immediate non-ARC-tuning fixes:
   - Marked `learned_online_minimal` as `role="falsification_gate_scaffold"` and `arc_competence_validated=False`.
   - Fixed information-gain targets to use replay/probe memory loss reduction instead of same-sample fitting.
   - Removed raw uncertainty from direct policy value; uncertainty remains diagnostic/question/exploration state only.

Validation:

- Learned-online focused tests:
  - `python -m pytest tests\test_learned_online_claim_boundary.py tests\test_learned_online_dense_surface.py tests\test_learned_online_update_loop.py tests\test_learned_online_no_sweep.py tests\test_learned_online_synthetic_gates.py tests\test_learned_online_ablation_gates.py tests\test_eval_path_constraints.py -q`
  - Result: `25 passed in 9.09s`.
- Combined targeted regression:
  - `python -m pytest tests\test_learned_online_claim_boundary.py tests\test_learned_online_dense_surface.py tests\test_learned_online_update_loop.py tests\test_learned_online_no_sweep.py tests\test_learned_online_synthetic_gates.py tests\test_learned_online_ablation_gates.py tests\test_action_spotlight.py tests\test_scientist_agent.py tests\test_scientist_training.py tests\test_arc_adapter_helpers.py tests\test_eval_path_constraints.py tests\test_arc_public_training.py -q`
  - Result: `110 passed in 127.47s`.

Current conclusion:

- No ARC success yet.
- The clean minimal path is preserved as scaffolding and gates, not as competence.
- Further ARC runs before recurrent synthetic gates would be misleading and would invite forbidden family/cost tuning.

## 2026-04-24 14:38 EDT

Operator actions:

1. Implemented `learned_online_recurrent_v1`:
   - `arcagi.learned_online.sequence`
   - `arcagi.learned_online.recurrent_model`
   - `arcagi.learned_online.recurrent_policy`
   - `arcagi.agents.learned_online_recurrent_agent`
2. Kept the same clean dependency boundary:
   - no Spotlight;
   - no TheoryManager;
   - no RuntimeRuleController;
   - no HybridPlanner;
   - no graph/frontier/search controller.
3. Added harder synthetic sequence tasks:
   - `VisibleMovementTrapTask`
   - `MovementRequiredAfterModeTask`
   - `DelayedUnlockTask`
4. Added recurrent boundary and hidden-state tests.
5. Added `scripts/train_learned_online_recurrent.py`.
6. Found a generic feature bug: unknown symbolic actions such as `trap` and `useful` collapsed to the same action features. Added a stable action-identity projection to the generic action feature vector.
7. Trained recurrent checkpoints:
   - `artifacts\learned_online_recurrent_synth_3000.pkl`: expanded curriculum `success_rate=0.8217`.
   - `artifacts\learned_online_recurrent_synth_10000.pkl`: expanded curriculum `success_rate=0.8381`.
   - `artifacts\learned_online_recurrent_synth_avail_10000.pkl`: expanded curriculum `success_rate=0.8385`.
   - `artifacts\learned_online_recurrent_synth_actionid_12000.pkl`: expanded curriculum `success_rate=0.8432`.
8. Heldout check for `learned_online_recurrent_synth_actionid_12000.pkl`:
   - `VisibleUsefulTrapTask`: `40/40`
   - `RandomizedBindingTask`: `40/40`
   - `DenseCoordinateGroundingTask`: `40/40`
   - `VisibleMovementTrapTask`: `40/40`
   - `MovementRequiredAfterModeTask`: `40/40`
   - `DelayedUnlockTask`: `40/40`

Validation:

- Recurrent/sequence focused tests:
  - `python -m pytest tests\test_learned_online_recurrent_gates.py tests\test_learned_online_sequence_curriculum.py tests\test_learned_online_claim_boundary.py tests\test_learned_online_dense_surface.py tests\test_learned_online_update_loop.py tests\test_learned_online_no_sweep.py tests\test_learned_online_synthetic_gates.py tests\test_learned_online_ablation_gates.py tests\test_eval_path_constraints.py -q`
  - Result: `32 passed in 9.03s`.
- Combined targeted regression:
  - `python -m pytest tests\test_learned_online_recurrent_gates.py tests\test_learned_online_sequence_curriculum.py tests\test_learned_online_claim_boundary.py tests\test_learned_online_dense_surface.py tests\test_learned_online_update_loop.py tests\test_learned_online_no_sweep.py tests\test_learned_online_synthetic_gates.py tests\test_learned_online_ablation_gates.py tests\test_action_spotlight.py tests\test_scientist_agent.py tests\test_scientist_training.py tests\test_arc_adapter_helpers.py tests\test_eval_path_constraints.py tests\test_arc_public_training.py -q`
  - Result: `117 passed in 127.87s`.

Current conclusion:

- Recurrent synthetic gates are now strong enough to justify one diagnostic 80-step ARC smoke.
- This is still not an ARC success claim.
- The next ARC run must be judged by objective progress first, and by learned-decision diagnostics second.

## 2026-04-24 14:46 EDT

ARC task success prototype:

1. Ran the first serious clean recurrent ARC attempt after recurrent synthetic gates passed:
   - `.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent learned_online_recurrent --checkpoint-path artifacts\learned_online_recurrent_synth_actionid_12000.pkl --game-id ar25-0c556536 --mode offline --max-steps 320 --progress-every 80 --trace-path artifacts\traces\learned_online_recurrent_synth12000_320.jsonl`
2. Result:
   - `success=true`
   - `won=false`
   - `return=1.0`
   - `levels_completed=1`
   - `steps=320`
   - `interaction_steps=37`
   - `reset_steps=2`
   - `dense_action_surface=true`
   - `scored_action_count=447`
   - `legal_action_count=447`
   - `controller_kind=learned_online_recurrent_v1`
   - `learned_online_controller=true`
   - `spotlight_feature_schema_version=0`
3. Interpretation:
   - This is a clean learned ARC task success prototype because the recurrent learned-online agent achieved real ARC level progress under the full dense action surface without Spotlight, RuntimeRuleController, HybridPlanner, TheoryManager, graph/frontier control, or scripted probe ownership.
   - It is not a full ARC game win because `won=false` and only `1` level was completed.
4. Follow-up already started:
   - A 1000-step continuation run is active to test cross-level carryover and full game completion:
     - `.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent learned_online_recurrent --checkpoint-path artifacts\learned_online_recurrent_synth_actionid_12000.pkl --game-id ar25-0c556536 --mode offline --max-steps 1000 --progress-every 100 --trace-path artifacts\traces\learned_online_recurrent_synth12000_1000.jsonl`

## 2026-04-24 15:12 EDT

ARC level-transition verification and 1000-step continuation:

1. The 1000-step continuation finished:
   - `success=true`
   - `won=false`
   - `return=1.0`
   - `levels_completed=1`
   - `steps=1000`
   - `interaction_steps=64`
   - `reset_steps=2`
   - `family_histogram={"move":747,"undo":187,"select":1,"click":63,"reset":2}`
   - `max_same_action_streak=183`
2. Interpretation:
   - The clean learned recurrent controller preserves the level-1 ARC task success prototype.
   - It does not yet win AR25 because it remains stuck after level 1.
   - The post-level failure mode is movement/undo-heavy action arbitration, not an environment reset to the same level.
3. Verified local ARC inventory:
   - OFFLINE mode has one cached local game: `ar25-0c556536`.
   - That game contains 8 internal levels: `win_score=8`, `_levels=8`, `_clean_levels=8`.
   - ONLINE mode can fetch 25 environment descriptors through the ARC API, but those are not local offline cached games.
4. Verified first level completion semantics by replaying the recorded action sequence:
   - step `299`, action `1`: `levels_completed` changes `0 -> 1`, `reward=1.0`, state remains `GameState.NOT_FINISHED`, no termination/truncation.
   - internal game `level_index` changes `0 -> 1`.
   - grid hash changes from reset `d1cf6c65a8c9510a` to post-boundary `23389de4e0a7e7b3`, then to `e793346c71d0d291` on the next level-1-index step.
   - Therefore the environment advances to the next internal AR25 level in the same session; it is not serving the same level again.
5. Fixed a repo CLI hygiene issue discovered during verification:
   - `python -m arcagi.cli list-arc-games` previously printed nothing because `arcagi.cli` did not call `main()` under module execution.
   - Added the module entrypoint so future inventory checks can use the CLI.

## 2026-04-24 15:43 EDT

GPT-Pro follow-up implementation batch:

1. Asked GPT-Pro to review the pushed level-1 success, 1000-step continuation failure, and verified AR25 level semantics.
2. Implemented the first validity-fix batch from the review:
   - pre-outcome recurrent feature snapshots;
   - hidden snapshots in memory/probe loss;
   - separate `pretrain_updates` from fresh `online_adapt_updates`;
   - `value` head;
   - partial sparse non-progress cost target;
   - retrospective current-level success credit;
   - session-global plus level-local belief stats;
   - recurrent policy diagnostics in traces;
   - mixed learner-owned training rollouts.
3. Validation:
   - focused learned-online tests: `19 passed in 0.35s`;
   - broad targeted regression: `122 passed in 139.96s`.
4. Synthetic result:
   - `artifacts\learned_online_recurrent_mixed_localbelief_2000.pkl`
   - mixed training success rate `0.985`
   - heldout synthetic gates all `40/40`, including the previously regressed movement-required-after-mode gate.
5. AR25 post-fix diagnostic:
   - command: `.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent learned_online_recurrent --checkpoint-path artifacts\learned_online_recurrent_mixed_localbelief_2000.pkl --game-id ar25-0c556536 --mode offline --max-steps 320 --progress-every 80 --trace-path artifacts\traces\learned_online_recurrent_mixed_localbelief_2000_ar25_320.jsonl`
   - result: `success=false`, `won=false`, `return=0.0`, `levels_completed=0`, `steps=320`
   - action histogram: `click:31:1=300`, `click:31:10=7`, `click:31:4=7`, `click:31:7=6`
   - family histogram: `click=320`
   - max same-action streak: `292`
   - final diagnostics: `all_negative_scores=true`, `mean_pred_cost=0.8813`, `mean_q_progress=0.0022`, `score_margin_top2=0.00158`, `score_entropy=6.076`
6. Current conclusion:
   - the original level-1 prototype remains logged as a real prior result;
   - the new validity-fixed checkpoint does not reproduce it;
   - the current failure mode is dense-click fixation under all-negative tiny-margin scores, so the next work must address learned policy calibration/exploration without scripted action-pattern search or dense-click caps.

## 2026-04-24 16:14 EDT

Factorized policy and dense-arbitration curriculum:

1. Asked GPT-Pro for a concrete next step after the validity-fixed checkpoint regressed into dense-click fixation.
2. Implemented the agreed next patch:
   - clean recurrent action features now disable exact full-action-string projection;
   - recurrent policy now defaults to factorized full-support stochastic selection;
   - family probability is count-normalized and action probability is conditional on family;
   - greedy remains diagnostic only;
   - added dense-family arbitration, mode-then-dense-click, and action-name-remap synthetic tasks;
   - added tests for full support, order equivariance, family count normalization, non-sweep behavior, and exact-action projection hygiene.
3. Validation:
   - focused tests: `27 passed in 11.99s`;
   - broad targeted regression: `130 passed in 237.90s`.
4. GPT-Pro requested flat-score data:
   - with `400` clicks plus reset/move/select/undo, factorized probabilities are `0.2` for each family;
   - total click mass is `0.2`;
   - individual click probability is `0.0005`.
5. Expanded synthetic checkpoint:
   - `artifacts\learned_online_recurrent_factorized_densemix_3000.pkl`
   - train success rate `0.99`
   - heldout seeds `80..119`: all nine tasks `40/40`
6. AR25 80-step diagnostic:
   - command: `.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent learned_online_recurrent --checkpoint-path artifacts\learned_online_recurrent_factorized_densemix_3000.pkl --game-id ar25-0c556536 --mode offline --max-steps 80 --progress-every 20 --trace-path artifacts\traces\learned_online_recurrent_factorized_densemix_3000_ar25_80.jsonl`
   - result: `success=false`, `won=false`, `return=0.0`, `levels_completed=0`
   - family histogram: `click=26`, `select=36`, `undo=10`, `move=8`
   - max same-action streak: `5`
   - final diagnostics: `all_negative_scores=true`, `mean_pred_cost=0.4991`, `mean_q_progress=0.00171`, `score_entropy=6.0918`, `score_margin_top2=0.00592`
7. Current conclusion:
   - dense-click fixation is fixed;
   - AR25 still fails with all-negative low-progress scoring;
   - next target is value/progress calibration and harder sparse delayed mixed-family curriculum, not action coverage.

## 2026-04-24 19:20 EDT

Learned recurrent ARC recovery regression and GPT-Pro escalation context:

1. Implemented the next recovery batch after factorized-policy diagnostics:
   - blended long-horizon return credit;
   - return credit trains `value`, not immediate `useful`;
   - observed-transition value ranking;
   - long sparse/carryover synthetic tasks;
   - trace fine-tuning script for offline ARC experience;
   - imitation/action-prior head for recurrent policy scoring.
2. Validation:
   - focused recurrent/sequence tests: `31 passed in 3.26s`;
   - broad targeted regression before imitation-head patch: `140 passed in 134.24s`.
3. Expanded synthetic checkpoint:
   - `artifacts\learned_online_recurrent_longcredit_1500.pkl`
   - train success rate `0.832`;
   - old nine heldout tasks remain near-solved;
   - post-boundary carryover heldout `19/20`;
   - arbitrary long sparse chain and dense decoy heldouts `0/20`, so these tasks exposed a long-horizon sequence-learning gap.
4. AR25 learned-agent diagnostics:
   - `artifacts\learned_online_recurrent_longcredit_1500.pkl`: failed `AR25` at 320 steps, `return=0.0`, `levels_completed=0`.
   - old `artifacts\learned_online_recurrent_synth_actionid_12000.pkl` under current clean runtime: failed `AR25` at 320 steps, `return=0.0`, `levels_completed=0`.
   - greedy and exact-action-projection diagnostic variants also failed.
5. Trace integrity check:
   - the original success trace `artifacts\traces\learned_online_recurrent_synth12000_320.jsonl` replays to `return=1.0`, `levels_completed=1` only when the diagnostic replay agent is not reset on in-game reset actions.
   - This validates the trace/environment but remains invalid as a solver. It is usable only as labeled offline ARC experience.
6. Offline trace fine-tuning results:
   - return-credit-only trace fine-tune reached teacher success `12/12` but normal learned-agent eval still failed.
   - dense all-negative imitation fine-tune produced `artifacts\learned_online_recurrent_longcredit_traceft_imitation_12.pkl` with teacher success `12/12`.
   - normal learned-agent eval of that checkpoint still failed: `success=false`, `return=0.0`, `levels_completed=0`, final `GameState.GAME_OVER`, full dense scoring `448/448`, family histogram `select=109`, `move=84`, `undo=58`, `click=67`, `reset=2`.
   - Final diagnostics showed reset over-weighting: family probability for reset near `0.61`; imitation signal remained weak (`mean_q_imitation ~= 0.001`).
7. Current conclusion:
   - the earlier level-1 behavior is not recovered by value credit or dense imitation;
   - the current scalar-head / heuristic-feature recurrent learner is not sufficient to robustly imitate or rediscover the validated trace;
   - GPT-Pro must review the pushed updated code and advise whether to continue with sampled/hard-negative action-prior recovery or pivot to a larger mechanism-level architecture.

## 2026-04-25 Parametric Object-Event Implementation And Diagnostic

Operator actions recorded for continuity:

1. Implemented the GPT-Pro-approved parametric object-event patch:
   - `arcagi/learned_online/object_event_curriculum.py`
   - `arcagi/learned_online/object_event_model.py`
   - `scripts/train_learned_online_object_event.py`
   - `arcagi/agents/learned_online_object_event_agent.py`
   - `tests/test_learned_online_object_event_agent_contracts.py`
   - `tests/test_object_event_parametric_action_surface.py`
   - `TASKS.MD`
   - `CONTEXT.MD`
2. Ran verification:
   - focused parametric/runtime/agent tests: `31 passed`
   - full object-event suite: `77 passed`
   - recurrent regression: `31 passed`
3. Ran the fast 447-action synthetic gate:
   - command used `--state-source extracted --action-surface arc_scale_parametric --action-surface-size 447 --coordinate-grid-size 64 --max-steps-per-level 5`
   - best checkpoint at step `120`
   - true runtime within-5 `0.84375`
   - failed exact/near repeat rates both `0.0`
4. Ran the 447-action promotion gate:
   - best checkpoint at step `200`
   - true runtime within-5 `0.8333333333333334`
   - next-level first try `0.8125`
   - failed exact repeat `0.0`
   - failed near repeat `0.10714285714285714`
   - max same-action streak `3.0`
   - artifact: `artifacts/object_event_parametric_runtime_probe.pkl`
5. Ran the dense 68-action regression:
   - final true runtime within-5 `1.0`
   - next-level first try `0.7083333333333334`
   - no metadata leakage
6. Ran the real ARC hygiene diagnostic:
   - command: `.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent learned_online_object_event --checkpoint-path artifacts/object_event_parametric_runtime_probe.pkl --mode offline --game-limit 1 --max-steps 48 --progress-every 8 --object-event-bridge-diagnostics`
   - result: no win, no level completion, full `447/447` scoring, no forbidden controller/oracle/replay/metadata leakage
   - failure signature: repeated coordinate-column clicks around `click:31:31`, `click:31:34`, `click:31:37`; `max_same_action_streak=11`
7. Operator constraint update:
   - the `48`-step ARC run is only a harsh hygiene cutoff for collapse/forbidden behavior
   - it must not be treated as a fair competence budget for a game that may require online learning over hundreds of steps
   - future real ARC checks should combine short hygiene probes with longer acquisition probes after collapse behavior is fixed
