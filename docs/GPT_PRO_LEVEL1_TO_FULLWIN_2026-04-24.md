# GPT-Pro Level-1 To Full-Win Review, 2026-04-24

Thread: `https://chatgpt.com/c/69eb9ed1-e514-8325-bc1b-2c503c8c2c25`

## Input State

- Latest pushed verification before the review: `f0e84a5`.
- Clean learned recurrent controller had one real AR25 level-progress prototype:
  - checkpoint: `artifacts/learned_online_recurrent_synth_actionid_12000.pkl`
  - `320` steps on `ar25-0c556536`
  - `success=true`
  - `won=false`
  - `return=1.0`
  - `levels_completed=1`
  - dense action surface: `447/447` scored/legal actions
- 1000-step continuation did not win:
  - `success=true`
  - `won=false`
  - `return=1.0`
  - `levels_completed=1`
  - `move=747`, `undo=187`, `click=63`
  - max same-action streak `183`
- Verified AR25 semantics:
  - local offline cache has one game, `ar25-0c556536`
  - AR25 has `8` internal levels
  - first reward advances internal level index `0 -> 1`
  - the environment does not replay the same level after completion

## GPT-Pro Diagnosis

GPT-Pro identified these likely blockers:

1. Recurrent path was more like a random recurrent feature reservoir plus online linear heads than a trained recurrent world model.
2. `on_transition()` used outcome-mutated belief features for the same transition update, inflating synthetic gates.
3. Loaded checkpoints inherited the synthetic `fast_heads.updates` count, so ARC online adaptation started with a learning rate decayed by roughly `89555` updates.
4. Memory probe loss reused old feature entries with the current hidden state, corrupting realized information gain.
5. Cost target treated most sparse non-progress as full cost, driving all-negative post-boundary scores.
6. No retrospective value credit existed for long pre-reward action sequences.
7. No session-global vs level-local belief partition existed, so level-1 local evidence contaminated later levels.
8. Training data came from random rollouts only, not mixed learner-owned rollouts.
9. Synthetic tasks overused semantically transparent action names and did not sufficiently test cross-level carryover or action-id remapping.

## Implemented First Fix Batch

Implemented:

- recurrent update now snapshots pre-outcome feature and hidden before belief/memory mutation
- fast-head learning-rate decay now uses `online_adapt_updates`, reset on episode load/reset, while preserving `pretrain_updates`
- memory entries now store hidden snapshots, level epoch/step, and return credit
- recurrent probe loss now evaluates memory entries against their stored hidden snapshots
- added `value` head
- lowered sparse no-progress cost target from full cost to partial cost; harm remains full cost
- added retrospective current-level success credit
- added session-global and level-local belief stats
- added local belief priors so same-level evidence affects scoring
- added policy diagnostics into traces:
  - top scores
  - score entropy
  - score margin
  - all-negative-score flag
  - mean predicted cost/progress/info
  - level epoch/step
  - pretrain/online adaptation counters
- fixed the minimal scaffold to use the same pre-outcome update ordering
- changed `scripts/train_learned_online_recurrent.py` from random-only to configurable `random`, `agent`, `epsilon_agent`, and `mixed` behavior policies

## Validation

Focused tests:

```powershell
python -m pytest tests\test_learned_online_recurrent_gates.py tests\test_learned_online_update_loop.py tests\test_learned_online_synthetic_gates.py tests\test_learned_online_ablation_gates.py -q
```

Result: `19 passed in 0.35s`.

Broad targeted regression:

```powershell
python -m pytest tests\test_learned_online_recurrent_gates.py tests\test_learned_online_sequence_curriculum.py tests\test_learned_online_claim_boundary.py tests\test_learned_online_dense_surface.py tests\test_learned_online_update_loop.py tests\test_learned_online_no_sweep.py tests\test_learned_online_synthetic_gates.py tests\test_learned_online_ablation_gates.py tests\test_action_spotlight.py tests\test_scientist_agent.py tests\test_scientist_training.py tests\test_arc_adapter_helpers.py tests\test_eval_path_constraints.py tests\test_arc_public_training.py -q
```

Result: `122 passed in 139.96s`.

Checkpoint counter check:

- old checkpoint loads with `pretrain_updates=89555`
- old checkpoint loads with `online_adapt_updates=0`
- first eval update increments `online_adapt_updates` to `1`

## Revised Synthetic Training

Random-only 1200-episode smoke after validity fixes:

- checkpoint: `artifacts\learned_online_recurrent_validityfix_smoke_1200.pkl`
- `success_rate=0.8583`
- heldout:
  - visible useful trap: `40/40`
  - randomized binding: `40/40`
  - dense coordinate grounding: `40/40`
  - visible movement trap: `40/40`
  - movement-required-after-mode: `0/40`
  - delayed unlock: `40/40`

Mixed-policy 1200-episode smoke:

- checkpoint: `artifacts\learned_online_recurrent_mixed_smoke_1200.pkl`
- `success_rate=0.9742`
- movement-required-after-mode still `0/40`

Mixed-policy plus local-belief-prior 2000-episode smoke:

- checkpoint: `artifacts\learned_online_recurrent_mixed_localbelief_2000.pkl`
- `success_rate=0.985`
- behavior counts:
  - `agent=2244`
  - `epsilon_agent=868`
  - `epsilon_random=212`
  - `random=2233`
- heldout:
  - visible useful trap: `40/40`, avg steps `1.0`
  - randomized binding: `40/40`, avg steps `2.525`
  - dense coordinate grounding: `40/40`, avg steps `1.0`
  - visible movement trap: `40/40`, avg steps `1.0`
  - movement-required-after-mode: `40/40`, avg steps `4.0`
  - delayed unlock: `40/40`, avg steps `2.0`

## Post-Fix AR25 Diagnostic

Command:

```powershell
.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent learned_online_recurrent --checkpoint-path artifacts\learned_online_recurrent_mixed_localbelief_2000.pkl --game-id ar25-0c556536 --mode offline --max-steps 320 --progress-every 80 --trace-path artifacts\traces\learned_online_recurrent_mixed_localbelief_2000_ar25_320.jsonl
```

Result:

- `success=false`
- `won=false`
- `return=0.0`
- `levels_completed=0`
- `steps=320`
- `interaction_steps=320`
- `reset_steps=0`
- `family_histogram={"click":320}`
- `action_histogram={"click:31:1":300,"click:31:10":7,"click:31:4":7,"click:31:7":6}`
- max same-action streak `292`
- final diagnostics:
  - `all_negative_scores=true`
  - `mean_pred_cost=0.8813`
  - `mean_q_progress=0.0022`
  - `score_margin_top2=0.00158`
  - `score_entropy=6.076`
  - `level_epoch=0`
  - `level_step=320`

Interpretation:

- The validity fixes and mixed-policy training repaired synthetic leakage/counter issues and restored heldout synthetic gates.
- The new checkpoint regressed the original AR25 level-1 success.
- The new failure mode is dense-click fixation under all-negative tiny-margin scores, not the older post-level movement/undo carryover failure.
- The next target is learned policy calibration/exploration under all-negative uncertainty without introducing forbidden action-pattern search or dense-click caps.

## GPT-Pro Follow-Up And Factorized Policy Patch

GPT-Pro reviewed the post-validity AR25 regression and diagnosed the new failure as a greedy-policy stop condition:

- the score surface was all-negative and nearly flat;
- `score_entropy=6.076` was close to `ln(447)`;
- `score_margin_top2=0.00158`;
- deterministic greedy argmax therefore turned arbitrary tiny score differences into a repeated dense click.

Implemented:

- disabled exact full-action-string stable projection for the clean recurrent action features while preserving role/family/geometry features;
- kept feature dimensionality unchanged by zeroing the exact-action projection slot;
- added factorized full-support stochastic selection for the recurrent policy:
  - every legal action is still scored;
  - every legal action has nonzero probability;
  - family probability is count-normalized through log-mean-exp aggregation;
  - action probability is `p(family) * p(action | family)`;
  - greedy remains available as a diagnostic mode;
- added policy diagnostics:
  - selection mode;
  - selected action probability;
  - selected family probability;
  - effective action/family support;
  - family/action temperatures;
  - family probabilities;
  - action feature config;
- added synthetic gates:
  - `DenseFamilyMassArbitrationTask`;
  - `ModeThenDenseClickTask`;
  - `ActionNameRemapHeldoutTask`;
- expanded recurrent trainer to sample nine tasks.

Focused tests:

```powershell
python -m pytest tests\test_learned_online_recurrent_gates.py tests\test_learned_online_sequence_curriculum.py tests\test_learned_online_update_loop.py tests\test_learned_online_no_sweep.py -q
```

Result: `27 passed in 11.99s`.

Broad targeted regression:

```powershell
python -m pytest tests\test_learned_online_recurrent_gates.py tests\test_learned_online_sequence_curriculum.py tests\test_learned_online_claim_boundary.py tests\test_learned_online_dense_surface.py tests\test_learned_online_update_loop.py tests\test_learned_online_no_sweep.py tests\test_learned_online_synthetic_gates.py tests\test_learned_online_ablation_gates.py tests\test_action_spotlight.py tests\test_scientist_agent.py tests\test_scientist_training.py tests\test_arc_adapter_helpers.py tests\test_eval_path_constraints.py tests\test_arc_public_training.py -q
```

Result: `130 passed in 237.90s`.

Flat-score data point requested by GPT-Pro:

- action surface: `400` clicks plus reset/move/select/undo
- family probabilities:
  - `click:none=0.2`
  - `reset:none=0.2`
  - `move:none=0.2`
  - `select:none=0.2`
  - `undo:none=0.2`
- total click mass: `0.2`
- one click probability: `0.0005`

Old checkpoint with exact action projection disabled:

- `artifacts\learned_online_recurrent_mixed_localbelief_2000.pkl` still passes the old six heldout gates at `40/40` each.

Expanded factorized synthetic training:

```powershell
.venv313\Scripts\python.exe -m scripts.train_learned_online_recurrent --output artifacts\learned_online_recurrent_factorized_densemix_3000.pkl --episodes 3000 --max-steps 16 --seed 94 --behavior-policy mixed
```

Result:

- `success_rate=0.99`
- `avg_return=0.99`
- `model_updates=23708`

Heldout result over seeds `80..119`:

- visible useful trap: `40/40`, avg steps `1.325`
- randomized binding: `40/40`, avg steps `2.525`
- dense coordinate grounding: `40/40`, avg steps `1.0`
- visible movement trap: `40/40`, avg steps `1.05`
- movement-required-after-mode: `40/40`, avg steps `5.925`
- delayed unlock: `40/40`, avg steps `2.0`
- dense family mass arbitration: `40/40`, avg steps `1.075`
- mode then dense click: `40/40`, avg steps `2.625`
- action name remap heldout: `40/40`, avg steps `2.25`

AR25 80-step diagnostic:

```powershell
.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent learned_online_recurrent --checkpoint-path artifacts\learned_online_recurrent_factorized_densemix_3000.pkl --game-id ar25-0c556536 --mode offline --max-steps 80 --progress-every 20 --trace-path artifacts\traces\learned_online_recurrent_factorized_densemix_3000_ar25_80.jsonl
```

Result:

- `success=false`
- `won=false`
- `return=0.0`
- `levels_completed=0`
- `steps=80`
- `interaction_steps=62`
- family histogram:
  - `click=26`
  - `select=36`
  - `undo=10`
  - `move=8`
- max same-action streak: `5`
- final diagnostics:
  - `selection_mode=factorized_softmax`
  - `all_negative_scores=true`
  - `mean_pred_cost=0.4991`
  - `mean_q_progress=0.00171`
  - `mean_q_info=0.00197`
  - `score_margin_top2=0.00592`
  - `score_entropy=6.0918`
  - family probabilities:
    - `select:none=0.5842`
    - `click:none=0.1645`
    - `move:none=0.1582`
    - `undo:none=0.0930`

Interpretation:

- Factorized full-support stochastic selection fixed the dense-click fixation.
- It did not solve AR25 level 1.
- The current failure is still all-negative low-progress scoring under sparse reward, now with diverse mixed-family action selection.
- The next correction should target value/progress calibration and harder sparse delayed mixed-family curriculum, not action-surface coverage or deterministic tie-breaking.
