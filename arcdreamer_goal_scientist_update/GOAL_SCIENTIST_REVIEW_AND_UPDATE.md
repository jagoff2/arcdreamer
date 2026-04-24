# ARC Dreamer goal-scientist review, training update, and code handoff

Date: 2026-04-22

## Executive result

I reviewed the current public GitHub state with the available web tools, then attempted to clone and run the repository locally. The live clone failed in the sandbox because the execution container could not resolve `github.com`. Because of that DNS failure, I could not run the repository's full test suite, install the optional `arc-agi` toolkit, or train/evaluate against official ARC-AGI-3 games in this environment.

I did not verify an official ARC-AGI-3 score and I am not claiming the agent now beats ARC-AGI-3. The current repo README still reports the official public online five-game slice as `0/5`, while also reporting synthetic success for the recurrent and hybrid agents. The patch below is therefore aimed at the next real blocker: converting the public ARC training loop from sparse, shallow transition fitting into a goal-scientist training curriculum that teaches delayed selector/control binding, diagnostic probing, causal/effect classification, and policy usefulness under black-box interaction.

The code returned here is a drop-in update:

1. `arcagi/training/goal_scientist_targets.py`
2. `arcagi/training/goal_scientist_public.py`
3. `tests/test_goal_scientist_targets.py`
4. `docs/GOAL_SCIENTIST_TRAINING.md`

It preserves the existing checkpoint format consumed by the current harness: `encoder`, `world_model`, and `language_model` state dictionaries.

## Review findings

### Current repo status

The repository is now an integrated ARC-AGI-3 attack stack rather than the earlier minimal scientist-agent patch. It includes object-centric extraction, explicit graph state tracking, recurrent world model heads, episodic memory, grounded language, synthetic training, public ARC training, and an optional ARC toolkit adapter.

The README reports:

- synthetic full-slice success of `1.0` for `recurrent` and `hybrid`;
- a long manual Stage 1 run reaching `hidden_modes` by epoch 4;
- partial hidden-mode competence, with `running_success_rate = 0.45072` at that promoted epoch-4 snapshot;
- current public online five-game ARC slice still `0/5`;
- the known current synthetic bottleneck is held-out compositional control / sequence binding, especially `selector_sequence_unlock`.

That status implies the remaining problem is not merely model capacity. The missing training pressure is the online scientist behavior ARC-AGI-3 actually rewards: use first-contact probes, notice delayed effects, bind selectors or contact actions to later state transitions, and avoid repeating stale no-effect actions.

### Current public trainer bottleneck

The existing `arcagi.training.arc_public` loop collects transitions and trains the recurrent world model, policy head, and language model. However, the inspected loop uses simple usefulness of the form approximately:

```python
usefulness = max(reward, 0.0) + 0.5 * delta_norm
```

and then derives a hard binary policy target from a threshold. This is too local for ARC-AGI-3. Public ARC success often requires actions that look locally useless but become useful as prefixes: click/select, interact, bind, move, then reward or visible state change. If the trainer labels the first prefix action as a no-effect negative, it actively trains the runtime agent away from the very experiments it needs.

The current world model already exposes richer heads: `return_value`, `usefulness`, `policy`, `causal_value`, `diagnostic_value`, and `effect_logits`. The existing public ARC trainer was not using that full structure strongly enough on public ARC transitions. The new trainer does.

### ARC-AGI-3 implications

ARC-AGI-3 is interactive. The official benchmark description emphasizes exploration, percept-plan-action loops, memory, goal acquisition, and alignment rather than static one-shot puzzle transformation. The action interface is also generic: directional actions, a generic action, and a coordinate action. A useful agent must treat those actions as experimental interventions, not as fixed per-game semantics.

That is why this update deliberately avoids game IDs, environment files, and hidden engine semantics. The new diagnostic collector uses only visible grids and advertised actions, and the relabeler uses only reward, terminal status, state-change vectors, action text, and hindsight from later black-box transitions.

## What the patch changes

### 1. Hindsight target construction

`goal_scientist_targets.py` introduces a pure-Python relabeling layer. It takes chronological episode samples and adds:

- `usefulness`
- `policy_target`
- `diagnostic_weight`
- `transition_credit`
- `credit_tags`
- `belief_tokens`
- `question_tokens`
- `plan_tokens`
- `hindsight_return`
- `future_event_distance`

The relabeler gives positive delayed credit to selector/contact prefixes when a future reward or state change occurs within a configurable horizon. This directly targets the repo's stated sequence-binding bottleneck.

It also penalizes repeated stale no-effect actions while preserving first-time no-effect probes as useful diagnostics. That distinction matters: a first null result is information; the tenth identical null result is waste.

### 2. Public ARC goal-scientist trainer

`goal_scientist_public.py` is a replacement/supplement for `arc_public.py`. It keeps the checkpoint format compatible with the existing harness and current agents. It adds:

- a `GoalScientistPublicTrainingConfig`;
- a `PublicDiagnosticAgendaAgent` collector;
- episode-level hindsight relabeling;
- causal target supervision;
- diagnostic target supervision;
- effect-kind supervision using the current model's `EFFECT_KINDS`;
- soft policy targets instead of a brittle usefulness threshold;
- sample weighting for high-information transitions;
- JSON epoch progress logs;
- interrupt-independent epoch snapshots.

The diagnostic collector is deliberately generic. It prefers coordinate probes when coordinate actions exist, then interaction actions, then motion/action contrasts. If the adapter exposes `click:x:y` actions, it uses those. If a parametric coordinate action is available, it synthesizes `click:x:y` probes from visible grid salience, such as color centroids and corners. This follows the current adapter convention that `click:x:y` maps to `ACTION6` plus coordinate parameters.

### 3. Tests

`tests/test_goal_scientist_targets.py` verifies the pure relabeling logic:

- ARC/toolkit action spellings map to action families;
- a no-effect click before a later reward is relabeled as a useful sequence prefix;
- repeated no-effect probes become hard negatives;
- summary metrics report selector/no-effect fractions.

Because the sandbox could not clone the live repo, I ran the pure module tests manually against the returned patch tree instead of the full repository suite.

## Validation performed in this sandbox

The execution container could not clone GitHub:

```text
fatal: unable to access 'https://github.com/jagoff2/arcdreamer.git/': Could not resolve host: github.com
```

I therefore validated the returned files directly.

Syntax check:

```text
/usr/bin/python3 -m py_compile \
  /mnt/data/arcdreamer_goal_scientist_update/arcagi/training/goal_scientist_targets.py \
  /mnt/data/arcdreamer_goal_scientist_update/arcagi/training/goal_scientist_public.py
py_compile_ok
```

Manual execution of the target tests:

```text
test_action_family_accepts_arc_and_adapter_spellings ok
test_hindsight_relabels_no_effect_selector_as_sequence_prefix ok
test_repeated_no_effect_probe_becomes_hard_negative ok
test_summary_reports_selector_and_no_effect_fractions ok
manual tests passed
```

I could not run ARC-AGI-3 training or scorecard evaluation in this sandbox. The public trainer requires the optional ARC toolkit and local cached environments or online credentials.

## Integration instructions

From the repository root:

```bash
unzip -o arcdreamer_goal_scientist_update.zip
python -m pytest tests/test_goal_scientist_targets.py
python -m py_compile arcagi/training/goal_scientist_targets.py arcagi/training/goal_scientist_public.py
```

Install ARC support in a Python 3.12+ environment, matching the repo's current README guidance:

```bash
python -m pip install -e .[dev,arc]
```

Run a short smoke collection/training pass locally:

```bash
python -m arcagi.training.goal_scientist_public \
  --mode offline \
  --game-limit 5 \
  --episodes-per-game 2 \
  --max-steps 96 \
  --epochs 1 \
  --init-checkpoint-path artifacts/manual_stage1.pt \
  --checkpoint-path artifacts/goal_scientist_public_smoke.pt \
  --behavior-policies hybrid,diagnostic,graph
```

Run the stronger public-ARC adaptation pass:

```bash
python -m arcagi.training.goal_scientist_public \
  --mode offline \
  --game-limit 25 \
  --episodes-per-game 16 \
  --max-steps 256 \
  --epochs 64 \
  --learning-rate 1e-4 \
  --init-checkpoint-path artifacts/manual_stage1.pt \
  --checkpoint-path artifacts/goal_scientist_public.pt \
  --behavior-policies hybrid,diagnostic,graph,learned,random \
  --hindsight-gamma 0.92 \
  --sequence-horizon 14 \
  --max-coordinate-probes 18
```

Evaluate with the current harness, using the existing hybrid path and the new checkpoint:

```bash
python -m arcagi.evaluation.harness \
  --agent hybrid \
  --mode offline \
  --game-limit 5 \
  --max-steps 256 \
  --checkpoint-path artifacts/goal_scientist_public.pt
```

Then run the same evaluation in online mode only when ready to generate an official scorecard:

```bash
python -m arcagi.evaluation.harness \
  --agent hybrid \
  --mode online \
  --game-limit 5 \
  --max-steps 256 \
  --checkpoint-path artifacts/goal_scientist_public.pt
```

## Recommended training protocol

1. Continue using the existing synthetic Stage 1 trainer until hidden-mode holdout is materially stable and teacher relabel dependence is lower. The repo's own README says the synthetic blocker has moved to hidden-mode stability and learner ownership.

2. Use `goal_scientist_public.py` as Stage 2. It should initialize from the best Stage 1 checkpoint and train on local public ARC environments. Use `hybrid,diagnostic,graph,learned,random` collection. The diagnostic policy is not a deployed controller; it is data augmentation by black-box experimentation.

3. Keep evaluation generic. Use the existing `hybrid` agent with the new checkpoint. Do not deploy the diagnostic collector as the ARC-facing agent unless you explicitly want a training-data collector rather than a learned runtime agent.

4. Track these metrics from the new trainer logs:

   - `credit_selector_binding_fraction`: should increase from near zero on games with coordinate/select interfaces.
   - `credit_no_effect_fraction`: should remain high enough to learn hard negatives, but not dominate all samples.
   - `credit_avg_policy_target`: should not collapse to zero.
   - `uncertainty`: should fall over epochs without eliminating useful exploration.
   - official/evaluation success: this is the only metric that matters after training.

5. If the official score remains `0/5`, inspect per-game traces for:

   - zero interaction/click count;
   - repeated no-effect action family;
   - selector click followed by no systematic contrast;
   - reward after a prefix not being propagated backward;
   - coordinate probes outside visible object centroids.

This patch addresses those failure classes in the training loop. It does not guarantee that the current model architecture is sufficient for all ARC-AGI-3 games.

## Files added

```text
arcagi/training/goal_scientist_targets.py
arcagi/training/goal_scientist_public.py
tests/test_goal_scientist_targets.py
docs/GOAL_SCIENTIST_TRAINING.md
```

No existing files must be edited for the patch to run. If you want to expose it in project docs or CLI menus, add a README command block pointing to `python -m arcagi.training.goal_scientist_public`.
