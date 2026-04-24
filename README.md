# arcagi

`arcagi` is a local ARC-AGI-3 attack stack aimed at consumer hardware. The design is explicitly hybrid:

- object-centric state extraction over grid observations
- explicit world-state graph for novelty, transitions, and search
- compact recurrent world model with uncertainty via ensemble disagreement, explicit causal-effect classification, and diagnostic value heads
- episodic memory for one-shot mechanic recall
- grounded internal language for hypotheses, diagnostic questions, theory summaries, and plan sketches

The repository now has a runnable core with synthetic hidden-rule benchmarks, strong baselines, and an optional adapter for the official ARC toolkit. The current priority is turning that core into a genuinely online experimental scientist instead of a narrow synthetic specialist.

Runtime requirement:

- the final agent must learn online while it plays
- checkpoint weights may provide priors, but the deployed learned eval path must update memory, action preferences, and planning state from its own transitions within the episode without consulting a hand-authored controller
- ARC-facing control logic should survive transfer to non-text games with similar observation/action structure
- ARC must be treated as a black-box runtime environment: no per-game tuning, no environment-file semantics, and no external task knowledge in the runtime path
- explicit rule-object containers are allowed, but their contents must be induced online from current observations and transitions

Hard forbidden mechanisms:

- no hand-coded action search patterns as the ARC-facing solver
- no fixed movement sweeps, counted movement probes, canned reset/replay loops, or action-pattern enumerators
- no graph search, shortest-path frontier expansion, BFS/DFS replay, or coverage search as the full hybrid learned agent's controller
- no "generic" scripted controller that solves by manually trying action schedules, even when it is not keyed by game id
- no pre-training action-space narrowing in the ARC-facing train/eval path: legal actions, including dense click parameters, must be exposed to the learner unless an explicitly labeled smoke/debug mode opts out
- graphs are allowed only as memory, diagnostics, retrieval substrate, and explicit baselines/ablations
- any ARC success produced by a forbidden mechanism is invalid for the core goal and must be labeled as such

Minimum online-learning prerequisites:

- maintain a live belief state over observed objects, relations, latent variables, goal hypotheses, action meanings, uncertainty, and self/action history
- close the same-session learning loop: predict, act, observe, compute surprise/error, update beliefs/model/policy, and change later action choice during the same run
- choose actions from learned or online-updated value, uncertainty, diagnostic utility, and world-model predictions, not scripted action-pattern enumeration
- generate internal questions from model uncertainty or rival hypotheses and convert them into diagnostic actions through learned/evidence-updated mechanisms
- use counterfactual/imagination only through learned dynamics or learned hypotheses rooted in observed evidence, not through a scripted search tree
- train and adapt on the full currently legal action surface before relying on learned pruning, learned confidence, or learned action abstraction
- prove small-data adaptation: a few observed transitions must measurably change beliefs and action preferences inside the same episode/session

Detailed runtime-learning design:

- see [docs/ONLINE_RUNTIME_LEARNING.md](</G:/arcagi/docs/ONLINE_RUNTIME_LEARNING.md>)
- see [docs/MINIMALLY_CONSCIOUS_ONLINE_AGENT.md](</G:/arcagi/docs/MINIMALLY_CONSCIOUS_ONLINE_AGENT.md>) for the operational threshold the learned online agent must satisfy
- see [docs/CHECKPOINT_LINEAGE.md](</G:/arcagi/docs/CHECKPOINT_LINEAGE.md>) for scientist checkpoint/runtime schema boundaries
- that document now defines:
  - what meaningful online self-correction means
  - how to learn "the rules" online from transitions
  - why counterfactual replay matters
  - the three learning timescales
  - a diffusion-inspired hypothesis-to-action-to-learning-to-proof loop

## Status

- integrity reset after the latest artifact audit:
  - the current `*_v3.pkl` scientist artifacts are not proven autonomous ARC-capable artifacts
  - newest local v3 probes still show zero teacher-free synthetic solves and zero capped AR25 progress
  - scientist training now writes auditable checkpoint metadata that separates learner-owned train metrics, sparse oracle relabeling rates, and autonomous holdout metrics
  - "best" scientist checkpoints are no longer promoted by default when all autonomous solve metrics are zero
  - the scientist world model is now an online recurrent bootstrap model rather than a linear-only head over hashed state/action features
  - the new post-integrity smoke artifact `artifacts/post_integrity_scientist_latest.pkl` was collected with learner-owned actions plus sparse oracle relabeling, was not promoted, and still scores zero on the bounded autonomous holdout and short AR25 probe
- core package scaffold: implemented
- generic graph baseline: implemented
- learned hybrid baseline: implemented
- synthetic benchmark: research-only instrumentation
- ARC toolkit adapter: validated against the official toolkit in an isolated Python `3.13` env
- current default graph baseline: generic within-episode rule induction, no ARC-shaped controller heuristics
- current official public online slice for the default graph baseline: `0/5` after deleting the old handcrafted controller
- current learned synthetic result after the runtime object-hypothesis controller landed:
  - `recurrent`: `1.0` success rate on the full synthetic eval slice
  - `hybrid`: `1.0` success rate on the full synthetic eval slice
- scientist-agent patch status:
  - the hypothesis-driven `arcagi.scientist` stack is now integrated, checkpointable, and runnable through the shared evaluation harness
  - the current scientist curriculum trainer is session-based and now trains a checkpointed spotlight stack with learned `habit`, `executive`, and `adaptation` scorers on top of the online world model and spotlight priors
  - the current scientist trainer now uses learner-owned rollouts by default; synthetic oracle labels are sparse bootstrap instrumentation and are tracked as query/feedback fractions
  - real offline ARC now has a genuine positive signal: the preserved legacy `spotlight_exec_curriculum_best_legacy_v1.pkl` checkpoint can complete `AR25` level 1 under an uncapped local harness run, but post-level transfer is still weak
- current runtime-learning status:
  - the learned agent now beats the synthetic benchmark through live runtime rule learning and explicit online hypothesis competition
  - the clean `HybridAgent` no longer constructs the hand-authored `RuntimeRuleController`; controller-backed runs are bootstrap/baseline evidence only
  - the runtime path now records proof and exception objects, executes online split / merge / relabel / rebind repair in the working structured state, and writes local model edits within the episode
  - control, objective, and selector-mode competition now use normalized posteriors over rival executable theories instead of only heuristic utility ranking
  - temporary options are induced online from successful control sequences such as path-to-objective chains, selector-followed-by-move behaviors, and bind-then-objective programs
  - the Stage 1 training loop now optimizes explicit causal/theory structure directly:
    - recurrent world model heads predict `effect_kind`, `causal_value`, and `diagnostic_value`
    - replay and dream phases train `theory` and `diagnostic` language modes explicitly
    - event-conditioned supervision and hindsight credit now shape usefulness/policy targets instead of dropping event semantics on the floor
  - dense warm-start supervision is now learner-owned by default:
    - dense supervision no longer creates full teacher-owned episodes
    - current CLI and programmatic defaults are aligned at `teacher_episode_fraction=0.1` and `teacher_takeover_prob=0.25`
    - a tiny end-to-end flat CLI run after the fix recorded `teacher_episode_count=0`, `teacher_step_fraction=0.0823`, and `teacher_relabel_fraction=0.6013`
  - a long manual Stage 1 run at `25,000` episodes per epoch cleared foundation and reached `hidden_modes` by epoch `4`
  - that promoted epoch-4 snapshot is still only partial competence in `hidden_modes`: `running_success_rate = 0.45072`, `running_avg_return = -0.0482464`, `running_avg_steps = 33.6464`, `teacher_step_fraction = 0.259249`, `teacher_relabel_fraction = 0.621525`
  - current synthetic bottleneck is held-out compositional control/sequence binding, especially `selector_sequence_unlock`; easy families hold up materially better than the sequence-binding family under flat joint training
  - the official public online 5-game slice now completes end-to-end under one shared scorecard in the harness; current result remains `0/5`, but first-contact activity is less passive:
    - `ar25`: `72` steps, `5` interaction steps
    - `bp35`: `22` steps, `0` interaction steps
    - `cd82`: `100` steps, `0` interaction steps
    - `cn04`: `75` steps, `1` interaction step, with `objective_competition` / `disambiguate_objective` in the trace
    - `dc22`: `128` steps, `16` interaction steps
  - the next useful work is held-out compositional generalization, lower relabel dependence, broader mechanic diversity, and broader transfer validation, not more shallow controller shaping

## Prize Eligibility Boundary

- The default ARC-facing path must stay generic.
- Synthetic-task-specific controller logic is not part of the default execution path.
- The learned eval path must not construct or consult the hand-authored `RuntimeRuleController`.
- Explicit causal theory objects are allowed only as learned/evidence-updated belief state, not as a scripted controller.
- The learned eval path does not use heuristic language/question fallbacks.
- Generic perception does not infer synthetic roles from raw color IDs.
- The ARC toolkit adapter is now offline-first. Prize-facing execution should use locally cached environments and avoid runtime network dependence.
- The submission path must adapt online from live experience; a frozen policy-only path is not sufficient.
- The submission path must not rely on benchmark environment source files, hidden engine internals, or per-game behavior shaping.
- The submission path must not hide legal ARC actions behind default hand-authored candidate caps before training or online evidence has learned that they are unhelpful.
- Previous controller-backed real-ARC positive signals are historical baselines, not success for the clean learned-agent goal.

## First Commands

```bash
python -m venv .venv
. .venv/Scripts/activate
python -m pip install -e .[dev]
python -m pytest
```

## Optional ARC Toolkit

```bash
py -3.13 -m venv .venv313
. .venv313/Scripts/activate
python -m pip install -e .[arc]
```

Notes:

- Current official `arc-agi` releases require Python `3.12+`.
- `arc-agi` brings in `arcengine`; it does not need to be listed separately here.
- The ARC adapter is kept optional so local research and testing do not hard-fail when the toolkit is absent.

## Scientist Agent

The repo now includes a separate hypothesis-driven `scientist` agent under `arcagi.scientist`.
This path is intentionally small and explicit:

- object-centric perception
- falsifiable online hypotheses
- surprise-weighted episodic memory
- a tiny bootstrap world model
- grounded internal language
- an information-gain planner

Current boundary:

- the scientist path now supports checkpoint save/load and a reproducible curriculum runner
 - its checkpoint now persists the learned spotlight `habit`, `executive`, and `adaptation` scorers together with the online world model and spotlight priors
 - its synthetic trainer now uses retryable sessions with resets and cross-level carryover instead of only one-shot episodes
 - its default collection path executes learner actions, not teacher actions; `--teacher-feedback-mode dense` exists only as an explicit ablation/debug mode
 - its latest checkpoint schema stores `checkpoint_metadata` with learner-owned train summaries, sparse oracle query/feedback rates, autonomous holdout summaries, and promotion eligibility
 - its trainer refuses to update the "best" checkpoint on all-zero autonomous holdout solve metrics unless `--allow-unproven-best-checkpoint` is explicitly passed for debugging
 - it is useful for rapid attack-loop experiments on online learning, but it is not yet a full replacement for the main Stage 1 / Stage 2 training stack

Checkpoint schema boundary:

- `spotlight_feature_schema_version = 1`
  - legacy baseline spotlight checkpoints
  - old checkpoints now auto-load in legacy compatibility mode
- `spotlight_feature_schema_version = 2`
  - current extended spotlight runtime
  - extended generic cost/budget/schema signals in the live spotlight feature map
- `spotlight_feature_schema_version = 3`
  - current reset-guard spotlight runtime
  - preserves the `v2` generic cost/schema surface and adds reset cooldown / reset-budget features plus reset-penalized update targets
- harness progress, ARC eval results, and scientist trainer logs now emit `spotlight_feature_schema_version`
- do not compare `v1`, `v2`, and `v3` checkpoints as if they were the same runtime

Current preserved legacy baselines:

- [spotlight_exec_curriculum_best_legacy_v1.pkl](/G:/arcagi/artifacts/spotlight_exec_curriculum_best_legacy_v1.pkl)
- [spotlight_exec_curriculum_latest_legacy_v1.pkl](/G:/arcagi/artifacts/spotlight_exec_curriculum_latest_legacy_v1.pkl)

Run the scientist smoke environment:

```bash
python -m arcagi.scientist.cli --seed 3 --max-steps 80
```

Run bounded scientist curriculum training:

```bash
python -m arcagi.scientist.train \
  --stage1-sessions 32 \
  --stage2-sessions 96 \
  --eval-every 32 \
  --holdout-simple-sessions 6 \
  --checkpoint-path artifacts/spotlight_exec_curriculum_best_v3.pkl \
  --latest-checkpoint-path artifacts/spotlight_exec_curriculum_latest_v3.pkl \
  --teacher-feedback-mode sparse_disagreement \
  --teacher-query-probability 0.35 \
  --teacher-feedback-probability 0.05 \
  --teacher-disagreement-probability 0.45 \
  --teacher-feedback-weight 0.35
```

By default this command always writes the latest checkpoint, but only promotes
`--checkpoint-path` when teacher-free holdout has nonzero solve evidence.
Unproven debug snapshots can still be forced with
`--allow-unproven-best-checkpoint`, but those are not valid "best" artifacts.

Fresh post-integrity smoke result on this tree:

- `artifacts/post_integrity_scientist_latest.pkl`
- training sessions: `4`
- teacher query fraction: `0.421875`
- teacher feedback fraction: `0.1484375`
- autonomous holdout:
  - `simple_success_rate = 0.0`
  - `rich_session_win_rate = 0.0`
  - `rich_avg_levels_completed = 0.0`
- promotion: false, with `artifacts/post_integrity_scientist_best.pkl.unpromoted.json`
- short local `AR25` offline probe, `64` steps:
  - success: false
  - return: `0.0`
  - levels completed: `0`
  - interaction steps after the stalled diagnostic-binding fix: `14`

Evaluate the trained scientist checkpoint on local ARC with the default capped budget:

```bash
python -m arcagi.evaluation.harness arc \
  --agent scientist \
  --checkpoint-path artifacts/spotlight_exec_curriculum_best_v3.pkl \
  --game-limit 1 \
  --mode offline
```

Run an uncapped local ARC probe when you want to know whether a level is ever solved rather than whether it is solved within the default `256`-step budget:

```bash
python -m arcagi.evaluation.harness arc \
  --agent scientist \
  --checkpoint-path artifacts/spotlight_exec_curriculum_best_v3.pkl \
  --game-id ar25-0c556536 \
  --mode offline \
  --max-steps 0 \
  --progress-every 50
```

Current measured scientist result on this tree:

- synthetic curriculum training now shows nonzero teacher-free attempt-improvement on both simple and rich holdouts, and the checkpointed spotlight stack now logs real move-37-style override candidates and validations
- the default capped holdout path is still pessimistic:
  - `simple_success_rate = 0.0`
  - `rich_session_win_rate = 0.0`
  - `rich_avg_levels_completed = 0.0`
- but the uncapped local offline ARC probe is now materially better than zero:
  - `artifacts/spotlight_exec_curriculum_best_legacy_v1.pkl` completed `AR25` level 1 at about step `300`
  - the same run still plateaued at `levels_completed = 1` through at least step `750`
  - so the remaining real-ARC bottleneck is post-level transfer and rapid next-level objective reacquisition, not total inability to solve any real level
- the same `AR25` first-level solve now reproduces in `--mode online` with the same step-level milestones through at least step `850`, which makes the result much less likely to be an offline-only adapter artifact
- after the later generic cost/schema feature expansion, the old checkpoint should now be treated as a preserved `v1` baseline rather than a valid `v2` artifact
- a direct compatibility check now loads `artifacts/spotlight_exec_curriculum_best_legacy_v1.pkl` as `spotlight_feature_schema_version = 1`
- a short online `AR25` probe under the compatibility fix no longer shows the worst repeated-`4` collapse that appeared during the regression, but full restoration of the older step-300 first-level solve still needs a longer re-check

So the scientist path now has real first-level ARC competence on `AR25`, but it still does not close the cross-level carryover gap.

## Manual Stage 1 Training

The synthetic trainer now supports long manual runs with live JSON progress, interrupt-safe checkpoints, and held-out gated stage promotion.

Current training-default corrections:

- the trainer default is now `behavior_policy=mixed` in both CLI and programmatic config construction
- `mixed` / `bootstrap` collection now use the current hybrid agent as the learner-side collector, not the graph baseline, while teacher episodes and teacher relabeling still provide scaffold data
- replay and dream now optimize explicit causal/theory structure, not only generic next-state/value losses
- dense warm-start supervision keeps trajectories learner-owned; teacher control defaults are now `teacher_episode_fraction=0.1` and `teacher_takeover_prob=0.25`
- `--secondary-device` is now the preferred name for the second GPU; it mirrors replay and dream optimization each epoch, and only falls back to async holdout when it cannot be used for real training work
- `--resume-checkpoint-path` is the correct way to continue a tracked run; `--init-checkpoint-path` is weights-only initialization unless explicitly overridden

Observed long-run milestone on `2026-04-20`:

- a manual run with `25,000` episodes per epoch advanced into `hidden_modes` by epoch `4`
- end-of-epoch running metrics at the promotion regime were:
  - `running_success_rate = 0.45072`
  - `running_avg_return = -0.0482464`
  - `running_avg_steps = 33.6464`
  - `samples_collected = 841160`
  - `teacher_step_fraction = 0.259249`
  - `teacher_relabel_fraction = 0.621525`
- this means foundation is no longer the active synthetic blocker; hidden-mode stability and learner ownership are now the main Stage 1 problems

Recommended staged training command:

```bash
python -m arcagi.training.synthetic \
  --epochs 64 \
  --episodes-per-epoch 256 \
  --learning-rate 2e-4 \
  --checkpoint-path artifacts/manual_stage1.pt \
  --behavior-policy mixed \
  --device cuda:0 \
  --secondary-device cuda:1 \
  --curriculum staged \
  --log-every-episodes 16 \
  --holdout-eval-every-epochs 4 \
  --holdout-episodes-per-variant 2 \
  --promotion-consecutive-evals 2
```

Trainer behavior:

- prints `episode_progress` JSON every `--log-every-episodes`
- prints `epoch` JSON at each completed epoch
- prints `holdout_eval` JSON at each held-out evaluation point
- prints `stage_advance` JSON when the held-out thresholds are cleared and the trainer promotes
- saves the rolling latest checkpoint at `--checkpoint-path`
- also saves epoch snapshots like `artifacts/manual_stage1.epoch_0007.pt`
- on `Ctrl-C`, saves `artifacts/manual_stage1.interrupt.pt` before exit
- `--curriculum staged` now means gated staged progression with replay, not the old fixed epoch-thirds schedule

Recommended flat joint-training command:

```bash
python -m arcagi.training.synthetic \
  --epochs 200 \
  --episodes-per-epoch 512 \
  --learning-rate 2e-4 \
  --checkpoint-path artifacts/flat_joint.pt \
  --behavior-policy mixed \
  --device cuda:0 \
  --secondary-device cuda:1 \
  --curriculum flat \
  --log-every-episodes 32
```

Flat-mode notes:

- `flat` trains across all synthetic mechanic families at once
- `flat` does not use staged promotion/holdout gating during training
- collection metrics in `flat` are not a held-out benchmark
- evaluate saved checkpoints separately when comparing flat runs or checking whether hard families such as `selector_sequence_unlock` are actually improving
