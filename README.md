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

Detailed runtime-learning design:

- see [docs/ONLINE_RUNTIME_LEARNING.md](</G:/arcagi/docs/ONLINE_RUNTIME_LEARNING.md>)
- that document now defines:
  - what meaningful online self-correction means
  - how to learn "the rules" online from transitions
  - why counterfactual replay matters
  - the three learning timescales
  - a diffusion-inspired hypothesis-to-action-to-learning-to-proof loop

## Status

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
  - the current scientist curriculum trainer persists the online world-model weights across episodes, but it does not yet train a separate persistent policy or hypothesis library
  - bounded scientist pretraining currently solves its own tiny hidden-rule holdout (`simple_success_rate = 1.0`) but still fails the richer repo synthetic families (`rich_success_rate = 0.0`) and does not yet improve local ARC `AR25`
- current runtime-learning status:
  - the learned agent now beats the synthetic benchmark through live runtime rule learning and explicit online hypothesis competition
  - the full hybrid agent constructs the generic runtime hypothesis controller by default
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
- The learned eval path may construct the generic runtime hypothesis controller.
- The learned eval path must not construct any synthetic-task-specific or benchmark-shaped controller.
- The learned eval path does not use heuristic language/question fallbacks.
- Generic perception does not infer synthetic roles from raw color IDs.
- The ARC toolkit adapter is now offline-first. Prize-facing execution should use locally cached environments and avoid runtime network dependence.
- The submission path must adapt online from live experience; a frozen policy-only path is not sufficient.
- The submission path must not rely on benchmark environment source files, hidden engine internals, or per-game behavior shaping.
- The current real-ARC success path uses only black-box observations, action roles exposed by the adapter, and within-episode induced rules.

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
- its persistent learned state is currently only the online world model
- it is therefore useful for rapid attack-loop experiments, but it is not yet a full replacement for the main Stage 1 / Stage 2 training stack

Run the scientist smoke environment:

```bash
python -m arcagi.scientist.cli --seed 3 --max-steps 80
```

Run bounded scientist curriculum training:

```bash
python -m arcagi.scientist.train \
  --stage1-episodes 32 \
  --stage2-episodes 96 \
  --eval-every 32 \
  --holdout-simple-episodes 6 \
  --checkpoint-path artifacts/scientist_curriculum_run1_best.pkl \
  --latest-checkpoint-path artifacts/scientist_curriculum_run1_latest.pkl
```

Evaluate the trained scientist checkpoint on local ARC:

```bash
python -m arcagi.evaluation.harness arc \
  --agent scientist \
  --checkpoint-path artifacts/scientist_curriculum_run1_best.pkl \
  --game-limit 1 \
  --mode offline
```

Current measured scientist result on this tree:

- bounded curriculum pretraining reached:
  - `simple_success_rate = 1.0`
  - `rich_success_rate = 0.0`
- local offline ARC `ar25-0c556536` still failed after pretraining:
  - `success = false`
  - `return = 0.0`
  - `steps = 256`
  - `interaction_steps = 20`

So the scientist edits improved the experimentation substrate, but they did not close the real ARC semantic-transfer gap yet.

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
