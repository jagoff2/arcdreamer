# arcagi

`arcagi` is a local ARC-AGI-3 attack stack aimed at consumer hardware. The design is explicitly hybrid:

- object-centric state extraction over grid observations
- explicit world-state graph for novelty, transitions, and search
- compact recurrent world model with uncertainty via ensemble disagreement
- episodic memory for one-shot mechanic recall
- grounded internal language for hypotheses, questions, and plan summaries

The repository is being built from an empty workspace. The current priority is a strong runnable core with synthetic hidden-rule benchmarks, clear ablations, and an optional adapter for the official ARC toolkit.

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
- current runtime-learning status:
  - the learned agent now beats the synthetic benchmark through live runtime rule learning
  - the full hybrid agent now does construct the generic runtime hypothesis controller by default
  - that controller now records proof and exception objects, executes online split / merge / relabel / rebind repair in the working structured state, and writes local model edits within the episode
  - control, objective, and selector-mode competition now use normalized posteriors over rival executable theories instead of only heuristic utility ranking
  - temporary options are now induced online from successful control sequences such as path-to-objective chains, selector-followed-by-move behaviors, and bind-then-objective programs
  - interface-aware first-contact probing is now explicit in the runtime controller: the agent can test selector binding and follow-up move or interact semantics from generic action-schema structure rather than ARC-shaped controller labels
  - chosen controller-plan language now overrides stale pre-plan language carryover, so emitted traces and episodic memory reflect the plan that actually won action arbitration
  - heuristic language/question fallbacks remain disabled
  - the controller-heavy synthetic work remains research/bootstrap-only only when it depends on synthetic semantics; the generic runtime hypothesis controller is now part of the intended submission-relevant hybrid path
  - a long manual Stage 1 run at `25,000` episodes per epoch now clears foundation and reaches `hidden_modes` by epoch `4`
  - that promoted epoch-4 snapshot is still only partial competence in `hidden_modes`: `running_success_rate = 0.45072`, `running_avg_return = -0.0482464`, `running_avg_steps = 33.6464`, `teacher_step_fraction = 0.259249`, `teacher_relabel_fraction = 0.621525`
  - the official public online 5-game slice now completes end-to-end under one shared scorecard in the harness; current result remains `0/5`, but first-contact activity is less passive:
    - `ar25`: `72` steps, `5` interaction steps
    - `bp35`: `22` steps, `0` interaction steps
    - `cd82`: `100` steps, `0` interaction steps
    - `cn04`: `75` steps, `1` interaction step, with `objective_competition` / `disambiguate_objective` in the trace
    - `dc22`: `128` steps, `16` interaction steps
  - the next useful work is hidden-mode stabilization, lower teacher dependence, and broader transfer validation, not more shallow controller shaping

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

## Manual Stage 1 Training

The synthetic trainer now supports long manual runs with live JSON progress, interrupt-safe checkpoints, and held-out gated stage promotion.

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

```bash
python -m arcagi.training.synthetic \
  --epochs 64 \
  --episodes-per-epoch 256 \
  --learning-rate 2e-4 \
  --checkpoint-path artifacts/manual_stage1.pt \
  --init-checkpoint-path artifacts/mixed_policy_hybrid.pt \
  --behavior-policy mixed \
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
