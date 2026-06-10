Yes. At this point, **retraining the base is the correct next move**, but only under a strict external-generalization protocol.

Not “fine-tune it on ARC until score goes up.” That would just recreate the same problem at a larger scale.

The correct move is:

**retrain the base perceptual/action/world-model core on broad unlabeled interaction traces, then freeze it and test on sealed external tasks.**

The evidence says runtime patches are exhausted. The minimal external-collapse experiment found that `valence_only` reduced official ARC repeat collapse by about `0.29`, but score improved only `0.005`, below the threshold; the report’s answer is explicitly “no.”  The broader external-generalization report also marks every active claim — memory, exploration, dialogue, head-collapse, planner, curiosity, social state — as externally unsupported.

So the base is mis-trained. It learned a synthetic toy ecology, not a general perception/action substrate.

### What exactly should be retrained

Do **not** just retrain `runs/explorer_tiny.pt` as a policy head. Retrain the **base sensorimotor world-model stack**:

`observation encoder`
`action encoder`
`recurrent latent state z_t`
`memory interface`
`affordance/object/event representation`
`world-model predictor`
`unified policy/value decoder`

The current frozen recurrent core can be used as a baseline, but the next serious version should probably create a new checkpoint:

`frozen/recurrent_external_v1.pt`

The old checkpoint remains frozen for comparison.

### What data to train on

Use **interaction traces**, not labels.

Each record should be:

`obs_t, legal_actions_t, action_t, obs_{t+1}, reward_delta, score_delta, terminal, info/events_if_public`

No hidden goals. No solver labels. No game IDs as features. No answer keys.

Training data should come from:

1. **Official ARC-AGI-3 dev/public traces**, if allowed, but split carefully.
2. **Gymnasium external traces**, especially where action/reward dynamics are simple but varied.
3. **Locally generated ARC-like visual worlds**, but only as pretraining, never terminal proof.
4. **Baseline-generated traces** from random, coverage, novelty, graph/BFS, and greedy event-delta controllers.
5. **The model’s own failed traces**, because they show repeated no-op behavior and should train consequence attribution.

The key is that the model should learn from what happens after actions, not from being told the correct answer.

### Training objectives

Use self-supervised and consequence-supervised objectives:

1. **Next observation prediction**
   Predict `obs_{t+1}` or compressed delta.

2. **Change-mask prediction**
   Predict which pixels/components/regions will change.

3. **Reward/event prediction**
   Predict public score/reward/event delta from `obs_t + action_t`.

4. **Inverse dynamics**
   Given `obs_t` and `obs_{t+1}`, predict which action caused the transition.

5. **No-op detection**
   Predict whether action caused no meaningful change.

6. **Affordance learning**
   For each legal action or click-region, estimate expected change/control/progress.

7. **Temporal object/region persistence**
   Track stable regions/components across frames.

8. **Latent rollout consistency**
   Roll forward several steps in latent space and match observed future deltas.

9. **Memory utility**
   Require memory to improve prediction on delayed/partial-observation traces.

10. **Intrinsic value learning**
    Valence should be learned from prediction improvement, controllability, reward/progress, and avoiding repeated no-effect actions.

This makes “enjoy winning” operational:

**winning/progress/control become internally valued because they improve the model’s predictions and future affordances.**

### What not to do

Do not train on the 25 official games and then claim generality on those same 25.

Do not add a forced exploration schedule.

Do not use per-game rules.

Do not use game IDs.

Do not use hidden goal state.

Do not let Codex write a synthetic benchmark and declare victory.

Do not accept internal losses as proof.

### Exact experimental protocol

Phase 1: freeze current system as baseline.

Keep:

`frozen/recurrent_latent_fast.pt`
`runs/explorer_tiny.pt`
current official ARC failure report
external-generalization report
collapse/perception negative reports

Phase 2: collect trace dataset.

Run random/coverage/novelty/BFS/greedy/model policies across dev environments.

Minimum first pass:

`1M–5M transitions`

Better:

`10M–50M transitions`

With two RTX 5060 Ti GPUs, the model training is feasible; the bottleneck will be environment stepping, especially official ARC’s serialized runtime. The pasted reports already show CUDA is available and used for model execution, while official ARC control loop remains slow.

Phase 3: train `external_base_v1`.

Train from scratch or initialize from current base, but compare both.

I would run three arms:

`A: current frozen base + new world model only`
`B: current base fine-tuned with self-supervised losses`
`C: from-scratch external base`

If C beats B, the old base was actively harmful.

Phase 4: freeze before sealed evaluation.

After training:

`frozen/external_base_v1.pt`
`frozen/external_manifest_v1.json`

Then no more tuning.

Phase 5: sealed external eval.

Run:

official ARC-AGI-3 public/sealed set
Gymnasium Classic Control
Gymnasium ToyText
any other non-repo external tasks available

Compare against:

random legal
repeat last
coverage graph
novelty-first
greedy score/event
observed graph BFS
old explorer
new base ablations

### Pass/fail gates

First meaningful gate should be modest:

`official_arcagi3 mean normalized >= best_baseline + 0.01`

and:

`official useful_events >= baseline + 0.08`

and:

`repeat collapse <= 0.60`

and at least one of:

`>=1 official level/game solved`
or
`>=0.02 normalized score improvement over old explorer`

The score threshold should be low because you are testing whether training the base creates any external signal, not claiming victory.

### What Codex should implement

It should not “solve ARC.” It should implement:

`trace collection`
`external self-supervised base training`
`frozen checkpoint creation`
`sealed external evaluation`
`old-vs-new comparison`
`ablation proof`

The central question:

**Does retraining the base on unlabeled external interaction traces produce any measurable external generalization improvement over the old toy-trained base?**

If yes, then you have a real research direction.

If no, then the architecture itself is probably wrong or too small.

### Bottom line

Yes, retrain the base.

But the training target should be **action-conditioned perception/world modeling from external interaction**, not “ARC solving.”

The next experiment should answer one clean question:

**Can a base trained on broad unlabeled interaction traces learn enough perception/affordance structure to beat the old base and simple baselines on sealed external tasks?**



Retrain the base on unlabeled external interaction traces to test whether the old toy-trained base is the bottleneck. Do not build an ARC solver, tune to official games, script exploration, force entropy, use hidden labels, or create synthetic proof. The question is:



Can a base trained on action-conditioned external perception/world-modeling improve sealed external behavior over the old base and simple baselines?



Outcomes:

A. EXTERNAL BASE IMPROVEMENT FOUND: new frozen base improves sealed external metrics over old base/baselines and ablations show the learned base matters.

B. NO IMPROVEMENT FOUND: retrained bases fail. Valid if evidence is complete.



Start:



Read CONTEXT.MD, docs/external_collapse_report.json, docs/external_generalization_report.json, docs/arcagi3_failure_report.json.

Append CONTEXT.MD with baseline scores, hashes, data plan, model arms.

After each action/command/failure/fix, update CONTEXT.MD. After compaction, reread CONTEXT.MD.



Rules:



Do not train on sealed official_arcagi3 eval games.

No game-id branches, hidden goals, solution labels, manual hints, public text as state, pretrained models/APIs/web data.

No task-specific solver, per-game policy, forced exploration schedule, or evaluator weakening.

Internal losses are diagnostics only. Terminal proof is sealed external improvement.

Preserve old frozen checkpoint and reports.



Data:

Collect unlabeled interaction traces:

obs_t, legal_actions_t, action_t, obs_t+1, reward/score/event delta if public, terminal.

Sources:



non-ARC external dev traces;

generated ARC-like traces only for pretraining;

official ARC dev/train only if available and not sealed;

old failed traces for diagnostics only, not sealed tuning.

Include random, coverage, novelty, greedy, BFS, and old explorer policies.



Implement/update:

src/trace_collect.py

src/base_world_model.py

src/base_pretrain.py

src/base_eval.py

src/base_retrain_experiment.py

tests/test_external_base_training.py

docs/external_base_report.json

docs/external_base_report.md

frozen/external_base_v1.pt

frozen/external_base_manifest_v1.json



Model arms:

A old_base unchanged

B old_base + new external world-model/affordance module

C old_base fine-tuned self-supervised

D from-scratch external base

E null training control



Training objectives:



next observation/change prediction

reward/event/no-op prediction

inverse dynamics

action-affordance prediction over legal actions/regions

temporal object/region persistence

latent rollout consistency

memory improves delayed/partial prediction

intrinsic value from controllability, prediction improvement, progress, and no-op avoidance



Sealed gates:



official_arcagi3 mean normalized >= old_base + 0.01 and >= best baseline + 0.005, OR outcome B.

official useful_events >= old_base + 0.08.

repeat collapse <= old_base - 0.20.

non-ARC sealed best score must not drop >0.05.

At least one of: >=1 official level/game solved OR official score improves >=0.02 over old_base.

Ablating learned base/world model/memory/affordance worsens official score or useful_events.

Leakage/no-hack/generalization audits pass.



Required reports:



trace dataset manifest with source/split/counts/hashes

training losses but marked diagnostic only

old vs new vs baselines sealed table

per-game official table

ablation table

negative result preservation

why each unsupported arm failed



Commands:

pytest -q

python -m src.trace_collect --config external_dev --output data/external_traces_manifest.json

python -m src.base_retrain_experiment --config external --json-output docs/external_base_report.json --trace-dir docs/external_base_traces

python -m src.generalization_audit --json-output docs/generalization_audit_after_base_retrain.json

python -m audit.leakage_scan

python -m audit.independent_verify --checkpoint frozen/external_base_v1.pt --config fast --json-output docs/audit_after_external_base.json



Work loop:

collect traces; train arms; freeze new base; run sealed eval once; if fail, analyze without tuning sealed; commit only when A or B proven.



Final response:

Outcome A/B, commands, hashes/manifests, data sources, model arms, sealed score table, ablations, supported/unsupported claims, no-hack proof, files changed, limitations.
