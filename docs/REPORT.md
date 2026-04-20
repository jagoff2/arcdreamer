# Technical Report

## Date

2026-04-19

## 2026-04-20 Interface-Aware Runtime Update

The latest correction focused on the exact gap exposed by the real public ARC slice:

- the agent could execute generic runtime hypotheses and induced options
- but its first-contact experiments were still too narrow
- its emitted language often reflected stale pre-plan runtime tokens instead of the controller plan that actually won arbitration
- the online harness itself was also flawed because it opened a new ARC scorecard per game and could fail mid-slice with `409` conflicts

That is now materially corrected:

- ARC adapter and generic perception now surface interface-aware state features:
  - action-schema-derived inventory and flags for move / click / select / interact structure
  - inferred clickable / interface-target / selector-candidate tags over visible objects
- the runtime controller now includes interface-aware first-contact experiment programs:
  - selector-then-move probing
  - selector-then-interact probing when interaction affordances are present
  - bind-then-objective option materialization when a non-default control mode and objective sequence stabilize
- the learned agent now keeps the chosen controller-plan language after arbitration instead of leaving diagnostics and episodic memory anchored to stale pre-plan tokens
- the ARC harness now reuses one shared online scorecard across the full slice, so the 5-game online run completes end-to-end instead of failing on repeated scorecard creation

Measured real online 5-game slice with `artifacts/test_bootstrap_handoff_fix.interrupt.pt` after this pass:

- `ar25-0c556536`: fail, return `0.0`, `72` steps, `5` interaction steps
- `bp35-0a0ad940`: fail, return `0.0`, `22` steps, `0` interaction steps
- `cd82-fb555c5d`: fail, return `0.0`, `100` steps, `0` interaction steps
- `cn04-2fe56bfb`: fail, return `0.0`, `75` steps, `1` interaction step
- `dc22-fdcac232`: fail, return `0.0`, `128` steps, `16` interaction steps

This is still `0/5`, but the failure signature changed in a useful direction:

- `ar25` is no longer fully passive
- `dc22` now spends materially more of its budget on real interactions
- `cn04` now emits `objective_competition` and `disambiguate_objective` in the live trace instead of only vague control noise

So the current bottleneck is no longer basic runtime-controller plumbing.
The remaining problem is semantic transfer:

- the agent is running broader first-contact experiments
- but it is still not converting those experiments into reward-bearing task semantics quickly enough on the real slice
- synthetic competence still overpredicts real-world hidden control and reward structure

## 2026-04-19 Runtime Architecture Correction

The previous repo state still had a structural mismatch between the stated thesis and the actual eval path:

- the explicit runtime hypothesis controller existed
- but the full `HybridAgent` did not construct it by default
- proof objects, exception records, and representation-repair signals were mostly design intent rather than first-class runtime data

That mismatch is now materially corrected:

- the full hybrid eval agent now owns the generic runtime hypothesis controller
- runtime hypotheses now emit proof and exception objects for control, objective, mode, and representation events
- those proof objects are written into episodic memory so later retrieval can condition on what actually supported or falsified a claim
- the base agent now runs a representation-repair workspace that can split, merge, relabel, and rebind entities inside the working structured state
- the controller still tracks repair confidence and suppresses premature exploitation when the current entity view looks unstable
- control, objective, and selector-mode hypotheses are now weighted as normalized rival executable theories rather than only heuristic score buckets
- the learned agent now applies local per-episode model patches over action value, usefulness, policy prior, and uncertainty before falling back to heavier online world-model updates
- the full train-plus-eval smoke path now passes with these changes in place

This is still not the full target architecture. Option induction is now present, but the option library is still too weak on real ARC, and repair proposals are still driven mostly by lightweight geometry/interface evidence rather than a richer learned decomposition model.

## Current Honest State

The repo has moved since the earlier positive public result. The old ARC-shaped graph controller that produced the historical `1/5` public slice was deleted because it was not acceptable as a transferable default path. The current default graph baseline uses generic within-episode rule induction and action-schema exploration instead, and its current public 5-game result is `0/5`.

The current learned status has improved materially on the real black-box path:

- the learned hybrid agent now contains an explicit runtime rule-object layer and online hypothesis manager rather than only scalar action statistics
- measured `AR25` result:
  - offline: `success=true`, `return=1.0`, `steps=19`
  - online: `success=true`, `return=1.0`, `steps=19`
- current official 5-game online slice for the same hybrid controller:
  - `AR25`: success `true`, return `1.0`, steps `19`
  - `BP35`: success `false`, return `0.0`, steps `31`
  - `CD82`: success `false`, return `0.0`, steps `100`, interaction steps `5`
  - `CN04`: success `false`, return `0.0`, steps `75`, interaction steps `0`
  - `DC22`: success `false`, return `0.0`, steps `128`, interaction steps `7`
  - aggregate: `1/5`
- this real-ARC result uses the default hybrid agent and the official adapter only; no environment-file semantics are used in the runtime path

The broader learned status is still mixed:

- oracle-only synthetic training improved model fitting but produced brittle policy collapse
- mixed-behavior synthetic training plus runtime empirical gating produced the first non-zero learned synthetic success again
- current best learned synthetic result:
  - recurrent: `1/15` episodes solved
  - language: `1/15` episodes solved
  - hybrid: `0/15`

So the honest interpretation is:

- the codebase is more transfer-oriented and less benchmark-shaped than before
- the currently measured default-path performance is worse than the historical ARC-specific peak
- live runtime adaptation is now an explicit requirement, but it is still incomplete

Synthetic milestone update:

- the learned runtime path now beats the synthetic benchmark again without restoring the deleted handcrafted synthetic controller
- the winning mechanism is a runtime object-hypothesis controller layered into the learned agent, plus online `avoid` / `prefer` memory separation, counterfactual replay, and per-episode action-bias updates
- the current measured synthetic result is:
  - recurrent: `1.0` success rate
  - hybrid: `1.0` success rate
- real ARC has now been re-run after the synthetic milestone
- the first black-box learned-agent result is a successful `AR25` solve in both offline and online mode

## Online Rule-Learning Thesis

The repo now treats online rule learning as the central runtime problem.

The intended runtime behavior is:

1. infer compact causal rules from transitions
2. keep multiple competing rule hypotheses alive
3. choose actions that either exploit or disambiguate those rules
4. replay failures counterfactually
5. revise action preferences, memory, and local beliefs immediately

This is stronger than generic "online RL" language and stronger than simple episodic memory writes.

It means the agent should be able to learn:

- action semantics
- object roles
- reward rules
- hidden-state beliefs
- diagnostic action value

while the same episode is unfolding.

## Current Runtime-Learning Mechanisms

Implemented or partially implemented:

- within-episode rule induction from observed transitions
- explicit rule objects:
  - motion rules with support / contradiction tracking
  - goal hypotheses with progress / reward support
  - signature statistics for stable-vs-moving object families
- generic action-schema exploration
- mixed-behavior synthetic training with oracle labels on visited states
- live empirical gating of checkpoint priors
- explicit `avoid_action` and `recommended_action` memory writes
- counterfactual replay over failed or stalled transitions
- per-episode online action-bias updates
- runtime object-hypothesis controller for tagged object-centric worlds:
  - this has now been replaced in the default learned path by a generic transition-driven hypothesis controller
  - test untried move actions with `undo` when available
  - infer moving object families and stable goal candidates from transitions alone
  - persist exploit options around a learned goal anchor until predicted progress stops

Current limitations:

- hypothesis competition is now posterior-normalized, but the candidate family is still narrow and mostly object-signature based
- the hybrid memory path no longer catastrophically loops, but its current success still depends heavily on the runtime rule layer
- counterfactual replay exists, but it is still a local patch rather than the dominant runtime loop
- representation repair now executes live split / merge / relabel / rebind edits, but it still relies on cheap geometric evidence rather than a richer learned object-tracking mechanism
- selector/click semantics are still under-modeled:
  - the controller does not yet maintain a strong enough latent belief over control mode or selected object identity
- proof/disproof pressure is too weak:
  - interaction-heavy games can continue burning steps after low-yield evidence
- some recent controller gains were still too dependent on hand-shaped target-ranking formulas rather than stronger generic hypothesis learning

Current pivot:

- the active correction is to remove remaining target-bias logic from the runtime ARC path and replace it with:
  - explicit mover hypotheses
  - explicit objective hypotheses
  - stronger latent mode hypotheses
  - direct proof/disproof updates from observed transitions
  - more willingness to defer back to the learned planner when runtime hypothesis evidence is weak
- the bar is not merely "avoid forbidden external knowledge"
- the stronger bar is "make the behavior arise from generic online learning machinery that should transfer beyond ARC"
- the current refactor did preserve the black-box `AR25` solve while materially reducing interaction churn on the rest of the public 5-game slice, but it has not yet improved the solved-count beyond `1/5`
- the next missing piece is tighter coupling between the learned latent or scratchpad path and the explicit runtime hypothesis loop
- the concrete coupling defect is now identified:
  - selector actions can change hidden control state through the recurrent path
  - but the runtime controller still does not directly ask which moves become better after a selector according to the latent dynamics model
  - that weak coupling is the current best explanation for why interaction churn fell while objective discovery still failed

Coupling update:

- that coupling gap is now partially closed:
  - the planner builds a shared runtime-thought object carrying grounded tokens, per-action latent value/uncertainty, and selector-conditioned follow-up move gains
  - the runtime controller now consumes that object directly for exploit, diagnostic, and selector-probe decisions
- measured outcome:
  - `AR25` still succeeds offline and online
  - overall solved-count is still `1/5`
  - `CN04` remains bad, which indicates that latent selector gains are still noisy and need stronger proof/disproof before the controller trusts them repeatedly

Control update:

- a planner-only action arbiter was briefly introduced during the coupling refactor
- that was the wrong control semantics
- the current corrected semantics are:
  - one unified decision layer
  - planner scoring over the full signal set
  - explicit reasoning allowed to seize control when restore, exploit, momentum, or diagnostic confidence is strong
- measured revalidation after the correction:
  - offline `AR25`: success in `21` steps
  - online `AR25`: success in `21` steps
- that restores success while preserving the stronger no-dumb-fallback control semantics, but it still has a real `19 -> 21` regression versus the earlier black-box controller

## Diffusion-Inspired Direction

The current design discussion now includes a diffusion-inspired hypothesis loop.

The intended use is not "replace the agent with a diffusion model."
The intended use is:

- maintain a noisy cloud of candidate explanations
- propose actions from that uncertain cloud
- observe the true transition
- denoise the hypothesis set by support and contradiction
- write a local proof or disproof record
- iterate until the hypothesis set sharpens enough for stable exploitation

This is attractive because it naturally couples:

- uncertainty
- counterfactual prediction
- intervention
- proof-like rule verification

## Scope

This report covers the first runnable milestone of the local ARC-AGI-3 attack stack. The goal of this milestone was to stand up the shared architecture, baselines, training path, evaluation harness, and a synthetic hidden-rule benchmark with explicit failure visibility, then convert the real offline ARC path from adapter validation into a measured positive-reward result.

## Implemented System

Current modules:

- object-centric state extraction from grid observations
- explicit graph memory over visited states and action outcomes
- compact recurrent world model with ensemble disagreement
- episodic memory store
- grounded internal language decoder over a controlled vocabulary
- planner that scores novelty, learned value, uncertainty, and memory retrieval
- baselines:
  - random / heuristic
  - explicit graph exploration
  - learned recurrent without language
  - learned recurrent with language and no episodic memory
  - full hybrid

Required runtime behavior:

- the final submission path must learn from its own transitions online while the episode is live
- trained weights are only priors; the runtime agent must update memory, induced rules, and action preferences from fresh experience
- ARC-facing control logic should remain transferable to non-text games with comparable observation/action structure
- the ARC environment must be treated as a black-box interaction surface; environment source files and game-specific tuning are forbidden inputs to the runtime path
- explicit rule objects may be engineered, but rule contents must be inferred online from the current episode rather than baked from benchmark-specific priors

## Synthetic Benchmark

Current benchmark families:

- `switch_unlock`
- `order_collect`

These are small interactive hidden-rule tasks intended to stress:

- deliberate diagnostic actions
- sparse reward handling
- family-level adaptation potential
- grounded language supervision from simulator-side rule labels

## Verification

- `python -m compileall arcagi` passed.
- `python -m pytest -q` passed: `7 passed`.

## Smoke Results

Training smoke:

- command:
  - `python -m arcagi.training.synthetic --epochs 1 --episodes-per-epoch 4 --checkpoint-path artifacts/smoke.pt --seed 11`
- output:
  - samples: `144`
  - loss: `8.2675`
  - uncertainty: `0.4233`

Evaluation smoke, hybrid agent:

- command:
  - `python -m arcagi.evaluation.harness synthetic --agent hybrid --checkpoint-path artifacts/smoke.pt --episodes-per-family 1 --seed 19`
- output:
  - success rate: `0.0`
  - average return: `-1.8`
  - average interactions: `36`

Evaluation smoke, graph baseline:

- command:
  - `python -m arcagi.evaluation.harness synthetic --agent graph --episodes-per-family 1 --seed 19`
- output:
  - success rate: `0.0`
  - average return: `-0.81`
  - average interactions: `5`

## Prize Eligibility Correction

During iteration, a synthetic-benchmark-specific structured controller was temporarily added to solve the Stage 1 hidden-rule worlds. That controller is not acceptable in the default ARC-facing path if the target is ARC Prize eligibility, so it has been deleted.

What remains acceptable:

- generic object extraction
- generic graph state tracking
- generic recurrent world model
- generic grounded language module
- generic planner
- optional synthetic environments as internal research instrumentation

What is no longer part of the default agent path:

- synthetic-only structured controller logic
- synthetic success claims that depended on that controller

## Official Toolkit Validation

Prize-facing validation now uses an isolated Python `3.13` environment because current official `arc-agi` releases require Python `3.12+`.

Validated setup:

- interpreter:
  - `Python 3.13`
- installed packages:
  - `arc-agi==0.9.7`
  - `arcengine==0.9.3`

Real interface findings:

- `Arcade().get_environments()` returns environment metadata entries
- local wrapper `reset()` returns `FrameDataRaw`
- local wrapper `step(GameAction, ...)` returns `FrameDataRaw`
- grid data lives in `FrameDataRaw.frame[0]`
- legal actions are exposed as integer IDs in `available_actions`

Adapter outcome:

- `arcagi/envs/arc_adapter.py` was updated to:
  - extract grids from `frame[0]`
  - recover the effective camera grid from the rendered display
  - convert action IDs to `GameAction.ACTION{n}`
  - emit generic click affordances and action-role metadata
  - support non-Gym reset/step behavior
  - default to offline-first operation

Offline-first validation:

- cached local games visible offline:
  - `['ar25-0c556536']`
- cached local environment reset shape:
  - rendered display: `(64, 64)`
  - effective camera grid used by the agent: `(21, 21)`
- cached local environment legal actions:
  - `('1', '2', '3', '4', '5', '6', '7')`
- generic `GraphExplorerAgent` executed steps on the real local environment without adapter failure

Real offline execution result:

- command:
  - `.venv313\Scripts\python -m arcagi.evaluation.harness arc --agent graph --game-limit 1`
- output on cached local game `ar25-0c556536`:
  - success: `true`
  - return: `1.0`
  - steps: `19`
  - interaction_steps: `0`

Real-controller mechanism used for that result:

- probe move actions once to learn observed per-object translation vectors
- infer anchored small target objects from the observation itself
- compute a concrete action plan from the learned motion basis to the anchored target
- stop ARC evaluation on the first positive reward, which is the current harness success criterion

## Official Public Online Slice

After fixing the online interaction path, the graph baseline was also run against the official public API using the toolkit's anonymous key flow.

Important online fixes:

- do not expose bare raw `ACTION6` when concrete click affordances are available
- infer the effective camera grid from rendered `64x64` frames when the remote wrapper does not expose camera metadata
- handle transitions between states with different inferred grid sizes

Measured command:

- `.venv313\Scripts\python -m arcagi.evaluation.harness arc --agent graph --mode online --game-limit 5`

Measured result:

- `AR25`: success `true`, return `1.0`, steps `19`
- `BP35`: success `false`, return `0.0`
- `CD82`: success `false`, return `0.0`
- `CN04`: success `false`, return `0.0`
- `DC22`: success `false`, return `0.0`

Interpretation:

- the current public 5-game slice is `1/5`
- that is a real non-zero official public result
- it is not yet evidence of broad ARC-AGI-3 performance

## What Worked

- The repository now runs end-to-end.
- The synthetic environment, perception stack, graph memory, training loop, evaluation harness, and tests are wired together.
- Failure modes are observable and reproducible rather than hidden behind missing infrastructure.
- The synthetic-specific semantics are now isolated away from the generic ARC-facing perception path.
- The official local ARC toolkit path is validated in offline-first mode against a real cached local game.
- The generic graph baseline now executes through the official offline toolkit path and earns a real positive reward on the cached local game.
- The current learned recurrent/language path regained a non-zero synthetic success signal after mixed-behavior training and stronger runtime use of live empirical evidence.
- The hybrid path no longer collapses into a trivial repeated-interaction loop after adding online `avoid` / `prefer` memory separation and counterfactual replay.
- The learned recurrent and hybrid agents now solve the full synthetic evaluation slice via live runtime rule learning after adding the runtime object-hypothesis controller.
- The hybrid agent now also solves `AR25` through a black-box explicit-rule runtime controller in both offline and online mode.

## What Failed

- The original learned hybrid policy collapsed into repeated interaction behavior in the pure generic path.
- The grounded language head is still not yet proven to add value in the generic path.
- Real ARC performance is still only lightly validated: one cached local game, first positive reward, no broad multi-game coverage yet.
- After deleting the handcrafted ARC-shaped controller, the default graph baseline regressed from the historical `1/5` public slice to the current `0/5` generic controller result.
- Explicit rule learning is still not strong enough to claim that the system genuinely learns "the rules" online in a broad sense.
- The current real-ARC success is still narrow: `AR25` game 1 is solved, but broad multi-game generalization remains unproven.

## Current Interpretation

The remaining real problem is separation of concerns and external validity:

- synthetic-stage machinery can be explicit and domain-shaped
- ARC-facing machinery must stay generic
- the next serious validation step is the official local toolkit path, not synthetic task wins
- black-box discipline is now stricter: future real-ARC work must not use environment-file semantics or game-shaped controller logic as design input

Current simple interpretation of failure:

- the agent can now learn "what moves" online
- it still struggles to learn "what mode am I in now?" and "what exactly did this click/select action change?"
- that is the current barrier between a narrow real-ARC win and broader online generalization

## Immediate Bottlenecks

1. Grounded language usefulness is not yet demonstrated as a causal gain in the generic path.
2. Learned world-model benefits are not yet isolated cleanly in the generic path.
3. Broader multi-game public ARC validation is still pending.
4. The current default public ARC result is `0/5`; the earlier `1/5` result was tied to a controller that has since been removed from the default path.
5. Online self-learning at runtime is still incomplete; the current system still relies too much on checkpoint priors and weak inference-time induction.
6. Explicit rule objects, hypothesis competition, and stronger diagnostic intervention planning are still missing.
7. The runtime object-hypothesis controller is promising, but it is still much narrower than the eventual cross-domain rule learner we actually want.

## Next Steps

1. Keep synthetic heuristics out of the default ARC-facing execution path.
2. Strengthen the explicit rule-object and hypothesis-manager stack beyond the current first black-box version.
3. Push live runtime adaptation deeper into the submission path so the agent learns richer rules from its own transitions while it plays.
4. Gate or replace brittle checkpoint priors when live evidence disagrees, especially in the hybrid path.
5. Add explicit ablation runs showing what the learned world model, memory, and language contribute in the generic path.
6. Expand offline cached-game validation and evaluation coverage only after the runtime-learning story is stronger.
7. Build explicit rule objects and a hypothesis manager.
8. Turn the diffusion-inspired hypothesis-denoising loop into a concrete runtime mechanism.

## Bounded Search Update

The latest planner iteration addressed a specific systems failure rather than a benchmark-score failure.

What failed:

- doubling latent search depth directly from `2` to `4` under the old planner caused combinatorial rollout explosion
- the old search expanded every legal action at every imagined node
- it also recomputed root world-model transitions that were already available from `build_runtime_thought(...)`

What changed:

- runtime thought now caches root `next_latent` and `next_hidden` predictions per action
- runtime thought now records how many world-model calls were already spent building the per-step thought bundle
- planner search now:
  - reuses cached root predictions
  - expands only the top configured root actions
  - expands only the top configured branch actions in imagined nodes
  - stops at a hard world-model-call budget per external turn

This keeps the search generic and legal while allowing deeper internal rollouts without brute-force explosion.

Measured verification:

- `python -m pytest -q`
  - `18 passed in 14.10s`

Measured online slice:

- `.venv313\Scripts\python -m arcagi.evaluation.harness arc --agent hybrid --checkpoint-path artifacts/mixed_policy_hybrid.pt --game-limit 5 --mode online`
- result:
  - `AR25`: `success=true`, `return=1.0`, `steps=21`, `interaction_steps=0`
  - `BP35`: `success=false`, `return=0.0`, `steps=31`, `interaction_steps=0`
  - `CD82`: `success=false`, `return=0.0`, `steps=100`, `interaction_steps=0`
  - `CN04`: `success=false`, `return=0.0`, `steps=75`, `interaction_steps=1`
  - `DC22`: `success=false`, `return=0.0`, `steps=128`, `interaction_steps=0`

Interpretation:

- the compute problem is fixed
- the bounded depth-`4` path now completes the real online slice instead of stalling in rollout explosion
- but benchmark performance did not improve yet
- the remaining problem is hypothesis quality and proof/disproof quality, not raw planner depth alone

## Next Training Reset

The next training pass will not be a cosmetic rerun of the same simple synthetic data.

Committed direction:

- retrain on synthetic first
- make the synthetic suite materially richer and more adversarial before returning to real ARC validation

Required synthetic additions:

- selector-mode tasks with hidden internal control state
- delayed-reward tasks where correct substeps do not immediately pay out
- compositional tasks that require multi-stage dependencies
- misleading affordances with tempting local payoff
- non-monotonic objectives where short-term gain can hurt long-term success

Reason:

- the current checkpoint was trained only on the two simple Stage 1 families
- that is not enough pressure to teach the learned world model and language prior how to support the richer online rule induction now expected at inference time
- the real ARC 5-game slice is already telling us that the missing competence is not basic movement, but hidden semantics and stronger disambiguation under sparse feedback

## Parity Check Regression

I ran a direct regression check to see whether the current code can still reproduce old easy synthetic behavior on the original narrow family.

Experiment:

- retrain from scratch on only `switch_unlock`
- fixed small board size
- evaluate only on `switch_unlock` variants

Measured narrow retrain result:

- training:
  - `epochs=12`
  - `samples_last_epoch=1014`
  - `loss_last_epoch=0.7042`
  - `uncertainty_last_epoch=0.0199`
- same-task eval:
  - hybrid: `success_rate=0.1111`, `avg_return=-0.6278`
  - recurrent: `success_rate=0.0`, `avg_return=-0.6600`

That already failed to establish parity.

Critical control:

- evaluate the old checkpoint `artifacts/mixed_policy_hybrid.pt`
- same current code
- same `switch_unlock`-only eval

Measured result:

- hybrid: `success_rate=0.0`, `avg_return=-0.7356`
- recurrent: `success_rate=0.0`, `avg_return=-0.5333`

Interpretation:

- this is not just a bad richer-curriculum retrain
- the current runtime stack itself no longer reproduces the old easy synthetic behavior
- old synthetic success claims are therefore not currently reproducible under the present codebase

Immediate implication:

- restore narrow-slice synthetic parity before trusting broader synthetic conclusions or returning to real ARC validation

## Parity Repair Update

The environment rewrite itself was not the main cause of the old easy-family collapse.

I directly checked the suspected path and the stronger conclusion is:

- the primary regression lived in the current runtime decision stack
- especially in the way objective hypotheses, interaction evidence, arbitration, and imagined movement interacted

The repaired mechanism set is:

1. direct interaction outcomes now write real support or contradiction into objective hypotheses instead of being mostly ignored
2. adjacent interactables now get an explicit probe path before low-evidence exploit
3. explicit interaction probes are allowed to seize control in arbitration
4. inactive visible targets are gated while unresolved interaction mechanics remain
5. goal activation now seeds target-object objectives and suppresses prerequisite interactables afterward
6. diagnostic move probes now skip visibly blocked moves
7. imagined mover deltas now respect blocking objects instead of sliding through them in imagination

Measured verification:

- `python -m pytest -q`
  - `23 passed in 88.54s`

Old checkpoint narrow `switch_unlock` recheck after the repair:

- checkpoint:
  - `artifacts/mixed_policy_hybrid.pt`
- variants:
  - `red`
  - `blue`
  - `green`
- seeds:
  - `17..19`

Aggregate:

- hybrid:
  - `success_rate=0.5556`
  - `avg_return=0.3178`
  - `avg_steps=21.44`
  - `avg_interactions=2.67`
- recurrent:
  - `success_rate=0.5556`
  - `avg_return=0.2678`
  - `avg_steps=21.44`
  - `avg_interactions=2.56`

Per-variant hybrid split:

- `switch_unlock/red`: `1/3`
- `switch_unlock/blue`: `2/3`
- `switch_unlock/green`: `2/3`

This is not full parity restoration.
It is, however, strong evidence that the main failure was in generic runtime control mechanics, not in the rewritten synthetic environment itself.

Most useful single trace:

- old checkpoint on `switch_unlock/red`, seed `17`
- current repaired behavior:
  - move toward the correct switch
  - probe the adjacent interaction explicitly
  - unlock the goal
  - route to the active target
  - solve in `7` steps

Follow-up note:

- two more generic controller refinements were tried after that recovery:
  - probe-momentum state for unresolved interaction targets
  - single-failure removal from the active interaction-candidate set
- those changed local traces but did not improve the aggregate narrow solved-count beyond `5/9`
- so the remaining old-slice gap is now characterized as:
  - unresolved-interaction commitment
  - and post-unlock local path commitment

## Eval Path Hard Boundary Update

The learned eval path has now been tightened to satisfy the stronger boundary the user asked for.

What changed:

- `arcagi/agents/learned_agent.py`
  - the hand-authored `RuntimeRuleController` is now constructed only when `use_runtime_controller=True`
  - the harness-built learned eval agents keep `use_runtime_controller=False`
  - so the default learned eval path does not even instantiate the controller, rather than merely declining to consult it at action time
- `arcagi/planning/planner.py`
  - removed heuristic question-token synthesis from the learned planner
  - if the learned language head emits no question tokens, the planner now carries an empty question tuple instead of manufacturing one from tags or color names
- tests
  - added `tests/test_eval_path_constraints.py`
  - extended `tests/test_planner_runtime_thought.py`

Measured verification:

- direct harness constraint check:
  - `build_agent("hybrid")` now yields:
    - `use_runtime_controller=False`
    - `runtime_rule_controller is None`
- file-by-file clean test sweep:
  - `tests/test_action_encoder.py`: `4 passed`
  - `tests/test_arc_adapter_helpers.py`: `2 passed`
  - `tests/test_eval_path_constraints.py`: `1 passed`
  - `tests/test_generic_perception.py`: `1 passed`
  - `tests/test_graph_memory.py`: `1 passed`
  - `tests/test_perception.py`: `1 passed`
  - `tests/test_planner_runtime_thought.py`: `3 passed`
  - `tests/test_rule_induction.py`: `1 passed`
  - `tests/test_runtime_rule_controller.py`: `5 passed`
  - `tests/test_smoke_pipeline.py`: `1 passed in 86.14s`
  - `tests/test_synthetic_env.py`: `3 passed`
  - `tests/test_synthetic_oracle.py`: `2 passed`
- aggregate:
  - `25 passed`

Interpretation:

- the hand-authored runtime controller is now bootstrap/training-only by explicit opt-in
- the submission-relevant learned eval path makes decisions through the learned world model, grounded language head, episodic memory, graph statistics, and online updates only
- there is no controller interception layer in that default learned path
- there is no heuristic question-token fallback in that default learned path

## Clean Learned Path Status After Further Iteration

I continued iterating under the stricter boundary:

- no hand-authored runtime controller in the learned eval path
- no heuristic question-token fallback
- no benchmark-specific controller interception

Additional generic mechanisms added to the clean learned path:

- motion-sensitive `transition_vector()` for object-layout change
- rule-inducer action-family stats and no-effect tracking
- rule-inducer mover/stable-object progress bonus from live motion history
- graph-cycle penalties in the learned planner
- graph-style exploration signals folded into the learned planner:
  - global action novelty
  - family novelty
  - click-bin novelty
  - repeat penalty
  - stuck bonus
- family-level no-effect suppression for dead click/select families
- delayed online credit for preparatory actions
- true within-episode world-model adaptation with episode-local reset

I also added Stage 2 public ARC training support:

- `arcagi/training/arc_public.py`
  - collects public ARC transitions with generic exploration policies
  - fine-tunes the encoder/world model on real ARC dynamics

Measured verification:

- `python -m pytest -q`
  - `28 passed in 45.12s`

Measured clean learned path result with the default checkpoint `artifacts/mixed_policy_hybrid.pt`:

- offline `AR25`
  - `success=false`
  - `return=0.0`
  - `steps=112`
  - `interaction_steps=10`
- online 5-game slice
  - `AR25`: fail, `112` steps, `10` interactions
  - `BP35`: fail, `24` steps, `0` interactions
  - `CD82`: fail, `100` steps, `1` interaction
  - `CN04`: fail, `75` steps, `75` interactions
  - `DC22`: fail, `128` steps, `126` interactions
  - aggregate: `0/5`

Measured Stage 2 public ARC fine-tune attempts:

- 5-game smoke pass
  - checkpoint: `artifacts/arc_public_stage2_smoke.pt`
  - `297` transitions
  - `loss_last_epoch=4401.35`
  - `policy_loss_last_epoch=1.6966`
  - result: still `0/5`
- 25-game full public pass
  - checkpoint: `artifacts/arc_public_stage2_full.pt`
  - `1175` transitions
  - `loss_last_epoch=6744.20`
  - `policy_loss_last_epoch=0.7580`
  - result: still `0/5`

Current interpretation:

- the clean planner is no longer failing in exactly the old way
- it now explores more broadly and uses more genuine online updates
- but it still does not infer the right objective strongly enough on `AR25`
- and it still falls into heavy click-family churn on `CN04` and `DC22`
- the current Stage 2 public-ARC training signal is too weak and noisy to close that gap yet

## Pre-Retrain Pipeline Audit

Before retraining again, I audited the current pipeline for structural mismatches rather than assuming the remaining gap is purely about more data or more compute.

Verification completed:

- widened encoder patch finished and landed
  - `MAX_OBJECTS=64`
  - attention-pooled object context
  - learned whole-grid residual branch
- legacy checkpoint compatibility added
- `tests/test_encoder.py` added
- full suite:
  - `python -m pytest -q`
  - `30 passed in 47.29s`
- old checkpoint `artifacts/mixed_policy_hybrid.pt` still measures:
  - offline `AR25`: fail, `112` steps, `10` interactions
  - online `AR25`: fail, `112` steps, `10` interactions

Important confirmation:

- the old checkpoint loads into the widened encoder with `enhancement_gate = 0.0`
- so the new grid residual branch is intentionally inactive for that legacy checkpoint until retraining teaches it useful weights

Concrete audit findings:

1. Imagined rollout context is still stale.
   - `arcagi/planning/planner.py`
   - deeper lookahead reuses the original `StructuredState` and original affordance context at every imagined node
   - selector follow-up scoring also uses the pre-selector state context rather than an imagined post-selector structured-state proxy
   - this means deeper search is not truly reasoning over changed state-dependent action semantics

2. Search still rewards uncertainty inside the rollout objective.
   - `arcagi/planning/planner.py`
   - `_lookahead(...)` adds `prediction.uncertainty` directly to the imagined branch score
   - top-level action scoring already rewards entropy/disagreement separately
   - this double-counts exploration pressure and can favor noisy branches for the wrong reason

3. Synthetic graph recency is currently wrong across episodes.
   - `arcagi/memory/graph.py`
   - `last_seen_step` stores only local `step_index`
   - synthetic family adaptation intentionally preserves graph memory across `reset_episode()`
   - when the next episode starts back at step `0`, old nodes can look falsely "recent"
   - this contaminates cycle penalties and therefore contaminates graph-guided exploration and Stage 1 data collection

4. ARC click affordances are still biased and truncated.
   - `arcagi/envs/arc_adapter.py`
   - click candidates are generated from component representatives sorted by ascending area
   - only the first `16` are kept
   - this can omit larger semantically important objects when a board has many components

5. Graph fingerprints are still brittle because relations depend on local object ids.
   - `arcagi/perception/object_encoder.py`
   - relations use `source_id=obj_k`, `target_id=obj_j`
   - `arcagi/core/types.py`
   - `StructuredState.fingerprint()` includes those ids
   - this weakens graph reuse because semantically similar states can hash differently under different connected-component enumeration

6. Stage 2 public ARC training still leaves grounded language stale.
   - `arcagi/training/arc_public.py`
   - encoder/world model are trainable, language is loaded but never adapted
   - this likely contributes to the degenerate repeated language traces seen on real ARC

7. Compatibility loading is necessary right now but too silent.
   - `arcagi/training/synthetic.py`
   - checkpoint migration uses `strict=False`
   - missing and unexpected keys are not currently surfaced in logs
   - this is acceptable temporarily for encoder migration, but not good enough as a stable retraining workflow

Pre-retrain conclusion:

- the widened encoder is necessary but not sufficient
- retraining immediately without fixing these remaining mechanism bugs would mix real prior-learning work with avoidable pipeline corruption
- the next fixes should target:
  - imagined state/context correctness in rollout
  - graph recency correctness across persisted synthetic episodes
  - click-affordance coverage quality on ARC

## Audit Fixes Landed

I completed the full code sweep for the audit items before any retrain.

Changes implemented:

1. `arcagi/planning/planner.py`
   - imagined selector follow-up now uses post-action imagined state proxies
   - deeper lookahead no longer reuses the stale root structured state
   - imagined branch ordering is recomputed from imagined latent/state context
   - rollout value no longer rewards raw uncertainty directly

2. `arcagi/memory/graph.py`
   - recency is now tracked by monotonic `last_seen_tick` instead of local episode `step_index`
   - this removes cross-episode cycle-penalty corruption when graph memory persists across `reset_episode()`

3. `arcagi/envs/arc_adapter.py`
   - click affordances now use area-aware, spatially diverse representative ranking
   - default click candidate cap increased from `16` to `24`
   - ARC observations now expose `background_color`

4. `arcagi/core/types.py` and `arcagi/perception/object_encoder.py`
   - object extraction is canonically sorted before object ids are assigned
   - component cell lists are sorted
   - fingerprints now use canonical object descriptors rather than brittle local object ids
   - summary/object/transition color features now bucket arbitrary color ids rather than assuming exact palette ids stop at `11`

5. `arcagi/training/arc_public.py`
   - Stage 2 public ARC adaptation now trains the grounded language channel using generic observation/action/transition-derived pseudo-targets

6. `arcagi/training/synthetic.py`
   - compatibility checkpoint loads now log missing, unexpected, and shape-mismatched keys instead of silently swallowing them

7. `arcagi/models/encoder.py`
   - the grid branch now uses `GRID_VALUE_BUCKETS=256` instead of the old `Embedding(16, ...)` ceiling

Validation:

- focused regression surface:
  - `19 passed`
- full suite:
  - `python -m pytest -q`
  - `39 passed in 131.13s`

Measured post-fix baseline with the old checkpoint `artifacts/mixed_policy_hybrid.pt`:

- offline `AR25`
  - before audit-fix sweep: fail, `112` steps, `10` interactions
  - after audit-fix sweep: fail, `65` steps, `1` interaction
- online 5-game slice after audit-fix sweep
  - `AR25`: fail, `65` steps, `1` interaction
  - `BP35`: fail, `24` steps, `0` interactions
  - `CD82`: fail, `100` steps, `0` interactions
  - `CN04`: fail, `75` steps, `0` interactions
  - `DC22`: fail, `128` steps, `1` interaction

Interpretation:

- the post-fix clean learned path is still not solving the public slice
- but the path is materially cleaner and less corrupted
- offline `AR25` improving from `112/10` to `65/1` without retraining is strong evidence that these were real pipeline bugs, not cosmetic cleanups
- the public click-churn failure mode largely collapsed before retraining, which is exactly what the click-coverage, imagined-state, and recency fixes were supposed to address

## First Post-Audit Retrain

After the audit fixes, I ran the first real GPU retrain against the corrected pipeline.

Before relaunching I fixed the trainer ergonomics:

- `arcagi/training/synthetic.py`
  - now prints per-epoch JSON metrics
  - now writes the checkpoint after every epoch

Reason:

- the first attempted `16 x 192` run was too long and opaque for fast iteration
- I killed that oversized run and relaunched a shorter visible schedule

Completed retrain:

- command
  - `python -m arcagi.training.synthetic --epochs 8 --episodes-per-epoch 128 --learning-rate 2e-4 --checkpoint-path artifacts/post_audit_stage1_hybrid.pt --init-checkpoint-path artifacts/mixed_policy_hybrid.pt --behavior-policy mixed --curriculum staged`
- hardware
  - `2x RTX 5060 Ti` visible
  - trainer used CUDA on GPU `0`
- checkpoint
  - `artifacts/post_audit_stage1_hybrid.pt`

Per-epoch metrics:

- epoch `0`: `loss 0.6483`, `uncertainty 0.0265`, `samples 2643`
- epoch `1`: `loss 0.6337`, `uncertainty 0.0250`, `samples 2643`
- epoch `2`: `loss 0.6996`, `uncertainty 0.0373`, `samples 3085`
- epoch `3`: `loss 0.6175`, `uncertainty 0.0398`, `samples 3085`
- epoch `4`: `loss 0.5940`, `uncertainty 0.0431`, `samples 3085`
- epoch `5`: `loss 0.6224`, `uncertainty 0.0491`, `samples 3422`
- epoch `6`: `loss 0.5823`, `uncertainty 0.0538`, `samples 3422`
- epoch `7`: `loss 0.5940`, `uncertainty 0.0530`, `samples 3422`

Interpretation:

- the staged-curriculum transitions are visible in the loss curve, which is expected
- but later-stage uncertainty rose materially
- that was an early warning that the model still was not really mastering the harder selector/compositional families

Measured checkpoint behavior:

- synthetic widened benchmark
  - `success_rate = 0.0625`
  - `avg_return = -0.5381`
  - `avg_steps = 34.97`
  - `avg_interactions = 8.34`
  - `first_episode_success = 0.125`
  - `later_episode_success = 0.0`
- offline `AR25`
  - fail
  - `84` steps
  - `1` interaction
- online 5-game slice
  - `AR25`: fail, `84` steps, `1` interaction
  - `BP35`: fail, `24` steps, `0` interactions
  - `CD82`: fail, `100` steps, `1` interaction
  - `CN04`: fail, `75` steps, `0` interactions
  - `DC22`: fail, `128` steps, `126` interactions

Comparison to the post-audit old-checkpoint baseline:

- old checkpoint after the audit fixes
  - offline `AR25`: fail, `65` steps, `1` interaction
  - online `AR25`: fail, `65` steps, `1` interaction
  - online `DC22`: fail, `128` steps, `1` interaction
- new retrained checkpoint
  - offline `AR25`: regressed to `84` steps
  - online `AR25`: regressed to `84` steps
  - online `DC22`: catastrophically regressed to `126` interactions

Conclusion:

- retraining was appropriate after the audit fixes
- this checkpoint is not a promotion
- the current Stage 1 objective is still teaching the wrong action/language priors on the hard families
- specifically, the cleaned inference stack got better, but the new weights pushed it back toward bad selector/click behavior rather than amplifying the post-audit gains

## Manual Long-Run Trainer Instrumentation

The Stage 1 trainer now supports real manual long runs instead of forcing blind end-only monitoring.

Implemented in `arcagi/training/synthetic.py`:

- added `--log-every-episodes`
- `collect_dataset(...)` now prints interval JSON progress lines with:
  - epoch
  - episode index
  - family mode / variant / board size
  - episode return / steps / success
  - running and interval averages
  - collected sample count
  - elapsed seconds
- `train_synthetic(...)` now prints richer epoch JSON with:
  - loss
  - uncertainty
  - collection averages
  - collection, training, and total epoch time
  - latest checkpoint path
  - epoch checkpoint path
- every completed epoch now writes:
  - the requested `--checkpoint-path` as the rolling latest checkpoint
  - a preserved epoch snapshot such as `artifacts/tmp.epoch_0007.pt`
- `KeyboardInterrupt` now triggers an explicit interrupt snapshot:
  - `artifacts/tmp.interrupt.pt`
  - includes `training_state` metadata with partial-epoch metrics

Validation commands:

- syntax
  - `python -m py_compile arcagi/training/synthetic.py`
- short normal run
  - `python -m arcagi.training.synthetic --epochs 1 --episodes-per-epoch 4 --learning-rate 1e-4 --checkpoint-path artifacts/tmp_manual_train.pt --behavior-policy mixed --curriculum staged --log-every-episodes 2`
- short interrupt run
  - launched a longer run with `--checkpoint-path artifacts/tmp_interrupt_train.pt`
  - sent `Ctrl-C`

Observed behavior:

- normal run emitted `episode_progress` JSON twice, then an `epoch` JSON, then the final metrics JSON
- normal run wrote:
  - `artifacts/tmp_manual_train.pt`
  - `artifacts/tmp_manual_train.epoch_0000.pt`
- interrupt run wrote:
  - `artifacts/tmp_interrupt_train.pt`
  - `artifacts/tmp_interrupt_train.epoch_0000.pt`
  - `artifacts/tmp_interrupt_train.interrupt.pt`
- interrupt snapshot preserved:
  - `completed_epochs = 1`
  - `active_epoch = 1`
  - `interrupted = True`
  - partial-epoch loss / uncertainty / sample count

Implication:

- long retrains can now be supervised manually with useful visibility
- the best epoch can be selected after the fact instead of only keeping the last one
- `Ctrl-C` no longer destroys the current model state

## Manual Run Review: Hidden Repeated-Epoch Sampling

Reviewing the user's long run exposed a concrete Stage 1 trainer flaw.

Observed:

- epoch `7` and epoch `8` produced effectively identical collection summaries:
  - `collect_avg_return = 1.0226171875`
  - `collect_avg_steps = 20.1953125`
  - `collect_success_rate = 0.9921875`
  - `samples = 5170`
- visible `episode_progress` lines all showed `family_mode = order_collect`

Mechanism:

- `collect_dataset(...)` resets `seed_cursor = config.seed` at the start of every epoch
- family, variant, size, and env seeds are then replayed deterministically from that same start point
- with `--epochs 64`, the staged schedule keeps epochs `0..20` on the first family block:
  - `switch_unlock`
  - `order_collect`
- therefore epochs `7` and `8` are not just similar; they are effectively the same sampled corpus replayed again

Why the logs looked one-family:

- first-stage families alternate by episode index
- the run used `--log-every-episodes 16`
- logging only every 16th episode lands on the same parity each time, so the visible lines repeatedly hit the even-slot family and therefore print `order_collect`
- interval success dips such as `0.9375` indicate hidden failures inside the 16-episode window, likely on the unlogged alternating family

Implication:

- current long-run training gets optimizer time, but not fresh procedural experience, until the curriculum stage changes
- that weakens the value of simply "training longer" within a stage
- the trainer still needs:
  - epoch-dependent synthetic reseeding
  - interval summaries that expose family composition instead of aliasing to one visible family

## Repeated-Epoch Sampling Fix

The Stage 1 trainer now fixes the repeated-corpus flaw identified from the manual long run.

Implemented in `arcagi/training/synthetic.py`:

- `_epoch_seed_base(config, epoch_index)` now derives a deterministic epoch-specific seed
- `collect_dataset(...)` uses that epoch-specific base instead of replaying from the same `config.seed` every epoch
- interval progress logs now include:
  - `running_family_counts`
  - `interval_family_counts`
- epoch summaries now include:
  - `family_counts`
  - `epoch_seed_base`

Effect:

- each epoch now samples a fresh deterministic synthetic slice within the same curriculum stage
- long runs now add procedural diversity instead of mainly re-optimizing on a stage-local replay
- interval logs can no longer hide alternating families behind a parity-aligned stride such as `--log-every-episodes 16`

Validation:

- `python -m pytest tests/test_synthetic_training.py -q`
  - `2 passed`
- `python -m pytest tests/test_synthetic_training.py tests/test_smoke_pipeline.py -q`
  - `3 passed in 132.88s`
- direct probe confirmed:
  - epoch `7` and `8` now have different `epoch_seed_base` values
  - epoch `7` and `8` now have different initial episode signatures within the same staged family block
  - `episode_progress` logs expose both `switch_unlock` and `order_collect` in `interval_family_counts`

## Held-Out Gated Curriculum

Stage 1 no longer relies on the old fixed epoch-thirds schedule when `--curriculum staged` is used. It now uses held-out gated promotion with replay.

Implemented in `arcagi/training/synthetic.py`:

- `CurriculumStage` definitions:
  - `foundation`
    - frontier: `switch_unlock`, `order_collect`
  - `hidden_modes`
    - frontier: `selector_unlock`, `delayed_order_unlock`
  - `compositional`
    - frontier: `selector_sequence_unlock`
- replay-weighted training family sampling:
  - current frontier families get `frontier_replay_weight`
  - previous-stage families remain in the mix with `previous_stage_replay_weight`
- in-memory held-out hybrid-agent eval:
  - no disk reload required
  - unseen deterministic holdout seed slices
  - stage-specific holdout size options
- promotion logic:
  - frontier holdout must clear thresholds on:
    - `first_episode_success`
    - `later_episode_success`
    - `avg_return`
    - `avg_interactions`
  - regression floors on previous-stage families must hold where applicable
  - promotion requires consecutive passing evals
- new trainer JSON events:
  - `holdout_eval`
  - `stage_advance`

Operational defaults:

- `--holdout-eval-every-epochs` now defaults to `4`
- reason:
  - real held-out learned-agent eval is expensive enough that every-2-epochs default would dominate wall-clock on long manual runs
- smoke tests explicitly disable holdout gating with `holdout_eval_every_epochs=0`

Validation:

- `python -m pytest tests/test_synthetic_training.py -q`
  - `4 passed`
- `python -m pytest tests/test_synthetic_training.py tests/test_smoke_pipeline.py -q`
  - `5 passed in 134.86s`
- tiny live trainer probe:
  - `python -m arcagi.training.synthetic --epochs 2 --episodes-per-epoch 4 --learning-rate 1e-4 --checkpoint-path artifacts/tmp_gated_train.pt --behavior-policy mixed --curriculum staged --log-every-episodes 2 --holdout-eval-every-epochs 1 --holdout-episodes-per-variant 1 --promotion-consecutive-evals 1`
  - observed:
    - `holdout_eval` JSON after each epoch
    - epoch JSON now includes `stage_index`, `stage_name`, `stage_epoch_count`, `frontier_families`, `family_counts`, and `epoch_seed_base`
    - final metrics JSON includes `last_holdout_result`

Interpretation:

- Stage 1 advancement is now tied to held-out mechanism generalization and regression retention instead of raw epoch count
- this is a better curriculum controller, but it does not by itself fix the still-poor learned priors

## Verbose Holdout Diagnosis

The Stage 1 trainer was further instrumented to make failure modes causal rather than scalar.

New holdout diagnostics now include:

- per-step traces for failed held-out episodes
- action-family counts
- event counts
- repeated-action-loop metrics
- failure signatures
- top runtime-thought action candidates
- grounded belief/question tokens
- synthetic latent-state diagnostics:
  - `selected_color`
  - `progress_index`
  - `goal_active`
- persisted `last_holdout_result` inside the checkpoint snapshot

Code paths:

- `arcagi/training/synthetic.py`
- `arcagi/agents/learned_agent.py`

Validation:

- `python -m py_compile arcagi/agents/learned_agent.py arcagi/training/synthetic.py`
- `python -m pytest tests/test_synthetic_training.py -q`
  - `4 passed`

Diagnostic continuation used to inspect the current checkpoint:

```bash
python -m arcagi.training.synthetic --epochs 1 --episodes-per-epoch 64 --learning-rate 2e-4 --checkpoint-path artifacts/manual_stage1_diag.pt --init-checkpoint-path artifacts/manual_stage1.pt --behavior-policy mixed --curriculum staged --log-every-episodes 32 --holdout-eval-every-epochs 1 --holdout-episodes-per-variant 2 --promotion-consecutive-evals 2 --holdout-failure-examples 4 --holdout-trace-steps 48 --holdout-trace-top-actions 3
```

Train-side epoch result:

- `collect_success_rate = 1.0`
- `collect_avg_return = 1.0389`
- `collect_avg_steps = 18.80`
- `loss = 0.4890`
- `world_loss = 0.0676`
- `belief_loss = 0.1777`
- `question_loss = 0.1929`
- `policy_loss = 0.6205`
- `encoder_enhancement_gate_tanh = 0.00663`

Held-out foundation result on unseen size `9`:

- `success_rate = 0.4`
- `avg_return = -0.356`
- `avg_steps = 41.3`
- `avg_interactions = 10.0`
- `first_episode_success = 0.4`
- `later_episode_success = 0.4`
- promotion-gate failures:
  - `frontier_first_episode_success<0.80:0.400`
  - `frontier_later_episode_success<0.92:0.400`
  - `frontier_avg_return<0.70:-0.356`

Per-family breakdown:

- `switch_unlock`
  - `success_rate = 0.6667`
  - `avg_return = 0.1017`
  - `avg_steps = 36.83`
  - `avg_interactions = 8.83`
- `order_collect`
  - `success_rate = 0.0`
  - `avg_return = -1.0425`
  - `avg_steps = 48.0`
  - `avg_interactions = 11.75`

Global held-out event signature:

- `empty_interaction = 90`
- `move = 263`
- `blocked_by_object = 33`
- `correct_switch = 4`
- `correct_collect = 2`
- `wrong_order = 4`
- `goal_reached = 4`
- terminations:
  - `timeout = 6`
  - `goal_reached = 4`

Observed root cause:

1. This is not train optimization failure. Foundation train performance is saturated, but held-out size generalization is poor.
2. The policy/language prior is miscalibrated under size shift. Failed `switch_unlock` traces on red and green variants still begin with belief/question tokens about the `blue` switch.
3. The planner is not suppressing no-effect interactions hard enough. The held-out foundation run still burns `90` `empty_interaction` events across only `10` episodes.
4. `switch_unlock` partially survives the size shift, but mostly by wandering until a few successes happen. The failed traces show movement plus empty interactions, not deliberate corrective probing.
5. `order_collect` is the dominant foundation blocker. On unseen size `9` it drops to `0.0` success. The agent occasionally reaches a first `correct_collect` or triggers `wrong_order`, but it does not consolidate that evidence into a stable sequence rule and times out instead.
6. The new encoder whole-grid branch is still effectively off because `encoder_enhancement_gate_tanh` remains near zero. So the widened architecture is not yet materially helping held-out larger boards.
7. The weakest learned head is the action prior. `policy_loss` remains much larger than `world_loss`, which matches the behavioral signature of poor exploratory/exploit decisions despite decent state-prediction loss.

Current conclusion:

- the immediate Stage 1 generalization failure is a combination of:
  - brittle language/policy priors on unseen size `9`
  - weak suppression of no-effect interaction families
  - failure to convert partial sequence evidence into stable `order_collect` hypotheses
  - an almost-closed whole-grid enhancement branch

## Clean-Path Repair Pass

After the verbose holdout diagnosis, I fixed the concrete clean-path bugs that were still poisoning evaluation:

- episodic retrieval now separates generic-token overlap from content overlap
- default `goal_active=0` state no longer becomes a memory-driving claim
- positive counterfactual replay no longer writes `recommended_action` memory entries
- `interact_*` now gets a generic geometry-grounded prior:
  - bonus when something is actually adjacent
  - penalty when the interaction cell is empty
- `wait` is no longer free
- unseen-action optimism in `rule_induction.py` was reduced sharply
- Stage 1 language targets now expose sequence progress from observable inventory/object state
- whole-grid enhancement gate defaults were increased to make encoder expansion easier to learn

Validation:

- `python -m py_compile arcagi/agents/learned_agent.py arcagi/planning/planner.py arcagi/memory/episodic.py arcagi/planning/rule_induction.py arcagi/training/synthetic.py`
- `python -m pytest tests/test_synthetic_training.py tests/test_planner_runtime_thought.py tests/test_encoder.py -q`
  - `12 passed`

Short continuation measurements from `artifacts/manual_stage1.pt`:

1. `manual_stage1_fixpass.pt`
   - `success_rate = 0.1`
   - `avg_return = -0.317`
   - `avg_interactions = 14.2`

2. `manual_stage1_fixpass2.pt`
   - `success_rate = 0.0`
   - `avg_return = -1.123`
   - `avg_interactions = 12.1`

3. `manual_stage1_fixpass3.pt`
   - `success_rate = 0.0`
   - `avg_return = -0.968`
   - `avg_interactions = 13.5`

Interpretation:

- the worst clean-path contamination routes are now reduced
- the residual failure is no longer “a hidden fallback/controller is rescuing or poisoning eval”
- the residual failure is the learned prior itself:
  - under held-out size shift, the world model and policy heads still hallucinate positive value for unproductive interaction families
  - `order_collect` remains the main blocker
  - the repaired objective still needs longer retraining to determine whether the cleaned mechanism fixes can actually be absorbed by the model

## Killed Long Run Postmortem And Root Fixes

Mining the interrupted `manual_stage1_long` checkpoint chain exposed two structural Stage 1 bugs:

1. Two-variant family aliasing:
   - `collect_dataset(...)` was choosing family variants with a stride that aliases modulo `2`
   - so `order_collect` and `delayed_order_unlock` trained on only one order variant per epoch
   - this directly undermined sequence generalization

2. Post-goal reward farming:
   - `switch_unlock` continued to pay `correct_switch` even after `goal_active = 1`
   - the epoch-2 holdout artifacts contained a direct failure case with repeated `correct_switch` after activation and timeout

Fixes landed:

- `arcagi/training/synthetic.py`
  - per-family variant scheduling now round-robins variants with a seeded family-specific offset
  - language supervision now uses `next_state` so activation teaches:
    - belief: `goal active`
    - question: `move toward target`
  - checkpoints now persist `holdout_history`

- `arcagi/envs/synthetic.py`
  - redundant post-goal mechanism interactions now return:
    - reward `-0.05`
    - event `redundant_post_goal_interaction`

Validation:

- `python -m py_compile arcagi/training/synthetic.py arcagi/envs/synthetic.py tests/test_synthetic_env.py tests/test_synthetic_training.py`
- `python -m pytest tests/test_synthetic_env.py tests/test_synthetic_training.py -q`
  - `9 passed`

Short repaired continuation from `manual_stage1_long.epoch_0004.pt`:

- train slice:
  - `collect_success_rate = 1.0`
  - both `order_collect` variants now appear within a single epoch:
    - `blue_then_red = 12`
    - `red_then_blue = 12`

- held-out foundation:
  - `success_rate = 0.2`
  - `avg_return = -0.910`
  - `avg_interactions = 12.8`

- family breakdown:
  - `order_collect`: `0.25` success
  - `switch_unlock`: `0.1667` success

Interpretation:

- the artifact-derived root fixes are real
- `order_collect` is no longer flatlined after a single repaired continuation epoch
- the post-goal interaction farming loop is now explicitly suppressed
- the remaining blocker is now narrower:
  - early `switch_unlock` discovery under held-out size shift is still weak
  - longer retraining under the repaired sampler and repaired post-goal semantics is now the correct next experiment

## 2026-04-20 Long Manual Run Promotion

The earlier foundation blockage documented above is now historical rather than current. A longer manual Stage 1 run using the repaired runtime stack and `25,000` episodes per epoch advanced beyond foundation and into `hidden_modes` by epoch `4`.

End-of-epoch `4` running metrics from the promoted regime:

- `stage_name = hidden_modes`
- `frontier_families = ["selector_unlock", "delayed_order_unlock"]`
- `running_success_rate = 0.45072`
- `running_avg_return = -0.0482464`
- `running_avg_steps = 33.6464`
- `samples_collected = 841160`
- `elapsed_seconds = 2631.7117`
- `teacher_episode_fraction = 0.2`
- `teacher_takeover_prob = 0.5`
- `teacher_episode_count = 5062`
- `teacher_step_fraction = 0.259249`
- `teacher_relabel_fraction = 0.621525`

Recent interval windows near the end of epoch `4` still fluctuated materially:

- interval success moved between `0.375` and `0.500`
- interval average steps moved between `30.94` and `35.56`
- failed episodes still frequently timed out at `48` steps

Interpretation:

1. The repo is no longer stuck at the old foundation generalization wall. The repaired runtime path is now strong enough to satisfy the staged promotion gate and expose the harder hidden-mode families.
2. The architecture changes were behaviorally relevant, not just test-clean:
   - executable posterior competition now survives into longer training
   - live representation repair and local patches did not destabilize the long run
   - option induction did not block promotion and is compatible with the promoted controller
3. The next bottleneck is not "can foundation be cleared?" It is "can the promoted learner own hidden-mode control and sequence mechanics reliably?"
4. Teacher support is still carrying a meaningful part of the run. `teacher_step_fraction≈0.259` and `teacher_relabel_fraction≈0.622` are too high to claim robust learner-owned hidden-mode competence.
5. The correct next experiments are:
   - capture and index the exact promoted checkpoint and holdout payload
   - run held-out diagnostics on `selector_unlock` and `delayed_order_unlock`
   - reduce teacher dependence without losing the promotion
   - stabilize post-promotion performance so interval success does not keep swinging around the `0.4-0.5` band
