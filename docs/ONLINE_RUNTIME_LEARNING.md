# Online Runtime Learning

## Purpose

This document defines what "live self-learning" should mean in this repo.

The requirement is not just:

- run a pretrained checkpoint
- collect a few statistics
- slowly improve after many offline training runs

The requirement is:

- infer rules online while the episode is live
- test those rules with interventions
- revise them immediately when prediction fails
- let the revised beliefs change action choice during the same episode

This is the central mechanism for any serious ARC-facing or broader non-text-game-facing agent.

## Core Thesis

Everything interactive is constrained by rules.

Those rules may be:

- simple or compositional
- visible or latent
- deterministic or stochastic
- local or temporally extended
- object-based, relational, inventory-based, or sequence-based

The runtime learner should therefore not be framed as "just do online RL" or "just update the policy head."

The right framing is:

1. infer compact causal rules from transitions
2. maintain uncertainty over competing rule hypotheses
3. choose actions that either exploit a good rule or disambiguate bad ones
4. replay failures mentally or computationally
5. revise the action prior, memory, and local dynamics model immediately

## What Counts As Meaningful Online Self-Correction

### 1. Action-Semantics Learning

The agent should learn what actions do online.

Examples:

- "`interact_right` toggles adjacent switches"
- "`click:x:y` affects the nearest small component"
- "`select` changes which object subsequent moves control"
- "`undo` restores a local latent state rather than just grid pixels"

This should be learned from precondition/effect pairs rather than assumed from benchmark-specific code.

### 2. Object-Role Learning

The agent should infer object roles online.

Examples:

- "this compact object behaves like a switch candidate"
- "this elongated object behaves like a wall or blocker"
- "this object changes reward-significantly when interacted with"
- "this object participates in sequence constraints"

Role assignments should be soft, revisable, and evidence-based.

### 3. Transition-Rule Learning

The agent should infer rules of the form:

`preconditions -> effects`

where preconditions and effects are expressed over compact grounded predicates derived from the current structured state.

Examples:

- `adjacent(agent,obj) + action=interact_right + obj.role=switch -> flag(goal_active)=1`
- `inventory.prefix=red + action=interact_up + target.color=blue -> reward positive`
- `action=move_right + selected_object=obj_k -> centroid(obj_k).x += 1`

### 4. Reward-Rule Learning

The agent should infer what produces reward or progress.

Examples:

- reward depends on touching target only after activation
- reward depends on collection order
- reward depends on moving the right object, not the agent avatar
- reward is sparse and only appears after a hidden sequence is satisfied

This matters because reward rules often define the task more directly than transition rules.

### 5. Hidden-State Belief Learning

The agent should maintain latent hypotheses when the world is partially observed.

Examples:

- required order is `{red, blue}` or `{blue, red}`
- currently selected object is one of several candidates
- switch state is active or inactive
- reward condition depends on an unseen bit changed by a past action

The point is not to pretend the hidden state is known.
The point is to update beliefs about it explicitly as evidence arrives.

### 6. Search-Control Learning

The agent should learn which actions are diagnostic.

Examples:

- "clicking different parameter bins is more informative than repeating movement"
- "interact next to the suspicious object is diagnostic"
- "selection change before movement is diagnostic"

This is rule learning about epistemic value, not only reward.

## Mechanisms For Online Rule Learning

### 1. Explicit Rule Objects

Do not keep only scalar action scores.

Maintain explicit rule candidates with fields such as:

- preconditions
- predicted effects
- confidence
- support count
- exception count
- scope or object family
- whether the rule is causal, correlational, or still ambiguous

### 2. Hypothesis Competition

The agent should not collapse too early to one story.

For many tasks there should be several live candidates at once.

Examples:

- "reward depends on color"
- "reward depends on order"
- "reward depends on adjacency"
- "reward depends on selection state"

Each transition should update confidence over these candidates.

### 3. Intervention Planning

The agent should deliberately choose actions that separate competing hypotheses.

Examples:

- if two rules predict the same reward but different object changes, choose the action that reveals the difference
- if one action tests color dependence and another tests ordering dependence, pick the one with higher expected information gain

Without intervention planning, the agent drifts into memorization or blind search.

### 4. Counterfactual Replay

After a bad or stalled action, the agent should revisit the same pre-action state and ask:

- what else could I have done here?
- what did my current model predict for those alternatives?
- which alternative seems more plausible now?

This replay should update:

- avoid memory
- prefer memory
- action bias
- local rule confidence

This is the machine analog of mentally replaying mistakes until a better action pattern emerges.

### 5. Fast Online Preference Updates

The agent should maintain per-episode action biases that update immediately from:

- observed reward
- observed stall
- counterfactual replay gap
- contradiction between expected and observed effects

These updates should be fast and local, not delayed until the next offline training run.

### 6. Online Model Editing

When the world model repeatedly mispredicts a local mechanic, the runtime agent should be able to patch a small local adapter or fast-weight layer.

Important constraint:

- this should be narrow and stable
- do not destabilize the entire model during one episode

Examples:

- small per-episode action embedding correction
- local low-rank adapter on transition prediction
- temporary object-role correction table

### 7. Representation Repair

Sometimes the rule learner fails because the state abstraction is wrong.

Examples:

- one perceived "object" actually contains two behaviorally distinct parts
- two identically colored components have different roles
- the relevant variable is a relation or ordering, not an object property

The agent must be allowed to refine its representation online, not just its policy.

### 8. Exception Memory

Some rules are mostly true with sparse exceptions.

The agent should keep both:

- the compressed general rule
- a sparse memory of exceptions

This prevents the system from either overgeneralizing or memorizing everything.

### 9. Temporal Abstraction

Some rules only become visible at the option level.

Examples:

- "approach object then interact"
- "cycle selection then move"
- "collect A before B"

The agent should infer and reuse short temporal chunks, not only primitive-action preferences.

## Three Learning Timescales

### 1. Instant Online Updates

These should happen within a few steps:

- action bias changes
- avoid/prefer memories
- rule confidence updates
- hidden-state posterior updates

### 2. Within-Episode Adaptation

These should happen during the episode but can use a larger computation budget:

- counterfactual replay
- local model patching
- branch simulation
- abstraction repair

### 3. Cross-Episode Consolidation

These happen offline between episodes:

- train the slow model on the fast learner's discoveries
- compress repeated local discoveries into general priors
- improve the world model, language model, and action priors

All three timescales are necessary.

If only the third exists, the runtime agent is effectively frozen.
If only the first exists, the agent remains shallow and fragile.

## What This Repo Is Still Missing

The current codebase has pieces of this, but not the full mechanism.

Missing or weak components include:

1. explicit rule objects rather than only statistics
2. real hypothesis competition
3. strong diagnostic intervention selection
4. robust counterfactual replay as a central loop rather than an auxiliary patch
5. online representation repair
6. reliable runtime policy correction for the hybrid memory path
7. tight coupling between the learned latent state, grounded scratchpad tokens, and the explicit hypothesis loop

Current measured evidence for these gaps:

- transport-style movement rules can now be learned online well enough to solve `AR25`
- the first public 5-game online slice is still only `1/5`
- the failed games show two dominant remaining gaps:
  - weak selector/click semantics learning
  - weak proof/disproof pressure on bad interaction hypotheses
- after refactoring away from hand-shaped target-ranking logic, interaction churn dropped sharply on the public 5-game slice without increasing solved-count
- that points to the next bottleneck:
  - the explicit runtime hypotheses are cleaner now, but the learned latent or scratchpad path is still too weakly integrated to complete harder objective inference
  - selector-induced hidden-state changes are not yet being evaluated through downstream latent move consequences strongly enough

## Concrete Near-Term Build Targets

### 1. Rule DSL

Add a compact rule language over grounded predicates.

Examples of predicate families:

- object color
- object size bucket
- object shape bucket
- adjacency
- containment
- selected object
- inventory bits
- recent action family
- temporal order marker

### 2. Hypothesis Manager

Maintain a live set of candidate rules with:

- support
- contradiction count
- compression score
- action suggestions
- diagnostic actions

### 3. Diagnostic Action Scorer

Score actions not only by expected reward but also by expected rule disambiguation.

### 4. Replay Buffer For Recent Bad States

Keep a small online buffer of failed or surprising transitions and revisit them for counterfactual comparison.

### 5. Runtime Patch Layer

Add a small per-episode action-preference or transition-correction module that can be updated online without destabilizing the base model.

### 6. Consistency Checker

If the rule learner, world model, memory, and empirical transition stats disagree sharply, force an experiment rather than letting one subsystem dominate silently.

## Diffusion-Inspired Hypothesis Loop

The user suggested taking inspiration from diffusion techniques for the hypothesis-to-action-to-learning-to-proof loop.

That suggestion is directionally strong, even if the final mechanism is not literally a standard image-diffusion model.

The useful diffusion-style idea is:

- start with a noisy, broad, low-confidence hypothesis cloud
- iteratively denoise it using prediction, intervention, and contradiction
- keep multiple candidates alive while progressively sharpening confidence
- let each refinement step propose better actions and better tests
- use successful tests as local proofs and failed tests as structured negative evidence

### A Practical Diffusion-Style Runtime Loop

1. Initialize a noisy hypothesis set.
The agent begins with several weak candidates for:

- object roles
- action semantics
- reward conditions
- hidden state

2. Sample candidate actions from the current hypothesis cloud.
Do not collapse too early to a single best action.
Retain diversity while uncertainty is high.

3. Predict effects under each candidate.
Use the world model, rule system, and local replay to estimate what each action would imply if each hypothesis were true.

4. Execute the most informative or promising action.
The action can be:

- exploitative
- exploratory
- disambiguating

5. Compare observed transition against predicted transition.
This is the denoising step.
Bad hypotheses lose mass.
Good hypotheses gain mass.

6. Write a local proof object.
A "proof" here means a compact record that a specific intervention supported or falsified a specific rule fragment.

7. Update action preferences and rule confidence.
The next action distribution should already be sharper.

8. Repeat until the rule cloud collapses enough to support stable exploitation or until the task is solved.

### Why This Is Attractive

This framing encourages:

- uncertainty-aware search
- iterative refinement rather than one-shot brittle commitment
- explicit use of negative evidence
- a clean bridge between world-model prediction and symbolic-ish rule induction

### What To Avoid

Do not over-interpret "diffusion" here as:

- requiring a large generative denoiser
- requiring continuous latent Gaussian noise
- replacing explicit rule learning with vague sampling tricks

The useful takeaway is the iterative denoising structure, not the branding.

## Non-Negotiable Design Constraint

Any rule-learning mechanism added here should satisfy all of the following:

- generic across ARC tasks
- transferable to broader non-text games with similar observation/action structure
- usable online within a single episode
- grounded in the agent's own observations and transitions
- compatible with consumer hardware
- not dependent on benchmark-specific heuristics
- not dependent on environment source inspection, hidden engine semantics, or per-game tuning

## Forbidden Inputs For The ARC-Facing Runtime Path

The user has now made these prohibitions explicit:

- do not treat benchmark environment files as knowledge available to the runtime agent
- do not hard-wire AR25-specific, level-specific, or task-specific semantics into rules, planners, or priors
- do not use external world knowledge or human-written task instructions at runtime
- do not claim "generalization" for a controller that was tuned to a known game

Permitted:

- generic action-interface facts exposed by the environment adapter
- generic inductive biases about objects, relations, transitions, memory, and uncertainty
- engineered rule-object formats whose contents are filled and revised online from live interaction

## Current Direction

The implementation path consistent with this document is:

1. strengthen explicit rule objects and hypothesis tracking
2. expand counterfactual replay and online action-bias updates
3. make live evidence override stale priors more aggressively
4. add hypothesis disambiguation as a first-class planning term
5. only then return to real ARC evaluation

Current implementation note:

- a first shared runtime-thought path now exists:
  - grounded tokens
  - per-action latent value/uncertainty
  - selector-conditioned hidden-state follow-up move gains
- that thought object is now consumed by both the planner and the explicit runtime hypothesis controller
- the remaining issue is not absence of coupling anymore
- the remaining issue is that selector-conditioned latent gains still need stronger proof/disproof against realized post-selector improvement

## Bounded Internal Rollouts

The runtime learner also needs a compute discipline. More internal thought is useful only if it stays inside a bounded, generic planning loop.

The rejected version:

- full-width recursive latent search
- recomputation of root predictions that were already available from the runtime thought
- no hard per-turn rollout budget

This is not acceptable on the target hardware because increasing search depth naively produces combinatorial blow-up.

The current accepted direction:

1. cache root world-model predictions in the runtime thought itself
2. rank root actions using the same grounded signals already available to the agent:
   - structured claims
   - latent action value
   - uncertainty
   - selector follow-up gain
   - graph novelty / entropy
3. expand only the top configured roots
4. expand only the top configured branch actions at imagined nodes
5. stop when the per-turn world-model-call budget is exhausted

This matters conceptually:

- the agent is still thinking further ahead
- but it is doing so through a bounded, prioritized, reusable internal search process
- this is closer to "concentrate thought where uncertainty and leverage are highest" than to "spray rollouts everywhere"

Important limitation:

- bounded search fixes rollout explosion
- it does not by itself fix bad hypotheses
- if the scratchpad claims and online rule objects are still wrong or too weak, deeper bounded rollouts will still search the wrong branches

So the next requirement after bounded search is richer live claims plus stronger proof/disproof pressure.
