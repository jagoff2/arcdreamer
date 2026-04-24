# Minimally Conscious Online Agent Threshold

This document defines the engineering threshold for a "minimally conscious" online learning agent in this repo. This is not a metaphysical claim. It is an operational requirement: the ARC-facing agent must maintain an integrated, self-updating control state that notices uncertainty, imagines action consequences, learns from surprising transitions, and uses those changes to select future actions.

## Core Criterion

A minimally conscious-enough agent must expose measurable internal variables that explain:

- what it currently believes
- what it expects an action to change
- what it is unsure about
- what question it is trying to answer
- why a specific action follows from those beliefs
- how the latest transition changed later action choice

Solving a task is not enough. A scripted planner, graph search controller, frozen policy, or lookup table may solve narrow tasks without meeting this threshold.

## Required Mechanisms

### 1. Live Belief State

The agent must maintain a structured belief state, not just hidden activations.

Minimum contents:

- observed objects and relations
- latent object states
- candidate goals
- candidate rules
- action affordances
- uncertainty over mechanics
- recent causal hypotheses
- known unknowns
- recent self-action history

Acceptance tests:

- identical observations with different transition histories produce different beliefs
- a surprising transition changes at least one explicit belief or uncertainty field
- a replay log can reconstruct the belief trajectory from observations and actions
- diagnostics can answer: current goal hypothesis, active tested rule, expected action outcome, and major uncertainty

### 2. Self And Agency Model

The agent must distinguish itself as an intervention source.

Required distinctions:

- self-controlled entity vs external objects
- self-caused changes vs autonomous/passive changes
- action success vs blocked/no-effect action
- controllable effects vs incidental visible changes

Acceptance tests:

- autonomous object movement is not automatically attributed to the agent
- repeated blocked actions reduce local action usefulness
- reliable action-object effects increase causal confidence
- ablating action-history input degrades rule inference or planning quality

### 3. Selective Global Workspace

The agent needs a small, inspectable workspace that conditions planning, memory, language, and world-model use.

Minimum workspace slots:

- selected object or object set
- active goal hypothesis
- top causal hypothesis
- current uncertainty or internal question
- candidate plan
- expected information-gain target
- relevant memory references

Selection priorities:

- novelty
- goal relevance
- uncertainty
- predicted value
- causal surprise

Acceptance tests:

- with distractor objects, workspace focuses on reward-relevant, changing, novel, or uncertain entities
- perturbing/removing workspace contents changes action choice
- workspace changes after surprising transitions
- workspace is compact enough to inspect, not a full environment dump

### 4. Agent-Owned Online Learning Loop

Every step must close this loop:

1. encode observation into structured state
2. update belief state and episodic memory
3. compare predicted vs actual transition
4. compute surprise, progress, novelty, uncertainty, and value
5. update world model, belief state, memory, and action preferences online
6. generate or update an internal question or hypothesis
7. retrieve relevant memories
8. imagine candidate action consequences
9. choose action by value, information gain, risk, and uncertainty
10. log belief, workspace, question, prediction, action, and outcome

Acceptance tests:

- online-enabled agent improves within a related episode family after one or a few exposures
- disabling online updates reduces adaptation
- prediction error drops over repeated interaction with the same mechanic
- internal questions become more specific after observations

### 5. Counterfactual Imagination

The agent must simulate candidate action consequences before acting.

Allowed sources:

- learned recurrent world model
- learned symbolic/rule hypotheses
- explicit remembered transitions as evidence, not as a scripted controller
- ensemble disagreement or uncertainty estimates

Required predicted fields:

- next latent state
- visible delta
- reward/progress
- uncertainty
- expected information gain
- likely belief update

Acceptance tests:

- in forked states, selected actions match better imagined consequences more often than chance
- ambiguous-rule tasks produce actions that distinguish rival hypotheses
- disabling rollouts reduces performance or increases steps-to-solve on tasks requiring action consequence prediction
- logs show at least two candidate counterfactuals with different predicted outcomes

### 6. Value, Affect, And Progress Signals

"Affect" means engineered control signals, not emotion simulation.

Required signals:

- reward
- progress
- novelty
- surprise
- uncertainty
- stagnation/frustration
- confidence
- curiosity or expected information gain
- irreversible-action risk/cost

Acceptance tests:

- repeated no-progress increases exploration or diagnostic pressure
- reliable goal inference increases exploitation
- high surprise triggers memory writes and belief revision
- removing curiosity/information gain reduces diagnostic action frequency

### 7. Episodic Memory And Consolidation

Memory layers:

- short-term step history
- episodic memory of surprising/useful transitions
- semantic/rule memory of consolidated mechanics
- graph memory for state/action/outcome accounting only

Memory keys should include:

- latent state
- action
- outcome delta
- active hypothesis
- internal language/query tokens
- surprise/value

Acceptance tests:

- one surprising transition is retrievable in a similar state
- repeated similar transitions strengthen a generalized rule
- memory-off ablation reduces adaptation on hidden-rule tasks
- consolidation reduces duplicates without erasing rare important cases

### 8. Metacognitive Uncertainty And Questions

The agent must estimate uncertainty over:

- current state
- goal
- transition effects
- action usefulness
- reward/progress meaning
- memory relevance

Internal questions must be grounded in current uncertainty and converted into action proposals by learned or online-updated mechanisms.

Acceptance tests:

- ambiguous tasks produce a question before a diagnostic action
- selected diagnostic action has higher expected hypothesis discrimination than random alternatives
- diagnostic result reduces uncertainty over at least one hypothesis
- language-off or question-off ablation reduces sample efficiency on tasks requiring hypothesis indexing

### 9. Grounded Internal Language

Language must be computation, not decoration.

Required vocabulary scope:

- objects
- positions
- relations
- actions
- outcomes
- hypotheses
- uncertainty
- goals
- plans
- questions

Required use:

- memory keys
- planner context
- hypothesis selection
- workspace state

Acceptance tests:

- same observation with different active hypotheses yields different internal language
- same language query retrieves relevant memories
- removing language tokens from planner/memory changes behavior
- generated strings correspond to actual state variables and hypotheses

## What Does Not Count

These can be useful baselines or infrastructure, but they do not meet the threshold:

- graph search alone
- shortest-path frontier exploration as controller
- BFS/DFS replay controller
- fixed movement sweeps
- counted movement probes
- hand-coded action-pattern enumerators
- lookup tables without causal abstraction and uncertainty
- frozen policies
- scripted planners that do not revise hypotheses
- pure next-state world models unused by agency, uncertainty, memory, and action selection
- language emitted only after decisions
- reward maximization without diagnostic behavior
- hidden neural state with no inspectable beliefs, uncertainty, memory, or questions
- default action-candidate caps that hide legal actions before the agent has trained or accumulated evidence for ignoring them

Any ARC success from these mechanisms is invalid for the core learned-agent goal.

## Minimum Implementation Package

Required modules or module-level equivalents:

- `BeliefState`: objects, relations, hypotheses, uncertainties, self-state
- `SelfModel`: controllability, action competence, causal attribution
- `Workspace`: limited active contents shared by planner, memory, language, and world model
- `EpisodicMemory`: fast writes, similarity retrieval, surprise/value indexing
- `WorldModel`: action-conditioned transition, reward/progress, and uncertainty prediction
- `QuestionGenerator`: uncertainty to grounded internal queries
- `CounterfactualPlanner`: candidate action comparison by value and information gain
- `ActionSurfaceAdapter`: exposes all currently legal actions and dense parameters to training/evaluation before learned pruning or abstraction is trusted
- `ProgressSignals`: novelty, surprise, stagnation, confidence, curiosity
- `Consolidator`: episodes to reusable rules and training samples
- `AblationRunner`: disables language, memory, workspace, imagination, online learning, and graph/search components

## Minimum Acceptance Suite

Synthetic tasks:

- hidden affordance: unknown object effect must be tested
- ambiguous goal: diagnostic action resolves two plausible reward rules
- partial observability: vanished relevant object must be remembered
- irreversible action: agent must imagine before committing
- distractor objects: workspace must select relevant entities
- mechanic transfer: after one exposure, similar task is solved faster

Required metrics:

- steps-to-solve
- diagnostic-action rate
- prediction error over time
- hypothesis entropy over time
- memory retrieval precision
- online adaptation gain
- ablation deltas
- workspace relevance
- question-action alignment
- language-usefulness delta

Pass threshold:

- full agent beats random baseline
- full agent beats graph-only baseline
- full agent beats frozen learned policy
- no-memory ablation is worse on hidden-rule tasks
- no-language ablation is worse on at least one task requiring hypothesis indexing
- no-imagination ablation is worse on at least one task requiring action consequence prediction
- full-action-surface evaluation is the default; any restricted-action smoke run is labeled as such and cannot establish core ARC success

## Operational Definition

For this repo, call an agent minimally conscious-enough only if:

- it maintains inspectable beliefs about the world and itself
- it knows when it does not know something
- it formulates an internal grounded question
- it chooses actions because they answer questions or exploit beliefs
- it updates beliefs from results
- it remembers and consolidates useful surprises
- it uses imagined consequences before acting
- it improves online under small experience
- it sees the legal action surface before learned evidence, not hand-authored caps, decides what to ignore
- its behavior degrades when these mechanisms are ablated

Anything less may still be useful ARC machinery, but it is not enough for the minimally conscious online-agent threshold.
