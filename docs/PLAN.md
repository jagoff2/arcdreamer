# Project Plan

## Objective

Build a local ARC-AGI-3 agent that uses explicit state tracking, compact learned dynamics, episodic memory, and grounded internal language instead of generic text-only scaling.

The plan now explicitly treats online rule learning as the center of the runtime architecture.
The target is not a frozen checkpoint with small runtime tweaks.
The target is an agent that infers, tests, revises, and exploits rules while the episode is live.

## Near-Term Milestones

1. Foundation
   - Typed data model for observations, structured state, transitions, plans, and language traces.
   - Synthetic hidden-rule environments with small action spaces and family-level adaptation opportunities.
   - CLI, configs, tests, and smoke-evaluation path.
2. Explicit Baselines
   - Random / heuristic baseline.
   - Graph exploration baseline with novelty and shortest-path support.
3. Learned Core
   - Object/state encoder.
   - Recurrent world model with reward, usefulness, and uncertainty heads.
   - Grounded language decoder over a controlled vocabulary.
   - Episodic memory keyed by latent state and question context.
4. Hybrid Planning
   - Action scoring by value, novelty, disagreement, and hypothesis disambiguation.
   - Small lookahead over manageable action branches.
   - Counterfactual replay over failed or stalled transitions.
   - Live action-preference updates that can override stale priors.
   - Explicit rule-hypothesis competition rather than single-story commitment.
5. Evaluation
   - Language-off and memory-off ablations.
   - Search-only vs learned comparison.
   - Synthetic family adaptation metrics.
   - ARC toolkit evaluation path when local SDK packages are installed.

## Online Rule-Learning Plan

1. Explicit Rule Objects
   - Represent grounded rule candidates with support, contradiction count, scope, and confidence.
   - Separate general rules from exception memory.
2. Hypothesis Manager
   - Maintain multiple live candidates for action semantics, reward conditions, and hidden-state beliefs.
   - Rank candidates by explanatory power and empirical consistency.
3. Diagnostic Intervention
   - Add an information-gain or disagreement term that directly rewards actions that separate plausible rule candidates.
4. Counterfactual Replay
   - Cache recent bad or surprising pre-action states.
   - Re-evaluate them against alternative actions.
   - Write `avoid` and `prefer` corrections back into runtime memory.
5. Runtime Patch Layer
   - Maintain per-episode action bias and local transition corrections.
   - Allow live evidence to override checkpoint priors.
6. Representation Repair
   - Allow the runtime learner to split or reinterpret entities when the current abstraction does not explain the observed transitions.

## Diffusion-Inspired Direction

Use diffusion as inspiration for iterative hypothesis denoising, not as a mandatory architecture family.

The intended loop is:

1. begin with a noisy cloud of weak hypotheses
2. propose actions from that cloud
3. observe the real transition
4. denoise the hypothesis set by contradiction and support
5. write a local "proof" or disproof object
6. repeat until the rule set becomes sharp enough for exploitation

## Scope Control

- Favor a runnable synthetic stack over premature ARC-specific specialization.
- Keep the language system compact and grounded. No pretrained LM dependencies.
- Use optional integration points for the ARC toolkit rather than hard-failing package imports.
- Keep architecture small enough for single- and dual-consumer-GPU setups.
- Do not accept runtime-learning mechanisms that would fail to transfer to non-text games outside ARC with similar observation/action structure.
