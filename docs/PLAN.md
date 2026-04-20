# Project Plan

## Objective

Build a local ARC-AGI-3 agent that uses explicit state tracking, compact learned dynamics, episodic memory, and grounded internal language instead of generic text-only scaling.

The plan now explicitly treats online rule learning as the center of the runtime architecture.
The target is not a frozen checkpoint with small runtime tweaks.
The target is an agent that infers, tests, revises, and exploits rules while the episode is live.
The full hybrid path now explicitly includes the generic runtime hypothesis controller rather than keeping that layer outside the main eval agent.

## Current Status

The long manual Stage 1 run has now cleared the old foundation bottleneck and promoted into `hidden_modes` by epoch `4` under `25,000` episodes per epoch.
That changes the immediate target:

1. keep the promoted checkpoint stable instead of regressing after promotion
2. push `hidden_modes` beyond the current `~0.45` running success regime
3. reduce teacher ownership so promoted behavior is actually learner-owned
4. verify that the same runtime machinery transfers beyond the promoted synthetic regime

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
   - Live action-preference updates and local model patches that can override stale priors.
   - Explicit rule-hypothesis competition with normalized posteriors over rival executable theories rather than single-story commitment.
   - Proof and exception objects carried into memory and later action choice.
   - Online representation repair that can split, merge, relabel, and rebind entities inside the working structured state.
5. Evaluation
   - Language-off and memory-off ablations.
   - Search-only vs learned comparison.
   - Synthetic family adaptation metrics.
   - ARC toolkit evaluation path when local SDK packages are installed.

## Experimental Scientist Loop

The runtime loop now needs to be judged against this concrete standard:

1. Maintain multiple grounded hypotheses about control, objective, mode, and representation, and weight them as normalized rival theories.
2. Pick actions partly for separation value between rival explanations, not just reward or novelty.
3. Write compact proof or contradiction objects from each informative intervention.
4. Patch local action semantics and policy priors immediately when evidence disagrees.
5. Execute representation repair when the current object decomposition or binding stops explaining transitions.
6. Compile short reusable action programs online and reuse them as temporary options when they improve control efficiency.

## Online Rule-Learning Plan

1. Explicit Rule Objects
   - Represent grounded rule candidates with support, contradiction count, scope, and confidence.
   - Separate general rules from exception memory.
2. Hypothesis Manager
   - Maintain multiple live candidates for action semantics, reward conditions, and hidden-state beliefs.
   - Rank candidates by explanatory power and empirical consistency using normalized posteriors rather than raw heuristic scores.
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
   - Allow the runtime learner to split, merge, relabel, or rebind entities when the current abstraction does not explain the observed transitions.

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
