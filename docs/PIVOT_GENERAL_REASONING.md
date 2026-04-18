# General Reasoning Pivot

Date: 2026-04-17

## Thesis

The target is not "solve the current game/harness."

The target is a transferable agent that can:

- infer latent task state from observation and action history
- form revisable beliefs
- choose diagnostic actions under uncertainty
- plan over abstract progress state
- execute robustly when interface, layout, mechanics, or task family shift

If a capability depends on privileged harness fields, hidden env state, family-specific shortcuts, or exact oracle move imitation, it does not count as the target capability.

## Operator Standard

This is the operative standard for the repo from this point forward:

- The agent must infer latent task state from observation-action history, not consume privileged harness fields and call that reasoning.
- The planner must operate over generic beliefs, uncertainty, subgoals, and action-outcome expectations, not over task-family-specific shortcuts.
- The policy target must reward equivalence-class competence and adaptive recovery, not exact oracle move imitation.
- Memory must store causal facts and phase state in a task-agnostic form, not brittle game labels.
- Evaluation must include interface, layout, mechanic, and task shifts. Otherwise we are just training a benchmark-specialized policy.

The recent state-retention fix was necessary but not sufficient. It repaired a real defect in state representation, but it does not satisfy this stronger requirement by itself.

In repo terms, the remaining wrong parts are still:

- synthetic extras expose too much internal state
- oracle-style action targets still teach "how to solve this family"
- success is still measured too much as task completion rather than transferable belief formation, planning, and execution quality

The right bar is:

1. Infer hidden phase/state, do not hand it in.
2. Represent it in a generic belief state that survives task/harness shift.
3. Plan diagnostic actions to reduce uncertainty.
4. Execute via generic action schemas, not family-specific reflexes.
5. Adapt online when the prior is wrong.
6. Measure transfer, not just score.

## Spatial Workspace Requirement

Generalized planning under layout, size, and interface shift needs an agent-owned allocentric spatial workspace.

This is not a raw second copy of the visible board and not neuroscience cosplay. It is a compact belief map that gives the control loop a stable notion of place across time so it can:

- preserve location identity across steps
- path-integrate and return to previously tested regions
- remember where diagnostic interactions happened
- separate spatial structure from mechanic hypotheses
- plan over topology instead of memorized move traces
- transfer across interface/layout changes because the representation is about space and control, not task-family labels

The spatial workspace should be hybridized with the existing object graph and symbolic belief state:

1. Object graph for entities, relations, affordances, and causal object roles.
2. Allocentric spatial workspace for explored regions, anchors, control-site hypotheses, effect hotspots, and uncertainty heat.
3. Symbolic belief state for progress phase, contradictions, pending subgoals, and control-mode hypotheses.

Pure object slots lose topology. Pure visible-grid replay loses abstraction. The target design needs both.

### Minimum workspace contents

The internal spatial workspace should carry compact layers such as:

- visible occupancy / color evidence
- explored vs unexplored confidence
- persistent object or landmark anchors
- interactable / control-location hypotheses
- target / progress likelihood
- contradiction / uncertainty heat
- recent causal-effect traces

This must be agent-owned, persistent across steps, action-updated, and planner-queryable. It should be built from observation-action history, not handed in by the environment.

## Why The Current Path Is Insufficient

The current synthetic pipeline still contains benchmark-shaped shortcuts that can inflate apparent competence without proving generalized reasoning.

### 1. Hidden synthetic state leaks into the learned path

Synthetic observations currently expose or recently exposed internal state such as:

- `inventory`
- `flags`
- `goal_active`
- `progress_index`
- `selected_color`
- `rule_tokens`
- `question_tokens`

That teaches the system to consume latent mechanic state from the harness instead of inferring it from transitions.

### 2. Training still over-centers exact oracle action structure

Stage 1 data collection still uses `oracle_action(env)` and `behavior_policy="mixed"` defaults, which keeps the dataset and induced prior too close to family-specific solution trajectories.

That is useful for bootstrapping throughput, but it does not produce the desired capability:

- generic hypothesis formation
- layout-robust planning
- equivalence-class action competence
- adaptive recovery when the prior is wrong

### 3. Policy supervision still rewards "teacher move fidelity" too much

The policy head is still trained around executed action labels and transition-local rewards. This is better than raw oracle action cloning, but it still needs to move further toward:

- abstract progress supervision
- diagnostic-value supervision
- equivalence over multiple valid moves
- explicit anti-overfitting pressure against exact path priors

### 4. Hidden-state reasoning is not yet agent-owned enough

If the env stops handing out latent mechanic fields, the agent still risks amnesia unless it owns a persistent inferred latent-state channel.

That channel must be built from:

- action history
- object changes
- reward deltas
- control-state experiments
- world-model disagreement

not from environment internals.

## Strategic Pivot

The project now pivots from "improve synthetic benchmark score" to:

"Make the agent's control loop depend on generalized reasoning, planning, and execution that survives removal of benchmark-specific crutches."

This does not mean abandoning the synthetic curriculum.

It means the synthetic curriculum must now train the right capability under the right information boundary.

## Hard Rules From This Point Forward

### Information boundary

- No hidden synthetic rule text in observation extras.
- No hidden synthetic inventory/flag state in observation extras if that state is not directly observable.
- No training path may rely on env-private latent variables at inference time.

### Reasoning boundary

- Latent progress must be inferred by the agent from transitions.
- The planner must operate on agent-owned belief state, not harness-owned mechanic state.
- The world model must learn partially observed control-state dynamics rather than being handed them.

### Supervision boundary

- Reduce dependence on exact oracle action imitation.
- Prefer progress, diagnostic value, and outcome-based supervision over exact-path supervision.
- Treat multiple valid move actions as equivalent where possible.

### Evaluation boundary

- Track transfer under size shift, layout shift, interface shift, and mechanic composition.
- Do not count train-family competence as evidence of generalized reasoning.
- The decisive eval is whether the same control loop works when task-specific assumptions are stripped away.

## Concrete Code Pivot

### A. Remove privileged synthetic observation leakage

Patch `arcagi/envs/synthetic.py` and the perception path so that synthetic observations no longer expose hidden rule/mechanic state through extras.

Remove from the agent-visible observation surface:

- `rule_tokens`
- `question_tokens`
- hidden `inventory`
- hidden `flags`
- hidden `progress_index`
- hidden `selected_color`

Keep only observation-grounded fields such as:

- grid
- legal actions
- visible action-role metadata if it is interface-level rather than task-solution-level
- visible cell tags only if they correspond to directly observable affordances

### B. Add agent-owned inferred latent state

Add a generic inferred-state tracker that is maintained by the agent across the episode and merged into planning state.

The inferred state should encode generic latent control/progress concepts such as:

- recent productive action family
- last strong causal interaction signature
- abstract phase/progress count
- candidate control mode
- contradiction count
- whether the current objective appears completed / advanced / reset

This must be derived from transitions, not read from env internals.

### B2. Add an allocentric spatial belief workspace

Add a compact spatial memory module that remaps observations into a persistent allocentric workspace and updates it after each action/transition.

It should support at minimum:

- agent position anchor
- explored vs unexplored memory
- tested interaction sites
- recent positive / negative effect locations
- return-to-place targets for follow-up testing
- uncertainty / contradiction hotspots

The planner and world model should be able to query this workspace directly when choosing exploration, diagnostic, and completion actions.

### C. Rework Stage 1 language targets

Grounded language should describe:

- uncertainty
- causal hypotheses
- control-mode uncertainty
- phase progress
- next diagnostic need

but must be generated from observable transition evidence, not hidden rule ids or hidden env state.

Current status after the latest language pass:

- the old ~29-token toy vocabulary has been replaced with a richer grounded predicate vocabulary covering structural fields like `belief/question/plan`, `action/focus/state/direction/color`, mechanic words, color buckets, and coarse progress/count buckets
- Stage 1 synthetic training and Stage 2 ARC-public training now both supervise a learned `plan` channel alongside `belief` and `question`
- planner runtime thought now decodes and carries `plan_tokens`
- episodic memory now stores and retrieves `plan_tokens`
- planner action scoring now includes generic plan-action alignment against learned plan tokens instead of treating plan language as inert logging

This still does not make the language module a large open-ended LM. It does make it a materially more serious grounded internal scratchpad rather than a tiny auxiliary caption head.

### D. Rework Stage 1 policy supervision

Move further away from exact teacher action supervision.

Stage 1 policy loss should prefer:

- actions that produced generic progress
- actions that reduced uncertainty
- actions that produced meaningful state change
- families of equivalent moves where appropriate

and should penalize:

- no-effect churn
- repeated misleading interactions
- brittle exact-path dependence

### E. Reduce oracle dependence in collection

Keep oracle only as a bounded bootstrap tool if still needed, not as the dominant trajectory source.

Default collection should shift toward:

- graph exploration
- learned policy roll-in
- planner-guided diagnostic exploration
- mixed replay emphasizing hard self-generated transitions

### F. Add a constrained dream/consolidation phase

The current repo already has bounded act-time imagination:

- planner lookahead through the recurrent world model
- tiny online counterfactual replay after bad actions

That is not enough. The system still cannot "sleep on" a mechanic and return with a better internal control program.

The missing capability is constrained latent rehearsal, not free-form fantasy.

The required Stage 1 mechanism is:

1. Anchor dreams in real memory.
2. Roll forward short latent futures without re-observing the environment at every step.
3. Reality-check those dream rollouts against later observed outcomes.
4. Distill grounded beliefs/questions/plans from the imagined futures.
5. Use this to stabilize long-horizon phase transitions and post-imitation control.

Hard rule:

- dream rollouts must be rooted in real collected trajectories or explicit memory anchors
- dream rollouts must stay short-horizon and reality-checked
- dream rollouts must not become a source of unconstrained self-generated supervision

The intended order is:

1. teach the agent a grounded way to think
2. then let it rehearse internally under constrained dreaming

not the reverse

## Bootstrap And Handoff Rule

The apprenticeship schedule must not hand control over to the learned policy on a fixed epoch clock if the policy is not yet stable.

The fresh non-fallback retrain on `2026-04-18` showed the failure mode clearly:

- epoch `0` train collection was strong while imitation/bootstrap support was dense
- epoch `1` remained strong under the same scaffold
- epoch `2` collapsed (`0.4531` train success, `0.0` held-out frontier success) as teacher density dropped

That means the early strength was scaffold-carried, not policy-owned.

From this point forward, Stage 1 teacher support should follow these rules:

1. Start with a real apprenticeship phase, not a token warm-start.
2. Keep a permanent nonzero teacher floor for the full run rather than decaying support to zero.
3. Prefer learner roll-ins with teacher relabeling on visited states over teacher-owned trajectories as the dominant data source.
4. Keep some pure teacher episodes for the full run to stabilize belief/question formation, phase transitions, and clean completion behavior.
5. Maintain denser teacher control on the hard early prefix of each episode where latent-state formation happens.
6. Make guidance dynamic rather than latched: when train or holdout competence regresses, teacher density should rise again.
7. Surface the live teacher-guidance state in metrics so a run can be diagnosed as:
   - scaffold-carried
   - partially self-owned
   - genuinely self-sustaining
8. Prefer DAgger-style relabeling and state-distribution correction over either:
   - long pure oracle cloning
   - abrupt early off-policy self-training

### Practical implication

The next Stage 1 trainer behavior should be:

- longer imitation/bootstrap defaults
- a permanent minority of pure teacher episodes
- learner roll-ins with teacher labels on every visited state
- denser early-prefix teacher takeover that decays only to a floor
- dynamic guidance that strengthens again when competence regresses
- visible guidance state in epoch logs and checkpoints

This preserves the generalized-reasoning thesis while avoiding the current failure mode where the policy is forced off the scaffold before it owns usable beliefs, goals, and phase-conditioned action priors.

### F. Add transfer-oriented tests

Add tests that fail if:

- hidden env state re-enters observations
- policy supervision collapses to exact single-action labels where equivalence should hold
- inferred latent state is missing after hidden state is removed
- planner cannot carry progress through partially observed control-mode tasks

## Immediate Execution Plan

### Phase 1: Information-boundary cleanup

1. Remove hidden synthetic state from observation extras.
2. Remove hidden rule/question token leakage from env observations.
3. Update tests to enforce the no-leak boundary.

### Phase 2: Agent-owned latent-state channel

1. Implement a generic inferred-state tracker in the learned agent path.
2. Merge inferred state into planning/world-model context without using env-private fields.
3. Add regression tests for inferred latent progress on hidden-mode synthetic tasks.

### Phase 2.5: Spatial belief substrate

1. Implement a compact allocentric spatial workspace owned by the agent.
2. Update it from observation/action/transition history rather than env-private fields.
3. Expose planner-queryable summaries such as explored coverage, nearby hotspots, and tested-control anchors.
4. Add regression tests showing persistence across steps and return-to-place capability after intermediate exploration.

### Phase 3: Training objective pivot

1. Remove `target_action` dependence from Stage 1 policy training.
2. Reduce `mixed` oracle dependence in default collection.
3. Rebuild language targets from transition evidence only.
4. Keep reward/usefulness/policy supervision centered on generic progress and diagnostic value.

### Phase 4: Retrain and re-evaluate

1. Retrain Stage 1 foundation under the new information boundary.
2. Recheck:
   - held-out `switch_unlock`
   - held-out `order_collect`
   - selector-based families
   - compositional families
3. Compare to the old path on:
   - hold-out success
   - interaction churn
   - transfer stability
   - dependence on task family priors

## Acceptance Criteria For The Pivot

The pivot is only considered real if all of the following hold:

1. The agent no longer receives hidden synthetic mechanic state in observations.
2. The learned path can still maintain latent progress/control beliefs across steps.
3. Policy and language training no longer depend on hidden rule labels.
4. Performance on held-out families survives or recovers under the stricter boundary.
5. The resulting control loop is more transferable, not just harder to train.
6. The agent owns a persistent spatial workspace that survives intermediate motion and helps it revisit diagnostic/control locations under layout shift.

## Non-Negotiable Reminder

We are not training it to beat the current task harness.

We are training it to perform generalized reasoning, planning, and execution at a level that does not depend on the harness, interface, gameplay, or task family.
