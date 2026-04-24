# GPT-Pro Advice, 2026-04-24

Source:

- GPT-Pro thread: `https://chatgpt.com/c/69eb6acc-d724-8329-bb12-19525fb56f06`
- Prompt context: clean ARC-AGI-3 online-learning agent, full legal action surface, no hand-coded ARC search patterns, no graph-search solver, no runtime external APIs, online learning with persistent state across ARC level transitions.

## Blunt Diagnosis

The likely failure is not lack of cleverness. The agent has almost no usable credit assignment over a huge legal action surface.

Dense clicks are exposed, movement no longer receives positive objective credit by default, reset/undo/wait are penalized, but without external reward the policy has no strong reason to become competent. A no-same-action-streak trace only shows it is not repeating one action. It does not show learning. A 50-step smoke run is trace sanity, not a learning verdict.

## Exact Next Order

1. Add hard honesty guards before more training.
2. Fix instrumentation before changing learning logic.
3. Validate environment/action contract.
4. Make checkpoints preserve actual learned state, not only weights.
5. Fix level transition handling so learned state is carried, not reset.
6. Replace objective-free churn with generic progress terms, not ARC tricks.
7. Make the policy score the full legal action surface.
8. Train the learned recurrent world model online.
9. Implement the minimal conscious-agent contract as traceable state.
10. Make causal theory learned/evidence-updated, not rule-scripted.
11. Add operator tracing and optional demonstration learning with source labels.
12. Use staged training gates: instrumented probe, public online signal check, longer public online meta-training, AR25 adaptation, then clean eval.
13. Enforce acceptance gates before claiming progress.
14. Do not add forbidden solvers, sweeps, caps, or per-game tuning.

## 1. Honesty Guards

Add tests and runtime checks that prevent clean training/eval from using forbidden paths:

- no per-game `ar25` / `0c556536` hardcoding in solver/training/model code
- no legacy `scientist/*` path in clean learned mode unless explicitly baseline/debug
- no runtime controller in clean mode
- no graph/frontier/BFS/DFS/beam/MCTS solver
- no `max_candidates`, candidate limits, top-k action pruning, or action-surface caps in clean action selection

Source review plus tests should distinguish diagnostics from clean action selection. Do not fail just because a diagnostic module contains a forbidden word; fail when forbidden behavior affects clean training/eval.

## 2. Instrumentation

No more silent training. Every environment step must emit JSONL with source, mode, run id, game id, level id, legal action count, chosen action id/index, policy logit/logprob/prob/rank/entropy, observation hashes, reward, progress terms, belief hash, recurrent state norm, memory size, internal tokens, optimizer step, losses, and gradient norm.

Terminology must be unambiguous:

- `executed_step_count`
- `legal_action_count`
- `mean_legal_action_count`

Do not confuse legal actions available with executed actions.

Add heartbeat logging independent of normal logger output. Stop long runs only on concrete health failure:

- no heartbeat
- no trace writes
- no env steps
- no optimizer steps
- repeated exceptions

Do not kill dense training merely because stdout is quiet.

## 3. Environment And Action Contract

Before training, prove the adapter is not corrupting learning:

- full legal surface returned every step
- raw legal action count equals exposed legal action count
- canonical action ids are stable for the same state
- dense click ids are structured and stable, not list-index based
- a 50-200 step trace can replay exactly: chosen actions, observation hashes, rewards, and level transitions

If trace replay fails, fix the adapter before training.

## 4. Checkpoint Learned State

Checkpoints must include online state, not only neural weights:

- model weights
- world model weights
- action encoder weights
- belief/recurrent state
- self model
- episodic memory
- progress normalizers
- optimizer state
- RNG state
- global and optimizer counters

Add resume equivalence tests:

- run N steps continuously
- run K steps, save, reload, run N-K steps
- compare action ids, observation hashes, optimizer step, and memory size

Small numerical tolerance is acceptable. Behavioral divergence is not.

## 5. Level Transition Handling

Level transitions must not reconstruct the agent or clear long-term learned state.

Allowed:

- detach recurrent hidden state for truncated BPTT
- record a boundary event
- update belief with boundary metadata

Forbidden at level transitions:

- reinitialize the agent
- clear memory
- zero semantic recurrent state
- reset optimizer
- reset self model

Trace level transitions with memory size before/after, belief hash before/after, recurrent norm before/after, and optimizer step.

Acceptance: memory does not shrink, optimizer step does not reset, belief changes as a boundary update rather than full reset.

## 6. Generic Progress Signal

Keep external reward and intrinsic progress separate.

External success is only:

- game reward
- level transition
- game-provided success/win

Intrinsic progress may train the agent, but must never be reported as ARC success.

Generic progress terms can include:

- external reward
- level transition
- done success
- non-movement direct effect
- novel effect after non-reset/non-undo/non-wait action
- learned potential delta
- information gain / learning progress
- no-op penalty
- wait penalty
- undo penalty
- reset penalty
- tiny step penalty

Movement alone should not get direct positive reward. Movement can become instrumentally useful only through learned future value/potential.

## 7. Full Legal Action Surface Policy

The policy must score every legal action and include every legal action in the sampling distribution.

Allowed:

- chunked scoring for memory/runtime efficiency

Forbidden:

- `legal_actions[:N]`
- heuristic top-k before policy scoring
- object-near click candidate restriction
- fixed click sweeps
- graph/frontier expansion as solver

Acceptance:

- `policy_logits.shape[0] == len(legal_actions)`
- chosen action index is in range
- all legal actions are in the denominator

## 8. Online Recurrent World Model Training

World model should predict action-conditioned outcomes from actual transitions:

- next latent
- effect probability
- progress
- uncertainty
- value

Train with:

- next-latent loss
- effect loss
- progress loss
- value loss
- policy gradient / actor-critic loss from observed progress and bootstrapped next value
- entropy regularization

Replay is allowed as online training data replay, not search. Suggested sampler:

- recent transitions
- positive/effective transitions
- surprising transitions
- random transitions

## 9. Traceable Minimal Conscious Contract

Do not spend time on philosophical labels. Make it inspectable.

The agent should carry these across level transitions/checkpoints:

- recurrent belief state
- current game/level id
- level index
- expected progress EMA
- effect rate EMA
- no-op rate EMA
- uncertainty EMA
- stuck probability
- action-family success/effect EMAs
- recent policy entropy
- recent value error
- grounded internal language token logits/tokens

Internal tokens should describe transition facts and learned predictions, for example:

- `OBS_CHANGED`
- `NO_EFFECT`
- `LEVEL_ADVANCED`
- `UNDO_USED`
- `RESET_USED`
- `EXPECT_EFFECT`
- `EXPECT_NO_EFFECT`
- `HIGH_UNCERTAINTY`
- `LOW_PROGRESS`
- `STUCK`
- `RECOVERING`

No pretrained LM. No runtime API. No text-world crutch.

## 10. Causal Theory

Use `reasoning/causal_theory.py` as an online causal-effect estimator, not a rule script.

It should estimate:

- `P(effect | belief, action)`
- `P(progress | belief, action)`
- expected information gain
- repeated no-op likelihood

Do not add rules such as click colored cell, fill grid, move to object, try all positions, or AR25-specific mechanics.

## 11. Operator Tracing And Demonstrations

Operator and agent actions should use the same action schema.

Trace operator actions with:

- source = `operator`
- chosen action id
- legal action count
- obs hash before/after
- external reward
- progress terms

If human demonstrations are allowed for training, label them by source and train auxiliary behavior cloning. Final autonomous acceptance must use source = `agent` only.

## 12. Training Regimen

### Phase A: 50-200 Step Instrumented AR25 Probe

Purpose:

- verify traces
- verify legal action surface
- verify optimizer activity
- verify dense scoring does not crash

Required:

- one JSONL record per step
- legal action count every step
- policy logits count equals legal action count
- optimizer step increases
- gradient norm nonzero on most update steps
- reward and progress logged separately

If this fails, do not train longer.

### Phase B: 2k-Step Public ARC Online Signal Check

Purpose:

- prove world model and policy receive nonzero learning signal without AR25 tuning

Required:

- world loss improves over initial baseline
- effect prediction beats constant/random baseline
- policy entropy changes due to learning
- undo/reset/wait rate decreases versus first 200-step baseline
- progress is not identically zero
- external reward reported honestly even if zero

### Phase C: 20k-50k Public Online Meta-Training

Purpose:

- learn generic action affordances
- learn recurrent state use
- learn objective grounding

Do not select lucky checkpoints. Use latest scheduled checkpoint or a preconfigured EMA model. If comparing checkpoints, report all.

### Phase D: AR25 Online Adaptation From Latest Learned State

Purpose:

- continue online learning on AR25 without AR25 hardcoding

Requirements:

- full action surface
- online updates
- preserve state across levels
- heartbeat and trace
- do not reset memory or belief at transitions

### Phase E: Clean Learned Evaluation

Must show:

- no scientist path
- no runtime controller
- no graph/frontier/BFS/beam solver
- no candidate cap
- full legal action scoring
- learned state loaded
- traceable actions

## 13. Acceptance Gates

1. Full action surface:
   - legal action count equals raw adapter legal count
   - policy logit count equals legal action count
   - no clean candidate cap

2. Trace replay:
   - 50-200 step trace replays action ids, obs hashes, reward, and level transitions exactly

3. Optimizer/gradient flow:
   - optimizer step increments
   - gradient norm nonzero
   - policy/world/value parameters change

4. Learned state persistence:
   - memory does not reset
   - optimizer step does not reset
   - belief/self model not reinitialized
   - checkpoint restores memory and normalizers

5. Progress sanity:
   - external reward may be zero
   - intrinsic progress not identically zero when observations change
   - movement alone not direct positive
   - undo/reset/wait penalties visible

6. World model learning:
   - effect prediction beats constant baseline
   - uncertainty decreases on familiar transitions
   - surprise remains high on novel transitions

7. Behavior improvement:
   - later windows improve versus first 200 steps
   - undo/reset/wait/no-op rates decrease or learned potential increases
   - chosen action value exceeds mean legal action value more often than chance
   - entropy changes without premature collapse

8. Public generality before AR25 claims:
   - show public-task improvement with same code/hyperparameters

9. AR25 external success:
   - only external reward, level transition, or game success counts

## 14. Likely Current Bugs

- Dense click surface is exposed, but the policy has no learned affordance model strong enough to rank it.
- Negative penalties remove bad actions but do not create a path to good actions.
- Movement needs bootstrapped future value. Without it, the agent cannot learn instrumental positioning.
- Disabling the runtime controller probably removed the only competent action selector. That is honest, but it exposes the learned policy as weak.
- Contrastive/full-action training may be improving representation without improving policy/value.
- Current clean checkpoints may not include learned memory, normalizers, recurrent belief, or action statistics.
- Killing the 2.5-minute dense run was wrong. The fix is heartbeat and JSONL progress, not shorter training.

## Do Not Do

- Do not add AR25-specific rules.
- Do not add fixed click sweeps.
- Do not add BFS/frontier/beam/MCTS/graph search as solver.
- Do not top-k or prune legal actions before policy scoring.
- Do not hide intrinsic progress inside reward and call it success.
- Do not select lucky checkpoints.
- Do not use legacy `scientist/*` path in clean mode.
- Do not claim minimal consciousness unless belief, self-model, internal language tokens, episodic memory, recurrent state, and world-model predictions are present in checkpoints and traces.

## Immediate Repo Work

The next best move is not another blind AR25 run. It is:

1. Add enforceable guard tests.
2. Add replayable step traces and heartbeats.
3. Add full-surface policy scoring assertions.
4. Add full online-state checkpointing and resume equivalence tests.
5. Add level-transition persistence tests.
6. Add generic progress terms separated from external reward.
7. Run public online signal checks.
8. Train public online meta-learning.
9. Adapt AR25 from the latest learned public state.
10. Evaluate clean learned path with full trace and no forbidden control.
