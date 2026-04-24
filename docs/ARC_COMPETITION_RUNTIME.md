# ARC Competition Runtime Semantics

This note records the verified runtime contract that the ARC-facing path must obey.

## Verified External Facts

- Competition mode allows only one `make(...)` call per environment instance.
- The environment contains multiple levels inside that one game session.
- Level resets are permitted inside the session through the explicit `RESET` action.
- After failure states such as `GAME_OVER`, the correct continuation is to issue `RESET` and keep playing the same environment.
- Scoring is tied to levels completed within the environment session, not to a single first positive reward event.

Primary sources:

- https://docs.arcprize.org/toolkit/competition_mode
- https://docs.arcprize.org/actions
- https://docs.arcprize.org/methodology
- https://docs.arcprize.org/api-reference/commands/execute-simple-action-1

## Engineering Consequences

The ARC-facing runtime must not model prize evaluation as one level per episode.

Required behavior:

1. Treat one environment instance as a persistent multi-level session.
2. Keep the agent alive after level failure so it can choose `RESET` and retry inside the same environment.
3. Preserve learned semantics across retries and across completed-level transitions.
4. Clear only volatile local workspace at level boundaries:
   - active intentions
   - pending binder probes
   - short-horizon navigation state
   - per-level action visit / no-effect churn penalties
5. Preserve cross-level learning state:
   - world-model weights
   - induced hypotheses
   - episodic memory contents
   - spotlight family-level binder priors

## Training Consequences

Stage 2 ARC training must use the same unit of interaction as prize evaluation.

That means:

1. The training unit is one persistent environment session, not one `env.reset(...)` episode.
2. The agent must stay alive through repeated fail -> `RESET` -> retry loops inside the same session.
3. The agent must carry forward what it learned on level `N` into later retries of level `N` and into levels `N+1`, `N+2`, and so on.
4. Level boundaries must be treated as soft resets:
   - clear volatile workspace
   - preserve learned semantics, memory, hypotheses, and family-level priors
5. Collector code must not treat `terminated` / `truncated` as unconditional end-of-training-sample if reset continuation is possible.
6. Positive reward is not a solved-training target by itself.
   - the primary targets are `levels_completed` and full-session win
   - reward is only an auxiliary signal
7. Stage 2 datasets should annotate session semantics explicitly:
   - `levels_completed_before`
   - `levels_completed_after`
   - `level_delta`
   - `reset_action`
   - `level_boundary`
   - `failure_terminal`
   - `session_terminal`
   - `won_session`
8. Stage 2 trainer metrics should be session-shaped:
   - session win rate
   - mean levels completed per session
   - resets used per session
   - steps to first level completion
   - steps between level completions
9. Current public ARC trainers that run `env.reset(...)` for every short episode are structurally misaligned with the benchmark and should be treated as provisional data-collection utilities, not correct prize-shaped training.

## Required Trainer Rewrite

The ARC public trainers now need a session-based collector/relabel path.

Minimum rewrite:

1. Replace `episodes_per_game` collection with `sessions_per_game`.
2. Inside each session, keep stepping the same environment until:
   - session win
   - non-resettable terminal
   - explicit session step budget
3. Keep the same live agent instance across all in-session resets and level transitions.
4. Call `reset_level()` at retry / level-complete boundaries, not `reset_episode()`.
5. Relabel samples with session-aware credit:
   - reward
   - state-change / effect
   - level-completion credit
   - retry-value credit
   - cross-level carryover value
6. Select best checkpoints by session outcomes, not by first positive reward or single-episode return.

## Repo-Level Decision

The public ARC harness and scientist runtime must therefore:

- expose `RESET` as a legal ARC affordance
- not terminate the session on the first positive reward
- not terminate permanently on `GAME_OVER` when reset is possible
- continue until true environment win, explicit session end without reset continuation, or step budget exhaustion

This is now a hard design constraint for all ARC-facing work in this repo.

## Forbidden Runtime Shortcuts

The ARC-facing learned agent must not use hand-coded control shortcuts to satisfy the session semantics above.

Forbidden:

- fixed or counted action-pattern search
- movement sweeps used as a solver
- reset/replay scripts used as a solver
- graph-search, shortest-path frontier expansion, BFS/DFS replay, or coverage search used as the solver
- per-game or environment-source-derived behavior

Allowed:

- `RESET` as an environment action under prize-session semantics
- state graphs and transition logs as memory or diagnostics
- explicit graph-search baselines and ablations that are not reported as learned-agent success
- learned-model lookahead where transition predictions, uncertainty, and action values come from learned or online-updated models

An ARC run that succeeds because of a forbidden shortcut is invalid for the core learned-agent objective and must be labeled as such in reports.

## Operator Command

To let the scientist agent keep retrying one specific offline ARC environment without a step cap:

```bash
.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent scientist --checkpoint-path artifacts/spotlight_exec_curriculum_best_v3.pkl --game-id ar25-0c556536 --mode offline --max-steps 0 --progress-every 500
```

Notes:

- `--max-steps 0` means no step limit.
- `--game-id ...` targets one environment instead of the default first `N` games.
- `--progress-every 500` emits periodic JSON progress lines during the long run.
- Stop the run manually with `Ctrl+C`.
