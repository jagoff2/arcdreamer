# ARC Level Transition Verification, 2026-04-24

## Local inventory

- Direct `arc_agi` OFFLINE query reports exactly one cached local environment:
  - `ar25-0c556536`
  - title: `AR25`
  - local dir: `environment_files\ar25\0c556536`
  - tags: `['keyboard_click']`
- The same environment contains multiple internal levels:
  - `win_score = 8`
  - `_levels` length = `8`
  - `_clean_levels` length = `8`
- ONLINE mode can fetch 25 environment descriptors through the ARC API, but those are not locally cached offline games. The runtime target remains offline/local unless explicitly performing availability diagnostics.

## 1000-step learned recurrent run

Command:

```powershell
.venv313\Scripts\python.exe -m arcagi.evaluation.harness arc --agent learned_online_recurrent --checkpoint-path artifacts\learned_online_recurrent_synth_actionid_12000.pkl --game-id ar25-0c556536 --mode offline --max-steps 1000 --progress-every 100 --trace-path artifacts\traces\learned_online_recurrent_synth12000_1000.jsonl
```

Final result:

- `success=true`
- `won=false`
- `return=1.0`
- `levels_completed=1`
- `steps=1000`
- `interaction_steps=64`
- `reset_steps=2`
- `dense_action_surface=true`
- `scored_action_count=447`
- `legal_action_count=447`
- `controller_kind=learned_online_recurrent_v1`
- `learned_online_controller=true`
- `arc_competence_validated=false`
- `family_histogram={"move":747,"undo":187,"select":1,"click":63,"reset":2}`
- `max_same_action_streak=183`

Interpretation:

- This preserves the clean learned level-1 success prototype.
- It is not a full game win.
- After level 1, the recurrent controller fails to solve level 2 and degenerates into mostly movement and undo repetition.

## Boundary semantics

The recorded trace has the first level boundary at step `299`:

- step `299`
- action `1`
- `before_levels_completed=0`
- `after_levels_completed=1`
- `reward=1.0`
- `before_game_state=GameState.NOT_FINISHED`
- `after_game_state=GameState.NOT_FINISHED`
- `terminated=false`
- `truncated=false`

Replay of the recorded action sequence through `ArcToolkitEnv` verifies the environment transition:

- reset:
  - `level_index=0`
  - grid hash `d1cf6c65a8c9510a`
  - `levels_completed=0`
- after step `298`:
  - `level_index=0`
  - grid hash `84eee96a9eb6e8d8`
  - `levels_completed=0`
- after step `299`:
  - `level_index=1`
  - grid hash `23389de4e0a7e7b3`
  - `levels_completed=1`
  - `reward=1.0`
- after step `300`:
  - `level_index=1`
  - grid hash `e793346c71d0d291`
  - `levels_completed=1`

Conclusion:

- The environment does not replay the same level after completion.
- The local AR25 game advances from internal level `0` to internal level `1` inside the same session.
- The correct next engineering target is cross-level carryover and post-boundary action arbitration, not re-solving level 1.

## Tooling note

`python -m arcagi.cli list-arc-games` previously printed nothing because `arcagi.cli` did not call `main()` when executed as a module. This was fixed so future offline inventory checks use the repo CLI rather than ad hoc snippets.
