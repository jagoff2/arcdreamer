# TODO Next

Last updated: 2026-06-12.

## Ranked Experiments

1. Implement a transition-graph next-attempt planner.
   - Build from full attempt traces: public observation hash, action, next observation hash, score/event delta, terminal flag.
   - Track which actions caused visible state transitions, which were blocked/no-effect, and which transitions preceded delayed positive events.
   - Plan next attempts as targeted experiments over untested state-action edges, not fixed cycling or entropy.
   - Success criterion: attempts 2 or 3 improve official useful events or score over attempt 1 without increasing invalid actions.

2. Add region/object causal hypotheses to attempt memory.
   - Track changed regions, contacts, pushes, carries, toggles, spawns, removals, and blocked motion using public frames only.
   - Link click/move actions to components and delayed frame changes.
   - Use contradictions to lower confidence in stale hypotheses.

3. Replace scalar action nudges with sequence-level plans.
   - Current `score_action()` can only bias a single next action.
   - Add compact recurrent plan state over a proposed multi-step experiment while preserving live recurrent state and ARC legality.

4. Train or adapt the JEPA encoder on legal failed-attempt traces without solution labels.
   - Use only public observations, legal actions, chosen actions, score/event deltas, and terminal flags.
   - Do not use game IDs as features or tune per game.
   - Treat internal losses as diagnostics only; proof remains official public runtime.

5. Add official trace analysis for the single recurring partial-score game.
   - Identify what public event patterns preceded the `lf52` partial score without hardcoding that game or branching on its ID.
   - Generalize only if the same mechanism appears in other traces.

## Immediate Next Step

Implement experiment 1: a transition-graph attempt planner that uses no-effect/effect evidence and delayed event proximity to choose next-attempt action sequences. Keep it generic, public-observation-only, and audited against fixed schedules and game-specific branches.
