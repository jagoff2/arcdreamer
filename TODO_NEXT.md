# TODO Next

Last updated: 2026-06-12.

## Ranked Experiments

1. Add region/object causal hypotheses to attempt memory.
   - Track changed regions, contacts, pushes, carries, toggles, spawns, removals, and blocked motion using public frames only.
   - Link click/move actions to components and delayed frame changes.
   - Use contradictions to lower confidence in stale hypotheses.
   - Success criterion: attempts 2 or 3 improve official useful events or score over attempt 1 without increasing invalid actions.

2. Replace scalar action nudges with sequence-level plans.
   - Current transition graph can bias a single current action, but it does not maintain a compact multi-step experiment across states.
   - Add recurrent plan state over a proposed public-evidence experiment while preserving live recurrent state and ARC legality.
   - Avoid fixed schedules, forced entropy, game-specific branches, and hardcoded action strings beyond generic action-family parsing.

3. Train or adapt the JEPA encoder on legal failed-attempt traces without solution labels.
   - Use only public observations, legal actions, chosen actions, score/event deltas, and terminal flags.
   - Do not use game IDs as features or tune per game.
   - Treat internal losses as diagnostics only; proof remains official public runtime.

4. Add official trace analysis for the single recurring partial-score game.
   - Identify what public event patterns preceded the `lf52` partial score without hardcoding that game or branching on its ID.
   - Generalize only if the same mechanism appears in other traces.

## Attempted This Run

Transition-graph next-attempt planner:
   - Build from full attempt traces: public observation hash, action, next observation hash, score/event delta, terminal flag.
   - Track which actions caused visible state transitions, which were blocked/no-effect, and which transitions preceded delayed positive events.
   - Plan next attempts as targeted experiments over untested state-action edges, not fixed cycling or entropy.
   - Result: implemented and audited, but official runtime remained `NO IMPROVEMENT FOUND`; attempts 2 and 3 did not improve over attempt 1.

## Immediate Next Step

Implement experiment 1: region/object causal hypotheses from public frame diffs. Focus on changed components, contacts, blocked moves, object appearances/removals, and delayed event links, then feed those hypotheses into a sequence-level next-attempt planner rather than another one-step scalar action nudge.
