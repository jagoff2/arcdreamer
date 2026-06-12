# TODO Next

Last updated: 2026-06-12.

## Ranked Experiments

1. Replace coarse 8x8 region hypotheses with component-level public causal graphs.
   - Track connected components, component colors, bounding boxes, centroids, contact relations, containment, alignment, disappearance, appearance, split/merge, and color transforms from public frames only.
   - Link generic action families and click cells to component-level effects without game IDs or per-game branches.
   - Use contradictions to lower stale component hypotheses.
   - Success criterion: attempts 2 or 3 improve official useful events or score over attempt 1 without increasing invalid actions.

2. Ground sequence candidates in predicted public state transitions.
   - Current sequence candidates replay prior positive-event windows, but they are not tied to expected public component changes.
   - Add compact plan state with predicted next public component relation after each action and abandon/replan when observed public changes contradict it.
   - Avoid fixed schedules, forced entropy, game-specific branches, and hardcoded action strings beyond generic action-family parsing.

3. Mine recurring public partial-score patterns as generic mechanisms.
   - Identify which public component relations preceded partial-score events such as the recurring `lf52` partial score.
   - Generalize only mechanisms that can be expressed without game ID, source inspection, hidden labels, solution labels, or per-game tuning.

4. Train or adapt the JEPA encoder on legal failed-attempt traces without solution labels.
   - Use only public observations, legal actions, chosen actions, score/event deltas, and terminal flags.
   - Do not use game IDs as features or tune per game.
   - Treat internal losses as diagnostics only; proof remains official public runtime.

## Attempted This Run

Region/object causal hypotheses and sequence candidate planner:
   - Store post-action `next_frame` in attempt timelines.
   - Infer public-frame movement, spawn, removal, toggle/transform, visual transform, and blocked/no-effect hypotheses.
   - Track changed regions, changed colors, delayed public-event region links, region/action-family values, and failed regions.
   - Score future legal actions by public region contact and action-family evidence.
   - Replay short prior-event action windows as compact sequence candidates.
   - Result: implemented and audited, but official runtime remained `NO IMPROVEMENT FOUND`; attempts 2 and 3 did not improve score or useful events over attempt 1.

Transition-graph next-attempt planner:
   - Build from full attempt traces: public observation hash, action, next observation hash, score/event delta, terminal flag.
   - Track which actions caused visible state transitions, which were blocked/no-effect, and which transitions preceded delayed positive events.
   - Plan next attempts as targeted experiments over untested state-action edges, not fixed cycling or entropy.
   - Result: implemented and audited, but official runtime remained `NO IMPROVEMENT FOUND`; attempts 2 and 3 did not improve over attempt 1.

## Immediate Next Step

Implement experiment 1: component-level public causal graphs from frame diffs. Focus on connected components and public relations instead of coarse 8x8 regions, then feed only supported component hypotheses into sequence-level plans with contradiction handling.
