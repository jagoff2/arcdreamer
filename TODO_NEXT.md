# TODO Next

Last updated: 2026-06-12.

## Ranked Experiments

1. Build generic transition-goal chain search over the component transition graph.
   - Use predicted component transitions as graph edges and search for short public-state experiment chains that move from current component relations toward previously goal-linked component mechanisms.
   - Prefer plans with public-state preconditions and expected postconditions, not single-step scalar bonuses.
   - Replan when observed public transitions contradict expected postconditions.
   - Success criterion: attempts 2 or 3 improve official useful events or score over attempt 1 without increasing invalid actions.

2. Add richer generic public component relations.
   - Track containment, alignment, color match, object removal, object transfer, split/merge, and repeated transform relations from public frames only.
   - Link generic action families and click cells to these relations without game IDs or per-game branches.
   - Use contradictions to lower stale component hypotheses.

3. Ground sequence candidates in predicted public state transitions.
   - Current sequence candidates carry component expectations, but they still mostly replay prior positive-event windows.
   - Add compact plan state with predicted next public component relation after each action and abandon/replan when observed public changes contradict it.
   - Avoid fixed schedules, forced entropy, game-specific branches, and hardcoded action strings beyond generic action-family parsing.

4. Mine recurring public partial-score patterns as generic mechanisms.
   - Identify which public component relations preceded partial-score events such as the recurring `lf52` partial score.
   - Generalize only mechanisms that can be expressed without game ID, source inspection, hidden labels, solution labels, or per-game tuning.

5. Train or adapt the JEPA encoder on legal failed-attempt traces without solution labels.
   - Use only public observations, legal actions, chosen actions, score/event deltas, and terminal flags.
   - Do not use game IDs as features or tune per game.
   - Treat internal losses as diagnostics only; proof remains official public runtime.

## Attempted This Run

Predicted component-transition planning and scoring cache:
   - Record component relation-to-mechanism predictions for movement, appearance, disappearance, color/shape transform, visual transform, split/merge, stable, and blocked/no-effect outcomes.
   - Score legal actions by expected productive public component transitions, goal-linked transition evidence, family-backed transition evidence, and contradiction penalties.
   - Penalize stale predictions after live public sequence contradictions.
   - Cache connected components once per observation and reuse component-target relations across legal actions.
   - Result: implemented and audited, but official runtime remained `NO IMPROVEMENT FOUND`; attempts 2 and 3 did not improve score or useful events over attempt 1.

Component-level public causal graph and grounded sequence check:
   - Extract connected components from public pre-action and post-action frames.
   - Infer component movement, appearance, disappearance, color/shape transform, and blocked/no-effect transitions.
   - Track component-target relations, component-family values, component-goal links, delayed public-event component links, and contradiction counts.
   - Attach expected component changes to sequence candidates and abort active replay when live public frame diffs contradict the expectation.
   - Result: implemented and audited, but official runtime remained `NO IMPROVEMENT FOUND`; attempts 2 and 3 did not improve score or useful events over attempt 1.

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

Implement experiment 1: generic transition-goal chain search over the existing predicted component-transition graph. Focus on public-state preconditions/postconditions and short experiment chains toward goal-linked mechanisms, then rerun tests, official evaluation, audits, compaction, and docs.
