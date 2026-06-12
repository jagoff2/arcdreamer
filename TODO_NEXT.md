# TODO Next

Last updated: 2026-06-12.

## Ranked Experiments

1. Convert component graph memory into predicted component-transition planning.
   - Score legal actions by expected next public component relation, not only accumulated component/contact value.
   - Represent plan state as predicted movement, appearance, disappearance, color/shape transform, or blocked/no-effect outcome per step.
   - Downweight stale hypotheses after live public contradictions and choose the next best supported experiment.
   - Success criterion: attempts 2 or 3 improve official useful events or score over attempt 1 without increasing invalid actions.

2. Cache component extraction and action-relation features inside legal-action scoring.
   - Current official run completed but component extraction inside per-action scoring was slow.
   - Cache frame components once per observation and reuse component-target relation features across legal actions.
   - Preserve behavior and rerun official evaluation after any performance patch.

3. Add richer generic public component relations.
   - Track containment, alignment, color match, object removal, object transfer, split/merge, and repeated transform relations from public frames only.
   - Link generic action families and click cells to these relations without game IDs or per-game branches.
   - Use contradictions to lower stale component hypotheses.

4. Ground sequence candidates in predicted public state transitions.
   - Current sequence candidates carry component expectations, but they still mostly replay prior positive-event windows.
   - Add compact plan state with predicted next public component relation after each action and abandon/replan when observed public changes contradict it.
   - Avoid fixed schedules, forced entropy, game-specific branches, and hardcoded action strings beyond generic action-family parsing.

5. Mine recurring public partial-score patterns as generic mechanisms.
   - Identify which public component relations preceded partial-score events such as the recurring `lf52` partial score.
   - Generalize only mechanisms that can be expressed without game ID, source inspection, hidden labels, solution labels, or per-game tuning.

6. Train or adapt the JEPA encoder on legal failed-attempt traces without solution labels.
   - Use only public observations, legal actions, chosen actions, score/event deltas, and terminal flags.
   - Do not use game IDs as features or tune per game.
   - Treat internal losses as diagnostics only; proof remains official public runtime.

## Attempted This Run

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

Implement experiment 1: predicted component-transition planning from the existing component graph. Focus on using expected next public component changes for action scoring and re-planning after contradictions, then rerun tests, official evaluation, audits, compaction, and docs.
