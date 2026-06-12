# TODO Next

Last updated: 2026-06-12.

## Ranked Experiments

1. Ground sequence candidates in predicted public relation transitions.
   - Current sequence candidates carry component expectations, but they still mostly replay prior positive-event windows.
   - Add compact plan state with predicted next public component relation delta after each action, not only action replay.
   - Prefer plans whose first predicted relation delta is reachable from the current public frame and abandon/replan when observed public changes contradict it.
   - Avoid fixed schedules, forced entropy, game-specific branches, and hardcoded action strings beyond generic action-family parsing.

2. Add relation-chain and relation-delta activation diagnostics.
   - Count how often relation-chain and relation-delta plans activate, resolve to legal actions, produce public changes, trigger useful events, and abort by contradiction.
   - Use those aggregate diagnostics to distinguish weak evidence from stale relation templates without inspecting game source or branching on game identity.

3. Mine recurring public partial-score patterns with plan-level context.
   - The single-action generic relation-delta miner was legal but insufficient.
   - Next mining should represent two-step or three-step public relation-delta chains such as contact -> transform -> event, movement -> alignment -> event, or repeated contact count -> event.
   - Success criterion: attempts 2 or 3 improve official useful events or score over attempt 1 without increasing invalid actions.

4. Train or adapt the JEPA encoder on legal failed-attempt traces without solution labels.
   - Use only public observations, legal actions, chosen actions, score/event deltas, and terminal flags.
   - Do not use game IDs as features or tune per game.
   - Treat internal losses as diagnostics only; proof remains official public runtime.

## Attempted This Run

Relation-level component goal chains:
   - Record public relation-state signatures before and after component transitions.
   - Abstract contact actions into relation templates that can resolve to currently legal clicks on matching component value/area relations.
   - Store relation-chain edges with delayed public-event credit, goal-state values, failures, contradictions, and live postcondition expectations.
   - Result: implemented and audited, but official runtime remained `NO IMPROVEMENT FOUND`; attempts 2 and 3 did not improve score or useful events over attempt 1.

Generic relation-delta event mechanism mining:
   - Convert public component before/after hypotheses into scoped generic delta tokens for movement direction, value transform, appearance, disappearance, visual transform, and blocked/no-effect.
   - Score current legal actions by matching public action templates, generalized relation keys, action family, and component value scopes.
   - Record delayed public-event goal credit and contradiction penalties for stale expected deltas.
   - Result: implemented and audited, but official runtime remained `NO IMPROVEMENT FOUND`; attempts 2 and 3 did not improve score or useful events over attempt 1.

Component transition-goal chain search:
   - Record exact public component-state signatures before and after each action.
   - Store component graph edges keyed by `(before_state, action, after_state)` with counts, values, failures, contradictions, expectations, delayed public-event credit, and goal-state values.
   - Search short action chains from the current public component state and activate the best chain as the live sequence plan.
   - Abort stale chains when observed public postconditions contradict the predicted after-state.
   - Result: implemented and audited, but official runtime remained `NO IMPROVEMENT FOUND`; attempts 2 and 3 did not improve score or useful events over attempt 1.

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

Implement experiment 1: ground sequence candidates in predicted public relation transitions, then rerun tests, official evaluation, audits, compaction, and docs. Do not use game IDs, game source, hidden labels, manual hints, or per-game branches.
