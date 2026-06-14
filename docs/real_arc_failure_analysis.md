# Real ARC failure analysis

The ar25 traces show the current agent can generate many transition labels that look useful while still ending with zero score and game_over.  That means the evaluation signal is not a motor-control problem; it is a goal-credit problem.

A visible state change is not progress.  Public controllability is not progress.  A reachable new state is not progress unless it is attached to a candidate goal whose distance or posterior actually improves.

The new invariant is:

```text
useful_effect = real progress OR terminal win OR attached candidate-goal improvement
```

Everything else must remain instrumental evidence, nuisance evidence, or negative evidence.  This is deliberately stricter than the previous tracker rows because ar25 punishes false usefulness harder than it rewards generic exploration.

The next target after this patch is not another policy checkpoint.  It is a planner that holds a candidate goal, computes goal-distance deltas under candidate actions, and executes a multi-step plan until contradiction or success.
