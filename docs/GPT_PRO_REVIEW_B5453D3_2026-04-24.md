# GPT-Pro Review for `b5453d3`, 2026-04-24

Context: requested after pushing `b5453d3514bc409349fa937970c34ed65d3390ef`.

Key diagnosis:

- The TD-target patch penalizes no-progress visible motion after the action, but pre-action scoring still turns visible change, uncertainty, coverage, memory, and movement hypotheses into positive action value.
- The agent has learned that motion is a consistent perceptual effect, but not that motion is diagnostically exhausted unless it yields objective or structural progress.
- `notify_transition()` treats visible-only nonprogress as too weak a failure.
- `planner.notify_transition(changed=record.delta.has_visible_effect or progress)` resets planner stall on reversible motion.
- `memory.write_transition()` can still let pure visible nonprogress become positive option/schema pressure.

Minimum next patch:

1. Add explicit `visible_only_failures` and `objective_failures` to `ActionEvidence`, persist them, and propagate them through exact/local/global family evidence.
2. Classify transitions as objective success, structural effect, or pure visible-only nonobjective motion.
3. Gate `information`, `change`, `uncertainty`, `coverage`, `memory`, and option value by visible-only trap pressure for non-binding/non-probe actions.
4. Add `visible_only_failure_rate`, `objective_failure_pressure`, and `movement_trap_pressure` into spotlight features and diagnostics.
5. Trigger diagnostic binding earlier when movement trap pressure is high, without scripted action schedules or action pruning.
6. In the agent wrapper, report planner `changed=True` only for structural/objective changes, not reversible visible motion.
7. Keep episodic memory for visible-only transitions, but prevent pure visible nonprogress from creating positive option/schema value.
8. If this does not suppress `expected_change` value inflation, split the world model into visible-change and useful-change heads later.

Metrics to watch next:

- 80-step clean eval should reduce `move <= 40`, `undo <= 6`, `reset == 0`, `max_same_action_streak <= 4`.
- Binding/targeted diagnostic actions should increase materially.
- `visible_only_failure_total >= 6` in zero-progress runs.
- Move/undo reliability should fall after repeated no-progress visible motion.
- `executive_last_target <= -0.45` on visible-only/undo transitions.
- `option_schema_bonus` and `memory_bonus` for move/undo should stay near zero in zero-progress runs.
