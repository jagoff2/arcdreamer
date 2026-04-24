# GPT-Pro Consensus: Learned Online Path

Date: 2026-04-24

## Decision

Stop patching the Spotlight/TheoryManager stack as the ARC-facing success-claiming agent.

The Spotlight stack remains useful as an instrumented heuristic baseline, diagnostics layer, and trace generator. It is not claim-eligible for the core goal because action selection is still owned by hand-shaped scoring, probe routines, and controller arbitration. Any ARC win from this path must be reported as baseline-only unless separately ablated and labeled.

## What Was Learned

- The recent Spotlight patches improved falsification telemetry and reduced some bad proxy behaviors: resets, undo collapse, and long same-action streaks.
- The latest 80-step ARC probe still had zero reward, zero level completion, and no win.
- The new failure mode is a dense coordinate click walk under the label `diagnostic_binding_probe`.
- Better action diversity is not ARC progress. Success, reward, level completion, and ablation-backed learned decision ownership are primary.

## Agreed Path

1. Commit/guard Spotlight as `instrumented_spotlight_baseline`, not as the claimed learned agent.
2. Build a minimal learned decision owner before any larger architecture:
   - full dense action scoring;
   - explicit online belief/action-semantics state;
   - useful, visible, reward, information-gain, uncertainty, and learned-cost heads;
   - online fast updates;
   - episodic memory as model input, not direct bonus;
   - grounded question tokens as model input, not action scripts.
3. Train and gate on randomized synthetic online-adaptation tasks before ARC claiming eval.
4. Use ARC 80-step runs as diagnostics only until gates pass.
5. Use 320-step ARC attempts only after synthetic gates show learned online decision ownership.

## Required Gates

- Every legal dense action is scored or represented by a learned full-support policy. Chunking is allowed; static candidate caps are not.
- Same-state action scores change after contradictory transition evidence.
- Online agent materially beats frozen agent on randomized binding and visible-vs-useful trap tasks.
- Learned agent beats family/exact-count baselines on context-conditioned dense grounding tasks.
- Action-order permutation tests and null-task controls reject coordinate/action-list sweep behavior.
- Question-token and memory ablations matter before claiming grounded language or memory usefulness.
- Clean learned eval cannot instantiate Spotlight, TheoryManager, RuntimeRuleController, HybridPlanner, graph/frontier control, or diagnostic binding probes as the primary controller.

## Stop Conditions

- Stop tuning ARC behavior if synthetic gates have not passed.
- Stop treating action diversity, click count, low reset count, or low same-action streak as progress when reward and level progress remain zero.
- Stop and redesign if a learned run is dominated by hand-authored score components, probe routines, graph/frontier control, or coordinate enumeration.
- Stop claiming language or memory as functional if shuffle/off ablations do not degrade controlled synthetic behavior.

## Immediate Implementation Order

1. Add boundary docs, metadata, and tests marking Scientist/Spotlight as non-claimable instrumentation.
2. Preserve the current dirty Spotlight telemetry patch behind that boundary.
3. Add `arcagi.learned_online` with a minimal model, fast belief state, question tokens, memory features, action features, and all-action chunked policy.
4. Add randomized synthetic online-adaptation tasks and baseline agents.
5. Add gates and run tests before any ARC claim.
