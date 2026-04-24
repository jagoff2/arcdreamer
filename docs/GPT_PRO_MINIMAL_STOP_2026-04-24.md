# GPT-Pro Stop Condition: Minimal Learned Online Agent

Date: 2026-04-24

## Decision

`learned_online_minimal` remains useful as a clean compliance scaffold, dense-action scorer, falsification harness, and online-update testbed. It is not validated ARC competence.

## Evidence

ARC diagnostics on `ar25-0c556536` stayed at zero reward and zero level completion:

- Untrained minimal: select repetition, `select=64`, max streak `63`.
- No-effect cost target: dense click walking, `click=74`, max streak `1`.
- Parametric click family/context evidence: visible movement loop, `move=73`, max streak `21`.

This is family substitution under local penalties, not increasing ARC competence.

## Required Correction

Stop ARC behavior tuning on the shallow minimal scorer.

Keep generic infrastructure:

- full dense legal action scoring;
- claim-boundary metadata and tests;
- generic no-effect / visible-only labels;
- parametric-click evidence inheritance;
- no-sweep/null dense tests;
- replay/probe information-gain target;
- removal of raw uncertainty from direct policy value.

Next implementation must be `learned_online_recurrent_v1`: a clean recurrent latent-state model with reward/useful/visible/cost/info heads, online fast-head updates, replay/probe information gain, synthetic sequence gates, and no imports from Spotlight, TheoryManager, RuntimeRuleController, HybridPlanner, graph-frontier control, or scripted probes.
