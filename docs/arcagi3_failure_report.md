# ARC-AGI-3 Failure Diagnosis

Terminal outcome: **ARC FAILURE DIAGNOSIS PROVEN**

## Official Runtime Status
Official runtime available: `True`. Diagnosis device: `cuda`. CUDA: `True`.

## Aggregate Score Table

| Name | Solve Rate | Mean Score | Mean Normalized | Steps | Entropy | Repeat Collapse |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| explorer | 0.000 | 0.000 | 0.000 | 93.92 | 0.285 | 0.919 |
| random_legal | 0.000 | 0.000 | 0.000 | 106.60 | 5.249 | 0.090 |
| repeat_last_action | 0.000 | 0.000 | 0.000 | 102.28 | 0.005 | 0.999 |
| coverage_graph_exploration | 0.000 | 0.004 | 0.004 | 110.32 | 5.403 | 0.075 |
| novelty_first | 0.000 | 0.004 | 0.004 | 114.76 | 4.973 | 0.249 |
| greedy_observable_score_delta | 0.000 | 0.004 | 0.004 | 114.84 | 0.765 | 0.863 |
| oracle_free_observed_graph_bfs | 0.000 | 0.004 | 0.004 | 110.32 | 5.492 | 0.064 |

## Failure Table

| Game | Score | Levels | Steps | Invalid | Entropy | Repeat | Primary | Secondary | Confidence |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |
| ar25-0c556536 | 0.000 | 0/8 | 164 | 0.000 | -0.000 | 1.000 | C | F, H, G | high |
| bp35-0a0ad940 | 0.000 | 0/9 | 64 | 0.000 | -0.000 | 1.000 | C | F, H, G | high |
| cd82-fb555c5d | 0.000 | 0/6 | 100 | 0.000 | -0.000 | 1.000 | C | F, H, G | high |
| cn04-2fe56bfb | 0.000 | 0/6 | 75 | 0.000 | 0.177 | 0.973 | C | F, H, G | high |
| dc22-fdcac232 | 0.000 | 0/6 | 128 | 0.000 | 1.110 | 0.695 | F | C, H | medium |
| ft09-0d8bbf25 | 0.000 | 0/6 | 32 | 0.000 | 0.997 | 0.531 | F | C, H | medium |
| g50t-5849a774 | 0.000 | 0/7 | 130 | 0.000 | -0.000 | 1.000 | C | F, G | high |
| ka59-38d34dbb | 0.000 | 0/7 | 100 | 0.000 | 0.194 | 0.970 | C | F, H, G | high |
| lf52-271a04aa | 0.000 | 0/10 | 64 | 0.000 | -0.000 | 1.000 | C | F, H, G | high |
| lp85-305b61c3 | 0.000 | 0/8 | 110 | 0.000 | -0.000 | 1.000 | C | F, H, G | high |
| ls20-9607627b | 0.000 | 0/7 | 129 | 0.000 | 0.065 | 0.992 | C | F, G | high |
| m0r0-492f87ba | 0.000 | 0/6 | 151 | 0.000 | 0.102 | 0.987 | C | F, H, G | high |
| r11l-495a7899 | 0.000 | 0/6 | 60 | 0.000 | 2.937 | 0.200 | F | C, H | medium |
| re86-8af5384d | 0.000 | 0/8 | 100 | 0.000 | -0.000 | 1.000 | C | G | high |
| s5i5-18d95033 | 0.000 | 0/8 | 50 | 0.000 | -0.000 | 1.000 | C | F, H, G | high |
| sb26-7fbdac44 | 0.000 | 0/8 | 92 | 0.000 | -0.000 | 1.000 | C | F, H, G | high |
| sc25-635fd71a | 0.000 | 0/6 | 52 | 0.000 | -0.000 | 1.000 | C | F, H, G | high |
| sk48-d8078629 | 0.000 | 0/8 | 196 | 0.000 | 0.404 | 0.934 | C | F, H, G | high |
| sp80-589a99af | 0.000 | 0/6 | 30 | 0.000 | -0.000 | 1.000 | C | F, H, G | high |
| su15-1944f8ab | 0.000 | 0/9 | 32 | 0.000 | -0.000 | 1.000 | C | F, H, G | high |
| tn36-ef4dde99 | 0.000 | 0/7 | 61 | 0.000 | -0.000 | 1.000 | C | F, H, G | high |
| tr87-cd924810 | 0.000 | 0/6 | 128 | 0.000 | -0.000 | 1.000 | C | F, G | high |
| tu93-0768757b | 0.000 | 0/9 | 50 | 0.000 | -0.000 | 1.000 | C | F, G | high |
| vc33-5430563c | 0.000 | 0/7 | 50 | 0.000 | 1.127 | 0.680 | F | C, H | medium |
| wa30-ee6fef47 | 0.000 | 0/9 | 200 | 0.000 | -0.000 | 1.000 | C | G | high |

## Top Bottlenecks
- action repetition collapse: 21 games. mean repeat collapse 0.919
- pixel-proxy observation bottleneck: 23 games. Observation audit flags dense visual proxies or large click surfaces.
- useful events not converted into completion: 0 games. Only games with useful events still failed to solve.
- no multi-step planner sequence: 21 games. Planner class assigned where entropy did not produce progress.

## No-Hack Proof
No-hack audit passes: `True`.
Scanned: `src\arcagi3_adapter.py, src\arcagi3_baselines.py, src\arcagi3_official.py, src\arcagi3_official_eval.py, src\arcagi3_trace_analysis.py, src\arcagi3_failure_taxonomy.py, src\arcagi3_diagnose.py`.

## Prior Properties

| Name | Outcome | Passes |
| --- | --- | --- |
| fixture_arc_report | ARC-AGI-3 ADAPTER PROVEN | True |
| post_arc_audit | AUDIT PROVEN | True |
| head_collapse | HEAD-COLLAPSE PROVEN | True |

## Representative Trace Excerpts

### ar25-0c556536
- step 81: action `3`, events `[]`, score_delta `-0.001`, baselines `{'random_legal': 'click:32:48', 'repeat_last_action': '1', 'coverage_graph_exploration': 'click:8:19', 'novelty_first': 'click:55:21', 'greedy_observable_score_delta': 'click:1:16', 'oracle_free_observed_graph_bfs': 'click:32:21'}`
- step 82: action `3`, events `[]`, score_delta `-0.001`, baselines `{'random_legal': 'click:32:22', 'repeat_last_action': '1', 'coverage_graph_exploration': 'click:31:19', 'novelty_first': 'click:6:22', 'greedy_observable_score_delta': 'click:1:16', 'oracle_free_observed_graph_bfs': 'click:55:21'}`
- step 83: action `3`, events `[]`, score_delta `-0.001`, baselines `{'random_legal': 'click:58:47', 'repeat_last_action': '1', 'coverage_graph_exploration': 'click:55:19', 'novelty_first': 'click:8:22', 'greedy_observable_score_delta': 'click:1:16', 'oracle_free_observed_graph_bfs': 'click:6:22'}`

### bp35-0a0ad940
- step 31: action `click:32:31`, events `[]`, score_delta `-0.001`, baselines `{'random_legal': 'click:22:35', 'repeat_last_action': '3', 'coverage_graph_exploration': 'click:15:63', 'novelty_first': 'click:15:63', 'greedy_observable_score_delta': 'click:22:38', 'oracle_free_observed_graph_bfs': 'click:15:63'}`
- step 32: action `click:32:31`, events `[]`, score_delta `-0.001`, baselines `{'random_legal': 'click:22:38', 'repeat_last_action': '3', 'coverage_graph_exploration': 'click:48:63', 'novelty_first': 'click:48:63', 'greedy_observable_score_delta': 'click:22:38', 'oracle_free_observed_graph_bfs': 'click:48:63'}`
- step 33: action `click:32:31`, events `[]`, score_delta `-0.001`, baselines `{'random_legal': 'click:1:22', 'repeat_last_action': '3', 'coverage_graph_exploration': 'click:16:63', 'novelty_first': 'click:16:63', 'greedy_observable_score_delta': 'click:22:38', 'oracle_free_observed_graph_bfs': 'click:16:63'}`

### cd82-fb555c5d
- step 49: action `1`, events `[]`, score_delta `-0.001`, baselines `{'random_legal': 'click:29:25', 'repeat_last_action': '1', 'coverage_graph_exploration': 'click:16:1', 'novelty_first': 'click:40:1', 'greedy_observable_score_delta': 'click:32:26', 'oracle_free_observed_graph_bfs': 'click:35:1'}`
- step 50: action `1`, events `[]`, score_delta `-0.001`, baselines `{'random_legal': 'click:31:2', 'repeat_last_action': '1', 'coverage_graph_exploration': 'click:15:63', 'novelty_first': 'click:15:63', 'greedy_observable_score_delta': 'click:32:26', 'oracle_free_observed_graph_bfs': 'click:15:63'}`
- step 51: action `1`, events `[]`, score_delta `-0.001`, baselines `{'random_legal': 'click:30:8', 'repeat_last_action': '1', 'coverage_graph_exploration': 'click:44:0', 'novelty_first': 'click:44:0', 'greedy_observable_score_delta': 'click:32:26', 'oracle_free_observed_graph_bfs': 'click:44:0'}`

### cn04-2fe56bfb
- step 36: action `1`, events `[]`, score_delta `-0.001`, baselines `{'random_legal': 'click:45:39', 'repeat_last_action': '1', 'coverage_graph_exploration': 'click:31:0', 'novelty_first': 'click:21:2', 'greedy_observable_score_delta': 'click:39:0', 'oracle_free_observed_graph_bfs': 'click:17:2'}`
- step 37: action `1`, events `[]`, score_delta `-0.001`, baselines `{'random_legal': 'click:21:10', 'repeat_last_action': '1', 'coverage_graph_exploration': 'click:40:0', 'novelty_first': 'click:24:0', 'greedy_observable_score_delta': 'click:40:0', 'oracle_free_observed_graph_bfs': 'click:24:0'}`
- step 38: action `1`, events `[]`, score_delta `-0.001`, baselines `{'random_legal': 'click:21:18', 'repeat_last_action': '1', 'coverage_graph_exploration': 'click:24:0', 'novelty_first': 'click:23:2', 'greedy_observable_score_delta': 'click:40:0', 'oracle_free_observed_graph_bfs': 'click:19:2'}`

### dc22-fdcac232
- step 63: action `2`, events `[]`, score_delta `-0.001`, baselines `{'random_legal': 'click:58:10', 'repeat_last_action': '1', 'coverage_graph_exploration': 'click:48:63', 'novelty_first': 'click:48:63', 'greedy_observable_score_delta': 'click:24:20', 'oracle_free_observed_graph_bfs': 'click:48:63'}`
- step 64: action `2`, events `[]`, score_delta `-0.001`, baselines `{'random_legal': 'click:54:6', 'repeat_last_action': '1', 'coverage_graph_exploration': 'click:38:0', 'novelty_first': 'click:57:0', 'greedy_observable_score_delta': 'click:24:20', 'oracle_free_observed_graph_bfs': 'click:57:0'}`
- step 65: action `2`, events `[]`, score_delta `-0.001`, baselines `{'random_legal': 'click:20:22', 'repeat_last_action': '1', 'coverage_graph_exploration': 'click:4:2', 'novelty_first': 'click:46:4', 'greedy_observable_score_delta': 'click:24:20', 'oracle_free_observed_graph_bfs': 'click:35:3'}`

## Concrete Next Steps
- Replace the pixel-proxy frame collapse with learned or explicitly evaluated object/affordance extraction, then rerun diagnosis before any policy change.
- Separate click-surface ranking from movement-action ranking so dense click games do not collapse to one click or one keyboard action.
- Add a diagnosis-only transition model probe that predicts event deltas from public frames and legal actions; keep it outside the policy until audited.
- Make memory audit stricter by recording event-token writes directly in adapter memory diagnostics rather than inferring them from post-event recall.
- Evaluate planner sequence formation on official trace replays with counterfactual legal-action rollouts, still without using game source or labels.

## Limitations
- Diagnosis does not build a solver and does not prove that any proposed next step will solve ARC-AGI-3.
