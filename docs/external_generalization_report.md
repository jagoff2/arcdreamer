# External Generalization Discipline

Terminal outcome: **EXTERNAL GENERALIZATION DISCIPLINE PROVEN**

## External Suites

| Suite | Available | Tasks | Split |
| --- | --- | --- | --- |
| official_arcagi3 | True | ar25-0c556536, bp35-0a0ad940, cd82-fb555c5d, cn04-2fe56bfb, dc22-fdcac232, ft09-0d8bbf25, g50t-5849a774, ka59-38d34dbb, lf52-271a04aa, lp85-305b61c3, ls20-9607627b, m0r0-492f87ba, r11l-495a7899, re86-8af5384d, s5i5-18d95033, sb26-7fbdac44, sc25-635fd71a, sk48-d8078629, sp80-589a99af, su15-1944f8ab, tn36-ef4dde99, tr87-cd924810, tu93-0768757b, vc33-5430563c, wa30-ee6fef47 | sealed_eval |
| gymnasium_classic_control | True | CartPole-v1 | dev_and_sealed_eval |
| gymnasium_toy_text | True | FrozenLake-v1 | dev_and_sealed_eval |

## Claim Registry

| Claim | Metric | Baseline | Ablation | Status |
| --- | --- | --- | --- | --- |
| memory | sealed_eval.mean_normalized_score | best external baseline | corrupt_memory | unsupported |
| exploration | sealed_eval.mean_normalized_score | best external baseline | zero_z | unsupported |
| dialogue | sealed_eval.mean_normalized_score | best external baseline | no_dialogue | unsupported |
| head_collapse | sealed_eval.mean_normalized_score | best external baseline | disable_planner_imagination | unsupported |
| planner | sealed_eval.mean_normalized_score | best external baseline | disable_planner_imagination | unsupported |
| curiosity | sealed_eval.mean_unique_states | best external baseline | corrupt_drive | unsupported |
| social_state | sealed_eval.mean_normalized_score | best external baseline | no_social_state | unsupported |

## Aggregate Scores

| Suite | Controller | Mean Normalized | Solve Rate | Steps | Entropy | Repeat |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| official_arcagi3 | explorer | 0.000 | 0.000 | 93.92 | 0.285 | 0.919 |
| official_arcagi3 | random_legal | 0.000 | 0.000 | 106.60 | 5.249 | 0.090 |
| official_arcagi3 | repeat_last_action | 0.000 | 0.000 | 102.28 | 0.005 | 0.999 |
| official_arcagi3 | coverage_graph_exploration | 0.004 | 0.000 | 110.32 | 5.403 | 0.075 |
| official_arcagi3 | novelty_first | 0.004 | 0.000 | 114.76 | 4.973 | 0.249 |
| official_arcagi3 | greedy_observable_score_delta | 0.004 | 0.000 | 114.84 | 0.765 | 0.863 |
| official_arcagi3 | oracle_free_observed_graph_bfs | 0.004 | 0.000 | 110.32 | 5.492 | 0.064 |
| gymnasium_classic_control | ablation_corrupt_drive | 0.117 | 0.000 | 9.33 | 0.000 | 1.000 |
| gymnasium_classic_control | ablation_corrupt_memory | 0.117 | 0.000 | 9.33 | 0.000 | 1.000 |
| gymnasium_classic_control | ablation_disable_planner_imagination | 0.117 | 0.000 | 9.33 | 0.000 | 1.000 |
| gymnasium_classic_control | ablation_no_dialogue | 0.117 | 0.000 | 9.33 | 0.000 | 1.000 |
| gymnasium_classic_control | ablation_no_private_tokens | 0.117 | 0.000 | 9.33 | 0.000 | 1.000 |
| gymnasium_classic_control | ablation_no_social_state | 0.117 | 0.000 | 9.33 | 0.000 | 1.000 |
| gymnasium_classic_control | ablation_shuffled_z | 0.117 | 0.000 | 9.33 | 0.000 | 1.000 |
| gymnasium_classic_control | ablation_zero_z | 0.117 | 0.000 | 9.33 | 0.000 | 1.000 |
| gymnasium_classic_control | coverage_graph_exploration | 0.483 | 0.000 | 38.67 | 0.999 | 0.513 |
| gymnasium_classic_control | explorer | 0.117 | 0.000 | 9.33 | 0.000 | 1.000 |
| gymnasium_classic_control | greedy_observable_score_delta | 0.125 | 0.000 | 10.00 | 0.138 | 0.972 |
| gymnasium_classic_control | novelty_first | 0.117 | 0.000 | 9.33 | 0.000 | 1.000 |
| gymnasium_classic_control | oracle_free_observed_graph_bfs | 0.483 | 0.000 | 38.67 | 0.999 | 0.513 |
| gymnasium_classic_control | random_legal | 0.325 | 0.000 | 26.00 | 0.989 | 0.544 |
| gymnasium_classic_control | repeat_last_action | 0.117 | 0.000 | 9.33 | 0.000 | 1.000 |
| gymnasium_toy_text | ablation_corrupt_drive | 0.000 | 0.000 | 32.00 | 0.201 | 0.969 |
| gymnasium_toy_text | ablation_corrupt_memory | 0.000 | 0.000 | 3.00 | 0.000 | 1.000 |
| gymnasium_toy_text | ablation_disable_planner_imagination | 0.000 | 0.000 | 32.00 | 0.201 | 0.969 |
| gymnasium_toy_text | ablation_no_dialogue | 0.000 | 0.000 | 3.00 | 0.000 | 1.000 |
| gymnasium_toy_text | ablation_no_private_tokens | 0.000 | 0.000 | 3.00 | 0.000 | 1.000 |
| gymnasium_toy_text | ablation_no_social_state | 0.000 | 0.000 | 3.00 | 0.000 | 1.000 |
| gymnasium_toy_text | ablation_shuffled_z | 0.000 | 0.000 | 3.00 | 0.000 | 1.000 |
| gymnasium_toy_text | ablation_zero_z | 0.000 | 0.000 | 3.00 | 0.000 | 1.000 |
| gymnasium_toy_text | coverage_graph_exploration | 0.000 | 0.000 | 6.00 | 1.918 | 0.333 |
| gymnasium_toy_text | explorer | 0.000 | 0.000 | 3.00 | 0.000 | 1.000 |
| gymnasium_toy_text | greedy_observable_score_delta | 0.000 | 0.000 | 32.00 | 0.134 | 0.979 |
| gymnasium_toy_text | novelty_first | 0.000 | 0.000 | 32.00 | 0.000 | 1.000 |
| gymnasium_toy_text | oracle_free_observed_graph_bfs | 0.000 | 0.000 | 3.00 | 1.585 | 0.333 |
| gymnasium_toy_text | random_legal | 0.000 | 0.000 | 6.33 | 1.322 | 0.444 |
| gymnasium_toy_text | repeat_last_action | 0.000 | 0.000 | 32.00 | 0.000 | 1.000 |

## Unsupported Claims
- memory: margin `0.0`, ablation drop `0.0`.
- exploration: margin `0.0`, ablation drop `0.0`.
- dialogue: margin `0.0`, ablation drop `0.0`.
- head_collapse: margin `0.0`, ablation drop `0.0`.
- planner: margin `0.0`, ablation drop `0.0`.
- curiosity: margin `0.0`, ablation drop `0.0`.
- social_state: margin `0.0`, ablation drop `0.0`.

## Trace Paths
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\ar25-0c556536.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\bp35-0a0ad940.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\cd82-fb555c5d.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\cn04-2fe56bfb.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\dc22-fdcac232.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\ft09-0d8bbf25.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\g50t-5849a774.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\ka59-38d34dbb.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\lf52-271a04aa.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\lp85-305b61c3.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\ls20-9607627b.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\m0r0-492f87ba.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\r11l-495a7899.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\re86-8af5384d.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\s5i5-18d95033.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\sb26-7fbdac44.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\sc25-635fd71a.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\sk48-d8078629.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\sp80-589a99af.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\su15-1944f8ab.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\tn36-ef4dde99.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\tr87-cd924810.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\tu93-0768757b.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\vc33-5430563c.official.normal.json`
- `docs\external_traces\official_arcagi3\sealed_eval\explorer\wa30-ee6fef47.official.normal.json`
- `docs\external_traces\gymnasium_classic_control\dev\explorer\CartPole-v1.seed_0.json`
- `docs\external_traces\gymnasium_classic_control\dev\explorer\CartPole-v1.seed_1.json`
- `docs\external_traces\gymnasium_classic_control\dev\random_legal\CartPole-v1.seed_0.json`
- `docs\external_traces\gymnasium_classic_control\dev\random_legal\CartPole-v1.seed_1.json`
- `docs\external_traces\gymnasium_classic_control\dev\repeat_last_action\CartPole-v1.seed_0.json`
- `docs\external_traces\gymnasium_classic_control\dev\repeat_last_action\CartPole-v1.seed_1.json`
- `docs\external_traces\gymnasium_classic_control\dev\coverage_graph_exploration\CartPole-v1.seed_0.json`
- `docs\external_traces\gymnasium_classic_control\dev\coverage_graph_exploration\CartPole-v1.seed_1.json`
- `docs\external_traces\gymnasium_classic_control\dev\novelty_first\CartPole-v1.seed_0.json`
- `docs\external_traces\gymnasium_classic_control\dev\novelty_first\CartPole-v1.seed_1.json`
- `docs\external_traces\gymnasium_classic_control\dev\greedy_observable_score_delta\CartPole-v1.seed_0.json`
- `docs\external_traces\gymnasium_classic_control\dev\greedy_observable_score_delta\CartPole-v1.seed_1.json`
- `docs\external_traces\gymnasium_classic_control\dev\oracle_free_observed_graph_bfs\CartPole-v1.seed_0.json`
- `docs\external_traces\gymnasium_classic_control\dev\oracle_free_observed_graph_bfs\CartPole-v1.seed_1.json`
- `docs\external_traces\gymnasium_classic_control\dev\ablation_zero_z\CartPole-v1.seed_0.json`
- ... 135 more traces

## Limitations
- No active claim is externally supported; all internal synthetic metrics are diagnostic only.
- Gymnasium wrappers are external smoke suites, not evidence that the ARC failure is solved.
