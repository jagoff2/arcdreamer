# Head Collapse Report

Terminal outcome: **HEAD-COLLAPSE PROVEN**

## Head-Disabled Metrics

| Metric | Value |
| --- | ---: |
| `informative_action_rate` | 0.9999328851699829 |
| `random_baseline_success` | 0.08580636978149414 |
| `repeat_collapse` | 0.08917236328125 |
| `self_directed_exploration_ticks` | 16384.0 |
| `next_state_prediction` | 0.977783203125 |
| `planner_accuracy` | 0.9999100565910339 |
| `reactive_baseline_accuracy` | 0.12370670586824417 |
| `counterfactual_choice` | 1.0 |
| `skill_accuracy` | 0.99993896484375 |
| `skill_count_learned` | 8.0 |
| `skill_transfer` | 0.99993896484375 |
| `project_accuracy` | 0.9998779296875 |
| `multi_step_projects` | 4.0 |
| `restart_resume` | 1.0 |
| `useful_question_under_uncertainty` | 0.9973913431167603 |
| `testimony_observation_conflict_resolution` | 0.9999999403953552 |
| `partner_history_restart_recall` | 0.997194230556488 |
| `no_forced_reply` | 0.9998082518577576 |
| `random_margin` | 0.9141265153884888 |
| `planner_margin` | 0.8762033507227898 |
| `core_score` | 0.9971836805343628 |

## Ablations

| Ablation | Delta |
| --- | ---: |
| `zero_z_delta` | 0.9260647984221577 |
| `shuffled_z_delta` | 0.5202318102121353 |
| `corrupt_memory_delta` | 0.7551816433668137 |
| `corrupt_drive_delta` | 0.9260647984221577 |

## Gate Checks

| Gate | Pass |
| --- | --- |
| `head_disabled_informative_ge_0_70` | True |
| `head_disabled_random_margin_ge_0_30` | True |
| `head_disabled_repeat_collapse_lt_0_20` | True |
| `head_disabled_next_state_ge_0_75` | True |
| `head_disabled_planner_margin_ge_0_25` | True |
| `head_disabled_counterfactual_ge_0_70` | True |
| `head_disabled_skills_ge_5` | True |
| `head_disabled_skill_transfer_ge_0_70` | True |
| `head_disabled_projects_ge_3` | True |
| `head_disabled_restart_ge_0_80` | True |
| `head_disabled_social_questions_ge_0_80` | True |
| `head_disabled_social_conflict_ge_0_80` | True |
| `head_disabled_partner_ge_0_80` | True |
| `head_disabled_no_forced_reply_ge_0_80` | True |
| `structured_disable_delta_le_0_05` | True |
| `probe_removal_delta_le_0_05` | True |
| `zero_z_degrades_ge_0_30` | True |
| `shuffled_z_degrades_ge_0_30` | True |
| `corrupt_memory_degrades_ge_0_40` | True |
| `corrupt_drive_degrades_ge_0_25` | True |
| `anti_crutch_source_scan_passes` | True |
| `runtime_trace_old_heads_not_used` | True |
| `pytest_passes` | True |
| `retention_eval_passes` | True |
| `memory_eval_passes` | True |
| `living_eval_passes` | True |
| `dialogue_eval_passes` | True |
| `explorer_eval_passes` | True |
| `leakage_scan_passes` | True |
| `hidden_target_canary_zero` | True |
| `independent_verify_passes` | True |
| `no_text_as_state_path` | True |

## Limitations
- None.
