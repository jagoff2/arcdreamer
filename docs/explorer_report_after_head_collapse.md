# Explorer Report

Terminal outcome: **EXPLORER CORE PROVEN**

## Exploration

| Metric | Value |
| --- | ---: |
| `informative_action_rate` | 1.0 |
| `random_baseline_success` | 0.08580636978149414 |
| `random_margin` | 0.9141936302185059 |
| `repeat_collapse` | 0.08892822265625 |
| `self_directed_exploration_ticks` | 16384.0 |

## Learning

| Metric | Value |
| --- | ---: |
| `next_state_accuracy` | 0.942626953125 |
| `next_state_without_intrinsic` | 0.0140380859375 |
| `prediction_uncertainty_improvement` | 0.9285888671875 |
| `uncertainty_accuracy` | 1.0 |
| `novelty_accuracy` | 0.99993896484375 |
| `novelty_distractor_rejection` | 1.0 |
| `noise_fixation` | 0.0 |

## Planning

| Metric | Value |
| --- | ---: |
| `next_state_prediction` | 0.942626953125 |
| `planner_accuracy` | 0.999730110168457 |
| `reactive_baseline_accuracy` | 0.12370670586824417 |
| `imagined_planner_margin` | 0.8760234043002129 |
| `counterfactual_choice` | 1.0 |
| `planner_repeat_collapse` | 0.31707763671875 |

## Skills

| Metric | Value |
| --- | ---: |
| `skill_accuracy` | 1.0 |
| `skill_count_learned` | 8.0 |
| `transfer_accuracy` | 1.0 |
| `skill_without_skill_memory` | 0.52313232421875 |
| `skill_ablation_delta` | 0.47686767578125 |

## Projects

| Metric | Value |
| --- | ---: |
| `project_accuracy` | 1.0 |
| `multi_step_projects` | 4.0 |
| `restart_resume_accuracy` | 1.0 |
| `restart_without_project_state` | 0.605344295501709 |
| `safety_preservation` | 0.9998054504394531 |

## Social

| Metric | Value |
| --- | ---: |
| `useful_question_under_uncertainty` | 1.0 |
| `testimony_observation_conflict_resolution` | 1.0 |
| `partner_history_restart_recall` | 0.9988777041435242 |
| `partner_without_social_state` | 0.25084176659584045 |
| `partner_social_delta` | 0.7480359375476837 |

## Mindlike

| Metric | Value |
| --- | ---: |
| `long_session_not_random_or_reactive` | True |
| `no_forced_reply` | 1.0 |
| `mind_action_accuracy` | 1.0 |
| `required_action_coverage_met` | True |

## Gate Checks

| Gate | Pass |
| --- | --- |
| `exploration_informative_rate_ge_0_70` | True |
| `exploration_random_margin_ge_0_30` | True |
| `exploration_repeat_collapse_lt_0_20` | True |
| `exploration_ticks_ge_10000` | True |
| `learning_improvement_ge_0_25` | True |
| `learning_distractor_rejection_ge_0_80` | True |
| `learning_noise_fixation_lt_0_10` | True |
| `planning_next_state_ge_0_75` | True |
| `planning_margin_ge_0_25` | True |
| `planning_counterfactual_ge_0_70` | True |
| `skills_count_ge_5` | True |
| `skills_transfer_ge_0_70` | True |
| `skills_ablation_delta_ge_0_25` | True |
| `projects_count_ge_3` | True |
| `projects_restart_ge_0_80` | True |
| `projects_safety_ge_0_80` | True |
| `social_questions_ge_0_70` | True |
| `social_conflict_ge_0_80` | True |
| `social_partner_restart_ge_0_80` | True |
| `mindlike_not_random_or_reactive` | True |
| `mindlike_no_forced_reply_ge_0_80` | True |
| `mindlike_action_coverage` | True |
| `retention_eval_passes` | True |
| `memory_eval_passes` | True |
| `living_eval_passes` | True |
| `dialogue_eval_passes` | True |
| `leakage_scan_passes` | True |
| `hidden_target_canary_zero` | True |
| `independent_verify_passes` | True |
| `no_text_as_state_path` | True |

## Limitations
- None.
