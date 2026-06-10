# Conversation Report

Terminal outcome: **GROUNDED CONVERSATION PROVEN**

## Gates

| Gate | Passed |
| --- | --- |
| `twenty_turn_success_ge_0_75` | True |
| `sixty_turn_memory_ge_0_70` | True |
| `same_utterance_state_ge_0_85` | True |
| `copy_rate_le_0_03` | True |
| `form_overlap_lt_0_10` | True |
| `heldout_free_text_ge_0_75` | True |
| `paraphrase_free_text_ge_0_75` | True |
| `parseable_nonempty_ge_0_90` | True |
| `zero_z_drop_ge_0_30` | True |
| `shuffled_z_drop_ge_0_30` | True |
| `relevant_memory_drop_ge_0_40` | True |
| `unrelated_memory_drop_lt_0_10` | True |
| `listener_drop_ge_0_25` | True |
| `private_drop_ge_0_15` | True |
| `source_reports_ge_0_80` | True |
| `contradiction_ge_0_80` | True |
| `uncertainty_missing_ge_0_75` | True |
| `wrong_cue_rejection_ge_0_80` | True |
| `non_role_behavior_ge_0_70` | True |
| `override_rejection_ge_0_70` | True |
| `role_dependency_absent` | True |
| `random_labels_no_effect` | True |
| `text_changes_state` | True |
| `retention_eval_passes` | True |
| `memory_eval_passes` | True |
| `living_eval_passes` | True |
| `dialogue_eval_passes` | True |
| `leakage_scan_passes` | True |
| `hidden_target_canary_zero` | True |
| `independent_verify_passes` | True |
| `no_text_as_state_path` | True |

## Heldout Metrics

| Metric | Value |
| --- | ---: |
| `task_success_20_turn` | 0.945563 |
| `memory_consistency_60_turn` | 0.802344 |
| `same_utterance_different_state_accuracy` | 1.000000 |
| `heldout_free_text_semantic_correctness` | 0.873321 |
| `parseable_nonempty_rate` | 0.996041 |
| `answer_head_accuracy` | 0.895377 |
| `exact_training_sentence_copy_rate` | 0.000000 |
| `template_form_overlap` | 0.000000 |
| `normal_memory_accuracy` | 0.825537 |
| `corrupt_memory_accuracy` | 0.300955 |
| `relevant_memory_corruption_delta` | 0.524582 |
| `unrelated_memory_corruption_delta` | 0.002232 |
| `zero_z_dialogue_delta` | 0.979911 |
| `shuffled_z_dialogue_delta` | 0.794643 |
| `listener_action_goal_accuracy` | 0.997159 |
| `listener_disabled_action_goal_accuracy` | 0.106534 |
| `listener_ablation_delta` | 0.890625 |
| `text_command_state_shift` | 3.906035 |
| `private_dialogue_accuracy` | 1.000000 |
| `private_disabled_accuracy` | 0.235838 |
| `private_speech_delta` | 0.764162 |
| `source_report_accuracy` | 1.000000 |
| `contradiction_resolution_accuracy` | 1.000000 |
| `uncertainty_missing_evidence` | 1.000000 |
| `wrong_cue_rejection` | 1.000000 |
| `silence_clarification_refusal_appropriate` | 0.988468 |
| `invalid_goal_override_rejection` | 1.000000 |
| `role_dependency_absent` | True |
| `random_label_max_abs_diff` | 0.000000 |

## Limitations
- None.
