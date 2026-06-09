# Retention Fix Report

Terminal outcome: **RETENTION FIX PROVEN**

## New Concepts

| Concept ID | Before | After | Improvement | Pass |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.000000 | 1.000000 | 1.000000 | True |
| 1 | 0.000000 | 1.000000 | 1.000000 | True |
| 2 | 0.000000 | 1.000000 | 1.000000 | True |
| 3 | 0.000000 | 1.000000 | 1.000000 | True |
| 4 | 0.000000 | 1.000000 | 1.000000 | True |

## Old-Task Retention

| Metric | Before | After |
| --- | ---: | ---: |
| `action_accuracy` | 0.981348 | 0.981348 |
| `delayed_memory_accuracy` | 1.000000 | 1.000000 |
| `object_color_accuracy` | 1.000000 | 1.000000 |
| `object_pos_accuracy` | 1.000000 | 1.000000 |
| `object_permanence_accuracy` | 1.000000 | 1.000000 |
| `provenance_accuracy` | 0.999707 | 0.999707 |
| `grounded_language_accuracy` | 0.994531 | 0.994531 |
| `self_world_continuity_accuracy` | 1.000000 | 1.000000 |
| `core_mean` | 0.996948 | 0.996948 |

## Restart

```json
{
  "memory_path": "runs\\retention_concepts_fast.pt",
  "loaded_new_concept_recall": 1.0,
  "zero_memory_new_concept_recall": 0.0,
  "corrupt_memory_new_concept_recall": 0.0,
  "old_task_core_after_reload": 0.996948243677616,
  "checks": {
    "loaded_new_recall_ge_0_85": true,
    "old_task_core_ge_0_90": true,
    "zero_memory_degrades_new_recall": true,
    "corrupt_memory_degrades_new_recall": true
  }
}
```

## Prior Living Preservation

```json
{
  "durable_restart_memory_ge_0_85": true,
  "memory_beats_zero_reset_by_0_40": true,
  "idle_memory_ge_0_95": true,
  "idle_object_pos_ge_0_95": true,
  "idle_goal_action_ge_0_70": true,
  "public_repetition_lt_0_40": true,
  "private_repetition_lt_0_40": true,
  "private_tokens_generated_and_causal": true,
  "richer_dynamics_all_present": true
}
```

## Audit Preservation

```json
{
  "leakage_scan_passes": true,
  "hidden_target_canary_diff_zero": true,
  "tensor_z_stream_no_text_loop": true,
  "zero_z_degrades": true,
  "shuffled_z_degrades": true,
  "no_language_degrades_grounding": true,
  "no_private_perturbs_outputs": true,
  "random_private_perturbs_outputs": true,
  "corrupt_tensor_memory_degrades": true
}
```

## Limitations
- None.
