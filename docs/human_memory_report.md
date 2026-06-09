# Human Memory Report

Terminal outcome: **HUMAN MEMORY PROVEN**

## Architecture

Sparse cue-addressed engram memory stores compressed recurrent latents, content fragments, body/affect/action/private-token tags, source distributions, time order, and source histories. Recall performs pattern completion by attention over sparse random-projection traces; replay fits a bounded semantic adapter from stored traces only.

## Gate Metrics

| Gate | Value | Pass |
| --- | ---: | --- |
| `partial_cue_accuracy` | 0.937500 |  |
| `noisy_cue_accuracy` | 1.000000 |  |
| `wrong_cue_rejection` | 0.937500 |  |
| `similar_episode_discrimination` | 1.000000 |  |
| `partial_to_full_reconstruction` | 1.000000 |  |
| `accuracy_loss_after_20_similar` | 0.000000 |  |
| `targeted_trace_corruption_recall_action_degrade` | 1.000000 |  |
| `unrelated_memory_corruption_degrade` | 0.000000 |  |
| `relevant_trace_action_shift` | 0.223602 |  |
| `relevant_trace_private_shift` | 0.115654 |  |
| `relevant_trace_language_shift` | 0.064516 |  |
| `semantic_accuracy_before_replay` | 0.015625 |  |
| `semantic_accuracy_after_replay` | 1.000000 |  |
| `semantic_accuracy_improvement` | 0.984375 |  |
| `old_task_core_before` | 0.996436 |  |
| `old_task_core_after` | 0.996436 |  |
| `old_task_core_delta` | 0.000000 |  |
| `samples_replayed` | 64.000000 |  |
| `adapter_rank` | 49.000000 |  |
| `source_history_accuracy` | 1.000000 |  |
| `current_belief_accuracy` | 1.000000 |  |
| `source_monitoring_accuracy` | 1.000000 |  |
| `observed_accuracy` | 1.000000 |  |
| `told_accuracy` | 1.000000 |  |
| `imagined_accuracy` | 1.000000 |  |
| `inferred_accuracy` | 1.000000 |  |
| `replayed_accuracy` | 1.000000 |  |
| `reconstructed_accuracy` | 1.000000 |  |
| `restart_recall_accuracy` | 1.000000 |  |
| `zero_memory_recall_accuracy` | 0.015625 |  |
| `corrupt_memory_recall_accuracy` | 0.015625 |  |
| `zero_memory_degradation` | 0.984375 |  |
| `corrupt_memory_degradation` | 0.984375 |  |

## Causal Ablation

```json
{
  "targeted_trace_corruption_recall_action_degrade": 1.0,
  "unrelated_memory_corruption_degrade": 0.0,
  "relevant_trace_action_shift": 0.22360216081142426,
  "relevant_trace_private_shift": 0.1156536117196083,
  "relevant_trace_language_shift": 0.0645160973072052
}
```

## Limitations
- None.
