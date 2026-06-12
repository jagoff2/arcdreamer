# ARC Affordance Baseline Report

Terminal outcome: **NO SIGNAL FOUND**
Answer: **no**
Selected variant: `state_graph_affordance`

## Gate
- `selected_variant`: `state_graph_affordance`
- `official_useful_event_gain_over_best_baseline`: `0.0`
- `official_score_gain_over_best_baseline`: `0.001`
- `official_invalid_action_rate`: `0.0`
- `official_repeat_collapse`: `0.22784841859240573`
- `best_existing_baseline_useful_events`: `0.04`
- `best_existing_baseline_score`: `0.004`
- `repeat_collapse_ok`: `True`
- `ablation_supports_if_needed`: `True`
- `passes`: `False`

## Aggregate Table
| Variant | Score | Useful | Repeat | Invalid |
| --- | ---: | ---: | ---: | ---: |
| `ablation_no_change_memory` | 0.000000 | 0.000000 | 0.188627 | 0.000000 |
| `ablation_no_component_memory` | 0.005000 | 0.040000 | 0.167517 | 0.000000 |
| `ablation_no_event_memory` | 0.000000 | 0.000000 | 0.141545 | 0.000000 |
| `change_memory_search` | 0.000000 | 0.000000 | 0.194595 | 0.000000 |
| `combined_affordance_search` | 0.000000 | 0.000000 | 0.159579 | 0.000000 |
| `component_click_search` | 0.000000 | 0.000000 | 0.225243 | 0.000000 |
| `coverage_graph_exploration` | 0.004000 | 0.040000 | 0.074917 | 0.000000 |
| `event_linked_ranking` | 0.000000 | 0.000000 | 0.236519 | 0.000000 |
| `greedy_observable_score_delta` | 0.004000 | 0.040000 | 0.863198 | 0.000000 |
| `novelty_first` | 0.004000 | 0.040000 | 0.249335 | 0.000000 |
| `object_persistence_search` | 0.000000 | 0.000000 | 0.219784 | 0.000000 |
| `old_explorer` | 0.004000 | 0.040000 | 0.857922 | 0.000000 |
| `oracle_free_observed_graph_bfs` | 0.004000 | 0.040000 | 0.063967 | 0.000000 |
| `random_legal` | 0.000000 | 0.000000 | 0.092644 | 0.000000 |
| `repeat_last_action` | 0.000000 | 0.000000 | 0.999344 | 0.000000 |
| `state_graph_affordance` | 0.005000 | 0.040000 | 0.227848 | 0.000000 |

## No-Hack Proof
```json
{
  "passes": true,
  "findings": [],
  "no_neural_training": true,
  "no_model_tuning": true,
  "no_external_data": true,
  "no_official_source_inspection": true,
  "no_game_specific_branches": true,
  "no_forced_cycle": true,
  "no_external_judge": true,
  "selection_protocol": "all non-neural variants and ablations are declared before official evaluation",
  "allowed_input_surface": "public observation grid, legal actions, public reward/score/event deltas, terminal flags"
}
```

## Bridge Assessment
Evidence is mixed: some generic signal moved, but not enough to clear both useful-event and score gates.

## Limitations
- Official ARC runtime is serialized and dominates wall-clock time.
- The baseline is intentionally non-neural and cannot learn reusable latent abstractions during evaluation.
- No tested generic hand-built affordance baseline cleared both official useful-event and score gates.
- Useful-event gain over the best existing baseline was below the required 0.08.
- Score gain over the best existing baseline was below the required 0.005.
