# External Base Retrain Report

Terminal outcome: **NO IMPROVEMENT FOUND**
Answer: **no**
Selected arm: `old_base_finetuned`

## Gate
- `selected_variant`: `old_base_finetuned`
- `official_score_gain_over_old_base`: `-0.004`
- `official_score_gain_over_best_baseline`: `-0.004`
- `official_useful_event_gain_over_old_base`: `-0.04`
- `official_repeat_collapse_drop`: `0.02462735707291852`
- `official_invalid_action_rate`: `0.0`
- `non_arc_best_score_drop`: `0.0`
- `best_existing_baseline_score`: `0.004`
- `solved_or_score_gain_clause`: `False`
- `passes`: `False`

## CUDA
- Requested device: `cuda`
- Resolved device: `cuda`
- CUDA available: `True`
- CUDA runtime: `12.8`

## Sealed Scores
| Suite | Variant | Score | Useful | Repeat | Invalid |
| --- | --- | ---: | ---: | ---: | ---: |
| `official_arcagi3` | `ablation_no_affordance` | 0.000000 | 0.000000 | 0.809339 | 0.000000 |
| `official_arcagi3` | `ablation_no_external_base` | 0.004000 | 0.040000 | 0.857922 | 0.000000 |
| `official_arcagi3` | `ablation_no_memory` | 0.000000 | 0.000000 | 0.834043 | 0.000000 |
| `official_arcagi3` | `ablation_no_world_model` | 0.000000 | 0.000000 | 0.830783 | 0.000000 |
| `official_arcagi3` | `from_scratch_external_base` | 0.000000 | 0.000000 | 0.863306 | 0.000000 |
| `official_arcagi3` | `null_training_control` | 0.004000 | 0.040000 | 0.857922 | 0.000000 |
| `official_arcagi3` | `old_base_finetuned` | 0.000000 | 0.000000 | 0.833294 | 0.000000 |
| `official_arcagi3` | `old_base_unchanged` | 0.004000 | 0.040000 | 0.857922 | 0.000000 |
| `official_arcagi3` | `old_base_world_model` | 0.000000 | 0.000000 | 0.857106 | 0.000000 |
| `gymnasium_classic_control` | `from_scratch_external_base` | 0.116667 | 9.333333 | 1.000000 | 0.000000 |
| `gymnasium_classic_control` | `null_training_control` | 0.116667 | 9.333333 | 1.000000 | 0.000000 |
| `gymnasium_classic_control` | `old_base_finetuned` | 0.116667 | 9.333333 | 1.000000 | 0.000000 |
| `gymnasium_classic_control` | `old_base_unchanged` | 0.116667 | 9.333333 | 1.000000 | 0.000000 |
| `gymnasium_classic_control` | `old_base_world_model` | 0.116667 | 9.333333 | 1.000000 | 0.000000 |
| `gymnasium_toy_text` | `from_scratch_external_base` | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| `gymnasium_toy_text` | `null_training_control` | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| `gymnasium_toy_text` | `old_base_finetuned` | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| `gymnasium_toy_text` | `old_base_unchanged` | 0.000000 | 0.000000 | 1.000000 | 0.000000 |
| `gymnasium_toy_text` | `old_base_world_model` | 0.000000 | 0.000000 | 1.000000 | 0.000000 |

## Trace Manifest
- Trace transitions: `1000763`
- Data hash: `AE718CF6FC7E725BF841D659C237D92F47224DEDD6FF8B254F1D15AA45D0BAB5`

## Ablations
```json
{
  "present": true,
  "selected_variant": "old_base_finetuned",
  "score_drop_vs_best_ablation": -0.004,
  "useful_event_drop_vs_best_ablation": -0.04,
  "supports_if_needed": false,
  "rows": {
    "ablation_no_affordance": {
      "solve_rate": 0.0,
      "mean_score": 0.0,
      "mean_normalized_score": 0.0,
      "mean_steps": 103.92,
      "mean_invalid_action_rate": 0.0,
      "mean_unique_states": 104.92,
      "mean_useful_events": 0.0,
      "mean_action_entropy": 0.7982650359524766,
      "mean_repeat_collapse": 0.8093393032780399
    },
    "ablation_no_external_base": {
      "solve_rate": 0.0,
      "mean_score": 0.004,
      "mean_normalized_score": 0.004,
      "mean_steps": 110.4,
      "mean_invalid_action_rate": 0.0,
      "mean_unique_states": 111.4,
      "mean_useful_events": 0.04,
      "mean_action_entropy": 0.7487791837206761,
      "mean_repeat_collapse": 0.8579215341806236
    },
    "ablation_no_memory": {
      "solve_rate": 0.0,
      "mean_score": 0.0,
      "mean_normalized_score": 0.0,
      "mean_steps": 102.6,
      "mean_invalid_action_rate": 0.0,
      "mean_unique_states": 103.6,
      "mean_useful_events": 0.0,
      "mean_action_entropy": 0.7002462731243316,
      "mean_repeat_collapse": 0.8340425910062989
    },
    "ablation_no_world_model": {
      "solve_rate": 0.0,
      "mean_score": 0.0,
      "mean_normalized_score": 0.0,
      "mean_steps": 102.6,
      "mean_invalid_action_rate": 0.0,
      "mean_unique_states": 103.6,
      "mean_useful_events": 0.0,
      "mean_action_entropy": 0.7336417028918827,
      "mean_repeat_collapse": 0.830782928050921
    }
  }
}
```

## Unsupported Arms
```json
[
  {
    "arm": "old_base_world_model",
    "selected_by_non_arc_dev": false,
    "official_score_gain": -0.004,
    "official_useful_event_gain": -0.04,
    "official_repeat_drop": 0.0008152643907409773,
    "unsupported_reasons": [
      "official score gain below 0.01",
      "official useful-event gain below 0.08",
      "repeat-collapse drop below 0.20"
    ]
  },
  {
    "arm": "old_base_finetuned",
    "selected_by_non_arc_dev": true,
    "official_score_gain": -0.004,
    "official_useful_event_gain": -0.04,
    "official_repeat_drop": 0.02462735707291852,
    "unsupported_reasons": [
      "official score gain below 0.01",
      "official useful-event gain below 0.08",
      "repeat-collapse drop below 0.20"
    ]
  },
  {
    "arm": "from_scratch_external_base",
    "selected_by_non_arc_dev": false,
    "official_score_gain": -0.004,
    "official_useful_event_gain": -0.04,
    "official_repeat_drop": -0.00538425097451023,
    "unsupported_reasons": [
      "official score gain below 0.01",
      "official useful-event gain below 0.08",
      "repeat-collapse drop below 0.20"
    ]
  },
  {
    "arm": "null_training_control",
    "selected_by_non_arc_dev": false,
    "official_score_gain": 0.0,
    "official_useful_event_gain": 0.0,
    "official_repeat_drop": 0.0,
    "unsupported_reasons": [
      "official score gain below 0.01",
      "official useful-event gain below 0.08",
      "repeat-collapse drop below 0.20",
      "null training control is not eligible for improvement claim"
    ]
  }
]
```

## No-Hack Proof
```json
{
  "passes": true,
  "findings": [],
  "model_action_source": "old ARCAGI3Adapter action_scores optionally rescored by frozen external base predictions",
  "selection_protocol": "primary arm selected from non-ARC dev before official sealed evaluation",
  "no_forced_cycle": true,
  "no_external_judge": true,
  "no_game_id_branch": true,
  "no_hidden_labels": true,
  "no_public_text_as_state": true
}
```

## Limitations
- No external-base arm passed the sealed official ARC score/useful/repeat gate.
- Generated ARC-like traces are pretraining data only and are not terminal proof.
- The first-pass dataset is dominated by generated pretraining transitions; sealed evaluation remains the proof source.
- Official ARC runtime remains serialized and slow even when model inference is CUDA-backed.
