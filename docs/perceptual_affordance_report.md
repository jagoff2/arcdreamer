# Perceptual Affordance Experiment

Terminal outcome: **NO IMPROVEMENT FOUND**

Answer: **no**

## Device
- Requested device: `cuda`
- Resolved device: `cuda`
- Torch: `2.11.0.dev20260120+cu128`
- CUDA available: `True`
- CUDA runtime: `12.8`
- CUDA devices: `NVIDIA GeForce RTX 5060 Ti, NVIDIA GeForce RTX 5060 Ti`

## Improvement Gate
- `selected_variant`: `full_perceptual_affordance`
- `official_score_gain_over_unchanged`: `-0.004`
- `official_score_gain_over_best_existing_baseline`: `-0.004`
- `official_useful_event_gain`: `-0.04`
- `official_repeat_collapse_delta`: `-0.21844845425513992`
- `official_invalid_action_rate`: `0.0`
- `non_arc_best_score_drop`: `0.0`
- `best_existing_baseline_score`: `0.004`
- `passes`: `False`

## Scores
| Suite | Variant | Mean Normalized | Useful Events | Repeat Collapse | Invalid |
| --- | --- | ---: | ---: | ---: | ---: |
| official_arcagi3 | ablation_no_components | 0.000 | 0.000 | 0.696 | 0.000 |
| official_arcagi3 | ablation_no_effect_memory | 0.000 | 0.000 | 0.666 | 0.000 |
| official_arcagi3 | ablation_no_prediction | 0.000 | 0.000 | 0.646 | 0.000 |
| official_arcagi3 | ablation_no_temporal_slots | 0.000 | 0.000 | 0.700 | 0.000 |
| official_arcagi3 | action_effect_memory | 0.004 | 0.040 | 0.693 | 0.000 |
| official_arcagi3 | baseline_unchanged | 0.004 | 0.040 | 0.858 | 0.000 |
| official_arcagi3 | component_only | 0.000 | 0.000 | 0.704 | 0.000 |
| official_arcagi3 | full_perceptual_affordance | 0.000 | 0.000 | 0.639 | 0.000 |
| official_arcagi3 | null_patch_control | 0.004 | 0.040 | 0.858 | 0.000 |
| official_arcagi3 | predictive_object_model | 0.004 | 0.040 | 0.694 | 0.000 |
| official_arcagi3 | temporal_slots | 0.000 | 0.000 | 0.690 | 0.000 |
| gymnasium_classic_control | action_effect_memory | 0.117 | 9.333 | 1.000 | 0.000 |
| gymnasium_classic_control | baseline_unchanged | 0.117 | 9.333 | 1.000 | 0.000 |
| gymnasium_classic_control | component_only | 0.117 | 9.333 | 1.000 | 0.000 |
| gymnasium_classic_control | full_perceptual_affordance | 0.117 | 9.333 | 1.000 | 0.000 |
| gymnasium_classic_control | null_patch_control | 0.117 | 9.333 | 1.000 | 0.000 |
| gymnasium_classic_control | predictive_object_model | 0.117 | 9.333 | 1.000 | 0.000 |
| gymnasium_classic_control | temporal_slots | 0.117 | 9.333 | 1.000 | 0.000 |
| gymnasium_toy_text | action_effect_memory | 0.000 | 0.000 | 1.000 | 0.000 |
| gymnasium_toy_text | baseline_unchanged | 0.000 | 0.000 | 1.000 | 0.000 |
| gymnasium_toy_text | component_only | 0.000 | 0.000 | 1.000 | 0.000 |
| gymnasium_toy_text | full_perceptual_affordance | 0.000 | 0.000 | 1.000 | 0.000 |
| gymnasium_toy_text | null_patch_control | 0.000 | 0.000 | 1.000 | 0.000 |
| gymnasium_toy_text | predictive_object_model | 0.000 | 0.000 | 1.000 | 0.000 |
| gymnasium_toy_text | temporal_slots | 0.000 | 0.000 | 1.000 | 0.000 |

## Perception Diagnostics
| Variant | Components | Stable Tracks | Effects | Prediction > Null | Perturb Change |
| --- | ---: | ---: | ---: | --- | ---: |
| ablation_no_components | 161.400 | 138.320 | 116 | `True` | 0.315 |
| ablation_no_effect_memory | 161.400 | 140.480 | 186 | `True` | 0.647 |
| ablation_no_prediction | 161.040 | 139.920 | 176 | `True` | 0.628 |
| ablation_no_temporal_slots | 161.480 | 141.240 | 162 | `True` | 0.628 |
| action_effect_memory | 117.457 | 99.629 | 140 | `True` | 0.192 |
| baseline_unchanged | 0.000 | 0.000 | 0 | `False` | 0.000 |
| component_only | 116.400 | 101.971 | 198 | `True` | 0.635 |
| full_perceptual_affordance | 116.029 | 100.800 | 190 | `True` | 0.618 |
| null_patch_control | 116.257 | 101.400 | 156 | `True` | 0.561 |
| predictive_object_model | 116.514 | 100.457 | 142 | `True` | 0.193 |
| temporal_slots | 115.714 | 101.200 | 204 | `True` | 0.619 |

## Trace Paths
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\ar25-0c556536.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\bp35-0a0ad940.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\cd82-fb555c5d.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\cn04-2fe56bfb.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\dc22-fdcac232.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\ft09-0d8bbf25.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\g50t-5849a774.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\ka59-38d34dbb.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\lf52-271a04aa.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\lp85-305b61c3.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\ls20-9607627b.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\m0r0-492f87ba.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\r11l-495a7899.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\re86-8af5384d.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\s5i5-18d95033.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\sb26-7fbdac44.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\sc25-635fd71a.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\sk48-d8078629.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\sp80-589a99af.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\su15-1944f8ab.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\tn36-ef4dde99.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\tr87-cd924810.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\tu93-0768757b.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\vc33-5430563c.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\baseline_unchanged\wa30-ee6fef47.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\ar25-0c556536.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\bp35-0a0ad940.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\cd82-fb555c5d.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\cn04-2fe56bfb.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\dc22-fdcac232.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\ft09-0d8bbf25.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\g50t-5849a774.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\ka59-38d34dbb.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\lf52-271a04aa.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\lp85-305b61c3.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\ls20-9607627b.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\m0r0-492f87ba.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\r11l-495a7899.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\re86-8af5384d.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\s5i5-18d95033.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\sb26-7fbdac44.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\sc25-635fd71a.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\sk48-d8078629.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\sp80-589a99af.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\su15-1944f8ab.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\tn36-ef4dde99.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\tr87-cd924810.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\tu93-0768757b.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\vc33-5430563c.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\component_only\wa30-ee6fef47.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\temporal_slots\ar25-0c556536.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\temporal_slots\bp35-0a0ad940.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\temporal_slots\cd82-fb555c5d.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\temporal_slots\cn04-2fe56bfb.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\temporal_slots\dc22-fdcac232.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\temporal_slots\ft09-0d8bbf25.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\temporal_slots\g50t-5849a774.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\temporal_slots\ka59-38d34dbb.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\temporal_slots\lf52-271a04aa.json`
- `docs\perceptual_affordance_traces\official_arcagi3\sealed_eval\temporal_slots\lp85-305b61c3.json`
- ... 285 more traces

## Limitations
- Perception is online and self-supervised; no model weights are trained or updated.
- Perception diagnostics are mechanistic evidence only; terminal outcome depends on sealed external scores.
- Baselines are comparison policies only and are not used to choose actions.
- No tested generic perceptual-affordance variant satisfied the official score and useful-event gate.
