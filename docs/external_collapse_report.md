# External Collapse Experiment

Terminal outcome: **NO MINIMAL IMPROVEMENT FOUND**

Answer: **no**

## Device
- Requested device: `cuda`
- Resolved device: `cuda`
- Torch: `2.11.0.dev20260120+cu128`
- CUDA available: `True`
- CUDA runtime: `12.8`
- CUDA devices: `NVIDIA GeForce RTX 5060 Ti, NVIDIA GeForce RTX 5060 Ti`
- Suite `gymnasium_classic_control` resolved device: `cuda`
- Suite `gymnasium_toy_text` resolved device: `cuda`
- Suite `official_arcagi3` resolved device: `cuda`

## Improvement Gate
- `selected_variant`: `valence_only`
- `official_repeat_drop`: `0.2898760892775156`
- `official_score_gain`: `0.005000000000000001`
- `non_arc_best_score_drop`: `0.0`
- `passes`: `False`

## Scores
| Suite | Variant | Mean Normalized | Solve Rate | Useful Events | Invalid |
| --- | --- | ---: | ---: | ---: | ---: |
| official_arcagi3 | ablation_corrupt_drive | 0.005 | 0.000 | 0.040 | 0.000 |
| official_arcagi3 | ablation_corrupt_memory | 0.005 | 0.000 | 0.040 | 0.000 |
| official_arcagi3 | ablation_zero_z | 0.005 | 0.000 | 0.040 | 0.000 |
| official_arcagi3 | affordance_only | 0.000 | 0.000 | 0.000 | 0.000 |
| official_arcagi3 | baseline_unchanged | 0.004 | 0.000 | 0.040 | 0.000 |
| official_arcagi3 | loop_aversion_only | 0.009 | 0.000 | 0.080 | 0.000 |
| official_arcagi3 | null_patch_control | 0.004 | 0.000 | 0.040 | 0.000 |
| official_arcagi3 | valence_affordance | 0.005 | 0.000 | 0.040 | 0.000 |
| official_arcagi3 | valence_affordance_loop | 0.005 | 0.000 | 0.040 | 0.000 |
| official_arcagi3 | valence_only | 0.009 | 0.000 | 0.080 | 0.000 |
| gymnasium_classic_control | affordance_only | 0.117 | 0.000 | 9.333 | 0.000 |
| gymnasium_classic_control | baseline_unchanged | 0.117 | 0.000 | 9.333 | 0.000 |
| gymnasium_classic_control | loop_aversion_only | 0.117 | 0.000 | 9.333 | 0.000 |
| gymnasium_classic_control | null_patch_control | 0.117 | 0.000 | 9.333 | 0.000 |
| gymnasium_classic_control | valence_affordance | 0.117 | 0.000 | 9.333 | 0.000 |
| gymnasium_classic_control | valence_affordance_loop | 0.117 | 0.000 | 9.333 | 0.000 |
| gymnasium_classic_control | valence_only | 0.117 | 0.000 | 9.333 | 0.000 |
| gymnasium_toy_text | affordance_only | 0.000 | 0.000 | 0.000 | 0.000 |
| gymnasium_toy_text | baseline_unchanged | 0.000 | 0.000 | 0.000 | 0.000 |
| gymnasium_toy_text | loop_aversion_only | 0.000 | 0.000 | 0.000 | 0.000 |
| gymnasium_toy_text | null_patch_control | 0.000 | 0.000 | 0.000 | 0.000 |
| gymnasium_toy_text | valence_affordance | 0.000 | 0.000 | 0.000 | 0.000 |
| gymnasium_toy_text | valence_affordance_loop | 0.000 | 0.000 | 0.000 | 0.000 |
| gymnasium_toy_text | valence_only | 0.000 | 0.000 | 0.000 | 0.000 |

## Collapse
| Suite | Variant | Repeat Collapse | Entropy | Unique States |
| --- | --- | ---: | ---: | ---: |
| official_arcagi3 | ablation_corrupt_drive | 0.561 | 1.810 | 97.360 |
| official_arcagi3 | ablation_corrupt_memory | 0.561 | 1.810 | 97.360 |
| official_arcagi3 | ablation_zero_z | 0.561 | 1.810 | 97.360 |
| official_arcagi3 | affordance_only | 0.875 | 0.647 | 104.880 |
| official_arcagi3 | baseline_unchanged | 0.858 | 0.749 | 111.400 |
| official_arcagi3 | loop_aversion_only | 0.513 | 2.586 | 109.120 |
| official_arcagi3 | null_patch_control | 0.858 | 0.749 | 111.400 |
| official_arcagi3 | valence_affordance | 0.490 | 2.710 | 97.560 |
| official_arcagi3 | valence_affordance_loop | 0.429 | 3.188 | 97.560 |
| official_arcagi3 | valence_only | 0.568 | 2.529 | 104.720 |
| gymnasium_classic_control | affordance_only | 1.000 | 0.000 | 10.333 |
| gymnasium_classic_control | baseline_unchanged | 1.000 | 0.000 | 10.333 |
| gymnasium_classic_control | loop_aversion_only | 1.000 | 0.000 | 10.333 |
| gymnasium_classic_control | null_patch_control | 1.000 | 0.000 | 10.333 |
| gymnasium_classic_control | valence_affordance | 1.000 | 0.000 | 10.333 |
| gymnasium_classic_control | valence_affordance_loop | 1.000 | 0.000 | 10.333 |
| gymnasium_classic_control | valence_only | 1.000 | 0.000 | 10.333 |
| gymnasium_toy_text | affordance_only | 1.000 | 0.000 | 4.000 |
| gymnasium_toy_text | baseline_unchanged | 1.000 | 0.000 | 4.000 |
| gymnasium_toy_text | loop_aversion_only | 1.000 | 0.000 | 4.000 |
| gymnasium_toy_text | null_patch_control | 1.000 | 0.000 | 4.000 |
| gymnasium_toy_text | valence_affordance | 1.000 | 0.000 | 4.000 |
| gymnasium_toy_text | valence_affordance_loop | 1.000 | 0.000 | 4.000 |
| gymnasium_toy_text | valence_only | 1.000 | 0.000 | 4.000 |

## Trace Paths
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\ar25-0c556536.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\bp35-0a0ad940.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\cd82-fb555c5d.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\cn04-2fe56bfb.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\dc22-fdcac232.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\ft09-0d8bbf25.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\g50t-5849a774.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\ka59-38d34dbb.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\lf52-271a04aa.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\lp85-305b61c3.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\ls20-9607627b.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\m0r0-492f87ba.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\r11l-495a7899.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\re86-8af5384d.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\s5i5-18d95033.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\sb26-7fbdac44.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\sc25-635fd71a.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\sk48-d8078629.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\sp80-589a99af.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\su15-1944f8ab.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\tn36-ef4dde99.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\tr87-cd924810.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\tu93-0768757b.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\vc33-5430563c.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\baseline_unchanged\wa30-ee6fef47.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\ar25-0c556536.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\bp35-0a0ad940.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\cd82-fb555c5d.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\cn04-2fe56bfb.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\dc22-fdcac232.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\ft09-0d8bbf25.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\g50t-5849a774.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\ka59-38d34dbb.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\lf52-271a04aa.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\lp85-305b61c3.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\ls20-9607627b.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\m0r0-492f87ba.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\r11l-495a7899.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\re86-8af5384d.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\s5i5-18d95033.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\sb26-7fbdac44.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\sc25-635fd71a.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\sk48-d8078629.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\sp80-589a99af.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\su15-1944f8ab.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\tn36-ef4dde99.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\tr87-cd924810.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\tu93-0768757b.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\vc33-5430563c.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\valence_only\wa30-ee6fef47.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\affordance_only\ar25-0c556536.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\affordance_only\bp35-0a0ad940.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\affordance_only\cd82-fb555c5d.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\affordance_only\cn04-2fe56bfb.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\affordance_only\dc22-fdcac232.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\affordance_only\ft09-0d8bbf25.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\affordance_only\g50t-5849a774.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\affordance_only\ka59-38d34dbb.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\affordance_only\lf52-271a04aa.json`
- `docs\external_collapse_traces\official_arcagi3\sealed_eval\affordance_only\lp85-305b61c3.json`
- ... 260 more traces

## Limitations
- Variants are generic runtime calibrations of model action scores; no weights are trained.
- Baselines are comparison policies only and are not used to choose variant actions.
- No tested minimal generic change satisfied the official ARC score and collapse improvement gate.
