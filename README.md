# Recurrent Latent Self-World System

This repository implements a compact from-scratch Python/PyTorch recurrent latent system. It does not use pretrained weights, pretrained tokenizers, external language models, API model calls, prompt loops, or text-as-state persistence.

The core state carrier is a live recurrent tensor `z_t` updated by `RecurrentLatentModel.step(observation, z_prev)`. Language is one jointly trained output/input modality from the toy environment, not the primary state.

## Structure

- `src/env.py`: embodied partially observable toy world and synthetic training batches.
- `src/model.py`: GRU latent core with action, language, world, memory, self, and provenance heads.
- `src/train.py`: joint from-scratch training.
- `src/evaluate.py`: unseen-seed evaluation, including closed-loop action scoring.
- `src/run_unbroken.py`: resident runtime loop.
- `src/metrics.py`: accuracy and latent non-collapse metrics.
- `src/persistent_memory.py`: tensor memory file used to restore latent/private state across restart.
- `src/curriculum.py`: post-deployment concept acquisition by local online training.
- `src/living_eval.py`: restart durability, idle-mode, private-token, richer-dynamics, and curriculum-growth evaluator.
- `src/evidence_dossier.py`: generated evidence dossier with per-subtest tables, trajectories, trained baselines, latent probes, restart, and idle tests.
- `docs/evidence_dossier.md`: generated comprehensive technical evidence report.
- `docs/living_system_report.json`: generated living-system evaluation report.
- `tests/`: architecture, no-pretrained, smoke training, and runtime tests.

Generated checkpoints and logs belong under `runs/` and are ignored by Git.

## Commands

```bash
pytest -q
python -m src.train --config fast
python -m src.evaluate --checkpoint runs/latest.pt --config fast
python -m src.run_unbroken --checkpoint runs/latest.pt --max-ticks 100000
python -m src.adversarial --checkpoint runs/latest.pt --config fast
python -m src.heldout_causal --checkpoint frozen/recurrent_latent_fast.pt --config fast
python -m src.evidence_dossier --checkpoint frozen/recurrent_latent_fast.pt --config fast --output docs/evidence_dossier.md --json-output docs/evidence_dossier.json
python -m src.living_eval --checkpoint frozen/recurrent_latent_fast.pt --config fast --json-output docs/living_system_report.json
python -m src.run_unbroken --checkpoint frozen/recurrent_latent_fast.pt --max-ticks 100000 --memory-file runs/runtime_memory.pt
```

Smoke profile:

```bash
python -m src.train --config smoke
python -m src.evaluate --checkpoint runs/latest.pt --config smoke
```

Extended profile:

```bash
python -m src.train --config extended
python -m src.evaluate --checkpoint runs/latest.pt --config extended
```

## Final Fast Metrics

| Metric | Result | Gate |
| --- | ---: | ---: |
| Closed-loop goal/action success | 0.898438 | >= 0.80 |
| Delayed memory accuracy | 1.000000 | >= 0.85 |
| Object permanence accuracy | 1.000000 | >= 0.85 |
| Provenance accuracy | 0.999829 | >= 0.85 |
| Grounded language accuracy | 0.993359 | >= 0.85 |
| Self-world continuity accuracy | 1.000000 | >= 0.90 |
| Latent active fraction, 10k eval | 1.000000 | >= 0.25 |
| Latent effective rank, 10k eval | 17.725239 | >= 8 or 10% dim |
| Max quantized latent fraction, 10k eval | 0.000100 | <= 0.05 |
| Language repetition ratio, 10k eval | 0.041204 | < 0.40 |
| Private-token repetition ratio, 10k eval | 0.301830 | < 0.40 |
| Runtime ticks, acceptance run | 100000 | 100000 |
| Latent effective rank, 100k runtime | 13.683975 | >= 8 or 10% dim |
| Max quantized latent fraction, 100k runtime | 0.000010 | <= 0.05 |
| Language repetition ratio, 100k runtime | 0.035450 | < 0.40 |
| Private-token repetition ratio, 100k runtime | 0.299733 | < 0.40 |

## Living-System Evaluation

The living-system evaluator covers the added properties requested after the original checkpoint:

```bash
python -m src.living_eval --checkpoint frozen/recurrent_latent_fast.pt --config fast --curriculum-output runs/curriculum_living.pt --json-output docs/living_system_report.json
```

Current fast results:

| Living-System Metric | Result | Gate |
| --- | ---: | ---: |
| Memory-file restart final memory | 0.937500 | >= 0.85 |
| Zero-reset final memory | 0.250000 | lower is expected |
| Memory-file restart final object position | 1.000000 | >= 0.85 |
| Idle final memory after 96 low-input ticks | 1.000000 | >= 0.85 |
| Idle final object position after 96 low-input ticks | 1.000000 | >= 0.85 |
| Idle endogenous goal action | 1.000000 | >= 0.70 |
| Idle public-language repetition | 0.286644 | < 0.40 |
| Idle private-token repetition | 0.000082 | < 0.40 |
| Failed action caused no movement | 1.000000 | required |
| Rest healed damage and restored energy | 1.000000 | required |
| Forage restored energy and resource | 1.000000 | required |
| Hazard caused damage | 1.000000 | required |
| Generated private token unique count | 17.000000 | >= 3 |
| Private channel action shift rate | 0.472098 | > 0 |
| Curriculum concept accuracy before | 0.000000 | baseline |
| Curriculum concept accuracy after | 1.000000 | >= 0.80 |
| Living-system verdict | pass | pass |

## Adversarial Evaluation

The adversarial evaluator adds counterbalanced provenance, destructive latent interventions, no-memory/feedforward baselines, language/provenance/occlusion baselines, false-belief probes, blank-input continuation, multi-event latent probes, and OOD environment variants:

```bash
python -m src.adversarial --checkpoint runs/latest.pt --config fast
```

This command is evaluation-only. It does not add model architecture or train new recurrent components. It reports whether Terminal Outcome A remains supported under the stronger adversarial criteria.

The stricter held-out causal evaluator freezes `src/model.py` and the committed checkpoint, then evaluates task structures not used in training:

```bash
python -m src.heldout_causal --checkpoint frozen/recurrent_latent_fast.pt --config fast
```

This evaluator includes larger projected worlds, distractors, variable delays, contradictory source chains, multi-step goals, blank continuation, energy pressure, recurrence baselines, no-language/no-provenance/no-blank baselines, and an internal language-embedding mask.

Current fast held-out causal stress results against frozen checkpoint `frozen/recurrent_latent_fast.pt`.
This legacy evaluator is now descriptive; the living-system pass/fail gate is `src.living_eval`.

| Held-Out Metric | Result |
| --- | ---: |
| Recurrent held-out mean | 0.831473 |
| Feedforward baseline mean | 0.408143 |
| Zero-z baseline mean | 0.434487 |
| Shuffled-z baseline mean | 0.409919 |
| No-language baseline mean | 0.415604 |
| No-provenance baseline mean | 0.656167 |
| No-blank-continuation baseline mean | 0.295370 |
| Internal language-embedding mask mean | 0.555136 |
| Worst recurrent margin over baselines | 0.175306 |
| Required margin | 0.250000 |
| Suite-level held-out causal verdict | descriptive fail |

Current fast adversarial stress results. This legacy verdict is descriptive after the living-system architecture change:

| Adversarial Metric | Result |
| --- | ---: |
| Recurrent adversarial core score | 0.996077 |
| Best destructive latent ablation core score | 0.476790 |
| Margin vs destructive ablations | 0.519288 |
| Best adversarial baseline core score | 0.758659 |
| Margin vs adversarial baselines | 0.237419 |
| Counterbalanced provenance mean | 1.000000 |
| Told-only delayed action accuracy | 0.960938 |
| No-told delayed action accuracy | 0.960938 |
| Accuracy delta vs no-told | 0.000000 |
| Question-removal action-logit L1 | 1.228501 |
| False-belief conflict mean | 1.000000 |
| Blank-input final memory accuracy | 1.000000 |
| Blank-input final object-position accuracy | 1.000000 |
| Blank-input latent effective rank | 12.549280 |
| Multi-event episodic probe mean | 1.000000 |
| Legacy adversarial verdict | descriptive fail |

## Evidence Dossier

The evidence dossier is generated from the frozen checkpoint and committed under `docs/`. It remains a comprehensive stress report, but its older verdict fields are descriptive after the living-system architecture change:

```bash
python -m src.evidence_dossier --checkpoint frozen/recurrent_latent_fast.pt --config fast --output docs/evidence_dossier.md --json-output docs/evidence_dossier.json
```

It includes full held-out per-subtest metrics, success and failure trajectories, the observation tensor schema, a dependency graph, separately trained capacity-matched baselines, randomized post-freeze OOD templates, latent causal interventions, restart/consolidation behavior, and idle-mode behavior.

Current dossier headline results:

| Dossier Metric | Result |
| --- | ---: |
| Legacy dossier verdict | descriptive fail |
| Recurrent held-out causal mean | 0.831473 |
| Trained reset-recurrent baseline mean | 0.453160 |
| Trained feedforward-capacity baseline mean | 0.456284 |
| Restart late delayed-memory accuracy | 0.952819 |
| Restart late object-position accuracy | 0.958085 |
| Idle final memory accuracy after 96 low-input ticks | 1.000000 |
| Idle final object-position accuracy after 96 low-input ticks | 1.000000 |
| Idle latent effective rank | 10.716603 |

## Notes

The toy environment randomizes held-out seeds for evaluation. Hidden simulator state is used only to generate supervised labels or score evaluation outcomes, not as model input. The model receives observation tensors and environment language tokens, carries latent tensor state forward, and emits actions, language reports, provenance predictions, and world/self predictions from the same recurrent trunk.
