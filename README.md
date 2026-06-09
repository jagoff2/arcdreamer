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
- `tests/`: architecture, no-pretrained, smoke training, and runtime tests.

Generated checkpoints and logs belong under `runs/` and are ignored by Git.

## Commands

```bash
pytest -q
python -m src.train --config fast
python -m src.evaluate --checkpoint runs/latest.pt --config fast
python -m src.run_unbroken --checkpoint runs/latest.pt --max-ticks 100000
python -m src.adversarial --checkpoint runs/latest.pt --config fast
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
| Closed-loop goal/action success | 1.000000 | >= 0.80 |
| Delayed memory accuracy | 1.000000 | >= 0.85 |
| Object permanence accuracy | 0.999974 | >= 0.85 |
| Provenance accuracy | 0.997485 | >= 0.85 |
| Grounded language accuracy | 0.999805 | >= 0.85 |
| Self-world continuity accuracy | 1.000000 | >= 0.90 |
| Latent active fraction, 10k eval | 1.000000 | >= 0.25 |
| Latent effective rank, 10k eval | 15.485283 | >= 8 or 10% dim |
| Max quantized latent fraction, 10k eval | 0.000100 | <= 0.05 |
| Language repetition ratio, 10k eval | 0.019602 | < 0.40 |
| Runtime ticks, acceptance run | 100000 | 100000 |
| Latent effective rank, 100k runtime | 15.631958 | >= 8 or 10% dim |
| Max quantized latent fraction, 100k runtime | 0.000010 | <= 0.05 |
| Language repetition ratio, 100k runtime | 0.032880 | < 0.40 |

## Adversarial Evaluation

The adversarial evaluator adds counterbalanced provenance, destructive latent interventions, no-memory/feedforward baselines, language/provenance/occlusion baselines, false-belief probes, blank-input continuation, multi-event latent probes, and OOD environment variants:

```bash
python -m src.adversarial --checkpoint runs/latest.pt --config fast
```

This command is evaluation-only. It does not add model architecture or train new recurrent components. It reports whether Terminal Outcome A remains supported under the stronger adversarial criteria.

Current fast adversarial results:

| Adversarial Metric | Result |
| --- | ---: |
| Recurrent adversarial core score | 0.998494 |
| Best destructive latent ablation core score | 0.386717 |
| Margin vs destructive ablations | 0.611778 |
| Best adversarial baseline core score | 0.452062 |
| Margin vs adversarial baselines | 0.546432 |
| Counterbalanced provenance mean | 0.999219 |
| Told-only delayed action accuracy | 0.996094 |
| No-told delayed action accuracy | 0.355469 |
| Accuracy delta vs no-told | 0.640625 |
| Question-removal action-logit L1 | 0.705947 |
| False-belief conflict mean | 1.000000 |
| Blank-input final memory accuracy | 1.000000 |
| Blank-input final object-position accuracy | 1.000000 |
| Blank-input latent effective rank | 9.006064 |
| Multi-event episodic probe mean | 1.000000 |
| Terminal A adversarial verdict | pass |

## Notes

The toy environment randomizes held-out seeds for evaluation. Hidden simulator state is used only to generate supervised labels or score evaluation outcomes, not as model input. The model receives observation tensors and environment language tokens, carries latent tensor state forward, and emits actions, language reports, provenance predictions, and world/self predictions from the same recurrent trunk.
