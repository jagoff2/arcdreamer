# Repository Guidelines

## Source of Truth

`GOAL.md` is authoritative. Treat it as the acceptance contract for architecture, implementation order, metrics, and completion format. Do not replace it with a planning document, README-first workflow, chatbot wrapper, prompt loop, or dashboard. When instructions conflict, follow `GOAL.md`.

## Project Structure & Module Organization

Implement the project as a compact Python/PyTorch package. Required runtime code belongs in `src/`:

- `src/env.py`: embodied, partially observable toy environment.
- `src/model.py`: recurrent latent model with `step(observation, z_prev)`.
- `src/train.py`: from-scratch joint training entrypoint.
- `src/evaluate.py`: unseen-seed evaluation.
- `src/metrics.py`: metric calculations and collapse checks.
- `src/run_unbroken.py`: resident runtime loop.

Tests belong in `tests/`. Store generated checkpoints and logs under `runs/`, which should not be committed unless a small fixture is explicitly required.

## Build, Test, and Development Commands

Use these commands as the stable interface:

```bash
python -m src.train --config fast
python -m src.evaluate --checkpoint runs/latest.pt --config fast
pytest -q
python -m src.run_unbroken --checkpoint runs/latest.pt --max-ticks 100000
```

`train` must initialize all weights locally. `evaluate` must use unseen seeds and no hidden simulator state leakage. `run_unbroken` must keep one live process and recurrent latent state until external interruption or `max_ticks`.

## Coding Style & Naming Conventions

Use clear Python modules, type hints where they clarify interfaces, and small functions with explicit tensor shapes when practical. Prefer deterministic seeds for tests and smoke runs. Use snake_case for functions, variables, files, and config keys. Avoid hand-coded cognitive controllers, symbolic memory databases, scripted policies, or hardcoded answers.

## Testing Guidelines

Tests must enforce architecture gates as well as behavior. Required files include `tests/test_no_pretrained.py`, `tests/test_architecture.py`, `tests/test_training_smoke.py`, and `tests/test_unbroken_runtime.py`. Name tests by the invariant they protect, for example `test_generated_language_is_not_state_carrier`.

## Commit & Pull Request Guidelines

The current history starts with `Initialize repository`. Continue with short imperative commits, for example `Add recurrent model` or `Implement unseen-seed evaluation`. Keep commits atomic: code, tests, and docs for one coherent change. Pull requests should include commands run, metric results, files changed, and any limitations that do not invalidate `GOAL.md`.

## Security & Configuration

Do not call external LLM APIs, download pretrained weights, or depend on pretrained tokenizers. Keep secrets out of the repository. Generated artifacts, large checkpoints, caches, and local environment files should remain untracked.
