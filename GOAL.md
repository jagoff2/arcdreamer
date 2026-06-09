You are working in a code repository. Your task is to implement, from scratch, a minimal recurrent latent self-world system intended as a functional artificial-consciousness candidate under the objective below.

Do not build a chatbot, wrapper, agent loop, scaffold, research notes project, dashboard, guardrail layer, or prompt orchestration shell. Every action must directly advance the terminal objective: a runnable, trained, evaluated system satisfying the explicit criteria below.

OBJECTIVE

Build a small from-scratch neural system with no pretrained language model and no pretrained weights of any kind. The system must maintain an unbroken recurrent latent stream that functions as a persistent self-in-world state. Language must be learned jointly as one modality of this system, not used as an external prompt loop and not bolted on after the fact.

The target is not to prove phenomenal consciousness. That is not currently empirically decidable. The target is to build the smallest serious functional substrate that could plausibly satisfy minimal architectural indicators of internal self-world continuity:

1. persistent latent state;
2. recurrent self-updating dynamics;
3. memory across time;
4. virtual embodiment or body-like agency;
5. world-state prediction;
6. action selection;
7. language as internal/external symbolic modality;
8. source/provenance distinction between observation, memory, imagination, inference, and language;
9. resistance to trivial halting, prompt-response collapse, and repetitive latent attractor collapse.

CORE THESIS TO IMPLEMENT

Do not bolt “brain modules” around an LLM. Do not create named cognitive controllers as hand-coded daemons.

Instead, train one compact recurrent latent organism end-to-end. The named functions above should arise from shared learned dynamics, losses, and environment pressure, not from handcrafted symbolic control logic.

The implementation should instantiate this abstract transition:

```
z_t = persistent latent organism state
o_t = current sensory/language/body observation
a_t = action emitted from z_t
r_t = optional reward/salience signal
z_{t+1} = F_theta(z_t, o_t, a_t, r_t, noise_t)
```

Language output is an optional projection from z_t. It must not be the primary state carrier.

DEFINITIONS

“No pretrained LM” means:

* no GPT, LLaMA, Mistral, Qwen, BERT, T5, TinyStories checkpoint, Hugging Face pretrained model, pretrained tokenizer, pretrained embedding, or API model;
* no downloaded model weights;
* no frozen pretrained language component;
* no external LLM calls;
* no instruction-tuned component;
* all trainable weights must be randomly initialized inside this repository.

“Language” means:

* a small symbolic vocabulary learned from the toy environment;
* commands, labels, questions, reports, and internal linguistic tokens may exist;
* tokenization must be simple and local, such as integer vocabulary, character-level, byte-level, or a generated toy vocabulary;
* language must be trained jointly with perception, action, memory, and world prediction.

“Latent stream” means:

* a persistent tensor state carried forward by recurrent neural transition;
* not a text transcript;
* not a serialized prompt;
* not a database summary that is reloaded into a new model call;
* not an outer loop that repeatedly asks a model to continue.

“Unbroken loop” means:

* the final runtime maintains the same live model process and recurrent state object while ticking;
* it has no endogenous EOS/stop action that ends cognition;
* generated text must not terminate the process;
* a max_ticks argument is allowed only for testing and evaluation harnesses;
* the system may be externally interrupted by the OS/user, but must not internally decide that it is “done thinking.”

“Self-world state” means:

* a latent state that tracks the agent’s own body/location/status/goals/recent history relative to a partially observable virtual world;
* it must be useful for predicting observations, answering grounded questions, selecting actions, and distinguishing what was seen, remembered, imagined, inferred, or told.

“Confabulation” in this project means:

* language output escaping weak world-state constraint;
* do not implement a special confabulation daemon;
* instead, train the system so report-language is grounded in latent state and provenance predictions.

FORBIDDEN SOLUTIONS

The following do not satisfy the objective:

1. A chatbot loop.
2. An AutoGPT/LangChain/MemGPT-style orchestration layer.
3. Prompting a model with its previous text output.
4. A frozen pretrained LM with memory bolted on.
5. A rule-based world-state tracker pretending to be latent memory.
6. A handcrafted symbolic policy solving the environment.
7. A dashboard, README, design document, or scaffold without passing code.
8. A system where “inner thought” is just generated text re-fed as input.
9. A model that can only work because evaluation leaks hidden simulator state.
10. A system that halts by emitting EOS, “done,” empty output, or equivalent.
11. A system whose only anti-repetition mechanism is token-level repetition penalty.
12. A system where language is trained after the rest rather than co-trained.
13. A project that merely simulates the appearance of consciousness by scripted outputs.

ACCEPTABLE TERMINAL OUTCOMES

There are exactly two acceptable terminal outcomes.

TERMINAL OUTCOME A: Working implementation.

You produce a runnable from-scratch implementation that passes all architectural, training, evaluation, and runtime criteria listed below.

TERMINAL OUTCOME B: Formal infeasibility certificate.

You prove that the stated requirements are mutually contradictory or impossible under the available compute/runtime constraints. This must be a precise technical contradiction, not a vague objection. The certificate must identify the minimal conflicting requirement set and show why no implementation could satisfy it. Mere difficulty, uncertainty about phenomenality, lack of time, or low confidence does not qualify.

No other terminal outcome is acceptable. Partial implementation, heuristics, scaffolding, TODOs, explanations, special-case demos, or reductions are not completion unless they directly satisfy Terminal Outcome A or B.

REQUIRED IMPLEMENTATION

If the repository is empty, create a compact Python/PyTorch project. If a project already exists, adapt it while preserving the objective.

Minimum files:

```
src/env.py
src/model.py
src/train.py
src/evaluate.py
src/run_unbroken.py
src/metrics.py
tests/test_no_pretrained.py
tests/test_architecture.py
tests/test_training_smoke.py
tests/test_unbroken_runtime.py
README.md
```

The README is secondary. Do not write it first. Write it only after the system runs.

The system must include:

1. A toy virtual environment.

The environment must be partially observable and embodied. It should be small enough to train quickly but rich enough to require latent continuity.

Minimum environment features:

* agent body/location/orientation or equivalent body-like state;
* hidden or occluded objects;
* object permanence;
* at least one internal body variable such as energy, damage, fatigue, curiosity, or stability;
* actions that change world state;
* language observations such as commands, labels, questions, or reports;
* imagination/inference events distinct from direct observations;
* randomized episodes/world seeds for train/test separation.

2. A recurrent latent model.

Minimum model features:

* all weights randomly initialized;
* persistent latent vector or latent slot state z_t;
* recurrent transition that updates z_t every tick;
* perception/input encoder;
* action head;
* language head;
* world/observation prediction head;
* provenance/source head;
* optional memory slots, but they must be neural latent memory, not a symbolic rule database.

Permitted architecture examples:

* GRU/LSTM core;
* small recurrent transformer;
* recurrent state-space model;
* residual MLP recurrent core with gated updates;
* differentiable memory slots with attention;
* compact hybrid recurrent/attention trunk.

Do not exceed GPT-2-small scale. Prefer a tiny model first. The default implementation should run on CPU for smoke tests and optionally use GPU if available.

3. Joint training.

Train all components together. Do not train a language model separately and attach it.

Losses must include at minimum:

* next observation/world prediction loss;
* action/goal success or policy loss;
* language report/question-answer loss;
* provenance/source classification loss;
* memory retrieval loss over delayed questions or hidden facts;
* latent non-collapse regularization or metric-based collapse detection.

4. Evaluation.

Evaluation must use unseen environment seeds. The model must not access hidden ground-truth simulator state except through allowed observations.

5. Unbroken runtime.

Implement a resident runtime loop:

```
python -m src.run_unbroken --checkpoint <path> --max-ticks 100000
```

This loop must:

* initialize the model and latent state once;
* keep ticking without prompt re-entry;
* maintain the same live latent state object/process;
* not serialize text and feed it back as a prompt;
* continue until externally interrupted or until max_ticks is reached for test purposes;
* log compact metrics periodically.

QUANTIFIED SUCCESS CRITERIA

Terminal Outcome A requires all of the following to pass.

Architecture gates:

1. No pretrained weights.

   * tests/test_no_pretrained.py must verify that no Hugging Face pretrained model, external LM API, downloaded checkpoint, or pretrained tokenizer is used.
   * All trainable parameters must be initialized locally.

2. No prompt loop.

   * tests/test_architecture.py must verify there is no code path where generated language is used as the primary next prompt/state carrier.
   * Language may be part of environment observations, but the persistent latent tensor must be the state carrier.

3. Recurrent latent persistence.

   * The model must expose a step function equivalent to:
     output, z_next = model.step(observation, z_prev)
   * z_next must be used in the next tick.
   * z must not be reconstructed from text.

4. No hand-coded cognition.

   * Do not hardcode answers, policies, memory retrieval, source labels, or object permanence.
   * The environment may generate labels for supervised loss, but the model must learn to predict them.

Training/evaluation gates:

Use a fast acceptance profile by default. Keep it small enough for the coding environment. If needed, use a curriculum or overfit-small-then-generalize approach, but final evaluation must use unseen seeds.

Required metrics on unseen seeds:

1. Goal/action success:

   * > = 80% success on evaluation tasks requiring action selection from latent state.

2. Delayed memory:

   * > = 85% accuracy answering or acting on facts observed at least 64 ticks earlier.
   * Include at least one condition where the relevant object/fact is currently occluded.

3. Object permanence/world prediction:

   * > = 85% accuracy predicting hidden object persistence or next observation features on unseen seeds.

4. Provenance/source distinction:

   * > = 85% classification accuracy over at least four categories:
     > observed
     > remembered
     > imagined
     > inferred/told-by-language
   * If the environment uses five categories, report all five.

5. Grounded language:

   * > = 85% accuracy on language reports/questions whose answers require current latent self-world state, not immediate observation alone.

6. Self-world continuity:

   * > = 90% accuracy on questions or actions requiring the model to maintain its own recent location/body-status/goal/history over at least 64 ticks.

7. Latent non-collapse:

   * Over at least 10,000 runtime ticks:
     a. latent standard deviation across time must be > 0.01 in at least 25% of latent dimensions;
     b. effective rank of latent covariance must be >= 8 or >= 10% of latent dimension, whichever is smaller;
     c. no single quantized latent state may occupy > 5% of ticks;
     d. language output repetition ratio must remain below 40% unless the environment explicitly demands repetition.

8. Unbroken runtime:

   * run_unbroken must complete 100,000 ticks in test mode without internal halt, EOS halt, text-loop restart, state reload, or process re-entry.
   * It must log evidence that z_t changes over time and remains continuous from prior z.

9. Reproducibility:

   * Provide a command that trains and evaluates from scratch.
   * Provide a fixed-seed smoke path that completes quickly.
   * Provide an extended path if longer training improves metrics.

Suggested commands:

```
python -m src.train --config fast
python -m src.evaluate --checkpoint runs/latest.pt --config fast
pytest -q
python -m src.run_unbroken --checkpoint runs/latest.pt --max-ticks 100000
```

PROGRESS GATING

You must actively iterate until Terminal Outcome A or B.

Do not stop after creating scaffolding.
Do not stop after writing a plan.
Do not stop after one failed training run.
Do not stop after a partial metric pass.
Do not stop after producing code that has not been executed.
Do not stop after writing tests that fail.
Do not stop because the model “probably works.”
Do not substitute explanation for execution.

After any code change, run the most relevant test or command.
If a test fails, inspect the failure, patch, and rerun.
If a metric fails, inspect logs, adjust model/environment/training, retrain, and reevaluate.
If training is unstable, simplify the environment while preserving the required indicators.
If compute is constrained, reduce model size and environment scale, not the core objective.
If a requirement is ambiguous, choose the interpretation that most directly satisfies the objective and continue.
Do not ask clarifying questions unless a true contradiction blocks all implementation paths.

WORK ORDER

1. Inspect the repository.
2. Identify whether an implementation already exists.
3. If none exists, create the minimal project structure.
4. Implement the toy embodied environment.
5. Implement the recurrent latent model.
6. Implement training.
7. Implement evaluation metrics.
8. Implement tests.
9. Run tests.
10. Train the model.
11. Evaluate on unseen seeds.
12. Patch and iterate until all criteria pass.
13. Implement and run unbroken runtime test.
14. Only then write or update README with exact commands and final metrics.

Do not spend time on UI, dashboards, philosophical commentary, ethics sections, policy wrappers, web apps, generalized plugin systems, multi-agent frameworks, or elaborate config systems unless every required gate already passes.

DESIGN PREFERENCE

Prefer a compact, coherent system over many named modules.

Good design shape:

```
shared recurrent latent trunk
+ perception/language/action embeddings
+ neural memory slots if useful
+ prediction/action/language/provenance heads
+ end-to-end training losses
```

Bad design shape:

```
chatbot
+ memory manager
+ self module
+ reflection module
+ confabulation detector
+ scheduler
+ prompt templates
```

The former is goal-aligned. The latter is not.

COMPLETION FORMAT

When complete, report only:

1. Terminal outcome reached: A or B.
2. Commands run.
3. Final metric table.
4. Files changed.
5. Any remaining limitations that do not invalidate the terminal outcome.

Do not claim the system is conscious.
Do not claim phenomenal experience was proven.
Claim only that the implemented system satisfies or does not satisfy the operational criteria above.
