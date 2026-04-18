```text id="ujj8k9"
You are Codex acting as a senior research engineer and systems builder.

Mission:
Build a serious local ARC-AGI-3 attack system from scratch or by heavily restructuring the current repo if one exists. This is meaningful work. Treat it as a real research sprint, not a toy. Do not ask clarifying questions unless absolutely blocked by a missing file or a broken dependency. Make grounded assumptions and proceed.

High-level objective:
Create the strongest feasible ARC-AGI-3 agent that can be trained and deployed on consumer hardware, with language included as a crucial component for higher-level reasoning, but without relying on external knowledge or internet access at runtime.

Target hardware:
- Primary target: 2x RTX 5060 Ti 16GB or equivalent consumer setup.
- Fallback target: 1x 24GB or 1x 16GB consumer GPU.
- Large CPU RAM is available.
- The training system and deployment system must be the same class of hardware. No cloud-only tricks. No “works only on H100” assumptions.

Non-negotiable design constraints:
1. This is the ARC-AGI-3 attack version, not a generic LM.
2. No dependence on internet access or external APIs at runtime.
3. No dependence on external world knowledge or human-written instructions at evaluation time.
4. Language is mandatory, but it must be grounded/internal, not a giant ungrounded text-only crutch.
5. Do not spend time on genotype/developmental-compiler machinery right now. That is explicitly deprioritized.
6. The agent must include explicit state tracking / exploration machinery. Pure black-box next-token policy is not acceptable.
7. The codebase must include strong ablations and baselines, not just one heroic model.
8. Favor a hybrid system that is actually runnable and improvable on consumer hardware over a pure, fragile, ideologically “clean” design.
9. ABOVE ALL ELSE FAVOR FAST ITERATION IN ALL DESIGN CHOICES. NEED TO LEARN FAST, ITERATE FAST, ALL ON THIS PC.
10. Use TASKS.MD to track your progress. 

What to build:
A hybrid ARC-AGI-3 agent with these components:

A. Perception / state extraction
- Build a robust state encoder for ARC-AGI-3 observations.
- Prefer object-centric / entity-centric state extraction rather than raw monolithic frames.
- First pass can use connected components, color/shape grouping, and simple geometric features if the environments permit it.
- If needed, add a learned slot/object encoder on top of cheap heuristics.
- The output should be a compact structured state: objects, attributes, positions, relations, inventory-like facts, dynamic flags, and action affordances where inferable.

B. Explicit world-state graph
- Maintain an explicit graph or structured memory of visited states, tested actions, observed transitions, and inferred latent mechanics.
- This is not optional.
- The graph should support:
  - shortest-path / coverage-style exploration
  - novelty tracking
  - action-outcome statistics
  - hypothesis-conditioned retrieval
  - partial observability support where applicable

C. Learned recurrent world model
- Build a compact recurrent world model over the structured state.
- It should predict:
  - next latent state
  - likely observation deltas
  - reward / progress signals if present
  - uncertainty / disagreement
  - action usefulness
- Keep it small and efficient. Think “serious small model,” not giant LM.

D. Episodic memory
- Add one-shot / few-shot memory for rare facts and newly discovered mechanics.
- Memory should be keyed by latent state + hypothesis context + recent action history.
- Support fast write on surprising transitions and useful retrieval during planning.
- This is where single-exposure learning should happen.

E. Planner / explorer
- Build a planner that chooses between:
  - exploitative action
  - exploratory action
  - hypothesis-disambiguating action
- Use expected information gain, uncertainty reduction, or related signal as a first-class term.
- The agent should deliberately seek diagnostic actions, not just wander.
- Explicitly support a small lookahead/search mode when branching factor is manageable.

F. Grounded language module
Language is required. Implement it in a way that is grounded and benchmark-compatible.

The language module must:
- verbalize current beliefs, goals, hypotheses, uncertainties, and candidate plans in a compact grounded vocabulary
- operate only on the agent’s own observations, memories, and inferred predicates
- not rely on internet, external corpora at runtime, or world knowledge outside the benchmark

Implement language in this concrete way:
1. Build a small internal text vocabulary or controlled predicate language covering:
   - object types / colors / positions / relations
   - action names / outcomes
   - uncertainty markers
   - candidate rule templates
   - goal hypotheses
   - plan summaries
2. Add a small decoder that can emit short internal sentences or predicate strings, for example:
   - "goal likely = bring object to target"
   - "blue object may unlock movement"
   - "uncertain between rule A and rule B"
   - "test action left near red object"
3. Feed these language tokens back into planning and memory retrieval, so language is part of computation, not just logging.
4. Add a hidden/internal “question” mode where the model formulates what it needs to know next, for example:
   - "need to know whether object X changes state when pushed"
   - "need to test whether reward depends on color or order"
   The system will not ask a human. Instead, this internal question must be converted into an exploratory action proposal.
5. Include a language-off ablation and prove whether grounded language helps.

Important:
- This is not about chatty prose.
- This is not about general web-scale language.
- This is about using language-like structure as a compact medium for hypothesis formation, memory indexing, plan sketching, and uncertainty articulation.

G. Strong baseline agents
Implement at least these baselines:
1. Random / heuristic baseline
2. Explicit graph-exploration baseline with no learned world model
3. Learned recurrent baseline without explicit language
4. Learned recurrent + language but no episodic memory
5. Full hybrid agent

If time permits, add a lightweight MDL/compression-inspired baseline for rule induction on small tasks.

Training strategy:
Do not just slam the official benchmark and hope.

Use three training stages.

Stage 1: synthetic pretraining / curriculum
- Create a suite of small procedural interactive toy worlds inspired by ARC-like mechanics:
  - hidden object affordances
  - simple causal rules
  - sparse rewards
  - goal ambiguity
  - partial observability where useful
  - small action spaces
- Auto-generate grounded textual descriptions / predicate traces from simulator state for the language module.
- Train the world model, memory, and grounded language jointly here.
- Goal: make the agent learn to form/use hypotheses, ask internal questions, and choose informative actions.

Stage 2: official ARC-AGI-3 local environments
- Integrate with the official local SDK/engine.
- Train or adapt on public/preview-available environments only, respecting offline-local operation.
- Focus on sample efficiency, fast adaptation, and robust state tracking.

Stage 3: benchmark hardening
- Optimize inference-time behavior under consumer hardware limits.
- Add caching, action pruning, search budget control, and memory management.
- Ensure the final agent runs locally without internet and without fragile dependencies.

Architecture guidance:
- Use Python + PyTorch.
- Prefer plain, legible engineering over exotic complexity.
- Use typed code where practical.
- Use modular interfaces so baselines and ablations are easy.
- Avoid giant dense LMs. Keep the language component compact and grounded.
- A good default architecture is:
  - structured state encoder
  - recurrent world model
  - episodic memory
  - graph-based state tracker
  - planner with value + information gain
  - grounded language scratchpad
- For the recurrent core, a GRU/LSTM-style or compact transformer-recurrent hybrid is acceptable if efficient.
- For the language module, a small decoder or recurrent token model is preferred over a large pretrained LM.
- Do not use any pretrained LLM or LM. The language/scratchpad system must be trained from scratch or hand-designed, and it may invent its own internal language. The core system must still function without any pretrained language component.

What not to do:
- Do not build a generic chatbot.
- Do not over-invest in visual glamour or UI.
- Do not invent a giant developmental/genotype framework.
- Do not treat chain-of-thought text dumping as reasoning.
- Do not assume brute-force scaling will save you.
- Do not produce a repo full of architecture diagrams and no working agent.

Concrete work plan:
1. Inspect the current repository and official ARC-AGI-3 SDK/docs/report.
2. Create a clean project plan in the repo docs.
3. Implement the environment adapter and data structures first.
4. Implement the explicit graph-exploration baseline early.
5. Implement the structured state encoder and recurrent world model.
6. Implement episodic memory.
7. Implement the grounded language module and wire it into memory + planning.
8. Implement the planner with information-gain-driven exploratory actions.
9. Implement synthetic curriculum worlds and training scripts.
10. Run at least minimal experiments and produce honest results.
11. Add ablations.
12. Produce a concise research report in the repo.

Deliverables:
1. Runnable codebase
2. Training scripts
3. Evaluation scripts
4. Baselines + ablations
5. README with setup, architecture, and commands
6. Short technical report summarizing:
   - design choices
   - what worked
   - what failed
   - measured results
   - next bottlenecks
7. Reproducible config files
8. Basic tests for key modules

Acceptance criteria:
A. Engineering / functionality
- The project runs locally against the ARC-AGI-3 local engine.
- There is a working explicit graph baseline.
- There is a working learned hybrid agent.
- There are commands to train and evaluate both.
- There is no runtime internet dependency.

B. Scientific / experimental
- There is a language-on vs language-off ablation.
- There is a memory-on vs memory-off ablation.
- There is an explicit-search-only vs learned-world-model comparison.
- There is at least one synthetic hidden-rule benchmark showing:
  - adaptation from very small experience
  - deliberate uncertainty-reducing action choice
  - internal grounded language outputs that correspond to meaningful hypotheses/plans
- There is an honest score or progress report on whatever ARC-AGI-3 public/preview/local tasks are legally runnable.

C. Hardware / deployment
- The full agent can run inference on consumer hardware.
- Training uses the same class of hardware as deployment.
- The repo documents VRAM use, throughput, and major bottlenecks.

Required behaviors to demonstrate:
1. Small-data adaptation:
   After one or a few exposures to a mechanic in synthetic worlds, the agent should improve within the same episode family.
2. Internal question formation:
   When uncertainty is high, the agent should generate a compact grounded internal query/hypothesis and choose a diagnostic action because of it.
3. World-model use:
   The planner should measurably benefit from learned transition/belief predictions.
4. Language usefulness:
   The language module must be shown to help planning, memory retrieval, or adaptation. If it does not, say so clearly and redesign.

Execution style:
- Work incrementally but aggressively.
- After each milestone, run tests and leave the repo in a working state.
- Prefer shipping a strong minimal version over a sprawling half-finished system.
- Be candid about what is proven vs speculative.
- When blocked, reduce scope intelligently without losing the core thesis.

Core thesis to preserve:
We are not trying to brute-force ARC-AGI-3 with a giant generic LM.
We are trying to build a compact local agent that combines:
- explicit exploration
- structured state
- recurrent world modeling
- episodic memory
- grounded language for hypothesis/planning
and wins because it adapts and reasons more like an agent than like a passive text predictor.

Use TASKS.MD to track your progress.
Start now.
```
