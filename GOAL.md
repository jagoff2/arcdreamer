I would not instantiate this as a “video transformer plus RL head.” That will almost certainly collapse into frame-action imitation, random-search scaffolding, or brittle puzzle heuristics. I would instantiate it as a continuously running **stateful scientific organism**: a neural dynamical system whose persistent state is a mutable world-belief substrate, not a context window. The core object is not a model call; it is a living state transition:

[
S_{t+1}=\Phi_\theta(S_t, o_t, a_{t-1}, \Delta o_t, m_t)
]

where (S_t) is the entire agent-state, (o_t) is the current frame or grid observation, (a_{t-1}) is the previous action, (\Delta o_t) is the measured environmental change, and (m_t) is metadata such as available actions, score, terminal state, or level transition.

For ARC-AGI-3 specifically, the benchmark is interactive, turn-based, and explicitly demands exploration, goal inference, world-model formation, and planning without instructions; humans solve all environments while frontier AI systems were reported below 1% as of the technical report. The official scoring is action-efficiency against human baselines, and internal computation does not count as an action, so the agent should spend large amounts of internal compute to minimize external experiments. ([arXiv][1]) ([ARC-AGI-3 Docs][2])

The architecture I would build is:

**CASSI: Continuous Analog State Scientist Intelligence.**

Its fundamental design principle is this: **do not store knowledge primarily in weights, and do not store it primarily in a hidden vector. Store it as a live posterior over causal hypotheses, grounded in an exact event journal, continuously distilled into plastic memory, graph memory, and latent world models.**

ARC-AGI-3’s action interface is small but information-poor: games expose standardized actions, including simple actions 1–5, coordinate action 6, undo action 7, and observations are 2D grid-like frames up to 64×64 with integer cell values. This makes blind brute force tempting, but the scoring punishes it quadratically through action inefficiency. ([ARC-AGI-3 Docs][3]) ([ARC-AGI-3 Docs][4]) ([ARC-AGI-3 Docs][2])

### 1. The persistent state is not one tensor

Define the live state as:

[
S_t = (z_t, E_t, G_t, M_t, H_t, Q_t, \Pi_t, A_t, C_t)
]

where:

(z_t) is the continuous neural field: a working latent state evolved by a continuous-time recurrent model.

(E_t) is an append-only event journal: every frame, action, delta, score change, available-action mask, terminal event, and prediction error.

(G_t) is a discovered transition graph: nodes are canonicalized observations or abstract states; edges are actions and their observed effects.

(M_t) is semantic memory: compressed relational facts, object identities, action affordances, invariants, counterexamples, and reusable skills.

(H_t) is fast plastic memory: Hebbian traces, low-rank adapters, and temporary parameters learned during the current game or run.

(Q_t(h)) is a posterior over hypotheses (h): candidate world rules, action semantics, goal definitions, object transformations, and level progression mechanisms.

(\Pi_t) is a library of discovered macro-actions and policies.

(A_t) is the active affordance model: what actions are likely to change what parts of the world.

(C_t) is context: game ID, level index if inferable, uncertainty, confidence gates, and reset/win/game-over state.

This is how you get the “unbroken analog stream” property. The hidden state (z_t) flows continuously, but long-term continuity comes from **state translation**, not from pretending a recurrent vector can remember everything. When a level ends or the game resets, the agent does not clear (S_t); it applies a boundary transform:

[
S_{t+1}=T_{\text{boundary}}(S_t,\text{event})
]

That boundary transform tags memories by context, lowers the confidence of fragile hypotheses, preserves verified rules, and moves useful patterns into semantic/procedural memory. A reset is not amnesia. It is a phase change.

### 2. The analog stream should be a continuous-time recurrent field

Use a continuous-time controller, preferably a liquid-time-constant or neural-ODE-style recurrent core:

[
\frac{dz}{d\tau}=f_\theta(z_\tau, e(o_t), r(M_t), r(G_t), q_t, a_{t-1})
]

and treat external observations/actions as impulses:

[
z_{\tau^+}=z_{\tau^-}+J_\theta(e(o_t), a_{t-1}, \Delta o_t)
]

Liquid time-constant networks are relevant because they explicitly model continuous-time recurrent dynamics with state-dependent time constants and stable bounded behavior. ([arXiv][5])

But this continuous state is only the **working fluid**. It is not the memory. Memory must be explicit, queryable, and revisable. Neural Turing Machines and memory-augmented neural networks showed the importance of differentiable external memory for algorithmic behavior and rapid assimilation of new data; differentiable plasticity shows how recurrent networks can be trained to keep learning after initial training through learned Hebbian-style update rules. ([arXiv][6]) ([arXiv][7]) ([arXiv][8])

### 3. Perception should become object/event algebra, not captions

The input may be video frames in general, but for ARC-AGI-3 you should exploit the actual grid representation. Build a dual perceptual front end:

[
o_t \rightarrow {P_t, O_t, R_t, B_t}
]

where (P_t) is patch/grid embedding, (O_t) is object slots, (R_t) is a relation graph, and (B_t) is a vector-symbolic binding of facts.

For ARC, object extraction should include connected components, color regions, bounding boxes, holes, adjacency, containment, symmetry, repetition, line/shape structure, motion/change vectors, and candidate “active” cells. This is not a hand-coded game solver; it is the core-knowledge perceptual substrate humans use: objectness, topology, numerosity, containment, geometry, and change.

The crucial representation is not “the frame.” It is the **delta**:

[
\Delta_t = \operatorname{EditScript}(o_{t-1}, o_t)
]

For every action, the system asks: what changed, what stayed invariant, which object caused the change, and which hypothesis predicted it?

Use vector-symbolic or hyperdimensional bindings to encode compositional facts like:

[
\text{bind}(\text{object_17}, \text{color_red})+
\text{bind}(\text{object_17}, \text{position_x7y4})+
\text{bind}(\text{time_t}, \text{action_3})
]

Vector Symbolic Architectures are specifically designed to combine distributed vector representations with symbolic structure through high-dimensional algebraic operations. ([arXiv][9])

### 4. The world model must be a posterior over causal edit programs

A normal learned world model predicts pixels. That is insufficient. ARC-AGI-3 needs low-data causal abstraction. So instantiate the world model as a mixture:

[
p(o_{t+1}\mid o_t,a_t,S_t)
==========================

\sum_h Q_t(h),p_h(o_{t+1}\mid o_t,a_t)
]

Each (h) is not merely a neural latent. It can be a typed stochastic edit program:

[
h:\quad \text{select objects} \rightarrow \text{apply transformation} \rightarrow \text{update relations} \rightarrow \text{test goal}
]

Examples of hypothesis families:

[
\text{ACTION3 moves controlled object left}
]

[
\text{ACTION6}(x,y)\text{ toggles object containing }(x,y)
]

[
\text{goal is relation-match}(O_i,O_j)
]

[
\text{level advances when all target heights equal reference pattern}
]

[
\text{object changes color according to contact or path history}
]

The posterior update is:

[
\log Q_{t+1}(h)
===============

\log Q_t(h)
-\beta,\mathcal{L}_{\text{pred}}(h;E_t)
-\lambda,\operatorname{DL}(h)
+\gamma,\operatorname{support}(h)
-\rho,\operatorname{counterexample}(h)
]

where (\operatorname{DL}(h)) is description length. This makes “theorizing” a compression process: the best theory is the shortest causal program that predicts observed transitions and goals.

This crosses Bayesian program learning with world models. Bayesian Program Learning represented visual concepts as probabilistic programs and combined compositionality, causality, and learning-to-learn for one-shot generalization; MuZero showed that planning can be coupled to a learned model without knowing the true environment rules; Dreamer-style agents learn world models and improve behavior by imagining futures. ([CMU School of Computer Science][10]) ([arXiv][11]) ([arXiv][12])

### 5. Action selection should be experimental design, not policy sampling

ARC-AGI-3 agents must spend actions like a scientist spends experiments. The policy should optimize:

[
a_t
===

\arg\max_a
\left[
\mathbb{E}*{h\sim Q_t}V_h(S_t,a)
+
\alpha,I(h;o*{t+1}\mid S_t,a)
-----------------------------

## c(a)

\eta,\operatorname{risk}(a)
\right]
]

where (I(h;o_{t+1}\mid S_t,a)) is expected information gain about the world rules, and (c(a)) is action cost.

This creates a strict hierarchy:

First, use internal simulation.

Second, test reversible or low-risk actions.

Third, test actions that maximally distinguish hypotheses.

Fourth, execute the shortest predicted path to win.

Fifth, when predictions fail, backtrack, revise the posterior, and test the smallest discriminating intervention.

This is where prior attempts often fail. They explore states; they do not explore **hypothesis space**. A good ARC-AGI-3 agent should ask:

“What action will most cheaply tell me whether this game is about movement, clicking, matching, counting, transformation, pathing, resource transfer, ordering, or constraint satisfaction?”

The preview competition data supports this emphasis: the winning preview agent scored 12.58% using action-learning for more efficient exploration, while another strong approach built a state graph from frames, pruned loops/no-change actions, and trained a small value model from progress. A separate graph-based exploration paper reported that explicit state/action tracking substantially outperformed frontier LLM-style agents on the preview challenge. ([ARC Prize][13]) ([ARC Prize][14]) ([arXiv][15])

### 6. The agent needs four coupled memories

A single memory mechanism will not work. Use four.

The **event journal** is lossless, append-only, and boring:

[
E_t = E_{t-1}\cup{o_{t-1},a_{t-1},o_t,\Delta_t,m_t,\epsilon_t}
]

This is the only layer that honestly satisfies “state is never lost,” assuming finite storage is sufficient for the run.

The **transition graph** is operational:

[
G_t=(V_t,E_t^{\text{graph}})
]

where nodes are abstracted states and edges are tested actions. It tracks visited states, no-ops, loops, reversible transitions, shortest paths, and unexplored state-action pairs.

The **semantic memory** stores compressed facts:

“Action 1 decreases y of controlled component.”

“Clicking centers changes color; clicking empty space no-ops.”

“Blue objects are obstacles.”

“Goal candidates involving equality of heights gained support.”

The **plastic neural memory** stores fast adaptation:

[
H_{t+1}=\gamma H_t+\eta,n_t,u_t v_t^\top
]

where (n_t) is neuromodulation from novelty, prediction error, reward, or terminal success. In implementation, (H_t) can be a mix of Hebbian matrices, LoRA adapters, small hypernetworks, and key-value memory. The base weights stay mostly frozen; the agent learns online through fast weights, adapters, posterior updates, graph expansion, and replay.

Continual-learning work repeatedly shows why this is necessary: ordinary networks trained sequentially tend to catastrophically forget prior knowledge, so stability/plasticity separation is not optional. ([arXiv][16])

### 7. Test-time training should be internal and self-supervised

ARC-AGI-1 saw major gains from test-time training; one MIT paper found that per-instance test-time training with augmentations and adapters significantly improved ARC performance, with a reported 53% public validation score for an 8B model in their setup. ([arXiv][17])

For ARC-AGI-3, the analogous move is not “fine-tune on the puzzle.” It is:

Train the current game’s adapter on self-supervised transition prediction.

Train inverse dynamics: infer which action caused a delta.

Train no-op/change discrimination.

Train object persistence across frames.

Train goal-proximity/value only after wins or progress signals.

Train a coordinate affordance map for ACTION6.

The test-time loss:

[
\mathcal{L}_{\text{TTT}}
========================

\mathcal{L}*{\text{next-delta}}
+
\mathcal{L}*{\text{inverse-action}}
+
\mathcal{L}*{\text{object-consistency}}
+
\mathcal{L}*{\text{affordance}}
+
\mathcal{L}*{\text{value-backup}}
+
\lambda\lVert \phi*{\text{fast}}\rVert^2
]

where (\phi_{\text{fast}}) are temporary adapter weights.

The agent should train on its own stream while it plays. But it should update only bounded fast-memory modules during evaluation, not destructively modify the whole base network.

### 8. Goal inference is inverse planning over observed consequences

ARC-AGI-3 does not provide natural-language goals. So goal inference must be explicit.

Maintain:

[
Q_t(g)
]

over candidate goals (g). Candidate goals are generated from observed structure and progress events: terminal wins, score increments, level transitions, object disappearance, pattern completion, equality constraints, reachability, matching, sorting, filling, clearing, alignment, containment, and transformation closure.

Update goals by asking:

“If (g) were the goal, would the observed state transitions, level changes, and human-efficient action paths make sense?”

[
\log Q_{t+1}(g)
===============

\log Q_t(g)
+
\operatorname{progress}*g(o_t,o*{t+1})
--------------------------------------

## \operatorname{inconsistency}_g(E_t)

\lambda\operatorname{DL}(g)
]

The planner should not wait for the true goal to be known. It should act under a distribution:

[
V(S,a)=\sum_g Q(g)\sum_h Q(h)V_{g,h}(S,a)
]

A practical trick: infer **subgoals before goals**. Humans often do this. Even without knowing the final win condition, the agent can identify controllable objects, reversible moves, target-like structures, changing objects, gates, counters, and constraints.

### 9. For ARC-AGI-3, use a specific runtime loop

At each turn:

1. Parse the frame into object slots, relations, raw grid hash, and salience map.

2. Compare with previous frame and compute a minimal edit script.

3. Update the event journal.

4. Update the state graph.

5. Update action affordances.

6. Update the hypothesis posterior over action semantics, world dynamics, and goals.

7. Train fast adapters on the accumulated transition data.

8. Generate candidate experiments and candidate win plans.

9. Use internal rollouts under the posterior ensemble.

10. Select the action with best value/information/action-cost tradeoff.

11. Execute one external action.

12. Preserve all state; never clear, only translate.

Pseudocode:

```python
while agent_is_running:
    obs, meta = receive_frame()

    percept = perceive(obs)
    delta = diff(prev_obs, obs) if prev_obs is not None else None

    S.event_log.append(prev_obs, prev_action, obs, delta, meta)
    S.scene = update_scene(S.scene, percept, delta)
    S.graph = update_transition_graph(S.graph, prev_obs, prev_action, obs, delta, meta)

    S.affordances = update_affordances(S.affordances, S.graph, delta, meta)
    S.hypotheses = bayesian_hypothesis_update(S.hypotheses, S.event_log, S.graph)
    S.goals = infer_goals(S.goals, S.event_log, meta)

    S.fast_weights = self_supervised_ttt(
        fast_weights=S.fast_weights,
        event_log=S.event_log,
        losses=[
            "next_delta",
            "inverse_action",
            "object_persistence",
            "affordance_map",
            "goal_progress_if_available",
        ],
    )

    experiments = propose_discriminating_experiments(S)
    plans = posterior_model_predictive_search(S, experiments)

    action = select_action(
        plans,
        objective="expected_win_minus_action_cost_plus_information_gain",
        risk_sensitive=True,
    )

    prev_obs = obs
    prev_action = action
    execute(action)
```

### 10. The architecture should learn action semantics by “causal spectroscopy”

Think of every action as a probe and every frame delta as a spectrum. The agent should discover the latent mechanics by perturbation.

For simple actions, test each available action once in the safest reversible state. For coordinate actions, never scan all coordinates. Generate a sparse candidate set: object centers, corners, boundaries, empty cells adjacent to objects, repeated-pattern anomalies, and high-salience cells. Then model:

[
p(\Delta o\neq 0\mid x,y,o_t)
]

as an affordance heatmap.

For undo action 7, immediately test whether it reverses the last transition if available. If undo works, the environment becomes much safer to probe.

For movement-like games, infer a controlled object by finding the object whose position changes consistently with action labels. For logic/orchestration games, infer controlled variables by interventions: which object, count, color, height, or relation changes when acted on?

This turns ARC-AGI-3 from “play unknown game” into “identify controllable variables, transition law, and terminal constraint under action budget.”

### 11. The neural part should propose; the symbolic/graph part should verify

Do not make the symbolic layer a fixed ARC DSL. That becomes benchmark overfitting. Make it a **typed causal grammar** that is domain-general:

Selectors: object by color, shape, relation, position, novelty, controllability, contact, containment.

Relations: same color, adjacent, aligned, inside, connected, between, count-equal, path-exists, symmetry, ordering.

Transforms: move, copy, delete, toggle, rotate, reflect, recolor, increment/decrement, attach/detach, fill, reveal, gate, swap.

Combinators: sequence, condition, repeat-until, all, exists, argmin/argmax, path plan, constraint satisfaction.

The neural model proposes candidate programs and parameters. The verifier tests them against the exact event log. The posterior rewards hypotheses that predict all observed transitions with low description length.

This is the main deviation from prior hacks: the system does not contain hand-written solutions to ARC games. It contains machinery for inducing small causal theories from interaction.

### 12. Training curriculum should be broad, not ARC-clone overfitting

The ARC-AGI-3 paper explicitly warns against domain-specific overfitting to ARC-AGI-3-like environments or handcrafted harness choices that inflate benchmark performance without measuring general intelligence. ([arXiv][1])

So train the agent on a broad “scientist curriculum”:

Object permanence games.

Grid physics and cellular automata.

Toy chemistry: objects transform on contact.

Navigation with hidden rules.

Inventory/attachment/detachment.

Logic circuits and switches.

Matching/copying under transformations.

Counting and equality constraints.

Resource transfer and conservation.

Games with unknown goals and sparse terminal feedback.

Games where actions have unknown semantics.

Games where some actions are no-ops, traps, or reversible.

The objective is not to memorize games. The objective is to meta-learn the loop: perceive, perturb, infer, compress, plan, test, consolidate.

Use video self-supervision for the visual backbone. JEPA-style approaches are relevant because they learn predictive video representations in latent space rather than pixel-level generation, and V-JEPA 2 specifically targets understanding, prediction, and planning from observation plus limited interaction data. ([arXiv][18])

### 13. The planner should run two searches at once

There should be a **solution search** and an **experiment search**.

Solution search:

[
\min_{\pi} \mathbb{E}[\text{actions to terminal win}]
]

Experiment search:

[
\max_a I(h,g;o_{t+1}\mid S_t,a)
]

The agent chooses experiments until one hypothesis has enough posterior mass, then switches to exploitation. But it should never fully stop experimenting; failed predictions are valuable.

Use MCTS/POMCP-style planning over the learned model, but with graph memory anchoring. MuZero-style latent planning is useful, but pure MuZero is not enough because ARC-AGI-3 requires rapid rule induction on a novel environment, not just model-based control from extensive training. ([arXiv][11])

### 14. Internal self-play should be imagination, not hallucination

The world model should imagine futures, but each imagined transition must carry uncertainty:

[
\hat{o}*{t+k},\ \sigma*{t+k},\ Q(h)
]

Plans that depend on high-uncertainty imagined transitions are treated as hypotheses, not facts. This prevents “dream collapse,” where the agent believes its own simulator more than the environment.

Use three rollout types:

Exact graph rollouts: transitions already observed.

Symbolic-hypothesis rollouts: transitions predicted by verified causal rules.

Neural rollouts: transitions predicted by learned latent dynamics.

Risk ordering should be:

[
\text{observed graph} > \text{verified symbolic} > \text{ensemble-agreed neural} > \text{speculative neural}
]

### 15. What “memorize” means here

The agent should memorize in three forms:

Raw episodic memory: “I did action 3 here and the red object moved.”

Compressed causal memory: “Action 3 probably means left.”

Procedural memory: “To solve this level family, move controlled object to transform symbol, then bring it to matching target.”

Across levels of the same game, this is decisive. ARC-AGI-3 games are series of levels; later levels test whether the agent has acquired the mechanics, not whether it can re-random-search from scratch. The official methodology weights later levels more heavily, so skill transfer within a game matters. ([ARC-AGI-3 Docs][2])

### 16. Minimal viable build

The smallest serious version would be:

A deterministic grid/object parser.

A state-transition graph with exact frame hashes and canonical object abstractions.

A coordinate-action affordance model.

A hypothesis posterior over typed edit rules.

A small neural encoder for object/relation embeddings.

A fast adapter trained online for transition prediction and action-effect prediction.

A planner that combines shortest-path graph search, information gain, and hypothesis-conditioned model-predictive control.

An event journal with replay and consolidation.

Do not start with a giant VLM. Start with a compact, exact, instrumented agent that cannot forget what happened and cannot take unaccounted actions. Then add neural proposal networks where search is too large.

### 17. The likely winning behavior pattern

On a new ARC-AGI-3 game, CASSI should behave like this:

It first identifies the controllable substrate: which objects respond to which actions.

It immediately prunes no-ops and loops.

It tests reversibility.

It infers coordinate affordances sparsely, not by brute-force clicking.

It forms 3–20 candidate theories.

It chooses actions that maximally separate those theories.

It identifies progress events or terminal conditions.

It compresses the discovered rule into a macro-policy.

It applies that macro-policy to later levels.

It notices when a later level violates the current theory, then minimally patches the theory rather than restarting.

That is the difference between action-efficient intelligence and brute force.

### 18. Why this is not just prior art glued together

The novel move is to make the agent’s enduring identity a **posterior-preserving dynamical memory system**, not a model checkpoint, a prompt, or a hidden state.

World models contribute imagination.

MuZero contributes planning with learned dynamics.

Test-time training contributes per-instance adaptation.

Memory-augmented networks contribute explicit read/write persistence.

Differentiable plasticity contributes fast within-life learning.

Bayesian program learning contributes compositional causal hypotheses.

Graph exploration contributes exact action-state accounting.

The synthesis is: **a continuously running neural-scientific state machine whose main currency is causal compression of its own interaction stream.**

That is the paradigm shift I would bet on for ARC-AGI-3: not “bigger model reasons harder,” but “an agent treats every action as an experiment, every delta as evidence, every level as curriculum, and every hypothesis as a mutable executable theory.”

[1]: https://arxiv.org/html/2603.24621v1 "ARC-AGI-3: A New Challenge for Frontier Agentic Intelligence"
[2]: https://docs.arcprize.org/methodology "ARC-AGI-3 Scoring Methodology - ARC-AGI-3 Docs"
[3]: https://docs.arcprize.org/actions "Actions - ARC-AGI-3 Docs"
[4]: https://docs.arcprize.org/game-schema "Game Schema - ARC-AGI-3 Docs"
[5]: https://arxiv.org/abs/2006.04439?utm_source=chatgpt.com "Liquid Time-constant Networks"
[6]: https://arxiv.org/abs/1410.5401?utm_source=chatgpt.com "Neural Turing Machines"
[7]: https://arxiv.org/abs/1605.06065?utm_source=chatgpt.com "One-shot Learning with Memory-Augmented Neural ..."
[8]: https://arxiv.org/abs/1804.02464?utm_source=chatgpt.com "Differentiable plasticity: training plastic neural networks with backpropagation"
[9]: https://arxiv.org/abs/2111.06077?utm_source=chatgpt.com "A Survey on Hyperdimensional Computing aka Vector Symbolic Architectures, Part I: Models and Data Transformations"
[10]: https://www.cs.cmu.edu/~rsalakhu/papers/LakeEtAl2015Science.pdf?utm_source=chatgpt.com "Human-level concept learning through probabilistic ..."
[11]: https://arxiv.org/abs/1911.08265 "[1911.08265] Mastering Atari, Go, Chess and Shogi by Planning with a Learned Model"
[12]: https://arxiv.org/abs/2301.04104 "[2301.04104] Mastering Diverse Domains through World Models"
[13]: https://arcprize.org/blog/arc-agi-3-preview-30-day-learnings "ARC-AGI-3 Preview: 30-Day Learnings | ARC Prize"
[14]: https://arcprize.org/competitions/arc-agi-3-preview-agents "ARC-AGI-3 Preview Agent Competition"
[15]: https://arxiv.org/abs/2512.24156 "[2512.24156] Graph-Based Exploration for ARC-AGI-3 Interactive Reasoning Tasks"
[16]: https://arxiv.org/html/2403.05175v1?utm_source=chatgpt.com "Continual Learning and Catastrophic Forgetting"
[17]: https://arxiv.org/html/2411.07279v1 "The Surprising Effectiveness of Test-Time Training for Abstract Reasoning"
[18]: https://arxiv.org/abs/2506.09985?utm_source=chatgpt.com "[2506.09985] V-JEPA 2: Self-Supervised Video Models ..."
