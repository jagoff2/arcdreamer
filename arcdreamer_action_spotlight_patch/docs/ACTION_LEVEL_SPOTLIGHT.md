# Action-Level Spotlight Scientist Patch

This patch is based on the working diagnosis that the current agent is not failing because it lacks another offline trainer. It is failing because real ARC-AGI-3 games require serial, accountable action commitments under partial observability. In particular, selector, click, and coordinate actions can change hidden control state without creating an immediate visible pixel delta. A planner that judges every action from one-step visible change will learn to suppress exactly the actions it must learn to use.

The patch implements a bounded action-level global workspace. It does not assert machine consciousness. It operationalizes the useful part: a small serial bottleneck that broadcasts one active intention at a time, records what the action is meant to prove, and updates local preference from support or falsification.

## Files

```text
arcagi/scientist/spotlight.py
arcagi/agents/spotlight_scientist_agent.py
arcagi/agents/scientist_agent.py
scripts/install_action_spotlight_patch.py
```

## Behavioral change

The new `ActionSpotlight` receives the same candidate actions that the current scientist planner would consider. It scores them with the hypothesis engine, online world model, memory, coverage pressure, no-effect memory, contradiction memory, and a new latent-binder term.

Every selected action becomes an explicit intention with:

```text
action
intent_kind
target
predicted_reward
predicted_change
falsifier
short commitment window
support / contradiction counters
```

This is the important rule: `ACTION5`, `ACTION6`, `click:*`, and selector-like actions are not penalized merely because they produce no immediate visible change. Instead, they create a pending binder probe. The next movement or interaction action is treated as the downstream test. Only if the downstream probe fails does the system write negative evidence against the binder.

## Why this targets the current failure mode

The repository status identifies selector/click semantics, proof pressure, and downstream selector-conditioned consequences as the remaining real-game bottlenecks. This patch directly attacks that bottleneck by adding delayed credit assignment for latent binding actions and by forcing the agent to answer “what am I testing with this action?” before each step.

## Install

From the repository root, unzip the patch so the files land under `arcagi/`, `docs/`, and `scripts/`.

Then run:

```bash
python scripts/install_action_spotlight_patch.py
```

The script only edits `arcagi/evaluation/harness.py`. The replacement makes `--agent scientist`, `--agent spotlight`, and `--agent spotlight_scientist` construct the spotlight scientist. It also makes checkpoint loading use the spotlight checkpoint loader instead of returning the old base scientist.

## Evaluate

Local synthetic smoke test:

```bash
python -m pytest
python -m arcagi.evaluation.harness synthetic --agent scientist --episodes-per-family 3
```

ARC offline run:

```bash
python -m arcagi.evaluation.harness arc --agent spotlight --mode offline --game-limit 5
```

If the harness patch is not applied, `--agent scientist` will still instantiate the new wrapper when no checkpoint path is used, but checkpoint loading may still return the old base scientist.

## What to inspect in traces

The useful diagnostics are inside `episode["diagnostics"]["spotlight"]`:

```text
active_intention
pending_binder_probe
last_broadcast
no_effect_count_total
binding_success_total
binding_failure_total
trace_tail
```

A good real-ARC trace should show:

```text
spotlight.focus=bind_or_test_hidden_control
spotlight.pending_probe_after=<ACTION5/ACTION6/click>
spotlight.focus=probe_after_latent_binding
```

That pattern means the agent is no longer treating latent selector actions as immediate no-ops.

## Limits

This patch is not sufficient proof of ARC-AGI-3 competence. It is a control architecture correction. It should be paired with evaluation against real offline games and replay inspection. If the agent still fails after this, the next likely bottleneck is representation repair: the perceived object model may not expose the variable whose transition actually matters.
