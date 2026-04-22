# Action-Level Spotlight Scientist Patch

The ARC-facing scientist agent path now includes an action-level spotlight
controller. This is not benchmark-specific tuning and it is not a claim about
consciousness. It is a generic serial accountability layer over the existing
hypothesis engine, world model, memory, and candidate planner.

## Why it exists

The failure mode it targets is generic to black-box control tasks with hidden
state: selector-like, click-like, and coordinate-bearing actions can alter
latent control state without producing an immediate visible delta. A one-step
scorer tends to suppress those actions because they look like no-ops.

The spotlight corrects that by:

- turning each chosen primitive action into an explicit short-lived intention
- storing the intended target, predicted reward, predicted change, and falsifier
- delaying negative evidence for latent binders until at least one downstream
  probe action has been attempted

## Behavioral effect

`ACTION5`, `ACTION6`, `click:*`, and selector-like actions are treated as
potential binders rather than instant failures. When one is chosen, the next
movement or interaction action is scored as the diagnostic probe. Only after the
probe fails does the spotlight write negative evidence against the binder.

That keeps the control loop generic:

- action semantics are inferred from transitions, not hardcoded task logic
- the spotlight only reasons over action families, state changes, progress, and
  uncertainty
- the existing hypothesis engine, world model, memory, and planner remain the
  source of candidate evidence

## Diagnostics

Inspect `episode["diagnostics"]["spotlight"]` during evaluation. Useful fields:

- `active_intention`
- `pending_binder_probe`
- `last_broadcast`
- `binding_success_total`
- `binding_failure_total`
- `trace_tail`

A healthy trace for hidden-control tasks often includes:

```text
spotlight.focus=bind_or_test_hidden_control
spotlight.pending_probe_after=<selector/click/binder>
spotlight.focus=probe_after_latent_binding
```
