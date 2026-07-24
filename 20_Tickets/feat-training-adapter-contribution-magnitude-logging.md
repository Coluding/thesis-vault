---
type: feat
scope: training
status: open
priority: high
created: 2026-07-09
updated: 2026-07-16
resolution:
resolution_note:
closed_at:
related: ["[[../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]", "[[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]]", "[[bug-adapter-gate-saturation-mask-mix]]"]
---

# feat: log adapter contribution magnitude ‖g·Δ‖ / ‖f_base‖

## Shipped (2026-07-16)

Implemented in `generative-flow-adapters` — `models/adapted_model.py`,
`training/trainer.py`, `training/wandb_logger.py`:

- `AdaptedModel._compose()` now stashes the post-sigmoid gate tensor on
  `self._last_gate` for `mask_mix`/`avid_mask_mix`/`gated_residual` (`None`
  for `add`/`replace`, which have no gate).
- `Trainer._forward_and_loss` logs `adapter_gate_mean` / `adapter_gate_std`
  every step when a gate exists, plus a **composition-agnostic**
  `adapter_rel_contribution = ‖composed − base‖ / ‖base‖` (batch-norm ratio)
  that works for every composition mode, including the ungated ones this
  ticket's original formula didn't cover.
- `Trainer.training_step` logs a wandb Histogram `adapter/gate_hist` (full
  per-pixel gate distribution, subsampled to 100k values) every optimizer
  step — the user asked specifically for the distribution, not just the
  mean, since a bimodal 0/1 gate and a unimodal 0.5 gate look identical on
  `gate_mean` alone but mean very different things architecturally.
- **Extended scope beyond the original ticket:** also added
  `adapter_grad_norm` — L2 norm over `model.adapter.parameters()` only,
  captured post-accumulation/pre-`zero_grad()`. Same "is the adapter dead or
  active" theme as this ticket's decision rule, but answers it one level
  further upstream (gradient reaching the adapter at all, vs. gradient
  reaching it but converging to something small).
- Fixed a real bug surfaced while adding this: `Trainer._forward_and_loss`
  referenced `base_output` outside the flow-only branch where it was
  assigned, which would `UnboundLocalError` on any diffusion-model run once
  the new gate/rel-contribution block was added. Predeclared it (matching
  the existing `base_only_loss_val` pattern) — verified via the full test
  suite (`pytest tests/`, all passing after the fix).
- Full field-by-field documentation: [[../10_now/training-hyperparameters]]
  §"What we log, per step".

Decision rule from the original ticket is now checkable directly — read
`adapter_gate_mean`/`adapter/gate_hist` and `adapter_grad_norm` together with
`adapter_rel_contribution` per the "read-together" note in
[[../10_now/training-hyperparameters]].

**Not yet done:** nothing has actually run with this logging live yet (all
three 2026-07-16 runs predate it) — the gatelow/overfit/replace analysis in
[[../30_Knowledge/experiments/20260716-wan-xattn-adapter-clones-base-not-actions]]
still relies on the older delta/probe metrics only. Next run of any of those
configs will be the first with gate histograms and grad norms visible.

## Why

For [[../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]] we
cannot currently tell whether the adapter is *inert* (contributing ≈0 to the
composed prediction) or *active but ignoring actions*. A flat `base_loss` is
consistent with both. One scalar disambiguates the "inert" branch.

## What

In the composition `f = f_base + g(d)·Δ_φ` (`AdaptedModel._compose`), log per
training step:

- `adapter/rel_contribution` = `‖g(d)·Δ‖ / ‖f_base‖` (batch-mean of the ratio, or
  ratio of batch norms).
- `adapter/gate_value` = the learned gate `g(d)` / `σ(gate)` for gated modes, so
  a dead zero-init gate is visible directly.

Emit to the same metrics dict as `base_loss` (`trainer.py:334`) → stdout / JSONL
/ W&B.

## Decision rule

- `rel_contribution ≈ 0` throughout ⇒ adapter is inert (dead gate, adapter LR too
  low, or grad not reaching it) — independent of the action question.
- `rel_contribution` grows but `base_loss` stays flat ⇒ adapter is active but its
  output isn't reducing the objective → conditioning content problem (pairs with
  [[experiments/exp-conditioning-action-shuffle-ablation]]).

## Scope

Metrics-only; no change to the training objective. Keep it cheap (norms on the
already-computed tensors).

## Update (2026-07-14) — now higher priority: gate is likely saturated

[[../30_Knowledge/tech/why-adapter-underlearns-diagnosis]] confirmed via the
composition math that `mask_mix` + `gate_bias=4.0` (both live Wan configs) puts
`gate ≈ 0.982` at init with no annealing schedule anywhere — a ~50× throttle on
the adapter's gradient, every step (see
[[bug-adapter-gate-saturation-mask-mix]]). `gate_value` logging (this ticket) is
the direct empirical check: does it move off 0.982 over training, or stay pinned?
That distinguishes "init artifact, self-corrects" from "confirmed ongoing
suppressor."

Complementary metrics already implemented this session (2026-07-14,
`training/trainer.py` `_forward_and_loss`/`_probe_eval`): `denoise_adapter_delta`
(paired base-vs-adapted denoising loss on the same batch) and
`probe_denoise_{base,adapted,delta}` (same, on a frozen low-variance probe batch).
These answer "is the adapter's *effect* on the loss growing" — `gate_value` +
`rel_contribution` (this ticket) answer the mechanistic "why/why not" underneath
that. Implement both; they're complementary, not redundant.
