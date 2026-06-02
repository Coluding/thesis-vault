---
date: 2026-05-28
type: insight
scope: D3  # shortcut adapters
status: action-required
related:
  - "[[../../20_Tickets/bug-training-shortcut-twostep-no-stepsize-variation]]"
  - "[[../../30_Knowledge/theory/heun-shortcut-target]]"
  - "[[../../30_Knowledge/theory/ddim-step-v-parameterisation]]"
  - "[[../../30_Knowledge/tech/shortcut-training-modes]]"
---

# `two_step` shortcut training has no step-size signal

## What we found

The `two_step` shortcut target as currently coded is **not a shortcut
target in the usual sense** — it is a Heun-quality velocity prior at a
single hardcoded scale (one timestep). The DDIM jump
`prev_t = (t - 1).clamp_min(0)` in
`shortcut_targets.py:45` and `trainer.py:528` decides this — once
the jump is fixed, the supervisory target carries no information about
step size.

Concretely:

- The injected `step_level` is sampled but the target ignores it.
- The adapter therefore receives **no gradient signal** that would
  cause it to behave differently at different `step_level` values.
- At inference, any few-step rollout (`N < T`) asks the adapter for
  jumps that were never trained — every `log_step_size_grid` panel
  produced so far is an out-of-distribution probe.

## How we found it

Reading through `_compute_two_step_target_v` while tracing the math of
`ddim_micro_step_v` end-to-end (decompose → recompose) for thesis
documentation. The math is fine, the function it wraps is fine, but
the *interval* over which Heun averages is wrong — one timestep
instead of `s` timesteps drawn from the schedule.

The earlier ticket
[[../../20_Tickets/bug-training-shortcut-target-timestep]] (which
flagged the wrong time argument on the second base call) is
**structurally resolved** by the switch to `ddim_micro_step_v`. That
ticket is now marked superseded by the new one.

## Implications for D3

- Every shortcut run logged so far (both `diffusion_avid_shortcut_metaworld`
  and `diffusion_hyperalign_shortcut_metaworld`) was trained under
  this no-step-size-variation regime. Numbers from those runs
  characterise *the adapter at the finest scale*, not few-step
  generation quality.
- The headline D3 framing "`two_step` = base-anchored shortcut training
  with no collapse risk" needs to be qualified or the bug fixed before
  the framing holds.
- The eval-side ticket
  [[../../20_Tickets/risk-shortcut-eval-steplevel-out-of-distribution]]
  is the symptom of the same root cause. Both should close together.

## Decision needed

The fix is small (sample `s` from the existing
`ShortcutStepSchedule`, scale the DDIM jump, inject the matching
`step_level` — full plan in
[[../../20_Tickets/bug-training-shortcut-twostep-no-stepsize-variation]]).
Open questions before landing:

- Per-batch vs per-sample `s` for the new `two_step` (paper convention
  is per-batch).
- Cap on the schedule's `max` to keep Heun's quadrature error
  acceptable at large `s` — empirically 1/8 is probably safe, but
  needs measurement.
- Whether to ship the trainer change together with the config update,
  or stagger them (the legacy fallback in step 1 of the plan keeps old
  configs producing old behaviour, so staggering is safe).

## What we updated in the vault

- New: [[../../30_Knowledge/theory/ddim-step-v-parameterisation]] —
  rigorous derivation of the DDIM single-step primitive in
  v-parameterisation (decompose + recompose, orthogonal rotation in the
  `(x_0, ε)` plane).
- Rewrote: [[../../30_Knowledge/theory/heun-shortcut-target]] — Heun
  derivation with full Taylor expansion, current-code mapping, and
  the new §5–6 documenting the limitation and the fix.
- Updated: [[../../30_Knowledge/tech/shortcut-training-modes]] — added
  the hardcoded-jump gotcha and bumped the priority of the
  follow-up to "ship before any D3 A/B."
- New ticket:
  [[../../20_Tickets/bug-training-shortcut-twostep-no-stepsize-variation]].
- Marked superseded:
  [[../../20_Tickets/bug-training-shortcut-target-timestep]].
