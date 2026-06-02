---
type: decision
status: decided
created: 2026-05-28
decided_at: 2026-05-28
updated: 2026-05-28
target_date:
scope: training
related:
  - "[[../../30_Knowledge/theory/heun-shortcut-target]]"
  - "[[../../30_Knowledge/theory/heun-smoothness-regularizer]]"
  - "[[../../30_Knowledge/theory/shortcut-training]]"
  - "[[../../30_Knowledge/tech/shortcut-training-modes]]"
  - "[[../../20_Tickets/refactor-shortcut-deprecate-twostep-add-heun-smoothness]]"
  - "[[../../20_Tickets/bug-training-shortcut-twostep-no-stepsize-variation]]"
---

# Decision: Deprecate `two_step` as a shortcut training mode; add a Heun-smoothness regularizer in its place

## Status

**Decided 2026-05-28.** Upstream of the implementation ticket
[[../../20_Tickets/refactor-shortcut-deprecate-twostep-add-heun-smoothness]].

## Context

The `two_step` shortcut training mode (current code:
`shortcut_targets.py:33-51`, dispatched at `trainer.py:372-383`) was
introduced as a base-anchored alternative to the paper-faithful
self-consistency target (`distillation` mode). The motivating story
was "the frozen base supplies a deterministic chord-velocity target,
so the adapter can be supervised without risking self-consistency
collapse." See [[../../30_Knowledge/tech/shortcut-training-modes]] for
the original framing.

Two structural problems were surfaced during the math review on
2026-05-28
([[../../60_Updates/entries/2026-05-28-twostep-shortcut-no-stepsize-variation]]):

1. **The DDIM jump is hardcoded to one timestep** (`prev_t = t - 1`).
   The Heun-averaged target therefore approximates the chord velocity
   over `1/T` of the trajectory — *not* over the step sizes the eval
   grid asks about (typically `s = T/N` for `N ≪ T`).
2. **`step_level` is decorative.** The adapter receives it via
   conditioning but the target is independent of it, so the adapter
   has no incentive to learn step-size-conditional behaviour.

The proposed fix in
[[../../20_Tickets/bug-training-shortcut-twostep-no-stepsize-variation]]
was to wire the existing `ShortcutStepSchedule` into the `two_step`
teacher path, sampling `s` per step and scaling the DDIM jump. That
fixes (1) and (2) mechanically, but exposes a deeper limitation that
no amount of plumbing repairs.

## The deeper problem

The frozen base is a **pointwise estimator** of instantaneous velocity
`v(x_t, t)`. It has no information about the chord velocity
`v̄(x_t, t, s) := (1/s) ∫_{t-s}^{t} v(x_τ, τ) dτ` at scales `s ≫ 0`.
Heun's averaging is a 2nd-order Taylor expansion of `v̄` *around the
current point*:

$$
\bar v(x_t, t, s) \;\approx\; \tfrac{1}{2}\big[v(x_t, t) + v(x_{t-s}, t-s)\big] + O(s^3 \cdot \text{curvature}),
$$

which is accurate only when `s` is small or the velocity field is
nearly straight along the chord. For the step sizes few-step inference
asks about (`s` corresponding to 8–32 timestep jumps), the Heun target
degrades to a noisy chord guess: the Euler predictor lands far from
the true endpoint, and `v_base` evaluated at that predicted endpoint
is no longer informative about the true chord velocity.

**Fixing the jump does not fix this.** The Heun-averaged base target
caps at "as good as the base's pointwise predictions allow", which is
fundamentally weaker than what shortcut training is supposed to
provide. The `distillation` mode does not have this ceiling — it
teaches the *adapted* model to produce real chord velocities through
self-consistent recursion, and the anchor branch keeps it grounded
against the data.

So even with the step-size-conditional repair, `two_step` would be:

- dominated by `distillation` at every step size where shortcut
  training matters (large `s`),
- redundant with `distillation`'s anchor branch at small `s` (both
  reduce to the standard diffusion/flow loss when `s` is finest),
- and the source of a category error in the framing of the methods
  chapter: "shortcut training" should mean "training that teaches the
  model about chord velocities at varying scales", which `two_step`
  cannot do regardless of how its jump is wired.

## Options considered

**A. Fix `two_step` as described in the prior ticket.**
Wire the schedule in, scale the jump, make `step_level` load-bearing.
Pro: minimal code change; preserves the option of a base-anchored
shortcut path. Con: solves a mechanical bug but leaves the deeper
"base cannot teach chord velocities" issue. Net: an A/B between
fixed-`two_step` and `distillation` would always favour
`distillation` and the comparison would be misleading.

**B. Delete `two_step` from the shortcut mode dispatch; add a
Heun-derived smoothness regularizer as a separate, generally
applicable loss.**
Pro: cleanly separates two concerns the original `two_step` was
conflating: (i) supervising chord velocities at varying scales —
`distillation`'s job; (ii) regularizing the velocity field to be
smooth for better ODE integration — a general property of *any*
trained predictor, not specific to shortcut training. The Heun
construction maps naturally onto (ii) as a self-distillation along the
predicted trajectory; see
[[../../30_Knowledge/theory/heun-smoothness-regularizer]]. Con: minor
loss of the "base-anchored shortcut mode" option in the future.

**C. Keep `two_step` as a small-weight regularizer alongside
`distillation`.**
Pro: minimum churn. Con: same conceptual confusion remains, and the
"two modes" framing in the YAML and docstrings keeps inviting the
question this decision answers.

## Decision

**Choose B.** Delete `two_step` from the shortcut training mode
dispatch entirely. `distillation` becomes the only shortcut training
mode. Add a separate `heun_smoothness_weight` knob (and matching
loss `compute_heun_smoothness_loss`) that wraps the same Heun
construction as a **self-distillation regularizer** along the
predicted trajectory — derived in
[[../../30_Knowledge/theory/heun-smoothness-regularizer]]. This term
can be on for any training run (vanilla diffusion, vanilla flow,
shortcut) and is orthogonal to `distillation`.

The Heun-smoothness term is structurally similar to the current
`two_step` target — both involve a DDIM micro-step and an MSE between
the predictor's current velocity and the velocity at the predicted
endpoint. The differences that matter:

1. **Teacher is the composed model, not the base.** Regularizes
   `f_θ = f_base + Δ_θ`, not `f_base`. Reflects what the inference
   sampler actually integrates.
2. **Stop-grad on the future-self call.** Symmetric "both with grad"
   formulation has trivial minimizers (equalize via both endpoints);
   stop-grad forces the optimizer to move the *current* prediction
   toward the *future* prediction.
3. **Sampled `s`, not fixed `s = 1`.** Step size drawn from the same
   `ShortcutStepSchedule` already used by `distillation` — multi-scale
   smoothness aligned with eval-time step counts.
4. **Framed as a regularizer (`λ_hs ≪ λ_base`), not a primary
   objective.** Its purpose is ODE-quality of the velocity field, not
   teaching chord velocities.

## Rationale

- **Conceptual clarity.** Shortcut training and ODE-smoothness
  regularization are different goals with different mathematical
  content; conflating them in one mode hid this from us for several
  weeks. The thesis methods chapter is cleaner if "shortcut training"
  means exactly one thing (`distillation`) and other regularizers
  compose on top.
- **Generality.** The smoothness regularizer is useful for any run —
  vanilla world-model training in D2, shortcut training in D3, the
  D4 combination — not just for shortcut. Surfacing it as a separate
  knob makes that obvious.
- **Mathematical honesty.** The framing "`two_step` is base-anchored
  shortcut training" is wrong; the base cannot supply chord-velocity
  supervision. Calling it Heun smoothness instead names what the
  construction actually does.
- **Implementation overhead is negligible.** All the primitives needed
  for the regularizer (`ddim_micro_step_v`, `ShortcutStepSchedule`,
  the `eval()`/`no_grad` plumbing) already exist. Deleting `two_step`
  removes ~50 lines of trainer code; adding the regularizer adds
  ~30. Net code reduction.

## Consequences and migration

- **The two live shortcut configs** must be updated:
  - `configs/diffusion_avid_shortcut_metaworld.yaml` currently sets
    `shortcut_target_method: two_step`. After the refactor: either
    switch to `shortcut_target_method: distillation` (paper-faithful)
    or remove the shortcut weight entirely and turn on
    `heun_smoothness_weight` (if the goal is just smoother
    trajectories, not few-step generation).
  - Same for `configs/diffusion_hyperalign_shortcut_metaworld.yaml`.
- **All shortcut runs logged before the refactor lands** were
  effectively "Heun-smoothness regularization at one fixed scale,
  with a confused name." They are valid baseline numbers for "what
  does the adapter learn when the only signal is smooth-velocity-at-
  finest-scale" — useful as a control in the D3 chapter, not as
  evidence of few-step generation capability.
- The earlier ticket
  [[../../20_Tickets/bug-training-shortcut-twostep-no-stepsize-variation]]
  is **superseded by this decision and the new ticket**. Marked
  `wont-fix-superseded`.
- **`shortcut_target_method` becomes a single-value field** (just
  `distillation`). The dataclass default `"linear"` (`config.py:68`,
  already stale per
  [[../../30_Knowledge/tech/shortcut-training-modes]]) gets removed
  alongside `"two_step"`.

## Open questions to resolve during implementation

- **Stop-grad placement on `v_1` only?** Yes, default. Symmetric
  variants are tracked as a follow-up ablation in
  [[../../30_Knowledge/theory/heun-smoothness-regularizer]] §2 and §7.
- **Default `λ_hs`?** Probably small (≤ 0.1) — regularizer, not
  primary objective. Set empirically on a small smoke run before any
  D2/D3 run picks up the new knob. Not part of this decision.
- **Default schedule for `s` in the regularizer?** Same
  `ShortcutStepSchedule` as `distillation` by default; falls back to
  fixed `s = 1` when no schedule is configured. Lets vanilla runs
  use the regularizer without standing up a schedule.

## Related

- [[../../30_Knowledge/theory/heun-shortcut-target]] — §5 was the
  derivation that exposed the limitation; §6 (the proposed fix) is
  the option this decision deprecates
- [[../../30_Knowledge/theory/heun-smoothness-regularizer]] — formal
  derivation of the replacement
- [[../../30_Knowledge/theory/shortcut-training]] — needs §4.2
  rewritten: `two_step` is no longer presented as a co-equal
  supervision regime
- [[../../30_Knowledge/tech/shortcut-training-modes]] — needs to be
  reduced to a single mode (`distillation`), with `two_step` moved to
  a "history" section
- [[../../20_Tickets/refactor-shortcut-deprecate-twostep-add-heun-smoothness]]
  — the implementation ticket
- [[../../60_Updates/entries/2026-05-28-twostep-shortcut-no-stepsize-variation]]
  — the finding entry that surfaced this
