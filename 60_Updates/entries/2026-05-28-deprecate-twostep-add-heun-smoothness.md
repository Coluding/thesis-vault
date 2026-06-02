---
date: 2026-05-28
type: decision
scope: D3  # shortcut adapters
status: decided
related:
  - "[[../../50_Decisions/decided/deprecate-twostep-shortcut-mode]]"
  - "[[../../20_Tickets/refactor-shortcut-deprecate-twostep-add-heun-smoothness]]"
  - "[[../../30_Knowledge/theory/heun-smoothness-regularizer]]"
  - "[[../../30_Knowledge/theory/heun-shortcut-target]]"
  - "[[2026-05-28-twostep-shortcut-no-stepsize-variation]]"  # the morning's finding
---

# Deprecate `two_step` shortcut mode; add a Heun-smoothness regularizer instead

## Decision

`two_step` is removed from the shortcut training mode dispatch.
`distillation` becomes the only shortcut training mode. A separate,
opt-in `heun_smoothness_weight` is added that wraps the same Heun
construction as a **self-distillation regularizer along the predicted
trajectory** — applicable to any training run, not just shortcut.

Full record: [[../../50_Decisions/decided/deprecate-twostep-shortcut-mode]].
Implementation plan: [[../../20_Tickets/refactor-shortcut-deprecate-twostep-add-heun-smoothness]].

## Why we changed our minds

This morning's finding
([[2026-05-28-twostep-shortcut-no-stepsize-variation]]) was that the
DDIM jump in `two_step` is hardcoded to 1 timestep, so `step_level` is
decorative and no step-size-conditional supervision exists. The
proposed fix was to wire the existing `ShortcutStepSchedule` into the
teacher path.

In the afternoon discussion we surfaced that the mechanical fix would
not address the actual problem:

- The frozen base is a **pointwise** velocity estimator. It has no
  information about chord velocities at scales `s ≫ 0`.
- Heun's averaging of two base samples is a 2nd-order Taylor expansion
  of the chord velocity *around the current point* — accurate only
  when `s` is small or the field is locally straight.
- At the step sizes few-step inference asks about (typically 8–32
  timestep jumps), the Euler predictor lands far from the true
  endpoint, and `v_base` at that point is no longer informative about
  the true chord velocity. The target degrades to a noisy guess.
- `distillation` does not have this ceiling — the *adapted* model can
  learn structure the base never had, and self-consistency bootstraps
  chord-velocity predictions through the recurrence
  `s(x, t, 2d) ≈ ½[s(x, t, d) + s(x', t-d, d)]`. The anchor branch
  keeps it grounded against real data.

So even after the mechanical fix, a `two_step` vs `distillation` A/B
would always favour `distillation`, and the comparison would mislead
about *why* (apparent: "the methods differ"; actual: "one of them
cannot do the task at all").

## What `two_step` actually does — re-framed

Stripping the misleading "shortcut" label, the `two_step` construction
is recognisable as **Heun-derived velocity-field smoothness
regularization**: it penalizes the model's deviation from the
Heun-averaged velocity over a one-timestep interval, which is the
discrete material derivative `Df/Dt` along the predicted trajectory.

That *is* a useful term — a smoother velocity field integrates more
accurately at any step count. The right place for it is as a
**separate, generally-applicable regularizer** with its own weight
knob, available for any run (vanilla diffusion, vanilla flow, shortcut).
Not as a shortcut training mode.

The formal derivation lives in the new theory note
[[../../30_Knowledge/theory/heun-smoothness-regularizer]].

## Differences between the new regularizer and old `two_step`

| Aspect | Old `two_step` | New `heun_smoothness` |
|---|---|---|
| Role | "Shortcut training mode" (primary objective) | Regularizer (small weight, opt-in) |
| Teacher | Frozen base `v_base` | Composed model `f_θ = f_base + Δ_θ` |
| Step size | Fixed `s = 1` timestep, hardcoded | Sampled from `ShortcutStepSchedule` |
| `step_level` injected? | Yes, but decorative | No — smoothness scale ≠ task knob |
| Stop-grad on the future call? | Implicit (target is `.detach()`-ed) | Explicit `sg(v_1)` — symmetric variant rejected to avoid trivial collapse |
| Default weight | 1.0 (loss-dominating) | 0.0 (opt-in); recommended ≤ 0.1 if turned on |
| Composes with `distillation`? | Mutually exclusive | Orthogonal — both can be on |

## Implications for runs and writing

- **All shortcut runs logged before the refactor lands** were
  effectively "Heun-smoothness regularization at one fixed scale,
  misnamed as shortcut training." They are valid as a **control
  baseline** in the D3 chapter — "what does the adapter learn when
  the only signal is local smoothness at the finest scale, without
  any step-size conditioning?" — not as evidence of few-step
  generation capability.
- The two live shortcut configs
  (`diffusion_avid_shortcut_metaworld.yaml`,
  `diffusion_hyperalign_shortcut_metaworld.yaml`) need to be migrated
  to `distillation`. Tracked in patch 2 of the implementation ticket.
- The D3 methods chapter framing is now cleaner:
  - **Shortcut training** = `distillation` mode, paper-faithful
    self-consistency, end of story.
  - **Heun smoothness** = orthogonal regularizer on velocity-field
    quality, applicable to any model, motivated by ODE-integration
    error analysis (see [[../../30_Knowledge/theory/heun-smoothness-regularizer]] §1).
  - These are different mathematical objects with different
    motivations; they happen to share the same underlying primitive
    (`ddim_micro_step_v` + an MSE on velocities) but supervise
    different things.

## Vault changes today

Already landed:

- New theory note:
  [[../../30_Knowledge/theory/ddim-step-v-parameterisation]] —
  rigorous derivation of the DDIM single-step primitive (this
  morning).
- Rewritten theory note:
  [[../../30_Knowledge/theory/heun-shortcut-target]] — Heun derivation
  with full Taylor expansion and limitation analysis (this morning).
- New theory note:
  [[../../30_Knowledge/theory/heun-smoothness-regularizer]] — formal
  derivation of the new regularizer (this afternoon).
- New decision record:
  [[../../50_Decisions/decided/deprecate-twostep-shortcut-mode]].
- New refactor ticket:
  [[../../20_Tickets/refactor-shortcut-deprecate-twostep-add-heun-smoothness]].
- Updated catalogue:
  [[../../30_Knowledge/tech/shortcut-training-modes]] — flagged the
  hardcoded-jump limitation; bumped the follow-up priority. Will
  collapse to one mode after the refactor lands.
- Superseded ticket:
  [[../../20_Tickets/bug-training-shortcut-target-timestep]] — closed
  as structurally fixed (jump-to-`prev_t` switch already landed in an
  earlier commit).
- Superseded ticket:
  [[../../20_Tickets/bug-training-shortcut-twostep-no-stepsize-variation]] —
  the "fix the jump" plan, replaced by the deprecation decision.

Pending (after the refactor patches land):

- [[../../30_Knowledge/theory/shortcut-training]] §4.2 — currently
  presents `two_step` and `distillation` as co-equal regimes; needs
  rewriting to reflect the deprecation.
- [[../../30_Knowledge/tech/shortcut-training-modes]] — collapse to
  one active mode + add a `heun_smoothness` section.
- [[../../30_Knowledge/writing/explainer-shortcut-training]] and
  the shortcut-training figure — relabel the Heun construction from
  "shortcut training" to "smoothness regularizer", or split into two
  explainers.
