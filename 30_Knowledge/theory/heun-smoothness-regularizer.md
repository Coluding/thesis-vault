---
type: theory
last_updated: 2026-06-04
sources:
  - "[[ddim-step-v-parameterisation]]"
  - "[[heun-shortcut-target]]"
  - "[[shortcut-training]]"
  - "[[prediction-objectives]]"
---

# Heun-derived velocity-field smoothness regularizer

> A loss term that pushes the composed model's predicted velocity field
> to be **locally smooth along its own predicted ODE trajectory**. The
> functional comes from the same Heun construction that motivated the
> `two_step` shortcut target, but the role is different: this is a
> general-purpose regularizer for ODE quality, applicable to any
> training run (vanilla diffusion / flow / shortcut), **not a shortcut
> training method**. The accompanying decision to deprecate the
> `two_step` shortcut mode is recorded in
> [[../../50_Decisions/decided/deprecate-twostep-shortcut-mode]].

> **Resolved framing (grilling session, 2026-06-03/04) — read this first.**
> The body below still carries the original "general multi-scale deployment
> regularizer" framing; the session refined it. The settled position:
>
> 1. **Loss form:** normalise by `s²` — `L = w(t,s)·‖v₀−sg(v₁)‖²/s² ≈ ‖Df/Dt‖²`.
>    The raw `‖v₀−v₁‖²` is `≈ s²‖Df/Dt‖²`, so it goes inert at small `s` (the
>    measured `~1e-7` at `jump=1` is this artifact, **not** a healthy small
>    signal) and is dominated by the top rung when `s` is sampled. `/s²` removes
>    the `s`-dependence and decouples `λ_hs` from the schedule range. See §2.
> 2. **`s` vs `d` are different things.** `s` = the finite-difference *probe
>    window* (lives only in `ddim_micro_step_v` + the `/s²`); the **model is
>    never conditioned on `s`**. `d` = the *step-size conditioning input* to the
>    model. In the regularizer, evaluate `v₀, v₁` at **`d=0` (instantaneous)** —
>    never feed `d=s`, which would smooth the *chord* field and fight the
>    shortcut mechanism.
> 3. **Division of labour (the key refinement).** Small-scale instantaneous
>    curvature → the Heun term (light, `d=0`). **Big-jump accuracy → the chord /
>    `d`-conditioning + self-consistency, NOT this term.** The chord velocity
>    *learns the averaged jump directly*, absorbing big-jump curvature instead of
>    flattening it. We do **not** extend Heun smoothness to large `s` to chase
>    big-jump smoothness — that would be the wrong (and base-fighting) tool.
> 4. **Frozen-base caveat.** `f_θ = f_base(frozen) + Δ`, and the instantaneous
>    curvature is **base-dominated and data-faithful**. Flattening the *composed*
>    field forces `Δ` to *cancel* the base's curvature → fidelity loss + waste of
>    scarce adapter capacity. So scope the term as "**don't let `Δ` *add*
>    curvature**" (an adapter-excess limiter, `‖DΔ/Dt‖²`), not "force the composed
>    field flat." Demote it to a **D2 baseline** regularizer / optional small D3
>    adapter prior; the chord conditioning is the D3/D4 few-step lever.
> 5. **Usable `s` band is *moderate*.** Tiny `s` → catastrophic cancellation
>    (differencing two `O(1)` velocities); huge `s` → `x_{t−s}` off-manifold,
>    `v₁` unreliable. So cap `high`, and replace the `jump=1` fallback with a
>    moderate default (or refuse-to-start when `λ_hs>0` with no schedule).

## Convention

Diffusion convention (see [[ddim-step-v-parameterisation]] §0):
`t = 0` data, `t = T` noise; sampling decreases `t`; step `s` is
positive in the sampling direction. The composed predictor under
v-parameterisation is

$$f_\theta(x, t, c) \;=\; f_{\text{base}}(x, t) \;+\; \Delta_\theta(x, t, c),$$

with `c` collecting any conditioning (actions, step size, etc.). The
base is frozen; only `Δ_θ` is trainable.

## 1. The objective — ODE-quality, not chord-velocity

The model defines a deterministic ODE $\dot x = -f_\theta(x, t, c)$
(sign convention follows the sampling direction). An N-step DDIM
sampler is the first-order exponential integrator for this ODE; its
global error after rolling from `t = T` to `t = 0` decomposes into

$$
\text{global error} \;\propto\; \sum_{n=1}^{N} \underbrace{O(h_n^2)}_{\text{local Euler error}} \cdot \underbrace{\Big|\frac{D f_\theta}{D t}\Big|}_{\text{velocity-field curvature}} + \text{(higher order)},
$$

where $D / D t = \partial_t + f \cdot \nabla_x$ is the **material
derivative** of `f_θ` along the predicted trajectory. The local
truncation error per step is bounded above by the magnitude of this
material derivative — flat (= constant along the trajectory) velocity
fields integrate exactly with one Euler step, regardless of step size;
curved fields accumulate error linearly in the step count.

For a *fixed* number of network calls, the only knob the trainer has to
reduce inference-time integration error is to **regularize the
material derivative of `f_θ` toward zero along its own predicted
trajectory**.

This is what `L_heun_smooth` does.

## 2. Discrete formulation

The material derivative is approximated to first order by a finite
difference along one DDIM step. At a sampled `(x_t, t, c)` with step
size `s` drawn from a schedule:

```
v_0     = f_θ(x_t, t, c)                                    [with grad]
x_{t-s} = ddim_micro_step_v(x_t, v_0, t, t-s)               [differentiable but no extra learning signal]
v_1     = f_θ(x_{t-s}, t-s, c)                              [no grad — used as a "future-self" reference]
```

Then

$$
\boxed{ L_{\text{heun-smooth}}(\theta) \;:=\; \big\| v_0 \;-\; \operatorname{sg}(v_1) \big\|^2. } \tag{S}
$$

`sg` denotes stop-gradient. The discrete material derivative of `f_θ`
along the one-step trajectory is exactly `(v_1 - v_0) / s`; squaring
and dropping the constant factor recovers (S).

--> This should decrease the curvature since (v1-v0)/s is the approximate derivative $D / D t$  if s is small enough. We just ignore the scaler s since it does not matter during optimization. 

  ==L_heun-smooth = ‖v₀ − sg(v₁)‖² isn't an arbitrary "make consecutive velocities agree" penalty. It's the discrete material derivative: (v₁ − v₀)/s ≈ Df/Dt, squared . And Df/Dt is  exactly the curvature term in the global-error bound from above.==


### Equivalence to "deviation from Heun average"

Penalizing the deviation of `v_0` from the Heun-averaged velocity is
equivalent up to a constant:

$$
\left\| v_0 - \tfrac{1}{2}(v_0 + v_1) \right\|^2 \;=\; \tfrac{1}{4} \| v_0 - v_1 \|^2.
$$

Either form is fine; (S) is preferred because it is the most direct
discrete material-derivative penalty and its scaling is interpretable
(`L_smooth / s²` ≈ squared material derivative).

### Design decision (2026-06-03): normalise by `s²` when `s` is sampled

We promote the `/s²` form from "interpretation" to the actual loss:

$$
L_{\text{heun-smooth}}(\theta) \;:=\; \frac{\big\| v_0 - \operatorname{sg}(v_1) \big\|^2}{s^2} \;\approx\; \Big\| \tfrac{D f_\theta}{D t} \Big\|^2.
$$

This refines the "we just ignore the scalar `s`" margin note above: **`s` is
ignorable only when it is fixed** (then it is a constant absorbed into
`λ_hs`). Once `s` is *sampled* per example, `/s²` varies sample-to-sample and
is no longer absorbable — it reweights examples, which is the behaviour we
want. Reasoning (grilling session, 2026-06-03):

- **`s` is not a model input in this regularizer.** `f_θ(x,t,c)` is the same
  instantaneous-velocity field regardless of which `s` is sampled (§3 — `s`
  sets the *interval*, not the *task*). So the curvature `Df/Dt` is a fixed
  local property and does **not** grow with `s`. The raw `‖v₀−v₁‖² ≈ s²‖Df/Dt‖²`
  growth is a **measurement-window artifact**, not "large steps designed to
  bend more." Dividing by `s²` removes exactly that artifact.
- **`ShortcutStepSchedule.sample()` draws each dyadic rung with equal
  probability** (`step_schedule.py:142`, `mode="log2"`). Under raw `‖v₀−v₁‖²`
  the expected loss is dominated by the top rung by a factor `~(high/low)²`,
  starving the finer scales. `/s²` makes every rung report the same geometric
  quantity, so all scales contribute comparably.
- **`λ_hs` decouples from the `low/high` range.** With raw, retuning `high`
  silently rescales the penalty; with `/s²` it does not. Main practical win.

**Honest cost (recorded, not yet resolved):** `/s²` de-emphasizes the coarse
steps that dominate *deployment* integration error (per-step local error is
`½s²·Df/Dt`, which raw self-weights toward). If D3 later commits to a specific
deploy step count `N`, bias the `s` *sampling* toward that `s` rather than
reverting to raw — keep the estimand (`Df/Dt`) scale-clean, move the sampling
weight instead. See the open `s`-sampling question in §4.

### Why stop-grad on `v_1`

Treating `v_1` as a no-grad reference makes (S) a *self-distillation*
of the velocity at the predicted endpoint into the velocity at the
current point. Three motivations:

1. **Avoids backprop through `ddim_micro_step_v`**, which would
   otherwise double the backward-pass cost and introduce second-order
   gradient pathways through the schedule lookups.
2. **Matches the established teacher–student pattern** used in
   `distillation` (`trainer.py:536-565`); the trainer's
   `eval()`/`no_grad` plumbing applies unchanged.
3. **Prevents trivial collapse via `v_1 ↘ v_0`**: with both endpoints
   trainable, the optimizer can equalize them by adjusting both, which
   does not necessarily reduce true field curvature. With `v_1`
   detached, the only way to reduce (S) is to move `v_0` toward the
   model's *current* prediction at the predicted endpoint — a real
   smoothness condition.

## 3. What this regularizer is *not*

- **Not a shortcut training objective.** It does not condition on step
  size in a way that lets the model produce different chord velocities
  at different `s` — the sampled `s` only changes the *interval* over
  which smoothness is enforced, not the *task* the model is supervised
  on. Shortcut training (`distillation`) supervises the model to
  produce chord velocities; (S) supervises the model to produce
  *consistent* instantaneous velocities along its trajectory.
- **Not equivalent to `two_step`.** `two_step` (now deprecated, see
  [[../../50_Decisions/decided/deprecate-twostep-shortcut-mode]]) is
  base-anchored: target `= ½(v_base(x_t,t) + v_base(x_mid, t-1))`.
  Replacing the base with `f_θ` and using `sg` on the second call would
  give Heun smoothness at a single hardcoded scale — almost (S) but
  fixed at `s = 1`. (S) generalises this to a sampled `s`.
- **Not Lipschitz regularization.** Lipschitz penalties bound
  `‖∇_x f‖` *uniformly*; (S) bounds the material derivative *along the
  predicted trajectory*. The latter is weaker (only the directional
  derivative along `f` is penalized) but cheaper and better-aligned
  with what the ODE solver actually integrates over.
- **Not a score-matching curvature term.** Score-matching variants
  that penalize `tr(∇_x s)` are about the divergence of the score
  field; (S) is about the rate of change of `f` along trajectories.
  These are distinct geometric properties.

## 4. Sampling `s` — three regimes

The step size `s` over which smoothness is enforced is a design knob:

| Regime | `s` | Cost | Signal characteristic |
|---|---|---|---|
| **Fixed-fine** | `s = 1` (one timestep) | Cheapest — both calls at adjacent timesteps; minimal numerical cost difference vs. standard loss | Always-on, low-magnitude — encodes "be locally smooth at the schedule resolution". Useful as a default baseline regularizer. |
| **Sampled from `ShortcutStepSchedule`** | `s ∼ schedule.sample()` | Same as `distillation` per call | Multi-scale smoothness — see corrected justification below. Aligned with eval-time step sizes. |
| **Fixed-coarse** | `s = T / N_eval` for a target eval step count | Same as above | Pre-trains smoothness at exactly the step size you intend to deploy. Bias-toward-deployment but no multi-scale defense. |

### Why multi-scale `s` (corrected justification, 2026-06-03)

> The earlier rationale — *"large `s` rungs carry stronger signal because the
> field gets more time to vary"* — is **stale under the `/s²` loss** (§2). Once
> we divide out the window, every rung reports an estimate of the *same*
> `‖Df/Dt‖²`, so large `s` carries *equal*, not stronger, signal. The reason to
> keep multi-scale is **arc-coverage**, not signal-strength.

`(v₁−v₀)/s` is the *average* curvature accumulated over the arc `[t−s, t]`
(mean-value theorem), **not** the pointwise curvature at `x_t`:

- **small `s`** supervises local smoothness — "is the field flat *right here*";
- **large `s`** supervises arc-averaged smoothness — "is the field flat
  *averaged over the long arc a coarse sampler jumps across*."

Deployment takes finite jumps, so we want smoothness over finite arcs of the
lengths we deploy at; multi-scale sampling covers the spectrum, with the fine
rungs pinning local smoothness as a floor.

**Why this does *not* argue for the raw (un-`/s²`) loss.** Larger steps do
incur larger integration error (`½s²κ`), but the loss's job is to estimate and
kill the *curvature* `κ` that causes it, not to *measure* the error. Drive
`κ → 0` and the large-`s` error vanishes for free (`½s²κ → 0`). So the
deployment-relevant error is best reduced by estimating `κ` *scale-cleanly*
(`/s²`), not by up-weighting large `s` — up-weighting only adds gradient
variance to the same curvature direction.

**Cap the top rung.** At large `s`, `x_{t−s}` is a far, Euler-*predicted*
endpoint that may drift off the data manifold; `v₁` there is a noisier, biased
`κ` proxy ("comparing velocities at non-comparable locations"). Bound `high` so
the secant stays a trustworthy curvature estimate — this is a reason to **cap
the range**, not to reweight scales.

Default recommendation: **sampled from the existing
`ShortcutStepSchedule`** (with a capped `high`) so the regularizer covers the
arc lengths inference will ask for. Coincidentally lets the same eval grid
(`log_step_size_grid`) measure regularized vs. unregularised runs on identical
step counts.

### Loss weighting `w(t, s)` (design decision, 2026-06-03)

The full per-sample term is an importance weight **on top of** the scale-clean
core:

$$
L_{\text{heun-smooth}} \;=\; w(t,s)\cdot\frac{\big\| v_0 - \operatorname{sg}(v_1)\big\|^2}{s^2}, \qquad w(t,s)\equiv 1 \text{ by default.}
$$

- **`/s²` stays in the core, `w` is separate.** `/s²` is *what we measure*
  (the curvature; §2) and is not optional. `w` is *how much we care about this
  `(t,s)` sample*. Folding `/s²` into `w` would tangle the curvature estimand
  with the importance weight and destabilise the meaning of `λ_hs`. Keep them
  distinct.
- **`t`-shaping — high at noise, taper toward data.** Convention reminder
  (§Convention): `t=0` data, `t=T` noise, sampling runs `T→0`. The chosen shape
  is **high weight near `t→T` (noise / start of sampling), decaying toward
  `t→0` (data / end)**. Rationale = the (a) mild-trim stance: smooth hardest
  where curvature is *affordable* (noise/mid region, which is also where a
  coarse sampler takes its big jumps, so the few-step payoff is largest) and
  back off where curvature is *irreducible and fidelity-critical* (near data,
  where smoothing would erase the mode structure and where `x_{t−s}` is most
  likely to overshoot onto the wrong mode). A flat `w` spends its harm budget
  in the worst place.
  - **Parameterise the decay on log-SNR, not raw `t`.** A monotone
    log-decay (high→low) keyed to **log-SNR** is invariant to the noise
    schedule; raw-`t` weighting silently changes meaning when the schedule
    changes. Same shape the user described ("inverse-logarithmic decay"), made
    portable.
- **`s`-dependence of `w` — default flat; reserve for a *soft cap*.** Since
  `/s²` already handles scale, `w` does not need an `s`-term for normalisation.
  Its one honest use is a soft version of the `high` cap (§ above): taper `w`
  down at large `s`, where `x_{t−s}` drifts off-manifold and the secant becomes
  an unreliable curvature proxy. Default `w` flat in `s`; the `t`-axis carries
  the shaping.
- **Implementation: a weight, not a resampling.** The shaping is applied as
  the multiplicative `w(t,s)` on the per-sample loss, *not* by biasing the
  `t`/`s` sampling distributions. (Resolves the earlier "sampling-side vs
  weight-side" sub-question in favour of weight-side — it keeps the eval-grid
  alignment intact and the loss explicit.)

**Honest hedge.** High-at-noise is a *bet* that few-step error is dominated by
smooth-region jumps. If the data-end curvature turns out to dominate the
coarse-step error, some near-data smoothing (with its fidelity cost) becomes
necessary. The `w(t,s)` profile is therefore an **ablation axis**, not a fixed
truth. _needs verification_ on the DynamiCrafter base / MetaWorld.

## 5. Interaction with the rest of the loss

The full training loss becomes

$$
L \;=\; L_{\text{base}} \;+\; \lambda_{\text{sc}} L_{\text{shortcut-direction}} \;+\; \lambda_{\text{hs}} L_{\text{heun-smooth}},
$$

where `L_base` is the standard diffusion/flow loss, `L_shortcut-direction`
is the `distillation` self-consistency term (or zero if shortcut training
is off), and the new `λ_hs L_heun-smooth` is the smoothness penalty.

- **`L_base` and `L_heun-smooth` are in genuine (but, at small `λ_hs`,
  mild) tension — not freely compatible (corrected 2026-06-03).** `Df/Dt = 0`
  everywhere means straight constant-speed trajectories (a *straight* / rectified
  transport). The true marginal-preserving field that `L_base` pins `f_θ` to has
  *irreducible* curvature (multimodal data ⇒ bending trajectories). So the two
  terms pull against each other at every `s`: `L_base` wants the data-coupling
  velocities, `L_heun-smooth` wants to straighten. This is a **fidelity ↔
  few-step-accuracy tradeoff**, dialled by `λ_hs`, *not* a free lunch.
  - **Design stance (2026-06-03): (a) mild trim, not straightening.** `λ_hs`
    is kept small so the term shaves only *excess, adapter-induced* wiggle that
    the data coupling does not require, buying few-step accuracy at a small
    fidelity cost. The "result" of the term is the **Pareto curve** (sample
    quality vs. step count), reported as such.
  - **What we are *not* doing: (b) aggressive straightening.** A large `λ_hs`
    would change the coupling and pull endpoints off the true marginals; doing
    that *correctly* needs a reflow/rectification step (regenerate (noise,data)
    pairs from the current model, retrain `L_base` on them) — a heavier,
    different method that would bloat D3's scope. Out of scope here.
  - Note the two tensions are on **different axes** and need **different
    counters**: the `s`-axis measurement-window growth is countered by `/s²`
    (§2); this `L_base` fidelity tension is countered only by small `λ_hs`.
    Neither counter substitutes for the other.
- **`L_shortcut-direction` and `L_heun-smooth` are compatible but
  redundantly constrain at small `s`.** The anchor branch of
  `distillation` already gives `L_base` at the smallest step;
  `L_heun-smooth` at the same step adds smoothness on top. At large
  `s`, the two are orthogonal — shortcut supervises *what* the
  averaged velocity should be, smoothness supervises *that the
  velocity field varies slowly* (which is a property of the integrand,
  not the integral).
- **No collapse risk.** Both `L_base` (in standard runs) and the
  anchor branch (in shortcut runs) keep `f_θ` tied to real data
  velocities. `L_heun-smooth` alone has trivial minimizers (constant
  fields), but those are excluded by the data-anchored term.

## 6. Implementation hooks

The regularizer needs four pieces, all already present in different
forms elsewhere in the trainer:

| Piece | Reuse from |
|---|---|
| Sampling `s` and converting to a timestep jump | `step_schedule.py:142` `ShortcutStepSchedule.sample` and `:134` `to_timestep_jump` — same path `distillation` uses |
| Running one DDIM step from `(x_t, v_0)` | `shortcut_targets.py:81-126` `ddim_micro_step_v` — already differentiable |
| Calling the composed model in `no_grad` for `v_1` | `trainer.py:552-564` `_compute_self_consistency_target_v` — same `eval()` toggle pattern |
| Adding a weighted MSE term to the loss | `trainer.py:200-214` weighted loss application pattern |

No new primitives; the regularizer is ≈30 lines of trainer code plus a
config field and a loss function in `losses/consistency.py`.

Implementation plan and integration with the existing trainer is in
[[../../20_Tickets/refactor-shortcut-deprecate-twostep-add-heun-smoothness]].

### Delta against the current implementation (observed 2026-06-04)

The `heun_smoothness` loss already exists but does **not** yet match the
resolved design above:

- **Currently raw, no `/s²`.** Called as
  `LossRegistry.get_consistency_loss("heun_smoothness")(v0, v1.float())` — no
  `s`, no `t`. Add the `/s²` normalisation and the `w(t,s)` weight (§2, §4).
- **`jump=1` fallback is doubly wrong.** When no `shortcut_step_schedule` is
  configured, `_compute_heun_smoothness` falls back to `jump=1` (one of 1000
  timesteps, `s≈1e-3`). Observed loss there is `~1e-7` — *inert* under raw (the
  `s²` artifact), and would be *precision-fragile* under `/s²` (catastrophic
  cancellation). Fix: moderate default jump, or refuse-to-start with
  `λ_hs>0` and no schedule. The fallback is "no step-size source configured,"
  not "smallest schedule level."
- **`d`-conditioning:** ensure both `v₀, v₁` evaluate at `d=0` when shortcut
  conditioning is active (do not inherit the batch's `d`).
- **Confirmed:** `ddim_micro_step_v(x, v, t, prev_t, …)` contains **no model
  call** — `s` enters only as the `t→t−s` timestep gap. The model never sees
  `s`. (`shortcut_targets.py:81-126`.)

## 7. What is *not* claimed here

- _needs verification_: empirical magnitude of the integration-error
  reduction from `L_heun-smooth` on a fixed inference step count.
  Plausibly meaningful at small `N`; plausibly negligible at large
  `N` where the per-step error is already in noise.
- _partially resolved (2026-06-03/04)_: the shortcut-interaction question is
  no longer "at which `d` do we smooth?" — settled as **`d=0`, small-scale
  only; the chord conditioning owns big jumps** (see Resolved-framing block,
  items 2–4). What remains _needs verification_ is the *empirical* sign and
  size of the effect: does a light `d=0` smoothness prior make self-consistency
  converge faster/cleaner, or is it net-neutral once the chord model is doing
  the integration work? Ablation needed (`λ_hs ∈ {0, small}` × shortcut on).
- _needs verification_: optimal `λ_hs` and the right baseline `s`
  schedule. Almost certainly `λ_hs ≪ λ_base` (regularizer, not
  primary objective). Probably `λ_hs ≪ λ_sc` when shortcut training
  is on. Numbers TBD.
- No quantitative claim that this term is better than alternative
  smoothness penalties (Lipschitz, Jacobian, score-curvature). Heun
  smoothness is *natural* for ODE-quality but is one of several
  options.

## Related

- [[ddim-step-v-parameterisation]] — the primitive `ddim_micro_step_v`
  used to compute `x_{t-s}` from `(x_t, v_0)`
- [[heun-shortcut-target]] — the Heun derivation; this regularizer is
  the same construction, re-scoped from "supervisory target" to
  "regularization penalty"
- [[shortcut-training]] — the shortcut training objective that (S) is
  *not* a substitute for; the two compose orthogonally
- [[../../50_Decisions/decided/deprecate-twostep-shortcut-mode]] — the
  decision to delete `two_step` as a shortcut mode in favour of this
  regularizer plus `distillation`
- [[../../20_Tickets/refactor-shortcut-deprecate-twostep-add-heun-smoothness]]
  — implementation plan
- Code: `src/generative_flow_adapters/training/shortcut_targets.py:81-126`
  (`ddim_micro_step_v`)
- Code: `src/generative_flow_adapters/training/step_schedule.py`
  (`ShortcutStepSchedule`)
- Code: `src/generative_flow_adapters/losses/consistency.py` (where
  the new loss function will land)
