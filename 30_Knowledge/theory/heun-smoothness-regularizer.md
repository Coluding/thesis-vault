---
type: theory
last_updated: 2026-05-28
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

### Equivalence to "deviation from Heun average"

Penalizing the deviation of `v_0` from the Heun-averaged velocity is
equivalent up to a constant:

$$
\left\| v_0 - \tfrac{1}{2}(v_0 + v_1) \right\|^2 \;=\; \tfrac{1}{4} \| v_0 - v_1 \|^2.
$$

Either form is fine; (S) is preferred because it is the most direct
discrete material-derivative penalty and its scaling is interpretable
(`L_smooth / s²` ≈ squared material derivative).

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
| **Sampled from `ShortcutStepSchedule`** | `s ∼ schedule.sample()` | Same as `distillation` per call | Multi-scale smoothness — large `s` rungs carry stronger signal because the field gets more time to vary. Aligned with eval-time step sizes. |
| **Fixed-coarse** | `s = T / N_eval` for a target eval step count | Same as above | Pre-trains smoothness at exactly the step size you intend to deploy. Bias-toward-deployment but no multi-scale defense. |

Default recommendation: **sampled from the existing
`ShortcutStepSchedule`** so the regularizer scales naturally with what
inference is going to ask. Coincidentally lets the same eval grid
(`log_step_size_grid`) measure regularized vs. unregularized runs on
identical step counts.

## 5. Interaction with the rest of the loss

The full training loss becomes

$$
L \;=\; L_{\text{base}} \;+\; \lambda_{\text{sc}} L_{\text{shortcut-direction}} \;+\; \lambda_{\text{hs}} L_{\text{heun-smooth}},
$$

where `L_base` is the standard diffusion/flow loss, `L_shortcut-direction`
is the `distillation` self-consistency term (or zero if shortcut training
is off), and the new `λ_hs L_heun-smooth` is the smoothness penalty.

- **`L_base` and `L_heun-smooth` are compatible.** The base loss
  supervises pointwise velocity from data; the smoothness term
  supervises pairwise velocity along the predicted trajectory. They
  can both be on at any nonzero weight.
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

## 7. What is *not* claimed here

- _needs verification_: empirical magnitude of the integration-error
  reduction from `L_heun-smooth` on a fixed inference step count.
  Plausibly meaningful at small `N`; plausibly negligible at large
  `N` where the per-step error is already in noise.
- _needs verification_: whether (S) helps or hurts shortcut training
  specifically — by smoothing the velocity field, it may make
  self-consistency easier (smooth fields are easier to learn averages
  of) or harder (the chord velocity at large `s` is *not* a smooth
  function of `(x, t)` in general). Ablation needed.
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
