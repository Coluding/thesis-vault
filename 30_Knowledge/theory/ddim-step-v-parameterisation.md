---
type: theory
last_updated: 2026-05-28
sources:
  - "code: src/generative_flow_adapters/training/shortcut_targets.py"
  - "[[prediction-objectives]]"
  - "[[../related-work/shortcut-models]]"
---

# The DDIM single-step primitive under v-parameterisation

> `ddim_micro_step_v` is the deterministic single-step ODE update that
> underpins every multi-step inference primitive and every shortcut
> teacher path in this codebase. This note derives the formula from
> first principles, fixes notation for the rest of the shortcut
> documentation, and pins down what the function *is* (a
> decompose-then-recompose along a fixed `(x_0, ε)` ray) and *is not*
> (a noising step).

## Convention

Throughout this note we use the **diffusion convention** (matches the
code):

- `t ∈ {0, 1, …, T}` integer; `t = 0` is clean data, `t = T` is pure
  noise.
- `α_t ∈ (0, 1]` is the cumulative product (`alphas_cumprod[t]`), with
  `α_0 = 1` and `α_T ≈ 0`. `√α_t` is the data-side weight and
  `√(1 - α_t)` the noise-side weight.
- **Sampling decreases `t`** (`x_T → x_{T-1} → … → x_0`). A "denoising
  step of size `s`" maps `t ↦ t - s`.
- Shorthand: `σ_t := √(1 - α_t)`.

The flow-matching convention used in [[shortcut-training]] flips the
time axis (`t = 0` noise, `t = 1` data). The underlying math is
identical — only the direction of `t` differs. Where this note refers
to a step size `s`, it is in the **sampling direction**, i.e. always
positive.

## 1. The forward process and v's definition

The forward (noising) process in v-parameterisation is

$$
x_t = \sqrt{\alpha_t}\, x_0 + \sqrt{1 - \alpha_t}\, \varepsilon,
\qquad \varepsilon \sim \mathcal{N}(0, I). \tag{F}
$$

The velocity target is defined by Salimans & Ho (2022) as

$$
v_t := \sqrt{\alpha_t}\, \varepsilon - \sqrt{1 - \alpha_t}\, x_0. \tag{V}
$$

Why this particular combination? Parametrise the forward process by
the angle $\phi_t := \arccos(\sqrt{\alpha_t})$, so that
$\cos \phi_t = \sqrt{\alpha_t}$ and $\sin \phi_t = \sqrt{1 - \alpha_t}$.
Then (F) reads $x_t = \cos\phi_t \cdot x_0 + \sin\phi_t \cdot \varepsilon$,
and differentiating with respect to $\phi$:

$$
\frac{\partial x_t}{\partial \phi_t} = -\sin\phi_t \cdot x_0 + \cos\phi_t \cdot \varepsilon = v_t. \tag{V'}
$$

So `v_t` is the *velocity of `x_t` along the noise-angle parametrisation*
— literally a derivative. This is the geometric content behind the
minus sign in (V).

## 2. The decompose step — recovering `(x_0, ε)` from `(x_t, v_t)`

Stack (F) and (V) into a linear system:

$$
\begin{pmatrix} x_t \\ v_t \end{pmatrix}
=
\begin{pmatrix} \sqrt{\alpha_t} & \sqrt{1-\alpha_t} \\ -\sqrt{1-\alpha_t} & \sqrt{\alpha_t} \end{pmatrix}
\begin{pmatrix} x_0 \\ \varepsilon \end{pmatrix}.
$$

Call this matrix `R(φ_t)`. Its determinant is
$\alpha_t + (1 - \alpha_t) = 1$, its rows are unit-norm, and they are
orthogonal:

$$
\sqrt{\alpha_t} \cdot (-\sqrt{1-\alpha_t}) + \sqrt{1-\alpha_t} \cdot \sqrt{\alpha_t} = 0.
$$

So `R(φ_t)` is a special orthogonal matrix — a **rotation** in the
`(x_0, ε)` plane. Concretely it is rotation by `-φ_t`:

$$
R(\phi_t) =
\begin{pmatrix} \cos\phi_t & \sin\phi_t \\ -\sin\phi_t & \cos\phi_t \end{pmatrix}.
$$

The inverse is rotation by `+φ_t`:

$$
R(\phi_t)^{-1} = R(\phi_t)^\top =
\begin{pmatrix} \cos\phi_t & -\sin\phi_t \\ \sin\phi_t & \cos\phi_t \end{pmatrix}.
$$

Multiplying out gives the **decompose identities**:

$$
\boxed{
\begin{aligned}
x_0          &= \sqrt{\alpha_t}\, x_t - \sqrt{1-\alpha_t}\, v_t, \\
\varepsilon  &= \sqrt{1-\alpha_t}\, x_t + \sqrt{\alpha_t}\, v_t.
\end{aligned}
} \tag{D}
$$

Code: `shortcut_targets.py:118-119`.

These are **exact algebraic identities**, not approximations. Given any
`(x_t, v_t)` pair that satisfies (F) and (V) for some `(x_0, ε)`,
equations (D) recover that pair uniquely. The model's prediction
`v̂_t ≈ v_t` substitutes into (D) to give predicted
`(x̂_0, ε̂)` — the model's belief about the diffusion ray through `x_t`.

The minus sign on `v_t` in the `x_0` formula traces directly back to
the minus in (V) — algebraically it falls out of the cofactor, and
geometrically it is the minus inside `-sin φ_t` of the rotation
matrix.

## 3. The recompose step — DDIM at a new timestep

Once we have predicted `(x̂_0, ε̂)`, evaluating (F) at any other
timestep `t'` gives the corresponding sample along the same ray:

$$
\boxed{
x_{t'} = \sqrt{\alpha_{t'}}\, \hat x_0 + \sqrt{1 - \alpha_{t'}}\, \hat \varepsilon.
} \tag{R}
$$

Code: `shortcut_targets.py:126`.

This is the **DDIM micro-step**. Notice what just happened: we held
`(x̂_0, ε̂)` fixed and re-evaluated (F) at a different `α`. The
*diffusion ray* is the invariant; only the position along the ray
changes.

In matrix form: `(x_0, ε) ↦ (x_{t'}, v_{t'})` is `R(φ_{t'})`, so a DDIM
step is the composition

$$
(x_t, v_t) \xrightarrow{R(\phi_t)^{-1}} (x_0, \varepsilon) \xrightarrow{R(\phi_{t'})} (x_{t'}, v_{t'}),
$$

i.e. a net rotation by `φ_{t'} - φ_t` in the `(x_0, ε)` plane. If
`t' < t` then `α_{t'} > α_t` so `φ_{t'} < φ_t`, and the net rotation is
**toward the data axis**.

## 4. Geometric picture

```
         ε  ↑
            │
            │      x_t   (angle φ_t)
            │     /
            │    /                  diffusion ray
            │   /                   (one (x_0, ε) pair)
            │  /
            │ /  x_{t'}   (angle φ_{t'} < φ_t)
            │/
            ●─────────────────→  x_0
          (x̂_0, ε̂)
```

Each diffusion ray is a half-line from the origin through one
`(x_0, ε)` point. As `t` decreases (sampling direction), `φ_t`
decreases, and the projection of the ray onto the standard basis
shifts weight from `ε` to `x_0`.

The DDIM step slides `x_t` to `x_{t'}` *along the same ray*. No new
randomness — the model's predicted `ε̂` is held fixed. This is what
makes DDIM deterministic and an exponential ODE integrator.

Contrast with **DDPM**: at each step, draw a fresh `ε' ∼ N(0, I)`,
landing on a different ray each time. Stochastic; not an ODE.

## 5. Connection to the probability-flow ODE

The probability-flow ODE for the variance-preserving diffusion process
(Song et al. 2021) can be written, under v-parameterisation, as a
deterministic dynamical system $dx/dt = u(x, t)$ for some learned
field. DDIM is the **first-order exponential integrator** for this
system — exact when the diffusion ray through `x_t` is constant along
the trajectory (i.e. the model's prediction is correct).

In practice the model is imperfect, so the diffusion ray reconstructed
from `(x_t, v̂_t)` is only an estimate. An N-step DDIM sampler
accumulates per-step ray-misidentification errors. The natural fix:
better per-step ray estimates. This is the entry point for

- **higher-order integrators** (Heun, RK4, DPM-Solver) — refine the
  velocity prediction with one or more extra model calls per step;
- **shortcut models** (Frans et al. 2024) — train the model to directly
  predict the *averaged* velocity over a finite interval, so one model
  call gives the right ray at large step sizes.

Both wrap *this* primitive; they do not replace it.

## 6. Per-sample timesteps and the dynamic-rescale branch

### Per-sample `t` and `t'`

`ddim_micro_step_v` accepts `t` and `prev_t` as `(B,)` long tensors,
indexed per-sample into the shared `alphas_cumprod` table by
`_gather`. Used by:

- the inference sampler (all samples share `t`, but the function signature
  is uniform);
- the `distillation` teacher path (one Python-int dyadic `d` per batch,
  but `t` varies per sample);
- the `two_step` teacher path (jump fixed to 1, `t` varies per sample).

`prev_t.clamp_min(0)` makes `t = 0` a no-op: `α_t = α_{t'}` ⇒ recompose
returns `x_t` unchanged.

### Dynamic rescale (DynamiCrafter)

DynamiCrafter (see [[../tech/dynamic-rescale]]) optionally normalises
`x_0`'s magnitude across timesteps via a `(T,)` table `scale_arr`.
Inside the DDIM step this rescales `x̂_0` between decompose and
recompose:

$$\hat x_0 \;\leftarrow\; \hat x_0 \cdot \frac{\text{scale}[t']}{\text{scale}[t]}.$$

The noise component `ε̂` is left alone. Code:
`shortcut_targets.py:121-124`. Most schedules pass `scale_arr=None` and
skip this branch.

## 7. Sanity checks

| Check | Argument | Status |
|---|---|---|
| **Identity at `t' = t`.** | `α_{t'} = α_t` ⇒ recompose reduces to (F) with `(x̂_0, ε̂)`. By construction of (D), this equals `x_t`. | ✓ |
| **Data limit `t' = 0`.** | `α_0 = 1` ⇒ recompose gives `x̂_0` directly. Canonical one-step-to-data. | ✓ |
| **Noise limit `t = T`, `t' = T`.** | `α_T ≈ 0` ⇒ recompose ≈ `ε̂`. Consistent with `x_T` being pure noise. | ✓ |
| **Orthogonality of `R(φ_t)`.** | $\langle (\sqrt{\alpha_t}, \sqrt{1-\alpha_t}), (-\sqrt{1-\alpha_t}, \sqrt{\alpha_t}) \rangle = 0$. | ✓ |
| **Determinant +1.** | $\alpha_t + (1 - \alpha_t) = 1$. Ray-preserving, no reflection. | ✓ |

## What this primitive is *not*

- **Not a noising step.** Although the recompose formula (R) has the
  same shape as the forward formula (F), here `t' < t` so `α_{t'} > α_t`.
  The data-side weight grows; the noise-side weight shrinks. Rotation
  is *toward* the data axis.
- **Not a chord-velocity estimator.** It outputs an endpoint `x_{t'}`,
  not a velocity. Chord-velocity estimators wrap this primitive: step
  to the predicted endpoint, then call the model again *there*.
- **Not a generic ODE solver.** It is first-order under the assumption
  that the diffusion ray is correctly identified by the model's
  velocity prediction at `x_t`. Higher-order corrections (Heun,
  DPM-Solver, …) must add extra structure on top.
- **Not random.** No fresh noise is drawn. DDIM is deterministic given
  the model's velocity prediction.

## Related

- [[prediction-objectives]] — v / ε / x_0 parameterisation; why
  v-parameterisation is the natural setting for the orthogonal
  decompose-recompose structure above
- [[heun-shortcut-target]] — Heun's method built on top of this
  primitive, used as the `two_step` shortcut target
- [[shortcut-training]] — the general shortcut training objective; this
  primitive is the underlying step in every teacher path
- [[../tech/dynamic-rescale]] — the optional `scale_arr` branch
- [[../tech/shortcut-training-modes]] — code-side catalogue of both
  teacher pathways that consume this primitive
- Code: `src/generative_flow_adapters/training/shortcut_targets.py:81-126`
  (the function itself)
- Code: `src/generative_flow_adapters/training/trainer.py:567-583`
  (`_diffusion_tables` — how `alphas_cumprod` and `scale_arr` are
  fetched)
