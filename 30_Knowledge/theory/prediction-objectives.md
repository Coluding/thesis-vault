---
type: theory
last_updated: 2026-05-19
sources:
  - "[[shortcut-training]]"
  - "[[../related-work/consistency-models]]"
---

# Prediction objectives — noise, velocity, and `x_0`

> **A diffusion model can be trained against three genuinely distinct
> regression targets — `ε`, `x_0`, or `v` — and the choice meaningfully
> changes training dynamics.** Flow matching does not have the same
> three-way choice in the same sense: its native target is the constant
> velocity along the straight path, and although the code interface
> exposes the same `noise` / `sample` / `velocity` keys via algebraic
> re-expression, those are alternate parameterisations of the single FM
> objective rather than three competing ones. The codebase reflects
> this asymmetry: `prediction_type` is a real configuration knob for
> diffusion backbones and almost always left at `"velocity"` for flow
> backbones (`models/base/interfaces.py:57-64`, default flow → velocity;
> `backbones/opensora/model.py:89`, Open-Sora hard-codes velocity).

## 1. The forward process — what `x_t` is made of

Both training objectives share the same general shape: a clean latent
`x_0` is linearly mixed with Gaussian noise `ε` to produce `x_t`. The
mixing coefficients differ.

**Diffusion** (`losses/diffusion.py:91-96`, `q_sample`):

```
x_t = sqrt(ᾱ_t) · x_0 + sqrt(1 - ᾱ_t) · ε
```

with `ᾱ_t = ∏(1 - β_s)` for a discrete `t ∈ {0, …, T-1}`. At `t=0`,
`x_t ≈ x_0`; at `t=T`, `x_t ≈ ε`.

**Flow matching** (`losses/flow_matching.py:112-130`, `q_sample`):

```
x_t = (1 - (1 - σ_min) · t) · x_0 + t · ε
```

with continuous `t ∈ [0, 1]`. **Convention flip:** `t=0` is clean data,
`t=1` is noise — the opposite of the diffusion convention. The docstring
at `losses/flow_matching.py:18-22` calls this out explicitly. The
flow-matching path is a straight line between `x_0` and `ε`; the
diffusion path is a curved arc through latent space.

In both cases, knowing any two of `{x_0, ε, x_t}` determines the third
(at a given `t`). That is exactly why the choice of regression target is
free — and why it still matters.

## 2. The three diffusion objectives

In the diffusion forward process the model sees a noisy `x_t` and a
timestep `t`, and must output something from which `(x_0, ε)` can be
recovered. There are three independently sensible choices, and the
trainer dispatches between them via a single signature
(`training/trainer.py:101-107`):

```python
target_tensor = self.diffusion_objective.get_target(
    prediction_type=prediction_type or "noise",
    x_start=target_scaled,
    x_t=x_t,
    t=t,
    noise=noise,
)
## Very important. We can predict either noise, starting data point or
## velocity. Velocity is a combination of the first two.
```

The dispatch itself is `losses/diffusion.py:98-106`. `model_type`
("diffusion") selects the *forward process*; `prediction_type`
selects the *regression target* and is a separate axis — these are
three meaningfully different training targets in the same noise
schedule.

### 2.1 ε-prediction (noise)

Target is the noise sample itself (`losses/diffusion.py:100-101`):

```python
if prediction_key in {"noise", "eps", "epsilon"}:
    return noise
```

**Rationale.** Historically the original DDPM choice. At any
`t`, `ε` has unit variance and is roughly *t-invariant in scale* — the
regression target does not get small or large as `t` moves, which keeps
the loss well-conditioned across timesteps with no per-`t` reweighting.
**Failure mode at `t ≈ 0`:** when `x_t ≈ x_0` and the remaining noise
component is tiny, predicting `ε` accurately is an extremely
high-frequency task — small errors in `ε` produce large errors in the
recovered `x_0` because the inversion divides by `sqrt(1 - ᾱ_t) → 0`
(`inference/diffusion.py:172`: `pred_x0 = (sample - sqrt_one_minus *
model_output) / sqrt_alpha_t`). In practice this makes ε-loss
disproportionately punishing of low-`t` errors.

### 2.2 `x_0`-prediction (sample / `x_start`)

Target is the clean latent (`losses/diffusion.py:102-103`):

```python
if prediction_key in {"sample", "x0", "x_start"}:
    return x_start
```

**Rationale.** Symmetric failure mode to `ε`: at `t ≈ 0` predicting
`x_0` is trivial (it equals `x_t`), but at `t ≈ T` the network has to
hallucinate the clean sample from pure noise — an extremely
*low-frequency* prediction with huge target variance. Useful when the
downstream consumer wants `x_0` directly (e.g. consistency-model-style
one-step samplers, see [[../related-work/consistency-models]]) or when
the data manifold has tight structure that the network can exploit more
easily than the noise structure.

### 2.3 Velocity (`v`)

Target is a t-dependent mix of `ε` and `x_0`
(`losses/diffusion.py:108-113`):

```python
def get_velocity(self, x_start, noise, t):
    sqrt_alphas = …       # sqrt(ᾱ_t)
    sqrt_one_minus = …    # sqrt(1 - ᾱ_t)
    return sqrt_alphas * noise - sqrt_one_minus * x_start
```

i.e. `v = sqrt(ᾱ_t)·ε − sqrt(1−ᾱ_t)·x_0`. This is the
Salimans-&-Ho-2022 "v-prediction" formulation that DynamiCrafter and
Stable Video Diffusion use.

**Rationale.** Velocity is the *convex combination* of the two extreme
targets that the trainer's own comment calls out: "velocity is a
combination of the first two." Concretely:

- At `t ≈ 0` (`sqrt(ᾱ_t) → 1`, `sqrt(1−ᾱ_t) → 0`) v reduces to `≈ ε`,
  inheriting ε-prediction's well-conditioned regression in the clean
  regime.
- At `t ≈ T` (`sqrt(ᾱ_t) → 0`, `sqrt(1−ᾱ_t) → 1`) it reduces to
  `≈ −x_0`, inheriting `x_0`-prediction's behaviour in the noisy
  regime.

So v-prediction interpolates between the two endpoint parameterisations
and avoids the worst-conditioned region of each. Empirically this is
why v-prediction tends to outperform ε for video diffusion at high
noise levels and for zero-terminal-SNR schedules
(`losses/diffusion.py:48-49`, the `rescale_betas_zero_snr` branch — at
zero terminal SNR `ᾱ_T = 0`, so ε-prediction at `t = T` is ill-defined
but v-prediction reduces cleanly to `−x_0`).

### 2.4 Summary of the diffusion three-way choice

The three are not stylistic variants: each picks a different
worst-conditioning regime and a different scale-vs-`t` profile of the
loss. Practical defaults in the wild:

| Backbone family                | Default | Why                                        |
|--------------------------------|---------|--------------------------------------------|
| Original DDPM / SD-1.x         | `ε`     | Historical; works well with non-zero terminal SNR |
| DynamiCrafter / SVD            | `v`     | Better behaviour at high `t`, zero-SNR-compatible |
| Consistency-style one-step     | `x_0`   | Sampler consumes `x_0` directly            |

## 3. Flow matching: same code keys, different meaning

Flow matching has a single natural target — the constant velocity that
transports `x_0` to `ε` along the linear interpolation path
(`losses/flow_matching.py:161-167`):

```python
def get_velocity(self, x_start, noise):
    # v = (1 - σ_min) · noise - x_0 — constant along the linear path
    return (1 - self.sigma_min) * noise - x_start
```

This is *time-independent* — there is no `t` argument, because along a
straight-line trajectory the velocity is the same at every point. There
is no analog of diffusion's "different target shape at different `t`"
trade-off: the FM objective is a single regression problem.

The FM `get_target` dispatch
(`losses/flow_matching.py:132-159`) accepts the same string keys —
`noise`/`eps`, `sample`/`x0`, `velocity`/`v` — but the rationale is
different:

- `velocity` is the native FM target.
- `noise` and `x0` are *algebraic re-expressions* of that same straight
  line. Given a fixed `(x_0, ε)` and the linear interpolation
  `x_t = (1 - (1-σ_min)·t)·x_0 + t·ε`, recovering `ε` or `x_0` is
  trivially equivalent to predicting `v`. Training against `noise` here
  is closer to an SD3/Flux-style "noise-prediction flow-matching
  variant" than to a genuinely different objective in the diffusion
  sense.

In short: for diffusion, picking `ε` vs `x_0` vs `v` is a substantive
choice about which regression problem the network solves. For flow
matching, the choice is between three labels for the same line, and
the code only carries them all for API symmetry.

This is why `infer_prediction_type` (`models/base/interfaces.py:57-64`)
hard-defaults flow → velocity and is rarely overridden, while
diffusion backbones in the wild ship with all three.

## 4. They are interconvertible at inference

The sampler does not care which target the network was trained on, as
long as it can convert the network output back to a usable quantity
(typically `(pred_x0, pred_eps)`). The DDIM step at
`inference/diffusion.py:164-185` does exactly that conversion:

```python
prediction_type = _map_prediction_type(self.prediction_type)
if prediction_type == "v_prediction":
    pred_x0  = sqrt_alpha_t * sample - sqrt_one_minus_alpha_t * model_output
    pred_eps = sqrt_alpha_t * model_output + sqrt_one_minus_alpha_t * sample
elif prediction_type == "epsilon":
    pred_x0  = (sample - sqrt_one_minus_alpha_t * model_output) / sqrt_alpha_t
    pred_eps = model_output
elif prediction_type == "sample":
    pred_x0  = model_output
    pred_eps = (sample - sqrt_alpha_t * model_output) / sqrt_one_minus_alpha_t
```

All three branches end up at the same `(pred_x0, pred_eps)` pair fed
into the same DDIM update. So the *only* axis on which the
parameterisation affects the final sample quality (with sufficient
training) is: **which target gives the cleanest gradient signal during
training, given the noise/timestep distribution actually sampled.**
That is an empirical question, and it is one of the reasons backbone
choices matter — DynamiCrafter ships v-prediction weights, Open-Sora
ships flow-matching/velocity, diffusers checkpoints can be any of the
three. The framework respects whatever the loaded backbone was trained
with (`losses/diffusion.py:100-106` + `models/base/interfaces.py:57-64`).

## 5. Why this matters for the adapter framing

For D1/D2 the adapter operates inside the base model's parameterisation
— if the base was trained as v-prediction, the adapter's output enters
the residual sum `f_base + g(d)·Δ_φ` *in v-space*, and the loss is in
v-space too. There is no parameterisation conversion happening inside
the adapter — the choice is inherited from the backbone.

For D3 (shortcut adapters, see [[shortcut-training]]) the choice
becomes load-bearing because the *target* for the shortcut adapter is
synthesised from base forward passes
(`training/shortcut_targets.py:_compute_two_step_shortcut_target`). The
two-step average `½(v_b1 + v_b2)` is well-defined in v-space and
sensible: averaging two velocities along a path approximates the chord
velocity. The same averaging in ε-space or `x_0`-space would not have
the same geometric interpretation — `½(ε_b1 + ε_b2)` is not a
meaningful "chord noise." This is one concrete reason the framework is
easier to reason about when working in velocity parameterisation, even
though it nominally supports all three.

## 6. What is *not* claimed here

- _needs verification_: empirical loss-curve comparison of `ε` vs `v`
  on a single adapter+backbone in this codebase. No such ablation has
  been run; the choice is currently inherited from the backbone
  ([[../../20_Tickets]] does not yet have a ticket for this).
- _needs verification_: whether the shortcut-target construction
  (Section 5 above) materially degrades when forced into ε-space on a
  DynamiCrafter-style backbone — plausible from the geometry but not
  measured.
- _needs verification_: the practical interaction of the
  zero-terminal-SNR branch (`rescale_betas_zero_snr=True`) with
  ε-prediction is a known instability in the literature but has not
  been re-tested here.

## Related

- [[shortcut-training]] — uses v-space averaging for the chord target
- [[../related-work/consistency-models]] · [[../related-work/self-distillation]] — `x_0`-prediction is the natural target for consistency-style one-step samplers
- Code: `src/generative_flow_adapters/losses/diffusion.py` (`get_target`, `get_velocity`)
- Code: `src/generative_flow_adapters/losses/flow_matching.py` (`get_target`, `get_velocity`)
- Code: `src/generative_flow_adapters/models/base/interfaces.py` (`infer_prediction_type`)
- Code: `src/generative_flow_adapters/inference/diffusion.py` (`_map_prediction_type`, DDIM conversion)
- Code: `src/generative_flow_adapters/training/trainer.py:101-107` (the explicit "very important" comment in the train loop)
