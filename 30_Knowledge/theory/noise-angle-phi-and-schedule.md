---
type: theory
last_updated: 2026-06-18
sources:
  - "[[ddim-step-v-parameterisation]]"
  - "[[shortcut-v-averaging-bias]]"
  - "[[shortcut-training]]"
  - "code: configs/base/dynamicrafter512.yaml (linear_start 0.00085, linear_end 0.012, rescale_betas_zero_snr: True, parameterization v, timesteps 1000)"
  - "code: src/external_deps/lvdm/models/utils_diffusion.py (make_beta_schedule, rescale_zero_terminal_snr)"
---

# The noise-angle φ and the timestep↔angle map

> The concrete bridge between the clean circle geometry
> ([[ddim-step-v-parameterisation]], [[shortcut-v-averaging-bias]]) and the
> actual DynamiCrafter schedule. This is the math behind the **secondary
> misalignment** (timesteps ≠ arc-angle) and behind the **arc-length /
> log-SNR reparametrisation** option (Option C in
> [[../../50_Decisions/open/shortcut-target-endpoint-vs-v-averaging]]).
> **It fixes the secondary issue only** — not the primary v-averaging bias;
> stack it on the endpoint/displacement fix.

## 1. Definition of φ

The VP forward process is `x_t = √ᾱ_t·x₀ + √(1−ᾱ_t)·ε`, with
`ᾱ_t = ∏_{i≤t}(1−β_i)`. Matching to `x = cosφ·x₀ + sinφ·ε`:

```
cosφ = √ᾱ_t ,   sinφ = √(1−ᾱ_t)     ⇒     φ(t) = arccos√ᾱ_t = arctan√((1−ᾱ_t)/ᾱ_t).
```

`cos²+sin² = 1` holds automatically because `ᾱ_t + (1−ᾱ_t) = 1` — that *is*
the "variance-preserving" constraint, and it is exactly what puts the
trajectory on the **unit circle**. φ sweeps `[0, π/2]` as t goes data→noise
(`ᾱ→1 ⇒ φ→0`; `ᾱ→0 ⇒ φ→π/2`).

## 2. Why φ is the geometrically natural angle (two readings)

- **SNR / log-SNR.** `tanφ = √((1−ᾱ)/ᾱ) = σ_t/α_t`, so
  `SNR = ᾱ/(1−ᾱ) = cot²φ` and `λ := log SNR = 2·log cot φ`. φ is a smooth
  monotone reparametrisation of log-SNR — "work in φ" and "work in log-SNR"
  are the same clock up to a monotone map.
- **Arc-length.** On the unit circle φ *is* arc-length (radius 1 ⇒ arc =
  angle). The tangent `v` has unit speed in φ, and φ is the coordinate in
  which **curvature is uniform** — the reason it is the "right" clock for the
  consistency identity.

## 3. The t↦φ map is nonlinear — the local rate

`t↦φ` is nonlinear (`ᾱ_t` is a product, ≈ exponential in t, then `arccos√·`
on top). Differentiating `cosφ = √ᾱ_t` and using the continuous-VP relation
`dᾱ/dt = −β(t)·ᾱ`, `sinφ=√(1−ᾱ)`, `cosφ=√ᾱ`:

```
dφ/dt = β(t)/2 · cotφ.
```

Behaviour:

- **Clean end** (`φ→0`): `cotφ→∞ ⇒ dφ/dt→∞`. A tiny t-step sweeps a large
  arc — angle moves fast.
- **Noisy end** (`φ→π/2`): `cotφ→0 ⇒ dφ/dt→0`. A t-step sweeps almost no
  arc — angle crawls.

So equal t-steps are wildly unequal in φ (front-loaded toward the clean
end), and `β(t)` linear modulates it further. This is precisely why "double
the step in timestep units" ≠ "double the arc," and why the halving in the
self-consistency identity is distorted depending on where on the trajectory
you are — the **secondary misalignment** of [[shortcut-v-averaging-bias]] §4.

**Conditioning consequence.** The shortcut head conditions on a step size in
`t`-units, but by the rate above the *same* conditioned step size sweeps a
**different arc at different `t`** — a large arc near the clean end, a tiny
one near noise. So "step size `d`" is not a consistent geometric instruction:
one scalar maps to many different jumps, and the doubling/halving the
consistency identity is built on is warped (worst where `dφ/dt` changes
fastest). Conditioning on `φ` (or `λ`) makes a given step size always sweep
the same arc — a true geometric halving, and an unambiguous conditioning
variable.

## 4. DynamiCrafter, computed (sourced)

`dynamicrafter512.yaml`: scaled-linear β (`make_beta_schedule("linear",…)` =
`linspace(√0.00085, √0.012, 1000)²`), `parameterization: v`, **and
`rescale_betas_zero_snr: True`** (Lin et al. 2023). Computed φ landmarks
(this is a computed result against the repo schedule, not a literature
number):

| | first 10% of t (clean) | last 10% of t (noisy) | ratio | ᾱ_T | φ_max | φ at t/T=0.5 |
|---|---|---|---|---|---|---|
| **actual (zero-SNR rescaled)** | 0.314 rad | 0.055 rad | **5.7×** | **0** | **π/2 = 1.571** | 1.058 (= 0.673 of arc) |
| raw scaled-linear (no rescale) | 0.302 rad | 0.052 rad | 5.9× | 0.00466 | 1.503 | 1.017 (0.648 of arc) |

Read-offs (actual schedule): the angle **front-loads hard toward the clean
end** (~5.7× faster there); by `t/T=0.5` you are already **67%** through the
arc. Because of the zero-terminal-SNR rescale, the trajectory **reaches pure
noise** (`ᾱ_T=0`, `φ_max=π/2`) — good for one-/few-step generation, whose
largest shortcut must start from pure noise.

**Corrections to an earlier hand-derivation** (kept for honesty): a prior
pass claimed DynamiCrafter is *not* zero-terminal-SNR (`φ_max≈1.531`,
`ᾱ_T≈0.00158`) and an ~11× clean/noisy ratio with `t/T=0.5 → 73.7%`. Neither
reproduces against this config — the qualitative front-loading is right, but
the rescale flag makes it zero-terminal (`φ_max=π/2` exactly) and the
scaled-linear schedule gives ~5.7×, not 11×. Always compute against the repo
betas (the discrete product + the zero-SNR rescale), not a continuous
idealisation.

## 5. The clock dictionary (for implementation)

```
cosφ = √ᾱ_t ,  sinφ = √(1−ᾱ_t) ,  φ = arccos√ᾱ_t
SNR  = cot²φ ,  λ = log SNR = 2 log cotφ ,  dφ/dt = β(t)/2 · cotφ
```

**Inverse (to build a φ-uniform grid):** pick equally spaced
`φ_k ∈ [0, φ_max]`, convert to signal level `ᾱ = cos²φ_k`, then find the
timestep `t` whose cumulative `ᾱ_t` matches by lookup/interpolation against
the precomputed `ᾱ` array (no closed form for `t(ᾱ)` with a discrete product
schedule). Query/step the network at those timesteps while *conditioning and
stepping uniformly in φ*.

## 6. Practical upshot (and its scope)

Condition the shortcut head on **φ** (or, better for engineering, **log-SNR
λ**, which is smoother near the endpoints and avoids the `arccos√·`
compression where φ bunches up near π/2), build the doubling grid uniform in
φ/λ, and convert to timesteps by the inverse lookup. That simultaneously:

- makes the holonomy factor uniform per doubling level, and
- removes the timestep-vs-angle misalignment (halving the step now halves the
  arc).

**Scope caveat (important).** This is Option C — the *arc-length
reparametrisation*. It fixes the **secondary** misalignment only. It does
**not** remove the primary v-averaging bias: the composition rule is still
tangent-averaging, whose fixed point is biased (see
[[shortcut-v-averaging-bias]] §4, the fixed-point criterion). Stack this on
the **endpoint-inversion or displacement** fix (Options A/B), do not use it
in place of them.

## Related

- [[ddim-step-v-parameterisation]] — the v-step primitive and the rotation form.
- [[shortcut-v-averaging-bias]] — the primary bias, the two errors, the fixed-point test.
- [[shortcut-training]] — general shortcut framing.
- [[../../50_Decisions/open/shortcut-target-endpoint-vs-v-averaging]] — Option C lives here.
