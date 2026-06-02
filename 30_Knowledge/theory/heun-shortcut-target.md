---
type: theory
last_updated: 2026-05-28
sources:
  - "code: src/generative_flow_adapters/training/shortcut_targets.py"
  - "code: src/generative_flow_adapters/training/trainer.py"
  - "[[ddim-step-v-parameterisation]]"
  - "[[shortcut-training]]"
  - "[[prediction-objectives]]"
  - "[[../related-work/shortcut-models]]"
---

# Heun's method as the `two_step` shortcut target — and why it is not a shortcut target

> The `two_step` shortcut target in this codebase is **Heun's method**
> (the 2nd-order explicit Runge-Kutta integrator a.k.a. improved Euler /
> explicit trapezoidal rule) applied to the frozen base model over a
> single DDIM micro-step. This note derives Heun rigorously from the ODE
> view, maps it onto the current code, and pins down the critical
> structural limitation: the DDIM jump is hardcoded to 1 timestep, so
> the supervision signal carries no information about step-size
> variation. As currently coded, `two_step` is a Heun-quality velocity
> prior at the finest scale — not a shortcut target.

## Convention

Diffusion convention (matches the code): `t = 0` clean data, `t = T`
noise; sampling decreases `t`; a denoising step of size `s` maps
`t ↦ t - s`. Builds on the decompose-recompose identities in
[[ddim-step-v-parameterisation]].

## 1. What an ideal shortcut target should be

The shortcut adapter is asked to predict a step of size `s` from `x_t`
in **one network call**. The "right" target for that prediction is the
**average velocity** along the true ODE trajectory over the interval
`[t - s, t]`:

$$
\bar v(x_t, t, s) := \frac{1}{s} \int_{t-s}^{t} v(x_\tau, \tau)\, d\tau. \tag{V̄}
$$

Equivalently, the chord direction from `x_t` to the true `x_{t-s}`,
scaled appropriately. If the model can produce `v̄` directly, then a
single DDIM-style step using `v̄` lands at the right place — no
multi-step rollout needed.

Two practical problems:

1. The true `v(x_\tau, \tau)` is unknown; we have a frozen base model
   $\hat v_{\text{base}}$ that approximates it pointwise.
2. The integral in (V̄) cannot be computed in closed form — we need a
   **quadrature rule** to estimate it from finite model samples.

Different rules give different bias–variance trade-offs. Heun is
2nd-order; the cheaper Euler rule is 1st-order; richer rules (RK4,
DPM-Solver) buy more accuracy at more compute. The choice of rule
defines the **supervision target** that the adapter is regressed
against.

## 2. Heun's method, rigorously

Consider an ODE $\dot y = f(t, y)$ with step size `h`. Euler is the
1st-order rule

$$
\tilde y_{n+1} = y_n + h \cdot f(t_n, y_n). \tag{E}
$$

Local truncation error: $O(h^2)$; global error $O(h)$. The error comes
from approximating the integral $\int_{t_n}^{t_n+h} f(t, y(t))\, dt$ by
the rectangle $h \cdot f(t_n, y_n)$, which ignores any change in `f`
across the interval.

**Heun's method** (improved Euler, explicit trapezoidal rule) corrects
this with a predictor-corrector:

$$
\begin{aligned}
\tilde y_{n+1} &= y_n + h \cdot f(t_n, y_n)
  &&\text{(Euler predictor)} \\
y_{n+1}        &= y_n + \frac{h}{2} \big[ f(t_n, y_n) + f(t_n + h, \tilde y_{n+1}) \big]
  &&\text{(trapezoidal corrector)}
\end{aligned} \tag{H}
$$

Local truncation error: $O(h^3)$; global error $O(h^2)$. One order of
accuracy in `h` is bought with one extra evaluation of `f`. The
**average slope over the step** is

$$
\frac{y_{n+1} - y_n}{h} = \tfrac{1}{2}\big[ f(t_n, y_n) + f(t_n + h, \tilde y_{n+1}) \big]. \tag{H'}
$$

This is the form that matters for shortcut training: (H') is exactly a
chord-velocity estimator over the interval `[t_n, t_n + h]`, obtained
from two velocity samples — one at the start, one at the predictor
endpoint.

### Why Heun cancels the leading error term

Taylor-expand the true increment:

$$
y(t_n + h) - y_n
  = h f(t_n, y_n) + \tfrac{h^2}{2}\, \dot f(t_n, y_n) + O(h^3),
$$

where $\dot f = \partial_t f + f \partial_y f$ (total derivative along
the trajectory). Now expand the Euler-predictor slope:

$$
f(t_n + h, \tilde y_{n+1})
  = f(t_n, y_n) + h \dot f(t_n, y_n) + O(h^2).
$$

Average it with $f(t_n, y_n)$:

$$
\tfrac{1}{2}\big[ f(t_n, y_n) + f(t_n + h, \tilde y_{n+1}) \big]
  = f(t_n, y_n) + \tfrac{h}{2}\, \dot f(t_n, y_n) + O(h^2).
$$

Multiplying by `h`, the Heun increment matches the Taylor expansion to
order `h²` — i.e. through the $\tfrac{h^2}{2} \dot f$ term — so the
leading error is $O(h^3)$. The Euler increment, by contrast, only
matches through order `h`.

This is **why** Heun's averaging works: it captures the first-order
curvature of `f` along the trajectory.

## 3. Mapping Heun onto the current code

The trajectory variable is `y → x`, the slope field is `f → v̂_base`,
and the step direction is *denoising* (sampling): `h → -s` in
mathematical terms, but the code parametrises it via DDIM (`t → t - s`)
rather than a raw additive update. Concretely, in
`shortcut_targets.py:33-51` and the inline copy at `trainer.py:510-534`:

```python
v0    = base_model(x_t, t, cond=cond)                    # f(t_n, y_n)
prev_t = (t - 1).clamp_min(0)                            # t_n + h, with h = -1
x_mid = ddim_micro_step_v(x=x_t, v=v0, t=t, prev_t=prev_t,
                         alphas_cumprod=alphas, scale_arr=scale_arr)
                                                         # ỹ_{n+1}
v1    = base_model(x_mid, prev_t, cond=cond)             # f(t_n + h, ỹ_{n+1})
return ((v0 + v1) / 2.0).detach()                        # ½(f₁ + f₂)
```

| Heun ingredient | Code line |
|---|---|
| Initial slope `f(t_n, y_n)` | `v0 = base_model(x_t, t)` |
| Predictor step `ỹ_{n+1}` (Euler in y, DDIM in x) | `x_mid = ddim_micro_step_v(x_t, v0, t, prev_t)` |
| Slope at predictor endpoint `f(t_n + h, ỹ_{n+1})` | `v1 = base_model(x_mid, prev_t)` |
| Averaged slope (chord velocity) `½(f₁ + f₂)` | `(v0 + v1) / 2.0` |

The predictor is **not** the naive linear step `x_t + h · v0` — that
would ignore the v-parameterisation algebra (see
[[ddim-step-v-parameterisation]]). Using a proper DDIM micro-step is
what makes `x_mid` actually land on a valid `x_{t-1}` along a single
predicted diffusion ray.

### Earlier implementation (now superseded)

A previous version of this note flagged a bug: the second base call was
evaluated at `(x_mid, t)` instead of `(x_mid, t - 1)`, breaking Heun's
time argument. **This is structurally fixed.** The current code calls
`base_model(x_mid, prev_t, …)` — the time argument advances with the
state. The associated ticket
[[../../20_Tickets/bug-training-shortcut-target-timestep]] is now
**obsolete** with respect to the time-argument issue. (A different
limitation has taken its place; see §5.)

## 4. The Heun-averaged target as supervision

The returned tensor `½(v0 + v1)` has the units of an instantaneous
velocity — it is the Heun-averaged slope *over the interval
`[t-1, t]`*. The adapter is regressed against this target via an MSE
head (`shortcut_direction_loss` / `local_consistency_loss` in
`losses/consistency.py:7-12`).

So the adapter is supervised to **output, in one call, the
Heun-averaged velocity that would result from two carefully chosen base
calls**. If the adapter learns this perfectly, then a single
adapter-call DDIM step *within one timestep* is second-order accurate,
whereas a single base-call DDIM step is only first-order.

Note carefully: the interval is *one timestep*. Not `s` timesteps. Not
the step size passed to the adapter via `step_level`.

## 5. The critical structural limitation

The DDIM jump in the current code is **hardcoded to 1 timestep**:

```python
prev_t = (t - 1).clamp_min(0)     # trainer.py:528, shortcut_targets.py:45
```

This single line decides the whole semantics of the supervision signal.

### 5.1 What this means

- Heun's `h` is fixed to one timestep, regardless of what the user puts
  in `shortcut_step_level_min/max` or what value of `step_level` is
  injected into `cond`.
- The averaged target `½(v0 + v1)` is the chord velocity over a
  *single-timestep* interval. For a `T = 1000` noise schedule, that is
  `1/1000` of the trajectory.
- The adapter is supervised to be a *better velocity prior at the
  finest possible scale* — it learns Heun-quality velocity over 1
  timestep, not a chord velocity over a finite step size.

### 5.2 `step_level` is decorative

The code does still inject `step_level` into `cond`, sampled uniformly
from `[shortcut_step_level_min, shortcut_step_level_max]`
(`_resolve_step_level`, `trainer.py:443-473`). But the target does
**not** depend on this value. The adapter sees `step_level` in its
conditioning but receives the same supervisory signal regardless of
what value was passed.

Consequence: the adapter has **no incentive** to learn step-size-aware
behaviour under `two_step`. Whatever pattern it learns at one
`step_level` is what it will produce at every other `step_level`. The
inference-time step-size knob is decoupled from anything the model was
trained to do.

### 5.3 Inference is out-of-distribution

At inference, the sampler is asked to take jumps of size `s = T / N`
for some user-chosen `N`. For typical `N = 50` and `T = 1000`, that is
a 20-timestep jump. The adapter has **never seen training supervision
for a jump of more than 1 timestep**. Any few-step rollout it produces
is extrapolating from a single training operating point.

The adapter may still generalise — the Heun-averaged target is *closer*
to the true chord velocity than the raw base output, so the adapter
inherits a slightly better velocity prior — but this is the "happy
accident" path, not a designed behaviour.

### 5.4 What `two_step` actually gives you

| Claim | Status |
|---|---|
| Better velocity prediction per call at fine scale | ✓ |
| Second-order accurate ODE integration when stepping by 1 timestep at inference | ✓ |
| Step-size-conditional behaviour (different output for different `step_level`) | ✗ — `step_level` is ignored by the target |
| Few-step rollout fidelity at `N << T` | ✗ — never supervised |
| Implementation of the shortcut-models paper's objective | ✗ — that lives in `distillation` mode |

This makes `two_step` a misnomer. It is not a shortcut training mode in
the sense of Frans et al. 2024; it is a **base-distilled second-order
velocity prior**. Useful for thesis claims about "the adapter improves
the base's per-call accuracy"; not useful for thesis claims about
"few-step generation".

## 6. The fix — superseded by deprecation decision

> **2026-05-28 update.** This section originally proposed wiring the
> `ShortcutStepSchedule` into the `two_step` teacher path to make the
> jump and `step_level` load-bearing. That fix is **no longer the
> path** — the decision recorded in
> [[../../50_Decisions/decided/deprecate-twostep-shortcut-mode]] is to
> remove `two_step` from the shortcut training mode dispatch entirely
> and add the Heun construction back as a separate
> velocity-field-smoothness regularizer
> ([[heun-smoothness-regularizer]]).
>
> The mechanical fix below is preserved for reference, but the deeper
> reason for deprecation is that the *frozen base* cannot supply
> chord-velocity supervision at large step sizes regardless of how the
> jump is wired — Heun's averaging is a 2nd-order Taylor expansion
> around the current point, which degrades to a noisy chord guess when
> the Euler predictor lands far from the true endpoint. Only
> `distillation` (the adapted-model self-consistency target) escapes
> this ceiling.

To turn `two_step` into a real shortcut target while keeping its
appeal (base as teacher, no self-consistency collapse risk), three
changes are required:

1. **Sample `s` per training step** from the same
   `ShortcutStepSchedule` already used by the `distillation` path
   (`step_schedule.py`). This is one line:

   ```python
   s = self.step_schedule.sample()
   jump = self.step_schedule.to_timestep_jump(s)
   ```

2. **Use `jump` in the DDIM micro-step** instead of the hardcoded 1:

   ```python
   prev_t = (t - jump).clamp_min(0)
   x_mid  = ddim_micro_step_v(x=x_t, v=v0, t=t, prev_t=prev_t, ...)
   v1     = base_model(x_mid, prev_t, cond=cond)
   ```

3. **Inject the matching `step_level = s`** into `cond` so the adapter
   knows which interval the target corresponds to. This makes the
   conditioning value functionally load-bearing.

The target $\tfrac12(v_0 + v_1)$ is now genuinely the Heun-averaged
chord velocity over the interval `[t - jump, t]`, and the adapter is
supervised on a family of targets indexed by the sampled step size.

### Why this is a clean variant

- Same base-as-teacher property — `v_0` and `v_1` are both no-grad
  calls of the frozen base, so the target is deterministic given
  `(x_t, t, s, c)` and carries no collapse risk.
- Same Heun-quality second-order accuracy — within each interval `s`,
  the target is `O(s^3)` accurate.
- Shares the same `ShortcutStepSchedule` as `distillation`, so the
  same eval grid (`log_step_size_grid`) measures both methods on the
  same step counts.
- Strictly more informative than the current `two_step`: at the
  smallest `s` in the schedule the new target reduces to the current
  one (both supervise a 1-timestep jump).

### Caveat — Heun fidelity degrades at large `s`

Heun's error is $O(s^3)$ *locally*, but the constant grows with the
curvature of the velocity field along the chord. At large `s` (e.g.
`s = 1`, one-step generation) the Euler predictor lands far from the
true endpoint, and the corrector slope `v_1` is evaluated at a
poorly-chosen `x_{t-s}`. The target degrades to a noisy chord estimate.

Two mitigations:

- **Cap the schedule.** Discrete `log2` schedule with `max = 1/8` or
  `1/4` keeps `s` in the range where Heun is empirically faithful.
- **Higher-order quadrature for large `s`.** E.g. evaluate the base at
  one or more intermediate points along the predicted ray and use a
  4-point Simpson or RK4 average. Cost grows linearly in the number of
  base calls per step.

Both are research questions; the minimum-viable fix is just (1)–(3)
above.

## 7. What is *not* claimed here

- _needs verification_: empirical magnitude of the inference-time
  generalisation gap for the current `two_step` (trained at 1-timestep
  jump, asked to roll out at 8-, 16-, 64-step jumps). The geometric
  argument predicts large degradation; a smoke-test on the existing
  configs would settle it.
- _needs verification_: whether the fixed `two_step` variant
  outperforms `distillation` at any specific step count. Plausibly yes
  at small step counts where base curvature is mild; plausibly no when
  the adapted model has accumulated useful structure that
  self-consistency can exploit.
- No quantitative claim about which integrator (Heun vs. RK4 vs.
  higher-order) is optimal as a shortcut target. Heun is what the
  shortcut-models paper uses (implicitly, through its averaging
  structure) and what the code does; deeper quadrature is a research
  direction, not a settled choice.

## Related

- [[ddim-step-v-parameterisation]] — the DDIM single-step primitive
  used as Heun's predictor; rigorous derivation of the
  decompose-recompose
- [[shortcut-training]] §3 — the chord-velocity target this integrator
  approximates; §4 — how the adapter framing changes the supervision
  story
- [[../tech/shortcut-training-modes]] — code-side catalogue of
  `two_step` vs. `distillation` (gotchas listed)
- [[prediction-objectives]] — why v-parameterisation is the natural
  setting for the Heun construction (orthogonal decompose-recompose)
- [[../related-work/shortcut-models]] — original Frans et al. 2024
  paper
- Code: `src/generative_flow_adapters/training/shortcut_targets.py:33-51`
  (`compute_two_step_target_v`)
- Code: `src/generative_flow_adapters/training/trainer.py:510-534`
  (`_compute_two_step_target_v` — the inline copy)
- Code: `src/generative_flow_adapters/training/step_schedule.py`
  (`ShortcutStepSchedule`, the per-batch step-size sampler that
  `two_step` should be wired into)
