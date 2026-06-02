---
type: theory
last_updated: 2026-05-19
sources:
  - "[[../related-work/shortcut-models]]"
  - "[[../related-work/consistency-models]]"
  - "[[../related-work/self-distillation]]"
---

# Shortcut training — general idea + our adapter framing

> The conceptual backbone of the D3 contribution. The standard
> shortcut formulation retrains the generative model from scratch with
> a self-consistency objective so it can sample in few steps; our
> contribution is to keep the base frozen and put the same objective
> into an adapter that additionally consumes the base's local
> prediction. This note records the general idea and our specific
> twist; the data-flow figure lives in
> [[../writing/figure-shortcut-training]], the pedagogical walkthrough
> in [[../writing/explainer-shortcut-training]].

## 1. Generative models as ODE trajectories

Both diffusion and flow matching can be sampled by integrating a
learned vector field along a trajectory between two endpoints — noise
at one end, data at the other. The model's job at each timestep is to
predict the *local direction of travel*; the sampler's job is to
integrate.

- **Flow matching:** the model predicts a velocity field
  `v_θ(x_t, t)` such that `dx/dt = v_θ(x_t, t)`. Sampling integrates
  this from `t=0` (noise) to `t=1` (data).
- **Diffusion:** the model predicts a noise direction `ε_θ(x_t, t)`
  (or `x_0`, or score); via the probability-flow ODE this can be
  rewritten in the same velocity form. The choice of regression
  target — `ε`, `x_0`, or `v` — is a separate axis from the forward
  process and matters for training dynamics: see
  [[prediction-objectives]].

In both cases the local prediction is just a *direction*; how the
sampler turns directions into an actual finite-displacement step is
the integrator's responsibility.

## 2. Euler integration and why few-step matters

The cheapest integrator is plain Euler:

```
x_{t+d} ≈ x_t + d · v_θ(x_t, t)
```

Each step costs one model forward pass. Halving `d` halves the
discretisation error but doubles the inference cost. For video
diffusion this is the dominant cost — a single base-model call is
already expensive, so reducing the step count from e.g. 50 → 4 is the
biggest available speedup lever short of switching backbones.

## 3. Shortcut models — predict the full step

Shortcut models (see [[../related-work/shortcut-models]]) make the
step size `d` an *input* to the model rather than an external knob of
the sampler:

```
s_θ(x_t, t, d)   approximates   (x_{t+d} - x_t) / d
```

i.e. the *average* velocity over `[t, t+d]`, which when scaled by `d`
gives the chord from `x_t` to `x_{t+d}` on the true trajectory. At
`d → 0` this reduces to the standard instantaneous velocity / score
prediction; at `d ≈ 1` it produces a one-step sampler.

The training objective is **self-consistency**: a step of size `2d`
should equal the average of two consecutive steps of size `d`:

```
s_θ(x_t, t, 2d)  ≈  ½ · s_θ(x_t,    t,     d)
                   +  ½ · s_θ(x_{t+d}, t+d, d)
```

This recurrence anchors the model at `d → 0` (where standard
denoising supervision applies) and propagates supervision upward to
larger step sizes. Training is end-to-end and from scratch — every
parameter of the generative model is in the optimisation problem.

## 4. Our extension — adapter, with the base's local prediction as input

Retraining a multi-billion-parameter video diffusion model from
scratch is not realistic for a master's thesis. The D3 contribution:
freeze the base and put step-size conditioning + the shortcut
objective into an **adapter**.

Two design choices matter here:

### 4.1 The base's local prediction is given to the adapter

The adapter takes as input the base model's local prediction
`v_base = f_base(x_t, t, c)` — the direction the frozen base would
have produced if asked the standard question, ignoring step size.

Code anchors:

- `adapters/output/shortcut_direction.py:20`
  (`include_base_direction: bool = True` default).
- `adapters/output/shortcut_direction.py:72–80`
  (concat `cond["base_direction"]` into the adapter's feature input).
- The general `DynamiCrafterOutputAdapter` does the same in spirit via
  channel-concatenation of `[x_t ‖ v_base]`
  (`adapters/output/dynamicrafter.py:118–119`, with
  `condition_on_base_outputs=True` default).

This is the key architectural difference from the from-scratch shortcut
formulation. In the paper, the model itself learns both the local and
the chord direction; here the base contributes the local direction and
the adapter learns the step-size-dependent correction.

### 4.2 The adapter is trained against a paper-faithful self-consistency target

The supervision target for the adapter is constructed by the
**adapted** model bootstrapping itself across the dyadic recurrence
from §3. At each training step, with probability
`shortcut_anchor_prob` the model is grounded by the standard
diffusion/flow loss at the smallest step size (the anchor); otherwise
a step size `s` is sampled from `ShortcutStepSchedule` and the target
at `step_level = s` is the average of two no-grad calls of the adapted
model at `step_level = s/2`, chained across one `s/2`-sized DDIM
micro-step:

$$
\hat s(x_t, t, s)
\;\leftarrow\;
\tfrac{1}{2}\big[\, \hat s_{\text{detached}}(x_t, t, s/2) + \hat s_{\text{detached}}(x_{t-s/2}, t-s/2, s/2)\,\big].
$$

This is `shortcut_target_method: distillation` in the trainer
(`training/trainer.py:385-414`,
`training/shortcut_targets.py:54-78`). It is the only shortcut
training mode in the trainer. The anchor branch — together with the
data-anchored standard loss it runs — is what rules out the trivial
fixed points of the self-consistency objective; see
[[../../50_Decisions/decided/shortcut-anchor-schedule]] and
[[../../20_Tickets/risk-shortcut-self-consistency-collapse]].

**Historical note — a previously-implemented `two_step` mode.** Earlier
versions of the trainer also offered a `shortcut_target_method:
two_step` regime that built the supervision target from two no-grad
calls of the **frozen base** (Heun-corrected over one DDIM micro-step).
This mode was **deprecated on 2026-05-28** — see
[[../../50_Decisions/decided/deprecate-twostep-shortcut-mode]] — on the
grounds that the frozen base, being a pointwise velocity estimator,
cannot supply chord-velocity supervision at the step sizes few-step
inference asks about. The Heun construction itself is preserved as a
separate **velocity-field smoothness regularizer**
([[heun-smoothness-regularizer]]); it is orthogonal to shortcut
training and applicable to any run. The full catalogue of the
remaining shortcut training mode — knobs, costs, collapse risk, line
numbers — lives in [[../tech/shortcut-training-modes]].

## 5. Why this framing might work

- **The base already encodes a strong local prior.** Asking the
  adapter to *correct* a useful starting direction is plausibly an
  easier learning problem than asking it to predict the chord
  direction from scratch.
- **The `d → 0` limit is well-defined.** When the step size is
  infinitesimal, the correct adapter output equals `v_base`. The
  architecture admits this as a trivial pass-through (the input
  concat path means the adapter can in principle just copy
  `v_base`). This gives a sensible default behaviour and an obvious
  inductive bias.
- **Adapter-family agnostic.** The same shortcut training scheme
  applies across the framework's adapter families (output / hidden /
  LoRA / hypernetwork). This is exactly the design surface D1 gives
  us, and the D3 vs. D2 vs. D4 comparison rests on running shortcut
  training across families and reporting the trade-offs.

## 6. What is *not* claimed here

- _needs verification_: whether the `d → 0 ⇒ adapter ≈ v_base`
  pass-through actually emerges from the optimisation in practice, or
  whether the adapter learns something else at small `d`. Worth a
  small ablation when D3 experiments start.
- _needs verification_: whether the base's local prediction is
  *strictly necessary* for the adapter, or whether the adapter alone
  on `(x_t, t, c, d)` can match it. Drop-the-base ablation, easy to
  run.
- No quantitative claims about speed–quality trade-offs at any
  specific step count — those are D3 experiments that have not yet
  been run.

## Related

- [[../tech/shortcut-training-modes]] — code-side catalogue of the two
  supervision regimes (`two_step` vs. `distillation`)
- [[../related-work/shortcut-models]] — the original paper, vendored
  PDF at `docs/paper/shortcut_models.pdf`
- [[../related-work/consistency-models]] · [[../related-work/self-distillation]] · [[../related-work/dpm-solver]] — neighbours in the few-step-sampling cluster
- [[../writing/figure-shortcut-training]] — forward-time training data-flow figure
- [[../writing/explainer-shortcut-training]] — pedagogical walkthrough (HTML doc with SVG visuals)
- [[../../20_Tickets/bug-training-shortcut-target-timestep]] — pending code fix
- Code: `src/generative_flow_adapters/adapters/output/shortcut_direction.py`
- Code: `src/generative_flow_adapters/training/shortcut_targets.py`
- Code: `src/generative_flow_adapters/losses/consistency.py`
- Repo doc: `docs/shortcut_action_summary.md`
