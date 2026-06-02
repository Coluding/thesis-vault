---
type: risk
scope: shortcut
status: done
priority: high
created: 2026-05-19
updated: 2026-05-22
resolution: shipped
resolution_note: User-confirmed resolved on 2026-05-22.
closed_at: 2026-05-22
related: []
---

# Self-consistency-only adapter loss has trivial fixed points

## Context

We are designing an **adapter-side shortcut loss**: treat the composed
prediction `s(x_t, t, d) = compose(base(x_t, t), Δ_φ(x_t, t, a, d))` as the
shortcut function and train the adapter `Δ_φ` so that the composed `s`
satisfies the paper's self-consistency equation (Eq. 4 of
[[../30_Knowledge/related-work/shortcut-models]]):

```
s(x_t, t, 2d) = ½·s(x_t, t, d) + ½·s(x'_{t+d}, t+d, d)
x'_{t+d}     = x_t + s(x_t, t, d)·d        (flow case)
             = DDIM_step(x_t, s(x_t, t, d), t → t+d)   (v-pred case)
```

The appeal: the frozen base is already a one-step velocity predictor, so it
plays the role of the paper's `d → 0` flow-matching solution **for free**.
The adapter only has to learn the multi-step corrections.

## The risk

Self-consistency, taken on its own, has **trivial fixed points** the loss
cannot rule out:

1. **Cancellation collapse.** `Δ_φ(x_t, t, a, d) ≡ −base(x_t, t)` for all
   `d` gives `s ≡ 0`. Equation 4 reads `0 = (0 + 0)/2`. Satisfied.
2. **Constant-field collapse.** `s(x_t, t, d) ≡ c(x_t, t)` independent of
   `d` and constant under the ODE step. Reads `c = (c + c)/2`. Satisfied.
3. **More subtly**, any pair `(adapter, ODE-step)` that is consistent with
   itself but inconsistent with the data distribution — the loss does not
   see the data.

The paper avoids this with an explicit **`d = 0` flow-matching anchor**:
`||s(x_t, t, 0) − (x_1 − x_0)||²`, which pins the smallest-step behaviour
to empirical velocity samples and grounds the recursive bootstrap. Without
that term, nothing in their loss touches the data; the family of consistent
solutions is much larger than the data-consistent one.

For us, the obvious thought is "the frozen base is the anchor." That is
only partly true:

- It is true **asymptotically**: as `d → 0`, the composed `s` is dominated
  by `base`, which is the right one-step velocity by construction.
- It is **not** true that this asymptote is *enforced* by the loss. The
  self-consistency loss never sees the base in isolation; it only sees the
  composed `s` at chosen step sizes. Nothing in the gradient prevents the
  adapter from learning to cancel the base at every queried `d`.

So we cannot just rely on "the base is frozen" to dodge the collapse
problem. The base supplies the *correct* small-`d` answer, but the loss
does not pull the adapter towards letting that answer through.

## Mitigation options to evaluate

### Option A — Architectural zero-asymptote (no data term)

Engineer `Δ_φ(x_t, t, a, d) → 0` as `d → 0`, by design:

- Multiply the adapter's output by a gate `g(d)` with `g(0) = 0` (e.g.
  `g(d) = sigmoid(α·d + β)` with `β ≪ 0`, or `g(d) = softplus(d)`).
- Or zero-init the final projection of `step_level_embed` and use a
  bias-free skip so the residual is identically zero at the trained-for
  `d = 0` token.

This makes "at small `d`, the composed `s` equals base" a **structural**
guarantee, not a learned one. It does not directly prevent cancellation
collapse at moderate `d`, but it forces the adapter's contribution to fade
where the base is known-correct, which constrains the family of fixed
points.

**Pros:** no extra forward passes; the d=0 grounding is automatic.
**Cons:** the constraint is only enforced at `d ≈ 0`; collapse at large
`d` is still possible. Also entangles the architecture with the loss.

### Option B — Data anchor at d=1 (paper-faithful)

Split each training batch:
- A fraction (paper uses ~3/4) trains with the **standard diffusion/flow
  loss** at the smallest non-trivial step (`d = 1` in integer step-level
  units), supervising the composed `s` against the empirical `v_target`.
- The remaining fraction trains the self-consistency term at dyadic
  `d ∈ {2, 4, 8, …, K}`.

This is the direct lift of the paper's recipe. The d=1 fraction pins the
composed `s` to data; the d>1 fraction propagates that grounding via
recursive halving.

**Pros:** structurally identical to the paper's argument. Same recipe we
have to defend in the thesis chapter anyway.
**Cons:** adds one extra path through the training step. The split ratio
is a hyperparameter.

### Option C — Both

Use Option A's architectural zero-gate **and** Option B's d=1 data anchor.
The architecture handles the `d → 0` asymptote; the data anchor handles
collapse at all `d`. Probably the safest combination, at the cost of one
extra gate parameter and one extra batch-split bookkeeping path.

## Plan

1. Default to **Option B** in the first implementation — it matches the
   paper, so it is straight-forward to argue about in D3.
2. Sanity-check for collapse by monitoring three diagnostics during
   training:
   - `||Δ_φ(x_t, t, a, d=1)||` averaged over a held-out batch. Should be
     small if the d=1 data anchor is working.
   - `||s_composed(x_t, t, d) − base(x_t, t)||` at increasing `d`. Should
     grow with `d` (the adapter's correction is supposed to scale with
     the step size). Stays flat ⇒ adapter is not learning d-dependence.
     Collapses to a negative-of-base shape ⇒ cancellation collapse.
   - The self-consistency loss alone vs. the d=1 anchor loss alone. If
     the self-consistency loss drops fast while the anchor loss stays
     flat or grows, the adapter is satisfying Eq. 4 in a way that the
     data does not agree with — early collapse signal.
3. If diagnostics show drift toward any of the trivial fixed points, add
   Option A's zero-gate as a follow-up.
4. Run the same diagnostics on the existing AVID and HyperAlign shortcut
   configs once the new loss is wired up, before any long training.

## Open questions

- For DynamiCrafter (v-pred, integer `t ∈ {0, …, 999}`), the smallest
  meaningful step is `d = 1` timestep. Does the dyadic schedule terminate
  at `K = 512`, or do we cap earlier (e.g. `K = 16`) and rely on inference-
  time chaining beyond that? Cf.
  [[../30_Knowledge/related-work/shortcut-models]] which uses
  `K = 128` continuous units.
- Does the cancellation collapse manifest faster for `composition: add`
  (HyperAlign) or `composition: avid_mask_mix` (AVID-style)? The mask-mix
  has a learned gate that *could* pin the d=1 case to base ≈ pass-through,
  but its gate is currently conditioned on the captured base features, not
  on `d`. Worth investigating both.
- The `d=1` data anchor uses the standard diffusion loss. Does the AVID
  output adapter's `step_level_embed` need to be on the *no-grad* path for
  the self-consistency target computation, or do we want gradient through
  the embedding for the chained `Δ_φ(x', t+d, a, d)` call? Paper detaches
  the target; we should match unless we find a reason to do otherwise.

## Related

- [[../30_Knowledge/related-work/shortcut-models]] — Eq. 4, the `d = 0`
  flow-matching anchor, and the explicit discussion in §3 about why
  bootstrapped self-consistency works (and what grounds it).
- [[bug-training-shortcut-target-timestep.md]] — separate bug on the
  `t + d` simplification in the existing `two_step` path. That ticket
  is about the **target construction**; this one is about the **loss
  formulation**.
- [[../30_Knowledge/related-work/consistency-models]] — consistency
  models also rely on a data-side anchor (the boundary condition at
  `t = ε`). Same structural reason.
- [[../30_Knowledge/related-work/self-distillation]] — collapse modes
  in self-distillation more broadly; the cancellation fixed point here
  is a special case of the general "student matches a degenerate
  teacher" failure mode.
- Code (to-be-written): `Trainer._self_consistency_target` and
  `Trainer._micro_step` in
  `src/generative_flow_adapters/training/trainer.py`.
- Source config: this risk applies to any future
  `*_self_consistency_metaworld.yaml`; track when that lands.
