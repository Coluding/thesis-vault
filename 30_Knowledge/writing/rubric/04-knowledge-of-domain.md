---
type: writing
status: living
last_updated: 2026-08-03
rubric_item: knowledge-of-study-domain
category: research
current_band: "8"
target_band: "8-9"
sources:
  - "[[_index]]"
  - "[[../../theory/shortcut-v-averaging-bias]]"
  - "[[../../related-work/avid]]"
  - "[[../../../50_Decisions/decided/shortcut-target-endpoint-vs-v-averaging]]"
---

# Rubric 4 — Knowledge of study domain

## The rows

| | |
|---|---|
| 10 | In-depth knowledge; profoundly new insights to the field |
| 9 | In-depth knowledge; able to **place the field under a new light** |
| **8** | **On top of subjects discussed in thesis; able to add new knowledge to the study domain** |
| 7 | Understands the subject matter and related research; can incorporate it for the problem at hand |
| 6 | Understands the subject matter on a **textbook level** |
| 1–5 | Does not understand all of the subject matter discussed |

## What it actually asks

The 6→7→8 ladder is *textbook* → *can apply it* → *can add to it*. This is
partly a **viva** item, not only a document item: the committee probes
whether you understand what you wrote. Everything in the thesis must be
defensible under questioning — which means **do not write anything you
cannot derive on a whiteboard.**

## Evidence inventory

**The curvature analysis is the flagship.** It is a derivation, not an
observation: eq. (4) averages two half-step velocities; the arithmetic mean
is the Euclidean centroid while the manifold wants the Riemannian one; the
gap is the sagitta (≈ κ·δ²/2); and — the part that makes it a real result —
**the true field is not a fixed point of the averaging rule**, so the bias
does not train away, it compounds up the doubling tower.

Corroborated numerically at *zero model error*: the averaged target lands
5.1% / 16.1% / 24.1% off at s = 1/4, 1/2, 3/4; the endpoint/displacement
target lands 0.000000 at every step. **Theory plus synthetic verification
at zero model error is the strongest form this item takes**, and we have it.

⚠ **Corrected 2026-08-07.** An earlier version of this paragraph claimed the
argument was "confirmed empirically by the flow-vs-diffusion shortcut
result". It is not. That experiment's diffusion arm ran
`endpoint_inversion`, the *exact* construction, so the theory predicts it
should have worked and it did not; the comparison is confounded and the
confound runs against the attribution. The A4 2×2 (`v_average` vs
`endpoint_inversion`, one base, one depth, config-only) is what would supply
empirical support.

**Domain understanding demonstrated by correction.** Recognising that
action conditioning on a frozen video base is *AVID's own contribution* —
and repositioning the entire thesis accordingly rather than claiming it —
is a knowledge signal. The inverse (claiming AVID's contribution as ours)
is the single most expensive error available on this item, and it was
avoided ([[../../related-work/avid]] §Correction).

**Mechanism-level understanding of the architecture.** Knowing *why* AdaLN
carries a signal that cross-attention drops — normalised multiplicative
modulation vs unnormalised additive residual — is architectural knowledge,
not empirical observation. Similarly: `qk_norm` rescues the logits while
the value pathway stays unnormalised.

## The traps that invert this item

Two of the domain's own gotchas (CLAUDE.md Part 12) sit directly on this
item. A single sloppy sentence turns a strength into demonstrated
misunderstanding, because a committee member **will** know:

1. **"Shortcut modelling fails on diffusion" is false.** Consistency
   distillation on VP diffusion demonstrably works, and our own endpoint
   fix is exact on the curved manifold. The correct claim is about **the
   published velocity-averaging target**, and it has two escapes: fix the
   coordinates (endpoint inversion) or fix the geometry (flow, κ=0).
2. **Diffusion vs flow prediction types.** ε / x₀ / v are different
   targets; the shortcut formulation differs. Always state `model_type`
   and `prediction_type`.

Also live: "shortcut" (Frans et al. sense) vs generic consistency
self-distillation — related, different derivations.

## Gaps to 9

The 9-row is "place the field under a new light". That requires
**generalising past our own setting**. We have the raw material for one
such claim:

> Adapting a frozen generative prior to a *new conditioning variable* is
> governed by two things the literature treats as implementation details:
> **how the conditioning enters relative to the residual stream** (it must
> be scale-free, or it drowns), and **what the training objective pays for
> that conditioning** (here ~0.45% of the loss, so appearance correction
> always outbids it). Neither is visible in any standard readout.

Stated at that altitude, it applies to conditioning-on-frozen-priors as a
class, not to action-conditioned video world models specifically. That is
the 9-row move, and it costs an afternoon of thinking rather than a GPU.

## Why diffusion appears better suited (2026-08-07) — measured, unexplained

Action following is demonstrated on four backbones across both objectives
(DC, EA V5, EA V5.1, Wan-distilled), with both diffusion backbones above
both flow backbones. **We report the difference and we do not have a
mechanism for it.**

**Two rival explanations are eliminated with existing data**, and reporting
those eliminations is the substance here:

- **Parameterisation.** `prediction_type: velocity` holds across *all four*
  cells, so it cannot explain a difference it does not vary with. The
  contrast is attributable to `model_type`, not to the prediction target.
  This is a control we obtained by construction rather than by design, and
  it should be stated as such.
- **Noise-level distribution.** Dead by the flat σ-sweep (H8 in
  [[../ablation-axes]]). Report it: it is the obvious first guess, and a
  reader who does not see it tested will assume it was not.

**What the measurement constrains.** Adaptation *capacity* is tied
(−74.9 % vs −73.6 %) while *specificity* differs (+36 %). So whatever the
mechanism is, it is not that flow bases are harder to adapt; it is that less
of the adaptation becomes action-conditioned.

⚠ **Do NOT extend the curvature argument to cover this.** The curvature
result is about the **shortcut target construction**: the arithmetic mean of
two tangents is not the chord, and the true field is not a fixed point of
the averaging rule. That argument involves no model and no conditional
distribution. A claim that "a curved trajectory leaves the base less
committed, so actions matter more" is a *different* proposition that happens
to share the word *curvature*; trajectory geometry and conditional entropy
are not the same quantity, and nothing we have measured links them. The
unified "one property explains both halves" story is attractive and
unsupported. Considered and rejected 2026-08-07.

**If a mechanism is wanted**, the measurable one is action-attributable
variance in the objective per backbone, which is what the **IDM ceiling**
(B5 in [[../open-experiments-for-thesis]]) reports, in minutes of GPU.
Until then the honest form is: *measured, robust across two independent
model families, mechanism open.*

## Optimisation queue

- [ ] **Write the generalised claim above** and place it in §7.1
      (Discussion — the boundary), then echo one sentence of it in §1.4.
      *(Q7-adjacent; the Originality and Knowledge upgrades are the same
      paragraph written at the right altitude.)*
- [ ] **Precision pass on the two traps.** Grep the whole draft for
      "shortcut", "diffusion", "flow", "consistency"; verify every sentence
      names the right object and states the prediction type where it
      matters. *(Cheap, and it protects against the item's worst outcome.)*
- [ ] **Derive, don't assert, in Ch4.** The curvature section should carry
      the sagitta derivation explicitly — it is the one node of the arc
      that is *proven*, and proofs are where this item is won.
- [ ] **Prepare the viva surface.** For each headline claim, be able to
      answer: what would falsify it, what is the mechanism, what did you
      not test. Worth a short private note per claim closer to the defence.
- [ ] **Do not overreach on the economics claim.** ~0.45% is measured on a
      *teacher-forced denoising* objective in specific cells; it is not a
      universal constant. State the scope or it becomes an attack surface.

## Where it lands in the thesis

- Ch4 §4.5 — the curvature derivation (the proven node; its own section)
- Ch2 — the AVID positioning, stated correctly and early
- Ch6 — the mechanism explanations (why AdaLN, why the residual add)
- §7.1 — the generalised claim (the 9-row move)
