---
type: paper
status: living
last_updated: 2026-08-07
title: "High-Order Matching for One-Step Shortcut Diffusion Models (HOMO)"
authors: ["Chen et al."]
venue: arXiv:2502.00688
year: 2025
url: https://arxiv.org/abs/2502.00688
local_pdf:
relevance: D3 theory — prior art on the curvature limitation of shortcut models
---

# HOMO — higher-order supervision for shortcut models, and prior art on curvature

> **⚠ Prior art for our curvature analysis.** Found 2026-08-07. Our
> derivation is not the first identification of a curvature problem in
> shortcut models, and the thesis must say so.

## What it does

Argues that shortcut models suffer from "erratic trajectories, poor
geometric alignment, and instability, especially in high-curvature regions",
and that the "velocity-only approach fails to capture intrinsic manifold
geometry". Proposes supervision at acceleration and jerk order, with
theoretical approximation-error results.

## Why it matters to us

**The qualitative claim is already in the literature.** We derive that the
published velocity-averaging target is exact only at zero curvature and
carries a second-order bias otherwise. HOMO says shortcut models degrade in
high-curvature regions and that a velocity-only treatment misses the
geometry. Same direction.

**What our treatment adds, and how to position it:**

- A **specific second-order form**: the discrepancy is the sagitta,
  ≈ ½·κ·δ², so the error is quadratic in the step and vanishes only as
  κ → 0
- A **fixed-point argument**: the true field is not a fixed point of the
  averaging rule, so the bias is re-injected at every rung of the doubling
  tower rather than being trained away. This is the part that explains why
  more training does not help
- **Numerical verification at zero model error**, isolating the target
  construction from any model
- **Two named escapes**: fix the coordinates (endpoint inversion) or fix the
  geometry (a straight interpolant)

**Position it as:** an independent and more precise statement of a
limitation already identified, with the fixed-point argument and the
endpoint-inversion escape as the additions. **Do not claim it as novel.**
Claiming novelty here is a reviewer risk out of proportion to what the claim
is worth, and the derivation is just as useful when correctly attributed.

⚠ **Read before leaning on it.** arXiv preprint, no verified venue. Verify
the specific claims above against the paper text rather than the abstract
before citing them as prior art in detail.

## Related

- [[../theory/shortcut-v-averaging-bias]] — our derivation
- [[meanflow]] — the exact average-vs-instantaneous identity
- [[shortcut-models]] — the target being analysed
