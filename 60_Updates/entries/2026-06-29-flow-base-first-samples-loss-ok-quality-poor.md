---
date: 2026-06-29
category: finding
deliverable: D3
meeting:
sources:
  - "[[../../30_Knowledge/experiments/20260629-flow-vs-diffusion-shortcut-samples]]"
  - "[[2026-06-19-pivot-flow-matching-base]]"
---

# First samples on the flow-matching base: loss converges, generation quality not there yet

## What

First sample-quality batch since the flow-matching pivot
([[2026-06-19-pivot-flow-matching-base]]). Three runs on MetaWorld, artifacts in
`data/results/20260629/`:

- **Flow matching + shortcut** (768×1280, ~15k steps)
- **Flow matching, no shortcut** — baseline (768×1280, ~15k steps)
- **Diffusion + inversion shortcut** (1536×1600, ~1.4k steps — much earlier)

## Finding

- **Training is healthy.** `base_loss` converges cleanly and stably on all three
  (flow ~0.4 → ~0.13–0.15; diffusion ~0.4 → ~0.07; numbers eyeballed off W&B
  chart axes, not logged scalars — see the experiment note). The flow base is
  κ=0, so the v-averaging shortcut bias that motivated the pivot is gone, and the
  per-rung shortcut loss is well-behaved.
- **Samples are still poor.** Across all three conditions the videos recover the
  coarse robot-arm + table structure but show heavy blur, fogging, collapse and
  colour-drift artifacts, degrading over the rollout.
- **Net:** loss health is no longer the blocker — **generation quality is**.
  Likely suspects: undertraining (esp. the ~1.4k-step diffusion run), resolution
  / VAE decode, action-conditioning strength, rollout drift.
- **Bonus payoff: flow is much faster.** The flow runs train and sample
  noticeably faster than the diffusion run (qualitative; straight κ=0 ODE → fewer
  sampling steps). Concrete dividend of the pivot for D3/D4 fast rollout — but not
  a clean measurement here (diffusion was higher-res + a different backbone; a
  controlled steps/sec + sampling-NFE comparison is still to be logged).

## Why it matters

The pivot delivered on its premise (clean objective, stable training) but has not
yet delivered usable D3 samples. The next push is sample quality, not loss — and
keep training longer before reading the diffusion-inversion variant.

## Status

Finding, not a result claim. No usable few-step D3 evidence yet; grid layout (GT
vs prediction columns / NFE) still _needs verification_, so flow-shortcut vs
no-shortcut cannot yet be compared at matched NFE. Detail + per-rung loss table
in [[../../30_Knowledge/experiments/20260629-flow-vs-diffusion-shortcut-samples]].
