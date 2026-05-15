---
type: paper
status: living
last_updated: 2026-05-15
title: "DPM-Solver — Fast High-Order ODE Solver for Diffusion Models"
authors: []
venue:
year:
url:
local_pdf: docs/paper/dpm_sovler.pdf
relevance: baseline
deliverable: D3, D4
---

# DPM-Solver

> Solver-side few-step inference for diffusion. Honest non-trained
> baseline against which our shortcut adapter must compete at matched
> step budgets. The "do nothing to the model, just sample better"
> reference point.

## Status of this note

**Stub.** Vendored PDF at `docs/paper/dpm_sovler.pdf` (note the typo in
the filename — "sovler"). Title, authors, venue, year, URL — _needs
verification from the PDF_.

## Why it matters for the thesis

- DPM-Solver is the **untrained baseline** for D3. It achieves few-step
  sampling without changing the model — just by using a smarter ODE
  solver on the learned score / velocity field.
- For the D3 chapter to be honest, the shortcut adapter's few-step
  rollout quality must be compared to DPM-Solver at matched step budgets
  on the *same* base model.
- This is the comparison that exposes whether D3's contribution is
  empirically worth it. If the shortcut adapter is within noise of
  DPM-Solver at every step budget, the D3 chapter's claim collapses to
  "controllable few-step generation" rather than "improved few-step
  quality."

## Honest-baseline question for D3

If the thesis runs a DPM-Solver baseline:
- Same base model, same `x_T`, same conditioning.
- Sweep step count: 1, 2, 4, 8, 16, 50.
- Report rollout quality (FID / MSE / action-following) as a function of
  steps.
- Compare against shortcut-adapter at the same step counts.

This curve is one of the headline figures in D3 and D4.

## Open questions for the chapter

- DPM-Solver's reported sweet spot (often 10–20 steps with quality
  matching 1000-step ancestral) — _verify the exact numbers_.
- Whether higher-order variants (DPM-Solver++ / DPM-Solver-2) are the
  right baseline reference and which ones are commonly used.
- Whether DPM-Solver applies to flow-matching velocity fields directly or
  requires the diffusion-noise parameterisation. Affects which of our
  configs can host the comparison.
- Whether the codebase has any DPM-Solver integration today (likely via
  the optional `diffusers` integration — _needs verification_).

## Related

- [[_MOC]]
- [[shortcut-models]] · [[consistency-models]] · [[self-distillation]] — the few-step-sampling cluster (trained variants)
- [[../../10_now/positioning]] — D3 / D4 deliverables
- [[../../10_now/architecture]] — see Inference
