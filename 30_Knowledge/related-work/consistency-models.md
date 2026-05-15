---
type: paper
status: living
last_updated: 2026-05-15
title: "Consistency Models"
authors: []
venue:
year:
url:
local_pdf: docs/paper/consistency_model.pdf
relevance: theory, baseline
deliverable: D3
---

# Consistency Models

> Few-step generative models trained with a consistency objective. The
> theoretical ancestor of the shortcut formulation. Our D3 borrows the
> loss form but applies it on a frozen base via the adapter rather than
> retraining the full model.

## Status of this note

**Stub.** Vendored PDF at `docs/paper/consistency_model.pdf`. Title,
authors, venue, year, URL — _needs verification from the PDF_.

## Why it matters for the thesis

- Consistency Models is the original few-step-sampling-via-consistency
  proposal. Any D3 chapter must cite it as ancestor.
- The thesis's anti-positioning is clean against this paper:
  - Consistency Models retrain the prior from scratch / via distillation.
  - The thesis keeps the prior frozen and bolts consistency behaviour onto
    the adapter.
- This makes Consistency Models both an **honest baseline** (does
  full-model retraining beat adapter-only at matched step budgets?) and
  a **theoretical reference** (the loss form descended from here).

## Key relationships to capture in the thesis

- How the consistency objective maps to the shortcut formulation (see
  [[shortcut-models]]).
- The role of the boundary condition `s(x_T, T, d) = x_0` (or its
  analogue) and whether it lifts cleanly to the adapter setting.
- Whether distillation from a teacher (one of the two regimes in this
  paper) is compatible with the frozen-base setup, or whether
  consistency-training-from-scratch (the other regime) is the closer
  analogue.

## Honest-baseline question for D3

If the thesis runs a head-to-head against a Consistency-Model baseline:
- Same base architecture, same data, same training budget.
- Full-model consistency vs. adapter-only consistency.
- Report few-step rollout quality as a function of `d`.

This is the harder version of the experiment than "just compare against
50-step DDIM," and it is the version that will hold up to careful
review.

## Open questions for the chapter

- Exact training objective and what trajectory pairs are used. _needs
  verification from the PDF_.
- Whether the paper's notion of "consistency" exactly matches the
  multistep self-consistency we implement, or differs in target form.
- The few-step quality numbers reported, for honest comparison.
- Whether the codebase has any consistency-models-aligned config that
  could serve as a baseline scaffolding (currently: the consistency
  *losses* exist as composable add-ons but no "consistency-from-scratch"
  config is in `configs/`).

## Related

- [[_MOC]]
- [[shortcut-models]] · [[self-distillation]] · [[dpm-solver]] — the few-step-sampling cluster
- [[../../10_now/positioning]] — D3 deliverable + anti-positioning
- `30_Knowledge/theory/` — derivations comparing consistency / shortcut / self-distillation losses (to be populated)
