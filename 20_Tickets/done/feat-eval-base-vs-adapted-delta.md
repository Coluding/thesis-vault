---
type: feat
scope: eval
status: done
priority: high
created: 2026-07-09
updated: 2026-07-15
resolution: shipped
resolution_note: >
  Implemented as part of the 2026-07-14/15 debugging session, not as a
  standalone quality-metrics feature: `AdaptedModel.forward(..., return_base=True)`
  + `Trainer._forward_and_loss`'s `denoise_adapter_delta` (per-step, paired
  base-vs-adapted denoising loss on the same batch) and `Trainer._probe_eval`'s
  `probe_denoise_{base,adapted,delta}` (same, on a frozen low-variance probe
  batch every eval) — both always-on whenever the model supports it, not
  opt-in. This is exactly the "report the delta explicitly, not just two
  absolute numbers" ask. Distribution-metric (FID/FVD) deltas both ways were
  already separately live via QualityMetricSuite
  (feat-eval-training-quality-metrics, done) — so both halves of this
  ticket's ask are covered.
closed_at: 2026-07-15
related: ["[[../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]"]
---

# feat: report base-vs-adapted delta as the headline adapter metric

## Why

With the 20260907 run using a **real pretrained WAN frozen base** (prior runs had
random base weights), the aggregate `base_loss` sitting flat ~0.15 is *expected* —
a strong prior already sits near its denoising floor, so the trend says almost
nothing about whether the adapter works. See
[[../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]].

The metric that actually answers "is the adapter doing anything?" is the
**frozen-base vs base+adapter delta** on the same eval — which the
`QualityMetricSuite` (`training/quality_metrics.py`, see [[../10_now/architecture]])
was built to compute (it scores both the adapted and frozen-base sampler).

## What

Surface the base-vs-adapted delta as a first-class, always-on eval readout for
this run (and going forward):

- Per-frame paired metrics (PSNR/SSIM/LPIPS/MSE) — adapted vs frozen-base.
- If feasible on the eval set, the distribution metrics (FID/FVD) both ways.
- Report the **delta** explicitly, not just the two absolute numbers, so
  "adapter ≈ base" is unmissable.

## Decision rule

- **Adapter ≈ frozen base** (delta ~0) ⇒ the adapter is contributing ~nothing over
  the prior → confirms the under-incentivised / inert-adapter hypothesis. Move to
  fixes: adapter LR / gate init, or forcing the base to need actions (base-path
  condition dropout, residual/base-subtracted target).
- **Adapter clearly better than base** ⇒ the adapter *is* adding value; then the
  open question narrows to whether that value is *action-driven* (→
  [[../experiments/exp-conditioning-action-shuffle-ablation]]).

## Notes

Cheapest and most reframing test — **run first**. Requires the VAE decoder +
inference sampler to be wired on the eval (per the quality-metrics cadence).
