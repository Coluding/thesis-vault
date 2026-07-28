---
date: 2026-07-24
category: feature
deliverable: D1
meeting:
sources: []
---

# Step-0 baseline eval is now config-gated (skip FID/FVD at init, keep inference + loss)

## What

On a fresh run the trainer runs a full baseline eval *before* the first
gradient step so every metric has an at-init reference. That baseline had
three hard-wired components:

1. `_run_eval_cycle` — held-out **loss** eval + paired quality metrics
   (psnr/ssim/lpips/mse)
2. `_native_eval_grid` — the native Wan generation grid (**inference**)
3. `_run_quality_eval(quality_dist_metrics)` — **FID/FVD** (loads Inception/I3D)

The FID/FVD pass at step 0 is expensive and often not wanted for the at-init
snapshot. Added three `TrainingConfig` toggles that gate each component of the
**step-0 baseline only** — the periodic mid-training cadences are untouched:

- `baseline_eval_loss` (default `true`)
- `baseline_eval_inference` (default `true`)
- `baseline_eval_quality` (default `true`) — gates both the paired quality
  metrics (via a new `run_quality` arg on `_run_eval_cycle`) and the FID/FVD
  distribution pass.

Set `baseline_eval_quality: false` to get the inference grid + loss reference
at step 0 without paying for the quality pass. All defaults preserve the
previous full-baseline behaviour.

## Where

- `config.py` — new fields on `TrainingConfig`.
- `training/trainer.py` — step-0 block honours the flags; `_run_eval_cycle`
  gained `run_quality: bool = True`.
- `configs/wan22/diffusion_wan22_avid_xattn_i2v_metaworld.yaml` — set as the
  example (`baseline_eval_quality: false`).

## Why

The step-0 quality eval (FID/FVD) dominates init-time wall clock and the at-init
distribution metric is rarely informative; the inference grid + loss reference
are what's actually useful at step 0.
