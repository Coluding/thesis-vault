---
date: 2026-07-01
category: added
deliverable: D2
meeting:
sources:
  - "[[../../20_Tickets/feat-eval-training-quality-metrics]]"
  - "[[2026-07-01-wan22-i2v-eval-loader]]"
---

# Quality metrics switched on in the Wan2.2 AVID i2v config

## What

The `QualityMetricSuite` infra has existed for a few days but was inert — every
config left `quality_metrics` empty. Turned it on in
`configs/diffusion_wan22_avid_i2v_metaworld.yaml`:

- **Paired (every eval cycle):** `quality_metrics: [psnr, ssim, lpips, mse]`,
  scored on decoded pixels vs the aligned ground-truth future.
  `quality_eval_num_batches: 4`, `quality_eval_num_steps: 10` (a cheaper sampler
  than the 50-step video panel). Both the adapted **and** the frozen-base
  rollout are scored, so wandb shows the base→adapted delta.
- **Distribution (own cadence):** `quality_dist_metrics: [fvd, fid]`,
  `quality_dist_num_batches: 16`.
- `data.val_fraction: 0.05` was added so the train script actually builds the
  held-out `eval_loader` — without it the whole `_run_eval_cycle` path (and thus
  paired metrics) is skipped. This closes the gap flagged in
  [[2026-07-01-wan22-i2v-eval-loader]].

## Why it matters

We were flying blind on sample quality during training — only losses were
logged. These give a quantitative, per-cycle base-vs-adapted quality signal
(the core D2/D4 adapter-benefit evidence) instead of eyeballing the video panel.

## Evidence

- Config loads and populates the `TrainingConfig` dataclass (all six quality
  fields resolve).
- `QualityMetricSuite(['psnr','ssim','mse'])` runs end-to-end on CPU over dummy
  uint8 `(B,T,3,H,W)` pixels and returns the expected keys. Deps present:
  `torchmetrics` 1.9, `cd-fvd`, `lpips`.
- No training-run numbers yet — a GPU run is still needed to confirm keys land
  in wandb/jsonl.

## Watch / next

- **Cadence is smoke-run-tuned, not production.** `eval_every_n_steps: 5` and
  `quality_dist_every_n_steps: 3` fire the (expensive) rollouts almost every
  step — fine to verify the keys appear, far too frequent for a real job. FID/FVD
  load Inception + I3D and roll out 16 batches; raise both cadences before a full
  training run.
- Confirm `eval_quality/adapted/*` + `eval_quality/base/*` land in wandb/jsonl on
  the first GPU run; then pick the production cadence.
