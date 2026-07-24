---
type: feat
scope: eval
status: done
priority: high
created: 2026-06-27
updated: 2026-07-15
resolution: shipped
resolution_note: >
  The "still pending" item (run on GPU, confirm keys land in wandb/jsonl,
  sanity-check FID/FVD) is done — confirmed 2026-07-15 by pulling
  train/metrics/{psnr,ssim,lpips,mse}_step_N and {fid,fvd_i3d}_epoch directly
  from a real wandb run (coluding/avid-metaworld/pg3x72uc and our own WAN
  runs), all logging cleanly with sane (non-NaN, non-degenerate) values across
  the per-rollout-step grid.
closed_at: 2026-07-15
related: []
---

# feat(eval): wire perceptual/distribution quality metrics into training eval

## Problem

Training-time evaluation currently computes **only losses** (`base_loss`,
consistency terms) in `_run_eval_cycle` (`trainer.py:499-533`). There are **no
image/video quality metrics** — the standard generative-visual eval signals
(PSNR, SSIM, LPIPS, FID, FVD) are entirely absent. We're flying blind on actual
sample quality during training and have no quantitative base-vs-adapted delta.

Metric definitions + exact computation: [[../30_Knowledge/tech/generative-visual-metrics]].

## What already exists (reuse, don't reimplement)

- `src/external_deps/avid_utils/metrics.py` — working `PSNR`, `SSIM`, `LPIPS`,
  `FID`, `FVD` (i3d/videomae), `MSE` classes. Currently unused by the trainer.
- Sample generation already runs at eval (`_maybe_generate_samples`,
  `generate_samples`) producing latents `(B, C, T, H, W)`.
- `wandb_logger._decode_to_uint8()` (`wandb_logger.py:232-238`) already
  VAE-decodes latents → uint8 RGB for the `GT | base | adapted` panels. The
  decoded pixels needed for metrics are already produced here.
- Deps needed by the AVID metrics (`torchmetrics`, `pytorch_fid`, `cdfvd`,
  `einops`) are **not yet in pyproject.toml** — must be added.

## Design

**Two-tier cadence** (FID/FVD are distribution metrics — noisy on small per-step
eval batches; paired metrics are reliable because the world-model eval has
aligned ground-truth future frames):

- **Per eval cycle:** PSNR, SSIM (+ optional LPIPS) — paired generated-vs-GT,
  averaged over `eval_num_batches`. Cheap, no big feature nets.
- **Separate `quality_eval_every_n_steps` (rarer):** FID, FVD — accumulate
  features over a larger fixed sample budget, compute once per cycle. Gated/off
  by default if too expensive.

**Base vs adapted:** compute every metric for both the frozen-base output and
the adapted output (the panel already decodes both). The delta is the
quantitative adapter-benefit evidence for D2/D4.

**Logging:** scalar keys `eval/psnr_adapted`, `eval/ssim_adapted`,
`eval/psnr_base`, ... via `wandb_logger.log_metrics` + `metrics.jsonl`.

## Touch points

- `config.py` (108-116): add `quality_metrics: list[str]`,
  `quality_eval_every_n_steps`, sample-budget knobs.
- `trainer.py` `_run_eval_cycle` (499-533): call a new metrics step on decoded
  pixels; accumulate FID/FVD across batches, compute at cycle end.
- new `training/quality_metrics.py`: thin adapter over the AVID metric classes
  (handle `(B,T,3,H,W)` → per-frame for FID, clips for FVD).
- `pyproject.toml`: add metric deps.

## Open choices (confirm before build)

1. LPIPS in the per-cycle tier or skip for now?
2. FID/FVD default on or off (cost / sample-budget)?

## Implementation status (2026-07-01)

Both open choices resolved: **skip LPIPS** in the per-cycle tier (PSNR/SSIM
only, LPIPS still selectable via config); **FID/FVD off by default**, opt-in per
config. Landed in `generative-flow-adapters`:

- `training/quality_metrics.py` — new `QualityMetricSuite`, a **native**
  implementation over `torchmetrics` (PSNR/SSIM/LPIPS/FID) + `cd-fvd` (FVD).
  **Does not import `external_deps`** (hard rule) — the vendored AVID module is
  untouched. Frames enter as uint8 `(B,T,3,H,W)`; image metrics get `[0,1]`
  `(B*T,3,H,W)`, FVD gets uint8 `(B,T,H,W,C)`. All six accumulate correctly
  across batches (this also fixes the AVID FVD's last-batch-only bug).
- `training/wandb_logger.py` — public `decode_to_uint8` + `can_decode`.
- `training/trainer.py` — `_generate_eval_rollout` (shared-noise adapted + base
  rollout) and `_run_quality_eval`; paired metrics wired into `_run_eval_cycle`,
  distribution metrics on their own cadence in `train()`.
- `config.py` — `quality_metrics`, `quality_eval_num_batches`,
  `quality_eval_num_steps`, `quality_dist_metrics`,
  `quality_dist_every_n_steps`, `quality_dist_num_batches`.
- `pyproject.toml` — `torchmetrics[image]` (pulls torch-fidelity + lpips) +
  `cd-fvd` as **required** core deps.

Verified: imports clean; paired suite (mse/psnr/ssim/lpips) run end-to-end on
CPU — random-pair and identical-pair (mse 0 / ssim 1) sanity both correct.

**Config activation (2026-07-01):** enabled in
`configs/diffusion_wan22_avid_i2v_metaworld.yaml` — paired
`quality_metrics: [psnr, ssim, lpips, mse]` (every eval cycle,
`quality_eval_num_batches: 4`, `quality_eval_num_steps: 10`) plus distribution
`quality_dist_metrics: [fvd, fid]` (`quality_dist_num_batches: 16`). `data`
gained `val_fraction: 0.05`, so the train script builds the required
`eval_loader` (see
[[../60_Updates/entries/2026-07-01-wan22-i2v-eval-loader]]) and the paired path
is live. Config loads and populates the dataclass; paired suite re-verified
end-to-end on CPU. Deps present (`torchmetrics` 1.9, `cd-fvd`, `lpips`).

Cadence is currently set for a **smoke run** to confirm the keys fire quickly:
`eval_every_n_steps: 5` (paired) and `quality_dist_every_n_steps: 3` (FID/FVD).
Both are far too frequent for a long job — the FID/FVD rollout (16 batches,
loads Inception + I3D) every 3 steps would dominate wall-clock. **Raise both
before a full training run.**

**Still pending (keep open):** run on GPU and confirm keys land in wandb/jsonl
(`eval_quality/adapted/*`, `eval_quality/base/*`); sanity-check FID/FVD once;
decide the production cadence + `quality_dist_num_batches` after seeing noise.

## Acceptance

- Per-cycle PSNR/SSIM logged for base + adapted on every eval.
- FID/FVD computed on their own cadence over a fixed budget, logged with sample
  count.
- No fabricated numbers; cadence + budget recorded so runs are reproducible.
