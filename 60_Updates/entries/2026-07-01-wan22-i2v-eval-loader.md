---
date: 2026-07-01
category: added
deliverable: D2
meeting:
sources:
  - "[[2026-06-30-wan22-variable-cond-frames]]"
---

# Wan2.2 I2V train script: wire up the held-out eval loader

## What

`scripts/train_wan22_i2v_metaworld.py` was calling `trainer.train(...)` without
an `eval_loader`, so it defaulted to `None`. Consequence: the trainer silently
skipped the periodic eval cycle, the quality-metric eval (FID/FVD/LPIPS from the
new `training/quality_metrics.py`), the final eval, and `best.pt` checkpointing
(which is gated on `eval_metric` improving). Only train loss was logged.

Mirrored the eval wiring already present in
`scripts/train_avid_shortcut_metaworld.py`, then made the eval **data source**
config-driven (not just CLI):

- New `DataConfig` fields: `eval_hdf5: str | None = None` (separate leak-free
  file) and `val_fraction: float = 0.05` (random window-level split of `hdf5`,
  in-distribution — adjacent windows from one episode can straddle the split).
  Set in `configs/diffusion_wan22_avid_i2v_metaworld.yaml` under `data:`.
- Cadence lives in `TrainingConfig` (`eval_every_n_steps`, `eval_num_batches`,
  `eval_metric`) — already there; YAML sets `eval_every_n_steps: 5`.
- CLI flags `--eval-hdf5` / `--val-fraction` / `--eval-every` / `--eval-batches`
  now **override the config only when passed** (defaults are `None`).
- `want_eval = bool(eval_every_n_steps) and (eval_hdf5 or val_fraction > 0)`.
  `eval_hdf5` wins over the split when both are set.
- Builds `eval_loader` and passes it to `trainer.train(..., eval_loader=...)`;
  prints `eval: ...` status (or `eval: disabled`).

## Why

Without a held-out loader there were no validation metrics and no best-checkpoint
tracking for the Wan2.2 diffusion-forcing runs — only raw train loss.

## Note / open question

The `--val-fraction` split is in-distribution (episode leakage across the split
boundary). For a clean held-out number use `--eval-hdf5` with a separate file.
Eval still requires the cadence set (`--eval-every` or YAML) to actually fire.
