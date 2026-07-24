---
type: bug
scope: eval
status: done
priority: medium
created: 2026-07-03
updated: 2026-07-03
resolution: fixed
resolution_note: >
  Added an `_updated` flag to `_TorchmetricPaired`/`_FID`/`_FVD`
  (quality_metrics.py), set True only after the underlying `update()` returns
  so a partially-failed (OOM'd) batch leaves it False. `compute()` returns
  `{}` when `_updated` is False instead of crashing on empty state. Follow-up
  not done: per-batch atomicity (adapted/base metric asymmetry if OOM hits
  mid-batch) — see body.
closed_at: 2026-07-03
related: []
---

# Native quality eval crashes with "No samples to concatenate" after eval OOM

## Symptom
During `_native_quality_eval`, an OOM is caught and logged
(`[eval] OOM during native quality eval ... reporting metrics from the batches
that fit`), but training then dies with:

```
ValueError: No samples to concatenate
  torchmetrics/image/lpip.py -> dim_zero_cat(self.all_scores)
```

## Root cause
In `trainer._native_quality_eval`, each batch does
`suites.setdefault("adapted", QualityMetricSuite(...)).update(...)`. `setdefault`
**inserts the suite into the dict first**, then `.update()` runs the metrics in
list order. LPIPS's `update` executes a VGG forward on GPU; if *that* OOMs, the
suite is already registered in `suites` but LPIPS accumulated **zero** scores.
The OOM is caught, then `suite.compute()` calls `LearnedPerceptualImagePatchSimilarity.compute()`
on empty state → `dim_zero_cat([])` raises.

`_MSE.compute()` already guarded this (`if self._count == 0: return {}`), but the
torchmetrics wrappers (`_TorchmetricPaired`, `_FID`) and `_FVD` did not.

## Fix
`src/generative_flow_adapters/training/quality_metrics.py`: added an `_updated`
flag to `_TorchmetricPaired`, `_FID`, `_FVD` (set `True` only *after* the
underlying `update` returns, so a partially-failed batch leaves it `False`).
`compute()` returns `{}` when `_updated` is `False`. The whole-cycle compute loop
then simply reports whatever metrics fully accumulated, and reports nothing (no
crash) when every eval batch OOM'd.

Verified: suite with no updates → `compute()` returns `{}`; after one update →
full metric dict incl. lpips.

## Follow-up (not done)
Per-batch atomicity: if OOM hits mid-batch, "adapted" may get mse/psnr for that
batch while "base" (computed later in the loop) never runs, giving asymmetric
adapted/base metrics. Could commit a batch only if both rollouts+updates succeed.
Real remedy for the OOM itself is lowering `training.extra.inference_max_area`.
