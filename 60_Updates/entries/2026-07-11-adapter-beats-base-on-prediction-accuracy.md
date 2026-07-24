---
date: 2026-07-11
category: finding
deliverable: D4
meeting:
sources:
  - "[[../../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]"
  - "[[../../10_now/product-state]]"
  - "[[../../50_Decisions/open/action-conditioning-injection-mechanism]]"
  - "[[../../20_Tickets/feat-eval-base-vs-adapted-delta]]"
  - "[[../../20_Tickets/experiments/exp-conditioning-action-shuffle-ablation]]"
---

# Adapter beats the frozen base on prediction accuracy — flat loss was a red herring

## What

First run on a **genuinely pretrained Wan2.2-TI2V-5B frozen base** (prior runs
loaded random WAN weights → the adapter was learning everything from scratch). On
the base-vs-adapted eval, the adapter **clearly beats the frozen base on
reconstruction and the gap widens with training**, while **degrading
perceptual/distribution metrics** — the classic **regression-to-the-mean** (blur)
signature.

## Why it matters

This **overturns the earlier "the adapter isn't doing much" read**, and fixes the
yardstick. With a strong frozen base, `base_loss` sitting flat is *expected* (the
base is already near its denoising floor) — the loss trend is uninformative. The
**base-vs-adapted delta is the right metric, and it's positive.** For a world model
aimed at **planning, prediction accuracy (PSNR/MSE) is the headline** → the adapter
helps; the realism regression is a separate, known MSE-objective tradeoff.

## Evidence / sources

From `data/results/20260907/button/{adapted_eval,base_eval}.png` (W&B eval curves,
button-press task). **Numbers eyeballed off the chart axes — not logged scalars;
wandb run id / ckpt / commit `_needs verification_`** (hard rule 8):

| metric | frozen base (flat) | adapted (trend) | verdict |
|---|---|---|---|
| PSNR ↑ | ~15.6 | ~15.7 → ~16.8, rising | adapter wins, widening |
| SSIM ↑ | ~0.80 | ~0.815 → ~0.833 | adapter wins |
| MSE ↓ | ~0.0275 | ~0.0265 → ~0.021 | adapter wins |
| LPIPS ↓ | ~0.357 | 0.345 → ~0.40 | adapter loses (degrades) |
| FVD ↓ | ~1250 | 1300 → ~1850 | adapter loses |
| FID ↓ | ~75 | 73 → ~100 | adapter loses |

Full reading in [[../../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]].

## Next

- **Cite the real numbers**: pull PSNR/SSIM base-vs-adapted straight from wandb
  (already logged every eval) → replace the eyeballed values.
- **Shuffle test** ([[../../20_Tickets/experiments/exp-conditioning-action-shuffle-ablation]]):
  is the win *action-following* or a better task prior?
- **Is the blur the shortcut `distillation` target?** — the action-free shortcut
  isolation + no-shortcut control would show how much realism the shortcut costs.
- **Headline-metric decision** (prediction accuracy vs generation realism) — decides
  how every result from here reads; still to open as a vault decision.
