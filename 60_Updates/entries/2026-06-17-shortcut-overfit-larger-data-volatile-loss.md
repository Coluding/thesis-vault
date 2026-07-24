---
date: 2026-06-17
category: finding
deliverable: D3
meeting:
sources:
  - "[[../../30_Knowledge/experiments/avid-shortcut-anchor045-volatile-loss]]"
  - "[[entries/2026-06-05-anchor-baseline-confirms-shortcut-fewstep-gain]]"
  - "[[../../20_Tickets/experiments/exp-shortcut-vs-image-only-anchor-baseline]]"
  - "[[../../50_Decisions/open/shortcut-collapse-mitigation-anchor-vs-gate]]"
---

# Shortcut few-step robustness was small-data overfit — degrades on larger data, shortcut loss is volatile

## What

First real-backbone shortcut run on the **larger** MetaWorld set
(AVID/DynamiCrafter, velocity, output adapter, `anchor_prob=0.45`). The
prediction **got worse**, not better, and the **`shortcut_direction_loss` is
volatile** throughout training while the honest `base_loss` converges cleanly.
This **walks back** the 2026-06-05 finding — that earlier "shortcut training
buys few-step robustness" call was a small, local, qualitative run, and the
robustness does not survive more data.

## Why it matters

The headline D3 claim (step-size-conditioned adapters give cheap, consistent
few-step rollout) currently has **no clean supporting evidence on real data** —
the apparent robustness was an overfitting artifact. The split signal is the
useful part: the base denoising objective is healthy; the instability is
isolated to the **self-consistency / shortcut term**. So the open question is
narrowed to "is the shortcut objective itself unstable, or are we just mixing
step-size loss magnitudes," not "is the whole approach broken."

## Evidence / sources

Run artifacts in `data/results/20261706/` (config + W&B chart + sample PNGs);
wandb project `avid-shortcut-metaworld-0.45`. Values eyeballed off the exported
charts (not logged scalars); wandb run id / commit / ckpt still `_needs
verification_`:

- `train/base_loss`: smooth, stable, ~0.06–0.07.
- `train/shortcut_direction_loss`: volatile, ~0.01–0.12, no clear downtrend.
- `train/loss` (total): ~0.07 with spikes to ~0.2, tracking the shortcut term.
- Samples: coherent arm frames mixed with collapsed/foggy + red-drift frames.

Detail: [[../../30_Knowledge/experiments/avid-shortcut-anchor045-volatile-loss]].

## Next

- **In progress:** logging shortcut loss **separately per step size** to test
  whether the volatility is just mixed loss magnitudes across step sizes vs
  genuine training instability.
- Anchor-vs-shortcut control ([[../../20_Tickets/experiments/exp-shortcut-vs-image-only-anchor-baseline]])
  is **on hold** pending that diagnosis — its premise no longer holds.
- Owed: the sourced MSE-vs-NFE curve, plus run id / commit / ckpt and the
  small-data comparison pointers to make the regression quantitative.
