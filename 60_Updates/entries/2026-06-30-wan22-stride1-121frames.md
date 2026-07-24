---
date: 2026-06-30
category: change
deliverable: D2
meeting:
sources:
  - "[[../../50_Decisions/decided/wan22-temporal-window-stride1]]"
  - "[[../../50_Decisions/decided/metaworld-frame-stride-load-time]]"
---

# Wan2.2 MetaWorld clips: contiguous stride-1, 121-frame window (5 → 31 latent frames)

## What

The live Wan2.2 training geometry was **17 pixel frames → 5 latent frames** at
`frame_stride: 4`, giving only a **4-latent-frame** prediction horizon
(`cond_frames: 1`). Changed both Wan2.2 configs to:

- `data.frame_stride: 4 → 1` (contiguous)
- `model.extra.temporal_length: 17 → 121` → `t_lat = 31` latent frames

Files: `configs/diffusion_wan22_i2v_metaworld.yaml`,
`configs/diffusion_wan22_avid_i2v_metaworld.yaml`.

## Why it matters

5 latent frames is too short to learn meaningful action-conditioned dynamics,
and the 4× striding put inter-frame motion outside Wan2.2's native 24fps
distribution. 121 contiguous frames matches the base's native window, covers
~40% of a 300-frame episode (vs ~21% under the old stride-4 trick), keeps motion
in-distribution, and uses raw per-frame actions (no summed-`Δgripper`). VRAM is
not the constraint — the 16×16 latent is only ~1984 tokens at 31 frames.

This **amends** [[../../50_Decisions/decided/metaworld-frame-stride-load-time]]
(stride-4 + action-SUM) **for Wan2.2 only**; that decision still holds for any
DynamiCrafter/AVID D1 configs.

## Watch

- Action magnitudes now ~4× smaller (raw vs summed-over-4) — normalisation is a
  watch-item, not yet changed.
- Throughput: frozen VAE now encodes 121 full-res frames/sample — if it
  bottlenecks, cache latents or drop to a 65f / 17-latent window.

## Next

- Run training at the new geometry; confirm action effect is visible in the
  eval video panel and watch action-norm stability.
