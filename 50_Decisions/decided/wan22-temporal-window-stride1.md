---
type: decision
status: decided
created: 2026-06-30
decided_at: 2026-06-30
updated: 2026-06-30
scope: data
amends: "[[metaworld-frame-stride-load-time]]"
related:
  - "[[metaworld-frame-stride-load-time]]"
  - "[[../../30_Knowledge/tech/frame-stride-conditioning]]"
  - "[[../../60_Updates/entries/2026-06-30-wan22-stride1-121frames]]"
  - "code: configs/diffusion_wan22_i2v_metaworld.yaml:16-17,32"
  - "code: configs/diffusion_wan22_avid_i2v_metaworld.yaml:16-17,31"
---

# Decision: Wan2.2 MetaWorld clips — contiguous (stride 1), long temporal window (121f / 31 latent)

## Status

**Decided & implemented 2026-06-30.** Scoped to the **Wan2.2** configs only.
**Amends** (does not globally supersede) [[metaworld-frame-stride-load-time]]:
that decision's `frame_stride: 4` + action-SUM remains in force for any
DynamiCrafter/AVID configs still used for the D1 adapter-taxonomy work. The
calculus that motivated stride-4 changed when the world-model base moved to
Wan2.2.

## Trigger

Audit of the live Wan2.2 training geometry revealed the clip was only
**17 pixel frames → 5 latent frames** (`t_lat = 1 + (17-1)/4`), and at
`frame_stride: 4` those frames were subsampled (4× jumps). With
`cond_frames: 1` that leaves a **4-latent-frame** prediction horizon — far too
short for action-conditioned dynamics, and the 4× striding pushes inter-frame
motion outside the base's native 24fps distribution.

## Decision

For Wan2.2 configs:

1. **`data.frame_stride: 1`** — contiguous frames. Keeps motion inside the
   frozen Wan2.2 base's native 24fps continuity (the backbone was trained on
   contiguous 121f @ 24fps), and makes the action-SUM path a no-op (each kept
   frame retains its raw delta-action — the awkward summed-`Δgripper` caveat of
   the prior decision disappears).
2. **`model.extra.temporal_length: 121`** → `t_lat = 1 + (121-1)/4 = 31` latent
   frames, matching the Wan2.2-TI2V-5B native window. `temporal_length` is kept
   `≡ 1 (mod 4)` so the VAE (stride-4 temporal) encodes with no truncation/pad.

## Why this is not the old "5% coverage" problem

The 2026-06-04 decision chose stride 4 because DynamiCrafter's 16-frame limit
made a *contiguous* clip cover only ~5% of a 300-frame episode (no visible
action effect). Wan2.2 removes that constraint by allowing a **long** window:

| | DynamiCrafter (stride 4) | Wan2.2 (stride 1, this decision) |
|---|---|---|
| Base native frames | 16 | 121 |
| Episode coverage (300f) | 16×4 = 64f (~21%) | 121×1 = 121f (~40%) |
| Inter-frame motion | 4× jumps (OOD) | contiguous (in-distribution) |
| Action handling | SUM dropped deltas | raw per-frame action |
| Prediction horizon (cond_frames=1) | — | 30 latent frames |

So coverage is obtained through **window length**, not subsampling — strictly
more coverage than the old trick, contiguous, with clean per-frame actions.

## Memory note

VRAM is **not** the binding constraint here: the MetaWorld latent is tiny
spatially (16×16 → 8×8 = 64 tokens/frame after patch_size 2), so 31 latent
frames is only ~1984 attention tokens. The heavier cost is the frozen VAE
encoding 121 full-res frames per sample (chunkable) and longer-sequence DiT
forward throughput.

## Caveats / watch in logging

- **Action magnitudes now ~4× smaller** than under stride-4 SUM (raw deltas vs
  summed-over-4). Action-input normalisation may want revisiting — **flagged as a
  watch-item**, not changed pre-emptively (mirrors the prior decision's k×
  magnitude-bump caveat).
- **Episode-length headroom:** episodes are 300 frames, so 121 contiguous fits
  with margin (action window bound `length*stride = 121 ≤ 300`). Re-check if a
  future dataset has shorter episodes — the `min(cond_frames, t_lat-1)` guard and
  the span filter both bind on clip length.
- **`fs_value`** stays `1` (Wan2.2 diffusion forcing conditions on the clean obs
  latent, not an fps channel) — unchanged.

## Future-revisit triggers

- Throughput from VAE-encoding 121 frames/sample becomes the training
  bottleneck → cache latents or drop to a shorter window (e.g. 65f / 17 latent).
- Action-norm instability traced to the 4× magnitude drop → revisit normalisation.
- Multi-frame conditioning (`cond_frames > 1`) adopted → re-confirm the horizon
  budget (`min(cond_frames, t_lat-1)` caps clean frames at 30). **Done
  2026-06-30:** variable-history training via a categorical `cond_frames_dist`
  — see [[../../60_Updates/entries/2026-06-30-wan22-variable-cond-frames]].

## Related

- [[metaworld-frame-stride-load-time]] — the amended decision (DynamiCrafter
  stride-4 + action-SUM; still in force for non-Wan2.2 configs).
- [[../../30_Knowledge/tech/frame-stride-conditioning]] — the frame-stride audit.
- Code: `configs/diffusion_wan22_i2v_metaworld.yaml`,
  `configs/diffusion_wan22_avid_i2v_metaworld.yaml` (`data.frame_stride: 1`,
  `model.extra.temporal_length: 121`).
