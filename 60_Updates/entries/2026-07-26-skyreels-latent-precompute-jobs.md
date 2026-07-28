---
date: 2026-07-26
category: feature
deliverable: D1
meeting:
sources: []
---

# SkyReels latent precompute (prewarm) script + cluster jobs

## What

SkyReels can't reuse the Wan2.2 latent cache — different VAE (SkyReels = Wan2.1
VAE, **16-ch, stride 8**; Wan2.2 world-model = **48-ch, stride 16**). SkyReels
latents were only ever cached on-the-fly during training. Added a standalone
prewarm path so the `<data>.skyreels.latents` cache can be filled before a run.

Key point: `WanBatchPreprocessor.precompute()` calls `_encode_z0`, which
`SkyReelsI2VPreprocessor` overrides (16-ch batched-tensor VAE) while inheriting
`_latent_keys` — so calling `.precompute()` on a SkyReels preprocessor caches z0
with keys **identical to training by construction**. (The existing
`scripts/precompute_latents.py` can't do SkyReels — it hard-wires the Wan2.2
diffusion-forcing preprocessor, wrong VAE convention + wrong keys.)

Added (all additive):
- `scripts/precompute_skyreels_latents.py` — mirrors `train_skyreels_acwm.py`
  setup exactly (geometry, `build_base_model`, `SkyReelsI2VPreprocessor`,
  cache-dir logic), enumerates `dataset.fixed_window_enumeration()`, calls
  `preprocessor.precompute()`. Both `--dataset metaworld|acwm_phys`.
- `configs/skyreels/diffusion_skyreels_xattn_metaworld.yaml` — MetaWorld variant
  (none existed): MetaWorld `data:` block, square stride-8 geometry (tl 17 → 5
  latent frames, 384×384, max_area 147456).
- `jobs/experiments_cluster/acwm_phys/skyreels/submit_precompute_skyreels_{robotarm,pushblock}.sh`
  (loop ACWM splits into `$ROOT/skyreels.latents.shared`, num-windows 8) and
  `jobs/experiments_cluster/metaworld/skyreels/submit_precompute_skyreels_metaworld.sh`.

## Verified

MetaWorld GPU gate: precompute encoded windows → wrote `.pt` to
`ds/metaworld_corner2.skyreels.latents/`; re-running `precompute()` at training
geometry reported `newly_encoded=0, cache_hits` → **keys match training**.

## Gotchas / caveats

- **frame_stride**: `train_skyreels_acwm.py` defaults `--frame-stride 1` and
  passes it, IGNORING `config.data.frame_stride` (which is 4 in the YAML). Both
  the training and precompute jobs omit `--frame-stride` → both use fs=1 → keys
  match. `frame_stride` is IN the cache key, so if anyone passes `--frame-stride`
  to training they must pass the same to precompute or it re-encodes.
- Precompute + training MUST share `--config` and `--num-windows` (both baked
  into keys). Job headers note this.

## ⚠️ Pre-existing blocker: SkyReels TRAINING is broken (separate from precompute)

`train_skyreels_acwm.py` smoke crashes in
`SkyReelsI2VPreprocessor._build_i2v_conditioning` → `clip.encode_video`:
`Input type (cuda.Float) and weight type (CPUBFloat16) should be the same` — the
offloaded CLIP (bf16, on CPU) is fed a GPU tensor. This is DOWNSTREAM of the
z0/cache path (z0 encode + cache hit run first and work), so the precompute
deliverable is unaffected, but the prewarmed cache isn't usable end-to-end until
this CLIP-offload dtype/device bug is fixed. The SkyReels configs are already
marked "DRAFT — validate the i2v seam" with GPU-VALIDATE notes. Needs a separate
fix (move/cast the CLIP conditioning frame to the CLIP's device+dtype, or keep
CLIP resident).
