---
date: 2026-07-01
category: added
deliverable: D2
meeting:
sources:
  - "[[2026-06-30-wan22-variable-cond-frames]]"
  - "[[2026-06-30-wan22-stride1-121frames]]"
---

# Standalone generation script: Wan2.2 diffusion-forcing cond-frame sweep

## What

Added `examples/wan22_generate_cond_frames.py` — a standalone script to
qualitatively inspect how the **frozen Wan2.2-TI2V-5B** base generates MetaWorld
video as a function of the number of clean observation (history) frames `k` it
conditions on (diffusion forcing). Sweeps `k ∈ {1,2,3,4,5,6}` by default.

## How

Reuses the training-path building blocks so the rollout matches eval exactly:
- `build_base_model` (frozen Wan2.2, provider `wan2.2`) — no adapter, exercises
  the base's *native* observation conditioning.
- `Wan22DiffusionForcingPreprocessor` to VAE-encode one real MetaWorld clip to
  the clean 48-ch latent `x0` and build the action `cond`.
- `FlowInferenceSampler` (rectified-flow UniPC, bf16 autocast). For each `k`,
  rebuild `frame_mask = (arange(t_lat) >= k)` and pass `{target, x0, frame_mask,
  cond}` — the sampler clamps the first `k` latent frames clean (timestep 0)
  every step and denoises the rest.
- **Shared noise draw** across all `k` so panels differ *only* in the amount of
  observed history.
- Decodes each rollout + the ground-truth clip to mp4 via `Wan2_2_VAE.decode` +
  `cache_video`.

`k` counts **latent** frames (VAE temporal stride 4): k=1 ≈ first pixel frame,
each extra latent frame ≈ 4 more pixel frames.

## Why

Wanted a fast visual check of the base model's image-conditioning quality before
reading into trained-adapter rollouts — the base does the observation
conditioning, the adapter only adds the action delta, so this isolates the base.

## Run

```
python examples/wan22_generate_cond_frames.py \
  --hdf5 ds/metaworld_corner2.hdf5 --ckpt-dir ckpts/Wan2.2-TI2V-5B \
  --cond-frames 1 2 3 4 5 6 --steps 50 --out-dir outputs/wan22_cond_sweep
```

Smoke-tested end-to-end (33px window / 8 steps, ~6 it/s). Full 121-frame / 256px
/ 50-step sweep runs at native config geometry.

## Open

- Base-only for now; a `--through-adapter` / `--adapter-ckpt` path is a natural
  extension once a trained Wan2.2 adapter checkpoint exists (none yet).
- `k` is latent-frame count, not pixel — consider exposing a pixel-frame flag if
  that framing is clearer for the thesis figures.
