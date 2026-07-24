---
date: 2026-07-02
category: added
deliverable: D2
meeting:
sources:
  - "[[../../50_Decisions/decided/wan22-i2v-diffusion-forcing]]"
  - "[[2026-07-02-basevideomodel-external-repo-design]]"
---

# Pre-training generation check: real AVID adapter through the BaseVideoModel seam

Rewrote `examples/wan22_generate_cond_frames.py` onto the new plug-and-play stack
(`WanTI2VVideoModel` external base + `AdaptedModel`), image-conditioned, to
sanity-check generation **before** training.

## What

- Builds `AdaptedModel(WanTI2VVideoModel, AVID adapter)` directly from the AVID
  config (provider forced to `wan2.2_external`, ckpt DIR, offload from CLI). No
  optimizer/wandb/checkpoint — inference only.
- Conditions on a single frame (`--image`, default `metaworld_frame0.png`; falls
  back to a MetaWorld hdf5 frame when empty).
- Generates **base-only** (`base_model.generate`, delegates to upstream
  `wan.WanTI2V.generate`) and **adapted** (`AdaptedModel.generate`, real untrained
  AVID adapter injected at the denoise seam), saves both mp4s + the cond frame,
  and prints max |adapted − base|.
- Constant action cond fed to the adapter (`--action`, default zeros; no
  `step_level` → adapter tolerates None). `condition_drop_prob=0` for determinism.

Defaults mirror the upstream `generate.py` invocation: `--size 1280*704`
(max_area=901120), `--frame-num 41`, `--guide-scale 1.0`, `--offload-model`,
convert_model_dtype implied by `model.extra.dtype=bf16`, shift 5.0, unipc.

## Why

The existing `wan22_base_vs_adapted_generation.py` only exercised the seam with a
zero-delta *ProbeAdapter* (ignores `t`). This is the first run of the **real**
AVID adapter through Wan's native loop, so it tests the adapter's timestep + cond
handling inside generation — the plumbing we rely on before spending a train run.

## Open question / risk (untested — needs GPU + external repo)

Inside the native loop the seam hands the adapter whatever timestep form Wan
feeds its DiT. The AVID adapter collapses `t.dim() > 1 -> amax`, but if Wan passes
a 1-D `[seq_len]` timestep the adapter forwards it as-is to a module expecting
`[B]`. If this mismatches, this script is exactly what surfaces it. Filename is
now a misnomer (no longer a cond-frames sweep) — consider renaming to
`wan22_generate_check.py`.
