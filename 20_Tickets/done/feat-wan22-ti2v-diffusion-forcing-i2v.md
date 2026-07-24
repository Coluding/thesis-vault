---
type: feat
scope: backbone
status: done
priority: high
created: 2026-06-24
updated: 2026-07-15
resolution: shipped
resolution_note: >
  The only remaining blocker ("GPU validation pending", real Wan2.2-TI2V-5B
  DiT + Wan2.2_VAE.pth) is done — this exact diffusion-forcing architecture
  (per-frame timestep, obs-frame clamp at t=0, masked velocity loss) has been
  the base for every WAN training run in the 2026-07-14/15 debugging session
  (AVID baseline, cross-attention arms, DC-UNet capacity runs, the AVID
  reference comparison), all real GPU runs against the real checkpoint. The
  zero-init-head "false bug" and latent-geometry gotchas noted in this ticket
  held up; no further issues surfaced.
closed_at: 2026-07-15
related:
  - "[[../30_Knowledge/theory/diffusion-forcing]]"
  - "[[fix-wan-flow-eval-video-grid-never-fired]]"
  - "[[feat-flow-inference-sampler-eval]]"
  - "[[feat-wan21-backbone-integration]]"
  - "[[feat-tiny-wan-action-adapter]]"
---

# Wan2.2-TI2V-5B image-to-video base via diffusion forcing

## Why we switched to I2V (and to the 5B)

The Wan2.1-T2V world model had no way to condition on the current observation
frame: training was action-only with `x_t` = noised full clip, so the eval grid
generated the whole MetaWorld scene from pure noise + a 4-dim action — an
under-determined task, which is why the first wandb panels were washed-out mush
(the GT column was correct; base/adapted were noise). T2V-1.3B cannot accept a
concat-image channel (in_dim=16, fixed) so DynamiCrafter-style conditioning is
impossible on the frozen base.

Options weighed: **Wan2.1-I2V-14B** (vendored, but ~28 GB → needs 80 GB GPU,
CLIP + 36-ch concat) vs **Wan2.2-TI2V-5B** (consumer-GPU, Apache-2.0, unified
text+image). Chose **5B**.

## Key finding: TI2V-5B conditions by DIFFUSION FORCING (no CLIP, no concat)

Reading `external_repos/wan22/wan/textimage2video.py` + `modules/model.py`:
- `in_dim == VAE z_dim == 48`; **no `clip_fea`, no `y` concat** for `model_type='ti2v'`.
- Image conditioning is done **outside** the model: the observation frame's
  clean latent is clamped into the sequence (`latent = (1-mask)*z + mask*latent`,
  re-applied every step) and those tokens get **timestep 0**, while the rest
  denoise. The DiT forward takes a **per-token timestep `[B, seq_len]`**.
- This is exactly the **UWM / diffusion-forcing** scheme flagged in
  [[../../memory/multimodal-adapter-broadening]] — the I2V decision and that plan
  converged. Wan2.2-VAE: z_dim=48, stride (4,16,16). Flow matching / velocity.

So the integration is *adapter-first preserved*: the frozen 5B base does the
observation conditioning natively; the trainable adapter still does the action.
But it changes the **training objective** from uniform flow matching to
diffusion forcing (obs frames clean at t=0, future frames noised, loss masked to
the future).

## Implementation (7 stages, all verified on CPU)

1. **Vendor** `backbones/wan/modules/model2_2.py` (per-token-timestep ti2v DiT)
   + `vae2_2.py` (Wan2.2-VAE) + `utils/diffusion_forcing.py` (`masks_like`).
   Reused the existing SDPA-adapted `attention.py` and `fm_solvers_unipc.py`
   (one-line `attention as flash_attention` swap for CPU).
2. VAE decode_fn — the generic `make_wan_decode_fn` already fits (same list API).
3. `models/base/wan2_2.py::Wan22DiTWrapper` — per-token timestep (accepts
   per-frame `[B,T']` and expands across patch tokens), null text, factory
   `provider: wan2.2`.
4. `data/wan22_batch_preprocessor.py::Wan22DiffusionForcingPreprocessor` —
   obs-frame clamp + per-frame timestep + `frame_mask` + `x0`.
5. Trainer `_flow_loss` (masked to predicted frames) + adapter per-sample
   timestep reduction (`t.dim()>1 → amax`); `FlowInferenceSampler` diffusion-
   forcing path (clamp obs each step + per-frame t).
6. Configs `base/wan2.2_ti2v_5B.yaml` (+ `_tiny`) + `diffusion_wan22_i2v_metaworld.yaml`
   (48ch, /16) + `scripts/train_wan22_i2v_metaworld.py`.
7. `tests/test_wan22_i2v.py` — CPU smoke (preprocessor mask, end-to-end masked
   training step, sampler obs-clamp) + GPU-gated I2V reconstruction.

Verified end to end on CPU: tiny base + adapter trains on a diffusion-forcing
batch with a masked velocity loss, frozen base gets 0 grad, sampler holds the
observation frame and generates the future. 33 tests green, no regressions.

## Gotchas / caveats

- **Zero-init head**: a freshly-built Wan DiT emits ~zero velocity (identity
  init) and is timestep-invariant at init — chased this nearly to a false "bug".
  CPU tests assert wiring (masked loss, frozen base, obs clamp), not learned
  response; perturb the head to test timestep response.
- **Latent geometry changed**: 48 ch, /16 spatial. 256 px → 16×16 latent (vs
  32×32 at /8). Adapter `feature_dim=48`, train-script VAE stride = 16.
- **Shortcut is OFF in v1** (`shortcut_direction_weight: 0`): shortcut self-
  consistency *on top of* diffusion forcing is a separate research question.
- **Conditioning-frame noise augmentation** (`masks_like` p=0.2) not yet wired —
  clean obs only. Multi-frame history (cond_frames>1) is supported but untested.
- **GPU validation pending**: needs the real Wan2.2-TI2V-5B DiT + Wan2.2_VAE.pth.
  Run `WAN22_CKPT_DIR=... pytest -s tests/test_wan22_i2v.py::test_wan22_i2v_reconstructs_future_from_observation`
  and report the i2v-vs-uncond MSE so the threshold can be tuned.
