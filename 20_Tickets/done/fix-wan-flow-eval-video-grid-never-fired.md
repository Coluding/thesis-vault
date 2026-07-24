---
type: fix
scope: training
status: done
priority: high
created: 2026-06-23
updated: 2026-06-23
resolution: completed
resolution_note: >
  The FlowInferenceSampler ([[feat-flow-inference-sampler-eval]]) existed and was
  auto-selected, but the WAN shortcut run still logged NO videos. Four gaps,
  all fixed: (1) two `model_type != "diffusion"` guards in trainer
  (`_maybe_generate_samples`, `generate_samples`) bailed for flow; relaxed to
  accept flow/flow_matching. (2) The wandb logger was never built for WAN
  (config had no `training.extra.wandb` block) and even when enabled it looked
  for the DynamiCrafter VAE interface (`decode_first_stage` + `first_stage_model`)
  which Wan21DiTWrapper does not expose. (3) GT-panel correctness bug: the flow
  batch's `target = noise - z0` is the VELOCITY, but the logger decodes
  `target_latents` as ground truth -> would render noise. (4) The training
  script never passed `wandb_logger` to the Trainer. Reuses the WanVAE already
  loaded for the pixel->latent encode (no extra VAE load).
closed_at: 2026-06-23
related:
  - "[[feat-flow-inference-sampler-eval]]"
  - "[[feat-wan21-backbone-integration]]"
  - "[[feat-tiny-wan-action-adapter]]"
  - "[[fix-wan-flow-timestep-scale]]"
  - "[[bug-eval-hyperalign-logged-videos-broken]]"
---

# WAN shortcut training logged no eval videos despite the flow sampler landing

[[feat-flow-inference-sampler-eval]] wired `FlowInferenceSampler` into the
trainer and auto-selects it for `model_type=="flow"`. But end to end, a WAN
shortcut run still produced **zero eval rollouts/videos**: the sampler was built
but never invoked, and even if invoked nothing could decode the latents. Found
while sanity-checking "are we doing inference eval by logging the video grid?"
— the answer was no.

## The four gaps

1. **Diffusion-only guards.** `trainer._maybe_generate_samples` opened with
   `if model_type != "diffusion": return None`, and `generate_samples` raised
   for non-diffusion. So `inference_every_n_steps` was dead config for flow.
   -> relaxed both to `in ("diffusion", "flow", "flow_matching")`.

2. **Logger never built / wrong VAE interface.** `_maybe_build_wandb_logger`
   only fires on `training.extra.wandb.enable: true` (absent from the WAN
   config), and it discovers the VAE via the LVDM convention
   (`base_model.decode_first_stage` + `first_stage_model`). `Wan21DiTWrapper`
   exposes neither — the Wan-VAE is a separate `WanVAE` object. -> added a
   `wandb` block with `require_vae: false`, a `WandbLogger.set_decode_fn(...)`
   setter, and a `make_wan_decode_fn(vae)` adapter
   (`models/base/wan.py`) that turns `WanVAE.decode(list)->list` into the
   logger's `[B,C,T,H,W] -> [B,3,T,H,W]` contract. The training script injects
   it using the **same VAE it already loaded** for the encode — no extra load.

3. **GT panel decoded the velocity (correctness bug).** The flow batch's
   `target = noise - z0` is the velocity, not the clean latent; the logger
   decodes `target_latents` as the ground-truth row, so the GT panel would have
   been pure noise. -> `WanBatchPreprocessor` now also returns the clean latent
   under `x0`; the grid + single-video paths decode `batch["x0"]` for GT and
   fall back to `target` for diffusion. Regression test:
   `FlowGridGroundTruthTest` in `tests/test_video_logging.py`.

4. **Logger not handed to the Trainer.** `scripts/train_wan_shortcut_metaworld.py`
   built the logger (via `build_experiment`) but constructed `Trainer(...)`
   without `wandb_logger=`, so even a built logger was orphaned. -> now passes
   it and injects the decode fn.

## What you get now

At every `inference_every_n_steps` the WAN shortcut run logs a **multi-step
shortcut grid** to wandb: rows = sampling-step counts (1/2/4/8), columns =
`[GT | base | adapted]`, all rollouts sharing one noise draw, with
`step_level = 1/num_steps` injected only into the adapted cond (matches the
dyadic `d = 2^-k` training levels). Driven by a new `eval_step_schedule` in the
config; `_eval_step_schedule` already supported the format.

## Files touched

- `src/generative_flow_adapters/training/trainer.py` — guards + decode `x0` for GT.
- `src/generative_flow_adapters/training/wandb_logger.py` — `set_decode_fn`.
- `src/generative_flow_adapters/models/base/wan.py` — `make_wan_decode_fn`.
- `src/generative_flow_adapters/data/wan_batch_preprocessor.py` — emit `x0`.
- `scripts/train_wan_shortcut_metaworld.py` — pass logger + inject decoder.
- `configs/diffusion_wan_shortcut_metaworld.yaml` — `wandb` + `eval_step_schedule`.
- `tests/test_video_logging.py` — `FlowGridGroundTruthTest`.

## Follow-on: flow shortcut now uses the shared step schedule (was dyadic-only)

While reviewing the eval grid we asked "why is WAN dyadic when DynamiCrafter
isn't?" — and the answer was: **implementation gap, not a constraint.**
`_maybe_prepare_shortcut` had two sub-paths in the diffusion branch
(continuous `shortcut_step_schedule` vs legacy dyadic), but the **flow** branch
only ever implemented the dyadic `shortcut_max_log2_steps` ladder (≤8-step,
down to 1/8). It never checked `self.step_schedule`.

Flow is actually the *cleaner* case for the continuous schedule: the diffusion
self-consistency target needs `alphas_cumprod` + a `to_timestep_jump`
conversion (DDIM curvature), whereas the rectified-flow Euler micro-step is a
straight line (κ=0 — the very reason for the flow-base pivot,
[[../60_Updates/entries/2026-06-19-pivot-flow-matching-base]]), so `d = s/2` is
fed to `compute_self_consistency_target_v_flow` directly. So there was no reason
to cap the depth.

- EDIT `trainer.py` `_maybe_prepare_shortcut`: flow branch now takes a
  `self.step_schedule is not None` sub-path mirroring the diffusion one
  (anchor at `smallest()`, sample `s_full` excl. smallest, supervise against two
  chained calls at `s_half = s_full/2`), falling back to the dyadic ladder when
  no schedule is set (non-breaking).
- CONFIG `diffusion_wan_shortcut_metaworld.yaml`: replaced
  `shortcut_max_log2_steps: 3` with the AVID-identical
  `shortcut_step_schedule` (normalised log2, min 1/128, max 1). The adapter is
  now supervised across the full multi-step range, so the eval grid's finer
  rows (incl. the 25-step / 0.04 reference) are **interpolation, not
  extrapolation**.
- TEST `test_flow_shortcut_uses_step_schedule_like_dynamicrafter` — asserts the
  flow path records step_levels drawn from the schedule rungs and reaches
  below 1/8 (impossible under the old `max_log2: 3`).

## Interpreting the logged grid (washed-out panels ≠ bug)

First real grid on wandb: the **GT column decodes correctly** (the `x0` fix
works — a clear MetaWorld arm), but the **base and adapted columns are
washed-out gray**. This is *expected*, not necessarily a pipeline bug:

- The eval grid samples from **pure noise** through the **frozen Wan T2V base
  with null text context** and `guidance_scale=1.0`. An unconditional,
  empty-prompt Wan rollout is not a MetaWorld scene — low-contrast mush is the
  honest output.
- The adapted column is action-conditioned-from-noise (no frame conditioning in
  this config); early in training the adapter cannot yet synthesise the scene.

So a bad-looking grid conflates "machinery broken" with "unconditional base +
undertrained adapter". The deterministic separator is a **frame-conditioned
SDEdit reconstruction vs GT** — the WAN analogue of DynamiCrafter's
`test_base_model_image_to_video_rollout_is_semantically_consistent` (which we
*had* for DynamiCrafter but not for WAN):

- NEW `tests/test_wan_generation_gpu.py::test_wan_base_sdedit_reconstructs_metaworld_clip`
  — encodes a real clip to `z0`, rolls the frozen base from the same noise via
  SDEdit (anchored on `z0`) vs unconditional (pure noise), and asserts
  `MSE(sdedit, GT) < 0.6·MSE(uncond, GT)` and `< 0.12`. Stores
  `[GT | sdedit | uncond]` to `outputs/`. GPU-gated (needs the 1.3B ckpt + VAE +
  HDF5). Run: `WAN_CKPT_DIR=ckpts/Wan2.1-T2V-1.3B pytest -s -q \
  tests/test_wan_generation_gpu.py::test_wan_base_sdedit_reconstructs_metaworld_clip`.
  If it passes → generation machinery is sound; if it fails → the sampler /
  timestep convention / VAE path is the culprit (start at the `[0,1]`-vs-`[0,1000]`
  note in [[feat-flow-inference-sampler-eval]]).

## Caveats / follow-ups

- Logging requires `pip install wandb` (the logger imports it lazily; with
  `enable: true` an offline/missing wandb will raise at construction).
- The AVID-variant config (`diffusion_wan_avid_shortcut_metaworld.yaml`) and
  `scripts/train_avid_shortcut_metaworld.py` were **not** updated — if that run
  also needs the WAN grid, mirror the `wandb`/`eval_step_schedule` block and the
  `set_decode_fn` injection there.
- The deeper `[0,1]`-vs-`[0,1000]` base-timestep distribution issue noted in
  [[feat-flow-inference-sampler-eval]] is still open and orthogonal to this fix.
