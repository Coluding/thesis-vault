---
date: 2026-07-25
category: finding
deliverable: D1
meeting:
sources: ["[[entries/2026-07-21-replace-eval-bug-fixed-adapter-action-blind]]"]
---

# DynamiCrafter compare script: base i2v was broken by two bugs (missing frame-0 anchoring + wrong fps), NOT CFG

## TL;DR (resolution)

Built `scripts/generate_dynamicrafter_compare.py` (DynamiCrafter analogue of the
Wan compare). The base row was a brown blob. Traced it against the AVID base repo
(`external_repos/avid/latent_diffusion`, lvdm DDIM) — the fix was **not** CFG
(AVID runs `unconditional_guidance_scale: 1.0`, i.e. no CFG, on MetaWorld). Two
real bugs in our generic DDIM path:

1. **Missing first-frame anchoring.** DynamiCrafter i2v pins frame 0 to the clean
   observation latent at every DDIM step (`img = x0*mask + (1-mask)*img`,
   `clean_cond=True`; lvdm `DDIMSampler.ddim_sampling`, `mask=cond_mask, x0=z`).
   Our `DiffusionInferenceSampler` didn't. Added opt-in `anchor_mask` /
   `anchor_latent` args (default None = old behaviour). Frame 0: blob → sharp.
2. **Wrong fps conditioning.** Our data pipeline feeds `data.fs_value: 1`; AVID's
   MetaWorld config uses `default_fs: 10` (`fps_condition_type: "fps"`). Feeding
   fs=1 degrades the temporal dynamics. Added a `--fs` override; **fs=10** makes
   the near-term frames recognizable (table + arm + shadow).

With both fixes the base row is coherent for frame 0 + the first few frames, then
blurs out over time — expected for the **zero-shot** DynamiCrafter base with a
**random-init** adapter (the adapter supplies MetaWorld dynamics; untrained here).

## Verified: no remaining pipeline bug (instrumented 2026-07-25)

Added `--debug` to the compare script (base-UNet input-channel hook + per-frame
latent drift). On the random-init run at fs=10:
- base UNet input = **8 channels** → concat (first-frame latent) IS applied every frame;
- latents healthy: target mean 0.20/std 0.89, base 0.12/**0.63** (slightly lower
  variance ⇒ blurrier), adapted 0.13/0.76 — no explosion/collapse;
- base per-frame L2 drift from frame 0: `0.0, 46.5, 50.8, …, 74.1` (monotone).

So conditioning + anchoring + decode + schedule all work. The base progressively
**drifts / blurs** because a frozen DynamiCrafter (pretrained on natural video)
zero-shot on OOD MetaWorld predicts wrong dynamics — which is precisely what the
**adapter is trained to correct**. The "super noisy" adapted column is the
`--random-init` perturbation (0.02·randn on adapter-only params) under
`avid_mask_mix`. ⇒ A meaningful base-vs-adapted video test needs a **trained**
adapter checkpoint; the pipeline itself is sound.

## Resolution: native generation via the real lvdm model (2026-07-25)

Rather than keep fixing our reimplemented DDIM, wrapped the REAL vendored lvdm
`LatentVisualDiffusion` as a `BaseVideoModel` and delegated generation to its own
`DDIMSampler` (Wan-style: adapter injected by wrapping `apply_model`, like Wan's
`_ComposedDiT`). New file `models/base/dynamicrafter_video.py:DynamiCrafterVideoModel`
(generation-only; training untouched). Base-only native generation is **coherent** —
arm+table+object stable & sharp across all 16 frames
(`outputs/dcvideo_native_validation.png`), a decisive win over the brown blob. The
plain white background still hallucinates cloudy texture (zero-shot natural-video
model on OOD MetaWorld) — cosmetic; a trained adapter should ground it.

Integration notes (all in the new file): config `target: lvdm.*` rewritten to
`external_deps.lvdm.*`; checkpoint loads 2520 tensors, unexpected=0, missing=1518
= all `model_ema.*` (no EMA in ckpt → `use_ema=False`, never `ema_scope`); load in
**bf16** (fp16 → UNet NaN), schedule buffers kept fp32; **kornia** installed into
the venv via `uv pip install` (note: this repo's package manager is **uv**;
`.venv/bin/pip` doesn't exist). Batch keys `prepare_batch_for_inference` needs:
`video [b,c,t,h,w]∈[-1,1] @320×512`, `caption list[str]`, `act [b,t,a]`, `fps [b]`.

**Root cause of the terminal-NaN class of bugs:** `rescale_betas_zero_snr` makes
`alphas_cumprod[999]=0.0` exactly ⇒ `sqrt_recip{,m1}_alphas_cumprod[999]=inf`, and
`uniform_trailing` starts at t=999 ⇒ `predict_start_from_noise` does `inf·x − inf·e
= NaN` on step 1. Fixed in the new model by clamping those two recip buffers'
terminal `inf` to max-finite (the two coeffs are asymptotically equal there ⇒
`pred_x0→0`). This is almost certainly why the OLD reimplemented path also
struggled with this schedule, and plausibly related to the `eval_denoise_base_only`
NaN.

## Migration done: native provider for compare + training (2026-07-25)

Wired `DynamiCrafterVideoModel` in as a real factory provider `dynamicrafter_video`
and made it serve BOTH generation and the training denoise seam (adapter coupling
is clean — the AVID adapter only consumes the base's final prediction via
`cat([x_t, base_output])`, no internal hooks). Changes: `factory.py` (+provider),
`dynamicrafter_video.py` (`denoise` now translates our preprocessor cond
`{context,concat,act,fs}`→`apply_model` under self-autocast; `_to_lvdm_cond`
helper; generate defaults fixed to eta=0/use_ema=False), `generate_dynamicrafter_compare.py`
(rewritten to native: base=`base_model.generate`, adapted=`model.generate(cond=…)`
via compose_fn), new config `configs/dynamicrafter/diffusion_avid_shortcut_metaworld_native.yaml`.
Training script needed NO changes (provider-agnostic).

Results: compare base row **coherent** (`outputs/dcvideo_compare_native.png`,
verified); 30-step training smoke on native loss 0.263→0.251 (old provider
0.307→0.290) — same scale, finite, no NaN/OOM; fits 24 GB at batch_size=1.

### ⚠️ Key finding — native base ≠ old base (a decision for the thesis)
GATE-1 equivalence FAILED: native (`external_deps/lvdm`, faithful upstream) vs old
(`backbones/dynamicrafter`, a MODIFIED copy with WAN per-frame-t + adapter_embedding
branches) UNet on identical (x_t,t,cond): **max_abs 0.475, cosine 0.974 (~2.6%)**,
action-invariant, intrinsic to the UNet blocks. So the two are DIFFERENT models.
The native one is the faithful base whose generation is coherent; the old one is
what the reimplemented (broken-generation) path used. ⇒ Training on
`dynamicrafter_video` is a **fresh base**; adapters are not weight-compatible
across the two. No existing DynamiCrafter adapter checkpoints exist, so nothing is
lost — but confirm the faithful base is the intended one (vs. the per-frame-t
modified base, if the shortcut/diffusion-forcing method needs it).

### In-training native gen-eval: FIXED (2026-07-25)
Added a batch-mode branch so the Trainer's periodic gen-eval works for DynamiCrafter
(was Wan-shaped: `generate(frame, max_area, frame_num…)` + Wan cond keys). Change is
**additive & isolated** — `trainer.py` +142 lines, **0 deletions**, Wan path
byte-identical:
- `BaseVideoModel.native_eval_batch_mode = False` (capability flag);
  `DynamiCrafterVideoModel` sets it True.
- `_native_eval_grid` early-branches to new `_native_dc_eval_grid` when the flag is
  set; `_native_quality_eval` returns `{}` for batch-mode bases (pixel quality
  metrics not wired — follow-up).
- `_native_dc_eval_grid`: per-clip `base.generate(batch, ddim_steps, fs=10)` +
  adapter-composed `model.generate(batch, cond={act,fs,concat})` (concat re-encoded
  from the native VAE), logged via the same `log_step_size_grid_pixels`. Reuses the
  step schedule + OOM/restore discipline of the Wan grid.
- eval knobs added to the native config so the grid fires.

Verified: real `train_avid_shortcut_metaworld.py` run on the native config with eval
ON ran to completion — step-0 grid fired (`_native_dc_eval_grid`), no crash/OOM,
training continued (loss 0.415→0.256). Base grid **coherent**
(`outputs/dctrain_eval_base.png`); adapted = noise (random-init, expected).

### Still deferred
- Pixel **quality metrics** (psnr/ssim/fvd) during training for DC (`_native_quality_eval`
  stubbed `{}`).
- Memory: native model + training preprocessor duplicate the VAE/CLIP (fits bs=1;
  larger batches tight).
- Adapter `compose_fn` path exercised only with random-init (adapted row = noise);
  a trained checkpoint is needed for a meaningful adapted row.

### Action item
`data.fs_value: 1` in the DynamiCrafter configs looks wrong — it feeds fps=1 to a
base whose i2v was trained around fps≈10. Changing it affects TRAINING too (the
fps embedding the model sees), so decide deliberately. Also align the eval
timestep spacing (`uniform_trailing`) and consider `guidance_rescale: 0.7`.

---

## (original finding, superseded above) base row is a blob — first suspected CFG

## What

Built `scripts/generate_dynamicrafter_compare.py` — the DynamiCrafter analogue of
`scripts/generate_wan22_i2v_compare.py`. It loads a trained adapter checkpoint
(or `--random-init`) on the frozen DynamiCrafter UNet and runs:

1. **loss seam** — `Trainer.evaluate` (adapted vs frozen-base denoise loss + delta);
2. **generation seam** — DDIM rollout, base vs adapted from a shared noise draw,
   decoded to pixels via the wrapper's `decode_first_stage`, saved as a
   `GT | base | adapted` mp4 + frame strip.

Unlike Wan (native `generate` loop), DynamiCrafter has no native generator, so
generation reuses the Trainer's DDIM `inference_sampler` / `base_inference_sampler`
— the same path the periodic wandb rollouts use.

Ran it in `--random-init` on `ds/metaworld_corner2.hdf5`, clip 0, 50 steps (3090):
- **GT** column decodes to a clean MetaWorld scene → dataset + VAE decode are fine.
- **base** column is a featureless brown blob → the frozen DynamiCrafter base is
  NOT generating coherent video.
- **adapted** column is pure noise → expected (untrained, perturbed AVID adapter
  under `avid_mask_mix`).

## Root cause (base blob)

`src/generative_flow_adapters/inference/diffusion.py:92-97` hard-disables CFG:

```python
cond_output = self.model(sample, t, cond)
if False: #TODO check if we need unconditional sampling
    ...cfg...
else:
    model_output = cond_output
```

The sampler's own docstring says SD-derived models like DynamiCrafter "otherwise
produce near-noise samples" without CFG. So the blob is expected given no guidance
— and because this is the SAME sampler the trainer uses, the **wandb base-rollout
panels during DynamiCrafter training are blobs too**. This is a pre-existing repo
limitation, not introduced by the compare script. Wan looked fine because its
native loop does real guidance (guide_scale 5.0).

## Open question (secondary)

`eval_denoise_base_only` came back **NaN** while the adapted denoise loss was
finite (0.427). Since `AdaptedModel.forward(return_base=True)` returns the same
`base_output` used in the composition, a finite composed loss + NaN base-only
loss is contradictory and unexplained — needs a direct check (is `base_output`
inf/NaN in bf16 autocast? does `_flow_loss` treat the two inputs differently?).

## Next step (undecided — asked the user)

To get a meaningful DynamiCrafter base-vs-adapted **video** test we need CFG in
the DDIM sampler (single-CFG, or DynamiCrafter's dual text+image CFG). Options:
enable/implement CFG in the shared sampler (affects training rollouts too) vs.
treat the script as pipeline-validated and move on. Also: no trained DynamiCrafter
adapter checkpoint exists yet (only Wan runs under `outputs/`).
