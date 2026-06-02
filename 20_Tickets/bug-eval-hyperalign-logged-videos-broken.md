---
type: bug
scope: eval
status: open
priority: medium
created: 2026-05-28
updated: 2026-05-28
resolution:
resolution_note:
closed_at:
related:
  - "[[bug-training-hyperalign-oom-flash-attention]]"
  - "[[../30_Knowledge/related-work/hyperalign]]"
---

# HyperAlign logged videos are broken (GT wrong colors, samples look like noise)

## Symptom

During HyperAlign training, the videos logged to wandb (or local logger)
are unusable:

- **Ground-truth panel:** RGB colors look wrong / "off" — the GT clip itself
  is misrendered, not just the prediction.
- **Generated panel:** completely off — looks like random noise rather than
  a degraded-but-recognizable rollout.

The **same logging code path works correctly with the output adapter** —
GT and samples both render as expected there. So the regression is
HyperAlign-specific, not a general logger bug.

_Exact run id / config / commit — needs verification._

## Why this matters

Without trustworthy video logging we cannot eyeball HyperAlign training
quality, which means we cannot tell whether the OOM-workaround configs
([[bug-training-hyperalign-oom-flash-attention]]) are still producing
sensible rollouts. Blocks any qualitative D1/D2 evidence from HyperAlign.

## Investigation candidates (analysed estimates — code-read 2026-05-28)

**Code map (verified — file:line cites).** Both adapter families go
through the **same** logger and decoder:

- `training/trainer.py:601-619` — `_maybe_generate_samples` reads
  `target = batch["target"]` and hands it to `wandb_logger.log_videos`.
  No adapter-family branching.
- `training/wandb_logger.py:82-89` — both `prediction_latents` and
  `target_latents` go through the same `_decode_to_uint8`.
- `training/wandb_logger.py:172-177` — `_decode_to_uint8` calls
  `self._decode_fn(latents)` then `.clamp(-1, 1).add(1).mul(127.5)`.
  `decode_fn` is `base_model.decode_first_stage` (`training/builders.py:61`).
- `models/base/dynamicrafter.py:98-106` — `decode_first_stage` is
  `@torch.no_grad()` and calls `first_stage_model.decode_video(latents)`.
- `data/latent_encoder.py:85,92` — VAE encode multiplies by
  `SD_VAE_SCALE_FACTOR=0.18215`; decode divides by it. Symmetric, identical
  for both adapter families.
- `data/batch_preprocessor.py:91-95,153` — `batch["target"]` is set once
  at preprocess time from `vae.encode_video(video)`. Not adapter-aware.

So `batch["target"]` is the **same tensor** for both adapter families,
and the decode path is identical. If GT is corrupted, **something is
mutating either `batch["target"]` or the VAE between the two runs**.

### Concrete differences HyperAlign introduces (none triggers GT decode directly, but flagged for verification)

1. **Structural LoRA injection into the UNet.** HyperAlign physically
   wraps target linear layers via `inject_hyperalign_lora_layers`
   (`adapters/hypernetworks/hyperalign.py:123-130`). The output adapter
   does NOT modify UNet layers — it adds a head on top
   (`adapters/output/dynamicrafter.py`). _This does not touch the VAE,
   so should not affect GT decode — but should be verified._

2. **LoRA factors retained across forward.** `HyperAlignAdapter.forward`
   (hyperalign.py:271, comment 264-269) deliberately leaves the per-step
   LoRA factors set on the wrapped layers after returning, so gradient
   checkpointing replays the same graph. They are only cleared at the
   **start** of the *next* `AdaptedModel.forward`
   (`models/adapted_model.py:65-66`).

3. **Persistent forward hooks on `module.input_blocks`.** The
   `_HyperAlignInputFeatureStore` registers hooks on every input block
   (hyperalign.py:702-703) once and never removes them. They append
   `output.detach()` to `self.input_activations` on **every** forward
   through the UNet, including the base sampler's. The hooks read but do
   not mutate, so this is a memory leak rather than a corruption — but
   `input_activations` grows without bound.

4. **No `diffusion_schedule_config` override.** The output adapter exposes
   its own schedule (`adapters/output/dynamicrafter.py:164-165`);
   HyperAlign falls back to the base model's
   (`adapters/hypernetworks/hyperalign.py:411`). If these schedules
   diverge on `use_dynamic_rescale`/`base_scale`/`turning_step`, the
   sampler's reverse-rescale path will behave differently — but again,
   this affects sampler output, not GT decode.

### **STRONGEST FINDING — explains the "base/sample noise" panels**

Base sampler contamination via stale LoRA factors:

- `_maybe_generate_samples` runs **two** samplers back-to-back
  (`training/trainer.py:605-612`): the `inference_sampler` (uses
  `AdaptedModel`) then `base_inference_sampler` (uses the raw `base_model`,
  built at trainer.py:62-68).
- Both share the **same** underlying UNet module — `base_inference_sampler`
  is wrapped around `base_model = getattr(model, "base_model", None)`
  (trainer.py:61).
- After the adapted sampler finishes its 50 denoising steps, the last
  `HyperAlign.forward` left LoRA factors **set** on the UNet's wrapped
  linear layers (no end-of-forward clear).
- The base sampler then runs through that **contaminated UNet**: its
  `model(sample, t, cond)` calls bypass `HyperAlignAdapter.forward`, so
  no `clear_dynamic_parameters()` ever runs, and every step applies the
  stale per-batch LoRA factors from the very last training/eval step.
- Result: the "base_model" panel is not the base at all — it is the
  adapted UNet with frozen-in factors from one batch ago. With early
  training (or with bad factors) this will look like noise.

This **does not happen for the output adapter** because it does not modify
UNet layers; the base sampler runs a truly clean base UNet.

### The GT-color question remains open

The code-read above does not find a mechanism that corrupts
`batch["target"]` or the VAE on the HyperAlign-only path. Possibilities
worth checking, in order:

1. **Misinterpretation of the panel layout.** The logger labels panels
   `left=ground_truth | middle=base_model | right=adapted`
   (wandb_logger.py:182). If the user is reading the *middle* panel as
   "ground truth," the wrong colors there are fully explained by the
   stale-LoRA contamination above. _Confirm which panel is "wrong-colored"
   on the actual wandb video._

2. **Config-level VAE divergence.** The HyperAlign config file may load a
   different VAE checkpoint or different `first_stage_config` from the
   output-adapter config. _Diff the two configs around
   `model.extra.load_first_stage_model` and `first_stage_config`._

3. **Subtle dtype/autocast issue in the VAE decode.** `decode_first_stage`
   is `@torch.no_grad()` but not wrapped in `autocast(enabled=False)`.
   If the trainer enters autocast somewhere upstream that's still active
   at log time for the HyperAlign config (e.g. bf16 amp), the VAE could
   silently downcast and produce a slight color shift. _Not seen in the
   trainer code path (autocast scope is `with self._autocast(): forward`,
   which exits before `_maybe_generate_samples`) but worth a sanity
   check on the actual run config._

4. **Real GT corruption.** If panels 1+2 are ruled out, instrument
   `target.min/max/mean` immediately before
   `wandb_logger.log_videos` and compare against a fresh
   `vae.encode_video(video)` round-trip baseline.

### What was ruled out

- **In-place mutation of `batch["target"]` by HyperAlign.** No `mul_`,
  `add_`, `div_`, `copy_` or `.data =` in hyperalign.py touches anything
  in the batch. `scale_x_start` (`losses/diffusion.py:68-79`) returns a
  new tensor.
- **VAE state mutation by HyperAlign.** Injection and hooks target the
  UNet (`module.input_blocks`), not `first_stage_model`. Decode is
  `@torch.no_grad()` and the SD VAE has no train/eval-sensitive layers
  (GroupNorm only) — so even if model was in train mode, decoded pixels
  shouldn't shift.
- **Different decode path per adapter.** Both go through the same
  `_decode_to_uint8` → `decode_first_stage` → `first_stage_model.decode_video`.

## Reproduction

- Train HyperAlign on the standard MetaWorld config used in
  [[bug-training-hyperalign-oom-flash-attention]] — _exact config path
  needs verification_.
- Inspect logged videos at first eval step.
- Compare side-by-side with an output-adapter run on the same data /
  logger settings — that path renders correctly.

## Definition of done

- Root cause identified (which of the candidates above, or another).
- HyperAlign-logged GT renders with correct colors.
- HyperAlign-logged samples render as actual decoded rollouts (good or
  bad — but not random noise from a space mismatch).
- A short note added to [[../30_Knowledge/related-work/hyperalign]] or a
  new `30_Knowledge/tech/` note if the fix exposes a general logging
  contract that other adapter families need to honour.

## Related

- [[bug-training-hyperalign-oom-flash-attention]] — sibling HyperAlign bug
- [[../30_Knowledge/related-work/hyperalign]] — adapter mechanics,
  double-forward, hidden-state capture
- code: `src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py`
- code: `src/generative_flow_adapters/training/trainer.py` (video logging hook)
