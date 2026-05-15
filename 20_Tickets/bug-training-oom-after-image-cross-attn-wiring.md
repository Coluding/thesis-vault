---
type: bug
scope: training
status: open
priority: high
created: 2026-05-15
updated: 2026-05-15
resolution:
resolution_note:
closed_at:
related: []
---

# OOM after wiring the image cross-attention branch into the preprocessor

## Symptom

Training crashes on the very first `training_step` after the recent preprocessor change. From the user-reported traceback:

```
File ".../scripts/train_hyperalign_metaworld.py", line 245, in main
    trainer.train(...)
File ".../training/trainer.py", line 230, in train
    metrics = self.training_step(batch)
File ".../training/trainer.py", line 97, in training_step
    prediction = self.model(x_t, t, batch.get("cond"))
...
File ".../backbones/dynamicrafter/modules/attention.py", line 113, in forward
    sim = torch.einsum("b i d, b j d -> b i j", q, k) * self.scale
torch.OutOfMemoryError: CUDA out of memory. Tried to allocate 3.91 GiB.
GPU 0 has a total capacity of 23.56 GiB of which 1.59 GiB is free.
```

Triggering conditions (verbatim from the run prints):

- `batch_size=2`, `temporal_length=16`, latent **320x512** pixels (latent 40x64).
- VAE loaded (`ckts/dynami512.ckpt`), OpenCLIP image embedder + Resampler loaded.
- 1.5B-param UNet, ~14M trainable adapter params, fp32.
- 24 GB card. ~21.8 GB already in use when the offending 3.91 GiB alloc fails.

## What changed

Today's preprocessor edits (single source of truth: `src/generative_flow_adapters/data/batch_preprocessor.py`):

1. **Added the image cross-attention branch.** `_encode_image_branch` runs OpenCLIP image embedder + Resampler on the conditioning frame, concatenates `(B, T*16, 1024)` tokens onto the text context. Context shape went from 77 -> 77 + 16*16 = 333 tokens, so every cross-attention matmul got wider.
2. **Forced `temporal_length: 16`** to match the dynamicrafter_512 checkpoint's Resampler/UNet (the build_dynamicrafter_resampler_from_checkpoint loader now refuses a mismatch). Previously the YAML had T=8.
3. **Added an OpenCLIP image embedder on GPU by default** (ViT-H/14, ~2.5 GB params).

So the same training step now: (a) doubles the temporal axis it noises through, (b) widens cross-attention sequence length, (c) holds ~2.5 GB of CLIP weights resident.

## Root cause (analysed)

Two compounding factors. **All numbers below are derived estimates with shown reasoning — they are not from a profiler run.**

**Factor A — N x N spatial attention matrix gets materialized.**

`CrossAttention.forward` in `src/generative_flow_adapters/backbones/dynamicrafter/modules/attention.py:113` runs `sim = einsum("b i d, b j d -> b i j", q, k)`. At the highest UNet level with latent 40x64 -> N=2560 spatial tokens, B=2, T=16, heads=5 (model_channels=320 / num_head_channels=64):

- sim shape = `(B * T * heads, N, N)` = `(2 * 16 * 5, 2560, 2560)` = `160 * 6.55e6` floats * 4 B = **~4.2 GB**.
- Matches the "Tried to allocate 3.91 GiB" line in the trace.

xformers is **not installed** in this env, so the `MemoryEfficientCrossAttention` fast-path defined in the same file never activates — every transformer block goes through the manual softmax.

**Factor B — the adapter runs the base UNet twice per step.**

`AdaptedModel.forward` (`src/.../models/adapted_model.py:71`) calls `self.base_model(...)` once to obtain `base_output`. Then `HyperAlign.forward` (`src/.../adapters/hypernetworks/hyperalign.py:279`) calls `self.base_model(...)` **again** with the adapter-augmented LoRA weights. Both forwards keep activations live for backward through the second pass. Activation memory is roughly 2x a vanilla diffusion step. Independent of Factor A; would still OOM at smaller batch with just A fixed.

## Mitigations applied today (need re-verification tomorrow)

1. **SDPA in `CrossAttention.forward`** (`src/.../backbones/dynamicrafter/modules/attention.py:111-150`). When `relative_position` is off and there's no explicit mask, we route through `F.scaled_dot_product_attention`. Flash / mem-efficient backend never materializes the (N,N) matrix. Numerical agreement vs manual softmax: 3e-8 (fp32 epsilon, measured). The temporal attention path (which uses `relative_position=True`) still goes through the manual softmax but N=T=16 there, so it doesn't matter.
2. **Image cross-attention path in the same function** also routed through SDPA when `k_ip is not None`.
3. **Config knob to place OpenCLIP image embedder on CPU.** Resolution chain: `--image-encoder-device` CLI flag > `conditioning.extra.image_encoder_device` (YAML) > `auto`. The YAML currently has `image_encoder_device: cuda` — flip to `cpu` if Factor B still leaves the run OOM after the SDPA patch.

The user has **not re-run the training script** after these mitigations landed. The first job tomorrow is to confirm whether they were sufficient.

## Reproduce

```bash
cd /home/lukas/projects/generative-flow-adapters
uv run python scripts/train_hyperalign_metaworld.py
```

Default config: `configs/diffusion_hyperalign_metaworld.yaml`. Settings: B=2, T=16, lr=1e-4, dynamicrafter_512 checkpoint at `ckts/dynami512.ckpt`.

## Plan for tomorrow

1. **Re-run training with the SDPA patch.** Expected: spatial-attention OOM gone. If it still OOMs but in a *different* location (e.g. activation pile from Factor B at backward time), proceed to (2).
2. **Flip `image_encoder_device: cpu` in `configs/diffusion_hyperalign_metaworld.yaml:69`.** Frees ~2.5 GB. Roundtrip cost ~50 ms/batch (1.3 MB CLIP-token tensor across PCIe + CPU forward).
3. If still OOM: **enable BF16 autocast** around the model forward in `Trainer.training_step` (`src/.../training/trainer.py:97`). Roughly halves activation memory. Adapter trains fine in BF16. Untested today.
4. If still OOM: **gradient checkpointing on the adapter-side UNet pass.** The two `base_model(...)` calls in the AdaptedModel + HyperAlign chain are where activations stack — wrapping the second call in `torch.utils.checkpoint.checkpoint` would trade compute for memory.
5. Last resort: `batch_size=1`. Halves activation memory linearly.

`PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is worth trying alongside any of the above — the trace itself suggests it, costs nothing.

## Not the cause (ruled out today)

- Resolution mismatch: the metaworld -> 320x512 stretch is the trained resolution, confirmed working on the rollout sanity test.
- `use_dynamic_rescale` math: only adds a scalar multiply per step, no memory impact.
- The Resampler size: it's ~150-line Perceiver, < 50 MB on GPU.

## Files of interest

- `src/generative_flow_adapters/data/batch_preprocessor.py` — the function that changed today.
- `src/generative_flow_adapters/backbones/dynamicrafter/modules/attention.py` — where the OOM line lives, and where today's SDPA patch landed.
- `src/generative_flow_adapters/models/adapted_model.py` + `src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py` — the double-base-model-forward path.
- `configs/diffusion_hyperalign_metaworld.yaml` — `image_encoder_device` knob, `temporal_length: 16` requirement.
