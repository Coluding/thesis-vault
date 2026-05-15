---
type: tech-note
status: living
last_updated: 2026-05-15
sources:
  - "code: src/generative_flow_adapters/data/batch_preprocessor.py"
  - "code: src/generative_flow_adapters/models/base/dynamicrafter.py"
  - "code: src/generative_flow_adapters/adapters/output/dynamicrafter.py"
  - "code: src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py"
  - "code: src/generative_flow_adapters/testing/fake_data.py"
  - "code: src/external_deps/lvdm/basics.py"
  - "code: src/external_deps/lvdm/models/ddpm3d.py"
  - "config: configs/test_dynamicrafter_hyperalign_unet.yaml"
  - "config: configs/diffusion_hyperalign_metaworld.yaml"
commit: 3a235c8
relevance: D2  # action-conditioned diffusion world models
---

# `cond["concat"]` — channel-concatenation conditioning

> Documents the `concat` entry in the conditioning dictionary that the
> trainer hands to the base model and the adapters. `cond["concat"]` is
> the DynamiCrafter image-conditioning channel pack: a tensor that is
> **concatenated to `x_t` along the channel axis** before the UNet's
> first input block. Its presence (or absence) changes the input-channel
> count the UNet sees, which is why it interacts with
> `allow_dummy_concat_condition` and the `in_channels` config field.

## TL;DR

`cond["concat"]` is the first-frame VAE latent, replicated across the
temporal axis, packed under the `"concat"` key of the conditioning dict
produced by `DynamiCrafterBatchPreprocessor`. Both the frozen base UNet
wrapper and the DynamiCrafter output adapter look for `cond["concat"]` and
`torch.cat` it onto `x_t` (or onto the adapter input) along `dim=1` before
calling the underlying `UNetModel`. The HyperAlign hypernetwork's
encoder-pass shim does the same thing after reshaping for the per-frame 2D
pass. The base DynamiCrafter UNet is built with `in_channels = 8`
(`4 latent + 4 concat`), so when an experiment runs without a real concat
tensor (e.g. MetaWorld actions only), the wrappers will zero-pad the
missing 4 channels iff `allow_dummy_concat_condition=True` — otherwise the
forward pass errors on channel mismatch.

## Where it lives

| Piece | File | Lines |
|---|---|---|
| Producer — `DynamiCrafterBatchPreprocessor` (writes `cond["concat"]`) | `src/generative_flow_adapters/data/batch_preprocessor.py` | 90–134 |
| Fake-data producer (tests) | `src/generative_flow_adapters/testing/fake_data.py` | 158–171 |
| Consumer — base UNet wrapper | `src/generative_flow_adapters/models/base/dynamicrafter.py` | 99–136 |
| Consumer — DynamiCrafter output adapter | `src/generative_flow_adapters/adapters/output/dynamicrafter.py` | 89–145 |
| Consumer — HyperAlign encoder-pass shim | `src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py` | 596–621 |
| `allow_dummy_concat_condition` plumbing | `src/generative_flow_adapters/models/base/factory.py:38`; `src/generative_flow_adapters/adapters/factory.py:40–41` | — |
| UNet `in_channels` (config) | `configs/test_dynamicrafter_hyperalign_unet.yaml` | 10 |
| Upstream naming reference (`"concat" → "c_concat"`) | `src/external_deps/lvdm/models/ddpm3d.py` | 36 |
| Upstream consumer (`xc = torch.cat([x] + c_concat, dim=1)`) | `src/external_deps/lvdm/models/ddpm3d.py` | 1436–1447 |

All line numbers are at commit `3a235c8` on `main`.

## What `cond["concat"]` actually is

The producer is `DynamiCrafterBatchPreprocessor.__call__` in
`data/batch_preprocessor.py`. The relevant block
(`batch_preprocessor.py:108–134`):

```python
cond_idx = int(config.cond_frame_index)
if config.rand_cond_frame and train:
    cond_idx = int(torch.randint(0, frames, (1,)).item())

# ...

if config.include_concat:
    frame_latent = z[:, :, cond_idx, :, :].unsqueeze(2)   # (B, 4, 1, h, w)
    frame_latent = frame_latent * image_keep              # CFG drop
    cond["concat"] = frame_latent.repeat(1, 1, frames, 1, 1)  # (B, 4, T, h, w)
```

So `cond["concat"]` is:

- **Shape:** `(B, C_latent, T, h, w)` — same shape as `target` and `x_t`.
  With the default DynamiCrafter VAE, `C_latent = 4`.
- **Content:** the VAE-encoded latent of a single conditioning frame
  (default `cond_frame_index=0`, i.e. the first frame), replicated across
  the time axis so every frame of `x_t` is concatenated with the *same*
  conditioning frame. When `rand_cond_frame=True` and `train=True`, the
  conditioning frame is sampled uniformly per batch.
- **CFG drop:** multiplied element-wise by `image_keep`
  (`batch_preprocessor.py:133, 214–215`). When CFG drops the image branch
  on a sample, the concat tensor becomes all-zero on that sample, so the
  base model sees "no image" through the concat channels for that example.
- **Toggle:** only written when `BatchPreprocessConfig.include_concat=True`
  (default `True`, `batch_preprocessor.py:53`).

The conditioning frame index is **shared** with the image
cross-attention branch (`batch_preprocessor.py:108–123`), so the
CLIP-image embedding and the concat-channel image always look at the same
frame. The CFG mask is also shared (same `image_keep`), so when CFG drops
the image, both branches drop together.

## How it gets consumed

### Base UNet wrapper (`models/base/dynamicrafter.py:99–136`)

```python
if isinstance(cond, dict):
    # ...
    concat = cond.get("concat")
    # ...
    if isinstance(concat, Tensor):
        model_input = torch.cat([model_input, concat.to(...)], dim=1)

if self.allow_dummy_concat_condition:
    expected_channels = getattr(self.module, "in_channels", model_input.shape[1])
    missing_channels = expected_channels - model_input.shape[1]
    if missing_channels > 0:
        dummy_concat = torch.zeros(..., missing_channels, ...)
        model_input = torch.cat([model_input, dummy_concat], dim=1)

return self.module(model_input, timesteps=t, context=..., act=..., fs=...)
```

Two paths:

1. **Real concat present:** `model_input = cat(x_t, concat)` along `dim=1`.
   For DynamiCrafter, this brings the channel count from `4` to `8`,
   matching `unet_config.params.in_channels = 8` from the upstream YAML.
2. **No concat, dummy padding allowed:** if `cond["concat"]` is missing or
   not a tensor and the wrapper was built with
   `allow_dummy_concat_condition=True`, the wrapper zero-pads up to
   `module.in_channels`. This is how MetaWorld training runs with the
   DynamiCrafter UNet without an image input.

If neither path applies (no concat tensor and
`allow_dummy_concat_condition=False`), the underlying convolution will
raise a channel-mismatch error on the first input block — there is no
silent fallback.

### DynamiCrafter output adapter (`adapters/output/dynamicrafter.py:111–145`)

Mirrors the base wrapper's behaviour on the adapter side. After
optionally cat-ing `base_output` onto `x_t` for the
`condition_on_base_outputs=True` case, it cats `cond["concat"]` and then
applies the same dummy-pad logic if its own
`allow_dummy_concat_condition` is set. The default cascade in the adapter
factory pulls the flag from the adapter's `extra` first, falling back to
the model's `extra` (`adapters/factory.py:40–41`).

### HyperAlign encoder-pass (`adapters/hypernetworks/hyperalign.py:596–621`)

`_apply_channel_concat_for_input_blocks` replays the same logic when the
HyperAlign hypernetwork manually re-runs the frozen UNet's encoder to
capture activations. The relevant difference is the layout: the
hypernetwork runs the encoder per-frame in a `(B*T, C, H, W)` 2D pass, so
the `(B, C_concat, T, H, W)` concat tensor is permuted and reshaped first
(`hyperalign.py:613`):

```python
flat = concat.permute(0, 2, 1, 3, 4).reshape(B * T, C_concat, H, W)
return torch.cat([h, flat], dim=1)
```

The fallback dummy-pad uses the wrapper's
`allow_dummy_concat_condition` flag (`hyperalign.py:615`), so this shim
inherits the same behaviour as the base.

## Config plumbing

`cond["concat"]` is exclusive to the DynamiCrafter family. Three
config-side handles control it:

- **`BatchPreprocessConfig.include_concat`** (Python, default `True`,
  `batch_preprocessor.py:53`) — flips the producer off if you want to feed
  a DynamiCrafter UNet without the concat branch (rare; you'd also need
  `allow_dummy_concat_condition=True` on the consumer side to avoid the
  channel-mismatch error).
- **`BatchPreprocessConfig.cond_frame_index` / `rand_cond_frame`**
  (`batch_preprocessor.py:48–49, 110–112`) — which frame is replicated.
- **`model.extra.allow_dummy_concat_condition`** (YAML, default `false`,
  e.g. `configs/diffusion_hyperalign_metaworld.yaml:16`) — toggles the
  zero-padding fallback. Used by every example/test that runs a
  DynamiCrafter UNet without real image inputs (MetaWorld, fake-data
  smoke tests).
- **`adapter.extra.allow_dummy_concat_condition`** (YAML, optional) —
  overrides the per-adapter flag when the adapter has its own input
  branch (`adapters/factory.py:40–41`). Defaults to whatever the model's
  flag is.

## Naming: why `"concat"`?

The upstream LVDM / DynamiCrafter codebase routes conditioning through a
dispatch table:

```python
# src/external_deps/lvdm/models/ddpm3d.py:36
__conditioning_keys__ = {"concat": "c_concat", "crossattn": "c_crossattn", "adm": "y"}
```

The string `"concat"` is the upstream's name for "channel-concatenated
conditioning" — image-to-video models like DynamiCrafter use it for the
first-frame image latent that anchors the generated video. This repo
keeps the same key name on the user-facing side (`cond["concat"]`) and
performs the actual `torch.cat` itself in the wrapper, so the
`c_concat`/`c_crossattn` packing of the upstream
`HybridConditioner` (`src/external_deps/lvdm/basics.py:93–101`) is bypassed.
The naming is therefore *historical*, not descriptive — `cond["concat"]`
is not "anything I want to concatenate", it's specifically the image
conditioning channel pack.

## Fake-data shape (tests)

For unit tests that don't run the real preprocessor, `make_fake_batch`
synthesises a matching cond dict
(`testing/fake_data.py:158–171`):

```python
if spec.cond_kind == "dynamicrafter":
    cond = {
        "context": torch.randn(spec.context_shape),
        "concat":  torch.randn(latent_channels, temporal_length, latent_height, latent_width),
    }
```

Same shape contract as the real producer.

## Gotchas

- **It changes the input-channel count of the UNet.** The base
  DynamiCrafter UNet is built with `in_channels = 8`. If `cond["concat"]`
  is `None` and `allow_dummy_concat_condition=False`, the first conv will
  fail with a channel-mismatch error. The two recovery paths are: provide
  a real concat tensor, or enable dummy padding.
- **Dummy padding is silent.** With
  `allow_dummy_concat_condition=True` and no concat tensor, the model
  sees four zero channels appended to `x_t`. The base weights were
  trained on real first-frame latents, so the model is being asked to
  generalise off-distribution on the image input. This is intentional
  for action-conditioned ablations (MetaWorld config), but the
  consequence on rollout quality is an open ablation, not a measured
  fact in this repo. _needs verification: zero-pad vs first-frame-latent
  rollouts._
- **The four channels are not arbitrary — they're VAE latents.** The
  producer goes through `self.vae.encode_video` first
  (`batch_preprocessor.py:94`). Any first-party caller that wants to feed
  raw pixels as the concat condition must encode them first; the
  consumer wrappers expect the latent layout.
- **Same shape as `target`/`x_t`.** Because the concat tensor is the VAE
  encoding of a single frame *replicated across time*, it has the same
  `(B, C_latent, T, h, w)` shape as `x_t`. After channel-cat this
  doubles the channel dim to `2 * C_latent`.
- **CFG drop happens at producer time, not consumer time.** The wrapper
  doesn't know whether a given sample had its image branch dropped — it
  just sees an all-zero concat tensor on those samples. If you build a
  custom preprocessor, replicate the multiplicative `image_keep` mask or
  CFG won't work end-to-end.
- **HyperAlign expects the (B, C, T, H, W) layout in the cond dict, even
  though it consumes it as (B*T, C, H, W) internally.** The producer's
  contract is the 5-D form; the per-frame reshape happens in
  `_apply_channel_concat_for_input_blocks`. Don't pre-flatten on the
  producer side.
- **`pass_cond_to_base=False` does not drop concat from the base.** The
  cond is filtered by the AdaptedModel, but `cond["concat"]` flows
  through the base UNet's path regardless of adapter flags — it is a
  pretraining-time input contract, not a conditioning signal the adapter
  owns. (See `pass_cond_to_base: false` in
  `configs/diffusion_hyperalign_metaworld.yaml:8` — concat still applies.)
- **Per-adapter override beats per-model.** `adapter.extra.allow_dummy_concat_condition`
  is read first; only when missing does the factory fall back to
  `model.extra.allow_dummy_concat_condition`
  (`adapters/factory.py:40–41`). Useful if the base accepts a real concat
  but a specific output adapter operates on a different channel layout.

## Defaults at a glance

| Param | Default | Source |
|---|---|---|
| `BatchPreprocessConfig.include_concat` | `True` | `batch_preprocessor.py:53` |
| `BatchPreprocessConfig.cond_frame_index` | `0` | `batch_preprocessor.py:48` |
| `BatchPreprocessConfig.rand_cond_frame` | `False` | `batch_preprocessor.py:49` |
| `model.extra.allow_dummy_concat_condition` | `False` (objective default); `True` in MetaWorld config | `models/base/factory.py:38`; `configs/diffusion_hyperalign_metaworld.yaml:16` |
| DynamiCrafter UNet `in_channels` | `8` (`4 + 4`) | `configs/test_dynamicrafter_hyperalign_unet.yaml:10` |
| VAE `scale_factor` | `0.18215` | `batch_preprocessor.py:19` (docstring); `models/base/dynamicrafter.py:171` |

## Related

- [[dynamic-rescale]] — separate data-side mechanism on `x_t`. They
  compose; `cond["concat"]` is its own channel branch.
- [[../../10_now/architecture]] — adapter and backbone composition.
  Worth adding a one-line note in the DynamiCrafter section that the
  base's `in_channels=8` contract is preserved through the concat key.

## Open follow-ups

- [ ] Ablation: real first-frame-latent concat vs dummy zero-pad on
      MetaWorld. _No run logged yet._
- [ ] Decide whether `allow_dummy_concat_condition=False` should raise
      with a clearer error message that names the flag, instead of
      surfacing as a conv channel-mismatch error.
      → consider opening `50_Decisions/open/concat-channel-guard.md`.
- [ ] Document the cross-attention image branch (`image_emb`,
      `_encode_image_branch`) — it shares the conditioning frame with
      `cond["concat"]` and the CFG mask but lands in `cond["context"]`,
      not `cond["concat"]`. Probably a sibling note: `context-condition.md`.
