---
type: refactor
scope: backbone
status: open
priority: high
created: 2026-05-22
updated: 2026-07-15
resolution:
resolution_note:
closed_at:
related: ["[[../10_now/architecture]]"]
---

# DynamiCrafter assumptions have leaked across the supposedly base-model-agnostic core

The framework's D1 selling point is that adapters compose with **any**
frozen generative backbone via the additive rule
`f(x_t, t) + g · Δ_φ(x_t, t, cond)`. In practice, DynamiCrafter
specifics (batch-dict keys, 5D `[B,C,T,H,W]` latent layout, U-Net
internal attribute names, SD2 VAE, OpenCLIP ViT-H-14, DC-flavoured
diffusion schedule) have leaked into the data layer, the adapters, the
loss, and the trainer. A second backbone (OpenSora is partially wired,
plus anything new) cannot drop in without touching all of these.

This ticket is the audit + refactor plan; sub-tickets to follow once
the protocol surfaces are agreed.

## Why it matters

- **D1 deliverable.** The framework chapter argues a clean
  composition interface across diffusion/flow backbones. The current
  code does not actually support that claim — every "agnostic" adapter
  reaches into DC U-Net internals.
- **D2/D3 generalisation.** Any second backbone (OpenSora, Wan, a
  smaller diffusion baseline for ablations, a flow-matching base) is
  currently blocked behind a refactor of the data layer and adapters.
- **HyperAlign is provider-gated.** `adapters/factory.py:135` literally
  raises if `provider != "dynamicrafter"` — a paper-aligned adapter
  family that cannot be evaluated on other backbones is not really a
  reusable adapter family.

## Update (2026-07-15) — a second backbone now exists, but via a different path than this ticket envisioned

Since this audit was written, **WAN2.2 (`provider: wan2.2`/`wan2.2_external`)
has become the primary backbone for the whole D2/D3 debugging effort** — a
frozen flow-matching video model, used end-to-end with its own dedicated
adapter family (`Wan21OutputAdapter`/`ActionWanModel`,
`backbones/wan/modules/action_model.py`) that does **not** reach into DC U-Net
internals. This partially satisfies this ticket's second "done when" bullet
("a second backbone has been used end-to-end against at least one adapter
family without touching DC code").

**But this happened via an alternate path, not the refactor envisioned here**:
a parallel, WAN-native adapter class was built from scratch
(`adapters/output/wan.py`), rather than the generic `ConditionSpec`/
`BaseGenerativeModel` protocol (items 1–2 in the refactor plan below) that
would let *any* adapter family — including HyperAlign — target any backbone.
Concretely still unaddressed:

- **The six refactor items are still not split into sub-tickets** (this
  ticket's first "done when" bullet).
- **`adapters/factory.py:135` still hard-guards HyperAlign to
  `provider == "dynamicrafter"`** — not re-checked this session, but nothing
  in the WAN work touched that guard or HyperAlign at all, so almost
  certainly still true.
- The DC-specific coupling audited below (batch/data layer, U-Net
  introspection, conditioning prep, training/losses) is about DynamiCrafter
  specifically — WAN's own adapter avoided the problem by not being built on
  top of DC's `DynamicCrafterOutputAdapter` at all, not by fixing the
  underlying leakage for adapters that *do* need to generalize across
  backbones (HyperAlign, UniCon).

Net: real evidence the framework *can* support a second backbone, but the
audit's actual architectural goal (one generic adapter/backbone interface,
not two parallel hard-coded ones) is unchanged. Still open.

## Audit — where the coupling lives

### Tier 1: hard coupling (real refactor needed)

**Batch / data layer — DC-specific dict keys (`context`, `concat`, `fs`,
`act`, `dropout_actions`) and 5D `[B,C,T,H,W]` latent shape are baked
in:**

- `src/generative_flow_adapters/data/batch_preprocessor.py:64-156` —
  class literally named `DynamiCrafterBatchPreprocessor`; builds
  `cond["concat"]` (first-frame replicate across `T`), `cond["fs"]`
  (frame stride), CLIP image cross-attention via the DC Resampler,
  hardcodes scale factor `0.18215`.
- `src/generative_flow_adapters/data/latent_encoder.py:22-38, 111-135`
  — `SD_VAE_DDCONFIG` hardcoded (z=4, res 256, SD2 `ch_mult`);
  `load_dynamicrafter_checkpoint()` looks for `first_stage_model.*`
  prefix.
- `src/generative_flow_adapters/data/clip.py:21-24` — frozen DC
  OpenCLIP constants: ViT-H-14 / laion2b_s32b_b79k / penultimate / 77.
- `src/generative_flow_adapters/data/_resampler.py` — entire module is
  the DynamiCrafter image-projection Resampler.

**Adapters reach into the DC U-Net internals:**

- `src/generative_flow_adapters/adapters/output/dynamicrafter.py:10-145`
  — calls `prepare_dynamicrafter_condition`, uses DC `UNetModel`,
  threads `context/act/fs/concat/dropout_actions` through the forward.
- `src/generative_flow_adapters/adapters/hidden_states/unicon.py:13-14,
  81-96, 431-482` — imports `timestep_embedding` / `prob_mask_like`
  from DC; deep-copies `module.output_blocks`, `module.out`; accesses
  `module.input_block_chans`, `module.middle_block[0].channels`,
  `module.time_embed[0].weight`, `module.action_embed`,
  `module.null_action_emb`, `module.add_act_time_emb`. Encodes the
  magic context split `context_tokens == 77 + frames * 16` (DC's text
  + image-cross-attn token layout).
- `src/generative_flow_adapters/adapters/hypernetworks/hyperalign.py:9,
  18-19, 200-273` — same pattern: DC condition prep + DC U-Net
  internals (`output_blocks`, `input_block_chans`, `middle_block`,
  `out`).
- `src/generative_flow_adapters/adapters/factory.py:118-120` —
  provider-sniff for `input_summary_dim`.
- `src/generative_flow_adapters/adapters/factory.py:134-166` — hard
  guard: paper-aligned HyperAlign raises if
  `provider != "dynamicrafter"`.

**Conditioning prep helper:**

- `src/generative_flow_adapters/conditioning/utils/dynamicrafter_conditioning.py:9-33`
  — `prepare_dynamicrafter_condition()`, consumed by the output
  adapter and HyperAlign.

**Backbone factory:**

- `src/generative_flow_adapters/models/base/factory.py:4, 27-40` — DC
  branch with `unet_config_path`, `allow_dummy_concat_condition`,
  `load_first_stage_model`.
- `src/generative_flow_adapters/models/base/dynamicrafter.py:14, 47-136`
  — the wrapper itself is fine; the leakage is that every
  adapter/trainer assumes _this_ wrapper's contract.

**Training / losses:**

- `src/generative_flow_adapters/training/trainer.py:32-42, 84-104` —
  pulls DC schedule keys (`linear_start`, `rescale_betas_zero_snr`,
  `use_dynamic_rescale`, `base_scale`, `turning_step`); line 91 carries
  the author's own TODO: "this is everything very dynamicrafter
  orientated atm".
- `src/generative_flow_adapters/training/wandb_logger.py:114-120` —
  `_decode_to_uint8` assumes a 5D `[B,3,T,H,W]` decoder output.
- `src/generative_flow_adapters/losses/diffusion.py:6-7, 57-96` —
  imports `extract_into_tensor`, `make_beta_schedule`,
  `rescale_zero_terminal_snr` from `backbones.dynamicrafter.*`;
  `scale_x_start()` is DC's data-side SNR rescaling; offset-noise
  branch keys off `x_start.dim() == 5`.

### Tier 2: soft coupling (naming, docs, configs, tests, scripts)

- DC-only configs: `configs/diffusion_hyperalign_metaworld.yaml`,
  `configs/diffusion_hyperalign_shortcut_metaworld.yaml`,
  `configs/diffusion_avid_shortcut_metaworld.yaml`. All set
  `provider: dynamicrafter` and reference
  `external_repos/avid/latent_diffusion/configs/train/dynamicrafter_512.yaml`.
- DC-only scripts: `scripts/train_hyperalign_metaworld.py`,
  `scripts/train_hyperalign_shortcut_metaworld.py`,
  `scripts/train_avid_shortcut_metaworld.py`.
- DC-bound tests: `tests/test_dynamicrafter_integration.py`,
  `tests/test_dynamicrafter_checkpoint_sanity.py`,
  `tests/test_batch_preprocessor.py`, `tests/test_video_logging.py`,
  `tests/test_hyperalign_architecture.py`,
  `tests/test_null_caption.py`.
- `src/external_deps/lvdm/` — vendored DC VAE; only imported by
  `data/latent_encoder.py:57-58` (well-isolated).
  `src/external_deps/avid_utils/` — not imported by core src.
- `pyproject.toml:22-27` — optional `[dynamicrafter]` extra (einops,
  pytorch-lightning, open_clip_torch). Base deps are clean.

## Refactor plan (priority order)

1. **Generalise the batch contract.** Replace the DC-keyed dict
   (`concat`/`fs`/`context`/`act`/`dropout_actions`) with a typed,
   backbone-declared `ConditionSpec`. The preprocessor produces
   whatever keys the wrapper declares it needs; DC-specific fields
   become a DC adapter on top of a generic
   `VideoLatentBatchPreprocessor`.
2. **Move U-Net introspection behind the wrapper.** Adapters must not
   reach into `module.output_blocks` / `module.input_block_chans` /
   `module.action_embed`. Add a small protocol on
   `BaseGenerativeModel` — e.g. `get_skip_channels()`,
   `get_decoder_blocks()`, `embed_timestep()`, `embed_action()` — and
   let each backbone implement it.
3. **Drop the HyperAlign provider check.** It exists because HyperAlign
   reads DC internals — once (2) lands, the guard at
   `adapters/factory.py:135` goes away.
4. **Move DC diffusion-schedule pieces out of `losses/diffusion.py`.**
   Either inline a generic `make_beta_schedule` in `losses/`, or have
   the wrapper own the noise scheduler and pass scalar primitives
   (`alpha_bar`, `sigma`) into the loss.
5. **Decouple VAE / CLIP from DC.** `latent_encoder.py` and `clip.py`
   should accept a codec / encoder object rather than hardcoding SD2 +
   ViT-H-14. The DC defaults stay available as a registered codec.
6. **Rename and split.** `DynamiCrafterBatchPreprocessor` →
   `VideoLatentBatchPreprocessor` (generic) + a thin DC adapter that
   fills in `concat` / `fs` / DC CLIP image tokens.

Each numbered item should become its own sub-ticket once we lock the
protocol surfaces (item 1 and item 2 are the load-bearing ones — they
unblock all of the others).

## Definition of done (this ticket)

This is the **audit** ticket. It's done when:

- The six refactor items above are split into sub-tickets with their
  own protocol sketches (at minimum: item 1 `ConditionSpec` and
  item 2 `BaseGenerativeModel` extensions).
- A second backbone has been used end-to-end against at least one
  adapter family without touching DC code (probably starts as OpenSora
  + a LoRA or output adapter — the smallest viable proof).
- `adapters/factory.py:135` no longer raises on non-DC providers for
  HyperAlign.

## Out of scope

- Removing DynamiCrafter as a supported backbone. DC is a first-class
  citizen and the main thesis runs are against it. The goal is to
  contain it, not delete it.
- Replacing the vendored `external_deps/lvdm/` VAE. It stays as the
  default codec, just registered behind a generic interface.
- Performance / kernel-level work. Pure interface refactor.

## Related

- `src/generative_flow_adapters/adapters/factory.py:135` — the provider
  guard that has to go.
- `src/generative_flow_adapters/training/trainer.py:91` — author TODO
  confirming the DC orientation.
- D1 framework deliverable in `Home.md`.
