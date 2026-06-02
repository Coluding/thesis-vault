---
type: paper
status: living
last_updated: 2026-05-28
title: "UniCon — Unified Hidden-State Conditioning for Diffusion"
authors: []
venue:
year:
url:
local_pdf: docs/paper/unicon.pdf
relevance: framework, baseline
deliverable: D1
---

# UniCon

> Hidden-state / skip-connection control adapter for diffusion. Direct
> reference for the thesis's hidden-state adapter family.

## Status of this note

PDF still pending verification (title/authors/venue/year/URL). What is
*verified* below is the implementation side: three UniCon-derived adapters
live in `src/generative_flow_adapters/adapters/hidden_states/unicon.py` and
all three are scheduled for testing next, with integration of the lessons
we picked up from the hypernetwork (HyperAlign) and output (AVID) families.

## Why it matters for the thesis

UniCon is the direct precedent for the **hidden-state adapter family** (D1).
The codebase reifies UniCon's Figure-3 panels as three discrete adapter
classes that share a common feature-capture path against a frozen
U-Net backbone (currently DynamiCrafter 512):

```
1. base.forward(x_t, t, cond)
   └── forward hooks capture: input_block activations,
       middle_block activation, output_block activations
2. adapter.forward(x_t, t, cond, base_output=None)
   └── uses captured features + a trainable replica of part of the U-Net
   └── returns an OutputAdapterResult (prediction + optional gate)
3. AdaptedModel composes base_output with adapter result
   (add / replace / mask_mix — same composition contract as HyperAlign/AVID)
```

## The three variants

### 1. `UniConHiddenStateAdapter` — Figure 3(d), decoder-focused

What it trains:
- Deep-copy of `module.output_blocks` (decoder ResBlocks + attention).
- Deep-copy of `module.out` (final RGB / latent head).
- Three families of zero-initialised connectors:
  - `middle_connector` — injects the captured middle activation as the
    decoder's starting state.
  - `skip_connectors[i]` — modulate each captured encoder skip before it's
    concatenated to the decoder.
  - `decoder_connectors[i]` — modulate each decoder-block output using the
    corresponding captured base output activation as the source signal.
- Optional `mask_head` (deep-copy of `module.out_mask` when
  `output_mask=True`) → produces a per-pixel gate consumed by
  `AdaptedModel`'s `mask_mix` composition.

Parameter cost: ~ decoder + heads + connectors. The encoder of the base is
never replicated; encoder skips are taken from the frozen base via hooks.

Intuition: "freeze the encoder, retrain the decoder, but condition every
decoder hidden state on what the frozen decoder would have produced."
The connectors are zero-init so at training step 0 the adapter is the
identity on top of the captured base activations.

Connector type (`connector_type`):
- `zeroconv` — additive zero-conv (ControlNet-style).
- `zeroft` — additive *and* multiplicative zero-init paths (default for
  UniCon): `out = target + add(source) + target * scale(source)`.

### 2. `ReplaceDecoderHiddenStateAdapter` — Figure 3(e), decoder swap

Same idea as variant 1 but **without** the connectors. The captured encoder
skips and middle activation are routed *unmodified* into a fresh,
trainable decoder. No zero-init guardrails — the adapter starts from a
random decoder copy and learns to denoise from scratch given the frozen
encoder's features.

Parameter cost: decoder + heads. Slightly cheaper than UniCon (no
connectors).

Intuition: ablation of the connectors. Tells us how much of UniCon's
behaviour comes from the connector machinery vs. simply having a trainable
decoder over frozen-encoder features.

### 3. `FullSkipLayerControlAdapter` — Figure 3(c), full ControlNet replica

What it trains:
- Deep-copy of `module.input_blocks` (the encoder).
- Deep-copy of `module.middle_block`.
- Deep-copy of `module.output_blocks` (the decoder).
- Deep-copy of `module.out` and optional `out_mask`.
- One zero-init connector per encoder block, the middle, and each decoder
  block. Connectors fuse the *replicated* activation with the *captured base*
  activation at the matching depth.

Parameter cost: ~full U-Net + connectors. By far the most expensive of the
three; closest to vanilla ControlNet's "trainable copy of the encoder
half plus the middle" but extended to the full network.

Intuition: the upper-bound expressivity in the hidden-state family. If the
cheaper variants underperform, this is the fallback that gives the
adapter enough capacity to override the base's behaviour anywhere.

## Shared infrastructure (all three)

- `_UNetFeatureStore` — forward hooks on `input_blocks`, `middle_block`,
  `output_blocks`. Records `.detach()`'d activations; the captured-features
  graph is implicitly the no-grad reference pass that `AdaptedModel`
  triggers before calling the adapter.
- `_prepare_unet_runtime` — rebuilds the U-Net's `emb` (time + optional
  action + optional fs) and `context` (text + img tokens) so the trainable
  decoder copy can be driven the same way the base would be. Currently
  reimplements the DynamiCrafter conditioning logic instead of calling the
  shared `prepare_dynamicrafter_condition` helper used by HyperAlign and
  AVID — *opportunity for consolidation*.
- `_prepare_adapter_conditioning` — builds an MLP that projects the
  structured condition embedding (e.g. action latents) into the
  time-embedding dimension and an `emb_fuse` head that adds the projection
  into `emb`. Zero-init on the final layer so training starts from the
  unmodified base time embedding.
- `OutputAdapterResult` — shared return type. When `output_mask=True` and
  composition is `mask_mix`, the gate is computed from the adapter's own
  `out_mask` head (deep-copy of the base's mask head).

## How it maps to our adapter taxonomy

- UniCon is a **hidden-state** adapter family registered under
  `adapter.type: hidden_state` with `extra.architecture ∈ {unicon,
  replace_decoder, full_skip_controlnet}`.
- All three return their composition through the same
  `OutputAdapterResult` contract, so they slot into the trainer's
  composition modes (`add`, `replace`, `mask_mix`) without bespoke logic.

## What we're carrying over from HyperAlign + AVID

The two adapter families we've already exercised end-to-end on DynamiCrafter
brought several lessons that UniCon doesn't yet have:

1. **Step-level conditioning for shortcut training** — adapter-side
   `Linear(1, h)→SiLU→Linear(h, cond_dim)` embedding of `cond[step_level]`,
   combined with the structured condition embedding via
   `prepare_dynamicrafter_condition` and `combine_adapter_embeddings`. UniCon
   currently has no step-level branch, so it cannot be trained under our
   shortcut/distillation loss without modification.
2. **Composition contract** — both HyperAlign and AVID accept
   `composition ∈ {add, replace, mask_mix, avid_mask_mix}` via
   `AdaptedModel`. UniCon already returns `OutputAdapterResult` so the wiring
   is correct; it just needs configs that pick the composition explicitly.
3. **`mask_mix` gating** — AVID emits a per-pixel mask via `out_mask`, and
   HyperAlign supports `channel` vs `spatial` gate kinds. UniCon's existing
   mask path mirrors AVID (per-pixel head); a `channel` head would be a
   cheap follow-up if the per-pixel one is unstable.
4. **Embedding-strip parity** — the frozen base was never trained on
   `cond['embedding']`. HyperAlign strips it before the encoder pass; AVID
   never passes it. UniCon captures via hooks during the *base* call, so it
   needs the same stripping (or the same `pass_cond_to_base=true` discipline
   AVID uses).
5. **bf16 amp + flash SDPA** — running the frozen base under
   `training.extra.amp_dtype: bf16` routes attention through the Flash
   kernel on Ampere+ at no quality cost (weights stay fp32, no GradScaler).
   Confirmed working with HyperAlign and AVID; should transfer to UniCon as-is.

## Open questions for the chapter

- Verify the diagram labels (Figure 3 panels c/d/e) from the PDF; the
  current panel→class mapping comes from the codebase's docstrings and
  needs to be pinned to a page number.
- Are the connector designs (`zeroconv`, `zeroft`) from UniCon proper or
  from adjacent ControlNet-family papers? Needs verification.
- Parameter-count ladder for the three variants (Replace ≤ UniCon ≤
  FullSkipControlNet) — measure with the FLOPs/params script once it's
  wired up (see `20_Tickets/feat-adapter-flops-per-step-estimator.md`).

## Related

- [[_MOC]]
- [[../../10_now/architecture]] — see Adapter families, hidden-state column
- [[avid]] · [[hyperalign]] · [[cafm]] — the other adapter-side neighbours
- [[../tech/mask-mix-gate]] · [[../tech/shortcut-training-modes]] · [[../tech/flash-attention-sdpa-bf16]]
- `src/generative_flow_adapters/adapters/hidden_states/unicon.py`
- `configs/diffusion_unicon_metaworld.yaml` (and `..._shortcut_...`) — to be created
