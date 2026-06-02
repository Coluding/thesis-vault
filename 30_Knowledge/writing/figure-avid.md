---
type: figure-design
status: living
last_updated: 2026-05-19
target: advisor-presentation
source: figures/avid.html
deliverable: D1, D2
---

# Figure — AVID architecture

> One-slide figure for the advisor presentation showing the AVID-style
> output-adapter architecture as implemented in this repo. Mirrors the
> two-pass layout used in [[figure-hyperalign]] for visual parallelism:
> frozen base produces a base prediction, trainable adapter produces a
> residual, the two combine to give the adapted output.

## What this figure asserts

Each claim is anchored to code.

1. **The output adapter is its own 3D U-Net** (same architecture family
   as the frozen base — `UNetModel` from
   `backbones/dynamicrafter/modules/networks/openaimodel3d.py`). It is
   trainable end-to-end. Source: `adapters/output/dynamicrafter.py:53`
   (`self.module = UNetModel(**params)`).
2. **The adapter's input is channel-concatenated `[x_t || v_base]`** when
   `condition_on_base_outputs=True` (the AVID-faithful default). The
   adapter input channels = base input channels + base output channels.
   Source: `dynamicrafter.py:50, 118–119`.
3. **The structural encoder's output `c = cond["embedding"]` is fused
   into the adapter's time embedding.** A learnable MLP
   (`adapter_condition_proj`: `Linear → SiLU → Linear`) projects `c` to
   the time-embed dimension; the result is *added to* (or concatenated
   with — depends on `add_act_time_emb`) the time embedding before it
   feeds the residual blocks. Source:
   `openaimodel3d.py:460–467` (proj definition) and `:769–776`
   (`cond_emb = self._prepare_adapter_embedding(...)`,
   `emb = time_emb + cond_emb`).
4. **The frozen base runs once, with no adapter influence**, to produce
   `v_base = f_base(x_t, t, c)`. The base U-Net is identical to the
   adapter's architecture but its weights are frozen.
5. **The adapter produces a residual** `δ = Δ_φ(x_t, t, c, v_base)`.
   With `condition_on_base_outputs=True` the adapter explicitly sees
   `v_base` as part of its input (this is the AVID-faithful behaviour).
6. **Composition is additive (default) or mask-mixed.** Default:
   `v_out = v_base + δ`. When `output_mask=True`, the adapter also emits
   a per-element gate `g`, and composition becomes the mask-mix form
   (this is the AVID gating extension). Source:
   `OutputAdapterResult(adapter_output=..., output_kind="prediction", gate=gate)`
   at `dynamicrafter.py:150–151`.

## What this figure deliberately omits

- The `act` and `fs` (frame-stride) paths into the time embedding —
  those are existing DynamiCrafter conditioning mechanisms, not the
  thesis's contribution. Mentioned in a small annotation.
- Classifier-free guidance dropout on `c` — handled by the structural
  encoder, separately figured.
- The `allow_dummy_concat_condition` zero-padding logic — implementation
  detail, not architectural.
- The `step_level` embedding fusion path — relevant for D3 (shortcut),
  shown in `figure-shortcut-training` instead.
- The mask-mix composition variant. Default `add` composition is shown;
  mask-mix is mentioned in caption only.

## Layout intent

Mirrors `figure-hyperalign` for visual parallelism.

- **Top zone (trainable adapter):** input row `[x_t, t, c]` and
  `v_base`; channel-concat `[x_t ‖ v_base]`; time embed; cross-attn
  context; **`c` → MLP → +emb fusion arrow** (the explicit injection
  point); 3D U-Net stack; output `δ`.
- **Bottom zone (frozen base):** input row `[x_t, t, c]`; frozen 3D
  U-Net; output `v_base`.
- **Right zone:** sum node `v_out = v_base + δ`.
- Curved arrow from `v_base` (bottom) up into the adapter input row
  shows the `condition_on_base_outputs=True` feedback.

## Decisions (2026-05-19)

- **Two-pass structure**: visual parallel with HyperAlign. Frozen base
  and trainable adapter drawn as two parallel paths converging at the
  composition node.
- **Adapter internals**: expanded — show channel-concat input, time
  embed + condition fusion, cross-attention context, U-Net core, output
  head. Not a black box.
- **Composition shown**: `add` only. `mask_mix` mentioned in caption.

## Related

- [[../related-work/avid]] — paper-side note
- [[../tech/structural-encoder]] — to be written
- [[figure-hyperalign]] · [[figure-shortcut-training]] · [[figure-structural-encoder]]
- Code: `src/generative_flow_adapters/adapters/output/dynamicrafter.py`
- Code: `src/generative_flow_adapters/conditioning/utils/dynamicrafter_conditioning.py`
- Code (UNet internals): `src/generative_flow_adapters/backbones/dynamicrafter/modules/networks/openaimodel3d.py`
  — `adapter_condition_proj` at line 460, fusion at line 769–776.
