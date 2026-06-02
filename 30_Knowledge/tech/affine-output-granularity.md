---
type: tech-note
status: living
last_updated: 2026-05-29
sources:
  - "code: src/generative_flow_adapters/adapters/output/format.py"
  - "code: src/generative_flow_adapters/adapters/output/output_head.py"
  - "code: src/generative_flow_adapters/adapters/output/affine.py"
  - "config: configs/diffusion_output_v2_affine_metaworld.yaml"
relevance: D1 / D2  # output-adapter design surface + the format ablation
---

# Affine output granularity — `dense` vs `channel`

> **`affine_granularity` controls the *shape* of the `(scale, shift)` that the
> affine output format applies to the base prediction — i.e. how finely the
> modulation can vary across the latent.** Both granularities produce a delta
> of the form `delta = base*scale + shift` (composed as `base*(1+scale)+shift`
> under `add`); they differ only in whether `scale`/`shift` are full-resolution
> per-element maps (`dense`) or one value per channel broadcast over space and
> time (`channel`). The knob is **inert for `output_format: direct`** — it only
> shapes the affine arm. This is the exact analogue, on the output-format axis,
> of the `mask_mix_gate_kind: channel | spatial` choice in
> [[mask-mix-gate]].

The latent is `[B, C, T, H, W]` (batch, channels, frames, height, width).

## `dense` — per-element scale/shift maps

The backbone emits a full-resolution `2C`-channel tensor; `build_output_result`
splits it into a `scale` and a `shift` **the same shape as the latent**:

```
scale, shift : [B, C, T, H, W]
delta        = base * scale + shift      # element-wise, no broadcast
```

Every `(channel, frame, pixel)` site gets its own scale and shift — the
modulation can amplify the base prediction in one region of a frame and
suppress it in another. DoF: `2·C·T·H·W` per sample.

### Dense affine is a strict superset of `direct`

Expand the dense affine path through the `add` composition (`reference = base`):

```
final = base + (base*scale + shift) = base*(1 + scale) + shift
```

Force `scale ≡ 0` and it collapses to `final = base + shift` — which is exactly
the `direct` path with **`shift` playing the role of the direct delta** (same
full-resolution `[B,C,T,H,W]` shape and role). So:

- `shift`        → the additive correction = the `direct` delta.
- `base * scale` → an *extra* per-element **multiplicative** gating of the base
  prediction, which `direct` cannot express.

That is the honest framing of the ablation (better than "capacity-matched":
the affine head does emit `2C` vs `direct`'s `C`, so it has one extra
projection's worth of parameters — trivial next to the shared backbone, but not
literally equal). **`direct = dense affine with scale forced to 0`**, so dense
affine can always match direct; the question is whether the multiplicative term
earns its keep. If `scale` learns to ≈0, the format has degenerated into
`direct` and the multiplicative term isn't helping (see
[[../../50_Decisions/open/output-format-affine-vs-direct]]).

## `channel` — global per-channel FiLM

The scale/shift maps are pooled to one value per channel (mean over `T,H,W`),
then broadcast back:

```
scale, shift : [B, C, 1, 1, 1]           # pooled over T, H, W (keepdim)
delta        = base * scale + shift       # broadcasts over all frames & pixels
```

A channel is uniformly scaled-and-shifted: the same `scale[c]` hits every frame
and pixel of channel `c`. This is classic FiLM and exactly what the original
tiny `AffineOutputAdapter` (`affine.py`) does. DoF: `2·C` per sample — orders of
magnitude smaller than `dense`.

## The one line that separates them

Both paths share `build_output_result`; `channel` only adds a pooling step
(`format.py`):

```python
scale, shift = raw.chunk(2, dim=channel_dim)
if granularity == "channel":
    scale = _pool_spatial(scale, channel_dim)   # mean over T,H,W, keepdim
    shift = _pool_spatial(shift, channel_dim)
delta = reference * scale + shift                # broadcasts if pooled
```

## Comparison

| | `dense` | `channel` |
|---|---|---|
| `scale`/`shift` shape | `[B,C,T,H,W]` | `[B,C,1,1,1]` |
| Degrees of freedom | `2·C·T·H·W` | `2·C` |
| Corrects spatially-localised base errors? | yes | no — global per-channel gain/bias only |
| Relation to `direct` delta | strict superset (`direct` = `scale≡0`; `shift` *is* the delta) | superset only on the per-channel slice |
| Equivalent to | a learned residual modulation map | FiLM / the original `AffineOutputAdapter` |

## What the choice is actually testing (analysed estimate, not a result)

- If the frozen base's errors are **spatially structured** (e.g. systematically
  wrong around moving objects), `dense` can express the fix and `channel`
  cannot — `channel` would underfit.
- If the base's errors are mostly a **global per-channel miscalibration**
  (slightly wrong magnitude/offset per latent channel), `channel` captures it
  with a tiny fraction of the parameters and `dense` buys little.

Reasoning, not measurement — the numbers come from
[[../../20_Tickets/exp-adapter-output-format-affine-vs-direct]].

## Backbone caveat

The `mlp` backbone is effectively `channel` **regardless of the flag**: it has
no spatial extent to vary over (it emits a `[B, 2C]` vector from a pooled
`(t, cond)` context, like the original affine head). `dense` only becomes
meaningful for the `transformer` and `unet` backbones, which produce
full-resolution feature maps. The factory does not reject `dense` on `mlp`; it
just degenerates to `channel` semantics via the broadcast.
