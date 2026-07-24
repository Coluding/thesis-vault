# Wan output adapter: additive delta → AVID mask_mix (prediction-blend)

Date: 2026-07-06
Config: `configs/diffusion_wan22_avid_i2v_metaworld.yaml`
Script: `scripts/train_wan22_i2v_metaworld_external.py`

## What changed

The Wan output adapter (`backbone: wan`, the 34M AVID-style delta on frozen
Wan2.2-TI2V-5B) composed as a **pure additive residual** despite the "AVID" name:

```
prediction = base + Δ          (composition="add", output_kind="delta")
```

It is now the **true AVID prediction-blend** with a learned per-pixel mask:

```
prediction = base·σ(gate+b) + pred·(1 − σ(gate+b))   (composition="mask_mix")
```

The adapter now emits a *standalone prediction* competing with the base plus a
1-channel spatial gate, instead of a small correction added on top.

## Implementation

- `ActionWanModel` (`backbones/wan/modules/action_model.py`): new
  `output_mask` flag. When set, adds a second zero-init `Head(dim, 1, ...)`
  gate head → returns `(prediction, gate)` with `gate` a `[B,1,T,H,W]`
  per-pixel mask logit. The **main head keeps its xavier init** (a real
  prediction, not a ~0 delta) — identity-at-init comes from the gate, not from
  a zeroed prediction head. `_unpatchify` generalised to take a channel count.
- `Wan21OutputAdapter` (`adapters/output/wan.py`): threads `output_mask`;
  returns `OutputAdapterResult(output_kind="prediction", gate=…)` when set,
  else the old delta path (backward compatible).
- `adapters/factory.py`: passes `output_mask` from `adapter.extra` (same key
  the DynamiCrafter/UNet adapters already use).
- Config: `adapter.composition: mask_mix`, `adapter.gate_bias: 4.0`,
  `adapter.extra.output_mask: true`.

No changes to `AdaptedModel` — the `mask_mix` branch and `gate_bias` plumbing
(`config.adapter.composition`/`gate_bias` → builder) already existed; only the
Wan head had to start producing a gate.

## Identity-at-init

Gate head is zero-init → gate logit 0 everywhere. With `gate_bias=4`,
σ(4)≈0.982, so the mask starts ≈1 (keep base). Measured leak at init:
mean|blend−base| ≈ 0.025 vs mean|base| ≈ 0.80 (~3%), from the 1.8% weight on
the random-init prediction. Raise `gate_bias` (~8 → σ≈0.9997) for tighter
identity if the early perturbation matters.

## Update — gated_residual A/B variant added

Added a second gated composition and made the head layout **derive from
`adapter.composition`** (single source of truth; the manual `extra.output_mask`
key is gone):

| composition | head | gate | output_kind | compose |
|---|---|---|---|---|
| `add` (legacy) | zero-init delta | — | delta | `base + Δ` |
| `gated_residual` | zero-init delta | yes | delta | `base + σ(gate)·Δ` |
| `mask_mix` | xavier prediction | yes | prediction | `base·σ(gate+b) + pred·(1-σ(gate+b))` |

`ActionWanModel` now takes `output_mask` (emit gate head) + `predict_full`
(main head is a real prediction vs a ~0 delta); the factory sets both from the
composition string. `gated_residual` needs **no gate_bias** — identity-at-init
is exact (Δ≈0 → measured leak 0.0000 vs base), whereas mask_mix leaks ~3.5% at
init from the random-init prediction weighted by 1−σ(4).

**A/B configs** (identical except `composition` + wandb project):
- `configs/diffusion_wan22_avid_i2v_metaworld.yaml` — mask_mix (replace)
- `configs/diffusion_wan22_gated_i2v_metaworld.yaml` — gated_residual (correct)

Semantics: mask_mix **replaces** the base where the mask opens (adapter's own
prediction takes over); gated_residual **corrects** the base (never discards it,
just gates how much residual to add — the thesis core rule f = base + g·Δ).

## Open questions / follow-ups

- **gate_bias sweep**: 4 vs 8. Higher = cleaner identity but less gradient to
  the prediction head early (grad scaled by 1−σ(gate)). Watch how fast the mask
  opens on predicted frames.
- **Gate granularity**: currently 1 channel broadcast over all 48 latent
  channels (spatial/temporal mask, AVID-faithful). Per-channel gate (out_dim
  channels) is a one-line change if we want channel-wise masking.
- **Prediction-head init/scale**: xavier prediction may not match the base
  velocity scale; it's mixed at ~1.8% at init so training absorbs it, but worth
  checking the prediction magnitude vs base once the mask opens.
- Compare mask_mix vs the old additive delta and vs `gated_residual`
  (`base + σ(gate)·Δ`) on the MetaWorld corner2 held-out loss + FID/FVD.
