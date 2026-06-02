---
type: decision
status: open
created: 2026-05-29
decided_at:
updated: 2026-05-29
target_date:
scope: architecture
related:
  - "[[../../30_Knowledge/theory/unicon-output-adapters-detached-backward]]"
  - "[[../../30_Knowledge/theory/interior-vs-output-adapters-backward-cost]]"
  - "[[../../30_Knowledge/tech/affine-output-granularity]]"
  - "[[../../20_Tickets/exp-adapter-output-format-affine-vs-direct]]"
  - "[[../../10_now/architecture]]"
---

# Decision: Output parameterisation — affine (scale+shift) vs direct delta

## Status

**Open — captured 2026-05-29.** Framework support is built (`output_v2`
family); the empirical question is unresolved and is what this decision
tracks. Nothing here is a result yet — see the ticket for the run.

## Context

The original `AffineOutputAdapter` is a super-lightweight head: a context
`MLP(timestep ⊕ cond)` produces a per-channel `(scale, shift)` applied to the
base prediction, returned as a delta so `AdaptedModel` composes
`base*(1+scale)+shift`. The observation that triggered this: the affine
*format* is interesting independent of the head's capacity. The
`DynamicCrafterOutputAdapter` we actually train with emits a **direct** delta
from a full UNet. So two design choices were silently entangled — the output
**format** (affine vs direct) and the backbone **capacity** (tiny head vs
full UNet).

## Question

Holding the backbone fixed, does the **affine `(scale, shift)` output format**
predict the action-conditioned dynamics better than emitting the **residual
delta directly**? And does the answer depend on backbone capacity
(mlp / transformer / unet) or on affine granularity (dense vs per-channel)?

## What was built (to enable the comparison, not to pre-judge it)

- `adapters/output/format.py` — shared format math. `direct` → delta is the
  raw `C`-channel output; `affine` → raw `2C` split into `(scale, shift)`,
  `delta = base*scale + shift`. Both return `output_kind="delta"` so the
  comparison is residual-vs-residual (fair). `affine_granularity ∈ {dense,
  channel}`.
- `adapters/output/output_head.py` — `OutputHeadAdapter` with `mlp` and
  `transformer` (DiT-style, patchified latents) backbones.
- `DynamicCrafterOutputAdapter` — gained `output_format`/`affine_granularity`
  (the `unet` backbone; `affine` doubles UNet `out_channels` to `2C`).
- Factory `output_v2` architecture; configs
  `diffusion_output_v2_{affine,direct}_metaworld.yaml`.
- All backbones zero-init the final projection → identity residual at init.

Verified against commit 57244cc (tests in `tests/test_output_format_heads.py`,
16/16 pass).

## Options on the table

1. **Direct delta is the default** (current `DynamicCrafterOutputAdapter`
   behaviour). Simple; the adapter predicts the residual outright.
2. **Affine `(scale, shift)` is the default.** Hypothesis: modulating the base
   prediction is an easier learning target than reconstructing the full delta,
   especially early in training (identity-at-init is exact, and the base
   already carries most of the signal).
3. **Format is task/backbone-dependent** — keep both, report the ablation,
   pick per-deliverable.

## Hypotheses to test (analysed estimates, not results)

- Affine may help most with **small backbones** (mlp): a per-channel gain on a
  strong base beats asking a tiny head to synthesise the delta from scratch.
- Affine may help **early-training stability** because `scale,shift → 0`
  is the exact identity, whereas a direct head must learn to output ~0.
- Dense affine ≈ direct in capacity, so any gap there is genuinely the format;
  per-channel affine is a much smaller hypothesis class and may underfit on
  spatially-structured corrections.

These are reasoning, not measurements — resolve via the ticket.

## Decision

_Pending the ablation run._ Promote to `decided/` once
[[../../20_Tickets/exp-adapter-output-format-affine-vs-direct]] produces
sourced numbers (wandb id + ckpt + commit).

## Consequences

_To fill on decision._ Touches the D2 headline-adapter choice (see the open
question in [[../../10_now/architecture]] about which output family is the
default for D2's ablation).
