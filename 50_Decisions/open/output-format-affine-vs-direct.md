---
type: decision
status: open
created: 2026-05-29
decided_at:
updated: 2026-06-03
target_date:
scope: architecture
related:
  - "[[../../30_Knowledge/theory/unicon-output-adapters-detached-backward]]"
  - "[[../../30_Knowledge/theory/interior-vs-output-adapters-backward-cost]]"
  - "[[../../30_Knowledge/tech/affine-output-granularity]]"
  - "[[../../20_Tickets/experiments/exp-adapter-output-format-affine-vs-direct]]"
  - "[[../../10_now/architecture]]"
---

# Decision: Output parameterisation — affine (scale+shift) vs direct delta

## Status

**Open — captured 2026-05-29; design refined 2026-06-03.** Framework support
is built (`output_v2` family); the empirical question is unresolved and is what
this decision tracks. Nothing here is a result yet — see the ticket for the run.

**Design refinement (2026-06-03): affine is now channel-wise only.** The
`affine_granularity` knob was removed from the code. Reasoning below (and in
[[../../30_Knowledge/tech/affine-output-granularity]]): a *dense* per-element
`(scale, shift)` makes `shift` full-rank, so `base + shift` can already
represent any residual — dense affine is just `direct` plus an
expressivity-redundant multiplicative term, which **collapses the format
distinction**. The "capacity-matched dense affine vs direct" framing was
therefore degenerate. The meaningful comparison is a difference in **inductive
bias**, not capacity:

- **direct** — a free, full-resolution residual delta.
- **affine (channel)** — a cheap, structured per-channel FiLM on the frozen
  base; cannot collapse into a free residual.

Same trunk in both arms; only the readout structure differs.

**Config consolidation (2026-06-04, decision [a]).** In the cleanup to a focused
config set, the standalone transformer-backbone arms
(`diffusion_output_v2_{affine,direct}_metaworld.yaml`) were **deleted**. The
affine idea moved into the AVID/unet **shortcut** output adapter, so the live
comparison is now:

- **direct** = `diffusion_avid_shortcut_metaworld.yaml` (AVID/unet, `avid_mask_mix`).
- **affine** = `diffusion_avid_shortcut_affine_metaworld.yaml` (AVID/unet, `affine` + `add`).

This is **no longer the clean single-axis test** described above — the two arms
differ in output format *and* composition (`add` vs `avid_mask_mix`), both under
shortcut. We accepted this (option [a]): treat it as the practical
affine-under-shortcut comparison and report the confound. A clean single-axis
run would need an added `direct`+`add` shortcut sibling. See the ticket
[[../../20_Tickets/experiments/exp-adapter-output-format-affine-vs-direct]].

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

Holding the backbone fixed, does the **per-channel affine `(scale, shift)`
output format** predict the action-conditioned dynamics better than emitting the
**residual delta directly**? And does the answer depend on backbone capacity
(mlp / transformer / unet)? (The affine-granularity sub-question is closed:
affine is channel-only — see Status.)

## What was built (to enable the comparison, not to pre-judge it)

- `adapters/output/format.py` — shared format math. `direct` → delta is the
  raw `C`-channel output; `affine` → raw `2C` split into `(scale, shift)`,
  each **pooled to one value per channel**, `delta = base*scale + shift`. Both
  return `output_kind="delta"` so the comparison is residual-vs-residual.
- `adapters/output/output_head.py` — `OutputHeadAdapter` with `mlp` and
  `transformer` (DiT-style, patchified latents) backbones.
- `DynamicCrafterOutputAdapter` — gained `output_format` (the `unet` backbone;
  `affine` doubles UNet `out_channels` to `2C`).
- Factory `output_v2` architecture (still reachable as a backbone via
  `OutputHeadAdapter`). The factory raises if a config requests
  `affine_granularity` other than `channel`.
- All backbones zero-init the final projection → identity residual at init.

Verified against commit 57244cc (tests in `tests/test_output_format_heads.py`).
**Update 2026-06-03:** `affine_granularity` removed; affine is channel-only;
tests updated (20/20 pass).
**Update 2026-06-04:** the `output_v2_{affine,direct}` configs were deleted in
the config cleanup; the affine arm now lives in
`diffusion_avid_shortcut_affine_metaworld.yaml` (AVID/unet, affine + `add`,
under shortcut) against `diffusion_avid_shortcut_metaworld.yaml` (direct,
`avid_mask_mix`). The transformer/mlp `OutputHeadAdapter` backbones remain in
code but have no live config. See the Status note on the resulting confound.

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
- Per-channel affine is a **much smaller hypothesis class** than a dense delta:
  it can only rescale/shift each base channel globally, so it should win when
  the base's errors are a per-channel miscalibration and lose when they are
  **spatially-structured** corrections that a free dense delta can express and a
  per-channel FiLM cannot. The ablation measures which regime MetaWorld is in.

These are reasoning, not measurements — resolve via the ticket.

## Decision

_Pending the ablation run._ Promote to `decided/` once
[[../../20_Tickets/experiments/exp-adapter-output-format-affine-vs-direct]] produces
sourced numbers (wandb id + ckpt + commit).

## Consequences

_To fill on decision._ Touches the D2 headline-adapter choice (see the open
question in [[../../10_now/architecture]] about which output family is the
default for D2's ablation).
