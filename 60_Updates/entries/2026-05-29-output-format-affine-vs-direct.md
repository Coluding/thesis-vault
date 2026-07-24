---
date: 2026-05-29
category: added
deliverable: D1
meeting:
sources:
  - "[[../../50_Decisions/open/output-format-affine-vs-direct]]"
  - "[[../../20_Tickets/experiments/exp-adapter-output-format-affine-vs-direct]]"
  - "[[../../30_Knowledge/theory/unicon-output-adapters-detached-backward]]"
  - "code: src/generative_flow_adapters/adapters/output/format.py"
  - "code: src/generative_flow_adapters/adapters/output/output_head.py"
  - "code: src/generative_flow_adapters/adapters/output/dynamicrafter.py"
  - "code: src/generative_flow_adapters/adapters/factory.py"
  - "code: tests/test_output_format_heads.py"
  - "config: configs/diffusion_output_v2_affine_metaworld.yaml"
  - "config: configs/diffusion_output_v2_direct_metaworld.yaml"
---

# New output-format axis: affine (scale+shift) vs direct delta

## What
Added an `output_v2` output-adapter family that makes the **output
parameterisation** an explicit, ablatable axis, orthogonal to backbone
capacity. Two independent knobs:

- `output_format`: `affine` — the backbone emits `2C` channels split into
  `(scale, shift)`, returned as `delta = base*scale + shift` (realised as
  `base*(1+scale)+shift` under `add`); `direct` — the backbone emits `C`
  channels = the delta directly. **Both return `output_kind="delta"`**, so the
  two arms are residual-vs-residual and the comparison is apples-to-apples.
- `backbone`: `mlp` (lightweight FiLM ≈ the original `affine` adapter),
  `transformer` (DiT-style over patchified video latents), or `unet` (routed
  to the existing `DynamicCrafterOutputAdapter`, which also gained the format
  knob).
- `affine_granularity`: `dense` (per-element scale/shift maps, capacity-matched
  to a dense delta) vs `channel` (pooled per-channel FiLM).

Shared format math lives in `adapters/output/format.py`; the new backbones in
`adapters/output/output_head.py`. All backbones zero-init their final
projection → identity residual at init. Matched configs
`diffusion_output_v2_{affine,direct}_metaworld.yaml` differ only in
`output_format`.

**Gating made a first-class, recipe-agnostic mixing concern.** Separated the
gate (composition layer) from how the adapter forms its output, and added a
second gated blend so both options exist:
`gated_residual` (`base + σ(gate)·Δ`, treats Δ as a *contribution* — the thesis
core `f = base + g·Δ`, identity at init for free) alongside the existing AVID
`mask_mix` (`base·σ(gate) + Δ·(1−σ(gate))`, treats Δ as a *standalone
prediction*). The `output_v2` heads emit the gate via
`gate_kind ∈ {none, channel, dense}`. This also dissolved the earlier
`affine`+`mask_mix` restriction at the principled level (an output adapter just
emits a full-size contribution; base-combination + gating belong to mixing).
See [[../../30_Knowledge/tech/mask-mix-gate]].

38/38 tests pass (`tests/test_output_format_heads.py`), existing `mask_mix`
adapters (unicon composition tests) unaffected.

## Why it matters
- The original `AffineOutputAdapter` entangled two choices — the affine
  *format* and the *tiny-head* capacity. `output_v2` separates them, so D1 can
  report the output family as a `backbone × format` grid instead of a handful
  of one-off classes.
- It sets up a concrete D2-relevant question: is modulating the frozen base
  prediction (`scale, shift`) an easier learning target than reconstructing the
  residual delta outright? Affine is exactly the identity at init (`scale,
  shift → 0`), which may help early-training stability — to be tested, not
  claimed.
- Reinforces the framework story (see the theory note): this is a new *size
  point* on the same detached-output family — heavier than affine, lighter than
  the full-UNet DynamiCrafter adapter — and like all of them keeps the frozen
  base out of the gradient graph.

## Evidence / sources
- Code verified at commit 57244cc (working tree). Factory builds all
  `backbone × format` combos; latent-channel resolution fixed so
  `model.feature_dim` (default 64) no longer shadows the 4-channel video latent.
- `tests/test_output_format_heads.py` — format math, both backbones × both
  formats, identity-at-init, channel-count doubling, a backward step, error
  cases. 16/16 green.
- _No experiment metrics yet — this is the framework capability + matched
  configs; the affine-vs-direct numbers come from the ticket run._

## Next
- Run [[../../20_Tickets/experiments/exp-adapter-output-format-affine-vs-direct]]:
  transformer backbone, `dense` affine first (fairest), then add the mlp and
  unet points. Equal seed/budget per arm.
- Resolve [[../../50_Decisions/open/output-format-affine-vs-direct]] once the
  runs produce sourced numbers; feeds the open "default output family for D2"
  question in [[../../10_now/architecture]].
- Housekeeping: removed a broken half-typed `ComplexAffineOutputAdapter` stub
  from `affine.py` that was breaking imports; the new `OutputHeadAdapter`
  supersedes it (rename available if preferred).
