---
date: 2026-06-01
category: finding
deliverable: D1
meeting:
sources:
  - "[[../../30_Knowledge/tech/mask-mix-gate]]"
  - "[[../../30_Knowledge/tech/affine-output-granularity]]"
  - "[[../../50_Decisions/open/avid-adapter-init]]"
---

# The composition interface has two gated blends, and a step-0 init subtlety that isn't base pass-through

## What

Audited how the shared composition rule is actually realised across adapter
families. Two findings worth surfacing:

1. **Two distinct gated blends, by what `Δ` *means*.** A new `gated_residual`
   composition (`base + σ(gate)·Δ`, added 2026-05-29) treats `Δ` as a
   *contribution* — this is exactly the thesis core rule `f = base + g·Δ`, and
   it is **identity-at-init for free** (Δ→0 ⇒ output=base, any gate). The older
   `mask_mix` (`base·σ(gate) + Δ·(1−σ(gate))`) treats `Δ` as a *standalone
   prediction competing* with the base, and therefore needs a `gate_bias` to be
   identity at init. The gate is a mixing-layer concern, agnostic to how `Δ` was
   formed (direct / affine / full UNet).

2. **`mask_mix` does NOT give base pass-through at step 0.** Both AVID and
   HyperAlign zero-init the gate head, so `gate = 0` ⇒ `σ(0) = 0.5` — a **50/50
   mix of base with a random adapter output**, not the base. The asymmetry (zero
   gate, *random* `y`) is deliberate: it keeps the sigmoid at its max-derivative
   point and keeps gradient flowing into the UNet body at step 0.

## Why it matters

For D1 framing this is the cleanest possible story: the framework's core
composition rule `f = base + g·Δ` now has a direct, identity-at-init
realisation (`gated_residual`), and we can state precisely how the AVID/
HyperAlign `mask_mix` path differs from it. The step-0 subtlety also means
"adapter starts as a no-op" is **false for `mask_mix`** — a trap for anyone
reading early-training curves.

## Evidence / sources

- Composition code and per-family gate paths catalogued in
  [[../../30_Knowledge/tech/mask-mix-gate]] (commit `7680e82` + uncommitted
  trainer edits), with code anchors in `models/adapted_model.py:74-96` and the
  AVID/HyperAlign gate heads.
- The gate's `channel | spatial` granularity is the **exact analogue** of the
  affine output format's `channel | dense` knob — see
  [[../../30_Knowledge/tech/affine-output-granularity]] (already logged as the
  2026-05-29 affine-vs-direct entry).
- Live configs: AVID uses `avid_mask_mix` (gate active); HyperAlign currently
  uses `add` (gate heads dormant). So gate dynamics observed on AVID do **not**
  automatically transfer to HyperAlign.

## Next

The AVID step-0 init choice is an open decision
([[../../50_Decisions/open/avid-adapter-init]]). If HyperAlign is later run with
`mask_mix`, a sibling decision is needed (its `y` is a LoRA-perturbed frozen
base, structurally cleaner than AVID's random UNet output, so the case for
changing init is weaker there).
