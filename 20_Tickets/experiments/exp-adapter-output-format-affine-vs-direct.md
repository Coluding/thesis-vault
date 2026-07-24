---
type: exp
scope: adapter
status: open
priority: medium
created: 2026-05-29
updated: 2026-06-04
resolution:
resolution_note:
closed_at:
related:
  - "[[../../50_Decisions/open/output-format-affine-vs-direct]]"
  - "[[../../30_Knowledge/theory/unicon-output-adapters-detached-backward]]"
---

# exp: Output-format ablation — affine (scale+shift) vs direct delta

## Goal

Measure whether a **cheap per-channel affine `(scale, shift)`** modulation of
the frozen base beats predicting a **free, full-resolution residual delta**,
holding the backbone fixed. This is a difference in **inductive bias**, not
capacity. Resolves [[../../50_Decisions/open/output-format-affine-vs-direct]].

## Setup

> **CONFIG CONSOLIDATION 2026-06-04 — the comparison moved into the shortcut
> output setting and is now CONFOUNDED.** The standalone transformer-backbone
> arms (`diffusion_output_v2_{affine,direct}_metaworld.yaml`) were **deleted**
> in the config cleanup to four-then-five focused experiments. The affine idea
> now lives in the AVID/unet shortcut output adapter. The surviving comparison
> is:
>
> - **direct arm** = `configs/diffusion_avid_shortcut_metaworld.yaml` — AVID/unet
>   output adapter, `composition: avid_mask_mix` (full prediction + per-pixel gate).
> - **affine arm** = `configs/diffusion_avid_shortcut_affine_metaworld.yaml` —
>   same AVID/unet adapter + shortcut, but `output_format: affine` (per-channel
>   scale/shift) and `composition: add` (affine returns a delta; mask_mix needs a
>   full prediction, so it cannot be used).
>
> **This is no longer a clean single-axis test.** The two arms differ on *two*
> axes — output format (affine vs direct) AND composition (`add` vs
> `avid_mask_mix`) — and both now run *under shortcut*. So "affine vs direct" is
> confounded with "add vs mask_mix". Decision [a] (2026-06-04) accepted this:
> read #4-vs-#1 as the practical affine-under-shortcut comparison, not the pure
> format ablation the transformer arms would have given. If a clean single-axis
> result is later needed, add a `direct`+`add` shortcut sibling (a 6th config).

## Grid to sweep

- **Headline (confounded):** affine+`add` vs direct+`avid_mask_mix`, both AVID/unet, under shortcut.
- Backbone capacity sweep (mlp/transformer) is dropped — those configs were removed.

> **Affine is channel-wise only — there is no `affine_granularity` axis.** A
> dense per-element affine subsumes the direct delta (its full-rank `shift` *is*
> a direct delta), so "dense affine vs direct" was degenerate. The knob was
> removed from the code; the factory rejects `affine_granularity` ≠ `channel`.
> See [[../../30_Knowledge/tech/affine-output-granularity]].

## Metrics

Validation MSE on held-out MetaWorld rollouts; training-loss trajectory (does
affine converge faster / more stably early?); a few qualitative rollout videos
per arm. Same seed, same step budget across arms.

## Done when

Both arms have logged runs (wandb id + ckpt + commit) at equal budget, the MSE
gap (if any) is reported with its sign and rough magnitude, and the decision
note is moved to `decided/` with the chosen default. **No numbers recorded
until the runs actually execute** (hard rule 8).

## Notes

- Affine + `output_mask`/`mask_mix` is intentionally disallowed (affine returns
  a delta, mask-mix needs a full prediction) — the affine arm uses `add`. This
  is exactly the source of the confound noted above.
- Framework code verified at commit 57244cc; `tests/test_output_format_heads.py`
  green (20/20 after the channel-only change).
- The expected story: per-channel affine should win when the base's errors are a
  global per-channel miscalibration and lose when they are spatially-structured
  corrections only a free dense delta can express. The ablation tells us which
  regime MetaWorld is in — bearing in mind the composition confound.
