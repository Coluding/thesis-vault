---
type: exp
scope: adapter
status: open
priority: medium
created: 2026-05-29
updated: 2026-05-29
resolution:
resolution_note:
closed_at:
related:
  - "[[../50_Decisions/open/output-format-affine-vs-direct]]"
  - "[[../30_Knowledge/theory/unicon-output-adapters-detached-backward]]"
---

# exp: Output-format ablation — affine (scale+shift) vs direct delta

## Goal

Measure whether the affine `(scale, shift)` output format beats predicting the
residual delta directly, holding the backbone fixed. Resolves
[[../50_Decisions/open/output-format-affine-vs-direct]].

## Setup (built, ready to run)

- Configs: `configs/diffusion_output_v2_affine_metaworld.yaml` and
  `configs/diffusion_output_v2_direct_metaworld.yaml` — identical except
  `adapter.extra.output_format` (`affine` vs `direct`). Transformer backbone,
  DynamiCrafter frozen base, MetaWorld, velocity prediction.
- Both zero-init at identity, `condition_on_base_outputs: true`.

## Grid to sweep

- `output_format`: affine | direct  (the headline axis)
- `backbone`: mlp | transformer | unet  (does the answer depend on capacity?)
- `affine_granularity`: dense | channel  (affine arm only)

Start with `backbone: transformer`, `affine_granularity: dense` (the
capacity-matched, fairest comparison), then add the mlp and unet points.

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
  a delta, mask-mix needs a full prediction) — keep composition `add` for this
  ablation.
- Framework code verified at commit 57244cc; `tests/test_output_format_heads.py`
  green.
