---
type: exp
scope: backbone
status: in-progress
priority: medium
created: 2026-07-25
updated: 2026-07-25
resolution:
resolution_note:
closed_at:
related: ["[[../../30_Knowledge/writing/ablation-axes]]", "[[exp-backbone-skyreels-robotarm-run]]", "[[exp-adapter-wan-cap50-warmup-pushblock-run]]"]
---

# exp: SkyReels-V2-1.3B · Push Cube (matrix run 2 — weak flow base, flat domain)

## Hypothesis

Weak-flow base (SkyReels-1.3B, 16-ch) on the flat 2D domain. Completes the
model × dataset grid, but **read with a caveat**: the SkyReels probe (2026-07-25)
showed the frozen base **style-drifts** on flat Push Cube — it re-renders the flat
vector shapes as glossy 3D blocks (its natural-video prior is OOD for flat art).
So on this cell both the base's residual *and* any adapter gain are confounded by
the style mismatch, unlike the clean Robot-Arm cell.

## Procedure

- Base SkyReels-V2-I2V-1.3B-540P; adapter feature_dim 16, mask_mix + cross-attn +
  base-input. Data ACWM push_block, input_dim 2.

## Decision rule

- Primarily a **grid-completeness** cell. Compare its base residual + adapter
  behaviour to the Wan Push-Cube runs; expect the flat-visuals ceiling
  (base near-perfect) to dominate regardless of base strength.

## Build status

**Training path CODE-COMPLETE + smoke-green (2026-07-25); GPU validation
pending** — config `configs/skyreels/diffusion_skyreels_xattn_acwm_pushblock.yaml`,
sbatch `.../skyreels/submit_train_skyreels_pushblock.sh`. Shares the SkyReels
wrapper + preprocessor + entrypoint with [[exp-backbone-skyreels-robotarm-run]];
the 8 GPU-validation items are listed there. Note the flat-2D style-drift
confound still applies to this cell.

## Notes

Report this cell together with the style-drift probe finding so the confound is
explicit (hard rule 8: no unqualified numbers). Axis 5 /
[[../../30_Knowledge/writing/ablation-axes]].
