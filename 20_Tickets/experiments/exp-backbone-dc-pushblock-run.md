---
type: exp
scope: backbone
status: open
priority: medium
created: 2026-07-25
updated: 2026-07-25
resolution:
resolution_note:
closed_at:
related: ["[[../../30_Knowledge/writing/ablation-axes]]", "[[exp-backbone-dc-robotarm-run]]", "[[exp-backbone-skyreels-pushblock-run]]"]
---

# exp: DynamiCrafter · Push Cube (matrix run 5 — weak diffusion base, flat domain)

## Hypothesis

DynamiCrafter (weak diffusion base, 4-ch) on flat 2D Push Cube — the diffusion
counterpart of the SkyReels Push-Cube cell, completing the model × dataset grid.
User adjusted the DynamiCrafter training run (the old one did not use the
BaseVideoModel interface); this is the re-run on the conforming path.

## Procedure

- Provider `dynamicrafter_video`; adapter feature_dim 4, mask_mix + cross-attn +
  base-input. Data ACWM push_block, input_dim 2. Live SD-VAE encode.

## Decision rule

- Grid-completeness + diffusion-vs-flow contrast on the flat domain. Expect the
  same flat-visuals ceiling (base residual small) that drives cloning — compare
  to the Wan and SkyReels Push-Cube cells.

## Build status

**Build done (2026-07-25), drafted + smoke-tested; ready to launch.**
- Config: `configs/dynamicrafter/diffusion_dc_acwm_pushblock.yaml` (provider
  `dynamicrafter_video`, 4-ch, `avid_mask_mix` + structured `act`, da=2). Loads clean.
- Shared data-path change in `scripts/train_avid_shortcut_metaworld.py`
  (`--dataset acwm_phys`); see [[exp-backbone-dc-robotarm-run]].
- Launch: same script, `--config configs/dynamicrafter/diffusion_dc_acwm_pushblock.yaml
  --data-dir ds/acwm-phys/rigid_dynamics/push_block/ind_train --target-height 512
  --target-width 512`.
- **Caveat:** DC512's real-world prior is likely OOD for flat 2D push_block — run
  a base-coherence probe first; expect the flat-visuals ceiling.

## Notes

Axis 5 / [[../../30_Knowledge/writing/ablation-axes]]. Report with the
flat-visuals residual caveat (Push Cube base loss ~0.036).
