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
related: ["[[../../30_Knowledge/writing/ablation-axes]]", "[[exp-backbone-wan-robotarm-run]]", "[[exp-backbone-skyreels-robotarm-run]]", "[[exp-backbone-dc-pushblock-run]]"]
---

# exp: DynamiCrafter · Robot Arm (matrix run 4 — weak diffusion base)

## Hypothesis

Base-strength axis, diffusion side: **DynamiCrafter** (weak diffusion base,
4-ch SD-VAE, v-prediction, zero-SNR) as the third base family alongside Wan-5B
(strong flow) and SkyReels-1.3B (weak flow). We know DynamiCrafter works as a
base from the MetaWorld campaign; this ports it to ACWM Robot Arm to see whether
a diffusion base leaves a usable residual for the action adapter on the rich
domain.

## Procedure

- Provider **`dynamicrafter_video`** (the BaseVideoModel-conforming path — the
  old `dynamicrafter` UNet wrapper was non-conforming; this is the migration).
- Config off template `configs/dynamicrafter/diffusion_avid_shortcut_metaworld_native.yaml`;
  adapter feature_dim 4, mask_mix + cross-attn + base-input. Data ACWM robot_arm,
  input_dim 7. DC encodes latents **live** via its SD-VAE (no Wan-style
  precompute).

## Decision rule

- **Adapter learns on the DC base** ⇒ the framework generalises across base
  families (D1) and gives a diffusion-side D2 datapoint.
- **DC base incoherent on ACWM Robot Arm** (domain OOD for DC's pretrain) ⇒
  note it as a base-coverage limit, not an adapter failure.

## Build status

**Build done (2026-07-25), drafted + smoke-tested; ready to launch.**
- Config: `configs/dynamicrafter/diffusion_dc_acwm_robotarm.yaml` (provider
  `dynamicrafter_video`, 4-ch, `avid_mask_mix` + structured `act` conditioning,
  da=7). Loads clean.
- Data path: `scripts/train_avid_shortcut_metaworld.py` now takes
  `--dataset acwm_phys --data-dir` → `build_acwmphys_clip_dataset` (same clip
  dict as MetaWorld, so live SD-VAE encode downstream is unchanged).
- Checkpoint present: `ckts/dynami512.ckpt` (10.4 GB).
- Launch: `python scripts/train_avid_shortcut_metaworld.py --config
  configs/dynamicrafter/diffusion_dc_acwm_robotarm.yaml --dataset acwm_phys
  --data-dir ds/acwm-phys/kinematics/robot_arm/ind_train --frame-stride 1
  --target-height 512 --target-width 512`.
- **Pending validation before trusting results:** (1) one real training step to
  confirm SD-VAE encode + concat shapes at the chosen geometry **and that the
  seam diagnostics now emit for DC** (see below); (2) a DC base-coherence probe
  on robot_arm (DC512 is a real-world I2V prior → arm should sit close to it,
  unlike flat push_block).
- **Seam logging fix (2026-07-26):** the shared `Trainer` only requested the
  frozen-base prediction in its *flow* branch, so DC (diffusion branch) logged
  none of the base-parity diagnostics. Fixed `training/trainer.py` diffusion
  branch to request `return_base=True` + compute the base-only diffusion loss,
  so DC now logs the SAME seam metrics as Wan/SkyReels: `adapter_gate_mean/std`,
  `adapter_base_cosine`, `adapter_pred_base_cosine`, `adapter_rel_contribution`,
  `denoise_base_only`, `denoise_adapter_delta`, `adapter_grad_norm`. Compiles +
  mirrors the flow branch; confirm the keys appear in wandb on the first step.

## Notes

"Migration to BaseVideoModel" = use the existing `dynamicrafter_video` provider,
not a new model class. Base-family peers: [[exp-backbone-wan-robotarm-run]],
[[exp-backbone-skyreels-robotarm-run]]. Axis 5 /
[[../../30_Knowledge/writing/ablation-axes]].
