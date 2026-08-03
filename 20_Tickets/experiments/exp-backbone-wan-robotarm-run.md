---
type: exp
scope: backbone
status: open
priority: high
created: 2026-07-25
updated: 2026-08-01
resolution:
resolution_note:
closed_at:
related: ["[[../../30_Knowledge/writing/ablation-axes]]", "[[exp-adapter-wan-cap50-warmup-pushblock-run]]", "[[exp-backbone-skyreels-robotarm-run]]", "[[exp-backbone-dc-robotarm-run]]"]
---

# exp: Wan2.2 · Robot Arm (matrix run 1 — clean baseline on the new arena)

## Hypothesis

Moving off flat Push Cube to the visually-rich **ACWM Robot Arm** gives the
frozen base a real residual to leave for the action-conditioned adapter, so the
adapter stops cloning the base. Motivation is measured, not assumed: at matched
17-frame / masked-MSE geometry with a random adapter (2026-07-25 loss probe),
frozen-base denoise loss was **0.314 on Robot Arm vs ~0.036 on Push Cube (≈8.7×
more residual)**. On Push Cube the base is near-perfect → nothing to learn → the
mask_mix gate saturates to a clone (the MetaWorld/Push-Cube base-parity failure).

## Procedure

- Config: `configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_robotarm.yaml`
  (provider wan2.2, 48-ch, temporal_length 97, max_area 589824, input_dim 7,
  cross-attention action injection, mask_mix + base-input, gate_cap 0.9,
  sigma_shift 5.0, pretrain_steps 0).
- Prereq (cluster): download + precompute robot_arm latents —
  `jobs/experiments_cluster/infra/download_acwmphys_robotarm.sh` then
  `submit_precompute_acwmphys_robotarm.sh`. T5 contexts already computed
  (`configs/prompts/acwm_robotarm.contexts.pt`, validated prompt).

## Decision rule

- **Adapted denoise loss descends well below 0.314 + pred-base cosine drops**
  ⇒ the residual is real and the adapter is using actions → dataset move works;
  this becomes the D2 Robot-Arm baseline.
- **Adapter stalls near base (pred-base cosine ~0.85, gate saturates)** ⇒
  base-parity is not purely a flat-visuals artifact → escalate to the Run-6
  interventions (cap 0.5 + warmup) on this arena too.

## Build status

Config + precompute scripts ready. **Blocked only on cluster precompute.**

## Notes

Base-strength axis partner runs: SkyReels-1.3B robot_arm
([[exp-backbone-skyreels-robotarm-run]]) and DynamiCrafter robot_arm
([[exp-backbone-dc-robotarm-run]]). SkyReels probe (2026-07-25) confirmed the
frozen weak base holds the Robot-Arm domain (coherent, real articulated motion)
once the prompt is de-branded.

## Cleanup 2026-08-01 — **DELIVERED**

Run as `ncztxyyo` (D2 matrix, effect_rel 0.0056) — see [[../../30_Knowledge/experiments/20260728-acwm-robotarm-matrix-action-blind]]. Mechanism fully explained by the 07-31 campaign.

*Proposed for close; awaiting confirmation (CLAUDE.md: never close without it).*
