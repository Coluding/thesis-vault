---
type: experiment
date: 2026-07-25
config: configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_robotarm.yaml (robot_arm); acwm_pushblock (push_cube, prior)
commit: uncommitted working tree @ 2026-07-25 (local diagnostic, not a training run)
wandb_run_id: none — local eval via scripts/generate_wan22_i2v_compare.py --random-init
ckpt_path: n/a (random-init adapter; frozen Wan2.2-TI2V-5B base)
status: completed
deliverable: D2
metrics:
  robotarm_base_masked_denoise_mse: 0.314
  robotarm_adapted_random_adapter: 1.164
  robotarm_adapter_rel_contribution: 0.743   # |pred-base|/|base|, random adapter (reference only)
  pushcube_base_masked_denoise_mse: 0.036    # PRIOR measurement — matched-geometry re-measure PENDING
notes: "Frozen-base residual is ~8.7x larger on Robot Arm than Push Cube -> base-parity collapse is a flat-visuals (near-zero-residual) problem, not base strength. Diagnostic, NOT a trained run."
---

# ACWM base-residual diagnostic — Robot Arm 0.314 vs Push Cube 0.036 (≈8.7×)

> **This is a LOCAL DIAGNOSTIC, not a wandb training run.** It measures the
> *frozen base's* denoise error (with a random-init adapter, so the adapter
> contributes nothing) to quantify how much residual the action-conditioned
> adapter has to learn on each domain. Cite it as a diagnostic, not a result
> table row.

## Method

`scripts/generate_wan22_i2v_compare.py --random-init` on a frozen
Wan2.2-TI2V-5B base, ACWM `robot_arm/ind_train`, **17-frame window**, masked
denoise MSE, `--num-windows 2 --loss-batches 2`, local RTX 3090, 2026-07-25.
The random-init adapter perturbs only adapter-only params (base left intact).

## Result

| domain | frozen-base masked denoise MSE | note |
|---|---|---|
| **Robot Arm** | **0.314** | this run (matched 17f/masked, random adapter) |
| Push Cube | ~0.036 | **prior** measurement; matched re-measure at identical 17f/masked geometry still **pending** |

Random-adapter reference: adapted loss 1.164, `adapter_rel_contribution` 0.743
(a random adapter is far from cloning — the trained-clone signature is the
opposite, `rel_contribution → 0`).

## Interpretation

On Push Cube the frozen base is near-perfect → almost no residual → the adapter
clones it (base-parity collapse, [[20260724-metaworld-cap-shift-triangle-base-parity]]).
On Robot Arm the base leaves ~8.7× more error for the action-conditioned adapter
to close → a genuine learning signal. Motivates the base × dataset matrix
([[ablation-axes]]) and the dataset move.

## Open

- **Matched Push-Cube re-measure** at the identical 17f/masked setup (the 0.036
  predates this protocol) — closes the strict apples-to-apples gap.
- These are base-residual numbers; the *trained*-adapter results per cell come
  from the matrix runs (Wan/SkyReels/DC × Push Cube/Robot Arm).
