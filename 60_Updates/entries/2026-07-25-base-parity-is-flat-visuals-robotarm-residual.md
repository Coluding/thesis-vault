---
date: 2026-07-25
category: finding
deliverable: D2
meeting:
sources: ["[[ablation-axes]]", "[[20260724-metaworld-cap-shift-triangle-base-parity]]", "[[exp-backbone-wan-robotarm-run]]", "[[exp-adapter-wan-cap50-warmup-pushblock-run]]", "[[exp-backbone-skyreels-robotarm-run]]", "[[exp-backbone-dc-robotarm-run]]"]
---

# Base-parity collapse is a flat-visuals problem, not a base-strength problem

## What

The adapter's "clone the frozen base instead of using actions" failure is driven
by the **flat visuals of Push Cube**, not by the Wan base being too strong.
Measured 2026-07-25: frozen-base masked denoise loss = **0.314 on ACWM Robot Arm
vs ~0.036 on Push Cube** — ~8.7× more residual on the visually-rich domain. On
Push Cube the base is near-perfect, so there is almost nothing for the adapter to
learn and the `mask_mix` gate saturates to a copy; on Robot Arm the base leaves a
real error for the action-conditioned adapter to close.

## Why it matters

This reframes the whole D2 base-parity story that dominated the MetaWorld/Push-Cube
campaign (see [[20260724-metaworld-cap-shift-triangle-base-parity]]): the fix is a
**dataset with a real residual**, not just gate/loss interventions. It justifies
the move to Robot Arm and sets up a clean **base × dataset ablation** (D2).

Corollary from three SkyReels-V2-1.3B zero-adapter probes: the weak flow base is
**coherent** and, once the prompt is de-branded, **holds the Robot Arm domain**
(correct neutral palette, real articulated motion). On flat Push Cube it instead
**style-drifts** (flat vector shapes → glossy 3D blocks). So SkyReels is a viable
*weak* base, and the base-strength axis belongs on Robot Arm where both priors
match the domain — not on flat art where both are OOD.

## Evidence / sources

- **0.314** — local diagnostic, `scripts/generate_wan22_i2v_compare.py
  --random-init`, ACWM robot_arm, 17-frame window, masked denoise MSE, random
  adapter, 3090, 2026-07-25. (Local eval, not a wandb training run.)
- **~0.036** — prior Push Cube base-loss measurement; a matched-geometry
  re-measure at the identical 17f/masked setup is _pending_ (the two numbers are
  directionally solid but not yet strictly apples-to-apples).
- SkyReels probes: 3 native-pipeline i2v generations (Push Cube, robot_arm ep33
  ×2). Prompt-poison caveat: "Franka **Panda**" → panda hallucination; fixed +
  validated `configs/prompts/acwm_robotarm.yaml` (neutral geometric description,
  T5-precomputed).
- Axis design: [[ablation-axes]].

## Next

Scaffolded the **base × dataset matrix** (6 runs) — manifest at
`jobs/experiments_cluster/acwm_phys/EXPERIMENTS.md`, tickets under
`20_Tickets/experiments/`:

- **Wan** Robot Arm ([[exp-backbone-wan-robotarm-run]]) + Push-Cube intervention
  cap 0.5 + warmup 500 ([[exp-adapter-wan-cap50-warmup-pushblock-run]], launch-ready).
- **DynamiCrafter** Robot Arm / Push Cube — migrated onto the `dynamicrafter_video`
  (BaseVideoModel) interface; configs + acwm_phys data path built.
- **SkyReels** Robot Arm / Push Cube — `SkyReelsVideoModel` + provider-aware
  precompute built; i2v preprocessor + training entrypoint in progress.

Immediate: matched Push-Cube re-measure (close the 0.036 rigor gap); DC
base-coherence probe per domain; robot_arm download+precompute for the Wan run.
