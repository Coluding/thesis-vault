---
type: experiment
date: 2026-07-31
config: probe of the NOBASE checkpoint (diffusion_wan22_avid_xattn_tokennorm_nobase_acwm_robotarm.yaml); job 25107536
commit: uncommitted working tree @ 2026-07-31 (probe: generate_wan22_i2v_compare.py --action-analysis)
wandb_run_id: probe of TOKENNORM-NOBASE (job 25088945) latest checkpoint
ckpt_path: /scratch-shared/lbierling/outputs/acwm-robotarm-tokennorm-nobase-run/checkpoints/ (latest at probe time, ~step 4000)
status: completed
deliverable: D2
metrics:
  steer_cos_sigma_050: -0.0021
  steer_cos_sigma_083: -0.0032
  steer_cos_std: 0.07
  temporal_alignment_score: 0.25
  temporal_alignment_chance: 0.28
  spatial_concentration_top10motion: 0.114
  spatial_concentration_chance: 0.10
notes: "WHY sensitivity-without-control, resolved: the action signal reaching Wan's output is a GLOBAL BAG — (A) steering cos = 0.00 at both sigmas (the action->output map is ARBITRARY, direction never learned); (B) temporal alignment 0.25 = chance — zeroing ANY pixel-frame bin of actions perturbs all latent frames uniformly (peak always at frame 0); the 4x pixel->latent token correspondence never formed; (C) spatial concentration 10.5-11.4% vs 10% chance — the effect is spatially uniform, not on the arm. Fix implemented: explicit temporal binning (adaptive_avg_pool1d px->latent) in the simple-adapter per-frame path; SIMPLE arm relaunched with it (25107616)."
---

# Wan's action signal is a global bag — arbitrary, unaligned, unfocused (D2)

> The "why sensitivity-without-control" probe, three measurements in one job on
> the best-recipe (NOBASE) checkpoint, σ ∈ {0.5, 0.83}, paired forwards.

## A. Steering direction — the map is arbitrary

`cos(pred(a_B) − pred(a_A), v_B − v_A)` with clip B's actions on clip A's
state, paired noise: **−0.002 / −0.003 (± 0.07)** at σ 0.5 / 0.83. Exactly
zero. Swapping actions moves the prediction (|Δpred|/|Δv| ≈ 0.06–0.08) in a
direction **uncorrelated** with the target difference. The adapter learned
*that* actions vary, never *what they mean*. (Caveat: v_B − v_A includes
appearance differences; on same-scene robot-arm data it is dominated by pose.)

## B. Temporal response — the pixel→latent correspondence never formed

Action tokens are per **pixel** frame (~97/window, `_action_sequence`
passthrough); the DiT denoises **latent** frames (~25, 4× compression).
Zeroing one pixel-frame bin at a time and reading per-latent-frame |Δpred|:

- every row's response is ~uniform (0.93–0.99 row-normalised) with the peak
  **always at latent frame 0**, regardless of which bin was zeroed;
- alignment score **0.25 vs 0.28 chance** — no diagonal whatsoever.

The trunk treats the action tokens as an unordered bag: perturbing *any* part
shifts a global signal. This also explains why `action_pos_emb` was the
fastest-moving parameter in the adapter (17.5%/400 steps) — training was
*straining* to build a correspondence it never achieved on a 0.45% gradient.

## C. Spatial map — the effect is uniform, not on the arm

Action-driven |Δpred| mass inside the top-10%-motion region: **10.5–11.4%**
(10% = unconcentrated). Spatially flat, like the gate (std ~0.002 vs AVID's
0.057). The correction is not aimed at the arm.

## Synthesis

The action information that reaches the output is a **global, temporally
unaligned, spatially uniform perturbation** — a bag-of-actions bias, not a
motor command. All three structure axes (direction, time, space) come back at
chance. This is the mechanism of "sensitivity without control", and it stacks
cleanly on the economics: a 0.45% gradient was never going to force the
model to learn three kinds of structure nobody wired in.

## Fixes this implies (in order of leverage)

1. **Enforce the temporal correspondence instead of hoping it is learned** —
   bin action tokens onto latent frames explicitly. **Implemented** in the
   simple-adapter per-frame path (`adaptive_avg_pool1d` px→latent,
   `output_head.py`; verified: perturbing px frames 40–43 lands on latent
   frame 10 at 23.6× peak/median). The SIMPLE arm was relaunched with it
   (job 25107616) — its original launch had silently fallen back to pooled
   global conditioning (97 ≠ 25 length mismatch), which this probe exposed.
   The DiT-clone path's equivalent (bin `action_seq` in the preprocessor,
   `action_seq_len = t_latent`) is a 1-line config once validated.
2. Spatial focus: a gate that can localise (or per-region loss weighting).
3. Direction: only an objective that *pays* for correct steering (action-CFG /
   rollout losses) forces the map itself — architecture cannot.

## Related

- [[20260731-wan-tokennorm-nobase-training-results]] — the checkpoint probed
- [[20260731-why-wan-copies-the-base-decomposed]] — the economics underneath
- [[20260731-wan-action-trace-value-pathway-drowns]] — transport, previously fixed
