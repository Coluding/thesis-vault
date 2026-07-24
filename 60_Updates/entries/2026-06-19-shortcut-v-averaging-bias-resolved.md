---
date: 2026-06-19
category: finding
deliverable: D3
meeting: 2026-06-19
sources:
  - "[[../../30_Knowledge/theory/shortcut-v-averaging-bias]]"
  - "[[../../30_Knowledge/theory/noise-angle-phi-and-schedule]]"
  - "[[../../30_Knowledge/theory/ddim-step-v-parameterisation]]"
  - "[[../../50_Decisions/open/shortcut-target-endpoint-vs-v-averaging]]"
  - "[[entries/2026-06-17-shortcut-overfit-larger-data-volatile-loss]]"
  - "[[../../30_Knowledge/experiments/avid-shortcut-anchor045-volatile-loss]]"
---

# Volatile shortcut loss resolved: step-size mixing plus a real geometric bias in the v-prediction target

## What

Per-rung logging (one curve per sampled step size) resolves the open question from
2026-06-17. Two things, confirmed:

1. **The aggregate `shortcut_direction_loss` is spiky because it marginalises
   over step size.** Per-rung magnitudes span about **50×** (`N064 ≈ 0.002`
   up to `N001 ≈ 0.1`), so one mixed curve swings by which `d` was drawn.
   Splitting shows **fine rungs converge low; coarse (few-step) rungs stay high
   and never settle.**
2. **That coarse-rung failure is a derived bias, not undertraining.** The
   self-consistency target `(v1+v2)/2` is exact for flow matching but **biased
   for v-prediction + DDIM**: the VP trajectory is a circular arc (noise-angle
   φ); averaging two tangents on a circle shrinks them by `cos(θ/2)` and rotates
   them (the sagitta), so the few-step target lands inside the arc. Verified on
   the real `ddim_micro_step_v` with **zero model error**: landing error
   ≈ 0% → 16% → 24% from fine steps to the 2-step / ¾-step jumps. Secondary: the
   step-size conditioning is in t-units and ambiguous because φ(t) is nonlinear
   (≈ 5.7× more arc per Δt near the data end than near noise).

## Why it matters

The headline D3 mechanism needs the few-step targets to be correct; by
construction they are not. **More training cannot fix it** — the loss's fixed
point is a biased velocity field. This converts "is the approach broken?" into a
specific, fixable target bug, and is the reason the qualitative few-step rollouts
are soft (recognizable but blurry / colour-smeared).

## Fix options (decision open)

Endpoint inversion (drop-in, exact), predict displacement (exact,
schedule-independent), arc-length / log-SNR reparam (secondary), or move to a
**flow-matching base** (κ=0 → bias gone). See
[[../../50_Decisions/open/shortcut-target-endpoint-vs-v-averaging]]. Full
derivation, the sagitta/fixed-point argument and the interactive visual are in
[[../../30_Knowledge/theory/shortcut-v-averaging-bias]].

## Evidence / sources

Per-rung W&B charts + run artifacts in `data/results/20261706/`
(`charts_with_step_size/`, `config_run_anchor_prob=045.yaml`); run
`diffusion_avid_shortcut_metaworld`, Snellius H100, bs 48, ~1600 steps. Per-rung
logging shipped to `training/trainer.py` 2026-06-17. Bias percentages computed
against `ddim_micro_step_v` (in the bias note). wandb run id / commit / ckpt
still `_needs verification_`.
