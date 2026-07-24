---
title: Wan2.2 AVID adapter â no-shortcut ablation
tags: [experiment, wan2.2, avid, shortcut, ablation]
date: 2026-06-27
---

# Wan2.2 AVID adapter, action-conditioned, NO shortcut consistency

## Goal
Baseline/ablation against the shortcut-trained AVID run: does the AVID-style
`backbone: wan` output-delta, action-conditioned on a frozen Wan2.2-TI2V-5B
base, learn a useful world model from **plain flow matching alone**, with no
shortcut self-consistency supervision?

## Setup
- Config: `configs/diffusion_wan22_avid_i2v_metaworld_noshortcut.yaml`
  (clone of `diffusion_wan22_avid_i2v_metaworld.yaml`).
- Job: `jobs/submit_train_wan22_avid_noshortcut.sh`
  (script `scripts/train_wan22_i2v_metaworld.py`, ckpt `ckpts/Wan2.2-TI2V-5B`).
- Only change vs the shortcut run: `shortcut_anchor_prob: 1.0`.

## Why anchor_prob=1.0 = no consistency loss
In `Trainer._maybe_prepare_shortcut`, the flow distillation branch computes
`do_anchor = anchor_prob >= 1.0 or ...`. With `anchor_prob=1.0` every step is an
## Setup
- Config: `configs/diffusion_wan22_avid_i2v_metaworld_noshortcut.yaml`
  (clone of `diffusion_wan22_avid_i2v_metaworld.yaml`).
- Job: `jobs/submit_train_wan22_avid_noshortcut.sh`
  (script `scripts/train_wan22_i2v_metaworld.py`, ckpt `ckpts/Wan2.2-TI2V-5B`).
- Only change vs the shortcut run: `shortcut_anchor_prob: 1.0`.

## Why anchor_prob=1.0 = no consistency loss
In `Trainer._maybe_prepare_shortcut`, the flow distillation branch computes
`do_anchor = anchor_prob >= 1.0 or ...`. With `anchor_prob=1.0` every step is an
anchor step â it returns a **None** shortcut target (and clamps `step_level` to
the smallest rung). `training_step` only adds the shortcut/consistency terms when
`"shortcut_target" in batch`, so with a None target the sole supervision is the
standard flow-matching loss. `local_consistency_weight` /
`multistep_consistency_weight` are 0 too.

step_level conditioning stays wired (constant smallest rung) so the adapter input
contract is unchanged â the only difference from the shortcut run is the absence
of the consistency target/loss. Eval grid (multi-N) is a pure diagnostic and is
unaffected.

## Compare against
The shortcut run `wan22-avid-i2v-metaworld` (same adapter/base, shortcut ON).
wandb project: `wan22-avid-i2v-metaworld-noshortcut`.