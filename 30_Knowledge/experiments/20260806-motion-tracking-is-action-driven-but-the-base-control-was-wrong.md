---
type: experiment
date: 2026-08-06
config: configs/wan22/diffusion_wan22turbo_action_robotarm.yaml
commit: (⚠️ WORKING TREE, not a commit — see "Reproducibility caveat")
wandb_run_id: mo3k2639
ckpt_path: outputs/wan22turbo-action-robotarm/checkpoints/{best.pt, step_00001000.pt}
status: completed
deliverable: D2
metrics:
  eval_loss_400: 0.18159
  eval_loss_800: 0.14317
  eval_loss_1200: 0.12689
  motion_corr_action_gain_draws: "+0.069, +0.168, +0.122, +0.213"   # 4/4 positive, mean +0.143
  motion_corr_adapted_minus_base: "+0.13, +0.10, +0.045, -0.034"    # sign-inconsistent
  eval_action_effect_rel_final: 0.00973
  eval_action_loss_gap_final: 0.00005
notes: |
  Slurm 25259766, TIMEOUT at 3:00:00 (the configured limit, not a failure), 1200 steps,
  batch 16, 49-frame windows, peak VRAM 89.5/93 GiB (96%). First run of the instrumented
  motion metric. wandb: Wan2.2-avid-xattn-acwm-robotarm/runs/mo3k2639.
---

# The action DOES drive per-clip motion — but the frozen-base control was measuring the wrong thing

**Base:** `Wan22TurboVideoModel` (quanhaol/Wan2.2-TI2V-5B-Turbo), frozen, 4-step distilled
grid `[1.0, 0.9375, 0.83333, 0.625]` at shift 5.0.
**Adapter:** output/wan, 101,292,484 trainable (1.99%), cross-attention injection,
`action_seq_len: latent` (13 tokens ↔ 13 latent frames).
**Data:** ACWM-Phys robot_arm, 49-frame windows.

This is the confirmation run for **Result 3** of
[[20260805-turbo-action-tokens-binned-to-latent-grid]], which was hand-measured from six
mp4s and explicitly marked *"do not cite until the instrumented run reports."* It has now
reported.

## The measurement

`Trainer._native_quality_eval` computes per-clip inter-frame motion for the ground truth,
the adapted rollout, the frozen base, and a **paired shuffled-action control** (same
weights, same conditioning frame, same seed — only another clip's actions), then
correlates each against GT. Two eval cycles, two independent draws of n=16 each:

| step | draw | corr(adapted, GT) | base | shuffled | **action gain** | ratio_ad |
|---|---|---|---|---|---|---|
| 400 | 1 | −0.082 [95% CI −0.60, +0.56] | −0.212 | −0.151 | **+0.069** | 1.13 |
| 400 | 2 | +0.494 [95% CI +0.13, +0.77] | +0.397 | +0.326 | **+0.168** | 1.05 |
| 800 | 1 | +0.196 [95% CI −0.29, +0.55] | +0.151 | +0.073 | **+0.122** | 1.16 |
| 800 | 2 | −0.105 [95% CI −0.60, +0.39] | −0.071 | −0.318 | **+0.213** | 0.92 |

## Result 1 (SOURCED): the paired control fires — 4/4

`motion_corr_action_gain` = corr(adapted) − corr(shuffled) is **positive in all four
draws**: +0.069, +0.168, +0.122, +0.213, mean **+0.143**. Sign test p = 1/16 ≈ 0.06
one-tailed.

Because the shuffled control differs from the adapted rollout **only** in which actions
were fed — identical weights, identical conditioning frame, identical seed, same clips —
this isolates the action. It is the causal quantity.

## Result 2 (SOURCED, negative): the frozen-base control does NOT fire

adapted − base across the same four draws: +0.13, +0.10, +0.045, **−0.034**.
Sign-inconsistent and centred near zero. **The frozen base, which never sees an action,
tracks per-clip GT motion about as well as the adapter does.**

The hand-measured preliminary claim (adapted r = +0.75 vs base r = +0.09, a gap of
**0.66**) does **not** survive. The logged gap is an order of magnitude smaller and
reverses in one draw.

## What this means

The reservation written into the preliminary note was the correct one, and it resolved
against the base:

> *"The frozen base is not a sufficient control — it differs from the adapter in every
> respect, not only action access."*

That objection was not academic; it was the entire effect. Had the base remained the
control, this would have been recorded as a strong positive result on the strength of a
comparison that is confounded by everything the two models differ in (capacity,
conditioning pathway, training). **The paired shuffled-action control is what turned a
false positive into a small true one.** Design lesson worth carrying to every future
action-conditioning claim in this thesis.

## Scope of the claim

Supported: **a modest, consistently-signed action effect on how much the arm moves per
clip.** Not supported: the magnitude of the hand measurement, or that `corr(adapted, GT)`
is distinguishable from zero — its CI spans zero in 3 of 4 draws.

This remains an effect on motion **magnitude**, not **correctness**:
`eval_action_loss_gap` finished at **0.00005** and `eval_action_cos` at 0.99995, i.e.
feeding the *correct* action still does not reduce error versus a wrong one. The
dissociation documented as Result 2 of the predecessor note is unchanged.

`eval_action_effect_rel` finished at 0.00973 and `eval_action_effect_vs_adapter` at
0.11524 — the same regime as the predecessor run, so this arm behaved like `jlnl7s1k`.

## Training

`eval_loss` 0.18159 (step 400) → 0.14317 (800) → **0.12689 (1200)**, still falling at the
3 h limit. Unlike `jlnl7s1k`, which overfit from step 1200, this run stopped before the
turn — the `--time=03:00:00` cap recommended by the predecessor note worked as intended.

## Caveats before citing

1. **No interval on the gain.** The 95% CIs above are on `corr(adapted, GT)`, not on the
   gain. `Trainer._gain_ci` now bootstraps the gain directly with paired resampling
   (added 2026-08-06, 5 tests), but it **cannot be applied retroactively** — this run
   logged only summary correlations, not per-clip series. **The next run on this arm
   produces the first gain intervals.**
2. n = 16 per draw, 4 draws, 2 cycles, **one run**, no seed replication.
3. Dashboard: **https://wandb.ai/coluding/Wan2.2-avid-xattn-acwm-robotarm/runs/mo3k2639**
   (corrected 2026-08-07 — an earlier revision of this note claimed the run produced no
   wandb entry. It did; the URL is printed to stderr, and the check only grepped stdout.)

## Reproducibility caveat

The remote working tree had ~135 uncommitted modified files; the run used rsynced
working-tree code, not a commit. Same caveat as
[[20260805-turbo-action-tokens-binned-to-latent-grid]]. **Commit before the next launch.**
