---
type: experiment
date: 2026-08-05
config: configs/wan22/diffusion_wan22turbo_action_robotarm.yaml
commit: 75721b7 (⚠️ WORKING TREE, not this commit — see "Reproducibility caveat")
wandb_run_id: jlnl7s1k
ckpt_path: outputs/wan22turbo-action-robotarm/checkpoints/{best.pt @1200, step_00004000.pt}
status: completed
deliverable: D2
metrics:
  eval_loss_best: 0.12709   # step 1200; rises to 0.19184 by step 4000
  action_effect_rel_at_1600: 0.01268
  action_effect_vs_adapter_at_1600: 0.17899
  action_loss_gap_all_evals: ~0   # |x| <= 0.00055 across all ten evals
  denoise_adapter_delta_at_4000: -0.00930   # negative throughout
notes: |
  Slurm 25240927, 10:00:25 wall (TIMEOUT = intended end of a 5M-step config, not
  a failure), 4000 steps, batch 16, peak VRAM 89.5/93 GiB (96%), no OOM.
---

# Binning action tokens to the latent grid raises action *effect* 4–6.5× — but not action *accuracy*

**Base:** `Wan22TurboVideoModel` (quanhaol/Wan2.2-TI2V-5B-Turbo), frozen, 4-step
distilled grid `[1.0, 0.9375, 0.83333, 0.625]` at shift 5.0.
**Adapter:** output/wan, 101,292,484 trainable (1.99%), `configs/base/wan_adapter_100m.yaml`,
cross-attention injection, mask_mix, `gate_cap 0.9`.
**Data:** ACWM-Phys robot_arm, 49-frame windows → 13 latent frames, 7371 tokens/clip.

Compared against [[20260805-turbo-49f-predecessor]] — run `zcjvjj5a` (slurm
25221298), the same arm at 34M / 97 frames with **unbinned** action tokens.

## The change under test

The previous run logged, at startup:

```
action tokens: PASSTHROUGH (1 per PIXEL frame, 97 tokens) | latent_frames=25
```

97 cross-attention action tokens for a DiT running on 25 latent frames — token *j*
corresponded to no latent frame. This is the documented 2026-08-02 failure mode
(RT-1 run `5w72bo01`: temporal alignment 0.25 = chance; see
[[20260731-wan-action-signal-is-a-global-bag]]). Cause: `_resolve_action_seq_len`
returns `None` when `action_seq_len` is absent **and** `action_per_frame: false`,
and the Turbo config set neither. Fix: `action_seq_len: latent` →

```
action tokens: 13 tokens, binned to the LATENT grid | latent_frames=13
```

## Result 1 (SOURCED, solid): the action reaches the adapter, 4–6.5× more

At matched step 1600, `jlnl7s1k` (binned) vs `zcjvjj5a` (unbinned):

| metric | unbinned, 34M | binned, 100M | ratio |
|---|---|---|---|
| `eval_action_effect_rel` | 0.00313 | **0.01268** | 4.1× |
| `eval_action_effect_vs_adapter` | 0.0277 | **0.179** | 6.5× |

Monotone growth over the run (all ten evals):

| step | 400 | 800 | 1200 | 1600 | 2000 | 2400 | 2800 | 3200 | 3600 | 4000 |
|---|---|---|---|---|---|---|---|---|---|---|
| `effect_rel` | .0048 | .0081 | .0105 | .0127 | .0160 | .0163 | .0255 | .0189 | .0225 | .0219 |
| `vs_adapter` | .043 | .080 | .124 | .179 | .228 | .224 | .307 | .230 | .285 | .272 |

⚠️ **Two variables changed at once** (token binning *and* 34M→100M adapter), so the
4–6.5× is not attributable to binning alone. A 34M + binned control is needed to
split them — not yet run.

## Result 2 (SOURCED, negative): it does not become action *accuracy*

`eval_action_loss_gap` across all ten evals: `+.0001, −.0003, +.0001, −.00003,
+.0001, +.0003, +.0005, +.0003, +.0006, +.0003`. No trend; |x| ≤ 0.00055 throughout.
`eval_action_cos` never leaves 0.9998.

**A clean dissociation: the action now changes the prediction, but changing it to
the *correct* value does not reduce error versus shuffling or zeroing it.** The
adapter consumes the action without having learned the true action-conditioned
dynamics. The misalignment was real and worth fixing; it was **not** the binding
constraint.

## Result 3 (PRELIMINARY — n=6, do NOT cite): motion may track ground truth

Measured by hand from the run's eval mp4s (`wandb/run-20260805_132346-jlnl7s1k/
files/media/videos/eval_step_grid/`), inter-frame motion in 0–255 grayscale,
panels are `[GT | base | adapted]`, 3 clips at step 1400 + 3 at step 4000:

| | GT | base | adapted |
|---|---|---|---|
| values | .445 .114 .505 .124 .105 .140 | .259 .356 .377 .294 .286 .338 | .182 .147 .282 .158 .092 .200 |
| **corr with GT** | — | **+0.09** | **+0.75** |
| mean ratio vs GT | 1.00 | 1.33 (over-moves) | 0.74 (under-moves) |

Reading *if it holds*: the frozen base emits near-constant motion regardless of
clip (r≈0.09 — correct, it never sees actions), while the adapter modulates motion
per clip (r≈0.75) — i.e. the action sets *how much* the arm moves even though MSE
cannot see it. That would explain the Result-2 dissociation, since squared error
penalises slightly-misplaced motion as hard as none.

⚠️ **Preliminary, but stronger than first assessed** (revised 2026-08-06).

Initially dismissed as "n=6 pooled across two eval draws, p≈0.08". Both parts of that
were wrong:

1. **Spearman, not Pearson, is the right statistic** here — GT motion clusters into
   fast/slow groups, which is precisely when Pearson is fragile. ρ = **0.943**, above
   the n=6 critical value of 0.886 → p ≈ 0.017.
2. **Pooling is not doing the work.** Decomposed per draw, the adapted rank order
   matches GT **perfectly in BOTH independent draws** (step 1400 `[2,1,3]` vs
   `[2,1,3]`; step 4000 `[2,1,3]` vs `[2,1,3]`; the draws use different clips — GT
   values differ entirely). The base matches 1/2, i.e. chance. P(both match by
   chance) = (1/6)² = **0.028**.

What remains open is **causal, not statistical**: is the adapter tracking the ACTION,
or the conditioning frame? Arm pose at frame 0 plausibly predicts how much motion
follows. The frozen base is not a sufficient control — it differs from the adapter in
every respect, not only action access. **Only the paired shuffled-action control
settles it** (same weights, same frame, same seed, another clip's actions):
`eval/motion_corr_action_gain` > 0 with a CI excluding zero ⇒ action-driven.

Still hand-measured from 6 mp4s rather than logged. **Do not cite until the
instrumented run reports.** `Trainer._clip_motion` / `_pearson` +
`eval/motion_corr_{adapted,base}` are now wired into `_native_quality_eval`
(16 clips/cycle, tests in `tests/test_motion_tracking_metric.py`) and fire
automatically on the next run — that is the confirmation.

### The instrumented run reported (2026-08-06) — the CONTROL was wrong, not the signal

Run `25259766`, quality-eval cycles at steps 400 and 800, two independent draws of
n=16 each:

| step | draw | corr(adapted, GT) | base | shuffled | **action gain** | ratio_ad |
|---|---|---|---|---|---|---|
| 400 | 1 | −0.082 [95% CI −0.60, +0.56] | −0.212 | −0.151 | **+0.069** | 1.13 |
| 400 | 2 | +0.494 [95% CI +0.13, +0.77] | +0.397 | +0.326 | **+0.168** | 1.05 |
| 800 | 1 | +0.196 [95% CI −0.29, +0.55] | +0.151 | +0.073 | **+0.122** | 1.16 |
| 800 | 2 | −0.105 [95% CI −0.60, +0.39] | −0.071 | −0.318 | **+0.213** | 0.92 |

**Two different comparisons live in this table and they say opposite things.**

**adapted vs base — dead.** The differences are +0.13, +0.10, +0.045, **−0.034**:
sign-inconsistent and centred near zero. The frozen base, which never sees an action,
tracks per-clip GT motion about as well as the adapter does. The hand-measured
preliminary claim above (adapted +0.75 vs base +0.09, gap **0.66**) rested entirely on
this comparison and **does not survive** — the gap is an order of magnitude smaller and
in one draw it reverses.

**adapted vs shuffled — consistently positive.** `motion_corr_action_gain` is positive
in **4/4 draws** (+0.069, +0.168, +0.122, +0.213; mean **+0.143**). Sign test:
p = 1/16 ≈ 0.06 one-tailed. Same weights, same conditioning frame, same seed — only the
actions differ — so this isolates the action in a way the frozen base never could.

So the earlier reservation stated at the top of Result 3 was the correct one, and it
resolved against the base: *"The frozen base is not a sufficient control — it differs
from the adapter in every respect, not only action access."* The instrumented run shows
that objection was not academic. It was the whole effect.

**What this supports, precisely:** a modest, consistently-signed action effect on how
much the arm moves per clip. It does **not** support the magnitude of the hand
measurement, and the adapted correlation itself is not distinguishable from zero in 3 of
4 draws (CIs span zero).

Gaps before this can be cited:

1. ~~**No interval on the gain itself.**~~ **FIXED 2026-08-06.** `Trainer._gain_ci`
   bootstraps `corr(adapted) − corr(shuffled)` directly, resampling clips **once per
   draw** and recomputing both correlations on the same resampled set — the pairing is
   the design (adapted and shuffled differ only in which actions were fed, on the same
   clips, weights and seed), and bootstrapping the two series independently would
   discard it and inflate the interval. Emitted as
   `eval/motion_corr_action_gain_ci_{lo,hi}` and printed on the `[motion]` line.
   Tests in `tests/test_motion_tracking_metric.py` (5 new, incl. the confound case where
   tracking survives shuffling and the interval must straddle zero).
   ⚠️ **Cannot be applied retroactively** to the four draws above — the run logged only
   the summary correlations, not the per-clip motion series. The next run on this arm
   produces the first gain intervals.
2. n=16 per draw, 4 draws, 2 cycles, one run. A weak hint of growth with training
   (mean gain +0.119 @400 → +0.168 @800) on two points is not a trend.
3. `eval_action_loss_gap` is still ≈0 (Result 2), so this remains an effect on motion
   *magnitude*, not evidence that the adapter predicts the *correct* motion.

**Status: Result 3 partially confirmed — the causal control, not the base comparison.**
The reading above has not been discussed with Lukas.

## Result 4 (SOURCED): overfits from step 1200

`eval_loss`: .1840@400 → **.1271@1200 (best)** → .1466 → .1632 → .1422 → .1641 →
.1688 → .1902 → .1918@4000, while train loss fell to 0.0487. `best.pt` is step 1200;
the remaining 2800 steps (~7 h, ~1340 SBU) were past the optimum. Predicted in
[[../../20_Tickets/experiments/exp-adapter-action-on-distilled-wan-turbo]]: this arm
sees only 4 noise levels and 13 latent frames — the narrowest training distribution
of any arm. **Next run on this arm should cap `--time` at ~3 h.**

`eval_denoise_adapter_delta` improves −0.0257 → −0.0093 but never crosses zero: the
adapter still *hurts* denoising at every eval. Gate healthy throughout
(`gate_mean` 0.82→0.88, `gate_std` 0.039–0.046) — the
[[../../20_Tickets/bug-adapter-gate-cap-equals-init-freezes-gate]] signature is gone.

## Reproducibility caveat

The remote working tree had **135 uncommitted modified files** at launch; the run
used rsynced working-tree code, not commit `75721b7`. Config, adapter config and
`action_seq_len` are captured above and in the startup log, but an exact
re-creation needs those files. **Commit before the next launch.**

## Also changed in this run (not isolated)

49-frame clips (was 97), fp32 eval VAE decode (was bf16), new 49f latent cache
(16 856 files / 45 GB at `/scratch-shared/lbierling1/latents-robotarm-49f`).
Both were cosmetic-path changes chasing eval-video quality; see the ticket for why
the gate's videos looked better (the gate *over*-moved by 4–8× vs ground truth).
