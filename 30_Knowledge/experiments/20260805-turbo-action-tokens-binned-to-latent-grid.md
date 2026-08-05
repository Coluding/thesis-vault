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

🛑 **This is not a result.** n=6, pooled across two *different* eval draws, GT values
fall in two clusters rather than spreading → p ≈ 0.08. Hand-measured, not logged.
**Confirm before citing anywhere.** `Trainer._clip_motion` / `_pearson` +
`eval/motion_corr_{adapted,base}` are now wired into `_native_quality_eval`
(16 clips/cycle, tests in `tests/test_motion_tracking_metric.py`) and fire
automatically on the next run — that is the confirmation.

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
