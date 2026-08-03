---
type: experiment
date: 2026-07-30
config: external_repos/avid/latent_diffusion/configs/train/avid/avid_11M_acwm_robotarm.yaml (ACWMVideoDataModule, kinematics/robot_arm/ind_train)
commit: uncommitted working tree @ 2026-07-30 (external_repos/ is gitignored; the tracked launcher pair is jobs/experiments_cluster/avid_official/submit_{train_avid_acwm_robotarm,probe_acwm_robotarm_action}.sh)
wandb_run_id: rqp4s3gp (project avid-acwm-robotarm-official)
ckpt_path: /scratch-shared/lbierling/avid-acwm-robotarm/avid_acwm_robotarm_11M/checkpoints/epoch=4-step=5000.ckpt (probed); run still training toward max_steps 15000
status: running
deliverable: D2
metrics:
  action_effect_rel_shuffle: 0.029475
  action_effect_rel_zero: 0.027991
  cos_true_shuffle: 0.998984
  cos_true_zero: 0.998975
  base_null_violation: 0.0
  adapter_rel_contribution: 0.069838
  gate_mask_mean: 0.875879
  effect_over_adapter_ratio: 0.422
  train_loss_step_early: 0.279136
  train_loss_step_late: 0.015114
  train_mask_mean_early: 0.500977
  train_mask_mean_late: 0.879883
  fid_epoch: 60.2196
  fvd_i3d_epoch: 216.467
  mse_epoch: 0.013551
notes: "The ORIGINAL AVID recipe FOLLOWS ACTIONS on ACWM Robot Arm — the same synthetic dataset where our Wan/DC/SkyReels adapters were action-blind. effect_rel 0.0295 vs our 0.0056/0.0034/0.0013 (5.3-22.7x), null exactly 0. SUPERSEDES the 2026-07-29 'blindness is a data/OOD problem' read: the data is NOT the blocker, the gap is our implementation or adapter design. CAVEAT: probed at step 5000 vs our runs' ~800-1200 — not training-matched."
---

# AVID follows actions on ACWM Robot Arm — it's our implementation, not the data (D2)

> Probe: `external_repos/avid/latent_diffusion/scripts/probe_action_sensitivity.py`
> (`--config avid_11M_acwm_robotarm.yaml --data-dir .../kinematics/robot_arm/ind_train`),
> reproducing our `eval_action_effect_rel` on `AVIDAdapter.apply_model`
> (v-prediction). 120 samples (8 batches × 5 timesteps × 3 paired draws),
> timesteps [100, 300, 500, 700, 900]. Checkpoint `epoch=4-step=5000.ckpt`.
> Dataset as reported by the probe: 9492 train / 500 val clips, 2002 episodes,
> traj_len 16 × stride 4, batch 8; 6 episodes clipped to decodable length.

## Result

| metric | value |
|---|---|
| `action_effect_rel` shuffle | **0.029475** |
| `action_effect_rel` zero | 0.027991 |
| cos(true, shuffle) | 0.998984 |
| **`base_null_violation`** | **0.000e+00** (PASS — frozen base invariant) |
| `adapter_rel_contribution` | 0.069838 |
| gate/mask mean | 0.875879 |
| **effect ÷ adapter contribution** | **≈ 0.42** |

## The grid this completes — same data, same probe, recipe varied

| run | data | effect_rel | effect ÷ adapter |
|---|---|---|---|
| **AVID · ACWM Robot Arm** `rqp4s3gp` @5000 | synthetic | **0.029475** | **0.42** |
| Wan · ACWM Robot Arm `ncztxyyo` | synthetic | 0.0056 | 0.056 |
| DC · ACWM Robot Arm `c3pcewxk` | synthetic | 0.0034 | — |
| SkyReels · ACWM Robot Arm `8zjjn7wl` | synthetic | 0.0013 | — |
| AVID · RT-1 `93qrvr5v` @15000 | real, in-dist | 0.0495 | 0.66 |
| AVID · ACWM Push Cube `423pjv8y` | synthetic, **confounded** | 0.0015 | — |

AVID is **5.3× / 8.7× / 22.7×** more action-sensitive than our three adapters on
the dataset they went blind on. Null control exactly 0 on both sides, so the
measurement is trustworthy in both directions.

> **Note on the DC number.** 0.0034 is the step-~801 snapshot recorded on
> 2026-07-28. `c3pcewxk`'s **final** summary (crashed @ 11.69 h) is
> `eval_action_effect_rel` **0.004238** (zero 0.003900, `loss_gap` 9.09e-06,
> null 0). Blind on either reading; use 0.004238 when citing the run's endpoint.

## The sharper contrast — it is not cloning, it is contributing the wrong thing

Ours and AVID diverge in *opposite* directions on how hard the adapter works vs
how much of that work is action-driven (both from the runs' own summaries):

| | our DC `c3pcewxk` (final) | AVID `rqp4s3gp` @5000 |
|---|---|---|
| `adapter_rel_contribution` | **0.319004** | 0.069838 |
| gate / mask mean | 0.596796 (≈40% adapter weight) | 0.875879 (≈12%) |
| `action_effect_rel` | 0.004238 | 0.029475 |
| **effect ÷ adapter** | **0.014588** | **0.422** |

Our adapter does **4.6× more work** than AVID's and spends **~1.5%** of it on
actions; AVID's does less and spends **~42%**. Composition form is identical
([[../tech/avid-vs-ours-action-conditioning]] row 8), so this is not the
"adapter collapses to cloning the base" failure mode seen on MetaWorld — ours is
contributing plenty, just nothing action-conditioned. That matches the 2026-07-21
verdict of "pure action-independent domain adjustment"
([[20260721-replace-fix-validation-sigma-sweep-action-probe]]) and is consistent
with the concat-vs-add mechanism: if the action rides superimposed on a much
larger time signal, the adapter still learns a large correction — a
time/appearance-driven one.

**Coverage gap found while pulling these:** `gxq7kxzp` (our DC × ACWM Push Cube,
16.96 h, killed), `kjgt3z0f` (DC Robot Arm, 5.61 h) and `t4bp8nki` (DC shortcut,
16.10 h) have **no action-sensitivity keys logged at all** — the probe was not
enabled. Our longest DC run has no action datapoint.

## Interpretation (analysed)

**The action-blindness is ours, not the data.** The unmodified AVID recipe —
same frozen DynamiCrafter-512 base weights (`ckts/dynami512.ckpt`), same
synthetic ACWM Robot Arm episodes, same probe — reaches effect_rel 0.0295 with
42% of its adapter's contribution action-driven, where our three adapters sit at
0.0013–0.0056 with ~5%. Varying only the *implementation* moves the metric by an
order of magnitude.

**This supersedes [[20260729-avid-rt1-follows-actions-control]].** That note
concluded "the blindness is a DATA/OOD problem, not the recipe" by comparing a
clean full-data RT-1 run against the 64-clip Push Cube smoke. With the clean
synthetic cell now filled, that inference does not hold: the recipe follows
actions on synthetic OOD data too. The RT-1 measurement itself stands — only the
conclusion drawn from it is replaced.

**Training health is not the signal.** The run also looks good on every
conventional metric (loss 0.279→0.015, `mask_mean` 0.501→0.88, FID 60.2, FVD
216.5 — vs Push Cube's FID 80.1 / FVD 832.0). But Push Cube looked healthy too
and probed blind at 0.0015, and our own 2026-07-21 replace run beat the base on
PSNR at every eval while shuffled actions moved the loss <1e-5
([[20260721-replace-fix-validation-sigma-sweep-action-probe]]). Only the probe
separates these.

**Eyeballed videos cannot settle it either.** The epoch-7 validation clips look
sharp and plausible, but they are rendered *with the true actions* on top of the
GT first frame, so first-frame + caption + task prior reproduces them without any
action use. Measured on that clip, the sample's frame-to-frame motion is flat
(~0.0040 from t=6 to t=15) while GT has a structured double-hump
(0.0067 peak → 0.0014 pause → 0.0068 peak), and cumulative displacement from
frame 0 is 0.0305 vs GT's 0.0141 — the sample moves further but not in the same
pattern. Crude grayscale proxy, one clip, mid-training — indicative only, and
listed here so it is not mistaken for evidence either way.

## ⚠ Open confound — NOT training-matched

The probe is at **step 5000**; our three comparison runs were snapshotted at
~1172 (Wan), ~801 (DC), ~897 (SkyReels, killed). AVID has had **4–6× more
training** than anything it is compared against. This is the same class of
confound that invalidated the Push Cube reference (64-clip memorization) — do
not promote the headline until it is closed.

**`epoch=4-step=5000` is the earliest checkpoint that exists** (the config saves
`every_n_epochs: 5`), so the match cannot be made from the AVID side without
re-running with denser checkpointing. **Match from our side instead:** push the
Wan/DC Robot Arm runs to ~5000 steps and re-probe there.

- Ours still ≈0.005 at 5000 ⇒ comparison clean, conclusion holds outright.
- Ours climbs toward 0.03 at 5000 ⇒ our runs were simply undertrained and the
  2026-07-28 blindness verdict was premature — check that before searching our
  code.

This also closes the 2026-07-28 open item "confirm the flat action signal isn't a
late bloomer" ([[20260728-acwm-robotarm-matrix-action-blind]]) from the other
direction.

## Setup notes (why this run exists at all)

A first attempt **failed**: wandb `iybbufly` (2026-07-28 10:45, local
lukas-station), 0 logged steps, log ending cleanly after model setup with **no
traceback** — stdout/stderr were never redirected. Cause never determined. Ruled
out by inspection: `action_dims` is correctly 7 (robot_arm actions verified
`[128, 7]` in `metadata.pt`; the config's header comment claiming a 7→2 change is
a stale copy-paste from the Push Cube variant), the short-video guard from
[[../../20_Tickets/done/bug-data-acwm-robotarm-short-videos]] *is* present
(`acwm.py:127-132`), and `num_workers` defaults to 0 so the decord fork deadlock
does not apply.

One real defect was fixed pre-emptively: the config shipped
`data.params.target_height: 384` while `act_cond_diffusion_11M_acwm_robotarm.yaml`
declares `image_size: [40, 64]` (= 320×512 ÷ 8) and the DynamiCrafter base is
natively 320×512. Push Cube — the arm that did train — uses 320. `image_size` is
read only at `ddpm3d.py:1119,1148` for sampling shape, so 384 would not break the
training forward, but it would break the `ImageLogger` and puts the frozen base
off-distribution. Forced to 320 by the launcher.

## Next

- **Step-matched probe** at ~1000 steps (above) — gate on this.
- Re-probe at step 15000 for the training-matched comparison against RT-1's
  `epoch=14-step=15000`.
- **Run the `concat` variant — the highest-value next run.** Our DC run
  `c3pcewxk` is already near-AVID (`avid_mask_mix`, `gate_bias: 0.0`,
  `condition_on_base_outputs: true`, velocity, AVID's own 11M UNet config). The
  single structural divergence was source-verified on 2026-07-27 —
  AVID concatenates time⊕action into orthogonal 64-dim halves, we *add*
  full-width 128s, so a large time signal can swamp the action
  ([[../tech/avid-vs-ours-action-conditioning]]). The toggle
  (`action_time_combine: concat`) and the config
  (`diffusion_dc_acwm_robotarm_concat.yaml`) have existed since 2026-07-27 and
  **have never been launched** (verified: `dc-acwm-robotarm` holds only
  `kjgt3z0f`, `u9u7kxia`, `c3pcewxk`, all `add`, all crashed). This result
  removes the data/OOD alternative that displaced it.
  ⚠ flip **only** `action_time_combine` — the shipped concat config also changes
  `use_step_level_conditioning` and `shortcut_anchor_prob`.
- Then the fuller match if concat doesn't explain it.
  → [[../../20_Tickets/experiments/exp-adapter-our-framework-avid-replication-robotarm]]

## Related

- [[20260729-avid-rt1-follows-actions-control]] — the run this supersedes
- [[20260728-acwm-robotarm-matrix-action-blind]] — the three blind adapters
- [[20260721-replace-fix-validation-sigma-sweep-action-probe]] — good metrics + total blindness
- [[avid-vs-ours-action-conditioning]]
