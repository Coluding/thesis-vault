---
type: experiment
date: 2026-08-01
config: configs/wan22/diffusion_wan22_action_rt1_tokennorm_nobase.yaml (job 25107236)
commit: uncommitted working tree @ 2026-07-31
wandb_run_id: 5w72bo01 (Wan2.2-action-rt1 / wan-rt1-TOKENNORM-NOBASE)
ckpt_path: /scratch-shared/lbierling/outputs/wan-rt1-tokennorm-nobase-run/checkpoints/
status: killed          # stopped 2026-08-01 @ 25.9 h (Wan) / 21.9 h (SkyReels); checkpoints retained
deliverable: D2
metrics:
  effect_rel_evals: "0.0234 0.0264 0.0274 0.0285 0.0299 0.0238 0.0332 0.0233 0.0249 0.0201 0.0211 0.0184 0.0185 0.0168 0.0193 0.0223 0.0200 0.0254 — SETTLES to a stable ~0.021 band, no continuing decay"
  effect_rel_mean_first5: 0.0271
  effect_rel_mean_last5: 0.0245
  effect_rel_peak: 0.0332
  acwm_wan_plateau: 0.011
  avid_rt1_reference: 0.0495
notes: "⚠ SUPERSEDED 2026-08-02 — SkyReels half RETRACTED (dataset_size=76 via temporal_length 97); all RT-1 numbers IN-SAMPLE; the 2.3x compares in-sample RT-1 to held-out ACWM. Original text: Wan + both fixes (token-norm, no oracle) on in-distribution RT-1: noisy PLATEAU at ~0.025 (peak 0.0332) over 11 evals — 2.3x the ACWM plateau, ~half the AVID RT-1 reference (which uses a full separate UNet). Establishes that the action-economics is DATASET-DEPENDENT: on diverse real manipulation data the same recipe sustains 2.3x the action-following it manages on scripted ACWM. Early videos rough (nobase trade: adapter must earn prediction quality without the oracle)."
---

# Wan × RT-1 — ⚠ HEAVILY QUALIFIED (D2)

> **READ THE THREE BANNERS BELOW BEFORE ANY NUMBER ON THIS PAGE.** As of
> 2026-08-02: the SkyReels half is **retracted** (trained on 76 episodes), every
> RT-1 number is **in-sample**, the headline "2.3×" is **not like-for-like**
> (in-sample RT-1 vs held-out ACWM), and `effect_rel` itself **cannot separate
> action information from fit instability**. The original text is kept below
> unedited for provenance — it is *superseded*, not correct.

> The long-planned in-distribution test (07-29 entry), run with this week's
> recipe: `action_token_norm: true` + `condition_on_base_outputs: false` +
> live gate. 11 evals over ~9 h; run continues.

## Result

`effect_rel`: 0.0234 → … → **peak 0.0332** → … → 0.0211 — a **noisy plateau
around 0.025** (first-half mean 0.0271, second-half 0.0245; RT-1's diverse
eval batches make single evals noticeably noisier than ACWM's).

| | effect_rel |
|---|---|
| Wan × ACWM, same recipe (best) | ~0.011 |
| **Wan × RT-1, same recipe** | **~0.025 (peak 0.033)** |
| AVID × RT-1 (full separate UNet) | 0.0495 |

## Reading

1. **The action economics is dataset-dependent.** Same frozen base, same
   adapter, same fixes — only the data changed, and sustained action-following
   is 2.3× higher. On scripted same-motion ACWM clips the future is largely
   predictable without actions (the 0.45%-of-loss measurement); on diverse
   real manipulation the actions buy more, the gradient pays for them, and
   the model keeps more of them. This *reframes* the 07-30 "not the data"
   verdict: ACWM was never OOD-broken, but its futures are too predictable to
   make actions load-bearing. Both statements are true: the *recipe* failures
   were ours (fixed), and the *ceiling* is data-dependent.
2. **A lightweight adapter reaches ~half the reference** — AVID's 0.0495 uses
   a full separate action UNet; ours is a bolt-on adapter. Whether the gap is
   capacity or the remaining structure problems (temporal bag, spatial
   uniformity — [[20260731-wan-action-signal-is-a-global-bag]]) is open.
3. **Early videos are rough** (step-800 check: adapted further from GT than
   base, over-moving) — the expected nobase trade; prediction quality must be
   earned without the oracle. Re-check videos at ≥5k steps.

## SkyReels × RT-1 (job 25112302, run `skyreels-rt1-TOKENNORM-NOBASE`)

Second backbone on the same axis, same two fixes: opens at **0.0450** — 91% of
the AVID full-UNet reference, and a **35× jump** over the same backbone's ACWM
value (0.0013, the blindest of the three) — then erodes 0.0450 → 0.0307 →
0.0284 over three evals.

**Prediction closed (evals 4–6): SkyReels flattened at ~0.0185** — three
consecutive flat evals, landing in the same band as Wan (~0.021). Both
backbones converge to a **shared RT-1 floor ≈ 0.02**, retained, from very
different peaks (0.045 vs 0.033). Wan's lone 0.0406 spike at eval 22 was noise
(next eval 0.0191).

Two conclusions harden: (1) **dataset-dependence of the action economics is
backbone-general** — the blindest ACWM backbone nearly matches the reference
on RT-1; (2) **the erosion signature is universal** — every backbone × dataset
combination peaks early and decays under the plain denoising objective. What
differs is only the level, set by the data; what is shared is the objective's
failure to retain.

## 🛑🛑 RETRACTED: THE SKYREELS × RT-1 SECTION IS INVALID (2026-08-02)

**Both SkyReels RT-1 runs trained on 76 episodes, not 5000.** Verified from the
job logs (`logs/skyreels/acwm-robotarm-skyreels-skyreels-rt1-{oracle-25133625,tn-nobase-25112302}.out`):

```
dataset_size=76 steps=5000000 batch_size=6 eval=on gen_eval=on
```

Cause: both SkyReels RT-1 configs carry `temporal_length: 97`
(`configs/skyreels/diffusion_skyreels_rt1_tokennorm_{nobase,oracle}.yaml:47,62`)
with a stale comment — "128-frame episodes" — copied from ACWM Robot Arm. RT-1
episodes are ~22–115 frames (the Wan config says exactly this and uses **17**),
so `data/dataset.py:71` **silently filtered out 98.5% of the dataset**. No error,
no warning: just a 66× smaller training set.

**Retracted from this page and from the ledger:**
- "SkyReels opens at 0.0450 = **91% of the AVID reference**"
- the "**35×** data-axis jump" (0.0013 → 0.045) — its numerator is a 76-episode run
- "the blindest ACWM backbone nearly matches the reference on RT-1"
- "dataset-dependence is **backbone-general**" — the second backbone is gone,
  so this rests on Wan alone
- the SkyReels rows of the quality table below

The SkyReels ACWM number (0.0013) is unaffected — that config uses the correct
window for ACWM.

`gi44pv5k` (the oracle arm) was **killed 2026-08-02** at 7h41 rather than run
another 26 h on 1.5% of the data. Both SkyReels RT-1 configs need
`temporal_length: 17` before any re-run.

Lesser bug found alongside: `train_skyreels_acwm.py` prints
`eval dataset: ACWM-Phys split …` regardless of `--dataset`. The **builder
dispatch is correct** (`:207`, `:221` select `build_rt1_clip_dataset` for
`--dataset rt1`) — only the log string is wrong. Misleading, not damaging.

## 🛑 EVERY NUMBER ON THIS PAGE IS IN-SAMPLE (verified 2026-08-01)

The RT-1 jobs pass the **same directory** for training and eval:

```
jobs/experiments_cluster/rt1/submit_train_wan_rt1_tokennorm_nobase.sh:47-48
    --data-dir "$RT1_DIR" --eval-data-dir "$RT1_DIR"
```

and in `scripts/train_wan22_i2v_metaworld_external.py` the `--eval-data-dir`
branch (**line 419**) builds the eval dataset from that dir with **no held-out
split**; the `val_fraction` random-split fallback is at **line 445** inside an
`elif` and is therefore never reached. The comment at line 420 ("eval on a real
held-out split dir") describes the intent, not what these jobs do.

**Scope, verified by grepping every submit script:** ACWM is **clean** — every
ACWM job passes `--data-dir $ROOT/ind_train --eval-data-dir $ROOT/ind_test`.
Affected: **RT-1** (all runs) and **OpenVid** (`submit_train_wan_shortcut_openvid.sh`).

Consequences for this page:
- the ~0.02 floor, the 0.045/0.033 peaks, the 35×, the "91% of the reference"
  and the quality table below are all **training-set** numbers;
- the quality result is if anything *worse* than it reads — the adapter loses
  to the frozen base on perceptual metrics **on data it trained on**;
- **RESOLVED 2026-08-02: the AVID RT-1 reference (0.0495) is *also* in-sample**
  — its probe calls `data.train_dataloader()`. So that particular comparison is
  at least like-for-like, though both sides are training-set numbers.
- **The headline "2.3×" is NOT like-for-like**: it compares an in-sample RT-1
  number against a genuinely held-out ACWM one (ACWM jobs use
  `ind_train`/`ind_test`). This is the most serious single mis-comparison found.

A held-out split is buildable (`$HOME/scratch-shared/rt1/full/` holds the whole
dataset in shards; `train/` is a 5000-episode subset), and the checkpoints are
retained, so **re-evaluation is offline and needs no retraining**. In progress
2026-08-01 → `00_Inbox/2026-08-01-rt1-heldout-split.md`.

## ⚠ Quality: the adapter is a net regression vs the frozen base (added 2026-08-01)

Final logged eval metrics (Wan run `5w72bo01` @ 25.9 h; SkyReels
`skyreels-rt1-TOKENNORM-NOBASE` @ 21.9 h — both stopped by us, wandb state
`crashed`):

| RT-1, nobase | Wan adapted | Wan base | SkyReels adapted | SkyReels base |
|---|---|---|---|---|
| FID ↓ | **132.51** | 113.20 | **161.44** | 145.07 |
| FVD-I3D ↓ | 1575.6 | 1616.7 | **2514.8** | 2256.7 |
| LPIPS ↓ | **0.3823** | 0.3650 | **0.4977** | 0.4610 |
| SSIM ↑ | 0.6587 | _not logged_ | **0.4616** | 0.4830 |
| PSNR ↑ | 17.533 | 16.529 | 12.697 | 11.762 |
| MSE ↓ | 0.01765 | 0.02224 | 0.05375 | 0.06664 |

(bold = adapted is **worse** than the frozen base)

**Every perceptual/distributional metric is worse than the base; every
pixel-wise metric is better.** That split is the mean-regression signature: the
adapter lowers L2 by hedging toward a blurrier, more average future — the
cheapest loss reduction available under an L2 denoising objective once the
oracle is removed and the actions are worth ~0.45% of the loss.

Note also that SkyReels' *base* is poor on RT-1 (FID 145, PSNR 11.8, FVD 2257)
— that cell is low-signal end to end. Wan is the only backbone producing
usable RT-1 frames.

## ⚠ CONFOUND: does the elevated effect_rel carry information, or fit-noise?

`effect_rel` measures **sensitivity** to permuting actions. A poorly-fit,
unstable model is more sensitive to *any* input perturbation — so instability
inflates it with no action understanding involved. The observed timing fits
that alternative uncomfortably well: **both RT-1 runs peak at the first eval,
when the fit is worst, and decay as loss falls** (SkyReels 0.045 → 0.019; Wan
0.033 → 0.021). It also reproduces the data axis for free — RT-1 is harder to
fit, so it retains more residual instability, so a higher floor.

So the two readings on this page are not yet separated:

| reading | predicts peak-then-settle | predicts the RT-1 > ACWM level |
|---|---|---|
| "data sets the level" (as written above) | yes | yes |
| "early-fit instability inflates effect_rel" | yes | yes |

**`effect_rel` alone cannot distinguish them, and the frozen-base null does not
help** — it controls for architecture (an action-free base is insensitive by
construction), not for fit quality. **Therefore the 35× and the "91% of the
reference" figures above are provisional**, and the two-factor law in the
storyline (§9) rests on an unverified premise.

What separates them is whether the action-driven delta is **structured**:
noise-driven sensitivity is unstructured by construction, while genuine action
use is directional, temporally aligned, and spatially concentrated. Probe job
**25143284** (`submit_probe_wan_rt1_information.sh`) runs `--action-analysis`
and `--rollout-action-swap` on the retained Wan × RT-1 checkpoint to settle it,
against the Wan × ACWM chance-level baselines in
[[20260731-wan-action-signal-is-a-global-bag]].

- structure present on RT-1, absent on ACWM ⇒ the data-axis claim survives and
  sharpens.
- at chance on RT-1 too ⇒ the ~0.02 floor is sensitivity **without** control on
  both datasets, the 35× is at least partly a fitting artifact, and this page
  must be rewritten.

## Open / next

- Rollout-action-swap on this checkpoint once ≥3k steps: is RT-1's higher
  sensitivity also *control*, unlike ACWM's?
- SkyReels × RT-1 (job 25112302) — second backbone on the same axis.
- **CORRECTED at 18 evals: Wan × RT-1 SETTLES, it does not keep eroding.**
  After the early peak (0.0332) it holds a flat ~0.021 band for 11 straight
  evals — the first configuration in the campaign that *retains* its
  action-following. Refined law: runs settle to a **data-dependent floor**
  (~0.021 on RT-1 vs ~0.008–0.011 on ACWM), not to zero; erosion is the
  *settling*, and its endpoint is set by what the data pays. SkyReels' steep
  early decline (0.045 → 0.019 in 4 evals) is plausibly the same settling from
  a higher peak — judge after it flattens (Wan also fell 0.033 → 0.017 before
  stabilising). The objective-change argument stands but sharpens: action-CFG /
  rollout losses are for raising the FLOOR, not preventing collapse.

## Related

- [[20260731-wan-tokennorm-nobase-training-results]] — the ACWM counterpart
- [[20260731-wan-action-signal-is-a-global-bag]] — remaining structure problems
- [[20260729-avid-rt1-follows-actions-control]] — the reference on this data
