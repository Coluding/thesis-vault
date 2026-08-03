---
type: experiment
date: 2026-07-30
config: configs/dynamicrafter/diffusion_dc_acwm_robotarm_{arm0_baseline,armA_concat,armB_stride1,armC_concat_stride1}.yaml
commit: uncommitted working tree @ 2026-07-30 (launcher jobs/experiments_cluster/acwm_phys/dc/submit_train_dc_avid_parity.sh)
wandb_run_id: n3dbgq4q (arm0 s0), l2jcz9nx (arm0 s1), hbuu4lwx (armA), 1e0fe9ei (armB), 2us8hugq (armC), t62nhyfu (armD)
ckpt_path: outputs/dc_acwm_robotarm_*/checkpoints/step_00001000.pt (all six, retained after cancel)
status: killed
deliverable: D2
metrics:
  arm0_action_effect_rel: 0.003288
  arm0S_action_effect_rel: 0.003533
  armA_action_effect_rel: 0.003540
  armB_action_effect_rel: 0.003663
  armC_action_effect_rel: 0.004310
  armD_action_effect_rel: 0.004121
  noise_floor_0_vs_0S: 0.000245
  ours_realised_over_rms: 0.0050
  avid_realised_over_rms: 0.238
  ours_cond_over_time_rms: 14.451
  avid_act_over_time_rms: 0.831
  ours_frame_var_ratio_init: 0.0238
  ours_frame_var_ratio_step1000: 0.00344
  ours_cond_rms_init: 0.0276
  ours_cond_rms_step1000: 2.895
notes: "Four parity treatments + two seed controls ALL null. Mechanism found instead: our action embedding is 14x LARGER than the time embedding but 99.7% CONSTANT (realised/RMS 0.0050 vs AVID's 0.238, 48x). Trajectory probe proves the pedestal is LEARNED, not architectural — at init cond/time is 0.182 and 2.4% varies; by step 600 magnitude has grown 106x while the varying fraction collapsed 7x."
---

# DC parity arms all null — the action embedding is a learned pedestal (D2)

> Six arms launched in parallel per
> [[../../50_Decisions/decided/reproduce-avid-on-dc-before-scaling-to-wan]], cancelled
> at ~6.5 h once the readout was clear. Checkpoints retained at step 1000.
> Probes: `scripts/eval_action_sensitivity.py --emb-scale [--timesteps] [--random-init]`,
> and AVID-side `probe_action_sensitivity.py --emb-scale`.

## 1. The parity arms: all null

Treatments off the `c3pcewxk` baseline, each single-variable, with two seed
controls establishing a noise floor.

| arm | change | `effect_rel` | vs control |
|---|---|---|---|
| **0** `n3dbgq4q` | control (add, stride 4) | 0.003288 | — |
| **0S** `l2jcz9nx` | control, seed 1 | 0.003533 | **floor = 0.000245** |
| **A** `hbuu4lwx` | `action_time_combine: concat` | 0.003540 | +0.000129 (inside floor) |
| **B** `1e0fe9ei` | `frame_stride: 1` | 0.003663 | +0.000252 (~1 floor) |
| **C** `2us8hugq` | both | 0.004310 | +0.000899 (~3.7 floor) |
| **D** `t62nhyfu` | both + 320×512 | 0.004121 | +0.000710 (~2.9 floor) |

Target was ≥0.02; AVID reaches **0.029475** on this data. Best arm is **7× short**.
`base_null_violation` exactly 0 on all six.

**The controls earned their GPU time.** Without arm 0/0S there was no noise floor,
and arm C's +0.0009 would have been uninterpretable. They also showed the
historical baseline (`c3pcewxk` 0.004238, a 2×2 probe on a crashed run)
reproduces at 0.0033–0.0035 under the tighter 4×3 probe.

**`concat` was genuinely active** — arms A/C/D have 11,947,397 trainable params vs
11,996,805 for add, a 49,408 difference from resizing the time embedding to
`embed_dim//2`. So arm A is a real null, not an inert flag.

**Treatments did take effect**: `base_loss` separates cleanly by stride
(0.042–0.044 at stride 4 vs 0.028–0.032 at stride 1).

## 2. Blindness is uniform across the noise schedule

Timestep-stratified probe on arm 0 (`--timesteps 100,300,500,700,900`):

| t | 100 | 300 | 500 | 700 | 900 |
|---|---|---|---|---|---|
| shuffle | 0.003291 | 0.003283 | 0.003395 | 0.003264 | 0.003402 |

Flat within ±2%; null 0 in every row. No band of hidden sensitivity — the
aggregate was honest.

**This also validated every prior our-vs-AVID comparison.** Our in-training probe
draws t uniformly (`losses/diffusion.py:82`); AVID's uses a fixed 5-point grid.
Running ours *on their grid* gives 0.0033 vs 0.0034 from uniform — the sampling
difference accounts for nothing, so the 07-28 matrix and 07-29 RT-1 numbers stand.

## 3. The mechanism — a 99.7% constant

Forward hooks on `adapter.module.{time_embed,adapter_condition_proj}` (ours) and
`diffusion_model.{time_embed,action_embed}` (AVID), same definitions both sides:

| | ours (arm 0 @1000) | AVID (@5000) | ratio |
|---|---|---|---|
| embedding ÷ time RMS | **14.45×** | **0.83×** | — |
| per-frame variation | 0.0034 | **0.167** | 49× |
| across-clip variation | 0.0028 | **0.0786** | 28× |
| ‖J‖_F | 1.711 | 3.624 | 2.1× |
| **realised ÷ RMS per element** | **0.0050** | **0.238** | **48×** |

`realised` = ‖J·diag(σ_act)‖ ÷ √C, i.e. how far the embedding moves under typical
action variation, per element. It is dimensionless, so it survives the different
widths (128 vs 64) and magnitudes.

**AVID's action embedding is ~24% action-driven. Ours is 0.5%.** Ours is a large
constant with a signal scratched on it; theirs is a signal.

Cross-check: the directly measured per-frame ratio (0.34%) and the Jacobian route
(0.50%) agree to within 1.5×, with the expected sign (σ_act spans within- and
across-clip variation, per-frame std only within).

**Second defect — misallocation.** Our Jacobian column norms are
`0.740 0.124 0.459 0.533 1.263 0.277 0.445` against input σ
`0.113 0.236 0.064 0.104 0.0115 0.332 0.185`: the **largest** sensitivity sits on
the **lowest**-variance DoF. Max/min column ratio 10.2× (AVID 3.3×, largest on a
mid-variance dim).

## 4. The pedestal is LEARNED, not architectural

Trajectory probe (`--random-init` for step 0; `keep_last_checkpoints: 3` left 600
as the earliest surviving checkpoint):

| | init | step 600 | step 1000 | AVID @5000 |
|---|---|---|---|---|
| cond_proj RMS | **0.0276** | 2.933 | 2.895 | 0.334 |
| cond ÷ time | **0.182** | 16.02 | 14.45 | 0.831 |
| frame-var ratio | **0.0238** | 0.00283 | 0.00344 | 0.167 |
| ‖J‖_F | 0.0568 | 1.629 | 1.711 | 3.624 |

**At init the architecture is fine** — output smaller than the time embedding
(0.182×, same side of 1 as AVID's 0.831×), 2.4% varying per frame.

**Training builds the pedestal.** By step 600 magnitude has grown **106×** while
the varying fraction *collapsed* 7× (0.0238 → 0.0028). The Jacobian grew ~30×, so
the encoder *is* becoming more action-sensitive — the constant simply grows ~3.5×
faster.

**Interpretation (analysed):** the encoder is being used as a **bias generator**.
A large constant added into every ResBlock's `emb` helps fit the denoising
objective without using actions — the mechanism behind the 2026-07-21 verdict of
"pure action-independent domain adjustment"
([[20260721-replace-fix-validation-sigma-sweep-action-probe]]).

**This retires the encoder-architecture hypothesis.** Narrow/shallow encoder arms
were premised on depth creating the pedestal. It doesn't — a 2-layer encoder
would start from the same healthy place and could inflate identically. The
85×-parameter and 6-vs-2-layer asymmetries are real but are no longer the natural
explanation, and whether AVID's shallowness *prevents* inflation is now a separate
open question.

## Backward direction (weaker, kept for completeness)

`condition_grad_norm` ÷ `adapter_grad_norm`, per-parameter RMS: ours **0.0124**
(arm 0) / **0.0316** (arm 0S — 2.75× seed spread on the raw norm), AVID
**0.0903** (`5e4m9dxz`, step 1014, matched). So AVID is **3–7× higher**, direction
consistent but straddling the pre-registered bands (≈0.012 kills / 0.2–1.0
confirms). Total action-pathway gradient is nearly identical (3.18e-4 vs 3.11e-4);
the difference is dilution across 85× more parameters. Both use AdamW at lr 1e-4,
which normalises per-parameter updates, so the *ratio* is the defensible reading,
not the absolute. **The forward measurement (48×) is the load-bearing one.**

## Corrections to earlier claims in this investigation

- The gradient deficit is **~80× per parameter**, not the ~300× the raw L2 norms
  suggested — norms over 792,576 vs 11,204,229 params are not comparable.
- The probe's printed `action-driven fraction ~0.088` is a **bad metric**
  (‖J‖_F × *mean* σ, ignoring per-dim weighting); properly weighted it is ~0.3–0.5%,
  matching the measured variance. Should be removed from the script.
- "octo does per-dim action standardisation, we don't" applies to the **RT-1 path
  only**. On ACWM Robot Arm, AVID's datamodule passes actions through raw
  (`acwm.py:181`), exactly as we do — so normalisation is *not* a
  reference-vs-ours difference here. It stands only on our own 29× σ spread.

## Next

- **Intervention: stop the encoder being a bias source.** Centre/normalise
  `cond_emb` before it enters `emb` so an input-independent component cannot
  inflate. → [[../../20_Tickets/experiments/exp-conditioning-decouple-encoder-bias]]
- **Hold** the narrow/shallow encoder arms (§4).
- Per-dim action standardisation for the misallocation (§3), independent of the above.
- Open: why AVID's stays at 0.83× while ours runs to 14×.

## Related

- [[20260730-avid-robotarm-follows-actions-recipe-not-data]] — the reference result
- [[20260728-acwm-robotarm-matrix-action-blind]] · [[20260721-replace-fix-validation-sigma-sweep-action-probe]]
- [[../tech/avid-vs-ours-action-conditioning]]
- [[../../50_Decisions/decided/reproduce-avid-on-dc-before-scaling-to-wan]]
