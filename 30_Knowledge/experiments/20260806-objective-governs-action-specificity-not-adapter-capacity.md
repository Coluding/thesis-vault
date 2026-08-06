---
type: experiment
date: 2026-08-06
config: configs/easyanimate/diffusion_ea_v5_tokennorm_nobase_ADD_acwm_robotarm.yaml
        vs configs/easyanimate/flow_ea_v51_tokennorm_nobase_ADD_acwm_robotarm.yaml
commit: uncommitted working tree @ 2026-08-06 (EasyAnimate integration)
wandb_run_id: project coluding/EasyAnimate-objective-acwm-robotarm
              — V5 slurm 25240257, V5.1 slurm 25241732
ckpt_path: /home/lbierling1/generative-flow-adapters/outputs/acwm-robotarm-EA-V5-tokennorm-nobase-ADD-run/checkpoints/
           and .../acwm-robotarm-EA-V51-tokennorm-nobase-ADD-run/checkpoints/
status: completed (V5); V5.1 running out its wall
deliverable: D2
metrics:
  matched_step: 14200
  v5_loss_reduction_vs_frozen_base: -0.838
  v51_loss_reduction_vs_frozen_base: -0.687
  v5_adapter_rel_contribution: 0.51673
  v51_adapter_rel_contribution: 0.49147
  v5_action_effect_rel: 0.05998
  v51_action_effect_rel: 0.03866
  v5_action_effect_vs_adapter: 0.11623
  v51_action_effect_vs_adapter: 0.07932
notes: "FINAL at matched step 14,200. (1) The Wan ceiling was NOT intrinsic to output
  adapters: the same adapter family cuts denoising loss 83.8% (diffusion) / 68.7% (flow)
  vs the frozen base and contributes ~0.5 of the prediction, against -3.3% and 0.047 on
  the best Wan arm; adapter_base_cosine 0.87 vs Wan's 0.9989. (2) The objective's effect
  is far larger on ACTION-SPECIFICITY (effect_rel +55%, effect_vs_adapter +47%) than on
  raw contribution (+5%). Diffusion led all 15 hourly evals and is ~4x more stable.
  Confounded: V5/V5.1 differ in text stack; n=1 per arm."
---

# The objective governs action-specificity, not adapter capacity (D2)

✅ **FINAL, AT MATCHED STEP 14,200.** V5 reached its 17 h wall (`TIMEOUT` at 17:00:07 —
the SBATCH limit, not a failure) with its last eval at step 14,200; V5.1 reached the
same step while still running. Both numbers below are that step. Direction held for
**fifteen consecutive hourly evals** with diffusion never once behind.

| | V5 diffusion | V5.1 flow | diffusion |
|---|---|---|---|
| base loss (frozen) | 0.22594 | 0.47049 | |
| adapted `eval_loss` | **0.03650** | **0.14746** | |
| **loss reduction** | **−83.8%** | **−68.7%** | **+15 pts** |
| `adapter_rel_contribution` | 0.5167 | 0.4915 | +5% |
| `adapter_base_cosine` | 0.8678 | 0.8827 | |
| **`action_effect_rel`** | **0.05998** | **0.03866** | **+55%** |
| **`action_effect_vs_adapter`** | **0.11623** | **0.07932** | **+47%** |
| `adapter_out_const_frac` | 0.0318 | 0.0090 | |
| `action_base_null_violation` | 0.00000 | 0.00000 | clean |

**Stability differs too, and it is part of the result.** Over the final five evals
diffusion held 0.0524–0.0600 (spread ±0.001 over three consecutive) while flow swung
0.0296–0.0444 — roughly **4× the variance**. Report the spread, not just the mean.

## Result 1 — the Wan ceiling was not intrinsic to output adapters

The same adapter family (34.9M output adapter, `composition: add`, cross-attention
action injection, `condition_on_base_outputs: false`) on a different frozen base:

| run | base loss | adapted | **Δ** | `adapter_base_cosine` | `rel_contrib` |
|---|---|---|---|---|---|
| Wan action-only + `add` (`25192286`) | 0.13362 | 0.13262 | **−0.75%** | 0.9997 | 0.023 |
| Wan token-norm + `add` (`25192313`) | 0.13362 | 0.12917 | **−3.3%** | 0.9989 | 0.047 |
| **EA V5 diffusion** (`25240257`) | 0.22594 | **0.03650** | **−83.8%** | 0.868 | **0.517** |
| **EA V5.1 flow** (`25241732`) | 0.47049 | **0.14746** | **−68.7%** | 0.883 | **0.491** |

On Wan, `adapter_base_cosine` 0.9989 means the adapted prediction was *numerically
almost the frozen base* — the adapter was cosmetic. On EasyAnimate it reshapes the
prediction substantially. **This directly contradicts the reading the Wan campaign was
heading toward** ([[20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]),
which risked concluding that output adapters simply cannot move a frozen video base.

## Result 2 — where the objective actually bites

At matched step 14,200 (final):

| | V5 diffusion | V5.1 flow | diffusion lead |
|---|---|---|---|
| loss reduction | −83.8% | −68.7% | **+15 pts** |
| `rel_contribution` | 0.5167 | 0.4915 | +5% |
| `action_effect_rel` | 0.05998 | 0.03866 | **+55%** |
| `action_effect_vs_adapter` | 0.11623 | 0.07932 | **+47%** |

⚠ **Revision vs the interim reading.** At the earlier (unmatched) 9000/8200 point the two
objectives looked *tied* on capacity (−74.9% vs −73.6%). At matched 14,200 they are not:
diffusion pulls ahead on loss reduction too (+15 points). The clean statement is that
**the objective's effect on action-specificity (+55% / +47%) is much larger than its
effect on raw contribution (+5%)** — not that capacity is unaffected. The earlier
"capacity is tied" phrasing was an artifact of comparing arms 800 steps apart.

**Action-specificity is where the objective bites hardest.** `effect_vs_adapter` is the cleaner
measure — it asks what fraction of the adapter's *own* output is action-driven, so it
is not inflated by the adapter simply being large.

Interpretation: the objective does not govern *how much* an adapter can reshape a
frozen base — it governs *how much of that reshaping is action-conditioned*.

## The cross-backbone pattern (why this is more than one pair)

| backbone | objective | action-following |
|---|---|---|
| DynamiCrafter 1.4B | diffusion | present — steering +0.117, alignment 1.000 |
| EasyAnimate V5 7B | diffusion | `effect_rel` **0.060** |
| EasyAnimate V5.1 7B | flow | `effect_rel` **0.039** |
| Wan2.2-TI2V 5B | flow | `effect_rel` 0.002–0.011 |

Both diffusion backbones sit above both flow backbones, across **independent model
families** — n=2 per objective rather than a single pair. That is materially harder to
dismiss than the EasyAnimate comparison alone.

⚠ **But it is "flow is weaker", not "flow fails".** EA-flow (0.039) is ~4× Wan-flow, so
the backbone clearly matters too. Objective and backbone are not separable from these
four points alone.

## Confounds — state these, do not bury them

- **EA vs Wan** differs in backbone, VAE, objective *and* text conditioning.
- **V5 vs V5.1** share a video backbone but **not a text backbone** (BERT+T5 vs
  Qwen2VL). CFG steers *through* text, so this is not peripheral — see
  [[../../20_Tickets/bug-diffusers-silently-drops-vae-weights]] for the four distinct
  crashes this caused.
- Per-eval `effect_rel` swung 20–80% mid-run; the final figures are single matched-step
  readings, not means over draws. Diffusion's last three evals were tight (±0.001), flow's
  were not (0.0296–0.0444) — quote the spread.
- n=1 per arm.

## What is still open

Only **8–12%** of the adapter's large contribution is action-driven
(`effect_vs_adapter` 0.116 diffusion / 0.079 flow — both roughly doubled over the run). So "the adapter is powerful" is established;
"powerful **at action conditioning**" is not. The domain-corrector reading from
[[20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]] may still hold —
now at ~10× the operating scale.

## Provenance note

These numbers only exist because the base was fixed first. Before 2026-08-05 the
EasyAnimate base was rendering **noise** while every shape/finiteness/file-existence
check passed; it was caught by the user looking at one frame. Three stacked defects:
zero text context instead of a real embedding (absmax 21), diffusers' pipeline dropping
the entire T5 stream, and CFG forced to 1.0. All `effect_rel` numbers logged before that
fix are **void**. Full account in [[../../10_now/compute-spend-ledger]].

## Related

- [[20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]] — the reading this
  revises
- [[20260803-concat-injection-does-not-help]] — injection site was not the constraint
- [[../../10_now/compute-spend-ledger]] — the debugging account and SBU cost
