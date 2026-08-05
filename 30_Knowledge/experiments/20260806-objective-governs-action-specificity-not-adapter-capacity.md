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
status: running
deliverable: D2
metrics:
  v5_diffusion_step: 9000
  v51_flow_step: 8200
  v5_loss_reduction_vs_frozen_base: -0.749
  v51_loss_reduction_vs_frozen_base: -0.736
  v5_adapter_rel_contribution: 0.521
  v51_adapter_rel_contribution: 0.484
  v5_action_effect_rel: 0.03893
  v51_action_effect_rel: 0.02861
  v5_action_effect_vs_adapter: 0.07559
  v51_action_effect_vs_adapter: 0.05984
notes: "TWO findings. (1) The Wan ceiling was NOT intrinsic to output adapters: on
  EasyAnimate the same adapter family cuts denoising loss ~74% vs the frozen base and
  contributes ~50% of the prediction, against -3.3% and 0.047 on the best Wan arm.
  (2) The training objective governs ACTION-SPECIFICITY, not adaptation capacity —
  the two objectives adapt equally well (-74.9% vs -73.6%) but diffusion leads on
  effect_rel (+36%) and effect_vs_adapter (+26%). NOT YET FINAL: arms still running,
  steps not exactly matched (9000 vs 8200)."
---

# The objective governs action-specificity, not adapter capacity (D2)

⚠ **Interim — both arms still running.** Numbers below are the latest eval at
**V5 step 9000 / V5.1 step 8200**, i.e. *not* exactly matched. The quotable figure is
the end-of-run matched-step value with spread across eval draws. Recorded now because
the finding is stable across eight consecutive hourly evals.

## Result 1 — the Wan ceiling was not intrinsic to output adapters

The same adapter family (34.9M output adapter, `composition: add`, cross-attention
action injection, `condition_on_base_outputs: false`) on a different frozen base:

| run | base loss | adapted | **Δ** | `adapter_base_cosine` | `rel_contrib` |
|---|---|---|---|---|---|
| Wan action-only + `add` (`25192286`) | 0.13362 | 0.13262 | **−0.75%** | 0.9997 | 0.023 |
| Wan token-norm + `add` (`25192313`) | 0.13362 | 0.12917 | **−3.3%** | 0.9989 | 0.047 |
| **EA V5 diffusion** (`25240257`) | 0.24290 | **0.06098** | **−74.9%** | 0.868 | **0.521** |
| **EA V5.1 flow** (`25241732`) | 0.44620 | **0.11759** | **−73.6%** | 0.886 | **0.484** |

On Wan, `adapter_base_cosine` 0.9989 means the adapted prediction was *numerically
almost the frozen base* — the adapter was cosmetic. On EasyAnimate it reshapes the
prediction substantially. **This directly contradicts the reading the Wan campaign was
heading toward** ([[20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]),
which risked concluding that output adapters simply cannot move a frozen video base.

## Result 2 — where the objective actually bites

| | V5 diffusion | V5.1 flow | diffusion lead |
|---|---|---|---|
| loss reduction | −74.9% | −73.6% | **~0 (tied)** |
| `rel_contribution` | 0.521 | 0.484 | +8% |
| `action_effect_rel` | 0.03893 | 0.02861 | **+36%** |
| `action_effect_vs_adapter` | 0.07559 | 0.05984 | **+26%** |

**Capacity is tied; action-specificity is not.** `effect_vs_adapter` is the cleaner
measure — it asks what fraction of the adapter's *own* output is action-driven, so it
is not inflated by the adapter simply being large.

Interpretation: the objective does not govern *how much* an adapter can reshape a
frozen base — it governs *how much of that reshaping is action-conditioned*.

## The cross-backbone pattern (why this is more than one pair)

| backbone | objective | action-following |
|---|---|---|
| DynamiCrafter 1.4B | diffusion | present — steering +0.117, alignment 1.000 |
| EasyAnimate V5 7B | diffusion | `effect_rel` 0.039 |
| EasyAnimate V5.1 7B | flow | `effect_rel` 0.029 |
| Wan2.2-TI2V 5B | flow | `effect_rel` 0.002–0.011 |

Both diffusion backbones sit above both flow backbones, across **independent model
families** — n=2 per objective rather than a single pair. That is materially harder to
dismiss than the EasyAnimate comparison alone.

⚠ **But it is "flow is weaker", not "flow fails".** EA-flow (0.029) is ~3× Wan-flow, so
the backbone clearly matters too. Objective and backbone are not separable from these
four points alone.

## Confounds — state these, do not bury them

- **EA vs Wan** differs in backbone, VAE, objective *and* text conditioning.
- **V5 vs V5.1** share a video backbone but **not a text backbone** (BERT+T5 vs
  Qwen2VL). CFG steers *through* text, so this is not peripheral — see
  [[../../20_Tickets/bug-diffusers-silently-drops-vae-weights]] for the four distinct
  crashes this caused.
- Steps not yet matched (9000 vs 8200); per-eval `effect_rel` swings 20–80% between
  consecutive evals even as the *direction* held for eight straight checks.
- n=1 per arm.

## What is still open

Only **6–8%** of the adapter's large contribution is action-driven
(`effect_vs_adapter` 0.076 / 0.060). So "the adapter is powerful" is established;
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
