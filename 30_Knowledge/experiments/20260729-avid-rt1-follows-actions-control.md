---
type: experiment
date: 2026-07-29
config: external_repos/avid/latent_diffusion/configs/train/avid/avid_11M.yaml (RTXDataModule, RT-1 fractal20220817_data)
commit: uncommitted probe (external_repos gitignored)
wandb_run_id: 93qrvr5v (avid-rt1-official training run)
ckpt_path: outputs/avid_rt1_11M/checkpoints/epoch=14-step=15000.ckpt (cluster)
status: completed
deliverable: D2
metrics:
  action_effect_rel_shuffle: 0.0495
  action_effect_rel_zero: 0.0443
  cos_true_variant: 0.997
  base_null_violation: 0.0
  adapter_rel_contribution: 0.0755
  gate_mask_mean: 0.905
  effect_over_adapter_ratio: 0.66
notes: "The DECISIVE control: the ORIGINAL AVID recipe on its OWN in-distribution real data (RT-1) DOES follow actions (~10x our ACWM runs; ~66% of the adapter's contribution is action-driven vs our ~5%). -> the ACWM action-blindness is a DATA/OOD problem, not the recipe or the probe."
---

# AVID on RT-1 follows actions — the control that says ACWM blindness is a data problem (D2)

> **⚠ CONCLUSION SUPERSEDED 2026-07-30.** The *measurement* below stands
> (effect_rel 0.0495, null 0). The *inference* — "the blindness is a DATA/OOD
> problem, not the recipe" — does not. It rested on comparing this clean
> full-data real-world run against the 64-clip memorization-confounded Push Cube
> smoke (`423pjv8y`), the only synthetic-side AVID datapoint then available.
> With the clean synthetic cell filled, the same recipe **also follows actions on
> synthetic ACWM Robot Arm** (effect_rel 0.029475, null 0) where our adapters are
> blind — so the gap is **our implementation**, not the data. See
> [[20260730-avid-robotarm-follows-actions-recipe-not-data]].

> Probe: `external_repos/avid/latent_diffusion/scripts/probe_action_sensitivity.py`
> (`--config avid_11M.yaml --ckpt-dir .../avid_rt1_11M/checkpoints --dataset-dir
> $RTX_DATA_DIR`), reproducing our `eval_action_effect_rel` on
> `AVIDAdapter.apply_model` (v-prediction). 120 samples (8 batches × 5 timesteps
> × 3 paired draws). Full RT-1, 15000 steps — NOT the 64-clip smoke confound.

## Result

| metric | value |
|---|---|
| `action_effect_rel` shuffle | **0.0495** |
| `action_effect_rel` zero | 0.0443 |
| cos(true, variant) | 0.997 |
| **`base_null_violation`** | **0.000** (PASS — trustworthy) |
| `adapter_rel_contribution` | 0.0755 |
| gate/mask mean | 0.905 (base-heavy) |
| **effect ÷ adapter contribution** | **≈ 0.66** |

## vs our ACWM runs (all action-blind)

| run | action_effect_rel | effect ÷ adapter |
|---|---|---|
| **AVID RT-1** `93qrvr5v` | **0.0495** | **0.66** |
| Wan · ACWM Arm `ncztxyyo` | 0.0056 | 0.056 |
| DC · ACWM Arm `c3pcewxk` | 0.0034 | — |
| SkyReels · ACWM Arm `8zjjn7wl` | 0.0013 | — |

AVID-RT1 is ~9–38× more action-sensitive, and of what its adapter *does*, ~66% is
action-driven (vs Wan's ~5%). Null control exactly 0 on both.

## Interpretation (analysed)

**The action-blindness is the data, not the recipe or the probe.** The same
family of approach (frozen base + trained action-conditioned correction) follows
actions on **in-distribution real-world RT-1** while collapsing to
action-independence on **OOD synthetic ACWM** (flat Push Cube, synthetic Robot
Arm). Corrects the earlier confounded read
([[20260728-acwm-robotarm-matrix-action-blind]] — that AVID smoke was 64-clip
memorization). Directly motivates running OUR adapters on RT-1 / OpenVid.

**Honest caveats:** (a) AVID's recipe is a *full separate action UNet*, not our
lightweight output adapter — this proves the data/setup supports action-following;
our-adapter-on-RT-1 is the direct test. (b) The absolute effect (0.05) is modest
(gate 0.905 base-heavy) — real and above the blind floor, but not dramatic.

## Next

- **Run our Wan/DC/SkyReels adapters on RT-1** (translator built:
  `data/translators/rt1.py`, `--dataset rt1`) — do they follow actions where ACWM
  failed? The direct our-adapter-in-distribution test.
- Consider the RT-1 per-dim action std-normalization (octo does it; ours doesn't)
  before trusting a weak our-adapter result.
