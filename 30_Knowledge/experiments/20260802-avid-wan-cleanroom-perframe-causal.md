---
type: experiment
date: 2026-08-02
config: external_repos/avid/wan_diffusion/configs/train/avid/avid_wan_rt1.yaml (arms via ARM=faithful|pooled)
commit: uncommitted working tree @ 2026-08-02 (external_repos/ is gitignored; branch is new)
wandb_run_id: avid_wan_rt1_47M_faithful / avid_wan_rt1_47M_pooled (project avid-wan-rt1)
ckpt_path: /scratch-shared/lbierling/avid-wan-rt1/avid_wan_rt1_47M_{faithful,pooled}/checkpoints/epoch=19-step=5000.ckpt
status: completed
deliverable: D2
metrics:
  faithful_effect_rel_shuffle: 0.017474
  faithful_effect_rel_shuffle_sem: 0.001531
  faithful_effect_rel_zero: 0.013930
  faithful_action_driven_share: 0.1532
  faithful_adapter_rel_contribution: 0.114054
  avid_ref_step5000_effect_rel_shuffle: 0.012541
  avid_ref_step5000_adapter_rel_contribution: 0.051383
  avid_ref_step5000_mask_mean: 0.906970
  avid_ref_step5000_action_driven_share: 0.2441
  pooled_effect_rel_shuffle: 0.010192
  pooled_effect_rel_shuffle_sem: 0.001587
  pooled_effect_rel_zero: 0.008706
  pooled_action_driven_share: 0.0914
  pooled_adapter_rel_contribution: 0.111481
  ratio_shuffle: 1.71
  welch_t_shuffle: 3.30
  base_null_violation: 0.0
  faithful_diag_concentration: 0.3900
  pooled_diag_concentration: 0.1987
  diag_concentration_chance: 0.2000
  step10000_faithful_effect_rel_shuffle: 0.027641
  step10000_pooled_effect_rel_shuffle: 0.011239
  step10000_faithful_action_driven_share: 0.2299
  step10000_pooled_action_driven_share: 0.0910
  step10000_faithful_diag_concentration: 0.4089
  step10000_pooled_diag_concentration: 0.1980
  step10000_ratio_shuffle: 2.46
  step10000_welch_t_shuffle: 7.34
  step12000_faithful_effect_rel_shuffle: 0.031779
  step12000_pooled_effect_rel_shuffle: 0.012788
  step12000_faithful_action_driven_share: 0.2529
  step12000_pooled_action_driven_share: 0.1019
  step12000_faithful_adapter_rel_contribution: 0.125675
  step12000_pooled_adapter_rel_contribution: 0.125455
  step12000_ratio_shuffle: 2.49
  step12000_welch_t_shuffle: 10.5
notes: "Single-variable A/B inside an AVID clean-room on a frozen Wan2.2 TI2V-5B base. (1) Per-frame action addressability is CAUSAL: 1.7x more action-driven contribution at matched depth, matched adapter share, clean null, Welch t=3.30. (2) At MATCHED step 5000 the faithful arm EXCEEDS the AVID/DynamiCrafter reference (0.01747 vs 0.01254) -- the often-quoted 0.0495 is a step-15000 number, so Wan is NOT the harder substrate. (3) What differs is purity (AVID 24.4% action-driven vs ours 15.3%), not size."
---

# Per-frame conditioning is causal on Wan — and Wan beats AVID's own base at matched depth (D2)

> First result from the AVID clean-room third branch
> ([[../../20_Tickets/experiments/exp-adapter-avid-wan-cleanroom-rt1]]).
> **Runs complete.** Both arms ended on their 6 h walltime (Slurm `TIMEOUT`,
> exit 0:0, no errors): faithful reached step 14000, pooled 13000. Measurements
> below are at matched steps 5000 / 10000 / 12000.

## Setup

`external_repos/avid/wan_diffusion/` — a third branch of the official AVID repo
beside `pixel_diffusion` and `latent_diffusion`. AVID's composition, mask, EMA,
freezing, optimiser and **their own RT-1 datamodule imported unmodified**; only
the frozen base changes (Wan2.2 TI2V-5B, rectified flow, diffusion forcing).

Two arms, identical in every respect except one switch:

| arm | action conditioning |
|---|---|
| **A `faithful`** | per-frame action → MLP → **concat** into a reserved half of the modulation (AVID's scheme) |
| **B `pooled`** | the same, but the action is **mean-pooled over latent frames** and broadcast — reproducing our framework's `ce.mean(dim=1)` + global AdaLN |

## Result — step 5000, 16 batches (240 samples) per arm, checkpoint pinned

| | arm A `faithful` | arm B `pooled` |
|---|---|---|
| `adapter_rel_contribution` | 0.1141 | 0.1115 |
| `mask mean` | 0.9534 | 0.9536 |
| `action_effect_rel` shuffle | **0.017474** ± 0.001531 | 0.010192 ± 0.001587 |
| `action_effect_rel` zero | **0.013930** ± 0.000953 | 0.008706 ± 0.001010 |
| **action-driven share** | **15.3%** | 9.1% |
| `base_null_violation` | 0.000e+00 | 0.000e+00 |

Welch t on the batch means: **3.30** (shuffle, p ≈ 0.002), **3.76** (zero,
p ≈ 0.0007), n = 16 batches per arm.

Sources: probes `25148083` (faithful) / `25148084` (pooled), both on
`epoch=19-step=5000.ckpt`; arm identity verified in-log
(`action_temporal_pool=False` vs `True` — the switch changes no parameter
shapes, so a mismatched load would have been silent).

## Reading

**1. Per-frame addressability is causal.** The arms are matched on data,
capacity, objective, depth — *and* on adapter contribution and mask, so they
differ only in **what** the adapter contributes, not how much. The faithful
arm's contribution is **1.7x more action-driven**. This converts divergences §2
and §3 of [[../tech/avid-vs-ours-wan-action-conditioning]] from a plausible
story into a measured cause.

**2. At MATCHED DEPTH the clean-room on Wan EXCEEDS the AVID reference.**

⚠️ **Correction made while writing this note.** The widely quoted reference
`action_effect_rel = 0.0495` (`93qrvr5v`) is measured at **step 15000**
(`epoch=14-step=15000.ckpt`, [[20260729-avid-rt1-follows-actions-control]]), not
5000. Comparing our step-5000 arms against it is a 3x training-depth mismatch —
the same error already flagged for the step-500 point, repeated. The AVID
reference was therefore re-probed **at step 5000** on its own checkpoint
(`epoch=4-step=5000.ckpt`, job `25148170`, same probe, 120 samples, null 0):

| @ step 5000 | AVID / DynamiCrafter | **faithful** / Wan | pooled / Wan |
|---|---|---|---|
| `action_effect_rel` shuffle | 0.012541 | **0.017474** | 0.010192 |
| `action_effect_rel` zero | 0.010474 | **0.013930** | 0.008706 |
| `adapter_rel_contribution` | 0.051383 | 0.114054 | 0.111481 |
| `mask mean` | 0.906970 | 0.953402 | 0.953613 |
| action-driven share | **24.4%** | 15.3% | 9.1% |
| `base_null_violation` | 0 | 0 | 0 |

**The clean-room faithful arm beats AVID-on-DynamiCrafter by 1.39x on the
headline metric at the same step.** The Wan latent space is therefore *not* the
harder substrate — hypothesis (b) is dead.

**3. What differs is the CHARACTER of the contribution, not its size.** AVID's
adapter contributes **less** in total (0.051 vs 0.114) but what it contributes is
**purer**: 24.4% action-driven vs 15.3%. Its mask also opens further (0.907 vs
0.953). Ours is a *larger but more diluted* contribution. Raising purity — not
raising contribution — is the lever that remains.

**4. Neither is at its ceiling.** AVID goes 0.0125 (5k) → 0.0495 (15k): a 4x gain
from depth alone. Our arms were at step 6500 and still rising when this was
written (6 h walltime should reach ~12000). Statements about final levels must
wait for matched deep checkpoints.

## Temporal control — the gain-vs-information confound, resolved

`--localisation` mode (jobs `25148466` / `25148467`, both pinned to step 5000,
6 batches): perturb **one latent frame's actions at a time** and record which
frames respond. Unlike `action_effect_rel`, the diagonal concentration is
**invariant to overall action-path gain** — it asks whether the model knows
*which* frame an action belongs to.

**arm A `faithful`** — diagonal concentration **0.3900** (chance 0.2000):

```
          resp f0  resp f1  resp f2  resp f3  resp f4
  pert f0   0.0472   0.0095   0.0087   0.0080   0.0083
  pert f1   0.0865   0.0517   0.0173   0.0147   0.0151
  pert f2   0.0536   0.0149   0.0516   0.0173   0.0146
  pert f3   0.0486   0.0102   0.0139   0.0543   0.0206
  pert f4   0.0522   0.0101   0.0117   0.0154   0.0562
```

**arm B `pooled`** — diagonal concentration **0.1987** (chance 0.2000):

```
          resp f0  resp f1  resp f2  resp f3  resp f4
  pert f1   0.0466   0.0110   0.0114   0.0120   0.0131
  pert f2   0.0466   0.0110   0.0114   0.0120   0.0131
  pert f3   0.0466   0.0110   0.0114   0.0120   0.0131
  pert f4   0.0466   0.0110   0.0114   0.0120   0.0131
```

**The pooled arm's four rows are bit-identical.** Perturbing frame 1 and frame 3
produce exactly the same output change, because mean-pooling maps them to the
same vector. Its diagonal concentration is chance to three decimals.

⇒ **The 1.7x `effect_rel` gap is information, not gain.** A gain difference
cannot produce identical rows on one side and a 3–5x diagonal (0.052–0.056 on
the diagonal vs 0.010–0.021 off it, among predicted frames) on the other. This
is the trained-model counterpart of the init-time 261x-vs-1.0x localisation
test, and it closes the caveat that `effect_rel` alone could not.

*Reading note:* the `resp f0` column is elevated for every perturbation in both
arms. Latent frame 0 is the clean observation frame, held at timestep 0 and
**masked out of the loss**, so its output is unconstrained. That inflates the
off-diagonal mass, making 0.3900 a conservative floor for the faithful arm.

## Depth trajectory 5000 → 10000 — the pooled arm hits an information ceiling

Both arms re-probed at **step 10000**, pinned and matched (magnitude jobs
`25149407`/`25149409`, 16 batches; temporal control `25149408`/`25149410`).
Adapter contribution and mask remain matched (0.1202 vs 0.1235; 0.9468 vs
0.9411), so the comparison stays clean.

| | step 5000 | step 10000 | change |
|---|---|---|---|
| **faithful** `action_effect_rel` shuffle | 0.017474 | **0.027641** ± 0.001871 | **+58%** |
| **pooled** `action_effect_rel` shuffle | 0.010192 | 0.011239 ± 0.001219 | +10% |
| ratio | 1.71x | **2.46x** | widening |
| **faithful** action-driven share | 15.3% | **23.0%** | climbing |
| **pooled** action-driven share | 9.1% | 9.1% | **flat** |
| **faithful** diagonal concentration | 0.3900 | **0.4089** | strengthening |
| **pooled** diagonal concentration | 0.1987 | 0.1980 | **pinned at chance** |

Welch t on the batch means at step 10000: **7.34** (shuffle).

The pooled arm's temporal-control rows are **still bit-identical** after 10000
steps:

```
  pert f1   0.0524   0.0129   0.0141   0.0139   0.0166
  pert f2   0.0524   0.0129   0.0141   0.0139   0.0166
  pert f3   0.0524   0.0129   0.0141   0.0139   0.0166
  pert f4   0.0524   0.0129   0.0141   0.0139   0.0166
```

### Final matched point — step 12000 (jobs `25149903` / `25149904`)

Adapter contributions are now **essentially identical** (0.125675 vs 0.125455,
0.2% apart) — the cleanest control of the series.

| step | faithful `effect_rel` | pooled `effect_rel` | ratio | faithful share | pooled share |
|---|---|---|---|---|---|
| 5000 | 0.017474 | 0.010192 | 1.71x | 15.3% | 9.1% |
| 10000 | 0.027641 | 0.011239 | 2.46x | 23.0% | 9.1% |
| **12000** | **0.031779** ± 0.001432 | 0.012788 ± 0.001105 | **2.49x** | **25.3%** | 10.2% |

Welch t at step 12000: **10.5**. Over the range faithful grew **+82%**; pooled
+25%, with its action-driven share essentially static (9.1% → 10.2%).

**Faithful's 25.3% action-driven share now EXCEEDS the AVID/DynamiCrafter
reference's 24.4%**, and its `effect_rel` 0.0318 @12000 is closing on AVID's
0.0495 @15000 — on the backbone that was suspected of being the problem.

### This is the clearest statement of the finding

**The faithful arm learns; the pooled arm cannot.** Doubling training moved the
faithful arm's action-following by 58% and *strengthened* its frame addressing;
it moved the pooled arm's not at all and left its addressing exactly at chance.
The pooled arm is not under-trained — it is against an **information ceiling**,
because mean-pooling destroys the frame correspondence *before the network sees
it*. No amount of optimisation recovers information that is not in the input.

That is the mechanism our own Wan framework has been sitting behind all along.

**Scale check:** faithful's 23.0% action-driven share at step 10000 has
essentially caught the AVID/DynamiCrafter reference's **24.4%** (measured at
step 5000), on a backbone whose 4x-temporally-compressed latent space was the
suspected culprit.

## Caveats

- Final depths differ (faithful 14000, pooled 13000) because the arms ran at slightly different rates; every *comparison* here is at a pinned, matched step.
- An earlier n=4 probe of the same checkpoint gave 1.9x; the n=16 rerun gives
  1.68x. **Quote the n=16 number** — the small-n estimate was noise-inflated.
- Cross-framework comparison to our own `5w72bo01` (~0.021) is loose: different
  probe harness, different config. The rigorous comparison is arm A vs arm B.
- `condition_adapter_on_base_outputs: True` is kept at AVID's default in both
  arms, so the base-oracle dynamics our own campaign documented are present here
  too and are not controlled for.
- ⚠️ **The `effect_rel` gain-vs-information caveat applies here too.** As flagged
  on our own runs ([[../experiments/_index]] rows for `5w72bo01` and the DC
  `condition_center` arms), `effect_rel` is monotone in action-path *gain*, so a
  higher value need not mean more action *information*. This A/B has a partial
  defence the others lack — the arms are matched on `adapter_rel_contribution`
  (0.114 vs 0.111), so total output gain is equalised — but mean-pooling could
  still lower the action-path gain specifically. The clean discriminator is a
  **control/steering** measurement (does swapping actions change the *right*
  frames), not a magnitude one. That is the next thing to run, and it is exactly
  what the init-time localisation test does for an untrained model.

## Related

- [[../tech/avid-vs-ours-wan-action-conditioning]] — the five divergences; §2/§3 are what this tests
- [[../../20_Tickets/experiments/exp-adapter-avid-wan-cleanroom-rt1]] — the ticket, full trajectory
- [[../../20_Tickets/feat-adapter-wan-per-frame-adaln]] — the fix this justifies (was proposed-for-close)
- [[20260731-wan-action-signal-is-a-global-bag]] — the signature this arm B reproduces deliberately
- [[20260729-avid-rt1-follows-actions-control]] — the 0.0495 reference (**step 15000**; re-probed here at 5000 as 0.01254)
