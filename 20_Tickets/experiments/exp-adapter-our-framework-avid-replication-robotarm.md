---
type: exp
scope: adapter
status: in-progress
priority: high
created: 2026-07-30
updated: 2026-08-01
resolution:
resolution_note:
closed_at:
related: ["[[../../50_Decisions/decided/reproduce-avid-on-dc-before-scaling-to-wan]]", "[[../../30_Knowledge/experiments/20260730-avid-robotarm-follows-actions-recipe-not-data]]", "[[../../30_Knowledge/experiments/20260728-acwm-robotarm-matrix-action-blind]]", "[[../../30_Knowledge/tech/avid-vs-ours-action-conditioning]]", "[[../../30_Knowledge/related-work/avid]]", "[[../bug-adapter-gate-saturation-mask-mix]]"]
---

> **Governed by [[../../50_Decisions/decided/reproduce-avid-on-dc-before-scaling-to-wan]]**
> (2026-07-30): reproduce AVID with our code on DC first; Wan action-conditioning
> is paused until this hits `action_effect_rel` ≳0.02 with a clean null.

# exp: our framework vs the AVID reference, same substrate (D1/D2)

## Why

[[../../30_Knowledge/experiments/20260730-avid-robotarm-follows-actions-recipe-not-data]]
established that the **unmodified AVID recipe follows actions on ACWM Robot Arm**
(effect_rel 0.029475, null 0) where **our three adapters are blind** (Wan 0.0056,
DC 0.0034, SkyReels 0.0013). Same data, same frozen base weights, same probe —
so the gap is **our implementation or our adapter design**.

Every "control" in this investigation so far has controlled for the *approach*
and never once for *our implementation*: both AVID datapoints (`93qrvr5v` RT-1,
`rqp4s3gp` Robot Arm) ran the official repo, not our framework. This ticket
closes that hole.

The prior on a wiring bug is not low. This codebase family has already produced
three silent action-conditioning drops: `get_batch_input` checking `"act"` while
the datamodule emitted `"action"` (would have trained with *zero* action
conditioning, no error), `action_dims` defaulting to 7 with no override
([[exp-adapter-avid-native-reference-run]] §"Real bug found and fixed"), and our
own eval path dropping `action_seq` for a single OOD token
([[../../30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe]]).

## Arm 1 (DO THIS FIRST) — flip `action_time_combine` to `concat`

Our DC run `c3pcewxk` is **already** near-AVID: `composition: avid_mask_mix`,
`gate_bias: 0.0`, `condition_on_base_outputs: true`, `prediction_type: velocity`,
`action_dim: 7`, and it loads AVID's own `act_cond_diffusion_11M.yaml` UNet
config. Composition form is identical
([[../../30_Knowledge/tech/avid-vs-ours-action-conditioning]] row 8).

[[../../30_Knowledge/tech/avid-vs-ours-action-conditioning]] (source-verified
2026-07-27) identifies **one** meaningful structural divergence:

- **AVID:** `emb = cat([time_emb₆₄, act_emb₆₄])` — orthogonal subspaces
  (`openaimodel3d.py:744`)
- **Ours:** `emb = time_emb₁₂₈ + cond_emb₁₂₈` — superimposed
  (`adapters/output/dynamicrafter.py:74`)

That note's verdict: the add is dimensionally correct, *not* a bug, but "a large
time signal can **swamp** the action, whereas concat reserves the action its own
subspace — the most plausible reason actions are relatively weaker in our DC
adapter."

The toggle exists (`adapter.extra.action_time_combine: add | concat`, implemented
2026-07-27) and `configs/dynamicrafter/diffusion_dc_acwm_robotarm_concat.yaml`
was written — **but never launched** (verified 2026-07-30: the `dc-acwm-robotarm`
wandb project holds only `kjgt3z0f`, `u9u7kxia`, `c3pcewxk`, all `add`, all
crashed). The 07-28 matrix declared blindness three days later and the
investigation went toward data/OOD instead. `rqp4s3gp` has now removed that
alternative.

⚠ **The shipped concat config is not a clean single-variable test.** Versus the
baseline it flips *three* things: `action_time_combine` add→concat,
`use_step_level_conditioning` false→true, `shortcut_anchor_prob` 1.0→0.55 (plus
eval/checkpoint cadence). If it follows actions you will not know which change
did it.

### A second divergence — `frame_stride`

Verified 2026-07-30: our DC config uses **`frame_stride: 4`**, and
`data/translators/acwm_phys.py:215` does
`actions = actions.reshape(length, stride, -1).sum(dim=1)` — so it trains on
**sums of 4 consecutive actions**. AVID's Robot Arm config uses
**`frame_stride: 1`** with the comment "consecutive frames, no stride-summing
(baseline parity)". Flagged LOW–MED in
[[../../30_Knowledge/tech/avid-vs-ours-action-conditioning]] row 9 with "keep
`frame_stride:1` for parity"; the config shipped 4 anyway.

Two candidates both weakening the action path means **flipping one may not
recover action-following**, and a null result on a single flip would wrongly
clear it.

### Three arms, single-variable, off the `c3pcewxk` baseline

| arm | change | isolates |
|---|---|---|
| **A** | `action_time_combine: concat` only | the time⊕action subspace |
| **B** | `frame_stride: 1` only | per-frame action detail |
| **C** | both | interaction / additive effect |

Also bring to parity while you're there: `target_height` 384 → **320** (DC's
native geometry — `act_cond_diffusion_11M_acwm_robotarm.yaml` declares
`image_size: [40, 64]`; affects base residual rather than action use directly).

**`action_sensitivity_probe: true` on every arm** — three of our four DC runs
(`gxq7kxzp` 17 h, `kjgt3z0f`, `t4bp8nki`) logged no action keys at all.

Decision rule: read at a step comparable to `c3pcewxk` (~801) *and* further out.
**Judge every treatment arm against arm 0, widened by the 0↔0S spread** — not
against the historical 0.004238.

- **Any arm ≳0.02 with null ≈0, and clear of arm 0 ± the 0↔0S spread** ⇒ that is
  the cause; clean D2 finding with the mechanism already written up.
- **All arms within the 0↔0S spread** ⇒ neither named divergence is it; proceed
  to the full match below.
- **Arm 0 itself lands far from 0.004238** ⇒ the historical number was probe
  noise or run-dependent, and the 07-28 blindness verdict needs re-reading before
  anything else is concluded.

Reference points: historical baseline `c3pcewxk` endpoint 0.004238 /
`effect ÷ adapter` 0.014588 / `adapter_rel_contribution` 0.319004. Target
`rqp4s3gp` 0.029475 / 0.422 / 0.069838.

## Arm 2 — full match, vary only the implementation

Our framework, configured as close to `avid_11M_acwm_robotarm.yaml` as our config
surface allows, on the **same** Robot Arm split, then the **same** probe.

| Axis | Reference (`rqp4s3gp`) | Ours (this run) |
|---|---|---|
| Base weights | DynamiCrafter-512, `ckts/dynami512.ckpt` | same file |
| Adapter family | output-level | output-level (`adapters/output/dynamicrafter.py`) |
| Composition | mask_mix, `learnt_mask: True` | `composition: mask_mix` |
| Gate init | `init_mask_bias: 0.0` (σ=0.5) | `gate_bias: 0.0` |
| Capacity | 11M tier | match 11M |
| Prediction | v-prediction | `prediction_type: velocity` |
| Action dim | 7 | 7 (verified: `metadata.pt` actions `[128, 7]`) |
| Data | `kinematics/robot_arm/ind_train`, traj_len 16 | identical split |
| Geometry | 320×512 | 320×512 |

**Resolved 2026-07-30 (was an open verification item):** our surface *does* expose
`condition_on_base_outputs` and `c3pcewxk` already sets it `true`. Remaining
known deltas beyond arm 1: action encoder depth (2-layer→64 vs our 4-layer→512→128),
stride-summed actions when `frame_stride > 1` (`acwm_phys.py:195` — keep
`frame_stride: 1` for parity), and our trainer/optimizer vs theirs. All rated
LOW–MED in [[../../30_Knowledge/tech/avid-vs-ours-action-conditioning]].

## Decision rule

Probe at a **step-matched** checkpoint against the reference (see the caveat in
the linked note — the 0.0295 is at step 5000 while our earlier runs were at
~800–1200; match before concluding).

- **Ours ≈ reference (≳0.02, null clean)** ⇒ our framework reproduces the
  reference on its own substrate. The earlier blindness was a config/recipe
  difference, not a framework defect — and this becomes **D1 validation
  evidence**, which the framework chapter currently has essentially none of.
- **Ours ≪ reference (≈0.005, null clean)** ⇒ a defect in our repo, with a
  diffable same-substrate reference. Bisect axes in order: action reaches the
  adapter at all (zero-gap > 0) → per-frame vs aggregated action → gate/composition
  → capacity → optimization (`sigma_shift`, gate cap).

A *small* gap is expected and is not a bug — different trainer, optimizer
defaults, VAE encode path, precision. The order-of-magnitude split above is the
threshold; **pre-register it and do not relax it after seeing the number.**

## Guardrails

- `base_null_violation ≈ 0` is a prerequisite for reading any effect_rel. That
  check is what made `93qrvr5v`, `423pjv8y` and `rqp4s3gp` trustworthy.
- Reference numbers are the *official repo's*, not our contribution — cite them
  as a control, never as D2 evidence for our adapters
  ([[exp-adapter-avid-native-reference-run]] §Guardrails).

## BUILT 2026-07-30 — ready to launch

**Configs** (generated from the `c3pcewxk` baseline
`configs/dynamicrafter/diffusion_dc_acwm_robotarm.yaml`; diffs verified
single-variable):

| file | `frame_stride` | `action_time_combine` |
|---|---|---|
| `diffusion_dc_acwm_robotarm_arm0_baseline.yaml` | 4 | add *(control)* |
| `diffusion_dc_acwm_robotarm_armA_concat.yaml` | 4 | **concat** |
| `diffusion_dc_acwm_robotarm_armB_stride1.yaml` | **1** | add |
| `diffusion_dc_acwm_robotarm_armC_concat_stride1.yaml` | **1** | **concat** |

### Controls (added 2026-07-30 — budget allows, and they are not optional)

Six arms run in parallel. Two are **controls**, and without them the treatment
arms are not interpretable:

- **Arm 0** — untreated baseline (`add` + `frame_stride 4`) re-measured under the
  **same 3×4 probe** as every treatment arm. The number the arms would otherwise
  be judged against — `c3pcewxk`'s 0.004238 — came from a `draws:2 × batches:2`
  probe **on a run that crashed**, so comparing against it confounds a
  measurement change with a treatment change.
- **Arm 0S** — arm 0 at `--seed 1`. The 0-vs-0S gap **is** the run-to-run noise
  floor for `eval_action_effect_rel`. Nothing in the vault currently establishes
  it, so a treatment arm reading e.g. 0.012 could not be called movement or
  noise. Judge treatments against arm 0 ± that spread, not against 0.004238.
- **Arm D** is no longer gated behind A/B/C — it launches with the rest.

Each also bumps the probe from 2×2 to `action_sensitivity_batches: 4` ×
`draws: 3` (measurement precision only — the reference used 120 samples, our
2×2 is noisy for a gap this size), sets a per-arm `output_dir`, and logs to
`wandb_project: dc-acwm-robotarm-avid-parity` so all arms land together.
`action_sensitivity_probe: true` + `action_sensitivity_keys: [act]` carried over
from the baseline — **the probe runs inside the eval cycle, so no separate probe
job is needed** (unlike the AVID side).

**Job:** `jobs/experiments_cluster/acwm_phys/dc/submit_train_dc_avid_parity.sh`,
parameterised by `ARM`:

```bash
cd ~/generative-flow-adapters
mkdir -p logs/dc-avid-parity
for a in 0 0S A B C D; do ARM=$a sbatch -J dc-parity-$a \
    jobs/experiments_cluster/acwm_phys/dc/submit_train_dc_avid_parity.sh; done
```

All six are independent; launch together. Arm dispatch verified by dry-run
2026-07-30 (each arm resolves to an existing config with the intended
stride/height/seed; an unknown `ARM` is rejected).

Arms A–C hold `--target-height 384` (what `c3pcewxk` ran) so the comparison stays
single-variable; **arm D** adds DC's native 320×512 — the geometry the reference
`rqp4s3gp` ran — as the full-parity attempt. D reuses arm C's config and differs
only by CLI geometry, so tell them apart by `--output-dir` / `-J`, not by the
wandb run name.

**Verified before building:** `action_time_combine` is fully wired
(`adapters/factory.py:281` → `adapters/output/dynamicrafter.py:84` sets
`add_act_time_emb = (combine == "add")`; both branches live in
`backbones/dynamicrafter/modules/networks/openaimodel3d.py:424,443,803`), CLI
`--frame-stride` overrides the config
(`scripts/train_avid_shortcut_metaworld.py:374`), and **DC encodes latents live —
no precompute is needed for a stride or geometry change**.

## Launchers (reference side, already tracked)

`jobs/experiments_cluster/avid_official/submit_train_avid_acwm_robotarm.sh` ·
`submit_probe_acwm_robotarm_action.sh`

## Cleanup 2026-08-01 — **DELIVERED**

All six arms ran; mechanism found (learned pedestal) and fixed. See [[../../30_Knowledge/experiments/20260730-dc-parity-arms-null-action-embedding-pedestal]] + [[../../30_Knowledge/experiments/20260731-dc-condition-center-accelerates-escape]].

*Proposed for close; awaiting confirmation (CLAUDE.md: never close without it).*
