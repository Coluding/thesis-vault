---
type: experiment
date: 2026-07-28
config: configs/wan22/diffusion_wan22_avid_xattn_gatelow_capshift_acwm_robotarm.yaml; configs/dynamicrafter/diffusion_dc_acwm_robotarm.yaml; configs/skyreels/diffusion_skyreels_xattn_acwm_robotarm.yaml; external_repos/avid avid_11M_acwm (reference)
commit: uncommitted working tree @ 2026-07-28 (diagnostics + toggle session)
wandb_run_id: ncztxyyo (Wan), c3pcewxk (DC), 8zjjn7wl (SkyReels), 423pjv8y (AVID ref)
ckpt_path: remote (wandb runs, coluding)
status: snapshot (Wan running @1172, DC running @801, SkyReels killed @897, AVID finished @1687)
deliverable: D2
metrics:
  wan_eval_action_effect_rel: 0.0056
  dc_eval_action_effect_rel: 0.0034
  skyreels_eval_action_effect_rel: 0.0013
  base_null_violation_all: 0
  wan_eval_base_loss: 0.171
  dc_eval_base_loss: 0.045
  skyreels_eval_base_loss: 0.065
notes: "All three adapter bases are ACTION-BLIND on ACWM robot arm (effect_rel ~0, cos~1, loss_gap~0, flat across eval points) despite the 8.7x residual. Base-null control = 0 -> measurement trustworthy. Three distinct starvation signatures."
---

# ACWM Robot Arm matrix — all three adapters action-blind (D2)

> **Snapshot of live runs**, wandb project `coluding`, read 2026-07-28 via the
> wandb API. Numbers are the latest eval-cycle values at the steps noted; runs
> were early (800–1700 steps) but the action signal is flat, not emerging.

## Headline

The move from flat Push Cube to visually-rich Robot Arm (motivated by the 8.7×
larger base residual, [[20260725-acwm-base-residual-diagnostic]]) did **NOT**
produce action-following. All three adapter bases converge to **action-blind**
solutions that lower the loss *action-independently*. Base-parity persists on
the rich domain.

## Action-sensitivity (the load-bearing measurement)

| Base | wandb | `eval_action_effect_rel` 400→800 | `eval_action_cos` | `eval_action_loss_gap` | `base_null_violation` |
|---|---|---|---|---|---|
| Wan2.2 | `ncztxyyo` | 0.0059 → 0.0056 (max 0.0059) | 0.99998 | 2.1e-5 | **0** |
| DynamiCrafter | `c3pcewxk` | 0 → 0.0034 (max 0.0034) | 0.99999 | −1.3e-6 | **0** |
| SkyReels-1.3B | `8zjjn7wl` | 0.0014 → 0.0013 (max 0.0014) | 0.9999992 | 1.3e-6 | **0** |

`effect_rel ≈ 0` (prediction barely moves when the action is shuffled/zeroed) +
`cos ≈ 1` + `loss_gap ≈ 0` (perturbing the action doesn't change the loss). The
`base_null_violation = 0` control confirms the frozen base is action-invariant —
**the harness is clean, so the ~0 action effect is real, not a leak.**

## Three distinct starvation signatures (from the grad/gate diagnostics)

| Base | gate_mean (std) | pred-vs-base cos | adapter_grad_norm | action-path grad | eval_base_loss | signature |
|---|---|---|---|---|---|---|
| Wan `ncztxyyo` | 0.50 (0.00) | 0.99 | 0.21 | action_inject 0.012 | 0.171 | **gate stuck at init**; pred clones base; adapter grad healthy but action-independent (loss fell 0.314→0.171) |
| DC `c3pcewxk` | 0.60 (0.067) | 0.69 | 0.13 | **condition 0.001** | 0.045 | gate *moves*, pred diverges most (least cloning), but the **action encoder is starved** — matches the concat-vs-add prediction ([[avid-vs-ours-action-conditioning]]) |
| SkyReels `8zjjn7wl` | 0.90 (0.0005) | 0.92 | 0.018 | action_inject 0.002 | 0.065 | **gate saturated to the cap** → adapter starved (classic gate-saturation trap) |

(DC has no cross-attn action path, so `action_inject_grad_norm` is absent by
design; `condition_grad_norm` is its action-gradient trace. Wan/SkyReels report
`action_inject_grad_norm`.)

## AVID reference (`423pjv8y`, finished, 187 epochs)

**Trained on ACWM Push Cube, NOT Robot Arm** (`data_dir=.../rigid_dynamics/push_block/ind_train`)
— so it is a *different-dataset* baseline, not a same-dataset match to the three
robot-arm runs above. Push Cube is still action-informative, so it tests whether
the *original* AVID recipe follows actions at all. Its own PyTorch-Lightning
trainer has **no** action-sensitivity probe, so only `train/loss_simple = 0.0163`
is available (not scale-comparable to our masked `eval_base_loss`). Checkpoints
saved under `external_repos/avid/latent_diffusion/outputs/avid_acwm_pushblock_11M/checkpoints/`.
**AVID-side action-sensitivity probe (2026-07-28)** — checkpoint
`epoch=184-step=1480.ckpt`, 60 samples (4 batches × 5 timesteps × 3 paired
draws), via a bespoke probe reproducing our metric on `AVIDAdapter.apply_model`
(v-prediction; script `external_repos/avid/latent_diffusion/scripts/probe_action_sensitivity.py`):

| variant | action_effect_rel | cos(true,variant) | base_null |
|---|---|---|---|
| shuffle | 0.0015 | ~1.0 | **0.0** ✓ |
| zero | 0.0015 | ~1.0 | **0.0** ✓ |

`adapter_rel_contribution = 0.047`, gate/mask mean 0.745. The **null control is
exactly 0** (base is bit-identical across action variants) → the probe is
trustworthy AVID-side.

**BUT this is NOT yet a usable AVID reference:** the run used `max_clips: 64`
(64 of 1500 push_block episodes) for 185 epochs — heavy overfitting on tiny data,
where action-blindness is *expected regardless of recipe*. So the 0.0015 does NOT
answer "does the original AVID recipe follow actions." It only confirms the probe
reproduces AVID-side. A real AVID reference needs **full data** (`max_clips` off)
+ more training, ideally on **Robot Arm** to match our runs.

## Interpretation (analysed)

- The residual-space hypothesis (richer domain → adapter must use actions) does
  **not** hold as hoped: the base leaves residual, but the adapter fills it with
  an **action-independent** better-than-base prediction.
- Three *different* mechanisms starve the action path (gate stuck / encoder
  starved / gate saturated), so there is no single fix — this elevates both the
  **concat-vs-add** experiment (DC's starved encoder) and the **gate
  interventions** (SkyReels' saturation, Wan's stuck gate).

## Next

- Run the DC **concat** variant ([[avid-vs-ours-action-conditioning]]) and watch
  `condition_grad_norm` + `eval_action_effect_rel`.
- Measure action-sensitivity on the AVID reference checkpoint (does the *original*
  recipe follow actions where ours doesn't?).
- Let the runs continue — confirm the flat action signal isn't a late bloomer.
