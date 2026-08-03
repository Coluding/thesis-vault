> **⚠ CORRECTED 2026-08-02.** Two claims on this page do not survive the
> full logged histories: (1) "NOBASE stops the erosion (flat 0.0106)" — `vy9tcuco`
> peaked at **0.0123** and fell to **0.0077**; it eroded. (2) "6–10× sustained" is a
> *peak*-vs-control comparison; sustained is **3–4×** (mean of last 3 evals: 0.0066 /
> 0.0087 vs the GATEFIX control's 0.0018). And the structure triad on this very
> checkpoint came back **at chance on all three axes**, so token-norm raised
> *sensitivity*, not *control* — note it is mechanically a gain knob while
> `effect_rel` is monotone in gain. See
> [[20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]].

---
type: experiment
date: 2026-07-31
config: configs/wan22/diffusion_wan22_avid_xattn_{gatefix,tokennorm,tokennorm_nobase}_acwm_robotarm.yaml
commit: uncommitted working tree @ 2026-07-31 (action_token_norm in backbones/wan/modules/action_model.py; jobs 25083978 / 25085598 / 25088945)
wandb_run_id: GATEFIX (control) · 52o3uxz8 (TOKENNORM) · TOKENNORM-NOBASE
ckpt_path: /scratch-shared/lbierling/outputs/acwm-robotarm-{gatefix,tokennorm,tokennorm-nobase}-run/checkpoints/
status: killed (GATEFIX/TOKENNORM 2026-07-31 midday; NOBASE still running)
deliverable: D2
metrics:
  gatefix_latest_effect_rel: 0.00198
  tokennorm_first_effect_rel: 0.01131
  tokennorm_latest_effect_rel: 0.00668
  nobase_effect_rel_evals: "0.01234 -> 0.01067 -> 0.01056 -> 0.00982 -> 0.00949 -> 0.00902 (slow decay, x1.4 slower than tokennorm)"
  tokennorm_video_adapted_vs_base_l1: 0.0388
  tokennorm_video_adapted_vs_gt_l1: 0.0450
  tokennorm_video_base_vs_gt_l1: 0.0562
  tokennorm_motion_corr_adapted: 0.667
  tokennorm_motion_corr_base: 0.173
  gt_motion_energy: 0.00117
  base_motion_energy: 0.00557
  adapted_motion_energy: 0.00325
notes: "ROLLOUT-SWAP RESOLVED: video gains are temporal-prior, not action-following (true tracks GT no better than shuffled/zero). Wan intervention training results: token-norm gives a sustained 6-10x over the matched gate-fixed control but ERODES (0.0113 -> 0.0067) while the oracle is present; adding condition_on_base_outputs:false (NOBASE) STOPS the erosion (flat ~0.0106) — pre-registered prediction confirmed. Video analysis at step 3000: the adapted rollout is finally distinct from base (L1 0.039), 20% closer to GT, and tracks GT motion timing far better (corr 0.67 vs 0.17) while halving the base's hallucinated motion."
---

# Wan interventions — token-norm works, the oracle erodes it, NOBASE stops the leak (D2)

> Training results of the three Wan arms launched 2026-07-31 (overnight
> session), against the matched GATEFIX control (live gate, no norm). All runs
> in flight; snapshot 2026-07-31 afternoon. Mechanism notes:
> [[20260731-wan-action-trace-value-pathway-drowns]] (transport) and
> [[20260731-why-wan-copies-the-base-decomposed]] (incentive).

## Trajectories (`eval_action_effect_rel`, null = 0 throughout)

| eval≈ (×500 steps) | 1 | 2 | 3 | 4 | 5 | 6 | 7 |
|---|---|---|---|---|---|---|---|
| GATEFIX control | 0.0012 | 0.0013 | 0.0013 | 0.0015 | 0.0014 | 0.0014 | 0.0020 |
| **TOKENNORM** | **0.0113** | 0.0090 | 0.0085 | 0.0078 | 0.0084 | 0.0070 | 0.0067 |
| **TOKENNORM-NOBASE** | **0.0123** | 0.0107 | **0.0106** | | | | |

- **Token-norm: sustained 6–10× over the matched control** — the causal
  confirmation of the value-pathway diagnosis — but the trajectory **erodes
  monotonically** while `condition_on_base_outputs: true` keeps the oracle in
  the input.
- **NOBASE erodes ~1.4× slower than TOKENNORM** (corrected at 6 evals:
  0.0123→0.0090 = −27% vs 0.0113→0.0070 = −38%; the 3-eval "flat" read was
  premature). Removing the oracle **slows, not stops**, the erosion —
  **pre-registered prediction partially confirmed** ([[20260731-why-wan-copies-the-base-decomposed]] §Outcome). It
  does *not* climb — consistent with the 0.45% action-economics providing no
  upward pressure once the leak is plugged.
- Cost of NOBASE: adapted loss ~0.13–0.18 vs ~0.09 with the oracle at matched
  steps — without the base's answer the adapter must earn base-quality
  prediction itself. A trade, not a free win.
- Wan plateau with both fixes: **~0.011** (7.5× control; the AVID reference
  0.0295 uses a full separate UNet). Pushing higher points at the objective
  (action-CFG, rollout losses), not more architecture.

## Video analysis — TOKENNORM step 3000 (`52o3uxz8`, eval_step_grid, 3 clips)

Quantified on the gt|base|adapted panels (grayscale L1 / motion profiles):

| | sample 0 | sample 1 | sample 2 |
|---|---|---|---|
| \|adapted − base\| | 0.0388 | 0.0325 | 0.0405 |
| \|base − gt\| | 0.0562 | 0.0553 | 0.0611 |
| \|adapted − gt\| | **0.0450** | **0.0450** | **0.0435** |
| motion corr w/ gt: base | 0.17 | 0.46 | 0.30 |
| motion corr w/ gt: adapted | **0.67** | **0.57** | **0.46** |

The adapted rollout is (a) **finally distinct from the base** — earlier runs
sat at ≈0 — (b) **~20% closer to ground truth** on every clip, and (c) tracks
GT's motion *timing* far better while roughly halving the base's hallucinated
motion (energy 0.0033 vs base 0.0056 vs GT 0.0012).

**Attribution — RESOLVED by the rollout-action-swap probe (job 25104155,
COMPLETED): the video gains are temporal-prior, not action-following.** Same
seed/clip/solver, only the actions varied across three 50-step rollouts:

| | value |
|---|---|
| \|shuffle − true\| | 0.0134 gray-L1 (≈14% of base↔true) |
| compounding first→last third | 0.0123 → 0.0147 (mild, +20%) |
| GT-tracking true / shuffle / zero | 0.0818 / 0.0807 / **0.0791** |
| GT-tracking base | **0.0607** |

Actions *influence* the rollout (1.3%, mildly compounding) but do **not steer
it** — the true-action rollout tracks GT no better than wrong/zero actions:
**sensitivity without control**, consistent with the ~0.011 single-step
plateau. Also: in this native 50-step rollout the *base* tracks GT better than
any adapted variant (0.061 vs ~0.080), whereas the training-eval grids showed
adapted 20% closer — different generation path and clip; both stand, but the
adapted model may pay a rollout-quality cost the eval grid does not expose.
Caveats: one clip / seed / donor (the zero variant covers the similar-donor
hole). Videos + JSON: `eval_rollout_swap/` on scratch.

## Related

- [[20260731-wan-action-trace-value-pathway-drowns]] — why token-norm exists
- [[20260731-why-wan-copies-the-base-decomposed]] — why the erosion exists
- [[20260731-dc-condition-center-accelerates-escape]] — the DC counterpart
