---
type: exp
scope: shortcut
status: open
priority: high
created: 2026-07-28
updated: 2026-07-28
resolution:
resolution_note:
closed_at:
related: ["[[../../30_Knowledge/writing/ablation-axes]]", "[[../../30_Knowledge/experiments/20260728-acwm-robotarm-matrix-action-blind]]"]
---

# exp: shortcut objective — flow (Wan) vs diffusion (DC), on OpenVid (D3)

## Hypothesis (the contribution structure)

Few-step **shortcut** generation should work on a **flow-matching** base and
struggle on a **diffusion** base — because flow matching has near-straight
probability-flow trajectories while the diffusion denoising trajectory is
**curved**, so a step-size-conditioned adapter can jump along a straight path
but not a curved one. This is the empirical test of the thesis's "diffusion is
ill-suited for shortcut" section.

Two outcomes, both valuable:
- **Wan (flow) shortcut works** → a **big contribution on its own**: a
  plug-and-play adapter turns a frozen flow-matching video base into a few-step
  world model.
- **Wan works, DC (diffusion) doesn't** → empirical validation of the
  curvature argument — the theory made concrete, not just argued.

SkyReels (weak flow) is a second flow datapoint (does base strength matter for
shortcut fidelity?).

## Procedure

Action-FREE shortcut adapters (`use_step_level_conditioning: true`,
`shortcut_direction_weight 1.0`, `multistep_consistency_weight 1.0`,
`shortcut_anchor_prob 0.5` — non-inert), TI2V (frame-0 anchor + caption), on the
SAME in-distribution captioned real-world data so the flow-vs-diffusion contrast
isn't confounded by base OOD drift:

- **Data: OpenVid-1M subset** (real captions, in-distribution for the video
  priors) — configs `configs/{wan22,dynamicrafter,skyreels}/*_shortcut_openvid.yaml`.
- Also runnable on ACWM Robot Arm (`*_shortcut_actionfree_robotarm.yaml`) as a
  secondary substrate.
- Metric: few-step rollout fidelity — quality at N ∈ {1,2,4,8,25,50} steps
  (the `eval_step_schedule`) vs the base's 50-step rollout; does the shortcut
  adapter reproduce the multi-step result in few steps?

## Decision rule

- **Wan few-step quality ≈ base 50-step at small N** ⇒ shortcut works on flow →
  headline D3 result.
- **DC few-step quality degrades sharply as N drops while Wan holds** ⇒
  curvature theory validated (flow-suited, diffusion-not).
- **Both fail** ⇒ the shortcut adapter / consistency objective needs work before
  the flow-vs-diffusion claim can be made (check `shortcut_anchor_prob` actually
  builds targets — earlier ACWM configs shipped the inert `1.0`).

## Sequencing (priority, 2026-07-28)

**Gate on Wan first.** The headline is "does few-step shortcut work on a flow
base (Wan)". Prove that before spending GPU on the DC comparison:

1. **Wan shortcut works** (step-1 smoke: shortcut loss non-inert → few-step eval
   sweep holds vs base 50-step). This alone is the big contribution.
2. **Only after (1):** the DC flow-vs-diffusion comparison, "just for metrics" —
   completes the 3-way table for the curvature section:
   - DC + `endpoint_inversion` (adjusted; **already configured**),
   - DC + `v_average` (naive flow target on diffusion — a 2-line clone;
     **deferred, add post-Wan**) to show the sagitta bias explicitly.

SkyReels (weak flow) rides along with Wan as the second flow datapoint.

## Status

**Wan step-1 gate PASSED (2026-07-28, local smoke, tiny geometry):**
`shortcut_direction_loss=0.0145` + `multistep_consistency_loss=0.0145` (both
non-zero → objective non-inert, the `anchor_prob 1.0` trap fixed);
`eval_adapter_pred_base_cosine=0.248`, `rel_contribution=0.616` (adapter is NOT
cloning — step-size-driven, as a shortcut adapter should be). base_loss 0.74.
Local run: `configs/wan22/diffusion_wan22_shortcut_actionfree_robotarm.yaml`,
temporal 9 / max_area 147456 (24 GB-3090 gate only). **Infra note: real
shortcut training needs the CLUSTER (H100 80 GB) — the shortcut consistency does
2–3× base forwards, so Wan-5B OOMs the local 3090 at real geometry.**

Configs built (action-free shortcut, ACWM Robot Arm ✅; OpenVid variants +
per-clip-caption path in progress 2026-07-28). DC `endpoint_inversion` in place;
DC `v_average` naive variant deferred to post-Wan. Pending: OpenVid subset
download + T5 per-caption precompute, then the **Wan** step-1 smoke (confirm
shortcut loss > 0, not inert), then the Wan few-step eval sweep. Note: SkyReels
`denoise` must thread `step_level` to the adapter at runtime (config-valid,
untested).
