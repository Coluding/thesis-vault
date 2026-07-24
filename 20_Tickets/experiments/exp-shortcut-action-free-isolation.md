---
type: exp
scope: shortcut
status: open
priority: high
created: 2026-07-09
updated: 2026-07-09
resolution:
resolution_note:
closed_at:
related: ["[[../../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]]", "[[exp-shortcut-zero-weight-control-run]]", "[[../../30_Knowledge/experiments/20260629-flow-vs-diffusion-shortcut-samples]]"]
---

# exp: action-free shortcut-only adapter training (pure D3 test)

## Idea (user, 2026-07-09)

Train an adapter that does **shortcut modeling only — no action conditioning**.
Strip the actions entirely; the adapter is step-size-conditioned (`step_level` /
`include_step_size`) but has **no action input**. Its whole job is to distill the
frozen WAN flow base into a good **few-step** generator.

## Why — decouple D3 from D2

In the current run ([[../../30_Knowledge/experiments/20260907-flow-shortcut-weak-action-signal]])
"does the shortcut work?" and "does action conditioning work?" are confounded — a
bad few-step grid could be blamed on either. Removing actions isolates the
**pure D3 question**: *can our shortcut adapter make the frozen base samplable in
few steps?*

Completes the ablation matrix — {action on/off} × {shortcut on/off}:

| | shortcut OFF | shortcut ON |
|---|---|---|
| action OFF | plain base distill (sanity) | **THIS — pure shortcut test** |
| action ON | action-only D2 ([[exp-shortcut-zero-weight-control-run]]) | current D4 run |

## Setup

- **Conditioning:** `include_step_size: true`, **no action modality** (drop the
  `action` condition from `ConditioningConfig`). Adapter is
  step-size-conditioned only. → likely needs a **new config** (e.g.
  `configs/flow_wan22_shortcut_only_metaworld.yaml`) forked from the current
  wan22 shortcut config with the action condition removed.
- **Losses:** keep the shortcut terms (`shortcut_direction`, `local_consistency`,
  `multistep_consistency`) + the base flow-matching loss. No action dependence.
- **Base / data / everything else:** identical to the current run (Wan2.2 TI2V-5B
  frozen base, `metaworld_corner2`), so the only removed variable is actions.

## Success criterion (the D3 result)

Read the **NFE-row grid** (rows = denoising steps 1→50, cols = GT | base |
adapted):

- **Shortcut works** ⇒ the **top (few-step) rows get good** — few-step adapted
  quality approaches the many-step rows, and clearly beats the **frozen base at
  the same few-step budget** (base has no few-step help).
- **Shortcut doesn't work** ⇒ top rows stay mush even without the action burden →
  the problem is the shortcut approach/impl, fix that first. A clean negative here
  is still a thesis-worthy D3 finding.

## Notes

Pair the eval with the frozen-base few-step rollout as the reference (the shortcut
adapter's *added value* over the raw base at low NFE is the headline number).
Quantify, don't just eyeball (hard rule 8) — few-step PSNR/SSIM/LPIPS/FVD vs GT,
adapted-vs-base, per step budget.
