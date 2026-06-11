---
date: 2026-06-05
category: finding
deliverable: D3
meeting:
sources:
  - "[[../../20_Tickets/exp-shortcut-vs-image-only-anchor-baseline]]"
  - "[[../../20_Tickets/exp-conditioning-add-actions-to-shortcut-adapter]]"
  - "[[entries/2026-06-01-shortcut-steplevel-ood-mitigated]]"
---

# Anchor-only baseline (anchor_prob=1) confirms shortcut training buys few-step robustness

## What

Ran the image-only control for the shortcut adapter: `shortcut_anchor_prob=1.0`,
which pins `step_level` to the finest step and removes the self-consistency
supervision entirely (standard diffusion loss only) — same step-level-conditioned
adapter and plumbing, only the shortcut target removed. Compared qualitatively
against the `anchor_prob=0.5` shortcut run from the prior local test.

## Why it matters

This is the **positive control passing.** The anchor-only adapter still beats the
frozen base (it learns the scene) but its **few-step samples are visibly worse** —
the robot arm comes out ghosted/motion-blurred, worst at low step counts and
tightening toward 25 steps. The `0.5` shortcut run stayed sharp and roughly
step-count-invariant. So the step-count robustness seen earlier is genuinely
produced by the **self-consistency training**, not just by the adapter being a
competent image predictor — which de-risks the core D3 claim.

## Evidence / sources

- **Qualitative only, local run** — no wandb id / ckpt / commit yet, so no
  metrics recorded (hard rule 8). Side-by-side stills: left = GT, middle = frozen
  base (garbage in both runs), right = adapter; rows = sampling steps 1→25.
- The quantitative deliverable is still owed: **MSE-vs-NFE curve for both arms**
  (the headline plot in [[../../20_Tickets/exp-shortcut-vs-image-only-anchor-baseline]]).
- Caveat to state explicitly: confirm the **inference step-level schedule is
  identical** across the two arms, else part of the anchor-arm blur is OOD
  conditioning rather than missing supervision.

## Next

- **HPC run** to produce the sourced MSE-vs-NFE curve for both arms (this was a
  local sanity check only).
- **Diagnose why action-conditioned dynamics still look random.**
  ([[../../20_Tickets/exp-conditioning-add-actions-to-shortcut-adapter]])
  Correction (2026-06-05): the existing AVID config **already conditions on
  `a_t`** — the adapter UNet is natively `action_conditioned: True` and `act` is
  passed to its action head. So the wandering dynamics are *not* from missing
  action conditioning; actions are wired but apparently not effective. The task is
  now to find out why (actions null/dropped at train or eval? action a_t not held
  fixed across the self-consistency micro-step? signal too weak / undertrained?),
  not to add wiring.
