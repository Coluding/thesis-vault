---
date: 2026-07-21
category: finding
deliverable: D2
meeting:
sources: ["[[../../30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe]]", "[[../../20_Tickets/done/bug-adapter-replace-generation-flat-since-init]]", "[[../../20_Tickets/done/exp-conditioning-action-shuffle-ablation]]", "[[../../20_Tickets/bug-adapter-gate-saturation-mask-mix]]", "[[../../20_Tickets/experiments/exp-adapter-replace-nobase-overfit]]"]
---

# Generation-eval bug fixed and validated; adapter measured fully action-blind

## What

The "replace-composition generation is pure noise" mystery is solved: every
generation-eval path dropped the per-frame `action_seq` tokens the
cross-attention adapter was trained on, collapsing its output (cos vs base
0.997 → 0.63). After the fix, the same pipeline produces coherent rollouts at
base-level FID (wandb `y1jrgxqp`: adapted FID 518 → 58 ≈ base 55 by step
600). Follow-up measurements then quantified the *real* problem: the trained
adapter is a near-total base-clone (cos ≥ 0.996 at every noise level) and
completely action-blind — feeding it another clip's actions or zero actions
changes its loss by less than 1e-4 at every σ.

## Why it matters

- Every generation metric from the earlier xattn runs (incl. the "worse on
  all 6 metrics" negative result) was measuring the bug, not the adapter —
  those conclusions are void and the eval pipeline is now trustworthy.
- The D2 blocker is now precisely characterized: not broken generation, but
  a copy-through optimization trap (gatelow single-clip overfit `uxrst2k5`
  couldn't overfit ONE clip — gate saturated 0.5 → 0.99 from a balanced
  init, adapter grad norm 4.4 → 0.003) plus a training objective (σ~U(0,1))
  and possibly a dataset (scripted MetaWorld demos) that barely reward
  action information.

## Evidence / sources

All numbers: [[../../30_Knowledge/experiments/20260721-replace-fix-validation-sigma-sweep-action-probe]]
(wandb `y1jrgxqp`, `uxrst2k5`; local σ-sweep + action probe on ckpt
`outputs/replace-metaworld-run/checkpoints/step_00001500.pt`, artifacts in
`outputs/replace_debug/`).

## Next

- No-base-input single-clip overfit (config ready) discriminates
  optimization-trap vs capacity ([[../../20_Tickets/experiments/exp-adapter-replace-nobase-overfit]]).
- `sigma_shift: 5.0` training option landed (concentrates supervision at
  high noise, matching Wan pretraining + the ACWM-DiT recipe) — enabled for
  the next replace-family run.
- Dataset decision pending: MetaWorld scripted demos may make actions
  redundant given the anchor frame; ACWM-Phys (Push Cube) is the candidate
  action-informative benchmark with published from-scratch baselines.
