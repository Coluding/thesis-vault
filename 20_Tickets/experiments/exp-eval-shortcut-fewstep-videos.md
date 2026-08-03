---
type: exp
scope: eval
status: open
priority: high
created: 2026-07-29
updated: 2026-08-01
resolution:
resolution_note:
closed_at:
related: ["[[exp-shortcut-flow-vs-diffusion-openvid]]", "[[../../30_Knowledge/experiments/20260729-shortcut-wan-vs-dc-curvature-signature]]"]
---

# exp: offline few-step video generation + step-size perturbation (D3)

## Hypothesis / purpose

The D3 payoff is **qualitative**: does the shortcut adapter reproduce the base's
50-step rollout in *few* steps? In-training `eval_step_grid` shows this to wandb,
but we need an **offline, reproducible** version from a chosen checkpoint — for
thesis figures and to inspect the *existing* runs (`pzmc2orq` Wan, `t4bp8nki` DC)
without restarting them.

## Procedure

Standalone `scripts/generate_shortcut_fewstep.py` (Wan/DC/SkyReels via the shared
base interface): load a shortcut checkpoint, generate each eval clip at
**N ∈ {1,2,4,8,25,50}**, save a **gt | base | adapted** grid per clip. Plus a
**step-size perturbation** strip (generate at a fixed/wrong `step_level`) — the
visual counterpart to `eval_stepsize_effect_rel`, to *see* whether the adapter is
step-conditioned or degenerate (condition-blind). sbatch wrapper for the cluster.

## Decision rule (what the videos should show)

- **adapted @ small N ≈ base @ 50** ⇒ few-step works — the headline (flow/Wan).
- **adapted @ small N degrades / diverges** ⇒ few-step fails (expected for DC if
  the curvature story holds).
- **perturbed-step-size video ≈ true-step-size video** ⇒ step-size-blind
  (degenerate collapse — the few-step is fake even if quality looks ok).

## Status

Tool being built 2026-07-29 (reuses the trainer's `_render_step_grid`
generation). Reads any shortcut checkpoint + config. Run remotely on the cluster
(GPU; shortcut generation is heavy). Configs already sweep N∈{1,2,4,8,25,50}.

## Cleanup 2026-08-01 — **SUPERSEDED**

The few-step sweep now runs inside training (`eval_step_schedule` N in {1,2,4,8,25,50}) on both arms of [[exp-shortcut-d3-fewstep-vs-noshortcut-control]]. Keep the offline tool for figures.

*Proposed for close; awaiting confirmation (CLAUDE.md: never close without it).*
