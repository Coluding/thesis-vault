---
type: exp
scope: data
status: open
priority: high
created: 2026-06-30
updated: 2026-06-30
resolution:
resolution_note:
closed_at:
related:
  - "[[../../30_Knowledge/experiments/20260629-flow-vs-diffusion-shortcut-samples]]"
  - "[[../../30_Knowledge/experiments/avid-shortcut-anchor045-volatile-loss]]"
  - "[[exp-shortcut-scale-episodes-longer-train]]"
---

# exp: single-environment dataset — does narrowing the task improve sample quality?

## Hypothesis

On the 2026-06-29 flow-base batch the loss converges but **samples are still
poor** ([[../../30_Knowledge/experiments/20260629-flow-vs-diffusion-shortcut-samples]]).
One suspect is that the model is spread too thin across **all MetaWorld envs and
cameras**. Restricting training to a **single environment / single task** (smaller,
narrower dataset) may let the adapter fit that one task's dynamics well enough to
produce **visibly better rollouts**.

## What to run

- Take the current flow-matching shortcut setup and **train on one environment
  only** (single task, single/fixed camera), reducing dataset size accordingly.
- Keep everything else matched to the 2026-06-29 flow-shortcut run so the only
  variable is dataset breadth (one env vs all envs).
- Optionally also run the no-shortcut baseline on the same single env, to keep the
  ±shortcut comparison alive at the narrow scope.

## What to measure

- Sample-video quality vs the all-envs run (same eval grid + layout).
- `base_loss` and per-rung shortcut loss (do they reach lower / cleaner values on
  the narrower distribution?).
- If a few-step comparison is possible: flow ±shortcut at matched NFE on the
  single env.

## Caveat / prior tension

Earlier small-data work flagged that an apparent "robust across NFE" result was a
**small-data overfitting artifact** that did **not** survive more data
([[../../30_Knowledge/experiments/avid-shortcut-anchor045-volatile-loss]]). So
"single-env looks better" must be read carefully: distinguish **genuinely better
learned dynamics** from **overfitting / memorisation** of a narrow set. Hold out
unseen rollouts within the single env to check generalisation, not just training
reconstruction.

## Done when

- A single-env run completes with logged loss + sample videos → promote to an
  experiment note under `30_Knowledge/experiments/`.
- Verdict recorded: does narrowing to one task improve sample quality, and is the
  gain real (held-out) or overfitting?
