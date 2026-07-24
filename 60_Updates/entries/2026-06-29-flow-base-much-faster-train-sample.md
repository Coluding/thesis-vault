---
date: 2026-06-29
category: finding
deliverable: D3
meeting:
sources:
  - "[[../../30_Knowledge/experiments/20260629-flow-vs-diffusion-shortcut-samples]]"
  - "[[2026-06-19-pivot-flow-matching-base]]"
---

# Flow-matching base trains and samples much faster than diffusion

## What

On the first post-pivot batch (`data/results/20260629/`), the flow-matching runs
are **noticeably faster than the diffusion run — both to train and to sample**
(qualitative observation; exact steps/sec and sampling wall-clock
_needs verification_).

## Why it matters

This is the practical payoff of the 2026-06-19 flow-matching pivot
([[2026-06-19-pivot-flow-matching-base]]), beyond the clean κ=0 objective. Fast
sampling is the whole point of D3/D4 — few-step shortcut rollout for planning —
so a base that already samples cheaply makes the few-step story easier to land.
The straight (κ=0) ODE trajectory needs far fewer integration steps than the
diffusion sampler, which is the expected mechanism.

## Evidence / sources

- Qualitative, from running the three 2026-06-29 jobs
  ([[../../30_Knowledge/experiments/20260629-flow-vs-diffusion-shortcut-samples]]).
- **Caveat — not a controlled comparison.** The diffusion run was at higher
  resolution (1536×1600 vs 768×1280) and a different backbone, so this batch
  conflates flow-vs-diffusion with model/resolution differences. No logged
  steps/sec or sampling latency yet.

## Next

Log steps/sec + few-step sampling latency (NFE + wall-clock) as a standard metric
on matched model/resolution to turn this into a sourced number — and pair it with
the open sample-quality blocker (loss converges, generation still poor).
