---
type: writeup
scope: writing
status: open
priority: medium
created: 2026-06-22
updated: 2026-06-22
resolution:
resolution_note:
closed_at:
related:
  - "[[../50_Decisions/decided/shortcut-target-endpoint-vs-v-averaging]]"
  - "[[bug-losses-shortcut-v-averaging-target]]"
  - "[[feat-shortcut-displacement-target-parametrisation]]"
---

# writeup: D3 thesis paragraph on the shortcut v-averaging bias + the two fixes

A short theory passage for the D3 chapter: naïvely porting the shortcut
self-consistency rule (Frans et al. 2024, eq. 4) to v-prediction diffusion is
**biased** — the VP trajectory is a circular arc (κ=1), so averaging two
tangents from different points lands inside the arc (the sagitta / Euclidean-vs-
Fréchet-centroid gap). Present the geometry, then the **two endpoint-consistent
fixes** (endpoint inversion in velocity space; displacement in secant space) as
the same idea in two coordinate choices, and note the structural escape (a
flow-matching base, κ=0, makes eq. 4 exact as written).

## Sources

- [[../30_Knowledge/theory/shortcut-v-averaging-bias]] (geometry, sagitta,
  fixed-point criterion, numbers)
- [[../30_Knowledge/theory/ddim-step-v-parameterisation]] (the DDIM rotation /
  inversion primitive)
- [[../50_Decisions/decided/shortcut-target-endpoint-vs-v-averaging]] (the
  decision + the ablation axis)

## Done when

Drafted into the D3 section via `/thesis-write` (destination
`70_Thesis/draft/`), with the before/after on diffusion once
[[bug-losses-shortcut-v-averaging-target]] produces it. Respects the
no-unsourced-numbers rule.
