---
date: 2026-06-19
category: decision
deliverable: D3
meeting: 2026-06-19
sources:
  - "[[../../50_Decisions/open/shortcut-target-endpoint-vs-v-averaging]]"
  - "[[../../30_Knowledge/theory/shortcut-v-averaging-bias]]"
---

# Evaluating a flow-matching base (Pyramid Flow, WAN) to sidestep the v-prediction shortcut bias

## What

The shortcut v-averaging bias is fundamentally a curvature effect (κ=1 on the VP
circle). A **flow-matching base has κ=0** — a straight trajectory — so averaging
is exact and Frans eq. (4) is faithful as written, with no target surgery. We are
evaluating two open flow-matching video models as the frozen base:

- **Pyramid Flow** — flow matching, but its pyramidal / temporally-autoregressive
  design complicates a clean frozen-`f_base` + adapter interface and risks
  rollout drift (assessing).
- **WAN (Wan2.1)** — flow-matching DiT, cleaner single-shot base, strong open
  weights — current front-runner.

## Why it matters

This is the **structural escape** (Option E in the decision) and is orthogonal to
fixing the diffusion target. It keeps the D1 framework's diffusion ↔
flow-matching span intact and could become the primary D3 / D4 backbone.

## Status

Forward-looking plan, not a result. Pyramid's specific blockers to be confirmed;
WAN currently leading. Tracked as Option E in
[[../../50_Decisions/open/shortcut-target-endpoint-vs-v-averaging]].
