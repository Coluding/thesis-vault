---
type: feat
scope: shortcut
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
---

# feat: displacement shortcut parametrisation (Option B) — config-selectable

## Context

Second endpoint-consistent target from
[[../50_Decisions/decided/shortcut-target-endpoint-vs-v-averaging]]. Displacements
compose **exactly and schedule-independently** (`Δ(2d) = Δ(d) + Δ′(d)` —
point-differencing telescopes regardless of curvature), so it is the conceptually
cleanest fix and a cross-check on the endpoint-inversion arm
([[bug-losses-shortcut-v-averaging-target]]). Derivation:
[[../30_Knowledge/theory/shortcut-v-averaging-bias]] §3 / §5 (Option B).

## Why it is more than a target swap

Unlike Option A (target-only), B reparametrises:

- **the head output** — predicts net displacement `Δ` instead of velocity `v`;
- **the few-step sampler** — composes additively (add displacements), no longer
  DDIM-recompose for that path;
- **the `d→0` grounding** — anchor on `≈ d·v_true`.

So this lands *after* A and shares A's config selector
(`shortcut_target_method` → add `displacement` as a value, or the sibling
selector A introduces).

## Done when

`displacement` is selectable via config, trains end-to-end, and the additive
sampler produces few-step rollouts; the displacement target passes the same
synthetic-ray exactness check as A (residual 0 at every rung). Compared against
`endpoint_inversion` and the `v_average` baseline as the D3 ablation axis. No
numbers before a run logs them (hard rule 8).
