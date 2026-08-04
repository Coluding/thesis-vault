---
type: exp
scope: shortcut
status: open
priority: high
created: 2026-08-03
updated: 2026-08-03
resolution:
resolution_note:
closed_at:
related: ["[[exp-shortcut-pdd-lora-distill-dc]]", "[[../../30_Knowledge/experiments/20260802-shortcut-works-on-flow-not-diffusion]]", "[[exp-shortcut-d3-fewstep-vs-noshortcut-control]]"]
---

# Parallel-decoding adapter on frozen Wan (action-free, D3)

> **⭐ Part of the efficiency axis** —
> [[../../50_Decisions/decided/efficiency-axis-as-thesis-spine]] (decided
> 2026-08-04). This ticket is **L2 — separable** (speed in a second adapter on the base).
> **Requires a matched conditioning-only control** (same adapter, base, data
> and depth, acceleration off) — without it the pre-registered comparison is
> unmeasurable. The predicted ordering is registered in that note **before**
> any level ran; do not restate it post hoc.


## Idea (Lukas, 2026-08-03)

Give the **adapter** PDD's parallel decoder: one forward pass emits `L` corrected
velocity directions instead of one, and the target is evaluated **on the states the
adapter's own predictions reach** rather than on a teacher rollout.

Source: Shaul, Liu, Vahdat & Berner, *Parallel Decoding Distillation*
(arXiv:2607.26004). Lesson + cheat sheet: `85_Learning/`.

## Why the adapter form suits this

1. **The frozen base is amortised.** `f_base(x_t,t)` is computed once and anchors all
   `L` predictions. PDD amortises only its backbone; we amortise the 5B base too.
2. **`condition_on_base_outputs: true` is already on** — the adapter already consumes
   the base output (`output_head.py`, `in_channels = feature_dim*2`).
3. **Zero-init gives the right prior for free.** All heads zero-init
   (`action_model.py:252-253`) ⇒ every `Δ^k = 0` ⇒ every predicted mean velocity
   equals the base's instantaneous velocity — a constant-velocity Euler rollout.
   Training only learns how interval `k`'s mean velocity *deviates*. PDD has to
   initialise heads from the teacher's `W` to get the analogous prior; the residual
   composition hands it to us.

## What changes

| Change | Location |
|---|---|
| `Head(dim, out_dim)` → `Head(dim, out_dim * N)`, split after unpatchify (one GEMM, not a ModuleList) | `action_model.py:207, 438` |
| Zero-init all N slices | `action_model.py:252-253` |
| Return `[B, N, C, T, H, W]` | `Wan21OutputAdapter.forward`, `OutputAdapterResult` |
| **Shared** mask for `mask_mix`, not per-head | `action_model.py:211, 440` |
| On-policy target: one adapted pass → N dirs → roll to `x_k` (detached) → **one** base RK step there | new branch, `trainer.py:1845` |
| Head fusion at inference (PDD Eq. 15) | sampler |

**Constraint — fusion requires a shared mask.** Composition must stay linear in the
adapter output: `Σ_k Δ_k·(base·m + Δ^k·(1−m)) = base·m + (Σ_k Δ_k Δ^k)·(1−m)` is exact
only if `m` does not depend on `k`. A per-head mask breaks PDD Eq. 15.

## Cost

`Head` = `Linear(448 → prod(1,2,2)·48 = 192)` ≈ 86.2k params (~87k with modulation).
- N=8 → +610k = **1.7%** of the 35,172,932-param adapter
- N=16 → +1.31M = **3.7%**

Per-step cost **drops**: on-policy PDD needs **1** base eval per loss vs the current
`substeps=4`.

Measured throughput (job `25166226`): **5.07 s/step** @ batch 1, `billing=192`.
3000 steps ≈ 4.2 h ≈ **~810 SBU**.

## Design choices still open

- **N = 8 or 16?** Leaning 8–16 with `step_level` conditioning **dropped entirely** —
  the head index replaces the second time coordinate, which is the point of PDD's
  architecture (paper §3: "without the need to introduce a second time coordinate").
- Euler vs Midpoint for the single teacher step. Paper's ablations (Fig. 6, Tables 3–5)
  favour Midpoint consistently, at 2 teacher evals instead of 1.

## Control

Reuse the existing action-free no-shortcut control
(`wan-noshortcut-control-actionfree-robotarm`) and the existing `v_average` arm as the
two comparison points. Metric: `consistency_cos` via `scripts/eval_stepsize_blindness.py`,
same probe as [[../../30_Knowledge/experiments/20260802-shortcut-works-on-flow-not-diffusion]].

## Prerequisite

Fix `trainer.py:470-471` first — it stores one target under both `shortcut_target` and
`self_consistency_target`, so `shortcut_direction_loss` and `multistep_consistency_loss`
are the same number (observed identical at 0.16743 in job `25166226`) and the objective
is silently applied at weight 2.0. Pre-existing; affects the `v_average` arm too.
