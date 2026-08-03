---
type: decision
status: decided
created: 2026-07-30
decided_at: 2026-07-30
updated: 2026-07-30
target_date:
scope: adapter
related:
  - "[[../../30_Knowledge/experiments/20260730-avid-robotarm-follows-actions-recipe-not-data]]"
  - "[[../../30_Knowledge/experiments/20260728-acwm-robotarm-matrix-action-blind]]"
  - "[[../../30_Knowledge/tech/avid-vs-ours-action-conditioning]]"
  - "[[../../20_Tickets/experiments/exp-adapter-our-framework-avid-replication-robotarm]]"
  - "config: configs/dynamicrafter/diffusion_dc_acwm_robotarm.yaml"
---

# Decision: reproduce AVID with our code on DynamiCrafter before scaling to Wan

## Status

**Decided 2026-07-30.** D2's near-term goal is to make **our** adapter reach
AVID-level action-following on **DynamiCrafter × ACWM Robot Arm**, measured by the
same probe. Wan work on action-conditioning is paused until that target is hit.

## Context

[[../../30_Knowledge/experiments/20260730-avid-robotarm-follows-actions-recipe-not-data]]
showed the unmodified AVID recipe reaches `action_effect_rel` **0.029475** (null
0, ~42% of adapter contribution action-driven) on ACWM Robot Arm, while our three
adapters on the same data sit at 0.0013–0.0056 with ~1.5–5.6% action-driven. Same
frozen base weights, same episodes, same probe — so the fault is on our side.

**There is no single shared cause to chase.** The two identified DC divergences
are both DC-specific or DC-only in practice:

- `action_time_combine: add` vs AVID's `concat` — lives in the DC ResBlock
  time-embedding path; Wan and SkyReels inject actions by cross-attention, so the
  toggle does not even apply to them.
- `frame_stride: 4` (actions stride-**summed**, `acwm_phys.py:215`) vs AVID's
  `frame_stride: 1` — but **Wan and SkyReels configs already use
  `frame_stride: 1`** and were blind anyway.

This confirms the 2026-07-28 read of "3 distinct starvation signatures". Chasing
all three bases at once means three simultaneous unknowns and no target number.

## Decision

**Reproduce first, generalise second.**

1. Make our DC adapter reach AVID parity on Robot Arm — a **pass/fail target**
   (`action_effect_rel` ≳0.02, `base_null_violation` ≈0, ideally
   effect ÷ adapter approaching 0.42) against a reference that provably achieves
   it on the identical substrate.
2. Only then port the lesson to Wan, which is where the thesis contribution
   actually lives (Wan is the flow-matching base carrying D3/D4).

## Why this ordering

- DC is the **only** base with a known-good reference. Debugging Wan means
  optimising against no target; debugging DC means closing a measured gap.
- The two candidate deltas are already named and source-verified — this is a
  short, bounded search, not an open-ended one.
- A DC reproduction is also **D1 evidence** (our framework reproduces a published
  recipe), which the framework chapter currently lacks almost entirely.
- It converts a five-week dead end into a positive result either way: the knobs
  that decide whether a plug-and-play adapter uses its conditioning.

## Consequences

- **Paused:** Wan/SkyReels action-conditioning debugging, and the ACWM
  optimisation-countermeasure ticket family (`exp-adapter-*-nobase-overfit`,
  `-gatelow-cap-sigmashift-`, `-wan-cap50-warmup-`) — those target a copy-through
  trap on a substrate whose diagnosis is now in flux.
- **Not paused:** D3 shortcut work (`pzmc2orq` / `t4bp8nki`, the few-step eval) —
  a separate deliverable on a separate axis.
- **Migration is not mechanical.** The DC fix (`action_time_combine`) has no Wan
  equivalent. What transfers to Wan is the *principle* (actions need their own
  representational subspace; don't aggregate away per-frame detail) plus the probe
  methodology — not the toggle. Budget a real port, not a config copy.
- **Every action-conditioned run from here logs the probe.** Three of our four DC
  runs (`gxq7kxzp` 17 h, `kjgt3z0f`, `t4bp8nki`) logged no action-sensitivity keys
  at all. `action_sensitivity_probe: true` is now mandatory for D2 runs.

## Derived tickets

- [[../../20_Tickets/experiments/exp-adapter-our-framework-avid-replication-robotarm]]
  — the three-arm reproduction (concat-only, frame_stride-1-only, both).
