---
type: exp
scope: shortcut
status: in-progress
priority: high
created: 2026-06-17
updated: 2026-06-17
resolution:
resolution_note:
closed_at:
related:
  - "[[exp-shortcut-scale-episodes-longer-train]]"
  - "[[exp-shortcut-vs-image-only-anchor-baseline]]"
  - "[[../../50_Decisions/open/shortcut-collapse-mitigation-anchor-vs-gate]]"
---

# exp: Diagnose volatile shortcut loss — log shortcut loss per step size

## Context

The first real-backbone shortcut run (AVID/DynamiCrafter, `anchor_prob=0.45`,
larger MetaWorld) showed a stable `base_loss` (~0.06–0.07) but a **volatile
`shortcut_direction_loss`** (~0.01–0.12, no downtrend) — the total-loss spikes
track the shortcut term, not the base term. See
[[../../30_Knowledge/experiments/avid-shortcut-anchor045-volatile-loss]].

**This is the gating diagnosis** for the scaled rerun
([[exp-shortcut-scale-episodes-longer-train]]) — until we know whether the
volatility is an artifact or real instability, we don't know whether the scaled
run needs a loss fix first.

## The question

Is the volatility a **logging/scaling artifact** — one aggregated
`shortcut_direction_loss` scalar pooling step sizes whose loss magnitudes
differ by orders of magnitude (the `log2` schedule spans `1/128`→`1`) — or
**genuine training instability** of the self-consistency objective?

## What's being done (in progress)

Logging `shortcut_direction_loss` **separately per step size** instead of only
the pooled scalar.

**The logging is already implemented in code** (verified 2026-06-19):
`training/trainer.py:277-285` re-logs the shortcut loss under a per-rung wandb
key `shortcut_direction_loss/N{steps}` (`s=1→N001`, `0.5→N002`, `0.25→N004`,
…). One `s_full` is drawn **per batch** in the schedule path
(`trainer.py:598-599` sets `_last_shortcut_step_level`), so each step
contributes to exactly one rung series — clean grouping, no within-batch mix.
Anchor steps and the reset (`:558`) leave it `None`, so they log to no rung.

Caveats:
- Only the **schedule-based distillation path** sets the per-rung key; the
  legacy no-schedule path (`trainer.py:613+`) does not. The `anchor_prob=0.45`
  config uses `shortcut_step_schedule` → it's covered.
- Per-rung curves are **sparse** (one `s` per batch) — gappy lines by design.

So the remaining work is to **run and read the per-rung curves**, not to write
logging code.

## Read the result as

- **Each per-step-size curve smooth, pooled scalar noisy** → the volatility is
  step-size mixing. Fix = normalise/reweight the per-step-size loss before
  pooling (spawn a `feat-shortcut` ticket for the reweighting), then the scaled
  run is safe to launch.
- **Individual per-step-size curves themselves volatile** (esp. the coarse
  levels) → genuine instability of the consistency objective at coarse steps.
  Fix lives upstream (warmup/anchor schedule, target method, coarse-step
  clipping) before any scaling.

## Outcome — 2026-06-24: Case A (step-size mixing)

The per-rung curves are in `data/results/20262206/loss_new.png` (six
`shortcut_direction_loss/N{steps}` series, ~1.6k steps). Loss magnitude scales
**monotonically with step size** — ~15× spread between coarsest (N002 ~0.03)
and finest (N064 ~0.002) rung — and each rung is individually well-behaved (no
within-rung order-of-magnitude jumps). Since one `s_full` is drawn per batch,
the pooled scalar bounces over that 15× range purely from which rung was
sampled, reproducing the 0.01–0.12 pooled volatility. **The volatility is a
pooling/scaling artifact, not genuine instability.** Full reading:
[[../../30_Knowledge/experiments/avid-shortcut-anchor045-volatile-loss]].

Per the Case-A branch above: spawned the reweighting fix
[[../feat-shortcut-per-stepsize-loss-reweighting]], and the scaled run
[[exp-shortcut-scale-episodes-longer-train]] is no longer gated on objective
stability (it should still fold in the reweighting + ideally the
endpoint-inversion target [[../bug-losses-shortcut-v-averaging-target]]).

Secondary note (out of scope for this ticket): coarse rungs **plateau** while
fine rungs decline — the v-averaging-bias signature, tracked by
[[../bug-losses-shortcut-v-averaging-target]], not a stability problem here.

## Done when

The per-step-size breakdown is logged for a run (wandb id + commit), and we can
state which of the two cases holds — feeding the go/fix decision for
[[exp-shortcut-scale-episodes-longer-train]]. No numbers recorded before the
run logs them (hard rule 8).
✅ **Met** (Case A). Status held at `in-progress` pending your OK to close.
