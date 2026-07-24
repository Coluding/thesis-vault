---
type: feat
scope: shortcut
status: open
priority: high
created: 2026-06-24
updated: 2026-06-24
resolution:
resolution_note:
closed_at:
related:
  - "[[experiments/exp-shortcut-per-stepsize-loss-diagnosis]]"
  - "[[experiments/exp-shortcut-scale-episodes-longer-train]]"
  - "[[bug-losses-shortcut-v-averaging-target]]"
---

# feat: normalise/reweight per-step-size shortcut loss before pooling

## Context

The diagnosis [[experiments/exp-shortcut-per-stepsize-loss-diagnosis]] resolved the volatile
`shortcut_direction_loss` as **step-size mixing** (Case A): per-rung loss
magnitude scales monotonically with step size — **~15× spread** between the
coarsest (N002 ~0.03) and finest (N064 ~0.002) rung
(`data/results/20262206/loss_new.png`,
[[../30_Knowledge/experiments/avid-shortcut-anchor045-volatile-loss]]). Because
one step level is drawn per batch, the pooled scalar (and the gradient) is
dominated by whichever rung was sampled.

This is not only a logging artifact: the optimizer sees the same uneven
magnitudes, so **coarse rungs get ~15× the effective gradient weight** of fine
rungs from the shortcut term, batch-to-batch. Normalising the per-step-size
loss before pooling makes the objective balanced and the total loss stable.

## What to build

1. **Per-rung normalisation/reweighting** of `shortcut_direction_loss` before it
   enters the total loss — so each step level contributes comparably regardless
   of its raw magnitude. Candidate schemes (decide in-ticket, cheap to compare):
   - running per-rung EMA of the loss magnitude → divide each rung's loss by its
     EMA (scale-free, adaptive);
   - fixed analytic weight from the step-size schedule (closed-form if the
     magnitude scaling is ~predictable from `d`);
   - uniform target after detached per-rung standardisation.
2. **Config flag** to toggle it (default off ⇒ no behaviour change), so the
   `v_average`/biased baseline and the old pooling remain reproducible.
3. Keep the per-rung logging (already in `trainer.py:277-285`) so the effect is
   visible.

## Before implementing — read the code first

- Confirm where the per-rung loss is currently pooled into the total
  (`training/trainer.py` shortcut path) — do not assume; the per-rung re-log at
  `:277-285` may be separate from the scalar that feeds `loss.backward()`.
- Decide reweighting vs the **endpoint-inversion target fix**
  ([[bug-losses-shortcut-v-averaging-target]]) ordering. They are orthogonal
  (one balances magnitudes across rungs; the other removes the per-rung bias
  that makes coarse rungs plateau) — but both touch the shortcut target/loss
  path, so land them in a known order to keep the ablation clean.

## Done when

The reweighting lands behind a config flag with the per-rung curves showing
**comparable magnitudes across rungs** and a **stable pooled/total loss** on a
real run (wandb id + ckpt + commit), feeding the scaled rerun
[[experiments/exp-shortcut-scale-episodes-longer-train]]. No numbers recorded before the run
logs them (hard rule 8).
