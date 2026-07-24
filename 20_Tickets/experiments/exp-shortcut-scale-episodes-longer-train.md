---
type: exp
scope: shortcut
status: open
priority: high
created: 2026-06-17
updated: 2026-06-24
resolution:
resolution_note:
closed_at:
related:
  - "[[exp-shortcut-per-stepsize-loss-diagnosis]]"
  - "[[exp-shortcut-vs-image-only-anchor-baseline]]"
  - "[[../../50_Decisions/open/shortcut-collapse-mitigation-anchor-vs-gate]]"
---

# exp: Scaled shortcut run — more episodes + longer training

## Context

The early "few-step robust" read was a **small-data overfitting artifact**: on
the larger MetaWorld set the AVID shortcut adapter (`anchor_prob=0.45`)
**degraded** (foggy/collapsed + drift frames). See
[[../../30_Knowledge/experiments/avid-shortcut-anchor045-volatile-loss]] and the
update [[../../60_Updates/entries/2026-06-17-shortcut-overfit-larger-data-volatile-loss]].
The direct test of the overfit hypothesis is: give it **more data + more
training** and see whether prediction quality recovers.

## Gate lifted — 2026-06-24

The diagnosis [[exp-shortcut-per-stepsize-loss-diagnosis]] resolved the loss
volatility as a **step-size-mixing artifact, not genuine instability** (Case A;
`data/results/20262206/loss_new.png`). The objective is stable, so this run is
**no longer blocked**. Before launching, fold in:
- the per-step-size loss reweighting
  [[../feat-shortcut-per-stepsize-loss-reweighting]] (balances the ~15× per-rung
  magnitude spread), and ideally
- the endpoint-inversion target [[../bug-losses-shortcut-v-averaging-target]]
  (fixes the coarse-rung plateau).

Both are quick loss-path changes; landing them first keeps the scaled run's
curves interpretable and the budget well spent.

## Arms / levers

Start from `data/results/20261706/config_run_anchor_prob=045.yaml` (AVID /
DynamiCrafter velocity, output adapter `avid_mask_mix`, step-level
conditioning, `anchor_prob=0.45`). Change only the scale axes:

| Axis | This run |
|---|---|
| Episodes / trajectories | **More** — same MetaWorld tasks + cameras, more rollouts per task |
| Training length | **Longer** — more steps; keep `eval_metric: base_loss`, `keep_last_checkpoints`, eval cadence |
| Loss handling | Apply any per-step-size normalisation from the diagnosis (else identical) |
| Everything else | Identical (seed, schedule, conditioning) for comparability vs the 0.45 run |

## Hypothesis

If the degradation was overfitting, more episodes + longer training should
**recover prediction quality** and shrink the gap to the frozen-base/anchor
arms — and the `base_loss` should keep improving without the shortcut term
destabilising it. If quality **still** degrades with more data, overfitting is
not the cause and the problem is the shortcut objective itself (escalate to
the collapse-mitigation decision).

## Metrics

- `base_loss` and per-step-size `shortcut_direction_loss` vs training step
  (does volatility shrink with more data + the fix?).
- Sample quality vs the 0.45 small-/larger-data runs, same eval step schedule
  {1,2,4,8,25}.
- Toward the owed **MSE-vs-NFE curve** on held-out rollouts.

## Done when

A scaled run is logged (wandb id + ckpt + commit), compared against the 0.45
run, and we can state plainly whether more data + longer training recovers
shortcut prediction quality. No numbers before the run logs them (hard rule 8).
