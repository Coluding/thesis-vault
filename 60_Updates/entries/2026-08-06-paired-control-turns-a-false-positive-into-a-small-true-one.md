---
date: 2026-08-06
category: finding
deliverable: D2
meeting:
sources:
  - "[[../../30_Knowledge/experiments/20260806-motion-tracking-is-action-driven-but-the-base-control-was-wrong]]"
  - "[[../../30_Knowledge/experiments/20260805-turbo-action-tokens-binned-to-latent-grid]]"
  - "[[../../20_Tickets/experiments/exp-adapter-action-on-distilled-wan-turbo]]"
---

# The distilled-Wan arm: a control we trusted was measuring the wrong thing

**One line:** on the distilled Wan Turbo action arm, the action *does* modulate how much
the arm moves — but the effect is ~10× smaller than a preliminary hand measurement
suggested, and the difference is entirely down to which control we compared against.

## What we thought we had

A hand measurement over six eval videos showed the adapted rollout tracking per-clip
ground-truth motion at **r = +0.75**, against the frozen base at **+0.09** — a gap of
0.66, and the first sign on this arm that the action was doing anything useful.

## What the instrumented run says

Run `mo3k2639`, 16 clips per draw, four draws across two eval cycles:

| comparison | draws | verdict |
|---|---|---|
| adapted − **frozen base** | +0.13, +0.10, +0.045, **−0.034** | sign-inconsistent, ≈ 0 |
| adapted − **shuffled actions** | +0.069, +0.168, +0.122, +0.213 | **positive 4/4**, mean +0.143 |

The frozen base — which never sees an action — tracks per-clip motion about as well as
the adapter does. The 0.66 gap does not survive.

## Why the two controls disagree, and which one is right

The frozen base *looks* like a control: it is the same rollout with the adapter removed.
But it differs from the adapter in capacity, conditioning pathway and training — not only
in action access. So "adapted beats base" can be explained by any of those.

The **paired shuffled-action control** holds all of them fixed: same weights, same
conditioning frame, same seed, same clips, and swaps in another clip's actions. It is the
only comparison in which the action is the sole variable.

**Had we kept the frozen base as the control, this would be recorded as a strong positive
result.** The paired control is what turned it into a small true one. That is the
transferable lesson, and it applies to every action-conditioning claim in the thesis.

## What the arm actually supports now

- A **modest, consistently-signed** action effect on motion **magnitude**.
- **Not** action correctness: `action_loss_gap` finished at 0.00005 — feeding the *right*
  action still does not reduce error versus a wrong one. The dissociation reported on
  2026-08-05 is unchanged.
- `corr(adapted, GT)` on its own spans zero in 3 of 4 draws; only the *gain* is consistent.

## Caveats to state at the meeting

- **No confidence interval on the gain yet.** The intervals logged are on the wrong
  series. The estimator now exists but cannot be applied retroactively — this run stored
  only summary correlations. The next run produces the first real intervals.
- n = 16 per draw, 4 draws, **one run, one seed**.

## Next

Re-run the arm to obtain gain intervals; that decides whether +0.143 is a result or noise
at this sample size.
