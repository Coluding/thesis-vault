---
type: decision
status: open
created: 2026-07-31
decided_at:
updated: 2026-07-31
target_date:
scope: conditioning
related:
  - "[[../../30_Knowledge/experiments/20260731-wan-tokennorm-nobase-training-results]]"
  - "[[../../30_Knowledge/experiments/20260731-why-wan-copies-the-base-decomposed]]"
  - "[[../../30_Knowledge/experiments/20260731-dc-condition-center-accelerates-escape]]"
  - "[[../decided/reproduce-avid-on-dc-before-scaling-to-wan]]"
---

# Decision: Wan action-following — change the objective, or lean on DC for D2?

## Context

The 2026-07-30/31 investigation fixed everything architectural on Wan and
measured the result honestly:

- transport fixed (`action_token_norm`): 6–10× single-step sensitivity
- incentive leak plugged (`condition_on_base_outputs: false`): erosion stops
- **ceiling: ~0.011 single-step, and the rollout-swap probe shows sensitivity
  without control** — the true-action rollout tracks GT no better than
  wrong/zero-action rollouts. The improved videos are temporal-prior gains.
- root economics: actions are worth **0.45% of the teacher-forced denoising
  loss**; no architecture change alters that.

Meanwhile DC + `condition_center` demonstrably follows actions at **3.6× the
AVID reference** (0.106) with lower loss than its blind control.

## The choice

**A — change the training objective on Wan** (make actions load-bearing):
action-CFG (train with action dropout, guide at inference), rollout-based /
multi-step losses, or action-magnified loss weighting. Real method work; new
experiment matrix; uncertain payoff; directly attacks the 0.45% economics and
would generalise across backbones (thesis-strongest if it works).

**B — declare Wan's plug-and-play limit a *finding* and carry D2 on DC:**
the scale-calibration + incentive + economics story is a complete,
well-evidenced negative-and-mechanism result for the cross-attention/DiT
backbone; DC carries the positive D2 evidence. Cheapest path to a defensible
chapter; leaves "can a DiT adapter follow actions at all" open.

**C — hybrid:** B for the D2 chapter now, A as one bounded experiment
(action-CFG only, it is the cheapest objective change) to test whether the
economics ceiling moves at all.

## Constraints

- D3/D4 (shortcut) run on Wan — a Wan world model that follows actions is
  needed for D4's combined story eventually.
- Thesis timeline favours B/C over open-ended A.

## Recommendation (analysed, not decided)

C. The mechanism story is publishable evidence either way; one action-CFG arm
bounds the open question without betting the chapter on it.
