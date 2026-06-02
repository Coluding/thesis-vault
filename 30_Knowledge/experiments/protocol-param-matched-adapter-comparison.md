---
type: experiment-protocol
status: draft
last_updated: 2026-05-22
deliverable: D1, D2
related:
  - "[[../../50_Decisions/open/param-matched-adapter-comparison-definition]]"
  - "[[../../20_Tickets/exp-adapter-param-matched-comparison]]"
  - "[[../related-work/avid]]"
  - "[[../related-work/hyperalign]]"
  - "[[../related-work/unicon]]"
---

# Protocol — Param-matched comparison across adapter families

> **Status: draft skeleton.** Several methodology calls are unresolved and
> are tracked in
> [[../../50_Decisions/open/param-matched-adapter-comparison-definition]].
> Do not launch runs from this protocol until that decision closes.
> Promoted from [[../../00_Inbox/2026-05-21|inbox 2026-05-21 16:17]].

## Hypothesis

When trainable-parameter budget is held constant across adapter families
(full fine-tuning of a subset, LoRA, hypernetwork, AVID-style output
residual), the resulting differences in prediction accuracy, rollout
stability, and inference cost reflect **where** the capacity is spent —
weight delta vs. hidden-state injection vs. hypernetwork-generated delta
vs. output-level residual — rather than how much capacity each family
receives by default.

Without budget-matching, any cross-family comparison is confounded by
parameter count. The user-observed datapoint that motivates this: the
current AVID-shortcut MetaWorld setup uses ~0.78% of backbone params as
trainable (_user-reported, needs verification — count the trainable
params under `configs/diffusion_avid_shortcut_metaworld.yaml` and record
the exact figure in this file before launching_). LoRA at default rank
and a hypernetwork at default hidden-size both land at very different
fractions; comparing them at their defaults conflates "which family" with
"how many params."

## Falsification conditions

The hypothesis is falsified — and the comparison becomes uninteresting in
its current framing — if either of the following holds at the resolution
this study runs at:

1. **Within-budget variance ≫ between-family variance.** Re-running the
   same family with different seeds at the same budget produces a metric
   spread comparable to or larger than the gap between families. Then the
   ranking is noise.
2. **Ranking flips across budgets.** If family A wins at 0.1% but loses
   at 3%, "which family is best" has no budget-independent answer; the
   chapter has to reframe around budget-dependent regimes instead of a
   single recommendation.

Both of these are themselves thesis-worthy observations, but they would
change the chapter's conclusion shape.

## Open methodology decisions (blocking)

These must be resolved in
[[../../50_Decisions/open/param-matched-adapter-comparison-definition]]
before the protocol is runnable. Each is here as a placeholder; the
decision file is where the answer is reasoned through.

1. **Definition of "matched."** Candidates: total trainable params,
   trainable params per backbone block, training-time FLOPs, training-time
   peak memory, inference-time FLOPs. These give *different* rankings —
   e.g. a hypernetwork is cheap at inference but its hyper-MLP is expensive
   to train; LoRA matches on params but its rank choice changes FLOPs.
   Pick one as primary and report the others as secondary observables.
2. **Single budget vs. sweep.** Anchor at the AVID 0.78% as the single
   matched budget, or sweep across {0.1%, ~0.8%, 3%} to test whether the
   ranking is budget-dependent? Sweep is the stronger contribution and
   the more honest answer, but 4 families × 3 budgets × N seeds is a lot
   of GPU-hours.
3. **Where this lives in the thesis.** D1 (framework characterisation,
   "the framework supports apples-to-apples comparison") or D2 (empirical
   adapter-trade-off result with action conditioning)? Or both, with D1
   carrying the methodology and D2 carrying the action-conditioned
   instance.
4. **Whether to include full fine-tuning as a row.** The thesis is
   explicitly *not* a fine-tuning paper
   ([[../../10_now/positioning#anti-positioning]]). But fine-tuning a
   parameter-matched subset of the base is the natural "trivial" baseline.
   Decide whether it is in scope as a baseline or excluded as out-of-scope.

## Design (skeleton, pending the decisions above)

### Axes held fixed

- **Backbone.** _Pending decision_ — most natural choice is the
  DynamiCrafter backbone used by the current AVID shortcut configs, so
  that AVID is its own family-row instead of being re-implemented on a
  different base.
- **Dataset.** MetaWorld (the D2 implemented choice — see
  `tests/test_metaworld_dataset.py`,
  [[../../10_now/positioning#D2]]).
- **Action conditioning interface.** Identical across families. The
  per-family adapter changes; the action embedding path does not.
- **Training schedule.** Same optimizer, LR, batch, total step count,
  seed set. Same anchor-step warmup schedule
  ([[../../50_Decisions/decided/shortcut-anchor-schedule]]) if shortcut
  supervision is in this study.
- **Shortcut on or off?** _Pending decision_ — turning shortcut on makes
  this a D4-style study; off makes it a clean D2 adapter-family
  comparison. Default lean: shortcut OFF for the first round, so we
  isolate "where the capacity is spent" without coupling to the
  consistency-loss dynamics. Revisit for a follow-up that adds shortcut.

### Axes varied

- **Adapter family.** {AVID-style output residual, LoRA on the base,
  HyperAlign-style hypernetwork → LoRA, UniCon-style hidden-state}. _Add
  full-FT-subset row pending decision (4)._
- **Budget.** Single point at the matched fraction, OR
  {0.1%, ~0.8%, 3%} pending decision (2).
- **Seeds.** ≥ 3 seeds per (family, budget) cell so within-cell variance
  is measurable. Fewer seeds make falsification condition 1 unverifiable.

### Metrics — primary

- Prediction accuracy: MSE of one-step prediction on held-out MetaWorld
  trajectories at matched (x_t, t, a_t). Single number, reported with
  cross-seed std.
- Rollout stability: per-step prediction error growth out to N steps on
  held-out trajectories (curve, not single number). N pending decision.
- Inference cost: FLOPs/step and wall-clock/step on a fixed GPU, both
  for the adapter alone and for adapter + base composition.

### Metrics — secondary (recorded but not headline)

- Training cost: GPU-hours to a fixed metric, plus peak training memory.
- The other "matched on X" budgets the families happen to land at when
  matched on the primary. So a row matched on trainable-params reports
  its FLOPs, memory, and inference cost as observed numbers, not as
  matched constraints.

### Sweep grid

```
{
  family:   [avid_output, lora, hyperalign_hyper, unicon_hidden]  # + full_ft_subset?
  budget:   [single anchor]  OR  [low, mid, high]
  seed:     [s1, s2, s3, ...]
}
```

Total cells: 4 (or 5) × {1 or 3} × ≥3 = **12 – 45 runs.** _Compute
budget feasibility check needed before committing to the sweep variant._

## Pre-registration / what we commit to before running

To keep this honest, we commit to the following before the first run
finishes:

- The primary metric (one of the candidates above).
- The headline figure shape (e.g. "bar chart, x = family, y = MSE,
  error bars = seed std" vs. "ranking-flip line plot across budgets").
- That we report **all** cells, not only the cells that support the
  hypothesis.
- That if the within-seed variance swamps the between-family signal, we
  say so plainly in the chapter rather than picking a favourable seed.

## Deliverable mapping

Resolves once decision (3) closes. Skeleton:

- If D1: the protocol itself is part of the framework story
  ("the framework lets you compare adapter families fairly because it
  exposes a single composition interface — and here's what that
  comparison looks like").
- If D2: the *result* is the empirical core of the adapter-trade-off
  analysis, and the protocol description goes in the chapter's methods
  section.

## What this protocol does NOT cover

- Multi-modality (D2/D4 extension). Out of scope for v1.
- Action-conditioned shortcut combined (D4). Out of scope until D2 +
  D3 individually have evidence ([[../../10_now/positioning#D4]]).
- Cross-backbone generalisation. The point of this protocol is the
  family-vs-family axis on *one* backbone; backbone-as-axis is a
  separate study.

## Related

- Decision: [[../../50_Decisions/open/param-matched-adapter-comparison-definition]]
- Ticket:   [[../../20_Tickets/exp-adapter-param-matched-comparison]]
- Inbox origin: [[../../00_Inbox/2026-05-21]] (16:17 entry)
- Adapter family neighbours: [[../related-work/avid]],
  [[../related-work/hyperalign]], [[../related-work/unicon]],
  [[../related-work/cafm]]
- Positioning: [[../../10_now/positioning#D1]], [[../../10_now/positioning#D2]]
- Code anchors (to verify trainable-param counts):
  - `configs/diffusion_avid_shortcut_metaworld.yaml`
  - `configs/diffusion_hyperalign_shortcut_metaworld.yaml`
  - `src/generative_flow_adapters/adapters/`
