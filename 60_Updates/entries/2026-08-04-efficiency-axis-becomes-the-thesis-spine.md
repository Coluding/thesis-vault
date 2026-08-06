---
date: 2026-08-04
category: decision
deliverable: D3
meeting:
sources:
  - "[[../../50_Decisions/decided/efficiency-axis-as-thesis-spine]]"
  - "[[../../30_Knowledge/writing/thesis-storyline]]"
  - "[[../../30_Knowledge/writing/rubric/01-originality]]"
  - "[[../../20_Tickets/experiments/exp-conditioning-add-actions-to-shortcut-adapter]]"
---

# Decision: efficiency of adapted world models becomes the thesis spine

## What

The thesis question changes from *when and why does adapter-based adaptation
work* to **where should few-step behaviour live when you adapt a frozen
prior**. The single "shortcut" node expands into **three acceleration
levels**, and the ending becomes a comparative analysis with a design
recommendation rather than a diagnosis.

The storyline chain barely moves — only two nodes:

```
DC + adapter works → AVID → too slow → flow → rectified flow
    → DISTILLATION TO FEW STEPS (three levels)
    → analysis: what worked, what did not, and why
```

## Why it matters

The three levels are not three implementations of one idea. They differ in
**where the few-step behaviour lives**, and therefore in whether the two
adaptations are entangled:

| Level | Speed lives in | Entanglement |
|---|---|---|
| **L1** shortcut adapter | the adapter | one adapter learns actions *and* step size |
| **L2** PDD / LoRA distillation | a second adapter | two adapters, sequential |
| **L3** distilled base | the base | acceleration free; only conditioning learned |

This mirrors the adapter taxonomy — *where does trainable capacity sit
relative to the frozen prior?* — asked of the **speed** axis instead of the
**conditioning** axis.

**The key point for the meeting:** this is not a shoot-out. The 0.45 %
loss-share result **predicts the ordering before the runs**. Actions are
already outbid by appearance correction; adding a consistency objective to
the *same* adapter adds a second, larger claim on the same gradient budget.
So the economics bound stops being a descriptive limit and becomes a
**predictive tool**.

The D2 mechanism campaign is not displaced — it becomes the *conditioning*
axis of a two-axis results chapter, which is where its 20+ runs already
point.

## Evidence / status

**Pre-registered 2026-08-04, before any level ran** (timestamped in Git):

- **H-E (primary):** the deficit is a **gradient-budget** effect, not a
  parameter-budget one. L1 below its matched conditioning-only control;
  L2/L3 at control.
- **H-C (secondary, scoped):** capacity sufficient for one objective may be
  insufficient for two. Distinct from the capacity hypothesis already
  **killed** in the ablation; testable via the 34.9M vs 47M arms.
- **Both branches are results.** If L1 matches its control, H-E is wrong at
  this scale, separability is unnecessary — and D4 is delivered.

Run status: **L1 has never been trained** (it is D4, ticketed since June);
L2 ticketed; **L3 has first data** as of 2026-08-05, which already forced a
protocol change.

## Next

- Write the common per-level measurement protocol — without it the three
  cells are not comparable and the pre-registration cannot be evaluated.
- **Every level needs a matched conditioning-only control**; the first L3
  data arrived without one and is confounded three ways as a result.
- L1 (= D4) is the load-bearing run: it is simultaneously the entangled
  level, the integration chapter and the prediction's target.
