---
type: exp
scope: eval
status: open
priority: high
created: 2026-08-03
updated: 2026-08-03
resolution:
resolution_note:
closed_at:
related:
  - "[[../../30_Knowledge/experiments/20260731-dc-condition-center-accelerates-escape]]"
  - "[[../../30_Knowledge/experiments/20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]"
  - "[[../../30_Knowledge/tech/probe-suite]]"
  - "[[../../30_Knowledge/writing/rubric/03-experimental-evaluation]]"
  - "[[../../30_Knowledge/writing/thesis-storyline]]"
---

# Rollout-action-swap on the DC working cell — quantify the control claim

**Why this is the highest-value remaining experiment.** The DC × ACWM cell
passes rungs 1 and 2 of the probe ladder ([[../../30_Knowledge/tech/probe-suite]]):
sensitivity at 3.9× the AVID reference and **all three structure probes
above chance** on held-out `ind_test`. Lukas reports **qualitative rollout
evidence that control also works** on this cell — which would close rung 3
on the one cell where rungs 1–2 already pass, and turn the thesis's positive
result from *structured sensitivity* into *demonstrated control*.

## ⚠ Clarified 2026-08-03 — the existing videos are NOT control evidence

Checkpoints confirmed: **arm E (`6oyu1inq`)** and **arm F (`86kb01su`)**.
The comparison they show is **adapted vs frozen base**.

**That comparison cannot demonstrate control.** A rollout that differs from
the frozen base differs for *some* reason — a better temporal prior, a
domain correction, or action-following — and the comparison cannot separate
them. The standing counterexample is in our own results: the Wan × ACWM cell
beats its frozen base on **6/6** quality metrics (FVD −64%) while every
structure probe sits **at chance**. Distinct-and-better is not
action-following.

**The cheap upgrade:** re-generate the *same clips at the same seeds* with
**wrong-clip** and **zero** action sequences. That is a generation pass, not
a training run, and it converts the existing artefacts into genuine
rung-3 evidence.

## What the existing videos *are* good for (do write this)

They fill a gap that was flagged as blocking the spine: **no DC run logs
quality metrics at all** (checked across all 18 runs in
`dc-acwm-robotarm-avid-parity`). These rollouts are therefore the **only**
evidence about the DC cell's output quality — the cell that carries the
positive D2 result. As a *qualitative quality demonstration* they belong in
the draft, labelled qualitative, making **no claim about actions**.

To be citable they still need:

- [ ] **Artifact location** — path or wandb link.
- [ ] **Step** for each of `6oyu1inq` / `86kb01su`.
- [ ] **Fixed seed + identical initial frame across arms**, stated in the
      caption ([[../../30_Knowledge/writing/thesis-style-guide]] §7).
- [ ] **Held-out clips** (`ind_test`), not training clips — otherwise the
      comparison is on memorised data.
- [ ] Confirmation the frozen-base arm used identical conditioning.

## Measurement options, ranked by strength per unit of compute

| Measure | What it proves | Cost |
|---|---|---|
| **Rollout-swap + tracking error** | true-action rollout tracks GT better than wrong-clip/zero, CIs over clips | generation only |
| **Per-dimension ablation** (ACWM Arm, `da`=7) | zero one action dim at a time; the effect should localise to the *right* joint | generation only |
| **Graded scaling** α·a, α ∈ {0, ½, 1, 1½} | monotone response — uses action *magnitude*, not just presence | generation only |
| **Action Error Ratio** (AVID §4.2) | actions are *decodable* from generated video; external + AVID-comparable | train a small action predictor |
| **End-state error vs commanded** | the cleanest control measure — ACWM is a sim, so ground-truth state exists | needs state extraction |
| **CEM planning** | control in the only sense a world model needs | expensive — [[exp-eval-planning-through-dc-world-model]] |

**Recommended bundle: rollout-swap + per-dimension ablation.** Both are
generation-only on retained checkpoints. Per-dimension ablation is the
underrated one: it is very hard to fake — a domain adapter cannot produce
joint-specific responses — and it upgrades the spatial-concentration probe
from *"localised somewhere"* to *"localised correctly"*.

## Tier 2 — the actual control experiment

The rollout-action-swap protocol already exists but **only on the Wan
path**: `scripts/generate_wan22_i2v_compare.py` (`_action_analysis` is where
the structure triad was ported *from*). There is no DC equivalent, so this
needs a small port.

Cheapest version: reuse the generation config behind the arm E / arm F
videos and vary only the action sequence.

Protocol, same shape as the Wan probe that returned null:

- Same seed, same initial frame, same clip; three action sequences —
  **true**, **wrong-clip** (donor), **zero**.
- Roll out each; measure trajectory tracking against ground truth.
- Report the *gap* between true and the two counterfactuals, with CIs over
  clips, plus the frozen-base null (must be identically 0).

**Pre-register before running** ([[../../30_Knowledge/writing/rubric/03-experimental-evaluation]]):

- [ ] The tracking metric and the threshold that counts as control.
- [ ] The number of clips and seeds.
- [ ] What result would be *uninterpretable* (e.g. all three rollouts
      diverge from GT so fast that no gap is measurable).

## Why it matters beyond one number

- **It closes the ladder on the working cell.** Currently the thesis must
  say "rollout-level control is null on Wan and untested on DC". With this,
  it can say what is actually true on each cell.
- **It is the strongest possible answer to the obvious viva question** —
  *"you measured that the prediction responds to actions; does the model
  actually do what it is told?"*
- **It makes the Wan negative sharper.** A rollout-swap that is positive on
  DC and null on Wan, with the same probe, is a far better contrast than a
  null on one side and silence on the other.
- **Arm 0 matters as much as arm E.** If the *untreated* control shows
  rollout control, the claim becomes "AVID-style output adapters on a weaker
  base produce controllable world models natively" — cleaner, and it does
  not depend on our intervention.

## Definition of done

- Artifacts located and recorded (tier 1), or the probe run (tier 2).
- Result promoted to `30_Knowledge/experiments/{slug}.md` + a ledger row in
  [[../../30_Knowledge/experiments/_index]].
- The per-cell control statement updated in
  [[../../30_Knowledge/writing/thesis-storyline]] §9,
  [[../../30_Knowledge/writing/rubric/_index]], and
  [[../../70_Thesis/outline]] — all three currently say "untested on DC".
- If tier 2 lands positive, §5.2 of the results chapter is upgraded from
  *structured sensitivity* to *demonstrated control*.

## Related

- [[../../30_Knowledge/tech/probe-suite]] §2 — the ladder this completes
- [[exp-eval-planning-through-dc-world-model]] — the natural follow-on:
  control is the precondition for planning
