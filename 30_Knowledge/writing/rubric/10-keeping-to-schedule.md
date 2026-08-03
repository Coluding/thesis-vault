---
type: writing
status: living
last_updated: 2026-08-03
rubric_item: keeping-to-schedule
category: attitude
current_band: "not yet determined"
target_band: "8-10"
sources:
  - "[[_index]]"
  - "[[../writing-plan-2026-08]]"
---

# Rubric 10 — Keeping to schedule

## The rows

| | |
|---|---|
| **8–10** | **Final version of thesis and colloquium finished within the planned period (or overdue but with good reason)** |
| 7 | Thesis or colloquium at most **10%** of nominal period overdue (without valid reasons) |
| 6 | at most **25%** overdue (without valid reason) |
| 1–5 | at most **50%** overdue (without a valid reason) |

## What it actually asks

The only near-binary item on the sheet, and the only one where **8, 9 and
10 share a single description** — there is no way to distinguish yourself
upward here, only to lose points. Treat it as a *defence* problem: the goal
is to not lose 2–4 points, not to gain any.

Two things follow:

1. **"Or overdue but with good reason" keeps the top band open.** A
   documented, communicated reason is worth up to three points. The
   documentation has to exist *before* the overrun, not after.
2. **Effort spent chasing another point elsewhere is wasted if it costs a
   band here.** This item is the budget constraint on every other note's
   optimisation queue.

## The live risk

> ⚠ `_needs verification_` — the nominal period, submission date, and
> colloquium date are not recorded in the vault. **Record them**; a
> deadline that only exists in someone's head cannot be planned against,
> and the 10%/25% thresholds are meaningless without the denominator.

**The structural risk is scope creep from the parallel experiment track.**
[[../writing-plan-2026-08]] committed on 2026-08-01 to writing while
experiments continue for ~2 weeks in parallel, on the reasoning that they
"can only improve numbers inside a story whose shape is already settled".
That reasoning holds **only while the shape stays settled** — and it has
since moved twice (the 08-02 reframe; the LoRA comparison now in flight
changing D1 from a complexity-analysis contribution to an empirical one).

Each reframe is individually correct and evidence-driven. Collectively they
are the main threat to this item.

## The rule this implies

> **A new result may change a *number* in a section that exists. It may not
> change the *shape* of the thesis.** Any finding that would restructure
> chapters after drafting begins goes to `50_Decisions/open/` and is
> deferred to future work unless it invalidates a claim already written.

The exception is invalidation: a result that shows a written claim is
*wrong* must be acted on immediately (hard rules 7–8 outrank the schedule).

## Optimisation queue

- [ ] **Record the dates** — nominal period start/end, submission,
      colloquium — in this note and in `10_now/product-state.md`.
- [ ] **Freeze the thesis shape** once [[07-thesis-organization]] Q1 (the
      reconciliation pass) lands. After that, storyline changes require a
      decision note, not an edit.
- [ ] **Set a hard cutoff date for new experimental results** entering the
      thesis. After it: results go in as future work or an appendix, not as
      restructuring. The LoRA comparison and any planning artefacts are the
      last two admitted by default.
- [ ] **Write the method and integrity chapters first** — they are the
      least dependent on in-flight results, so they cannot be invalidated
      by them.
- [ ] **Log any slip and its reason in `60_Updates/` as it happens.** This
      is what converts an overrun into "overdue but with good reason" and
      keeps the top band open. Retroactive explanations do not.

## Interaction with the other items

The queues in [[05-reflection]] (Q6), [[03-experimental-evaluation]] (Q2,
Q4) and [[06-literature]] (Q5) are all **writing and reading work with no
GPU dependency** — they can proceed regardless of what the parallel runs
do. That is deliberate: the highest-return rubric work is also the work
least exposed to schedule risk. Prioritise accordingly.

The GPU-dependent items (Q8 seeds, Q9 Action Error Ratio, the LoRA
comparison, planning artefacts) are the ones to time-box.
