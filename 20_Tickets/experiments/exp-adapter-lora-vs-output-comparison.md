---
type: exp
scope: adapter
status: in-progress
priority: high
created: 2026-08-03
updated: 2026-08-03
resolution:
resolution_note:
closed_at:
related:
  - "[[../../30_Knowledge/writing/rubric/01-originality]]"
  - "[[../../30_Knowledge/writing/rubric/03-experimental-evaluation]]"
  - "[[../../30_Knowledge/writing/ablation-axes]]"
  - "[[../../50_Decisions/decided/param-matched-adapter-comparison-definition]]"
  - "[[exp-adapter-param-matched-comparison]]"
---

# LoRA vs. output adapter — head-to-head

**In flight as of 2026-08-03** (reported by Lukas). Compares a **LoRA**
weight-update adapter against the **AVID-style output adapter** on the same
frozen base, to test whether the output adapter is actually the better
choice or merely the one we committed to.

> ⚠ `_needs verification_` — run parameters not yet in the vault: base
> backbone, dataset, matched budget (params / FLOPs / GPU-hours), step
> count, wandb run ids, seeds, metric set. **Fill these before any result
> is promoted to `30_Knowledge/experiments/`** (hard rule 6).

## Why it matters beyond the run

This is **not** the large FLOPs-matched sweep in
[[exp-adapter-param-matched-comparison]] (39–48 runs, blocked since
2026-05-22 on the FLOPs estimator). It is a focused two-family comparison —
but it has outsized consequences for the thesis:

1. **It upgrades D1.** [[../../30_Knowledge/writing/ablation-axes]] Axis 2
   currently states "LoRA … is not run. No LoRA config," which forced D1 to
   be a **software + complexity-analysis** contribution and made
   [[../../70_Thesis/outline]] §3.3 declare "cost only — no quality
   comparison (underpowered; ruled out by Axis 2)". A landed LoRA
   comparison lets Ch3 carry an **empirical** family claim instead of a
   cost argument alone.
2. **It closes an open ablation axis** — "the adapter family is wrong" is
   one of the candidate explanations for the action-blindness result and is
   currently untested (see the hypothesis table in
   [[../../30_Knowledge/writing/rubric/03-experimental-evaluation]]).
3. **It answers the obvious viva question** — *why an additive output
   adapter rather than PEFT on the base?* Currently answered on principle
   (frozen-base-with-output-access-only; detached backward), not on
   evidence.

## ⚠ Pre-register the decision rule — before it lands

Per [[../../30_Knowledge/writing/rubric/03-experimental-evaluation]], the
pre-registration is worth more to the *Experimental evaluation* rubric item
than the outcome is. Write down **now**, timestamped:

- [ ] What is held matched (trainable params / training FLOPs / wall-clock)
      and what is merely reported.
- [ ] The primary metric and the threshold that would count as "the output
      adapter is better" — and the one that would count as "LoRA is
      better". Both directions, stated in advance.
- [ ] Whether action-sensitivity (probe suite) or generation quality
      (FID/FVD) is primary. **They can disagree** — the 08-02 result shows
      a cell that wins 6/6 on perceptual metrics with probes at chance
      ([[../../30_Knowledge/experiments/20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]).
- [ ] Seed count. Single-seed leaves the comparison open to the same
      criticism as the rest of the campaign.
- [ ] What result would **not** be interpretable (e.g. LoRA under-trained
      at matched steps because it converges differently).

## Definition of done

- Run parameters and wandb ids recorded in this ticket.
- Both arms run to a logged outcome at the matched budget.
- Result promoted to `30_Knowledge/experiments/{slug}.md` + a row in
  [[../../30_Knowledge/experiments/_index]].
- [[../../30_Knowledge/writing/ablation-axes]] Axis 2 rewritten (it is
  currently stale in the opposite direction).
- Decide whether
  [[../../50_Decisions/decided/param-matched-adapter-comparison-definition]]
  needs an update — it ruled out a quality comparison as underpowered, and
  this run partially reopens that.

## Out of scope

- The full FLOPs-matched grid across all families
  ([[exp-adapter-param-matched-comparison]], still blocked).
- Hypernetwork and hidden-state families — they stay in the
  complexity-analysis argument.
