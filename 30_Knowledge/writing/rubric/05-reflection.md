---
type: writing
status: living
last_updated: 2026-08-03
rubric_item: reflection
category: research
current_band: "6-7"
target_band: "9"
sources:
  - "[[_index]]"
  - "[[../../experiments/20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]"
  - "[[../../experiments/_index]]"
  - "[[../thesis-style-guide]]"
---

# Rubric 5 — Reflection

> **⭐ The highest-return item in the thesis.** The largest gap between
> what exists (a lot) and what is written (almost nothing). Currently 6–7
> because the reflection lives in vault notes rather than in
> `70_Thesis/draft/`. A **9** is achievable with writing alone — no GPU, no
> new experiments.

## The rows

| | |
|---|---|
| 10 | In full sync with their work and in harmony with its limitations, resulting in completely new insights |
| **9** | **Reflect and learn from their own work in such a way that other researchers in the field can be helped** |
| 8 | Reflect on their work, understanding the pros and cons, as well as insights for improvement |
| 7 | Reflect on the outcomes, with common lessons learned |
| 6 | **Minimal** reflections |
| 1–5 | Not capable of reflecting |

## What it actually asks

The 9-row is unambiguous and unusually generous: **"other researchers in
the field can be helped."** It does not ask for a successful method. It
asks for a *transferable lesson* extracted from your own experience —
which is exactly what a rigorous negative campaign produces and a
successful one often does not.

The 8→9 delta is scope: 8 reflects on *this work*; 9 produces something
*someone else can use*.

## The headline reflection — write this

The sharpest sentence in the vault, and it is not in the draft:

> Loss, gate statistics, FID and sample quality are **all blind** to
> whether a conditioned model uses its conditioning. Our clearest
> demonstration: on Wan × ACWM the adapter **beats its frozen base on 6/6
> metrics (FVD −64%, 1118 → 406)** while all three structure probes sit
> **at chance**. The same adapter is a *domain* adapter on one base and an
> *action* conditioner on another — never both. Anyone training a
> conditioned generative model on standard readouts alone cannot tell
> these apart.

That is the 9-row, verbatim: a lesson that helps other researchers,
derived from our own work, and backed by a matched control
([[../../experiments/20260802-adapter-is-a-domain-adapter-not-an-action-conditioner]]).

## Evidence inventory — the raw material is unusually strong

The vault's own history is a reflection record. All of it currently
invisible to a grader:

| Reflective act | What happened |
|---|---|
| **A headline retracted** | the SkyReels cell voided by our own config audit — the 0.0450, the "91% of reference", the 35× all withdrawn |
| **An inference superseded, its measurement preserved** | the 07-29 "blindness is a data/OOD problem" reading, killed by the 07-30 AVID-on-our-data control; the number stands, the conclusion does not |
| **A level claim retired** | DC arm E's 3.5× fell to 2.5× without converging; only the ~6× *acceleration* was kept |
| **Our own primary metric distrusted** | `effect_rel` identified as monotone in gain **by us**, then discriminated with a purpose-built temporal control |
| **A framing abandoned** | "DC is the spine, Wan is the failure branch" — superseded when the evidence stopped supporting it |
| **Hypotheses that died** | σ-mismatch (flat sweep), "our data is too hard" (AVID control), "capacity is the problem" (the 7.5M simple arm), "binning fixes it" (bought nothing) |

**Reporting the dead hypotheses is what makes the surviving one
credible.** A reader who sees only the winning explanation assumes it was
the first one tried.

## The honest limit — state it, do not bury it

> **Rollout-level control is not demonstrated.** On Wan the rollout-swap
> probe is null — same seed, same clip, swapped actions, and the
> true-action rollout tracks ground truth no better than a wrong-action or
> zero-action one. On DC, where the structure probes *do* clear chance,
> the rollout-swap has not been run. What we establish is a ladder —
> sensitivity, then structure — and where each cell falls on it; the top
> rung is measured only on the cell that fails it.

⚠ **Do not write "control is demonstrated nowhere."** It understates the DC
cell, whose structure triad is above chance on all three axes on held-out
data, and it is contradicted by our own probe table. The defensible
statements are *per cell* — see [[_index]].

This belongs in the **abstract**. Under this rubric a clearly stated,
well-characterised limit scores; a vague or hidden one destroys the item.
It is also what keeps [[01-originality]] honest.

## Gaps

1. **None of the above is in `70_Thesis/draft/`.** The item is scored on
   the document.
2. **No dedicated section exists for it.** Reflection scattered through a
   Discussion chapter reads as hedging; concentrated in a named section it
   reads as method.
3. **No "what we would do differently" content.** The 8-row asks for
   "insights for improvement" — we have them (held-out splits from day one,
   probe before scaling, instrument validation before trusting a metric)
   and they are unwritten.

## Optimisation queue

- [ ] **Q6 — Write the blindness section.** Its own chapter section, not a
      Discussion paragraph. Contents: the standard readouts and what each
      is blind to; the FVD −64% ÷ probes-at-chance contrast as the
      demonstration; the probe suite as the alternative; what it cost us to
      find out. *(~2 h, the single best writing investment in the thesis.)*
- [ ] **Write the dead-hypothesis inventory** — each with the measurement
      that killed it. Pairs with [[03-experimental-evaluation]] Q2.
- [ ] **Write "what we would do differently"** — held-out splits enforced
      at the data layer; validate the instrument before trusting it;
      structure probes before quality metrics; pre-register the decision
      rule before the run.
- [ ] **Put the control limit in the abstract**, in the author's own words,
      before a reader can find it themselves.
- [ ] **Reflect on the methodology as a reusable contribution**, not as an
      apology: the probe suite is the thing another group would actually
      adopt from this thesis.

## Where it lands in the thesis

- **Abstract** — the control limit, stated plainly
- **Ch6 (results), own section** — "standard metrics are blind" + the probes
- **Ch5 or Appendix** — methods integrity (shared with [[03-experimental-evaluation]])
- **§7.3 Limitations** — concentrated, thorough, once. Not sprinkled
- **§8.2 Future work** — what would move the boundary: an objective that
  pays for actions (action-CFG, rollout losses, action-conditional
  consistency) + the structural repairs
