---
type: writing
status: living
last_updated: 2026-08-03
rubric_item: thesis-organization
category: thesis
current_band: "7-8"
target_band: "9"
sources:
  - "[[_index]]"
  - "[[../../../70_Thesis/outline]]"
  - "[[../../../70_Thesis/index]]"
  - "[[../thesis-storyline]]"
---

# Rubric 7 — Thesis organization

## The rows

| | |
|---|---|
| 10 | Perfect structure and balance. No section too long or short. All sections in harmony |
| **9** | **Each section has a clear and unique function. Hierarchy correct. Ordering logical. All information in the correct place. Level of detail appropriate throughout** |
| 8 | Clear and unique function; hierarchy correct; ordering mostly logical; correct placement with few exceptions |
| 7 | Main structure mostly correct; placement illogical in certain places |
| 6 | Sections not logical in places; **overlapping functions** → ambiguity in placement; **level of detail varies widely** |
| 1–5 | Structure incorrect in places; placement illogical in many places |

## What it actually asks

Three separable things: **unique function per section** (no two sections
doing the same job), **correct placement** (every fact has exactly one
home), and **uniform level of detail**. The last is called out explicitly
in the 6-row and is our specific risk — see "Balance" below.

## Where we stand

**The structure itself is good, and the hard problem is already solved.**
[[../../../70_Thesis/outline]] resolves the arc-vs-deliverable tension
explicitly: *the arc is told in Ch1; the later chapters keep their evidence
deliverable-separated and merely order themselves along it.* That is
9-row thinking — each section has a clear and unique function.

The risk is not the structure. It is **narrative churn**: the storyline has
been reframed repeatedly in a short window, and the layers have not caught
up with one another.

## ⚠ The stale-layer list — fix before any prose is written

Drafting from any of these produces chapters that contradict the vault's
own evidence. **Cost of fixing now: ~1 h. Cost of fixing after two
chapters: those two chapters.**

| Layer | Stale claim | Superseded by |
|---|---|---|
| [[../ablation-axes]] Axis 2 | "LoRA … is not run. No LoRA config." | the LoRA comparison **in flight** |
| [[../ablation-axes]] (whole) | axis map predates the 07-30 → 08-02 campaign | [[03-experimental-evaluation]] §axis inventory |
| [[../writing-plan-2026-08]] | "Wan is the generality branch — it collapses" | [[../thesis-storyline]] line 154, superseded 08-02 |
| [[../../../70_Thesis/index]] | Method is **Ch3** | outline: Adapters is Ch3, Method is Ch4 |
| [[../../../70_Thesis/outline]] §3.3 | "cost only — no quality comparison" | the LoRA run enables one |
| [[../../../10_now/positioning]] | planning is "a sanity check, not the contribution" | storyline §2 elevates it (scoped) |
| [[../../../70_Thesis/outline]] §6.6 | "base-parity collapse on Wan" as the framing | "Wan works; the pathway decides" |

Also: `draft/25-adapters.md` is referenced by the outline but **does not
exist**.

## ⚠ The 40-page limit changes everything (template received 2026-08-03)

The UvA MSc AI template sets a **40-page upper limit**, at `report`/`12pt`
with 2 cm margins, with **appendices as additional optional pages**. Full
constraints: [[../thesis-formal-rules]] §1.

**The 8-chapter outline does not fit.** Ch3 (adapters) alone was budgeted
6–8 pp, and the old structure had Abstract + 8 chapters. At ~500 words a
page, 40 pages is roughly 20k words *including figures* — and a diagnostic
thesis loses 6–8 pages to plots.

Two consequences, both of which resolve open questions in this note:

1. **The chapters must merge.** The template's own suggested spine is
   Introduction / Related work / Method / Experiments / Conclusions. The
   adapter-families chapter folds into Method; Discussion folds into
   Conclusion.
2. **The appendix is the pressure valve.** It sits outside the 40 pages.
   The run inventory, formal probe definitions, and the long tail of
   methods-integrity detail belong there — which lets the D2 dominance
   problem below be solved by *placement* rather than by cutting evidence.

### Page budget — **signed off 2026-08-03**

Implemented in `70_Thesis/latex/main.tex` as comments on each `\include`.

| Chapter | Budget | Function |
|---|---|---|
| 1 Introduction | 4–5 pp | the arc, told once; contributions |
| 2 Related work | 5 pp | organised by tension; makes our question next |
| 3 Method | 10 pp | D1 framework **+ adapter families** + curvature derivation |
| 4 Experiments | 7 pp | protocol, **probe suite**, hypothesis-first ablation, integrity |
| 5 Results | 10 pp | D2 mechanism study, D3, the blindness section |
| 6 Discussion & conclusion | 4 pp | the boundary, limitations, future work |
| **Total** | **40 pp** | |
| Appendix | +N | run inventory, probe definitions |

`_needs verification_`: whether the 40 counts the roman-numbered front
matter (TOC + abstract) or only the arabic body. Plan for body-only and
confirm with the supervisor.

## Balance — the specific 6-row risk

The rubric penalises "level of detail varies widely", and our evidence is
lopsided by construction:

| Deliverable | Evidence |
|---|---|
| D1 (framework) | strong — 3 backbone families + the AVID-repo port; LoRA comparison in flight |
| D2 (action world models) | **20+ runs, 13 axes** — dominant |
| D3 (shortcut) | one clean positive, one confound flagged |
| D4 (combined) | none — descoped or explicitly stated as such |

**Chapters must not inherit that proportion literally**, or Ch6 becomes 70%
D2 and the thesis reads as unbalanced regardless of quality. Two viable
resolutions:

- **(a) Budget pages by argument weight, not run count.** D2's 20 runs
  collapse into ~4 mechanism claims; report the claims, put the run
  inventory in an appendix.
- **(b) Restructure so the D2 mechanism work is presented as *the study*,**
  with D1/D3 as supporting chapters and D4 explicitly descoped. This
  matches the 08-02 reframe (an empirical study, not a systems build) and
  is probably the honest structure.

**(b) is the recommendation** — but it is a structural decision that should
be recorded in `50_Decisions/` before chapters are written, not made
implicitly by whichever chapter gets drafted first.

## Placement rules (the "unique function" half)

Enforced in [[../thesis-style-guide]] §3. Restated because this is where
placement errors get graded:

| Material | Home |
|---|---|
| What others did + its limitation | Ch2 Related Work |
| *Why* we chose an adapter family | Ch3 Adapters (a derivation, not a survey) |
| What our system *is* | Ch4 Method — no results, no motivation |
| How we would know | Ch5 Experiments — protocol, metrics, controls, thresholds |
| What we found | Ch6 Results — numbers with receipts |
| What it means / where it breaks | Ch7 Discussion |

Known overlap risk: Ch3 (adapter families, D1) and Ch2 §2.1 (adapters/PEFT)
have adjacent functions. The outline already resolves it — §2.1 is a
**pointer only**, the substance is in Ch3 — but that resolution must
survive drafting, or it becomes the 6-row's "overlapping functions leading
to ambiguity in placement".

## Optimisation queue

- [ ] **Q1 — Reconciliation pass.** Fix every row in the stale-layer table.
      Blocking: nothing else in this note is safe to do first. *(~1 h)*
- [ ] **Decide the D2-dominance resolution** (a) or (b), and record it in
      `50_Decisions/open/` → `decided/` rather than letting drafting decide.
- [ ] **Sign off the 6-chapter structure + page budget** above, then
      rewrite `70_Thesis/outline.md` against it. The old 8-chapter outline
      (and its `draft/25-adapters.md` reference) is superseded by the
      40-page limit — do not create that file.
- [ ] **Port the existing Markdown prose to LaTeX** — the `fs` boundary,
      the shortcut target construction and the computational profile in
      `draft/30-method.md` are real prose and must be moved, not rewritten.
- [ ] **Confirm whether front matter counts** toward the 40 pages.
- [ ] **One-sentence function statement per section**, written into the
      outline before drafting. If two sections' statements overlap, merge
      or re-scope them now.

## Where it lands in the thesis

Everywhere — this item is the outline itself. The artifacts are
`70_Thesis/outline.md`, `70_Thesis/index.md`, and the page budget.
