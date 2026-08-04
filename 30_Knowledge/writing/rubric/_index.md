---
type: writing
status: living
last_updated: 2026-08-03
sources:
  - "[[../thesis-grading-rubric]]"
  - "[[../thesis-style-guide]]"
  - "[[../thesis-storyline]]"
  - "[[../ablation-axes]]"
  - "[[../../experiments/_index]]"
---

# Rubric optimisation — hub

> **Working method (decided 2026-08-03).** The thesis is graded on **10
> rubric items**. Rather than write prose and hope it scores, we take each
> item as an **optimisation target with its own note**: what the rows
> literally ask, what evidence we already have, what is missing, and the
> ordered actions that move it. **We optimise each item first, then write
> the coherent text** — the prose becomes an assembly step over material
> that already satisfies the criteria.

Rubric verbatim + the structural read: [[../thesis-grading-rubric]].
Which evidence moves which item: [[../evidence-map]].
How the prose must argue: [[../thesis-style-guide]].

**Bands below are an analysed estimate** (CLAUDE.md hard rule 7b) — a
judgement about *fit to the rubric's wording*, not a predicted mark.
Inputs: [[../../experiments/_index]] (25 sourced runs),
[[../thesis-storyline]] §REFRAME, [[../../../70_Thesis/outline]].

## The ten items

| # | Item | Category | Current | Target | The one thing that moves it | Note |
|---|---|---|---|---|---|---|
| 1 | Originality of the research | Research | 7–8 | **8** | State the pathway result as a scoped **design principle**, not a Wan bug | [[01-originality]] |
| 2 | Technical skills | Research | 8–9 | **9** | Seeds on the headline A/B + write the probe suite as a *validated instrument* | [[02-technical-skills]] |
| 3 | Experimental evaluation | Research | 7–8 | **8–9** | Hypothesis-first axis table + a methods-integrity subsection | [[03-experimental-evaluation]] |
| 4 | Knowledge of study domain | Research | 8 | **8–9** | Generalise pathway + economics into a claim about conditioning frozen priors | [[04-knowledge-of-domain]] |
| 5 | Reflection | Research | 6–7 | **9** | The "standard metrics are blind" section — **highest return in the thesis** | [[05-reflection]] |
| 6 | Use of literature | Thesis | 6–7 | **8** | Write the missing notes (`lora`, `controlnet`, world models, few-step) | [[06-literature]] |
| 7 | Thesis organization | Thesis | 7–8 | **9** | Reconcile the stale layers, then budget pages by argument weight | [[07-thesis-organization]] |
| 8 | Writing | Thesis | — | **8–9** | Probe figures that read without the text | [[08-writing]] |
| 9 | Independence | Attitude | 8–9 | **9** | Attribute every design choice to its reasoning in-text | [[09-independence]] |
| 10 | Keeping to schedule | Attitude | — | **8–10** | Freeze scope; parallel runs must not move the target | [[10-keeping-to-schedule]] |

Items 1–5 and 9 are supervisor/committee-facing judgements informed by the
document; 6–8 are the document itself; 10 is calendar.

## The consolidated queue

Ordered by rubric return per hour. **Nothing here is prose-writing** —
this is the material that must exist before coherent text is worth writing.

| # | Action | Moves | Cost | Status |
|---|---|---|---|---|
| Q1 | **Reconcile the stale layers** to the 08-02 reframe (see below) | 7, 1 | ~1 h | ✅ 2026-08-03 — all 7 layers |
| Q2 | **Rewrite `ablation-axes.md`** — it predates the entire 07-30→08-02 campaign and the LoRA run | 3, 7 | ~2 h | ✅ 2026-08-03 — hypothesis-first, 11 hypotheses × 13 axes, verdict-bearing |
| Q3 | **Probe-suite instrument spec** — each probe, its null, its chance level, its validation | 2, 5, 3 | ~2 h | ✅ 2026-08-03 → [[../../tech/probe-suite]] |
| Q4 | **Methods-integrity inventory** — every retraction/bug/confound, detection→consequence | 3, 5 | ~2 h | ✅ 2026-08-03 → [[../methods-integrity]] |
| Q5 | **Missing literature notes** (`lora`, `controlnet`, FiLM/adaLN, world models, few-step baselines) | 6, 4 | bounded | ☐ **needs sourcing from the papers themselves** (hard rule 7a) |
| Q6 | **The blindness claim, written** — standard metrics vs probes | 5, 4, 1 | ~2 h | ✅ 2026-08-03 — drafted as prose into `latex/chapters/50-results.tex` §5.6 |
| Q7 | **Design-principle statement** for §1.4 (scope of validity included) | 1, 4 | ~1 h | ✅ 2026-08-03 — §1.4 drafted in `latex/chapters/10-introduction.tex`, 6 contributions + scope + what-we-do-not-claim |
| Q8 | Seeds on the headline clean-room A/B | 2, 3 | GPU | ☐ |
| Q9 | Action Error Ratio as the second, external readout | 3, 6 | needs a run | ☐ |
| Q10 | Export probe figures (propagation trace, pedestal, structure triad) | 8 | moderate | ☐ |

**Correction logged 2026-08-03:** "control is demonstrated nowhere" was an
overstatement propagated through several of these notes. DC arm E clears
chance on all three structure axes on held-out data. See the framing box
below — state the ceiling **per cell**.

## The stale-layer list (Q1)

**✅ Q1 CLOSED 2026-08-03 — all seven reconciled.** Kept as the record of
what was wrong and why. Two of these notes are now *banner-superseded*
rather than deleted, so they still read as authoritative until you reach the
banner — the standing trap when drafting from Git history or from a stale
tab.

| Layer | Stale claim | Superseded by | Fixed |
|---|---|---|---|
| [[../ablation-axes]] Axis 2 | "LoRA … is not run. No LoRA config." | LoRA-vs-adapter comparison **in flight** (2026-08-03) | ✅ 2026-08-03 |
| [[../ablation-axes]] (whole note) | Axis map predates 07-30→08-02; missing injection-pathway, scale-calibration and capacity axes | [[../thesis-storyline]] §REFRAME | ✅ **rewritten** 2026-08-03 |
| [[../writing-plan-2026-08]] | **only** its "Spine correction" — Wan as the collapse branch, and the global headline that follows from it | [[../thesis-storyline]] §REFRAME | ✅ **partially** superseded 2026-08-03 — its §D2-e downgrade and in-sample warning were *correct* and still hold |
| [[../../../70_Thesis/outline]] | **8-chapter structure**, separate adapters chapter, `draft/25-adapters.md` | the **40-page limit** — 6 chapters, see [[07-thesis-organization]] | ✅ 2026-08-03 |
| [[../../../70_Thesis/outline]] §3.3 | "Cost only — no quality comparison (ruled out by Axis 2)" | the LoRA run makes a quality comparison available | ✅ 2026-08-03 |
| [[../../../10_now/positioning]] anti-positioning | planning is "a sanity check, not the contribution" | storyline §2 elevates it (scoped) | ✅ 2026-08-03 |
| [[../../../70_Thesis/index]] | Markdown draft is the source of truth; Method is Ch3 | LaTeX tree + 6-chapter proposal | ✅ 2026-08-03 |

**Template constraints now known** (2026-08-03): UvA MSc AI, `report`/12pt,
**40-page upper limit**, appendices additional, citation style free. Full
list in [[../thesis-formal-rules]] §1.

## The framing that all ten items depend on

Settled 2026-08-02, restated here because six of the notes below reference
it and one stale phrasing would damage several items at once:

> **The method works.** On DynamiCrafter × ACWM the adapter is a genuine
> action conditioner: `effect_rel` **0.11479** (3.9× the AVID reference),
> **all three structure probes above chance on held-out data**, and the
> action-following solution is **better on the denoising loss itself**
> (0.0357 vs the control's 0.0433). The untreated control clears the AVID
> reference *unaided* — the cell works natively.
>
> **On Wan, it depends on the pathway — and that is the whole point.**
> State it per cell, never per backbone:
>
> | cell | injection | verdict |
> |---|---|---|
> | Wan × ACWM | cross-attention | **domain corrector** — 6/6 quality (FVD −64%), all three structure probes **at chance** |
> | Wan × RT-1 (clean-room) | **per-frame AdaLN** | **follows actions** — 2.49× @12000 (Welch t=10.5), diagonal concentration 0.409 vs chance 0.200 |
> | DC × ACWM | concat + centring | follows actions — triad above chance |
>
> **The decisive contrast is within Wan**, not between backbones: the same
> base, the same data, matched adapter contribution and matched mask, and
> only the injection pathway differs. That eliminates base strength as a
> confound entirely — which the DC-vs-Wan comparison cannot do.

**This is a positive result with a matched architectural negative control**
— not a failed method with a good post-mortem. Lead with it in that order.

**⚠ Weighting (decided 2026-08-03): Wan is the contribution; DC is the
positive control.** The evidence mass is on Wan — 13 axes, the mechanism
campaign, the depth trace, the decomposition, the clean-room A/B. DC is an
*existence proof* that the method works and the reference the Wan analysis
is measured against. **Do not inflate DC into the headline**: it is one
cell, cancelled pre-convergence, with no quality metrics and no control
measurement. Keep §5.2 compact; the pages belong to the Wan mechanism work.

**The distinction to make sharp — what the action signal *is* in each cell:**

| | DC (works) | Wan (fails) |
|---|---|---|
| direction | steering +0.117 — **directional** | cos ≈ 0.00 — **arbitrary** |
| time | alignment 1.000 (chance 0.313) — **frame-addressable** | **at chance** — px→latent correspondence never formed |
| space | concentration 0.470 (chance 0.100) — **localised** | ≈ chance — **uniform** |

Wan's signal is a **global bag**: present, measurable, and structureless.
DC's is structured on all three axes. *That* is the contrast the thesis
turns on — not "one works and one doesn't".

**⚠ Two phrasings to never use** (both corrected 2026-08-03):
- ~~"the approach does not work on Wan"~~ — Wan wins 6/6 on quality.
- ~~"control is demonstrated nowhere"~~ — DC's structure triad clears
  chance on all three axes. State the ceiling **per cell**: rollout-level
  control (rung 3) is *null on Wan*; on DC there is **qualitative evidence,
  artefacts pending** ([[../../../20_Tickets/experiments/exp-eval-rollout-action-swap-dc-arme]]),
  and the quantitative probe exists only on the Wan path so it has not been
  run there. Nothing about DC control enters the draft until it lands.

**Live limits to state alongside:** no DC run logs quality metrics (all 18
checked), so the working cell's perceptual quality is unknown; the runs were
cancelled pre-convergence, so quote `condition_center`'s **~6×
acceleration**, never a level; `effect_rel` is monotone in gain, so the
structure triad is what carries the DC claim.

Never write "the approach does not work on Wan"
([[../thesis-grading-rubric]] §4.1). It contradicts our own tables and
trades the thesis's most original claim for a weaker one.
