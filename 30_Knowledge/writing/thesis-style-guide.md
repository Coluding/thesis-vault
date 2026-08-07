---
type: writing
status: living
last_updated: 2026-08-03
sources:
  - "[[thesis-formal-rules]]"
  - "[[thesis-grading-rubric]]"
  - "[[writing-plan-2026-08]]"
  - "[[thesis-storyline]]"
  - "[[../../70_Thesis/outline]]"
---

# Thesis style guide — academic register and rhetoric

> **Scope.** *How the prose argues.* Mechanics (LaTeX, labels, notation,
> units, citation keys) live in [[thesis-formal-rules]]. What the grader
> rewards lives in [[thesis-grading-rubric]]. Read all three before a
> writing session; `/thesis-write` loads them automatically.

The single sentence this guide exists to protect:

> **The thesis is judged on whether every claim sits at exactly the
> evidence level it has earned — no higher, and no lower.**

Higher is fraud (CLAUDE.md hard rules 7–8). Lower throws away the
contribution. This thesis's results are *mostly negative and partly
provisional*, so the whole grade rides on getting this calibration right —
see [[thesis-grading-rubric]] §Reflection and §Originality.

---

## 1. Voice and register

| Rule | Decision |
|---|---|
| **Person** | **First-person plural — "we".** Used for authorial acts: *we train*, *we measure*, *we argue*, *we choose*. Never for the reader ("we now see that…" → "…"). Single author; "we" is the ML-paper convention and reads naturally for method description. |
| **Spelling** | **British English** — matches the existing vault and draft prose (*analyse, behaviour, normalisation, optimiser, modelling*). Be consistent; do not mix in *behavior*/*normalization*. Exception: **verbatim identifiers keep their source spelling** (`condition_center`, `action_token_norm`, `normalize_action` — these are code, not prose). |
| **Passive voice** | Allowed *only* when the agent is genuinely irrelevant or unknown ("the checkpoint was corrupted"). Default to active. "It was decided that…" is banned — decisions have an author, and this thesis's decisions are a contribution. |
| **Contractions** | None. |
| **Register** | Formal but not stiff. Short declarative sentences beat long qualified ones. Technical precision is the register — pomposity is not. |

**Tense.**

- **Present** for what is permanently true: the method, the composition
  rule, what a cited paper claims, what a figure shows.
  *"The adapter contributes a residual on the frozen base."*
  *"Frans et al. average velocities at the two endpoints."*
  *"Figure 5.3 shows the gate saturating within 2k steps."*
- **Past** for what we did on a particular occasion: runs, measurements,
  decisions, observed outcomes.
  *"We trained the adapter for 40k steps on ACWM push-block."*
  *"The gap fell to 2.5× by the final evaluation."*
- **Present perfect** only for the state of the field ("shortcut models
  have since been extended to…").

Do not narrate the document ("in this chapter we will first…, then we
will…") beyond §1.5 *Thesis structure* and one signpost sentence per
chapter opening. Roadmap paragraphs at every section head are padding and
cost points on *Thesis organization*.

---

## 2. The claim ladder — the central rule

Every factual sentence in the thesis sits on exactly one rung. **The rung
determines the permitted verb and the required apparatus.** This is the
operational form of CLAUDE.md hard rules 7–8 and the readiness table in
[[writing-plan-2026-08]].

| Rung | When it applies | Permitted phrasing | Required apparatus |
|---|---|---|---|
| **1 — Measured** | A completed run with logged outputs, on data the model did not train on, with the confound ruled out by a control | *shows, is, reaches, reduces, we measure* | Provenance receipt (wandb id + ckpt + commit) on the number, and a named control |
| **2 — Measured, single-cell** | Completed and sourced, but one seed / one dataset / one backbone | *in this cell, X reaches…* ; *we observe* | Receipt **plus** an explicit scope clause ("on ACWM push-block with the 5B base"). Never generalise the verb |
| **3 — Provisional** | Completed and sourced, but a named alternative explanation is **not yet excluded** | *is consistent with* ; *we cannot yet distinguish X from Y* | Receipt **plus** a sentence naming the competing explanation **plus** the experiment that would settle it |
| **4 — In flight** | Run launched, not settled | *at the time of writing (snapshot 2026-08-01), the run is at…* | Snapshot date. Never a final endpoint |
| **5 — Analysed estimate** | An inherently judgement-based quantity (expected cost, likely scaling) | *we estimate* ; *an order-of-magnitude argument gives* | Inputs listed, sources cited, reasoning shown, labelled "estimate" in the same sentence |
| **6 — Blocked / excluded** | Evidence is invalidated (in-sample eval, known bug) | **Does not appear as a result.** May appear in Limitations as a named threat | The reason, so the reader sees we found it rather than missed it |

**Three hard consequences for this thesis specifically:**

1. **`effect_rel` is monotone in action-pathway gain.** Any sentence that
   converts an `effect_rel` movement into an *action-information* claim is
   rung 3, not rung 1, until the structure triad returns
   ([[../../00_Inbox/2026-08-01-effect-rel-is-a-gain-metric]]). Write
   *"consistent with a mechanism fix, but a pure gain increase predicts the
   same movement; the arm-E-vs-arm-0 triad discriminates them."*
2. **All RT-1 and OpenVid numbers are in-sample** — rung 6. They do not
   appear in Results in any form until the held-out re-eval lands
   ([[../../00_Inbox/2026-08-01-rt1-heldout-split]]).
3. **The D3 curvature comparison is confounded** by the frozen-gate bug —
   rung 6 until re-run
   ([[../../20_Tickets/bug-adapter-gate-cap-equals-init-freezes-gate]]).

**Rung-mixing is the failure mode to hunt in review.** A paragraph that
opens at rung 1 and quietly extends to a rung-3 conclusion in its last
sentence is the single most damaging thing this document can contain.

### Hedging discipline

Hedge on the **rung**, never on the **fact**. A hedge that expresses real
epistemic status is precision; a hedge that expresses timidity is noise.

- ✅ *"The pedestal mechanism is measured; whether `condition_center`
  repairs it or merely raises the gain is not yet distinguished."*
- ❌ *"The results seem to suggest that the adapter may perhaps be
  somewhat action-insensitive."* — three hedges, no epistemic content.
- ❌ *"The adapter is action-blind."* — banned phrasing, see §4.

Never hedge a number. A number is measured or it is not in the thesis.

---

## 3. Paragraph and section architecture

**One claim per paragraph, stated in the first sentence.** The rest of the
paragraph is evidence, mechanism, or qualification for that claim. If the
paragraph's claim only becomes clear at the end, it is built backwards —
invert it. A reader must be able to read only the first sentence of every
paragraph in a chapter and come away with the chapter's argument.

**The four-move paragraph** (default shape for a results paragraph):

1. **Claim** — what is true, at its rung.
2. **Evidence** — the number, with its receipt, and the control.
3. **Mechanism** — *why* it is true, or explicitly that the mechanism is
   unknown.
4. **Consequence** — what it forces next (the next section, a limitation,
   a design change).

Move 3 is where *Knowledge of study domain* and *Reflection* are earned.
A results paragraph that stops at move 2 is a lab notebook entry.

**Section openings** state the question the section answers, in one
sentence. **Section closings** state what was established and what it
forces. No "In this section we have seen…" summaries in short sections.

**Level of detail must be uniform** within a chapter — the rubric penalises
variance explicitly (*Thesis organization*, 6: "level of detail varies
widely"). If §5.4 has a measured cost table and §5.2 has two hand-wavy
sentences, the fix is to cut §5.4's detail to a table + pointer, not to
inflate §5.2.

**Placement discipline.** Every fact has exactly one home:

| Material | Home |
|---|---|
| What someone else did, and its limitation | Related Work (Ch2) |
| Why we chose an adapter family | Adapters (Ch3) — a *derivation*, not a survey |
| What our system is | Method (Ch4) — no results, no motivation |
| How we would know | Experiments (Ch5) — protocol, metrics, controls, thresholds |
| What we found | Results (Ch6) — numbers with receipts |
| What it means and where it breaks | Discussion (Ch7) |

A number in the Method chapter, or a design justification in Results, is a
placement error and is graded as one.

---

## 4. Terminology discipline

The thesis has a small closed vocabulary. Fix each term once, use it
everywhere, never use a synonym for variety. Elegant variation is a
*defect* in technical prose: two words for one thing implies two things.

**Fixed terms** (define on first use, then never paraphrase):

`frozen base` · `adapter` · `composition rule` · `output adapter` ·
`hidden-state adapter` · `hypernetwork adapter` · `gate` · `step-size
conditioning` · `shortcut target` · `self-consistency` · `action
sensitivity` · `base parity` · `action-informativeness` · `base strength`

**Banned or restricted phrasings:**

| Banned | Use instead | Why |
|---|---|---|
| "action-blind" | **"the adapter learns to correct the base rather than to incorporate actions"** | The measured statement. The RT-1 floor qualifies absolute blindness ([[writing-plan-2026-08]]) |
| "shortcut modelling fails on diffusion" | "the velocity-averaging target is exact for straight interpolants and carries a sagitta bias on a VP arc" | The blunt version is false and attackable — [[../theory/shortcut-v-averaging-bias]] |
| "the model" (unqualified) | name the side: *the frozen base* / *the adapted model* / *the adapter* | The composition rule has three objects; ambiguity here is fatal to the method chapter |
| "diffusion model" where flow matching is meant (or vice versa) | be explicit; state `model_type` and `prediction_type` | CLAUDE.md Part 12, the recurring trap |
| "shortcut" for generic consistency distillation | "self-distillation" / "consistency model" | Different derivations — [[../related-work/shortcut-models]] vs [[../related-work/consistency-models]] |
| "obviously", "clearly", "trivially", "of course" | delete | If it were obvious it would not need saying; if it is not, the word is a bluff |
| "significantly" (non-statistical) | "substantially", or give the number | Reserve for statistical significance only |
| "state-of-the-art" as a claim about our work | name the comparison and the metric | Unsourceable as written |
| "it should be noted that", "in order to", "due to the fact that" | delete / "to" / "because" | Padding |

### Punctuation: no em dashes

**Do not use em dashes (`—`), en dashes as parenthetical breaks, or the
"thinking dash" construction.** Stated preference, 2026-08-07; applies to
the LaTeX draft and to every artefact written for this thesis.

Replace with whichever fits the actual relation between the clauses, which
is usually clearer than the dash was:

| Dash was doing | Use instead |
|---|---|
| introducing an explanation or expansion | a colon |
| joining two closely related independent clauses | a semicolon |
| fencing an aside | parentheses, or commas |
| trailing an afterthought | a new sentence |

If none of those fit, the sentence is carrying two ideas and should be
split. In LaTeX, note that `---` renders as an em dash; use `:`, `;`, `(`
or a full stop instead.

**Ranges keep the en dash** (`5--10`, `\SI{}` ranges, page spans); the rule
is about the dash as a *rhetorical* break, not about numeric ranges.

**Acronyms:** expand on first use in the body (the abstract expands
separately), then use the acronym exclusively. D1–D4 are defined in §1.4
and used everywhere after. Keep a notation table
([[thesis-formal-rules]] §4) and do not introduce a symbol outside it.

---

## 5. Writing a negative result so it counts as a contribution

Most of this thesis's D2 evidence is a failure to reproduce action
sensitivity on a strong base. That is graded well or badly depending
almost entirely on rhetoric. The rubric rewards *diagnosis*, not outcome
(see [[thesis-grading-rubric]] §Reflection, §Experimental evaluation).

**The five-part shape** every negative result in Ch6 must follow:

1. **The expectation, and where it came from.** State the hypothesis and
   cite the prior work or argument that licensed it. A failure is only
   informative against a stated expectation.
2. **The observation, at its rung**, with its control. *The control is
   what makes it a result rather than a bug report.* State the control
   before the reader asks: `anchor_prob: 1.0` for "the consistency loss is
   not at fault", the null-action arm for "the pedestal is real".
3. **The candidate explanations — all of them, enumerated.** Include the
   boring ones (pipeline bug, insufficient capacity, insufficient
   training). A reader who thinks of an explanation we did not list stops
   believing the chapter.
4. **The discriminating measurement**, per explanation, with its
   pre-registered threshold and its outcome — including the hypotheses that
   *survived* and the probes that came back null. **Report the failed
   hypotheses.** They are the evidence that the surviving explanation was
   not simply the first one tried.
5. **What is now known, and what would move the boundary.** End on a
   claim, not an absence ([[../../70_Thesis/outline]] §7.1).

**Never write a failure apologetically.** "Unfortunately, the adapter did
not learn to use actions" is a lab-notebook sentence. The thesis sentence
is: "On a base of this strength, the residual that minimises the training
objective is a correction to the base, not an action-conditioned one; §6.6
measures the three mechanisms that make this so." Same fact, and the second
one is a contribution.

**Never write a failure defensively either.** Do not pre-empt criticism by
listing every possible limitation in the results text — that belongs in
§7.3, once, thoroughly. Sprinkling apologies dilutes the measured claims.

**The methodological finding is a first-class contribution.** Loss, gate
statistics, FID and sample quality are all blind to action-blindness;
diagnosis required purpose-built probes. Write that as a result with its
own subsection, not as an aside ([[writing-plan-2026-08]]).

---

## 6. Using the literature

Related Work is an **argument**, not an annotated bibliography. Its job is
to make our question the obvious next one.

- **Organise by problem, not by paper.** A subsection per *tension in the
  field*, with papers as evidence inside it. A paragraph that begins
  "Chen et al. (2024) propose…" and ends without connecting to the next
  paper is a catalogue entry.
- **Cite the concept, name the actor.** Use `\citep`-style parenthetical
  citation when the *idea* carries the sentence; use the authors as a
  grammatical subject only when the *actors* matter (a disagreement, a
  direct comparison, a method we build on directly).
- **Every related-work paragraph ends in a limitation or a delta.** What
  does this line of work not do that our question requires?
- **Be accurate about what a paper claims.** Hard rule 7a applies to
  citations with full force: no "the paper probably means…". If the paper
  is ambiguous, say the paper is ambiguous, and cite the page.
- **AVID is the working starting point, not a gap we fill.** Action
  conditioning on a frozen base is AVID's contribution; our delta is
  elsewhere ([[../../70_Thesis/outline]], [[../../10_now/positioning]]).
  Getting this wrong reads as not understanding the closest prior work —
  the most expensive possible error on *Use of literature*.
- **Cite the negative space.** Where we searched and found nothing (e.g. a
  flow-matching video model with a temporal VAE —
  [[../../00_Inbox/2026-08-01-flow-model-no-temporal-vae-search]]), say so
  explicitly. An absence stated with its search scope is a finding; an
  absence left silent looks like an omission.

---

## 7. Figures, tables and captions as prose

Treated as mechanics in [[thesis-formal-rules]] §6; the rhetorical rules:

- **Every figure makes exactly one point, stated in the caption's first
  sentence.** Caption structure: *claim* → *what is plotted* → *provenance*.
  A caption that only says what is plotted wastes the highest-attention
  real estate in the document.
- **Every figure and table is referenced in the text and interpreted
  there.** "Figure 6.2 shows the results" is not an interpretation.
- **Axes, units, and n on every plot.** Error bars or a statement of why
  there are none (single seed → say "single seed" in the caption; the
  rubric asks explicitly for quantitative treatment of uncertainty).
- **Sample grids need a fixed seed and an identical prompt/action across
  arms**, stated in the caption, or they are decorative.
- Prefer one honest, dense table over three sparse ones.

---

## 8. Pre-commit checklist (run before marking any section `draft-complete`)

1. Does every paragraph's first sentence carry its claim?
2. Is every number attached to a provenance receipt
   ([[thesis-formal-rules]] §5)?
3. Is every claim at its rung, with rung-3 claims naming both the
   competing explanation and the discriminating experiment?
4. Does any paragraph start at rung 1 and end at rung 3?
5. Are `D1`/`D2`/`D3`/`D4` evidence kept in their own sections?
6. Is "the model" used unqualified anywhere?
7. Does every figure/table have a claim-first caption, provenance, and an
   in-text interpretation?
8. Does the section state the question it answers and what it forces next?
9. Any banned phrasing from §4 left in?
10. Is anything in the section on rung 6 (blocked/in-sample/confounded)?

Items 2, 3 and 10 are the ones that end careers. Check them twice.
