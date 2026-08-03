---
type: writing
status: living
last_updated: 2026-08-03
rubric_item: use-of-literature
category: thesis
current_band: "6-7"
target_band: "8"
sources:
  - "[[_index]]"
  - "[[../../related-work/_MOC]]"
  - "[[../../../70_Thesis/outline]]"
---

# Rubric 6 — Use of literature and theoretical background

> **The weakest item, and the most mechanically fixable.** Its score is
> currently set by a *missing-file count*, not by judgement. Bounded work
> with a known list — the highest-certainty point gain available.

## The rows

| | |
|---|---|
| 10 | Clear, complete, relevant background, **perfectly tailored** / Literature complete |
| 9 | Clear, complete, and relevant background / Literature complete |
| **8** | **Relevant background used, nicely synthesized, successfully tailored to the research at hand / Literature complete (optionally overcomplete)** |
| 7 | Relevant background used, description shows minimal errors / literature study **almost** complete |
| 6 | Relevant background used, **occasional errors** / literature study **not complete** |
| 1–5 | Serious errors and/or limitations |

## What it actually asks

Two independent gates, and **completeness is the binding one for us**:

1. **Completeness** — 8 and 9 both require "complete"; 7 is "almost
   complete"; 6 is "not complete". This is closer to a checklist than a
   judgement.
2. **Synthesis + tailoring** — 8 says "nicely synthesized and successfully
   tailored to the research at hand". A catalogue of summaries does not
   reach 8 no matter how many entries it has.

## Where we stand

**Have (13 notes in `30_Knowledge/related-work/`):** avid, cafm,
consistency-models, dpm-solver, dreamzero-wam, hyperalign,
self-distillation, shortcut-models, surfing-uncertainty, unicon,
unified-world-models, weaver, plus the MOC.

**Quality of what exists is good** — the AVID note carries a Correction
section that repositioned the entire thesis, which is 8-row behaviour
("successfully tailored to the research at hand"). The problem is coverage,
not depth.

**⚠ Structural exposure:** [[../../../70_Thesis/outline]] §3.1 requires
`controlnet.md` and `lora.md` for the adapter-taxonomy chapter, and
**neither exists**. The D1 taxonomy contribution currently rests on two
unwritten notes. With the LoRA-vs-adapter comparison in flight, `lora.md`
also becomes an *experimental* dependency, not just a citation.

## The gap list — candidates to verify and write

⚠ **Every entry below is a search target, not a sourced claim.** Verify the
paper exists, read it, and write the note per the paper-note frontmatter
contract before citing. Do not cite from this list.

**(a) Adapter / PEFT families — blocks Ch3 (D1)**

| Note | Why it is required |
|---|---|
| `lora.md` | Axis-3 family in the taxonomy **and** the in-flight comparison baseline |
| `controlnet.md` | AVID's own baseline; named in outline §3.1 and in AVID's comparison table |
| FiLM / feature-wise modulation | **Directly underwrites the pathway claim** — our AdaLN-vs-cross-attention finding is a statement about multiplicative modulation of normalised features. Citing the lineage turns an empirical result into a principled one |
| DiT / adaLN-Zero | the conditioning mechanism our winning arm uses; the architectural context for "scale-free relative to the residual stream" |
| NLP adapter lineage (bottleneck adapters, prefix-tuning) | the taxonomy's origin; cheap completeness |
| T2I-Adapter / IP-Adapter | image-side conditioning adapters; adjacent family |

**(b) World models / action-conditioned video — the title's own field**

Thin relative to a thesis titled around world models. Candidates to check
for coverage: classical latent world models (Ha & Schmidhuber; the Dreamer
line), diffusion/transformer world models for interactive settings
(DIAMOND, GameNGen, Genie, iVideoGPT), large-scale learned simulators
(UniSim-style, Cosmos-style world foundation models), and Diffusion
Forcing. **Goal is not breadth for its own sake** — it is to place our
frozen-base-adapter setting against the retrain-from-scratch alternative,
which is the comparison a committee will raise.

**(c) Few-step sampling — makes the D3 baseline honest**

Have: consistency-models, self-distillation, dpm-solver, shortcut-models.
[[../thesis-storyline]] §3 already flags that 50-step DDIM alone is not a
fair baseline. Check coverage of progressive/step distillation and
rectified-flow-style reflow so the D3 comparison cannot be called a straw
man. Note there is already an open decision on this
([[../../../50_Decisions/open/d3-positioning-vs-weaver-reflow]]).

**(d) The negative space — cite it explicitly**

Where we searched and found nothing (e.g. a flow-matching video model with
a temporal VAE —
[[../../../00_Inbox/2026-08-01-flow-model-no-temporal-vae-search]]), say so
**with the search scope stated**. An absence with its scope is a finding; a
silent absence looks like an omission.

## Optimisation queue

- [ ] **Q5a — Write `lora.md` and `controlnet.md`.** Blocks Ch3 and the
      LoRA comparison writeup. Highest priority in this note.
- [ ] **Q5b — Write the FiLM / adaLN-Zero conditioning lineage note.**
      This one does double duty: it is a completeness gap *and* it upgrades
      [[01-originality]] by grounding the pathway principle in known
      theory rather than presenting it as a lucky ablation.
- [ ] **Q5c — Audit world-model coverage** against list (b); write the
      3–5 notes that place our setting against retrain-from-scratch.
- [ ] **Q5d — Close the few-step baseline gap** so the D3 comparison is
      honest; resolve
      [[../../../50_Decisions/open/d3-positioning-vs-weaver-reflow]].
- [ ] **Restructure Ch2 by *tension in the field*, not by paper.** Each
      subsection = one unresolved tension, papers as evidence inside it,
      every paragraph ending in a limitation or delta
      ([[../thesis-style-guide]] §6). This is the "nicely synthesized"
      half of the 8-row and costs nothing but ordering.
- [ ] **Resolve the `cafm.md` placeholder** — the note is currently a slot
      with the title and venue marked `_needs verification_`. An unverified
      entry in the bibliography is an "occasional error" (the 6-row).
- [ ] **Accuracy pass.** Hard rule 7a applies with full force to citations:
      no "the paper probably means…". Where a paper is ambiguous, say so
      and cite the page.

## Where it lands in the thesis

- Ch2 — organised by tension, every paragraph ending in a delta
- Ch3 §3.1 — the adapter families (blocked on `lora`, `controlnet`, FiLM)
- Ch4 §4.5 — the shortcut/consistency lineage for the curvature argument
- Ch5 §5.3 — AVID's Action Error Ratio + their baseline table
- §7.3 / §8.2 — the negative space and what the field still lacks
